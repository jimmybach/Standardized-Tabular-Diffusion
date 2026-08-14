"""Private subprocess worker for one standardized benchmark stage."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import ExperimentConfig, load_experiment_config
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file
from standardized_tabular_diffusion.orchestration.process import redact_text
from standardized_tabular_diffusion.runner import build_run_context, run_action, validate_action_inputs

_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret|credential)")


class WorkerError(RuntimeError):
    """Raised when a private worker cannot produce a safe stage artifact."""


def _relative_output(root: Path, path: str | Path | None, *, required: bool = False) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        if required:
            raise WorkerError("A required adapter output was written outside the run root")
        return None
    if candidate.is_symlink() or not candidate.is_file():
        if required:
            raise WorkerError("A required adapter output is missing or unsafe")
        return None
    return relative


def _relative_bundle(root: Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return None
    if candidate.is_symlink() or not candidate.is_dir() or not (candidate / "manifest.json").is_file():
        return None
    return relative


def _portable(value: Any, *, run_root: Path, repo_root: Path, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(item_key): _portable(item, run_root=run_root, repo_root=repo_root, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item, run_root=run_root, repo_root=repo_root) for item in value]
    if isinstance(value, Path):
        return _portable(str(value), run_root=run_root, repo_root=repo_root)
    if isinstance(value, str):
        candidate = Path(value)
        looks_absolute = candidate.is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        if looks_absolute:
            try:
                return candidate.resolve().relative_to(run_root).as_posix()
            except (OSError, ValueError):
                pass
            try:
                relative = candidate.resolve().relative_to(repo_root).as_posix()
                return f"<repo-root>/{relative}"
            except (OSError, ValueError):
                return "<external-path>"
        return redact_text(value, path_aliases={str(run_root): "<run-root>", str(repo_root): "<repo-root>"})
    return value


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".orchestration":
            continue
        stat = path.stat()
        result[relative.as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return result


def _changed_outputs(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    return sorted(relative for relative, identity in after.items() if before.get(relative) != identity)


def _synchronize_accelerator(config: ExperimentConfig, *, reset: bool) -> tuple[int | None, list[str]]:
    if "cuda" not in config.train.device.casefold():
        return None, []
    try:
        import torch
    except ImportError:
        return None, ["accelerator_synchronization_unavailable"]
    if not torch.cuda.is_available():
        return None, ["requested_accelerator_unavailable"]
    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return None, ["accelerator_timing_synchronized"]
    return int(torch.cuda.max_memory_allocated()), ["accelerator_timing_synchronized"]


def _write_stage_result(root: Path, operation: str, payload: dict[str, Any]) -> str:
    relative = {
        "prepare": "artifacts/execution/run_context.json",
        "train": "artifacts/execution/train.json",
        "sample": "artifacts/execution/sample.json",
        "validate": "artifacts/execution/validation.json",
        "evaluate": "artifacts/execution/evaluate.json",
        "aggregate": "pipeline_result.json",
        "report": "benchmark_report.json",
    }[operation]
    atomic_write_json(root / relative, payload)
    return relative


def _bundle_result(operation: str, config: ExperimentConfig, root: Path, repo_root: Path) -> tuple[dict[str, Any], int | None, bool]:
    action_config = config
    if operation == "evaluate":
        sample_result_path = root / "artifacts" / "execution" / "sample.json"
        if sample_result_path.is_file():
            from standardized_tabular_diffusion.evaluation.serialization import read_json

            sample_result = read_json(sample_result_path)
            if isinstance(sample_result, dict) and isinstance(sample_result.get("generated_sample_path"), str):
                action_config = copy.deepcopy(config)
                sample_path = str(root / sample_result["generated_sample_path"])
                action_config.evaluation.extra["sample_path"] = sample_path
                action_config.sample.extra["sample_path"] = sample_path
    bundle = run_action(action_config, operation)
    generated = _relative_output(root, bundle.generated_sample_path, required=operation == "sample")
    upstream_metrics = _relative_output(root, bundle.upstream_metrics_path)
    evaluation_bundle = _relative_bundle(root, bundle.evaluation_bundle_path)
    standardized_summary = _relative_output(root, bundle.standardized_summary_path)
    actual_rows = _count_rows(root / generated) if generated is not None else None
    payload = {
        "stage_result_schema_version": "1.0.0",
        "operation": operation,
        "model": bundle.model,
        "dataset": bundle.dataset,
        "generated_sample_path": generated,
        "upstream_metrics_path": upstream_metrics,
        "evaluation_bundle_path": evaluation_bundle,
        "standardized_summary_path": standardized_summary,
        "notes": _portable(bundle.notes, run_root=root, repo_root=repo_root),
    }
    if operation == "sample":
        cache_safe = generated is not None
    elif operation == "evaluate":
        cache_safe = evaluation_bundle is not None
    else:
        cache_safe = True
    return payload, actual_rows, cache_safe


def _count_rows(path: Path) -> int | None:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError:
            return None
        return int(parquet.ParquetFile(path).metadata.num_rows)
    return None


def _sample_path(config: ExperimentConfig, root: Path) -> tuple[Path, bool]:
    from standardized_tabular_diffusion.evaluation.serialization import read_json

    sample_result = root / "artifacts" / "execution" / "sample.json"
    if sample_result.is_file():
        payload = read_json(sample_result)
        if isinstance(payload, dict) and isinstance(payload.get("generated_sample_path"), str):
            return root / payload["generated_sample_path"], True
    supplied = config.evaluation.extra.get("sample_path") or config.sample.extra.get("sample_path")
    if not isinstance(supplied, str) or not supplied:
        raise WorkerError("No generated or externally supplied sample is available for validation")
    return Path(supplied), False


def _execute(operation: str, config: ExperimentConfig, root: Path, repo_root: Path) -> tuple[str, int | None, bool]:
    if operation == "prepare":
        context = _portable(build_run_context(config), run_root=root, repo_root=repo_root)
        relative = _write_stage_result(
            root,
            operation,
            {"stage_result_schema_version": "1.0.0", "operation": operation, "context": context},
        )
        return relative, None, True
    if operation in {"train", "sample", "evaluate"}:
        payload, actual_rows, preliminary_cache_safe = _bundle_result(operation, config, root, repo_root)
        relative = _write_stage_result(root, operation, payload)
        return relative, actual_rows, preliminary_cache_safe
    if operation == "validate":
        sample, local = _sample_path(config, root)
        if not sample.is_file() or sample.is_symlink():
            raise WorkerError("Synthetic sample must be a regular, non-symlinked file")
        actual_rows = _count_rows(sample)
        payload = {
            "stage_result_schema_version": "1.0.0",
            "operation": operation,
            "sample_artifact": {
                "location": sample.resolve().relative_to(root).as_posix() if local else "external-content-addressed",
                "sha256": sha256_file(sample),
                "byte_size": sample.stat().st_size,
                "row_count": actual_rows,
            },
            "structural_gate": "regular-file-and-readable-row-count" if actual_rows is not None else "regular-file",
            "scientific_repair_applied": False,
        }
        relative = _write_stage_result(root, operation, payload)
        return relative, actual_rows, local
    if operation == "aggregate":
        stage_results: dict[str, Any] = {}
        from standardized_tabular_diffusion.evaluation.serialization import read_json

        for name in ("prepare", "train", "sample", "validate", "evaluate"):
            path = root / "artifacts" / "execution" / ("run_context.json" if name == "prepare" else f"{name}.json")
            if path.is_file() and not path.is_symlink():
                stage_results[name] = read_json(path)
        relative = _write_stage_result(
            root,
            operation,
            {
                "pipeline_result_schema_version": "2.0.0",
                "aggregation_scope": "operational-stage-artifact-index",
                "scientific_or_leaderboard_aggregation_performed": False,
                "stages": stage_results,
            },
        )
        return relative, None, True
    if operation == "report":
        from standardized_tabular_diffusion.evaluation.serialization import read_json

        pipeline_path = root / "pipeline_result.json"
        pipeline = read_json(pipeline_path) if pipeline_path.is_file() else None
        relative = _write_stage_result(
            root,
            operation,
            {
                "benchmark_report_schema_version": "1.0.0",
                "report_scope": "operational-only",
                "leaderboard_source_of_truth": False,
                "pipeline_result_present": pipeline is not None,
                "completed_stage_artifacts": sorted((pipeline or {}).get("stages", {})),
            },
        )
        return relative, None, True
    raise WorkerError(f"Unsupported orchestration worker operation: {operation}")


def run_worker(*, operation: str, config_path: Path, run_root: Path, result_path: Path) -> dict[str, Any]:
    root = run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(config_path)
    if Path(config.output_dir).resolve() != root:
        raise WorkerError("The configuration output_dir must equal the orchestration run root")
    repo_root = Path(__file__).resolve().parents[2]
    before = _snapshot(root)
    _, synchronization = _synchronize_accelerator(config, reset=True)
    started = time.perf_counter()
    fixed_output, actual_rows, cache_safe = _execute(operation, config, root, repo_root)
    if operation == "train":
        if not config.sample.enabled:
            cache_safe = False
            synchronization.append("training_cache_disabled_without_sampling_contract")
        else:
            readiness = validate_action_inputs(config, "sample", repo_root=repo_root)
            if not readiness["ready"]:
                cache_safe = False
                synchronization.append("post_training_sampling_contract_not_ready")
            checkpoint = readiness.get("checked", {}).get("checkpoint_path")
            if isinstance(checkpoint, str):
                try:
                    Path(checkpoint).resolve().relative_to(root)
                except (OSError, ValueError):
                    cache_safe = False
                    synchronization.append("checkpoint_outside_run_root")
    accelerator_peak, end_synchronization = _synchronize_accelerator(config, reset=False)
    action_wall = time.perf_counter() - started
    after = _snapshot(root)
    outputs = _changed_outputs(before, after)
    if fixed_output not in outputs:
        outputs.append(fixed_output)
    dynamic_outputs = [path for path in outputs if path != fixed_output]
    if operation == "train" and not dynamic_outputs:
        cache_safe = False
        synchronization.append("no_model_artifact_captured")
    result = {
        "worker_result_schema_version": "1.0.0",
        "status": "succeeded",
        "action_wall_seconds": action_wall,
        "actual_rows": actual_rows,
        "outputs": sorted(set(outputs)),
        "accelerator_peak_bytes": accelerator_peak,
        "reliability": sorted(set([*synchronization, *end_synchronization])),
        "failure": None,
        "cache_safe": cache_safe,
    }
    atomic_write_json(result_path, result)
    return result


def _failure(category: str, reason_code: str, detail: str) -> dict[str, str]:
    return {"category": category, "reason_code": reason_code, "detail": detail[:500]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        required=True,
        choices=["prepare", "train", "sample", "validate", "evaluate", "aggregate", "report"],
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    result_value = os.environ.get("STD_ORCHESTRATION_WORKER_RESULT")
    if not result_value:
        raise SystemExit("STD_ORCHESTRATION_WORKER_RESULT is required")
    result_path = Path(result_value)
    try:
        run_worker(
            operation=args.operation,
            config_path=args.config,
            run_root=args.run_root,
            result_path=result_path,
        )
    except MemoryError as exc:
        atomic_write_json(
            result_path,
            {
                "worker_result_schema_version": "1.0.0",
                "status": "failed",
                "action_wall_seconds": None,
                "actual_rows": None,
                "outputs": [],
                "accelerator_peak_bytes": None,
                "reliability": [],
                "failure": _failure("out-of-memory", "python_memory_error", type(exc).__name__),
                "cache_safe": False,
            },
        )
        traceback.print_exc()
        raise SystemExit(1) from exc
    except ImportError as exc:
        atomic_write_json(
            result_path,
            {
                "worker_result_schema_version": "1.0.0",
                "status": "failed",
                "action_wall_seconds": None,
                "actual_rows": None,
                "outputs": [],
                "accelerator_peak_bytes": None,
                "reliability": [],
                "failure": _failure("dependency", "worker_dependency_import_failure", type(exc).__name__),
                "cache_safe": False,
            },
        )
        traceback.print_exc()
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            result_path,
            {
                "worker_result_schema_version": "1.0.0",
                "status": "failed",
                "action_wall_seconds": None,
                "actual_rows": None,
                "outputs": [],
                "accelerator_peak_bytes": None,
                "reliability": [],
                "failure": _failure("implementation", "worker_execution_failure", type(exc).__name__),
                "cache_safe": False,
            },
        )
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
