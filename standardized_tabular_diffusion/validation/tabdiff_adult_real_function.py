from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import tomllib
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter
from standardized_tabular_diffusion.validation.tabdiff import verify_sources

PROTOCOL_ID = "tabdiff-adult-real-function-windows-v1"
MODEL_ID = "tabdiff"
DATASET_ID = "adult"
TRAINING_SEED = 0
GENERATION_SEEDS = (3, 4, 5)
EXPECTED_TRAIN_ROWS = 32_561
EXPECTED_TEST_ROWS = 16_281
EXPERIMENT_NAME = "adult-windows-rtx5080-real-function-v2"
CONFIG_PATH = Path("configs/validation/tabdiff-adult-real-function-v1.toml")
OFFICIAL_CONFIG_PATH = Path("TabDiff-main/tabdiff/configs/tabdiff_configs.toml")
ALLOWED_CONFIG_DIFFERENCES = {
    "data.dequant_dist",
    "sample.batch_size",
    "train.main.check_val_every",
    "train.main.steps",
}


class TabDiffRealFunctionError(RuntimeError):
    """Raised when the bounded real-function acceptance protocol fails closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def _validate_config(repo_root: Path) -> dict[str, Any]:
    official_path = repo_root / OFFICIAL_CONFIG_PATH
    runtime_path = repo_root / CONFIG_PATH
    with official_path.open("rb") as stream:
        official = tomllib.load(stream)
    with runtime_path.open("rb") as stream:
        runtime = tomllib.load(stream)
    official_flat = _flatten(official)
    runtime_flat = _flatten(runtime)
    all_keys = set(official_flat) | set(runtime_flat)
    differences = sorted(key for key in all_keys if official_flat.get(key) != runtime_flat.get(key))
    if set(differences) != ALLOWED_CONFIG_DIFFERENCES:
        raise TabDiffRealFunctionError(
            f"Real-function config differs from official config outside the approved boundary: {differences}"
        )
    return {
        "official_path": OFFICIAL_CONFIG_PATH.as_posix(),
        "official_sha256": _sha256_file(official_path),
        "runtime_path": CONFIG_PATH.as_posix(),
        "runtime_sha256": _sha256_file(runtime_path),
        "differences": {key: {"official": official_flat[key], "runtime": runtime_flat[key]} for key in differences},
        "claim": "bounded real-function acceptance; not an official-quality training run",
    }


def _hardware() -> dict[str, Any]:
    import torch

    if platform.system() != "Windows":
        raise TabDiffRealFunctionError("This retained protocol requires Windows.")
    if sys.version_info[:2] != (3, 11):
        raise TabDiffRealFunctionError(
            f"This retained protocol requires Python 3.11, observed {platform.python_version()}."
        )
    if not torch.cuda.is_available():
        raise TabDiffRealFunctionError("CUDA is unavailable.")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA GeForce RTX 5080":
        raise TabDiffRealFunctionError(f"Expected NVIDIA GeForce RTX 5080, observed {device_name}.")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": device_name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_count": torch.cuda.device_count(),
    }


def _validate_adult_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import pandas as pd

    data_root = repo_root / "TabDiff-main" / "data" / DATASET_ID
    synthetic_root = repo_root / "TabDiff-main" / "synthetic" / DATASET_ID
    info_path = data_root / "info.json"
    train_path = data_root / "train.csv"
    test_path = data_root / "test.csv"
    real_path = synthetic_root / "real.csv"
    metric_test_path = synthetic_root / "test.csv"
    required = [info_path, train_path, test_path, real_path, metric_test_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise TabDiffRealFunctionError(f"Materialized Adult inputs are missing: {missing}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    expected_columns = info["column_names"]
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise TabDiffRealFunctionError("Adult row counts differ from the reviewed official split.")
    if list(train.columns) != expected_columns or list(test.columns) != expected_columns:
        raise TabDiffRealFunctionError("Adult columns differ from the reviewed TabDiff materialization.")
    if train.isna().any().any() or test.isna().any().any():
        raise TabDiffRealFunctionError("Adult model inputs contain missing values after train-only imputation.")
    numerical_columns = [expected_columns[index] for index in info["num_col_idx"]]
    if not np.isfinite(train[numerical_columns].to_numpy(dtype=float)).all():
        raise TabDiffRealFunctionError("Adult training inputs contain non-finite numerical values.")

    evidence = {
        "info_sha256": _sha256_file(info_path),
        "train_csv_sha256": _sha256_file(train_path),
        "test_csv_sha256": _sha256_file(test_path),
        "metric_real_csv_sha256": _sha256_file(real_path),
        "metric_test_csv_sha256": _sha256_file(metric_test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "columns": expected_columns,
        "missing_values": 0,
        "materialization": "reviewed official UCI split with train-only imputation",
    }
    return info, evidence


def _validate_sample(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    columns = info["column_names"]
    numerical_columns = [columns[index] for index in info["num_col_idx"]]
    integer_columns = [columns[index] for index in info["int_col_idx"]]
    categorical_indices = [*info["cat_col_idx"], *info["target_col_idx"]]
    categorical_columns = [columns[index] for index in categorical_indices]
    if len(frame) != EXPECTED_TRAIN_ROWS or list(frame.columns) != columns:
        raise TabDiffRealFunctionError(f"Generated table has an invalid shape or column order: {path}")
    if frame.isna().any().any():
        raise TabDiffRealFunctionError(f"Generated table contains missing values: {path}")
    numeric = frame[numerical_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise TabDiffRealFunctionError(f"Generated table contains invalid numerical values: {path}")
    non_integral = {column: int((numeric[column] != np.rint(numeric[column])).sum()) for column in integer_columns}
    if any(non_integral.values()):
        raise TabDiffRealFunctionError(f"Generated integer columns contain fractional values: {non_integral}")
    out_of_range: dict[str, int] = {}
    for column in numerical_columns:
        column_info = info["column_info"][column]
        outside = (numeric[column] < column_info["min"]) | (numeric[column] > column_info["max"])
        if outside.any():
            out_of_range[column] = int(outside.sum())
    if out_of_range:
        raise TabDiffRealFunctionError(f"Generated numerical values violate reviewed ranges: {out_of_range}")

    invalid_categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        allowed = set(info["column_info"][column]["categories"])
        observed = set(frame[column].astype(str).unique())
        unexpected = sorted(observed - allowed)
        if unexpected:
            invalid_categories[column] = unexpected
    if invalid_categories:
        raise TabDiffRealFunctionError(
            f"Generated table contains categories outside the reviewed domains: {invalid_categories}"
        )

    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_values": 0,
        "finite_numerical_values": True,
        "integer_columns_integral": True,
        "numerical_ranges_valid": True,
        "categorical_domains_valid": True,
    }


def run_protocol(repo_root: Path, output_root: Path, evidence_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    evidence_path = evidence_path.resolve()
    if output_root.exists():
        raise TabDiffRealFunctionError(f"Refusing to overwrite protocol output: {output_root}")
    checkpoint_root = repo_root / "TabDiff-main" / "tabdiff" / "ckpt" / DATASET_ID / EXPERIMENT_NAME
    result_root = repo_root / "TabDiff-main" / "tabdiff" / "result" / DATASET_ID / EXPERIMENT_NAME
    if checkpoint_root.exists() or result_root.exists():
        raise TabDiffRealFunctionError("Refusing to reuse an existing upstream TabDiff experiment directory.")
    output_root.mkdir(parents=True)

    started_at = _utc_now()
    started = time.perf_counter()
    source = verify_sources(repo_root)
    config = _validate_config(repo_root)
    hardware = _hardware()
    info, dataset = _validate_adult_inputs(repo_root)
    adapter = TabDiffAdapter(repo_root)
    runtime_config_path = repo_root / CONFIG_PATH

    print(f"[{_utc_now()}] Training TabDiff on the complete Adult training split", flush=True)
    train_output = output_root / "train"
    adapter.train(
        RunSpec(
            model=MODEL_ID,
            dataset=DATASET_ID,
            output_dir=train_output,
            device="cuda:0",
            seed=TRAINING_SEED,
            upstream_config_path=runtime_config_path,
            extra={"deterministic": True, "exp_name": EXPERIMENT_NAME},
        )
    )
    checkpoint = checkpoint_root / "model_20.pt"
    if not checkpoint.is_file():
        raise TabDiffRealFunctionError(f"TabDiff training did not emit the declared checkpoint: {checkpoint}")

    sample_results: list[dict[str, Any]] = []
    for seed in GENERATION_SEEDS:
        print(f"[{_utc_now()}] Generating a complete Adult table with seed {seed}", flush=True)
        sample_output = output_root / f"seed-{seed}"
        bundle = adapter.sample(
            RunSpec(
                model=MODEL_ID,
                dataset=DATASET_ID,
                output_dir=sample_output,
                device="cuda:0",
                seed=seed,
                num_samples=EXPECTED_TRAIN_ROWS,
                checkpoint_path=checkpoint,
                upstream_config_path=runtime_config_path,
                extra={
                    "allow_unsafe_external_checkpoint": True,
                    "deterministic": True,
                    "exp_name": EXPERIMENT_NAME,
                },
            )
        )
        if bundle.generated_sample_path is None:
            raise TabDiffRealFunctionError(f"TabDiff seed {seed} did not expose a generated table.")
        sample = _validate_sample(bundle.generated_sample_path, info)
        sample["generation_seed"] = seed
        sample["run_metadata_sha256"] = _sha256_file(sample_output / "tabdiff_run.json")
        sample_results.append(sample)

    hashes = [record["sha256"] for record in sample_results]
    if len(set(hashes)) != len(GENERATION_SEEDS):
        raise TabDiffRealFunctionError("Distinct generation seeds did not produce three distinct tables.")

    evidence = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "repository": {
            "head": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "working_tree_dirty": bool(_git(repo_root, "status", "--short")),
        },
        "environment": {
            "hardware": hardware,
            "dependencies": {
                name: _version(name) for name in ("numpy", "pandas", "scikit-learn", "sdmetrics", "torch", "xgboost")
            },
        },
        "model": {
            "model_id": MODEL_ID,
            "training_seed": TRAINING_SEED,
            "generation_seeds": list(GENERATION_SEEDS),
            "training_runs": 1,
            "checkpoint_reused_across_seeds": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "source_integrity": source,
            "configuration": config,
            "train_run_metadata_sha256": _sha256_file(train_output / "tabdiff_run.json"),
        },
        "dataset": dataset,
        "samples": sample_results,
        "assertions": {
            "real_official_dataset": True,
            "real_tabdiff_training": True,
            "windows_python311_rtx5080": True,
            "one_training_run": True,
            "three_distinct_generation_seeds": True,
            "complete_row_count_each_seed": True,
            "all_structural_checks_passed": True,
            "official_results_admitted": False,
            "release_support_admitted": False,
        },
        "claim_boundary": (
            "Development real-function acceptance only. The shortened training preset does not measure final "
            "model quality and does not admit TabDiff, Adult, or these samples to Official Results."
        ),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded TabDiff/Adult Windows real-function protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_protocol(args.repo_root, args.output_root, args.evidence_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(
                json.dumps(
                    {
                        "evidence_schema_version": "1.0.0",
                        "protocol_id": PROTOCOL_ID,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
