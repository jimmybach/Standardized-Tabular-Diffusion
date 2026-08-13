"""P6 orchestration, cache, resume, and resource-boundary exit validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.contracts import AtomicResult, MetricState, RawDirection
from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    content_fingerprint,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.orchestration.engine import StageSpec, execute_plan, validate_orchestration_run
from standardized_tabular_diffusion.orchestration.hardware import (
    IncompatibleHardwareProfileError,
    assert_efficiency_compatible,
    capture_hardware_profile,
)
from standardized_tabular_diffusion.orchestration.process import ProcessOutcome
from standardized_tabular_diffusion.platform_support import is_primary_release_family_environment

PROTOCOL_ID = "p6-orchestration-exit-gate-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FINGERPRINT = "0" * 64


def _repository_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _software_profile() -> dict[str, Any]:
    repository = {"commit": "fixture", "dirty": False, "patch_sha256": "2" * 64}
    material = {
        "python": {"implementation": "CPython", "version": platform.python_version()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": [],
        "material_environment": {},
        "repository": repository,
        "locale": "p6-fixture",
    }
    return {
        "software_profile_schema_version": "1.0.0",
        **material,
        "fingerprint": content_fingerprint(material),
        "code_fingerprint": content_fingerprint(repository),
    }


def _write_stage(
    root: Path,
    name: str,
    output: str,
    content: str,
    *,
    dependencies: tuple[str, ...] = (),
    **changes: Any,
) -> StageSpec:
    command = (
        sys.executable,
        "-c",
        "from pathlib import Path; import os; Path(os.environ['P6_OUTPUT']).write_text(os.environ['P6_CONTENT'])",
    )
    values: dict[str, Any] = {
        "name": name,
        "command": command,
        "action": f"write deterministic {name} validation artifact",
        "dependencies": dependencies,
        "identity_inputs": {"content": content},
        "output_paths": (output,),
        "timeout_seconds": 10.0,
        "environment_updates": {"P6_OUTPUT": str(root / output), "P6_CONTENT": content},
    }
    values.update(changes)
    return StageSpec(**values)


def _execute(root: Path, stages: list[StageSpec], hardware: dict[str, Any], **changes: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_root": root,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "repo_root": REPO_ROOT,
        "hardware_profile": hardware,
        "software_profile": _software_profile(),
    }
    values.update(changes)
    return execute_plan(stages, **values)


def _records(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in (root / ".orchestration" / "attempts").glob("*.json"):
        payload = read_json(path)
        assert isinstance(payload, dict)
        result[payload["stage_id"]] = payload
    return result


def _latest(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = _records(root)
    return [records[stage_id] for stage_id in manifest["current_stage_ids"]]


def _locked_file_hashes() -> dict[str, str]:
    roots = (
        REPO_ROOT / "standardized_tabular_diffusion" / "orchestration",
        REPO_ROOT / "tests" / "orchestration",
    )
    paths = {
        path
        for root in roots
        for path in root.rglob("*.py")
        if path.is_file()
    }
    paths.update(
        {
            REPO_ROOT / "standardized_tabular_diffusion" / "schemas" / "evaluation" / "hardware-profile.schema.json",
            REPO_ROOT
            / "standardized_tabular_diffusion"
            / "schemas"
            / "evaluation"
            / "orchestration-stage-record.schema.json",
            REPO_ROOT
            / "standardized_tabular_diffusion"
            / "schemas"
            / "evaluation"
            / "orchestration-run.schema.json",
            REPO_ROOT
            / "standardized_tabular_diffusion"
            / "schemas"
            / "evaluation"
            / "software-profile.schema.json",
            REPO_ROOT / ".github" / "workflows" / "p6-orchestration-validation.yml",
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "standardized_tabular_diffusion" / "cli.py",
            REPO_ROOT / "standardized_tabular_diffusion" / "evaluation" / "schema.py",
            REPO_ROOT / "standardized_tabular_diffusion" / "validation" / "core_ci.py",
            REPO_ROOT / "tests" / "evaluation" / "test_contracts_and_schemas.py",
            REPO_ROOT / "tests" / "test_cli.py",
            Path(__file__).resolve(),
        }
    )
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def run_validation(output: Path, *, require_primary_family_environment: bool = False) -> dict[str, Any]:
    hardware = capture_hardware_profile(profile_id="p6-validation-observed-host")
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P6",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Validates resource-aware process orchestration, content-addressed reuse, resume ancestry, "
            "failure isolation, and observed-profile efficiency diagnostics. It performs no model-quality "
            "assessment, leaderboard aggregation, hardware normalization, or Official Results admission."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "primary_family_environment_required": require_primary_family_environment,
            "hardware_profile_id": hardware["profile_id"],
            "hardware_comparison_key": hardware["comparison_key"],
        },
    }
    try:
        if require_primary_family_environment and not is_primary_release_family_environment():
            raise AssertionError("Primary-family P6 evidence requires Windows with Python 3.11")
        with tempfile.TemporaryDirectory(prefix="std-tabular-p6-") as temporary:
            root = Path(temporary)

            cache_root = root / "cache"
            first = _execute(
                cache_root,
                [_write_stage(cache_root, "prepare", "artifact.txt", "stable")],
                hardware,
            )
            first_decision = _latest(cache_root, first)[0]["cache"]["decision"]
            second = _execute(
                cache_root,
                [_write_stage(cache_root, "prepare", "artifact.txt", "stable")],
                hardware,
            )
            second_record = _latest(cache_root, second)[0]
            assert second_record["cache"]["decision"] == "hit"
            assert second_record["resource_usage"]["efficiency_eligible"] is False
            changed = _execute(
                cache_root,
                [_write_stage(cache_root, "prepare", "artifact.txt", "changed")],
                hardware,
            )
            changed_decision = _latest(cache_root, changed)[0]["cache"]["decision"]
            assert changed_decision == "miss"

            stale_root = root / "stale"
            stale_first = _execute(
                stale_root,
                [_write_stage(stale_root, "prepare", "artifact.txt", "stable")],
                hardware,
            )
            stale_record = _latest(stale_root, stale_first)[0]
            digest = stale_record["outputs"][0]["sha256"]
            blob = stale_root / ".orchestration" / "cache" / "artifacts" / digest[:2] / digest
            blob.write_text("corrupt", encoding="utf-8")
            (stale_root / "artifact.txt").unlink()
            stale_second = _execute(
                stale_root,
                [_write_stage(stale_root, "prepare", "artifact.txt", "stable")],
                hardware,
            )
            stale_decision = _latest(stale_root, stale_second)[0]["cache"]["decision"]
            assert stale_decision == "invalid"
            assert validate_orchestration_run(stale_root)["valid"] is True

            timeout_root = root / "timeout"
            timeout_stage = StageSpec(
                name="evaluate",
                command=(sys.executable, "-c", "import time; time.sleep(5)"),
                action="force timeout",
                timeout_seconds=0.2,
                identity_inputs={"fixture": "timeout"},
            )
            timeout_manifest = _execute(timeout_root, [timeout_stage], hardware)
            timeout_record = _latest(timeout_root, timeout_manifest)[0]
            assert timeout_record["failure"]["category"] == "timeout"

            memory_root = root / "memory"
            memory_stage = StageSpec(
                name="evaluate",
                command=(
                    sys.executable,
                    "-c",
                    "import time; payload=bytearray(64*1024*1024); time.sleep(5)",
                ),
                action="force process-tree memory boundary",
                memory_limit_bytes=8 * 1024 * 1024,
                identity_inputs={"fixture": "memory"},
            )
            memory_manifest = _execute(memory_root, [memory_stage], hardware)
            memory_record = _latest(memory_root, memory_manifest)[0]
            assert memory_record["failure"]["category"] == "out-of-memory"

            retry_root = root / "retry"
            marker = root / "retry.marker"
            retry_stage = StageSpec(
                name="train",
                command=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import os,sys; marker=Path(os.environ['MARKER']); "
                    "out=Path(os.environ['OUT']); "
                    "(marker.write_text('x'),sys.exit(2)) if not marker.exists() else out.write_text('ok')",
                ),
                action="force one retry",
                output_paths=("model.txt",),
                identity_inputs={"fixture": "retry"},
                max_retries=1,
                environment_updates={"MARKER": str(marker), "OUT": str(retry_root / "model.txt")},
            )
            retry_manifest = _execute(retry_root, [retry_stage], hardware)
            retry_records = [record for record in _records(retry_root).values() if record["stage_name"] == "train"]
            assert retry_manifest["status"] == "success"
            assert {record["status"] for record in retry_records} == {"failed", "succeeded"}

            partial_root = root / "partial"
            computed_result = AtomicResult(
                run_id="run-p6-fixture",
                protocol_version="1.0.0",
                dataset_id="diagnostic-fixture",
                dataset_version="1.0.0",
                dataset_view="canonical-v1",
                split_id="split-v1",
                model_id="external",
                comparison_track="native",
                generation_seed=42,
                metric_id="fixture-shape",
                metric_version="1.0.0",
                dimension="fidelity",
                scope_type="column",
                scope_id="feature-a",
                state=MetricState.COMPUTED,
                raw_direction=RawDirection.MAXIMIZE,
                weight=1.0,
                n_reference=4,
                n_synthetic=4,
                n_valid=8,
                n_excluded=0,
                computed_at="2026-08-13T00:00:00Z",
                raw_value=0.75,
            )
            computed_payload = json.dumps(computed_result.to_dict(), sort_keys=True)
            good = _write_stage(partial_root, "evaluate.good", "atomic-result.json", computed_payload)
            optional = StageSpec(
                name="evaluate.optional",
                command=(sys.executable, "-c", "raise RuntimeError('optional backend failure')"),
                action="force optional backend failure",
                required=False,
                identity_inputs={"fixture": "optional"},
            )
            aggregate = _write_stage(
                partial_root,
                "aggregate",
                "aggregate.json",
                "partial-preserved",
                dependencies=("evaluate.good", "evaluate.optional"),
                allow_failed_dependencies=True,
            )
            partial_manifest = _execute(partial_root, [good, optional, aggregate], hardware)
            assert partial_manifest["status"] == "partial"
            preserved_result = json.loads((partial_root / "atomic-result.json").read_text(encoding="utf-8"))
            AtomicResult.from_dict(preserved_result)
            validate_instance("atomic-result", preserved_result)

            interrupted_root = root / "interrupted"

            def interrupted_runner(*args: Any, **kwargs: Any) -> ProcessOutcome:
                return ProcessOutcome(
                    exit_code=None,
                    wall_seconds=0.01,
                    cpu_seconds=0.0,
                    peak_rss_bytes=1,
                    peak_accelerator_memory_bytes=None,
                    timed_out=False,
                    memory_limit_exceeded=False,
                    interrupted=True,
                    launch_error=None,
                    reliability=(),
                    diagnostic_tail=(),
                )

            interrupted_manifest = _execute(
                interrupted_root,
                [
                    _write_stage(interrupted_root, "train", "model.txt", "never"),
                    _write_stage(
                        interrupted_root,
                        "sample",
                        "sample.csv",
                        "never",
                        dependencies=("train",),
                    ),
                ],
                hardware,
                process_runner=interrupted_runner,
            )
            assert interrupted_manifest["status"] == "cancelled"
            assert [record["status"] for record in _latest(interrupted_root, interrupted_manifest)] == [
                "cancelled",
                "skipped",
            ]

            efficiency_observations: list[dict[str, Any]] = []
            for index in range(3):
                efficiency_root = root / f"efficiency-{index}"
                stage = StageSpec(
                    name="sample",
                    command=(
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import os,time; time.sleep(0.25); "
                        "Path(os.environ['OUT']).write_text('rows')",
                    ),
                    action="fixed observed-profile timing workload",
                    output_paths=("sample.txt",),
                    identity_inputs={"fixture": "efficiency"},
                    environment_updates={"OUT": str(efficiency_root / "sample.txt")},
                    requested_rows=4,
                    warmup_policy="fixed-validation-workload",
                )
                manifest = _execute(efficiency_root, [stage], hardware, use_cache=False)
                record = _latest(efficiency_root, manifest)[0]
                efficiency_observations.append(record["resource_usage"])
            wall = [float(item["wall_seconds"]) for item in efficiency_observations]
            rss = [int(item["peak_rss_bytes"]) for item in efficiency_observations]
            assert max(wall) - min(wall) <= 0.5
            assert max(rss) / max(1, min(rss)) <= 1.5
            assert_efficiency_compatible([hardware, hardware, hardware])
            incompatible = dict(hardware)
            incompatible["comparison_key"] = "f" * 64
            try:
                assert_efficiency_compatible([hardware, incompatible])
            except IncompatibleHardwareProfileError:
                cross_profile_rejected = True
            else:
                raise AssertionError("Cross-profile efficiency comparison was accepted")

        evidence["result_summary"] = {
            "cache_first_decision": first_decision,
            "cache_repeat_decision": second_record["cache"]["decision"],
            "changed_identity_decision": changed_decision,
            "stale_cache_decision": stale_decision,
            "timeout_category": timeout_record["failure"]["category"],
            "memory_category": memory_record["failure"]["category"],
            "retry_attempt_count": len(retry_records),
            "partial_status": partial_manifest["status"],
            "completed_atomic_result_preserved": True,
            "interruption_status": interrupted_manifest["status"],
            "efficiency_observation_count": len(efficiency_observations),
            "efficiency_wall_seconds": wall,
            "efficiency_peak_rss_bytes": rss,
            "efficiency_tolerance": {"wall_absolute_seconds": 0.5, "peak_rss_ratio": 1.5},
            "cross_profile_efficiency_rejected": cross_profile_rejected,
        }
        evidence["exit_gates"] = {
            "forced_timeout": "pass",
            "simulated_out_of_memory": "pass",
            "interruption_and_dependency_skip": "pass",
            "retry_with_ancestry": "pass",
            "stale_cache_rejection": "pass",
            "exact_identity_cache_reuse": "pass",
            "partial_success_preserves_completed_result": "pass",
            "named_hardware_profile_efficiency_tolerance": "pass",
            "cross_profile_ranking_prohibited": "pass",
            "structured_run_validation": "pass",
        }
        evidence["locked_files"] = _locked_file_hashes()
        evidence["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"P6 orchestration validation failed; inspect {output}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-primary-family-environment",
        action="store_true",
        help="Fail unless validation runs on the primary Windows/Python 3.11 family",
    )
    args = parser.parse_args()
    evidence = run_validation(
        args.output,
        require_primary_family_environment=args.require_primary_family_environment,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
