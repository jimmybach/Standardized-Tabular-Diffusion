"""P8 legacy-isolation, central-evaluation, quickstart, and release exit gate."""

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

from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.leaderboard import validate_snapshot_bundle
from standardized_tabular_diffusion.evaluation.legacy import import_legacy_summary, validate_legacy_import
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.evaluation.service import evaluate_table_files
from standardized_tabular_diffusion.platform_support import is_primary_release_family_environment
from standardized_tabular_diffusion.quickstart import run_quickstart

PROTOCOL_ID = "p8-migration-release-exit-gate-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _is_exact_native_windows_11(
    *, build: str, product_type: str, machine: str, python_version: tuple[int, int]
) -> bool:
    try:
        build_number = int(build)
    except ValueError:
        return False
    return (
        build_number >= 22000
        and product_type == "WinNT"
        and machine.lower() in {"amd64", "x86_64"}
        and python_version == (3, 11)
    )


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


def _legacy_fixture() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_name": "tabstruct-aligned-v1",
        "model": "tabddpm",
        "dataset": "adult",
        "metrics": {
            "density": {"shape_score": None, "trend_score": None, "overall_score": None, "status": "missing"},
            "ml_efficacy": {
                "primary_metric_name": None,
                "primary_metric_value": None,
                "task_type": "classification",
                "details": {},
            },
            "detection": {"logistic_detection": None, "status": "missing"},
            "privacy": {"dcr_score": None, "status": "not_requested"},
            "structural_fidelity": {"global_utility": None, "status": "missing"},
        },
        "tabstruct_alignment": {
            "density_fidelity": "legacy",
            "ml_efficacy": "legacy",
            "detection": "legacy",
            "privacy": "legacy",
            "structural_fidelity": "legacy",
        },
    }


def _locked_file_hashes() -> dict[str, str]:
    paths = {
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "MANIFEST.in",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "RELEASE_CHECKLIST.md",
        REPO_ROOT / ".github" / "workflows" / "p8-release-validation.yml",
        REPO_ROOT / "standardized_tabular_diffusion" / "evaluation" / "legacy.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "evaluation" / "service.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "quickstart.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "runner.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "orchestration" / "pipeline.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "orchestration" / "worker.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "validation" / "core_ci.py",
        REPO_ROOT / "tests" / "evaluation" / "test_p8_legacy_migration.py",
        REPO_ROOT / "tests" / "evaluation" / "test_p8_quickstart.py",
        Path(__file__).resolve(),
    }
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def _validate_docs() -> None:
    required = {
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "CONTRIBUTORS.md",
        "LICENSE",
        "NOTICE",
        "RELEASE_CHECKLIST.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/ARCHITECTURE.md",
        "docs/ARCHITECTURE.zh-CN.md",
        "docs/DATASET_CARDS.md",
        "docs/DATASET_CARDS.zh-CN.md",
        "docs/METRIC_CARDS.md",
        "docs/METRIC_CARDS.zh-CN.md",
        "docs/QUICKSTART.md",
        "docs/QUICKSTART.zh-CN.md",
        "docs/TROUBLESHOOTING.md",
        "docs/TROUBLESHOOTING.zh-CN.md",
        "docs/evaluation/P8_MIGRATION_AND_RELEASE.md",
        "docs/evaluation/P8_MIGRATION_AND_RELEASE.zh-CN.md",
    }
    missing = sorted(path for path in required if not (REPO_ROOT / path).is_file())
    if missing:
        raise AssertionError(f"P8 release documentation is incomplete: {missing}")
    for relative in required:
        if relative.endswith((".md", ".cff")):
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            if "C:\\Users\\" in text or "/home/runner/" in text:
                raise AssertionError(f"Developer-local path leaked into {relative}")


def _native_windows_release() -> dict[str, str | bool]:
    """Describe exact native-Windows eligibility without conflating Windows Server.

    Windows continues to report the NT 10.0 API version on Windows 11.  The
    consumer/workstation product type and build number are therefore both
    required; ``platform.release()`` alone is not a valid Windows 11 check.
    """

    details: dict[str, str | bool] = {
        "build": "unavailable",
        "display_version": "unavailable",
        "product_name": "unavailable",
        "product_type": "unavailable",
        "is_exact_native_windows_11_python_311_x86_64": False,
    }
    if platform.system() != "Windows":
        return details
    try:
        registry: Any = __import__("winreg")

        with registry.OpenKey(
            registry.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as key:
            details["build"] = str(registry.QueryValueEx(key, "CurrentBuildNumber")[0])
            details["display_version"] = str(registry.QueryValueEx(key, "DisplayVersion")[0])
            details["product_name"] = str(registry.QueryValueEx(key, "ProductName")[0])
        with registry.OpenKey(
            registry.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\ProductOptions"
        ) as key:
            details["product_type"] = str(registry.QueryValueEx(key, "ProductType")[0])
    except (OSError, ValueError):
        return details
    details["is_exact_native_windows_11_python_311_x86_64"] = _is_exact_native_windows_11(
        build=str(details["build"]),
        product_type=str(details["product_type"]),
        machine=platform.machine(),
        python_version=sys.version_info[:2],
    )
    return details


def run_validation(
    output: Path,
    *,
    require_primary_family_environment: bool = False,
    require_native_windows_11: bool = False,
) -> dict[str, Any]:
    native_windows = _native_windows_release()
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P8",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Validates P8 software migration and release-candidate engineering only. It does not admit a model, "
            "dataset, metric, protocol, run, suite, result, or adapter to Official Results or release support."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "python": platform.python_version(),
            "primary_family_environment_required": require_primary_family_environment,
            "exact_native_windows_11_required": require_native_windows_11,
            "native_windows_release": native_windows,
        },
    }
    try:
        if require_primary_family_environment and not is_primary_release_family_environment():
            raise AssertionError("Primary-family P8 evidence requires Windows with Python 3.11")
        if require_native_windows_11 and not native_windows["is_exact_native_windows_11_python_311_x86_64"]:
            raise AssertionError("Exact native release evidence requires Windows 11 x86-64 with Python 3.11")
        _validate_docs()
        with tempfile.TemporaryDirectory(prefix="std-tabular-p8-") as temporary:
            root = Path(temporary)
            quickstart_root = root / "quickstart"
            quickstart = run_quickstart(quickstart_root, seed=17)
            result = validate_result_bundle(quickstart_root / "evaluation-result")
            snapshot = validate_snapshot_bundle(quickstart_root / "diagnostic-snapshot")
            assert result.finalization_status == "finalized"
            assert snapshot["publication_class"] == "partial-diagnostic"
            assert all(item["rank"] is None for item in snapshot["leaderboard"])
            assert not (quickstart_root / "adapter" / "standardized_summary.json").exists()

            table_only = evaluate_table_files(
                reference_path=quickstart_root / "inputs" / "train.csv",
                synthetic_path=quickstart_root / "inputs" / "train.csv",
                dataset_profile_path=quickstart_root / "inputs" / "dataset-profile.json",
                protocol_id="p3-validity",
                output_dir=root / "table-only-result",
                expected_rows=24,
            )
            assert table_only.report.finalization_status == "finalized"

            source = root / "standardized_summary.json"
            atomic_write_json(source, _legacy_fixture())
            legacy_root = root / "legacy-import"
            legacy = import_legacy_summary(source, legacy_root)
            assert legacy == validate_legacy_import(legacy_root)
            record = read_json(legacy_root / "legacy_import_record.json")
            assert record["conversion"]["official_results_allowed"] is False
            assert record["conversion"]["atomic_evidence_available"] is False

        evidence["result_summary"] = {
            "adapter": quickstart["model"],
            "generation_seed": quickstart["generation_seed"],
            "result_bundle_id": quickstart["result_bundle_id"],
            "diagnostic_snapshot_id": quickstart["snapshot_id"],
            "legacy_source_sha256": legacy["source_sha256"],
            "table_only_request_fingerprint": table_only.request.fingerprint,
        }
        evidence["exit_gates"] = {
            "legacy_schema_frozen": "pass",
            "legacy_import_read_only_and_checksum_bound": "pass",
            "legacy_official_promotion_impossible": "pass",
            "adapter_evaluation_uses_central_result_bundle": "pass",
            "table_only_evaluation_finalized": "pass",
            "official_smote_quickstart_completed": "pass",
            "diagnostic_snapshot_valid_and_unranked": "pass",
            "release_assets_present_and_portable": "pass",
        }
        evidence["locked_files"] = _locked_file_hashes()
        evidence["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"P8 release validation failed; inspect {output}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-primary-family-environment", action="store_true")
    parser.add_argument("--require-native-windows11", action="store_true")
    args = parser.parse_args()
    evidence = run_validation(
        args.output,
        require_primary_family_environment=args.require_primary_family_environment,
        require_native_windows_11=args.require_native_windows11,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
