"""Authoritative P1 contracts-and-identity exit-gate validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from standardized_tabular_diffusion.evaluation.bundle import IncompleteRunBundleWriter, validate_result_bundle
from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    EvaluationRequest,
    MetricState,
    RawDirection,
)
from standardized_tabular_diffusion.evaluation.profiles import list_dataset_profiles, list_protocol_profiles
from standardized_tabular_diffusion.evaluation.registry import load_metric_registry
from standardized_tabular_diffusion.evaluation.schema import list_schemas, load_schema, validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    SerializationError,
    atomic_write_json,
    read_json,
    read_yaml_safe,
    sha256_file,
)

PROTOCOL_ID = "p1-contracts-identity-foundation-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SHA256_FIXTURE = "0" * 64


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _assert_primary_environment() -> None:
    if platform.system() != "Linux" or platform.python_version_tuple()[:2] != ("3", "11"):
        raise AssertionError("Authoritative P1 evidence requires Linux and Python 3.11")


def _repository_commit() -> str:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _request(*, generation_seed: int = 42, reverse: bool = False) -> EvaluationRequest:
    metrics = (
        {"metric_id": "fixture-shape", "metric_version": "1.0.0"},
        {"metric_id": "fixture-trend", "metric_version": "1.0.0"},
    )
    evaluator_seeds = (7, 11)
    if reverse:
        metrics = tuple(reversed(metrics))
        evaluator_seeds = tuple(reversed(evaluator_seeds))
    return EvaluationRequest(
        subject_type="external-synthetic-table",
        sample_artifact={
            "artifact_id": "diagnostic-sample",
            "media_type": "text/csv",
            "sha256": SHA256_FIXTURE,
            "row_count": 4,
        },
        dataset_profile={
            "dataset_id": "diagnostic-fixture",
            "dataset_profile_version": "1.0.0",
            "sha256": SHA256_FIXTURE,
        },
        protocol={
            "protocol_id": "development-p1",
            "protocol_version": "0.1.0",
            "sha256": SHA256_FIXTURE,
        },
        metrics=metrics,
        comparison_track="native",
        generation_seed=generation_seed,
        evaluator_seeds=evaluator_seeds,
    )


def _validate_duplicate_key_rejection(root: Path) -> dict[str, bool]:
    duplicate_json = root / "duplicate.json"
    duplicate_json.write_text('{"identity": 1, "identity": 2}\n', encoding="utf-8")
    duplicate_yaml = root / "duplicate.yaml"
    duplicate_yaml.write_text("identity: 1\nidentity: 2\n", encoding="utf-8")
    results: dict[str, bool] = {}
    for kind, path, reader in (
        ("json", duplicate_json, read_json),
        ("yaml", duplicate_yaml, read_yaml_safe),
    ):
        try:
            reader(path)
        except SerializationError:
            results[kind] = True
        else:
            raise AssertionError(f"Duplicate {kind.upper()} keys were accepted")
    return results


def _validate_atomic_states() -> list[str]:
    states: list[str] = []
    for state in MetricState:
        computed = state is MetricState.COMPUTED
        result = AtomicResult(
            run_id="run-diagnostic",
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
            state=state,
            raw_direction=RawDirection.MAXIMIZE,
            weight=1.0,
            n_reference=4,
            n_synthetic=4,
            n_valid=4,
            n_excluded=0,
            computed_at="2026-08-05T00:00:00Z",
            raw_value=0.75 if computed else None,
            reason_code=None if computed else "diagnostic_state",
            reason_detail=None if computed else "The P1 validator exercised a non-computed state.",
        )
        payload = result.to_dict()
        AtomicResult.from_dict(payload)
        validate_instance("atomic-result", payload)
        states.append(state.value)
    return states


def _locked_file_hashes() -> dict[str, str]:
    roots = (
        REPO_ROOT / "standardized_tabular_diffusion" / "evaluation",
        REPO_ROOT / "standardized_tabular_diffusion" / "schemas" / "evaluation",
        REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "evaluation",
        REPO_ROOT / "configs" / "datasets",
        REPO_ROOT / "tests" / "evaluation",
    )
    paths = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json"}
        and path != REPO_ROOT / "standardized_tabular_diffusion" / "evaluation" / "tabstruct.py"
    }
    paths.update(
        {
            REPO_ROOT / ".github" / "workflows" / "p1-foundation-validation.yml",
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "standardized_tabular_diffusion" / "cli.py",
            Path(__file__).resolve(),
        }
    )
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def run_validation(output: Path, *, require_primary_environment: bool = False) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P1",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Validates contracts, identity, registry/profile loading, and incomplete-bundle safety only. "
            "It executes no scientific metric, claims no source parity, and does not finalize a result bundle. "
            "TabStruct remains research reference material and a clearly isolated legacy diagnostic path."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "jsonschema": _distribution_version("jsonschema"),
            "pyyaml": _distribution_version("PyYAML"),
            "primary_environment_required": require_primary_environment,
        },
    }
    try:
        if require_primary_environment:
            _assert_primary_environment()

        schemas = list_schemas()
        for schema_name in schemas:
            Draft202012Validator.check_schema(load_schema(schema_name))

        registry = load_metric_registry()
        legacy_registry = [record for record in registry if record.metric_id.startswith("legacy-")]
        assert legacy_registry and all(
            record.payload["planned_leaderboard_role"] == "legacy-diagnostic" for record in legacy_registry
        )
        assert all(record.payload["admission"]["official_results_allowed"] is False for record in registry)

        protocols = list_protocol_profiles()
        assert protocols and all(profile.payload["official_results_allowed"] is False for profile in protocols)
        dataset_profiles = list_dataset_profiles(REPO_ROOT / "configs" / "datasets")
        assert dataset_profiles and all(profile.payload["official_eligible"] is False for profile in dataset_profiles)

        first = _request()
        reordered = _request(reverse=True)
        changed = _request(generation_seed=43)
        assert first.fingerprint == reordered.fingerprint
        assert first.fingerprint != changed.fingerprint

        with tempfile.TemporaryDirectory(prefix="std-tabular-p1-") as temporary:
            temporary_root = Path(temporary)
            duplicate_rejection = _validate_duplicate_key_rejection(temporary_root)
            first_report = IncompleteRunBundleWriter(temporary_root / "attempt-a").create(
                first,
                environment={"validation": PROTOCOL_ID},
            )
            second_report = IncompleteRunBundleWriter(temporary_root / "attempt-b").create(
                reordered,
                environment={"validation": PROTOCOL_ID},
            )
            assert first_report.bundle_id != second_report.bundle_id
            assert first_report.finalization_status == second_report.finalization_status == "incomplete"
            assert first_report.pending_files > 0 and second_report.pending_files > 0
            validate_result_bundle(first_report.root)
            validate_result_bundle(second_report.root)

        evidence["result_summary"] = {
            "schemas_validated": len(schemas),
            "metric_records_validated": len(registry),
            "protocol_profiles_validated": len(protocols),
            "dataset_profiles_validated": len(dataset_profiles),
            "metric_states_validated": _validate_atomic_states(),
            "duplicate_key_rejection": duplicate_rejection,
            "equivalent_request_fingerprints_equal": True,
            "scientifically_distinct_request_fingerprints_different": True,
            "repeated_attempt_run_ids_distinct": True,
            "incomplete_bundles_valid": True,
            "finalized_bundle_created": False,
            "scientific_metrics_executed": False,
        }
        evidence["exit_gates"] = {
            "invalid_contracts_fail_deterministically": "pass",
            "schema_and_round_trip_tests": "pass",
            "identity_fingerprints": "pass",
            "interruption_safe_incomplete_bundle": "pass",
            "registry_lifecycle_fail_closed": "pass",
        }
        evidence["workflow_preconditions"] = [
            "python -m pytest tests/evaluation",
            "python -m ruff check standardized_tabular_diffusion/evaluation "
            "standardized_tabular_diffusion/cli.py "
            "standardized_tabular_diffusion/validation/p1_foundation.py tests/evaluation",
            "python -m mypy",
            "python -m build",
        ]
        evidence["locked_files"] = _locked_file_hashes()
        evidence["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"P1 foundation validation failed; inspect {output}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the P1 evaluation contracts and identity foundation")
    parser.add_argument("--output", required=True, type=Path, help="Evidence JSON output path")
    parser.add_argument(
        "--require-primary-environment",
        action="store_true",
        help="Fail unless validation runs on the primary Linux/Python 3.11 environment",
    )
    args = parser.parse_args()
    evidence = run_validation(args.output, require_primary_environment=args.require_primary_environment)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
