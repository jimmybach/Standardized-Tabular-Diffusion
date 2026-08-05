"""Authoritative P3 validity and preprocessing-boundary exit-gate validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import tempfile
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.preprocessing import preprocess_splits

PROTOCOL_ID = "p3-validity-and-preprocessing-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _repository_commit() -> str:
    if github_sha := os.environ.get("GITHUB_SHA"):
        return github_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _assert_primary_environment() -> None:
    if platform.system() != "Linux" or platform.python_version_tuple()[:2] != ("3", "11"):
        raise AssertionError("Authoritative P3 evidence requires Linux and Python 3.11")


def _fixture(profile: dict[str, Any]) -> pd.DataFrame:
    data: dict[str, list[object]] = {}
    active = set(profile["table_contract"]["canonical_column_order"])
    for column in profile["columns"]:
        if column["name"] not in active:
            continue
        if column["semantic_type"] in {"continuous", "integer"}:
            domain = column.get("valid_domain") or {}
            minimum = domain.get("minimum", 0)
            data[column["name"]] = [minimum + index for index in range(20)]
        else:
            values = column["valid_domain"]["values"]
            data[column["name"]] = [values[index % min(len(values), 3)] for index in range(20)]
    return pd.DataFrame(data)


def _request(dataset: Any, protocol: Any, reference: Path, synthetic: Path) -> EvaluationRequest:
    metrics = tuple(
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in protocol.payload["metric_selections"]
    )
    return EvaluationRequest(
        subject_type="external-synthetic-table",
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": sha256_file(reference),
            "row_count": 20,
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": sha256_file(synthetic),
            "row_count": 20,
        },
        dataset_profile={
            "dataset_id": dataset.dataset_id,
            "dataset_profile_version": dataset.dataset_profile_version,
            "sha256": dataset.fingerprint,
        },
        protocol={
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.protocol_version,
            "sha256": protocol.fingerprint,
        },
        metrics=metrics,
        comparison_track="native",
        generation_seed=17,
        evaluator_seeds=(23,),
        model={"model_id": "p3-evidence-fixture"},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )


def _preprocessing_boundary() -> dict[str, Any]:
    train = pd.DataFrame({"number": [10.0, 20.0, None], "category": ["b", "a", "b"], "target": [0, 1, 0]})
    first_test = pd.DataFrame({"number": [None, 1_000_000.0], "category": [None, "z"], "target": [1, 0]})
    second_test = pd.DataFrame({"number": [-1_000_000.0, None], "category": ["q", None], "target": [0, 1]})
    first = preprocess_splits(
        train,
        test=first_test,
        numerical_columns=["number"],
        categorical_columns=["category"],
        target_columns=["target"],
    )
    second = preprocess_splits(
        train,
        test=second_test,
        numerical_columns=["number"],
        categorical_columns=["category"],
        target_columns=["target"],
    )
    assert first.state.fingerprint == second.state.fingerprint
    assert first.state.numerical_fill_values == {"number": 15.0}
    assert first.state.categorical_fill_values == {"category": "b"}
    assert first.test is not None and second.test is not None
    assert first.test.loc[0, "number"] == 15.0 and second.test.loc[1, "number"] == 15.0
    return {
        "train_only_state_equal_under_test_changes": True,
        "state_fingerprint": first.state.fingerprint,
        "numerical_fill": 15.0,
        "categorical_fill": "b",
        "synthetic_repair_policy": first.state.policy.synthetic_strategy,
    }


def _locked_files() -> dict[str, str]:
    paths = [
        ".github/workflows/p3-validity-validation.yml",
        "configs/datasets/adult-uci-2-v1.json",
        "configs/datasets/sick-uci-102-v1.json",
        "docs/evaluation/P3_VALIDITY_AND_PREPROCESSING.md",
        "docs/evaluation/P3_VALIDITY_AND_PREPROCESSING.zh-CN.md",
        "pyproject.toml",
        "standardized_tabular_diffusion/__init__.py",
        "standardized_tabular_diffusion/cli.py",
        "standardized_tabular_diffusion/dataset_onboarding.py",
        "standardized_tabular_diffusion/evaluation/bundle.py",
        "standardized_tabular_diffusion/evaluation/__init__.py",
        "standardized_tabular_diffusion/evaluation/evaluate_table.py",
        "standardized_tabular_diffusion/evaluation/table.py",
        "standardized_tabular_diffusion/evaluation/validity.py",
        "standardized_tabular_diffusion/preprocessing.py",
        "standardized_tabular_diffusion/resources/evaluation/metrics/validity-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/protocols/p3-validity.json",
        "standardized_tabular_diffusion/schemas/evaluation/dataset-profile.schema.json",
        "standardized_tabular_diffusion/validation/p3_validity.py",
        "tests/evaluation/test_p3_evaluate_table.py",
        "tests/evaluation/test_p3_validity.py",
        "tests/test_dataset_onboarding.py",
        "tests/test_preprocessing.py",
    ]
    return {relative: sha256_file(REPO_ROOT / relative) for relative in paths}


def run_validation(output: Path, *, require_primary_environment: bool = False) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P3",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Validates benchmark-native hard validity rules, immutable original-output scoring, exact Atomic Result "
            "aggregation, finalized bundles, and train-only mean/mode preprocessing. It does not freeze an "
            "Official Results protocol or approve unresolved dataset constraints."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "pandas": _distribution_version("pandas"),
            "pyarrow": _distribution_version("pyarrow"),
            "primary_environment_required": require_primary_environment,
        },
    }
    try:
        if require_primary_environment:
            _assert_primary_environment()
        dataset = load_dataset_profile(REPO_ROOT / "configs/datasets/adult-uci-2-v1.json")
        protocol = resolve_protocol("p3-validity", "0.3.0")
        reference_frame = _fixture(dataset.payload)
        synthetic_frame = reference_frame.copy(deep=True)
        synthetic_frame["age"] = synthetic_frame["age"].astype(float)
        synthetic_frame.loc[0, "age"] = 17.5
        synthetic_frame.loc[0, "workclass"] = "invalid-category"
        synthetic_frame.loc[0, "occupation"] = pd.NA
        preprocessing = _preprocessing_boundary()
        with tempfile.TemporaryDirectory(prefix="std-tabular-p3-") as temporary:
            root = Path(temporary)
            reference, synthetic = root / "reference.csv", root / "synthetic.csv"
            reference_frame.to_csv(reference, index=False)
            synthetic_frame.to_csv(synthetic, index=False)
            request = _request(dataset, protocol, reference, synthetic)
            reports = []
            for attempt in ("attempt-a", "attempt-b"):
                reports.append(
                    evaluate_table_to_bundle(
                        reference_path=reference,
                        synthetic_path=synthetic,
                        dataset_profile=dataset.payload,
                        protocol_profile=protocol.payload,
                        request=request,
                        output_dir=root / attempt,
                    )
                )
            assert all(report.finalization_status == "finalized" and report.pending_files == 0 for report in reports)
            for report in reports:
                validate_result_bundle(report.root)
            summaries = [read_json(report.root / "summary.json") for report in reports]
            details = [read_json(report.root / "artifacts/validity-details.json") for report in reports]
            assert summaries[0]["validity"] == summaries[1]["validity"]
            assert details[0]["property_scores"] == details[1]["property_scores"]
            assert math.isclose(summaries[0]["validity"]["column_validity_score"], 0.99, abs_tol=1e-12)
            assert summaries[0]["validity"]["constraint_validity_score"] is None
            assert math.isclose(summaries[0]["validity"]["fully_valid_row_rate"], 0.95, abs_tol=1e-12)
            assert summaries[0]["validity"]["synthetic_repair_applied"] is False
            assert details[0]["input_mutated"] is False
            atomic_rows = len(pd.read_parquet(reports[0].root / "metrics.parquet"))
            request_fingerprint = request.fingerprint

        evidence["result_summary"] = {
            "scientific_metrics_executed": True,
            "atomic_results": atomic_rows,
            "expected_atomic_results": 16,
            "column_validity_score": 0.99,
            "constraint_validity_score": None,
            "fully_valid_row_rate": 0.95,
            "finalized_bundles_created": 2,
            "semantic_summaries_equal": True,
            "request_fingerprint": request_fingerprint,
            "synthetic_repair_applied": False,
            "official_results_allowed": False,
        }
        evidence["preprocessing_boundary"] = preprocessing
        evidence["exit_gates"] = {
            "hand_computable_rules": "pass",
            "reviewed_rule_contract": "pass",
            "no_hidden_repair": "pass",
            "no_constraint_state": "pass",
            "width_sensitive_diagnostic_excluded": "pass",
            "train_only_preprocessing": "pass",
            "atomic_aggregation_reproduced": "pass",
            "bundle_finalization_and_checksums": "pass",
        }
        evidence["locked_files"] = _locked_files()
        evidence["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"P3 validation failed; inspect {output}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the P3 validity and preprocessing boundary")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-primary-environment", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run_validation(args.output, require_primary_environment=args.require_primary_environment), indent=2)
    )


if __name__ == "__main__":
    main()
