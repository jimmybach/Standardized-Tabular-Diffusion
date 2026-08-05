"""Generate bounded P4 diagnostic exit-gate evidence."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, content_fingerprint, sha256_file
from standardized_tabular_diffusion.evaluation.table import validate_utility_tables
from standardized_tabular_diffusion.evaluation.utility import (
    GLOBAL_TARGET_RATIO_METRIC_ID,
    LOCAL_RETENTION_METRIC_ID,
    GlobalBackendResult,
    evaluate_utility,
    global_target_ratio,
    load_p4_evaluator_profile,
    local_retention,
    p4_evaluator_profile_reference,
    validate_utility_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fixture(profile: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data: dict[str, list[object]] = {}
    rows = 120
    for column in profile["columns"]:
        if column["name"] not in profile["table_contract"]["canonical_column_order"]:
            continue
        if column["semantic_type"] in {"continuous", "integer"}:
            minimum = (column.get("valid_domain") or {}).get("minimum", 0)
            data[column["name"]] = [minimum + (index % 60) for index in range(rows)]
        else:
            values = column["valid_domain"]["values"]
            data[column["name"]] = [values[index % min(3, len(values))] for index in range(rows)]
    train = pd.DataFrame(data)
    test = train.iloc[:60].copy(deep=True)
    synthetic = train.copy(deep=True)
    for frame in (train, test, synthetic):
        frame["age"] = [17 + (index % 60) for index in range(len(frame))]
        frame["income"] = frame["age"].map(lambda value: ">50K" if value >= 45 else "<=50K")
    return train, test, synthetic


def _source_boundary_stub(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task_type: str,
    seed: int,
    time_limit_seconds: int,
    arm: str,
) -> GlobalBackendResult:
    del train, test, target, seed, time_limit_seconds
    score = (0.8 if arm == "trtr" else 0.6) if task_type == "classification" else (
        2.0 if arm == "trtr" else 2.5
    )
    predictors = ("KNeighbors", "TabPFN", "XGBoost")
    return GlobalBackendResult(score, predictors, {name: score for name in predictors})


def _artifact(artifact_id: str, fingerprint: str, rows: int) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "media_type": "application/vnd.apache.parquet",
        "sha256": fingerprint,
        "row_count": rows,
    }


def generate_evidence(*, require_primary_environment: bool) -> dict[str, Any]:
    primary = platform.system() == "Linux" and sys.version_info[:2] == (3, 11)
    if require_primary_environment and not primary:
        raise RuntimeError("P4 primary evidence requires Linux with Python 3.11")
    dataset = load_dataset_profile(REPO_ROOT / "configs" / "datasets" / "adult-uci-2-v1.json")
    protocol = resolve_protocol("p4-utility", "0.4.0")
    evaluator = load_p4_evaluator_profile()
    validate_utility_profile(dataset.payload, evaluator)
    train, test, synthetic = _fixture(dataset.payload)
    tables = validate_utility_tables(
        train,
        test,
        synthetic,
        dataset.payload,
        expected_synthetic_rows=len(synthetic),
    )
    metrics = tuple(
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in protocol.payload["metric_selections"]
    )
    request = EvaluationRequest(
        subject_type="external-synthetic-table",
        reference_artifact=_artifact("reference-table", content_fingerprint(train.to_dict("records")), len(train)),
        real_test_artifact=_artifact("real-test-table", content_fingerprint(test.to_dict("records")), len(test)),
        sample_artifact=_artifact("synthetic-table", content_fingerprint(synthetic.to_dict("records")), len(synthetic)),
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
        evaluator_profile=p4_evaluator_profile_reference(),
        model={"model_id": "p4-validation-fixture"},
        resource_limits={"global_time_limit_per_target_seconds": 1},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )
    outcome = evaluate_utility(
        request,
        dataset.payload,
        tables,
        run_id="run-p4-validation",
        global_scorer=_source_boundary_stub,
    )
    retentions = [atom for atom in outcome.atomic_results if atom.metric_id == LOCAL_RETENTION_METRIC_ID]
    ratios = [atom for atom in outcome.atomic_results if atom.metric_id == GLOBAL_TARGET_RATIO_METRIC_ID]
    exit_gates = {
        "three_local_families": "pass" if len(retentions) == 3 else "fail",
        "raw_arms_and_retention": "pass" if outcome.local_summary["retention"] == 1.0 else "fail",
        "all_target_global_formula": "pass" if len(ratios) == 15 and all(atom.raw_value is not None for atom in ratios) else "fail",
        "held_out_test_boundary": "pass" if outcome.details["input_boundary"]["real_test_fit_allowed"] is False else "fail",
        "unclipped_formulas": "pass" if local_retention(0.2, 0.8, 0.92, higher_is_better=True, tolerance=1e-12) > 1 and global_target_ratio(0.8, 0.88, task_type="classification") > 1 else "fail",
        "diagnostic_admission_only": "pass" if evaluator["official_results_allowed"] is False else "fail",
    }
    status = "pass" if set(exit_gates.values()) == {"pass"} else "fail"
    locked = [
        "configs/datasets/adult-uci-2-v1.json",
        "configs/datasets/sick-uci-102-v1.json",
        "standardized_tabular_diffusion/evaluation/utility.py",
        "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-utility-pilot-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/metrics/utility-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/protocols/p4-utility.json",
        "tests/evaluation/test_p4_utility.py",
        "tests/evaluation/test_p4_evaluate_table.py",
    ]
    return {
        "evidence_schema_version": "1.0.0",
        "protocol_id": "p4-utility-diagnostic-v1",
        "phase": "P4",
        "status": status,
        "claim_boundary": (
            "Validates P4 formulas, local official-package boundary, held-out-test isolation, Atomic Result coverage, "
            "and the pinned Global backend call contract. The real AutoGluon/XGB/KNN/TabPFN source runtime was not executed."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "primary_environment_required": require_primary_environment,
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "pyarrow": _package_version("pyarrow"),
        },
        "exit_gates": exit_gates,
        "result_summary": {
            "atomic_results": len(outcome.atomic_results),
            "local_retentions": len(retentions),
            "global_target_ratios": len(ratios),
            "global_utility": outcome.global_summary["global_utility"],
            "official_results_allowed": False,
        },
        "global_source_runtime": {
            "executed": False,
            "reason": "CI diagnostic evidence uses a deterministic call-boundary stub; source-runtime parity is a separate gate.",
            "profile": evaluator["global"]["profile_id"],
            "revision": evaluator["global"]["implementation_source"]["revision"],
        },
        "locked_files": {relative: sha256_file(REPO_ROOT / relative) for relative in locked},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-primary-environment", action="store_true")
    args = parser.parse_args()
    payload = generate_evidence(require_primary_environment=args.require_primary_environment)
    atomic_write_json(args.output, payload)
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
