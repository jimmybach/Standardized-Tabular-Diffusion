"""End-to-end protocol evaluation of a decoded synthetic table."""

from __future__ import annotations

import io
import platform
import subprocess
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from standardized_tabular_diffusion.evaluation.backends.sdmetrics import (
    SDMetricsBackendError,
    SDMetricsSourceError,
)
from standardized_tabular_diffusion.evaluation.bundle import (
    BundleError,
    BundleValidationReport,
    IncompleteRunBundleWriter,
)
from standardized_tabular_diffusion.evaluation.contracts import (
    ContractError,
    EvaluationRequest,
    StageRecord,
    StageStatus,
    utc_timestamp,
)
from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    content_fingerprint,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.evaluation.shape_trend import (
    DETAILS_ARTIFACT_PATH,
    P2_METRICS,
    ShapeTrendError,
    ShapeTrendOutcome,
    evaluate_shape_trend,
)
from standardized_tabular_diffusion.evaluation.table import (
    TableValidationError,
    ValidatedTables,
    ValidatedUtilityTables,
    validate_tables,
    validate_utility_tables,
)
from standardized_tabular_diffusion.evaluation.utility import (
    P4_METRICS,
    UTILITY_DETAILS_ARTIFACT_PATH,
    UtilityError,
    UtilityOutcome,
    evaluate_utility,
)
from standardized_tabular_diffusion.evaluation.validity import (
    P3_METRICS,
    VALIDITY_DETAILS_ARTIFACT_PATH,
    ValidityError,
    ValidityOutcome,
    evaluate_validity,
)


class TableEvaluationError(RuntimeError):
    """Raised when a table request cannot produce a trustworthy bundle."""


def _stage(
    name: str,
    *,
    status: StageStatus,
    dependencies: tuple[str, ...],
    inputs: dict[str, str],
    action: str,
    started_at: str | None,
    ended_at: str | None,
    elapsed_seconds: float | None,
    outputs: tuple[dict[str, Any], ...] = (),
    reason_code: str | None = None,
    failure_category: str | None = None,
) -> StageRecord:
    return StageRecord(
        stage_name=name,
        stage_version="1.0.0",
        status=status,
        dependency_stage_ids=dependencies,
        input_fingerprints=inputs,
        resolved_action=action,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed_seconds,
        process_exit_code=0 if status is StageStatus.SUCCEEDED else None,
        log_refs=("logs/events.jsonl",),
        outputs=outputs,
        warning_codes=(),
        failure_category=failure_category,
        failure_reason_code=reason_code,
        cache_decision="not-requested",
        retry_count=0,
        resume_ancestry=(),
    )


def _output(path: str, media_type: str, digest: str) -> dict[str, str]:
    return {"path": path, "media_type": media_type, "sha256": digest}


def _write_stage(writer: IncompleteRunBundleWriter, record: StageRecord, *, required: bool = True) -> str:
    return writer.write_json(
        f"stages/{record.stage_name}.json",
        record.to_dict(),
        required=required,
        schema_name="stage-record",
    )


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "sdmetrics"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _producer() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", True
    return {
        "repository": "https://github.com/jimmybach/Standardized-Tabular-Diffusion",
        "commit": commit,
        "dirty": dirty,
    }


def _artifact(
    artifact_id: str,
    role: str,
    media_type: str,
    *,
    path: str | None,
    external_uri: str | None,
    byte_size: int | None,
    sha256: str,
    producer_stage: str | None,
    publication_class: str,
    rights_classification: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "role": role,
        "media_type": media_type,
        "path": path,
        "external_uri": external_uri,
        "byte_size": byte_size,
        "sha256": sha256,
        "producer_stage": producer_stage,
        "retention_class": "release-evidence",
        "publication_class": publication_class,
        "rights_classification": rights_classification,
        "access_requirements": {},
    }


def _atomic_parquet(outcome: ShapeTrendOutcome | ValidityOutcome | UtilityOutcome) -> bytes:
    records = [result.to_dict() for result in outcome.atomic_results]
    for record in records:
        validate_instance("atomic-result", record)
    frame = pd.DataFrame.from_records(records)
    buffer = io.BytesIO()
    try:
        frame.to_parquet(buffer, engine="pyarrow", compression="zstd", index=False)
    except ImportError as exc:
        raise TableEvaluationError("Finalized table bundles require the pinned pyarrow dependency") from exc
    return buffer.getvalue()


def _terminal_payloads(
    request: EvaluationRequest,
    profile: dict[str, Any],
    outcome: ShapeTrendOutcome,
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    reference_rows: int,
    synthetic_rows: int,
    artifact_refs: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_counts = Counter(result.state.value for result in outcome.atomic_results)
    noncomputed = [
        result for result in outcome.atomic_results if result.state.value not in {"computed", "not_applicable"}
    ]
    terminal_status = (
        "success"
        if not noncomputed and all(value is not None for value in outcome.property_scores.values())
        else "partial"
    )
    summary = {
        "summary_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "terminal_status": terminal_status,
        "validity": {"structural_gate": "passed", "p3_validity_scoring": "not-in-p2"},
        "dimensions": {
            "fidelity": {
                "column_shapes": outcome.property_scores["column_shapes"],
                "column_pair_trends": outcome.property_scores["column_pair_trends"],
                "combined_score": None,
                "combined_score_policy": "not-defined-in-p2",
            }
        },
        "local_utility": {},
        "global_utility": {},
        "privacy_risk": {},
        "efficiency": {},
        "metric_state_counts": dict(sorted(state_counts.items())),
        "denominator_counts": outcome.denominator_counts,
        "warnings": sorted({warning for result in outcome.atomic_results for warning in result.warning_codes}),
        "failures": [
            {
                "metric_id": result.metric_id,
                "scope_id": result.scope_id,
                "state": result.state.value,
                "reason_code": result.reason_code,
                "reason_detail": result.reason_detail,
            }
            for result in noncomputed
        ],
        "atomic_result_refs": [f"metrics.parquet#row={index}" for index in range(len(outcome.atomic_results))],
        "aggregation": {
            "implementation": "source-score-reproduction-from-atomic-contributions",
            "version": "1.0.0",
            "reproducible_from_atomic_results": True,
        },
        "dataset_aggregation_eligible": False,
    }
    metadata = {
        "metadata_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "protocol": request.protocol,
        "dataset": request.dataset_profile,
        "model": request.model or {"subject_type": request.subject_type, "model_id": "external"},
        "implementation": {
            "evaluation_subsystem": "p2-shape-trend-vertical-slice",
            "metrics_executed": True,
            "source": outcome.source,
        },
        "comparison_track": request.comparison_track,
        "seeds": {"generation": request.generation_seed, "evaluators": list(request.evaluator_seeds)},
        "evaluator": {"profile": request.evaluator_profile, "hardware_profile": request.hardware_profile},
        "execution": {
            "requested_action": "evaluate-table",
            "started_at": started_at,
            "ended_at": ended_at,
            "terminal_phase": "report",
            "run_status": terminal_status,
            "requested_synthetic_rows": request.sample_artifact.get("row_count", reference_rows),
            "actual_synthetic_rows": synthetic_rows,
            "resource_limits": request.resource_limits,
            "interrupted": False,
            "resume_ancestry": [],
            "warning_codes": summary["warnings"],
            "failure_category": None,
            "failure_reason_code": None,
            "artifact_refs": artifact_refs,
        },
        "coverage": {
            "requested_metrics": list(request.metrics),
            "computed": state_counts.get("computed", 0),
            "states": dict(sorted(state_counts.items())),
            "denominators": outcome.denominator_counts,
        },
        "provenance": {
            "reference_artifact": request.reference_artifact,
            "sample_artifact": request.sample_artifact,
            "reference_rows": reference_rows,
            "dataset_profile_status": profile["status"],
        },
        "review": {"status": "source-parity-validated-p2", "official_results_allowed": False},
        "status": "finalized",
    }
    validate_instance("summary", summary)
    validate_instance("metadata", metadata)
    return summary, metadata


def _validity_terminal_payloads(
    request: EvaluationRequest,
    profile: dict[str, Any],
    outcome: ValidityOutcome,
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    reference_rows: int,
    synthetic_rows: int,
    artifact_refs: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_counts = Counter(result.state.value for result in outcome.atomic_results)
    noncomputed = [
        result for result in outcome.atomic_results if result.state.value not in {"computed", "not_applicable"}
    ]
    terminal_status = "success" if not noncomputed else "partial"
    validity = {
        "structural_gate": "passed",
        "input_view": "original-decoded-synthetic-output",
        "synthetic_repair_applied": False,
        **outcome.property_scores,
        "aggregation": {
            "column_validity": "equal-column mean",
            "constraint_validity": "equal-reviewed-applicable-constraint mean or null",
            "overall": "equal 0.5 components when constraints apply; otherwise column validity",
            "fully_valid_row_rate": "reported only because it is width-sensitive",
        },
    }
    summary = {
        "summary_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "terminal_status": terminal_status,
        "validity": validity,
        "dimensions": {"validity": validity},
        "local_utility": {},
        "global_utility": {},
        "privacy_risk": {},
        "efficiency": {},
        "metric_state_counts": dict(sorted(state_counts.items())),
        "denominator_counts": outcome.denominator_counts,
        "warnings": sorted({warning for result in outcome.atomic_results for warning in result.warning_codes}),
        "failures": [
            {
                "metric_id": result.metric_id,
                "scope_id": result.scope_id,
                "state": result.state.value,
                "reason_code": result.reason_code,
                "reason_detail": result.reason_detail,
            }
            for result in noncomputed
        ],
        "atomic_result_refs": [f"metrics.parquet#row={index}" for index in range(len(outcome.atomic_results))],
        "aggregation": {
            "implementation": "benchmark-native-validity-from-atomic-contributions",
            "version": "1.0.0",
            "reproducible_from_atomic_results": True,
        },
        "dataset_aggregation_eligible": False,
    }
    metadata = {
        "metadata_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "protocol": request.protocol,
        "dataset": request.dataset_profile,
        "model": request.model or {"subject_type": request.subject_type, "model_id": "external"},
        "implementation": {
            "evaluation_subsystem": "p3-validity-and-preprocessing-boundary",
            "metrics_executed": True,
            "source": outcome.source,
        },
        "comparison_track": request.comparison_track,
        "seeds": {"generation": request.generation_seed, "evaluators": list(request.evaluator_seeds)},
        "evaluator": {"profile": request.evaluator_profile, "hardware_profile": request.hardware_profile},
        "execution": {
            "requested_action": "evaluate-table",
            "started_at": started_at,
            "ended_at": ended_at,
            "terminal_phase": "report",
            "run_status": terminal_status,
            "requested_synthetic_rows": request.sample_artifact.get("row_count", reference_rows),
            "actual_synthetic_rows": synthetic_rows,
            "resource_limits": request.resource_limits,
            "interrupted": False,
            "resume_ancestry": [],
            "warning_codes": summary["warnings"],
            "failure_category": None,
            "failure_reason_code": None,
            "artifact_refs": artifact_refs,
        },
        "coverage": {
            "requested_metrics": list(request.metrics),
            "computed": state_counts.get("computed", 0),
            "states": dict(sorted(state_counts.items())),
            "denominators": outcome.denominator_counts,
        },
        "provenance": {
            "reference_artifact": request.reference_artifact,
            "sample_artifact": request.sample_artifact,
            "reference_rows": reference_rows,
            "dataset_profile_status": profile["status"],
            "original_synthetic_output_preserved": True,
            "evaluation_repair_applied": False,
        },
        "review": {"status": "unit-validated-p3-development", "official_results_allowed": False},
        "status": "finalized",
    }
    validate_instance("summary", summary)
    validate_instance("metadata", metadata)
    return summary, metadata


def _utility_terminal_payloads(
    request: EvaluationRequest,
    profile: dict[str, Any],
    outcome: UtilityOutcome,
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    real_train_rows: int,
    real_test_rows: int,
    synthetic_rows: int,
    artifact_refs: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_counts = Counter(result.state.value for result in outcome.atomic_results)
    noncomputed = [result for result in outcome.atomic_results if result.state.value != "computed"]
    terminal_status = (
        "success"
        if not noncomputed
        and outcome.local_summary["retention"] is not None
        and outcome.global_summary["global_utility"] is not None
        else "partial"
    )
    summary = {
        "summary_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "terminal_status": terminal_status,
        "validity": {
            "structural_gate": "passed",
            "input_view": "strict-canonical-model-view",
            "synthetic_repair_applied": False,
        },
        "dimensions": {
            "local-utility": outcome.local_summary,
            "global-utility": outcome.global_summary,
        },
        "local_utility": outcome.local_summary,
        "global_utility": outcome.global_summary,
        "privacy_risk": {},
        "efficiency": {},
        "metric_state_counts": dict(sorted(state_counts.items())),
        "denominator_counts": outcome.denominator_counts,
        "warnings": sorted({warning for result in outcome.atomic_results for warning in result.warning_codes}),
        "failures": [
            {
                "metric_id": result.metric_id,
                "scope_id": result.scope_id,
                "state": result.state.value,
                "reason_code": result.reason_code,
                "reason_detail": result.reason_detail,
            }
            for result in noncomputed
        ],
        "atomic_result_refs": [f"metrics.parquet#row={index}" for index in range(len(outcome.atomic_results))],
        "aggregation": {
            "implementation": "p4-raw-arms-to-unclipped-local-retention-and-tabstruct-global-ratios",
            "version": "1.0.0",
            "reproducible_from_atomic_results": True,
        },
        "dataset_aggregation_eligible": False,
    }
    metadata = {
        "metadata_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "protocol": request.protocol,
        "dataset": request.dataset_profile,
        "model": request.model or {"subject_type": request.subject_type, "model_id": "external"},
        "implementation": {
            "evaluation_subsystem": "p4-local-global-utility",
            "metrics_executed": True,
            "source": outcome.source,
        },
        "comparison_track": request.comparison_track,
        "seeds": {"generation": request.generation_seed, "evaluators": list(request.evaluator_seeds)},
        "evaluator": {"profile": request.evaluator_profile, "hardware_profile": request.hardware_profile},
        "execution": {
            "requested_action": "evaluate-table",
            "started_at": started_at,
            "ended_at": ended_at,
            "terminal_phase": "report",
            "run_status": terminal_status,
            "requested_synthetic_rows": request.sample_artifact.get("row_count", real_train_rows),
            "actual_synthetic_rows": synthetic_rows,
            "resource_limits": request.resource_limits,
            "interrupted": False,
            "resume_ancestry": [],
            "warning_codes": summary["warnings"],
            "failure_category": None,
            "failure_reason_code": None,
            "artifact_refs": artifact_refs,
        },
        "coverage": {
            "requested_metrics": list(request.metrics),
            "computed": state_counts.get("computed", 0),
            "states": dict(sorted(state_counts.items())),
            "denominators": outcome.denominator_counts,
        },
        "provenance": {
            "reference_artifact": request.reference_artifact,
            "real_test_artifact": request.real_test_artifact,
            "sample_artifact": request.sample_artifact,
            "real_train_rows": real_train_rows,
            "real_test_rows": real_test_rows,
            "dataset_profile_status": profile["status"],
            "original_synthetic_output_preserved": True,
            "evaluation_repair_applied": False,
            "test_used_for_fit": False,
        },
        "review": {
            "status": "unit-validated-p4-development",
            "official_results_allowed": False,
            "global_source_parity_claimed": False,
        },
        "status": "finalized",
    }
    validate_instance("summary", summary)
    validate_instance("metadata", metadata)
    return summary, metadata


def evaluate_table_to_bundle(
    *,
    reference_path: str | Path,
    synthetic_path: str | Path,
    real_test_path: str | Path | None = None,
    dataset_profile: dict[str, Any],
    protocol_profile: dict[str, Any],
    request: EvaluationRequest,
    output_dir: str | Path,
) -> BundleValidationReport:
    """Evaluate decoded table inputs and finalize a self-validating bundle."""

    reference_path = Path(reference_path)
    synthetic_path = Path(synthetic_path)
    resolved_test_path = Path(real_test_path) if real_test_path is not None else None
    validate_instance("dataset-profile", dataset_profile)
    validate_instance("protocol-profile", protocol_profile)
    if request.reference_artifact is None:
        raise TableEvaluationError("evaluate-table requires reference_artifact provenance")
    if sha256_file(reference_path) != request.reference_artifact["sha256"]:
        raise TableEvaluationError("Reference table checksum differs from the Evaluation Request")
    if sha256_file(synthetic_path) != request.sample_artifact["sha256"]:
        raise TableEvaluationError("Synthetic table checksum differs from the Evaluation Request")
    if request.real_test_artifact is not None:
        if resolved_test_path is None:
            raise TableEvaluationError("Evaluation Request declares real_test_artifact but no real_test_path was supplied")
        if sha256_file(resolved_test_path) != request.real_test_artifact["sha256"]:
            raise TableEvaluationError("Real test table checksum differs from the Evaluation Request")
    elif resolved_test_path is not None:
        raise TableEvaluationError("real_test_path requires checksum-bound real_test_artifact provenance")
    if content_fingerprint(dataset_profile) != request.dataset_profile["sha256"]:
        raise TableEvaluationError("Dataset Profile fingerprint differs from the Evaluation Request")
    if content_fingerprint(protocol_profile) != request.protocol["sha256"]:
        raise TableEvaluationError("Protocol Profile fingerprint differs from the Evaluation Request")
    protocol_metrics = [
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in protocol_profile["metric_selections"]
    ]
    if protocol_metrics != list(request.metrics):
        raise TableEvaluationError("Protocol metric selections differ from the Evaluation Request")
    requested_metrics = {(item["metric_id"], item["metric_version"]) for item in request.metrics}
    p2_metrics = {(item["metric_id"], item["metric_version"]) for item in P2_METRICS}
    p3_metrics = {(item["metric_id"], item["metric_version"]) for item in P3_METRICS}
    p4_metrics = {(item["metric_id"], item["metric_version"]) for item in P4_METRICS}
    if requested_metrics == p2_metrics:
        phase: Literal["p2", "p3", "p4"] = "p2"
        content_mode: Literal["strict", "preserve"] = "strict"
    elif requested_metrics == p3_metrics:
        phase = "p3"
        content_mode = "preserve"
    elif requested_metrics == p4_metrics:
        phase = "p4"
        content_mode = "strict"
        if request.real_test_artifact is None or resolved_test_path is None:
            raise TableEvaluationError("P4 Utility requires a checksum-bound held-out real test table")
    else:
        raise TableEvaluationError(
            "evaluate-table supports exactly the registered P2 Shape/Trend, P3 Validity, or P4 Utility metric set"
        )

    writer = IncompleteRunBundleWriter(output_dir)
    writer.create(request, environment=_environment(), producer=_producer())
    manifest = read_json(Path(output_dir) / "manifest.json")
    run_id = manifest["bundle_id"]
    run_started = utc_timestamp()

    stage_start = utc_timestamp()
    timer = time.perf_counter()
    tables: ValidatedTables | ValidatedUtilityTables
    prepare = _stage(
        "prepare",
        status=StageStatus.SUCCEEDED,
        dependencies=(),
        inputs={"request": request.fingerprint},
        action="resolve immutable request, profiles, and input artifacts",
        started_at=stage_start,
        ended_at=utc_timestamp(),
        elapsed_seconds=time.perf_counter() - timer,
    )
    _write_stage(writer, prepare)
    for stage_name, dependency in (("train", "prepare"), ("sample", "train")):
        skipped = _stage(
            stage_name,
            status=StageStatus.SKIPPED,
            dependencies=(dependency,),
            inputs={"request": request.fingerprint},
            action="use supplied external decoded table",
            started_at=None,
            ended_at=None,
            elapsed_seconds=None,
            reason_code="external_table_supplied",
        )
        _write_stage(writer, skipped, required=False)

    stage_start = utc_timestamp()
    timer = time.perf_counter()
    try:
        if phase == "p4":
            if resolved_test_path is None:
                raise TableEvaluationError("P4 real test path was not resolved")
            tables = validate_utility_tables(
                reference_path,
                resolved_test_path,
                synthetic_path,
                dataset_profile,
                expected_synthetic_rows=request.sample_artifact.get("row_count"),
            )
        else:
            tables = validate_tables(
                reference_path,
                synthetic_path,
                dataset_profile,
                expected_synthetic_rows=request.sample_artifact.get("row_count"),
                content_mode=content_mode,
            )
    except TableValidationError as exc:
        failed = _stage(
            "validate",
            status=StageStatus.FAILED,
            dependencies=("prepare", "sample"),
            inputs={
                "dataset-profile": request.dataset_profile["sha256"],
                "reference-table": request.reference_artifact["sha256"],
                "synthetic-table": request.sample_artifact["sha256"],
                **(
                    {"real-test-table": request.real_test_artifact["sha256"]}
                    if request.real_test_artifact is not None
                    else {}
                ),
            },
            action=f"apply the {phase.upper()} structural validation gate",
            started_at=stage_start,
            ended_at=utc_timestamp(),
            elapsed_seconds=time.perf_counter() - timer,
            reason_code=exc.reason_code,
            failure_category="input-validation",
        )
        _write_stage(writer, failed)
        writer.append_event(
            severity="error",
            stage="validate",
            component=f"{phase}-structural-gate",
            event_code="validation.failed",
            details={"reason_code": exc.reason_code, "detail": exc.detail},
        )
        raise TableEvaluationError(
            f"Structural validation failed ({exc.reason_code}): {exc.detail}; incomplete evidence bundle: {output_dir}"
        ) from exc
    structural_hash = writer.write_json(
        "artifacts/structural-validation.json",
        tables.report,
        required=True,
    )
    validate_record = _stage(
        "validate",
        status=StageStatus.SUCCEEDED,
        dependencies=("prepare", "sample"),
        inputs={
            "dataset-profile": request.dataset_profile["sha256"],
            "reference-table": request.reference_artifact["sha256"],
            "synthetic-table": request.sample_artifact["sha256"],
            **(
                {"real-test-table": request.real_test_artifact["sha256"]}
                if request.real_test_artifact is not None
                else {}
            ),
        },
        action=f"apply the {phase.upper()} structural validation gate",
        started_at=stage_start,
        ended_at=utc_timestamp(),
        elapsed_seconds=time.perf_counter() - timer,
        outputs=(_output("artifacts/structural-validation.json", "application/json", structural_hash),),
    )
    _write_stage(writer, validate_record)

    stage_start = utc_timestamp()
    timer = time.perf_counter()
    outcome: ShapeTrendOutcome | ValidityOutcome | UtilityOutcome
    try:
        if phase == "p2":
            if not isinstance(tables, ValidatedTables):
                raise TableEvaluationError("P2 resolved an incompatible structural table view")
            shape_outcome = evaluate_shape_trend(request, dataset_profile, tables, run_id=run_id)
            outcome = shape_outcome
            details_path = DETAILS_ARTIFACT_PATH
            details_payload = shape_outcome.source_details
            evaluation_action = "run pinned SDMetrics properties and map every scope to Atomic Result"
            source_input = shape_outcome.source["python_source_tree_sha256"]
            source_input_name = "sdmetrics-source"
        elif phase == "p3":
            if not isinstance(tables, ValidatedTables):
                raise TableEvaluationError("P3 resolved an incompatible structural table view")
            validity_outcome = evaluate_validity(request, dataset_profile, tables, run_id=run_id)
            outcome = validity_outcome
            details_path = VALIDITY_DETAILS_ARTIFACT_PATH
            details_payload = validity_outcome.details
            evaluation_action = "evaluate reviewed hard column rules and cross-column constraints"
            source_input = validity_outcome.source["dataset_validity_contract_sha256"]
            source_input_name = "validity-contract"
        else:
            if not isinstance(tables, ValidatedUtilityTables):
                raise TableEvaluationError("P4 resolved an incompatible structural table view")
            utility_outcome = evaluate_utility(request, dataset_profile, tables, run_id=run_id)
            outcome = utility_outcome
            details_path = UTILITY_DETAILS_ARTIFACT_PATH
            details_payload = utility_outcome.details
            evaluation_action = "run Local Dummy/TRTR/TSTR and TabStruct-profile Global Utility over held-out real test"
            source_input = content_fingerprint(utility_outcome.source)
            source_input_name = "utility-source-profile"
    except (SDMetricsBackendError, ShapeTrendError, ValidityError, UtilityError, ContractError) as exc:
        if isinstance(exc, SDMetricsSourceError):
            reason_code = "source_attestation_failure"
        elif isinstance(exc, SDMetricsBackendError):
            reason_code = "upstream_metric_execution_failure"
        elif isinstance(exc, ValidityError):
            reason_code = "validity_contract_failure"
        elif isinstance(exc, UtilityError):
            reason_code = "utility_contract_failure"
        else:
            reason_code = "metric_contract_failure"
        failed = _stage(
            "evaluate",
            status=StageStatus.FAILED,
            dependencies=("validate",),
            inputs={"structural-validation": structural_hash},
            action={
                "p2": "run pinned SDMetrics properties and map every scope to Atomic Result",
                "p3": "evaluate reviewed hard column rules and cross-column constraints",
                "p4": "run Local and Global Utility over one held-out real test set",
            }[phase],
            started_at=stage_start,
            ended_at=utc_timestamp(),
            elapsed_seconds=time.perf_counter() - timer,
            reason_code=reason_code,
            failure_category="metric-execution",
        )
        _write_stage(writer, failed)
        writer.append_event(
            severity="error",
            stage="evaluate",
            component=f"{phase}-evaluation",
            event_code="evaluation.failed",
            details={"reason_code": reason_code, "detail": str(exc)},
        )
        raise TableEvaluationError(
            f"Metric evaluation failed ({reason_code}): {exc}; incomplete evidence bundle: {output_dir}"
        ) from exc
    details_hash = writer.write_json(details_path, details_payload, required=True)
    metrics_hash = writer.write_bytes(
        "metrics.parquet",
        _atomic_parquet(outcome),
        media_type="application/vnd.apache.parquet",
        required=True,
    )
    evaluate_record = _stage(
        "evaluate",
        status=StageStatus.SUCCEEDED,
        dependencies=("validate",),
        inputs={
            "structural-validation": structural_hash,
            source_input_name: source_input,
        },
        action=evaluation_action,
        started_at=stage_start,
        ended_at=utc_timestamp(),
        elapsed_seconds=time.perf_counter() - timer,
        outputs=(
            _output("metrics.parquet", "application/vnd.apache.parquet", metrics_hash),
            _output(details_path, "application/json", details_hash),
        ),
    )
    _write_stage(writer, evaluate_record)

    artifact_refs = [
        "artifacts/structural-validation.json",
        details_path,
        "metrics.parquet",
    ]
    run_ended = utc_timestamp()
    if isinstance(outcome, ShapeTrendOutcome):
        if not isinstance(tables, ValidatedTables):
            raise TableEvaluationError("P2 terminal payload received an incompatible table view")
        summary, metadata = _terminal_payloads(
            request,
            dataset_profile,
            outcome,
            run_id=run_id,
            started_at=run_started,
            ended_at=run_ended,
            reference_rows=len(tables.reference),
            synthetic_rows=len(tables.synthetic),
            artifact_refs=artifact_refs,
        )
    elif isinstance(outcome, ValidityOutcome):
        if not isinstance(tables, ValidatedTables):
            raise TableEvaluationError("P3 terminal payload received an incompatible table view")
        summary, metadata = _validity_terminal_payloads(
            request,
            dataset_profile,
            outcome,
            run_id=run_id,
            started_at=run_started,
            ended_at=run_ended,
            reference_rows=len(tables.reference),
            synthetic_rows=len(tables.synthetic),
            artifact_refs=artifact_refs,
        )
    else:
        if not isinstance(tables, ValidatedUtilityTables):
            raise TableEvaluationError("P4 terminal payload received an incompatible table view")
        summary, metadata = _utility_terminal_payloads(
            request,
            dataset_profile,
            outcome,
            run_id=run_id,
            started_at=run_started,
            ended_at=run_ended,
            real_train_rows=len(tables.real_train),
            real_test_rows=len(tables.real_test),
            synthetic_rows=len(tables.synthetic),
            artifact_refs=artifact_refs,
        )
    stage_start = utc_timestamp()
    timer = time.perf_counter()
    summary_hash = writer.write_json("summary.json", summary, schema_name="summary")
    aggregate_record = _stage(
        "aggregate",
        status=StageStatus.SUCCEEDED,
        dependencies=("evaluate",),
        inputs={"atomic-results": metrics_hash},
        action={
            "p2": "reproduce source property scores from Atomic Result contributions",
            "p3": "reproduce validity component scores from Atomic Result contributions",
            "p4": "reproduce Local retention and strict all-target Global Utility from Atomic Results",
        }[phase],
        started_at=stage_start,
        ended_at=utc_timestamp(),
        elapsed_seconds=time.perf_counter() - timer,
        outputs=(_output("summary.json", "application/json", summary_hash),),
    )
    _write_stage(writer, aggregate_record)

    stage_start = utc_timestamp()
    timer = time.perf_counter()
    metadata_hash = writer.write_json("metadata.json", metadata, schema_name="metadata")
    index = {
        "artifact_index_schema_version": "1.0.0",
        "artifacts": [
            _artifact(
                "reference-table",
                "reference decoded table",
                request.reference_artifact["media_type"],
                path=None,
                external_uri=f"urn:sha256:{request.reference_artifact['sha256']}",
                byte_size=reference_path.stat().st_size,
                sha256=request.reference_artifact["sha256"],
                producer_stage=None,
                publication_class="external-not-copied",
                rights_classification="dataset-profile-governed",
            ),
            _artifact(
                "synthetic-table",
                "synthetic decoded table",
                request.sample_artifact["media_type"],
                path=None,
                external_uri=f"urn:sha256:{request.sample_artifact['sha256']}",
                byte_size=synthetic_path.stat().st_size,
                sha256=request.sample_artifact["sha256"],
                producer_stage=None,
                publication_class="external-not-copied",
                rights_classification="generated-data",
            ),
            *(
                [
                    _artifact(
                        "real-test-table",
                        "held-out real test decoded table",
                        request.real_test_artifact["media_type"],
                        path=None,
                        external_uri=f"urn:sha256:{request.real_test_artifact['sha256']}",
                        byte_size=resolved_test_path.stat().st_size,
                        sha256=request.real_test_artifact["sha256"],
                        producer_stage=None,
                        publication_class="external-not-copied",
                        rights_classification="dataset-profile-governed",
                    )
                ]
                if request.real_test_artifact is not None and resolved_test_path is not None
                else []
            ),
            _artifact(
                "structural-validation",
                "structural gate evidence",
                "application/json",
                path="artifacts/structural-validation.json",
                external_uri=None,
                byte_size=(Path(output_dir) / "artifacts/structural-validation.json").stat().st_size,
                sha256=structural_hash,
                producer_stage="validate",
                publication_class="bundle-public",
                rights_classification="benchmark-generated",
            ),
            _artifact(
                {"p2": "sdmetrics-details", "p3": "validity-details", "p4": "utility-details"}[phase],
                {
                    "p2": "verbatim source metric details",
                    "p3": "hard-rule validity evidence",
                    "p4": "raw-arm, support, predictor, ratio, and held-out-boundary evidence",
                }[phase],
                "application/json",
                path=details_path,
                external_uri=None,
                byte_size=(Path(output_dir) / details_path).stat().st_size,
                sha256=details_hash,
                producer_stage="evaluate",
                publication_class="bundle-public",
                rights_classification=(
                    "benchmark-generated-from-source-output" if phase == "p2" else "benchmark-generated"
                ),
            ),
            _artifact(
                "atomic-results",
                "one row per Atomic Result",
                "application/vnd.apache.parquet",
                path="metrics.parquet",
                external_uri=None,
                byte_size=(Path(output_dir) / "metrics.parquet").stat().st_size,
                sha256=metrics_hash,
                producer_stage="evaluate",
                publication_class="bundle-public",
                rights_classification="benchmark-generated",
            ),
        ],
    }
    index_hash = writer.write_json("artifacts/index.json", index, schema_name="artifact-index")
    report_record = _stage(
        "report",
        status=StageStatus.SUCCEEDED,
        dependencies=("aggregate",),
        inputs={"summary": summary_hash},
        action="write provenance, artifact index, and final report metadata",
        started_at=stage_start,
        ended_at=utc_timestamp(),
        elapsed_seconds=time.perf_counter() - timer,
        outputs=(
            _output("metadata.json", "application/json", metadata_hash),
            _output("artifacts/index.json", "application/json", index_hash),
        ),
    )
    _write_stage(writer, report_record)
    writer.append_event(
        severity="info",
        stage="report",
        component=f"{phase}-evaluate-table",
        event_code="evaluation.completed",
        details={"run_id": run_id, "terminal_status": summary["terminal_status"]},
    )
    try:
        return writer.finalize()
    except BundleError as exc:
        raise TableEvaluationError(f"{phase.upper()} bundle finalization failed: {exc}") from exc
