from __future__ import annotations

import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    ContractError,
    EvaluationRequest,
    MetricState,
    RawDirection,
    StageRecord,
    StageStatus,
)
from standardized_tabular_diffusion.evaluation.schema import (
    SchemaValidationError,
    list_schemas,
    load_schema,
    validate_instance,
)
from standardized_tabular_diffusion.evaluation.serialization import (
    SerializationError,
    canonical_json_bytes,
    read_json,
    read_yaml_safe,
    validate_bundle_relative_path,
)

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

SHA256 = "0" * 64


def make_request(**changes: object) -> EvaluationRequest:
    values: dict[str, object] = {
        "subject_type": "external-synthetic-table",
        "sample_artifact": {"artifact_id": "sample", "media_type": "text/csv", "sha256": SHA256},
        "dataset_profile": {"dataset_id": "fixture", "dataset_profile_version": "1.0.0", "sha256": SHA256},
        "protocol": {"protocol_id": "development-p1", "protocol_version": "0.1.0", "sha256": SHA256},
        "metrics": ({"metric_id": "fixture-shape", "metric_version": "1.0.0"},),
        "comparison_track": "native",
        "generation_seed": 42,
        "evaluator_seeds": (7, 11),
    }
    values.update(changes)
    return EvaluationRequest(**values)  # type: ignore[arg-type]


def make_atomic_result(**changes: object) -> AtomicResult:
    values: dict[str, object] = {
        "run_id": "run-fixture",
        "protocol_version": "1.0.0",
        "dataset_id": "fixture",
        "dataset_version": "1.0.0",
        "dataset_view": "canonical-v1",
        "split_id": "split-v1",
        "model_id": "model-a",
        "comparison_track": "native",
        "generation_seed": 42,
        "metric_id": "fixture-shape",
        "metric_version": "1.0.0",
        "dimension": "fidelity",
        "scope_type": "column",
        "scope_id": "age",
        "state": MetricState.COMPUTED,
        "raw_direction": RawDirection.MAXIMIZE,
        "weight": 1.0,
        "n_reference": 20,
        "n_synthetic": 20,
        "n_valid": 20,
        "n_excluded": 0,
        "computed_at": "2026-08-03T12:00:00Z",
        "raw_value": 0.75,
    }
    values.update(changes)
    return AtomicResult(**values)  # type: ignore[arg-type]


def test_all_packaged_schemas_are_valid_draft_2020_12() -> None:
    assert set(list_schemas()) == {
        "artifact-index",
        "atomic-result",
        "dataset-profile",
        "evaluation-request",
        "manifest",
        "metadata",
        "metric-registry-entry",
        "protocol-profile",
        "stage-record",
        "summary",
    }
    for name in list_schemas():
        Draft202012Validator.check_schema(load_schema(name))


def test_request_round_trip_and_fingerprint_are_deterministic() -> None:
    request = make_request(resource_limits={"memory_gib": 8, "timeout_seconds": 60})
    reordered = EvaluationRequest.from_dict(
        {
            "failure_policy": {},
            "resource_limits": {"timeout_seconds": 60, "memory_gib": 8},
            **{
                key: value
                for key, value in request.to_dict().items()
                if key not in {"failure_policy", "resource_limits"}
            },
        }
    )

    validate_instance("evaluation-request", request.to_dict())
    assert EvaluationRequest.from_dict(request.to_dict()) == request
    assert reordered.fingerprint == request.fingerprint
    assert make_request(generation_seed=43).fingerprint != request.fingerprint


def test_request_fingerprint_normalizes_set_like_metric_and_seed_order() -> None:
    first = make_request(
        metrics=(
            {"metric_id": "fixture-trend", "metric_version": "1.0.0"},
            {"metric_id": "fixture-shape", "metric_version": "1.0.0"},
        ),
        evaluator_seeds=(11, 7),
    )
    second = make_request(
        metrics=tuple(reversed(first.metrics)),
        evaluator_seeds=tuple(reversed(first.evaluator_seeds)),
    )
    assert first.to_dict() != second.to_dict()
    assert first.fingerprint == second.fingerprint


def test_request_rejects_duplicate_seeds_missing_wire_fields_and_unbound_adapter_run() -> None:
    with pytest.raises(ContractError, match="duplicates"):
        make_request(evaluator_seeds=(7, 7))
    with pytest.raises(ContractError, match="model provenance"):
        make_request(subject_type="adapter-run")
    with pytest.raises(ContractError, match="Missing"):
        EvaluationRequest.from_dict({key: value for key, value in make_request().to_dict().items() if key != "model"})


def test_request_accepts_audited_row_count_and_rejects_nonfinite_nested_policy() -> None:
    request = make_request(
        sample_artifact={
            "artifact_id": "sample",
            "media_type": "text/csv",
            "sha256": SHA256,
            "row_count": 0,
        }
    )
    validate_instance("evaluation-request", request.to_dict())
    with pytest.raises(ContractError, match="canonically serializable"):
        make_request(resource_limits={"timeout_seconds": math.inf})
    invalid_artifact = {"artifact_id": 7, "media_type": "text/csv", "sha256": SHA256}
    with pytest.raises(ContractError, match="portable identifier"):
        make_request(sample_artifact=invalid_artifact)


def test_request_unknown_field_and_version_fail_closed() -> None:
    payload = make_request().to_dict()
    payload["surprise"] = True
    with pytest.raises(ContractError, match="Unknown"):
        EvaluationRequest.from_dict(payload)
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_instance("evaluation-request", payload)
    with pytest.raises(ContractError, match="Unsupported"):
        make_request(request_schema_version="2.0.0")


def test_request_rejects_embedded_credentials_and_local_file_uris() -> None:
    with pytest.raises(ContractError, match="credential"):
        make_request(model={"api_token": "must-not-be-serialized"})
    with pytest.raises(ContractError, match="non-file URI"):
        make_request(
            sample_artifact={
                "artifact_id": "sample",
                "media_type": "text/csv",
                "sha256": SHA256,
                "uri": "file:///tmp/sample.csv",
            }
        )


@pytest.mark.parametrize(
    "state",
    [
        MetricState.MATHEMATICALLY_UNDEFINED,
        MetricState.INSUFFICIENT_SUPPORT,
        MetricState.NOT_APPLICABLE,
        MetricState.IMPLEMENTATION_FAILURE,
        MetricState.RESOURCE_FAILURE,
    ],
)
def test_every_noncomputed_state_requires_null_values_and_reason(state: MetricState) -> None:
    result = make_atomic_result(
        state=state,
        raw_value=None,
        normalized_value=None,
        aggregate_contribution=None,
        reason_code="fixture_reason",
        reason_detail="Fixture intentionally did not compute a number.",
    )
    validate_instance("atomic-result", result.to_dict())


def test_noncomputed_target_result_does_not_require_an_unavailable_reference() -> None:
    result = make_atomic_result(
        state=MetricState.RESOURCE_FAILURE,
        raw_direction=RawDirection.TARGET,
        raw_value=None,
        reason_code="resource_unavailable",
        reason_detail="The evaluator did not start, so no reference was observed.",
    )
    validate_instance("atomic-result", result.to_dict())


def test_atomic_result_rejects_nonfinite_and_state_value_conflation() -> None:
    for nonfinite in (math.nan, math.inf, -math.inf):
        with pytest.raises(ContractError, match="finite"):
            make_atomic_result(raw_value=nonfinite)
        with pytest.raises(SerializationError, match="Non-finite"):
            canonical_json_bytes({"value": nonfinite})

    with pytest.raises(ContractError, match="null raw"):
        make_atomic_result(
            state=MetricState.NOT_APPLICABLE,
            raw_value=0.0,
            reason_code="wrong_domain",
            reason_detail="Not applicable.",
        )
    with pytest.raises(ContractError, match="raw_value"):
        make_atomic_result(raw_value=None)


@pytest.mark.parametrize(
    "path",
    ["../escape.json", "stages/../../escape", "/absolute.json", "C:/absolute.json", r"logs\events.jsonl", "./x"],
)
def test_bundle_paths_reject_traversal_and_nonportable_forms(path: str) -> None:
    with pytest.raises(SerializationError):
        validate_bundle_relative_path(path)


def test_atomic_result_round_trip_preserves_raw_and_derived_fields() -> None:
    result = make_atomic_result(normalized_value=0.8, aggregate_contribution=0.4, reference_value=1.0)
    restored = AtomicResult.from_dict(result.to_dict())
    validate_instance("atomic-result", restored.to_dict())
    assert restored == result
    assert restored.raw_value == 0.75
    assert restored.normalized_value == 0.8
    assert restored.aggregate_contribution == 0.4


def test_atomic_result_enforces_support_direction_and_aggregation_invariants() -> None:
    with pytest.raises(ContractError, match="valid observation"):
        make_atomic_result(n_valid=0)
    with pytest.raises(ContractError, match="cannot exceed"):
        make_atomic_result(n_reference=1, n_synthetic=1, n_valid=2, n_excluded=1)
    with pytest.raises(ContractError, match="normalized_value"):
        make_atomic_result(aggregate_contribution=0.5)
    with pytest.raises(ContractError, match="reference_value"):
        make_atomic_result(raw_direction=RawDirection.TARGET)
    with pytest.raises(ContractError, match="both be set"):
        make_atomic_result(evaluator_id="fixture-evaluator")
    payload = make_atomic_result().to_dict()
    payload["warning_codes"] = [7]
    with pytest.raises(ContractError, match="reason code"):
        AtomicResult.from_dict(payload)


def test_skipped_stage_requires_a_stable_reason() -> None:
    values = {
        "stage_name": "evaluate",
        "stage_version": "1.0.0",
        "status": StageStatus.SKIPPED,
        "dependency_stage_ids": (),
        "input_fingerprints": {},
        "resolved_action": "skip",
        "started_at": None,
        "ended_at": None,
        "elapsed_seconds": None,
        "process_exit_code": None,
        "log_refs": (),
        "outputs": (),
        "warning_codes": (),
        "failure_category": None,
        "failure_reason_code": None,
        "cache_decision": "not-requested",
        "retry_count": 0,
        "resume_ancestry": (),
    }
    with pytest.raises(ContractError, match="skipped"):
        StageRecord(**values)  # type: ignore[arg-type]
    values["failure_reason_code"] = "not_requested"
    record = StageRecord(**values)  # type: ignore[arg-type]
    validate_instance("stage-record", record.to_dict())


def test_stage_record_round_trip_and_terminal_state_invariants() -> None:
    record = StageRecord(
        stage_name="evaluate",
        stage_version="1.0.0",
        status=StageStatus.SUCCEEDED,
        dependency_stage_ids=("validate",),
        input_fingerprints={"request": SHA256},
        resolved_action="evaluate diagnostic fixture",
        started_at="2026-08-03T12:00:00Z",
        ended_at="2026-08-03T12:00:01Z",
        elapsed_seconds=1.0,
        process_exit_code=0,
        log_refs=("logs/events.jsonl",),
        outputs=({"path": "artifacts/result.json", "media_type": "application/json", "sha256": SHA256},),
        warning_codes=(),
        failure_category=None,
        failure_reason_code=None,
        cache_decision="miss",
        retry_count=0,
        resume_ancestry=(),
    )
    restored = StageRecord.from_dict(record.to_dict())
    assert restored == record
    validate_instance("stage-record", restored.to_dict())

    with pytest.raises(ContractError, match="complete timing"):
        StageRecord.from_dict({**record.to_dict(), "ended_at": None})
    with pytest.raises(ContractError, match="lowercase SHA-256"):
        StageRecord.from_dict({**record.to_dict(), "input_fingerprints": {"request": "bad"}})
    with pytest.raises(ContractError, match="input_fingerprints"):
        StageRecord.from_dict({**record.to_dict(), "input_fingerprints": []})
    invalid_output = [{"path": 7, "media_type": "application/json", "sha256": SHA256}]
    with pytest.raises(SerializationError, match="path"):
        StageRecord.from_dict({**record.to_dict(), "outputs": invalid_output})


def test_structured_loaders_reject_duplicate_and_unicode_equivalent_keys(tmp_path: Path) -> None:
    duplicate_json = tmp_path / "duplicate.json"
    duplicate_json.write_text('{"metric": 1, "metric": 2}\n', encoding="utf-8")
    with pytest.raises(SerializationError, match="Duplicate JSON"):
        read_json(duplicate_json)

    unicode_json = tmp_path / "unicode-duplicate.json"
    unicode_json.write_text('{"é": 1, "é": 2}\n', encoding="utf-8")
    with pytest.raises(SerializationError, match="Unicode normalization"):
        read_json(unicode_json)

    duplicate_yaml = tmp_path / "duplicate.yaml"
    duplicate_yaml.write_text("metric: 1\nmetric: 2\n", encoding="utf-8")
    with pytest.raises(SerializationError, match="Duplicate YAML"):
        read_yaml_safe(duplicate_yaml)
