"""Strict Python contracts for versioned evaluation wire records."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from standardized_tabular_diffusion.evaluation.serialization import (
    SerializationError,
    content_fingerprint,
    validate_bundle_relative_path,
)

ATOMIC_RESULT_SCHEMA_VERSION = "1.0.0"
EVALUATION_REQUEST_SCHEMA_VERSION = "1.0.0"
STAGE_RECORD_SCHEMA_VERSION = "1.0.0"

_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY = re.compile(r"(?:password|secret|token|api[_-]?key|credential)", re.IGNORECASE)


class ContractError(ValueError):
    """Raised when a Python evaluation record violates a scientific contract."""


class MetricState(StrEnum):
    COMPUTED = "computed"
    MATHEMATICALLY_UNDEFINED = "mathematically_undefined"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    NOT_APPLICABLE = "not_applicable"
    IMPLEMENTATION_FAILURE = "implementation_failure"
    RESOURCE_FAILURE = "resource_failure"


class RawDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    DISTRIBUTIONAL = "distributional"
    DESCRIPTIVE = "descriptive"


class MetricLifecycle(StrEnum):
    REGISTERED = "registered"
    DEFINITION_REVIEWED = "definition-reviewed"
    IMPLEMENTATION_COMPLETE = "implementation-complete"
    UNIT_VALIDATED = "unit-validated"
    SOURCE_PARITY_VALIDATED = "source-parity-validated"
    PROTOCOL_FROZEN = "protocol-frozen"
    RELEASE_SUPPORTED = "release-supported"


class DefinitionOrigin(StrEnum):
    SOURCE_DEFINED = "source-defined"
    SOURCE_PARAMETERIZED = "source-parameterized"
    BENCHMARK_DERIVED = "benchmark-derived"
    BENCHMARK_NATIVE = "benchmark-native"


class FinalizationStatus(StrEnum):
    INCOMPLETE = "incomplete"
    FINALIZED = "finalized"
    INVALIDATED = "invalidated"
    WITHDRAWN = "withdrawn"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase portable identifier, got {value!r}")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _require_reason_code(name: str, value: str) -> None:
    if not isinstance(value, str) or not _REASON_CODE.fullmatch(value):
        raise ContractError(f"{name} must be a stable lowercase reason code")


def _require_finite(name: str, value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
    ):
        raise ContractError(f"{name} must be a finite number or null")


def _require_exact_fields(record_name: str, payload: dict[str, Any], expected: set[str]) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise ContractError(f"Missing {record_name} fields: {sorted(missing)}")
    if unknown:
        raise ContractError(f"Unknown {record_name} fields: {sorted(unknown)}")


def _parse_utc_timestamp(name: str, value: str) -> datetime:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be an ISO 8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{name} must be an ISO 8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{name} must identify UTC")
    return parsed


def _reject_embedded_secrets(value: Any, location: str = "request") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)) and not str(key).lower().endswith(("_reference", "-reference")):
                raise ContractError(f"Embedded credential material is prohibited at {location}.{key}")
            _reject_embedded_secrets(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_embedded_secrets(item, f"{location}[{index}]")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class EvaluationRequest:
    subject_type: str
    sample_artifact: dict[str, Any]
    dataset_profile: dict[str, str]
    protocol: dict[str, str]
    metrics: tuple[dict[str, str], ...]
    comparison_track: str
    generation_seed: int
    evaluator_seeds: tuple[int, ...]
    reference_artifact: dict[str, Any] | None = None
    evaluator_profile: dict[str, str] | None = None
    hardware_profile: dict[str, str] | None = None
    model: dict[str, Any] | None = None
    resource_limits: dict[str, Any] = field(default_factory=dict)
    failure_policy: dict[str, Any] = field(default_factory=dict)
    request_schema_version: str = EVALUATION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.request_schema_version != EVALUATION_REQUEST_SCHEMA_VERSION:
            raise ContractError(f"Unsupported Evaluation Request schema: {self.request_schema_version}")
        if self.subject_type not in {"adapter-run", "external-synthetic-table"}:
            raise ContractError(f"Unsupported subject_type: {self.subject_type!r}")
        if self.comparison_track not in {"native", "standardized-tuning"}:
            raise ContractError(f"Unsupported comparison_track: {self.comparison_track!r}")
        if isinstance(self.generation_seed, bool) or not isinstance(self.generation_seed, int):
            raise ContractError("generation_seed must be an integer")
        if not isinstance(self.evaluator_seeds, tuple) or not self.evaluator_seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.evaluator_seeds
        ):
            raise ContractError("evaluator_seeds must contain at least one integer")
        if len(set(self.evaluator_seeds)) != len(self.evaluator_seeds):
            raise ContractError("evaluator_seeds must not contain duplicates")
        self._validate_artifact("sample_artifact", self.sample_artifact)
        if self.reference_artifact is not None:
            self._validate_artifact("reference_artifact", self.reference_artifact)
            if self.reference_artifact["artifact_id"] == self.sample_artifact["artifact_id"]:
                raise ContractError("reference and sample artifacts must use different artifact_id values")
        self._validate_identity_ref("dataset_profile", self.dataset_profile, "dataset_id", "dataset_profile_version")
        self._validate_identity_ref("protocol", self.protocol, "protocol_id", "protocol_version")
        if not isinstance(self.metrics, tuple) or not self.metrics:
            raise ContractError("metrics must contain at least one exact metric identity")
        seen: set[tuple[str, str]] = set()
        for metric in self.metrics:
            if not isinstance(metric, dict):
                raise ContractError("Each metric selection must be an object")
            if set(metric) != {"metric_id", "metric_version"}:
                raise ContractError("Each metric selection must contain only metric_id and metric_version")
            _require_identifier("metric_id", metric["metric_id"])
            _require_identifier("metric_version", metric["metric_version"])
            identity = (metric["metric_id"], metric["metric_version"])
            if identity in seen:
                raise ContractError(f"Duplicate metric selection: {identity[0]}@{identity[1]}")
            seen.add(identity)
        if self.model is not None:
            if not isinstance(self.model, dict) or not self.model:
                raise ContractError("model must be a non-empty object when present")
            model_id = self.model.get("model_id")
            if model_id is not None:
                _require_identifier("model.model_id", model_id)
        if self.subject_type == "adapter-run" and (self.model is None or "model_id" not in self.model):
            raise ContractError("adapter-run requests require model provenance with model_id")
        for name, reference in (
            ("evaluator_profile", self.evaluator_profile),
            ("hardware_profile", self.hardware_profile),
        ):
            if reference is not None:
                if not isinstance(reference, dict):
                    raise ContractError(f"{name} must be an object or null")
                if set(reference) != {"profile_id", "profile_version", "sha256"}:
                    raise ContractError(f"{name} must contain profile_id, profile_version, and sha256")
                _require_identifier(f"{name}.profile_id", reference["profile_id"])
                _require_identifier(f"{name}.profile_version", reference["profile_version"])
                _require_sha256(f"{name}.sha256", reference["sha256"])
        if not isinstance(self.resource_limits, dict) or not isinstance(self.failure_policy, dict):
            raise ContractError("resource_limits and failure_policy must be objects")
        payload = self.to_dict()
        _reject_embedded_secrets(payload)
        try:
            content_fingerprint(payload)
        except SerializationError as exc:
            raise ContractError(f"Evaluation Request is not canonically serializable: {exc}") from exc

    @staticmethod
    def _validate_artifact(name: str, artifact: dict[str, Any]) -> None:
        if not isinstance(artifact, dict):
            raise ContractError(f"{name} must be an object")
        required = {"artifact_id", "media_type", "sha256"}
        optional = {"uri", "row_count"}
        if not required.issubset(artifact) or not set(artifact).issubset(required | optional):
            raise ContractError(
                f"{name} requires artifact_id, media_type, sha256, and optional uri or row_count"
            )
        _require_identifier(f"{name}.artifact_id", artifact["artifact_id"])
        _require_sha256(f"{name}.sha256", artifact["sha256"])
        if not isinstance(artifact["media_type"], str) or "/" not in artifact["media_type"]:
            raise ContractError(f"{name}.media_type must be an Internet media type")
        uri = artifact.get("uri")
        if uri is not None and (not isinstance(uri, str) or not uri):
            raise ContractError(f"{name}.uri must be a non-empty string when present")
        if uri is not None:
            parsed = urlsplit(uri)
            if not parsed.scheme or parsed.scheme.lower() == "file" or parsed.username or parsed.password:
                raise ContractError(f"{name}.uri must be a credential-free, non-file URI")
        row_count = artifact.get("row_count")
        if row_count is not None and (
            isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0
        ):
            raise ContractError(f"{name}.row_count must be a non-negative integer when present")

    @staticmethod
    def _validate_identity_ref(name: str, reference: dict[str, str], id_key: str, version_key: str) -> None:
        if not isinstance(reference, dict):
            raise ContractError(f"{name} must be an object")
        if set(reference) != {id_key, version_key, "sha256"}:
            raise ContractError(f"{name} must contain {id_key}, {version_key}, and sha256")
        _require_identifier(f"{name}.{id_key}", reference[id_key])
        _require_identifier(f"{name}.{version_key}", reference[version_key])
        _require_sha256(f"{name}.sha256", reference["sha256"])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.reference_artifact is None:
            payload.pop("reference_artifact")
        payload["metrics"] = [dict(item) for item in self.metrics]
        payload["evaluator_seeds"] = list(self.evaluator_seeds)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationRequest:
        if not isinstance(payload, dict):
            raise ContractError("Evaluation Request must be an object")
        known = set(cls.__dataclass_fields__)
        missing_allowed = {"reference_artifact"}
        unknown = set(payload) - known
        missing = known - set(payload) - missing_allowed
        if missing:
            raise ContractError(f"Missing Evaluation Request fields: {sorted(missing)}")
        if unknown:
            raise ContractError(f"Unknown Evaluation Request fields: {sorted(unknown)}")
        converted = dict(payload)
        try:
            converted["metrics"] = tuple(dict(item) for item in payload["metrics"])
            converted["evaluator_seeds"] = tuple(payload["evaluator_seeds"])
        except (TypeError, ValueError) as exc:
            raise ContractError("Evaluation Request metrics and evaluator_seeds must be arrays") from exc
        return cls(**converted)

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        payload["metrics"] = sorted(
            payload["metrics"],
            key=lambda item: (item["metric_id"], item["metric_version"]),
        )
        payload["evaluator_seeds"] = sorted(payload["evaluator_seeds"])
        return content_fingerprint(payload)


@dataclass(frozen=True)
class AtomicResult:
    run_id: str
    protocol_version: str
    dataset_id: str
    dataset_version: str
    dataset_view: str
    split_id: str
    model_id: str
    comparison_track: str
    generation_seed: int
    metric_id: str
    metric_version: str
    dimension: str
    scope_type: str
    scope_id: str
    state: MetricState
    raw_direction: RawDirection
    weight: float
    n_reference: int
    n_synthetic: int
    n_valid: int
    n_excluded: int
    computed_at: str
    raw_value: float | None = None
    normalized_value: float | None = None
    aggregate_contribution: float | None = None
    reference_value: float | None = None
    unit: str | None = None
    evaluator_id: str | None = None
    evaluator_version: str | None = None
    task_type: str | None = None
    reason_code: str | None = None
    reason_detail: str | None = None
    warning_codes: tuple[str, ...] = ()
    artifact_ref: str | None = None
    result_schema_version: str = ATOMIC_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.result_schema_version != ATOMIC_RESULT_SCHEMA_VERSION:
            raise ContractError(f"Unsupported Atomic Result schema: {self.result_schema_version}")
        if not isinstance(self.state, MetricState):
            raise ContractError("state must be a MetricState value")
        if not isinstance(self.raw_direction, RawDirection):
            raise ContractError("raw_direction must be a RawDirection value")
        for name in (
            "run_id",
            "protocol_version",
            "dataset_id",
            "dataset_version",
            "dataset_view",
            "split_id",
            "model_id",
            "metric_id",
            "metric_version",
            "dimension",
            "scope_id",
        ):
            _require_identifier(name, getattr(self, name))
        if self.comparison_track not in {"native", "standardized-tuning"}:
            raise ContractError(f"Unsupported comparison_track: {self.comparison_track!r}")
        if self.scope_type not in {"column", "pair", "target", "evaluator", "attack", "group", "phase", "dataset"}:
            raise ContractError(f"Unsupported scope_type: {self.scope_type!r}")
        if self.task_type not in {None, "classification", "regression"}:
            raise ContractError(f"Unsupported task_type: {self.task_type!r}")
        if (self.evaluator_id is None) != (self.evaluator_version is None):
            raise ContractError("evaluator_id and evaluator_version must either both be set or both be null")
        if self.evaluator_id is not None:
            _require_identifier("evaluator_id", self.evaluator_id)
            _require_identifier("evaluator_version", self.evaluator_version or "")
        if isinstance(self.generation_seed, bool) or not isinstance(self.generation_seed, int):
            raise ContractError("generation_seed must be an integer")
        for name in ("weight", "raw_value", "normalized_value", "aggregate_contribution", "reference_value"):
            _require_finite(name, getattr(self, name))
        if self.weight < 0:
            raise ContractError("weight must be non-negative")
        for name in ("n_reference", "n_synthetic", "n_valid", "n_excluded"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        _parse_utc_timestamp("computed_at", self.computed_at)
        if not isinstance(self.warning_codes, tuple):
            raise ContractError("warning_codes must be a tuple of stable reason codes")
        for warning in self.warning_codes:
            _require_reason_code("warning code", warning)
        if len(set(self.warning_codes)) != len(self.warning_codes):
            raise ContractError("warning_codes must not contain duplicates")
        if self.artifact_ref is not None:
            validate_bundle_relative_path(self.artifact_ref)
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit.strip()):
            raise ContractError("unit must be a non-empty string or null")
        total_observations = self.n_reference + self.n_synthetic
        if self.n_valid + self.n_excluded > total_observations:
            raise ContractError("n_valid plus n_excluded cannot exceed total reference and synthetic observations")

        if self.state is MetricState.COMPUTED:
            if self.raw_value is None and self.artifact_ref is None:
                raise ContractError("computed state requires raw_value or a structured artifact_ref")
            if self.reason_code is not None or self.reason_detail is not None:
                raise ContractError("computed state cannot carry a failure reason")
            if self.n_valid == 0:
                raise ContractError("computed state requires at least one valid observation")
            if self.aggregate_contribution is not None and self.normalized_value is None:
                raise ContractError("aggregate_contribution requires a separately recorded normalized_value")
            if self.weight == 0 and self.aggregate_contribution not in {None, 0, 0.0}:
                raise ContractError("zero-weight results cannot have a non-zero aggregate contribution")
            if self.raw_direction is RawDirection.TARGET and self.reference_value is None:
                raise ContractError("target-direction results require a reference_value")
        else:
            if any(value is not None for value in (self.raw_value, self.normalized_value, self.aggregate_contribution)):
                raise ContractError("non-computed states require null raw, normalized, and aggregate values")
            if self.reason_code is None:
                raise ContractError("non-computed state requires a stable lowercase reason_code")
            _require_reason_code("reason_code", self.reason_code)
            if not isinstance(self.reason_detail, str) or not self.reason_detail.strip():
                raise ContractError("non-computed state requires a human-readable reason_detail")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warning_codes"] = list(self.warning_codes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AtomicResult:
        if not isinstance(payload, dict):
            raise ContractError("Atomic Result must be an object")
        known = set(cls.__dataclass_fields__)
        _require_exact_fields("Atomic Result", payload, known)
        converted = dict(payload)
        try:
            converted["state"] = MetricState(payload["state"])
            converted["raw_direction"] = RawDirection(payload["raw_direction"])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Atomic Result contains an unsupported enum value: {exc}") from exc
        try:
            converted["warning_codes"] = tuple(payload["warning_codes"])
        except TypeError as exc:
            raise ContractError("Atomic Result warning_codes must be an array") from exc
        return cls(**converted)


@dataclass(frozen=True)
class StageRecord:
    stage_name: str
    stage_version: str
    status: StageStatus
    dependency_stage_ids: tuple[str, ...]
    input_fingerprints: dict[str, str]
    resolved_action: str
    started_at: str | None
    ended_at: str | None
    elapsed_seconds: float | None
    process_exit_code: int | None
    log_refs: tuple[str, ...]
    outputs: tuple[dict[str, Any], ...]
    warning_codes: tuple[str, ...]
    failure_category: str | None
    failure_reason_code: str | None
    cache_decision: str
    retry_count: int
    resume_ancestry: tuple[str, ...]
    stage_schema_version: str = STAGE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.stage_schema_version != STAGE_RECORD_SCHEMA_VERSION:
            raise ContractError(f"Unsupported Stage Record schema: {self.stage_schema_version}")
        if not isinstance(self.status, StageStatus):
            raise ContractError("status must be a StageStatus value")
        for name in ("dependency_stage_ids", "log_refs", "outputs", "warning_codes", "resume_ancestry"):
            if not isinstance(getattr(self, name), tuple):
                raise ContractError(f"{name} must be a tuple")
        if not isinstance(self.input_fingerprints, dict):
            raise ContractError("input_fingerprints must be an object")
        _require_identifier("stage_name", self.stage_name)
        _require_identifier("stage_version", self.stage_version)
        for dependency in self.dependency_stage_ids:
            _require_identifier("dependency_stage_ids item", dependency)
        if len(set(self.dependency_stage_ids)) != len(self.dependency_stage_ids):
            raise ContractError("dependency_stage_ids must not contain duplicates")
        for input_name, fingerprint in self.input_fingerprints.items():
            _require_identifier("input_fingerprints key", input_name)
            _require_sha256(f"input_fingerprints.{input_name}", fingerprint)
        if not isinstance(self.resolved_action, str) or not self.resolved_action.strip():
            raise ContractError("resolved_action must be a non-empty string")
        if self.elapsed_seconds is not None:
            _require_finite("elapsed_seconds", self.elapsed_seconds)
            if self.elapsed_seconds < 0:
                raise ContractError("elapsed_seconds must be non-negative")
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ContractError("retry_count must be a non-negative integer")
        if self.process_exit_code is not None and (
            isinstance(self.process_exit_code, bool) or not isinstance(self.process_exit_code, int)
        ):
            raise ContractError("process_exit_code must be an integer or null")
        if self.cache_decision not in {"not-requested", "miss", "hit", "bypassed", "invalid"}:
            raise ContractError(f"Unsupported cache_decision: {self.cache_decision!r}")
        if self.failure_category is not None and (
            not isinstance(self.failure_category, str) or not self.failure_category.strip()
        ):
            raise ContractError("failure_category must be a non-empty string or null")
        if self.status is StageStatus.SKIPPED and not self.failure_reason_code:
            raise ContractError("A skipped stage requires a stable reason code")
        if self.failure_reason_code is not None:
            _require_reason_code("failure_reason_code", self.failure_reason_code)
        for warning in self.warning_codes:
            _require_reason_code("warning code", warning)
        if len(set(self.warning_codes)) != len(self.warning_codes):
            raise ContractError("warning_codes must not contain duplicates")
        for path in self.log_refs:
            validate_bundle_relative_path(path)
        if len(set(self.log_refs)) != len(self.log_refs):
            raise ContractError("log_refs must not contain duplicates")
        output_paths: set[str] = set()
        for output in self.outputs:
            if not isinstance(output, dict):
                raise ContractError("Each stage output must be an object")
            if set(output) != {"path", "media_type", "sha256"}:
                raise ContractError("Each stage output must contain only path, media_type, and sha256")
            path = validate_bundle_relative_path(output["path"])
            if path in output_paths:
                raise ContractError(f"Duplicate stage output path: {path}")
            output_paths.add(path)
            if not isinstance(output["media_type"], str) or "/" not in output["media_type"]:
                raise ContractError("Stage output media_type must be an Internet media type")
            _require_sha256(f"outputs.{path}.sha256", output["sha256"])
        for ancestor in self.resume_ancestry:
            _require_identifier("resume_ancestry item", ancestor)
        if len(set(self.resume_ancestry)) != len(self.resume_ancestry):
            raise ContractError("resume_ancestry must not contain duplicates")

        started = _parse_utc_timestamp("started_at", self.started_at) if self.started_at is not None else None
        ended = _parse_utc_timestamp("ended_at", self.ended_at) if self.ended_at is not None else None
        if started is not None and ended is not None and ended < started:
            raise ContractError("ended_at cannot precede started_at")

        if self.status is StageStatus.PENDING:
            if any(value is not None for value in (started, ended, self.elapsed_seconds, self.process_exit_code)):
                raise ContractError("pending stages cannot declare timing or process outcome")
        elif self.status is StageStatus.RUNNING:
            if started is None or any(value is not None for value in (ended, self.elapsed_seconds, self.process_exit_code)):
                raise ContractError("running stages require started_at and no terminal outcome")
        elif self.status is StageStatus.SUCCEEDED:
            if started is None or ended is None or self.elapsed_seconds is None:
                raise ContractError("succeeded stages require complete timing")
            if self.process_exit_code not in {None, 0}:
                raise ContractError("succeeded stages cannot carry a non-zero process exit code")
        elif self.status in {StageStatus.FAILED, StageStatus.CANCELLED, StageStatus.INVALIDATED}:
            if ended is None or self.failure_reason_code is None or self.failure_category is None:
                raise ContractError(f"{self.status.value} stages require end time, failure category, and reason code")

        if self.status in {StageStatus.PENDING, StageStatus.RUNNING, StageStatus.SUCCEEDED} and (
            self.failure_category is not None or self.failure_reason_code is not None
        ):
            raise ContractError(f"{self.status.value} stages cannot carry failure fields")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "dependency_stage_ids",
            "log_refs",
            "outputs",
            "warning_codes",
            "resume_ancestry",
        ):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StageRecord:
        if not isinstance(payload, dict):
            raise ContractError("Stage Record must be an object")
        known = set(cls.__dataclass_fields__)
        _require_exact_fields("Stage Record", payload, known)
        converted = dict(payload)
        try:
            converted["status"] = StageStatus(payload["status"])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Stage Record contains an unsupported status: {payload['status']!r}") from exc
        try:
            for name in ("dependency_stage_ids", "log_refs", "warning_codes", "resume_ancestry"):
                converted[name] = tuple(payload[name])
            converted["outputs"] = tuple(dict(item) for item in payload["outputs"])
        except (TypeError, ValueError) as exc:
            raise ContractError("Stage Record array fields must contain values of the declared shape") from exc
        return cls(**converted)
