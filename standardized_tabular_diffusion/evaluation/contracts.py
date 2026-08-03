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
    if not _SHA256.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _require_finite(name: str, value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
    ):
        raise ContractError(f"{name} must be a finite number or null")


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
        if not self.evaluator_seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.evaluator_seeds
        ):
            raise ContractError("evaluator_seeds must contain at least one integer")
        self._validate_artifact()
        self._validate_identity_ref("dataset_profile", self.dataset_profile, "dataset_id", "dataset_profile_version")
        self._validate_identity_ref("protocol", self.protocol, "protocol_id", "protocol_version")
        if not self.metrics:
            raise ContractError("metrics must contain at least one exact metric identity")
        seen: set[tuple[str, str]] = set()
        for metric in self.metrics:
            if set(metric) != {"metric_id", "metric_version"}:
                raise ContractError("Each metric selection must contain only metric_id and metric_version")
            _require_identifier("metric_id", metric["metric_id"])
            _require_identifier("metric_version", metric["metric_version"])
            identity = (metric["metric_id"], metric["metric_version"])
            if identity in seen:
                raise ContractError(f"Duplicate metric selection: {identity[0]}@{identity[1]}")
            seen.add(identity)
        for name, reference in (
            ("evaluator_profile", self.evaluator_profile),
            ("hardware_profile", self.hardware_profile),
        ):
            if reference is not None:
                if set(reference) != {"profile_id", "profile_version", "sha256"}:
                    raise ContractError(f"{name} must contain profile_id, profile_version, and sha256")
                _require_identifier(f"{name}.profile_id", reference["profile_id"])
                _require_identifier(f"{name}.profile_version", reference["profile_version"])
                _require_sha256(f"{name}.sha256", reference["sha256"])
        _reject_embedded_secrets(self.to_dict())

    def _validate_artifact(self) -> None:
        required = {"artifact_id", "media_type", "sha256"}
        if not required.issubset(self.sample_artifact) or not set(self.sample_artifact).issubset(required | {"uri"}):
            raise ContractError("sample_artifact requires artifact_id, media_type, sha256, and optional uri")
        _require_identifier("sample_artifact.artifact_id", str(self.sample_artifact["artifact_id"]))
        _require_sha256("sample_artifact.sha256", str(self.sample_artifact["sha256"]))
        if not isinstance(self.sample_artifact["media_type"], str) or "/" not in self.sample_artifact["media_type"]:
            raise ContractError("sample_artifact.media_type must be an Internet media type")
        uri = self.sample_artifact.get("uri")
        if uri is not None and (not isinstance(uri, str) or not uri):
            raise ContractError("sample_artifact.uri must be a non-empty string when present")
        if uri is not None:
            parsed = urlsplit(uri)
            if not parsed.scheme or parsed.scheme.lower() == "file" or parsed.username or parsed.password:
                raise ContractError("sample_artifact.uri must be a credential-free, non-file URI")

    @staticmethod
    def _validate_identity_ref(name: str, reference: dict[str, str], id_key: str, version_key: str) -> None:
        if set(reference) != {id_key, version_key, "sha256"}:
            raise ContractError(f"{name} must contain {id_key}, {version_key}, and sha256")
        _require_identifier(f"{name}.{id_key}", reference[id_key])
        _require_identifier(f"{name}.{version_key}", reference[version_key])
        _require_sha256(f"{name}.sha256", reference["sha256"])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = [dict(item) for item in self.metrics]
        payload["evaluator_seeds"] = list(self.evaluator_seeds)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationRequest:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(payload) - known
        if unknown:
            raise ContractError(f"Unknown Evaluation Request fields: {sorted(unknown)}")
        converted = dict(payload)
        converted["metrics"] = tuple(dict(item) for item in payload.get("metrics", ()))
        converted["evaluator_seeds"] = tuple(payload.get("evaluator_seeds", ()))
        return cls(**converted)

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())


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
        try:
            parsed = datetime.fromisoformat(self.computed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError("computed_at must be an ISO 8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ContractError("computed_at must identify UTC")
        for warning in self.warning_codes:
            if not _REASON_CODE.fullmatch(warning):
                raise ContractError(f"Invalid warning code: {warning!r}")
        if self.artifact_ref is not None:
            validate_bundle_relative_path(self.artifact_ref)

        if self.state is MetricState.COMPUTED:
            if self.raw_value is None and self.artifact_ref is None:
                raise ContractError("computed state requires raw_value or a structured artifact_ref")
            if self.reason_code is not None or self.reason_detail is not None:
                raise ContractError("computed state cannot carry a failure reason")
        else:
            if any(value is not None for value in (self.raw_value, self.normalized_value, self.aggregate_contribution)):
                raise ContractError("non-computed states require null raw, normalized, and aggregate values")
            if self.reason_code is None or not _REASON_CODE.fullmatch(self.reason_code):
                raise ContractError("non-computed state requires a stable lowercase reason_code")
            if self.reason_detail is None or not self.reason_detail.strip():
                raise ContractError("non-computed state requires a human-readable reason_detail")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warning_codes"] = list(self.warning_codes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AtomicResult:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(payload) - known
        if unknown:
            raise ContractError(f"Unknown Atomic Result fields: {sorted(unknown)}")
        converted = dict(payload)
        converted["state"] = MetricState(payload["state"])
        converted["raw_direction"] = RawDirection(payload["raw_direction"])
        converted["warning_codes"] = tuple(payload.get("warning_codes", ()))
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
        _require_identifier("stage_name", self.stage_name)
        _require_identifier("stage_version", self.stage_version)
        if self.elapsed_seconds is not None:
            _require_finite("elapsed_seconds", self.elapsed_seconds)
            if self.elapsed_seconds < 0:
                raise ContractError("elapsed_seconds must be non-negative")
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ContractError("retry_count must be a non-negative integer")
        if self.status is StageStatus.SKIPPED and not self.failure_reason_code:
            raise ContractError("A skipped stage requires a stable reason code")
        if self.failure_reason_code is not None and not _REASON_CODE.fullmatch(self.failure_reason_code):
            raise ContractError("Invalid failure_reason_code")
        for path in self.log_refs:
            validate_bundle_relative_path(path)
        for output in self.outputs:
            if "path" in output:
                validate_bundle_relative_path(str(output["path"]))

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
