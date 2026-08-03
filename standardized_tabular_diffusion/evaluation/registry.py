"""Data-driven Metric Registry loading and lifecycle enforcement."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from standardized_tabular_diffusion.evaluation.contracts import MetricLifecycle
from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, read_json

REGISTRY_RESOURCE_PACKAGE = "standardized_tabular_diffusion.resources.evaluation.metrics"

_LIFECYCLE_EVIDENCE = {
    MetricLifecycle.DEFINITION_REVIEWED: "definition_review",
    MetricLifecycle.IMPLEMENTATION_COMPLETE: "implementation",
    MetricLifecycle.UNIT_VALIDATED: "unit_validation",
    MetricLifecycle.SOURCE_PARITY_VALIDATED: "source_parity",
    MetricLifecycle.PROTOCOL_FROZEN: "protocol_freeze",
    MetricLifecycle.RELEASE_SUPPORTED: "release_approval",
}

_LIFECYCLE_ORDER: tuple[MetricLifecycle, ...] = (
    MetricLifecycle.REGISTERED,
    MetricLifecycle.DEFINITION_REVIEWED,
    MetricLifecycle.IMPLEMENTATION_COMPLETE,
    MetricLifecycle.UNIT_VALIDATED,
    MetricLifecycle.SOURCE_PARITY_VALIDATED,
    MetricLifecycle.PROTOCOL_FROZEN,
    MetricLifecycle.RELEASE_SUPPORTED,
)


class MetricRegistryError(ValueError):
    """Raised when registry data is invalid, ambiguous, or overclaims evidence."""


@dataclass(frozen=True)
class MetricRecord:
    payload: dict[str, Any]
    source: str

    @property
    def metric_id(self) -> str:
        return str(self.payload["metric_id"])

    @property
    def metric_version(self) -> str:
        return str(self.payload["metric_version"])

    @property
    def identity(self) -> tuple[str, str]:
        return self.metric_id, self.metric_version

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_lifecycle(record: dict[str, Any]) -> None:
    try:
        status = MetricLifecycle(record["lifecycle_status"])
    except ValueError as exc:
        raise MetricRegistryError(f"Unknown lifecycle status: {record.get('lifecycle_status')!r}") from exc
    evidence = record["lifecycle_evidence"]
    for stage in _LIFECYCLE_ORDER[1 : _LIFECYCLE_ORDER.index(status) + 1]:
        evidence_name = _LIFECYCLE_EVIDENCE[stage]
        if not isinstance(evidence.get(evidence_name), dict) or not evidence[evidence_name]:
            raise MetricRegistryError(
                f"{record['metric_id']}@{record['metric_version']} cannot claim {status.value}: "
                f"missing non-empty {evidence_name} evidence required by {stage.value}"
            )

    if status in {
        MetricLifecycle.SOURCE_PARITY_VALIDATED,
        MetricLifecycle.PROTOCOL_FROZEN,
        MetricLifecycle.RELEASE_SUPPORTED,
    }:
        if not record["validation"]["source_parity_evidence"]:
            raise MetricRegistryError(
                f"{record['metric_id']} claims source parity without a validation evidence reference"
            )

    official_allowed = record["admission"]["official_results_allowed"]
    if official_allowed and status not in {MetricLifecycle.PROTOCOL_FROZEN, MetricLifecycle.RELEASE_SUPPORTED}:
        raise MetricRegistryError("Only protocol-frozen or release-supported metrics may affect Official Results")
    if official_allowed and record["planned_leaderboard_role"] != "official-component":
        raise MetricRegistryError("An officially admitted metric must have the official-component role")
    if status is MetricLifecycle.RELEASE_SUPPORTED and record["validation"]["release_decision"] != "approved":
        raise MetricRegistryError("A release-supported metric requires an approved validation release decision")


def _validate_semantics(record: dict[str, Any]) -> None:
    semantics = record["semantics"]
    if semantics["raw_direction"] == "target" and semantics["target_value"] is None:
        raise MetricRegistryError("A target-direction metric must declare target_value")
    if semantics["raw_direction"] != "target" and semantics["target_value"] is not None:
        raise MetricRegistryError("target_value is only valid for target-direction metrics")
    raw_range = semantics["raw_range"]
    if raw_range is not None:
        lower, upper = raw_range["minimum"], raw_range["maximum"]
        if lower is not None and upper is not None and lower > upper:
            raise MetricRegistryError("raw_range.minimum cannot exceed raw_range.maximum")


def validate_metric_record(record: dict[str, Any], *, source: str = "<memory>") -> MetricRecord:
    validate_instance("metric-registry-entry", record)
    _validate_lifecycle(record)
    _validate_semantics(record)
    return MetricRecord(payload=copy.deepcopy(record), source=source)


def _records_from_document(document: Any, source: str) -> list[MetricRecord]:
    if not isinstance(document, dict):
        raise MetricRegistryError(f"Metric Registry document {source} must be an object")
    if "records" not in document:
        return [validate_metric_record(document, source=source)]
    allowed = {"registry_collection_schema_version", "common", "records"}
    unknown = set(document) - allowed
    if unknown:
        raise MetricRegistryError(f"Unknown collection fields in {source}: {sorted(unknown)}")
    if document.get("registry_collection_schema_version") != "1.0.0":
        raise MetricRegistryError(f"Unsupported registry collection schema in {source}")
    common = document.get("common", {})
    records = document["records"]
    if not isinstance(common, dict) or not isinstance(records, list):
        raise MetricRegistryError(f"Invalid collection structure in {source}")
    return [validate_metric_record(_deep_merge(common, record), source=source) for record in records]


def _packaged_documents() -> Iterable[tuple[str, Any]]:
    for item in sorted(resources.files(REGISTRY_RESOURCE_PACKAGE).iterdir(), key=lambda candidate: candidate.name):
        if item.name.endswith(".json"):
            with resources.as_file(item) as path:
                yield f"package:{item.name}", read_json(path)


def _directory_documents(directory: Path) -> Iterable[tuple[str, Any]]:
    if not directory.is_dir():
        raise MetricRegistryError(f"Metric Registry directory does not exist: {directory}")
    for path in sorted(directory.glob("*.json")):
        yield str(path), read_json(path)


def load_metric_registry(directory: str | Path | None = None) -> tuple[MetricRecord, ...]:
    documents = _packaged_documents() if directory is None else _directory_documents(Path(directory))
    indexed: dict[tuple[str, str], MetricRecord] = {}
    for source, document in documents:
        for record in _records_from_document(document, source):
            if record.identity in indexed:
                previous = indexed[record.identity]
                raise MetricRegistryError(
                    f"Duplicate metric identity {record.metric_id}@{record.metric_version} in {previous.source} and {source}"
                )
            indexed[record.identity] = record
    return tuple(indexed[key] for key in sorted(indexed))


def get_metric_record(metric_id: str, metric_version: str, directory: str | Path | None = None) -> MetricRecord:
    for record in load_metric_registry(directory):
        if record.identity == (metric_id, metric_version):
            return record
    raise KeyError(f"Unknown metric identity: {metric_id}@{metric_version}")
