"""Auditable incomplete Run Result bundles with atomic publication semantics."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest, utc_timestamp
from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    read_yaml_safe,
    sha256_file,
    validate_bundle_relative_path,
)

BUNDLE_SCHEMA_VERSION = "1.0.0"

_INVENTORY_TEMPLATE: tuple[tuple[str, bool, str, str, str | None], ...] = (
    ("manifest.json", True, "application/json", "present", None),
    ("metadata.json", True, "application/json", "pending", "bundle_initializing"),
    ("config.yaml", True, "application/yaml", "pending", "bundle_initializing"),
    ("environment.json", True, "application/json", "pending", "bundle_initializing"),
    ("metrics.parquet", True, "application/vnd.apache.parquet", "pending", "evaluation_not_run"),
    ("summary.json", True, "application/json", "pending", "bundle_initializing"),
    ("checksums.sha256", True, "text/plain", "pending", "bundle_not_finalized"),
    ("logs/events.jsonl", True, "application/x-ndjson", "pending", "bundle_initializing"),
    ("logs/stdout.log", False, "text/plain", "not-applicable", "no_external_process"),
    ("logs/stderr.log", False, "text/plain", "not-applicable", "no_external_process"),
    ("stages/prepare.json", True, "application/json", "pending", "stage_not_run"),
    ("stages/train.json", False, "application/json", "pending", "stage_not_resolved"),
    ("stages/sample.json", False, "application/json", "pending", "stage_not_resolved"),
    ("stages/validate.json", True, "application/json", "pending", "stage_not_run"),
    ("stages/evaluate.json", True, "application/json", "pending", "stage_not_run"),
    ("stages/aggregate.json", True, "application/json", "pending", "stage_not_run"),
    ("stages/report.json", True, "application/json", "pending", "stage_not_run"),
    ("artifacts/index.json", True, "application/json", "pending", "bundle_initializing"),
)


class BundleError(ValueError):
    """Raised when bundle creation or cross-file validation fails."""


@dataclass(frozen=True)
class BundleValidationReport:
    root: Path
    bundle_id: str
    finalization_status: str
    present_files: int
    pending_files: int
    not_applicable_files: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "bundle_id": self.bundle_id,
            "finalization_status": self.finalization_status,
            "present_files": self.present_files,
            "pending_files": self.pending_files,
            "not_applicable_files": self.not_applicable_files,
        }


def _initial_inventory() -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "required": required,
            "status": status,
            "media_type": media_type,
            "sha256": None,
            "reason_code": reason,
        }
        for path, required, media_type, status, reason in _INVENTORY_TEMPLATE
    ]


def _redact(value: Any) -> Any:
    secret = re.compile(r"(?:password|secret|token|api[_-]?key|credential)", re.IGNORECASE)
    if isinstance(value, dict):
        return {key: "<redacted>" if secret.search(str(key)) else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class IncompleteRunBundleWriter:
    """Create and update only incomplete P1 bundles.

    Finalization belongs to the P2 vertical slice. This writer first publishes a
    valid incomplete manifest, so a later interruption cannot leave a bundle
    marked as finalized.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, relative: str) -> Path:
        validate_bundle_relative_path(relative)
        destination = self.root.joinpath(*relative.split("/"))
        root_resolved = self.root.resolve()
        if (
            destination.resolve(strict=False) != root_resolved
            and root_resolved not in destination.resolve(strict=False).parents
        ):
            raise BundleError(f"Bundle path escapes root: {relative}")
        current = self.root
        for part in relative.split("/")[:-1]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise BundleError(f"Symlinked bundle directory is prohibited: {current}")
        return destination

    def create(
        self,
        request: EvaluationRequest,
        *,
        environment: dict[str, Any],
        producer: dict[str, Any] | None = None,
    ) -> BundleValidationReport:
        validate_instance("evaluation-request", request.to_dict())
        if self.root.exists():
            if self.root.is_symlink() or not self.root.is_dir():
                raise BundleError(f"Bundle root must be a new regular directory: {self.root}")
            if any(self.root.iterdir()):
                raise BundleError(f"Refusing to overwrite non-empty bundle root: {self.root}")
        else:
            self.root.mkdir(parents=True)
        for directory in ("logs", "stages", "artifacts"):
            self._path(directory + "/.placeholder").parent.mkdir(exist_ok=True)

        fingerprint = request.fingerprint
        run_id = f"run-{fingerprint[:24]}"
        created_at = utc_timestamp()
        producer_record = producer or {
            "repository": "https://github.com/jimmybach/Standardized-Tabular-Diffusion",
            "commit": "unknown",
            "dirty": True,
        }
        if set(producer_record) != {"repository", "commit", "dirty"}:
            raise BundleError("producer must contain exactly repository, commit, and dirty")
        inventory = _initial_inventory()
        manifest = {
            "manifest_schema_version": "1.0.0",
            "bundle_id": run_id,
            "bundle_type": "run-result",
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": created_at,
            "finalized_at": None,
            "finalization_status": "incomplete",
            "identity": {"request_fingerprint": fingerprint, "run_id": run_id},
            "files": inventory,
            "external_artifacts": [copy.deepcopy(request.sample_artifact)],
            "supersedes": [],
            "invalidates": [],
            "producer": copy.deepcopy(producer_record),
            "checksum_algorithm": "sha256",
        }
        validate_instance("manifest", manifest)
        atomic_write_json(self._path("manifest.json"), manifest)

        metadata = _build_metadata(request, run_id, created_at)
        summary = _build_summary(run_id, fingerprint)
        artifact_index = {"artifact_index_schema_version": "1.0.0", "artifacts": []}
        validate_instance("metadata", metadata)
        validate_instance("summary", summary)
        validate_instance("artifact-index", artifact_index)

        atomic_write_json(self._path("metadata.json"), metadata)
        atomic_write_json(self._path("config.yaml"), request.to_dict())
        atomic_write_json(self._path("environment.json"), environment)
        atomic_write_json(self._path("summary.json"), summary)
        atomic_write_json(self._path("artifacts/index.json"), artifact_index)
        self.append_event(
            severity="info",
            stage="bundle",
            component="incomplete-writer",
            event_code="bundle.created",
            details={"run_id": run_id, "request_fingerprint": fingerprint},
        )

        present = {
            "metadata.json",
            "config.yaml",
            "environment.json",
            "summary.json",
            "logs/events.jsonl",
            "artifacts/index.json",
        }
        for item in inventory:
            if item["path"] in present:
                item["status"] = "present"
                item["reason_code"] = None
                item["sha256"] = sha256_file(self._path(item["path"]))
        validate_instance("manifest", manifest)
        atomic_write_json(self._path("manifest.json"), manifest)
        return validate_result_bundle(self.root)

    def append_event(
        self,
        *,
        severity: str,
        stage: str,
        component: str,
        event_code: str,
        details: dict[str, Any],
    ) -> None:
        if severity not in {"debug", "info", "warning", "error", "critical"}:
            raise BundleError(f"Unsupported event severity: {severity}")
        event = {
            "event_schema_version": "1.0.0",
            "timestamp": utc_timestamp(),
            "severity": severity,
            "stage": stage,
            "component": component,
            "event_code": event_code,
            "details": _redact(details),
        }
        payload = canonical_json_bytes(event) + b"\n"
        path = self._path("logs/events.jsonl")
        if path.exists() and path.is_symlink():
            raise BundleError(f"Refusing to append to symlink: {path}")
        manifest_path = self._path("manifest.json")
        manifest: dict[str, Any] | None = None
        if manifest_path.is_file():
            loaded_manifest = read_json(manifest_path)
            if not isinstance(loaded_manifest, dict):
                raise BundleError("manifest.json must be an object")
            if loaded_manifest.get("finalization_status") != "incomplete":
                raise BundleError("Events cannot be appended after bundle finalization")
            manifest = loaded_manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if manifest is not None:
            for item in manifest.get("files", []):
                if item.get("path") == "logs/events.jsonl" and item.get("status") == "present":
                    item["sha256"] = sha256_file(path)
                    validate_instance("manifest", manifest)
                    atomic_write_json(manifest_path, manifest)
                    break


def _build_metadata(request: EvaluationRequest, run_id: str, created_at: str) -> dict[str, Any]:
    requested_rows = request.sample_artifact.get("row_count")
    return {
        "metadata_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": request.fingerprint},
        "protocol": copy.deepcopy(request.protocol),
        "dataset": copy.deepcopy(request.dataset_profile),
        "model": copy.deepcopy(request.model or {"subject_type": request.subject_type, "model_id": "external"}),
        "implementation": {"evaluation_subsystem": "p1-contracts-only", "metrics_executed": False},
        "comparison_track": request.comparison_track,
        "seeds": {"generation": request.generation_seed, "evaluators": list(request.evaluator_seeds)},
        "evaluator": {
            "profile": copy.deepcopy(request.evaluator_profile),
            "hardware_profile": copy.deepcopy(request.hardware_profile),
        },
        "execution": {
            "requested_action": "evaluate",
            "started_at": created_at,
            "ended_at": None,
            "terminal_phase": None,
            "run_status": "pending",
            "requested_synthetic_rows": requested_rows,
            "actual_synthetic_rows": None,
            "resource_limits": copy.deepcopy(request.resource_limits),
            "interrupted": False,
            "resume_ancestry": [],
            "warning_codes": [],
            "failure_category": None,
            "failure_reason_code": None,
            "artifact_refs": [],
        },
        "coverage": {"requested_metrics": [copy.deepcopy(metric) for metric in request.metrics], "computed": 0},
        "provenance": {"sample_artifact": copy.deepcopy(request.sample_artifact)},
        "review": {"status": "not-reviewed"},
        "status": "incomplete",
    }


def _build_summary(run_id: str, fingerprint: str) -> dict[str, Any]:
    return {
        "summary_schema_version": "1.0.0",
        "identity": {"run_id": run_id, "request_fingerprint": fingerprint},
        "terminal_status": "pending",
        "validity": {},
        "dimensions": {},
        "local_utility": {},
        "global_utility": {},
        "privacy_risk": {},
        "efficiency": {},
        "metric_state_counts": {},
        "denominator_counts": {},
        "warnings": [],
        "failures": [],
        "atomic_result_refs": [],
        "aggregation": {"implementation": "not-run", "version": "1.0.0", "reproducible_from_atomic_results": True},
        "dataset_aggregation_eligible": False,
    }


def _load_json_object(path: Path, schema_name: str) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise BundleError(f"{path.name} must be a JSON object")
    validate_instance(schema_name, payload)
    return payload


def validate_result_bundle(root: str | Path) -> BundleValidationReport:
    bundle_root = Path(root)
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        raise BundleError(f"Result bundle must be a regular directory: {bundle_root}")
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BundleError("Result bundle is missing a regular manifest.json")
    manifest = _load_json_object(manifest_path, "manifest")

    seen: set[str] = set()
    inventory_status: dict[str, str] = {}
    counts = {"present": 0, "pending": 0, "not-applicable": 0}
    for item in manifest["files"]:
        relative = validate_bundle_relative_path(item["path"])
        if relative in seen:
            raise BundleError(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        inventory_status[relative] = item["status"]
        counts[item["status"]] += 1
        path = bundle_root.joinpath(*relative.split("/"))
        if item["status"] == "present":
            if not path.is_file() or path.is_symlink():
                raise BundleError(f"Manifest marks missing or unsafe file as present: {relative}")
            if item["sha256"] is not None and sha256_file(path) != item["sha256"]:
                raise BundleError(f"Checksum mismatch for {relative}")
        elif item["status"] == "not-applicable" and not item["reason_code"]:
            raise BundleError(f"Not-applicable inventory item requires reason_code: {relative}")
        if item["required"] and item["status"] == "not-applicable":
            raise BundleError(f"Required inventory item cannot be not-applicable: {relative}")

    required_inventory = {path for path, *_ in _INVENTORY_TEMPLATE}
    if not required_inventory.issubset(seen):
        raise BundleError(f"Manifest inventory is incomplete: {sorted(required_inventory - seen)}")

    config_path = bundle_root / "config.yaml"
    metadata_path = bundle_root / "metadata.json"
    summary_path = bundle_root / "summary.json"
    artifact_index_path = bundle_root / "artifacts" / "index.json"
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    if inventory_status.get("config.yaml") == "present":
        config = read_yaml_safe(config_path)
        if not isinstance(config, dict):
            raise BundleError("config.yaml must contain an Evaluation Request object")
        validate_instance("evaluation-request", config)
    if inventory_status.get("metadata.json") == "present":
        metadata = _load_json_object(metadata_path, "metadata")
    if inventory_status.get("summary.json") == "present":
        summary = _load_json_object(summary_path, "summary")
    if inventory_status.get("artifacts/index.json") == "present":
        _load_json_object(artifact_index_path, "artifact-index")
    for stage_name in ("prepare", "train", "sample", "validate", "evaluate", "aggregate", "report"):
        relative = f"stages/{stage_name}.json"
        if inventory_status.get(relative) == "present":
            _load_json_object(bundle_root / "stages" / f"{stage_name}.json", "stage-record")
    if inventory_status.get("logs/events.jsonl") == "present":
        _validate_event_log(bundle_root / "logs" / "events.jsonl")

    if config is not None and metadata is not None and summary is not None:
        request = EvaluationRequest.from_dict(config)
        expected_identity = manifest["identity"]
        if request.fingerprint != expected_identity["request_fingerprint"]:
            raise BundleError("Evaluation Request fingerprint does not match manifest")
        if metadata["identity"] != expected_identity:
            raise BundleError("metadata identity does not match manifest")
        if summary["identity"] != expected_identity:
            raise BundleError("summary identity does not match manifest")
        if metadata["comparison_track"] != request.comparison_track:
            raise BundleError("metadata comparison_track does not match config")

    if manifest["finalization_status"] == "finalized":
        if counts["pending"]:
            raise BundleError("A finalized bundle cannot contain pending inventory items")
        if not (bundle_root / "checksums.sha256").is_file():
            raise BundleError("A finalized bundle requires checksums.sha256")
        _validate_final_checksums(bundle_root)
    elif manifest["finalized_at"] is not None:
        raise BundleError("A non-finalized bundle cannot declare finalized_at")

    return BundleValidationReport(
        root=bundle_root.resolve(),
        bundle_id=manifest["bundle_id"],
        finalization_status=manifest["finalization_status"],
        present_files=counts["present"],
        pending_files=counts["pending"],
        not_applicable_files=counts["not-applicable"],
    )


def _validate_event_log(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BundleError(f"Cannot read events.jsonl: {exc}") from exc
    required = {"event_schema_version", "timestamp", "severity", "stage", "component", "event_code", "details"}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise BundleError(f"events.jsonl contains an empty record at line {line_number}")
        try:
            event = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise BundleError(f"events.jsonl line {line_number} is invalid JSON") from exc
        if not isinstance(event, dict) or set(event) != required:
            raise BundleError(f"events.jsonl line {line_number} has an invalid event contract")
        if event["event_schema_version"] != "1.0.0" or event["severity"] not in {
            "debug",
            "info",
            "warning",
            "error",
            "critical",
        }:
            raise BundleError(f"events.jsonl line {line_number} has an invalid version or severity")
        canonical_json_bytes(event)


def _validate_final_checksums(bundle_root: Path) -> None:
    checksum_path = bundle_root / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BundleError(f"Cannot read checksums.sha256: {exc}") from exc
    parsed: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise BundleError(f"Malformed checksums.sha256 line {line_number}")
        digest, relative = match.groups()
        validate_bundle_relative_path(relative)
        if relative == "checksums.sha256" or relative in parsed:
            raise BundleError(f"Invalid or duplicate checksum path: {relative}")
        parsed[relative] = digest
    if list(parsed) != sorted(parsed):
        raise BundleError("checksums.sha256 paths must use deterministic lexical order")
    regular_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != checksum_path
    }
    if set(parsed) != regular_files:
        raise BundleError("checksums.sha256 does not cover every finalized regular file exactly once")
    for relative, expected in parsed.items():
        path = bundle_root.joinpath(*relative.split("/"))
        if path.is_symlink() or sha256_file(path) != expected:
            raise BundleError(f"Final checksum mismatch for {relative}")
