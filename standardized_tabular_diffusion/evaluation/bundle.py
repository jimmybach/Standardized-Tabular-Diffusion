"""Auditable incomplete Run Result bundles with atomic publication semantics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    ContractError,
    EvaluationRequest,
    utc_timestamp,
)
from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    SerializationError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    parse_json_text,
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
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _request_external_artifacts(request: EvaluationRequest) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if request.reference_artifact is not None:
        artifacts.append(copy.deepcopy(request.reference_artifact))
    if request.real_test_artifact is not None:
        artifacts.append(copy.deepcopy(request.real_test_artifact))
    artifacts.append(copy.deepcopy(request.sample_artifact))
    return artifacts


class IncompleteRunBundleWriter:
    """Transactionally build a Run Result and atomically publish final status."""

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
        run_id = f"run-{uuid.uuid4().hex}"
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
            "external_artifacts": _request_external_artifacts(request),
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
        atomic_write_json(self._path("environment.json"), _redact(environment))
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

    def _load_incomplete_manifest(self) -> dict[str, Any]:
        manifest = read_json(self._path("manifest.json"))
        if not isinstance(manifest, dict) or manifest.get("finalization_status") != "incomplete":
            raise BundleError("Bundle mutation requires a valid incomplete manifest")
        return manifest

    @staticmethod
    def _inventory_item(manifest: dict[str, Any], relative: str) -> dict[str, Any] | None:
        return next((item for item in manifest["files"] if item["path"] == relative), None)

    def write_bytes(
        self,
        relative: str,
        payload: bytes,
        *,
        media_type: str,
        required: bool = True,
    ) -> str:
        """Publish one inventoried file without exposing a false present state."""

        if relative in {"manifest.json", "checksums.sha256"}:
            raise BundleError(f"Reserved bundle file requires dedicated handling: {relative}")
        destination = self._path(relative)
        manifest = self._load_incomplete_manifest()
        item = self._inventory_item(manifest, relative)
        if item is None:
            item = {
                "path": relative,
                "required": required,
                "status": "pending",
                "media_type": media_type,
                "sha256": None,
                "reason_code": "file_write_in_progress",
            }
            manifest["files"].append(item)
        else:
            if item["required"] != required or item["media_type"] != media_type:
                raise BundleError(f"Inventory contract differs for {relative}")
            item.update(status="pending", sha256=None, reason_code="file_write_in_progress")
        validate_instance("manifest", manifest)
        atomic_write_json(self._path("manifest.json"), manifest)
        atomic_write_bytes(destination, payload)
        item.update(status="present", sha256=sha256_file(destination), reason_code=None)
        validate_instance("manifest", manifest)
        atomic_write_json(self._path("manifest.json"), manifest)
        return item["sha256"]

    def write_json(
        self,
        relative: str,
        payload: Any,
        *,
        media_type: str = "application/json",
        required: bool = True,
        schema_name: str | None = None,
    ) -> str:
        if schema_name is not None:
            validate_instance(schema_name, payload)
        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )
        return self.write_bytes(relative, serialized, media_type=media_type, required=required)

    def mark_not_applicable(self, relative: str, *, reason_code: str) -> None:
        manifest = self._load_incomplete_manifest()
        item = self._inventory_item(manifest, relative)
        if item is None:
            raise BundleError(f"Cannot resolve an absent inventory path: {relative}")
        if item["required"]:
            raise BundleError(f"Required bundle file cannot be not-applicable: {relative}")
        path = self._path(relative)
        if path.exists():
            raise BundleError(f"Cannot mark an existing file not-applicable: {relative}")
        item.update(status="not-applicable", sha256=None, reason_code=reason_code)
        validate_instance("manifest", manifest)
        atomic_write_json(self._path("manifest.json"), manifest)

    def finalize(self) -> BundleValidationReport:
        """Commit final checksums and the final manifest as the last atomic marker."""

        validate_result_bundle(self.root)
        manifest = self._load_incomplete_manifest()
        unresolved = [
            item["path"]
            for item in manifest["files"]
            if item["path"] not in {"checksums.sha256"} and item["status"] == "pending"
        ]
        if unresolved:
            raise BundleError(f"Cannot finalize with unresolved inventory items: {sorted(unresolved)}")
        checksum_item = self._inventory_item(manifest, "checksums.sha256")
        if checksum_item is None:
            raise BundleError("Manifest does not inventory checksums.sha256")
        checksum_item.update(status="present", sha256=None, reason_code=None)
        manifest_item = self._inventory_item(manifest, "manifest.json")
        if manifest_item is None:
            raise BundleError("Manifest does not inventory itself")
        manifest_item.update(status="present", sha256=None, reason_code=None)
        manifest["finalized_at"] = utc_timestamp()
        manifest["finalization_status"] = "finalized"
        validate_instance("manifest", manifest)
        final_manifest = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n"
        )

        regular_files = {
            path.relative_to(self.root).as_posix(): path
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink() and path.name != "checksums.sha256"
        }
        if set(regular_files) != {item["path"] for item in manifest["files"] if item["status"] == "present"} - {
            "checksums.sha256"
        }:
            raise BundleError("Final inventory does not exactly match regular bundle files")
        checksums: list[str] = []
        for relative, path in sorted(regular_files.items()):
            digest = hashlib.sha256(final_manifest).hexdigest() if relative == "manifest.json" else sha256_file(path)
            checksums.append(f"{digest}  {relative}")
        atomic_write_bytes(self._path("checksums.sha256"), ("\n".join(checksums) + "\n").encode("utf-8"))
        atomic_write_bytes(self._path("manifest.json"), final_manifest)
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
        if manifest is not None:
            for item in manifest.get("files", []):
                if item.get("path") == "logs/events.jsonl":
                    item["status"] = "pending"
                    item["sha256"] = None
                    item["reason_code"] = "event_log_updating"
                    validate_instance("manifest", manifest)
                    atomic_write_json(manifest_path, manifest)
                    break
            else:
                raise BundleError("manifest.json does not inventory logs/events.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_bytes() if path.exists() else b""
        atomic_write_bytes(path, existing + payload)
        if manifest is not None:
            for item in manifest["files"]:
                if item["path"] == "logs/events.jsonl":
                    item["status"] = "present"
                    item["sha256"] = sha256_file(path)
                    item["reason_code"] = None
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
        "provenance": {
            "reference_artifact": copy.deepcopy(request.reference_artifact),
            "real_test_artifact": copy.deepcopy(request.real_test_artifact),
            "sample_artifact": copy.deepcopy(request.sample_artifact),
        },
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
    if manifest["bundle_id"] != manifest["identity"]["run_id"]:
        raise BundleError("manifest bundle_id does not match identity.run_id")

    actual_files: set[str] = set()
    for path in bundle_root.rglob("*"):
        relative = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            raise BundleError(f"Symlinks are prohibited inside result bundles: {relative}")
        if path.is_file():
            actual_files.add(relative)

    seen: set[str] = set()
    inventory_status: dict[str, str] = {}
    inventory_checksum: dict[str, str | None] = {}
    inventory_media_type: dict[str, str] = {}
    counts = {"present": 0, "pending": 0, "not-applicable": 0}
    for item in manifest["files"]:
        relative = validate_bundle_relative_path(item["path"])
        if relative in seen:
            raise BundleError(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        inventory_status[relative] = item["status"]
        inventory_checksum[relative] = item["sha256"]
        inventory_media_type[relative] = item["media_type"]
        counts[item["status"]] += 1
        path = bundle_root.joinpath(*relative.split("/"))
        if item["status"] == "present":
            if not path.is_file() or path.is_symlink():
                raise BundleError(f"Manifest marks missing or unsafe file as present: {relative}")
            if relative not in {"manifest.json", "checksums.sha256"} and item["sha256"] is None:
                raise BundleError(f"Present inventory item requires a checksum: {relative}")
            if item["reason_code"] is not None:
                raise BundleError(f"Present inventory item cannot carry reason_code: {relative}")
            if item["sha256"] is not None and sha256_file(path) != item["sha256"]:
                raise BundleError(f"Checksum mismatch for {relative}")
        else:
            if item["sha256"] is not None:
                raise BundleError(f"Non-present inventory item cannot carry a checksum: {relative}")
            if not item["reason_code"]:
                raise BundleError(f"{item['status']} inventory item requires reason_code: {relative}")
        if item["required"] and item["status"] == "not-applicable":
            raise BundleError(f"Required inventory item cannot be not-applicable: {relative}")

    required_inventory = {path for path, *_ in _INVENTORY_TEMPLATE}
    if not required_inventory.issubset(seen):
        raise BundleError(f"Manifest inventory is incomplete: {sorted(required_inventory - seen)}")
    unexpected_files = actual_files - seen
    if unexpected_files:
        raise BundleError(f"Result bundle contains files absent from the manifest: {sorted(unexpected_files)}")

    config_path = bundle_root / "config.yaml"
    metadata_path = bundle_root / "metadata.json"
    summary_path = bundle_root / "summary.json"
    artifact_index_path = bundle_root / "artifacts" / "index.json"
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    artifact_index: dict[str, Any] | None = None
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
        artifact_index = _load_json_object(artifact_index_path, "artifact-index")
    if inventory_status.get("environment.json") == "present":
        environment = read_json(bundle_root / "environment.json")
        if not isinstance(environment, dict):
            raise BundleError("environment.json must be a JSON object")
    stage_records: list[dict[str, Any]] = []
    for stage_name in ("prepare", "train", "sample", "validate", "evaluate", "aggregate", "report"):
        relative = f"stages/{stage_name}.json"
        if inventory_status.get(relative) == "present":
            stage = _load_json_object(bundle_root / "stages" / f"{stage_name}.json", "stage-record")
            if stage["stage_name"] != stage_name:
                raise BundleError(f"Stage record name does not match path: {relative}")
            stage_records.append(stage)
    for stage in stage_records:
        for output in stage["outputs"]:
            relative = output["path"]
            if inventory_status.get(relative) != "present":
                raise BundleError(f"Stage output is not a present inventory item: {relative}")
            if inventory_checksum.get(relative) != output["sha256"]:
                raise BundleError(f"Stage output checksum differs from manifest: {relative}")

    if artifact_index is not None:
        indexed_ids: set[str] = set()
        stages_by_name = {stage["stage_name"]: stage for stage in stage_records}
        external_by_id: dict[str, dict[str, Any]] = {}
        if config is not None:
            request_for_artifacts = EvaluationRequest.from_dict(config)
            external_by_id = {
                artifact["artifact_id"]: artifact for artifact in _request_external_artifacts(request_for_artifacts)
            }
        for artifact in artifact_index["artifacts"]:
            artifact_id = artifact["artifact_id"]
            if artifact_id in indexed_ids:
                raise BundleError(f"Duplicate artifact index identity: {artifact_id}")
            indexed_ids.add(artifact_id)
            relative = artifact["path"]
            if relative is None:
                request_artifact = external_by_id.get(artifact_id)
                if request_artifact is None or any(
                    artifact[field] != request_artifact[field] for field in ("media_type", "sha256")
                ):
                    raise BundleError(f"External artifact index entry differs from request: {artifact_id}")
                if artifact["external_uri"] != f"urn:sha256:{artifact['sha256']}":
                    raise BundleError(f"External artifact URI is not content-addressed: {artifact_id}")
                continue
            if inventory_status.get(relative) != "present":
                raise BundleError(f"Indexed local artifact is not present in the manifest: {relative}")
            path = bundle_root.joinpath(*relative.split("/"))
            if artifact["sha256"] != inventory_checksum.get(relative):
                raise BundleError(f"Artifact index checksum differs from manifest: {relative}")
            if artifact["media_type"] != inventory_media_type.get(relative):
                raise BundleError(f"Artifact index media type differs from manifest: {relative}")
            if artifact["byte_size"] != path.stat().st_size:
                raise BundleError(f"Artifact index byte_size differs from file: {relative}")
            producer_stage = artifact["producer_stage"]
            if producer_stage is not None:
                producer_record = stages_by_name.get(producer_stage)
                if producer_record is None or relative not in {output["path"] for output in producer_record["outputs"]}:
                    raise BundleError(f"Artifact producer stage does not declare its output: {relative}")
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
        if metadata["protocol"] != request.protocol:
            raise BundleError("metadata protocol does not match config")
        if metadata["dataset"] != request.dataset_profile:
            raise BundleError("metadata dataset does not match config")
        if metadata["seeds"] != {
            "generation": request.generation_seed,
            "evaluators": list(request.evaluator_seeds),
        }:
            raise BundleError("metadata seeds do not match config")
        if metadata["coverage"].get("requested_metrics") != list(request.metrics):
            raise BundleError("metadata requested metrics do not match config")
        if metadata["provenance"].get("reference_artifact") != request.reference_artifact:
            raise BundleError("metadata reference provenance does not match config")
        if metadata["provenance"].get("real_test_artifact") != request.real_test_artifact:
            raise BundleError("metadata real-test provenance does not match config")
        if metadata["provenance"].get("sample_artifact") != request.sample_artifact:
            raise BundleError("metadata sample provenance does not match config")
        if manifest["external_artifacts"] != _request_external_artifacts(request):
            raise BundleError("manifest external artifact does not match config")

    if manifest["finalization_status"] == "finalized":
        if counts["pending"]:
            raise BundleError("A finalized bundle cannot contain pending inventory items")
        if not (bundle_root / "checksums.sha256").is_file():
            raise BundleError("A finalized bundle requires checksums.sha256")
        if metadata is None or metadata["status"] != "finalized" or metadata["execution"]["ended_at"] is None:
            raise BundleError("A finalized bundle requires terminal finalized metadata")
        if summary is None or summary["terminal_status"] == "pending":
            raise BundleError("A finalized bundle requires a terminal summary")
        _validate_parquet_magic(bundle_root / "metrics.parquet")
        if config is None or metadata is None:
            raise BundleError("A finalized bundle requires config and metadata")
        _validate_final_atomic_results(
            bundle_root / "metrics.parquet",
            manifest=manifest,
            request=EvaluationRequest.from_dict(config),
            metadata=metadata,
            summary=summary,
        )
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
            event = parse_json_text(line, source=f"events.jsonl line {line_number}")
        except SerializationError as exc:
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
        if any(
            not isinstance(event[field], str) or not event[field].strip()
            for field in ("timestamp", "stage", "component", "event_code")
        ) or not isinstance(event["details"], dict):
            raise BundleError(f"events.jsonl line {line_number} has invalid field types")
        try:
            timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise BundleError(f"events.jsonl line {line_number} has an invalid timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise BundleError(f"events.jsonl line {line_number} timestamp must identify UTC")
        canonical_json_bytes(event)


def _validate_parquet_magic(path: Path) -> None:
    try:
        if path.stat().st_size < 8:
            raise BundleError("metrics.parquet is too short to be a Parquet file")
        with path.open("rb") as stream:
            prefix = stream.read(4)
            stream.seek(-4, 2)
            suffix = stream.read(4)
    except OSError as exc:
        raise BundleError(f"Cannot inspect metrics.parquet: {exc}") from exc
    if prefix != b"PAR1" or suffix != b"PAR1":
        raise BundleError("metrics.parquet is not a structurally valid Parquet file")


def _validate_final_atomic_results(
    path: Path,
    *,
    manifest: dict[str, Any],
    request: EvaluationRequest,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise BundleError("Finalized bundle validation requires pyarrow") from exc
    try:
        records = parquet.read_table(path).to_pylist()
        results = [AtomicResult.from_dict(record) for record in records]
    except (OSError, ValueError, TypeError, ContractError) as exc:
        raise BundleError(f"metrics.parquet contains invalid Atomic Results: {exc}") from exc
    if not results:
        raise BundleError("A finalized metrics.parquet must contain at least one Atomic Result")

    requested = {(item["metric_id"], item["metric_version"]) for item in request.metrics}
    model_id = (request.model or {}).get("model_id", "external")
    seen_scopes: set[tuple[str, str, str]] = set()
    for result in results:
        if (result.metric_id, result.metric_version) not in requested:
            raise BundleError(f"Atomic Result uses an unrequested metric: {result.metric_id}")
        if (
            result.run_id != manifest["bundle_id"]
            or result.protocol_version != request.protocol["protocol_version"]
            or result.dataset_id != request.dataset_profile["dataset_id"]
            or result.model_id != model_id
            or result.comparison_track != request.comparison_track
            or result.generation_seed != request.generation_seed
        ):
            raise BundleError("Atomic Result scientific identity differs from the bundle request")
        scope = (result.metric_id, result.scope_type, result.scope_id)
        if scope in seen_scopes:
            raise BundleError(f"Duplicate Atomic Result scope: {scope}")
        seen_scopes.add(scope)

    states = dict(sorted(Counter(result.state.value for result in results).items()))
    warnings = sorted({warning for result in results for warning in result.warning_codes})
    if summary["metric_state_counts"] != states:
        raise BundleError("Summary metric_state_counts do not match metrics.parquet")
    if summary["warnings"] != warnings:
        raise BundleError("Summary warnings do not match metrics.parquet")
    if summary["atomic_result_refs"] != [f"metrics.parquet#row={index}" for index in range(len(results))]:
        raise BundleError("Summary Atomic Result references do not cover metrics.parquet exactly")
    if metadata["coverage"].get("states") != states:
        raise BundleError("Metadata state coverage does not match metrics.parquet")
    if metadata["coverage"].get("computed") != states.get("computed", 0):
        raise BundleError("Metadata computed coverage does not match metrics.parquet")

    p2_summary_keys = {
        "sdmetrics-column-shapes": "column_shapes",
        "sdmetrics-column-pair-trends": "column_pair_trends",
    }
    fidelity = summary["dimensions"].get("fidelity", {})
    if requested == {(metric_id, "1.0.0") for metric_id in p2_summary_keys}:
        for metric_id, summary_key in p2_summary_keys.items():
            contributions = [
                result.aggregate_contribution
                for result in results
                if result.metric_id == metric_id and result.aggregate_contribution is not None
            ]
            reconstructed = sum(contributions) if contributions else None
            reported = fidelity.get(summary_key)
            if reconstructed is None:
                if reported is not None:
                    raise BundleError(f"Summary {summary_key} must be null without contributions")
            elif not isinstance(reported, (int, float)) or not math.isclose(
                reconstructed,
                float(reported),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise BundleError(f"Summary {summary_key} is not reproducible from Atomic Results")
        if fidelity.get("combined_score") is not None:
            raise BundleError("P2 finalized bundles must not contain a combined Fidelity score")

    p3_metric_ids = {"std-tabular-column-validity", "std-tabular-constraint-validity"}
    if requested == {(metric_id, "1.0.0") for metric_id in p3_metric_ids}:
        validity = summary["validity"]
        if summary["dimensions"].get("validity") != validity:
            raise BundleError("P3 dimensions.validity must exactly match the canonical validity summary")
        p3_column_results = [result for result in results if result.metric_id == "std-tabular-column-validity"]
        p3_constraint_results = [result for result in results if result.metric_id == "std-tabular-constraint-validity"]
        expected_column_weight = 1.0 / len(p3_column_results) if p3_column_results else None
        if expected_column_weight is None or any(
            result.state.value != "computed"
            or result.scope_type != "column"
            or result.dimension != "validity"
            or result.raw_value is None
            or not math.isclose(result.weight, expected_column_weight, rel_tol=1e-12, abs_tol=1e-12)
            or result.n_valid != result.n_synthetic
            or result.n_excluded != 0
            for result in p3_column_results
        ):
            raise BundleError("P3 column Atomic Results do not implement the equal-column contract")
        computed_constraint_results = [result for result in p3_constraint_results if result.state.value == "computed"]
        expected_constraint_weight = 1.0 / len(computed_constraint_results) if computed_constraint_results else 0.0
        if any(
            result.scope_type != "dataset"
            or result.dimension != "validity"
            or (
                result.state.value == "computed"
                and (
                    result.raw_value is None
                    or not math.isclose(
                        result.weight,
                        expected_constraint_weight,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            )
            or (result.state.value == "not_applicable" and result.weight != 0.0)
            or result.state.value not in {"computed", "not_applicable"}
            for result in p3_constraint_results
        ):
            raise BundleError("P3 constraint Atomic Results do not implement the equal-constraint contract")
        column_contributions = [
            result.aggregate_contribution for result in p3_column_results if result.aggregate_contribution is not None
        ]
        if not column_contributions:
            raise BundleError("P3 finalized bundles require computed per-column validity contributions")
        column_score = sum(column_contributions)
        constraint_contributions = [
            result.aggregate_contribution
            for result in p3_constraint_results
            if result.aggregate_contribution is not None
        ]
        constraint_score = sum(constraint_contributions) if constraint_contributions else None
        validity_score = 0.5 * column_score + 0.5 * constraint_score if constraint_score is not None else column_score
        expected_scores = {
            "column_validity_score": column_score,
            "constraint_validity_score": constraint_score,
            "validity_score": validity_score,
        }
        for key, expected in expected_scores.items():
            reported = validity.get(key)
            if expected is None:
                if reported is not None:
                    raise BundleError(f"P3 summary {key} must be null without computed contributions")
            elif not isinstance(reported, (int, float)) or not math.isclose(
                expected,
                float(reported),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise BundleError(f"P3 summary {key} is not reproducible from Atomic Results")
        if validity.get("synthetic_repair_applied") is not False:
            raise BundleError("P3 validity must be computed on the unrepaired decoded synthetic output")

        details_path = path.parent / "artifacts" / "validity-details.json"
        try:
            details = read_json(details_path)
        except (OSError, ValueError, SerializationError) as exc:
            raise BundleError(f"Cannot validate P3 validity details: {exc}") from exc
        if not isinstance(details, dict):
            raise BundleError("P3 validity details must be a JSON object")
        if (
            details.get("input_view") != "original-decoded-synthetic-output"
            or details.get("input_mutated") is not False
            or details.get("synthetic_repair_applied") is not False
        ):
            raise BundleError("P3 validity details do not prove evaluation of the immutable original output")
        detail_scores = details.get("property_scores")
        if not isinstance(detail_scores, dict):
            raise BundleError("P3 validity details lack property_scores")
        for key, expected in expected_scores.items():
            observed = detail_scores.get(key)
            if expected is None:
                if observed is not None:
                    raise BundleError(f"P3 detail {key} must be null")
            elif not isinstance(observed, (int, float)) or not math.isclose(
                expected, float(observed), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise BundleError(f"P3 detail {key} differs from Atomic Results")

        column_results = {result.scope_id: result for result in p3_column_results}
        column_details = details.get("columns")
        if not isinstance(column_details, list) or len(column_details) != len(column_results):
            raise BundleError("P3 detail columns do not cover Atomic Results exactly")
        seen_column_details: set[str] = set()
        for detail in column_details:
            if not isinstance(detail, dict) or not isinstance(detail.get("column_id"), str):
                raise BundleError("P3 column detail is malformed")
            column_id = detail["column_id"]
            if column_id in seen_column_details or column_id not in column_results:
                raise BundleError("P3 column detail scope is duplicate or unknown")
            seen_column_details.add(column_id)
            result = column_results[column_id]
            observed = detail.get("valid_cell_rate")
            if (
                result.raw_value is None
                or not isinstance(observed, (int, float))
                or not math.isclose(result.raw_value, float(observed), rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise BundleError("P3 column detail rate differs from its Atomic Result")
            valid_cells, invalid_cells = detail.get("valid_cells"), detail.get("invalid_cells")
            if (
                isinstance(valid_cells, bool)
                or not isinstance(valid_cells, int)
                or isinstance(invalid_cells, bool)
                or not isinstance(invalid_cells, int)
                or valid_cells < 0
                or invalid_cells < 0
                or valid_cells + invalid_cells != result.n_synthetic
            ):
                raise BundleError("P3 column detail cell counts are inconsistent")
        if seen_column_details != set(column_results):
            raise BundleError("P3 detail columns omit Atomic Result scopes")

        constraint_results = {
            result.scope_id: result for result in p3_constraint_results if result.scope_id != "no-reviewed-constraints"
        }
        constraint_details = details.get("cross_column_constraints")
        if not isinstance(constraint_details, list) or len(constraint_details) != len(constraint_results):
            raise BundleError("P3 constraint details do not cover reviewed constraint Atomic Results")
        all_constraint_scopes = {result.scope_id for result in p3_constraint_results}
        if (constraint_results and "no-reviewed-constraints" in all_constraint_scopes) or (
            not constraint_results and all_constraint_scopes != {"no-reviewed-constraints"}
        ):
            raise BundleError("P3 no-reviewed-constraints sentinel does not match the reviewed constraint set")
        seen_constraint_details: set[str] = set()
        for detail in constraint_details:
            if not isinstance(detail, dict) or not isinstance(detail.get("constraint_id"), str):
                raise BundleError("P3 constraint detail is malformed")
            constraint_id = detail["constraint_id"]
            if constraint_id in seen_constraint_details or constraint_id not in constraint_results:
                raise BundleError("P3 constraint detail scope is duplicate or unknown")
            seen_constraint_details.add(constraint_id)
            result = constraint_results[constraint_id]
            observed = detail.get("satisfaction_rate")
            if result.state.value == "computed":
                if (
                    result.raw_value is None
                    or not isinstance(observed, (int, float))
                    or not math.isclose(result.raw_value, float(observed), rel_tol=1e-12, abs_tol=1e-12)
                ):
                    raise BundleError("P3 constraint detail rate differs from its Atomic Result")
            elif observed is not None:
                raise BundleError("A non-computed P3 constraint detail must have a null rate")
            applicable = detail.get("applicable_rows")
            satisfied = detail.get("satisfied_rows")
            violating = detail.get("violating_rows")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (applicable, satisfied, violating)
            ):
                raise BundleError("P3 constraint detail row counts are inconsistent")
            applicable_rows = cast(int, applicable)
            satisfied_rows = cast(int, satisfied)
            violating_rows = cast(int, violating)
            if (
                satisfied_rows + violating_rows != applicable_rows
                or applicable_rows != result.n_valid
                or result.n_valid + result.n_excluded != result.n_synthetic
            ):
                raise BundleError("P3 constraint detail row counts are inconsistent")
        if seen_constraint_details != set(constraint_results):
            raise BundleError("P3 constraint details omit Atomic Result scopes")

        fully_valid_rows = details.get("fully_valid_rows")
        fully_valid_rate = details.get("fully_valid_row_rate")
        reported_rate = validity.get("fully_valid_row_rate")
        synthetic_rows = details.get("rows")
        if (
            isinstance(fully_valid_rows, bool)
            or not isinstance(fully_valid_rows, int)
            or isinstance(synthetic_rows, bool)
            or not isinstance(synthetic_rows, int)
            or synthetic_rows <= 0
            or not 0 <= fully_valid_rows <= synthetic_rows
            or not isinstance(fully_valid_rate, (int, float))
            or not isinstance(reported_rate, (int, float))
            or not math.isclose(fully_valid_rows / synthetic_rows, float(fully_valid_rate), abs_tol=1e-12)
            or not math.isclose(float(fully_valid_rate), float(reported_rate), abs_tol=1e-12)
        ):
            raise BundleError("P3 fully-valid-row evidence is inconsistent")
        synthetic_row_count = synthetic_rows
        fully_valid_row_count = fully_valid_rows
        expected_denominators = {
            "requested_columns": len(p3_column_results),
            "evaluated_columns": len(p3_column_results),
            "reviewed_cross_column_constraints": len(constraint_results),
            "computed_cross_column_constraints": len(computed_constraint_results),
            "synthetic_rows": synthetic_row_count,
            "fully_valid_rows": fully_valid_row_count,
        }
        if (
            summary.get("denominator_counts") != expected_denominators
            or metadata.get("coverage", {}).get("denominators") != expected_denominators
            or any(result.n_synthetic != synthetic_row_count for result in results)
        ):
            raise BundleError("P3 denominator evidence is not reproducible from Atomic Results and details")

    from standardized_tabular_diffusion.evaluation.utility import (
        GLOBAL_TARGET_RATIO_METRIC_ID,
        LOCAL_METRIC_IDS,
        LOCAL_RETENTION_METRIC_ID,
        P4_METRICS,
        global_target_ratio,
        local_retention,
    )

    p4_requested = {(item["metric_id"], item["metric_version"]) for item in P4_METRICS}
    if requested == p4_requested:
        local_summary = summary["local_utility"]
        global_summary = summary["global_utility"]
        if summary["dimensions"].get("local-utility") != local_summary or summary["dimensions"].get(
            "global-utility"
        ) != global_summary:
            raise BundleError("P4 canonical Local/Global Utility summaries differ from dimensions")
        if request.real_test_artifact is None:
            raise BundleError("P4 finalized bundles require a checksum-bound held-out real test artifact")
        details_path = path.parent / "artifacts" / "utility-details.json"
        try:
            details = read_json(details_path)
        except (OSError, ValueError, SerializationError) as exc:
            raise BundleError(f"Cannot validate P4 utility details: {exc}") from exc
        if not isinstance(details, dict) or details.get("utility_details_schema_version") != "1.0.0":
            raise BundleError("P4 utility details use an invalid contract")
        boundary = details.get("input_boundary")
        if not isinstance(boundary, dict) or boundary != {
            "real_train_fit_allowed": True,
            "synthetic_train_fit_allowed_only_for_tstr": True,
            "real_test_fit_allowed": False,
            "same_real_test_for_all_arms": True,
            "synthetic_repair_applied": False,
        }:
            raise BundleError("P4 utility details do not prove the held-out-test boundary")
        profile_ref = details.get("evaluator_profile")
        if not isinstance(profile_ref, dict) or {
            key: profile_ref.get(key) for key in ("profile_id", "profile_version", "sha256")
        } != request.evaluator_profile:
            raise BundleError("P4 details evaluator profile differs from the immutable request")

        local_runs = details.get("local_runs")
        global_runs = details.get("global_runs")
        if not isinstance(local_runs, list) or not local_runs or not isinstance(global_runs, list) or not global_runs:
            raise BundleError("P4 details must preserve every Local and Global raw-arm run")
        by_metric_scope = {(result.metric_id, result.scope_id): result for result in results}
        primary_name = local_summary.get("primary_metric")
        primary_metric_id = LOCAL_METRIC_IDS.get(primary_name)
        if primary_metric_id is None:
            raise BundleError("P4 Local Utility summary declares an unknown primary metric")
        test_fingerprints = {run.get("test_fingerprint") for run in local_runs}
        if test_fingerprints != {request.real_test_artifact["sha256"]}:
            raise BundleError(
                "P4 Local arms do not attest the checksum-bound held-out real test artifact"
            )
        computed_local_retentions: list[AtomicResult] = []
        for run in local_runs:
            required_run = {
                "task_id",
                "target_column_id",
                "task_type",
                "evaluator_id",
                "seed",
                "primary_metric",
                "raw_arms",
                "retention",
                "retention_state",
                "retention_reason_code",
                "test_fingerprint",
            }
            if not isinstance(run, dict) or set(run) != required_run or run["primary_metric"] != primary_name:
                raise BundleError("P4 Local run detail is malformed")
            seed_text = f"neg-{abs(run['seed'])}" if run["seed"] < 0 else str(run["seed"])
            base = f"{run['task_id']}--{run['evaluator_id']}--seed-{seed_text}"
            raw_arms = run["raw_arms"]
            if not isinstance(raw_arms, dict) or set(raw_arms) != {"dummy", "trtr", "tstr"}:
                raise BundleError("P4 Local run does not retain exactly Dummy, TRTR, and TSTR")
            for arm, detail_value in raw_arms.items():
                atom = by_metric_scope.get((primary_metric_id, f"{base}--{arm}"))
                if atom is None:
                    raise BundleError("P4 Local detail omits a primary raw-arm Atomic Result")
                if atom.weight != 0 or atom.aggregate_contribution is not None:
                    raise BundleError("P4 raw Local arms must not contribute directly to aggregation")
                if atom.state.value == "computed":
                    if detail_value is None or atom.raw_value is None or not math.isclose(
                        float(detail_value), atom.raw_value, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        raise BundleError("P4 Local raw-arm detail differs from its Atomic Result")
                elif detail_value is not None:
                    raise BundleError("A non-computed P4 Local raw arm must have a null detail value")
            retention_atom = by_metric_scope.get((LOCAL_RETENTION_METRIC_ID, base))
            if retention_atom is None or retention_atom.state.value != run["retention_state"]:
                raise BundleError("P4 Local retention detail differs from its Atomic Result state")
            if retention_atom.state.value == "computed":
                try:
                    reconstructed = local_retention(
                        float(raw_arms["dummy"]),
                        float(raw_arms["trtr"]),
                        float(raw_arms["tstr"]),
                        higher_is_better=run["task_type"] == "classification",
                        tolerance=1e-12,
                    )
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    raise BundleError(f"Cannot reconstruct P4 Local retention: {exc}") from exc
                if retention_atom.raw_value is None or not math.isclose(
                    reconstructed, retention_atom.raw_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise BundleError("P4 Local retention is not reproducible from raw arms")
                if run["retention"] is None or not math.isclose(
                    float(run["retention"]), retention_atom.raw_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise BundleError("P4 Local retention detail differs from its Atomic Result")
                computed_local_retentions.append(retention_atom)
            elif run["retention"] is not None:
                raise BundleError("A non-computed P4 Local retention must have a null detail value")

        expected_local = local_summary.get("expected_retentions")
        computed_local = local_summary.get("computed_retentions")
        if expected_local != len(local_runs) or computed_local != len(computed_local_retentions):
            raise BundleError("P4 Local retention denominator counts are inconsistent")
        if computed_local == expected_local:
            expected_weight = 1.0 / expected_local
            local_contributions: list[float] = []
            for atom in computed_local_retentions:
                if atom.raw_value is None or atom.aggregate_contribution is None:
                    raise BundleError("A computed P4 Local retention lacks a value or contribution")
                if (
                    not math.isclose(atom.weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12)
                    or atom.normalized_value != atom.raw_value
                    or not math.isclose(
                        atom.aggregate_contribution,
                        atom.raw_value * expected_weight,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise BundleError("P4 Local retention weights or contributions violate the equal-run contract")
                local_contributions.append(atom.aggregate_contribution)
            reconstructed_local = sum(local_contributions)
            if local_summary.get("retention") is None or not math.isclose(
                reconstructed_local, float(local_summary["retention"]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise BundleError("P4 Local summary is not reproducible from Atomic Results")
        elif local_summary.get("retention") is not None:
            raise BundleError("P4 strict Local summary must be null when any requested retention is unavailable")

        computed_global_ratios: list[AtomicResult] = []
        for run in global_runs:
            if not isinstance(run, dict) or set(run) != {
                "target_column_id",
                "target_name",
                "task_type",
                "seed",
                "trtr",
                "tstr",
                "ratio",
                "state",
                "reason_code",
                "predictors",
                "predictor_scores",
            }:
                raise BundleError("P4 Global run detail is malformed")
            seed_text = f"neg-{abs(run['seed'])}" if run["seed"] < 0 else str(run["seed"])
            base = f"{run['target_column_id']}--seed-{seed_text}"
            ratio_atom = by_metric_scope.get((GLOBAL_TARGET_RATIO_METRIC_ID, base))
            if ratio_atom is None or ratio_atom.state.value != run["state"]:
                raise BundleError("P4 Global ratio detail differs from its Atomic Result state")
            if ratio_atom.state.value == "computed":
                try:
                    reconstructed = global_target_ratio(
                        float(run["trtr"]), float(run["tstr"]), task_type=run["task_type"]
                    )
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    raise BundleError(f"Cannot reconstruct P4 Global target ratio: {exc}") from exc
                if ratio_atom.raw_value is None or not math.isclose(
                    reconstructed, ratio_atom.raw_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise BundleError("P4 Global target ratio is not reproducible from raw arms")
                if run["ratio"] is None or not math.isclose(
                    float(run["ratio"]), ratio_atom.raw_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise BundleError("P4 Global ratio detail differs from its Atomic Result")
                computed_global_ratios.append(ratio_atom)
            elif run["ratio"] is not None:
                raise BundleError("A non-computed P4 Global ratio must have a null detail value")

        expected_global = global_summary.get("expected_target_seed_ratios")
        computed_global = global_summary.get("computed_target_seed_ratios")
        if expected_global != len(global_runs) or computed_global != len(computed_global_ratios):
            raise BundleError("P4 Global target denominator counts are inconsistent")
        if computed_global == expected_global:
            expected_weight = 1.0 / expected_global
            global_contributions: list[float] = []
            for atom in computed_global_ratios:
                if atom.raw_value is None or atom.aggregate_contribution is None:
                    raise BundleError("A computed P4 Global ratio lacks a value or contribution")
                if (
                    not math.isclose(atom.weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12)
                    or atom.normalized_value != atom.raw_value
                    or not math.isclose(
                        atom.aggregate_contribution,
                        atom.raw_value * expected_weight,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise BundleError("P4 Global ratio weights violate equal target/seed aggregation")
                global_contributions.append(atom.aggregate_contribution)
            reconstructed_global = sum(global_contributions)
            if global_summary.get("global_utility") is None or not math.isclose(
                reconstructed_global,
                float(global_summary["global_utility"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise BundleError("P4 Global summary is not reproducible from Atomic Results")
        elif global_summary.get("global_utility") is not None:
            raise BundleError("P4 strict Global Utility must be null when any target/seed ratio is unavailable")
        if local_summary.get("retention_clipped") is not False or global_summary.get("ratio_clipped") is not False:
            raise BundleError("P4 finalized bundles must retain unclipped Local and Global values")
        denominator_counts = details.get("denominator_counts")
        if (
            denominator_counts != summary.get("denominator_counts")
            or denominator_counts != metadata.get("coverage", {}).get("denominators")
            or denominator_counts.get("local_requested_task_evaluator_seeds") != expected_local
            or denominator_counts.get("local_computed_retentions") != computed_local
            or denominator_counts.get("global_requested_target_seeds") != expected_global
            or denominator_counts.get("global_computed_target_seed_ratios") != computed_global
        ):
            raise BundleError("P4 denominator evidence differs across Atomic Results, details, summary, or metadata")


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
