"""Content-addressed stage orchestration with resumable, auditable attempts."""

from __future__ import annotations

import copy
import json
import mimetypes
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    SerializationError,
    atomic_write_json,
    content_fingerprint,
    read_json,
    sha256_file,
    validate_bundle_relative_path,
)
from standardized_tabular_diffusion.orchestration.environment import capture_software_profile
from standardized_tabular_diffusion.orchestration.hardware import capture_hardware_profile
from standardized_tabular_diffusion.orchestration.process import (
    ProcessLimits,
    ProcessOutcome,
    redact_text,
    run_isolated_process,
)

_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret|credential)")
_FAILURE_STATUSES = {"failed", "cancelled", "invalidated"}


class OrchestrationError(RuntimeError):
    """Raised when an execution cannot preserve P6 identity or audit guarantees."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    action: str
    dependencies: tuple[str, ...] = ()
    version: str = "1.0.0"
    required: bool = True
    enabled: bool = True
    allow_failed_dependencies: bool = False
    identity_inputs: dict[str, Any] = field(default_factory=dict)
    input_fingerprints: dict[str, str] = field(default_factory=dict)
    output_paths: tuple[str, ...] = ()
    cacheable: bool = True
    timeout_seconds: float | None = None
    memory_limit_bytes: int | None = None
    requested_rows: int | None = None
    warmup_policy: str = "not-applicable"
    uses_accelerator: bool = False
    max_retries: int = 0
    retry_categories: tuple[str, ...] = ("timeout", "implementation")
    environment_updates: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("version", self.version)):
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"Stage {label} is not a portable identifier: {value!r}")
        if not self.command or any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError("Stage command must contain non-empty strings")
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("Stage action must be a non-empty description")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"Stage {self.name} has duplicate dependencies")
        for dependency in self.dependencies:
            if not _IDENTIFIER.fullmatch(dependency):
                raise ValueError(f"Stage dependency is not portable: {dependency!r}")
        for relative in self.output_paths:
            validate_bundle_relative_path(relative)
        if len(set(self.output_paths)) != len(self.output_paths):
            raise ValueError(f"Stage {self.name} has duplicate output paths")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_limit_bytes is not None and self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        if self.requested_rows is not None and self.requested_rows < 0:
            raise ValueError("requested_rows must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        for name, digest in self.input_fingerprints.items():
            if not _IDENTIFIER.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid input fingerprint {name!r}: {digest!r}")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_root(root: str | Path) -> Path:
    path = Path(root)
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise OrchestrationError(f"Run root must be a regular directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _within_root(root: Path, relative: str) -> Path:
    validate_bundle_relative_path(relative)
    destination = root.joinpath(*relative.split("/"))
    resolved = destination.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise OrchestrationError(f"Output path escapes run root: {relative}")
    current = root
    for part in relative.split("/")[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise OrchestrationError(f"Symlinked output directory is prohibited: {relative}")
    return destination


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_SECRET_KEY.search(str(key)) or _contains_secret_key(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


def _redact_identity_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted-secret-input>" if _SECRET_KEY.search(str(key)) else _redact_identity_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_identity_secrets(item) for item in value]
    return value


def _portable_identity_value(value: Any) -> Any:
    value = _redact_identity_secrets(value)
    if isinstance(value, dict):
        return {str(key): _portable_identity_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_identity_value(item) for item in value]
    if isinstance(value, str) and (Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)):
        return "<absolute-path-redacted-use-input-fingerprint>"
    return value


def _empty_resource_usage(spec: StageSpec, *, mode: str = "not-run") -> dict[str, Any]:
    return {
        "wall_seconds": None,
        "action_wall_seconds": None,
        "cpu_seconds": None,
        "peak_rss_bytes": None,
        "peak_accelerator_memory_bytes": None,
        "requested_rows": spec.requested_rows,
        "actual_rows": None,
        "rows_per_second": None,
        "warmup_policy": spec.warmup_policy,
        "excluded_setup_seconds": None,
        "measurement_mode": mode,
        "reliability": [],
        "efficiency_eligible": False,
    }


def _empty_process(spec: StageSpec) -> dict[str, Any]:
    return {
        "exit_code": None,
        "timed_out": False,
        "memory_limit_exceeded": False,
        "interrupted": False,
        "timeout_seconds": spec.timeout_seconds,
        "memory_limit_bytes": spec.memory_limit_bytes,
    }


def _output_descriptor(root: Path, relative: str) -> dict[str, Any]:
    path = _within_root(root, relative)
    if not path.is_file() or path.is_symlink():
        raise OrchestrationError(f"Stage output is missing or unsafe: {relative}")
    media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    return {
        "path": relative,
        "media_type": media_type,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _collect_outputs(root: Path, relative_paths: Sequence[str]) -> tuple[dict[str, Any], ...]:
    descriptors = [_output_descriptor(root, relative) for relative in sorted(set(relative_paths))]
    return tuple(descriptors)


def _attempt_records(state_root: Path) -> list[dict[str, Any]]:
    attempts = state_root / "attempts"
    if not attempts.is_dir() or attempts.is_symlink():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(attempts.glob("*.json")):
        if path.is_symlink():
            raise OrchestrationError(f"Symlinked stage attempt is prohibited: {path.name}")
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise OrchestrationError(f"Stage attempt is not an object: {path.name}")
        validate_instance("orchestration-stage-record", payload)
        records.append(payload)
    return records


def _record_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": record["stage_id"],
        "stage_name": record["stage_name"],
        "status": record["status"],
        "identity_fingerprint": record["identity_fingerprint"],
        "path": f".orchestration/attempts/{record['stage_id']}.json",
    }


def _write_record(state_root: Path, record: dict[str, Any]) -> None:
    validate_instance("orchestration-stage-record", record)
    atomic_write_json(state_root / "attempts" / f"{record['stage_id']}.json", record)


def _cache_paths(cache_root: Path, stage_name: str, key: str) -> tuple[Path, Path]:
    entry = cache_root / "entries" / stage_name / f"{key}.json"
    artifact_root = cache_root / "artifacts"
    return entry, artifact_root


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _materialize_cache_entry(
    *,
    entry_path: Path,
    artifact_root: Path,
    run_root: Path,
    stage_name: str,
    key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not entry_path.is_file() or entry_path.is_symlink():
        return None, None
    try:
        entry = read_json(entry_path)
        if not isinstance(entry, dict):
            return None, "cache_entry_not_object"
        if set(entry) != {
            "cache_entry_schema_version",
            "cache_key",
            "stage_name",
            "stage_version",
            "identity_fingerprint",
            "source_stage_id",
            "outputs",
            "scientific_metadata",
        }:
            return None, "cache_entry_shape_invalid"
        if (
            entry["cache_entry_schema_version"] != "1.0.0"
            or entry["cache_key"] != key
            or entry["identity_fingerprint"] != key
            or entry["stage_name"] != stage_name
        ):
            return None, "cache_identity_mismatch"
        outputs = entry["outputs"]
        if not isinstance(outputs, list):
            return None, "cache_outputs_invalid"
        for descriptor in outputs:
            if not isinstance(descriptor, dict) or set(descriptor) != {"path", "media_type", "byte_size", "sha256"}:
                return None, "cache_output_shape_invalid"
            relative = validate_bundle_relative_path(descriptor["path"])
            digest = descriptor["sha256"]
            blob = artifact_root / digest[:2] / digest
            if (
                not blob.is_file()
                or blob.is_symlink()
                or blob.stat().st_size != descriptor["byte_size"]
                or sha256_file(blob) != digest
            ):
                return None, "cache_artifact_missing_or_corrupt"
            destination = _within_root(run_root, relative)
            if (
                destination.is_file()
                and not destination.is_symlink()
                and destination.stat().st_size == descriptor["byte_size"]
                and sha256_file(destination) == digest
            ):
                continue
            if destination.exists() and (destination.is_symlink() or not destination.is_file()):
                return None, "cache_destination_unsafe"
            _atomic_copy(blob, destination)
        verified = _collect_outputs(run_root, [item["path"] for item in outputs])
        if list(verified) != outputs:
            return None, "cache_materialization_verification_failed"
        return entry, None
    except (OSError, SerializationError, OrchestrationError, TypeError, ValueError):
        return None, "cache_entry_unreadable"


def _store_cache_entry(
    *,
    cache_root: Path,
    stage: StageSpec,
    key: str,
    stage_id: str,
    outputs: Sequence[dict[str, Any]],
    run_root: Path,
    scientific_metadata: dict[str, Any],
) -> None:
    entry_path, artifact_root = _cache_paths(cache_root, stage.name, key)
    for descriptor in outputs:
        source = _within_root(run_root, descriptor["path"])
        digest = descriptor["sha256"]
        blob = artifact_root / digest[:2] / digest
        if blob.exists():
            if blob.is_symlink() or not blob.is_file():
                raise OrchestrationError(f"Content cache blob is unsafe or corrupt: {digest}")
            if sha256_file(blob) != digest:
                _atomic_copy(source, blob)
                if sha256_file(blob) != digest:
                    raise OrchestrationError(f"Content cache repair failed verification: {digest}")
        else:
            _atomic_copy(source, blob)
            if sha256_file(blob) != digest:
                raise OrchestrationError(f"Content cache copy failed verification: {digest}")
    entry = {
        "cache_entry_schema_version": "1.0.0",
        "cache_key": key,
        "stage_name": stage.name,
        "stage_version": stage.version,
        "identity_fingerprint": key,
        "source_stage_id": stage_id,
        "outputs": list(outputs),
        "scientific_metadata": copy.deepcopy(scientific_metadata),
    }
    atomic_write_json(entry_path, entry)


def _worker_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not path.is_file() or path.is_symlink():
        return {}, warnings
    try:
        payload = read_json(path)
    except (OSError, SerializationError):
        return {}, ["worker_result_unreadable"]
    if not isinstance(payload, dict):
        return {}, ["worker_result_invalid"]
    allowed = {
        "worker_result_schema_version",
        "status",
        "action_wall_seconds",
        "actual_rows",
        "outputs",
        "accelerator_peak_bytes",
        "reliability",
        "failure",
        "cache_safe",
    }
    if set(payload) != allowed or payload.get("worker_result_schema_version") != "1.0.0":
        return {}, ["worker_result_invalid"]
    return payload, warnings


def _failure_from_outcome(outcome: ProcessOutcome, worker: Mapping[str, Any]) -> dict[str, str] | None:
    detail = next((line for line in reversed(outcome.diagnostic_tail) if line.strip()), "Stage process failed")
    detail = detail[:500]
    if outcome.interrupted:
        return {"category": "interrupted", "reason_code": "user_interruption", "detail": "Execution was interrupted"}
    if outcome.timed_out:
        return {"category": "timeout", "reason_code": "stage_timeout", "detail": "Stage exceeded its time limit"}
    if outcome.memory_limit_exceeded:
        return {
            "category": "out-of-memory",
            "reason_code": "stage_memory_limit",
            "detail": "Stage exceeded its process-tree memory limit",
        }
    if outcome.launch_error:
        return {"category": "dependency", "reason_code": "process_launch_failure", "detail": outcome.launch_error[:500]}
    worker_failure = worker.get("failure")
    if isinstance(worker_failure, dict) and set(worker_failure) == {"category", "reason_code", "detail"}:
        category = worker_failure.get("category")
        if category in {"out-of-memory", "dependency", "implementation"}:
            reason_code = str(worker_failure["reason_code"])
            if not re.fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", reason_code):
                reason_code = "worker_declared_failure"
            return {
                "category": str(category),
                "reason_code": reason_code,
                "detail": str(worker_failure["detail"])[:500],
            }
    if worker.get("status") == "failed":
        return {
            "category": "implementation",
            "reason_code": "worker_declared_failure",
            "detail": "Worker declared failure without a valid structured failure record",
        }
    folded = "\n".join(outcome.diagnostic_tail).casefold()
    if "out of memory" in folded or "cuda out of memory" in folded or outcome.exit_code in {137, -9, 3221225495}:
        return {"category": "out-of-memory", "reason_code": "process_out_of_memory", "detail": detail}
    if "modulenotfounderror" in folded or "importerror" in folded:
        return {"category": "dependency", "reason_code": "dependency_import_failure", "detail": detail}
    if outcome.exit_code not in {0, None}:
        return {"category": "implementation", "reason_code": "stage_process_failure", "detail": detail}
    return None


def _stage_identity(
    stage: StageSpec,
    *,
    dependency_records: Sequence[Mapping[str, Any]],
    code_fingerprint: str,
    software_fingerprint: str,
    hardware_key: str,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    dependency_inputs = {
        record["stage_name"]: {
            "status": record["status"],
            "output_fingerprint": record["output_fingerprint"],
            "identity_fingerprint": record["identity_fingerprint"],
        }
        for record in dependency_records
    }
    inputs = {
        "stage": _portable_identity_value(stage.identity_inputs),
        "declaration": {
            "output_paths": list(stage.output_paths),
            "required": stage.required,
            "enabled": stage.enabled,
            "allow_failed_dependencies": stage.allow_failed_dependencies,
            "timeout_seconds": stage.timeout_seconds,
            "memory_limit_bytes": stage.memory_limit_bytes,
            "requested_rows": stage.requested_rows,
            "warmup_policy": stage.warmup_policy,
            "uses_accelerator": stage.uses_accelerator,
            "max_retries": stage.max_retries,
            "retry_categories": list(stage.retry_categories),
        },
        "dependencies": dependency_inputs,
    }
    fingerprints = dict(stage.input_fingerprints)
    fingerprints["stage-command"] = content_fingerprint(_portable_identity_value(list(stage.command)))
    fingerprints["stage-environment"] = content_fingerprint(
        _portable_identity_value(stage.environment_updates)
    )
    for record in dependency_records:
        fingerprints[f"dependency.{record['stage_name']}"] = record["output_fingerprint"]
    material = {
        "stage_name": stage.name,
        "stage_version": stage.version,
        "identity_inputs": inputs,
        "input_fingerprints": fingerprints,
        "code_fingerprint": code_fingerprint,
        "software_fingerprint": software_fingerprint,
        "hardware_comparison_key": hardware_key,
    }
    return content_fingerprint(material), inputs, fingerprints


def _stage_attempt_number(records: Sequence[Mapping[str, Any]], stage_name: str) -> int:
    return 1 + sum(1 for record in records if record["stage_name"] == stage_name)


def _stage_id(stage_name: str, identity: str, attempt: int) -> str:
    return f"{stage_name}-{identity[:12]}-a{attempt}"


def _base_record(
    stage: StageSpec,
    *,
    stage_id: str,
    attempt: int,
    dependency_ids: Sequence[str],
    identity: str,
    identity_inputs: dict[str, Any],
    input_fingerprints: dict[str, str],
    code_fingerprint: str,
    software_fingerprint: str,
    hardware_key: str,
    cache_decision: str,
    retry_count: int,
    ancestry: Sequence[str],
) -> dict[str, Any]:
    return {
        "orchestration_stage_schema_version": "1.0.0",
        "stage_id": stage_id,
        "stage_name": stage.name,
        "stage_version": stage.version,
        "attempt": attempt,
        "status": "running",
        "required": stage.required,
        "enabled": stage.enabled,
        "allow_failed_dependencies": stage.allow_failed_dependencies,
        "dependency_stage_ids": list(dependency_ids),
        "identity_fingerprint": identity,
        "identity_inputs": copy.deepcopy(identity_inputs),
        "input_fingerprints": dict(sorted(input_fingerprints.items())),
        "code_fingerprint": code_fingerprint,
        "software_fingerprint": software_fingerprint,
        "hardware_comparison_key": hardware_key,
        "resolved_action": stage.action,
        "started_at": _utc_timestamp(),
        "ended_at": None,
        "resource_usage": _empty_resource_usage(stage),
        "process": _empty_process(stage),
        "log_ref": None,
        "outputs": [],
        "output_fingerprint": content_fingerprint([]),
        "warning_codes": [],
        "failure": None,
        "cache": {
            "decision": cache_decision,
            "cache_key": identity,
            "identity_verified": cache_decision in {"miss", "bypassed", "not-requested"},
            "outputs_verified": False,
            "reused_stage_id": None,
        },
        "retry_count": retry_count,
        "resume_ancestry": list(ancestry),
    }


def _skipped_record(
    base: dict[str, Any],
    stage: StageSpec,
    *,
    dependency_failed: bool,
) -> dict[str, Any]:
    record = copy.deepcopy(base)
    record.update(status="skipped", started_at=None, ended_at=None)
    record["cache"].update(decision="not-requested", outputs_verified=True)
    record["warning_codes"] = ["dependency_failed" if dependency_failed else "not_requested"]
    if dependency_failed:
        record["failure"] = {
            "category": "dependency-failed",
            "reason_code": "dependency_failed",
            "detail": "A required dependency did not succeed",
        }
    return record


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    validate_instance("orchestration-run", payload)
    atomic_write_json(path, payload)


def _validate_hardware_profile_integrity(profile: dict[str, Any]) -> None:
    validate_instance("hardware-profile", profile)
    material = {
        key: profile[key]
        for key in ("operating_system", "python", "cpu", "memory", "accelerators", "runtime")
    }
    if content_fingerprint(material) != profile["comparison_key"]:
        raise OrchestrationError("Hardware profile comparison key is not reproducible from its material fields")


def _validate_software_profile_integrity(profile: dict[str, Any]) -> None:
    validate_instance("software-profile", profile)
    material = {
        key: value
        for key, value in profile.items()
        if key not in {"software_profile_schema_version", "fingerprint", "code_fingerprint"}
    }
    if content_fingerprint(material) != profile["fingerprint"]:
        raise OrchestrationError("Software profile fingerprint is not reproducible from its material fields")
    if content_fingerprint(profile["repository"]) != profile["code_fingerprint"]:
        raise OrchestrationError("Software profile code fingerprint is not reproducible from repository identity")


def _summarize_status(current: Sequence[Mapping[str, Any]]) -> tuple[str, list[str], list[str]]:
    failed_required: list[str] = []
    failed_optional: list[str] = []
    cancelled = False
    for record in current:
        failed = record["status"] in _FAILURE_STATUSES or (
            record["status"] == "skipped" and record.get("failure") is not None
        )
        if not failed:
            continue
        if record["status"] == "cancelled":
            cancelled = True
        target = failed_required if record["required"] else failed_optional
        target.append(record["stage_name"])
    if cancelled:
        status = "cancelled"
    elif failed_required:
        status = "failed"
    elif failed_optional:
        status = "partial"
    else:
        status = "success"
    return status, sorted(set(failed_required)), sorted(set(failed_optional))


def execute_plan(
    stages: Sequence[StageSpec],
    *,
    run_root: str | Path,
    config_fingerprint: str,
    repo_root: str | Path,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    resume: bool = True,
    hardware_profile: dict[str, Any] | None = None,
    software_profile: dict[str, Any] | None = None,
    process_runner: Callable[..., ProcessOutcome] = run_isolated_process,
) -> dict[str, Any]:
    """Execute an ordered DAG; every dependency must precede its consumer."""

    stage_names = [stage.name for stage in stages]
    if len(set(stage_names)) != len(stage_names):
        raise OrchestrationError("An execution plan cannot contain duplicate stage names")
    observed_names: set[str] = set()
    for stage in stages:
        missing = set(stage.dependencies) - observed_names
        if missing:
            raise OrchestrationError(f"Stage {stage.name} has unordered or unknown dependencies: {sorted(missing)}")
        observed_names.add(stage.name)

    root = _safe_root(run_root)
    repository_root = Path(repo_root).resolve()
    state_root = root / ".orchestration"
    if state_root.exists() and state_root.is_symlink():
        raise OrchestrationError("Symlinked orchestration state is prohibited")
    manifest_path = state_root / "run.json"
    previous: dict[str, Any] | None = None
    if manifest_path.is_file():
        loaded = read_json(manifest_path)
        if not isinstance(loaded, dict):
            raise OrchestrationError("Existing orchestration manifest is invalid")
        validate_instance("orchestration-run", loaded)
        previous = loaded
        if not resume:
            raise OrchestrationError("--no-resume requires an output directory without prior orchestration state")
        if loaded["config_fingerprint"] != config_fingerprint:
            raise OrchestrationError("Refusing to resume an output directory with a different configuration identity")
    state_root.mkdir(parents=True, exist_ok=True)
    for directory in ("attempts", "invocations", "logs", "profiles", "worker-results"):
        (state_root / directory).mkdir(exist_ok=True)

    profile = hardware_profile or capture_hardware_profile()
    _validate_hardware_profile_integrity(profile)
    software = software_profile or capture_software_profile(repository_root)
    _validate_software_profile_integrity(software)
    if not re.fullmatch(r"[0-9a-f]{64}", str(software.get("fingerprint"))):
        raise OrchestrationError("Software profile is missing a valid fingerprint")
    if not re.fullmatch(r"[0-9a-f]{64}", str(software.get("code_fingerprint"))):
        raise OrchestrationError("Software profile is missing a valid code fingerprint")
    hardware_document_fingerprint = content_fingerprint(profile)
    hardware_relative = f".orchestration/profiles/hardware-{hardware_document_fingerprint}.json"
    software_relative = f".orchestration/profiles/software-{software['fingerprint']}.json"
    atomic_write_json(_within_root(root, hardware_relative), profile)
    atomic_write_json(_within_root(root, software_relative), software)

    cache_root = _safe_root(cache_dir or state_root / "cache")
    cache_location_class = "run-local" if cache_root == (state_root / "cache").resolve() else "external"
    prior_records = _attempt_records(state_root)
    prior_refs = [_record_ref(record) for record in prior_records]
    run_id = previous["run_id"] if previous is not None else f"orun-{uuid.uuid4().hex}"
    invocation_id = f"exec-{uuid.uuid4().hex}"
    previous_invocation_id = previous["invocation_id"] if previous is not None else None
    if previous is not None:
        atomic_write_json(state_root / "invocations" / f"{previous_invocation_id}.json", previous)
    invocation_ancestry = list(previous.get("resume_ancestry", [])) if previous is not None else []
    if previous_invocation_id is not None:
        invocation_ancestry.append(previous_invocation_id)
    invocation_ancestry = list(dict.fromkeys(invocation_ancestry))
    manifest: dict[str, Any] = {
        "orchestration_run_schema_version": "1.0.0",
        "run_id": run_id,
        "invocation_id": invocation_id,
        "previous_invocation_id": previous_invocation_id,
        "status": "running",
        "started_at": _utc_timestamp(),
        "ended_at": None,
        "config_fingerprint": config_fingerprint,
        "code_fingerprint": software["code_fingerprint"],
        "software_fingerprint": software["fingerprint"],
        "software_profile_ref": software_relative,
        "hardware_profile_ref": hardware_relative,
        "hardware_comparison_key": profile["comparison_key"],
        "cache_policy": {
            "enabled": use_cache,
            "content_addressed": True,
            "artifact_verification": "sha256-and-size",
            "cache_location_class": cache_location_class,
        },
        "stage_records": prior_refs,
        "current_stage_ids": [],
        "failed_required_stages": [],
        "failed_optional_stages": [],
        "resume_ancestry": invocation_ancestry,
        "warning_codes": [],
    }
    _write_manifest(manifest_path, manifest)

    current: list[dict[str, Any]] = []
    latest_by_name: dict[str, dict[str, Any]] = {}
    all_records = list(prior_records)
    stop_after_interruption = False
    for stage in stages:
        dependency_records = [latest_by_name[name] for name in stage.dependencies]
        identity, identity_inputs, input_fingerprints = _stage_identity(
            stage,
            dependency_records=dependency_records,
            code_fingerprint=software["code_fingerprint"],
            software_fingerprint=software["fingerprint"],
            hardware_key=profile["comparison_key"],
        )
        same_identity = [
            record
            for record in all_records
            if record["stage_name"] == stage.name and record["identity_fingerprint"] == identity
        ]
        ancestry = [record["stage_id"] for record in same_identity]
        retry_count = sum(1 for record in same_identity if record["status"] in _FAILURE_STATUSES)
        dependency_failed = any(
            record["status"] != "succeeded" and not (record["status"] == "skipped" and record["failure"] is None)
            for record in dependency_records
        )
        secret_bearing_stage = (
            _contains_secret_key(stage.identity_inputs)
            or _contains_secret_key(stage.environment_updates)
            or any(_SECRET_KEY.search(part) for part in stage.command)
        )
        cache_allowed = use_cache and stage.cacheable and not secret_bearing_stage
        entry_path, artifact_root = _cache_paths(cache_root, stage.name, identity)
        cached_entry: dict[str, Any] | None = None
        cache_error: str | None = None
        cache_decision = "bypassed" if not cache_allowed else "miss"
        cache_started = time.perf_counter()
        if cache_allowed and stage.enabled and (not dependency_failed or stage.allow_failed_dependencies):
            cached_entry, cache_error = _materialize_cache_entry(
                entry_path=entry_path,
                artifact_root=artifact_root,
                run_root=root,
                stage_name=stage.name,
                key=identity,
            )
            if cached_entry is not None:
                cache_decision = "hit"
            elif cache_error is not None:
                cache_decision = "invalid"

        attempt = _stage_attempt_number(all_records, stage.name)
        stage_identifier = _stage_id(stage.name, identity, attempt)
        base = _base_record(
            stage,
            stage_id=stage_identifier,
            attempt=attempt,
            dependency_ids=[record["stage_id"] for record in dependency_records],
            identity=identity,
            identity_inputs=identity_inputs,
            input_fingerprints=input_fingerprints,
            code_fingerprint=software["code_fingerprint"],
            software_fingerprint=software["fingerprint"],
            hardware_key=profile["comparison_key"],
            cache_decision=cache_decision,
            retry_count=retry_count,
            ancestry=ancestry,
        )
        if secret_bearing_stage:
            base["warning_codes"].append("cache_bypassed_secret_input")
        if cache_error is not None:
            base["warning_codes"].append("stale_cache_rejected")

        if stop_after_interruption or not stage.enabled or (dependency_failed and not stage.allow_failed_dependencies):
            record = _skipped_record(
                base,
                stage,
                dependency_failed=stop_after_interruption or dependency_failed,
            )
            _write_record(state_root, record)
            current.append(record)
            all_records.append(record)
            latest_by_name[stage.name] = record
            manifest["stage_records"].append(_record_ref(record))
            manifest["current_stage_ids"].append(record["stage_id"])
            _write_manifest(manifest_path, manifest)
            continue

        if cached_entry is not None:
            ended = _utc_timestamp()
            outputs = cached_entry["outputs"]
            scientific = cached_entry.get("scientific_metadata") or {}
            resource = _empty_resource_usage(stage, mode="cache-reuse")
            resource["wall_seconds"] = time.perf_counter() - cache_started
            resource["actual_rows"] = scientific.get("actual_rows")
            resource["reliability"] = ["cache_hit_not_efficiency_observation"]
            record = copy.deepcopy(base)
            record.update(
                status="succeeded",
                ended_at=ended,
                resource_usage=resource,
                outputs=outputs,
                output_fingerprint=content_fingerprint(outputs),
            )
            record["cache"].update(
                identity_verified=True,
                outputs_verified=True,
                reused_stage_id=cached_entry["source_stage_id"],
            )
            _write_record(state_root, record)
            current.append(record)
            all_records.append(record)
            latest_by_name[stage.name] = record
            manifest["stage_records"].append(_record_ref(record))
            manifest["current_stage_ids"].append(record["stage_id"])
            _write_manifest(manifest_path, manifest)
            continue

        retries_remaining = stage.max_retries
        while True:
            if base["attempt"] != attempt:
                raise AssertionError("Internal attempt identity drift")
            log_relative = f".orchestration/logs/{stage_identifier}.jsonl"
            worker_relative = f".orchestration/worker-results/{stage_identifier}.json"
            log_path = _within_root(root, log_relative)
            worker_path = _within_root(root, worker_relative)
            base["log_ref"] = log_relative
            _write_record(state_root, base)
            environment = os.environ.copy()
            environment.update(stage.environment_updates)
            environment.setdefault("PYTHONIOENCODING", "utf-8")
            environment["STD_ORCHESTRATION_WORKER_RESULT"] = str(worker_path)
            try:
                outcome = process_runner(
                    stage.command,
                    cwd=repository_root,
                    log_path=log_path,
                    stage_id=stage_identifier,
                    limits=ProcessLimits(
                        timeout_seconds=stage.timeout_seconds,
                        memory_limit_bytes=stage.memory_limit_bytes,
                    ),
                    environment=environment,
                    path_aliases={
                        str(repository_root): "<repo-root>",
                        str(root): "<run-root>",
                        str(cache_root): "<cache-root>",
                    },
                )
            except RuntimeError as exc:
                safe = redact_text(
                    str(exc),
                    path_aliases={str(repository_root): "<repo-root>", str(root): "<run-root>"},
                )
                outcome = ProcessOutcome(
                    exit_code=None,
                    wall_seconds=0.0,
                    cpu_seconds=None,
                    peak_rss_bytes=None,
                    peak_accelerator_memory_bytes=None,
                    timed_out=False,
                    memory_limit_exceeded=False,
                    interrupted=False,
                    launch_error=safe,
                    reliability=("process_boundary_dependency_unavailable",),
                    diagnostic_tail=(safe,),
                )
            worker, worker_warnings = _worker_metadata(worker_path)
            warnings = sorted(set([*base["warning_codes"], *worker_warnings]))
            if not log_path.is_file() or log_path.is_symlink():
                base["log_ref"] = None
                warnings.append("stage_log_unavailable")
            dynamic_outputs = worker.get("outputs", []) if isinstance(worker.get("outputs", []), list) else []
            relative_outputs = list(stage.output_paths)
            for item in dynamic_outputs:
                if isinstance(item, str):
                    try:
                        validate_bundle_relative_path(item)
                    except SerializationError:
                        warnings.append("worker_output_path_rejected")
                    else:
                        relative_outputs.append(item)
            present_outputs: list[str] = []
            for relative in sorted(set(relative_outputs)):
                path = _within_root(root, relative)
                if path.is_file() and not path.is_symlink():
                    present_outputs.append(relative)
            missing_expected = sorted(set(stage.output_paths) - set(present_outputs))
            try:
                outputs = _collect_outputs(root, present_outputs)
            except OrchestrationError:
                outputs = ()
                missing_expected = list(stage.output_paths)
                warnings.append("output_verification_failure")
            failure = _failure_from_outcome(outcome, worker)
            if failure is not None:
                failure["detail"] = redact_text(
                    failure["detail"],
                    path_aliases={
                        str(repository_root): "<repo-root>",
                        str(root): "<run-root>",
                        str(cache_root): "<cache-root>",
                        str(Path.home()): "<user-home>",
                    },
                )
            if failure is None and missing_expected:
                failure = {
                    "category": "implementation",
                    "reason_code": "missing_stage_output",
                    "detail": "Stage reported success without all declared outputs",
                }
            action_wall = worker.get("action_wall_seconds")
            if not isinstance(action_wall, (int, float)) or isinstance(action_wall, bool) or action_wall < 0:
                action_wall = None
            actual_rows = worker.get("actual_rows")
            if not isinstance(actual_rows, int) or isinstance(actual_rows, bool) or actual_rows < 0:
                actual_rows = None
            accelerator_peak = worker.get("accelerator_peak_bytes")
            if not isinstance(accelerator_peak, int) or isinstance(accelerator_peak, bool) or accelerator_peak < 0:
                accelerator_peak = None
            observed_accelerator = outcome.peak_accelerator_memory_bytes
            if accelerator_peak is not None:
                observed_accelerator = (
                    accelerator_peak
                    if observed_accelerator is None
                    else max(observed_accelerator, accelerator_peak)
                )
            reliability = set(outcome.reliability)
            worker_reliability = worker.get("reliability", [])
            if isinstance(worker_reliability, list):
                reliability.update(
                    item
                    for item in worker_reliability
                    if isinstance(item, str)
                    and re.fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", item)
                )
            status = "succeeded" if failure is None else ("cancelled" if outcome.interrupted else "failed")
            rows_per_second = (
                float(actual_rows) / float(action_wall)
                if actual_rows is not None and action_wall is not None and action_wall > 0
                else None
            )
            efficiency_eligible = (
                status == "succeeded"
                and action_wall is not None
                and outcome.cpu_seconds is not None
                and outcome.peak_rss_bytes is not None
                and (not stage.uses_accelerator or observed_accelerator is not None)
                and not {
                    "requested_accelerator_unavailable",
                    "accelerator_synchronization_unavailable",
                }
                & reliability
                and not (
                    stage.uses_accelerator
                    and accelerator_peak is None
                    and "accelerator_memory_sampling_unavailable" in reliability
                )
            )
            resource = {
                "wall_seconds": outcome.wall_seconds,
                "action_wall_seconds": action_wall,
                "cpu_seconds": outcome.cpu_seconds,
                "peak_rss_bytes": outcome.peak_rss_bytes,
                "peak_accelerator_memory_bytes": observed_accelerator,
                "requested_rows": stage.requested_rows,
                "actual_rows": actual_rows,
                "rows_per_second": rows_per_second,
                "warmup_policy": stage.warmup_policy,
                "excluded_setup_seconds": (
                    max(0.0, outcome.wall_seconds - float(action_wall)) if action_wall is not None else None
                ),
                "measurement_mode": "fresh-process",
                "reliability": sorted(reliability),
                "efficiency_eligible": efficiency_eligible,
            }
            record = copy.deepcopy(base)
            record.update(
                status=status,
                ended_at=_utc_timestamp(),
                resource_usage=resource,
                process={
                    "exit_code": outcome.exit_code,
                    "timed_out": outcome.timed_out,
                    "memory_limit_exceeded": outcome.memory_limit_exceeded,
                    "interrupted": outcome.interrupted,
                    "timeout_seconds": stage.timeout_seconds,
                    "memory_limit_bytes": stage.memory_limit_bytes,
                },
                outputs=list(outputs),
                output_fingerprint=content_fingerprint(outputs),
                warning_codes=sorted(set(warnings)),
                failure=failure,
            )
            record["cache"].update(identity_verified=True, outputs_verified=failure is None)
            worker_cache_safe = worker.get("cache_safe") is True if worker else bool(stage.output_paths)
            if status == "succeeded" and cache_allowed and not worker_cache_safe:
                record["warning_codes"] = sorted(
                    set([*record["warning_codes"], "stage_outputs_not_cacheable"])
                )
            _write_record(state_root, record)
            all_records.append(record)
            manifest["stage_records"].append(_record_ref(record))
            if status == "succeeded" and cache_allowed and worker_cache_safe:
                try:
                    _store_cache_entry(
                        cache_root=cache_root,
                        stage=stage,
                        key=identity,
                        stage_id=stage_identifier,
                        outputs=outputs,
                        run_root=root,
                        scientific_metadata={"actual_rows": actual_rows},
                    )
                except (OSError, OrchestrationError):
                    record["warning_codes"] = sorted(set([*record["warning_codes"], "cache_write_failure"]))
                    _write_record(state_root, record)
            if status == "cancelled":
                stop_after_interruption = True
            if (
                status == "failed"
                and retries_remaining > 0
                and failure is not None
                and failure["category"] in stage.retry_categories
            ):
                _write_manifest(manifest_path, manifest)
                retries_remaining -= 1
                ancestry = [*ancestry, stage_identifier]
                attempt = _stage_attempt_number(all_records, stage.name)
                stage_identifier = _stage_id(stage.name, identity, attempt)
                base = _base_record(
                    stage,
                    stage_id=stage_identifier,
                    attempt=attempt,
                    dependency_ids=[item["stage_id"] for item in dependency_records],
                    identity=identity,
                    identity_inputs=identity_inputs,
                    input_fingerprints=input_fingerprints,
                    code_fingerprint=software["code_fingerprint"],
                    software_fingerprint=software["fingerprint"],
                    hardware_key=profile["comparison_key"],
                    cache_decision=cache_decision,
                    retry_count=sum(
                        1
                        for item in all_records
                        if item["stage_name"] == stage.name
                        and item["identity_fingerprint"] == identity
                        and item["status"] in _FAILURE_STATUSES
                    ),
                    ancestry=ancestry,
                )
                continue
            current.append(record)
            latest_by_name[stage.name] = record
            manifest["current_stage_ids"].append(record["stage_id"])
            _write_manifest(manifest_path, manifest)
            break

    status, failed_required, failed_optional = _summarize_status(current)
    manifest.update(
        status=status,
        ended_at=_utc_timestamp(),
        failed_required_stages=failed_required,
        failed_optional_stages=failed_optional,
    )
    _write_manifest(manifest_path, manifest)
    return manifest


def load_run_status(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root) / ".orchestration" / "run.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise OrchestrationError("Orchestration run manifest must be an object")
    validate_instance("orchestration-run", payload)
    return payload


def _scan_for_secret_text(value: str) -> bool:
    return bool(
        re.search(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b", value)
        or re.search(r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token)\s*[:=]\s*(?!<redacted>)\S+", value)
    )


def validate_orchestration_run(run_root: str | Path) -> dict[str, Any]:
    """Validate schemas, identities, attempts, outputs, and redacted JSONL logs."""

    root = _safe_root(run_root)
    manifest = load_run_status(root)
    hardware_path = _within_root(root, manifest["hardware_profile_ref"])
    hardware = read_json(hardware_path)
    if not isinstance(hardware, dict):
        raise OrchestrationError("Hardware profile must be an object")
    _validate_hardware_profile_integrity(hardware)
    expected_hardware_ref = f".orchestration/profiles/hardware-{content_fingerprint(hardware)}.json"
    if manifest["hardware_profile_ref"] != expected_hardware_ref:
        raise OrchestrationError("Hardware profile reference is not content-addressed")
    if hardware["comparison_key"] != manifest["hardware_comparison_key"]:
        raise OrchestrationError("Hardware profile comparison key differs from run manifest")
    software = read_json(_within_root(root, manifest["software_profile_ref"]))
    if not isinstance(software, dict):
        raise OrchestrationError("Software profile must be an object")
    _validate_software_profile_integrity(software)
    expected_software_ref = f".orchestration/profiles/software-{software['fingerprint']}.json"
    if manifest["software_profile_ref"] != expected_software_ref:
        raise OrchestrationError("Software profile reference is not content-addressed")
    if software.get("fingerprint") != manifest["software_fingerprint"]:
        raise OrchestrationError("Software profile fingerprint differs from run manifest")
    if software.get("code_fingerprint") != manifest["code_fingerprint"]:
        raise OrchestrationError("Code fingerprint differs from run manifest")
    records = _attempt_records(root / ".orchestration")
    by_id = {record["stage_id"]: record for record in records}
    if len(by_id) != len(records):
        raise OrchestrationError("Duplicate stage attempt identity")
    manifest_ids = [item["stage_id"] for item in manifest["stage_records"]]
    if set(manifest_ids) != {record["stage_id"] for record in records} or len(manifest_ids) != len(records):
        raise OrchestrationError("Run manifest stage history differs from attempt files")
    current_ids = set(manifest["current_stage_ids"])
    for item in manifest["stage_records"]:
        record = by_id[item["stage_id"]]
        if item != _record_ref(record):
            raise OrchestrationError(f"Stage reference differs from attempt record: {item['stage_id']}")
        for dependency in record["dependency_stage_ids"]:
            if dependency not in by_id:
                raise OrchestrationError(f"Stage dependency does not exist: {dependency}")
        for ancestor in record["resume_ancestry"]:
            if ancestor not in by_id:
                raise OrchestrationError(f"Stage resume ancestor does not exist: {ancestor}")
        if content_fingerprint(record["outputs"]) != record["output_fingerprint"]:
            raise OrchestrationError(f"Stage output fingerprint mismatch: {record['stage_id']}")
        if record["stage_id"] in current_ids:
            verified_outputs = _collect_outputs(root, [output["path"] for output in record["outputs"]])
            if list(verified_outputs) != record["outputs"]:
                raise OrchestrationError(f"Current stage output checksum mismatch: {record['stage_id']}")
        if record["cache"]["decision"] == "hit":
            if not record["cache"]["identity_verified"] or not record["cache"]["outputs_verified"]:
                raise OrchestrationError("Cache hit lacks identity and output verification")
            if record["resource_usage"]["efficiency_eligible"]:
                raise OrchestrationError("Cache reuse cannot be an efficiency observation")
        log_ref = record["log_ref"]
        if log_ref is not None:
            log_path = _within_root(root, log_ref)
            if not log_path.is_file() or log_path.is_symlink():
                raise OrchestrationError(f"Stage log is missing or unsafe: {log_ref}")
            for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OrchestrationError(f"Invalid stage log JSON at {log_ref}:{line_number}") from exc
                if not isinstance(event, dict) or event.get("stage_id") != record["stage_id"]:
                    raise OrchestrationError(f"Stage log identity mismatch at {log_ref}:{line_number}")
                if _scan_for_secret_text(line):
                    raise OrchestrationError(f"Unredacted secret pattern in stage log: {log_ref}:{line_number}")
    current = [by_id[stage_id] for stage_id in manifest["current_stage_ids"]]
    for invocation_id in manifest["resume_ancestry"]:
        invocation_path = root / ".orchestration" / "invocations" / f"{invocation_id}.json"
        ancestor = read_json(invocation_path)
        if not isinstance(ancestor, dict):
            raise OrchestrationError(f"Invocation ancestor is missing or invalid: {invocation_id}")
        validate_instance("orchestration-run", ancestor)
        if ancestor["invocation_id"] != invocation_id:
            raise OrchestrationError(f"Invocation ancestor identity mismatch: {invocation_id}")
    status, failed_required, failed_optional = _summarize_status(current)
    if (status, failed_required, failed_optional) != (
        manifest["status"],
        manifest["failed_required_stages"],
        manifest["failed_optional_stages"],
    ):
        raise OrchestrationError("Run terminal status does not match current stage attempts")
    return {
        "valid": True,
        "run_id": manifest["run_id"],
        "invocation_id": manifest["invocation_id"],
        "status": manifest["status"],
        "attempt_count": len(records),
        "current_stage_count": len(current),
        "hardware_comparison_key": manifest["hardware_comparison_key"],
    }
