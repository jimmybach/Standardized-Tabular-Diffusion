"""Observed hardware profiles and strict efficiency compatibility checks."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, read_json
from standardized_tabular_diffusion.orchestration.environment import MATERIAL_ENVIRONMENT_KEYS


class IncompatibleHardwareProfileError(ValueError):
    """Raised when efficiency observations cannot be compared."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _cpu_model() -> str:
    model = platform.processor().strip()
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            if result:
                model = result
        except (OSError, subprocess.SubprocessError):
            pass
    elif Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.casefold().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return model or "unknown-cpu"


def _memory_and_affinity() -> tuple[int | None, int, list[str]]:
    warnings: list[str] = []
    logical = os.cpu_count() or 1
    permitted = logical
    total_memory: int | None = None
    try:
        import psutil

        total_memory = int(psutil.virtual_memory().total)
        try:
            affinity = psutil.Process().cpu_affinity()
            if affinity:
                permitted = len(affinity)
        except (AttributeError, psutil.Error):
            warnings.append("cpu_affinity_unavailable")
    except ImportError:
        warnings.extend(("total_memory_unavailable", "cpu_affinity_unavailable"))
    return total_memory, permitted, warnings


def _accelerators() -> tuple[list[dict[str, Any]], str | None, list[str]]:
    warnings: list[str] = []
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return [], None, ["accelerator_inventory_unavailable"]
    accelerators: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            warnings.append("accelerator_inventory_parse_failure")
            continue
        name, memory_mib, driver = fields
        try:
            memory_bytes: int | None = int(float(memory_mib) * 1024 * 1024)
        except ValueError:
            memory_bytes = None
            warnings.append("accelerator_memory_unavailable")
        accelerators.append(
            {
                "backend": "nvidia-cuda",
                "name": name,
                "memory_total_bytes": memory_bytes,
                "driver_version": driver or None,
            }
        )
    cuda_version: str | None = None
    try:
        overview = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True, timeout=10
        ).stdout
        match = re.search(r"CUDA(?: UMD)? Version:\s*([^|\s]+)", overview)
        if match:
            cuda_version = match.group(1)
        else:
            warnings.append("cuda_runtime_version_unavailable")
    except (OSError, subprocess.SubprocessError):
        warnings.append("cuda_runtime_version_unavailable")
    return accelerators, cuda_version, warnings


def capture_hardware_profile(*, profile_id: str | None = None) -> dict[str, Any]:
    """Capture a host-safe profile; this alone never grants Official eligibility."""

    total_memory, permitted_count, warnings = _memory_and_affinity()
    accelerators, cuda_version, accelerator_warnings = _accelerators()
    warnings.extend(accelerator_warnings)
    thread_limits = {key: os.environ[key] for key in MATERIAL_ENVIRONMENT_KEYS if "THREAD" in key and key in os.environ}
    material = {
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_count": os.cpu_count() or 1,
            "permitted_count": permitted_count,
            "thread_limits": thread_limits,
        },
        "memory": {"total_bytes": total_memory},
        "accelerators": accelerators,
        "runtime": {
            "cuda_version": cuda_version,
            "material_environment": {
                key: os.environ[key]
                for key in MATERIAL_ENVIRONMENT_KEYS
                if key in os.environ and key in {"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"}
            },
        },
    }
    comparison_key = content_fingerprint(material)
    resolved_id = profile_id or f"observed-{comparison_key[:12]}"
    profile = {
        "hardware_profile_schema_version": "1.0.0",
        "profile_id": resolved_id,
        "profile_version": "1.0.0",
        "profile_origin": "declared-observed" if profile_id else "observed",
        "captured_at": _utc_timestamp(),
        **material,
        "comparison_key": comparison_key,
        "official_efficiency_eligible": False,
        "warning_codes": sorted(set(warnings)),
    }
    validate_instance("hardware-profile", profile)
    return profile


def load_hardware_profile(path: str | Path) -> dict[str, Any]:
    profile = read_json(path)
    if not isinstance(profile, dict):
        raise TypeError("Hardware profile must be a JSON object")
    validate_instance("hardware-profile", profile)
    return profile


def assert_efficiency_compatible(profiles_or_records: Iterable[dict[str, Any]]) -> str:
    """Return the shared comparison key or reject cross-profile efficiency use."""

    keys: set[str] = set()
    for item in profiles_or_records:
        key = item.get("comparison_key") or item.get("hardware_comparison_key")
        if not isinstance(key, str):
            raise IncompatibleHardwareProfileError("Every efficiency record requires a hardware comparison key")
        keys.add(key)
    if not keys:
        raise IncompatibleHardwareProfileError("At least one hardware profile is required")
    if len(keys) != 1:
        raise IncompatibleHardwareProfileError(
            "Efficiency records from different hardware profiles cannot be ranked or aggregated"
        )
    return next(iter(keys))
