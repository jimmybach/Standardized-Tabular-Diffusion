from __future__ import annotations

from typing import Any

import pytest

from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint


@pytest.fixture
def hardware_profile() -> dict[str, Any]:
    material = {
        "operating_system": {"system": "FixtureOS", "release": "1", "version": "1.0", "machine": "x86_64"},
        "python": {"implementation": "CPython", "version": "3.11.0"},
        "cpu": {"model": "Fixture CPU", "logical_count": 4, "permitted_count": 4, "thread_limits": {}},
        "memory": {"total_bytes": 8 * 1024**3},
        "accelerators": [],
        "runtime": {"cuda_version": None, "material_environment": {}},
    }
    return {
        "hardware_profile_schema_version": "1.0.0",
        "profile_id": "fixture-cpu",
        "profile_version": "1.0.0",
        "profile_origin": "declared-observed",
        "captured_at": "2026-08-13T00:00:00Z",
        **material,
        "comparison_key": content_fingerprint(material),
        "official_efficiency_eligible": False,
        "warning_codes": [],
    }


@pytest.fixture
def software_profile() -> dict[str, Any]:
    repository = {"commit": "fixture", "dirty": False, "patch_sha256": "2" * 64}
    material = {
        "python": {"implementation": "CPython", "version": "3.11.0"},
        "platform": {"system": "FixtureOS", "release": "1", "machine": "x86_64"},
        "packages": [],
        "material_environment": {},
        "repository": repository,
        "locale": "fixture",
    }
    return {
        "software_profile_schema_version": "1.0.0",
        **material,
        "fingerprint": content_fingerprint(material),
        "code_fingerprint": content_fingerprint(repository),
    }
