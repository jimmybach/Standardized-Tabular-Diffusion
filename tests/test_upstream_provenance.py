from __future__ import annotations

import json
from pathlib import Path

from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def _load_source_lock() -> dict[str, object]:
    return json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))


def test_source_lock_matches_primary_adapter_registry() -> None:
    payload = _load_source_lock()
    components = payload["components"]

    assert isinstance(components, dict)
    assert set(components) == {"tabddpm", "tabdiff", "tabsyn"}
    for component_id, component in components.items():
        assert isinstance(component, dict)
        spec = get_adapter_spec(component_id)
        assert spec.upstream_repository == component["authoritative_repository"]
        assert spec.upstream_revision == component["upstream_commit"]
        assert list(spec.patch_set_ids) == [patch["patch_set_id"] for patch in component["patch_sets"]]
        assert (REPO_ROOT / component["license_path"]).is_file()


def test_source_lock_patch_ids_are_unique_and_classified() -> None:
    payload = _load_source_lock()
    components = payload["components"]
    patch_ids: list[str] = []

    assert isinstance(components, dict)
    for component in components.values():
        assert isinstance(component, dict)
        for patch in component["patch_sets"]:
            assert patch["classification"] in {"adapter-only", "compatibility-patched", "semantic-patched"}
            assert patch["status"] != "approved-parity-validated"
            patch_ids.append(patch["patch_set_id"])

    assert len(patch_ids) == len(set(patch_ids))


def test_audited_primary_adapters_fail_closed_for_release_claims() -> None:
    for model_id in ("tabddpm", "tabdiff", "tabsyn"):
        spec = get_adapter_spec(model_id)
        assert spec.upstream_revision is not None
        assert spec.benchmark_track == "experimental"
        assert spec.support_level == "unsupported"
        assert "native-parity-validated" not in spec.validation_level.value


def test_removed_unverified_checkpoints_are_recorded_and_absent() -> None:
    payload = _load_source_lock()
    components = payload["components"]

    assert isinstance(components, dict)
    tabsyn = components["tabsyn"]
    assert isinstance(tabsyn, dict)
    artifacts = tabsyn["removed_unverified_artifacts"]
    assert len(artifacts) == 7
    assert sum(artifact["bytes"] for artifact in artifacts) == 93_479_868
    for artifact in artifacts:
        assert len(artifact["sha256"]) == 64
        assert not (REPO_ROOT / artifact["original_path"]).exists()
