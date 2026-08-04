from __future__ import annotations

import hashlib
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
    assert set(components) == {
        "ctab-gan-plus",
        "ctgan",
        "nrgboost",
        "smote",
        "tabddpm",
        "tabdiff",
        "tabsyn",
        "tvae",
    }
    for component_id, component in components.items():
        assert isinstance(component, dict)
        spec = get_adapter_spec(component_id)
        assert spec.upstream_repository == component["authoritative_repository"]
        assert spec.upstream_revision == component["upstream_commit"]
        assert list(spec.patch_set_ids) == [patch["patch_set_id"] for patch in component["patch_sets"]]
        if component["distribution_form"] == "source":
            assert (REPO_ROOT / component["license_path"]).is_file()
        elif component["distribution_form"] == "package":
            assert component["package_lock"]["sha256"]
            assert component["license_url"].startswith("https://")
        else:
            assert component["distribution_form"] == "source-on-demand"
            assert component["license"] == "NONE-DECLARED"
            assert component["license_review"]["redistribution_status"] == "not-authorized"
            assert (REPO_ROOT / component["source_manifest"]).is_file()


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
    evidence_paths = {
        "ctgan": "docs/evidence/ctgan/native-parity-run-30910275922.json",
        "nrgboost": "docs/evidence/nrgboost/native-parity-run-30922326384.json",
        "smote": "docs/evidence/smote/native-parity-run-30918785254.json",
        "tabddpm": "docs/evidence/tabddpm/native-parity-run-30863212268.json",
        "tabdiff": "docs/evidence/tabdiff/native-parity-run-30866879879.json",
        "tabsyn": "docs/evidence/tabsyn/native-parity-run-30871758645.json",
        "tvae": "docs/evidence/tvae/native-parity-run-30913867621.json",
    }
    for model_id in ("ctgan", "nrgboost", "smote", "tabddpm", "tabdiff", "tabsyn", "tvae"):
        spec = get_adapter_spec(model_id)
        assert spec.upstream_revision is not None
        assert spec.validation_level.value == "native-parity-validated"
        assert spec.modification_status == "adapter-only"
        assert spec.patch_set_ids == ()
        assert evidence_paths[model_id] in spec.evidence_records
        assert spec.benchmark_track == "experimental"
        assert spec.support_level == "unsupported"

    tabddpm = get_adapter_spec("tabddpm")
    assert tabddpm.revision_status == "pinned-complete-native-parity-validated"

    smote = get_adapter_spec("smote")
    assert smote.revision_status == "pinned-canonical-package-native-parity-validated"
    assert smote.reproduction_target == "classical-oversampling-reference"

    # Restoring the primary TabSyn path does not audit its separately vendored baselines.
    assert get_adapter_spec("codi").modification_status == "compatibility-patched"


def test_tabddpm_source_lock_records_native_parity_without_overclaiming() -> None:
    payload = _load_source_lock()
    tabddpm = payload["components"]["tabddpm"]
    assert isinstance(tabddpm, dict)
    validation = tabddpm["validation"]

    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30863212268
    assert validation["result_summary"]["seed_cases_passed"] == 3
    assert tabddpm["official_eligibility"] == "pending-separate-official-track-review"


def test_ctgan_package_lock_is_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    ctgan = payload["components"]["ctgan"]
    assert isinstance(ctgan, dict)

    assert ctgan["distribution_form"] == "package"
    assert ctgan["license"] == "BUSL-1.1"
    assert ctgan["package_lock"] == {
        "filename": "ctgan-0.12.1-py3-none-any.whl",
        "name": "ctgan",
        "pypi_url": "https://pypi.org/project/ctgan/0.12.1/",
        "sha256": "38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18",
        "trusted_publishing_source_commit": "826da23f8f9385ad15fd206ecad691e04cb0ccdc",
        "version": "0.12.1",
    }
    validation = ctgan["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30910275922
    assert validation["result_summary"]["seed_cases_passed"] == 3
    assert validation["artifact"]["evidence_file_sha256"] == (
        "748501c8671c272a1e5d54c85fdb6550182d0e5578d550a3ca7681cc712f4570"
    )
    assert str(ctgan["official_eligibility"]).startswith("blocked-pending-license")


def test_tvae_package_lock_and_retained_validation_are_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    tvae = payload["components"]["tvae"]
    assert isinstance(tvae, dict)

    assert tvae["distribution_form"] == "package"
    assert tvae["license"] == "BUSL-1.1"
    assert tvae["package_lock"] == payload["components"]["ctgan"]["package_lock"]
    validation = tvae["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30913867621
    assert validation["result_summary"]["seed_cases_passed"] == 3
    assert validation["artifact"]["evidence_file_sha256"] == (
        "ad539ffdb637084a25dc3ab4ec5d54374ff6831525ca63adca2cfa48c3ef95f7"
    )
    assert str(tvae["official_eligibility"]).startswith("blocked-pending-license")


def test_nrgboost_package_lock_and_retained_validation_are_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    nrgboost = payload["components"]["nrgboost"]
    assert isinstance(nrgboost, dict)

    assert nrgboost["distribution_form"] == "package"
    assert nrgboost["license"] == "MIT"
    assert nrgboost["package_lock"] == {
        "filename": "nrgboost-0.0.3-cp311-cp311-manylinux_2_28_x86_64.whl",
        "name": "nrgboost",
        "pypi_url": "https://pypi.org/project/nrgboost/0.0.3/",
        "sha256": "dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb",
        "trusted_publishing_source_commit": "feef73a3edb20b911c2f7214b13f810909ef20ad",
        "version": "0.0.3",
        "wheel_license_sha256": "3693dc7c451fe74ffead14c00964ac00a1123242c9fc3d8cb13c8fef3091b945",
    }
    validation = nrgboost["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30922326384
    assert validation["repository_commit"] == "4cd32c8beedd116c6385463d41cf9cba8b1d5438"
    assert validation["result_summary"]["parity_cases_passed"] == 6
    assert validation["result_summary"]["checkpoint_bytes_exact"] is True
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert validation["artifact"]["evidence_file_sha256"] == (
        "5958c67261e8c25e60d58891efd5d27f8e8bb6439852862064e831f630cbe56c"
    )
    evidence_path = REPO_ROOT / validation["artifact"]["permanent_evidence_path"]
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == validation["artifact"][
        "evidence_file_sha256"
    ]
    spec = get_adapter_spec("nrgboost")
    assert spec.validation_level.value == "native-parity-validated"
    assert spec.revision_status == "pinned-canonical-package-native-parity-validated"
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"


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
