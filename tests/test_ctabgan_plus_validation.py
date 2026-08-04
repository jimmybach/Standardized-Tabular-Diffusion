from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.registry import AdapterValidationLevel, get_adapter_spec
from standardized_tabular_diffusion.validation import ctabgan_plus

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_identity_and_cases_are_frozen() -> None:
    assert ctabgan_plus.PROTOCOL_ID == "ctabgan-plus-native-parity-v1"
    assert ctabgan_plus.UPSTREAM_COMMIT == "6a6f90188cca3dac2c533fd5e8e7f20de074365b"
    assert ctabgan_plus.VARIANTS == ("classification", "regression")
    assert ctabgan_plus.SEED_CASES == (0, 19, 73)
    assert ctabgan_plus.TRAINING_PARAMETERS["epochs"] == 1
    assert ctabgan_plus.SAMPLE_ROWS == 13


def test_fixtures_cover_mixed_type_classification_and_regression() -> None:
    classification = ctabgan_plus._fixture_frame("classification")
    regression = ctabgan_plus._fixture_frame("regression")

    assert list(classification.columns) == ["continuous", "count", "segment", "target"]
    assert set(classification["target"]) == {"no", "yes"}
    assert regression["target"].dtype.kind == "f"
    assert not classification.isna().any().any()
    assert not regression.isna().any().any()


def test_legacy_semantically_modified_source_is_not_distributed() -> None:
    legacy_root = REPO_ROOT / "TabDDPM-main" / "CTAB-GAN-Plus"
    assert not legacy_root.exists() or not any(path.is_file() for path in legacy_root.rglob("*"))


def test_registry_pins_official_source_without_release_claim() -> None:
    spec = get_adapter_spec("ctab-gan-plus")

    assert spec.upstream_repository == ctabgan_plus.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == ctabgan_plus.UPSTREAM_COMMIT
    assert spec.modification_status == "adapter-only"
    assert spec.install_extra == "ctab-gan-plus"
    assert spec.validation_level is AdapterValidationLevel.NATIVE_PARITY_VALIDATED
    assert spec.revision_status == "pinned-official-source-native-parity-validated"
    assert "docs/evidence/ctabgan-plus/native-parity-run-30926267432.json" in spec.evidence_records
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"
