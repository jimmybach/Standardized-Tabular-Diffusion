from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.registry import AdapterValidationLevel, get_adapter_spec
from standardized_tabular_diffusion.validation import ctabgan

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_identity_and_cases_are_frozen() -> None:
    assert ctabgan.PROTOCOL_ID == "ctabgan-native-parity-v1"
    assert ctabgan.UPSTREAM_COMMIT == "73d4e315a2a51cf16c97ed8a00d2dad456cfce8a"
    assert ctabgan.VARIANTS == ("binary-classification", "multiclass-classification")
    assert ctabgan.SEED_CASES == (0, 19, 73)
    assert ctabgan.TRAINING_PARAMETERS["epochs"] == 1
    assert ctabgan.SAMPLE_ROWS == 13


def test_fixtures_cover_mixed_type_binary_and_multiclass_targets() -> None:
    binary = ctabgan._fixture_frame("binary-classification")
    multiclass = ctabgan._fixture_frame("multiclass-classification")

    assert list(binary.columns) == ["continuous", "count", "segment", "target"]
    assert set(binary["target"]) == {"no", "yes"}
    assert set(multiclass["target"]) == {"class-a", "class-b", "class-c", "class-d"}
    assert not binary.isna().any().any()
    assert not multiclass.isna().any().any()


def test_legacy_semantic_fork_is_removed_and_selected_official_source_is_present() -> None:
    source_root = REPO_ROOT / "TabDDPM-main" / "CTAB-GAN"
    assert (source_root / "LICENSE").is_file()
    assert (source_root / "License.txt").is_file()
    assert not (source_root / "columns.json").exists()
    assert not (source_root / "pipeline_ctabgan.py").exists()
    assert not (source_root / "train_sample_ctabgan.py").exists()
    assert not (source_root / "tune_ctabgan.py").exists()
    assert not (source_root / "model" / "eval" / "evaluation.py").exists()


def test_registry_pins_classification_only_official_source_pending_linux_evidence() -> None:
    spec = get_adapter_spec("ctab-gan")

    assert spec.upstream_repository == ctabgan.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == ctabgan.UPSTREAM_COMMIT
    assert spec.modification_status == "adapter-only"
    assert spec.install_extra == "ctab-gan"
    assert spec.task_types == ("classification",)
    assert spec.validation_level is AdapterValidationLevel.ADAPTER_COMPLETE
    assert spec.revision_status == "pinned-official-source-pending-native-parity"
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"
