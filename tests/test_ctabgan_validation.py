from __future__ import annotations

import hashlib
import json
from pathlib import Path

from standardized_tabular_diffusion.registry import AdapterValidationLevel, get_adapter_spec
from standardized_tabular_diffusion.validation import ctabgan

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
WINDOWS_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "ctabgan" / "windows-v2-real-function-17fc74e.json"
WINDOWS_EVIDENCE_SHA256 = "22144b71cf3eea455ba77631553377fe769b5fd543a125073977fa2f778898fa"


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


def test_registry_pins_classification_only_official_source_with_retained_linux_evidence() -> None:
    spec = get_adapter_spec("ctab-gan")

    assert spec.upstream_repository == ctabgan.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == ctabgan.UPSTREAM_COMMIT
    assert spec.modification_status == "adapter-only"
    assert spec.install_extra == "ctab-gan"
    assert spec.task_types == ("classification",)
    assert spec.validation_level is AdapterValidationLevel.NATIVE_PARITY_VALIDATED
    assert spec.revision_status == "pinned-official-source-native-parity-validated"
    assert "docs/evidence/ctabgan/native-parity-run-30930939961.json" in spec.evidence_records
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"


def test_retained_ctabgan_windows_v2_evidence_is_exact_and_complete() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "17fc74e2f9be8a507ec1f921bb3b509881937154"
    assert evidence["entry"]["device"] == "cpu"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["hardware"]["torch"] == "2.3.0+cpu"
    assert evidence["environment"]["hardware"]["cuda_available"] is False
    assert evidence["environment"]["pip_check"]["status"] == "pass"
    assert evidence["environment_lock"]["sha256"] == (
        "d89bb9ceec70c0de16d38b5652a57dbff58f097925a524d67deb0cb2677b60fb"
    )
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [16, 16]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True
    assert evidence["tracked_repository_unchanged"] is True
    assert evidence["central_evaluation"]["status"] == "pass"
    assert evidence["central_evaluation"]["validation"]["pending_files"] == 0

    component = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["ctab-gan"]
    windows = component["windows_real_function"]
    assert windows["level"] == "minimal-real-passed"
    assert windows["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert windows["repository_commit"] == evidence["repository_commit"]
    assert windows["source_code_modified"] is False
    assert "docs/evidence/ctabgan/windows-v2-real-function-17fc74e.json" in get_adapter_spec(
        "ctab-gan"
    ).evidence_records
