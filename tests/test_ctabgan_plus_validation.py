from __future__ import annotations

import hashlib
import json
from pathlib import Path

from standardized_tabular_diffusion.registry import AdapterValidationLevel, get_adapter_spec
from standardized_tabular_diffusion.validation import ctabgan_plus

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
WINDOWS_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "ctabgan-plus" / "windows-v2-real-function-17fc74e.json"
WINDOWS_EVIDENCE_SHA256 = "ce7834d0abab284f026230fdcd64bb91f9738b3d65fc77ed01c01fbe1e534127"


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


def test_retained_ctabgan_plus_windows_v2_evidence_is_exact_and_complete() -> None:
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
        "5eca4d5b6f590ab4916a3cbb829a8fc5aa25dc4b24834e6120c74a4815f52949"
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

    component = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["ctab-gan-plus"]
    windows = component["windows_real_function"]
    assert windows["level"] == "minimal-real-passed"
    assert windows["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert windows["repository_commit"] == evidence["repository_commit"]
    assert windows["source_code_modified"] is False
    assert "docs/evidence/ctabgan-plus/windows-v2-real-function-17fc74e.json" in get_adapter_spec(
        "ctab-gan-plus"
    ).evidence_records
