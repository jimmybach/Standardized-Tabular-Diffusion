from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.models.paper_gap_baselines import TabSDSAdapter
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.upstream_sources import load_source_manifest, validate_upstream_source
from standardized_tabular_diffusion.validation import tabsds as tabsds_validation

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabsds" / "native-parity-run-30974574593.json"
EVIDENCE_SHA256 = "11cfa96a3221944ebb6d423fdddf8660f278e7f6b108dff500fe39a1f9b07b66"
WINDOWS_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabsds" / "windows-v2-real-function-a5f83ae.json"
WINDOWS_EVIDENCE_SHA256 = "b5288ff5f0d0ea5556028ed9b859e8f666abe570e2e61cc0e69df8343370f4a3"
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def test_tabsds_method_author_source_lock_and_protocol() -> None:
    manifest = load_source_manifest("tabsds")
    spec = get_adapter_spec("tabsds")
    assert manifest["upstream_commit"] == TabSDSAdapter.upstream_commit
    assert len(manifest["runtime_files"]) == 2
    assert manifest["license"]["redistribution_status"] == "not-authorized"
    assert spec.source_authority == "method-author"
    assert spec.modification_status == "adapter-only"
    assert spec.install_extra == "tabsds"
    assert tabsds_validation.SEEDS == (0, 19, 73)
    assert tabsds_validation.VARIANTS == ("binary", "multiclass", "regression")


def test_tabsds_materialized_source_matches_lock() -> None:
    source_root = Path(".cache/upstream-sources/tabsds") / TabSDSAdapter.upstream_commit
    if not source_root.is_dir():
        pytest.skip("The checksum-locked source is materialized in the authoritative workflow")
    result = validate_upstream_source("tabsds", source_root)
    assert result["runtime_files_verified"] == 2


def test_tabsds_rejects_missing_values(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    pd.DataFrame({"x": [1.0, None], "target": ["a", "b"]}).to_csv(train, index=False)
    dataset = DatasetSpec(
        name="missing",
        task_type="classification",
        column_names=["x", "target"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["target"],
        metadata_path=tmp_path / "dataset.json",
        train_data_path=train,
    )
    with pytest.raises(ValueError, match="imputed"):
        TabSDSAdapter(tmp_path)._load_training_frame(dataset)


def test_tabsds_retained_native_parity_evidence_is_exact() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert len(evidence["cases"]) == 9
    assert evidence["result_summary"]["sample_csv_bytes_exact"] is True
    validation = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tabsds"]["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["workflow_run_id"] == 30974574593
    assert validation["artifact"]["evidence_file_sha256"] == EVIDENCE_SHA256
    assert get_adapter_spec("tabsds").validation_level.value == "native-parity-validated"


def test_tabsds_retained_windows_v2_evidence_is_exact_and_complete() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "a5f83ae4b7035ebe87bff0230160a77d80a21252"
    assert evidence["train"]["status"] == "pass"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["platform"].startswith("Windows-")
    assert evidence["environment"]["hardware"] == {
        "cuda_available": False,
        "cuda_runtime": None,
        "gpu": None,
        "torch": None,
    }
    assert evidence["environment_lock"]["sha256"] == (
        "d03d355188ff0037407241a8b4573d368d25b27ee1f82d1d85fe6252a0f524d9"
    )
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [32, 32]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True
    assert evidence["tracked_repository_unchanged"] is True
    assert evidence["central_evaluation"]["status"] == "pass"
    assert evidence["central_evaluation"]["finalization_status"] == "finalized"
    assert evidence["central_evaluation"]["validation"]["pending_files"] == 0

    component = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tabsds"]
    windows = component["windows_real_function"]
    assert windows["level"] == "minimal-real-passed"
    assert windows["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert windows["repository_commit"] == evidence["repository_commit"]
    assert windows["source_code_modified"] is False
    assert get_adapter_spec("tabsds").evidence_records == (
        "docs/UPSTREAM_SOURCE_AUDIT.md",
        "docs/TABSDS_VALIDATION.md",
        "docs/evidence/tabsds/native-parity-run-30974574593.json",
        "docs/evidence/tabsds/windows-v2-real-function-a5f83ae.json",
        "standardized_tabular_diffusion/resources/upstream/tabsds-source-manifest.json",
        "standardized_tabular_diffusion/resources/upstream/source-lock.json",
        ".github/workflows/tabsds-validation.yml",
    )
