from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.models.tabula import TabulaAdapter
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.upstream_sources import load_source_manifest
from standardized_tabular_diffusion.validation import tabula as tabula_validation

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabula" / "native-parity-run-30974574505.json"
EVIDENCE_SHA256 = "35b9c8bdab2828763a72fe3fa55aa6c9fa6308dc36740217d6479c296da3ca1c"
WINDOWS_EVIDENCE_PATH = (
    REPO_ROOT / "docs" / "evidence" / "tabula" / "windows-v2-real-function-8d72ee8.json"
)
WINDOWS_EVIDENCE_SHA256 = "8bfa58cfde52ab0f4b5d5d61ea42d4b7e1d39818444da550302667534d5d13ec"
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def test_tabula_method_author_source_lock_and_protocol() -> None:
    manifest = load_source_manifest("tabula")
    spec = get_adapter_spec("tabula")
    assert manifest["upstream_commit"] == TabulaAdapter.upstream_commit
    assert len(manifest["runtime_files"]) == 6
    assert manifest["license"]["license_file_present"] is False
    assert spec.source_authority == "method-author"
    assert spec.modification_status == "adapter-only"
    assert spec.install_extra == "tabula"
    assert tabula_validation.PROTOCOL_ID == TabulaAdapter.protocol_id
    assert tabula_validation.SEEDS == (0, 19, 73)


def test_tabula_locked_archive_when_provided() -> None:
    archive = os.environ.get("TABULA_ARCHIVE_PATH")
    if archive is None:
        pytest.skip("TABULA_ARCHIVE_PATH is provided by the authoritative validation workflow")
    result = tabula_validation._verify_archive(Path(archive))
    assert result["sha256"] == tabula_validation.ARCHIVE_SHA256
    assert result["source_commit"] == TabulaAdapter.upstream_commit


def test_tabula_integrity_manifest_rejects_tampering(tmp_path: Path) -> None:
    model_root = tmp_path / "tabula_model"
    model_root.mkdir()
    state = model_root / "tabula-state.json"
    state.write_text("{}", encoding="utf-8")
    TabulaAdapter._write_integrity_manifest(model_root)
    state.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        TabulaAdapter._validate_safe_model_root(model_root)


def test_tabula_exact_row_boundary_repeats_filtered_official_batches() -> None:
    class FilteredOfficialSampler:
        def __init__(self) -> None:
            self.requests: list[int] = []
            self.outputs = [
                pd.DataFrame({"value": ["a", "b"]}),
                pd.DataFrame({"value": []}),
                pd.DataFrame({"value": ["c"]}),
                pd.DataFrame({"value": ["d", "e"]}),
            ]

        def sample(self, *, n_samples: int, **_kwargs: object) -> pd.DataFrame:
            self.requests.append(n_samples)
            return self.outputs.pop(0)

    sampler = FilteredOfficialSampler()
    result = TabulaAdapter._sample_exact_rows(
        sampler,
        requested=5,
        start_col="target",
        start_dist={"0": 0.5, "1": 0.5},
        temperature=0.5,
        k=5,
        max_length=64,
        device="cpu",
        max_empty_batches=3,
    )
    assert sampler.requests == [5, 3, 3, 2]
    assert result["value"].tolist() == ["a", "b", "c", "d", "e"]


def test_tabula_exact_row_boundary_rejects_repeated_empty_batches() -> None:
    class EmptyOfficialSampler:
        @staticmethod
        def sample(**_kwargs: object) -> pd.DataFrame:
            return pd.DataFrame({"value": []})

    with pytest.raises(RuntimeError, match="repeatedly returned zero"):
        TabulaAdapter._sample_exact_rows(
            EmptyOfficialSampler(),
            requested=1,
            start_col="",
            start_dist=None,
            temperature=0.5,
            k=1,
            max_length=64,
            device="cpu",
            max_empty_batches=2,
        )


def test_tabula_retained_native_parity_evidence_is_exact() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert len(evidence["cases"]) == 3
    assert evidence["result_summary"]["state_tensors_exact"] is True
    assert evidence["result_summary"]["sample_csv_bytes_exact"] is True
    validation = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tabula"]["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["workflow_run_id"] == 30974574505
    assert validation["artifact"]["evidence_file_sha256"] == EVIDENCE_SHA256
    assert get_adapter_spec("tabula").validation_level.value == "native-parity-validated"


def test_tabula_windows_real_function_evidence_is_complete_and_immutable() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["model_id"] == "tabula"
    assert evidence["repository_commit"] == "8d72ee85e85712c473d7fcf9b7eda3a3cbf9cf62"
    assert evidence["adapter"]["upstream_revision"] == TabulaAdapter.upstream_commit
    hardware = evidence["environment"]["hardware"]
    assert hardware["gpu"] == "NVIDIA GeForce RTX 5080"
    assert hardware["torch"] == "2.8.0+cu128"
    assert hardware["cuda_available"] is True
    assert evidence["seed_outputs_distinct"] is True
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert all(sample["rows"] == 4 for sample in evidence["samples"])
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert len({sample["sha256"] for sample in evidence["samples"]}) == 2
    central = evidence["central_evaluation"]
    assert central["status"] == "pass"
    assert central["protocol"] == "p3-validity"
    assert central["finalization_status"] == "finalized"

    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    validation = source_lock["components"]["tabula"]["windows_real_function"]
    assert validation["status"] == "pass"
    assert validation["level"] == "minimal-real-passed"
    assert validation["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert validation["repository_commit"] == evidence["repository_commit"]
    assert validation["permanent_evidence_path"] in get_adapter_spec("tabula").evidence_records
