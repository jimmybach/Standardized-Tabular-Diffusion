from __future__ import annotations

import os
from pathlib import Path

import pytest

from standardized_tabular_diffusion.models.tabula import TabulaAdapter
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.upstream_sources import load_source_manifest
from standardized_tabular_diffusion.validation import tabula as tabula_validation

pytestmark = pytest.mark.core


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
