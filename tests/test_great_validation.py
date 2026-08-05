from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from standardized_tabular_diffusion.models.great import (
    GREAT_PACKAGE_VERSION,
    GREAT_UPSTREAM_COMMIT,
    GREAT_WHEEL_SHA256,
    GReaTAdapter,
    verify_great_distribution,
)
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.validation import great as great_validation

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "great" / "native-parity-run-30974574472.json"
EVIDENCE_SHA256 = "c8b70277dc43eac109533e82e970a4fceecc742c13f420a20838e15cdf16b2bf"
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def test_great_authority_and_protocol_are_locked() -> None:
    spec = get_adapter_spec("great")
    assert spec.source_authority == "method-author"
    assert spec.distribution_form == "package"
    assert spec.modification_status == "adapter-only"
    assert spec.install_extra == "great"
    assert spec.upstream_revision == GREAT_UPSTREAM_COMMIT
    assert great_validation.PROTOCOL_ID == GReaTAdapter.protocol_id
    assert great_validation.SEEDS == (0, 19, 73)
    assert GREAT_PACKAGE_VERSION == "0.0.14"
    assert GREAT_WHEEL_SHA256 == "4f6384ec4a736177ae2d1e6146951cfdfc764b1cc041ae5c2b155a99dd18cb74"


def test_great_locked_wheel_when_provided() -> None:
    wheel = os.environ.get("GREAT_WHEEL_PATH")
    if wheel is None:
        pytest.skip("GREAT_WHEEL_PATH is provided by the authoritative validation workflow")
    result = great_validation._verify_wheel(Path(wheel))
    assert result["wheel_files_verified"] == 19
    assert result["package_files_verified"] == 14
    assert verify_great_distribution()["version"] == GREAT_PACKAGE_VERSION


def test_great_integrity_manifest_rejects_executable_files(tmp_path: Path) -> None:
    model_root = tmp_path / "great_model"
    model_root.mkdir()
    (model_root / "model.bin").write_bytes(b"unsafe")
    GReaTAdapter._write_integrity_manifest(model_root)
    with pytest.raises(ValueError, match="forbidden"):
        GReaTAdapter._validate_safe_model_root(model_root)


def test_great_retained_native_parity_evidence_is_exact() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert len(evidence["cases"]) == 3
    assert evidence["result_summary"]["state_tensors_exact"] is True
    assert evidence["result_summary"]["sample_csv_bytes_exact"] is True
    validation = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["great"]["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["workflow_run_id"] == 30974574472
    assert validation["artifact"]["evidence_file_sha256"] == EVIDENCE_SHA256
    assert get_adapter_spec("great").validation_level.value == "native-parity-validated"
