from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion.validation import p4_global_source as validation

pytestmark = [pytest.mark.evaluation, pytest.mark.source_parity]


def _available_source() -> Path:
    if configured := os.environ.get("P4_TABEVAL_SOURCE_PATH"):
        return Path(configured)
    local = (
        Path(__file__).resolve().parents[2]
        / "research_inputs/evaluation/code/upstream/tabeval/src/tabeval/metrics/eval_structure.py"
    )
    if not local.exists():
        pytest.skip("The immutable TabEval research snapshot is not present")
    return local


def test_source_manifest_distinguishes_official_source_from_reconstructed_runtime() -> None:
    manifest = validation._manifest()
    assert manifest["revision"] == "dba19a4ee7aa391621cbeb464609285fd515dece"
    assert manifest["source_sha256"] == "1861a7573949e50b360c722f4e73110f2c3d014c412693b66c704d070df62743"
    assert manifest["upstream_dependency_disclosure"] == {
        "status": "not-reproducibly-locked",
        "autogluon_declared": False,
        "xgboost_constraint": "unbounded",
        "tabpfn_constraint": "unbounded",
    }
    assert manifest["approved_pilot_runtime"]["status"] == "benchmark-approved-not-upstream-official"


def test_source_attestation_is_line_ending_stable(tmp_path: Path) -> None:
    source = _available_source()
    verified = validation.verify_tabeval_source(source)
    assert verified["source_sha256"] == validation._manifest()["source_sha256"]

    crlf_copy = tmp_path / "eval_structure.py"
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    crlf_copy.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert validation.verify_tabeval_source(crlf_copy)["source_sha256"] == verified["source_sha256"]


def test_source_attestation_fails_closed_on_one_byte_change(tmp_path: Path) -> None:
    source = _available_source()
    changed = tmp_path / "eval_structure.py"
    changed.write_bytes(source.read_bytes() + b"\n# changed\n")
    with pytest.raises(validation.P4GlobalSourceValidationError, match="source hash"):
        validation.verify_tabeval_source(changed)


def test_exact_source_file_imports_with_real_selected_autogluon() -> None:
    pytest.importorskip("autogluon.tabular")
    module = validation.load_tabeval_source_module(_available_source())
    assert module.UtilityPerFeature.timestamp(None) == "2025-08-09"
    assert module.CustomTabPFNModel.__module__ == validation.SOURCE_MODULE_NAME
    assert sys.modules[validation.SOURCE_MODULE_NAME] is module


def test_checkpoint_manifest_has_immutable_revisions_and_sha256() -> None:
    checkpoints = validation._manifest()["tabpfn_checkpoints"]
    for checkpoint in checkpoints.values():
        assert len(checkpoint["revision"]) == 40
        assert len(checkpoint["sha256"]) == 64
        assert checkpoint["filename"].endswith(".ckpt")


def test_manifest_is_canonical_json_resource() -> None:
    resource = (
        Path(validation.__file__).resolve().parents[1]
        / "resources/evaluation/upstream/tabeval-p4-source.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    assert payload == validation._manifest()
