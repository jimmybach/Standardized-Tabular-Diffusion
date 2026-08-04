from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from standardized_tabular_diffusion.registry import AdapterValidationLevel, get_adapter_spec
from standardized_tabular_diffusion.validation.realtabformer import (
    FIT_CONTROLS,
    SAMPLE_ROWS,
    SEED_CASES,
    TRAIN_ROWS,
    VARIANTS,
    WHEEL_FILENAME,
    _load_manifest,
    _verify_wheel,
    _write_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "realtabformer" / "native-parity-run-30950369908.json"


def test_locked_wheel_manifest_and_protocol_scope_are_exact() -> None:
    manifest = _load_manifest(REPO_ROOT)

    assert manifest["model_id"] == "realtabformer"
    assert manifest["package"]["filename"] == WHEEL_FILENAME
    assert manifest["package"]["record_hashed_files"] == len(manifest["installed_files"]) == 16
    assert manifest["wheel_source_comparison"] == {
        "shared_source_files": 11,
        "exact_shared_source_files": 11,
        "wheel_only_file": "realtabformer/rtf_tokenizer.py",
        "wheel_only_file_bytes": 0,
    }
    assert SEED_CASES == (0, 19, 73)
    assert VARIANTS == ("binary", "multiclass", "regression")
    assert FIT_CONTROLS["n_critic"] == 0
    assert SAMPLE_ROWS == 7


@pytest.mark.parametrize("variant", VARIANTS)
def test_parity_fixture_covers_declared_types_without_missing_values(tmp_path: Path, variant: str) -> None:
    dataset_spec, record = _write_fixture(tmp_path, variant)

    assert dataset_spec.train_data_path is not None
    assert dataset_spec.train_data_path.is_file()
    assert dataset_spec.column_names == ["first", "second", "group", "target"]
    assert dataset_spec.numerical_columns == ["first", "second"]
    assert dataset_spec.categorical_columns == ["group"]
    assert dataset_spec.target_columns == ["target"]
    assert record["training_rows"] == TRAIN_ROWS
    assert record["missing_values"] == 0
    assert dataset_spec.task_type == ("regression" if variant == "regression" else "classification")


def test_downloaded_official_wheel_matches_every_locked_member() -> None:
    configured = os.environ.get("REALTABFORMER_WHEEL_PATH")
    local_audit = REPO_ROOT / ".cache" / "realtabformer-audit-0.2.4" / WHEEL_FILENAME
    wheel_path = Path(configured) if configured else local_audit
    if not wheel_path.is_file():
        pytest.skip("locked official wheel is supplied only in the formal validation job")

    observed = _verify_wheel(REPO_ROOT, wheel_path)

    assert observed["filename"] == WHEEL_FILENAME
    assert observed["archive_members"] == 17
    assert observed["record_rows"] == 17
    assert observed["record_hashed_files"] == 16


def test_retained_native_parity_evidence_is_exact_and_conservatively_scoped() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)
    spec = get_adapter_spec("realtabformer")
    source_lock = json.loads(
        (REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json").read_text(
            encoding="utf-8"
        )
    )["components"]["realtabformer"]

    assert hashlib.sha256(evidence_bytes).hexdigest() == (
        "0c6047efc3463aa21fa4b2e6aeed66858cbc29bfd5a9e836f330d975ec0cfa07"
    )
    assert evidence_bytes.endswith(b"\n")
    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "fb2f03dd579bb4d1847fa18395696ed698c8ce58"
    assert evidence["package"]["wheel"]["sha256"] == (
        "852436c5c82a0bf470ca7e9063e5a4f3e250b3ff5b9c8f6c50113c1e9ba76486"
    )
    assert evidence["package"]["installed_distribution"]["installed_files_verified"] == 16
    assert evidence["package_unchanged_after_validation"] is True
    assert len(evidence["cases"]) == 9
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(case["comparisons"]["checkpoint"]["tensors_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["checkpoint"]["file_bytes_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["raw_samples_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["sample_bytes_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["saved_config_semantics_exact"] for case in evidence["cases"])
    assert evidence["validated_scope"] == {
        "model_type": "tabular",
        "relational_mode": "outside-the-current-single-table-contract",
        "sensitivity_stopping": "not-native-parity-validated",
        "tasks": ["binary-classification", "multiclass-classification", "regression"],
        "training_path": "official fit with n_critic=0",
    }
    assert spec.validation_level is AdapterValidationLevel.NATIVE_PARITY_VALIDATED
    assert spec.modification_status == "compatibility-patched"
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"
    validation = source_lock["validation"]
    assert validation["workflow_run_id"] == 30950369908
    assert validation["pull_request_head_commit"] == "7db46e00452ce5cc25d28d8b484c9d6ee14de5b3"
    assert validation["result_summary"]["parity_cases_passed"] == 9
    assert validation["result_summary"]["parity_cases_total"] == 9
    assert validation["artifact"]["evidence_file_sha256"] == (
        "0c6047efc3463aa21fa4b2e6aeed66858cbc29bfd5a9e836f330d975ec0cfa07"
    )
    assert validation["artifact"]["downloaded_evidence_sha256"] == validation["artifact"]["evidence_file_sha256"]
