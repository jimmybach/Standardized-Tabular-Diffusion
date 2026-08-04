from __future__ import annotations

import os
from pathlib import Path

import pytest

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
