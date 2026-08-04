from __future__ import annotations

import os
from pathlib import Path

import pytest

from standardized_tabular_diffusion.validation.tabularargn import (
    SAMPLE_ROWS,
    SEED_CASES,
    SOURCE_ARCHIVE_FILENAME,
    TRAIN_ROWS,
    VARIANTS,
    WHEEL_FILENAME,
    _load_manifest,
    _verify_source_archive,
    _verify_wheel,
    _write_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_locked_release_manifest_and_protocol_scope_are_exact() -> None:
    manifest = _load_manifest(REPO_ROOT)
    assert manifest["model_id"] == "tabularargn"
    assert manifest["package"]["filename"] == WHEEL_FILENAME
    assert manifest["package"]["record_hashed_files"] == len(manifest["installed_files"]) == 53
    assert manifest["wheel_source_comparison"]["exact_shared_source_files"] == 50
    assert SEED_CASES == (0, 19, 73)
    assert VARIANTS == ("binary", "multiclass", "regression")
    assert SAMPLE_ROWS == 7


@pytest.mark.parametrize("variant", VARIANTS)
def test_parity_fixture_covers_declared_types_without_missing_values(tmp_path: Path, variant: str) -> None:
    dataset_spec, record = _write_fixture(tmp_path, variant)
    assert dataset_spec.train_data_path is not None and dataset_spec.train_data_path.is_file()
    assert dataset_spec.column_names == ["first", "second", "group", "target"]
    assert dataset_spec.numerical_columns == ["first", "second"]
    assert dataset_spec.categorical_columns == ["group"]
    assert dataset_spec.target_columns == ["target"]
    assert record["training_rows"] == TRAIN_ROWS
    assert record["missing_values"] == 0
    assert dataset_spec.task_type == ("regression" if variant == "regression" else "classification")


def test_downloaded_official_wheel_and_source_archive_match_every_locked_member() -> None:
    configured_wheel = os.environ.get("TABULARARGN_WHEEL_PATH")
    configured_source = os.environ.get("TABULARARGN_SOURCE_ARCHIVE_PATH")
    audit = REPO_ROOT / ".cache" / "tabularargn-audit-2.6.2"
    wheel_path = Path(configured_wheel) if configured_wheel else audit / WHEEL_FILENAME
    source_path = Path(configured_source) if configured_source else audit / SOURCE_ARCHIVE_FILENAME
    if not wheel_path.is_file() or not source_path.is_file():
        pytest.skip("locked official release files are supplied only in the formal validation job")
    wheel = _verify_wheel(REPO_ROOT, wheel_path)
    source = _verify_source_archive(REPO_ROOT, source_path, wheel_path)
    assert wheel["record_hashed_files"] == 53
    assert source["exact_shared_package_source_files"] == 50
