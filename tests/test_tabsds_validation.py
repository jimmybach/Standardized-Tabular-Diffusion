from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.models.paper_gap_baselines import TabSDSAdapter
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.upstream_sources import load_source_manifest, validate_upstream_source
from standardized_tabular_diffusion.validation import tabsds as tabsds_validation

pytestmark = pytest.mark.core


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
