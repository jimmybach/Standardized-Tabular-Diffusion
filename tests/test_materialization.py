from __future__ import annotations

import json
from pathlib import Path

from standardized_tabular_diffusion import materialization
from standardized_tabular_diffusion.interfaces import DatasetSpec


def test_materialization_status_is_recursively_json_serializable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    nested_path = tmp_path / "nested" / "artifact.json"
    spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["feature", "target"],
        numerical_columns=["feature"],
        categorical_columns=[],
        target_columns=["target"],
        metadata_path=tmp_path / "info.json",
        extra={"artifacts": {"nested_path": nested_path}},
    )
    monkeypatch.setattr(materialization, "get_dataset_spec", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        materialization,
        "load_manifest",
        lambda *_args, **_kwargs: {"output_paths": [nested_path]},
    )

    status = materialization.materialization_status("adult", repo_root=tmp_path)

    assert status["base_spec"]["extra"]["artifacts"]["nested_path"] == str(nested_path)
    assert status["manifest"]["output_paths"] == [str(nested_path)]
    json.dumps(status)
