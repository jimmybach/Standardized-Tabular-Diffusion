from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec


def test_public_interface_payloads_recursively_serialize_paths(tmp_path: Path) -> None:
    dataset = DatasetSpec(
        name="fixture",
        task_type="classification",
        column_names=["feature", "target"],
        numerical_columns=["feature"],
        categorical_columns=["target"],
        target_columns=["target"],
        metadata_path=tmp_path / "metadata.json",
        train_data_path=tmp_path / "train.csv",
        extra={
            "synthetic": {
                "real": tmp_path / "synthetic" / "real.csv",
                "partitions": [PurePosixPath("portable/validation.csv"), (tmp_path / "test.csv",)],
            }
        },
    )
    run = RunSpec(
        model="tabula",
        dataset="fixture",
        output_dir=tmp_path / "output",
        extra={"dataset_spec": dataset.to_dict(), "cache_paths": (tmp_path / "cache",)},
    )
    bundle = ArtifactBundle(
        model="tabula",
        dataset="fixture",
        output_dir=tmp_path / "output",
        upstream_workdir=tmp_path / "source",
        generated_sample_path=tmp_path / "output" / "samples.csv",
    )

    dataset_payload = dataset.to_dict()
    run_payload = run.to_dict()
    bundle_payload = bundle.to_dict()

    assert dataset_payload["extra"]["synthetic"] == {
        "real": str(tmp_path / "synthetic" / "real.csv"),
        "partitions": ["portable/validation.csv", [str(tmp_path / "test.csv")]],
    }
    assert run_payload["extra"]["cache_paths"] == [str(tmp_path / "cache")]
    assert bundle_payload["generated_sample_path"] == str(tmp_path / "output" / "samples.csv")
    json.dumps(
        {"dataset": dataset_payload, "run": run_payload, "bundle": bundle_payload},
        allow_nan=False,
    )
