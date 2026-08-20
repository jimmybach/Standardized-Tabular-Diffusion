from __future__ import annotations

from pathlib import Path

import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.sample_baselines import CTGANAdapter, SMOTEAdapter
from standardized_tabular_diffusion.output_decoding import declared_integer_columns


def _integer_dataset(tmp_path: Path) -> DatasetSpec:
    frame = pd.DataFrame(
        {
            "count": [1, 2, 3, 4, 5, 6],
            "group": ["a", "b", "a", "b", "a", "b"],
            "label": ["no", "yes", "no", "yes", "no", "yes"],
        }
    )
    train_path = tmp_path / "train.csv"
    metadata_path = tmp_path / "info.json"
    frame.to_csv(train_path, index=False)
    metadata_path.write_text('{"int_columns":["count"]}\n', encoding="utf-8")
    return DatasetSpec(
        name="integer-output-test",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=["count"],
        categorical_columns=["group"],
        target_columns=["label"],
        metadata_path=metadata_path,
        train_data_path=train_path,
        extra={"column_info": {"count": {"semantic_type": "integer"}}},
    )


def test_integer_declaration_resolution_deduplicates_reviewed_sources(tmp_path: Path) -> None:
    dataset = _integer_dataset(tmp_path)
    dataset.extra["integer_columns"] = ["count"]
    assert declared_integer_columns(dataset, model_name="test") == ["count"]


def test_smote_integer_decoding_retains_unmodified_native_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _integer_dataset(tmp_path)
    adapter = SMOTEAdapter(tmp_path)

    class FakeSampler:
        def fit_resample(self, features, target):
            native = features.copy()
            native["count"] = native["count"].astype(float) + 0.4
            return native, target.copy()

    monkeypatch.setattr(adapter, "_create_sampler", lambda **_kwargs: (FakeSampler(), "SMOTENC"))
    output = tmp_path / "output"
    spec = RunSpec(
        model="smote",
        dataset=dataset.name,
        output_dir=output,
        device="cpu",
        seed=17,
        num_samples=6,
        extra={"dataset_spec": dataset.to_dict(), "k_neighbors": 1},
    )

    bundle = adapter.sample(spec)

    decoded = pd.read_csv(bundle.generated_sample_path)
    native = pd.read_csv(output / "smote_native_samples.csv")
    metadata = read_json(output / "smote_metadata.json")
    assert decoded["count"].tolist() == native["count"].round().astype(int).tolist()
    assert (native["count"] % 1 != 0).all()
    assert metadata["integer_decoding"]["count"]["changed_rows"] == 6
    assert metadata["integer_decoding"]["count"]["clipped_rows"] == 0
    assert metadata["native_sample_sha256"]


def test_official_ctgan_package_adapter_decodes_declared_integers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _integer_dataset(tmp_path)
    adapter = CTGANAdapter(tmp_path)

    class FakeModel:
        def set_random_state(self, _seed):
            return None

        def sample(self, rows):
            frame = pd.read_csv(dataset.train_data_path).iloc[:rows].copy()
            frame["count"] = frame["count"].astype(float) + 0.25
            return frame

    output = tmp_path / "ctgan-output"
    output.mkdir()
    (output / adapter.checkpoint_filename).write_bytes(b"test-checkpoint")
    monkeypatch.setattr(adapter, "_load_model", lambda _spec, _path: FakeModel())
    spec = RunSpec(
        model="ctgan",
        dataset=dataset.name,
        output_dir=output,
        device="cuda",
        seed=29,
        num_samples=6,
        extra={"dataset_spec": dataset.to_dict()},
    )

    bundle = adapter.sample(spec)

    assert pd.read_csv(bundle.generated_sample_path)["count"].tolist() == [1, 2, 3, 4, 5, 6]
    assert (pd.read_csv(output / "ctgan_native_samples.csv")["count"] % 1 != 0).all()
    metadata = read_json(output / "ctgan_sample_metadata.json")
    assert metadata["integer_decoding"]["count"]["changed_rows"] == 6
    assert metadata["checkpoint_sha256"]
