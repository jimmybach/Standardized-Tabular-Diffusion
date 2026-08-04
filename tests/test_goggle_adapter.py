from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import standardized_tabular_diffusion.models.goggle as goggle_module
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.goggle import GoggleAdapter
from standardized_tabular_diffusion.upstream_sources import UpstreamSourceIntegrityError


def _classification_fixture(tmp_path: Path) -> tuple[GoggleAdapter, DatasetSpec, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.5, 4.0, 5.5, 7.0, 8.5],
            "segment": ["b", "a", "c", "a", "b", "c"],
            "target": [1, 0, 1, 0, 1, 0],
        }
    )
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    metadata_path = tmp_path / "info.json"
    metadata_path.write_text("{}", encoding="utf-8")
    spec = DatasetSpec(
        name="fixture",
        task_type="classification",
        column_names=["amount", "segment", "target"],
        numerical_columns=["amount"],
        categorical_columns=["segment"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )
    return GoggleAdapter(tmp_path), spec, frame


def test_goggle_preprocessing_is_train_fitted_and_reversible(tmp_path: Path) -> None:
    adapter, dataset_spec, expected = _classification_fixture(tmp_path)

    frame = adapter._load_training_frame(dataset_spec)
    transformed, metadata = adapter._transform_training_frame(frame, dataset_spec)
    recovered = adapter._inverse_transform(transformed.to_numpy(), metadata)

    assert metadata["fit_scope"] == "real-training-split-only"
    assert metadata["input_dim"] == 5
    assert metadata["training_rows"] == len(expected)
    assert recovered.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(recovered["amount"], expected["amount"])
    assert recovered["segment"].tolist() == expected["segment"].tolist()
    assert recovered["target"].tolist() == expected["target"].tolist()


def test_goggle_training_defaults_match_method_author_source(tmp_path: Path) -> None:
    adapter = GoggleAdapter(tmp_path)
    config = adapter._training_config(
        RunSpec(model="goggle", dataset="fixture", output_dir=tmp_path / "out", seed=19)
    )

    assert config == {
        "encoder_dim": 64,
        "encoder_l": 2,
        "het_encoding": True,
        "decoder_dim": 64,
        "decoder_l": 2,
        "threshold": 0.1,
        "decoder_arch": "gcn",
        "graph_prior": None,
        "prior_mask": None,
        "alpha": 0.1,
        "beta": 0.1,
        "iter_opt": True,
        "learning_rate": 0.005,
        "weight_decay": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "patience": 50,
        "logging": 100,
        "num_threads": 1,
    }


def test_goggle_rejects_missing_values_before_training(tmp_path: Path) -> None:
    adapter, dataset_spec, frame = _classification_fixture(tmp_path)
    frame.loc[2, "amount"] = np.nan
    frame.to_csv(dataset_spec.train_data_path, index=False)

    with pytest.raises(ValueError, match="centralized train-split-fitted"):
        adapter._load_training_frame(dataset_spec)


def test_goggle_validates_prior_shape_and_mask(tmp_path: Path) -> None:
    adapter = GoggleAdapter(tmp_path)
    config = adapter._training_config(
        RunSpec(
            model="goggle",
            dataset="fixture",
            output_dir=tmp_path / "out",
            extra={"graph_prior": [[0.0, 1.0], [1.0, 0.0]], "prior_mask": [[1.0, 0.0], [0.0, 0.5]]},
        )
    )

    with pytest.raises(ValueError, match="only zero and one"):
        adapter._validate_prior(config, input_dim=2)


def test_goggle_rejects_unknown_controls_and_unpaired_prior(tmp_path: Path) -> None:
    adapter = GoggleAdapter(tmp_path)
    unknown = RunSpec(model="goggle", dataset="fixture", output_dir=tmp_path / "out", extra={"mystery": 1})
    with pytest.raises(ValueError, match="Unsupported Goggle train controls"):
        adapter._validate_extra(unknown, action="train")

    unpaired = RunSpec(
        model="goggle",
        dataset="fixture",
        output_dir=tmp_path / "out",
        extra={"graph_prior": [[0.0]]},
    )
    with pytest.raises(ValueError, match="both be supplied"):
        adapter._training_config(unpaired)


def _source_record(source_root: Path) -> dict[str, object]:
    return {
        "model_id": "goggle",
        "source_dir": str(source_root),
        "repository": "https://github.com/vanderschaarlab/GOGGLE",
        "upstream_commit": GoggleAdapter.upstream_commit,
        "upstream_tree": "2d6a54f6d6f4d156890bf4e035119dbb483a46d0",
        "upstream_model_tree": "6dcaae801859f63e173537445548a50cd1f8625b",
        "runtime_files_verified": 18,
        "runtime_files": [],
        "license": {"declared_expression": "MIT"},
        "manifest_sha256": "a" * 64,
    }


def test_goggle_adapter_confines_artifacts_and_honors_requested_rows(tmp_path: Path, monkeypatch) -> None:
    adapter, dataset_spec, _ = _classification_fixture(tmp_path)
    source_root = adapter.upstream_root
    source_root.mkdir(parents=True)
    source = _source_record(source_root)
    commands: list[list[str]] = []
    monkeypatch.setattr(goggle_module, "validate_upstream_source", lambda model, path: source)

    def fake_run(args: list[str]) -> None:
        commands.append(args)
        if args[args.index("--action") + 1] == "train":
            Path(args[args.index("--checkpoint") + 1]).write_bytes(b"trusted-goggle-state")
        else:
            raw_path = Path(args[args.index("--raw-output") + 1])
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            input_dim = json.loads((tmp_path / "out" / "goggle-runtime-config.json").read_text())["input_dim"]
            np.save(raw_path, np.zeros((5, input_dim), dtype=np.float32), allow_pickle=False)

    monkeypatch.setattr(adapter, "_run_goggle", fake_run)
    output_dir = tmp_path / "out"
    common = {"dataset_spec": dataset_spec.to_dict(), "source_dir": str(source_root)}
    adapter.train(
        RunSpec(
            model="goggle",
            dataset="fixture",
            output_dir=output_dir,
            seed=19,
            extra={**common, "epochs": 1, "batch_size": 4, "encoder_dim": 8, "decoder_dim": 8},
        )
    )
    bundle = adapter.sample(
        RunSpec(
            model="goggle",
            dataset="fixture",
            output_dir=output_dir,
            seed=19,
            num_samples=5,
            extra={**common, "num_threads": 1},
        )
    )

    metadata = json.loads((output_dir / "goggle-model-metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"]["runtime_files_verified"] == 18
    assert metadata["execution_config"]["epochs"] == 1
    assert bundle.generated_sample_path == output_dir / "samples.csv"
    assert len(pd.read_csv(bundle.generated_sample_path)) == 5
    assert commands[1][commands[1].index("--num-samples") + 1] == "5"
    assert not any(path.name.startswith(".goggle-sample-") for path in output_dir.iterdir())
    assert not (source_root / "tmp").exists()


def test_goggle_adapter_rejects_checkpoint_tampering(tmp_path: Path, monkeypatch) -> None:
    adapter, dataset_spec, _ = _classification_fixture(tmp_path)
    source_root = adapter.upstream_root
    source_root.mkdir(parents=True)
    monkeypatch.setattr(
        goggle_module, "validate_upstream_source", lambda model, path: _source_record(source_root)
    )
    output_dir = tmp_path / "trained"

    def fake_train(args: list[str]) -> None:
        Path(args[args.index("--checkpoint") + 1]).write_bytes(b"original")

    monkeypatch.setattr(adapter, "_run_goggle", fake_train)
    adapter.train(
        RunSpec(
            model="goggle",
            dataset="fixture",
            output_dir=output_dir,
            extra={"dataset_spec": dataset_spec.to_dict(), "source_dir": str(source_root), "epochs": 1},
        )
    )
    (output_dir / "model.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checkpoint checksum"):
        adapter.sample(
            RunSpec(
                model="goggle",
                dataset="fixture",
                output_dir=output_dir,
                num_samples=2,
                extra={"dataset_spec": dataset_spec.to_dict(), "source_dir": str(source_root)},
            )
        )


def test_goggle_adapter_revalidates_source_after_training(tmp_path: Path, monkeypatch) -> None:
    adapter, dataset_spec, _ = _classification_fixture(tmp_path)
    source_root = adapter.upstream_root
    source_root.mkdir(parents=True)
    validations = 0

    def validate(model: str, path: Path) -> dict[str, object]:
        nonlocal validations
        validations += 1
        if validations <= 2:
            return _source_record(source_root)
        raise UpstreamSourceIntegrityError("source changed during execution")

    monkeypatch.setattr(goggle_module, "validate_upstream_source", validate)

    def fake_train(args: list[str]) -> None:
        Path(args[args.index("--checkpoint") + 1]).write_bytes(b"checkpoint")

    monkeypatch.setattr(adapter, "_run_goggle", fake_train)
    with pytest.raises(UpstreamSourceIntegrityError, match="source changed during execution"):
        adapter.train(
            RunSpec(
                model="goggle",
                dataset="fixture",
                output_dir=tmp_path / "out",
                extra={"dataset_spec": dataset_spec.to_dict(), "source_dir": str(source_root), "epochs": 1},
            )
        )
