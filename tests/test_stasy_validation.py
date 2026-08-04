from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import standardized_tabular_diffusion.models.vendored_baselines as vendored_baselines
from standardized_tabular_diffusion.compat.stasy_launcher import _install_sklearn_onehot_bridge
from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.vendored_baselines import STaSyAdapter
from standardized_tabular_diffusion.upstream_sources import UpstreamSourceIntegrityError, validate_upstream_source

pytestmark = pytest.mark.adapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(root: Path, *, missing: bool = False) -> None:
    data_dir = root / "TabSyn-main" / "data" / "fixture"
    data_dir.mkdir(parents=True)
    numerical = np.array([[1.0], [np.nan if missing else 2.0]], dtype=np.float32)
    categorical = np.array([["a"], ["b"]])
    target = np.array([0, 1], dtype=np.int64)
    for split in ("train", "test"):
        np.save(data_dir / f"X_num_{split}.npy", numerical)
        np.save(data_dir / f"X_cat_{split}.npy", categorical)
        np.save(data_dir / f"y_{split}.npy", target)
    (data_dir / "info.json").write_text(
        json.dumps(
            {
                "task_type": "binclass",
                "train_num": 2,
                "num_col_idx": [0],
                "cat_col_idx": [1],
                "target_col_idx": [2],
            }
        ),
        encoding="utf-8",
    )


def _source_record(root: Path) -> dict[str, object]:
    return {
        "model_id": "stasy",
        "source_dir": str(root / "TabSyn-main"),
        "repository": "https://github.com/amazon-science/tabsyn",
        "upstream_commit": "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7",
        "upstream_tree": "cb10c6da6e4b5c6f27261dfa0e4c593df9cc19ca",
        "upstream_model_tree": "4f56a7223d71d6b75c1698824c5d0245bf716bc6",
        "runtime_files_verified": 30,
        "runtime_files": [],
        "license": {"declared_expression": "Apache-2.0"},
        "manifest_sha256": "a" * 64,
    }


def test_stasy_distributed_execution_scope_is_checksum_locked() -> None:
    result = validate_upstream_source("stasy", REPO_ROOT / "TabSyn-main")

    assert result["runtime_files_verified"] == 30
    assert result["upstream_model_tree"] == "4f56a7223d71d6b75c1698824c5d0245bf716bc6"


def test_stasy_sklearn_bridge_only_renames_the_dense_output_keyword() -> None:
    sklearn_preprocessing, official_encoder = _install_sklearn_onehot_bridge()
    try:
        encoder = sklearn_preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
        assert encoder.sparse_output is False
        assert encoder.handle_unknown == "ignore"
        with pytest.raises(TypeError, match="both sparse and sparse_output"):
            sklearn_preprocessing.OneHotEncoder(sparse=False, sparse_output=False)
    finally:
        sklearn_preprocessing.OneHotEncoder = official_encoder


def test_stasy_adapter_confines_checkpoint_and_honors_requested_rows(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    adapter = STaSyAdapter(tmp_path)
    source = _source_record(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", lambda model, path: source)

    def fake_run(args: list[str], *, seed: int) -> None:
        assert seed == 19
        commands.append(args)
        output_dir = Path(args[args.index("--output-dir") + 1])
        if args[args.index("--action") + 1] == "train":
            checkpoint = output_dir / "ckpt" / "fixture" / "model.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"trusted-stasy-checkpoint")
        else:
            sample_path = Path(args[args.index("--save-path") + 1])
            pd.DataFrame({"0": [1.0] * 5, "1": ["a"] * 5, "2": [0] * 5}).to_csv(
                sample_path, index=False
            )

    monkeypatch.setattr(adapter, "_run_stasy", fake_run)
    output_dir = tmp_path / "artifacts" / "stasy"
    train_spec = RunSpec(
        model="stasy",
        dataset="fixture",
        output_dir=output_dir,
        device="cpu",
        seed=19,
        extra={
            "epochs": 1,
            "batch_size": 8,
            "nf": 8,
            "hidden_dims": [16, 16],
            "num_scales": 2,
            "num_workers": 0,
            "num_threads": 1,
            "sampler": "pc",
        },
    )
    adapter.train(train_spec)
    sample_bundle = adapter.sample(
        RunSpec(
            model="stasy",
            dataset="fixture",
            output_dir=output_dir,
            device="cpu",
            seed=19,
            num_samples=5,
            extra={"sampler": "pc", "num_threads": 1},
        )
    )

    checkpoint = output_dir / "ckpt" / "fixture" / "model.pth"
    metadata = json.loads((output_dir / "stasy-model-metadata.json").read_text(encoding="utf-8"))
    sample_metadata = json.loads((output_dir / "stasy-sample-metadata.json").read_text(encoding="utf-8"))
    assert checkpoint.is_file()
    assert not (tmp_path / "TabSyn-main" / "baselines" / "stasy" / "ckpt").exists()
    assert metadata["training_config"]["hidden_dims"] == [16, 16]
    assert metadata["source"]["runtime_files_verified"] == 30
    assert sample_metadata["rows"] == 5
    assert sample_bundle.generated_sample_path == output_dir.resolve() / "samples.csv"
    assert "--num-samples" in commands[1]
    assert commands[1][commands[1].index("--num-samples") + 1] == "5"


def test_stasy_adapter_rejects_missing_values_before_execution(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path, missing=True)
    adapter = STaSyAdapter(tmp_path)
    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", lambda model, path: _source_record(tmp_path))
    monkeypatch.setattr(adapter, "_run_stasy", lambda args, seed: pytest.fail("launcher must not execute"))

    with pytest.raises(ValueError, match="centralized train-split-fitted imputer"):
        adapter.train(RunSpec(model="stasy", dataset="fixture", output_dir=tmp_path / "out"))


@pytest.mark.parametrize("dataset", ["../fixture", "nested/fixture", r"nested\fixture", "C:fixture"])
def test_stasy_adapter_rejects_dataset_paths(tmp_path: Path, dataset: str) -> None:
    adapter = STaSyAdapter(tmp_path)

    with pytest.raises(ValueError, match="single safe dataset identifier"):
        adapter.train(RunSpec(model="stasy", dataset=dataset, output_dir=tmp_path / "out"))


def test_stasy_adapter_revalidates_source_after_upstream_execution(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    adapter = STaSyAdapter(tmp_path)
    source = _source_record(tmp_path)
    validations = 0

    def validate(model: str, path: Path) -> dict[str, object]:
        nonlocal validations
        validations += 1
        if validations == 1:
            return source
        raise UpstreamSourceIntegrityError("source changed during execution")

    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", validate)
    monkeypatch.setattr(adapter, "_run_stasy", lambda args, seed: None)

    with pytest.raises(UpstreamSourceIntegrityError, match="source changed during execution"):
        adapter.train(RunSpec(model="stasy", dataset="fixture", output_dir=tmp_path / "out"))


def test_stasy_adapter_rejects_unknown_controls_and_checkpoint_tampering(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    adapter = STaSyAdapter(tmp_path)
    source = _source_record(tmp_path)
    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", lambda model, path: source)

    with pytest.raises(ValueError, match="Unsupported STaSy train controls"):
        adapter.train(
            RunSpec(model="stasy", dataset="fixture", output_dir=tmp_path / "out", extra={"bogus": 1})
        )

    output_dir = tmp_path / "trained"

    def fake_train(args: list[str], *, seed: int) -> None:
        checkpoint = output_dir / "ckpt" / "fixture" / "model.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"original")

    monkeypatch.setattr(adapter, "_run_stasy", fake_train)
    adapter.train(RunSpec(model="stasy", dataset="fixture", output_dir=output_dir))
    (output_dir / "ckpt" / "fixture" / "model.pth").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checkpoint checksum"):
        adapter.sample(RunSpec(model="stasy", dataset="fixture", output_dir=output_dir, num_samples=2))
