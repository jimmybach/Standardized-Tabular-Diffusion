from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import standardized_tabular_diffusion.models.vendored_baselines as vendored_baselines
from standardized_tabular_diffusion.compat.codi_launcher import _TorchProxy
from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.vendored_baselines import CoDiAdapter
from standardized_tabular_diffusion.upstream_sources import UpstreamSourceIntegrityError, validate_upstream_source

pytestmark = pytest.mark.adapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(root: Path, *, missing: bool = False, numerical_features: bool = True) -> None:
    data_dir = root / "TabSyn-main" / "data" / "fixture"
    data_dir.mkdir(parents=True)
    for split, offset in (("train", 0), ("test", 10)):
        rows = np.arange(offset, offset + 4)
        numerical = (rows * 0.25 + 1).astype(np.float32).reshape(-1, 1)
        if missing and split == "train":
            numerical[1, 0] = np.nan
        categorical = np.array([f"g{value % 2}" for value in rows]).reshape(-1, 1)
        target = (rows % 2).astype(np.int64)
        if numerical_features:
            frame = pd.DataFrame({"num": numerical[:, 0], "cat": categorical[:, 0], "target": target})
            np.save(data_dir / f"X_num_{split}.npy", numerical)
            num_idx = [0]
            cat_idx = [1]
            target_idx = [2]
        else:
            frame = pd.DataFrame({"cat": categorical[:, 0], "target": target})
            num_idx = []
            cat_idx = [0]
            target_idx = [1]
        np.save(data_dir / f"X_cat_{split}.npy", categorical)
        np.save(data_dir / f"y_{split}.npy", target)
        frame.to_csv(data_dir / f"{split}.csv", index=False)
    columns = list(frame.columns)
    (data_dir / "info.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "task_type": "binclass",
                "num_col_idx": num_idx,
                "cat_col_idx": cat_idx,
                "target_col_idx": target_idx,
                "idx_name_mapping": {str(index): name for index, name in enumerate(columns)},
                "train_num": 4,
                "test_num": 4,
            }
        ),
        encoding="utf-8",
    )


def _source_record(root: Path) -> dict[str, object]:
    return {
        "model_id": "codi",
        "source_dir": str(root / "TabSyn-main"),
        "repository": "https://github.com/amazon-science/tabsyn",
        "upstream_commit": "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7",
        "upstream_tree": "cb10c6da6e4b5c6f27261dfa0e4c593df9cc19ca",
        "upstream_model_tree": "85c16ccfb76fbf00db6b30450ca47e9928efa8d3",
        "runtime_files_verified": 24,
        "runtime_files": [],
        "license": {"declared_expression": "Apache-2.0"},
        "manifest_sha256": "a" * 64,
    }


def _small_config() -> dict[str, object]:
    return {
        "training_batch_size": 4,
        "eval_batch_size": 4,
        "T": 2,
        "total_epochs_both": 1,
        "sample_step": 1,
        "nf_con": 4,
        "nf_dis": 4,
        "encoder_dim_con": [8, 8],
        "encoder_dim_dis": [8, 8],
        "num_threads": 1,
    }


def test_codi_distributed_execution_scope_is_checksum_locked() -> None:
    result = validate_upstream_source("codi", REPO_ROOT / "TabSyn-main")

    assert result["runtime_files_verified"] == 24
    assert result["upstream_model_tree"] == "85c16ccfb76fbf00db6b30450ca47e9928efa8d3"


def test_codi_cpu_proxy_exposes_one_logical_loader_device_without_global_patch() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 8

    class FakeTorch:
        cuda = FakeCuda()
        marker = object()

    original = FakeTorch()
    proxy = _TorchProxy(original, cuda_enabled=False)

    assert proxy.cuda.is_available() is False
    assert proxy.cuda.device_count() == 1
    assert proxy.marker is original.marker
    assert original.cuda.is_available() is True
    assert original.cuda.device_count() == 8


def test_codi_adapter_confines_checkpoint_pair_and_honors_requested_rows(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    adapter = CoDiAdapter(tmp_path)
    source = _source_record(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", lambda model, path: source)

    def fake_run(args: list[str], *, seed: int) -> None:
        assert seed == 19
        commands.append(args)
        output_dir = Path(args[args.index("--output-dir") + 1])
        if args[args.index("--action") + 1] == "train":
            checkpoint_root = output_dir / "ckpt" / "fixture"
            checkpoint_root.mkdir(parents=True)
            (checkpoint_root / "model_con.pt").write_bytes(b"trusted-continuous")
            (checkpoint_root / "model_dis.pt").write_bytes(b"trusted-discrete")
        else:
            sample_path = Path(args[args.index("--save-path") + 1])
            pd.DataFrame({"num": [1.0] * 5, "cat": ["g0"] * 5, "target": [0] * 5}).to_csv(
                sample_path, index=False
            )

    monkeypatch.setattr(adapter, "_run_codi", fake_run)
    output_dir = tmp_path / "artifacts" / "codi"
    adapter.train(
        RunSpec(
            model="codi",
            dataset="fixture",
            output_dir=output_dir,
            device="cpu",
            seed=19,
            extra=_small_config(),
        )
    )
    bundle = adapter.sample(
        RunSpec(
            model="codi",
            dataset="fixture",
            output_dir=output_dir,
            device="cpu",
            seed=19,
            num_samples=5,
            extra={"num_threads": 1},
        )
    )

    metadata = json.loads((output_dir / "codi-model-metadata.json").read_text(encoding="utf-8"))
    sample_metadata = json.loads((output_dir / "codi-sample-metadata.json").read_text(encoding="utf-8"))
    assert (output_dir / "ckpt" / "fixture" / "model_con.pt").is_file()
    assert (output_dir / "ckpt" / "fixture" / "model_dis.pt").is_file()
    assert not (tmp_path / "TabSyn-main" / "baselines" / "codi" / "ckpt").exists()
    assert metadata["source"]["runtime_files_verified"] == 24
    assert metadata["training_config"]["encoder_dim_con"] == [8, 8]
    assert sample_metadata["rows"] == 5
    assert bundle.generated_sample_path == output_dir.resolve() / "samples.csv"
    assert commands[1][commands[1].index("--num-samples") + 1] == "5"


def test_codi_adapter_rejects_missing_values_and_one_sided_diffusion_data(
    tmp_path: Path, monkeypatch
) -> None:
    _write_dataset(tmp_path, missing=True)
    adapter = CoDiAdapter(tmp_path)
    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", lambda model, path: _source_record(tmp_path))
    monkeypatch.setattr(adapter, "_run_codi", lambda args, seed: pytest.fail("launcher must not execute"))

    with pytest.raises(ValueError, match="centralized train-split-fitted imputer"):
        adapter.train(RunSpec(model="codi", dataset="fixture", output_dir=tmp_path / "out"))

    second_root = tmp_path / "one-sided"
    _write_dataset(second_root, numerical_features=False)
    with pytest.raises(ValueError, match="continuous diffusion column"):
        CoDiAdapter(second_root).train(
            RunSpec(model="codi", dataset="fixture", output_dir=second_root / "out")
        )


@pytest.mark.parametrize("dataset", ["../fixture", "nested/fixture", r"nested\fixture", "C:fixture"])
def test_codi_adapter_rejects_dataset_paths(tmp_path: Path, dataset: str) -> None:
    with pytest.raises(ValueError, match="single safe dataset identifier"):
        CoDiAdapter(tmp_path).train(RunSpec(model="codi", dataset=dataset, output_dir=tmp_path / "out"))


def test_codi_adapter_revalidates_source_after_execution(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    adapter = CoDiAdapter(tmp_path)
    validations = 0

    def validate(model: str, path: Path) -> dict[str, object]:
        nonlocal validations
        validations += 1
        if validations == 1:
            return _source_record(tmp_path)
        raise UpstreamSourceIntegrityError("source changed during execution")

    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", validate)
    monkeypatch.setattr(adapter, "_run_codi", lambda args, seed: None)

    with pytest.raises(UpstreamSourceIntegrityError, match="source changed during execution"):
        adapter.train(
            RunSpec(model="codi", dataset="fixture", output_dir=tmp_path / "out", extra=_small_config())
        )


def test_codi_adapter_rejects_unknown_controls_and_checkpoint_tampering(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    adapter = CoDiAdapter(tmp_path)
    monkeypatch.setattr(vendored_baselines, "validate_upstream_source", lambda model, path: _source_record(tmp_path))

    with pytest.raises(ValueError, match="Unsupported CoDi train controls"):
        adapter.train(
            RunSpec(model="codi", dataset="fixture", output_dir=tmp_path / "out", extra={"bogus": 1})
        )

    output_dir = tmp_path / "trained"

    def fake_train(args: list[str], *, seed: int) -> None:
        checkpoint_root = output_dir / "ckpt" / "fixture"
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "model_con.pt").write_bytes(b"original-con")
        (checkpoint_root / "model_dis.pt").write_bytes(b"original-dis")

    monkeypatch.setattr(adapter, "_run_codi", fake_train)
    adapter.train(
        RunSpec(model="codi", dataset="fixture", output_dir=output_dir, extra=_small_config())
    )
    (output_dir / "ckpt" / "fixture" / "model_con.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="continuous checkpoint checksum"):
        adapter.sample(RunSpec(model="codi", dataset="fixture", output_dir=output_dir, num_samples=2))
