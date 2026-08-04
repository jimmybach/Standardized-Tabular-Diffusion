from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, SampleConfig, TrainConfig
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.realtabformer import REaLTabFormerAdapter

pytestmark = pytest.mark.adapter


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        _FakeTorch.seeds.append(("cuda", seed))


class _FakeTorch:
    cuda = _FakeCuda()
    seeds: list[tuple[str, int]] = []

    @staticmethod
    def manual_seed(seed: int) -> None:
        _FakeTorch.seeds.append(("cpu", seed))

    @staticmethod
    def load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}


class _FakeGPT2Config:
    def __init__(self, **kwargs: Any) -> None:
        self.n_layer = int(kwargs.get("n_layer", 6))
        self.n_head = int(kwargs.get("n_head", 12))
        self.n_embd = int(kwargs.get("n_embd", 768))
        self.n_positions = int(kwargs.get("n_positions", 1024))
        self.payload = dict(kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "n_positions": self.n_positions,
            "id2label": {0: "LABEL_0"},
            **self.payload,
        }


class _FakeRTF:
    instances: list[_FakeRTF] = []
    observed_fit_rows: list[pd.DataFrame] = []
    observed_fit_kwargs: list[dict[str, Any]] = []
    observed_sample_kwargs: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.full_save_dir = Path(kwargs.get("full_save_dir", "."))
        self.__class__.instances.append(self)

    def fit(self, frame: pd.DataFrame, **kwargs: Any) -> None:
        self.__class__.observed_fit_rows.append(frame.copy())
        self.__class__.observed_fit_kwargs.append(dict(kwargs))

    def save(self, path: str) -> None:
        model_dir = Path(path) / "id0001"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "rtf_config.json").write_text('{"model_type":"tabular"}', encoding="utf-8")
        (model_dir / "rtf_model.pt").write_bytes(b"locked-state-dict")

    @classmethod
    def load_from_dir(cls, path: str) -> _FakeRTF:
        return cls(model_type="tabular", full_save_dir=Path(path) / "full-save")

    def sample(self, n_samples: int, **kwargs: Any) -> pd.DataFrame:
        self.__class__.observed_sample_kwargs.append(dict(kwargs))
        return pd.DataFrame(
            {
                "value": [float(index) for index in range(n_samples)],
                "group": ["a" if index % 2 == 0 else "b" for index in range(n_samples)],
                "target": ["yes" if index % 2 == 0 else "no" for index in range(n_samples)],
            }
        )


def _package_record(tmp_path: Path) -> dict[str, Any]:
    return {
        "distribution_root": str(tmp_path / "site-packages"),
        "package_version": "0.2.4",
        "wheel_sha256": "wheel-sha",
        "manifest_sha256": "manifest-sha",
        "upstream_commit": "73f239643f9ea5abc877f685ce927e986302ac2d",
    }


def _dataset(tmp_path: Path, *, missing: bool = False) -> DatasetSpec:
    train_path = tmp_path / "train.csv"
    rows = "value,group,target\n1.0,a,yes\n2.0,b,no\n3.0,a,yes\n4.0,b,no\n"
    if missing:
        rows = "value,group,target\n1.0,a,yes\n,b,no\n"
    train_path.write_text(rows, encoding="utf-8")
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text("{}", encoding="utf-8")
    return DatasetSpec(
        name="fixture",
        task_type="classification",
        column_names=["value", "group", "target"],
        numerical_columns=["value"],
        categorical_columns=["group"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )


@pytest.fixture(autouse=True)
def _reset_fake_runtime() -> None:
    _FakeTorch.seeds.clear()
    _FakeRTF.instances.clear()
    _FakeRTF.observed_fit_rows.clear()
    _FakeRTF.observed_fit_kwargs.clear()
    _FakeRTF.observed_sample_kwargs.clear()


def _patch_runtime(adapter: REaLTabFormerAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "_import_runtime",
        lambda: (_FakeRTF, _FakeGPT2Config, _FakeTorch, _package_record(tmp_path)),
    )


def test_realtabformer_train_and_sample_are_output_local_and_integrity_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = REaLTabFormerAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    output_dir = tmp_path / "output"
    train_config = ExperimentConfig(
        model="realtabformer",
        dataset="fixture",
        output_dir=str(output_dir),
        train=TrainConfig(
            seed=19,
            device="cpu",
            extra={
                "epochs": 2,
                "batch_size": 4,
                "n_critic": 0,
                "num_bootstrap": 0,
                "tabular_config": {"n_layer": 1, "n_head": 1, "n_embd": 8, "n_positions": 64},
                "report_to": "none",
            },
        ),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    adapter.train_from_config(train_config, dataset_spec=dataset)

    model = _FakeRTF.instances[0]
    assert model.kwargs["random_state"] == 19
    assert model.kwargs["epochs"] == 2
    assert model.kwargs["batch_size"] == 4
    assert model.kwargs["use_cpu"] is True
    for key in ("checkpoints_dir", "samples_save_dir", "full_save_dir"):
        assert Path(model.kwargs[key]).resolve().is_relative_to(output_dir.resolve())
    assert _FakeRTF.observed_fit_kwargs[0]["device"] == "cpu"
    assert _FakeRTF.observed_fit_kwargs[0]["n_critic"] == 0
    assert _FakeRTF.observed_fit_rows[0]["target"].dtype == object
    training_metadata = json.loads((output_dir / "realtabformer-model-metadata.json").read_text(encoding="utf-8"))
    assert training_metadata["training"]["constructor"]["tabular_config"]["id2label"] == {"0": "LABEL_0"}

    metadata_path = output_dir / "realtabformer-model-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["package"]["wheel_sha256"] == "wheel-sha"
    assert metadata["source_rows"] == metadata["training_rows"] == 4
    assert metadata["saved_model_manifest_sha256"]
    assert metadata["saved_model_dir"] == "realtabformer_model/id0001"

    sample_config = ExperimentConfig(
        model="realtabformer",
        dataset="fixture",
        output_dir=str(output_dir),
        train=TrainConfig(seed=19, device="cpu", enabled=False),
        sample=SampleConfig(enabled=True, num_samples=3, extra={"gen_batch": 2}),
        evaluation=EvaluationConfig(enabled=False),
    )
    bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset)
    assert bundle.generated_sample_path == output_dir / "samples.csv"
    assert pd.read_csv(bundle.generated_sample_path).shape == (3, 3)
    assert _FakeRTF.observed_sample_kwargs[-1] == {"device": "cpu", "gen_batch": 2}
    sample_metadata = json.loads((output_dir / "realtabformer-sample-metadata.json").read_text(encoding="utf-8"))
    assert sample_metadata["requested_rows"] == 3
    assert sample_metadata["checkpoint_manifest_sha256"] == metadata["saved_model_manifest_sha256"]
    assert ("cpu", 19) in _FakeTorch.seeds


def test_realtabformer_train_defaults_match_official_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = REaLTabFormerAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    spec = RunSpec(
        model="realtabformer",
        dataset="fixture",
        output_dir=tmp_path / "default-output",
        device="cpu",
        seed=0,
        extra={"dataset_spec": dataset.to_dict()},
    )
    adapter.train(spec)
    constructor = _FakeRTF.instances[0].kwargs
    fit = _FakeRTF.observed_fit_kwargs[0]
    assert constructor["epochs"] == 1000
    assert constructor["batch_size"] == 8
    assert constructor["train_size"] == 1.0
    assert constructor["numeric_precision"] == 4
    assert fit["num_bootstrap"] == 500
    assert fit["n_critic"] == 5


def test_realtabformer_tiny_row_limit_is_seeded_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = REaLTabFormerAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    spec = RunSpec(
        model="realtabformer",
        dataset="fixture",
        output_dir=tmp_path / "limited-output",
        device="cpu",
        seed=73,
        extra={
            "dataset_spec": dataset.to_dict(),
            "max_train_rows": 2,
            "n_critic": 0,
            "num_bootstrap": 0,
        },
    )
    adapter.train(spec)
    observed = _FakeRTF.observed_fit_rows[0]
    expected = pd.read_csv(dataset.train_data_path).sample(n=2, random_state=73).reset_index(drop=True)
    expected["group"] = expected["group"].astype(str)
    expected["target"] = expected["target"].astype(str)
    pd.testing.assert_frame_equal(observed, expected)
    metadata = json.loads((spec.output_dir / adapter.metadata_filename).read_text(encoding="utf-8"))
    assert metadata["source_rows"] == 4
    assert metadata["training_rows"] == 2


def test_realtabformer_rejects_missing_values_before_importing_runtime(tmp_path: Path) -> None:
    adapter = REaLTabFormerAdapter(tmp_path)
    dataset = _dataset(tmp_path, missing=True)
    spec = RunSpec(
        model="realtabformer",
        dataset="fixture",
        output_dir=tmp_path / "output",
        extra={"dataset_spec": dataset.to_dict()},
    )
    with pytest.raises(ValueError, match="centralized train-only mean/mode imputer"):
        adapter.train(spec)


def test_realtabformer_rejects_unknown_or_invalid_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = REaLTabFormerAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    base = {
        "model": "realtabformer",
        "dataset": "fixture",
        "output_dir": tmp_path / "output",
    }
    with pytest.raises(ValueError, match="Unknown REaLTabFormer training controls"):
        adapter.train(RunSpec(**base, extra={"dataset_spec": dataset.to_dict(), "epohs": 1}))
    with pytest.raises(ValueError, match="n_embd must be divisible"):
        adapter.train(
            RunSpec(
                **base,
                extra={
                    "dataset_spec": dataset.to_dict(),
                    "tabular_config": {"n_layer": 1, "n_head": 3, "n_embd": 8},
                },
            )
        )
    with pytest.raises(ValueError, match="reporting is disabled"):
        adapter.train(RunSpec(**base, extra={"dataset_spec": dataset.to_dict(), "report_to": "wandb"}))


def test_realtabformer_detects_checkpoint_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = REaLTabFormerAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    output_dir = tmp_path / "output"
    train_spec = RunSpec(
        model="realtabformer",
        dataset="fixture",
        output_dir=output_dir,
        extra={"dataset_spec": dataset.to_dict(), "n_critic": 0, "num_bootstrap": 0},
    )
    adapter.train(train_spec)
    (output_dir / "realtabformer_model" / "id0001" / "rtf_model.pt").write_bytes(b"tampered")
    sample_spec = RunSpec(
        model="realtabformer",
        dataset="fixture",
        output_dir=output_dir,
        num_samples=2,
        extra={"dataset_spec": dataset.to_dict()},
    )
    with pytest.raises(RuntimeError, match="integrity manifest"):
        adapter.sample(sample_spec)


def test_realtabformer_manifest_records_exact_official_release() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        repo_root / "standardized_tabular_diffusion" / "resources" / "upstream" / "realtabformer-wheel-manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["package"]["version"] == "0.2.4"
    assert payload["package"]["sha256"] == ("852436c5c82a0bf470ca7e9063e5a4f3e250b3ff5b9c8f6c50113c1e9ba76486")
    assert payload["source"]["commit"] == "73f239643f9ea5abc877f685ce927e986302ac2d"
    assert payload["source"]["license"] == "MIT"
    assert len(payload["installed_files"]) == payload["package"]["record_hashed_files"] == 16
    assert payload["wheel_source_comparison"]["exact_shared_source_files"] == 11
