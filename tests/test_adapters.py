from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

pytestmark = pytest.mark.adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, SampleConfig, TrainConfig
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.models.final_wave_baselines import ARFAdapter, GReaTAdapter, TabEBMAdapter
from standardized_tabular_diffusion.models.next_wave_baselines import (
    CTABGANPlusAdapter,
    NRGBoostAdapter,
    REaLTabFormerAdapter,
)
from standardized_tabular_diffusion.models.paper_gap_baselines import TabSDSAdapter, TabularARGNAdapter
from standardized_tabular_diffusion.models.sample_baselines import CTGANAdapter, SMOTEAdapter
from standardized_tabular_diffusion.models.structured_baselines import BNAdapter, GoggleAdapter, NFlowAdapter
from standardized_tabular_diffusion.models.tabddpm import TabDDPMAdapter
from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter
from standardized_tabular_diffusion.models.tabsyn import TabSynAdapter
from standardized_tabular_diffusion.models.tabula import TabulaAdapter
from standardized_tabular_diffusion.models.vendored_baselines import CoDiAdapter, CTABGANAdapter, STaSyAdapter
from standardized_tabular_diffusion.runner import validate_action_inputs


class PickleableFakeSynth:
    def fit(self, train_data, discrete_columns=()):
        self.train_shape = train_data.shape
        self.discrete_columns = list(discrete_columns)

    def sample(self, samples):
        return pd.DataFrame({"x": [10] * samples, "y": [1] * samples})


class PickleableFakeCTABGAN:
    def fit(self):
        return None

    def generate_samples(self, samples, seed=0):
        return pd.DataFrame({"0": [10] * samples, "y": [1] * samples})


class FakeTabulaTokenizer:
    generated_text = "age=42<|col|>city=Boston<|col|>label=1<|endrow|>"

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def add_special_tokens(self, payload):
        self.special_tokens = payload
        return len(payload.get("additional_special_tokens", []))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")

    def __len__(self):
        return 32

    def __call__(self, texts, truncation=False, padding=False, max_length=None, return_tensors=None):
        if isinstance(texts, str):
            return {"input_ids": [1, 2, 3]}
        seq_len = max_length or 8
        rows = len(texts)
        input_ids = torch.ones((rows, seq_len), dtype=torch.long)
        attention_mask = torch.ones((rows, seq_len), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=False):
        return self.generated_text

    def convert_tokens_to_ids(self, token):
        return 7


class FakeTabulaConfig:
    n_layer = 2
    n_head = 2
    n_embd = 32
    n_positions = 64

    @classmethod
    def from_pretrained(cls, path):
        return cls()


class FakeTabulaModel:
    def __init__(self):
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    @classmethod
    def from_config(cls, config):
        model = cls()
        model.config = config
        return model

    def resize_token_embeddings(self, size):
        self.embedding_size = size

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.bin").write_text("stub")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return None

    def generate(self, **kwargs):
        return torch.tensor([[999]], dtype=torch.long)


class FakeTrainingArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTrainer:
    def __init__(self, model, args, train_dataset, data_collator):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator

    def train(self):
        return None


class FakeTabularARGN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, df):
        self.train_df = df.copy()

    def sample(self, n_samples):
        return pd.DataFrame({"x": [11] * n_samples, "y": [1] * n_samples})


def test_tabsyn_train_uses_authoritative_epoch_default_and_does_not_reuse_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "TabSyn-main").mkdir()
    adapter = TabSynAdapter(tmp_path)
    commands: list[tuple[list[str], bool]] = []

    def fake_run_python(args: list[str], _cwd: Path, *, module: bool = False) -> None:
        commands.append((args, module))

    monkeypatch.setattr(adapter, "_run_python", fake_run_python)
    spec = ExperimentConfig(
        model="tabsyn",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts"),
        train=TrainConfig(),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec("train")

    adapter.train(spec)

    assert commands[0] == (["tabsyn.vae.main", "--dataname", "adult", "--gpu", "0"], True)
    assert commands[1][1] is False
    assert commands[1][0][-2:] == ["--num_epochs", "10001"]


def test_tabdiff_sample_infers_generated_sample_path_and_builds_expected_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    upstream_root = repo_root / "TabDiff-main"
    ckpt_dir = upstream_root / "tabdiff" / "ckpt" / "adult" / "exp-smoke"
    result_dir = upstream_root / "tabdiff" / "result" / "adult" / "exp-smoke" / "7"
    ckpt_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)

    checkpoint_path = ckpt_dir / "best_ema_model_7.pt"
    checkpoint_path.write_text("stub")
    sample_path = result_dir / "samples.csv"
    sample_path.write_text("a,b\n1,2\n")

    adapter = TabDiffAdapter(repo_root)
    commands: list[tuple[list[str], Path]] = []

    def fake_run_python(args: list[str], cwd: Path, *, module: bool = False) -> None:
        assert not module
        commands.append((args, cwd))

    monkeypatch.setattr(adapter, "_run_python", fake_run_python)
    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["a", "b"],
        numerical_columns=["a"],
        categorical_columns=[],
        target_columns=["b"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("a,b\n1,0\n")
    dataset_spec.test_data_path.write_text("a,b\n1,0\n")

    config = ExperimentConfig(
        model="tabdiff",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabdiff-sample"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(
            enabled=True,
            num_samples=512,
            extra={"exp_name": "exp-smoke", "gpu": 1, "no_wandb": True},
        ),
        evaluation=EvaluationConfig(enabled=False),
    )

    bundle = adapter.sample_from_config(config, dataset_spec=dataset_spec)

    assert commands == [
        (
            [
                "main.py",
                "--dataname",
                "adult",
                "--mode",
                "test",
                "--gpu",
                "1",
                "--exp_name",
                "exp-smoke",
                "--ckpt_path",
                str(checkpoint_path),
                "--num_samples_to_generate",
                "512",
                "--no_wandb",
            ],
            upstream_root,
        )
    ]
    assert bundle.generated_sample_path == sample_path
    assert json.loads((bundle.output_dir / "artifacts.json").read_text())["generated_sample_path"] == str(sample_path)


def test_tabddpm_train_and_sample_require_upstream_config_and_evaluate_normalizes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    upstream_root = repo_root / "TabDDPM-main"
    upstream_root.mkdir(parents=True)
    config_path = tmp_path / "adult.toml"
    config_path.write_text("seed = 42\n")

    adapter = TabDDPMAdapter(repo_root)
    commands: list[tuple[list[str], Path]] = []

    def fake_run_python(args: list[str], cwd: Path, *, module: bool = False) -> None:
        assert not module
        commands.append((args, cwd))

    monkeypatch.setattr(adapter, "_run_python", fake_run_python)
    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["a", "b"],
        numerical_columns=["a"],
        categorical_columns=[],
        target_columns=["b"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("a,b\n1,0\n")
    dataset_spec.test_data_path.write_text("a,b\n1,0\n")

    normalized_calls: list[tuple[str, Path, dict[str, Path | None]]] = []

    def fake_normalize(dataset: str, output_path: Path, metrics_paths: dict[str, Path | None]) -> None:
        normalized_calls.append((dataset, output_path, metrics_paths))
        output_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "metrics_paths": {k: None if v is None else str(v) for k, v in metrics_paths.items()},
                },
                indent=2,
            )
        )

    monkeypatch.setattr("standardized_tabular_diffusion.models.tabddpm.normalize_tabddpm_summary", fake_normalize)

    train_config = ExperimentConfig(
        model="tabddpm",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabddpm-train"),
        upstream_config_path=str(config_path),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="tabddpm",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabddpm-sample"),
        upstream_config_path=str(config_path),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    )

    train_bundle = adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    sample_bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    catboost_path = tmp_path / "catboost.json"
    catboost_path.write_text("{}")
    eval_config = ExperimentConfig(
        model="tabddpm",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabddpm-eval"),
        upstream_config_path=str(config_path),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=True, extra={"results_catboost_path": str(catboost_path)}),
    )
    eval_bundle = adapter.evaluate_from_config(eval_config, dataset_spec=dataset_spec)

    assert commands == [
        (["scripts/pipeline.py", "--config", str(config_path), "--train"], upstream_root),
        (["scripts/pipeline.py", "--config", str(config_path), "--sample"], upstream_root),
    ]
    assert train_bundle.output_dir.joinpath("artifacts.json").exists()
    assert sample_bundle.output_dir.joinpath("artifacts.json").exists()
    assert eval_bundle.standardized_summary_path == Path(eval_config.output_dir) / "standardized_summary.json"
    assert normalized_calls == [
        (
            "adult",
            Path(eval_config.output_dir) / "standardized_summary.json",
            {
                "catboost": catboost_path,
                "mlp": None,
                "privacy": None,
                "simple": None,
            },
        )
    ]


def test_validate_action_inputs_covers_tabddpm_and_tabdiff_contracts(tmp_path: Path) -> None:
    sample_path = tmp_path / "samples.csv"
    sample_path.write_text("x\n1\n")
    config_path = tmp_path / "config.toml"
    config_path.write_text("seed = 1\n")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}")

    dataset_spec = type(
        "Spec",
        (),
        {
            "name": "adult",
            "task_type": "classification",
            "metadata_path": tmp_path / "info.json",
            "train_data_path": tmp_path / "train.csv",
            "val_data_path": None,
            "test_data_path": tmp_path / "test.csv",
        },
    )()
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x\n1\n")
    dataset_spec.test_data_path.write_text("x\n1\n")

    tabdiff_eval = ExperimentConfig(
        model="tabdiff",
        dataset="adult",
        output_dir=str(tmp_path / "out-tabdiff"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=True, extra={"sample_path": str(sample_path)}),
    )
    tabddpm_eval = ExperimentConfig(
        model="tabddpm",
        dataset="adult",
        output_dir=str(tmp_path / "out-tabddpm"),
        upstream_config_path=str(config_path),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=True, extra={"results_catboost_path": str(metrics_path)}),
    )

    ready_tabdiff = validate_action_inputs(tabdiff_eval, "evaluate", dataset_spec=dataset_spec)
    ready_tabddpm = validate_action_inputs(tabddpm_eval, "evaluate", dataset_spec=dataset_spec)
    ready_tabddpm_train = validate_action_inputs(tabddpm_eval, "train", dataset_spec=dataset_spec)

    assert ready_tabdiff["ready"] is True
    assert ready_tabddpm["ready"] is True
    assert ready_tabddpm_train["ready"] is True


def test_ctgan_train_and_sample_use_local_pickle_checkpoint(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "TabDDPM-main").mkdir(parents=True)
    adapter = CTGANAdapter(repo_root)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n2,1\n")
    dataset_spec.test_data_path.write_text("x,y\n3,1\n")

    monkeypatch.setattr(adapter, "_build_synthesizer", lambda spec: PickleableFakeSynth())

    train_config = ExperimentConfig(
        model="ctgan",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "ctgan"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="ctgan",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "ctgan"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=3),
        evaluation=EvaluationConfig(enabled=False),
    )

    adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    assert (Path(train_config.output_dir) / "model.pkl").exists()
    assert bundle.generated_sample_path is not None
    assert pd.read_csv(bundle.generated_sample_path).shape == (3, 2)


def test_smote_sample_generates_requested_rows_and_requires_classification(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "TabSyn-main").mkdir(parents=True)
    adapter = SMOTEAdapter(repo_root)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n2,0\n3,0\n10,1\n11,1\n")
    dataset_spec.test_data_path.write_text("x,y\n4,0\n")

    class FakeSMOTE:
        def __init__(self, random_state=None, k_neighbors=5):
            self.random_state = random_state
            self.k_neighbors = k_neighbors

        def fit_resample(self, x_train, y_train):
            x_df = pd.DataFrame(x_train).reset_index(drop=True)
            y_series = pd.Series(y_train).reset_index(drop=True)
            return pd.concat([x_df, x_df.iloc[[0]]], ignore_index=True), pd.concat(
                [y_series, y_series.iloc[[0]]], ignore_index=True
            )

    fake_imblearn = types.ModuleType("imblearn")
    fake_over_sampling = types.ModuleType("imblearn.over_sampling")
    fake_over_sampling.SMOTE = FakeSMOTE
    fake_imblearn.over_sampling = fake_over_sampling
    monkeypatch.setitem(sys.modules, "imblearn", fake_imblearn)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", fake_over_sampling)

    config = ExperimentConfig(
        model="smote",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "smote"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=4, extra={"k_neighbors": 1}),
        evaluation=EvaluationConfig(enabled=False),
    )
    bundle = adapter.sample_from_config(config, dataset_spec=dataset_spec)

    assert bundle.generated_sample_path is not None
    assert pd.read_csv(bundle.generated_sample_path).shape == (4, 2)


def test_smote_uses_smotenc_for_mixed_type_features(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "TabSyn-main").mkdir(parents=True)
    adapter = SMOTEAdapter(tmp_path)
    dataset_spec = DatasetSpec(
        name="mixed",
        task_type="classification",
        column_names=["x", "color", "target"],
        numerical_columns=["x"],
        categorical_columns=["color"],
        target_columns=["target"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "color": ["red", "blue", "red", "blue", "red", "blue"],
            "target": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(dataset_spec.train_data_path, index=False)
    dataset_spec.test_data_path.write_text("x,color,target\n4,red,0\n")

    class FakeSMOTENC:
        def __init__(self, categorical_features, random_state=None, k_neighbors=5):
            assert categorical_features == [1]

        def fit_resample(self, x_train, y_train):
            x_frame = pd.DataFrame(x_train).reset_index(drop=True)
            assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in x_frame.dtypes)
            y_series = pd.Series(y_train).reset_index(drop=True)
            return x_frame, y_series

    fake_imblearn = types.ModuleType("imblearn")
    fake_over_sampling = types.ModuleType("imblearn.over_sampling")
    fake_over_sampling.SMOTENC = FakeSMOTENC
    fake_imblearn.over_sampling = fake_over_sampling
    monkeypatch.setitem(sys.modules, "imblearn", fake_imblearn)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", fake_over_sampling)

    config = ExperimentConfig(
        model="smote",
        dataset="mixed",
        output_dir=str(tmp_path / "artifacts" / "smote-mixed"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=6, extra={"k_neighbors": 1}),
        evaluation=EvaluationConfig(enabled=False),
    )
    bundle = adapter.sample_from_config(config, dataset_spec=dataset_spec)

    sampled = pd.read_csv(bundle.generated_sample_path)
    metadata = json.loads((Path(config.output_dir) / "smote_metadata.json").read_text())
    assert set(sampled["color"]) <= {"red", "blue"}
    assert metadata["sampler"] == "SMOTENC"
    assert metadata["categorical_indices"] == [1]


def test_ctab_gan_plus_train_and_sample_use_pickle_checkpoint(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "TabDDPM-main" / "CTAB-GAN-Plus").mkdir(parents=True, exist_ok=True)
    (repo_root / "TabDDPM-main").mkdir(parents=True, exist_ok=True)
    columns_path = repo_root / "TabDDPM-main" / "CTAB-GAN-Plus" / "columns.json"
    columns_path.write_text(
        json.dumps(
            {
                "adult": {
                    "categorical_columns": ["y"],
                    "mixed_columns": {},
                    "integer_columns": ["0"],
                    "general_columns": [],
                    "problem_type": {"Classification": "y"},
                }
            }
        )
    )
    adapter = CTABGANPlusAdapter(repo_root)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "target"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["target"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,target\n1,0\n2,1\n")
    dataset_spec.test_data_path.write_text("x,target\n3,1\n")

    monkeypatch.setattr(adapter, "_load_training_frame", lambda dataset_spec: pd.DataFrame({"0": [1, 2], "y": [0, 1]}))
    monkeypatch.setattr(adapter, "_build_model", lambda train_df, spec: PickleableFakeCTABGAN())

    train_config = ExperimentConfig(
        model="ctab-gan-plus",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "ctab-gan-plus"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="ctab-gan-plus",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "ctab-gan-plus"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=3),
        evaluation=EvaluationConfig(enabled=False),
    )

    adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    assert (Path(train_config.output_dir) / "ctabgan_plus.pkl").exists()
    assert bundle.generated_sample_path is not None
    assert pd.read_csv(bundle.generated_sample_path).shape == (3, 2)


def test_realtabformer_train_and_sample_with_stubbed_package(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "TabSyn-main").mkdir(parents=True)
    adapter = REaLTabFormerAdapter(repo_root)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n2,1\n")
    dataset_spec.test_data_path.write_text("x,y\n3,1\n")

    class FakeRTF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, df, **kwargs):
            self.df = df
            self.fit_kwargs = kwargs

        def save(self, path):
            model_dir = Path(path) / "id0001"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "marker.txt").write_text("ok")

        def sample(self, n_samples, **kwargs):
            return pd.DataFrame({"x": [10] * n_samples, "y": [1] * n_samples})

        @classmethod
        def load_from_dir(cls, path):
            return cls(loaded_from=path)

    fake_module = types.ModuleType("realtabformer")
    fake_module.REaLTabFormer = FakeRTF
    monkeypatch.setitem(sys.modules, "realtabformer", fake_module)

    train_config = ExperimentConfig(
        model="realtabformer",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "realtabformer"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="realtabformer",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "realtabformer"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=2),
        evaluation=EvaluationConfig(enabled=False),
    )

    adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    assert bundle.generated_sample_path is not None
    assert pd.read_csv(bundle.generated_sample_path).shape == (2, 2)


def test_realtabformer_can_limit_training_rows_for_tiny_smoke_runs(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "TabSyn-main").mkdir(parents=True)
    adapter = REaLTabFormerAdapter(repo_root)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n")
    dataset_spec.test_data_path.write_text("x,y\n5,1\n")

    observed: dict[str, object] = {}

    class FakeRTF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, df, **kwargs):
            observed["fit_rows"] = len(df)

        def save(self, path):
            model_dir = Path(path) / "id0001"
            model_dir.mkdir(parents=True, exist_ok=True)

    fake_module = types.ModuleType("realtabformer")
    fake_module.REaLTabFormer = FakeRTF
    monkeypatch.setitem(sys.modules, "realtabformer", fake_module)

    train_config = ExperimentConfig(
        model="realtabformer",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "realtabformer-tiny"),
        train=TrainConfig(enabled=True, extra={"max_train_rows": 2, "n_critic": 0, "num_bootstrap": 0}),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )

    adapter.train_from_config(train_config, dataset_spec=dataset_spec)

    assert observed["fit_rows"] == 2


@pytest.mark.skipif(torch is None, reason="TabuLa tensor contract requires the optional PyTorch runtime")
def test_tabula_train_and_sample_with_stubbed_transformers(tmp_path: Path, monkeypatch) -> None:
    adapter = TabulaAdapter(tmp_path)
    imports = types.SimpleNamespace(
        AutoConfig=FakeTabulaConfig,
        AutoModelForCausalLM=FakeTabulaModel,
        AutoTokenizer=FakeTabulaTokenizer,
        Trainer=FakeTrainer,
        TrainingArguments=FakeTrainingArguments,
        default_data_collator=object(),
    )
    monkeypatch.setattr(adapter, "_import_transformer_bits", lambda: imports)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["age", "city", "label"],
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["label"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    pd.DataFrame(
        {
            "age": [21, 35, 42],
            "city": ["Austin", "Boston", "Chicago"],
            "label": [0, 1, 0],
        }
    ).to_csv(dataset_spec.train_data_path, index=False)
    pd.DataFrame({"age": [30], "city": ["Austin"], "label": [0]}).to_csv(dataset_spec.test_data_path, index=False)

    train_config = ExperimentConfig(
        model="tabula",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabula"),
        train=TrainConfig(enabled=True, extra={"epochs": 1, "max_length": 16}),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="tabula",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabula"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=2, extra={"max_tries": 4}),
        evaluation=EvaluationConfig(enabled=False),
    )

    train_bundle = adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    sample_bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    model_root = Path(train_config.output_dir) / "tabula_model"
    assert train_bundle.output_dir == Path(train_config.output_dir)
    assert model_root.exists()
    assert (model_root / "adapter_metadata.json").exists()
    assert sample_bundle.generated_sample_path is not None
    sampled = pd.read_csv(sample_bundle.generated_sample_path)
    assert list(sampled.columns) == ["age", "city", "label"]
    assert len(sampled) == 2


def test_tabula_checkpoint_convention(tmp_path: Path) -> None:
    adapter = TabulaAdapter(tmp_path)
    spec = ExperimentConfig(
        model="tabula",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabula"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()

    assert adapter._model_root(spec).name == "tabula_model"
    assert adapter._metadata_path(adapter._model_root(spec)).name == "adapter_metadata.json"


def test_tabsds_train_and_sample_round_trip(tmp_path: Path) -> None:
    adapter = TabSDSAdapter(tmp_path)
    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["age", "city", "label"],
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["label"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    pd.DataFrame(
        {
            "age": [21, 35, 42, 28],
            "city": ["Austin", "Boston", "Chicago", "Austin"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(dataset_spec.train_data_path, index=False)
    pd.DataFrame({"age": [30], "city": ["Austin"], "label": [0]}).to_csv(dataset_spec.test_data_path, index=False)

    train_config = ExperimentConfig(
        model="tabsds",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabsds"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="tabsds",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabsds"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=3),
        evaluation=EvaluationConfig(enabled=False),
    )

    train_bundle = adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    sample_bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    checkpoint_path = Path(train_config.output_dir) / "tabsds.pkl"
    assert train_bundle.output_dir == Path(train_config.output_dir)
    assert checkpoint_path.exists()
    assert sample_bundle.generated_sample_path is not None
    sampled = pd.read_csv(sample_bundle.generated_sample_path)
    assert list(sampled.columns) == ["age", "city", "label"]
    assert len(sampled) == 3


def test_tabularargn_train_and_sample_with_stubbed_package(tmp_path: Path, monkeypatch) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    fake_engine = types.ModuleType("mostlyai.engine")
    fake_engine.TabularARGN = FakeTabularARGN
    monkeypatch.setitem(sys.modules, "mostlyai.engine", fake_engine)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n2,1\n")
    dataset_spec.test_data_path.write_text("x,y\n3,1\n")

    train_config = ExperimentConfig(
        model="tabularargn",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabularargn"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="tabularargn",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabularargn"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=2),
        evaluation=EvaluationConfig(enabled=False),
    )

    train_bundle = adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    sample_bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    checkpoint_path = Path(train_config.output_dir) / "tabularargn.pkl"
    assert train_bundle.output_dir == Path(train_config.output_dir)
    assert checkpoint_path.exists()
    assert sample_bundle.generated_sample_path is not None
    assert pd.read_csv(sample_bundle.generated_sample_path).shape == (2, 2)


def test_nrgboost_train_and_sample_with_stubbed_package(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    (repo_root / "TabSyn-main").mkdir(parents=True)
    adapter = NRGBoostAdapter(repo_root)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n2,1\n")
    dataset_spec.test_data_path.write_text("x,y\n3,1\n")

    class FakeDataset:
        def __init__(self, df):
            self.df = df

    class FakeBoosterInstance:
        def save(self, path):
            Path(path).write_text("saved")

        def sample(self, n_samples, num_steps=100):
            return pd.DataFrame({"x": [10] * n_samples, "y": [1] * n_samples})

    class FakeBooster:
        @staticmethod
        def fit(dataset, params, seed=0):
            return FakeBoosterInstance()

        @staticmethod
        def load(path):
            return FakeBoosterInstance()

    fake_module = types.ModuleType("nrgboost")
    fake_module.Dataset = FakeDataset
    fake_module.NRGBooster = FakeBooster
    monkeypatch.setitem(sys.modules, "nrgboost", fake_module)

    train_config = ExperimentConfig(
        model="nrgboost",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "nrgboost"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="nrgboost",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "nrgboost"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=2),
        evaluation=EvaluationConfig(enabled=False),
    )

    adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    bundle = adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    assert (Path(train_config.output_dir) / "model.nrgboost").exists()
    assert bundle.generated_sample_path is not None
    assert pd.read_csv(bundle.generated_sample_path).shape == (2, 2)


def test_bn_and_nflow_use_default_pickle_checkpoint_names(tmp_path: Path) -> None:
    bn_adapter = BNAdapter(tmp_path)
    nflow_adapter = NFlowAdapter(tmp_path)

    bn_spec = ExperimentConfig(
        model="bn",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "bn"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()
    nflow_spec = ExperimentConfig(
        model="nflow",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "nflow"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()

    assert bn_adapter._resolve_checkpoint_path(bn_spec).name == "model.pkl"
    assert nflow_adapter._resolve_checkpoint_path(nflow_spec).name == "model.pkl"


def test_goggle_uses_model_pt_checkpoint_name(tmp_path: Path) -> None:
    adapter = GoggleAdapter(tmp_path)
    spec = ExperimentConfig(
        model="goggle",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "goggle"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()

    assert adapter._resolve_checkpoint_path(spec).name == "model.pt"


def test_great_arf_and_tabebm_checkpoint_conventions(tmp_path: Path) -> None:
    great_adapter = GReaTAdapter(tmp_path)
    arf_adapter = ARFAdapter(tmp_path)
    tabebm_adapter = TabEBMAdapter(tmp_path)

    great_spec = ExperimentConfig(
        model="great",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "great"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()
    arf_spec = ExperimentConfig(
        model="arf",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "arf"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()
    tabebm_spec = ExperimentConfig(
        model="tabebm",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "tabebm"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True),
        evaluation=EvaluationConfig(enabled=False),
    ).to_run_spec()

    assert great_adapter._model_root(great_spec).name == "great_model"
    assert arf_adapter._resolve_checkpoint_path(arf_spec).name == "model.pkl"
    assert tabebm_adapter._resolve_checkpoint_path(tabebm_spec).name == "model.pkl"
    assert great_adapter._metadata_path(great_adapter._model_root(great_spec)).name == "adapter_metadata.json"


def test_tabebm_surrogate_negatives_cover_one_feature_and_are_seeded() -> None:
    one_feature = np.array([[0.1], [0.2]], dtype=np.float64)
    augmented, labels = TabEBMAdapter._add_surrogate_negative_samples(
        one_feature,
        5.0,
        rng=np.random.default_rng(7),
    )
    assert augmented.shape == (4, 1)
    assert labels.tolist() == [0, 0, 1, 1]

    multi_feature = np.zeros((2, 4), dtype=np.float64)
    first, _ = TabEBMAdapter._add_surrogate_negative_samples(
        multi_feature,
        5.0,
        rng=np.random.default_rng(11),
    )
    second, _ = TabEBMAdapter._add_surrogate_negative_samples(
        multi_feature,
        5.0,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(first, second)

    with pytest.raises(ValueError, match="at least one feature"):
        TabEBMAdapter._add_surrogate_negative_samples(
            np.empty((2, 0)),
            5.0,
            rng=np.random.default_rng(1),
        )


def test_code_executing_checkpoint_loads_fail_closed_outside_output_dir(tmp_path: Path) -> None:
    adapter = ARFAdapter(tmp_path)
    external_checkpoint = tmp_path / "external.pkl"
    external_checkpoint.write_bytes(b"not-loaded")
    config = ExperimentConfig(
        model="arf",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "arf"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, checkpoint_path=str(external_checkpoint)),
        evaluation=EvaluationConfig(enabled=False),
    )
    spec = config.to_run_spec(action="sample")

    with pytest.raises(PermissionError, match="can execute code"):
        adapter._validate_trusted_executable_artifact(spec, external_checkpoint, format_name="pickle")

    config.sample.extra["allow_unsafe_external_checkpoint"] = True
    trusted_spec = config.to_run_spec(action="sample")
    assert (
        adapter._validate_trusted_executable_artifact(
            trusted_spec,
            external_checkpoint,
            format_name="pickle",
        )
        == external_checkpoint.resolve()
    )


def test_stasy_and_codi_build_expected_tabsyn_dispatch_commands(tmp_path: Path, monkeypatch) -> None:
    stasy_adapter = STaSyAdapter(tmp_path)
    codi_adapter = CoDiAdapter(tmp_path)
    commands: list[tuple[list[str], Path]] = []

    def fake_run_python(args: list[str], cwd: Path, *, module: bool = False) -> None:
        assert not module
        commands.append((args, cwd))

    monkeypatch.setattr(stasy_adapter, "_run_python", fake_run_python)
    monkeypatch.setattr(codi_adapter, "_run_python", fake_run_python)

    train_config = ExperimentConfig(
        model="stasy",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "stasy"),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    sample_config = ExperimentConfig(
        model="codi",
        dataset="adult",
        output_dir=str(tmp_path / "artifacts" / "codi"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, num_samples=32, extra={"steps": 25}),
        evaluation=EvaluationConfig(enabled=False),
    )

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n")
    dataset_spec.test_data_path.write_text("x,y\n1,0\n")

    stasy_adapter.train_from_config(train_config, dataset_spec=dataset_spec)
    bundle = codi_adapter.sample_from_config(sample_config, dataset_spec=dataset_spec)

    assert commands == [
        (
            ["main.py", "--method", "stasy", "--mode", "train", "--dataname", "adult", "--gpu", "0"],
            tmp_path / "TabSyn-main",
        ),
        (
            [
                "main.py",
                "--method",
                "codi",
                "--mode",
                "sample",
                "--dataname",
                "adult",
                "--gpu",
                "0",
                "--steps",
                "25",
                "--save_path",
                str((Path(sample_config.output_dir) / "samples.csv").resolve()),
                "--num-samples",
                "32",
            ],
            tmp_path / "TabSyn-main",
        ),
    ]
    assert bundle.generated_sample_path == (Path(sample_config.output_dir) / "samples.csv").resolve()


def test_ctabgan_sample_restores_column_names(tmp_path: Path) -> None:
    adapter = CTABGANAdapter(tmp_path)
    checkpoint_path = tmp_path / "artifacts" / "ctab-gan" / "ctabgan.pkl"
    checkpoint_path.parent.mkdir(parents=True)
    with checkpoint_path.open("wb") as handle:
        pickleable = PickleableFakeCTABGAN()
        pickle.dump(pickleable, handle)

    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "y"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["y"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
        test_data_path=tmp_path / "test.csv",
    )
    dataset_dir = dataset_spec.train_data_path.parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x,y\n1,0\n")
    dataset_spec.test_data_path.write_text("x,y\n1,0\n")
    np.save(dataset_dir / "X_num_train.npy", np.array([[1.0]], dtype=np.float32))
    np.save(dataset_dir / "y_train.npy", np.array([[0]], dtype=np.int64))

    config = ExperimentConfig(
        model="ctab-gan",
        dataset="adult",
        output_dir=str(checkpoint_path.parent),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, checkpoint_path=str(checkpoint_path), num_samples=3),
        evaluation=EvaluationConfig(enabled=False),
    )

    bundle = adapter.sample_from_config(config, dataset_spec=dataset_spec)

    assert bundle.generated_sample_path is not None
    assert list(pd.read_csv(bundle.generated_sample_path).columns) == ["x", "y"]


def test_validate_action_inputs_accepts_extended_baseline_sample_contracts(tmp_path: Path, monkeypatch) -> None:
    dataset_spec = type(
        "Spec",
        (),
        {
            "name": "adult",
            "task_type": "classification",
            "metadata_path": tmp_path / "info.json",
            "train_data_path": tmp_path / "train.csv",
            "val_data_path": None,
            "test_data_path": tmp_path / "test.csv",
        },
    )()
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x\n1\n")
    dataset_spec.test_data_path.write_text("x\n1\n")

    for model_name, checkpoint_name in [
        ("bn", "model.pkl"),
        ("ctab-gan", "ctabgan.pkl"),
        ("nflow", "model.pkl"),
        ("goggle", "model.pt"),
        ("great", "great_model"),
        ("tabsds", "tabsds.pkl"),
        ("tabularargn", "tabularargn.pkl"),
        ("tabula", "tabula_model"),
        ("arf", "model.pkl"),
    ]:
        checkpoint = tmp_path / checkpoint_name
        if checkpoint_name.endswith(".pkl") or checkpoint_name.endswith(".pt"):
            checkpoint.write_text("stub")
        else:
            checkpoint.mkdir(exist_ok=True)
        config = ExperimentConfig(
            model=model_name,
            dataset="adult",
            output_dir=str(tmp_path),
            train=TrainConfig(enabled=False),
            sample=SampleConfig(enabled=True, checkpoint_path=str(checkpoint)),
            evaluation=EvaluationConfig(enabled=False),
        )
        ready = validate_action_inputs(config, "sample", dataset_spec=dataset_spec)
        assert ready["ready"] is True

    fake_runner_path = tmp_path / "standardized_tabular_diffusion" / "runner.py"
    fake_runner_path.parent.mkdir(parents=True, exist_ok=True)
    fake_runner_path.write_text("# test shim\n")
    monkeypatch.setattr("standardized_tabular_diffusion.runner.__file__", str(fake_runner_path))
    repo_root = tmp_path
    stasy_ckpt = repo_root / "TabSyn-main" / "baselines" / "stasy" / "ckpt" / "adult"
    codi_ckpt = repo_root / "TabSyn-main" / "baselines" / "codi" / "ckpt" / "adult"
    stasy_ckpt.mkdir(parents=True, exist_ok=True)
    codi_ckpt.mkdir(parents=True, exist_ok=True)
    (stasy_ckpt / "model.pth").write_text("stub")
    (codi_ckpt / "model_con.pt").write_text("stub")
    (codi_ckpt / "model_dis.pt").write_text("stub")

    for model_name in ["stasy", "codi"]:
        config = ExperimentConfig(
            model=model_name,
            dataset="adult",
            output_dir=str(tmp_path),
            train=TrainConfig(enabled=False),
            sample=SampleConfig(enabled=True),
            evaluation=EvaluationConfig(enabled=False),
        )
        ready = validate_action_inputs(config, "sample", dataset_spec=dataset_spec)
        assert ready["ready"] is True

    tabebm_checkpoint = tmp_path / "model.pkl"
    tabebm_checkpoint.write_text("stub")
    tabebm_config = ExperimentConfig(
        model="tabebm",
        dataset="adult",
        output_dir=str(tmp_path),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, checkpoint_path=str(tabebm_checkpoint), extra={"allow_gated_model": True}),
        evaluation=EvaluationConfig(enabled=False),
    )
    ready = validate_action_inputs(tabebm_config, "sample", dataset_spec=dataset_spec)
    assert ready["ready"] is True


def test_validate_action_inputs_rejects_tabebm_sample_without_opt_in(tmp_path: Path) -> None:
    dataset_spec = type(
        "Spec",
        (),
        {
            "name": "adult",
            "task_type": "classification",
            "metadata_path": tmp_path / "info.json",
            "train_data_path": tmp_path / "train.csv",
            "val_data_path": None,
            "test_data_path": tmp_path / "test.csv",
        },
    )()
    dataset_spec.metadata_path.write_text("{}")
    dataset_spec.train_data_path.write_text("x\n1\n")
    dataset_spec.test_data_path.write_text("x\n1\n")
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_text("stub")

    config = ExperimentConfig(
        model="tabebm",
        dataset="adult",
        output_dir=str(tmp_path),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, checkpoint_path=str(checkpoint)),
        evaluation=EvaluationConfig(enabled=False),
    )

    ready = validate_action_inputs(config, "sample", dataset_spec=dataset_spec)

    assert ready["ready"] is False
    assert "allow_gated_model" in " ".join(ready["missing"])
