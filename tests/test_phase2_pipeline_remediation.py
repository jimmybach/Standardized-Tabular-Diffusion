from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.config import (
    EvaluationConfig,
    ExperimentConfig,
    SampleConfig,
    TrainConfig,
)
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models import goggle as goggle_module
from standardized_tabular_diffusion.models.goggle import GoggleAdapter
from standardized_tabular_diffusion.models.great import GReaTAdapter
from standardized_tabular_diffusion.models.sample_baselines import CTGANAdapter
from standardized_tabular_diffusion.models.tabddpm import TabDDPMAdapter
from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter
from standardized_tabular_diffusion.models.tabsyn import TabSynAdapter
from standardized_tabular_diffusion.models.tabula import TabulaAdapter
from standardized_tabular_diffusion.registry import get_adapter
from standardized_tabular_diffusion.runner import save_run_context
from standardized_tabular_diffusion.runtime_contracts import (
    claim_output_identity,
    dataset_content_identity,
    observe_torch_model_device,
    resolve_torch_training_device,
    validate_action_controls,
)

AFFECTED_CONTROL_MODELS = (
    "arf",
    "bn",
    "ctab-gan",
    "ctab-gan-plus",
    "ctgan",
    "great",
    "nflow",
    "nrgboost",
    "smote",
    "tabddpm",
    "tabdiff",
    "tabebm",
    "tabsds",
    "tabsyn",
    "tabula",
    "tvae",
)


@pytest.mark.parametrize("model", AFFECTED_CONTROL_MODELS)
def test_phase2_unknown_top_level_controls_fail_closed(model: str) -> None:
    with pytest.raises(ValueError, match="misspelled_control"):
        validate_action_controls(model, "train", {"misspelled_control": 1})


def _dataset(tmp_path: Path) -> DatasetSpec:
    metadata = tmp_path / "data" / "toy" / "info.json"
    train = metadata.with_name("train.csv")
    test = metadata.with_name("test.csv")
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"name":"toy"}', encoding="utf-8")
    train.write_text("x,cat,label\n1,a,no\n2,b,yes\n", encoding="utf-8")
    test.write_text("x,cat,label\n3,a,no\n", encoding="utf-8")
    return DatasetSpec(
        name="toy",
        task_type="classification",
        column_names=["x", "cat", "label"],
        numerical_columns=["x"],
        categorical_columns=["cat"],
        target_columns=["label"],
        metadata_path=metadata,
        train_data_path=train,
        test_data_path=test,
        extra={"column_info": {"x": "int", "cat": "str", "label": "str"}},
    )


def test_phase2_runspec_binds_dataset_bytes_and_detects_late_mutation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    adapter = CTGANAdapter(tmp_path)
    config = ExperimentConfig(
        model="ctgan",
        dataset="toy",
        output_dir=str(tmp_path / "output"),
        train=TrainConfig(),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    spec = adapter.build_run_spec(config, dataset_spec=dataset, action="train")
    identity = spec.extra["dataset_identity"]
    assert identity["files"]["train"]["bytes"] == dataset.train_data_path.stat().st_size
    assert len(identity["files"]["train"]["sha256"]) == 64

    dataset.train_data_path.write_text("x,cat,label\n9,z,no\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset content changed"):
        adapter.resolve_dataset_spec(spec)


def test_phase2_output_identity_rejects_a_different_seed_and_merges_actions(tmp_path: Path) -> None:
    output = tmp_path / "run"
    claim_output_identity(
        output,
        model="ctgan",
        dataset="toy",
        actions={"train": {"seed": 3}},
    )
    claim_output_identity(
        output,
        model="ctgan",
        dataset="toy",
        actions={"sample": {"seed": 11}},
    )
    with pytest.raises(FileExistsError, match="different sample identity"):
        claim_output_identity(
            output,
            model="ctgan",
            dataset="toy",
            actions={"sample": {"seed": 12}},
        )
    identity = json.loads((output / ".standardized-run-identity.json").read_text(encoding="utf-8"))
    assert set(identity["actions"]) == {"train", "sample"}

    foreign = tmp_path / "foreign"
    (foreign / ".orchestration").mkdir(parents=True)
    (foreign / "unrelated.txt").write_text("not a run", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty output_dir"):
        claim_output_identity(
            foreign,
            model="ctgan",
            dataset="toy",
            actions={"train": {"seed": 3}},
        )


def test_phase2_saved_context_claims_a_compatible_output_identity(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = ExperimentConfig(
        model="ctgan",
        dataset="toy",
        output_dir=str(output),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    context = {
        "config": config.to_dict(),
        "run_spec": {"extra": {"dataset_identity": {"schema_version": 1}}},
    }
    save_run_context(context, output)
    claim_output_identity(
        output,
        model="ctgan",
        dataset="toy",
        actions={"train": {"seed": 0}},
    )
    identity = json.loads((output / ".standardized-run-identity.json").read_text(encoding="utf-8"))
    assert set(identity["actions"]) == {"dataset", "train"}
    with pytest.raises(FileExistsError, match="different dataset identity"):
        claim_output_identity(
            output,
            model="ctgan",
            dataset="toy",
            actions={"dataset": {"schema_version": 1, "dataset_identity": {"schema_version": 2}}},
        )


def test_phase2_trainer_adapters_use_and_observe_explicit_device_contracts() -> None:
    contract = resolve_torch_training_device("cpu")
    assert contract["requested"] == "cpu"
    assert contract["trainer_use_cpu"] is True

    class Parameter:
        device = "cpu"

    class Model:
        @staticmethod
        def parameters():
            return iter((Parameter(),))

    assert observe_torch_model_device(Model(), "cpu") == "cpu"
    for adapter_class in (GReaTAdapter, TabulaAdapter):
        source = inspect.getsource(adapter_class.train)
        assert "resolve_torch_training_device(spec.device)" in source
        assert 'parameters["train_kwargs"]["use_cpu"]' in source
        assert "observe_torch_model_device(model" in source


@pytest.mark.parametrize("model", ("nrgboost", "smote", "tabsds"))
def test_phase2_cpu_only_adapters_reject_cuda_before_model_execution(model: str, tmp_path: Path) -> None:
    adapter = get_adapter(model, repo_root=tmp_path)
    with pytest.raises(ValueError, match="CPU-only"):
        adapter.train(RunSpec(model=model, dataset="toy", output_dir=tmp_path / model, device="cuda"))


def _write_tabddpm_native_arrays(dataset: DatasetSpec) -> None:
    root = dataset.metadata_path.parent
    np.save(root / "X_num_train.npy", np.array([[1.0], [2.0]]), allow_pickle=False)
    np.save(root / "X_num_test.npy", np.array([[3.0]]), allow_pickle=False)
    np.save(root / "X_cat_train.npy", np.array([["a"], ["b"]]), allow_pickle=False)
    np.save(root / "X_cat_test.npy", np.array([["a"]]), allow_pickle=False)
    np.save(root / "y_train.npy", np.array([0, 1]), allow_pickle=False)
    np.save(root / "y_test.npy", np.array([0]), allow_pickle=False)


def _write_tabddpm_config(path: Path) -> None:
    path.write_text(
        """parent_dir = "upstream-output"
real_data_path = "upstream-data"
num_numerical_features = 1
model_type = "mlp"
seed = 99
device = "cuda:7"

[model_params]
num_classes = 2
is_y_cond = true

[diffusion_params]
num_timesteps = 2
gaussian_loss_type = "mse"

[train.main]
steps = 1
lr = 0.001
weight_decay = 0.0
batch_size = 2

[train.T]
seed = 99
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = 99
batch_size = 2
seed = 99
""",
        encoding="utf-8",
    )


def test_phase2_tabddpm_binds_controls_decodes_output_and_owns_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    _write_tabddpm_native_arrays(dataset)
    upstream = tmp_path / "TabDDPM-main"
    upstream.mkdir()
    config_path = tmp_path / "tabddpm.toml"
    _write_tabddpm_config(config_path)
    output = tmp_path / "run"
    config = ExperimentConfig(
        model="tabddpm",
        dataset="toy",
        output_dir=str(output),
        upstream_config_path=str(config_path),
        train=TrainConfig(seed=3, device="cpu"),
        sample=SampleConfig(seed=11, num_samples=2),
        evaluation=EvaluationConfig(enabled=False),
    )
    adapter = TabDDPMAdapter(tmp_path)
    observed: list[tuple[str, dict[str, object]]] = []

    def fake_pipeline(runtime_config: Path, action: str) -> None:
        with runtime_config.open("rb") as stream:
            payload = tomllib.load(stream)
        observed.append((action, payload))
        runtime_root = Path(payload["parent_dir"])
        if action == "train":
            (runtime_root / "model.pt").write_bytes(b"model")
            (runtime_root / "model_ema.pt").write_bytes(b"ema")
        else:
            np.save(runtime_root / "X_num_train.npy", np.array([[1.2], [2.8]]), allow_pickle=False)
            np.save(runtime_root / "X_cat_train.npy", np.array([["a"], ["b"]]), allow_pickle=False)
            np.save(runtime_root / "y_train.npy", np.array([0, 1]), allow_pickle=False)

    monkeypatch.setattr(adapter, "_run_pipeline", fake_pipeline)
    adapter.train_from_config(config, dataset_spec=dataset)
    bundle = adapter.sample_from_config(config, dataset_spec=dataset)

    train_payload = observed[0][1]
    sample_payload = observed[1][1]
    assert train_payload["train"]["main"]["seed"] == 3
    assert train_payload["train"]["T"]["seed"] == 3
    assert train_payload["device"] == "cpu"
    assert sample_payload["sample"] == {"num_samples": 2, "batch_size": 2, "seed": 11}
    assert Path(train_payload["parent_dir"]).is_relative_to(output)
    assert Path(train_payload["real_data_path"]).is_relative_to(output)
    assert bundle.generated_sample_path == output / "samples-seed-11.csv"
    generated = pd.read_csv(bundle.generated_sample_path)
    assert generated.to_dict(orient="list") == {
        "x": [1, 3],
        "cat": ["a", "b"],
        "label": ["no", "yes"],
    }
    assert not list(upstream.rglob("*.pt"))


def test_phase2_goggle_sampling_passes_the_independent_sample_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / "model.pt"
    checkpoint.write_bytes(b"weights")
    (output / "goggle-runtime-config.json").write_text("{}", encoding="utf-8")
    adapter = GoggleAdapter(tmp_path)
    dataset_metadata = tmp_path / "dataset-metadata.json"
    dataset_metadata.write_text("{}", encoding="utf-8")
    dataset_spec = DatasetSpec(
        name="toy",
        task_type="regression",
        column_names=["x"],
        numerical_columns=[],
        categorical_columns=[],
        target_columns=["x"],
        metadata_path=dataset_metadata,
    )
    source_record = {"manifest_sha256": "a" * 64, "upstream_commit": adapter.upstream_commit}
    monkeypatch.setattr(goggle_module, "validate_upstream_source", lambda *_args, **_kwargs: source_record)
    monkeypatch.setattr(adapter, "_source_root", lambda _spec: source)
    monkeypatch.setattr(adapter, "_validate_trusted_executable_artifact", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(adapter, "resolve_dataset_spec", lambda _spec: dataset_spec)
    monkeypatch.setattr(
        adapter,
        "_load_metadata",
        lambda *_args, **_kwargs: {
            "transform": {
                "training_rows": 2,
                "input_dim": 1,
                "column_names": ["x"],
                "task_type": "regression",
                "integer_columns": [],
            }
        },
    )
    monkeypatch.setattr(adapter, "_inverse_transform", lambda raw, _transform: pd.DataFrame({"x": raw[:, 0]}))
    calls: list[list[str]] = []

    def fake_run(args: list[str]) -> None:
        calls.append(args)
        raw_path = Path(args[args.index("--raw-output") + 1])
        np.save(raw_path, np.ones((2, 1)), allow_pickle=False)

    monkeypatch.setattr(adapter, "_run_goggle", fake_run)
    adapter.sample(
        RunSpec(
            model="goggle",
            dataset="toy",
            output_dir=output,
            seed=41,
            num_samples=2,
            extra={"source_dir": str(source)},
        )
    )
    assert calls[0][calls[0].index("--seed") + 1] == "41"


def test_phase2_ctgan_resets_loaded_model_random_state_before_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CTGANAdapter(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / "model.pkl"
    checkpoint.write_bytes(b"checkpoint")
    states: list[int] = []

    class Model:
        @staticmethod
        def set_random_state(seed: int) -> None:
            states.append(seed)

        @staticmethod
        def sample(rows: int) -> pd.DataFrame:
            return pd.DataFrame({"x": range(rows), "label": ["a"] * rows})

    dataset = DatasetSpec(
        name="toy",
        task_type="classification",
        column_names=["x", "label"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["label"],
        metadata_path=tmp_path / "info.json",
        train_data_path=tmp_path / "train.csv",
    )
    dataset.metadata_path.write_text("{}", encoding="utf-8")
    dataset.train_data_path.write_text("x,label\n1,a\n2,b\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "_load_model", lambda *_args: Model())
    monkeypatch.setattr(adapter, "resolve_dataset_spec", lambda _spec: dataset)
    adapter.sample(
        RunSpec(
            model="ctgan",
            dataset="toy",
            output_dir=output,
            checkpoint_path=checkpoint,
            seed=37,
            num_samples=2,
        )
    )
    assert states == [37]


def test_phase2_tabdiff_and_tabsyn_runtime_paths_are_run_owned(tmp_path: Path) -> None:
    spec = RunSpec(model="tabdiff", dataset="toy", output_dir=tmp_path / "run")
    assert TabDiffAdapter(tmp_path)._runtime_root(spec).is_relative_to(spec.output_dir)
    tabsyn = TabSynAdapter(tmp_path)
    assert tabsyn._vae_ckpt_dir(spec).is_relative_to(spec.output_dir)
    assert tabsyn._diffusion_ckpt_dir(spec).is_relative_to(spec.output_dir)
    assert dataset_content_identity(_dataset(tmp_path / "identity"))["schema_version"] == 1


def test_phase2_tabdiff_materializes_complete_run_owned_dataset_views(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "TabDiff-main")
    default_config = tmp_path / "TabDiff-main" / "tabdiff" / "configs" / "tabdiff_configs.toml"
    default_config.parent.mkdir(parents=True)
    default_config.write_text("[data]\ndequant_dist = 'round'\n", encoding="utf-8")
    spec = RunSpec(
        model="tabdiff",
        dataset=dataset.name,
        output_dir=tmp_path / "run",
        extra={
            "dataset_spec": dataset.to_dict(),
            "dataset_identity": dataset_content_identity(dataset),
        },
    )
    adapter = TabDiffAdapter(tmp_path)

    binding = adapter._prepare_runtime(spec)

    runtime = spec.output_dir / "tabdiff-runtime"
    assert (runtime / "data" / "toy" / "info.json").read_bytes() == dataset.metadata_path.read_bytes()
    assert (runtime / "data" / "toy" / "train.csv").read_bytes() == dataset.train_data_path.read_bytes()
    assert (runtime / "synthetic" / "toy" / "real.csv").read_bytes() == dataset.train_data_path.read_bytes()
    assert (runtime / "synthetic" / "toy" / "test.csv").read_bytes() == dataset.test_data_path.read_bytes()
    assert binding["synthetic_view"]["files"]["real.csv"]["canonical_path"] == str(
        dataset.train_data_path.resolve()
    )

    (runtime / "synthetic" / "toy" / "real.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs from its canonical source"):
        adapter._prepare_runtime(spec)
