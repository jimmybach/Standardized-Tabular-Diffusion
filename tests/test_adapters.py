from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, SampleConfig, TrainConfig
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.models.tabddpm import TabDDPMAdapter
from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter
from standardized_tabular_diffusion.runner import validate_action_inputs


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
        output_path.write_text(json.dumps({"dataset": dataset, "metrics_paths": {k: None if v is None else str(v) for k, v in metrics_paths.items()}}, indent=2))

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
