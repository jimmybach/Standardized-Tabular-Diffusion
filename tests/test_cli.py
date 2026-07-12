from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, SampleConfig, TrainConfig
from standardized_tabular_diffusion.interfaces import ArtifactBundle


def test_cli_list_model_inventory_filters_by_benchmark(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-model-inventory", "--benchmark", "tabstruct-2026"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    names = {entry["name"] for entry in payload["models"]}

    assert "tabddpm" in names
    assert "smote" in names
    assert "ctab-gan-plus" not in names


def test_cli_show_model_inventory_prints_expected_entry(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "realtabformer"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "realtabformer"
    assert payload["family"] == "llm"
    assert payload["runnable_recommendation"] == "yes"


def test_cli_list_models_includes_new_sample_based_adapters(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-models"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "ctgan" in payload["models"]
    assert "tvae" in payload["models"]
    assert "smote" in payload["models"]
    assert "ctab-gan-plus" in payload["models"]
    assert "realtabformer" in payload["models"]
    assert "nrgboost" in payload["models"]
    assert "bn" in payload["models"]
    assert "nflow" in payload["models"]
    assert "goggle" in payload["models"]
    assert "great" in payload["models"]
    assert "arf" in payload["models"]
    assert "tabebm" in payload["models"]


def test_cli_run_command_dispatches_pipeline_and_saves_result(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "artifacts" / "run"
    config_path.write_text(
        json.dumps(
            {
                "model": "tabsyn",
                "dataset": "adult",
                "output_dir": str(output_dir),
                "train": {"enabled": False},
                "sample": {"enabled": False},
                "evaluation": {"enabled": False},
            }
        )
    )

    observed: dict[str, object] = {}

    def fake_run_pipeline(config: ExperimentConfig):
        observed["run_pipeline_model"] = config.model
        return {"context": {"dataset": config.dataset}, "phases": {"train": {"model": config.model}}}

    def fake_save_pipeline_result(result: dict, out_dir: str | Path):
        observed["saved_result"] = result
        observed["saved_output_dir"] = str(out_dir)
        return Path(out_dir) / "pipeline_result.json"

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli, "save_pipeline_result", fake_save_pipeline_result)
    monkeypatch.setattr(sys, "argv", ["std-cli", "run", "--config", str(config_path)])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert observed["run_pipeline_model"] == "tabsyn"
    assert observed["saved_output_dir"] == str(output_dir)
    assert payload["phases"]["train"]["model"] == "tabsyn"


def test_cli_run_action_builds_context_saves_it_and_prints_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "artifacts" / "action"
    config_path.write_text(
        json.dumps(
            {
                "model": "tabddpm",
                "dataset": "adult",
                "output_dir": str(output_dir),
                "upstream_config_path": str(tmp_path / "adult.toml"),
                "train": {"enabled": False},
                "sample": {"enabled": False},
                "evaluation": {"enabled": False},
            }
        )
    )

    (tmp_path / "adult.toml").write_text("seed = 42\n")

    observed: dict[str, object] = {}
    bundle = ArtifactBundle(
        model="tabddpm",
        dataset="adult",
        output_dir=output_dir,
        upstream_workdir=tmp_path / "TabDDPM-main",
        notes=["ok"],
    )

    def fake_build_run_context(config: ExperimentConfig):
        observed["context_model"] = config.model
        return {"config": {"model": config.model}, "action_readiness": {"evaluate": {"ready": True}}}

    def fake_save_run_context(context: dict, out_dir: str | Path):
        observed["saved_context"] = context
        observed["saved_context_dir"] = str(out_dir)
        return Path(out_dir) / "run_context.json"

    def fake_run_action(config: ExperimentConfig, action: str):
        observed["action"] = action
        return bundle

    monkeypatch.setattr(cli, "build_run_context", fake_build_run_context)
    monkeypatch.setattr(cli, "save_run_context", fake_save_run_context)
    monkeypatch.setattr(cli, "run_action", fake_run_action)
    monkeypatch.setattr(sys, "argv", ["std-cli", "run-action", "--config", str(config_path), "--action", "evaluate"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert observed["context_model"] == "tabddpm"
    assert observed["action"] == "evaluate"
    assert observed["saved_context_dir"] == str(output_dir)
    assert payload["model"] == "tabddpm"
    assert payload["notes"] == ["ok"]


def test_cli_compare_writes_csv_and_prints_sorted_rows(tmp_path: Path, monkeypatch, capsys) -> None:
    csv_path = tmp_path / "comparison.csv"
    observed: dict[str, object] = {}
    frame = pd.DataFrame(
        [
            {"dataset": "adult", "model": "tabdiff", "ml_primary_value": 0.5},
            {"dataset": "adult", "model": "tabsyn", "ml_primary_value": 0.6},
        ]
    )

    def fake_compare_summaries(summary_paths):
        observed["summary_paths"] = [str(path) for path in summary_paths]
        return frame

    monkeypatch.setattr(cli, "compare_summaries", fake_compare_summaries)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-cli",
            "compare",
            "--summary",
            str(tmp_path / "a.json"),
            "--summary",
            str(tmp_path / "b.json"),
            "--csv",
            str(csv_path),
        ],
    )

    cli.main()
    captured = capsys.readouterr()

    assert observed["summary_paths"] == [str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    assert csv_path.exists()
    assert "tabdiff" in captured.out
    assert "tabsyn" in captured.out
