from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.config import ExperimentConfig
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


def test_cli_list_model_inventory_filters_tabula_benchmark(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-model-inventory", "--benchmark", "tabula-2025"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    names = {entry["name"] for entry in payload["models"]}

    assert "tabula" in names
    assert "tabicl" not in names


def test_cli_list_model_inventory_filters_tabforge_benchmark(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-model-inventory", "--benchmark", "tabforge-2026"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    names = {entry["name"] for entry in payload["models"]}

    assert "tabsds" in names
    assert "cdtd" in names
    assert "tabularargn" in names
    assert "tabula" not in names


def test_cli_list_model_inventory_filters_foundation_family(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-model-inventory", "--family", "foundation"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    names = {entry["name"] for entry in payload["models"]}

    assert "tabpfn" in names
    assert "tabfm" in names
    assert "ctgan" not in names


def test_cli_list_model_inventory_filters_not_implemented_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-model-inventory", "--status", "registered"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    names = {entry["name"] for entry in payload["models"]}

    assert "tabpfn" in names
    assert "tabula" not in names
    assert "ctgan" not in names


def test_cli_show_model_inventory_prints_expected_entry(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "realtabformer"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "realtabformer"
    assert payload["family"] == "llm"
    assert payload["runnable_recommendation"] == "yes"


def test_cli_show_model_inventory_can_describe_not_yet_integrated_method(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabicl"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabicl"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_tabiclv2(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabiclv2"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabiclv2"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_tabpfn(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabpfn"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabpfn"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_tabula(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabula"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabula"
    assert payload["validation_level"] == "adapter-complete"
    assert payload["family"] == "llm"


def test_cli_show_model_inventory_can_describe_tabdpt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabdpt"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabdpt"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_tabsds(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabsds"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabsds"
    assert payload["validation_level"] == "adapter-complete"
    assert payload["family"] == "traditional"


def test_cli_show_model_inventory_can_describe_tabularargn(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabularargn"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabularargn"
    assert payload["validation_level"] == "native-parity-validated"
    assert payload["family"] == "autoregressive"


def test_cli_show_model_inventory_can_describe_realtabpfn(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "realtabpfn"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "realtabpfn"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_tabfm(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabfm"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabfm"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_mothernet(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "mothernet"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "mothernet"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_gamformer(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "gamformer"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "gamformer"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_causalfm(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "causalfm"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "causalfm"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_tabflex(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "tabflex"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "tabflex"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_show_model_inventory_can_describe_transtab(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "show-model-inventory", "--model", "transtab"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "transtab"
    assert payload["validation_level"] == "registered"
    assert payload["family"] == "foundation"


def test_cli_list_models_includes_new_sample_based_adapters(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-models"])

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "ctgan" in payload["models"]
    assert "tvae" in payload["models"]
    assert "smote" in payload["models"]
    assert "ctab-gan" in payload["models"]
    assert "codi" in payload["models"]
    assert "ctab-gan-plus" in payload["models"]
    assert "realtabformer" in payload["models"]
    assert "nrgboost" in payload["models"]
    assert "bn" in payload["models"]
    assert "nflow" in payload["models"]
    assert "stasy" in payload["models"]
    assert "tabsds" in payload["models"]
    assert "tabularargn" in payload["models"]
    assert "goggle" in payload["models"]
    assert "great" in payload["models"]
    assert "tabula" in payload["models"]
    assert "arf" in payload["models"]
    assert "tabebm" in payload["models"]


def test_cli_list_models_details_exposes_validation_without_release_claims(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-models", "--details"])

    cli.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["models"]["tabdiff"]["validation_level"] == "native-parity-validated"
    assert payload["models"]["tabdiff"]["benchmark_track"] == "experimental"
    assert payload["models"]["tabdiff"]["support_level"] == "unsupported"
    assert payload["models"]["tabddpm"]["evaluation_input"] == "upstream-artifacts"


def test_cli_lists_checksum_pinned_dataset_sources(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["std-cli", "list-dataset-sources"])

    cli.main()
    payload = json.loads(capsys.readouterr().out)

    records = {record["dataset_id"]: record for record in payload["dataset_sources"]}
    assert set(records) == {"adult", "sick"}
    assert len(records["adult"]["sha256"]) == 64


def test_cli_download_dataset_dispatches_safe_fetch(monkeypatch, capsys, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_fetch(dataset: str, *, cache_dir: str | None, refresh: bool, timeout_seconds: float):
        observed.update(
            dataset=dataset,
            cache_dir=cache_dir,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
        )
        return {"download": {"cached": False}, "extraction": {"cached": False}}

    monkeypatch.setattr(cli, "_fetch_dataset_source", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-cli",
            "download-dataset",
            "--dataset",
            "adult",
            "--cache-dir",
            str(tmp_path),
            "--refresh",
            "--timeout-seconds",
            "12",
        ],
    )

    cli.main()
    payload = json.loads(capsys.readouterr().out)

    assert observed == {
        "dataset": "adult",
        "cache_dir": str(tmp_path),
        "refresh": True,
        "timeout_seconds": 12.0,
    }
    assert payload["download"]["cached"] is False


def test_cli_materialize_model_source_dispatches_locked_acquisition(monkeypatch, capsys, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_materialize(model_id, *, repo_root, destination, refresh, timeout_seconds):
        observed.update(
            model_id=model_id,
            repo_root=repo_root,
            destination=destination,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
        )
        return {"status": "ready", "upstream_commit": "locked"}

    monkeypatch.setattr(cli, "materialize_upstream_source", fake_materialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-cli",
            "materialize-model-source",
            "--model",
            "ctab-gan-plus",
            "--repo-root",
            str(tmp_path),
            "--destination",
            str(tmp_path / "source"),
            "--refresh",
            "--timeout-seconds",
            "12",
        ],
    )

    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert observed == {
        "model_id": "ctab-gan-plus",
        "repo_root": str(tmp_path),
        "destination": str(tmp_path / "source"),
        "refresh": True,
        "timeout_seconds": 12.0,
    }


def test_cli_materialize_dataset_forwards_official_source_controls(monkeypatch, capsys, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_materialize(
        dataset: str,
        *,
        cache_root: str | None,
        refresh: bool,
        timeout_seconds: float,
    ) -> dict[str, object]:
        observed.update(
            dataset=dataset,
            cache_root=cache_root,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
        )
        return {"dataset": dataset, "materialized_by": "official-uci-builder"}

    monkeypatch.setattr(cli, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-cli",
            "materialize-dataset",
            "--dataset",
            "sick",
            "--cache-dir",
            str(tmp_path),
            "--refresh",
            "--timeout-seconds",
            "15",
        ],
    )

    cli.main()
    payload = json.loads(capsys.readouterr().out)

    assert observed == {
        "dataset": "sick",
        "cache_root": str(tmp_path),
        "refresh": True,
        "timeout_seconds": 15.0,
    }
    assert payload["materialized_by"] == "official-uci-builder"


def test_cli_preprocesses_missing_values_with_train_only_state(monkeypatch, capsys, tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_dir = tmp_path / "processed"
    pd.DataFrame({"age": [10, 30, None], "city": ["NY", "NY", "CA"], "target": [0, 1, 0]}).to_csv(
        train_path,
        index=False,
    )
    pd.DataFrame({"age": [None], "city": [None], "target": [1]}).to_csv(test_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-cli",
            "preprocess-missing-values",
            "--train-csv",
            str(train_path),
            "--test-csv",
            str(test_path),
            "--output-dir",
            str(output_dir),
            "--numerical-column",
            "age",
            "--categorical-column",
            "city",
            "--target-column",
            "target",
        ],
    )

    cli.main()
    payload = json.loads(capsys.readouterr().out)
    transformed = pd.read_csv(output_dir / "test.csv")

    assert payload["fitted_on_split"] == "train"
    assert transformed.loc[0, "age"] == 20.0
    assert transformed.loc[0, "city"] == "NY"


def test_cli_example_config_can_save_to_file(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "generated-config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-cli",
            "example-config",
            "--model",
            "tabsyn",
            "--dataset",
            "adult",
            "--output-dir",
            str(tmp_path / "artifacts" / "tabsyn"),
            "--save-config",
            str(config_path),
        ],
    )

    cli.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    saved_payload = json.loads(config_path.read_text())

    assert config_path.exists()
    assert payload == saved_payload
    assert saved_payload["model"] == "tabsyn"
    assert saved_payload["dataset"] == "adult"


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
