from __future__ import annotations

import json
from pathlib import Path

import pytest

from standardized_tabular_diffusion.config import ExperimentConfig
from standardized_tabular_diffusion.orchestration.engine import validate_orchestration_run
from standardized_tabular_diffusion.orchestration.pipeline import build_benchmark_plan, run_benchmark_pipeline

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_pipeline_plan_has_seven_ordered_stages_and_terminal_partial_tolerance(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = ExperimentConfig(model="tabsyn", dataset="adult", output_dir=str(tmp_path / "run"))
    plan, fingerprint = build_benchmark_plan(
        config,
        config_path=config_path,
        repo_root=Path(__file__).resolve().parents[2],
        timeout_seconds=None,
        memory_limit_bytes=None,
        max_retries=1,
    )
    assert [stage.name for stage in plan] == [
        "prepare",
        "train",
        "sample",
        "validate",
        "evaluate",
        "aggregate",
        "report",
    ]
    assert len(fingerprint) == 64
    assert plan[-2].allow_failed_dependencies is True
    assert plan[-1].allow_failed_dependencies is True
    assert plan[1].max_retries == 1


def test_pipeline_configuration_rejects_embedded_secrets(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = ExperimentConfig(model="tabsyn", dataset="adult", output_dir=str(tmp_path / "run"))
    config.train.extra["api_token"] = "must-not-be-embedded"
    with pytest.raises(ValueError, match="Secrets are prohibited"):
        build_benchmark_plan(
            config,
            config_path=config_path,
            repo_root=Path(__file__).resolve().parents[2],
            timeout_seconds=None,
            memory_limit_bytes=None,
            max_retries=0,
        )


def test_pipeline_identity_is_portable_but_tracks_external_sample_content(tmp_path: Path) -> None:
    first_config_path = tmp_path / "first.json"
    second_config_path = tmp_path / "second.json"
    first_config_path.write_text('{"notes": "first"}\n', encoding="utf-8")
    second_config_path.write_text('{"notes": "second"}\n', encoding="utf-8")
    sample = tmp_path / "sample.csv"
    sample.write_text("feature\n1\n", encoding="utf-8")
    first = ExperimentConfig(model="tabsyn", dataset="adult", output_dir=str(tmp_path / "run-a"))
    second = ExperimentConfig(model="tabsyn", dataset="adult", output_dir=str(tmp_path / "run-b"))
    first.evaluation.extra["sample_path"] = str(sample)
    second.evaluation.extra["sample_path"] = str(sample)

    first_plan, first_identity = build_benchmark_plan(
        first,
        config_path=first_config_path,
        repo_root=Path(__file__).resolve().parents[2],
        timeout_seconds=None,
        memory_limit_bytes=None,
        max_retries=0,
    )
    second_plan, second_identity = build_benchmark_plan(
        second,
        config_path=second_config_path,
        repo_root=Path(__file__).resolve().parents[2],
        timeout_seconds=None,
        memory_limit_bytes=None,
        max_retries=0,
    )
    assert first_identity == second_identity
    assert first_plan[0].input_fingerprints == second_plan[0].input_fingerprints

    sample.write_text("feature\n2\n", encoding="utf-8")
    changed_plan, changed_identity = build_benchmark_plan(
        second,
        config_path=second_config_path,
        repo_root=Path(__file__).resolve().parents[2],
        timeout_seconds=None,
        memory_limit_bytes=None,
        max_retries=0,
    )
    assert changed_identity != second_identity
    assert changed_plan[0].input_fingerprints["external-sample"] != second_plan[0].input_fingerprints["external-sample"]


@pytest.mark.integration
def test_disabled_scientific_actions_still_produce_a_complete_operational_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.json"
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
        ),
        encoding="utf-8",
    )
    manifest = run_benchmark_pipeline(
        config_path,
        hardware_profile_id="disabled-pipeline-fixture",
        repo_root=Path(__file__).resolve().parents[2],
    )
    assert manifest["status"] == "success"
    assert len(manifest["current_stage_ids"]) == 7
    assert (output_dir / "pipeline_result.json").is_file()
    assert (output_dir / "benchmark_report.json").is_file()
    assert validate_orchestration_run(output_dir)["valid"] is True
