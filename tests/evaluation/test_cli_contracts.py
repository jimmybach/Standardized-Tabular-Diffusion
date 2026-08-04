from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.evaluation.bundle import IncompleteRunBundleWriter
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.profiles import import_legacy_dataset_spec, write_dataset_profile
from standardized_tabular_diffusion.interfaces import DatasetSpec

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

SHA256 = "0" * 64


def test_cli_lists_validated_metric_records(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["std-tabular-diffusion", "validate-metric-registry"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"valid": True, "record_count": 8}


def test_cli_lists_protocol_profiles(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["std-tabular-diffusion", "list-protocols"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert {item["protocol_id"] for item in payload["protocols"]} == {"development-p1", "legacy-tabstruct-aligned"}


def test_cli_validates_dataset_profile_and_result_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = tmp_path / "info.json"
    metadata.write_text("{}\n", encoding="utf-8")
    spec = DatasetSpec(
        name="fixture",
        task_type="classification",
        column_names=["x", "target"],
        numerical_columns=["x"],
        categorical_columns=["target"],
        target_columns=["target"],
        metadata_path=metadata,
    )
    profile_path = tmp_path / "profile.json"
    write_dataset_profile(import_legacy_dataset_spec(spec), profile_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["std-tabular-diffusion", "validate-dataset-profile", "--profile", str(profile_path)],
    )
    cli.main()
    profile_output = json.loads(capsys.readouterr().out)
    assert profile_output["valid"] is True
    assert profile_output["official_eligible"] is False

    request = EvaluationRequest(
        subject_type="external-synthetic-table",
        sample_artifact={"artifact_id": "sample", "media_type": "text/csv", "sha256": SHA256},
        dataset_profile={"dataset_id": "fixture", "dataset_profile_version": "0.1.0-legacy", "sha256": SHA256},
        protocol={"protocol_id": "development-p1", "protocol_version": "0.1.0", "sha256": SHA256},
        metrics=({"metric_id": "fixture-shape", "metric_version": "1.0.0"},),
        comparison_track="native",
        generation_seed=42,
        evaluator_seeds=(7,),
    )
    bundle = tmp_path / "bundle"
    IncompleteRunBundleWriter(bundle).create(request, environment={})
    monkeypatch.setattr(sys, "argv", ["std-tabular-diffusion", "validate-result", "--bundle", str(bundle)])
    cli.main()
    result_output = json.loads(capsys.readouterr().out)
    assert result_output["valid"] is True
    assert result_output["finalization_status"] == "incomplete"
