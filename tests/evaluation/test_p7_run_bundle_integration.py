from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.leaderboard import load_run_observation, load_snapshot_request
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file

pytestmark = [pytest.mark.integration, pytest.mark.evaluation]


def _write_p3_bundle(tmp_path: Path, adult_profile, p3_protocol, adult_frames) -> tuple[Path, EvaluationRequest]:
    pytest.importorskip("pyarrow")
    reference, synthetic = adult_frames
    reference_path = tmp_path / "reference.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    reference.to_csv(reference_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    metrics = tuple(
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in p3_protocol.payload["metric_selections"]
    )
    request = EvaluationRequest(
        subject_type="external-synthetic-table",
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": sha256_file(reference_path),
            "row_count": len(reference),
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": sha256_file(synthetic_path),
            "row_count": len(synthetic),
        },
        dataset_profile={
            "dataset_id": adult_profile.dataset_id,
            "dataset_profile_version": adult_profile.dataset_profile_version,
            "sha256": adult_profile.fingerprint,
        },
        protocol={
            "protocol_id": p3_protocol.protocol_id,
            "protocol_version": p3_protocol.protocol_version,
            "sha256": p3_protocol.fingerprint,
        },
        metrics=metrics,
        comparison_track="native",
        generation_seed=17,
        evaluator_seeds=(0,),
        model={"model_id": "fixture-model"},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )
    bundle = tmp_path / "run-bundle"
    evaluate_table_to_bundle(
        reference_path=reference_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p3_protocol.payload,
        request=request,
        output_dir=bundle,
    )
    return bundle, request


def _snapshot_request(request: EvaluationRequest) -> dict:
    return {
        "snapshot_request_schema_version": "1.0.0",
        "repository_release": "p7-integration",
        "published_at": "2026-08-13T00:00:00Z",
        "publication_class": "partial-diagnostic",
        "comparison_track": request.comparison_track,
        "protocol": request.protocol,
        "dataset_suite": {
            "suite_id": "adult-diagnostic",
            "suite_version": "1.0.0",
            "datasets": [request.dataset_profile],
        },
        "metric": {
            "metric_id": "std-tabular-column-validity",
            "metric_version": "1.0.0",
            "dimension": "validity",
            "direction": "maximize",
        },
        "expected_generation_seeds": [17],
        "aggregation": {
            "aggregation_id": "atomic-contribution-hierarchy",
            "aggregation_version": "1.0.0",
            "run_rule": "sum-declared-atomic-contributions",
            "seed_rule": "equal-seed-mean",
            "dataset_rule": "equal-dataset-macro-mean",
        },
        "bootstrap": {
            "method": "hierarchical-percentile",
            "replicates": 100,
            "confidence_level": 0.95,
            "random_seed": 11,
        },
        "ties": {"method": "disabled", "equivalence_margin": 0.0},
        "compatibility": {"hardware_profile": "exact", "evaluator_profile": "exact"},
        "community_submission": None,
    }


def test_p7_loads_a_real_finalized_bundle_and_cli_publishes_snapshot(
    tmp_path: Path,
    adult_profile,
    p3_protocol,
    adult_frames,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, evaluation_request = _write_p3_bundle(tmp_path, adult_profile, p3_protocol, adult_frames)
    request_path = tmp_path / "snapshot-request.json"
    atomic_write_json(request_path, _snapshot_request(evaluation_request))
    snapshot_request = load_snapshot_request(request_path)
    observation = load_run_observation(bundle, snapshot_request)

    assert observation.model_id == "fixture-model"
    assert observation.generation_seed == 17
    assert observation.score == pytest.approx(1.0)

    output = tmp_path / "snapshot"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-tabular-diffusion",
            "build-leaderboard",
            "--request",
            str(request_path),
            "--bundle",
            str(bundle),
            "--output",
            str(output),
        ],
    )
    cli.main()
    built = json.loads(capsys.readouterr().out)
    assert built["valid"] is True
    assert built["publication_class"] == "partial-diagnostic"
    assert built["entry_count"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["std-tabular-diffusion", "validate-leaderboard", "--snapshot", str(output)],
    )
    cli.main()
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["snapshot_id"] == built["snapshot_id"]
