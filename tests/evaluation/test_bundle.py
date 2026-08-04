from __future__ import annotations

import json
from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation import serialization
from standardized_tabular_diffusion.evaluation.bundle import (
    BundleError,
    IncompleteRunBundleWriter,
    validate_result_bundle,
)
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

SHA256 = "0" * 64


def make_request(seed: int = 42) -> EvaluationRequest:
    return EvaluationRequest(
        subject_type="external-synthetic-table",
        sample_artifact={"artifact_id": "sample", "media_type": "text/csv", "sha256": SHA256},
        dataset_profile={"dataset_id": "fixture", "dataset_profile_version": "1.0.0", "sha256": SHA256},
        protocol={"protocol_id": "development-p1", "protocol_version": "0.1.0", "sha256": SHA256},
        metrics=({"metric_id": "fixture-shape", "metric_version": "1.0.0"},),
        comparison_track="native",
        generation_seed=seed,
        evaluator_seeds=(7,),
    )


def test_incomplete_bundle_is_valid_auditable_and_not_finalized(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = IncompleteRunBundleWriter(root)
    report = writer.create(make_request(), environment={"python": "3.11.15", "operating_system": "linux"})
    writer.append_event(
        severity="info",
        stage="test",
        component="fixture",
        event_code="redaction.checked",
        details={"api_token": "must-not-leak", "safe": "visible"},
    )

    assert report.finalization_status == "incomplete"
    assert report.pending_files > 0
    assert not (root / "metrics.parquet").exists()
    assert not (root / "checksums.sha256").exists()
    manifest = read_json(root / "manifest.json")
    assert manifest["finalized_at"] is None
    assert manifest["finalization_status"] == "incomplete"
    events = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["details"]["api_token"] == "<redacted>"
    assert "must-not-leak" not in (root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert validate_result_bundle(root).finalization_status == "incomplete"


def test_equivalent_requests_share_fingerprint_but_scientific_change_does_not() -> None:
    assert make_request().fingerprint == EvaluationRequest.from_dict(make_request().to_dict()).fingerprint
    assert make_request().fingerprint != make_request(seed=43).fingerprint


def test_cross_file_tampering_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    IncompleteRunBundleWriter(root).create(make_request(), environment={})
    config = read_json(root / "config.yaml")
    config["generation_seed"] = 99
    atomic_write_json(root / "config.yaml", config)

    with pytest.raises(BundleError, match="Checksum mismatch"):
        validate_result_bundle(root)


def test_flipping_manifest_status_cannot_false_finalize_bundle(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    IncompleteRunBundleWriter(root).create(make_request(), environment={})
    manifest = read_json(root / "manifest.json")
    manifest["finalization_status"] = "finalized"
    manifest["finalized_at"] = "2026-08-03T12:00:00Z"
    atomic_write_json(root / "manifest.json", manifest)

    with pytest.raises(BundleError, match="pending"):
        validate_result_bundle(root)


def test_interrupted_writer_keeps_bootstrap_manifest_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "interrupted"
    real_replace = serialization.os.replace
    replacement_count = 0

    def fail_third_replace(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        nonlocal replacement_count
        replacement_count += 1
        if replacement_count == 3:
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr(serialization.os, "replace", fail_third_replace)
    with pytest.raises(OSError, match="simulated"):
        IncompleteRunBundleWriter(root).create(make_request(), environment={})

    manifest = read_json(root / "manifest.json")
    assert manifest["finalization_status"] == "incomplete"
    assert manifest["finalized_at"] is None
    report = validate_result_bundle(root)
    assert report.finalization_status == "incomplete"


def test_writer_refuses_to_overwrite_nonempty_directory(tmp_path: Path) -> None:
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "user-data.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(BundleError, match="Refusing to overwrite"):
        IncompleteRunBundleWriter(root).create(make_request(), environment={})
    assert (root / "user-data.txt").read_text(encoding="utf-8") == "preserve me"
