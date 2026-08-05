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
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file

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
    environment = read_json(root / "environment.json")
    assert environment == {"operating_system": "linux", "python": "3.11.15"}
    assert validate_result_bundle(root).finalization_status == "incomplete"


def test_equivalent_requests_share_fingerprint_but_scientific_change_does_not() -> None:
    assert make_request().fingerprint == EvaluationRequest.from_dict(make_request().to_dict()).fingerprint
    assert make_request().fingerprint != make_request(seed=43).fingerprint


def test_repeated_scientific_request_gets_distinct_attempt_ids(tmp_path: Path) -> None:
    first = IncompleteRunBundleWriter(tmp_path / "first").create(make_request(), environment={})
    second = IncompleteRunBundleWriter(tmp_path / "second").create(make_request(), environment={})
    first_manifest = read_json(first.root / "manifest.json")
    second_manifest = read_json(second.root / "manifest.json")
    assert first.bundle_id != second.bundle_id
    assert first_manifest["identity"]["request_fingerprint"] == second_manifest["identity"]["request_fingerprint"]


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


def test_interrupted_event_update_leaves_a_valid_pending_inventory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    writer = IncompleteRunBundleWriter(root)
    writer.create(make_request(), environment={})
    real_replace = serialization.os.replace
    replacement_count = 0

    def fail_final_manifest_replace(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        nonlocal replacement_count
        replacement_count += 1
        if replacement_count == 3:
            raise OSError("simulated event transaction interruption")
        real_replace(source, destination)

    monkeypatch.setattr(serialization.os, "replace", fail_final_manifest_replace)
    with pytest.raises(OSError, match="event transaction"):
        writer.append_event(
            severity="info",
            stage="test",
            component="fixture",
            event_code="interruption.checked",
            details={"safe": True},
        )

    manifest = read_json(root / "manifest.json")
    event_item = next(item for item in manifest["files"] if item["path"] == "logs/events.jsonl")
    assert event_item == {
        "media_type": "application/x-ndjson",
        "path": "logs/events.jsonl",
        "reason_code": "event_log_updating",
        "required": True,
        "sha256": None,
        "status": "pending",
    }
    report = validate_result_bundle(root)
    assert report.finalization_status == "incomplete"
    assert report.pending_files > 0


def test_bundle_rejects_unmanifested_files_and_cross_file_identity_drift(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    IncompleteRunBundleWriter(root).create(make_request(), environment={})
    (root / "untracked.txt").write_text("not declared\n", encoding="utf-8")
    with pytest.raises(BundleError, match="absent from the manifest"):
        validate_result_bundle(root)
    (root / "untracked.txt").unlink()

    metadata = read_json(root / "metadata.json")
    metadata["protocol"]["protocol_version"] = "0.2.0"
    atomic_write_json(root / "metadata.json", metadata)
    manifest = read_json(root / "manifest.json")
    metadata_item = next(item for item in manifest["files"] if item["path"] == "metadata.json")
    metadata_item["sha256"] = sha256_file(root / "metadata.json")
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(BundleError, match="metadata protocol"):
        validate_result_bundle(root)


def test_bundle_rejects_manifest_identity_drift_and_duplicate_event_keys(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    IncompleteRunBundleWriter(root).create(make_request(), environment={"api_token": "must-not-leak"})
    assert read_json(root / "environment.json")["api_token"] == "<redacted>"

    manifest = read_json(root / "manifest.json")
    manifest["bundle_id"] = "run-different"
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(BundleError, match="bundle_id"):
        validate_result_bundle(root)

    manifest["bundle_id"] = manifest["identity"]["run_id"]
    event_path = root / "logs" / "events.jsonl"
    event_path.write_text(
        '{"event_schema_version":"1.0.0","timestamp":"2026-08-05T00:00:00Z",'
        '"severity":"info","stage":"test","component":"fixture","event_code":"duplicate",'
        '"event_code":"shadowed","details":{}}\n',
        encoding="utf-8",
    )
    event_item = next(item for item in manifest["files"] if item["path"] == "logs/events.jsonl")
    event_item["sha256"] = sha256_file(event_path)
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(BundleError, match="invalid JSON"):
        validate_result_bundle(root)


def test_writer_refuses_to_overwrite_nonempty_directory(tmp_path: Path) -> None:
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "user-data.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(BundleError, match="Refusing to overwrite"):
        IncompleteRunBundleWriter(root).create(make_request(), environment={})
    assert (root / "user-data.txt").read_text(encoding="utf-8") == "preserve me"
