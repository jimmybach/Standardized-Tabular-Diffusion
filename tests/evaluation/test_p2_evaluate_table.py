from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import standardized_tabular_diffusion.evaluation.evaluate_table as evaluate_table_module
from standardized_tabular_diffusion.evaluation import bundle as bundle_module
from standardized_tabular_diffusion.evaluation.backends.sdmetrics import SDMetricsExecutionError
from standardized_tabular_diffusion.evaluation.bundle import BundleError, validate_result_bundle
from standardized_tabular_diffusion.evaluation.evaluate_table import (
    TableEvaluationError,
    evaluate_table_to_bundle,
)
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file

pytestmark = [pytest.mark.evaluation, pytest.mark.source_parity]


def _file_request(request, reference: Path, synthetic: Path):
    return replace(
        request,
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": sha256_file(reference),
            "row_count": 20,
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": sha256_file(synthetic),
            "row_count": 20,
        },
    )


def test_evaluate_table_finalizes_a_self_validating_bundle(
    tmp_path: Path, adult_profile, p2_protocol, adult_frames, p2_request
) -> None:
    pytest.importorskip("sdmetrics")
    pytest.importorskip("pyarrow")
    reference, synthetic = adult_frames
    real_path, synthetic_path = tmp_path / "real.csv", tmp_path / "synthetic.csv"
    reference.to_csv(real_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p2_request, real_path, synthetic_path)
    root = tmp_path / "bundle"
    report = evaluate_table_to_bundle(
        reference_path=real_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p2_protocol.payload,
        request=request,
        output_dir=root,
    )
    assert report.finalization_status == "finalized"
    assert report.pending_files == 0
    assert validate_result_bundle(root).finalization_status == "finalized"
    manifest = read_json(root / "manifest.json")
    assert manifest["finalization_status"] == "finalized"
    assert all(item["status"] != "pending" for item in manifest["files"])
    checksums = (root / "checksums.sha256").read_text(encoding="utf-8")
    assert "  manifest.json\n" in checksums
    assert "  checksums.sha256\n" not in checksums
    atomic = pd.read_parquet(root / "metrics.parquet")
    assert len(atomic) == 120
    assert set(atomic["scope_type"]) == {"column", "pair"}
    summary = read_json(root / "summary.json")
    assert summary["dimensions"]["fidelity"]["combined_score"] is None
    assert summary["dataset_aggregation_eligible"] is False

    index = read_json(root / "artifacts/index.json")
    local = next(artifact for artifact in index["artifacts"] if artifact["path"] is not None)
    local["byte_size"] += 1
    atomic_write_json(root / "artifacts/index.json", index)
    index_hash = sha256_file(root / "artifacts/index.json")
    report_stage = read_json(root / "stages/report.json")
    next(output for output in report_stage["outputs"] if output["path"] == "artifacts/index.json")[
        "sha256"
    ] = index_hash
    atomic_write_json(root / "stages/report.json", report_stage)
    manifest = read_json(root / "manifest.json")
    next(item for item in manifest["files"] if item["path"] == "artifacts/index.json")["sha256"] = index_hash
    next(item for item in manifest["files"] if item["path"] == "stages/report.json")["sha256"] = sha256_file(
        root / "stages/report.json"
    )
    atomic_write_json(root / "manifest.json", manifest)
    with pytest.raises(BundleError, match="byte_size differs"):
        validate_result_bundle(root)


def test_structural_failure_stops_before_metrics_and_records_reason(
    tmp_path: Path, adult_profile, p2_protocol, adult_frames, p2_request
) -> None:
    reference, synthetic = adult_frames
    synthetic = synthetic.iloc[:-1]
    real_path, synthetic_path = tmp_path / "real.csv", tmp_path / "short.csv"
    reference.to_csv(real_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p2_request, real_path, synthetic_path)
    root = tmp_path / "failed-bundle"
    with pytest.raises(TableEvaluationError, match="row_count_mismatch"):
        evaluate_table_to_bundle(
            reference_path=real_path,
            synthetic_path=synthetic_path,
            dataset_profile=adult_profile.payload,
            protocol_profile=p2_protocol.payload,
            request=request,
            output_dir=root,
        )
    assert not (root / "metrics.parquet").exists()
    stage = read_json(root / "stages/validate.json")
    assert stage["status"] == "failed"
    assert stage["failure_reason_code"] == "row_count_mismatch"
    assert read_json(root / "manifest.json")["finalization_status"] == "incomplete"
    assert validate_result_bundle(root).finalization_status == "incomplete"


def test_interruption_cannot_publish_false_finalized_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adult_profile, p2_protocol, adult_frames, p2_request
) -> None:
    pytest.importorskip("sdmetrics")
    pytest.importorskip("pyarrow")
    reference, synthetic = adult_frames
    real_path, synthetic_path = tmp_path / "real.csv", tmp_path / "synthetic.csv"
    reference.to_csv(real_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p2_request, real_path, synthetic_path)
    root = tmp_path / "interrupted"
    real_write = bundle_module.atomic_write_bytes

    def interrupt_final_manifest(path, payload):
        if Path(path).name == "manifest.json" and b'"finalization_status": "finalized"' in payload:
            raise OSError("simulated final commit interruption")
        real_write(path, payload)

    monkeypatch.setattr(bundle_module, "atomic_write_bytes", interrupt_final_manifest)
    with pytest.raises(OSError, match="final commit"):
        evaluate_table_to_bundle(
            reference_path=real_path,
            synthetic_path=synthetic_path,
            dataset_profile=adult_profile.payload,
            protocol_profile=p2_protocol.payload,
            request=request,
            output_dir=root,
        )
    manifest = read_json(root / "manifest.json")
    assert manifest["finalization_status"] == "incomplete"
    assert manifest["finalized_at"] is None


def test_upstream_execution_failure_is_distinct_from_source_attestation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adult_profile,
    p2_protocol,
    adult_frames,
    p2_request,
) -> None:
    reference, synthetic = adult_frames
    real_path, synthetic_path = tmp_path / "real.csv", tmp_path / "synthetic.csv"
    reference.to_csv(real_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p2_request, real_path, synthetic_path)
    root = tmp_path / "upstream-failure"

    def fail_upstream(*args, **kwargs):
        raise SDMetricsExecutionError("simulated attested-source execution failure")

    monkeypatch.setattr(evaluate_table_module, "evaluate_shape_trend", fail_upstream)
    with pytest.raises(TableEvaluationError, match="upstream_metric_execution_failure"):
        evaluate_table_to_bundle(
            reference_path=real_path,
            synthetic_path=synthetic_path,
            dataset_profile=adult_profile.payload,
            protocol_profile=p2_protocol.payload,
            request=request,
            output_dir=root,
        )
    stage = read_json(root / "stages/evaluate.json")
    assert stage["status"] == "failed"
    assert stage["failure_reason_code"] == "upstream_metric_execution_failure"
    assert read_json(root / "manifest.json")["finalization_status"] == "incomplete"
