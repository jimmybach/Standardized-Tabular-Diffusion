from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.evaluation.bundle import (
    BundleError,
    _validate_final_atomic_results,
    validate_result_bundle,
)
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = [pytest.mark.evaluation, pytest.mark.integration]


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


def test_p3_finalizes_original_output_validity_bundle(
    tmp_path: Path,
    adult_profile,
    adult_frames,
    p3_protocol,
    p3_request,
) -> None:
    reference, synthetic = adult_frames
    synthetic.loc[0, "workclass"] = "invalid-category"
    synthetic["age"] = synthetic["age"].astype(float)
    synthetic.loc[0, "age"] = 17.5
    reference_path = tmp_path / "reference.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    reference.to_csv(reference_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p3_request, reference_path, synthetic_path)
    root = tmp_path / "bundle"

    report = evaluate_table_to_bundle(
        reference_path=reference_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p3_protocol.payload,
        request=request,
        output_dir=root,
    )

    assert report.finalization_status == "finalized"
    assert validate_result_bundle(root).finalization_status == "finalized"
    summary = read_json(root / "summary.json")
    metadata = read_json(root / "metadata.json")
    details = read_json(root / "artifacts" / "validity-details.json")
    atoms = pd.read_parquet(root / "metrics.parquet")
    assert summary["validity"]["column_validity_score"] < 1.0
    assert summary["validity"]["constraint_validity_score"] is None
    assert summary["validity"]["synthetic_repair_applied"] is False
    assert details["input_mutated"] is False
    assert details["fully_valid_row_rate_aggregation_role"] == "reported-only-width-sensitive"
    assert metadata["provenance"]["original_synthetic_output_preserved"] is True
    assert len(atoms) == 16
    assert set(atoms["dimension"]) == {"validity"}
    assert sha256_file(synthetic_path) == request.sample_artifact["sha256"]


def test_p3_bundle_validation_detects_validity_summary_tampering(
    tmp_path: Path,
    adult_profile,
    adult_frames,
    p3_protocol,
    p3_request,
) -> None:
    reference, synthetic = adult_frames
    reference_path = tmp_path / "reference.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    reference.to_csv(reference_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p3_request, reference_path, synthetic_path)
    root = tmp_path / "bundle"
    evaluate_table_to_bundle(
        reference_path=reference_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p3_protocol.payload,
        request=request,
        output_dir=root,
    )
    summary = read_json(root / "summary.json")
    summary["validity"]["validity_score"] = 0.123
    (root / "summary.json").write_text(__import__("json").dumps(summary), encoding="utf-8")

    with pytest.raises(BundleError):
        validate_result_bundle(root)


def test_p3_bundle_validation_rejects_non_equal_atomic_weights(
    tmp_path: Path,
    adult_profile,
    adult_frames,
    p3_protocol,
    p3_request,
) -> None:
    reference, synthetic = adult_frames
    reference_path = tmp_path / "reference.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    reference.to_csv(reference_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p3_request, reference_path, synthetic_path)
    root = tmp_path / "bundle"
    evaluate_table_to_bundle(
        reference_path=reference_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p3_protocol.payload,
        request=request,
        output_dir=root,
    )
    metrics_path = root / "metrics.parquet"
    atoms = pd.read_parquet(metrics_path)
    column_rows = atoms.index[atoms["metric_id"] == "std-tabular-column-validity"].tolist()
    first, second = column_rows[:2]
    delta = 0.01
    atoms.loc[first, "weight"] += delta
    atoms.loc[second, "weight"] -= delta
    atoms.loc[first, "aggregate_contribution"] = atoms.loc[first, "raw_value"] * atoms.loc[first, "weight"]
    atoms.loc[second, "aggregate_contribution"] = atoms.loc[second, "raw_value"] * atoms.loc[second, "weight"]
    atoms.to_parquet(metrics_path, index=False)

    with pytest.raises(BundleError, match="equal-column contract"):
        _validate_final_atomic_results(
            metrics_path,
            manifest=read_json(root / "manifest.json"),
            request=request,
            metadata=read_json(root / "metadata.json"),
            summary=read_json(root / "summary.json"),
        )


def test_evaluate_table_cli_selects_the_versioned_p3_protocol(
    tmp_path: Path,
    adult_frames,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference, synthetic = adult_frames
    reference_path = tmp_path / "reference.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    reference.to_csv(reference_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    bundle = tmp_path / "cli-bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-tabular-diffusion",
            "evaluate-table",
            "--protocol",
            "p3-validity",
            "--reference",
            str(reference_path),
            "--synthetic",
            str(synthetic_path),
            "--dataset-profile",
            "configs/datasets/adult-uci-2-v1.json",
            "--output",
            str(bundle),
            "--expected-rows",
            "20",
        ],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["finalization_status"] == "finalized"
    assert read_json(bundle / "metadata.json")["protocol"]["protocol_id"] == "p3-validity"
