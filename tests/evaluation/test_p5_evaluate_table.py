from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.high_order_privacy import P5_METRICS, p5_evaluator_profile_reference
from standardized_tabular_diffusion.evaluation.profiles import resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file
from tests.evaluation.test_p5_high_order_privacy import p5_frames, p5_request

pytestmark = [pytest.mark.evaluation, pytest.mark.integration]


def test_cli_accepts_p5_and_requires_the_explicit_three_table_interface() -> None:
    args = cli.build_parser().parse_args(
        [
            "evaluate-table",
            "--protocol",
            "p5-high-order-privacy",
            "--reference",
            "train.csv",
            "--real-test",
            "test.csv",
            "--synthetic",
            "synthetic.csv",
            "--dataset-profile",
            "profile.json",
            "--output",
            "bundle",
        ]
    )
    assert args.protocol == "p5-high-order-privacy"
    assert args.real_test == "test.csv"


def test_p5_evaluate_table_finalizes_and_cross_validates_every_diagnostic(
    tmp_path: Path,
    adult_profile,
) -> None:
    train, test, synthetic = p5_frames(adult_profile)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = replace(
        p5_request(adult_profile),
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": sha256_file(train_path),
            "row_count": len(train),
        },
        real_test_artifact={
            "artifact_id": "real-test-table",
            "media_type": "text/csv",
            "sha256": sha256_file(test_path),
            "row_count": len(test),
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": sha256_file(synthetic_path),
            "row_count": len(synthetic),
        },
    )
    protocol = resolve_protocol("p5-high-order-privacy", "1.0.0")
    bundle = tmp_path / "bundle"

    report = evaluate_table_to_bundle(
        reference_path=train_path,
        real_test_path=test_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=protocol.payload,
        request=request,
        output_dir=bundle,
    )

    assert report.finalization_status == "finalized"
    assert validate_result_bundle(bundle).finalization_status == "finalized"
    summary = read_json(bundle / "summary.json")
    metadata = read_json(bundle / "metadata.json")
    details = read_json(bundle / "artifacts" / "p5-details.json")
    atoms = pd.read_parquet(bundle / "metrics.parquet")
    assert summary["terminal_status"] == "success"
    assert summary["dimensions"]["high-order-fidelity"]["overall_fidelity_score"] is None
    assert summary["privacy_risk"]["formal_privacy_guarantee"] is False
    assert metadata["review"]["official_results_allowed"] is False
    assert metadata["provenance"]["formal_privacy_guarantee"] is False
    assert details["privacy"]["domias_threat_model"]["model_access"] == "none"
    assert set(atoms["metric_id"]) == {item["metric_id"] for item in P5_METRICS}
    assert atoms["weight"].eq(0).all()
    assert request.evaluator_profile == p5_evaluator_profile_reference()
