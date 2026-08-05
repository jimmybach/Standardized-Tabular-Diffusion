from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from standardized_tabular_diffusion import cli
from standardized_tabular_diffusion.evaluation import evaluate_table as evaluate_table_module
from standardized_tabular_diffusion.evaluation.bundle import BundleError, validate_result_bundle
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file
from standardized_tabular_diffusion.evaluation.utility import GlobalBackendResult
from standardized_tabular_diffusion.evaluation.utility import evaluate_utility as real_evaluate_utility

pytestmark = [pytest.mark.evaluation, pytest.mark.integration]


def _frames(adult_frames):
    reference, _ = adult_frames
    train = pd.concat([reference] * 6, ignore_index=True)
    test = pd.concat([reference] * 3, ignore_index=True)
    synthetic = train.copy(deep=True)
    for frame in (train, test, synthetic):
        frame["age"] = [17 + (index % 60) for index in range(len(frame))]
        frame["income"] = frame["age"].map(lambda value: ">50K" if value >= 45 else "<=50K")
    return train, test, synthetic


def _source_stub(train, test, target, task_type, seed, time_limit_seconds, arm):
    del train, test, target, seed, time_limit_seconds
    value = (0.8 if arm == "trtr" else 0.6) if task_type == "classification" else (
        2.0 if arm == "trtr" else 2.5
    )
    predictors = ("KNeighbors", "TabPFN", "XGBoost")
    return GlobalBackendResult(value, predictors, {name: value for name in predictors})


def _patch_global(monkeypatch: pytest.MonkeyPatch) -> None:
    def implementation(request, dataset_profile, tables, *, run_id):
        return real_evaluate_utility(
            request,
            dataset_profile,
            tables,
            run_id=run_id,
            global_scorer=_source_stub,
        )

    monkeypatch.setattr(evaluate_table_module, "evaluate_utility", implementation)


def _file_request(request, reference: Path, test: Path, synthetic: Path, rows: int):
    return replace(
        request,
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": sha256_file(reference),
            "row_count": rows,
        },
        real_test_artifact={
            "artifact_id": "real-test-table",
            "media_type": "text/csv",
            "sha256": sha256_file(test),
            "row_count": 60,
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": sha256_file(synthetic),
            "row_count": rows,
        },
    )


def test_p4_finalizes_raw_arms_ratios_and_held_out_test_provenance(
    tmp_path: Path,
    adult_profile,
    adult_frames,
    p4_protocol,
    p4_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_global(monkeypatch)
    train, test, synthetic = _frames(adult_frames)
    train_path, test_path, synthetic_path = (
        tmp_path / "train.csv",
        tmp_path / "test.csv",
        tmp_path / "synthetic.csv",
    )
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p4_request, train_path, test_path, synthetic_path, len(train))
    root = tmp_path / "bundle"

    report = evaluate_table_to_bundle(
        reference_path=train_path,
        real_test_path=test_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p4_protocol.payload,
        request=request,
        output_dir=root,
    )

    assert report.finalization_status == "finalized"
    assert validate_result_bundle(root).finalization_status == "finalized"
    summary = read_json(root / "summary.json")
    metadata = read_json(root / "metadata.json")
    details = read_json(root / "artifacts" / "utility-details.json")
    assert {run["test_fingerprint"] for run in details["local_runs"]} == {
        request.real_test_artifact["sha256"]
    }
    atoms = pd.read_parquet(root / "metrics.parquet")
    assert summary["terminal_status"] == "success"
    assert summary["local_utility"]["retention"] == pytest.approx(1.0)
    assert summary["global_utility"]["global_utility"] == pytest.approx((9 * 0.75 + 6 * 0.8) / 15)
    assert set(atoms["dimension"]) == {"local-utility", "global-utility"}
    assert metadata["provenance"]["real_test_artifact"] == request.real_test_artifact
    assert metadata["provenance"]["test_used_for_fit"] is False
    assert details["input_boundary"]["same_real_test_for_all_arms"] is True
    assert {artifact["artifact_id"] for artifact in read_json(root / "artifacts" / "index.json")["artifacts"]} >= {
        "reference-table",
        "real-test-table",
        "synthetic-table",
    }


def test_p4_bundle_validation_detects_global_summary_tampering(
    tmp_path: Path,
    adult_profile,
    adult_frames,
    p4_protocol,
    p4_request,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_global(monkeypatch)
    train, test, synthetic = _frames(adult_frames)
    train_path, test_path, synthetic_path = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "syn.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    request = _file_request(p4_request, train_path, test_path, synthetic_path, len(train))
    root = tmp_path / "bundle"
    evaluate_table_to_bundle(
        reference_path=train_path,
        real_test_path=test_path,
        synthetic_path=synthetic_path,
        dataset_profile=adult_profile.payload,
        protocol_profile=p4_protocol.payload,
        request=request,
        output_dir=root,
    )
    summary = read_json(root / "summary.json")
    summary["global_utility"]["global_utility"] = 0.123
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(BundleError):
        validate_result_bundle(root)


def test_evaluate_table_cli_runs_p4_with_an_explicit_real_test(
    tmp_path: Path,
    adult_frames,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_global(monkeypatch)
    train, test, synthetic = _frames(adult_frames)
    train_path, test_path, synthetic_path = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "syn.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    synthetic.to_csv(synthetic_path, index=False)
    bundle = tmp_path / "cli-bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "std-tabular-diffusion",
            "evaluate-table",
            "--protocol",
            "p4-utility",
            "--reference",
            str(train_path),
            "--real-test",
            str(test_path),
            "--synthetic",
            str(synthetic_path),
            "--dataset-profile",
            "configs/datasets/adult-uci-2-v1.json",
            "--output",
            str(bundle),
            "--expected-rows",
            str(len(train)),
            "--evaluator-seeds",
            "23",
        ],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["finalization_status"] == "finalized"
    assert read_json(bundle / "metadata.json")["protocol"]["protocol_id"] == "p4-utility"
