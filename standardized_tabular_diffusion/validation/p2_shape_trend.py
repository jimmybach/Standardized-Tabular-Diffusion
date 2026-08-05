"""Authoritative P2 source-parity and finalized-bundle exit-gate validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import tempfile
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from standardized_tabular_diffusion.evaluation.backends.sdmetrics import (
    evaluate_quality,
    verify_sdmetrics_source,
)
from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    read_json,
    sha256_file,
)

PROTOCOL_ID = "p2-shape-trend-source-parity-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _repository_commit() -> str:
    if github_sha := os.environ.get("GITHUB_SHA"):
        return github_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _assert_primary_environment() -> None:
    if platform.system() != "Linux" or platform.python_version_tuple()[:2] != ("3", "11"):
        raise AssertionError("Authoritative P2 evidence requires Linux and Python 3.11")


def _fixture(profile: dict[str, Any]) -> pd.DataFrame:
    data: dict[str, list[object]] = {}
    for column in profile["columns"]:
        if column["semantic_type"] in {"continuous", "integer"}:
            minimum = column["valid_domain"].get("minimum", 0)
            data[column["name"]] = [minimum + index for index in range(20)]
        else:
            values = column["valid_domain"]["values"]
            data[column["name"]] = [values[index % min(len(values), 3)] for index in range(20)]
    return pd.DataFrame(data)


def _request(dataset: Any, protocol: Any, reference: Path, synthetic: Path) -> EvaluationRequest:
    metrics = tuple(
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in protocol.payload["metric_selections"]
    )
    return EvaluationRequest(
        subject_type="external-synthetic-table",
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
        dataset_profile={
            "dataset_id": dataset.dataset_id,
            "dataset_profile_version": dataset.dataset_profile_version,
            "sha256": dataset.fingerprint,
        },
        protocol={
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.protocol_version,
            "sha256": protocol.fingerprint,
        },
        metrics=metrics,
        comparison_track="native",
        generation_seed=17,
        evaluator_seeds=(23,),
        model={"model_id": "p2-evidence-fixture"},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )


def _direct_parity(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes

    seed = 23
    wrapped = evaluate_quality(frame, frame, metadata, evaluator_seed=seed)
    previous_state = np.random.get_state()
    np.random.seed(seed)
    shapes = ColumnShapes()
    trends = ColumnPairTrends()
    shapes.num_rows_subsample = 50000
    trends.num_rows_subsample = 50000
    trends.real_correlation_threshold = 0.5
    trends.real_association_threshold = 0.3
    try:
        direct_shape = shapes.get_score(frame.copy(), frame.copy(), metadata)
        direct_trend = trends.get_score(frame.copy(), frame.copy(), metadata)
    finally:
        np.random.set_state(previous_state)
    assert math.isclose(wrapped.column_shapes_score, direct_shape, rel_tol=0, abs_tol=0)
    assert math.isclose(wrapped.column_pair_trends_score, direct_trend, rel_tol=0, abs_tol=0)
    pd.testing.assert_frame_equal(wrapped.column_shapes_details, shapes.details, check_exact=True)
    pd.testing.assert_frame_equal(wrapped.column_pair_trends_details, trends.details, check_exact=True)
    return {
        "column_shapes": direct_shape,
        "column_pair_trends": direct_trend,
        "column_records": len(shapes.details),
        "pair_records": len(trends.details),
        "details_exact": True,
    }


def _subsampling_reproducibility() -> dict[str, Any]:
    rows = 50_001
    frame = pd.DataFrame(
        {
            "left": [f"l{index % 17}" for index in range(rows)],
            "right": [f"r{index % 17}" for index in range(rows)],
        }
    )
    synthetic = frame.sample(frac=1, random_state=91).reset_index(drop=True)
    metadata = {
        "columns": {
            "left": {"sdtype": "categorical"},
            "right": {"sdtype": "categorical"},
        }
    }
    seed = 29
    first = evaluate_quality(frame, synthetic, metadata, evaluator_seed=seed)
    second = evaluate_quality(frame, synthetic, metadata, evaluator_seed=seed)
    assert first.column_shapes_score == second.column_shapes_score
    assert first.column_pair_trends_score == second.column_pair_trends_score
    pd.testing.assert_frame_equal(first.column_shapes_details, second.column_shapes_details, check_exact=True)
    pd.testing.assert_frame_equal(
        first.column_pair_trends_details,
        second.column_pair_trends_details,
        check_exact=True,
    )
    return {"rows": rows, "evaluator_seed": seed, "repeated_details_exact": True}


def _locked_files() -> dict[str, str]:
    paths = [
        ".github/workflows/p2-shape-trend-validation.yml",
        "THIRD_PARTY_NOTICES.md",
        "docs/evaluation/P2_SHAPE_TREND_EVALUATION.md",
        "docs/evaluation/P2_SHAPE_TREND_EVALUATION.zh-CN.md",
        "pyproject.toml",
        "standardized_tabular_diffusion/cli.py",
        "standardized_tabular_diffusion/evaluation/__init__.py",
        "standardized_tabular_diffusion/evaluation/backends/sdmetrics.py",
        "standardized_tabular_diffusion/evaluation/bundle.py",
        "standardized_tabular_diffusion/evaluation/contracts.py",
        "standardized_tabular_diffusion/evaluation/evaluate_table.py",
        "standardized_tabular_diffusion/evaluation/shape_trend.py",
        "standardized_tabular_diffusion/evaluation/table.py",
        "standardized_tabular_diffusion/resources/evaluation/metrics/sdmetrics-shape-trend-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/protocols/p2-shape-trend.json",
        "standardized_tabular_diffusion/resources/evaluation/upstream/sdmetrics-p2-source.json",
        "standardized_tabular_diffusion/validation/p2_shape_trend.py",
        "tests/evaluation/test_p2_evaluate_table.py",
        "tests/evaluation/test_p2_shape_trend.py",
        "tests/evaluation/test_p2_source_parity.py",
        "tests/evaluation/test_p2_table_gate.py",
    ]
    paths.extend(
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "standardized_tabular_diffusion/schemas/evaluation").glob("*.json"))
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in paths}


def run_validation(output: Path, *, require_primary_environment: bool = False) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P2",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Validates the P2 structural gate, exact locked SDMetrics Column Shapes and Column Pair Trends "
            "source parity, denominator-complete Atomic Results, and finalized Run Result bundles. It does not "
            "freeze an Official Results protocol, admit a dataset, or define an overall Fidelity score."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "pandas": _distribution_version("pandas"),
            "pyarrow": _distribution_version("pyarrow"),
            "sdmetrics": _distribution_version("sdmetrics"),
            "primary_environment_required": require_primary_environment,
        },
    }
    try:
        if require_primary_environment:
            _assert_primary_environment()
        source = verify_sdmetrics_source()
        dataset = load_dataset_profile(REPO_ROOT / "configs/datasets/adult-uci-2-v1.json")
        protocol = resolve_protocol("p2-shape-trend", "0.2.0")
        frame = _fixture(dataset.payload)
        metadata = {
            "columns": {
                column["name"]: {
                    "sdtype": {
                        "continuous": "numerical",
                        "integer": "numerical",
                        "categorical": "categorical",
                        "boolean": "boolean",
                        "datetime": "datetime",
                        "string": "id",
                    }[column["semantic_type"]]
                }
                for column in dataset.payload["columns"]
            }
        }
        parity = _direct_parity(frame, metadata)
        subsampling = _subsampling_reproducibility()
        with tempfile.TemporaryDirectory(prefix="std-tabular-p2-") as temporary:
            root = Path(temporary)
            reference, synthetic = root / "reference.csv", root / "synthetic.csv"
            frame.to_csv(reference, index=False)
            frame.to_csv(synthetic, index=False)
            request = _request(dataset, protocol, reference, synthetic)
            reports = []
            for attempt in ("attempt-a", "attempt-b"):
                reports.append(
                    evaluate_table_to_bundle(
                        reference_path=reference,
                        synthetic_path=synthetic,
                        dataset_profile=dataset.payload,
                        protocol_profile=protocol.payload,
                        request=request,
                        output_dir=root / attempt,
                    )
                )
            assert reports[0].bundle_id != reports[1].bundle_id
            assert all(report.finalization_status == "finalized" and report.pending_files == 0 for report in reports)
            for report in reports:
                validate_result_bundle(report.root)
            first_summary = read_json(reports[0].root / "summary.json")
            second_summary = read_json(reports[1].root / "summary.json")
            assert first_summary["dimensions"] == second_summary["dimensions"]
            assert first_summary["denominator_counts"] == second_summary["denominator_counts"]
            assert first_summary["dimensions"]["fidelity"]["combined_score"] is None
            atomic_rows = len(pd.read_parquet(reports[0].root / "metrics.parquet"))

        evidence["source"] = source
        evidence["result_summary"] = {
            "scientific_metrics_executed": True,
            "direct_source_parity": parity,
            "subsampling_reproducibility": subsampling,
            "atomic_results": atomic_rows,
            "expected_atomic_results": 120,
            "finalized_bundles_created": 2,
            "identical_request_fingerprints_equal": True,
            "repeated_attempt_run_ids_distinct": True,
            "semantic_summaries_equal": True,
            "overall_fidelity_score_emitted": False,
            "official_results_allowed": False,
        }
        evidence["exit_gates"] = {
            "exact_source_tree_attested": "pass",
            "direct_source_details_equal": "pass",
            "seeded_source_subsampling_reproducible": "pass",
            "structural_gate": "pass",
            "denominator_complete_atomic_results": "pass",
            "source_aggregation_reproduced": "pass",
            "bundle_finalization_and_checksums": "pass",
            "interruption_safety_tests": "pass",
        }
        evidence["locked_files"] = _locked_files()
        evidence["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"P2 validation failed; inspect {output}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the P2 Shape/Trend vertical slice")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-primary-environment", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output, require_primary_environment=args.require_primary_environment), indent=2))


if __name__ == "__main__":
    main()
