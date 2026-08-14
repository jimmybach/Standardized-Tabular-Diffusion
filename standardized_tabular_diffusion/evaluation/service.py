"""One public construction path for versioned table-evaluation Result Bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from standardized_tabular_diffusion.evaluation.bundle import BundleValidationReport
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.evaluate_table import evaluate_table_to_bundle
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import sha256_file

PROTOCOL_VERSIONS = {
    "p2-shape-trend": "0.2.0",
    "p3-validity": "0.3.0",
    "p4-utility": "1.0.0",
    "p5-high-order-privacy": "1.0.0",
}
THREE_TABLE_PROTOCOLS = {"p4-utility", "p5-high-order-privacy"}


@dataclass(frozen=True)
class EvaluationOutcome:
    request: EvaluationRequest
    report: BundleValidationReport


def _media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".parquet", ".pq"}:
        return "application/vnd.apache.parquet"
    raise ValueError(f"Expected a .csv, .parquet, or .pq table: {path}")


def evaluate_table_files(
    *,
    reference_path: str | Path,
    synthetic_path: str | Path,
    dataset_profile_path: str | Path,
    output_dir: str | Path,
    protocol_id: str = "p3-validity",
    real_test_path: str | Path | None = None,
    comparison_track: str = "native",
    generation_seed: int = 0,
    evaluator_seeds: tuple[int, ...] | None = None,
    expected_rows: int | None = None,
    model_id: str | None = None,
) -> EvaluationOutcome:
    """Checksum-bind table inputs and execute one registered central protocol."""

    try:
        protocol_version = PROTOCOL_VERSIONS[protocol_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported central evaluation protocol: {protocol_id!r}") from exc
    reference = Path(reference_path)
    synthetic = Path(synthetic_path)
    real_test = None if real_test_path is None else Path(real_test_path)
    if protocol_id in THREE_TABLE_PROTOCOLS and real_test is None:
        raise ValueError(f"A held-out real test table is required for {protocol_id}")
    if protocol_id not in THREE_TABLE_PROTOCOLS and real_test is not None:
        raise ValueError("A held-out real test table is only valid for P4 or P5")

    dataset = load_dataset_profile(dataset_profile_path)
    protocol = resolve_protocol(protocol_id, protocol_version)
    seeds = evaluator_seeds
    if seeds is None:
        seeds = (0, 1, 2, 3, 4) if protocol_id in THREE_TABLE_PROTOCOLS else (0,)
    if protocol_id == "p5-high-order-privacy" and seeds != (0, 1, 2, 3, 4):
        raise ValueError("p5-high-order-privacy requires evaluator seeds 0,1,2,3,4")

    evaluator_profile: dict[str, str] | None = None
    if protocol_id == "p4-utility":
        from standardized_tabular_diffusion.evaluation.utility import p4_evaluator_profile_reference

        evaluator_profile = p4_evaluator_profile_reference()
    elif protocol_id == "p5-high-order-privacy":
        from standardized_tabular_diffusion.evaluation.high_order_privacy import (
            p5_evaluator_profile_reference,
        )

        evaluator_profile = p5_evaluator_profile_reference()

    request = EvaluationRequest(
        subject_type="adapter-run" if model_id is not None else "external-synthetic-table",
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": _media_type(reference),
            "sha256": sha256_file(reference),
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": _media_type(synthetic),
            "sha256": sha256_file(synthetic),
            **({"row_count": expected_rows} if expected_rows is not None else {}),
        },
        real_test_artifact=(
            {
                "artifact_id": "real-test-table",
                "media_type": _media_type(real_test),
                "sha256": sha256_file(real_test),
            }
            if real_test is not None
            else None
        ),
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
        metrics=tuple(
            {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
            for item in protocol.payload["metric_selections"]
        ),
        comparison_track=comparison_track,
        generation_seed=generation_seed,
        evaluator_seeds=seeds,
        evaluator_profile=evaluator_profile,
        model={"model_id": model_id} if model_id is not None else None,
        resource_limits={},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )
    report = evaluate_table_to_bundle(
        reference_path=reference,
        synthetic_path=synthetic,
        real_test_path=real_test,
        dataset_profile=dataset.payload,
        protocol_profile=protocol.payload,
        request=request,
        output_dir=output_dir,
    )
    return EvaluationOutcome(request=request, report=report)


def evaluate_adapter_output(
    *,
    model_id: str,
    generation_seed: int,
    synthetic_path: str | Path,
    output_dir: str | Path,
    protocol_id: str,
    dataset_profile_path: str | Path,
    reference_path: str | Path,
    real_test_path: str | Path | None,
    comparison_track: str,
    evaluator_seeds: tuple[int, ...] | None,
    expected_rows: int | None,
) -> EvaluationOutcome:
    """Evaluate an adapter-produced decoded table without invoking adapter-local metrics."""

    return evaluate_table_files(
        reference_path=reference_path,
        synthetic_path=synthetic_path,
        real_test_path=real_test_path,
        dataset_profile_path=dataset_profile_path,
        protocol_id=protocol_id,
        output_dir=output_dir,
        comparison_track=comparison_track,
        generation_seed=generation_seed,
        evaluator_seeds=evaluator_seeds,
        expected_rows=expected_rows,
        model_id=model_id,
    )
