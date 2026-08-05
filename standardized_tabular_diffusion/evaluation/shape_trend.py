"""P2 Column Shapes and Column Pair Trends Atomic Result mapping."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from standardized_tabular_diffusion.evaluation.backends.sdmetrics import evaluate_quality
from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    EvaluationRequest,
    MetricState,
    RawDirection,
    utc_timestamp,
)
from standardized_tabular_diffusion.evaluation.table import ValidatedTables, model_view_column_specs

COLUMN_SHAPES_METRIC_ID = "sdmetrics-column-shapes"
COLUMN_PAIR_TRENDS_METRIC_ID = "sdmetrics-column-pair-trends"
P2_METRIC_VERSION = "1.0.0"
P2_METRICS = (
    {"metric_id": COLUMN_SHAPES_METRIC_ID, "metric_version": P2_METRIC_VERSION},
    {"metric_id": COLUMN_PAIR_TRENDS_METRIC_ID, "metric_version": P2_METRIC_VERSION},
)
DETAILS_ARTIFACT_PATH = "artifacts/sdmetrics-details.json"


class ShapeTrendError(RuntimeError):
    """Raised when source output cannot satisfy the benchmark contract."""


@dataclass(frozen=True)
class ShapeTrendOutcome:
    atomic_results: tuple[AtomicResult, ...]
    property_scores: dict[str, float | None]
    denominator_counts: dict[str, int]
    source_details: dict[str, Any]
    source: dict[str, Any]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _json_value(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(key): _json_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _identity(request: EvaluationRequest, profile: dict[str, Any], run_id: str) -> dict[str, Any]:
    split = profile.get("split", {})
    split_id = split.get("split_id", "external-evaluation")
    model_id = (request.model or {}).get("model_id", "external")
    return {
        "run_id": run_id,
        "protocol_version": request.protocol["protocol_version"],
        "dataset_id": profile["dataset_id"],
        "dataset_version": profile["dataset_version"],
        "dataset_view": profile["dataset_view"],
        "split_id": split_id,
        "model_id": model_id,
        "comparison_track": request.comparison_track,
        "generation_seed": request.generation_seed,
    }


def _counts_for_column(tables: ValidatedTables, name: str) -> tuple[int, int]:
    valid = int(tables.reference[name].notna().sum() + tables.synthetic[name].notna().sum())
    total = len(tables.reference) + len(tables.synthetic)
    return valid, total - valid


def _counts_for_pair(tables: ValidatedTables, first: str, second: str) -> tuple[int, int]:
    real_valid = tables.reference[[first, second]].notna().all(axis=1).sum()
    synthetic_valid = tables.synthetic[[first, second]].notna().all(axis=1).sum()
    valid = int(real_valid + synthetic_valid)
    total = len(tables.reference) + len(tables.synthetic)
    return valid, total - valid


def _state_for_source_score(score: Any, error: Any) -> tuple[MetricState, str | None, str | None]:
    if _finite(score):
        return MetricState.COMPUTED, None, None
    if isinstance(error, str) and error.strip():
        return MetricState.IMPLEMENTATION_FAILURE, "upstream_metric_failure", error.strip()
    return (
        MetricState.MATHEMATICALLY_UNDEFINED,
        "upstream_nonfinite_result",
        "The pinned upstream implementation returned a non-finite score without an exception.",
    )


def _state_for_trend_row(row: dict[str, Any]) -> tuple[MetricState, str | None, str | None]:
    state = _state_for_source_score(row.get("Score"), row.get("Error"))
    if state[0] is not MetricState.MATHEMATICALLY_UNDEFINED:
        return state
    real_correlation = row.get("Real Correlation")
    real_association = row.get("Real Association")
    if (_finite(real_correlation) and abs(cast(float, real_correlation)) <= 0.5) or (
        _finite(real_association) and cast(float, real_association) <= 0.3
    ):
        return (
            MetricState.NOT_APPLICABLE,
            "below_source_threshold",
            "The real-data relationship does not exceed the pinned SDMetrics contribution threshold.",
        )
    return state


def _atomic(
    *,
    identity: dict[str, Any],
    metric_id: str,
    scope_type: str,
    scope_id: str,
    state: MetricState,
    score: Any,
    weight: float,
    n_reference: int,
    n_synthetic: int,
    n_valid: int,
    n_excluded: int,
    computed_at: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    warning_codes: tuple[str, ...] = (),
) -> AtomicResult:
    raw_value = float(score) if state is MetricState.COMPUTED else None
    return AtomicResult(
        **identity,
        metric_id=metric_id,
        metric_version=P2_METRIC_VERSION,
        dimension="fidelity",
        scope_type=scope_type,
        scope_id=scope_id,
        state=state,
        raw_direction=RawDirection.MAXIMIZE,
        weight=weight,
        n_reference=n_reference,
        n_synthetic=n_synthetic,
        n_valid=n_valid,
        n_excluded=n_excluded,
        computed_at=computed_at,
        raw_value=raw_value,
        normalized_value=raw_value,
        aggregate_contribution=raw_value * weight if raw_value is not None else None,
        unit="similarity",
        evaluator_id="sdmetrics",
        evaluator_version="0.28.3.dev0",
        reason_code=reason_code,
        reason_detail=reason_detail,
        warning_codes=warning_codes,
        artifact_ref=DETAILS_ARTIFACT_PATH,
    )


def _assert_source_aggregate(name: str, expected: float, actual: float) -> None:
    if math.isnan(expected) and math.isnan(actual):
        return
    if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
        raise ShapeTrendError(f"{name} Atomic Result contributions do not reproduce source score")


def evaluate_shape_trend(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    tables: ValidatedTables,
    *,
    run_id: str,
) -> ShapeTrendOutcome:
    """Execute the source backend and preserve every column and pair decision."""

    requested = {(item["metric_id"], item["metric_version"]) for item in request.metrics}
    supported = {(item["metric_id"], item["metric_version"]) for item in P2_METRICS}
    if requested != supported:
        raise ShapeTrendError(f"P2 requires exactly the two frozen metric identities, got {sorted(requested)}")
    if len(request.evaluator_seeds) != 1:
        raise ShapeTrendError("P2 requires exactly one evaluator seed")
    upstream = evaluate_quality(
        tables.reference,
        tables.synthetic,
        tables.metadata,
        evaluator_seed=request.evaluator_seeds[0],
    )
    identity = _identity(request, dataset_profile, run_id)
    computed_at = utc_timestamp()
    atoms: list[AtomicResult] = []

    shapes = upstream.column_shapes_details
    shape_rows = {row["Column"]: row for row in shapes.to_dict("records")}
    shape_computed = sum(_finite(row.get("Score")) for row in shape_rows.values())
    shape_weight = 1.0 / shape_computed if shape_computed else 0.0
    model_columns = model_view_column_specs(dataset_profile)
    for spec in model_columns:
        name = spec["name"]
        valid, excluded = _counts_for_column(tables, name)
        row = shape_rows.get(name)
        if row is None:
            atoms.append(
                _atomic(
                    identity=identity,
                    metric_id=COLUMN_SHAPES_METRIC_ID,
                    scope_type="column",
                    scope_id=spec["column_id"],
                    state=MetricState.NOT_APPLICABLE,
                    score=None,
                    weight=0.0,
                    n_reference=len(tables.reference),
                    n_synthetic=len(tables.synthetic),
                    n_valid=valid,
                    n_excluded=excluded,
                    computed_at=computed_at,
                    reason_code="unsupported_semantic_type",
                    reason_detail=f"SDMetrics Column Shapes does not evaluate {spec['semantic_type']} columns.",
                )
            )
            continue
        state, reason_code, reason_detail = _state_for_source_score(row.get("Score"), row.get("Error"))
        atoms.append(
            _atomic(
                identity=identity,
                metric_id=COLUMN_SHAPES_METRIC_ID,
                scope_type="column",
                scope_id=spec["column_id"],
                state=state,
                score=row.get("Score"),
                weight=shape_weight if state is MetricState.COMPUTED else 0.0,
                n_reference=len(tables.reference),
                n_synthetic=len(tables.synthetic),
                n_valid=valid,
                n_excluded=excluded,
                computed_at=computed_at,
                reason_code=reason_code,
                reason_detail=reason_detail,
            )
        )

    trends = upstream.column_pair_trends_details
    trend_rows = {(row["Column 1"], row["Column 2"]): row for row in trends.to_dict("records")}
    contributing = sum(_finite(row.get("Score")) and bool(row.get("Meets Threshold?")) for row in trend_rows.values())
    trend_weight = 1.0 / contributing if contributing else 0.0
    for first, second in itertools.combinations(model_columns, 2):
        first_name, second_name = first["name"], second["name"]
        valid, excluded = _counts_for_pair(tables, first_name, second_name)
        row = trend_rows.get((first_name, second_name))
        scope_id = f"{first['column_id']}--{second['column_id']}"
        if row is None:
            atoms.append(
                _atomic(
                    identity=identity,
                    metric_id=COLUMN_PAIR_TRENDS_METRIC_ID,
                    scope_type="pair",
                    scope_id=scope_id,
                    state=MetricState.NOT_APPLICABLE,
                    score=None,
                    weight=0.0,
                    n_reference=len(tables.reference),
                    n_synthetic=len(tables.synthetic),
                    n_valid=valid,
                    n_excluded=excluded,
                    computed_at=computed_at,
                    reason_code="unsupported_semantic_type",
                    reason_detail="SDMetrics Column Pair Trends does not evaluate this semantic-type pair.",
                )
            )
            continue
        state, reason_code, reason_detail = _state_for_trend_row(row)
        meets_threshold = bool(row.get("Meets Threshold?")) if state is MetricState.COMPUTED else False
        weight = trend_weight if meets_threshold else 0.0
        warnings = ("below_source_threshold",) if reason_code == "below_source_threshold" else ()
        atoms.append(
            _atomic(
                identity=identity,
                metric_id=COLUMN_PAIR_TRENDS_METRIC_ID,
                scope_type="pair",
                scope_id=scope_id,
                state=state,
                score=row.get("Score"),
                weight=weight,
                n_reference=len(tables.reference),
                n_synthetic=len(tables.synthetic),
                n_valid=valid,
                n_excluded=excluded,
                computed_at=computed_at,
                reason_code=reason_code,
                reason_detail=reason_detail,
                warning_codes=warnings,
            )
        )

    shape_contribution = (
        sum(atom.aggregate_contribution or 0.0 for atom in atoms if atom.metric_id == COLUMN_SHAPES_METRIC_ID)
        if shape_computed
        else math.nan
    )
    trend_contribution = (
        sum(atom.aggregate_contribution or 0.0 for atom in atoms if atom.metric_id == COLUMN_PAIR_TRENDS_METRIC_ID)
        if contributing
        else math.nan
    )
    _assert_source_aggregate("Column Shapes", upstream.column_shapes_score, shape_contribution)
    _assert_source_aggregate("Column Pair Trends", upstream.column_pair_trends_score, trend_contribution)

    property_scores = {
        "column_shapes": upstream.column_shapes_score if _finite(upstream.column_shapes_score) else None,
        "column_pair_trends": (
            upstream.column_pair_trends_score if _finite(upstream.column_pair_trends_score) else None
        ),
    }
    source_details = {
        "details_schema_version": "1.0.0",
        "source": upstream.source,
        "property_scores": property_scores,
        "column_shapes": _records(shapes),
        "column_pair_trends": _records(trends),
    }
    return ShapeTrendOutcome(
        atomic_results=tuple(atoms),
        property_scores=property_scores,
        denominator_counts={
            "requested_columns": len(model_columns),
            "source_evaluated_columns": len(shape_rows),
            "shape_contributing_columns": shape_computed,
            "requested_column_pairs": len(model_columns) * (len(model_columns) - 1) // 2,
            "source_evaluated_column_pairs": len(trend_rows),
            "trend_contributing_column_pairs": contributing,
        },
        source_details=source_details,
        source=upstream.source,
    )
