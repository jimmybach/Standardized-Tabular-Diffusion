from __future__ import annotations

import math
from dataclasses import replace

import pytest

from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.shape_trend import (
    COLUMN_PAIR_TRENDS_METRIC_ID,
    COLUMN_SHAPES_METRIC_ID,
    ShapeTrendError,
    evaluate_shape_trend,
)
from standardized_tabular_diffusion.evaluation.table import validate_tables

pytestmark = [pytest.mark.evaluation, pytest.mark.source_parity]


def test_every_column_and_pair_has_an_atomic_result(adult_profile, adult_frames, p2_request) -> None:
    pytest.importorskip("sdmetrics")
    reference, synthetic = adult_frames
    independent_income = ["<=50K", "<=50K", ">50K", ">50K"] * 5
    reference["income"] = independent_income
    synthetic["income"] = independent_income
    tables = validate_tables(reference, synthetic, adult_profile.payload, expected_synthetic_rows=20)
    outcome = evaluate_shape_trend(p2_request, adult_profile.payload, tables, run_id="run-fixture")
    column_count = len(adult_profile.payload["columns"])
    assert len(outcome.atomic_results) == column_count + column_count * (column_count - 1) // 2
    identities = {(result.metric_id, result.scope_id) for result in outcome.atomic_results}
    assert len(identities) == len(outcome.atomic_results)
    for result in outcome.atomic_results:
        validate_instance("atomic-result", result.to_dict())
        if result.state.value != "computed":
            assert result.reason_code and result.reason_detail


def test_atomic_contributions_exactly_reproduce_source_properties(
    adult_profile, adult_frames, p2_request
) -> None:
    pytest.importorskip("sdmetrics")
    reference, synthetic = adult_frames
    independent_income = ["<=50K", "<=50K", ">50K", ">50K"] * 5
    reference["income"] = independent_income
    synthetic["income"] = independent_income
    tables = validate_tables(reference, synthetic, adult_profile.payload, expected_synthetic_rows=20)
    outcome = evaluate_shape_trend(p2_request, adult_profile.payload, tables, run_id="run-fixture")
    shape = sum(
        result.aggregate_contribution or 0.0
        for result in outcome.atomic_results
        if result.metric_id == COLUMN_SHAPES_METRIC_ID
    )
    trend = sum(
        result.aggregate_contribution or 0.0
        for result in outcome.atomic_results
        if result.metric_id == COLUMN_PAIR_TRENDS_METRIC_ID
    )
    assert math.isclose(shape, outcome.property_scores["column_shapes"], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(trend, outcome.property_scores["column_pair_trends"], rel_tol=1e-12, abs_tol=1e-12)
    below_threshold = [
        result for result in outcome.atomic_results if "below_source_threshold" in result.warning_codes
    ]
    assert below_threshold
    assert all(result.weight == 0 and result.aggregate_contribution is None for result in below_threshold)


def test_p2_requires_one_unambiguous_evaluator_seed(adult_profile, adult_frames, p2_request) -> None:
    reference, synthetic = adult_frames
    tables = validate_tables(reference, synthetic, adult_profile.payload, expected_synthetic_rows=20)
    request = replace(p2_request, evaluator_seeds=(23, 29))
    with pytest.raises(ShapeTrendError, match="exactly one evaluator seed"):
        evaluate_shape_trend(request, adult_profile.payload, tables, run_id="run-fixture")
