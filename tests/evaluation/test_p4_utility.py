from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.contracts import MetricState
from standardized_tabular_diffusion.evaluation.table import validate_utility_tables
from standardized_tabular_diffusion.evaluation.utility import (
    GLOBAL_TARGET_RATIO_METRIC_ID,
    LOCAL_METRIC_IDS,
    LOCAL_RETENTION_METRIC_ID,
    GlobalBackendResult,
    UtilityProfileError,
    evaluate_utility,
    global_target_ratio,
    load_p4_evaluator_profile,
    local_retention,
    validate_evaluator_profile,
    validate_utility_profile,
)

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def _learnable_adult_frames(adult_frames):
    reference, _ = adult_frames
    real_train = pd.concat([reference] * 6, ignore_index=True)
    real_test = pd.concat([reference] * 3, ignore_index=True)
    synthetic = real_train.copy(deep=True)
    for frame in (real_train, real_test, synthetic):
        frame["age"] = [17 + (index % 60) for index in range(len(frame))]
        frame["income"] = frame["age"].map(lambda value: ">50K" if value >= 45 else "<=50K")
    return real_train, real_test, synthetic


def _source_stub(train, test, target, task_type, seed, time_limit_seconds, arm):
    del train, test, target, seed, time_limit_seconds
    score = (0.8 if arm == "trtr" else 0.6) if task_type == "classification" else (
        2.0 if arm == "trtr" else 2.5
    )
    predictors = ("KNeighbors", "TabPFN", "XGBoost")
    return GlobalBackendResult(
        score=score,
        predictors=predictors,
        predictor_scores={name: score for name in predictors},
    )


def test_local_retention_and_tabstruct_target_ratios_are_hand_computable_and_unclipped() -> None:
    assert local_retention(0.35, 0.80, 0.71, higher_is_better=True, tolerance=1e-12) == pytest.approx(0.8)
    assert local_retention(10.0, 4.0, 5.5, higher_is_better=False, tolerance=1e-12) == pytest.approx(0.75)
    assert local_retention(0.2, 0.8, 0.92, higher_is_better=True, tolerance=1e-12) == pytest.approx(1.2)
    assert global_target_ratio(0.8, 0.6, task_type="classification") == pytest.approx(0.75)
    assert global_target_ratio(2.0, 2.5, task_type="regression") == pytest.approx(0.8)
    assert global_target_ratio(0.8, 0.88, task_type="classification") == pytest.approx(1.1)
    with pytest.raises(ZeroDivisionError):
        local_retention(0.5, 0.5, 0.7, higher_is_better=True, tolerance=1e-12)
    with pytest.raises(ZeroDivisionError):
        global_target_ratio(0.0, 0.2, task_type="classification")
    with pytest.raises(ZeroDivisionError):
        global_target_ratio(2.0, 0.0, task_type="regression")


def test_p4_executes_three_local_families_and_equal_target_global_formula(
    adult_profile,
    adult_frames,
    p4_request,
) -> None:
    real_train, real_test, synthetic = _learnable_adult_frames(adult_frames)
    tables = validate_utility_tables(
        real_train,
        real_test,
        synthetic,
        adult_profile.payload,
        expected_synthetic_rows=len(synthetic),
    )
    request = replace(p4_request, sample_artifact={**p4_request.sample_artifact, "row_count": len(synthetic)})

    outcome = evaluate_utility(
        request,
        adult_profile.payload,
        tables,
        run_id="run-p4-complete",
        global_scorer=_source_stub,
    )

    retention_atoms = [atom for atom in outcome.atomic_results if atom.metric_id == LOCAL_RETENTION_METRIC_ID]
    assert len(retention_atoms) == 3
    assert {atom.evaluator_id for atom in retention_atoms} == {
        "logistic-regression",
        "random-forest",
        "hist-gradient-boosting",
    }
    assert all(atom.state is MetricState.COMPUTED for atom in retention_atoms)
    assert outcome.local_summary["retention"] == pytest.approx(1.0)
    assert outcome.local_summary["retention_clipped"] is False
    ratio_atoms = [atom for atom in outcome.atomic_results if atom.metric_id == GLOBAL_TARGET_RATIO_METRIC_ID]
    assert len(ratio_atoms) == 15
    assert all(atom.state is MetricState.COMPUTED for atom in ratio_atoms)
    # Adult has nine categorical and six numerical canonical targets.
    assert outcome.global_summary["global_utility"] == pytest.approx((9 * 0.75 + 6 * 0.8) / 15)
    assert outcome.global_summary["ratio_clipped"] is False
    assert outcome.denominator_counts["global_fully_computed_targets"] == 15
    assert {run["test_fingerprint"] for run in outcome.details["local_runs"]} == {
        request.real_test_artifact["sha256"]
    }


def test_missing_synthetic_class_is_explicit_and_never_receives_favorable_global_default(
    adult_profile,
    adult_frames,
    p4_request,
) -> None:
    real_train, real_test, synthetic = _learnable_adult_frames(adult_frames)
    synthetic["income"] = "<=50K"
    tables = validate_utility_tables(
        real_train,
        real_test,
        synthetic,
        adult_profile.payload,
        expected_synthetic_rows=len(synthetic),
    )
    request = replace(p4_request, sample_artifact={**p4_request.sample_artifact, "row_count": len(synthetic)})

    outcome = evaluate_utility(
        request,
        adult_profile.payload,
        tables,
        run_id="run-p4-missing-class",
        global_scorer=_source_stub,
    )

    local_tstr = [
        atom
        for atom in outcome.atomic_results
        if atom.metric_id == LOCAL_METRIC_IDS["macro-f1"] and atom.scope_id.endswith("--tstr")
    ]
    assert local_tstr and all(atom.state is MetricState.INSUFFICIENT_SUPPORT for atom in local_tstr)
    assert {atom.reason_code for atom in local_tstr} == {"insufficient_target_support"}
    income_ratio = next(
        atom
        for atom in outcome.atomic_results
        if atom.metric_id == GLOBAL_TARGET_RATIO_METRIC_ID and atom.scope_id.startswith("income--")
    )
    assert income_ratio.state is MetricState.INSUFFICIENT_SUPPORT
    assert income_ratio.raw_value is None
    assert outcome.local_summary["retention"] is None
    assert outcome.global_summary["global_utility"] is None
    assert outcome.global_summary["target_ratios"]["income"] is None


def test_regression_local_utility_uses_rmse_and_lower_is_better_retention(
    adult_profile,
    adult_frames,
    p4_request,
) -> None:
    profile = copy.deepcopy(adult_profile.payload)
    for column in profile["columns"]:
        if column["column_id"] == "age":
            column["roles"] = ["primary_target"]
        elif column["column_id"] == "income":
            column["roles"] = ["feature"]
    profile["predictive_tasks"] = [
        {
            "task_id": "predict-age",
            "target_column_id": "age",
            "task_type": "regression",
            "status": "reviewed",
            "positive_class": None,
            "label_mapping": {},
            "primary_metric": "rmse",
            "secondary_metrics": ["mae", "r2"],
            "minimum_support": {"real_train_rows": 2, "synthetic_train_rows": 2},
            "dummy_strategy": "mean",
            "retention_metric_id": "std-local-utility-retention",
            "retention_metric_version": "1.0.0",
        }
    ]
    profile["utility"]["local"]["primary_task_id"] = "predict-age"
    validate_utility_profile(profile, load_p4_evaluator_profile())
    real_train, real_test, synthetic = _learnable_adult_frames(adult_frames)
    for frame in (real_train, real_test, synthetic):
        frame["age"] = frame["hours.per.week"].astype(float) * 2.0
    tables = validate_utility_tables(real_train, real_test, synthetic, profile, expected_synthetic_rows=len(synthetic))
    request = replace(
        p4_request,
        dataset_profile={**p4_request.dataset_profile, "sha256": "a" * 64},
        sample_artifact={**p4_request.sample_artifact, "row_count": len(synthetic)},
    )

    outcome = evaluate_utility(request, profile, tables, run_id="run-p4-regression", global_scorer=_source_stub)

    rmse_atoms = [atom for atom in outcome.atomic_results if atom.metric_id == LOCAL_METRIC_IDS["rmse"]]
    assert len(rmse_atoms) == 9
    assert all(atom.raw_direction.value == "minimize" for atom in rmse_atoms)
    assert outcome.local_summary["primary_metric"] == "rmse"
    assert outcome.local_summary["retention"] == pytest.approx(1.0)


def test_global_profile_requires_every_model_view_target_or_a_reasoned_exclusion(adult_profile) -> None:
    profile = copy.deepcopy(adult_profile.payload)
    profile["utility"]["global"]["included_target_column_ids"].remove("age")
    with pytest.raises(UtilityProfileError, match="explicitly included or excluded"):
        validate_utility_profile(profile, load_p4_evaluator_profile())

    profile["utility"]["global"]["excluded_targets"] = [
        {"column_id": "age", "reason_code": "unsupported-target-type", "reason_detail": "Fixture exclusion."}
    ]
    validate_utility_profile(profile, load_p4_evaluator_profile())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("official_results_allowed",), True),
        (("default_evaluator_seeds",), [0]),
        (("local", "classification", "primary_metric"), "accuracy"),
        (("local", "retention", "clipping"), "zero-one"),
        (("global", "predictors"), ["xgb", "knn"]),
        (("global", "source_parity_claimed"), True),
    ],
)
def test_p4_evaluator_profile_rejects_admission_formula_and_source_drift(path, value) -> None:
    profile = copy.deepcopy(load_p4_evaluator_profile())
    cursor = profile
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(UtilityProfileError):
        validate_evaluator_profile(profile)
