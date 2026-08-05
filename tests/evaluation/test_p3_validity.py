from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.contracts import MetricState
from standardized_tabular_diffusion.evaluation.table import validate_tables
from standardized_tabular_diffusion.evaluation.validity import (
    COLUMN_VALIDITY_METRIC_ID,
    CONSTRAINT_VALIDITY_METRIC_ID,
    ValidityProfileError,
    evaluate_validity,
    validate_validity_profile,
)

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_reviewed_adult_column_rules_are_hand_computable_and_do_not_repair(
    adult_profile,
    adult_frames,
    p3_request,
) -> None:
    reference, synthetic = adult_frames
    synthetic = synthetic.copy(deep=True)
    synthetic["age"] = synthetic["age"].astype(float)
    synthetic.loc[0, "age"] = 17.5
    synthetic.loc[0, "workclass"] = "unreviewed-category"
    synthetic.loc[0, "occupation"] = pd.NA
    original = synthetic.copy(deep=True)

    tables = validate_tables(
        reference,
        synthetic,
        adult_profile.payload,
        expected_synthetic_rows=20,
        content_mode="preserve",
    )
    outcome = evaluate_validity(p3_request, adult_profile.payload, tables, run_id="run-p3-fixture")

    column_atoms = [atom for atom in outcome.atomic_results if atom.metric_id == COLUMN_VALIDITY_METRIC_ID]
    constraint_atoms = [atom for atom in outcome.atomic_results if atom.metric_id == CONSTRAINT_VALIDITY_METRIC_ID]
    assert len(column_atoms) == 15
    assert {atom.scope_id: atom.raw_value for atom in column_atoms}["age"] == pytest.approx(0.95)
    assert {atom.scope_id: atom.raw_value for atom in column_atoms}["workclass"] == pytest.approx(0.95)
    assert {atom.scope_id: atom.raw_value for atom in column_atoms}["occupation"] == pytest.approx(0.95)
    assert outcome.property_scores["column_validity_score"] == pytest.approx(0.99)
    assert outcome.property_scores["constraint_validity_score"] is None
    assert outcome.property_scores["validity_score"] == pytest.approx(0.99)
    assert outcome.property_scores["fully_valid_row_rate"] == pytest.approx(0.95)
    assert len(constraint_atoms) == 1
    assert constraint_atoms[0].state is MetricState.NOT_APPLICABLE
    assert constraint_atoms[0].reason_code == "no_reviewed_constraints"
    assert outcome.details["synthetic_repair_applied"] is False
    pd.testing.assert_frame_equal(synthetic, original)


def test_observed_adult_ranges_remain_soft_without_an_explicit_hard_rule(
    adult_profile,
    adult_frames,
    p3_request,
) -> None:
    reference, synthetic = adult_frames
    synthetic.loc[0, "age"] = 1000
    tables = validate_tables(reference, synthetic, adult_profile.payload, content_mode="preserve")

    outcome = evaluate_validity(p3_request, adult_profile.payload, tables, run_id="run-p3-soft-range")

    age = next(
        atom
        for atom in outcome.atomic_results
        if atom.metric_id == COLUMN_VALIDITY_METRIC_ID and atom.scope_id == "age"
    )
    assert age.raw_value == 1.0
    assert "observed-range-exceedance" in adult_profile.payload["validity"]["soft_diagnostics"]


def test_p3_content_preservation_does_not_round_large_nullable_integers(
    adult_profile,
    adult_frames,
    p3_request,
) -> None:
    reference, synthetic = adult_frames
    exact = 2**53 + 1
    ages: list[object] = [exact, pd.NA, *synthetic["age"].iloc[2:].tolist()]
    synthetic["age"] = pd.Series(ages, dtype="object")

    tables = validate_tables(reference, synthetic, adult_profile.payload, content_mode="preserve")
    outcome = evaluate_validity(p3_request, adult_profile.payload, tables, run_id="run-p3-large-integer")

    assert tables.synthetic.loc[0, "age"] == exact
    age = next(
        atom
        for atom in outcome.atomic_results
        if atom.metric_id == COLUMN_VALIDITY_METRIC_ID and atom.scope_id == "age"
    )
    assert age.raw_value == pytest.approx(0.95)


def _reviewed_constraint(
    constraint_id: str,
    constraint_type: str,
    columns: list[str],
    parameters: dict,
) -> dict:
    return {
        "constraint_id": constraint_id,
        "constraint_type": constraint_type,
        "columns": columns,
        "parameters": parameters,
        "applicability": {"type": "all_rows"},
        "missing_behavior": "violation",
        "evidence": {"source_type": "recorded-human-review", "reference": "Hand-computable test contract."},
        "severity": "hard",
        "version": "1.0.0",
    }


def _evidence() -> dict[str, str]:
    return {"source_type": "recorded-human-review", "reference": "Hand-computable test contract."}


def _rule(rule_id: str, rule_type: str, selector: dict, parameters: dict) -> dict:
    return {
        "rule_id": rule_id,
        "rule_type": rule_type,
        "selector": selector,
        "parameters": parameters,
        "evidence": _evidence(),
        "severity": "hard",
        "version": "1.0.0",
    }


def test_bounds_formats_lengths_datetime_ranges_and_uniqueness_are_hand_computable(p3_request) -> None:
    columns = [
        {
            "name": "number",
            "column_id": "number",
            "semantic_type": "integer",
            "roles": ["feature"],
            "nullable_model_input": False,
            "valid_domain": {"minimum": 0, "maximum": 10},
        },
        {
            "name": "code",
            "column_id": "code",
            "semantic_type": "string",
            "roles": ["feature"],
            "nullable_model_input": False,
            "valid_domain": None,
        },
        {
            "name": "event_date",
            "column_id": "event-date",
            "semantic_type": "datetime",
            "roles": ["feature"],
            "nullable_model_input": False,
            "valid_domain": None,
        },
        {
            "name": "unique_id",
            "column_id": "unique-id",
            "semantic_type": "string",
            "roles": ["feature"],
            "nullable_model_input": False,
            "valid_domain": None,
        },
        {
            "name": "category",
            "column_id": "category",
            "semantic_type": "categorical",
            "roles": ["feature"],
            "nullable_model_input": False,
            "valid_domain": {"values": ["x", "y"]},
        },
    ]
    profile = {
        "dataset_id": "rule-fixture",
        "dataset_profile_version": "1.0.0",
        "dataset_version": "1.0.0",
        "dataset_view": "model-view",
        "table_contract": {"canonical_column_order": [column["name"] for column in columns]},
        "columns": columns,
        "split": {"split_id": "fixture-split"},
        "validity": {
            "contract_schema_version": "1.0.0",
            "status": "reviewed-diagnostic",
            "hard_column_rules": [
                _rule("not-null", "not_null", {"nullable_model_input": False}, {}),
                _rule("finite", "finite", {"semantic_types": ["integer"]}, {}),
                _rule("integer", "integer", {"semantic_types": ["integer"]}, {}),
                _rule("bounds", "bounds", {"column_ids": ["number"]}, {"source": "valid_domain"}),
                _rule("code-pattern", "regex", {"column_ids": ["code"]}, {"pattern": "^[A-Z]+$"}),
                _rule("code-length", "length", {"column_ids": ["code"]}, {"minimum": 2, "maximum": 2}),
                _rule(
                    "date-range",
                    "datetime_range",
                    {"column_ids": ["event-date"]},
                    {"minimum": "2020-01-01", "maximum": "2020-12-31"},
                ),
                _rule("identifier-unique", "unique", {"column_ids": ["unique-id"]}, {}),
                _rule(
                    "category-domain", "allowed_values", {"column_ids": ["category"]}, {"source": "valid_domain.values"}
                ),
            ],
            "cross_column_constraints": [],
            "soft_diagnostics": [],
            "unresolved_reviews": [],
        },
    }
    reference = pd.DataFrame(
        {
            "number": [1, 2, 3, 4],
            "code": ["AA", "AB", "AC", "AD"],
            "event_date": ["2020-06-01"] * 4,
            "unique_id": ["r1", "r2", "r3", "r4"],
            "category": ["x", "y", "x", "y"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "number": [1.0, 2.5, 11.0, 4.0],
            "code": ["AA", "a", "AAA", "BB"],
            "event_date": ["2020-06-01", "2019-12-31", "2021-01-01", "2020-06-01"],
            "unique_id": ["a", "a", "b", "c"],
            "category": ["x", "z", "y", "x"],
        }
    )
    tables = validate_tables(reference, synthetic, profile, content_mode="preserve")
    request = replace(
        p3_request,
        dataset_profile={"dataset_id": "rule-fixture", "dataset_profile_version": "1.0.0", "sha256": "a" * 64},
        sample_artifact={**p3_request.sample_artifact, "row_count": 4},
    )

    outcome = evaluate_validity(request, profile, tables, run_id="run-p3-rule-families")

    rates = {
        atom.scope_id: atom.raw_value for atom in outcome.atomic_results if atom.metric_id == COLUMN_VALIDITY_METRIC_ID
    }
    assert rates == {"number": 0.5, "code": 0.5, "event-date": 0.5, "unique-id": 0.5, "category": 0.75}
    assert outcome.property_scores["column_validity_score"] == pytest.approx(0.55)
    assert outcome.property_scores["fully_valid_row_rate"] == pytest.approx(0.25)


def test_cross_column_constraints_have_stable_scopes_applicability_and_equal_weights(
    adult_profile,
    adult_frames,
    p3_request,
) -> None:
    profile = copy.deepcopy(adult_profile.payload)
    profile["validity"]["cross_column_constraints"] = [
        _reviewed_constraint(
            "capital-components-mutually-exclusive",
            "mutual_exclusion",
            ["capital-gain", "capital-loss"],
            {"truthy_values": [1], "maximum_true": 1},
        ),
        _reviewed_constraint(
            "sex-income-allowed-combinations",
            "allowed_combinations",
            ["sex", "income"],
            {
                "allowed": [
                    ["Female", "<=50K"],
                    ["Female", ">50K"],
                    ["Male", "<=50K"],
                    ["Male", ">50K"],
                ]
            },
        ),
    ]
    reference, synthetic = adult_frames
    synthetic["capital.gain"] = 0
    synthetic["capital.loss"] = 0
    synthetic.loc[0, ["capital.gain", "capital.loss"]] = 1
    tables = validate_tables(reference, synthetic, profile, content_mode="preserve")

    outcome = evaluate_validity(p3_request, profile, tables, run_id="run-p3-constraints")

    atoms = [atom for atom in outcome.atomic_results if atom.metric_id == CONSTRAINT_VALIDITY_METRIC_ID]
    assert [atom.scope_id for atom in atoms] == [
        "capital-components-mutually-exclusive",
        "sex-income-allowed-combinations",
    ]
    assert [atom.raw_value for atom in atoms] == pytest.approx([0.95, 1.0])
    assert [atom.weight for atom in atoms] == [0.5, 0.5]
    assert outcome.property_scores["constraint_validity_score"] == pytest.approx(0.975)
    assert outcome.property_scores["validity_score"] == pytest.approx(0.9875)


@pytest.mark.parametrize(
    ("constraint_type", "columns", "parameters"),
    [
        ("comparison", ["capital-gain", "capital-loss"], {"operator": "ge"}),
        (
            "conditional_domain",
            ["sex", "relationship"],
            {"if_values": ["Male"], "then_allowed_values": ["Husband", "Not-in-family"]},
        ),
        (
            "functional_dependency",
            ["education", "education-num"],
            {"mapping": [{"determinant": ["Preschool"], "dependent": 1}]},
        ),
        (
            "sum_equals",
            ["capital-gain", "capital-loss", "fnlwgt"],
            {"absolute_tolerance": 0.0, "relative_tolerance": 0.0},
        ),
    ],
)
def test_supported_constraint_families_execute_without_dynamic_code(
    adult_profile,
    adult_frames,
    p3_request,
    constraint_type: str,
    columns: list[str],
    parameters: dict,
) -> None:
    profile = copy.deepcopy(adult_profile.payload)
    profile["validity"]["cross_column_constraints"] = [
        _reviewed_constraint(f"fixture-{constraint_type.replace('_', '-')}", constraint_type, columns, parameters)
    ]
    reference, synthetic = adult_frames
    tables = validate_tables(reference, synthetic, profile, content_mode="preserve")

    outcome = evaluate_validity(p3_request, profile, tables, run_id=f"run-p3-{constraint_type}")

    atom = next(atom for atom in outcome.atomic_results if atom.metric_id == CONSTRAINT_VALIDITY_METRIC_ID)
    assert atom.state is MetricState.COMPUTED
    assert atom.raw_value is not None


def test_constraint_with_no_applicable_rows_is_explicit_not_applicable(
    adult_profile,
    adult_frames,
    p3_request,
) -> None:
    profile = copy.deepcopy(adult_profile.payload)
    constraint = _reviewed_constraint(
        "conditional-comparison",
        "comparison",
        ["capital-gain", "capital-loss"],
        {"operator": "ge"},
    )
    constraint["applicability"] = {"type": "column_equals", "column_id": "sex", "value": "not-present"}
    profile["validity"]["cross_column_constraints"] = [constraint]
    reference, synthetic = adult_frames
    tables = validate_tables(reference, synthetic, profile, content_mode="preserve")

    outcome = evaluate_validity(p3_request, profile, tables, run_id="run-p3-no-applicable")

    atom = next(atom for atom in outcome.atomic_results if atom.metric_id == CONSTRAINT_VALIDITY_METRIC_ID)
    assert atom.state is MetricState.NOT_APPLICABLE
    assert atom.reason_code == "no_applicable_rows"
    assert outcome.property_scores["constraint_validity_score"] is None


def test_fully_valid_row_rate_is_width_sensitive_and_never_changes_aggregate_formula(
    adult_profile,
    adult_frames,
    p3_request,
) -> None:
    reference, base = adult_frames
    same_row = base.copy(deep=True)
    different_rows = base.copy(deep=True)
    same_row.loc[0, ["workclass", "occupation"]] = "invalid"
    different_rows.loc[0, "workclass"] = "invalid"
    different_rows.loc[1, "occupation"] = "invalid"

    outcomes = []
    for run_id, synthetic in (("same", same_row), ("different", different_rows)):
        tables = validate_tables(reference, synthetic, adult_profile.payload, content_mode="preserve")
        outcomes.append(evaluate_validity(p3_request, adult_profile.payload, tables, run_id=run_id))

    assert outcomes[0].property_scores["column_validity_score"] == pytest.approx(
        outcomes[1].property_scores["column_validity_score"]
    )
    assert outcomes[0].property_scores["validity_score"] == pytest.approx(outcomes[1].property_scores["validity_score"])
    assert outcomes[0].property_scores["fully_valid_row_rate"] == pytest.approx(0.95)
    assert outcomes[1].property_scores["fully_valid_row_rate"] == pytest.approx(0.90)


def test_malformed_or_incompletely_sourced_validity_profiles_fail_closed(adult_profile) -> None:
    profile = copy.deepcopy(adult_profile.payload)
    profile["validity"]["hard_column_rules"][1]["rule_id"] = "model-input-not-null"
    with pytest.raises(ValidityProfileError, match="Duplicate hard column rule_id"):
        validate_validity_profile(profile)

    profile = copy.deepcopy(adult_profile.payload)
    del profile["validity"]["hard_column_rules"][0]["evidence"]["reference"]
    with pytest.raises(ValidityProfileError, match="evidence"):
        validate_validity_profile(profile)

    profile = copy.deepcopy(adult_profile.payload)
    profile["validity"]["hard_column_rules"].append(
        _rule("inverted-age-bounds", "bounds", {"column_ids": ["age"]}, {"minimum": 90, "maximum": 17})
    )
    with pytest.raises(ValidityProfileError, match="minimum must not exceed maximum"):
        validate_validity_profile(profile)

    profile = copy.deepcopy(adult_profile.payload)
    profile["validity"]["hard_column_rules"][3]["parameters"] = {"values": ["duplicate", "duplicate"]}
    with pytest.raises(ValidityProfileError, match="must not contain duplicate values"):
        validate_validity_profile(profile)


def test_sick_audit_only_columns_are_not_model_view_denominators() -> None:
    from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile
    from standardized_tabular_diffusion.evaluation.table import model_view_column_specs

    profile = load_dataset_profile("configs/datasets/sick-uci-102-v1.json")
    validate_validity_profile(profile.payload)
    columns = model_view_column_specs(profile.payload)
    assert len(columns) == 29
    assert {column["name"] for column in columns}.isdisjoint({"TBG", "record_id"})
