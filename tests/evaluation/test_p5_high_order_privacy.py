from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.backends.sdmetrics import calculate_dcr
from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest, MetricState, RawDirection
from standardized_tabular_diffusion.evaluation.high_order_privacy import (
    C2ST_AUROC_METRIC_ID,
    C2ST_FIDELITY_METRIC_ID,
    DOMIAS_AUROC_METRIC_ID,
    EXACT_TRAIN_COLLISION_METRIC_ID,
    P5_METRICS,
    SDMETRICS_DCR_METRIC_ID,
    SYNTHETIC_DUPLICATE_METRIC_ID,
    domias_density_ratio,
    domias_metrics,
    evaluate_high_order_privacy,
    exact_row_tokens,
    p5_evaluator_profile_reference,
)
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.table import validate_utility_tables

pytestmark = [pytest.mark.evaluation]


def p5_frames(profile, *, train_rows: int = 120, test_rows: int = 100):
    def frame(rows: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        payload: dict[str, object] = {}
        for spec in profile.payload["columns"]:
            name = spec["name"]
            semantic_type = spec["semantic_type"]
            if semantic_type in {"continuous", "integer"}:
                minimum = int(spec["valid_domain"].get("minimum", 0))
                maximum = int(spec["valid_domain"].get("maximum", minimum + 1000))
                values = rng.integers(minimum, maximum + 1, size=rows)
                payload[name] = values.astype("int64")
            elif semantic_type == "boolean":
                payload[name] = rng.choice([False, True], size=rows)
            else:
                values = spec["valid_domain"]["values"]
                payload[name] = rng.choice(values, size=rows)
        return pd.DataFrame(payload)

    train = frame(train_rows, 10)
    test = frame(test_rows, 20)
    synthetic = frame(train_rows, 30)
    return train, test, synthetic


def p5_request(profile, *, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> EvaluationRequest:
    protocol = resolve_protocol("p5-high-order-privacy", "0.1.0")
    return EvaluationRequest(
        subject_type="external-synthetic-table",
        reference_artifact={"artifact_id": "reference-table", "media_type": "text/csv", "sha256": "0" * 64},
        real_test_artifact={"artifact_id": "real-test-table", "media_type": "text/csv", "sha256": "2" * 64},
        sample_artifact={"artifact_id": "synthetic-table", "media_type": "text/csv", "sha256": "1" * 64},
        dataset_profile={
            "dataset_id": profile.dataset_id,
            "dataset_profile_version": profile.dataset_profile_version,
            "sha256": profile.fingerprint,
        },
        protocol={
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.protocol_version,
            "sha256": protocol.fingerprint,
        },
        metrics=tuple(P5_METRICS),
        comparison_track="native",
        generation_seed=7,
        evaluator_seeds=seeds,
        evaluator_profile=p5_evaluator_profile_reference(),
        model={"model_id": "fixture-model"},
    )


def test_exact_row_tokens_separate_train_collision_from_internal_duplication(adult_profile) -> None:
    train, _, synthetic = p5_frames(adult_profile, train_rows=40, test_rows=40)
    synthetic.iloc[0] = train.iloc[0]
    synthetic.iloc[-1] = synthetic.iloc[-2]
    specs = adult_profile.payload["columns"]
    train_tokens = set(exact_row_tokens(train, specs))
    tokens = exact_row_tokens(synthetic, specs)

    assert sum(token in train_tokens for token in tokens) == 1
    assert len(tokens) - len(set(tokens)) == 1


def test_sick_source_boolean_tokens_are_losslessly_canonicalized() -> None:
    profile = load_dataset_profile("configs/datasets/sick-uci-102-v1.json")
    row: dict[str, object] = {}
    for spec in profile.payload["columns"]:
        if spec["name"] not in profile.payload["table_contract"]["canonical_column_order"]:
            continue
        if spec["semantic_type"] in {"continuous", "integer"}:
            row[spec["name"]] = 1
        elif spec["semantic_type"] == "boolean":
            row[spec["name"]] = "f"
        else:
            row[spec["name"]] = spec["valid_domain"]["values"][0]
    frame = pd.DataFrame([row, {**row, **{"on_thyroxine": "t"}}])
    tables = validate_utility_tables(frame, frame, frame, profile.payload)

    assert tables.real_train["on_thyroxine"].tolist() == [False, True]
    assert tables.real_train["query_on_thyroxine"].tolist() == [False, False]


def test_sdmetrics_dcr_wrapper_matches_exact_upstream_null_and_zero_range_fixture() -> None:
    pytest.importorskip("sdmetrics")
    from sdmetrics.single_table.privacy.dcr_utils import calculate_dcr as upstream

    dataset = pd.DataFrame({"number": [1.0, 2.0, np.nan], "category": ["a", "b", None]})
    reference = pd.DataFrame({"number": [1.0, 1.0, np.nan], "category": ["a", "b", None]})
    metadata = {"columns": {"number": {"sdtype": "numerical"}, "category": {"sdtype": "categorical"}}}
    expected = upstream(dataset, reference, metadata, chunk_size=2).to_numpy()
    observed = calculate_dcr(dataset, reference, metadata, chunk_size=2)

    np.testing.assert_allclose(observed.distances, expected, rtol=0, atol=0)
    assert observed.source["implementation_symbol"].endswith("calculate_dcr")


def test_domias_formula_and_source_metrics_are_hand_computable() -> None:
    p_g = np.array([0.2, 0.4])
    p_r = np.array([0.1, 0.2])
    np.testing.assert_allclose(domias_density_ratio(p_g, p_r), p_g / (p_r + 1e-10))
    metrics = domias_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics == {"accuracy": 1.0, "auroc": 1.0, "advantage": 1.0, "tpr_at_fpr_01": 1.0}


def test_p5_is_deterministic_preserves_boundaries_and_emits_no_overall_score(adult_profile) -> None:
    pytest.importorskip("sdmetrics")
    train, test, synthetic = p5_frames(adult_profile)
    tables = validate_utility_tables(
        train,
        test,
        synthetic,
        adult_profile.payload,
        expected_synthetic_rows=len(synthetic),
    )
    request = p5_request(adult_profile)
    first = evaluate_high_order_privacy(request, adult_profile.payload, tables, run_id="p5-first")
    second = evaluate_high_order_privacy(request, adult_profile.payload, tables, run_id="p5-second")

    assert first.details == second.details
    assert first.high_order_summary["overall_fidelity_score"] is None
    assert first.privacy_summary["overall_privacy_score"] is None
    assert first.privacy_summary["formal_privacy_guarantee"] is False
    assert first.details["input_boundary"] == {
        "real_train_transform_fit_allowed": True,
        "real_test_transform_fit_allowed": False,
        "synthetic_transform_fit_allowed": False,
        "domias_auxiliary_reference_fit_only": True,
        "synthetic_repair_applied": False,
    }
    assert len(first.details["high_order"]["c2st_runs"]) == 5
    assert len(first.details["privacy"]["domias_runs"]) == 5
    assert all(run["real_test_used_for_transform_fit"] is False for run in first.details["high_order"]["c2st_runs"])
    assert all(
        run["transform_fit_source"] == "auxiliary-real-reference-only"
        for run in first.details["privacy"]["domias_runs"]
    )
    atoms = first.atomic_results
    assert len(atoms) == 34
    assert {atom.metric_id for atom in atoms} == {item["metric_id"] for item in P5_METRICS}
    assert all(atom.weight == 0 and atom.aggregate_contribution is None for atom in atoms)
    c2st_auc = [atom for atom in atoms if atom.metric_id == C2ST_AUROC_METRIC_ID]
    assert all(atom.raw_direction is RawDirection.TARGET and atom.reference_value == 0.5 for atom in c2st_auc)
    assert all(atom.state is MetricState.COMPUTED for atom in atoms)
    assert len([atom for atom in atoms if atom.metric_id == C2ST_FIDELITY_METRIC_ID]) == 5
    assert len([atom for atom in atoms if atom.metric_id == DOMIAS_AUROC_METRIC_ID]) == 5
    dcr = next(atom for atom in atoms if atom.metric_id == SDMETRICS_DCR_METRIC_ID)
    assert dcr.raw_direction is RawDirection.DISTRIBUTIONAL and dcr.raw_value is None
    assert next(atom for atom in atoms if atom.metric_id == EXACT_TRAIN_COLLISION_METRIC_ID).raw_value == 0
    assert next(atom for atom in atoms if atom.metric_id == SYNTHETIC_DUPLICATE_METRIC_ID).raw_value == 0


def test_p5_requires_the_complete_five_seed_diagnostic_schedule(adult_profile) -> None:
    train, test, synthetic = p5_frames(adult_profile)
    tables = validate_utility_tables(train, test, synthetic, adult_profile.payload)
    request = p5_request(adult_profile)
    request = replace(request, evaluator_seeds=(0,))
    with pytest.raises(Exception, match="seed schedule"):
        evaluate_high_order_privacy(request, adult_profile.payload, tables, run_id="p5-bad-seeds")
