from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.backends import sdmetrics as backend

pytestmark = [pytest.mark.evaluation, pytest.mark.source_parity]


def _source_available() -> None:
    pytest.importorskip("sdmetrics")


def test_backend_matches_locked_official_properties_exactly() -> None:
    _source_available()
    from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes

    rows = 24
    real = pd.DataFrame(
        {
            "number": list(range(rows)),
            "category": ["a", "b", "c"] * 8,
            "flag": [True, False] * 12,
            "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "constant": [1.0] * rows,
        }
    )
    synthetic = real.iloc[::-1].reset_index(drop=True)
    metadata = {
        "columns": {
            "number": {"sdtype": "numerical"},
            "category": {"sdtype": "categorical"},
            "flag": {"sdtype": "boolean"},
            "date": {"sdtype": "datetime"},
            "constant": {"sdtype": "numerical"},
        }
    }
    seed = 7
    actual = backend.evaluate_quality(real, synthetic, metadata, evaluator_seed=seed)
    previous_state = np.random.get_state()
    np.random.seed(seed)
    shapes = ColumnShapes()
    trends = ColumnPairTrends()
    shapes.num_rows_subsample = 50000
    trends.num_rows_subsample = 50000
    trends.real_correlation_threshold = 0.5
    trends.real_association_threshold = 0.3
    try:
        expected_shape = shapes.get_score(real.copy(), synthetic.copy(), metadata)
        expected_trend = trends.get_score(real.copy(), synthetic.copy(), metadata)
    finally:
        np.random.set_state(previous_state)

    assert math.isclose(actual.column_shapes_score, expected_shape, rel_tol=0, abs_tol=0)
    assert math.isclose(actual.column_pair_trends_score, expected_trend, rel_tol=0, abs_tol=0)
    pd.testing.assert_frame_equal(actual.column_shapes_details, shapes.details, check_exact=True)
    pd.testing.assert_frame_equal(actual.column_pair_trends_details, trends.details, check_exact=True)
    assert actual.source["revision"] == "ba8842f2ba04ce914f698cc1cf746ca12338ab0e"
    assert actual.source["execution"]["evaluator_seed"] == seed


def test_backend_controls_upstream_subsampling_and_restores_caller_rng() -> None:
    _source_available()
    from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes

    rows = 50_001
    real = pd.DataFrame(
        {
            "left": [f"l{index % 17}" for index in range(rows)],
            "right": [f"r{index % 17}" for index in range(rows)],
        }
    )
    synthetic = real.sample(frac=1, random_state=91).reset_index(drop=True)
    metadata = {
        "columns": {
            "left": {"sdtype": "categorical"},
            "right": {"sdtype": "categorical"},
        }
    }
    np.random.seed(1234)
    caller_state = np.random.get_state()
    first = backend.evaluate_quality(real, synthetic, metadata, evaluator_seed=29)
    restored_state = np.random.get_state()
    second = backend.evaluate_quality(real, synthetic, metadata, evaluator_seed=29)

    direct_state = np.random.get_state()
    np.random.seed(29)
    shapes = ColumnShapes()
    trends = ColumnPairTrends()
    shapes.num_rows_subsample = 50000
    trends.num_rows_subsample = 50000
    trends.real_correlation_threshold = 0.5
    trends.real_association_threshold = 0.3
    try:
        direct_shape = shapes.get_score(real.copy(), synthetic.copy(), metadata)
        direct_trend = trends.get_score(real.copy(), synthetic.copy(), metadata)
    finally:
        np.random.set_state(direct_state)

    assert caller_state[0] == restored_state[0]
    assert all(
        np.array_equal(before, after)
        for before, after in zip(caller_state[1:], restored_state[1:], strict=True)
    )
    assert first.column_shapes_score == second.column_shapes_score
    assert first.column_pair_trends_score == second.column_pair_trends_score
    assert first.column_shapes_score == direct_shape
    assert first.column_pair_trends_score == direct_trend
    pd.testing.assert_frame_equal(first.column_shapes_details, second.column_shapes_details, check_exact=True)
    pd.testing.assert_frame_equal(
        first.column_pair_trends_details,
        second.column_pair_trends_details,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(first.column_shapes_details, shapes.details, check_exact=True)
    pd.testing.assert_frame_equal(first.column_pair_trends_details, trends.details, check_exact=True)


def test_backend_rejects_any_source_tree_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    _source_available()
    monkeypatch.setattr(backend, "_source_tree_digest", lambda _: (121, "0" * 64))
    with pytest.raises(backend.SDMetricsBackendError, match="do not match"):
        backend.verify_sdmetrics_source()


def test_backend_rejects_missing_or_changed_upstream_license(monkeypatch: pytest.MonkeyPatch) -> None:
    _source_available()
    monkeypatch.setattr(backend, "_installed_license_digest", lambda: "0" * 64)
    with pytest.raises(backend.SDMetricsSourceError, match="license does not match"):
        backend.verify_sdmetrics_source()


def test_p2_backend_never_imports_legacy_tabstruct() -> None:
    source = Path(backend.__file__).read_text(encoding="utf-8")
    assert "evaluation.tabstruct" not in source
