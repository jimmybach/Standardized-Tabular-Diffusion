from __future__ import annotations

import json
import sys
import types
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standardized_tabular_diffusion.evaluation import tabstruct

TABDIFF_ROOT = REPO_ROOT / "TabDiff-main"
if str(TABDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(TABDIFF_ROOT))

mle = import_module("eval.mle.mle")


def test_prepare_ml_problem_uses_deterministic_split() -> None:
    train = np.array(
        [
            [0.0, "a", 0],
            [1.0, "b", 1],
            [2.0, "a", 0],
            [3.0, "b", 1],
            [4.0, "a", 0],
            [5.0, "b", 1],
            [6.0, "a", 0],
            [7.0, "b", 1],
            [8.0, "a", 0],
        ],
        dtype=object,
    )
    test = train.copy()
    info = {
        "task_type": "binclass",
        "num_col_idx": [0],
        "cat_col_idx": [1],
        "target_col_idx": [2],
    }

    first = mle.prepare_ml_problem(train, test, info, seed=42)
    second = mle.prepare_ml_problem(train, test, info, seed=42)
    third = mle.prepare_ml_problem(train, test, info, seed=7)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_array_equal(first[3], second[3])

    assert not np.array_equal(first[0], third[0]) or not np.array_equal(first[2], third[2])


def test_normalized_summary_is_byte_stable_across_repeated_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    tabdiff_root = repo_root / "TabDiff-main"
    data_dir = tabdiff_root / "data" / "toy"
    synthetic_dir = tabdiff_root / "synthetic" / "toy"
    data_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)

    info = {
        "task_type": "binclass",
        "num_col_idx": [0],
        "cat_col_idx": [],
        "target_col_idx": [1],
    }
    (data_dir / "info.json").write_text(json.dumps(info))

    real_df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0, 1, 0]})
    sample_df = pd.DataFrame({"x": [0.1, 0.9, 1.8], "y": [0, 1, 0]})
    real_df.to_csv(synthetic_dir / "real.csv", index=False)
    real_df.to_csv(synthetic_dir / "test.csv", index=False)
    sample_path = tmp_path / "samples.csv"
    sample_df.to_csv(sample_path, index=False)

    fake_metrics_module = types.ModuleType("tabdiff.metrics")

    class FakeTabMetrics:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def evaluate(self, syn_df: pd.DataFrame):
            assert list(syn_df.columns) == ["x", "y"]
            return (
                {
                    "density/Shape": 0.75,
                    "density/Trend": 0.50,
                    "density/Overall": 0.625,
                    "mle": 0.8,
                    "c2st": 0.25,
                },
                {
                    "mle": {
                        "best_auroc_scores": {
                            "XGBClassifier": {
                                "roc_auc": 0.8,
                            }
                        }
                    }
                },
            )

    fake_metrics_module.TabMetrics = FakeTabMetrics
    monkeypatch.setitem(sys.modules, "tabdiff.metrics", fake_metrics_module)
    monkeypatch.setattr(
        tabstruct,
        "compute_tabstruct_structural_fidelity",
        lambda **kwargs: {
            "global_utility": 1.25,
            "local_utility": 1.0,
            "status": "computed",
            "implementation": {
                "predictors": ["XGB", "KNN"],
                "predictor_policy": {
                    "tabpfn_enabled": False,
                },
            },
        },
    )

    first_output = tmp_path / "summary-first.json"
    second_output = tmp_path / "summary-second.json"

    tabstruct.normalize_tabdiff_or_tabsyn_summary(
        repo_root=repo_root,
        model_name="tabsyn",
        dataset="toy",
        sample_path=sample_path,
        output_path=first_output,
    )
    tabstruct.normalize_tabdiff_or_tabsyn_summary(
        repo_root=repo_root,
        model_name="tabsyn",
        dataset="toy",
        sample_path=sample_path,
        output_path=second_output,
    )

    assert first_output.read_bytes() == second_output.read_bytes()

    payload = json.loads(first_output.read_text())
    assert payload["metrics"]["ml_efficacy"]["task_type"] == "classification"
    assert payload["metrics"]["ml_efficacy"]["primary_metric_value"] == 0.8
    assert payload["metrics"]["structural_fidelity"]["global_utility"] == 1.25
