from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


METRIC_DEFINITIONS = {
    "protocol_name": "tabstruct-aligned-v1",
    "source": {
        "paper": "TabStruct: Measuring Structural Fidelity of Tabular Data",
        "dimensions": [
            "density_fidelity",
            "ml_efficacy",
            "detection",
            "privacy",
            "structural_fidelity",
        ],
    },
    "fields": {
        "density.shape_score": "SDMetrics column-shape fidelity score on reordered real/synthetic tables.",
        "density.trend_score": "SDMetrics column-pair trend fidelity score on reordered real/synthetic tables.",
        "density.overall_score": "Mean of shape and trend fidelity.",
        "ml_efficacy.primary_metric_value": "Primary downstream efficacy metric emitted by the evaluator.",
        "detection.logistic_detection": "Logistic detection score from the standardized evaluator when available.",
        "privacy.dcr_score": "Distance-to-closest-record style privacy score when available.",
        "structural_fidelity.global_utility": "Mean relative per-variable utility following the TabStruct global utility definition.",
        "structural_fidelity.utility_per_feature": "TabStruct UtilityPerFeature scores using the official tiny-default predictor ensemble.",
    },
}
DEFAULT_BENCHMARK_SEED = 42


def _normalize_task_type(task_type: str) -> str:
    mapping = {
        "binclass": "classification",
        "multiclass": "classification",
        "regression": "regression",
    }
    return mapping.get(task_type, task_type)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _tabpfn_enabled() -> bool:
    value = os.environ.get("STANDARDIZED_TABULAR_DIFFUSION_ENABLE_TABPFN", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]) -> Any:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _quiet_benchmark_context() -> Any:
    cache_root = Path(tempfile.gettempdir()) / "standardized-tabular-diffusion"
    cache_root.mkdir(parents=True, exist_ok=True)
    env_updates = {
        "DO_NOT_TRACK": "1",
        "DISABLE_TELEMETRY": "1",
        "POSTHOG_DISABLED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MPLCONFIGDIR": str(cache_root / "mplconfig"),
        "TABPFN_MODEL_CACHE_DIR": str(cache_root / "tabpfn-cache"),
        "XDG_CACHE_HOME": str(cache_root / "xdg-cache"),
    }
    Path(env_updates["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env_updates["TABPFN_MODEL_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env_updates["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    with (
        _temporary_env(env_updates),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        yield


@contextlib.contextmanager
def _seeded_benchmark_context(seed: int = DEFAULT_BENCHMARK_SEED) -> Any:
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        with _quiet_benchmark_context():
            yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def _try_import_autogluon() -> tuple[Any, Any]:
    from autogluon.core.models import AbstractModel
    from autogluon.features.generators import LabelEncoderFeatureGenerator
    from autogluon.tabular import TabularPredictor

    return TabularPredictor, (AbstractModel, LabelEncoderFeatureGenerator)


try:
    _, (_AUTOGLUON_ABSTRACT_MODEL, _AUTOGLUON_LABEL_ENCODER_GENERATOR) = _try_import_autogluon()
except ModuleNotFoundError:
    _AUTOGLUON_ABSTRACT_MODEL = None
    _AUTOGLUON_LABEL_ENCODER_GENERATOR = None


if _AUTOGLUON_ABSTRACT_MODEL is not None:

    class CustomTabPFNModel(_AUTOGLUON_ABSTRACT_MODEL):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._feature_generator = None

        def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
            X = super()._preprocess(X, **kwargs)

            if is_train:
                self._feature_generator = _AUTOGLUON_LABEL_ENCODER_GENERATOR(verbosity=0)
                self._feature_generator.fit(X=X)
            if self._feature_generator.features_in:
                X = X.copy()
                X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
            return X.fillna(0).to_numpy(dtype=np.float32)

        def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
            if X.shape[0] > 10000:
                X = X.sample(n=10000, random_state=42)
                y = y.loc[X.index]
            with _seeded_benchmark_context():
                from tabpfn import TabPFNClassifier, TabPFNRegressor

                if self.problem_type in ["regression", "softclass"]:
                    model_cls = TabPFNRegressor
                else:
                    model_cls = TabPFNClassifier
                    if len(y.unique()) > 10:
                        raise ValueError("TabPFN only supports up to 10 classes.")

                X = self.preprocess(X, is_train=True)
                params = self._get_model_params()
                self.model = model_cls(**params)
                self.model.fit(X, y)

        def _set_default_params(self) -> None:
            self._set_default_param_value("n_estimators", 1)

        def _get_default_auxiliary_params(self) -> dict[str, Any]:
            default_auxiliary_params = super()._get_default_auxiliary_params()
            default_auxiliary_params.update(
                {
                    "valid_raw_types": ["int", "float", "category"],
                }
            )
            return default_auxiliary_params

else:
    CustomTabPFNModel = None


def _write_summary(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _build_column_groups(info: dict[str, Any], columns: list[str]) -> tuple[list[str], list[str]]:
    numerical = [columns[idx] for idx in info["num_col_idx"]]
    categorical = [columns[idx] for idx in info["cat_col_idx"]]
    target = [columns[idx] for idx in info["target_col_idx"]]

    if info["task_type"] == "regression":
        numerical = numerical + target
    else:
        categorical = categorical + target

    return numerical, categorical


def _prepare_original_tables(
    real_train_df: pd.DataFrame,
    eval_real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    columns = list(real_train_df.columns)
    numerical_cols, categorical_cols = _build_column_groups(info, columns)

    real_train_df = real_train_df.copy()[columns]
    eval_real_df = eval_real_df.copy()[columns]
    synthetic_df = synthetic_df.copy()[columns]

    for frame in (real_train_df, eval_real_df, synthetic_df):
        frame.columns = columns

    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=float,
        )
        real_cat = real_train_df[categorical_cols].fillna("__nan__").astype(str)
        eval_cat = eval_real_df[categorical_cols].fillna("__nan__").astype(str)
        syn_cat = synthetic_df[categorical_cols].fillna("__nan__").astype(str)
        encoder.fit(real_cat)

        real_train_df[categorical_cols] = encoder.transform(real_cat)
        eval_real_df[categorical_cols] = encoder.transform(eval_cat)
        synthetic_df[categorical_cols] = encoder.transform(syn_cat)

    for col in numerical_cols:
        real_train_df[col] = pd.to_numeric(real_train_df[col], errors="coerce")
        eval_real_df[col] = pd.to_numeric(eval_real_df[col], errors="coerce")
        synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors="coerce")

    return real_train_df, eval_real_df, synthetic_df, numerical_cols, categorical_cols


def _get_tabstruct_predictor_bits() -> tuple[Any, Any]:
    try:
        from autogluon.tabular import TabularPredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TabStruct structural-fidelity metrics require `autogluon.tabular` and its "
            "dependencies. Install AutoGluon to compute global utility."
        ) from exc
    return TabularPredictor, CustomTabPFNModel


def _evaluate_tabstruct_utility_per_feature(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    column_list: list[str],
    time_limit_per_feature: int,
) -> dict[str, dict[str, float]]:
    TabularPredictor, CustomTabPFNModel = _get_tabstruct_predictor_bits()
    custom_hyperparameters: dict[Any, dict[str, Any]] = {
        "XGB": {
            "random_state": DEFAULT_BENCHMARK_SEED,
            "seed": DEFAULT_BENCHMARK_SEED,
        },
        "KNN": {},
    }
    if _tabpfn_enabled() and CustomTabPFNModel is not None:
        custom_hyperparameters[CustomTabPFNModel] = {}

    classification_scores: dict[str, float] = {}
    regression_negative_rmse_scores: dict[str, float] = {}
    regression_rmse_scores: dict[str, float] = {}

    for col in column_list:
        if train_df[col].nunique(dropna=False) == 1:
            classification_scores[col] = 1.0
            continue

        with tempfile.TemporaryDirectory(prefix="tabstruct-utility-") as predictor_dir:
            with _seeded_benchmark_context():
                predictor = TabularPredictor(
                    label=col,
                    path=predictor_dir,
                    log_to_file=True,
                    verbosity=0,
                ).fit(
                    train_data=train_df,
                    tuning_data=None,
                    hyperparameters=custom_hyperparameters,
                    fit_weighted_ensemble=False,
                    presets="medium_quality",
                    time_limit=time_limit_per_feature,
                )

                if predictor.problem_type == "regression":
                    leaderboard = predictor.leaderboard(eval_df, extra_metrics=["root_mean_squared_error"])
                    regression_negative_rmse_scores[col] = float(leaderboard["score_test"].mean())
                    if "root_mean_squared_error" in leaderboard.columns:
                        regression_rmse_scores[col] = float(leaderboard["root_mean_squared_error"].mean())
                else:
                    leaderboard = predictor.leaderboard(eval_df, extra_metrics=["balanced_accuracy"])
                    classification_scores[col] = float(leaderboard["balanced_accuracy"].mean())

    return {
        "balanced_accuracy": classification_scores,
        "negative_RMSE": regression_negative_rmse_scores,
        "RMSE": regression_rmse_scores,
    }


def compute_tabstruct_structural_fidelity(
    real_train_df: pd.DataFrame,
    eval_real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    info: dict[str, Any],
    total_time_limit_seconds: int = 900,
) -> dict[str, Any]:
    column_list = list(real_train_df.columns)
    try:
        real_train_ord, eval_real_ord, synthetic_ord, numerical_cols, categorical_cols = _prepare_original_tables(
            real_train_df=real_train_df,
            eval_real_df=eval_real_df,
            synthetic_df=synthetic_df,
            info=info,
        )

        time_limit_per_feature = max(1, int(total_time_limit_seconds // max(1, len(column_list))))
        reference_scores = _evaluate_tabstruct_utility_per_feature(
            train_df=real_train_ord,
            eval_df=eval_real_ord,
            column_list=column_list,
            time_limit_per_feature=time_limit_per_feature,
        )
        synthetic_scores = _evaluate_tabstruct_utility_per_feature(
            train_df=synthetic_ord,
            eval_df=eval_real_ord,
            column_list=column_list,
            time_limit_per_feature=time_limit_per_feature,
        )
    except RuntimeError as exc:
        return {
            "global_utility": None,
            "status": "dependency_missing",
            "message": str(exc),
        }

    relative_per_feature: dict[str, float] = {}
    feature_types: dict[str, str] = {}
    target_col = column_list[info["target_col_idx"][0]]
    for col in column_list:
        if col in synthetic_scores["balanced_accuracy"]:
            ref_score = reference_scores["balanced_accuracy"].get(col)
            syn_score = synthetic_scores["balanced_accuracy"].get(col)
            if ref_score not in (None, 0):
                relative_per_feature[col] = float(syn_score / ref_score)
                feature_types[col] = "classification"
        else:
            ref_rmse = reference_scores["RMSE"].get(col)
            syn_rmse = synthetic_scores["RMSE"].get(col)
            if ref_rmse not in (None, 0) and syn_rmse not in (None, 0):
                relative_per_feature[col] = float(ref_rmse / syn_rmse)
                feature_types[col] = "regression"

    global_utility = None
    if relative_per_feature:
        global_utility = float(np.mean(list(relative_per_feature.values())))

    local_utility = relative_per_feature.get(target_col)

    return {
        "global_utility": global_utility,
        "local_utility": local_utility,
        "status": "computed",
        "implementation": {
            "upstream_metric": "tabeval.metrics.eval_structure.UtilityPerFeature",
            "upstream_timestamp": "2025-08-09",
            "predictor_ensemble": "tiny-default-opt-in-tabpfn",
            "predictors": ["XGB", "KNN"] + (["TabPFN"] if _tabpfn_enabled() else []),
            "predictor_policy": {
                "tabpfn_enabled": _tabpfn_enabled(),
                "tabpfn_default": "disabled",
                "tabpfn_enable_env_var": "STANDARDIZED_TABULAR_DIFFUSION_ENABLE_TABPFN",
                "tabpfn_optional_when_unavailable": True,
                "tabpfn_unavailability_reasons": [
                    "unsupported_class_count",
                    "gated_huggingface_model",
                    "missing_dependency",
                ],
                "benchmark_runtime_mode": "quiet_local",
            },
            "autogluon_presets": "medium_quality",
            "fit_weighted_ensemble": False,
            "time_limit_total_seconds": total_time_limit_seconds,
            "time_limit_per_feature_seconds": time_limit_per_feature,
        },
        "utility_per_feature": {
            "reference": reference_scores,
            "synthetic": synthetic_scores,
            "relative": relative_per_feature,
            "feature_types": feature_types,
        },
        "column_groups": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "target": target_col,
        },
    }


def normalize_tabdiff_or_tabsyn_summary(
    repo_root: Path,
    model_name: str,
    dataset: str,
    sample_path: Path,
    output_path: Path,
) -> None:
    tabdiff_root = repo_root / "TabDiff-main"
    if str(tabdiff_root) not in sys.path:
        sys.path.insert(0, str(tabdiff_root))

    from tabdiff.metrics import TabMetrics  # pylint: disable=import-error

    info_path = repo_root / "TabDiff-main" / "data" / dataset / "info.json"
    real_path = repo_root / "TabDiff-main" / "synthetic" / dataset / "real.csv"
    test_path = repo_root / "TabDiff-main" / "synthetic" / dataset / "test.csv"
    val_path = repo_root / "TabDiff-main" / "synthetic" / dataset / "val.csv"
    if not val_path.exists():
        val_path = None

    info = json.loads(info_path.read_text())
    syn_df = pd.read_csv(sample_path)
    raw_metrics: dict[str, Any] = {}
    raw_details: dict[str, Any] = {}
    metric_failures: dict[str, str] = {}

    for metric_name in ["density", "mle", "c2st"]:
        metrics = TabMetrics(
            real_data_path=str(real_path),
            test_data_path=str(test_path),
            val_data_path=None if val_path is None else str(val_path),
            info=info,
            device="cpu",
            metric_list=[metric_name],
        )
        try:
            metric_values, metric_details = metrics.evaluate(syn_df.copy())
            raw_metrics.update(metric_values)
            raw_details.update(metric_details)
        except Exception as exc:  # pylint: disable=broad-except
            metric_failures[metric_name] = f"{type(exc).__name__}: {exc}"

    structural_fidelity = compute_tabstruct_structural_fidelity(
        real_train_df=pd.read_csv(real_path),
        eval_real_df=pd.read_csv(test_path),
        synthetic_df=syn_df,
        info=info,
    )

    payload = {
        "schema_version": "1.0",
        "protocol_name": METRIC_DEFINITIONS["protocol_name"],
        "model": model_name,
        "dataset": dataset,
        "sample_path": str(sample_path),
        "metrics": {
            "density": {
                "shape_score": raw_metrics.get("density/Shape"),
                "trend_score": raw_metrics.get("density/Trend"),
                "overall_score": raw_metrics.get("density/Overall"),
                "status": "computed" if "density" not in metric_failures else "failed",
                "error": metric_failures.get("density"),
            },
            "ml_efficacy": {
                "primary_metric_name": "xgb_rmse" if info["task_type"] == "regression" else "xgb_auroc",
                "primary_metric_value": raw_metrics.get("mle"),
                "task_type": _normalize_task_type(info["task_type"]),
                "details": raw_details.get("mle", {}),
                "status": "computed" if "mle" not in metric_failures else "failed",
                "error": metric_failures.get("mle"),
            },
            "detection": {
                "logistic_detection": raw_metrics.get("c2st"),
                "status": "computed" if "c2st" not in metric_failures else "failed",
                "error": metric_failures.get("c2st"),
            },
            "privacy": {
                "dcr_score": None,
                "status": "not_requested",
            },
            "structural_fidelity": structural_fidelity,
        },
        "evaluation_failures": metric_failures,
        "tabstruct_alignment": {
            "density_fidelity": "exact current implementation",
            "ml_efficacy": "exact current implementation",
            "detection": "exact current implementation",
            "privacy": "available through DCR path but not run in this summary",
            "structural_fidelity": "implemented with TabStruct UtilityPerFeature and per-variable relative utility aggregation",
        },
    }
    _write_summary(payload, output_path)


def normalize_tabddpm_summary(
    dataset: str,
    output_path: Path,
    metrics_paths: dict[str, Path | None],
) -> None:
    def load_json(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text())

    catboost = load_json(metrics_paths.get("catboost"))
    mlp = load_json(metrics_paths.get("mlp"))
    privacy = load_json(metrics_paths.get("privacy"))
    simple = load_json(metrics_paths.get("simple"))

    primary_metric_name = None
    primary_metric_value = None
    task_type = None

    if catboost is not None:
        synthetic_test = catboost.get("synthetic", {}).get("test", {})
        if "roc_auc-mean" in synthetic_test:
            task_type = "classification"
            primary_metric_name = "catboost_auroc"
            primary_metric_value = synthetic_test["roc_auc-mean"]
        elif "rmse-mean" in synthetic_test:
            task_type = "regression"
            primary_metric_name = "catboost_rmse"
            primary_metric_value = synthetic_test["rmse-mean"]

    payload = {
        "schema_version": "1.0",
        "protocol_name": METRIC_DEFINITIONS["protocol_name"],
        "model": "tabddpm",
        "dataset": dataset,
        "metrics": {
            "density": {
                "shape_score": None,
                "trend_score": None,
                "overall_score": None,
                "status": "not_emitted_by_upstream_tabddpm_pipeline",
            },
            "ml_efficacy": {
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "task_type": task_type,
                "details": {
                    "catboost": catboost,
                    "mlp": mlp,
                    "simple": simple,
                },
            },
            "detection": {
                "logistic_detection": None,
                "status": "not_emitted_by_upstream_tabddpm_pipeline",
            },
            "privacy": {
                "dcr_score": None if privacy is None else privacy.get("privacy"),
                "details": privacy,
            },
            "structural_fidelity": {
                "global_utility": None,
                "status": "not_available_without_real_and_synthetic_tables",
            },
        },
        "tabstruct_alignment": {
            "density_fidelity": "missing in current upstream outputs",
            "ml_efficacy": "normalized from upstream evaluator outputs",
            "detection": "missing in current upstream outputs",
            "privacy": "normalized from upstream privacy output when provided",
            "structural_fidelity": "requires table-level artifacts and AutoGluon-based UtilityPerFeature evaluation",
        },
    }
    _write_summary(payload, output_path)
