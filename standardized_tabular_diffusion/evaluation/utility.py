"""P4 Local and Global Utility with explicit held-out-test semantics."""

from __future__ import annotations

import math
import re
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from typing import Any, Callable, NoReturn

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    EvaluationRequest,
    MetricState,
    RawDirection,
    utc_timestamp,
)
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, read_json
from standardized_tabular_diffusion.evaluation.table import ValidatedUtilityTables

LOCAL_METRIC_IDS = {
    "macro-f1": "std-local-utility-macro-f1",
    "balanced-accuracy": "std-local-utility-balanced-accuracy",
    "roc-auc": "std-local-utility-roc-auc",
    "pr-auc": "std-local-utility-pr-auc",
    "rmse": "std-local-utility-rmse",
    "mae": "std-local-utility-mae",
    "r2": "std-local-utility-r2",
}
LOCAL_RETENTION_METRIC_ID = "std-local-utility-retention"
GLOBAL_BALANCED_ACCURACY_METRIC_ID = "tabstruct-global-balanced-accuracy"
GLOBAL_RMSE_METRIC_ID = "tabstruct-global-rmse"
GLOBAL_TARGET_RATIO_METRIC_ID = "tabstruct-global-target-utility"
P4_METRIC_VERSION = "1.0.0"
P4_METRICS = tuple(
    {"metric_id": metric_id, "metric_version": P4_METRIC_VERSION}
    for metric_id in (
        *LOCAL_METRIC_IDS.values(),
        LOCAL_RETENTION_METRIC_ID,
        GLOBAL_BALANCED_ACCURACY_METRIC_ID,
        GLOBAL_RMSE_METRIC_ID,
        GLOBAL_TARGET_RATIO_METRIC_ID,
    )
)
UTILITY_DETAILS_ARTIFACT_PATH = "artifacts/utility-details.json"
UTILITY_IMPLEMENTATION_VERSION = "1.0.0"
EVALUATOR_RESOURCE_PACKAGE = "standardized_tabular_diffusion.resources.evaluation.evaluators"
EVALUATOR_RESOURCE_NAME = "p4-utility-pilot-v1.json"

_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_NUMERIC_TYPES = {"continuous", "integer"}
_CLASSIFICATION_TYPES = {"categorical", "boolean"}


class UtilityError(RuntimeError):
    """Base class for P4 contract and execution failures."""


class UtilityProfileError(UtilityError):
    """Raised when a dataset or evaluator lacks a complete P4 declaration."""


class UtilityResourceError(UtilityError):
    """Raised when an authoritative optional predictor backend cannot run."""


class UtilityImplementationError(UtilityError):
    """Raised when a predictor backend violates its declared result contract."""


@dataclass(frozen=True)
class GlobalBackendResult:
    score: float
    predictors: tuple[str, ...]
    predictor_scores: dict[str, float]


GlobalScorer = Callable[
    [pd.DataFrame, pd.DataFrame, str, str, int, int, str],
    GlobalBackendResult,
]


@dataclass(frozen=True)
class UtilityOutcome:
    atomic_results: tuple[AtomicResult, ...]
    local_summary: dict[str, Any]
    global_summary: dict[str, Any]
    denominator_counts: dict[str, int]
    details: dict[str, Any]
    source: dict[str, Any]


@dataclass(frozen=True)
class _ArmResult:
    state: MetricState
    scores: dict[str, float | None]
    undefined: dict[str, str]
    warnings: tuple[str, ...]
    reason_code: str | None = None
    reason_detail: str | None = None


def load_p4_evaluator_profile() -> dict[str, Any]:
    item = resources.files(EVALUATOR_RESOURCE_PACKAGE).joinpath(EVALUATOR_RESOURCE_NAME)
    with resources.as_file(item) as path:
        payload = read_json(path)
    validate_evaluator_profile(payload)
    return payload


def p4_evaluator_profile_reference() -> dict[str, str]:
    payload = load_p4_evaluator_profile()
    return {
        "profile_id": payload["profile_id"],
        "profile_version": payload["profile_version"],
        "sha256": content_fingerprint(payload),
    }


def _fail(detail: str) -> NoReturn:
    raise UtilityProfileError(detail)


def _require_exact(name: str, payload: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != fields:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        _fail(f"{name} fields differ from the P4 contract; expected={sorted(fields)}, observed={observed}")
    return payload


def _require_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{name} must be a portable lowercase identifier")
    return value


def validate_evaluator_profile(profile: dict[str, Any]) -> None:
    required = {
        "profile_schema_version",
        "profile_id",
        "profile_version",
        "status",
        "official_results_allowed",
        "default_evaluator_seeds",
        "local",
        "global",
    }
    _require_exact("P4 Evaluator Profile", profile, required)
    if profile["profile_schema_version"] != "1.0.0":
        _fail("Unsupported P4 Evaluator Profile schema version")
    _require_id("profile_id", profile["profile_id"])
    _require_id("profile_version", profile["profile_version"])
    if (
        profile["profile_id"] != "p4-utility-pilot"
        or profile["profile_version"] != "0.1.0"
        or profile["status"] != "unit-validated-diagnostic"
        or profile["official_results_allowed"] is not False
    ):
        _fail("P4 pilot identity, lifecycle, or diagnostic admission boundary has drifted")
    seeds = profile["default_evaluator_seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        _fail("default_evaluator_seeds must be a non-empty unique integer array")
    if seeds != [0, 1, 2, 3, 4]:
        _fail("P4 pilot default evaluator seeds must remain exactly 0 through 4")

    local = _require_exact(
        "P4 Evaluator Profile local",
        profile["local"],
        {
            "profile_id",
            "profile_version",
            "fit_boundary",
            "categorical_encoding",
            "numeric_scaling",
            "classification",
            "regression",
            "retention",
        },
    )
    if local["profile_id"] != "std-local-three-family" or local["profile_version"] != "0.1.0":
        _fail("P4 pilot requires std-local-three-family@0.1.0")
    if not isinstance(local["fit_boundary"], str) or not local["fit_boundary"].strip():
        _fail("P4 Local Utility requires a non-empty fit-boundary declaration")
    if local["categorical_encoding"] != {
        "implementation": "sklearn.preprocessing.OneHotEncoder",
        "handle_unknown": "ignore",
        "sparse_output": False,
        "maximum_real_train_cardinality": 256,
    }:
        _fail("P4 Local categorical encoding differs from the frozen train-fitted contract")
    if local["numeric_scaling"] != {
        "implementation": "sklearn.preprocessing.StandardScaler",
        "with_mean": True,
        "with_std": True,
    }:
        _fail("P4 Local numeric scaling differs from the frozen train-fitted contract")

    expected_tasks = {
        "classification": {
            "primary_metric": "macro-f1",
            "secondary_metrics": ["balanced-accuracy", "roc-auc", "pr-auc"],
            "dummy": {"implementation": "sklearn.dummy.DummyClassifier", "strategy": "most_frequent"},
            "evaluators": [
                {
                    "evaluator_id": "logistic-regression",
                    "implementation": "sklearn.linear_model.LogisticRegression",
                    "parameters": {
                        "C": 1.0,
                        "solver": "lbfgs",
                        "max_iter": 1000,
                        "class_weight": None,
                    },
                },
                {
                    "evaluator_id": "random-forest",
                    "implementation": "sklearn.ensemble.RandomForestClassifier",
                    "parameters": {
                        "n_estimators": 200,
                        "max_depth": None,
                        "min_samples_leaf": 1,
                        "class_weight": None,
                        "n_jobs": 1,
                    },
                },
                {
                    "evaluator_id": "hist-gradient-boosting",
                    "implementation": "sklearn.ensemble.HistGradientBoostingClassifier",
                    "parameters": {
                        "learning_rate": 0.1,
                        "max_iter": 100,
                        "max_leaf_nodes": 31,
                        "l2_regularization": 0.0,
                    },
                },
            ],
        },
        "regression": {
            "primary_metric": "rmse",
            "secondary_metrics": ["mae", "r2"],
            "dummy": {"implementation": "sklearn.dummy.DummyRegressor", "strategy": "mean"},
            "evaluators": [
                {
                    "evaluator_id": "ridge",
                    "implementation": "sklearn.linear_model.Ridge",
                    "parameters": {"alpha": 1.0},
                },
                {
                    "evaluator_id": "random-forest",
                    "implementation": "sklearn.ensemble.RandomForestRegressor",
                    "parameters": {
                        "n_estimators": 200,
                        "max_depth": None,
                        "min_samples_leaf": 1,
                        "n_jobs": 1,
                    },
                },
                {
                    "evaluator_id": "hist-gradient-boosting",
                    "implementation": "sklearn.ensemble.HistGradientBoostingRegressor",
                    "parameters": {
                        "learning_rate": 0.1,
                        "max_iter": 100,
                        "max_leaf_nodes": 31,
                        "l2_regularization": 0.0,
                    },
                },
            ],
        },
    }
    for task_type, expected in expected_tasks.items():
        if local[task_type] != expected:
            _fail(f"Local {task_type} evaluator panel differs from the frozen P4 contract")
    if local["retention"] != {
        "metric_version": "1.0.0",
        "denominator_absolute_tolerance": 1e-12,
        "clipping": "none",
    }:
        _fail("P4 Local retention definition differs from the frozen unclipped contract")

    global_profile = _require_exact(
        "P4 Evaluator Profile global",
        profile["global"],
        {
            "profile_id",
            "profile_version",
            "formula_source",
            "implementation_source",
            "predictors",
            "autogluon_presets",
            "fit_weighted_ensemble",
            "default_time_limit_per_target_seconds",
            "classification_metric",
            "regression_metric",
            "target_weighting",
            "seed_weighting",
            "ratio_clipping",
            "constant_synthetic_target_policy",
            "dependency_failure_policy",
            "source_parity_claimed",
            "known_deviations",
        },
    )
    expected_source = {
        "repository": "https://github.com/SilenceX12138/TabEval.git",
        "revision": "dba19a4ee7aa391621cbeb464609285fd515dece",
        "symbol": "tabeval.metrics.eval_structure.UtilityPerFeature",
        "source_sha256": "edd2ab2ad576e5ef46c55bac01ac3366d8eec91b40398da48bf2e1c061e2d90c",
    }
    expected_global_values = {
        "profile_id": "tabeval-tiny-default",
        "profile_version": "2025-08-09-pinned",
        "formula_source": "TabStruct Equation 4",
        "implementation_source": expected_source,
        "predictors": ["xgb", "knn", "tabpfn"],
        "autogluon_presets": "medium_quality",
        "fit_weighted_ensemble": False,
        "default_time_limit_per_target_seconds": 300,
        "classification_metric": "balanced-accuracy",
        "regression_metric": "rmse",
        "target_weighting": "equal",
        "seed_weighting": "equal",
        "ratio_clipping": "none",
        "constant_synthetic_target_policy": "insufficient_support",
        "dependency_failure_policy": "resource_failure-no-substitution",
        "source_parity_claimed": False,
    }
    if any(global_profile[key] != value for key, value in expected_global_values.items()):
        _fail("P4 Global Utility source, predictors, formulas, or failure policy have drifted")
    deviations = global_profile["known_deviations"]
    if not isinstance(deviations, list) or not deviations or any(
        not isinstance(item, str) or not item.strip() for item in deviations
    ):
        _fail("P4 Global Utility known deviations must remain explicit and non-empty")


def validate_utility_profile(dataset_profile: dict[str, Any], evaluator_profile: dict[str, Any]) -> None:
    validate_evaluator_profile(evaluator_profile)
    utility = _require_exact(
        "Dataset Profile utility",
        dataset_profile.get("utility"),
        {"contract_schema_version", "status", "local", "global"},
    )
    if utility["contract_schema_version"] != "1.0.0" or utility["status"] != "reviewed-diagnostic":
        _fail("Dataset utility must be reviewed-diagnostic under contract 1.0.0")
    expected_profile = (evaluator_profile["profile_id"], evaluator_profile["profile_version"])
    local = _require_exact(
        "Dataset Profile utility.local",
        utility["local"],
        {"primary_task_id", "evaluator_profile_id", "evaluator_profile_version"},
    )
    global_block = _require_exact(
        "Dataset Profile utility.global",
        utility["global"],
        {
            "included_target_column_ids",
            "excluded_targets",
            "evaluator_profile_id",
            "evaluator_profile_version",
        },
    )
    for block_name, block in (("local", local), ("global", global_block)):
        if (block["evaluator_profile_id"], block["evaluator_profile_version"]) != expected_profile:
            _fail(f"utility.{block_name} evaluator identity differs from the immutable request profile")

    columns = dataset_profile.get("columns")
    canonical_names = dataset_profile.get("table_contract", {}).get("canonical_column_order")
    if not isinstance(columns, list) or not isinstance(canonical_names, list):
        _fail("Dataset Profile lacks canonical columns")
    by_id = {column.get("column_id"): column for column in columns if isinstance(column, dict)}
    by_name = {column.get("name"): column for column in columns if isinstance(column, dict)}
    canonical = [by_name.get(name) for name in canonical_names]
    if any(column is None for column in canonical):
        _fail("Dataset Profile canonical model view references unknown columns")

    tasks = dataset_profile.get("predictive_tasks")
    if not isinstance(tasks, list):
        _fail("Dataset Profile predictive_tasks must be an array")
    task = next((item for item in tasks if item.get("task_id") == local["primary_task_id"]), None)
    required_task = {
        "task_id",
        "target_column_id",
        "task_type",
        "status",
        "positive_class",
        "label_mapping",
        "primary_metric",
        "secondary_metrics",
        "minimum_support",
        "dummy_strategy",
        "retention_metric_id",
        "retention_metric_version",
    }
    task = _require_exact("Primary Local Utility task", task, required_task)
    if task["status"] != "reviewed" or task["target_column_id"] not in by_id:
        _fail("Primary Local Utility task must be reviewed and reference a known target")
    target = by_id[task["target_column_id"]]
    if "primary_target" not in target.get("roles", ()):
        _fail("Primary Local Utility task must reference the declared primary target")
    expected_task_type = _task_type(target)
    if task["task_type"] != expected_task_type:
        _fail("Primary Local Utility task type differs from target semantics")
    expected_primary = "macro-f1" if expected_task_type == "classification" else "rmse"
    if task["primary_metric"] != expected_primary:
        _fail(f"Primary Local Utility metric must be {expected_primary}")
    if task["retention_metric_id"] != LOCAL_RETENTION_METRIC_ID or task["retention_metric_version"] != "1.0.0":
        _fail("Primary task must identify the benchmark-derived Local Utility retention")
    if expected_task_type == "classification" and task["positive_class"] not in target["valid_domain"]["values"]:
        _fail("Classification positive_class must belong to the reviewed target domain")
    if expected_task_type == "regression" and task["positive_class"] is not None:
        _fail("Regression tasks cannot declare a positive class")

    included = global_block["included_target_column_ids"]
    excluded = global_block["excluded_targets"]
    if not isinstance(included, list) or len(included) != len(set(included)):
        _fail("Global Utility included targets must be a unique array")
    if not isinstance(excluded, list) or any(
        not isinstance(item, dict) or set(item) != {"column_id", "reason_code", "reason_detail"}
        for item in excluded
    ):
        _fail("Every Global Utility exclusion requires column_id, reason_code, and reason_detail")
    excluded_ids = [item["column_id"] for item in excluded]
    if len(excluded_ids) != len(set(excluded_ids)) or set(included) & set(excluded_ids):
        _fail("Global Utility target inclusions and exclusions must be unique and disjoint")
    canonical_ids = {column["column_id"] for column in canonical if column is not None}
    if set(included) | set(excluded_ids) != canonical_ids:
        _fail("Every canonical model-view column must be explicitly included or excluded from Global Utility")
    for column_id in included:
        column = by_id.get(column_id)
        if column is None or set(column.get("roles", ())) & {"identifier", "ignored", "audit_only"}:
            _fail(f"Global Utility target {column_id!r} is absent or has an excluded role")
        _task_type(column)
    for item in excluded:
        _require_id("Global Utility exclusion reason_code", item["reason_code"])
        if not isinstance(item["reason_detail"], str) or not item["reason_detail"].strip():
            _fail("Global Utility exclusion reason_detail must be non-empty")


def _task_type(column: dict[str, Any]) -> str:
    semantic_type = column.get("semantic_type")
    if semantic_type in _NUMERIC_TYPES:
        return "regression"
    if semantic_type in _CLASSIFICATION_TYPES:
        return "classification"
    _fail(
        f"Column {column.get('column_id')!r} with semantic type {semantic_type!r} requires an explicit "
        "Global Utility exclusion until datetime/string target handling is frozen"
    )


def _seed_id(seed: int) -> str:
    return f"neg-{abs(seed)}" if seed < 0 else str(seed)


def _atomic(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    *,
    run_id: str,
    metric_id: str,
    dimension: str,
    scope_id: str,
    evaluator_id: str | None,
    task_type: str,
    state: MetricState,
    raw_direction: RawDirection,
    n_reference: int,
    n_synthetic: int,
    raw_value: float | None = None,
    normalized_value: float | None = None,
    aggregate_contribution: float | None = None,
    reference_value: float | None = None,
    weight: float = 0.0,
    unit: str | None = None,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    warning_codes: tuple[str, ...] = (),
) -> AtomicResult:
    computed = state is MetricState.COMPUTED
    return AtomicResult(
        run_id=run_id,
        protocol_version=request.protocol["protocol_version"],
        dataset_id=dataset_profile["dataset_id"],
        dataset_version=dataset_profile["dataset_version"],
        dataset_view=dataset_profile["dataset_view"],
        split_id=dataset_profile["split"]["split_id"],
        model_id=(request.model or {}).get("model_id", "external"),
        comparison_track=request.comparison_track,
        generation_seed=request.generation_seed,
        metric_id=metric_id,
        metric_version=P4_METRIC_VERSION,
        dimension=dimension,
        scope_type="target",
        scope_id=scope_id,
        state=state,
        raw_direction=raw_direction,
        weight=weight,
        n_reference=n_reference,
        n_synthetic=n_synthetic,
        n_valid=n_reference if computed else 0,
        n_excluded=0 if computed else n_reference,
        computed_at=utc_timestamp(),
        raw_value=raw_value,
        normalized_value=normalized_value,
        aggregate_contribution=aggregate_contribution,
        reference_value=reference_value,
        unit=unit,
        evaluator_id=evaluator_id,
        evaluator_version=(
            "2025-08-09-pinned"
            if evaluator_id == "tabeval-tiny-default"
            else ("0.1.0" if evaluator_id is not None else None)
        ),
        task_type=task_type,
        reason_code=reason_code,
        reason_detail=reason_detail,
        warning_codes=tuple(sorted(set(warning_codes))),
        artifact_ref=UTILITY_DETAILS_ARTIFACT_PATH,
    )


def local_retention(
    dummy: float,
    trtr: float,
    tstr: float,
    *,
    higher_is_better: bool,
    tolerance: float,
) -> float:
    """Return the unclipped benchmark-derived retention or raise if undefined."""

    values = (dummy, trtr, tstr, tolerance)
    if any(not math.isfinite(float(value)) for value in values) or tolerance < 0:
        raise ValueError("Retention inputs and tolerance must be finite, with a non-negative tolerance")
    denominator = (trtr - dummy) if higher_is_better else (dummy - trtr)
    if denominator <= tolerance:
        raise ZeroDivisionError("TRTR does not improve on Dummy beyond the frozen tolerance")
    numerator = (tstr - dummy) if higher_is_better else (dummy - tstr)
    return float(numerator / denominator)


def global_target_ratio(reference: float, synthetic: float, *, task_type: str) -> float:
    """Implement TabStruct Equation 4 without clipping."""

    if not all(math.isfinite(float(value)) for value in (reference, synthetic)):
        raise ValueError("Global Utility raw arm values must be finite")
    denominator = reference if task_type == "classification" else synthetic
    if denominator == 0:
        raise ZeroDivisionError("Global Utility denominator is zero")
    if task_type == "classification":
        return float(synthetic / reference)
    if task_type == "regression":
        return float(reference / synthetic)
    raise ValueError(f"Unsupported Global Utility task type: {task_type}")


def _feature_frames(
    tables: ValidatedUtilityTables,
    target_name: str,
    local_profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    specs = [spec for spec in tables.column_specs if spec["name"] != target_name]
    if not specs:
        raise UtilityProfileError("Local Utility requires at least one predictor feature")
    numeric = [spec["name"] for spec in specs if spec["semantic_type"] in _NUMERIC_TYPES]
    datetimes = [spec["name"] for spec in specs if spec["semantic_type"] == "datetime"]
    categorical = [spec["name"] for spec in specs if spec["semantic_type"] in _CLASSIFICATION_TYPES | {"string"}]
    prepared: list[pd.DataFrame] = []
    for source in (tables.real_train, tables.synthetic, tables.real_test):
        frame = source.drop(columns=[target_name]).copy()
        for name in datetimes:
            frame[name] = pd.to_datetime(frame[name], errors="raise").astype("int64").astype("float64")
        for name in categorical:
            frame[name] = frame[name].astype("string")
        prepared.append(frame)
    numeric = [*numeric, *datetimes]
    maximum_cardinality = int(local_profile["categorical_encoding"]["maximum_real_train_cardinality"])
    too_wide = {
        name: int(prepared[0][name].nunique(dropna=False))
        for name in categorical
        if prepared[0][name].nunique(dropna=False) > maximum_cardinality
    }
    if too_wide:
        raise UtilityProfileError(f"Local Utility categorical cardinality exceeds the frozen limit: {too_wide}")
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                categorical,
            )
        )
    transformer = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    real_x = transformer.fit_transform(prepared[0])
    synthetic_x = transformer.transform(prepared[1])
    test_x = transformer.transform(prepared[2])
    return np.asarray(real_x), np.asarray(synthetic_x), np.asarray(test_x)


def _classification_scores(model: Any, test_x: np.ndarray, y_test: pd.Series, labels: list[Any], positive: Any) -> _ArmResult:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    predictions = model.predict(test_x)
    scores: dict[str, float | None] = {
        "macro-f1": float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "balanced-accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc-auc": None,
        "pr-auc": None,
    }
    undefined: dict[str, str] = {}
    observed_test = set(pd.unique(y_test))
    if len(observed_test) < 2:
        undefined.update({"roc-auc": "real test contains one class", "pr-auc": "real test contains one class"})
        return _ArmResult(MetricState.COMPUTED, scores, undefined, ())
    probabilities = np.asarray(model.predict_proba(test_x), dtype=float)
    model_classes = list(model.classes_)
    missing = [label for label in labels if label not in model_classes]
    if missing:
        undefined.update(
            {
                "roc-auc": f"predictor probability output omits labels {missing!r}",
                "pr-auc": f"predictor probability output omits labels {missing!r}",
            }
        )
        return _ArmResult(MetricState.COMPUTED, scores, undefined, ())
    order = [model_classes.index(label) for label in labels]
    probabilities = probabilities[:, order]
    if len(labels) == 2:
        if positive not in labels:
            undefined.update({"roc-auc": "positive class is absent", "pr-auc": "positive class is absent"})
        else:
            positive_index = labels.index(positive)
            truth = np.asarray(y_test == positive, dtype=int)
            scores["roc-auc"] = float(roc_auc_score(truth, probabilities[:, positive_index]))
            scores["pr-auc"] = float(average_precision_score(truth, probabilities[:, positive_index]))
    else:
        truth = label_binarize(y_test, classes=labels)
        scores["roc-auc"] = float(
            roc_auc_score(y_test, probabilities, labels=labels, multi_class="ovr", average="macro")
        )
        scores["pr-auc"] = float(average_precision_score(truth, probabilities, average="macro"))
    return _ArmResult(MetricState.COMPUTED, scores, undefined, ())


def _regression_scores(model: Any, test_x: np.ndarray, y_test: pd.Series) -> _ArmResult:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    predictions = np.asarray(model.predict(test_x), dtype=float)
    scores: dict[str, float | None] = {
        "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": None,
    }
    undefined: dict[str, str] = {}
    if len(y_test) < 2:
        undefined["r2"] = "R-squared requires at least two real test rows"
    else:
        value = float(r2_score(y_test, predictions))
        if math.isfinite(value):
            scores["r2"] = value
        else:
            undefined["r2"] = "R-squared is non-finite for a constant real test target"
    return _ArmResult(MetricState.COMPUTED, scores, undefined, ())


def _build_estimator(definition: dict[str, Any], task_type: str, seed: int) -> Any:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge

    implementations = {
        "sklearn.linear_model.LogisticRegression": LogisticRegression,
        "sklearn.linear_model.Ridge": Ridge,
        "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
        "sklearn.ensemble.RandomForestRegressor": RandomForestRegressor,
        "sklearn.ensemble.HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "sklearn.ensemble.HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    }
    implementation = definition["implementation"]
    if implementation not in implementations:
        raise UtilityImplementationError(f"Unapproved Local Utility estimator: {implementation}")
    parameters = dict(definition["parameters"])
    if implementation != "sklearn.linear_model.Ridge":
        parameters["random_state"] = seed
    model = implementations[implementation](**parameters)
    expected_suffix = "Classifier" if task_type == "classification" else "Regressor"
    if expected_suffix not in implementation and not (
        task_type == "classification" and implementation.endswith("LogisticRegression")
    ) and not (task_type == "regression" and implementation.endswith("Ridge")):
        raise UtilityImplementationError("Local evaluator implementation differs from the declared task type")
    return model


def _fit_arm(
    model: Any,
    train_x: np.ndarray,
    y_train: pd.Series,
    test_x: np.ndarray,
    y_test: pd.Series,
    *,
    task_type: str,
    labels: list[Any] | None,
    positive: Any,
) -> _ArmResult:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train_x, y_train)
        warning_codes = tuple(
            sorted(
                {
                    "predictor-convergence-warning"
                    for item in caught
                    if "conver" in item.category.__name__.lower() or "conver" in str(item.message).lower()
                }
            )
        )
        result = (
            _classification_scores(model, test_x, y_test, labels or [], positive)
            if task_type == "classification"
            else _regression_scores(model, test_x, y_test)
        )
        return _ArmResult(result.state, result.scores, result.undefined, warning_codes)
    except MemoryError as exc:
        return _ArmResult(
            MetricState.RESOURCE_FAILURE,
            {},
            {},
            (),
            "predictor_memory_exhausted",
            f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # predictor libraries expose heterogeneous failures
        return _ArmResult(
            MetricState.IMPLEMENTATION_FAILURE,
            {},
            {},
            (),
            "predictor_execution_failure",
            f"{type(exc).__name__}: {exc}",
        )


def _dummy(task_type: str) -> Any:
    if task_type == "classification":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent")
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(strategy="mean")


def _metric_direction(metric_name: str) -> RawDirection:
    return RawDirection.MINIMIZE if metric_name in {"rmse", "mae"} else RawDirection.MAXIMIZE


def _metric_unit(metric_name: str) -> str:
    return "target-unit" if metric_name in {"rmse", "mae"} else "score"


def _raw_atom(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    *,
    run_id: str,
    metric_name: str,
    scope_id: str,
    evaluator_id: str,
    task_type: str,
    arm_result: _ArmResult,
    n_test: int,
    n_synthetic: int,
) -> AtomicResult:
    value = arm_result.scores.get(metric_name)
    if arm_result.state is not MetricState.COMPUTED:
        return _atomic(
            request,
            dataset_profile,
            run_id=run_id,
            metric_id=LOCAL_METRIC_IDS[metric_name],
            dimension="local-utility",
            scope_id=scope_id,
            evaluator_id=evaluator_id,
            task_type=task_type,
            state=arm_result.state,
            raw_direction=_metric_direction(metric_name),
            n_reference=n_test,
            n_synthetic=n_synthetic,
            unit=_metric_unit(metric_name),
            reason_code=arm_result.reason_code,
            reason_detail=arm_result.reason_detail,
            warning_codes=arm_result.warnings,
        )
    if value is None:
        return _atomic(
            request,
            dataset_profile,
            run_id=run_id,
            metric_id=LOCAL_METRIC_IDS[metric_name],
            dimension="local-utility",
            scope_id=scope_id,
            evaluator_id=evaluator_id,
            task_type=task_type,
            state=MetricState.MATHEMATICALLY_UNDEFINED,
            raw_direction=_metric_direction(metric_name),
            n_reference=n_test,
            n_synthetic=n_synthetic,
            unit=_metric_unit(metric_name),
            reason_code="metric_undefined",
            reason_detail=arm_result.undefined.get(metric_name, "Metric is mathematically undefined"),
            warning_codes=arm_result.warnings,
        )
    return _atomic(
        request,
        dataset_profile,
        run_id=run_id,
        metric_id=LOCAL_METRIC_IDS[metric_name],
        dimension="local-utility",
        scope_id=scope_id,
        evaluator_id=evaluator_id,
        task_type=task_type,
        state=MetricState.COMPUTED,
        raw_direction=_metric_direction(metric_name),
        n_reference=n_test,
        n_synthetic=n_synthetic,
        raw_value=value,
        unit=_metric_unit(metric_name),
        warning_codes=arm_result.warnings,
    )


def _evaluate_local(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    tables: ValidatedUtilityTables,
    evaluator_profile: dict[str, Any],
    *,
    run_id: str,
) -> tuple[list[AtomicResult], dict[str, Any], list[dict[str, Any]]]:
    if request.real_test_artifact is None:
        raise UtilityProfileError("P4 Local Utility requires an immutable real_test_artifact")
    test_fingerprint = request.real_test_artifact["sha256"]
    utility = dataset_profile["utility"]
    task_id = utility["local"]["primary_task_id"]
    task = next(item for item in dataset_profile["predictive_tasks"] if item["task_id"] == task_id)
    by_id = {column["column_id"]: column for column in tables.column_specs}
    target = by_id[task["target_column_id"]]
    target_name = target["name"]
    task_type = task["task_type"]
    local_profile = evaluator_profile["local"]
    task_profile = local_profile[task_type]
    metrics = [task_profile["primary_metric"], *task_profile["secondary_metrics"]]
    real_x, synthetic_x, test_x = _feature_frames(tables, target_name, local_profile)
    real_y = tables.real_train[target_name]
    synthetic_y = tables.synthetic[target_name]
    test_y = tables.real_test[target_name]
    labels = sorted(pd.unique(pd.concat([real_y, test_y], ignore_index=True)).tolist(), key=repr) if task_type == "classification" else None
    real_classes = set(pd.unique(real_y)) if task_type == "classification" else set()
    test_only = set(pd.unique(test_y)) - real_classes if task_type == "classification" else set()
    missing_synthetic = real_classes - set(pd.unique(synthetic_y)) if task_type == "classification" else set()
    atoms: list[AtomicResult] = []
    runs: list[dict[str, Any]] = []
    primary_name = task_profile["primary_metric"]
    primary_values: dict[str, list[float]] = {"dummy": [], "trtr": [], "tstr": []}
    retention_values: list[float] = []
    expected_retention = len(request.evaluator_seeds) * len(task_profile["evaluators"])
    tolerance = float(local_profile["retention"]["denominator_absolute_tolerance"])

    for seed in request.evaluator_seeds:
        for definition in task_profile["evaluators"]:
            evaluator_id = definition["evaluator_id"]
            dummy_result = _fit_arm(
                _dummy(task_type),
                real_x,
                real_y,
                test_x,
                test_y,
                task_type=task_type,
                labels=labels,
                positive=task["positive_class"],
            )
            trtr_result = _fit_arm(
                _build_estimator(definition, task_type, seed),
                real_x,
                real_y,
                test_x,
                test_y,
                task_type=task_type,
                labels=labels,
                positive=task["positive_class"],
            )
            if test_only:
                tstr_result = _ArmResult(
                    MetricState.INSUFFICIENT_SUPPORT,
                    {},
                    {},
                    (),
                    "real_train_target_support_incomplete",
                    f"Real test contains labels absent from real train: {sorted(test_only, key=repr)!r}",
                )
            elif missing_synthetic:
                tstr_result = _ArmResult(
                    MetricState.INSUFFICIENT_SUPPORT,
                    {},
                    {},
                    (),
                    "insufficient_target_support",
                    f"Synthetic target omits real-train labels: {sorted(missing_synthetic, key=repr)!r}",
                )
            else:
                tstr_result = _fit_arm(
                    _build_estimator(definition, task_type, seed),
                    synthetic_x,
                    synthetic_y,
                    test_x,
                    test_y,
                    task_type=task_type,
                    labels=labels,
                    positive=task["positive_class"],
                )
            arm_results = {"dummy": dummy_result, "trtr": trtr_result, "tstr": tstr_result}
            primary_for_run: dict[str, float | None] = {}
            for arm, result in arm_results.items():
                scope = f"{task_id}--{evaluator_id}--seed-{_seed_id(seed)}--{arm}"
                for metric_name in metrics:
                    atom = _raw_atom(
                        request,
                        dataset_profile,
                        run_id=run_id,
                        metric_name=metric_name,
                        scope_id=scope,
                        evaluator_id=evaluator_id,
                        task_type=task_type,
                        arm_result=result,
                        n_test=len(test_y),
                        n_synthetic=len(synthetic_y),
                    )
                    atoms.append(atom)
                    if metric_name == primary_name and atom.state is MetricState.COMPUTED:
                        if atom.raw_value is None:
                            raise UtilityImplementationError("A computed Local Utility atom lacks a raw value")
                        primary_values[arm].append(atom.raw_value)
                        primary_for_run[arm] = atom.raw_value
                primary_for_run.setdefault(arm, None)

            retention_scope = f"{task_id}--{evaluator_id}--seed-{_seed_id(seed)}"
            dummy_value = primary_for_run.get("dummy")
            trtr_value = primary_for_run.get("trtr")
            tstr_value = primary_for_run.get("tstr")
            if dummy_value is not None and trtr_value is not None and tstr_value is not None:
                try:
                    retention = local_retention(
                        dummy_value,
                        trtr_value,
                        tstr_value,
                        higher_is_better=task_type == "classification",
                        tolerance=tolerance,
                    )
                except ZeroDivisionError as exc:
                    retention_atom = _atomic(
                        request,
                        dataset_profile,
                        run_id=run_id,
                        metric_id=LOCAL_RETENTION_METRIC_ID,
                        dimension="local-utility",
                        scope_id=retention_scope,
                        evaluator_id=evaluator_id,
                        task_type=task_type,
                        state=MetricState.MATHEMATICALLY_UNDEFINED,
                        raw_direction=RawDirection.MAXIMIZE,
                        n_reference=len(test_y),
                        n_synthetic=len(synthetic_y),
                        unit="ratio",
                        reason_code="weak_trtr_baseline",
                        reason_detail=str(exc),
                    )
                else:
                    retention_values.append(retention)
                    retention_atom = _atomic(
                        request,
                        dataset_profile,
                        run_id=run_id,
                        metric_id=LOCAL_RETENTION_METRIC_ID,
                        dimension="local-utility",
                        scope_id=retention_scope,
                        evaluator_id=evaluator_id,
                        task_type=task_type,
                        state=MetricState.COMPUTED,
                        raw_direction=RawDirection.MAXIMIZE,
                        n_reference=len(test_y),
                        n_synthetic=len(synthetic_y),
                        raw_value=retention,
                        normalized_value=retention,
                        aggregate_contribution=retention / expected_retention,
                        weight=1.0 / expected_retention,
                        unit="ratio",
                    )
            else:
                source_states = [result.state for result in arm_results.values() if result.state is not MetricState.COMPUTED]
                state = source_states[0] if source_states else MetricState.MATHEMATICALLY_UNDEFINED
                code = next((result.reason_code for result in arm_results.values() if result.reason_code), "raw_arm_undefined")
                detail = next(
                    (result.reason_detail for result in arm_results.values() if result.reason_detail),
                    "At least one primary raw arm is not computable",
                )
                retention_atom = _atomic(
                    request,
                    dataset_profile,
                    run_id=run_id,
                    metric_id=LOCAL_RETENTION_METRIC_ID,
                    dimension="local-utility",
                    scope_id=retention_scope,
                    evaluator_id=evaluator_id,
                    task_type=task_type,
                    state=state,
                    raw_direction=RawDirection.MAXIMIZE,
                    n_reference=len(test_y),
                    n_synthetic=len(synthetic_y),
                    unit="ratio",
                    reason_code=code,
                    reason_detail=detail,
                )
            atoms.append(retention_atom)
            runs.append(
                {
                    "task_id": task_id,
                    "target_column_id": target["column_id"],
                    "task_type": task_type,
                    "evaluator_id": evaluator_id,
                    "seed": seed,
                    "primary_metric": primary_name,
                    "raw_arms": primary_for_run,
                    "retention": retention_atom.raw_value,
                    "retention_state": retention_atom.state.value,
                    "retention_reason_code": retention_atom.reason_code,
                    # The request checksum is verified against the supplied file
                    # before table evaluation.  Retaining it on every arm makes
                    # the shared held-out view independently reconstructable at
                    # bundle finalization time.
                    "test_fingerprint": test_fingerprint,
                }
            )

    summary = {
        "status": "computed" if len(retention_values) == expected_retention else "partial",
        "profile_id": local_profile["profile_id"],
        "profile_version": local_profile["profile_version"],
        "primary_task_id": task_id,
        "target_column_id": target["column_id"],
        "task_type": task_type,
        "primary_metric": primary_name,
        "raw_arm_means": {
            arm: (float(np.mean(values)) if len(values) == expected_retention else None)
            for arm, values in primary_values.items()
        },
        "retention": float(np.mean(retention_values)) if len(retention_values) == expected_retention else None,
        "retention_clipped": False,
        "computed_retentions": len(retention_values),
        "expected_retentions": expected_retention,
        "evaluator_seeds": list(request.evaluator_seeds),
        "test_rows": len(test_y),
    }
    return atoms, summary, runs


def _prepare_global_frames(tables: ValidatedUtilityTables) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import OrdinalEncoder

    frames = [tables.real_train.copy(), tables.synthetic.copy(), tables.real_test.copy()]
    categorical = [
        spec["name"]
        for spec in tables.column_specs
        if spec["semantic_type"] in _CLASSIFICATION_TYPES | {"string"}
    ]
    datetimes = [spec["name"] for spec in tables.column_specs if spec["semantic_type"] == "datetime"]
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64)
        real_categories = frames[0][categorical].astype("string")
        encoder.fit(real_categories)
        for frame in frames:
            frame[categorical] = encoder.transform(frame[categorical].astype("string"))
    for frame in frames:
        for name in datetimes:
            frame[name] = pd.to_datetime(frame[name], errors="raise").astype("int64").astype("float64")
    return frames[0], frames[1], frames[2]


def _default_global_scorer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task_type: str,
    seed: int,
    time_limit_seconds: int,
    arm: str,
) -> GlobalBackendResult:
    try:
        from autogluon.tabular import TabularPredictor
    except ModuleNotFoundError as exc:
        raise UtilityResourceError(
            "tabeval-tiny-default requires AutoGluon, XGBoost, and TabPFN; no fallback predictor is permitted"
        ) from exc
    for dependency in ("xgboost", "tabpfn"):
        try:
            __import__(dependency)
        except (ImportError, OSError) as exc:
            raise UtilityResourceError(
                f"tabeval-tiny-default requires an importable {dependency} runtime; no fallback is permitted"
            ) from exc
    try:
        from standardized_tabular_diffusion.evaluation.tabstruct import CustomTabPFNModel
    except ModuleNotFoundError as exc:
        raise UtilityResourceError("The pinned TabPFN wrapper dependencies are unavailable") from exc
    if CustomTabPFNModel is None:
        raise UtilityResourceError("The pinned CustomTabPFNModel could not be constructed from installed dependencies")
    hyperparameters: dict[Any, dict[str, Any]] = {
        "XGB": {"random_state": seed, "seed": seed},
        "KNN": {},
        CustomTabPFNModel: {},
    }
    problem_type = "regression" if task_type == "regression" else ("binary" if train[target].nunique() == 2 else "multiclass")
    extra_metric = "root_mean_squared_error" if task_type == "regression" else "balanced_accuracy"
    try:
        with tempfile.TemporaryDirectory(prefix=f"p4-{arm}-{target}-") as workspace:
            predictor = TabularPredictor(
                label=target,
                path=workspace,
                problem_type=problem_type,
                verbosity=0,
                log_to_file=True,
            ).fit(
                train_data=train,
                tuning_data=None,
                hyperparameters=hyperparameters,
                fit_weighted_ensemble=False,
                presets="medium_quality",
                time_limit=time_limit_seconds,
            )
            leaderboard = predictor.leaderboard(test, extra_metrics=[extra_metric])
    except (OSError, PermissionError, TimeoutError) as exc:
        raise UtilityResourceError(f"Authoritative Global Utility backend resource failure: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise UtilityImplementationError(f"Authoritative Global Utility backend failed: {type(exc).__name__}: {exc}") from exc
    if "model" not in leaderboard or extra_metric not in leaderboard or leaderboard.empty:
        raise UtilityImplementationError("AutoGluon leaderboard lacks the declared model or score columns")
    predictor_scores: dict[str, float] = {}
    for _, row in leaderboard.iterrows():
        name = str(row["model"])
        value = float(row[extra_metric])
        if task_type == "regression":
            value = abs(value)
        if not math.isfinite(value):
            raise UtilityImplementationError(f"Non-finite Global Utility score for predictor {name}")
        predictor_scores[name] = value
    names = tuple(sorted(predictor_scores))
    families = {
        "xgb": any("xgboost" in name.lower() for name in names),
        "knn": any("neighbor" in name.lower() or "knn" in name.lower() for name in names),
        "tabpfn": any("tabpfn" in name.lower() for name in names),
    }
    if not families["xgb"] or not families["knn"]:
        raise UtilityImplementationError(f"Global Utility source backend omitted a required XGB/KNN family: {names}")
    # TabPFN supports at most ten classes in this pinned snapshot. AutoGluon
    # may source-faithfully skip it for a higher-cardinality target; the exact
    # trained model set is retained and must match between TRTR and TSTR.
    if not families["tabpfn"] and not (task_type == "classification" and train[target].nunique() > 10):
        raise UtilityImplementationError(f"Global Utility source backend unexpectedly omitted TabPFN: {names}")
    score = float(np.mean(list(predictor_scores.values())))
    return GlobalBackendResult(score=score, predictors=names, predictor_scores=predictor_scores)


def _global_failure_atom(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    *,
    run_id: str,
    metric_id: str,
    scope_id: str,
    task_type: str,
    state: MetricState,
    reason_code: str,
    reason_detail: str,
    n_test: int,
    n_synthetic: int,
    direction: RawDirection,
) -> AtomicResult:
    return _atomic(
        request,
        dataset_profile,
        run_id=run_id,
        metric_id=metric_id,
        dimension="global-utility",
        scope_id=scope_id,
        evaluator_id="tabeval-tiny-default",
        task_type=task_type,
        state=state,
        raw_direction=direction,
        n_reference=n_test,
        n_synthetic=n_synthetic,
        unit="score" if task_type == "classification" else "target-unit",
        reason_code=reason_code,
        reason_detail=reason_detail,
    )


def _evaluate_global(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    tables: ValidatedUtilityTables,
    evaluator_profile: dict[str, Any],
    *,
    run_id: str,
    scorer: GlobalScorer,
) -> tuple[list[AtomicResult], dict[str, Any], list[dict[str, Any]]]:
    by_id = {column["column_id"]: column for column in tables.column_specs}
    target_ids = dataset_profile["utility"]["global"]["included_target_column_ids"]
    targets = [by_id[column_id] for column_id in target_ids]
    real_global, synthetic_global, test_global = _prepare_global_frames(tables)
    global_profile = evaluator_profile["global"]
    time_limit = int(
        request.resource_limits.get(
            "global_time_limit_per_target_seconds",
            global_profile["default_time_limit_per_target_seconds"],
        )
    )
    if time_limit <= 0:
        raise UtilityProfileError("global_time_limit_per_target_seconds must be positive")
    expected_ratios = len(targets) * len(request.evaluator_seeds)
    ratio_weight = 1.0 / expected_ratios
    atoms: list[AtomicResult] = []
    runs: list[dict[str, Any]] = []
    ratios: list[float] = []
    target_ratios: dict[str, list[float]] = {target["column_id"]: [] for target in targets}

    for seed in request.evaluator_seeds:
        for target in targets:
            column_id, name = target["column_id"], target["name"]
            task_type = _task_type(target)
            raw_metric_id = (
                GLOBAL_BALANCED_ACCURACY_METRIC_ID if task_type == "classification" else GLOBAL_RMSE_METRIC_ID
            )
            raw_direction = RawDirection.MAXIMIZE if task_type == "classification" else RawDirection.MINIMIZE
            raw_scope_base = f"{column_id}--seed-{_seed_id(seed)}"
            support_missing = set(pd.unique(tables.real_train[name])) - set(pd.unique(tables.synthetic[name])) if task_type == "classification" else set()
            arm_results: dict[str, GlobalBackendResult] = {}
            arm_failures: dict[str, tuple[MetricState, str, str]] = {}
            for arm, train in (("trtr", real_global), ("tstr", synthetic_global)):
                if arm == "tstr" and support_missing:
                    arm_failures[arm] = (
                        MetricState.INSUFFICIENT_SUPPORT,
                        "insufficient_target_support",
                        f"Synthetic target omits real-train labels: {sorted(support_missing, key=repr)!r}",
                    )
                    continue
                try:
                    arm_results[arm] = scorer(train, test_global, name, task_type, seed, time_limit, arm)
                except UtilityResourceError as exc:
                    arm_failures[arm] = (MetricState.RESOURCE_FAILURE, "global_backend_resource_failure", str(exc))
                except UtilityImplementationError as exc:
                    arm_failures[arm] = (
                        MetricState.IMPLEMENTATION_FAILURE,
                        "global_backend_execution_failure",
                        str(exc),
                    )
                except Exception as exc:
                    arm_failures[arm] = (
                        MetricState.IMPLEMENTATION_FAILURE,
                        "global_backend_unexpected_failure",
                        f"{type(exc).__name__}: {exc}",
                    )
            for arm in ("trtr", "tstr"):
                scope = f"{raw_scope_base}--{arm}"
                if arm in arm_results:
                    result = arm_results[arm]
                    atoms.append(
                        _atomic(
                            request,
                            dataset_profile,
                            run_id=run_id,
                            metric_id=raw_metric_id,
                            dimension="global-utility",
                            scope_id=scope,
                            evaluator_id="tabeval-tiny-default",
                            task_type=task_type,
                            state=MetricState.COMPUTED,
                            raw_direction=raw_direction,
                            n_reference=len(test_global),
                            n_synthetic=len(synthetic_global),
                            raw_value=result.score,
                            unit="score" if task_type == "classification" else "target-unit",
                            warning_codes=(
                                ("source-predictor-set-reduced",)
                                if not any("tabpfn" in item.lower() for item in result.predictors)
                                else ()
                            ),
                        )
                    )
                else:
                    state, code, detail = arm_failures[arm]
                    atoms.append(
                        _global_failure_atom(
                            request,
                            dataset_profile,
                            run_id=run_id,
                            metric_id=raw_metric_id,
                            scope_id=scope,
                            task_type=task_type,
                            state=state,
                            reason_code=code,
                            reason_detail=detail,
                            n_test=len(test_global),
                            n_synthetic=len(synthetic_global),
                            direction=raw_direction,
                        )
                    )
            ratio_scope = raw_scope_base
            ratio_atom: AtomicResult
            if set(arm_results) == {"trtr", "tstr"}:
                reference = arm_results["trtr"]
                synthetic = arm_results["tstr"]
                if reference.predictors != synthetic.predictors:
                    ratio_atom = _global_failure_atom(
                        request,
                        dataset_profile,
                        run_id=run_id,
                        metric_id=GLOBAL_TARGET_RATIO_METRIC_ID,
                        scope_id=ratio_scope,
                        task_type=task_type,
                        state=MetricState.IMPLEMENTATION_FAILURE,
                        reason_code="predictor_set_mismatch",
                        reason_detail=(
                            f"TRTR predictors {reference.predictors!r} differ from TSTR predictors "
                            f"{synthetic.predictors!r}"
                        ),
                        n_test=len(test_global),
                        n_synthetic=len(synthetic_global),
                        direction=RawDirection.MAXIMIZE,
                    )
                else:
                    try:
                        ratio = global_target_ratio(reference.score, synthetic.score, task_type=task_type)
                    except ZeroDivisionError as exc:
                        ratio_atom = _global_failure_atom(
                            request,
                            dataset_profile,
                            run_id=run_id,
                            metric_id=GLOBAL_TARGET_RATIO_METRIC_ID,
                            scope_id=ratio_scope,
                            task_type=task_type,
                            state=MetricState.MATHEMATICALLY_UNDEFINED,
                            reason_code="zero_global_utility_denominator",
                            reason_detail=str(exc),
                            n_test=len(test_global),
                            n_synthetic=len(synthetic_global),
                            direction=RawDirection.MAXIMIZE,
                        )
                    else:
                        ratios.append(ratio)
                        target_ratios[column_id].append(ratio)
                        ratio_atom = _atomic(
                            request,
                            dataset_profile,
                            run_id=run_id,
                            metric_id=GLOBAL_TARGET_RATIO_METRIC_ID,
                            dimension="global-utility",
                            scope_id=ratio_scope,
                            evaluator_id="tabeval-tiny-default",
                            task_type=task_type,
                            state=MetricState.COMPUTED,
                            raw_direction=RawDirection.MAXIMIZE,
                            n_reference=len(test_global),
                            n_synthetic=len(synthetic_global),
                            raw_value=ratio,
                            normalized_value=ratio,
                            aggregate_contribution=ratio * ratio_weight,
                            reference_value=reference.score,
                            weight=ratio_weight,
                            unit="ratio",
                        )
            else:
                failure = next(iter(arm_failures.values()))
                ratio_atom = _global_failure_atom(
                    request,
                    dataset_profile,
                    run_id=run_id,
                    metric_id=GLOBAL_TARGET_RATIO_METRIC_ID,
                    scope_id=ratio_scope,
                    task_type=task_type,
                    state=failure[0],
                    reason_code=failure[1],
                    reason_detail=failure[2],
                    n_test=len(test_global),
                    n_synthetic=len(synthetic_global),
                    direction=RawDirection.MAXIMIZE,
                )
            atoms.append(ratio_atom)
            runs.append(
                {
                    "target_column_id": column_id,
                    "target_name": name,
                    "task_type": task_type,
                    "seed": seed,
                    "trtr": arm_results["trtr"].score if "trtr" in arm_results else None,
                    "tstr": arm_results["tstr"].score if "tstr" in arm_results else None,
                    "ratio": ratio_atom.raw_value,
                    "state": ratio_atom.state.value,
                    "reason_code": ratio_atom.reason_code,
                    "predictors": {
                        arm: list(result.predictors) for arm, result in arm_results.items()
                    },
                    "predictor_scores": {
                        arm: result.predictor_scores for arm, result in arm_results.items()
                    },
                }
            )

    complete = len(ratios) == expected_ratios
    summary = {
        "status": "computed" if complete else "partial",
        "profile_id": global_profile["profile_id"],
        "profile_version": global_profile["profile_version"],
        "predictors": list(global_profile["predictors"]),
        "global_utility": float(np.mean(ratios)) if complete else None,
        "target_ratios": {
            column_id: (float(np.mean(values)) if len(values) == len(request.evaluator_seeds) else None)
            for column_id, values in target_ratios.items()
        },
        "computed_target_seed_ratios": len(ratios),
        "expected_target_seed_ratios": expected_ratios,
        "target_count": len(targets),
        "evaluator_seeds": list(request.evaluator_seeds),
        "ratio_clipped": False,
        "all_target_denominator_required": True,
    }
    return atoms, summary, runs


def evaluate_utility(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    tables: ValidatedUtilityTables,
    *,
    run_id: str,
    global_scorer: GlobalScorer | None = None,
) -> UtilityOutcome:
    """Evaluate P4 without allowing held-out test data into any fit operation."""

    evaluator_profile = load_p4_evaluator_profile()
    validate_utility_profile(dataset_profile, evaluator_profile)
    expected_reference = p4_evaluator_profile_reference()
    if request.evaluator_profile != expected_reference:
        raise UtilityProfileError("Evaluation Request must bind the exact packaged P4 Evaluator Profile")
    if request.real_test_artifact is None:
        raise UtilityProfileError("P4 Utility requires an immutable real_test_artifact")
    local_atoms, local_summary, local_runs = _evaluate_local(
        request,
        dataset_profile,
        tables,
        evaluator_profile,
        run_id=run_id,
    )
    global_atoms, global_summary, global_runs = _evaluate_global(
        request,
        dataset_profile,
        tables,
        evaluator_profile,
        run_id=run_id,
        scorer=global_scorer or _default_global_scorer,
    )
    atoms = tuple([*local_atoms, *global_atoms])
    states = Counter(atom.state.value for atom in atoms)
    denominator_counts = {
        "local_requested_task_evaluator_seeds": local_summary["expected_retentions"],
        "local_computed_retentions": local_summary["computed_retentions"],
        "global_requested_target_seeds": global_summary["expected_target_seed_ratios"],
        "global_computed_target_seed_ratios": global_summary["computed_target_seed_ratios"],
        "global_requested_targets": global_summary["target_count"],
        "global_fully_computed_targets": sum(
            value is not None for value in global_summary["target_ratios"].values()
        ),
        "computed_atomic_results": states.get("computed", 0),
        "noncomputed_atomic_results": len(atoms) - states.get("computed", 0),
    }
    details = {
        "utility_details_schema_version": "1.0.0",
        "implementation_version": UTILITY_IMPLEMENTATION_VERSION,
        "input_boundary": {
            "real_train_fit_allowed": True,
            "synthetic_train_fit_allowed_only_for_tstr": True,
            "real_test_fit_allowed": False,
            "same_real_test_for_all_arms": True,
            "synthetic_repair_applied": False,
        },
        "evaluator_profile": {
            **expected_reference,
            "official_results_allowed": evaluator_profile["official_results_allowed"],
        },
        "local_runs": local_runs,
        "global_runs": global_runs,
        "denominator_counts": denominator_counts,
    }
    source = {
        "local": {
            "definition": "benchmark-native Local Utility panel and benchmark-derived unclipped retention",
            "implementation": "official scikit-learn package estimators through a frozen repository profile",
            "profile": f"{evaluator_profile['local']['profile_id']}@{evaluator_profile['local']['profile_version']}",
        },
        "global": {
            "formula": "TabStruct Equation 4",
            "predictor_profile": (
                f"{evaluator_profile['global']['profile_id']}@{evaluator_profile['global']['profile_version']}"
            ),
            "source": evaluator_profile["global"]["implementation_source"],
            "source_parity_claimed": False,
        },
    }
    return UtilityOutcome(
        atomic_results=atoms,
        local_summary=local_summary,
        global_summary=global_summary,
        denominator_counts=denominator_counts,
        details=details,
        source=source,
    )
