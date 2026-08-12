"""P5 high-order fidelity and empirical privacy diagnostics.

The module deliberately keeps fidelity and privacy as separate dimensions.
It does not produce an overall score and it never interprets empirical attacks
as a formal privacy guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Iterable

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.backends.sdmetrics import calculate_dcr
from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    EvaluationRequest,
    MetricState,
    RawDirection,
    utc_timestamp,
)
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, read_json
from standardized_tabular_diffusion.evaluation.table import ValidatedUtilityTables

C2ST_AUROC_METRIC_ID = "std-c2st-rf-auroc"
C2ST_FIDELITY_METRIC_ID = "std-c2st-fidelity"
EXACT_TRAIN_COLLISION_METRIC_ID = "std-exact-train-collision-rate"
SYNTHETIC_DUPLICATE_METRIC_ID = "std-synthetic-internal-duplicate-rate"
SDMETRICS_DCR_METRIC_ID = "sdmetrics-dcr-distribution"
DCR_CALIBRATION_METRIC_ID = "std-dcr-heldout-wasserstein"
DOMIAS_AUROC_METRIC_ID = "domias-kde-attack-auroc"
DOMIAS_ACCURACY_METRIC_ID = "domias-kde-attack-accuracy"
DOMIAS_ADVANTAGE_METRIC_ID = "domias-kde-attack-advantage"
DOMIAS_TPR_METRIC_ID = "domias-kde-tpr-at-fpr-01"
P5_METRIC_VERSION = "1.0.0"
P5_METRICS = tuple(
    {"metric_id": metric_id, "metric_version": P5_METRIC_VERSION}
    for metric_id in (
        C2ST_AUROC_METRIC_ID,
        C2ST_FIDELITY_METRIC_ID,
        EXACT_TRAIN_COLLISION_METRIC_ID,
        SYNTHETIC_DUPLICATE_METRIC_ID,
        SDMETRICS_DCR_METRIC_ID,
        DCR_CALIBRATION_METRIC_ID,
        DOMIAS_AUROC_METRIC_ID,
        DOMIAS_ACCURACY_METRIC_ID,
        DOMIAS_ADVANTAGE_METRIC_ID,
        DOMIAS_TPR_METRIC_ID,
    )
)
P5_DETAILS_ARTIFACT_PATH = "artifacts/p5-details.json"
P5_EVALUATOR_RESOURCE = "p5-high-order-privacy-v1.json"
P5_SOURCE_RESOURCE = "p5-sources.json"


class HighOrderPrivacyError(RuntimeError):
    """Raised when P5 identities or scientific boundaries drift."""


@dataclass(frozen=True)
class P5Outcome:
    atomic_results: tuple[AtomicResult, ...]
    high_order_summary: dict[str, Any]
    privacy_summary: dict[str, Any]
    denominator_counts: dict[str, int]
    details: dict[str, Any]
    source: dict[str, Any]


def _resource_json(package: str, name: str) -> dict[str, Any]:
    item = resources.files(package).joinpath(name)
    with resources.as_file(item) as path:
        payload = read_json(path)
    if not isinstance(payload, dict):
        raise HighOrderPrivacyError(f"Packaged P5 resource {name} must be an object")
    return payload


def load_p5_evaluator_profile() -> dict[str, Any]:
    profile = _resource_json(
        "standardized_tabular_diffusion.resources.evaluation.evaluators",
        P5_EVALUATOR_RESOURCE,
    )
    validate_p5_evaluator_profile(profile)
    return profile


def p5_evaluator_profile_reference() -> dict[str, str]:
    profile = load_p5_evaluator_profile()
    return {
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "sha256": content_fingerprint(profile),
    }


def load_p5_source_manifest() -> dict[str, Any]:
    return _resource_json(
        "standardized_tabular_diffusion.resources.evaluation.upstream",
        P5_SOURCE_RESOURCE,
    )


def validate_p5_evaluator_profile(profile: dict[str, Any]) -> None:
    expected = {
        "profile_schema_version",
        "profile_id",
        "profile_version",
        "status",
        "official_results_allowed",
        "default_evaluator_seeds",
        "c2st",
        "dcr",
        "domias",
        "excluded_metrics",
    }
    if set(profile) != expected:
        raise HighOrderPrivacyError("P5 evaluator profile fields have drifted")
    if (
        profile["profile_schema_version"] != "1.0.0"
        or profile["profile_id"] != "p5-high-order-privacy"
        or profile["profile_version"] != "0.1.0"
        or profile["status"] != "validated-diagnostic"
        or profile["official_results_allowed"] is not False
        or profile["default_evaluator_seeds"] != [0, 1, 2, 3, 4]
    ):
        raise HighOrderPrivacyError("P5 evaluator identity or admission boundary has drifted")
    c2st = profile["c2st"]
    if c2st != {
        "comparison": "heldout-real-vs-synthetic",
        "transform_fit": "real-train-only",
        "categorical": "one-hot-handle-unknown-ignore",
        "numeric": "standard-scaler",
        "balance": "equal-seeded-canonical-subsample",
        "test_fraction": 0.3,
        "classifier": {
            "implementation": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
        "bootstrap_replicates": 1000,
        "minimum_rows_per_class": 20,
    }:
        raise HighOrderPrivacyError("P5 C2ST profile has drifted")
    if profile["dcr"] != {
        "implementation": "sdmetrics.single_table.privacy.dcr_utils.calculate_dcr",
        "query_max_rows": 2000,
        "reference_max_rows": 5000,
        "sampling_seed": 0,
        "chunk_size": 250,
        "calibration": "wasserstein-synthetic-vs-heldout",
    }:
        raise HighOrderPrivacyError("P5 DCR profile has drifted")
    if profile["domias"] != {
        "threat_model": "black-box-released-table-with-auxiliary-real-reference",
        "member_source": "real-train",
        "nonmember_source": "heldout-real-test",
        "reference_source": "disjoint-heldout-real-test",
        "transform_fit": "auxiliary-reference-only",
        "mixed_table_adaptation": "one-hot-standardize-pca-kde",
        "pca_max_components": 8,
        "maximum_rows_per_group": 1000,
        "minimum_rows_per_group": 20,
        "bootstrap_replicates": 1000,
        "accuracy_threshold": "pooled-score-median-strict-greater",
        "tpr_fpr_ceiling": 0.01,
    }:
        raise HighOrderPrivacyError("P5 DOMIAS profile has drifted")
    expected_excluded = {
        "alaa-integrated-alpha-precision",
        "alaa-integrated-beta-recall",
        "alaa-authenticity-paper",
        "alaa-authenticity-repository",
        "synthcity-delta-presence",
        "attribute-inference",
        "great-rf-discriminator-accuracy",
    }
    if set(profile["excluded_metrics"]) != expected_excluded:
        raise HighOrderPrivacyError("P5 excluded-metric boundary has drifted")


def _identity(request: EvaluationRequest, profile: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "protocol_version": request.protocol["protocol_version"],
        "dataset_id": profile["dataset_id"],
        "dataset_version": profile["dataset_version"],
        "dataset_view": profile["dataset_view"],
        "split_id": profile.get("split", {}).get("split_id", "external-evaluation"),
        "model_id": (request.model or {}).get("model_id", "external"),
        "comparison_track": request.comparison_track,
        "generation_seed": request.generation_seed,
    }


def _atomic(
    *,
    identity: dict[str, Any],
    metric_id: str,
    dimension: str,
    scope_id: str,
    direction: RawDirection,
    state: MetricState,
    value: float | None,
    n_reference: int,
    n_synthetic: int,
    n_valid: int,
    computed_at: str,
    evaluator_id: str,
    evaluator_version: str,
    unit: str,
    reference_value: float | None = None,
    artifact_only: bool = False,
    reason_code: str | None = None,
    reason_detail: str | None = None,
) -> AtomicResult:
    raw = None if artifact_only or state is not MetricState.COMPUTED else value
    return AtomicResult(
        **identity,
        metric_id=metric_id,
        metric_version=P5_METRIC_VERSION,
        dimension=dimension,
        scope_type="attack" if dimension == "privacy-risk" else "evaluator",
        scope_id=scope_id,
        state=state,
        raw_direction=direction,
        weight=0.0,
        n_reference=n_reference,
        n_synthetic=n_synthetic,
        n_valid=n_valid,
        n_excluded=0,
        computed_at=computed_at,
        raw_value=raw,
        normalized_value=raw,
        aggregate_contribution=None,
        reference_value=reference_value,
        unit=unit,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        reason_code=reason_code,
        reason_detail=reason_detail,
        artifact_ref=P5_DETAILS_ARTIFACT_PATH,
    )


def _derived_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(f"p5:{label}:{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _canonical_subsample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    ordered = frame.sort_values(list(frame.columns), kind="mergesort", na_position="first").reset_index(drop=True)
    if len(ordered) <= maximum:
        return ordered
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(ordered), size=maximum, replace=False))
    return ordered.iloc[selected].reset_index(drop=True)


def _transformer(column_specs: Iterable[dict[str, Any]]):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    specs = list(column_specs)
    numeric = [item["name"] for item in specs if item["semantic_type"] in {"continuous", "integer"}]
    categorical = [item["name"] for item in specs if item["semantic_type"] in {"categorical", "boolean"}]
    blocks: list[tuple[str, Any, list[str]]] = []
    if numeric:
        blocks.append(("numeric", StandardScaler(), numeric))
    if categorical:
        blocks.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                categorical,
            )
        )
    if not blocks:
        raise HighOrderPrivacyError("P5 has no supported model-view columns")
    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=0.0)


def _bootstrap_auc(y: np.ndarray, scores: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    from sklearn.metrics import roc_auc_score

    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        values[index] = roc_auc_score(y[sample], scores[sample])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _c2st_run(
    tables: ValidatedUtilityTables,
    *,
    seed: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    config = profile["c2st"]
    size = min(len(tables.real_test), len(tables.synthetic))
    if size < config["minimum_rows_per_class"]:
        raise ValueError(f"C2ST requires at least {config['minimum_rows_per_class']} rows per class")
    real = _canonical_subsample(tables.real_test, size, _derived_seed("c2st-real", seed))
    synthetic = _canonical_subsample(tables.synthetic, size, _derived_seed("c2st-synthetic", seed))
    transformer = _transformer(tables.column_specs)
    transformer.fit(tables.real_train)
    real_x = np.asarray(transformer.transform(real), dtype=np.float64)
    synthetic_x = np.asarray(transformer.transform(synthetic), dtype=np.float64)
    x = np.vstack([real_x, synthetic_x])
    y = np.concatenate([np.zeros(size, dtype=np.int8), np.ones(size, dtype=np.int8)])
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["test_fraction"],
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    classifier = RandomForestClassifier(
        n_estimators=config["classifier"]["n_estimators"],
        max_depth=config["classifier"]["max_depth"],
        min_samples_leaf=config["classifier"]["min_samples_leaf"],
        n_jobs=config["classifier"]["n_jobs"],
        random_state=seed,
    )
    classifier.fit(x_train, y_train)
    scores = classifier.predict_proba(x_test)[:, list(classifier.classes_).index(1)]
    auc = float(roc_auc_score(y_test, scores))
    adjusted_auc = max(auc, 1.0 - auc)
    fidelity = 2.0 * (1.0 - adjusted_auc)
    auc_ci = _bootstrap_auc(
        y_test,
        scores,
        seed=_derived_seed("c2st-bootstrap", seed),
        replicates=config["bootstrap_replicates"],
    )
    return {
        "seed": seed,
        "rows_per_class": size,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "feature_count": int(x.shape[1]),
        "raw_auroc": auc,
        "label_invariant_auroc": adjusted_auc,
        "fidelity": fidelity,
        "raw_auroc_ci_95": auc_ci,
        "transform_fit_rows": len(tables.real_train),
        "real_test_used_for_transform_fit": False,
        "synthetic_used_for_transform_fit": False,
    }


def _row_value(value: Any, semantic_type: str) -> tuple[str, Any]:
    if pd.isna(value):
        return ("null", None)
    scalar = value.item() if hasattr(value, "item") else value
    if semantic_type == "continuous":
        return ("number", float(scalar).hex())
    if semantic_type == "integer":
        return ("integer", int(scalar))
    if semantic_type == "boolean":
        return ("boolean", bool(scalar))
    if semantic_type == "datetime":
        return ("datetime", pd.Timestamp(scalar).isoformat())
    return (semantic_type, str(scalar))


def exact_row_tokens(frame: pd.DataFrame, column_specs: Iterable[dict[str, Any]]) -> list[str]:
    """Return collision-resistant, type-aware tokens for exact semantic rows."""

    specs = list(column_specs)
    tokens: list[str] = []
    for row in frame[[item["name"] for item in specs]].itertuples(index=False, name=None):
        payload = [_row_value(value, spec["semantic_type"]) for value, spec in zip(row, specs, strict=True)]
        tokens.append(hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest())
    return tokens


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "population_standard_deviation": float(np.std(values)),
        "zero_rate": float(np.mean(values == 0)),
    }


def _dcr(tables: ValidatedUtilityTables, profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from scipy.stats import wasserstein_distance

    config = profile["dcr"]
    seed = config["sampling_seed"]
    reference = _canonical_subsample(tables.real_train, config["reference_max_rows"], seed)
    synthetic = _canonical_subsample(tables.synthetic, config["query_max_rows"], seed)
    heldout = _canonical_subsample(tables.real_test, config["query_max_rows"], seed)
    metadata = {
        "columns": {
            item["name"]: {
                "sdtype": {
                    "continuous": "numerical",
                    "integer": "numerical",
                    "categorical": "categorical",
                    "boolean": "boolean",
                    "datetime": "datetime",
                    "string": "id",
                }[item["semantic_type"]]
            }
            for item in tables.column_specs
        }
    }
    synthetic_result = calculate_dcr(
        synthetic,
        reference,
        metadata,
        chunk_size=config["chunk_size"],
    )
    heldout_result = calculate_dcr(
        heldout,
        reference,
        metadata,
        chunk_size=config["chunk_size"],
    )
    calibration = float(wasserstein_distance(synthetic_result.distances, heldout_result.distances))
    details = {
        "sampling_seed": seed,
        "reference_rows": len(reference),
        "synthetic_query_rows": len(synthetic),
        "heldout_query_rows": len(heldout),
        "synthetic_to_train": {
            "values": synthetic_result.distances.tolist(),
            "summary": _distance_summary(synthetic_result.distances),
        },
        "heldout_to_train": {
            "values": heldout_result.distances.tolist(),
            "summary": _distance_summary(heldout_result.distances),
        },
        "wasserstein_distance": calibration,
        "heldout_threshold_calibration": {
            "heldout_q01": float(np.quantile(heldout_result.distances, 0.01)),
            "heldout_q05": float(np.quantile(heldout_result.distances, 0.05)),
            "synthetic_fraction_at_or_below_heldout_q01": float(
                np.mean(synthetic_result.distances <= np.quantile(heldout_result.distances, 0.01))
            ),
            "synthetic_fraction_at_or_below_heldout_q05": float(
                np.mean(synthetic_result.distances <= np.quantile(heldout_result.distances, 0.05))
            ),
        },
        "interpretation": "Distributional calibration only; raw DCR is not monotonically ranked as privacy.",
    }
    return details, synthetic_result.source


def domias_density_ratio(p_g: np.ndarray, p_r: np.ndarray) -> np.ndarray:
    """DOMIAS source formula p_G(x)/(p_R(x)+1e-10), exposed for parity tests."""

    return np.asarray(p_g, dtype=np.float64) / (np.asarray(p_r, dtype=np.float64) + 1e-10)


def domias_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Reproduce the source median-threshold accuracy and attack AUROC."""

    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores > np.median(scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    allowed = tpr[fpr <= 0.01]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "advantage": float(np.max(tpr - fpr)),
        "tpr_at_fpr_01": float(np.max(allowed)) if len(allowed) else 0.0,
    }


def _domias_run(
    tables: ValidatedUtilityTables,
    *,
    seed: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    config = profile["domias"]
    heldout_pool = _canonical_subsample(
        tables.real_test,
        min(len(tables.real_test), 2 * config["maximum_rows_per_group"]),
        _derived_seed("domias-heldout", seed),
    )
    permutation = np.random.default_rng(_derived_seed("domias-partition", seed)).permutation(len(heldout_pool))
    shuffled_test = heldout_pool.iloc[permutation].reset_index(drop=True)
    split = len(shuffled_test) // 2
    nonmembers, reference = shuffled_test.iloc[:split], shuffled_test.iloc[split : 2 * split]
    group_size = min(
        config["maximum_rows_per_group"],
        len(tables.real_train),
        len(nonmembers),
        len(reference),
    )
    if group_size < config["minimum_rows_per_group"]:
        raise ValueError(f"DOMIAS requires at least {config['minimum_rows_per_group']} rows per group")
    members = _canonical_subsample(tables.real_train, group_size, _derived_seed("domias-member", seed))
    nonmembers = _canonical_subsample(nonmembers, group_size, _derived_seed("domias-nonmember", seed))
    reference = _canonical_subsample(reference, group_size, _derived_seed("domias-reference", seed))
    synthetic = _canonical_subsample(
        tables.synthetic,
        min(len(tables.synthetic), config["maximum_rows_per_group"]),
        _derived_seed("domias-synthetic", seed),
    )
    if len(synthetic) < config["minimum_rows_per_group"]:
        raise ValueError("DOMIAS has insufficient synthetic rows for KDE")

    transformer = _transformer(tables.column_specs)
    transformer.fit(reference)
    reference_x = np.asarray(transformer.transform(reference), dtype=np.float64)
    maximum_components = min(config["pca_max_components"], len(reference_x) - 1, reference_x.shape[1])
    if maximum_components < 1:
        raise ValueError("DOMIAS reference representation has no usable component")
    pca = PCA(n_components=maximum_components, svd_solver="full")
    pca.fit(reference_x)
    tolerance = np.finfo(np.float64).eps * max(reference_x.shape) * max(float(pca.singular_values_[0]), 1.0)
    components = int(np.sum(pca.singular_values_ > tolerance))
    if components < 1:
        raise ValueError("DOMIAS auxiliary reference has no non-constant PCA direction")
    pca = PCA(n_components=components, svd_solver="full").fit(reference_x)

    def encode(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(pca.transform(np.asarray(transformer.transform(frame), dtype=np.float64)), dtype=np.float64)

    reference_z = encode(reference)
    synthetic_z = encode(synthetic)
    candidates = pd.concat([nonmembers, members], ignore_index=True)
    labels = np.concatenate([np.zeros(group_size, dtype=np.int8), np.ones(group_size, dtype=np.int8)])
    candidate_z = encode(candidates)
    p_g = gaussian_kde(synthetic_z.T)(candidate_z.T)
    p_r = gaussian_kde(reference_z.T)(candidate_z.T)
    scores = domias_density_ratio(p_g, p_r)
    metrics = domias_metrics(labels, scores)
    ci = _bootstrap_auc(
        labels,
        scores,
        seed=_derived_seed("domias-bootstrap", seed),
        replicates=config["bootstrap_replicates"],
    )
    return {
        "seed": seed,
        "group_rows": group_size,
        "synthetic_density_rows": len(synthetic),
        "encoded_feature_count": int(reference_x.shape[1]),
        "pca_components": components,
        **metrics,
        "auroc_ci_95": ci,
        "score_summary": _distance_summary(scores),
        "transform_fit_source": "auxiliary-real-reference-only",
        "candidate_members": "real-train",
        "candidate_nonmembers": "disjoint-heldout-real-test",
        "model_access": "none",
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_high_order_privacy(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    tables: ValidatedUtilityTables,
    *,
    run_id: str,
) -> P5Outcome:
    requested = {(item["metric_id"], item["metric_version"]) for item in request.metrics}
    supported = {(item["metric_id"], item["metric_version"]) for item in P5_METRICS}
    if requested != supported:
        raise HighOrderPrivacyError("P5 requires exactly the registered high-order/privacy metric set")
    profile = load_p5_evaluator_profile()
    if request.evaluator_profile != p5_evaluator_profile_reference():
        raise HighOrderPrivacyError("P5 request evaluator profile differs from the packaged immutable profile")
    if list(request.evaluator_seeds) != profile["default_evaluator_seeds"]:
        raise HighOrderPrivacyError("P5 requires the complete frozen diagnostic seed schedule 0 through 4")

    identity = _identity(request, dataset_profile, run_id)
    computed_at = utc_timestamp()
    atoms: list[AtomicResult] = []
    c2st_runs: list[dict[str, Any]] = []
    for seed in request.evaluator_seeds:
        scope = f"rf-seed-{seed}"
        try:
            run = _c2st_run(tables, seed=seed, profile=profile)
        except (ValueError, np.linalg.LinAlgError) as exc:
            for metric_id in (C2ST_AUROC_METRIC_ID, C2ST_FIDELITY_METRIC_ID):
                atoms.append(
                    _atomic(
                        identity=identity,
                        metric_id=metric_id,
                        dimension="high-order-fidelity",
                        scope_id=scope,
                        direction=RawDirection.TARGET if metric_id == C2ST_AUROC_METRIC_ID else RawDirection.MAXIMIZE,
                        state=MetricState.INSUFFICIENT_SUPPORT,
                        value=None,
                        n_reference=len(tables.real_test),
                        n_synthetic=len(tables.synthetic),
                        n_valid=0,
                        computed_at=computed_at,
                        evaluator_id="c2st-rf",
                        evaluator_version="1.0.0",
                        unit="score",
                        reason_code="insufficient_c2st_support",
                        reason_detail=str(exc),
                    )
                )
            continue
        c2st_runs.append(run)
        for metric_id, value, direction in (
            (C2ST_AUROC_METRIC_ID, run["raw_auroc"], RawDirection.TARGET),
            (C2ST_FIDELITY_METRIC_ID, run["fidelity"], RawDirection.MAXIMIZE),
        ):
            atoms.append(
                _atomic(
                    identity=identity,
                    metric_id=metric_id,
                    dimension="high-order-fidelity",
                    scope_id=scope,
                    direction=direction,
                    state=MetricState.COMPUTED,
                    value=value,
                    n_reference=run["rows_per_class"],
                    n_synthetic=run["rows_per_class"],
                    n_valid=2 * run["rows_per_class"],
                    computed_at=computed_at,
                    evaluator_id="c2st-rf",
                    evaluator_version="1.0.0",
                    unit="score",
                    reference_value=0.5 if metric_id == C2ST_AUROC_METRIC_ID else None,
                )
            )

    train_tokens = set(exact_row_tokens(tables.real_train, tables.column_specs))
    synthetic_tokens = exact_row_tokens(tables.synthetic, tables.column_specs)
    collision_count = sum(token in train_tokens for token in synthetic_tokens)
    duplicate_count = len(synthetic_tokens) - len(set(synthetic_tokens))
    matched_train_tokens = set(synthetic_tokens).intersection(train_tokens)
    matched_multiplicities: dict[str, int] = {}
    for token in synthetic_tokens:
        if token in train_tokens:
            matched_multiplicities[token] = matched_multiplicities.get(token, 0) + 1
    collision_rate = collision_count / len(synthetic_tokens)
    duplicate_rate = duplicate_count / len(synthetic_tokens)
    for metric_id, value, scope in (
        (EXACT_TRAIN_COLLISION_METRIC_ID, collision_rate, "train-collision"),
        (SYNTHETIC_DUPLICATE_METRIC_ID, duplicate_rate, "internal-duplication"),
    ):
        atoms.append(
            _atomic(
                identity=identity,
                metric_id=metric_id,
                dimension="privacy-risk",
                scope_id=scope,
                direction=RawDirection.MINIMIZE,
                state=MetricState.COMPUTED,
                value=value,
                n_reference=len(tables.real_train),
                n_synthetic=len(tables.synthetic),
                n_valid=len(tables.synthetic),
                computed_at=computed_at,
                evaluator_id="exact-row-audit",
                evaluator_version="1.0.0",
                unit="rate",
            )
        )

    dcr_details, dcr_source = _dcr(tables, profile)
    atoms.extend(
        [
            _atomic(
                identity=identity,
                metric_id=SDMETRICS_DCR_METRIC_ID,
                dimension="privacy-risk",
                scope_id="synthetic-to-real-train",
                direction=RawDirection.DISTRIBUTIONAL,
                state=MetricState.COMPUTED,
                value=None,
                artifact_only=True,
                n_reference=dcr_details["reference_rows"],
                n_synthetic=dcr_details["synthetic_query_rows"],
                n_valid=dcr_details["synthetic_query_rows"],
                computed_at=computed_at,
                evaluator_id="sdmetrics-dcr",
                evaluator_version="0.28.3.dev0",
                unit="mixed-distance",
            ),
            _atomic(
                identity=identity,
                metric_id=DCR_CALIBRATION_METRIC_ID,
                dimension="privacy-risk",
                scope_id="synthetic-vs-heldout",
                direction=RawDirection.MINIMIZE,
                state=MetricState.COMPUTED,
                value=dcr_details["wasserstein_distance"],
                n_reference=dcr_details["heldout_query_rows"],
                n_synthetic=dcr_details["synthetic_query_rows"],
                n_valid=dcr_details["heldout_query_rows"] + dcr_details["synthetic_query_rows"],
                computed_at=computed_at,
                evaluator_id="sdmetrics-dcr-calibration",
                evaluator_version="1.0.0",
                unit="mixed-distance",
            ),
        ]
    )

    domias_runs: list[dict[str, Any]] = []
    domias_metric_fields = (
        (DOMIAS_AUROC_METRIC_ID, "auroc"),
        (DOMIAS_ACCURACY_METRIC_ID, "accuracy"),
        (DOMIAS_ADVANTAGE_METRIC_ID, "advantage"),
        (DOMIAS_TPR_METRIC_ID, "tpr_at_fpr_01"),
    )
    for seed in request.evaluator_seeds:
        scope = f"kde-seed-{seed}"
        try:
            run = _domias_run(tables, seed=seed, profile=profile)
        except (ValueError, np.linalg.LinAlgError) as exc:
            for metric_id, _ in domias_metric_fields:
                atoms.append(
                    _atomic(
                        identity=identity,
                        metric_id=metric_id,
                        dimension="privacy-risk",
                        scope_id=scope,
                        direction=RawDirection.MINIMIZE,
                        state=MetricState.INSUFFICIENT_SUPPORT,
                        value=None,
                        n_reference=len(tables.real_train) + len(tables.real_test),
                        n_synthetic=len(tables.synthetic),
                        n_valid=0,
                        computed_at=computed_at,
                        evaluator_id="domias-kde-mixed",
                        evaluator_version="1.0.0",
                        unit="attack-score",
                        reason_code="insufficient_domias_support",
                        reason_detail=str(exc),
                    )
                )
            continue
        domias_runs.append(run)
        for metric_id, field in domias_metric_fields:
            atoms.append(
                _atomic(
                    identity=identity,
                    metric_id=metric_id,
                    dimension="privacy-risk",
                    scope_id=scope,
                    direction=RawDirection.MINIMIZE,
                    state=MetricState.COMPUTED,
                    value=run[field],
                    n_reference=3 * run["group_rows"],
                    n_synthetic=run["synthetic_density_rows"],
                    n_valid=2 * run["group_rows"],
                    computed_at=computed_at,
                    evaluator_id="domias-kde-mixed",
                    evaluator_version="1.0.0",
                    unit="attack-score",
                )
            )

    high_order = {
        "c2st_raw_auroc_mean": _mean([run["raw_auroc"] for run in c2st_runs]),
        "c2st_fidelity_mean": _mean([run["fidelity"] for run in c2st_runs]),
        "computed_seeds": len(c2st_runs),
        "expected_seeds": len(request.evaluator_seeds),
        "overall_fidelity_score": None,
        "overall_score_policy": "prohibited-in-p5",
    }
    privacy = {
        "exact_train_collision_rate": collision_rate,
        "synthetic_internal_duplicate_rate": duplicate_rate,
        "dcr_wasserstein_heldout_calibration": dcr_details["wasserstein_distance"],
        "domias_attack_auroc_mean": _mean([run["auroc"] for run in domias_runs]),
        "domias_attack_accuracy_mean": _mean([run["accuracy"] for run in domias_runs]),
        "domias_attack_advantage_mean": _mean([run["advantage"] for run in domias_runs]),
        "domias_tpr_at_fpr_01_mean": _mean([run["tpr_at_fpr_01"] for run in domias_runs]),
        "computed_domias_seeds": len(domias_runs),
        "expected_domias_seeds": len(request.evaluator_seeds),
        "overall_privacy_score": None,
        "formal_privacy_guarantee": False,
    }
    source_manifest = load_p5_source_manifest()
    details = {
        "p5_details_schema_version": "1.0.0",
        "evaluator_profile": {**p5_evaluator_profile_reference(), "official_results_allowed": False},
        "input_boundary": {
            "real_train_transform_fit_allowed": True,
            "real_test_transform_fit_allowed": False,
            "synthetic_transform_fit_allowed": False,
            "domias_auxiliary_reference_fit_only": True,
            "synthetic_repair_applied": False,
        },
        "high_order": {"c2st_runs": c2st_runs, "summary": high_order},
        "privacy": {
            "exact_rows": {
                "train_collision_count": collision_count,
                "distinct_train_rows_matched": len(matched_train_tokens),
                "maximum_matched_train_row_multiplicity": max(matched_multiplicities.values(), default=0),
                "synthetic_duplicate_count": duplicate_count,
                "synthetic_rows": len(synthetic_tokens),
            },
            "dcr": dcr_details,
            "domias_threat_model": {
                "attacker_knowledge": "released synthetic table plus disjoint auxiliary real reference sample",
                "member_definition": "row from the real training split",
                "nonmember_definition": "row from the held-out real test split",
                "model_access": "none",
                "score": "p_G(x)/(p_R(x)+1e-10)",
                "higher_score_means": "higher inferred membership risk",
                "representation": "reference-fitted mixed-table transform and PCA; benchmark adaptation",
            },
            "domias_runs": domias_runs,
            "summary": privacy,
        },
        "excluded_metrics": profile["excluded_metrics"],
        "claim_boundary": "Empirical diagnostics only; no differential-privacy or other formal privacy guarantee.",
        "sources": {"manifest": source_manifest, "sdmetrics_dcr_runtime": dcr_source},
    }
    return P5Outcome(
        atomic_results=tuple(atoms),
        high_order_summary=high_order,
        privacy_summary=privacy,
        denominator_counts={
            "expected_c2st_seeds": len(request.evaluator_seeds),
            "computed_c2st_seeds": len(c2st_runs),
            "synthetic_rows_exact_audit": len(tables.synthetic),
            "dcr_synthetic_query_rows": dcr_details["synthetic_query_rows"],
            "dcr_heldout_query_rows": dcr_details["heldout_query_rows"],
            "expected_domias_seeds": len(request.evaluator_seeds),
            "computed_domias_seeds": len(domias_runs),
        },
        details=details,
        source={"manifest": source_manifest, "sdmetrics_dcr_runtime": dcr_source},
    )
