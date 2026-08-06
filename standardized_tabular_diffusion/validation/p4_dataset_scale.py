"""Run and finalize the preregistered P4 Adult/Sick dataset-scale pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import threading
import time
import traceback
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from standardized_tabular_diffusion.evaluation.profiles import DatasetProfile, load_dataset_profile
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    content_fingerprint,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.evaluation.utility import (
    _default_global_scorer,
    _prepare_global_frames,
    _task_type,
    global_target_ratio,
    load_p4_evaluator_profile,
    validate_utility_profile,
)
from standardized_tabular_diffusion.validation import p4_global_source

PROTOCOL_ID = "p4-dataset-scale-admission-pilot-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


class P4DatasetScaleValidationError(RuntimeError):
    """Raised when the preregistered dataset-scale protocol cannot be established."""


def _pilot_manifest() -> dict[str, Any]:
    resource = resources.files("standardized_tabular_diffusion").joinpath(
        "resources/evaluation/evaluators/p4-dataset-scale-pilot-v1.json"
    )
    with resource.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise P4DatasetScaleValidationError("The packaged P4 dataset-scale pilot manifest is invalid")
    return payload


def _profile_for_dataset(dataset: str, manifest: dict[str, Any]) -> DatasetProfile:
    try:
        relative = manifest["coverage"]["datasets"][dataset]["dataset_profile"]
    except KeyError as exc:
        raise P4DatasetScaleValidationError(f"Unknown pilot dataset: {dataset}") from exc
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise P4DatasetScaleValidationError("Dataset profile path escapes the repository")
    return load_dataset_profile(path)


def validate_pilot_manifest(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the scientific schedule before any expensive model fit starts."""

    payload = manifest or _pilot_manifest()
    expected_fields = {
        "pilot_schema_version",
        "pilot_id",
        "pilot_version",
        "status",
        "official_results_allowed",
        "source_runtime_manifest",
        "evaluator_profile_id",
        "evaluator_profile_version",
        "environment",
        "surrogate",
        "coverage",
        "stability",
        "resources",
        "predictor_policy",
        "claim_boundary",
    }
    if set(payload) != expected_fields:
        raise P4DatasetScaleValidationError("P4 dataset-scale pilot fields have drifted")
    if (
        payload["pilot_schema_version"] != "1.0.0"
        or payload["pilot_id"] != "p4-dataset-scale-admission-pilot"
        or payload["pilot_version"] != "0.1.0"
        or payload["status"]
        not in {"preregistered-diagnostic", "dataset-scale-pilot-validated-diagnostic"}
        or payload["official_results_allowed"] is not False
    ):
        raise P4DatasetScaleValidationError("Pilot identity or Official Results boundary has drifted")
    evaluator = load_p4_evaluator_profile()
    if (payload["evaluator_profile_id"], payload["evaluator_profile_version"]) != (
        evaluator["profile_id"],
        evaluator["profile_version"],
    ):
        raise P4DatasetScaleValidationError("Pilot evaluator identity differs from P4")
    environment = payload["environment"]
    if environment != {
        "platform": "Linux x86_64 CPU",
        "python": "3.11",
        "dependency_lock": "requirements-p4-global-source-validation.txt",
    }:
        raise P4DatasetScaleValidationError("Pilot environment identity has drifted")
    surrogate = payload["surrogate"]
    if (
        surrogate.get("type") != "full-row-multiset-preserving-permutation"
        or surrogate.get("preserves_all_rows") is not True
        or surrogate.get("preserves_all_per-column_support") is not True
        or surrogate.get("quality_result_publishable") is not False
    ):
        raise P4DatasetScaleValidationError("Pilot surrogate must preserve the full row multiset")
    datasets = payload["coverage"].get("datasets")
    if payload["coverage"].get("seed") != 0 or not isinstance(datasets, dict) or set(datasets) != {
        "adult",
        "sick",
    }:
        raise P4DatasetScaleValidationError("Pilot coverage must bind Adult and Sick under seed zero")
    for dataset, expected in {
        "adult": ("1.2.0-reviewed", 32561, 16281, 3),
        "sick": ("1.3.0-reviewed", 2800, 972, 4),
    }.items():
        record = datasets[dataset]
        if (
            record.get("dataset_profile_version"),
            record.get("train_rows"),
            record.get("test_rows"),
            record.get("shards"),
        ) != expected:
            raise P4DatasetScaleValidationError(f"Pilot {dataset} coverage identity has drifted")
        profile = _profile_for_dataset(dataset, payload)
        if profile.dataset_profile_version != expected[0]:
            raise P4DatasetScaleValidationError(f"Pilot {dataset} Dataset Profile version does not resolve")
        validate_utility_profile(profile.payload, evaluator)
    stability = payload["stability"]
    if stability.get("seeds") != [0, 1, 2, 3, 4]:
        raise P4DatasetScaleValidationError("Pilot stability seeds must remain preregistered at 0-4")
    sentinels = stability.get("sentinel_target_column_ids")
    if sentinels != {
        "adult": ["income", "native-country", "fnlwgt"],
        "sick": ["class", "referral-source", "tsh"],
    }:
        raise P4DatasetScaleValidationError("Pilot sentinel targets have drifted")
    for key in ("maximum_absolute_identity_ratio_deviation", "maximum_seed_ratio_range"):
        if not isinstance(stability.get(key), (int, float)) or not 0 < float(stability[key]) < 1:
            raise P4DatasetScaleValidationError(f"Invalid preregistered stability bound: {key}")
    resource_limits = payload["resources"]
    for key in (
        "autogluon_fit_time_limit_per_arm_seconds",
        "maximum_observed_arm_wall_seconds",
        "maximum_observed_process_tree_peak_rss_gib",
        "workflow_timeout_minutes",
    ):
        if not isinstance(resource_limits.get(key), (int, float)) or float(resource_limits[key]) <= 0:
            raise P4DatasetScaleValidationError(f"Invalid resource limit: {key}")
    policy = payload["predictor_policy"]
    if (
        policy.get("required_families") != ["xgb", "knn", "tabpfn"]
        or policy.get("tabpfn_maximum_classes") != 10
        or policy.get("arm_model_sets_must_match") is not True
    ):
        raise P4DatasetScaleValidationError("Pilot predictor policy has drifted")
    return payload


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def full_row_permutation(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a deterministic row-order surrogate with the exact input multiset."""

    if train.empty:
        raise P4DatasetScaleValidationError("The permutation surrogate requires a non-empty train table")
    return train.sample(frac=1.0, replace=False, random_state=seed).copy(deep=True)


def _load_materialized_dataset(
    dataset: str, manifest: dict[str, Any]
) -> tuple[DatasetProfile, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    profile = _profile_for_dataset(dataset, manifest)
    materialization_path = REPO_ROOT / "materialized_datasets" / dataset / "manifest.json"
    materialization = read_json(materialization_path)
    if not isinstance(materialization, dict) or materialization.get("dataset") != dataset:
        raise P4DatasetScaleValidationError(f"Missing official {dataset} materialization manifest")
    configured = manifest["coverage"]["datasets"][dataset]
    paths: dict[str, Path] = {}
    artifacts: dict[str, Any] = {}
    for split in ("train", "test"):
        key = f"{split}_data_path"
        relative = materialization.get(key)
        if not isinstance(relative, str):
            raise P4DatasetScaleValidationError(f"Materialized {dataset} lacks {key}")
        path = (REPO_ROOT / relative).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
            raise P4DatasetScaleValidationError(f"Unsafe or absent materialized {dataset} {split} path")
        artifact = materialization.get("artifacts", {}).get(relative)
        if not isinstance(artifact, dict) or sha256_file(path) != artifact.get("sha256"):
            raise P4DatasetScaleValidationError(f"Materialized {dataset} {split} checksum drift")
        paths[split] = path
        artifacts[split] = {
            "repository_path": relative,
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"],
        }
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    canonical = profile.payload["table_contract"]["canonical_column_order"]
    if list(train.columns) != canonical or list(test.columns) != canonical:
        raise P4DatasetScaleValidationError(f"Materialized {dataset} column order differs from the profile")
    if (len(train), len(test)) != (configured["train_rows"], configured["test_rows"]):
        raise P4DatasetScaleValidationError(f"Materialized {dataset} row counts differ from the pilot")
    if train.isna().any().any() or test.isna().any().any():
        raise P4DatasetScaleValidationError(f"Materialized {dataset} still contains missing model values")
    by_id = {column["column_id"]: column for column in profile.payload["columns"]}
    for target_id in profile.payload["utility"]["global"]["included_target_column_ids"]:
        name = by_id[target_id]["name"]
        if train[name].nunique(dropna=False) < 2:
            raise P4DatasetScaleValidationError(f"Included {dataset} target {target_id} is constant")
    identity = {
        "dataset_id": dataset,
        "dataset_profile_version": profile.dataset_profile_version,
        "dataset_profile_fingerprint": profile.fingerprint,
        "dataset_view": materialization["dataset_view"],
        "source_manifest_fingerprint": materialization["source_manifest_fingerprint"],
        "build_spec_fingerprint": materialization["build_spec_fingerprint"],
        "train_rows": len(train),
        "test_rows": len(test),
        "columns": list(train.columns),
        "artifacts": artifacts,
    }
    return profile, train, test, identity


def task_schedule(
    dataset: str,
    mode: str,
    shard_index: int,
    shard_count: int,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve one deterministic matrix shard from the preregistered schedule."""

    payload = validate_pilot_manifest(manifest)
    profile = _profile_for_dataset(dataset, payload)
    included = profile.payload["utility"]["global"]["included_target_column_ids"]
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise P4DatasetScaleValidationError("Invalid shard index/count")
    if mode == "coverage":
        expected_shards = payload["coverage"]["datasets"][dataset]["shards"]
        if shard_count != expected_shards:
            raise P4DatasetScaleValidationError(f"Coverage shard count for {dataset} must be {expected_shards}")
        return [
            {"dataset": dataset, "mode": mode, "target_column_id": target, "seed": 0}
            for position, target in enumerate(included)
            if position % shard_count == shard_index
        ]
    if mode == "stability":
        if (shard_index, shard_count) != (0, 1):
            raise P4DatasetScaleValidationError("Stability uses exactly one shard per dataset")
        coverage_seed = payload["coverage"]["seed"]
        return [
            {"dataset": dataset, "mode": mode, "target_column_id": target, "seed": seed}
            for target in payload["stability"]["sentinel_target_column_ids"][dataset]
            for seed in payload["stability"]["seeds"]
            if seed != coverage_seed
        ]
    raise P4DatasetScaleValidationError(f"Unknown pilot mode: {mode}")


class _ProcessTreeSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.baseline_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _rss_bytes() -> int:
        import psutil

        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return rss

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._rss_bytes())

    def __enter__(self) -> _ProcessTreeSampler:
        self.baseline_rss_bytes = self._rss_bytes()
        self.peak_rss_bytes = self.baseline_rss_bytes
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._rss_bytes())


def _families(predictors: Iterable[str]) -> dict[str, bool]:
    normalized = [name.lower() for name in predictors]
    return {
        "xgb": any("xgboost" in name for name in normalized),
        "knn": any("neighbor" in name or "knn" in name for name in normalized),
        "tabpfn": any("tabpfn" in name for name in normalized),
    }


def _run_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_name: str,
    task_type: str,
    seed: int,
    time_limit_seconds: int,
    arm: str,
) -> dict[str, Any]:
    sampler = _ProcessTreeSampler()
    started = time.perf_counter()
    try:
        with sampler:
            result = _default_global_scorer(
                train,
                test,
                target_name,
                task_type,
                seed,
                time_limit_seconds,
                arm,
            )
        return {
            "status": "pass",
            "score": result.score,
            "predictors": list(result.predictors),
            "predictor_scores": result.predictor_scores,
            "families": _families(result.predictors),
            "wall_seconds": time.perf_counter() - started,
            "baseline_rss_bytes": sampler.baseline_rss_bytes,
            "peak_rss_bytes": sampler.peak_rss_bytes,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
            "baseline_rss_bytes": sampler.baseline_rss_bytes,
            "peak_rss_bytes": sampler.peak_rss_bytes,
        }


def _run_task(
    task: dict[str, Any],
    *,
    profile: DatasetProfile,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    time_limit_seconds: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    target_id = task["target_column_id"]
    seed = int(task["seed"])
    by_id = {column["column_id"]: column for column in profile.payload["columns"]}
    column = by_id[target_id]
    target_name = column["name"]
    task_type = _task_type(column)
    surrogate = full_row_permutation(real_train, seed)
    train_global, surrogate_global, test_global = _prepare_global_frames(
        type(
            "PilotTables",
            (),
            {
                "real_train": real_train,
                "synthetic": surrogate,
                "real_test": real_test,
                "column_specs": tuple(
                    column_spec
                    for column_spec in profile.payload["columns"]
                    if column_spec["name"] in profile.payload["table_contract"]["canonical_column_order"]
                ),
            },
        )()
    )
    target_classes = int(real_train[target_name].nunique(dropna=False))
    high_cardinality = task_type == "classification" and target_classes > int(
        manifest["predictor_policy"]["tabpfn_maximum_classes"]
    )
    arms = {
        "trtr": _run_arm(
            train_global,
            test_global,
            target_name=target_name,
            task_type=task_type,
            seed=seed,
            time_limit_seconds=time_limit_seconds,
            arm="trtr",
        ),
        "tstr": _run_arm(
            surrogate_global,
            test_global,
            target_name=target_name,
            task_type=task_type,
            seed=seed,
            time_limit_seconds=time_limit_seconds,
            arm="tstr",
        ),
    }
    failures: list[str] = []
    if any(arm["status"] != "pass" for arm in arms.values()):
        failures.append("arm-execution-failed")
    else:
        for name, arm in arms.items():
            families = arm["families"]
            if not families["xgb"] or not families["knn"]:
                failures.append(f"{name}-missing-xgb-or-knn")
            if high_cardinality and families["tabpfn"]:
                failures.append(f"{name}-unexpected-high-cardinality-tabpfn")
            if not high_cardinality and not families["tabpfn"]:
                failures.append(f"{name}-missing-tabpfn")
        if arms["trtr"]["predictors"] != arms["tstr"]["predictors"]:
            failures.append("predictor-set-mismatch")
    ratio = None
    if not failures:
        ratio = global_target_ratio(
            float(arms["trtr"]["score"]),
            float(arms["tstr"]["score"]),
            task_type=task_type,
        )
        if not math.isfinite(ratio):
            failures.append("non-finite-ratio")
    resource_limits = manifest["resources"]
    resource_failures: list[str] = []
    for name, arm in arms.items():
        if float(arm["wall_seconds"]) > float(resource_limits["maximum_observed_arm_wall_seconds"]):
            resource_failures.append(f"{name}-wall-time")
        if float(arm["peak_rss_bytes"]) / GIB > float(
            resource_limits["maximum_observed_process_tree_peak_rss_gib"]
        ):
            resource_failures.append(f"{name}-peak-rss")
    failures.extend(resource_failures)
    return {
        **task,
        "task_key": f"{task['dataset']}:{target_id}:seed-{seed}",
        "target_name": target_name,
        "task_type": task_type,
        "target_real_train_cardinality": target_classes,
        "high_cardinality_tabpfn_omission_expected": high_cardinality,
        "surrogate": {
            "type": manifest["surrogate"]["type"],
            "rows": len(surrogate),
            "row_order_fingerprint": content_fingerprint([int(index) for index in surrogate.index]),
            "table_fingerprint": _frame_fingerprint(surrogate),
            "per_column_support_preserved": all(
                set(real_train[name].unique()) == set(surrogate[name].unique()) for name in real_train.columns
            ),
        },
        "arms": arms,
        "ratio": ratio,
        "resource_gate": "pass" if not resource_failures else "fail",
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def _system_identity() -> dict[str, Any]:
    import psutil

    return {
        "platform": f"{platform.system()} / {platform.machine()}",
        "python": platform.python_version(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_bytes": psutil.virtual_memory().total,
        "github_runner_environment": {
            key: os.environ.get(key)
            for key in ("RUNNER_ARCH", "RUNNER_NAME", "RUNNER_OS")
            if os.environ.get(key) is not None
        },
    }


def run_shard(
    output: Path,
    *,
    dataset: str,
    mode: str,
    shard_index: int,
    shard_count: int,
    classifier_checkpoint: Path,
    regressor_checkpoint: Path,
    require_primary_environment: bool = False,
) -> dict[str, Any]:
    """Execute one resumable matrix shard and retain failure evidence."""

    manifest = validate_pilot_manifest()
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "evidence_type": "dataset-scale-shard",
        "status": "fail",
        "repository_commit": p4_global_source._repository_commit(),
        "pilot_manifest_fingerprint": content_fingerprint(manifest),
        "shard": {
            "dataset": dataset,
            "mode": mode,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
        "claim_boundary": manifest["claim_boundary"],
        "results": [],
    }
    atomic_write_json(output, evidence)
    started = time.perf_counter()
    try:
        if require_primary_environment:
            p4_global_source._assert_primary_environment()
        runtime = p4_global_source.verify_pilot_runtime()
        source_manifest = p4_global_source._manifest()
        checkpoints = {
            "classifier": p4_global_source._verify_checkpoint(
                classifier_checkpoint, source_manifest["tabpfn_checkpoints"]["classifier"]
            ),
            "regressor": p4_global_source._verify_checkpoint(
                regressor_checkpoint, source_manifest["tabpfn_checkpoints"]["regressor"]
            ),
        }
        profile, real_train, real_test, dataset_identity = _load_materialized_dataset(dataset, manifest)
        schedule = task_schedule(dataset, mode, shard_index, shard_count, manifest)
        if not schedule:
            raise P4DatasetScaleValidationError("Resolved shard has no tasks")
        evidence.update(
            {
                "environment": _system_identity(),
                "runtime": runtime,
                "source": {
                    "repository": source_manifest["repository"],
                    "revision": source_manifest["revision"],
                    "source_sha256": source_manifest["source_sha256"],
                },
                "checkpoints": checkpoints,
                "dataset": dataset_identity,
                "schedule": schedule,
                "installed_distributions": p4_global_source._pip_freeze(),
            }
        )
        time_limit = int(manifest["resources"]["autogluon_fit_time_limit_per_arm_seconds"])
        for task in schedule:
            result = _run_task(
                task,
                profile=profile,
                real_train=real_train,
                real_test=real_test,
                time_limit_seconds=time_limit,
                manifest=manifest,
            )
            evidence["results"].append(result)
            evidence["elapsed_seconds"] = time.perf_counter() - started
            atomic_write_json(output, evidence)
            gc.collect()
        evidence["status"] = (
            "pass"
            if len(evidence["results"]) == len(schedule)
            and all(result["status"] == "pass" for result in evidence["results"])
            else "fail"
        )
    except Exception as exc:
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    evidence["elapsed_seconds"] = time.perf_counter() - started
    atomic_write_json(output, evidence)
    return evidence


def _expected_task_keys(manifest: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for dataset, record in manifest["coverage"]["datasets"].items():
        for shard_index in range(record["shards"]):
            keys.update(
                task["dataset"] + ":" + task["target_column_id"] + f":seed-{task['seed']}"
                for task in task_schedule(dataset, "coverage", shard_index, record["shards"], manifest)
            )
        keys.update(
            task["dataset"] + ":" + task["target_column_id"] + f":seed-{task['seed']}"
            for task in task_schedule(dataset, "stability", 0, 1, manifest)
        )
    return keys


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise P4DatasetScaleValidationError("Cannot summarize an empty resource sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _locked_files() -> dict[str, str]:
    relative_paths = (
        ".github/workflows/p4-dataset-scale-validation.yml",
        "configs/datasets/adult-uci-2-v1.json",
        "configs/datasets/sick-uci-102-v1.json",
        "requirements-p4-global-source-validation.txt",
        "standardized_tabular_diffusion/evaluation/tabstruct.py",
        "standardized_tabular_diffusion/evaluation/utility.py",
        "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-dataset-scale-pilot-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-utility-pilot-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/upstream/tabeval-p4-source.json",
        "standardized_tabular_diffusion/validation/p4_dataset_scale.py",
        "tests/evaluation/test_p4_dataset_scale_validation.py",
    )
    return {path: sha256_file(REPO_ROOT / path) for path in relative_paths}


def finalize_shards(shard_paths: Iterable[Path], output: Path) -> dict[str, Any]:
    """Fail closed unless every preregistered task appears once and passes."""

    manifest = validate_pilot_manifest()
    expected_keys = _expected_task_keys(manifest)
    shards = [read_json(path) for path in sorted(shard_paths)]
    final: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P4 Adult/Sick dataset-scale and stability admission pilot",
        "status": "fail",
        "repository_commit": p4_global_source._repository_commit(),
        "pilot_manifest_fingerprint": content_fingerprint(manifest),
        "claim_boundary": manifest["claim_boundary"],
        "official_results_allowed": False,
    }
    try:
        expected_shards = sum(
            record["shards"] + 1 for record in manifest["coverage"]["datasets"].values()
        )
        if len(shards) != expected_shards:
            raise P4DatasetScaleValidationError(
                f"Expected {expected_shards} shard artifacts, observed {len(shards)}"
            )
        commit_ids = {shard.get("repository_commit") for shard in shards if isinstance(shard, dict)}
        fingerprints = {shard.get("pilot_manifest_fingerprint") for shard in shards if isinstance(shard, dict)}
        if len(commit_ids) != 1 or None in commit_ids:
            raise P4DatasetScaleValidationError("Shard repository commits differ")
        if fingerprints != {content_fingerprint(manifest)}:
            raise P4DatasetScaleValidationError("Shard pilot identities differ")
        results: list[dict[str, Any]] = []
        dataset_identities: dict[str, dict[str, Any]] = {}
        runtime_identities: list[dict[str, Any]] = []
        seen_shards: set[tuple[str, str, int, int]] = set()
        for shard in shards:
            if shard.get("protocol_id") != PROTOCOL_ID or shard.get("evidence_type") != "dataset-scale-shard":
                raise P4DatasetScaleValidationError("Unexpected shard evidence identity")
            identity = shard["shard"]
            shard_key = (
                identity["dataset"],
                identity["mode"],
                identity["shard_index"],
                identity["shard_count"],
            )
            if shard_key in seen_shards:
                raise P4DatasetScaleValidationError(f"Duplicate shard: {shard_key}")
            seen_shards.add(shard_key)
            if shard.get("status") != "pass":
                raise P4DatasetScaleValidationError(f"Shard did not pass: {shard_key}")
            dataset_identity = shard["dataset"]
            existing = dataset_identities.get(identity["dataset"])
            if existing is not None and existing != dataset_identity:
                raise P4DatasetScaleValidationError("Dataset identity differs across shards")
            dataset_identities[identity["dataset"]] = dataset_identity
            runtime_identities.append(shard["runtime"])
            results.extend(shard["results"])
        task_keys = [result.get("task_key") for result in results]
        if len(task_keys) != len(set(task_keys)) or set(task_keys) != expected_keys:
            missing = sorted(expected_keys - set(task_keys))
            unexpected = sorted(set(task_keys) - expected_keys)
            raise P4DatasetScaleValidationError(
                f"Task coverage differs from the preregistration; missing={missing}, unexpected={unexpected}"
            )
        if any(result.get("status") != "pass" for result in results):
            raise P4DatasetScaleValidationError("At least one target/seed task failed")
        if any(runtime != runtime_identities[0] for runtime in runtime_identities[1:]):
            raise P4DatasetScaleValidationError("Runtime identity differs across shards")

        stability: dict[str, Any] = {}
        max_deviation = float(manifest["stability"]["maximum_absolute_identity_ratio_deviation"])
        max_range = float(manifest["stability"]["maximum_seed_ratio_range"])
        for dataset, targets in manifest["stability"]["sentinel_target_column_ids"].items():
            stability[dataset] = {}
            for target in targets:
                target_results = sorted(
                    (
                        result
                        for result in results
                        if result["dataset"] == dataset and result["target_column_id"] == target
                    ),
                    key=lambda result: result["seed"],
                )
                ratios = [float(result["ratio"]) for result in target_results]
                seeds = [int(result["seed"]) for result in target_results]
                if seeds != manifest["stability"]["seeds"]:
                    raise P4DatasetScaleValidationError(f"Incomplete stability seeds for {dataset}/{target}")
                record = {
                    "seeds": seeds,
                    "ratios": ratios,
                    "mean": statistics.fmean(ratios),
                    "population_standard_deviation": statistics.pstdev(ratios),
                    "range": max(ratios) - min(ratios),
                    "maximum_absolute_deviation_from_identity": max(abs(value - 1.0) for value in ratios),
                }
                record["gate"] = (
                    "pass"
                    if record["range"] <= max_range
                    and record["maximum_absolute_deviation_from_identity"] <= max_deviation
                    else "fail"
                )
                stability[dataset][target] = record
        if any(record["gate"] != "pass" for dataset in stability.values() for record in dataset.values()):
            raise P4DatasetScaleValidationError("Preregistered sentinel stability bound failed")

        arms = [arm for result in results for arm in result["arms"].values()]
        walls = [float(arm["wall_seconds"]) for arm in arms]
        peaks = [float(arm["peak_rss_bytes"]) / GIB for arm in arms]
        resources_summary = {
            "arm_count": len(arms),
            "wall_seconds": {
                "median": statistics.median(walls),
                "p95": _percentile(walls, 0.95),
                "maximum": max(walls),
                "sum": sum(walls),
            },
            "process_tree_peak_rss_gib": {
                "median": statistics.median(peaks),
                "p95": _percentile(peaks, 0.95),
                "maximum": max(peaks),
            },
            "preregistered_limits": manifest["resources"],
        }
        high_cardinality = [
            {
                "dataset": result["dataset"],
                "target_column_id": result["target_column_id"],
                "class_count": result["target_real_train_cardinality"],
                "seeds": sorted(
                    other["seed"]
                    for other in results
                    if other["dataset"] == result["dataset"]
                    and other["target_column_id"] == result["target_column_id"]
                ),
                "tabpfn_omitted_in_all_arms": all(
                    not arm["families"]["tabpfn"]
                    for other in results
                    if other["dataset"] == result["dataset"]
                    and other["target_column_id"] == result["target_column_id"]
                    for arm in other["arms"].values()
                ),
            }
            for result in results
            if result["high_cardinality_tabpfn_omission_expected"]
        ]
        deduplicated_high_cardinality = {
            (record["dataset"], record["target_column_id"]): record for record in high_cardinality
        }
        if not deduplicated_high_cardinality or any(
            not record["tabpfn_omitted_in_all_arms"]
            for record in deduplicated_high_cardinality.values()
        ):
            raise P4DatasetScaleValidationError("Dataset-scale high-cardinality omission did not pass")

        final.update(
            {
                "repository_commit": next(iter(commit_ids)),
                "environment": shards[0]["environment"],
                "runtime": runtime_identities[0],
                "source": shards[0]["source"],
                "checkpoints": shards[0]["checkpoints"],
                "datasets": dataset_identities,
                "schedule": {
                    "coverage_seed": manifest["coverage"]["seed"],
                    "stability_seeds": manifest["stability"]["seeds"],
                    "expected_tasks": len(expected_keys),
                    "completed_tasks": len(results),
                    "completed_arms": len(arms),
                },
                "constant_target_exclusions": {
                    "sick": _profile_for_dataset("sick", manifest).payload["utility"]["global"][
                        "excluded_targets"
                    ]
                },
                "high_cardinality_targets": list(deduplicated_high_cardinality.values()),
                "stability": stability,
                "resource_summary": resources_summary,
                "results": sorted(results, key=lambda result: result["task_key"]),
                "installed_distributions": shards[0]["installed_distributions"],
                "locked_files": _locked_files(),
                "exit_gates": {
                    "official_dataset_materializations_attested": "pass",
                    "all_reviewed_nonconstant_targets_covered_at_full_split_size": "pass",
                    "five_seed_stratified_sentinel_stability": "pass",
                    "xgb_knn_tabpfn_policy_enforced": "pass",
                    "high_cardinality_tabpfn_omission_executed": "pass",
                    "preregistered_resource_safety_bounds": "pass",
                    "constant_target_excluded_with_reason": "pass",
                    "generator_quality_assessed": "not-assessed",
                    "official_results_admission": "not-assessed",
                },
                "status": "pass",
            }
        )
    except Exception as exc:
        final["error_type"] = type(exc).__name__
        final["error"] = str(exc)
        final["traceback"] = traceback.format_exc()
    atomic_write_json(output, final)
    return final


def _discover_shards(directory: Path) -> list[Path]:
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise P4DatasetScaleValidationError(f"No JSON shards found under {directory}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("run-shard", help="Run one preregistered dataset/mode shard")
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--dataset", choices=("adult", "sick"), required=True)
    shard.add_argument("--mode", choices=("coverage", "stability"), required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--classifier-checkpoint", type=Path, required=True)
    shard.add_argument("--regressor-checkpoint", type=Path, required=True)
    shard.add_argument("--require-primary-environment", action="store_true")

    finalize = subparsers.add_parser("finalize", help="Finalize all shard evidence")
    finalize.add_argument("--shards-dir", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run-shard":
        evidence = run_shard(
            args.output,
            dataset=args.dataset,
            mode=args.mode,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            classifier_checkpoint=args.classifier_checkpoint.resolve(),
            regressor_checkpoint=args.regressor_checkpoint.resolve(),
            require_primary_environment=args.require_primary_environment,
        )
    else:
        evidence = finalize_shards(_discover_shards(args.shards_dir), args.output)
    if evidence["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
