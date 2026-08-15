"""Run and retain the native-Windows minimal-real cross-baseline protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, load_experiment_config
from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.runner import build_run_context, run_action, save_run_context

PROTOCOL_ID = "pipeline-v2-native-windows-v1"
IDENTITY_FILENAME = ".standardized-run-identity.json"
COPY_EXCLUSIONS = {
    IDENTITY_FILENAME,
    "artifact_bundle.json",
    "artifacts.json",
    "pipeline_result.json",
    "run_context.json",
}


class PipelineV2Error(RuntimeError):
    """Raised when a V2 acceptance condition fails closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True, encoding="utf-8").strip()


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0" or payload.get("protocol_id") != PROTOCOL_ID:
        raise PipelineV2Error(f"Unsupported V2 plan identity: {path}")
    models = payload.get("models")
    if not isinstance(models, list) or len({row.get("model_id") for row in models}) != len(models):
        raise PipelineV2Error("V2 plan models must be a unique list")
    seeds = payload.get("generation_seeds")
    if not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2:
        raise PipelineV2Error("V2 plan requires exactly two distinct generation seeds")
    return payload


def _model_entry(plan: dict[str, Any], model_id: str) -> dict[str, Any]:
    matches = [row for row in plan["models"] if row["model_id"] == model_id]
    if len(matches) != 1:
        raise PipelineV2Error(f"Model is not uniquely declared by the V2 plan: {model_id}")
    return matches[0]


_PIP_CHECK_CONFLICT = re.compile(
    r"^(?P<distribution>\S+) (?P<version>\S+) has requirement "
    r"(?P<requirement>.+), but you have (?P<installed_dependency>\S+) "
    r"(?P<installed_dependency_version>\S+)\.$"
)


def _review_pip_check(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    packages: dict[str, str],
    waivers: list[dict[str, str]],
) -> dict[str, Any]:
    """Fail closed except for exact, plan-declared stale-metadata conflicts."""

    from packaging.utils import canonicalize_name

    lines = [line.strip() for line in f"{stdout}\n{stderr}".splitlines() if line.strip()]
    if returncode == 0:
        if lines not in ([], ["No broken requirements found."]):
            raise PipelineV2Error(f"pip check returned success with unexpected output: {lines}")
        if waivers:
            raise PipelineV2Error("Declared pip-check waivers were not exercised")
        return {"status": "pass", "output": lines, "conflicts": [], "reviewed_waivers": []}

    if not lines:
        raise PipelineV2Error("pip check failed without a diagnostic")
    if not waivers:
        raise PipelineV2Error(f"pip check failed: {'; '.join(lines)}")

    required_fields = {
        "distribution",
        "version",
        "requirement",
        "installed_dependency",
        "installed_dependency_version",
        "reason",
    }
    normalized_waivers: list[dict[str, str]] = []
    for waiver in waivers:
        if set(waiver) != required_fields or not all(
            isinstance(waiver[field], str) and waiver[field].strip() for field in required_fields
        ):
            raise PipelineV2Error("Each pip-check waiver must contain only the six required non-empty fields")
        normalized_waivers.append(dict(waiver))

    consumed: set[int] = set()
    conflicts: list[dict[str, str]] = []
    for line in lines:
        match = _PIP_CHECK_CONFLICT.fullmatch(line)
        if match is None:
            raise PipelineV2Error(f"Unrecognized pip-check failure cannot be waived: {line}")
        conflict = match.groupdict()
        installed_distribution = packages.get(canonicalize_name(conflict["distribution"]))
        installed_dependency = packages.get(canonicalize_name(conflict["installed_dependency"]))
        if installed_distribution != conflict["version"] or installed_dependency != conflict[
            "installed_dependency_version"
        ]:
            raise PipelineV2Error(f"pip-check diagnostic does not match the recorded environment: {line}")
        matches = [
            index
            for index, waiver in enumerate(normalized_waivers)
            if index not in consumed
            and all(waiver[field] == conflict[field] for field in required_fields - {"reason"})
        ]
        if len(matches) != 1:
            raise PipelineV2Error(f"pip-check failure lacks one exact reviewed waiver: {line}")
        consumed.add(matches[0])
        conflicts.append(conflict)
    if len(consumed) != len(normalized_waivers):
        raise PipelineV2Error("One or more declared pip-check waivers were not exercised")
    return {
        "status": "pass-with-reviewed-waiver",
        "conflicts": conflicts,
        "reviewed_waivers": normalized_waivers,
    }


def _environment(repo_root: Path, *, pip_check_waivers: list[dict[str, str]] | None = None) -> dict[str, Any]:
    if platform.system() != "Windows" or sys.version_info[:2] != (3, 11):
        raise PipelineV2Error(
            "Retained V2 evidence requires native Windows and Python 3.11; "
            f"observed {platform.platform()} / {platform.python_version()}"
        )
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    pip_check_result = _review_pip_check(
        returncode=pip_check.returncode,
        stdout=pip_check.stdout,
        stderr=pip_check.stderr,
        packages=packages,
        waivers=[] if pip_check_waivers is None else pip_check_waivers,
    )
    hardware: dict[str, Any] = {"torch": None, "cuda_runtime": None, "cuda_available": False, "gpu": None}
    try:
        import torch

        hardware.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count(),
            }
        )
    except ImportError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": dict(sorted(packages.items())),
        "packages_sha256": _sha256_json(dict(sorted(packages.items()))),
        "pip_check": pip_check_result,
        "hardware": hardware,
    }


def _require_device(entry: dict[str, Any], environment: dict[str, Any], plan: dict[str, Any]) -> None:
    requested = entry["device"]
    if requested == "cpu":
        return
    hardware = environment["hardware"]
    expected_gpu = plan["target_environment"]["gpu_name"]
    if not hardware["cuda_available"] or hardware["gpu"] != expected_gpu:
        raise PipelineV2Error(
            f"{entry['model_id']} requires CUDA on {expected_gpu}; observed {hardware}"
        )


def _verify_environment_lock(
    lock_path: Path,
    environment: dict[str, Any],
    required_packages: dict[str, str],
) -> dict[str, Any]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = {
        canonicalize_name(name): version
        for name, version in environment["packages"].items()
    }
    checked: dict[str, dict[str, str]] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        observed = installed.get(name)
        if observed is None or (requirement.specifier and observed not in requirement.specifier):
            raise PipelineV2Error(
                f"Environment does not satisfy {lock_path.name}: required {requirement}, observed {observed}"
            )
        checked[name] = {"required": str(requirement.specifier) or "present", "observed": observed}
    for package, expected in required_packages.items():
        name = canonicalize_name(package)
        observed = installed.get(name)
        if observed is None:
            raise PipelineV2Error(f"Required V2 runtime package is missing: {package}=={expected}")
        normalized_observed = observed.partition("+")[0] if "+" not in expected else observed
        if normalized_observed != expected:
            raise PipelineV2Error(
                f"V2 runtime package mismatch for {package}: expected {expected}, observed {observed}"
            )
        checked[name] = {"required": f"=={expected}", "observed": observed}
    return {
        "status": "pass",
        "path": str(lock_path.resolve()),
        "sha256": sha256_file(lock_path),
        "packages": dict(sorted(checked.items())),
    }


def _greedy_coverage_indices(frame: Any, categorical: list[str], count: int, seed: int) -> list[int]:
    import numpy as np

    if count > len(frame):
        raise PipelineV2Error(f"Fixture requests {count} rows from a {len(frame)}-row split")
    tokens = {
        (column, str(value))
        for column in categorical
        for value in frame[column].astype(str).unique().tolist()
    }
    row_tokens = [
        {(column, str(frame.iloc[index][column])) for column in categorical}
        for index in range(len(frame))
    ]
    order = np.random.default_rng(seed).permutation(len(frame)).tolist()
    selected: list[int] = []
    remaining = set(order)
    uncovered = set(tokens)
    while uncovered and len(selected) < count:
        best = max(order, key=lambda index: len(row_tokens[index] & uncovered) if index in remaining else -1)
        if not (row_tokens[best] & uncovered):
            break
        selected.append(best)
        remaining.remove(best)
        uncovered -= row_tokens[best]
    if uncovered:
        raise PipelineV2Error(f"Fixture row budget cannot cover declared categorical values: {sorted(uncovered)}")
    target = categorical[-1]
    class_counts = frame.iloc[selected][target].astype(str).value_counts().to_dict()
    for index in order:
        if len(selected) >= count:
            break
        if index not in remaining:
            continue
        label = str(frame.iloc[index][target])
        if class_counts.get(label, 0) < 8:
            selected.append(index)
            remaining.remove(index)
            class_counts[label] = class_counts.get(label, 0) + 1
    for index in order:
        if len(selected) >= count:
            break
        if index in remaining:
            selected.append(index)
            remaining.remove(index)
    if len(selected) != count:
        raise PipelineV2Error("Fixture selection did not produce the declared row count")
    return sorted(selected)


def _tree_manifest(root: Path, *, exclusions: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = exclusions or set()
    records: dict[str, dict[str, Any]] = {}
    if not root.is_dir() or root.is_symlink():
        raise PipelineV2Error(f"Manifest root is missing or unsafe: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative.split("/", maxsplit=1)[0] in excluded:
            continue
        if path.is_symlink():
            raise PipelineV2Error(f"Validation tree contains a prohibited symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PipelineV2Error(f"Validation tree contains a non-regular file: {path}")
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return records


def materialize_fixture(repo_root: Path, plan: dict[str, Any], fixture_root: Path) -> DatasetSpec:
    import numpy as np
    import pandas as pd

    fixture = plan["fixture"]
    fixture_id = fixture["fixture_id"]
    source_root = repo_root / "TabSyn-main" / "data" / fixture["source_dataset"]
    output_root = fixture_root / fixture_id
    manifest_path = output_root / "fixture-manifest.json"
    native_root = repo_root / "TabSyn-main" / "data" / fixture_id
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_fixture") != fixture:
            raise PipelineV2Error("Existing V2 fixture was created from a different plan")
        expected = manifest.get("files", {})
        if _tree_manifest(output_root, exclusions={"fixture-manifest.json"}) != expected:
            raise PipelineV2Error("Existing V2 fixture differs from its retained manifest")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f"Refusing a non-empty unmanifested V2 fixture: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        info = json.loads((source_root / "info.json").read_text(encoding="utf-8"))
        train = pd.read_csv(source_root / "train.csv")
        test = pd.read_csv(source_root / "test.csv")
        categorical = [info["column_names"][index] for index in info["cat_col_idx"]]
        categorical.extend(info["column_names"][index] for index in info["target_col_idx"])
        train_indices = _greedy_coverage_indices(
            train,
            categorical,
            int(fixture["train_rows"]),
            int(fixture["selection_seed"]),
        )
        test_indices = _greedy_coverage_indices(
            test,
            categorical,
            int(fixture["test_rows"]),
            int(fixture["selection_seed"]) + 1,
        )
        selected_train = train.iloc[train_indices].reset_index(drop=True)
        selected_test = test.iloc[test_indices].reset_index(drop=True)
        selected_train.to_csv(output_root / "train.csv", index=False)
        selected_test.to_csv(output_root / "test.csv", index=False)
        for prefix, indices in (("train", train_indices), ("test", test_indices)):
            for stem in ("X_num", "X_cat", "y"):
                values = np.load(source_root / f"{stem}_{prefix}.npy", allow_pickle=False)
                np.save(output_root / f"{stem}_{prefix}.npy", values[indices], allow_pickle=False)
        info["name"] = fixture_id
        info["dataset_view"] = fixture_id
        info["data_path"] = f"data/{fixture_id}/train.csv"
        info["test_path"] = f"data/{fixture_id}/test.csv"
        info["train_num"] = len(selected_train)
        info["test_num"] = len(selected_test)
        info["val_num"] = 0
        for column in categorical:
            info["column_info"][column]["categories"] = sorted(selected_train[column].astype(str).unique().tolist())
        for column in info["int_columns"]:
            info["column_info"][column]["min"] = float(selected_train[column].min())
            info["column_info"][column]["max"] = float(selected_train[column].max())
        atomic_write_json(output_root / "info.json", info)
        files = _tree_manifest(output_root)
        manifest = {
            "schema_version": "1.0.0",
            "fixture_id": fixture_id,
            "plan_fixture": fixture,
            "source": {
                "info_sha256": sha256_file(source_root / "info.json"),
                "train_csv_sha256": sha256_file(source_root / "train.csv"),
                "test_csv_sha256": sha256_file(source_root / "test.csv"),
            },
            "selection": {
                "train_indices_sha256": _sha256_json(train_indices),
                "test_indices_sha256": _sha256_json(test_indices),
            },
            "files": files,
        }
        atomic_write_json(manifest_path, manifest)
    if native_root.exists():
        if _tree_manifest(native_root) != _tree_manifest(output_root, exclusions={"fixture-manifest.json"}):
            raise PipelineV2Error("Existing TabSyn-native V2 fixture differs from the canonical fixture")
    else:
        native_root.mkdir(parents=True)
        for path in output_root.iterdir():
            if path.name != "fixture-manifest.json":
                shutil.copy2(path, native_root / path.name)
    info = json.loads((output_root / "info.json").read_text(encoding="utf-8"))
    column_names = list(info["column_names"])
    numerical = [column_names[index] for index in info["num_col_idx"]]
    categorical = [column_names[index] for index in info["cat_col_idx"]]
    targets = [column_names[index] for index in info["target_col_idx"]]
    return DatasetSpec(
        name=fixture_id,
        task_type="classification" if info["task_type"] in {"binclass", "multiclass"} else "regression",
        column_names=column_names,
        numerical_columns=numerical,
        categorical_columns=categorical,
        target_columns=targets,
        metadata_path=output_root / "info.json",
        train_data_path=output_root / "train.csv",
        test_data_path=output_root / "test.csv",
        provenance=[str(manifest_path.resolve())],
        extra={
            "column_info": {
                column: "int" if column in info["int_columns"] else "str"
                for column in column_names
            },
            "validation_fixture": manifest,
        },
    )


def _copy_training_artifacts(source: Path, destination: Path) -> dict[str, dict[str, Any]]:
    source_manifest = _tree_manifest(source, exclusions=COPY_EXCLUSIONS)
    for relative in source_manifest:
        source_path = source / Path(relative)
        target_path = destination / Path(relative)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            raise FileExistsError(f"Derived sample workspace already contains a training artifact: {target_path}")
        shutil.copy2(source_path, target_path)
    copied = {
        relative: {"bytes": (destination / Path(relative)).stat().st_size, "sha256": sha256_file(destination / Path(relative))}
        for relative in source_manifest
    }
    if copied != source_manifest:
        raise PipelineV2Error("Copied training artifacts differ from their source bytes")
    return source_manifest


def _assert_manifest_unchanged(root: Path, expected: dict[str, dict[str, Any]]) -> None:
    for relative, record in expected.items():
        path = root / Path(relative)
        if not path.is_file() or path.is_symlink():
            raise PipelineV2Error(f"Sampling removed or made unsafe a training artifact: {path}")
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed != record:
            raise PipelineV2Error(f"Sampling mutated a copied training artifact: {path}")


def _action_config(
    base: ExperimentConfig,
    *,
    dataset: str,
    output_dir: Path,
    device: str,
    training_seed: int,
    sample_seed: int | None,
) -> ExperimentConfig:
    config = deepcopy(base)
    config.dataset = dataset
    config.output_dir = str(output_dir.resolve())
    config.train.seed = training_seed
    config.train.device = device
    config.sample.seed = sample_seed
    config.evaluation.enabled = False
    return config


def _validate_sample(path: Path, dataset_spec: DatasetSpec, expected_rows: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    if path.is_symlink() or not path.is_file():
        raise PipelineV2Error(f"Generated sample is missing or unsafe: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != dataset_spec.column_names or len(frame) != expected_rows:
        raise PipelineV2Error(
            f"Generated sample shape/schema mismatch: rows={len(frame)}, columns={list(frame.columns)}"
        )
    if bool(frame.isna().any().any()):
        raise PipelineV2Error("Generated sample contains missing values")
    train = pd.read_csv(dataset_spec.train_data_path)
    for column in [*dataset_spec.categorical_columns, *dataset_spec.target_columns]:
        unexpected = sorted(set(frame[column].astype(str)) - set(train[column].astype(str)))
        if unexpected:
            raise PipelineV2Error(f"Generated column {column!r} contains out-of-domain values: {unexpected}")
    for column in dataset_spec.numerical_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not bool(np.isfinite(values).all()):
            raise PipelineV2Error(f"Generated numerical column {column!r} contains non-finite values")
        if dataset_spec.extra["column_info"].get(column) == "int" and not bool(np.equal(values, np.rint(values)).all()):
            raise PipelineV2Error(f"Generated integer column {column!r} contains non-integral values")
    return {
        "path": str(path.resolve()),
        "rows": len(frame),
        "columns": list(frame.columns),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "missing_cells": 0,
        "schema_valid": True,
    }


def run_probe(
    repo_root: Path,
    plan_path: Path,
    model_id: str,
    work_root: Path,
    record_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan_path = plan_path.resolve()
    work_root = work_root.resolve()
    record_path = record_path.resolve()
    plan = _load_plan(plan_path)
    entry = _model_entry(plan, model_id)
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "model_id": model_id,
        "status": "running",
        "started_at": _utc_now(),
        "repository_commit": _git(repo_root, "rev-parse", "HEAD"),
        "claim_boundary": plan["claim_boundary"],
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "entry": entry,
        "attempts": [],
    }
    try:
        if _git(repo_root, "status", "--porcelain", "--untracked-files=no"):
            raise PipelineV2Error("V2 probes require a clean tracked worktree")
        environment = _environment(repo_root, pip_check_waivers=entry.get("pip_check_waivers", []))
        _require_device(entry, environment, plan)
        record["environment"] = environment
        config_path = repo_root / entry["config_path"]
        lock_path = repo_root / entry["environment_lock"]
        record["configuration"] = {"path": entry["config_path"], "sha256": sha256_file(config_path)}
        record["environment_lock"] = _verify_environment_lock(
            lock_path,
            environment,
            entry["required_packages"],
        )
        fixture_spec = materialize_fixture(repo_root, plan, work_root / "shared-fixture")
        record["dataset_spec"] = fixture_spec.to_dict()
        record["fixture_manifest"] = json.loads(
            (fixture_spec.metadata_path.parent / "fixture-manifest.json").read_text(encoding="utf-8")
        )
        adapter_record = get_adapter_spec(model_id).to_dict(model_id)
        record["adapter"] = adapter_record
        model_root = work_root / model_id
        if model_root.exists() and any(model_root.iterdir()):
            raise FileExistsError(f"V2 model workspace must be empty: {model_root}")
        model_root.mkdir(parents=True, exist_ok=True)
        base = load_experiment_config(config_path)
        train_output = model_root / "train"
        train_config = _action_config(
            base,
            dataset=fixture_spec.name,
            output_dir=train_output,
            device=entry["device"],
            training_seed=int(plan["training_seed"]),
            sample_seed=None,
        )
        train_config.train.enabled = True
        train_config.sample.enabled = False
        train_started = time.monotonic()
        train_bundle = run_action(train_config, "train", repo_root=repo_root, dataset_spec=fixture_spec)
        save_run_context(
            build_run_context(train_config, repo_root=repo_root, dataset_spec=fixture_spec),
            train_output,
        )
        train_manifest = _tree_manifest(train_output, exclusions=COPY_EXCLUSIONS)
        record["train"] = {
            "status": "pass",
            "elapsed_seconds": time.monotonic() - train_started,
            "output_dir": str(train_output),
            "bundle": train_bundle.to_dict(),
            "artifact_manifest": train_manifest,
            "artifact_manifest_sha256": _sha256_json(train_manifest),
        }
        samples: list[dict[str, Any]] = []
        for seed in plan["generation_seeds"]:
            sample_output = model_root / f"sample-seed-{seed}"
            sample_config = _action_config(
                base,
                dataset=fixture_spec.name,
                output_dir=sample_output,
                device=entry["device"],
                training_seed=int(plan["training_seed"]),
                sample_seed=int(seed),
            )
            sample_config.train.enabled = False
            sample_config.sample.enabled = True
            save_run_context(
                build_run_context(sample_config, repo_root=repo_root, dataset_spec=fixture_spec),
                sample_output,
            )
            copied_manifest = _copy_training_artifacts(train_output, sample_output)
            sample_started = time.monotonic()
            bundle = run_action(sample_config, "sample", repo_root=repo_root, dataset_spec=fixture_spec)
            save_run_context(
                build_run_context(sample_config, repo_root=repo_root, dataset_spec=fixture_spec),
                sample_output,
            )
            _assert_manifest_unchanged(sample_output, copied_manifest)
            if bundle.generated_sample_path is None:
                raise PipelineV2Error(f"{model_id} did not expose generated_sample_path")
            sample = _validate_sample(bundle.generated_sample_path, fixture_spec, int(sample_config.sample.num_samples))
            sample.update(
                {
                    "seed": seed,
                    "elapsed_seconds": time.monotonic() - sample_started,
                    "output_dir": str(sample_output),
                    "copied_training_manifest_sha256": _sha256_json(copied_manifest),
                    "training_artifacts_unchanged": True,
                }
            )
            samples.append(sample)
        if samples[0]["sha256"] == samples[1]["sha256"]:
            raise PipelineV2Error("Distinct declared generation seeds produced byte-identical sample tables")
        record["samples"] = samples
        record["seed_outputs_distinct"] = True
        tracked_changes = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
        if tracked_changes:
            raise PipelineV2Error(f"Authoritative probe modified tracked repository files: {tracked_changes}")
        record["tracked_repository_unchanged"] = True
        record["status"] = "probe-pass-evaluation-pending"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "fail"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    record["completed_at"] = _utc_now()
    record["elapsed_seconds"] = time.monotonic() - started
    record_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(record_path, record)
    if record["status"] == "fail":
        raise PipelineV2Error(f"{model_id} V2 probe failed; inspect {record_path}")
    return record


def _dataset_from_record(payload: dict[str, Any]) -> DatasetSpec:
    data = dict(payload["dataset_spec"])
    for key in ("metadata_path", "train_data_path", "val_data_path", "test_data_path"):
        data[key] = None if data[key] is None else Path(data[key])
    return DatasetSpec(**data)


def finalize_probe(
    repo_root: Path,
    plan_path: Path,
    probe_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan = _load_plan(plan_path.resolve())
    probe = json.loads(probe_path.resolve().read_text(encoding="utf-8"))
    if probe.get("status") != "probe-pass-evaluation-pending":
        raise PipelineV2Error(f"Cannot finalize a non-passing probe: {probe_path}")
    if probe.get("repository_commit") != _git(repo_root, "rev-parse", "HEAD"):
        raise PipelineV2Error("Probe commit differs from the current repository commit")
    entry = _model_entry(plan, probe["model_id"])
    environment = _environment(repo_root, pip_check_waivers=entry.get("pip_check_waivers", []))
    dataset_spec = _dataset_from_record(probe)
    base = load_experiment_config(repo_root / entry["config_path"])
    first_sample = probe["samples"][0]
    config = _action_config(
        base,
        dataset=dataset_spec.name,
        output_dir=Path(first_sample["output_dir"]),
        device=entry["device"],
        training_seed=int(plan["training_seed"]),
        sample_seed=int(first_sample["seed"]),
    )
    evaluation = plan["central_evaluation"]
    config.train.enabled = False
    config.sample.enabled = False
    config.evaluation = EvaluationConfig(
        enabled=True,
        protocol=evaluation["protocol"],
        dataset_profile_path=str((repo_root / evaluation["dataset_profile_path"]).resolve()),
        reference_path=str(dataset_spec.train_data_path.resolve()),
        comparison_track=evaluation["comparison_track"],
        extra={"sample_path": first_sample["path"]},
    )
    started = time.monotonic()
    bundle = run_action(config, "evaluate", repo_root=repo_root, dataset_spec=dataset_spec)
    if bundle.evaluation_bundle_path is None:
        raise PipelineV2Error("Central evaluation did not expose an evaluation bundle path")
    validation = validate_result_bundle(bundle.evaluation_bundle_path)
    result = deepcopy(probe)
    result["status"] = "pass"
    result["central_evaluation"] = {
        "status": "pass",
        "protocol": evaluation["protocol"],
        "elapsed_seconds": time.monotonic() - started,
        "bundle_path": str(bundle.evaluation_bundle_path.resolve()),
        "bundle_id": validation.bundle_id,
        "finalization_status": validation.finalization_status,
        "present_files": validation.present_files,
        "validation": validation.to_dict(),
        "environment": environment,
    }
    result["finalized_at"] = _utc_now()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(evidence_path, result)
    return result


def aggregate_evidence(repo_root: Path, plan_path: Path, evidence_dir: Path, output: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan = _load_plan(plan_path.resolve())
    records: list[dict[str, Any]] = []
    for entry in plan["models"]:
        path = evidence_dir / f"{entry['model_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("model_id") != entry["model_id"]:
            raise PipelineV2Error(f"Missing or non-passing V2 evidence for {entry['model_id']}: {path}")
        records.append(
            {
                "model_id": entry["model_id"],
                "state": "minimal-real-passed",
                "evidence": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
                "train_seconds": payload["train"]["elapsed_seconds"],
                "sample_seconds": [row["elapsed_seconds"] for row in payload["samples"]],
                "sample_sha256": [row["sha256"] for row in payload["samples"]],
                "central_bundle_id": payload["central_evaluation"]["bundle_id"],
            }
        )
    inventory = records + deepcopy(plan["already_passed"]) + deepcopy(plan["external_blocks"])
    if len(inventory) != 21 or len({row["model_id"] for row in inventory}) != 21:
        raise PipelineV2Error("Aggregated V2 inventory must cover exactly 21 unique models")
    aggregate = {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": "pass-with-one-external-block",
        "repository_commit": _git(repo_root, "rev-parse", "HEAD"),
        "claim_boundary": plan["claim_boundary"],
        "plan": {
            "path": plan_path.resolve().relative_to(repo_root).as_posix(),
            "sha256": sha256_file(plan_path.resolve()),
        },
        "summary": {
            "registered_models": 21,
            "new_minimal_real_passed": len(records),
            "prior_representative_real_passed": len(plan["already_passed"]),
            "externally_blocked": len(plan["external_blocks"]),
            "failed": 0,
        },
        "models": inventory,
        "generated_at": _utc_now(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--repo-root", type=Path, default=Path.cwd())
    probe.add_argument("--plan", type=Path, required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--work-root", type=Path, required=True)
    probe.add_argument("--record", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", type=Path, default=Path.cwd())
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--probe", type=Path, required=True)
    finalize.add_argument("--evidence", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--repo-root", type=Path, default=Path.cwd())
    aggregate.add_argument("--plan", type=Path, required=True)
    aggregate.add_argument("--evidence-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        run_probe(args.repo_root, args.plan, args.model, args.work_root, args.record)
    elif args.command == "finalize":
        finalize_probe(args.repo_root, args.plan, args.probe, args.evidence)
    else:
        aggregate_evidence(args.repo_root, args.plan, args.evidence_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
