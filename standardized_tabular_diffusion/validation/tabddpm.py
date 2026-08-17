from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.tabddpm import TabDDPMAdapter, build_tabddpm_environment
from standardized_tabular_diffusion.runtime_contracts import dataset_content_identity

PROTOCOL_ID = "tabddpm-native-parity-v2"
MANIFEST_RELATIVE_PATH = Path(
    "standardized_tabular_diffusion/resources/upstream/tabddpm-source-manifest.json"
)
UPSTREAM_ENTRYPOINT = Path("scripts/pipeline.py")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_lf(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"
    return _sha256_bytes(normalized)


def verify_sources(repo_root: Path) -> dict[str, Any]:
    """Fail closed unless the local sources match the predeclared authorities."""

    manifest_path = repo_root / MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream_root = repo_root / "TabDDPM-main"
    failures: list[dict[str, str]] = []

    for record in manifest["files"]:
        path = upstream_root / record["path"]
        actual = "missing" if not path.is_file() else _sha256_lf(path)
        if actual != record["sha256_lf"]:
            failures.append({"path": str(path), "expected": record["sha256_lf"], "actual": actual})

    libzero = manifest["dependencies"]["libzero"]
    for record in libzero["vendored_modules"]:
        path = upstream_root / record["path"]
        actual = "missing" if not path.is_file() else _sha256_file(path)
        if actual != record["sha256"]:
            failures.append({"path": str(path), "expected": record["sha256"], "actual": actual})

    license_path = repo_root / libzero["license_path"]
    actual_license = "missing" if not license_path.is_file() else _sha256_lf(license_path)
    if actual_license != libzero["license_sha256_lf"]:
        failures.append(
            {
                "path": str(license_path),
                "expected": libzero["license_sha256_lf"],
                "actual": actual_license,
            }
        )

    if failures:
        raise RuntimeError(f"TabDDPM source-integrity validation failed: {json.dumps(failures, indent=2)}")

    return {
        "manifest_path": str(MANIFEST_RELATIVE_PATH),
        "manifest_sha256": _sha256_file(manifest_path),
        "upstream_files_verified": len(manifest["files"]),
        "libzero_modules_verified": len(libzero["vendored_modules"]),
        "upstream_commit": manifest["upstream_commit"],
        "libzero_distribution_sha256": libzero["distribution_sha256"],
    }


def _write_fixture(data_dir: Path) -> tuple[dict[str, Any], DatasetSpec]:
    import numpy as np

    data_dir.mkdir(parents=True)
    features = {
        "train": np.array(
            [[float(i), float((i * 3) % 11), float((i * i) % 7)] for i in range(24)], dtype=np.float32
        ),
        "val": np.array([[float(i), float((i * 2) % 9), float((i + 3) % 7)] for i in range(8)], dtype=np.float32),
        "test": np.array([[float(i), float((i * 5) % 13), float((i + 1) % 7)] for i in range(8)], dtype=np.float32),
    }
    targets = {
        "train": np.array([i % 2 for i in range(24)], dtype=np.int64),
        "val": np.array([i % 2 for i in range(8)], dtype=np.int64),
        "test": np.array([i % 2 for i in range(8)], dtype=np.int64),
    }
    for split in ("train", "val", "test"):
        np.save(data_dir / f"X_num_{split}.npy", features[split])
        np.save(data_dir / f"y_{split}.npy", targets[split])
        header = "feature_0,feature_1,feature_2,target\n"
        rows = "".join(
            f"{values[0]},{values[1]},{values[2]},{int(target)}\n"
            for values, target in zip(features[split], targets[split])
        )
        (data_dir / f"{split}.csv").write_text(header + rows, encoding="utf-8")
    info = {"name": "tabddpm-parity-fixture", "task_type": "binclass", "n_classes": 2}
    (data_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    summary = {
        "kind": "deterministic-numeric-binary-classification",
        "train_rows": 24,
        "validation_rows": 8,
        "test_rows": 8,
        "numerical_features": 3,
        "categorical_features": 0,
    }
    dataset_spec = DatasetSpec(
        name="tabddpm-parity-fixture",
        task_type="classification",
        column_names=["feature_0", "feature_1", "feature_2", "target"],
        numerical_columns=["feature_0", "feature_1", "feature_2"],
        categorical_columns=[],
        target_columns=["target"],
        metadata_path=data_dir / "info.json",
        train_data_path=data_dir / "train.csv",
        val_data_path=data_dir / "val.csv",
        test_data_path=data_dir / "test.csv",
        provenance=["deterministic-tabddpm-native-parity-fixture"],
        extra={
            "column_info": {
                "feature_0": "float",
                "feature_1": "float",
                "feature_2": "float",
                "target": "int",
            }
        },
    )
    return summary, dataset_spec


def _toml_string(value: Path | str) -> str:
    return json.dumps(str(value))


def _write_config(
    path: Path,
    *,
    data_dir: Path,
    parent_dir: Path,
    training_seed: int,
    sampling_seed: int,
) -> None:
    config = f"""seed = {training_seed}
parent_dir = {_toml_string(parent_dir)}
real_data_path = {_toml_string(data_dir)}
num_numerical_features = 3
model_type = "mlp"
device = "cpu"

[model_params]
num_classes = 2
is_y_cond = true

[model_params.rtdl_params]
d_layers = [16, 16]
dropout = 0.0

[diffusion_params]
num_timesteps = 4
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 3
lr = 0.001
weight_decay = 0.0
batch_size = 8
seed = {training_seed}

[train.T]
seed = {training_seed}
normalization = "minmax"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = 12
batch_size = 6
seed = {sampling_seed}

[eval.type]
eval_model = "simple"
eval_type = "synthetic"

[eval.T]
seed = {training_seed}
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"
"""
    path.write_text(config, encoding="utf-8")


def _run_native(upstream_root: Path, config_path: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    environment = os.environ.copy()
    environment.update(build_tabddpm_environment(upstream_root))
    for flag in ("--train", "--sample"):
        command = [sys.executable, str(UPSTREAM_ENTRYPOINT), "--config", str(config_path), flag]
        subprocess.run(command, cwd=upstream_root, check=True, env=environment)
        commands.append(command)
    return commands


def _run_adapter(
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    dataset_spec: DatasetSpec,
    *,
    training_seed: int,
    sampling_seed: int,
) -> dict[str, Any]:
    adapter = TabDDPMAdapter(repo_root)
    dataset_extra = {
        "dataset_spec": dataset_spec.to_dict(),
        "dataset_identity": dataset_content_identity(dataset_spec),
    }
    train_spec = RunSpec(
        model="tabddpm",
        dataset=dataset_spec.name,
        output_dir=output_dir,
        device="cpu",
        seed=training_seed,
        upstream_config_path=config_path,
        extra=dict(dataset_extra),
    )
    sample_spec = RunSpec(
        model="tabddpm",
        dataset=dataset_spec.name,
        output_dir=output_dir,
        device="cpu",
        seed=sampling_seed,
        num_samples=12,
        upstream_config_path=config_path,
        extra=dict(dataset_extra),
    )
    snapshots = output_dir / "validation-manifests"
    train_bundle = adapter.train(train_spec)
    train_manifest = train_bundle.output_dir / "artifacts.json"
    train_snapshot = snapshots / "train-artifacts.json"
    atomic_write_bytes(train_snapshot, train_manifest.read_bytes())
    sample_bundle = adapter.sample(sample_spec)
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("TabDDPM adapter did not expose its decoded generated table.")
    sample_manifest = sample_bundle.output_dir / "artifacts.json"
    sample_snapshot = snapshots / "sample-artifacts.json"
    atomic_write_bytes(sample_snapshot, sample_manifest.read_bytes())
    manifests = [train_snapshot, sample_snapshot]
    manifests_valid = True
    for action, path in zip(("train", "sample"), manifests):
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_sample_path = None if action == "train" else str(sample_bundle.generated_sample_path)
        valid = (
            payload["model"] == "tabddpm"
            and payload["dataset"] == dataset_spec.name
            and payload["output_dir"] == str(output_dir)
            and payload["generated_sample_path"] == expected_sample_path
        )
        manifests_valid = manifests_valid and valid
    runtime_root = adapter._runtime_root(train_spec)
    return {
        "commands": [
            [
                sys.executable,
                str(UPSTREAM_ENTRYPOINT),
                "--config",
                str(runtime_root / f"config-train-seed-{training_seed}.toml"),
                "--train",
            ],
            [
                sys.executable,
                str(UPSTREAM_ENTRYPOINT),
                "--config",
                str(runtime_root / f"config-sample-seed-{sampling_seed}.toml"),
                "--sample",
            ],
        ],
        "generated_sample_path": str(sample_bundle.generated_sample_path),
        "manifests": [str(path) for path in manifests],
        "manifests_valid": manifests_valid,
        "output_dir": str(output_dir),
        "runtime_root": str(runtime_root),
        "runtime_configs": {
            "train": str(runtime_root / f"config-train-seed-{training_seed}.toml"),
            "sample": str(runtime_root / f"config-sample-seed-{sampling_seed}.toml"),
        },
    }


def _compare_configs(native_config: Path, adapter_config: Path, *, action: str) -> bool:
    with native_config.open("rb") as stream:
        native = tomllib.load(stream)
    with adapter_config.open("rb") as stream:
        adapter = tomllib.load(stream)
    for payload in (native, adapter):
        payload.pop("parent_dir")
        payload.pop("real_data_path")
        if action == "sample":
            # The global seed is consumed only by evaluation; sampling uses sample.seed.
            payload.pop("seed")
    return native == adapter


def _inspect_decoded_sample(path: Path, dataset_spec: DatasetSpec, expected_rows: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    train = pd.read_csv(dataset_spec.train_data_path)
    columns_exact = list(frame.columns) == dataset_spec.column_names
    rows_exact = len(frame) == expected_rows
    missing_values = int(frame.isna().sum().sum())
    numerical_finite = bool(
        np.isfinite(frame[dataset_spec.numerical_columns].to_numpy(dtype=float)).all()
    )
    target = dataset_spec.target_columns[0]
    target_domain_valid = set(frame[target]).issubset(set(train[target]))
    return {
        "columns_exact": columns_exact,
        "rows": len(frame),
        "rows_exact": rows_exact,
        "missing_values": missing_values,
        "numerical_finite": numerical_finite,
        "target_domain_valid": target_domain_valid,
        "sha256": sha256_file(path),
        "valid": (
            columns_exact
            and rows_exact
            and missing_values == 0
            and numerical_finite
            and target_domain_valid
        ),
    }


def _compare_state_dicts(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import torch

    native = torch.load(native_path, map_location="cpu")
    adapter = torch.load(adapter_path, map_location="cpu")
    keys_equal = list(native) == list(adapter)
    tensor_mismatches = [key for key in native if key not in adapter or not torch.equal(native[key], adapter[key])]
    return {
        "keys_equal": keys_equal,
        "tensor_values_exact": not tensor_mismatches,
        "tensor_mismatches": tensor_mismatches,
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _compare_arrays(native_root: Path, adapter_root: Path) -> dict[str, Any]:
    import numpy as np

    native_names = sorted(path.name for path in native_root.glob("*.npy"))
    adapter_names = sorted(path.name for path in adapter_root.glob("*.npy"))
    if native_names != adapter_names:
        raise AssertionError(f"Generated array inventories differ: {native_names} != {adapter_names}")
    records: dict[str, Any] = {}
    for name in native_names:
        native = np.load(native_root / name, allow_pickle=False)
        adapter = np.load(adapter_root / name, allow_pickle=False)
        exact = np.array_equal(native, adapter)
        finite = bool(np.isfinite(native).all()) if np.issubdtype(native.dtype, np.number) else True
        records[name] = {
            "exact": exact,
            "finite": finite,
            "shape": list(native.shape),
            "dtype": str(native.dtype),
            "native_sha256": _sha256_file(native_root / name),
            "adapter_sha256": _sha256_file(adapter_root / name),
        }
    return records


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "vendored-or-unavailable"


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(repo_root: Path, output_dir: Path, evidence_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_evidence = verify_sources(repo_root)
    fixture, dataset_spec = _write_fixture(output_dir / "data")
    seed_cases = [(0, 23), (17, 47), (101, 89)]
    cases: list[dict[str, Any]] = []
    for index, (training_seed, sampling_seed) in enumerate(seed_cases, start=1):
        case_root = output_dir / f"case-{index:02d}"
        native_root = case_root / "native"
        adapter_output = case_root / "adapter-run"
        native_config = case_root / "native.toml"
        adapter_config = case_root / "adapter.toml"
        case_root.mkdir(parents=True)
        _write_config(
            native_config,
            data_dir=output_dir / "data",
            parent_dir=native_root,
            training_seed=training_seed,
            sampling_seed=sampling_seed,
        )
        _write_config(
            adapter_config,
            data_dir=output_dir / "data",
            parent_dir=adapter_output / "source-config-placeholder",
            training_seed=training_seed,
            sampling_seed=sampling_seed,
        )
        native_commands = _run_native(repo_root / "TabDDPM-main", native_config)
        adapter_run = _run_adapter(
            repo_root,
            adapter_config,
            adapter_output,
            dataset_spec,
            training_seed=training_seed,
            sampling_seed=sampling_seed,
        )
        adapter_root = Path(adapter_run["runtime_root"])
        train_config_exact = _compare_configs(
            native_config,
            Path(adapter_run["runtime_configs"]["train"]),
            action="train",
        )
        sample_config_exact = _compare_configs(
            native_config,
            Path(adapter_run["runtime_configs"]["sample"]),
            action="sample",
        )
        model = _compare_state_dicts(native_root / "model.pt", adapter_root / "model.pt")
        ema_model = _compare_state_dicts(native_root / "model_ema.pt", adapter_root / "model_ema.pt")
        arrays = _compare_arrays(native_root, adapter_root)
        loss_exact = (native_root / "loss.csv").read_bytes() == (adapter_root / "loss.csv").read_bytes()
        sample_rows = int(arrays["y_train.npy"]["shape"][0])
        decoded_sample = _inspect_decoded_sample(
            Path(adapter_run["generated_sample_path"]),
            dataset_spec,
            expected_rows=12,
        )
        output_isolated = adapter_root.is_relative_to(adapter_output)
        case_passed = (
            train_config_exact
            and sample_config_exact
            and model["keys_equal"]
            and model["tensor_values_exact"]
            and ema_model["keys_equal"]
            and ema_model["tensor_values_exact"]
            and all(record["exact"] and record["finite"] for record in arrays.values())
            and loss_exact
            and sample_rows == 12
            and decoded_sample["valid"]
            and adapter_run["manifests_valid"]
            and output_isolated
        )
        cases.append(
            {
                "case": index,
                "status": "pass" if case_passed else "fail",
                "training_seed": training_seed,
                "sampling_seed": sampling_seed,
                "native_commands": native_commands,
                "adapter_commands": adapter_run["commands"],
                "comparisons": {
                    "effective_train_config_exact": train_config_exact,
                    "effective_sample_config_exact": sample_config_exact,
                    "model": model,
                    "ema_model": ema_model,
                    "generated_arrays": arrays,
                    "loss_csv_exact": loss_exact,
                    "sample_rows": sample_rows,
                    "decoded_sample": decoded_sample,
                    "adapter_manifests_valid": adapter_run["manifests_valid"],
                    "output_isolated": output_isolated,
                },
                "adapter_manifests": adapter_run["manifests"],
            }
        )
    source_after = verify_sources(repo_root)
    source_remained_exact = source_after == source_evidence
    passed = source_remained_exact and all(case["status"] == "pass" for case in cases)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabddpm",
        "status": "pass" if passed else "fail",
        "comparison_policy": {
            "deterministic_config_mapping": (
                "exact effective train/sample runtime configuration after excluding only output/data paths "
                "and the sample-irrelevant global evaluation seed"
            ),
            "dataset_binding": "embedded DatasetSpec plus byte identity; run-owned native mirror",
            "model_state": "exact tensor equality",
            "generated_arrays": "exact element equality",
            "decoded_table": "exact schema/row count, finite numerical values, no missing values, valid target domain",
            "numeric_integrity": "all generated numeric values must be finite",
            "seed_cases": [
                {"training": training_seed, "sampling": sampling_seed}
                for training_seed, sampling_seed in seed_cases
            ],
        },
        "source": source_evidence,
        "source_remained_exact": source_remained_exact,
        "repository_commit": _repository_commit(repo_root),
        "environment_lock": {
            "path": "requirements-tabddpm-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-tabddpm-validation.txt"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _version("torch"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit_learn": _version("scikit-learn"),
            "rtdl": _version("rtdl"),
            "libzero": "0.0.8-exact-vendored-source",
        },
        "fixture": fixture,
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("TabDDPM native-parity protocol failed; inspect the evidence record.")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TabDDPM native-parity validation protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": PROTOCOL_ID,
                        "model_id": "tabddpm",
                        "status": "fail",
                        "repository_commit": _repository_commit(args.repo_root.resolve()),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
