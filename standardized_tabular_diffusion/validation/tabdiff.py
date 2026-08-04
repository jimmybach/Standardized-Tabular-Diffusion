from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter

PROTOCOL_ID = "tabdiff-native-parity-v1"
MANIFEST_RELATIVE_PATH = Path(
    "standardized_tabular_diffusion/resources/upstream/tabdiff-source-manifest.json"
)
DATASET_NAME = "tabdiff_parity_dcr"
EXPERIMENT_NAME = "native-parity-v1"
EXPECTED_SAMPLE_ROWS = 12


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_lf(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"
    return _sha256_bytes(normalized)


def verify_sources(repo_root: Path) -> dict[str, Any]:
    """Fail closed unless all scoped TabDiff files match the pinned authority."""

    manifest_path = repo_root / MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream_root = repo_root / "TabDiff-main"
    failures: list[dict[str, str]] = []
    for record in manifest["files"]:
        path = upstream_root / record["path"]
        actual = "missing" if not path.is_file() else _sha256_lf(path)
        if actual != record["sha256_lf"]:
            failures.append({"path": str(path), "expected": record["sha256_lf"], "actual": actual})
    if failures:
        raise RuntimeError(f"TabDiff source-integrity validation failed: {json.dumps(failures, indent=2)}")
    return {
        "manifest_path": str(MANIFEST_RELATIVE_PATH),
        "manifest_sha256": _sha256_file(manifest_path),
        "upstream_files_verified": len(manifest["files"]),
        "upstream_commit": manifest["upstream_commit"],
        "upstream_tree": manifest["upstream_tree"],
    }


def _copy_verified_source(repo_root: Path, destination: Path) -> None:
    manifest = json.loads((repo_root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    source_root = repo_root / "TabDiff-main"
    for record in manifest["files"]:
        source = source_root / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_fixture(upstream_root: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    data_dir = upstream_root / "data" / DATASET_NAME
    synthetic_dir = upstream_root / "synthetic" / DATASET_NAME
    data_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)

    def rows(count: int, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        indices = np.arange(offset, offset + count)
        continuous = (indices * 0.37 + (indices % 3) * 0.11).astype(np.float32)
        integer = ((indices * 5) % 17).astype(np.float32)
        group = np.array([f"g{value % 3}" for value in indices])
        target = ((indices + (indices // 3)) % 2).astype(np.int64)
        numerical = np.column_stack([continuous, integer]).astype(np.float32)
        categorical = group.reshape(-1, 1)
        labels = target.reshape(-1, 1)
        frame = pd.DataFrame(
            {"0": continuous, "1": integer, "2": group, "3": target},
            columns=["0", "1", "2", "3"],
        )
        return numerical, categorical, labels, frame

    train = rows(30, 0)
    test = rows(14, 100)
    for split, payload in (("train", train), ("test", test)):
        numerical, categorical, labels, frame = payload
        np.save(data_dir / f"X_num_{split}.npy", numerical)
        np.save(data_dir / f"X_cat_{split}.npy", categorical)
        np.save(data_dir / f"y_{split}.npy", labels)
        frame.to_csv(data_dir / f"{split}.csv", index=False)
    train[3].to_csv(synthetic_dir / "real.csv", index=False)
    test[3].to_csv(synthetic_dir / "test.csv", index=False)

    identity_mapping = {str(index): index for index in range(4)}
    info = {
        "name": DATASET_NAME,
        "task_type": "binclass",
        "column_names": ["0", "1", "2", "3"],
        "num_col_idx": [0, 1],
        "cat_col_idx": [2],
        "target_col_idx": [3],
        "int_col_idx": [1],
        "int_col_idx_wrt_num": [1],
        "idx_mapping": identity_mapping,
        "inverse_idx_mapping": identity_mapping,
        "idx_name_mapping": {str(index): str(index) for index in range(4)},
        "train_num": 30,
        "test_num": 14,
        "val_num": 0,
        "n_classes": 2,
        "metadata": {
            "columns": {
                "0": {"sdtype": "numerical", "computer_representation": "Float"},
                "1": {"sdtype": "numerical", "computer_representation": "Float"},
                "2": {"sdtype": "categorical"},
                "3": {"sdtype": "categorical"},
            }
        },
    }
    (data_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {
        "kind": "deterministic-mixed-type-binary-classification",
        "dataset_name": DATASET_NAME,
        "train_rows": 30,
        "test_rows": 14,
        "numerical_features": 2,
        "categorical_features": 1,
        "target_features": 1,
        "missing_values": 0,
    }


def _configure_runtime(upstream_root: Path) -> dict[str, Any]:
    import tomli_w

    config_path = upstream_root / "tabdiff" / "configs" / "tabdiff_configs.toml"
    original_sha256 = _sha256_file(config_path)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    overrides = {
        "unimodmlp_params.num_layers": 1,
        "unimodmlp_params.factor": 4,
        "unimodmlp_params.dim_t": 64,
        "diffusion_params.num_timesteps": 4,
        "train.main.steps": 4,
        "train.main.batch_size": 32,
        "train.main.check_val_every": 2,
        "sample.batch_size": 32,
    }
    config["unimodmlp_params"].update({"num_layers": 1, "factor": 4, "dim_t": 64})
    config["diffusion_params"]["num_timesteps"] = 4
    config["train"]["main"].update({"steps": 4, "batch_size": 32, "check_val_every": 2})
    config["sample"]["batch_size"] = 32
    config_path.write_text(tomli_w.dumps(config), encoding="utf-8")
    return {
        "path": "tabdiff/configs/tabdiff_configs.toml",
        "base_sha256": original_sha256,
        "execution_sha256": _sha256_file(config_path),
        "overrides": overrides,
        "treatment": "validation-only hyperparameter override in an isolated verified source copy",
    }


def _common_command(mode: str, checkpoint_path: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--dataname",
        DATASET_NAME,
        "--mode",
        mode,
        "--exp_name",
        EXPERIMENT_NAME,
    ]
    if checkpoint_path is not None:
        command.extend(
            ["--ckpt_path", str(checkpoint_path), "--num_samples_to_generate", str(EXPECTED_SAMPLE_ROWS)]
        )
        command.extend(["--report", "--num_runs", "1"])
    command.extend(["--gpu", "-1", "--no_wandb", "--deterministic"])
    return command


def _run_command(command: list[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def _snapshot_training_outputs(upstream_root: Path, destination: Path) -> dict[str, Path]:
    checkpoint_root = upstream_root / "tabdiff" / "ckpt" / DATASET_NAME / EXPERIMENT_NAME
    result_root = upstream_root / "tabdiff" / "result" / DATASET_NAME / EXPERIMENT_NAME / "4"
    records = {
        "checkpoint": (
            checkpoint_root / "model_4.pt",
            destination / "ckpt" / "model_4.pt",
        ),
        "config": (
            checkpoint_root / "config.pkl",
            destination / "ckpt" / "config.pkl",
        ),
        "samples": (
            result_root / "samples.csv",
            destination / "training-result" / "samples.csv",
        ),
        "metrics": (
            result_root / "all_results.json",
            destination / "training-result" / "all_results.json",
        ),
    }
    copied: dict[str, Path] = {}
    for name, (source, target) in records.items():
        if not source.is_file():
            raise FileNotFoundError(f"TabDiff training did not produce {name}: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[name] = target
    return copied


def _load_config(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _normalized_cached_config(path: Path) -> Any:
    config = _jsonable(_load_config(path))
    config["model_save_path"] = "<runtime-model-output>"
    config["result_save_path"] = "<runtime-result-output>"
    return config


def _compare_checkpoints(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import torch

    native = torch.load(native_path, map_location="cpu")
    adapter = torch.load(adapter_path, map_location="cpu")
    mismatches: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                mismatches.append(path)
            return
        if isinstance(left, dict):
            if not isinstance(right, dict) or list(left) != list(right):
                mismatches.append(f"{path}.__keys__")
                return
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
            return
        if left != right:
            mismatches.append(path)

    compare(native, adapter, "checkpoint")
    return {
        "tensor_values_exact": not mismatches,
        "mismatches": mismatches,
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _compare_csv(native_path: Path, adapter_path: Path, expected_rows: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    native = pd.read_csv(native_path)
    adapter = pd.read_csv(adapter_path)
    exact_bytes = native_path.read_bytes() == adapter_path.read_bytes()
    exact_frame = native.equals(adapter)
    numerical = native[["0", "1"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    return {
        "exact_bytes": exact_bytes,
        "exact_frame": exact_frame,
        "rows": len(native),
        "columns": list(native.columns),
        "finite_numerical_values": bool(np.isfinite(numerical).all()),
        "expected_rows": expected_rows,
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


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

    native_repo = output_dir / "native-repo"
    adapter_repo = output_dir / "adapter-repo"
    native_root = native_repo / "TabDiff-main"
    adapter_root = adapter_repo / "TabDiff-main"
    _copy_verified_source(repo_root, native_root)
    _copy_verified_source(repo_root, adapter_root)
    fixture = _write_fixture(native_root)
    _write_fixture(adapter_root)
    native_runtime_config = _configure_runtime(native_root)
    adapter_runtime_config = _configure_runtime(adapter_root)
    if native_runtime_config != adapter_runtime_config:
        raise AssertionError("Native and adapter runtime config overrides differ.")

    native_train_command = _common_command("train")
    _run_command(native_train_command, native_root)
    native_training = _snapshot_training_outputs(native_root, output_dir / "native-training")
    native_sample_command = _common_command("test", native_training["checkpoint"])
    _run_command(native_sample_command, native_root)
    native_report_root = native_root / "eval" / "report_runs" / EXPERIMENT_NAME / DATASET_NAME
    native_sample = native_report_root / "all_samples" / "samples_0.csv"
    native_metrics = native_report_root / "4" / "all_results.json"

    os.environ["PYTHONHASHSEED"] = "0"
    adapter = TabDiffAdapter(adapter_repo)
    train_bundle = adapter.train(
        RunSpec(
            model="tabdiff",
            dataset=DATASET_NAME,
            output_dir=output_dir / "adapter-manifests" / "train",
            device="cpu",
            seed=0,
            extra={"deterministic": True, "exp_name": EXPERIMENT_NAME},
        )
    )
    adapter_training = _snapshot_training_outputs(adapter_root, output_dir / "adapter-training")
    sample_bundle = adapter.sample(
        RunSpec(
            model="tabdiff",
            dataset=DATASET_NAME,
            output_dir=output_dir / "adapter-manifests" / "sample",
            device="cpu",
            seed=0,
            num_samples=EXPECTED_SAMPLE_ROWS,
            checkpoint_path=adapter_training["checkpoint"],
            extra={
                "allow_unsafe_external_checkpoint": True,
                "deterministic": True,
                "exp_name": EXPERIMENT_NAME,
                "num_runs": 1,
                "report": True,
            },
        )
    )
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("TabDiff adapter did not record the generated sample path.")
    adapter_sample = sample_bundle.generated_sample_path
    adapter_metrics = adapter_root / "eval" / "report_runs" / EXPERIMENT_NAME / DATASET_NAME / "4" / "all_results.json"

    config_exact = _normalized_cached_config(native_training["config"]) == _normalized_cached_config(
        adapter_training["config"]
    )
    checkpoint = _compare_checkpoints(native_training["checkpoint"], adapter_training["checkpoint"])
    training_samples = _compare_csv(native_training["samples"], adapter_training["samples"], 30)
    generated_samples = _compare_csv(native_sample, adapter_sample, EXPECTED_SAMPLE_ROWS)
    training_metrics_exact = json.loads(native_training["metrics"].read_text()) == json.loads(
        adapter_training["metrics"].read_text()
    )
    generated_metrics_exact = json.loads(native_metrics.read_text()) == json.loads(adapter_metrics.read_text())
    manifests = [train_bundle.output_dir / "artifacts.json", sample_bundle.output_dir / "artifacts.json"]
    manifests_valid = all(
        json.loads(path.read_text(encoding="utf-8"))["model"] == "tabdiff" for path in manifests
    )
    passed = (
        config_exact
        and checkpoint["tensor_values_exact"]
        and training_samples["exact_bytes"]
        and training_samples["finite_numerical_values"]
        and generated_samples["exact_bytes"]
        and generated_samples["finite_numerical_values"]
        and generated_samples["rows"] == EXPECTED_SAMPLE_ROWS
        and generated_samples["columns"] == ["0", "1", "2", "3"]
        and training_metrics_exact
        and generated_metrics_exact
        and manifests_valid
    )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabdiff",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "source": source_evidence,
        "environment_lock": {
            "path": "requirements-tabdiff-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-tabdiff-validation.txt"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _version("torch"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit_learn": _version("scikit-learn"),
            "sdmetrics": _version("sdmetrics"),
        },
        "fixture": fixture,
        "runtime_config": native_runtime_config,
        "seed_contract": {
            "official_cli_capability": "deterministic mode fixes Python, NumPy, and PyTorch to seed 0",
            "validated_seed": 0,
            "configurable_seed_claim": False,
        },
        "native_commands": [native_train_command, native_sample_command],
        "adapter_commands": [
            _common_command("train"),
            _common_command("test", adapter_training["checkpoint"]),
        ],
        "comparisons": {
            "config_exact": config_exact,
            "checkpoint": checkpoint,
            "training_samples": training_samples,
            "generated_samples": generated_samples,
            "training_metrics_exact": training_metrics_exact,
            "generated_metrics_exact": generated_metrics_exact,
            "adapter_manifests_valid": manifests_valid,
        },
        "adapter_manifests": [str(path) for path in manifests],
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("TabDiff native-parity protocol failed; inspect the evidence record.")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TabDiff native-parity validation protocol.")
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
                        "model_id": "tabdiff",
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
