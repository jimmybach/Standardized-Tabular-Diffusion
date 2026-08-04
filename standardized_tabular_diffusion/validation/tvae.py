"""Executable native-parity protocol for TVAE from the official CTGAN package."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.sample_baselines import TVAEAdapter
from standardized_tabular_diffusion.validation import ctgan as package_validation

PROTOCOL_ID = "tvae-native-parity-v1"
PACKAGE_NAME = package_validation.PACKAGE_NAME
PACKAGE_VERSION = package_validation.PACKAGE_VERSION
WHEEL_FILENAME = package_validation.WHEEL_FILENAME
WHEEL_SHA256 = package_validation.WHEEL_SHA256
UPSTREAM_REPOSITORY = package_validation.UPSTREAM_REPOSITORY
UPSTREAM_TAG = package_validation.UPSTREAM_TAG
UPSTREAM_COMMIT = package_validation.UPSTREAM_COMMIT
UPSTREAM_TREE = package_validation.UPSTREAM_TREE
LICENSE_EXPRESSION = package_validation.LICENSE_EXPRESSION
EXPECTED_SAMPLE_ROWS = 12
SEED_CASES = (0, 19, 73)
EXPECTED_DISTRIBUTION_VERSIONS = package_validation.EXPECTED_DISTRIBUTION_VERSIONS


def _write_fixture(root: Path) -> tuple[pd.DataFrame, DatasetSpec, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    rows = 40
    row_ids = np.arange(rows)
    frame = pd.DataFrame(
        {
            "continuous": np.round(np.cos(row_ids / 4.0) + row_ids / 25.0, 6),
            "count": ((row_ids * 5) % 19).astype(int),
            "segment": np.asarray(["alpha", "beta", "gamma", "delta"])[row_ids % 4],
            "label": np.where((row_ids % 6) < 3, "positive", "negative"),
        }
    )
    train_path = root / "train.csv"
    metadata_path = root / "info.json"
    frame.to_csv(train_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "name": "tvae-parity-fixture",
                "task_type": "binclass",
                "column_names": list(frame.columns),
                "num_col_idx": [0, 1],
                "cat_col_idx": [2],
                "target_col_idx": [3],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_spec = DatasetSpec(
        name="tvae-parity-fixture",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=["continuous", "count"],
        categorical_columns=["segment"],
        target_columns=["label"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )
    return frame, dataset_spec, {
        "kind": "deterministic-mixed-type-binary-classification",
        "rows": rows,
        "columns": list(frame.columns),
        "numerical_columns": dataset_spec.numerical_columns,
        "categorical_columns": dataset_spec.categorical_columns,
        "target_columns": dataset_spec.target_columns,
        "missing_values": int(frame.isna().sum().sum()),
        "train_csv_sha256": package_validation._sha256_file(train_path),
    }


def _constructor_kwargs() -> dict[str, Any]:
    return {
        "embedding_dim": 16,
        "compress_dims": (16,),
        "decompress_dims": (16,),
        "l2scale": 1e-5,
        "batch_size": 20,
        "epochs": 1,
        "loss_factor": 2.0,
        "enable_gpu": False,
        "verbose": False,
    }


def _adapter_extra(dataset_spec: DatasetSpec) -> dict[str, Any]:
    kwargs = _constructor_kwargs()
    kwargs.pop("enable_gpu")
    kwargs["compress_dims"] = list(kwargs["compress_dims"])
    kwargs["decompress_dims"] = list(kwargs["decompress_dims"])
    kwargs["dataset_spec"] = dataset_spec.to_dict()
    return kwargs


def _load_official(path: Path):
    from ctgan import TVAE

    model = TVAE.load(path)
    model.set_device("cpu")
    return model


def _run_native(
    frame: pd.DataFrame,
    discrete_columns: list[str],
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    from ctgan import TVAE

    output_dir.mkdir(parents=True, exist_ok=True)
    model = TVAE(**_constructor_kwargs())
    model.set_random_state(seed)
    model.fit(frame.copy(), discrete_columns=discrete_columns)
    checkpoint = output_dir / "model.pkl"
    model.save(checkpoint)
    loaded = _load_official(checkpoint)
    samples = loaded.sample(EXPECTED_SAMPLE_ROWS)
    sample_path = output_dir / "samples.csv"
    samples.to_csv(sample_path, index=False)
    return loaded, pd.read_csv(sample_path), {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": package_validation._sha256_file(checkpoint),
        "sample_path": str(sample_path),
        "sample_sha256": package_validation._sha256_file(sample_path),
    }


def _run_adapter(
    repo_root: Path,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    adapter = TVAEAdapter(repo_root)
    common = {
        "model": "tvae",
        "dataset": dataset_spec.name,
        "output_dir": output_dir,
        "device": "cpu",
        "seed": seed,
        "extra": _adapter_extra(dataset_spec),
    }
    train_bundle = adapter.train(RunSpec(**common))
    train_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    checkpoint = output_dir / adapter.checkpoint_filename
    sampled_models: list[Any] = []
    original_load_model = adapter._load_model

    def capture_loaded_model(sample_spec: RunSpec, sample_checkpoint: Path) -> Any:
        model = original_load_model(sample_spec, sample_checkpoint)
        sampled_models.append(model)
        return model

    adapter._load_model = capture_loaded_model  # type: ignore[method-assign]
    sample_bundle = adapter.sample(RunSpec(**common, num_samples=EXPECTED_SAMPLE_ROWS))
    sample_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("TVAE adapter did not declare a generated sample path")
    if len(sampled_models) != 1:
        raise AssertionError("TVAE adapter did not load exactly one model while sampling")
    samples = pd.read_csv(sample_bundle.generated_sample_path)
    loaded = sampled_models[0]
    manifests_valid = (
        train_bundle.model == "tvae"
        and train_manifest["model"] == "tvae"
        and train_manifest["dataset"] == dataset_spec.name
        and sample_manifest["generated_sample_path"] == str(sample_bundle.generated_sample_path)
    )
    return loaded, samples, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": package_validation._sha256_file(checkpoint),
        "sample_path": str(sample_bundle.generated_sample_path),
        "sample_sha256": package_validation._sha256_file(sample_bundle.generated_sample_path),
        "manifests_valid": manifests_valid,
        "upstream_root": str(adapter.upstream_root.resolve()),
    }


def _compare_decoder_state(left: Any, right: Any) -> dict[str, Any]:
    import torch

    left_state = left.decoder.state_dict()
    right_state = right.decoder.state_dict()
    keys_exact = list(left_state) == list(right_state)
    tensors_exact = keys_exact and all(torch.equal(left_state[key], right_state[key]) for key in left_state)
    finite = all(torch.isfinite(tensor).all().item() for tensor in left_state.values())
    return {
        "keys_exact": keys_exact,
        "tensor_values_exact": tensors_exact,
        "finite": bool(finite),
        "tensor_count": len(left_state),
        "sigma_finite": bool(torch.isfinite(left.decoder.sigma).all().item()),
    }


def _compare_random_states(left: Any, right: Any) -> dict[str, bool]:
    import torch

    left_numpy, left_torch = left.random_states
    right_numpy, right_torch = right.random_states
    left_numpy_state = left_numpy.get_state()
    right_numpy_state = right_numpy.get_state()
    numpy_exact = (
        left_numpy_state[0] == right_numpy_state[0]
        and np.array_equal(left_numpy_state[1], right_numpy_state[1])
        and left_numpy_state[2:] == right_numpy_state[2:]
    )
    return {
        "numpy_exact": bool(numpy_exact),
        "torch_exact": bool(torch.equal(left_torch.get_state(), right_torch.get_state())),
    }


def _compare_models(left: Any, right: Any, frame: pd.DataFrame) -> dict[str, Any]:
    left_transformed = left.transformer.transform(frame.copy())
    right_transformed = right.transformer.transform(frame.copy())
    constructor_attributes = (
        "embedding_dim",
        "compress_dims",
        "decompress_dims",
        "l2scale",
        "batch_size",
        "epochs",
        "loss_factor",
        "verbose",
        "_enable_gpu",
    )
    return {
        "constructor_exact": bool(
            all(getattr(left, name) == getattr(right, name) for name in constructor_attributes)
        ),
        "device_exact": str(left._device) == str(right._device) == "cpu",
        "decoder": _compare_decoder_state(left, right),
        "transformer_exact": bool(np.array_equal(left_transformed, right_transformed)),
        "transformed_shape": list(left_transformed.shape),
        "random_state": _compare_random_states(left, right),
        "loss_values_exact": bool(left.loss_values.equals(right.loss_values)),
    }


def _compare_samples(left: pd.DataFrame, right: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    columns_exact = list(left.columns) == list(right.columns) == list(frame.columns)
    finite_numerical = bool(
        np.isfinite(left[["continuous", "count"]].to_numpy(dtype=float)).all()
        and np.isfinite(right[["continuous", "count"]].to_numpy(dtype=float)).all()
    )
    categorical_domains_valid = all(
        set(left[column]).issubset(set(frame[column])) and set(right[column]).issubset(set(frame[column]))
        for column in ("segment", "label")
    )
    return {
        "rows": len(left),
        "columns_exact": columns_exact,
        "frame_exact": left.equals(right),
        "finite_numerical": finite_numerical,
        "categorical_domains_valid": categorical_domains_valid,
        "missing_values": int(left.isna().sum().sum() + right.isna().sum().sum()),
    }


def _case_passed(comparisons: dict[str, Any]) -> bool:
    model = comparisons["model"]
    decoder = model["decoder"]
    random_state = model["random_state"]
    samples = comparisons["samples"]
    return bool(
        comparisons["adapter_manifests_valid"]
        and comparisons["sample_bytes_exact"]
        and model["constructor_exact"]
        and model["device_exact"]
        and decoder["keys_exact"]
        and decoder["tensor_values_exact"]
        and decoder["finite"]
        and decoder["sigma_finite"]
        and model["transformer_exact"]
        and random_state["numpy_exact"]
        and random_state["torch_exact"]
        and model["loss_values_exact"]
        and samples["rows"] == EXPECTED_SAMPLE_ROWS
        and samples["columns_exact"]
        and samples["frame_exact"]
        and samples["finite_numerical"]
        and samples["categorical_domains_valid"]
        and samples["missing_values"] == 0
    )


def _verify_environment() -> dict[str, str]:
    observed = {
        name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTION_VERSIONS
    }
    normalized = {**observed, "torch": observed["torch"].split("+")[0]}
    if normalized != EXPECTED_DISTRIBUTION_VERSIONS:
        raise RuntimeError(
            "TVAE validation environment does not match its frozen lock: "
            f"expected={EXPECTED_DISTRIBUTION_VERSIONS}, observed={observed}"
        )
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative TVAE validation requires Linux and Python 3.11; "
            f"observed platform={platform.platform()!r}, python={python_version!r}"
        )
    return observed


def _verify_source(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    legacy_root = repo_root / "TabDDPM-main" / "CTGAN"
    legacy_markers = (
        legacy_root / "CTGAN" / "ctgan" / "__init__.py",
        legacy_root / "CTGAN" / "setup.py",
        legacy_root / "train_sample_tvae.py",
    )
    remaining_markers = [str(path) for path in legacy_markers if path.exists()]
    if remaining_markers:
        raise ValueError(
            "Legacy CTGAN/TVAE source or wrappers remain after migration: "
            f"{remaining_markers}"
        )
    source = package_validation.verify_package(repo_root, wheel_path)
    from ctgan import TVAE

    if TVAE.__module__ != "ctgan.synthesizers.tvae" or TVAE.__name__ != "TVAE":
        raise ValueError("Installed package does not expose the locked official TVAE class")
    source["synthesizer"] = {
        "class": TVAE.__name__,
        "module": TVAE.__module__,
        "legacy_snapshot_absent": True,
    }
    return source


def run_validation(
    repo_root: Path,
    output_dir: Path,
    evidence_path: Path,
    wheel_path: Path,
) -> dict[str, Any]:
    """Run three exact official-package-versus-adapter TVAE comparisons."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_versions = _verify_environment()
    source = _verify_source(repo_root, wheel_path.resolve())
    frame, dataset_spec, fixture = _write_fixture(output_dir / "fixture")
    discrete_columns = [*dataset_spec.categorical_columns, *dataset_spec.target_columns]
    cases: list[dict[str, Any]] = []
    for seed in SEED_CASES:
        native_model, native_samples, native_artifacts = _run_native(
            frame, discrete_columns, output_dir / f"seed-{seed}" / "native", seed
        )
        adapter_model, adapter_samples, adapter_artifacts = _run_adapter(
            repo_root, dataset_spec, output_dir / f"seed-{seed}" / "adapter", seed
        )
        comparisons = {
            "adapter_manifests_valid": adapter_artifacts["manifests_valid"],
            "model": _compare_models(native_model, adapter_model, frame),
            "samples": _compare_samples(native_samples, adapter_samples, frame),
            "sample_bytes_exact": (
                native_artifacts["sample_sha256"] == adapter_artifacts["sample_sha256"]
            ),
        }
        cases.append(
            {
                "seed": seed,
                "status": "pass" if _case_passed(comparisons) else "fail",
                "native_artifacts": native_artifacts,
                "adapter_artifacts": adapter_artifacts,
                "comparisons": comparisons,
            }
        )

    passed = all(case["status"] == "pass" for case in cases)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tvae",
        "status": "pass" if passed else "fail",
        "repository_commit": package_validation._repository_commit(repo_root),
        "source": source,
        "environment_lock": {
            "path": "requirements-tvae-validation.txt",
            "sha256": package_validation._sha256_file(
                repo_root / "requirements-tvae-validation.txt"
            ),
            "torch_install": "torch==2.3.0 from https://download.pytorch.org/whl/cpu",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **actual_versions,
        },
        "fixture": fixture,
        "runtime_config": {
            **_constructor_kwargs(),
            "compress_dims": list(_constructor_kwargs()["compress_dims"]),
            "decompress_dims": list(_constructor_kwargs()["decompress_dims"]),
            "sample_rows": EXPECTED_SAMPLE_ROWS,
        },
        "seed_cases": list(SEED_CASES),
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("TVAE native-parity protocol failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TVAE native-parity validation protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path, args.wheel_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": PROTOCOL_ID,
                        "model_id": "tvae",
                        "status": "fail",
                        "repository_commit": package_validation._repository_commit(
                            args.repo_root.resolve()
                        ),
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
