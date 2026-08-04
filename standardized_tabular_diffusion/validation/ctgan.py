"""Executable native-parity protocol for the official CTGAN package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import traceback
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.sample_baselines import CTGANAdapter

PROTOCOL_ID = "ctgan-native-parity-v1"
PACKAGE_NAME = "ctgan"
PACKAGE_VERSION = "0.12.1"
WHEEL_FILENAME = "ctgan-0.12.1-py3-none-any.whl"
WHEEL_SHA256 = "38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18"
UPSTREAM_REPOSITORY = "https://github.com/sdv-dev/CTGAN"
UPSTREAM_TAG = "v0.12.1"
UPSTREAM_COMMIT = "826da23f8f9385ad15fd206ecad691e04cb0ccdc"
UPSTREAM_TREE = "164a4e877a6db2ca51b3cd7dbb22cbc18af536cb"
LICENSE_EXPRESSION = "BUSL-1.1"
EXPECTED_SAMPLE_ROWS = 12
SEED_CASES = (0, 19, 73)
EXPECTED_DISTRIBUTION_VERSIONS = {
    "Faker": "37.12.0",
    "ctgan": "0.12.1",
    "joblib": "1.5.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "rdt": "1.20.0",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "six": "1.17.0",
    "threadpoolctl": "3.6.0",
    "torch": "2.3.0",
    "tqdm": "4.66.5",
    "tzdata": "2026.3",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_record_hash(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify_wheel(wheel_path: Path) -> dict[str, Any]:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(f"CTGAN wheel must be a regular non-symlinked file: {wheel_path}")
    if wheel_path.name != WHEEL_FILENAME:
        raise ValueError(f"Expected wheel {WHEEL_FILENAME!r}, observed {wheel_path.name!r}")
    observed_sha256 = _sha256_file(wheel_path)
    if observed_sha256 != WHEEL_SHA256:
        raise ValueError(
            f"CTGAN wheel checksum mismatch: expected={WHEEL_SHA256}, observed={observed_sha256}"
        )

    metadata_name = f"ctgan-{PACKAGE_VERSION}.dist-info/METADATA"
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe path in CTGAN wheel: {name!r}")
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        if metadata.get("Name") != PACKAGE_NAME or metadata.get("Version") != PACKAGE_VERSION:
            raise ValueError("CTGAN wheel metadata does not match the locked package identity")
        if metadata.get("License-Expression") != LICENSE_EXPRESSION:
            raise ValueError("CTGAN wheel license expression differs from the locked BUSL-1.1 release")

    return {
        "filename": wheel_path.name,
        "sha256": observed_sha256,
        "bytes": wheel_path.stat().st_size,
        "archive_members": len(names),
        "license_expression": metadata.get("License-Expression"),
    }


def _verify_installed_distribution(repo_root: Path) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    if distribution.version != PACKAGE_VERSION:
        raise ValueError(
            f"Installed CTGAN version mismatch: expected={PACKAGE_VERSION}, observed={distribution.version}"
        )
    if distribution.metadata.get("License-Expression") != LICENSE_EXPRESSION:
        raise ValueError("Installed CTGAN metadata does not declare the locked BUSL-1.1 license")

    import ctgan

    module_path = Path(ctgan.__file__).resolve()
    resolved_root = repo_root.resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    legacy_root = (resolved_root / "TabDDPM-main" / "CTGAN" / "CTGAN").resolve()
    if not module_path.is_relative_to(distribution_root) or module_path.is_relative_to(legacy_root):
        raise ValueError(f"CTGAN resolved to repository-local source instead of the official package: {module_path}")
    if getattr(ctgan, "__version__", None) != PACKAGE_VERSION:
        raise ValueError("Imported CTGAN module version differs from installed distribution metadata")

    verified_files = 0
    for package_path in distribution.files or ():
        file_hash = package_path.hash
        if file_hash is None:
            continue
        if file_hash.mode != "sha256":
            raise ValueError(f"Unexpected installed RECORD hash mode for {package_path}: {file_hash.mode}")
        installed_path = Path(package_path.locate()).resolve()
        if not installed_path.is_file():
            raise ValueError(f"Installed CTGAN RECORD path is missing: {installed_path}")
        observed = bytes.fromhex(_sha256_file(installed_path))
        expected = _decode_record_hash(file_hash.value)
        if observed != expected:
            raise ValueError(f"Installed CTGAN RECORD checksum mismatch: {package_path}")
        verified_files += 1
    if verified_files == 0:
        raise ValueError("Installed CTGAN distribution exposed no verifiable RECORD entries")

    return {
        "version": distribution.version,
        "module_path": str(module_path),
        "distribution_root": str(distribution_root),
        "record_files_verified": verified_files,
        "license_expression": distribution.metadata.get("License-Expression"),
    }


def verify_package(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    """Fail closed unless both the downloaded wheel and installed package match the lock."""

    return {
        "authority": "method-author",
        "distribution_form": "package",
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "wheel": _verify_wheel(wheel_path.resolve()),
        "installed_distribution": _verify_installed_distribution(repo_root.resolve()),
    }


def _write_fixture(root: Path) -> tuple[pd.DataFrame, DatasetSpec, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    rows = 40
    row_ids = np.arange(rows)
    frame = pd.DataFrame(
        {
            "continuous": np.round(np.sin(row_ids / 3.0) + row_ids / 20.0, 6),
            "count": ((row_ids * 7) % 23).astype(int),
            "segment": np.asarray(["alpha", "beta", "gamma", "delta"])[row_ids % 4],
            "label": np.where((row_ids % 5) < 2, "positive", "negative"),
        }
    )
    train_path = root / "train.csv"
    metadata_path = root / "info.json"
    frame.to_csv(train_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "name": "ctgan-parity-fixture",
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
        name="ctgan-parity-fixture",
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
        "train_csv_sha256": _sha256_file(train_path),
    }


def _constructor_kwargs() -> dict[str, Any]:
    return {
        "embedding_dim": 16,
        "generator_dim": (16,),
        "discriminator_dim": (16,),
        "batch_size": 20,
        "discriminator_steps": 1,
        "log_frequency": True,
        "verbose": False,
        "epochs": 1,
        "pac": 10,
        "enable_gpu": False,
    }


def _adapter_extra(dataset_spec: DatasetSpec) -> dict[str, Any]:
    kwargs = _constructor_kwargs()
    kwargs.pop("enable_gpu")
    kwargs["generator_dim"] = list(kwargs["generator_dim"])
    kwargs["discriminator_dim"] = list(kwargs["discriminator_dim"])
    kwargs["dataset_spec"] = dataset_spec.to_dict()
    return kwargs


def _load_official(path: Path):
    from ctgan import CTGAN

    model = CTGAN.load(path)
    model.set_device("cpu")
    return model


def _run_native(
    frame: pd.DataFrame,
    discrete_columns: list[str],
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    from ctgan import CTGAN

    output_dir.mkdir(parents=True, exist_ok=True)
    model = CTGAN(**_constructor_kwargs())
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
        "checkpoint_sha256": _sha256_file(checkpoint),
        "sample_path": str(sample_path),
        "sample_sha256": _sha256_file(sample_path),
    }


def _run_adapter(
    repo_root: Path,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    adapter = CTGANAdapter(repo_root)
    common = {
        "model": "ctgan",
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
        raise AssertionError("CTGAN adapter did not declare a generated sample path")
    if len(sampled_models) != 1:
        raise AssertionError("CTGAN adapter did not load exactly one model while sampling")
    samples = pd.read_csv(sample_bundle.generated_sample_path)
    loaded = sampled_models[0]
    manifests_valid = (
        train_bundle.model == "ctgan"
        and train_manifest["model"] == "ctgan"
        and train_manifest["dataset"] == dataset_spec.name
        and sample_manifest["generated_sample_path"] == str(sample_bundle.generated_sample_path)
    )
    return loaded, samples, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "sample_path": str(sample_bundle.generated_sample_path),
        "sample_sha256": _sha256_file(sample_bundle.generated_sample_path),
        "manifests_valid": manifests_valid,
        "upstream_root": str(adapter.upstream_root.resolve()),
    }


def _compare_tensor_state(left: Any, right: Any) -> dict[str, Any]:
    import torch

    left_state = left._generator.state_dict()
    right_state = right._generator.state_dict()
    keys_exact = list(left_state) == list(right_state)
    tensors_exact = keys_exact and all(torch.equal(left_state[key], right_state[key]) for key in left_state)
    finite = all(torch.isfinite(tensor).all().item() for tensor in left_state.values())
    return {
        "keys_exact": keys_exact,
        "tensor_values_exact": tensors_exact,
        "finite": bool(finite),
        "tensor_count": len(left_state),
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


def _compare_sampler(left: Any, right: Any) -> dict[str, Any]:
    array_attributes = (
        "_discrete_column_matrix_st",
        "_discrete_column_cond_st",
        "_discrete_column_n_category",
        "_discrete_column_category_prob",
    )
    arrays_exact = all(
        np.array_equal(getattr(left._data_sampler, name), getattr(right._data_sampler, name))
        for name in array_attributes
    )
    row_ids_exact = all(
        np.array_equal(left_ids, right_ids)
        for left_column, right_column in zip(
            left._data_sampler._rid_by_cat_cols,
            right._data_sampler._rid_by_cat_cols,
            strict=True,
        )
        for left_ids, right_ids in zip(left_column, right_column, strict=True)
    )
    scalars_exact = all(
        getattr(left._data_sampler, name) == getattr(right._data_sampler, name)
        for name in ("_data_length", "_n_discrete_columns", "_n_categories")
    )
    return {
        "arrays_exact": bool(arrays_exact),
        "row_ids_exact": bool(row_ids_exact),
        "scalars_exact": bool(scalars_exact),
    }


def _compare_models(left: Any, right: Any, frame: pd.DataFrame) -> dict[str, Any]:
    left_transformed = left._transformer.transform(frame.copy())
    right_transformed = right._transformer.transform(frame.copy())
    transformer_exact = np.array_equal(left_transformed, right_transformed)
    constructor_attributes = (
        "_embedding_dim",
        "_generator_dim",
        "_discriminator_dim",
        "_generator_lr",
        "_generator_decay",
        "_discriminator_lr",
        "_discriminator_decay",
        "_batch_size",
        "_discriminator_steps",
        "_log_frequency",
        "_epochs",
        "pac",
    )
    constructor_exact = all(getattr(left, name) == getattr(right, name) for name in constructor_attributes)
    loss_exact = left.loss_values.equals(right.loss_values)
    return {
        "constructor_exact": bool(constructor_exact),
        "generator": _compare_tensor_state(left, right),
        "transformer_exact": bool(transformer_exact),
        "transformed_shape": list(left_transformed.shape),
        "data_sampler": _compare_sampler(left, right),
        "random_state": _compare_random_states(left, right),
        "loss_values_exact": bool(loss_exact),
    }


def _compare_samples(left: pd.DataFrame, right: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    columns_exact = list(left.columns) == list(right.columns) == list(frame.columns)
    frame_exact = left.equals(right)
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
        "frame_exact": frame_exact,
        "finite_numerical": finite_numerical,
        "categorical_domains_valid": categorical_domains_valid,
        "missing_values": int(left.isna().sum().sum() + right.isna().sum().sum()),
    }


def _case_passed(comparisons: dict[str, Any]) -> bool:
    model = comparisons["model"]
    sampler = model["data_sampler"]
    random_state = model["random_state"]
    samples = comparisons["samples"]
    return bool(
        comparisons["adapter_manifests_valid"]
        and comparisons["sample_bytes_exact"]
        and model["constructor_exact"]
        and model["generator"]["keys_exact"]
        and model["generator"]["tensor_values_exact"]
        and model["generator"]["finite"]
        and model["transformer_exact"]
        and sampler["arrays_exact"]
        and sampler["row_ids_exact"]
        and sampler["scalars_exact"]
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


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _verify_environment() -> dict[str, str]:
    observed = {name: _version(name) for name in EXPECTED_DISTRIBUTION_VERSIONS}
    normalized = {**observed, "torch": observed["torch"].split("+")[0]}
    if normalized != EXPECTED_DISTRIBUTION_VERSIONS:
        raise RuntimeError(
            "CTGAN validation environment does not match its frozen lock: "
            f"expected={EXPECTED_DISTRIBUTION_VERSIONS}, observed={observed}"
        )
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative CTGAN validation requires Linux and Python 3.11; "
            f"observed platform={platform.platform()!r}, python={python_version!r}"
        )
    return observed


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(
    repo_root: Path,
    output_dir: Path,
    evidence_path: Path,
    wheel_path: Path,
) -> dict[str, Any]:
    """Run three exact official-package-versus-adapter comparisons."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_versions = _verify_environment()
    source = verify_package(repo_root, wheel_path.resolve())
    frame, dataset_spec, fixture = _write_fixture(output_dir / "fixture")
    discrete_columns = [*dataset_spec.categorical_columns, *dataset_spec.target_columns]
    cases: list[dict[str, Any]] = []
    for seed in SEED_CASES:
        native_model, native_samples, native_artifacts = _run_native(
            frame,
            discrete_columns,
            output_dir / f"seed-{seed}" / "native",
            seed,
        )
        adapter_model, adapter_samples, adapter_artifacts = _run_adapter(
            repo_root,
            dataset_spec,
            output_dir / f"seed-{seed}" / "adapter",
            seed,
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
    environment = {"platform": platform.platform(), "python": platform.python_version(), **actual_versions}
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "ctgan",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "environment_lock": {
            "path": "requirements-ctgan-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-ctgan-validation.txt"),
            "torch_install": "torch==2.3.0 from https://download.pytorch.org/whl/cpu",
        },
        "environment": environment,
        "fixture": fixture,
        "runtime_config": {
            **_constructor_kwargs(),
            "generator_dim": list(_constructor_kwargs()["generator_dim"]),
            "discriminator_dim": list(_constructor_kwargs()["discriminator_dim"]),
            "sample_rows": EXPECTED_SAMPLE_ROWS,
        },
        "seed_cases": list(SEED_CASES),
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("CTGAN native-parity protocol failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CTGAN native-parity validation protocol.")
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
                        "model_id": "ctgan",
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
