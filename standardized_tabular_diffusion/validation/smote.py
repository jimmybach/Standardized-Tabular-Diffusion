"""Executable native-parity protocol for the official imbalanced-learn SMOTE package."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import traceback
import zipfile
from collections import Counter
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.sample_baselines import SMOTEAdapter

PROTOCOL_ID = "smote-native-parity-v1"
PACKAGE_NAME = "imbalanced-learn"
PACKAGE_VERSION = "0.14.2"
WHEEL_FILENAME = "imbalanced_learn-0.14.2-py3-none-any.whl"
WHEEL_SHA256 = "f9b81c47231aa1e3a71a1e4b3cc85b42e3b14f85e3a36922f3323c4da23605ef"
UPSTREAM_REPOSITORY = "https://github.com/scikit-learn-contrib/imbalanced-learn"
UPSTREAM_TAG = "0.14.2"
UPSTREAM_COMMIT = "8504e95f0160f61d1b617ca66f779646d2ee609e"
UPSTREAM_TREE = "af452de62e0f5c3d7e65fdc44a32dc97078152f2"
LICENSE_EXPRESSION = "MIT"
SOURCE_LICENSE_FILE_SHA256 = "7e586fd494f7470067defb818540a1d34b773ee62a16baaaccc511cdbc753181"
WHEEL_LICENSE_FILE_SHA256 = "fbcab88e3daf0f2f9248f5a774e97915b7a1a36da76a78c6a5baaa07095ef644"
EXPECTED_ARCHIVE_MEMBERS = 124
EXPECTED_RECORD_FILES = 123
EXPECTED_SOURCE_ROWS = 18
EXPECTED_BALANCED_ROWS = 24
EXPECTED_SAMPLE_ROWS = 20
K_NEIGHBORS = 3
SAMPLING_STRATEGY = "auto"
SEED_CASES = (0, 19, 73)
VARIANTS = ("smote", "smotenc", "smoten")
EXPECTED_DISTRIBUTION_VERSIONS = {
    "imbalanced-learn": "0.14.2",
    "joblib": "1.5.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "six": "1.17.0",
    "sklearn-compat": "0.1.6",
    "threadpoolctl": "3.6.0",
    "tzdata": "2026.3",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_record_hash(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify_wheel(wheel_path: Path) -> dict[str, Any]:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(f"imbalanced-learn wheel must be a regular non-symlinked file: {wheel_path}")
    if wheel_path.name != WHEEL_FILENAME:
        raise ValueError(f"Expected wheel {WHEEL_FILENAME!r}, observed {wheel_path.name!r}")
    observed_sha256 = _sha256_file(wheel_path)
    if observed_sha256 != WHEEL_SHA256:
        raise ValueError(
            "imbalanced-learn wheel checksum mismatch: "
            f"expected={WHEEL_SHA256}, observed={observed_sha256}"
        )

    dist_info = f"imbalanced_learn-{PACKAGE_VERSION}.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    license_name = f"{dist_info}/licenses/LICENSE"
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe path in imbalanced-learn wheel: {name!r}")
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        if metadata.get("Name") != PACKAGE_NAME or metadata.get("Version") != PACKAGE_VERSION:
            raise ValueError("imbalanced-learn wheel metadata does not match the locked package identity")
        if metadata.get("Requires-Python") != ">=3.10":
            raise ValueError("imbalanced-learn wheel Python requirement differs from the locked release")
        license_files = metadata.get_all("License-File", [])
        if "LICENSE" not in license_files:
            raise ValueError("imbalanced-learn wheel metadata does not declare its LICENSE file")
        license_sha256 = _sha256_bytes(archive.read(license_name))
        if license_sha256 != WHEEL_LICENSE_FILE_SHA256:
            raise ValueError("imbalanced-learn wheel LICENSE differs from the locked MIT license")
    if len(names) != EXPECTED_ARCHIVE_MEMBERS:
        raise ValueError(
            "imbalanced-learn wheel member count mismatch: "
            f"expected={EXPECTED_ARCHIVE_MEMBERS}, observed={len(names)}"
        )

    return {
        "filename": wheel_path.name,
        "sha256": observed_sha256,
        "bytes": wheel_path.stat().st_size,
        "archive_members": len(names),
        "license": LICENSE_EXPRESSION,
        "license_file_sha256": license_sha256,
    }


def _verify_installed_distribution(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    if distribution.version != PACKAGE_VERSION:
        raise ValueError(
            "Installed imbalanced-learn version mismatch: "
            f"expected={PACKAGE_VERSION}, observed={distribution.version}"
        )

    import imblearn
    from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
    from imblearn.over_sampling._smote.base import SMOTE as NativeSMOTE
    from imblearn.over_sampling._smote.base import SMOTEN as NativeSMOTEN
    from imblearn.over_sampling._smote.base import SMOTENC as NativeSMOTENC

    module_path = Path(imblearn.__file__).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not module_path.is_relative_to(distribution_root):
        raise ValueError(
            "imbalanced-learn did not resolve inside its installed distribution: "
            f"{module_path}"
        )
    if imblearn.__version__ != PACKAGE_VERSION:
        raise ValueError("Imported imbalanced-learn version differs from installed distribution metadata")
    if (SMOTE, SMOTENC, SMOTEN) != (NativeSMOTE, NativeSMOTENC, NativeSMOTEN):
        raise ValueError("Public imbalanced-learn sampler exports differ from their official source classes")

    record_name = f"imbalanced_learn-{PACKAGE_VERSION}.dist-info/RECORD"
    with zipfile.ZipFile(wheel_path) as archive:
        record_rows = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        expected_record_hashes = {
            path: encoded_hash.partition("=")[2]
            for path, encoded_hash, _ in record_rows
            if encoded_hash
        }

    verified_paths: set[str] = set()
    license_path: Path | None = None
    for package_path in distribution.files or ():
        installed_path = Path(package_path.locate()).resolve()
        if package_path.name == "LICENSE" and "licenses" in package_path.parts:
            license_path = installed_path
        expected_hash = expected_record_hashes.get(str(package_path).replace("\\", "/"))
        if expected_hash is None:
            continue
        file_hash = package_path.hash
        if file_hash is None:
            raise ValueError(f"Installed imbalanced-learn RECORD hash is missing: {package_path}")
        if file_hash.mode != "sha256":
            raise ValueError(f"Unexpected installed RECORD hash mode for {package_path}: {file_hash.mode}")
        if not installed_path.is_file():
            raise ValueError(f"Installed imbalanced-learn RECORD path is missing: {installed_path}")
        installed_digest = bytes.fromhex(_sha256_file(installed_path))
        if installed_digest != _decode_record_hash(expected_hash):
            raise ValueError(f"Installed imbalanced-learn file differs from the locked wheel: {package_path}")
        if installed_digest != _decode_record_hash(file_hash.value):
            raise ValueError(f"Installed imbalanced-learn RECORD checksum mismatch: {package_path}")
        verified_paths.add(str(package_path).replace("\\", "/"))
    if verified_paths != set(expected_record_hashes):
        raise ValueError(
            "Installed imbalanced-learn files do not match the locked wheel RECORD: "
            f"missing={sorted(set(expected_record_hashes) - verified_paths)}"
        )
    if license_path is None or _sha256_file(license_path) != WHEEL_LICENSE_FILE_SHA256:
        raise ValueError("Installed imbalanced-learn LICENSE does not match the locked MIT license")

    return {
        "version": distribution.version,
        "module_path": str(module_path),
        "distribution_root": str(distribution_root),
        "record_files_verified": len(verified_paths),
        "license": LICENSE_EXPRESSION,
        "license_file_sha256": WHEEL_LICENSE_FILE_SHA256,
        "sampler_classes": {
            "SMOTE": SMOTE.__module__,
            "SMOTENC": SMOTENC.__module__,
            "SMOTEN": SMOTEN.__module__,
        },
    }


def verify_package(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    """Fail closed unless the downloaded wheel and installed package match the lock."""

    return {
        "authority": "canonical-third-party-reference-package",
        "distribution_form": "package",
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "source_license_file_sha256": SOURCE_LICENSE_FILE_SHA256,
        "wheel": _verify_wheel(wheel_path.resolve()),
        "installed_distribution": _verify_installed_distribution(
            repo_root.resolve(), wheel_path.resolve()
        ),
    }


def _fixture_frame(variant: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    row_ids = np.arange(EXPECTED_SOURCE_ROWS)
    target = np.asarray(["majority"] * 12 + ["minority"] * 6)
    if variant == "smote":
        frame = pd.DataFrame(
            {
                "continuous": np.round(np.cos(row_ids / 2.5) + row_ids / 10.0, 6),
                "count": ((row_ids * 5) % 17).astype(int),
                "label": target,
            }
        )
        return frame, ["continuous", "count"], []
    if variant == "smotenc":
        frame = pd.DataFrame(
            {
                "continuous": np.round(np.sin(row_ids / 3.0) + row_ids / 12.0, 6),
                "count": ((row_ids * 7) % 19).astype(int),
                "segment": np.asarray(["alpha", "beta", "gamma"])[row_ids % 3],
                "label": target,
            }
        )
        return frame, ["continuous", "count"], ["segment"]
    if variant == "smoten":
        frame = pd.DataFrame(
            {
                "segment": np.asarray(["alpha", "beta", "gamma"])[row_ids % 3],
                "tier": np.asarray(["low", "medium", "high", "premium"])[row_ids % 4],
                "label": target,
            }
        )
        return frame, [], ["segment", "tier"]
    raise ValueError(f"Unknown SMOTE validation variant: {variant}")


def _write_fixtures(root: Path) -> list[tuple[str, pd.DataFrame, DatasetSpec, dict[str, Any]]]:
    fixtures: list[tuple[str, pd.DataFrame, DatasetSpec, dict[str, Any]]] = []
    for variant in VARIANTS:
        fixture_root = root / variant
        fixture_root.mkdir(parents=True, exist_ok=True)
        frame, numerical_columns, categorical_columns = _fixture_frame(variant)
        train_path = fixture_root / "train.csv"
        metadata_path = fixture_root / "info.json"
        atomic_write_bytes(train_path, frame.to_csv(index=False).encode("utf-8"))
        metadata_path.write_text(
            json.dumps(
                {
                    "name": f"smote-{variant}-parity-fixture",
                    "task_type": "binclass",
                    "column_names": list(frame.columns),
                    "num_col_idx": [frame.columns.get_loc(column) for column in numerical_columns],
                    "cat_col_idx": [frame.columns.get_loc(column) for column in categorical_columns],
                    "target_col_idx": [frame.columns.get_loc("label")],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        dataset_spec = DatasetSpec(
            name=f"smote-{variant}-parity-fixture",
            task_type="classification",
            column_names=list(frame.columns),
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target_columns=["label"],
            metadata_path=metadata_path,
            train_data_path=train_path,
        )
        loaded = pd.read_csv(train_path)[dataset_spec.column_names]
        fixtures.append(
            (
                variant,
                loaded,
                dataset_spec,
                {
                    "variant": variant,
                    "rows": len(loaded),
                    "columns": list(loaded.columns),
                    "numerical_columns": numerical_columns,
                    "categorical_columns": categorical_columns,
                    "target_columns": ["label"],
                    "class_counts": dict(Counter(loaded["label"])),
                    "missing_values": int(loaded.isna().sum().sum()),
                    "train_csv_sha256": _sha256_file(train_path),
                },
            )
        )
    return fixtures


def _native_sampler(variant: str, dataset_spec: DatasetSpec, seed: int) -> Any:
    from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC

    common = {
        "sampling_strategy": SAMPLING_STRATEGY,
        "random_state": seed,
        "k_neighbors": K_NEIGHBORS,
    }
    if variant == "smote":
        return SMOTE(**common)
    if variant == "smotenc":
        return SMOTENC(categorical_features=dataset_spec.categorical_columns, **common)
    if variant == "smoten":
        return SMOTEN(**common)
    raise ValueError(f"Unknown SMOTE validation variant: {variant}")


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _assemble_resampled_frame(
    x_resampled: Any,
    y_resampled: Any,
    dataset_spec: DatasetSpec,
) -> pd.DataFrame:
    target = dataset_spec.target_columns[0]
    feature_columns = [column for column in dataset_spec.column_names if column != target]
    frame = pd.DataFrame(x_resampled, columns=feature_columns)
    frame[target] = y_resampled
    return frame[dataset_spec.column_names]


def _run_native(
    variant: str,
    frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_spec.target_columns[0]
    feature_columns = [column for column in dataset_spec.column_names if column != target]
    before_state = np.random.get_state()
    sampler = _native_sampler(variant, dataset_spec, seed)
    x_resampled, y_resampled = sampler.fit_resample(frame[feature_columns].copy(), frame[target].copy())
    after_state = np.random.get_state()
    balanced = _assemble_resampled_frame(x_resampled, y_resampled, dataset_spec)
    selected = balanced.sample(
        n=EXPECTED_SAMPLE_ROWS,
        replace=len(balanced) < EXPECTED_SAMPLE_ROWS,
        random_state=seed,
    ).reset_index(drop=True)
    sample_path = output_dir / "samples.csv"
    atomic_write_bytes(sample_path, selected.to_csv(index=False).encode("utf-8"))
    return sampler, pd.read_csv(sample_path), {
        "sample_path": str(sample_path),
        "sample_sha256": _sha256_file(sample_path),
        "balanced_rows": len(balanced),
        "balanced_class_counts": dict(Counter(balanced[target])),
        "global_numpy_state_unchanged": _numpy_state_equal(before_state, after_state),
    }


def _run_adapter(
    repo_root: Path,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    adapter = SMOTEAdapter(repo_root)
    captured_samplers: list[Any] = []
    original_create_sampler = adapter._create_sampler

    def capture_sampler(**kwargs: Any) -> tuple[Any, str]:
        sampler, sampler_name = original_create_sampler(**kwargs)
        captured_samplers.append(sampler)
        return sampler, sampler_name

    adapter._create_sampler = capture_sampler  # type: ignore[method-assign]
    common = {
        "model": "smote",
        "dataset": dataset_spec.name,
        "output_dir": output_dir,
        "device": "cpu",
        "seed": seed,
        "extra": {
            "dataset_spec": dataset_spec.to_dict(),
            "k_neighbors": K_NEIGHBORS,
            "sampling_strategy": SAMPLING_STRATEGY,
        },
    }
    train_bundle = adapter.train(RunSpec(**common))
    train_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    before_state = np.random.get_state()
    sample_bundle = adapter.sample(RunSpec(**common, num_samples=EXPECTED_SAMPLE_ROWS))
    after_state = np.random.get_state()
    sample_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "smote_metadata.json").read_text(encoding="utf-8"))
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("SMOTE adapter did not declare a generated sample path")
    if len(captured_samplers) != 1:
        raise AssertionError("SMOTE adapter did not construct exactly one official sampler")
    manifests_valid = (
        train_bundle.model == "smote"
        and train_manifest["model"] == "smote"
        and train_manifest["dataset"] == dataset_spec.name
        and sample_manifest["generated_sample_path"] == str(sample_bundle.generated_sample_path)
    )
    return captured_samplers[0], pd.read_csv(sample_bundle.generated_sample_path), {
        "sample_path": str(sample_bundle.generated_sample_path),
        "sample_sha256": _sha256_file(sample_bundle.generated_sample_path),
        "metadata": metadata,
        "manifests_valid": manifests_valid,
        "upstream_root": str(adapter.upstream_root.resolve()),
        "global_numpy_state_unchanged": _numpy_state_equal(before_state, after_state),
    }


def _matrix_equal(left: Any, right: Any) -> bool:
    if sparse.issparse(left) or sparse.issparse(right):
        if not sparse.issparse(left) or not sparse.issparse(right) or left.shape != right.shape:
            return False
        return bool((left != right).nnz == 0)
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _encoder_categories_equal(left: Any, right: Any) -> bool:
    left_encoder = getattr(left, "categorical_encoder_", None)
    right_encoder = getattr(right, "categorical_encoder_", None)
    if left_encoder is None or right_encoder is None:
        return left_encoder is right_encoder
    return bool(
        len(left_encoder.categories_) == len(right_encoder.categories_)
        and all(
            np.array_equal(left_categories, right_categories)
            for left_categories, right_categories in zip(
                left_encoder.categories_, right_encoder.categories_, strict=True
            )
        )
    )


def _optional_array_equal(left: Any, right: Any, name: str) -> bool:
    left_value = getattr(left, name, None)
    right_value = getattr(right, name, None)
    if left_value is None or right_value is None:
        return left_value is right_value
    return bool(np.array_equal(left_value, right_value))


def _compare_samplers(left: Any, right: Any) -> dict[str, Any]:
    left_neighbors = getattr(getattr(left, "nn_k_", None), "_fit_X", None)
    right_neighbors = getattr(getattr(right, "nn_k_", None), "_fit_X", None)
    neighbor_state_exact = (
        left_neighbors is not None
        and right_neighbors is not None
        and _matrix_equal(left_neighbors, right_neighbors)
    )
    feature_names_exact = _optional_array_equal(left, right, "feature_names_in_")
    return {
        "class_exact": type(left) is type(right),
        "module_exact": type(left).__module__ == type(right).__module__,
        "params_exact": left.get_params(deep=False) == right.get_params(deep=False),
        "sampling_strategy_exact": left.sampling_strategy_ == right.sampling_strategy_,
        "n_features_exact": left.n_features_in_ == right.n_features_in_,
        "feature_names_exact": feature_names_exact,
        "neighbor_state_exact": neighbor_state_exact,
        "neighbor_fit_shape": list(left_neighbors.shape),
        "categorical_features_exact": _optional_array_equal(left, right, "categorical_features_"),
        "continuous_features_exact": _optional_array_equal(left, right, "continuous_features_"),
        "encoder_categories_exact": _encoder_categories_equal(left, right),
        "median_std_exact": getattr(left, "median_std_", None) == getattr(right, "median_std_", None),
    }


def _expected_metadata(
    variant: str,
    dataset_spec: DatasetSpec,
    seed: int,
) -> dict[str, Any]:
    target = dataset_spec.target_columns[0]
    feature_columns = [column for column in dataset_spec.column_names if column != target]
    names = {"smote": "SMOTE", "smotenc": "SMOTENC", "smoten": "SMOTEN"}
    return {
        "sampler": names[variant],
        "package": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "random_state": seed,
        "k_neighbors": K_NEIGHBORS,
        "sampling_strategy": SAMPLING_STRATEGY,
        "categorical_columns": dataset_spec.categorical_columns,
        "categorical_indices": [
            feature_columns.index(column) for column in dataset_spec.categorical_columns
        ],
        "source_rows": EXPECTED_SOURCE_ROWS,
        "balanced_rows": EXPECTED_BALANCED_ROWS,
        "output_rows": EXPECTED_SAMPLE_ROWS,
    }


def _compare_samples(
    left: pd.DataFrame,
    right: pd.DataFrame,
    source: pd.DataFrame,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    numerical_columns = dataset_spec.numerical_columns
    categorical_columns = [*dataset_spec.categorical_columns, *dataset_spec.target_columns]
    finite_numerical = not numerical_columns or bool(
        np.isfinite(left[numerical_columns].to_numpy(dtype=float)).all()
        and np.isfinite(right[numerical_columns].to_numpy(dtype=float)).all()
    )
    categorical_domains_valid = all(
        set(left[column]).issubset(set(source[column]))
        and set(right[column]).issubset(set(source[column]))
        for column in categorical_columns
    )
    return {
        "rows": len(left),
        "columns_exact": list(left.columns) == list(right.columns) == dataset_spec.column_names,
        "frame_exact": left.equals(right),
        "finite_numerical": finite_numerical,
        "categorical_domains_valid": categorical_domains_valid,
        "target_classes_present": set(left[dataset_spec.target_columns[0]]) == set(source["label"]),
        "missing_values": int(left.isna().sum().sum() + right.isna().sum().sum()),
    }


def _case_passed(comparisons: dict[str, Any]) -> bool:
    sampler = comparisons["sampler"]
    samples = comparisons["samples"]
    return bool(
        comparisons["adapter_manifests_valid"]
        and comparisons["adapter_metadata_exact"]
        and comparisons["sample_bytes_exact"]
        and comparisons["native_balanced_rows_exact"]
        and comparisons["native_balanced_classes_exact"]
        and comparisons["native_global_numpy_state_unchanged"]
        and comparisons["adapter_global_numpy_state_unchanged"]
        and sampler["class_exact"]
        and sampler["module_exact"]
        and sampler["params_exact"]
        and sampler["sampling_strategy_exact"]
        and sampler["n_features_exact"]
        and sampler["feature_names_exact"]
        and sampler["neighbor_state_exact"]
        and sampler["categorical_features_exact"]
        and sampler["continuous_features_exact"]
        and sampler["encoder_categories_exact"]
        and sampler["median_std_exact"]
        and samples["rows"] == EXPECTED_SAMPLE_ROWS
        and samples["columns_exact"]
        and samples["frame_exact"]
        and samples["finite_numerical"]
        and samples["categorical_domains_valid"]
        and samples["target_classes_present"]
        and samples["missing_values"] == 0
    )


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _verify_environment() -> dict[str, str]:
    observed = {name: _version(name) for name in EXPECTED_DISTRIBUTION_VERSIONS}
    if observed != EXPECTED_DISTRIBUTION_VERSIONS:
        raise RuntimeError(
            "SMOTE validation environment does not match its frozen lock: "
            f"expected={EXPECTED_DISTRIBUTION_VERSIONS}, observed={observed}"
        )
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative SMOTE validation requires Linux and Python 3.11; "
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
    """Run exact official-package-versus-adapter comparisons for all sampler variants."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_versions = _verify_environment()
    source = verify_package(repo_root, wheel_path.resolve())
    fixtures = _write_fixtures(output_dir / "fixtures")
    fixture_records = [fixture for _, _, _, fixture in fixtures]
    cases: list[dict[str, Any]] = []
    for variant, frame, dataset_spec, _ in fixtures:
        for seed in SEED_CASES:
            native_sampler, native_samples, native_artifacts = _run_native(
                variant,
                frame,
                dataset_spec,
                output_dir / variant / f"seed-{seed}" / "native",
                seed,
            )
            adapter_sampler, adapter_samples, adapter_artifacts = _run_adapter(
                repo_root,
                dataset_spec,
                output_dir / variant / f"seed-{seed}" / "adapter",
                seed,
            )
            comparisons = {
                "adapter_manifests_valid": adapter_artifacts["manifests_valid"],
                "adapter_metadata_exact": (
                    adapter_artifacts["metadata"] == _expected_metadata(variant, dataset_spec, seed)
                ),
                "sample_bytes_exact": (
                    native_artifacts["sample_sha256"] == adapter_artifacts["sample_sha256"]
                ),
                "native_balanced_rows_exact": (
                    native_artifacts["balanced_rows"] == EXPECTED_BALANCED_ROWS
                ),
                "native_balanced_classes_exact": (
                    native_artifacts["balanced_class_counts"]
                    == {"majority": 12, "minority": 12}
                ),
                "native_global_numpy_state_unchanged": native_artifacts[
                    "global_numpy_state_unchanged"
                ],
                "adapter_global_numpy_state_unchanged": adapter_artifacts[
                    "global_numpy_state_unchanged"
                ],
                "sampler": _compare_samplers(native_sampler, adapter_sampler),
                "samples": _compare_samples(
                    native_samples, adapter_samples, frame, dataset_spec
                ),
            }
            cases.append(
                {
                    "variant": variant,
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
        "model_id": "smote",
        "status": "pass" if passed else "fail",
        "claim_boundary": (
            "Classification-only classical oversampling reference; not a joint tabular generator."
        ),
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "environment_lock": {
            "path": "requirements-smote-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-smote-validation.txt"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **actual_versions,
        },
        "fixtures": fixture_records,
        "runtime_config": {
            "k_neighbors": K_NEIGHBORS,
            "sampling_strategy": SAMPLING_STRATEGY,
            "sample_rows": EXPECTED_SAMPLE_ROWS,
        },
        "variants": list(VARIANTS),
        "seed_cases": list(SEED_CASES),
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("SMOTE native-parity protocol failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SMOTE native-parity validation protocol.")
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
                        "model_id": "smote",
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
