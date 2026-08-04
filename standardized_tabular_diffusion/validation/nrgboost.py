"""Executable native-parity protocol for the official NRGBoost package."""

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
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.next_wave_baselines import NRGBoostAdapter

PROTOCOL_ID = "nrgboost-native-parity-v1"
PACKAGE_NAME = "nrgboost"
PACKAGE_VERSION = "0.0.3"
WHEEL_FILENAME = "nrgboost-0.0.3-cp311-cp311-manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb"
UPSTREAM_REPOSITORY = "https://github.com/Ajoo/nrgboost"
UPSTREAM_TAG = "v0.0.3"
UPSTREAM_COMMIT = "feef73a3edb20b911c2f7214b13f810909ef20ad"
UPSTREAM_TREE = "e3e84bacc7236a36af93c3d214de14bd308d2767"
LICENSE_EXPRESSION = "MIT"
SOURCE_LICENSE_FILE_SHA256 = "999f9e9b53bb8aae7225dac0599ff28bbcb61f8604da1bcb201cf767543998bf"
WHEEL_LICENSE_FILE_SHA256 = "3693dc7c451fe74ffead14c00964ac00a1123242c9fc3d8cb13c8fef3091b945"
C_EXTENSION_SHA256 = "c7574e824be9f116f95142c64335ea6078f7effdc9279f797cf0503534ba09c7"
BUNDLED_OPENMP_SHA256 = "ec3543cb6fa11f34258fe3d59082e4bd9740a18349e097fc42a7bc692a21bf2c"
EXPECTED_ARCHIVE_MEMBERS = 29
EXPECTED_RECORD_FILES = 22
EXPECTED_SOURCE_ROWS = 36
EXPECTED_SAMPLE_ROWS = 16
SEED_CASES = (0, 19, 73)
VARIANTS = ("classification", "regression")
EXPECTED_DISTRIBUTION_VERSIONS = {
    "cffi": "1.17.1",
    "joblib": "1.5.3",
    "llvmlite": "0.44.0",
    "numba": "0.61.2",
    "nrgboost": "0.0.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "pycparser": "2.22",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "scipy": "1.13.1",
    "six": "1.17.0",
    "tqdm": "4.66.5",
    "tzdata": "2026.3",
}
DATASET_PARAMS = {
    "num_bins": 32,
    "infer_fixed_point": True,
    "discretization_types": None,
    "infer_ordered_categoricals": False,
    "infer_continuous_ordered_categoricals": False,
}
TRAINING_PARAMS = {
    "num_trees": 3,
    "shrinkage": 0.15,
    "line_search": True,
    "max_leaves": 8,
    "max_ratio_in_leaf": 2.0,
    "min_data_in_leaf": 0.0,
    "initial_uniform_mixture": 0.1,
    "categorical_split_one_vs_all": False,
    "feature_frac": 1.0,
    "splitter": "best",
    "num_model_samples": 256,
    "p_refresh": 0.2,
    "num_chains": 4,
    "burn_in": 8,
    "temperature": 1.0,
    "initial_samples": "data",
    "min_gain": 0.0,
    "jit_all": False,
    "num_threads": 1,
}
SAMPLING_PARAMS = {
    "num_steps": 12,
    "num_rounds": None,
    "temperature": 1.0,
    "num_threads": 1,
    "output_full_chain": False,
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
        raise ValueError(f"NRGBoost wheel must be a regular non-symlinked file: {wheel_path}")
    if wheel_path.name != WHEEL_FILENAME:
        raise ValueError(f"Expected wheel {WHEEL_FILENAME!r}, observed {wheel_path.name!r}")
    observed_sha256 = _sha256_file(wheel_path)
    if observed_sha256 != WHEEL_SHA256:
        raise ValueError(f"NRGBoost wheel checksum mismatch: expected={WHEEL_SHA256}, observed={observed_sha256}")

    dist_info = f"nrgboost-{PACKAGE_VERSION}.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    wheel_name = f"{dist_info}/WHEEL"
    license_name = f"{dist_info}/licenses/LICENSE"
    extension_name = "_eval.abi3.so"
    openmp_name = "nrgboost.libs/libgomp-24e2ab19.so.1.0.0"
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe path in NRGBoost wheel: {name!r}")
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        if metadata.get("Name") != PACKAGE_NAME or metadata.get("Version") != PACKAGE_VERSION:
            raise ValueError("NRGBoost wheel metadata does not match the locked package identity")
        if metadata.get("Requires-Python") != ">=3.10":
            raise ValueError("NRGBoost wheel Python requirement differs from the locked release")
        if metadata.get_all("Requires-Dist", []) != [
            "cffi>=1", "numpy", "scipy", "numba", "tqdm", "joblib", "pandas"
        ]:
            raise ValueError("NRGBoost wheel dependencies differ from the audited release")
        if "LICENSE" not in metadata.get_all("License-File", []):
            raise ValueError("NRGBoost wheel metadata does not declare its LICENSE file")
        wheel_metadata = archive.read(wheel_name).decode("utf-8")
        if "Root-Is-Purelib: false" not in wheel_metadata or (
            "Tag: cp311-cp311-manylinux_2_28_x86_64" not in wheel_metadata
        ):
            raise ValueError("NRGBoost wheel ABI/platform tag differs from the Linux/Python 3.11 lock")
        license_sha256 = _sha256_bytes(archive.read(license_name))
        extension_sha256 = _sha256_bytes(archive.read(extension_name))
        openmp_sha256 = _sha256_bytes(archive.read(openmp_name))
    if len(names) != EXPECTED_ARCHIVE_MEMBERS:
        raise ValueError(
            f"NRGBoost wheel member count mismatch: expected={EXPECTED_ARCHIVE_MEMBERS}, observed={len(names)}"
        )
    if license_sha256 != WHEEL_LICENSE_FILE_SHA256:
        raise ValueError("NRGBoost wheel LICENSE differs from the locked MIT license")
    if extension_sha256 != C_EXTENSION_SHA256 or openmp_sha256 != BUNDLED_OPENMP_SHA256:
        raise ValueError("NRGBoost compiled runtime differs from the audited Linux wheel")
    return {
        "filename": wheel_path.name,
        "sha256": observed_sha256,
        "bytes": wheel_path.stat().st_size,
        "archive_members": len(names),
        "license": LICENSE_EXPRESSION,
        "license_file_sha256": license_sha256,
        "c_extension_sha256": extension_sha256,
        "bundled_openmp_sha256": openmp_sha256,
    }


def _verify_installed_distribution(wheel_path: Path) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    if distribution.version != PACKAGE_VERSION:
        raise ValueError(
            f"Installed NRGBoost version mismatch: expected={PACKAGE_VERSION}, observed={distribution.version}"
        )

    import nrgboost
    from nrgboost.dataset import Dataset as NativeDataset
    from nrgboost.wrapper import NRGBooster as NativeNRGBooster

    if nrgboost.Dataset is not NativeDataset or nrgboost.NRGBooster is not NativeNRGBooster:
        raise ValueError("Public NRGBoost exports differ from the official source classes")
    module_path = Path(nrgboost.__file__).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not module_path.is_relative_to(distribution_root):
        raise ValueError(f"NRGBoost did not resolve inside its installed distribution: {module_path}")

    record_name = f"nrgboost-{PACKAGE_VERSION}.dist-info/RECORD"
    with zipfile.ZipFile(wheel_path) as archive:
        rows = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        expected_hashes = {
            path: encoded.partition("=")[2]
            for path, encoded, _ in rows
            if encoded
        }
    verified_paths: set[str] = set()
    license_path: Path | None = None
    for package_path in distribution.files or ():
        normalized = str(package_path).replace("\\", "/")
        installed_path = Path(package_path.locate()).resolve()
        if package_path.name == "LICENSE" and "licenses" in package_path.parts:
            license_path = installed_path
        expected_hash = expected_hashes.get(normalized)
        if expected_hash is None:
            continue
        if not installed_path.is_file():
            raise ValueError(f"Installed NRGBoost RECORD path is missing: {installed_path}")
        observed = bytes.fromhex(_sha256_file(installed_path))
        if observed != _decode_record_hash(expected_hash):
            raise ValueError(f"Installed NRGBoost file differs from the locked wheel: {package_path}")
        record_hash = package_path.hash
        if record_hash is None or record_hash.mode != "sha256" or observed != _decode_record_hash(record_hash.value):
            raise ValueError(f"Installed NRGBoost RECORD checksum mismatch: {package_path}")
        verified_paths.add(normalized)
    if verified_paths != set(expected_hashes) or len(verified_paths) != EXPECTED_RECORD_FILES:
        raise ValueError(
            "Installed NRGBoost files do not match the locked wheel RECORD: "
            f"missing={sorted(set(expected_hashes) - verified_paths)}"
        )
    if license_path is None or _sha256_file(license_path) != WHEEL_LICENSE_FILE_SHA256:
        raise ValueError("Installed NRGBoost LICENSE does not match the locked MIT license")
    return {
        "version": distribution.version,
        "module_path": str(module_path),
        "distribution_root": str(distribution_root),
        "record_files_verified": len(verified_paths),
        "public_exports_exact": True,
        "license": LICENSE_EXPRESSION,
        "license_file_sha256": WHEEL_LICENSE_FILE_SHA256,
    }


def verify_package(wheel_path: Path) -> dict[str, Any]:
    return {
        "authority": "method-author",
        "distribution_form": "package",
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "release_provenance": "PyPI Trusted Publishing from the locked tag commit",
        "source_license_file_sha256": SOURCE_LICENSE_FILE_SHA256,
        "wheel": _verify_wheel(wheel_path.resolve()),
        "installed_distribution": _verify_installed_distribution(wheel_path.resolve()),
    }


def _fixture_frame(variant: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    row_ids = np.arange(EXPECTED_SOURCE_ROWS)
    frame = pd.DataFrame(
        {
            "continuous": np.round(np.sin(row_ids / 4.0) + row_ids / 20.0, 4),
            "count": ((row_ids * 7) % 23).astype(int),
            "segment": np.asarray(["alpha", "beta", "gamma"])[row_ids % 3],
        }
    )
    if variant == "classification":
        frame["target"] = np.asarray(["no", "yes"])[(row_ids % 5 == 0).astype(int)]
        return frame, ["continuous", "count"], ["segment"], ["target"]
    if variant == "regression":
        frame["target"] = np.round(0.35 * frame["count"] - 0.8 * frame["continuous"] + row_ids / 50.0, 4)
        return frame, ["continuous", "count", "target"], ["segment"], ["target"]
    raise ValueError(f"Unknown NRGBoost validation variant: {variant}")


def _write_fixtures(root: Path) -> list[tuple[str, pd.DataFrame, DatasetSpec, dict[str, Any]]]:
    fixtures = []
    for variant in VARIANTS:
        fixture_root = root / variant
        fixture_root.mkdir(parents=True, exist_ok=True)
        frame, numerical, categorical, targets = _fixture_frame(variant)
        train_path = fixture_root / "train.csv"
        metadata_path = fixture_root / "info.json"
        atomic_write_bytes(train_path, frame.to_csv(index=False).encode("utf-8"))
        metadata_path.write_text(
            json.dumps({"name": f"nrgboost-{variant}-parity-fixture", "task_type": variant}, indent=2) + "\n",
            encoding="utf-8",
        )
        spec = DatasetSpec(
            name=f"nrgboost-{variant}-parity-fixture",
            task_type=variant,
            column_names=list(frame.columns),
            numerical_columns=numerical,
            categorical_columns=categorical,
            target_columns=targets,
            metadata_path=metadata_path,
            train_data_path=train_path,
        )
        loaded = pd.read_csv(train_path)[spec.column_names]
        fixtures.append(
            (
                variant,
                loaded,
                spec,
                {
                    "variant": variant,
                    "rows": len(loaded),
                    "columns": list(loaded.columns),
                    "numerical_columns": numerical,
                    "categorical_columns": categorical,
                    "target_columns": targets,
                    "missing_values": int(loaded.isna().sum().sum()),
                    "train_csv_sha256": _sha256_file(train_path),
                },
            )
        )
    return fixtures


def _typed_frame(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    result = frame.copy()
    category_columns = [*spec.categorical_columns]
    if spec.task_type == "classification":
        category_columns.extend(spec.target_columns)
    for column in category_columns:
        result[column] = result[column].astype("category")
    return result


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:])


def _run_native(
    frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from nrgboost import Dataset, NRGBooster

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.nrgboost"
    before_train = np.random.get_state()
    dataset = Dataset(_typed_frame(frame, dataset_spec), **dict(DATASET_PARAMS))
    model = NRGBooster.fit(dataset, dict(TRAINING_PARAMS), seed=seed)
    model.save(str(checkpoint_path))
    after_train = np.random.get_state()
    before_sample = np.random.get_state()
    loaded = NRGBooster.load(str(checkpoint_path))
    samples = loaded.sample(EXPECTED_SAMPLE_ROWS, **SAMPLING_PARAMS, seed=seed)
    after_sample = np.random.get_state()
    sample_path = output_dir / "samples.csv"
    atomic_write_bytes(sample_path, samples[dataset_spec.column_names].to_csv(index=False).encode("utf-8"))
    serialized = joblib.load(checkpoint_path)
    return pd.read_csv(sample_path), {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "sample_path": str(sample_path),
        "sample_sha256": _sha256_file(sample_path),
        "serialized_version": serialized.get("version"),
        "serialized_tree_count": len(serialized["booster"]["trees"]),
        "transform_columns": list(serialized["transform"].columns),
        "global_numpy_state_unchanged": _numpy_state_equal(before_train, after_train)
        and _numpy_state_equal(before_sample, after_sample),
    }


def _adapter_extra() -> dict[str, Any]:
    return {
        **DATASET_PARAMS,
        **TRAINING_PARAMS,
        "training_temperature": TRAINING_PARAMS["temperature"],
        **{key: value for key, value in SAMPLING_PARAMS.items() if key != "output_full_chain"},
    }


def _run_adapter(
    repo_root: Path,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = NRGBoostAdapter(repo_root)
    common = {
        "model": "nrgboost",
        "dataset": dataset_spec.name,
        "output_dir": output_dir,
        "device": "cpu",
        "seed": seed,
        "extra": {"dataset_spec": dataset_spec.to_dict(), **_adapter_extra()},
    }
    before_train = np.random.get_state()
    train_bundle = adapter.train(RunSpec(**common))
    after_train = np.random.get_state()
    train_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    checkpoint_path = output_dir / "model.nrgboost"
    before_sample = np.random.get_state()
    sample_bundle = adapter.sample(RunSpec(**common, num_samples=EXPECTED_SAMPLE_ROWS))
    after_sample = np.random.get_state()
    sample_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "nrgboost_metadata.json").read_text(encoding="utf-8"))
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("NRGBoost adapter did not declare a generated sample path")
    serialized = joblib.load(checkpoint_path)
    manifests_valid = (
        train_bundle.model == "nrgboost"
        and train_manifest["model"] == "nrgboost"
        and train_manifest["dataset"] == dataset_spec.name
        and sample_manifest["generated_sample_path"] == str(sample_bundle.generated_sample_path)
        and Path(train_manifest["upstream_workdir"]).resolve() == repo_root.resolve()
    )
    return pd.read_csv(sample_bundle.generated_sample_path), {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "sample_path": str(sample_bundle.generated_sample_path),
        "sample_sha256": _sha256_file(sample_bundle.generated_sample_path),
        "serialized_version": serialized.get("version"),
        "serialized_tree_count": len(serialized["booster"]["trees"]),
        "transform_columns": list(serialized["transform"].columns),
        "metadata": metadata,
        "manifests_valid": manifests_valid,
        "global_numpy_state_unchanged": _numpy_state_equal(before_train, after_train)
        and _numpy_state_equal(before_sample, after_sample),
    }


def _expected_metadata(dataset_spec: DatasetSpec, output_dir: Path, seed: int) -> dict[str, Any]:
    category_columns = [*dataset_spec.categorical_columns]
    if dataset_spec.task_type == "classification":
        category_columns.extend(dataset_spec.target_columns)
    return {
        "package": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "seed": seed,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "columns": dataset_spec.column_names,
        "categorical_columns": category_columns,
        "dataset_params": DATASET_PARAMS,
        "training_params": TRAINING_PARAMS,
        "checkpoint_path": str(output_dir / "model.nrgboost"),
        "sampling": {
            "requested_rows": EXPECTED_SAMPLE_ROWS,
            **SAMPLING_PARAMS,
            "seed": seed,
        },
        "sample_path": str(output_dir / "samples.csv"),
    }


def _compare_samples(
    native: pd.DataFrame,
    adapter: pd.DataFrame,
    source: pd.DataFrame,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    numerical = dataset_spec.numerical_columns
    categorical = [*dataset_spec.categorical_columns]
    if dataset_spec.task_type == "classification":
        categorical.extend(dataset_spec.target_columns)
    finite = not numerical or bool(
        np.isfinite(native[numerical].to_numpy(dtype=float)).all()
        and np.isfinite(adapter[numerical].to_numpy(dtype=float)).all()
    )
    domains_valid = all(
        set(native[column]).issubset(set(source[column]))
        and set(adapter[column]).issubset(set(source[column]))
        for column in categorical
    )
    return {
        "rows": len(native),
        "columns_exact": list(native.columns) == list(adapter.columns) == dataset_spec.column_names,
        "frame_exact": native.equals(adapter),
        "finite_numerical": finite,
        "categorical_domains_valid": domains_valid,
        "missing_values": int(native.isna().sum().sum() + adapter.isna().sum().sum()),
    }


def _case_passed(comparisons: dict[str, Any]) -> bool:
    samples = comparisons["samples"]
    return bool(
        comparisons["adapter_manifests_valid"]
        and comparisons["adapter_metadata_exact"]
        and comparisons["checkpoint_bytes_exact"]
        and comparisons["sample_bytes_exact"]
        and comparisons["checkpoint_structure_exact"]
        and comparisons["native_global_numpy_state_unchanged"]
        and comparisons["adapter_global_numpy_state_unchanged"]
        and samples["rows"] == EXPECTED_SAMPLE_ROWS
        and samples["columns_exact"]
        and samples["frame_exact"]
        and samples["finite_numerical"]
        and samples["categorical_domains_valid"]
        and samples["missing_values"] == 0
    )


def _verify_environment() -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTION_VERSIONS}
    if observed != EXPECTED_DISTRIBUTION_VERSIONS:
        raise RuntimeError(
            f"NRGBoost validation environment does not match its frozen lock: expected={EXPECTED_DISTRIBUTION_VERSIONS}, observed={observed}"
        )
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative NRGBoost validation requires Linux and Python 3.11; "
            f"observed platform={platform.platform()!r}, python={python_version!r}"
        )
    return observed


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(repo_root: Path, output_dir: Path, evidence_path: Path, wheel_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = _verify_environment()
    source = verify_package(wheel_path.resolve())
    fixtures = _write_fixtures(output_dir / "fixtures")
    cases: list[dict[str, Any]] = []
    for variant, frame, dataset_spec, _ in fixtures:
        for seed in SEED_CASES:
            native_samples, native_artifacts = _run_native(
                frame, dataset_spec, output_dir / variant / f"seed-{seed}" / "native", seed
            )
            adapter_dir = output_dir / variant / f"seed-{seed}" / "adapter"
            adapter_samples, adapter_artifacts = _run_adapter(repo_root, dataset_spec, adapter_dir, seed)
            comparisons = {
                "adapter_manifests_valid": adapter_artifacts["manifests_valid"],
                "adapter_metadata_exact": adapter_artifacts["metadata"]
                == _expected_metadata(dataset_spec, adapter_dir, seed),
                "checkpoint_bytes_exact": native_artifacts["checkpoint_sha256"]
                == adapter_artifacts["checkpoint_sha256"],
                "sample_bytes_exact": native_artifacts["sample_sha256"] == adapter_artifacts["sample_sha256"],
                "checkpoint_structure_exact": (
                    native_artifacts["serialized_version"]
                    == adapter_artifacts["serialized_version"]
                    == "0.0"
                    and native_artifacts["serialized_tree_count"]
                    == adapter_artifacts["serialized_tree_count"]
                    == TRAINING_PARAMS["num_trees"]
                    and native_artifacts["transform_columns"]
                    == adapter_artifacts["transform_columns"]
                    == dataset_spec.column_names
                ),
                "native_global_numpy_state_unchanged": native_artifacts["global_numpy_state_unchanged"],
                "adapter_global_numpy_state_unchanged": adapter_artifacts["global_numpy_state_unchanged"],
                "samples": _compare_samples(native_samples, adapter_samples, frame, dataset_spec),
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
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "nrgboost",
        "status": "pass" if passed else "fail",
        "claim_boundary": "Native adapter parity only; benchmark eligibility and release support remain separate gates.",
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "environment_lock": {
            "path": "requirements-nrgboost-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-nrgboost-validation.txt"),
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), **versions},
        "fixtures": [record for _, _, _, record in fixtures],
        "runtime_config": {
            "dataset_params": DATASET_PARAMS,
            "training_params": TRAINING_PARAMS,
            "sampling_params": SAMPLING_PARAMS,
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
        raise AssertionError("NRGBoost native-parity protocol failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the NRGBoost native-parity validation protocol.")
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
                        "model_id": "nrgboost",
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
