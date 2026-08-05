"""Executable native-parity protocol for the method-author official arfpy package."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tarfile
import traceback
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.final_wave_baselines import ARFAdapter

PROTOCOL_ID = "arfpy-official-package-parity-v1"
PACKAGE_NAME = "arfpy"
PACKAGE_VERSION = "0.1.1"
SDIST_FILENAME = "arfpy-0.1.1.tar.gz"
SDIST_BYTES = 11841
SDIST_SHA256 = "88170d5e72638b0dbfec28cfbdfee02e97bd6a06d5a636e960acd5d90d480707"
SDIST_URL = (
    "https://files.pythonhosted.org/packages/95/6f/"
    "a61794959d3860e23f5f2de5886b61154d40c246b38eedebf19d22e4cc35/arfpy-0.1.1.tar.gz"
)
UPSTREAM_REPOSITORY = "https://github.com/bips-hb/arfpy"
METHOD_REPOSITORY = "https://github.com/bips-hb/arf"
UPSTREAM_COMMIT = "6f737baaaa589f7ac3ff59f0d739ce04b0f1381c"
UPSTREAM_TREE = "68b6fc5d28578a5c21bef560bd28f4c0d2d6401c"
LICENSE_EXPRESSION = "MIT"
LICENSE_SHA256 = "8f97b1e0e6c2a7c7b539e63e8a5c81c85d040556940a07b348596cd9674283ec"
EXPECTED_ARCHIVE_MEMBERS = 20
EXPECTED_ARCHIVE_FILES = {
    "LICENSE": LICENSE_SHA256,
    "PKG-INFO": "53f0cb2d90a1b1f6fc611203fa4f01e3f84cc7809c6d885affa8b6e53676b745",
    "README.md": "3bbe9fc947063769f23640578cf5912eac53776531bcad460f9d31001d88e661",
    "arfpy/__init__.py": ARFAdapter.runtime_file_sha256["__init__.py"],
    "arfpy/arf.py": ARFAdapter.runtime_file_sha256["arf.py"],
    "arfpy/utils.py": ARFAdapter.runtime_file_sha256["utils.py"],
    "arfpy.egg-info/PKG-INFO": "53f0cb2d90a1b1f6fc611203fa4f01e3f84cc7809c6d885affa8b6e53676b745",
    "arfpy.egg-info/SOURCES.txt": "f386d26df3974a1dc6b4a9af6ac88357b7e3c4853151b1220fea8749b40b948d",
    "arfpy.egg-info/dependency_links.txt": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "arfpy.egg-info/requires.txt": "4e0f71c79211138041f39934eac9909547a44b4e93dff4e96057a8bf7c4620f1",
    "arfpy.egg-info/top_level.txt": "b1e98ea395c359faf8bd15fe9c1b5dc043cb2fbc3703095a01773e3700cc86b3",
    "setup.cfg": "1c473cbaee8da5fc46e7f0158794af5cea4414c34a3cf3f180c2001f5e38bd3e",
    "setup.py": "ec0b8249a1edc52267eea895ae7b643847fa4e86a1912f8dc6f8ad5d0f723841",
    "tests/test.py": "cdb92d5cd8d4731641b275b2c769bd31352d5a27b2781a09f43b332d83a47a47",
    "tests/test_diabetes.py": "bf5c1235209b77e9aaadfe4fa57e0e7f2ec322782ffa7fd25a15d174d3861a79",
    "tests/test_iris.py": "414664409ae8e0ddbf137169eed73036d3552cdb42db5cfed9b32551100da3c8",
}
EXPECTED_GIT_BLOBS = {
    "LICENSE": "b4d4d1b1b589f798c1cab65fec7efac6a55ca60e",
    "README.md": "13bd2f89ca2165d5ea93ce5af090a174fd1cda5b",
    "arfpy/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "arfpy/arf.py": "c763d1cadb0344bc6457f1adb7741c52c686ed17",
    "arfpy/utils.py": "0fd7a4339e8cbe24473fb42ddba603cb518eedf7",
    "setup.py": "6267bec23bac24fafd922c3bb283bd208096f492",
}
EXPECTED_REQUIREMENTS = [
    "numpy>=1.20.3",
    "pandas>=1.5",
    "scikit-learn>=0.24",
    "scipy>=1.4",
]
EXPECTED_DISTRIBUTION_VERSIONS = {
    "arfpy": "0.1.1",
    "joblib": "1.5.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "setuptools": "80.10.2",
    "six": "1.17.0",
    "threadpoolctl": "3.6.0",
    "tzdata": "2026.3",
    "wheel": "0.45.1",
}
EXPECTED_SOURCE_ROWS = 60
EXPECTED_SAMPLE_ROWS = 13
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
TRAINING_PARAMS = {
    "num_trees": 20,
    "delta": 0.0,
    "max_iters": 1,
    "early_stop": False,
    "verbose": False,
    "min_node_size": 2,
    "n_jobs": 1,
}
FORDE_PARAMS = {"dist": "truncnorm", "oob": False, "alpha": 0.0}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()  # noqa: S324 - Git object identity is SHA-1 by definition.


def _decode_record_hash(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _archive_member_path(name: str) -> str:
    root_name = f"arfpy-{PACKAGE_VERSION}"
    if name == root_name:
        return ""
    root = f"{root_name}/"
    if not name.startswith(root):
        raise ValueError(f"Unexpected root in arfpy source archive: {name!r}")
    return name[len(root) :]


def _verify_sdist(sdist_path: Path) -> dict[str, Any]:
    if sdist_path.is_symlink() or not sdist_path.is_file():
        raise ValueError(f"arfpy source distribution must be a regular non-symlinked file: {sdist_path}")
    if sdist_path.name != SDIST_FILENAME:
        raise ValueError(f"Expected source archive {SDIST_FILENAME!r}, observed {sdist_path.name!r}")
    observed_sha256 = sha256_file(sdist_path)
    if observed_sha256 != SDIST_SHA256 or sdist_path.stat().st_size != SDIST_BYTES:
        raise ValueError(
            "arfpy source archive identity differs from the PyPI lock: "
            f"sha256={observed_sha256}, bytes={sdist_path.stat().st_size}"
        )

    files: dict[str, bytes] = {}
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != EXPECTED_ARCHIVE_MEMBERS:
            raise ValueError(
                f"arfpy archive member count mismatch: expected={EXPECTED_ARCHIVE_MEMBERS}, observed={len(members)}"
            )
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe path in arfpy source archive: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported link/device member in arfpy source archive: {member.name!r}")
            relative = _archive_member_path(member.name)
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read arfpy archive member: {member.name!r}")
                files[relative] = extracted.read()
    if set(files) != set(EXPECTED_ARCHIVE_FILES):
        raise ValueError(
            "arfpy source archive file set differs from the lock: "
            f"missing={sorted(set(EXPECTED_ARCHIVE_FILES) - set(files))}, "
            f"extra={sorted(set(files) - set(EXPECTED_ARCHIVE_FILES))}"
        )
    observed_file_hashes = {name: _sha256_bytes(payload) for name, payload in files.items()}
    if observed_file_hashes != EXPECTED_ARCHIVE_FILES:
        changed = sorted(
            name for name, expected in EXPECTED_ARCHIVE_FILES.items() if observed_file_hashes.get(name) != expected
        )
        raise ValueError(f"arfpy source archive files differ from the checksum lock: {changed}")

    metadata = Parser().parsestr(files["PKG-INFO"].decode("utf-8"))
    if (
        metadata.get("Name") != PACKAGE_NAME
        or metadata.get("Version") != PACKAGE_VERSION
        or metadata.get("License") != LICENSE_EXPRESSION
        or metadata.get("Home-page") != UPSTREAM_REPOSITORY
    ):
        raise ValueError("arfpy PKG-INFO differs from the locked package identity")
    classifiers = metadata.get_all("Classifier", [])
    if "Programming Language :: Python :: 3.11" not in classifiers:
        raise ValueError("arfpy 0.1.1 does not declare the audited Python 3.11 classifier")
    requirements = files["arfpy.egg-info/requires.txt"].decode("utf-8").splitlines()
    if requirements != EXPECTED_REQUIREMENTS:
        raise ValueError(f"arfpy source requirements differ from the lock: {requirements}")
    git_blobs = {name: _git_blob_sha1(files[name]) for name in EXPECTED_GIT_BLOBS}
    if git_blobs != EXPECTED_GIT_BLOBS:
        raise ValueError("arfpy PyPI source files do not match the recorded method-author Git blobs")
    return {
        "filename": sdist_path.name,
        "bytes": sdist_path.stat().st_size,
        "sha256": observed_sha256,
        "archive_members": len(members),
        "regular_files_verified": len(files),
        "metadata": {
            "name": metadata.get("Name"),
            "version": metadata.get("Version"),
            "license": metadata.get("License"),
            "home_page": metadata.get("Home-page"),
            "python_3_11_classifier": True,
            "requires_python": metadata.get("Requires-Python"),
            "requirements": requirements,
        },
        "git_blob_matches": git_blobs,
        "source_commit": UPSTREAM_COMMIT,
        "source_tree": UPSTREAM_TREE,
        "source_relation": "All six recorded release files are byte-exact Git blobs from the locked commit.",
    }


def _verify_installed_distribution() -> dict[str, Any]:
    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    metadata = distribution.metadata
    if (
        distribution.version != PACKAGE_VERSION
        or metadata.get("Name") != PACKAGE_NAME
        or metadata.get("License") != LICENSE_EXPRESSION
        or metadata.get("Home-page") != UPSTREAM_REPOSITORY
        or list(distribution.requires or []) != EXPECTED_REQUIREMENTS
    ):
        raise ValueError("Installed arfpy metadata differs from the locked official source distribution")

    import arfpy
    import arfpy.arf as arf_module
    import arfpy.utils as utils_module
    from arfpy.arf import arf as model_cls

    package_root = Path(arfpy.__file__).resolve().parent
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not package_root.is_relative_to(distribution_root):
        raise ValueError(f"arfpy did not resolve inside its installed distribution: {package_root}")
    modules = {"__init__.py": arfpy.__file__, "arf.py": arf_module.__file__, "utils.py": utils_module.__file__}
    runtime_hashes = {}
    for name, module_file in modules.items():
        if module_file is None:
            raise ValueError(f"Installed arfpy module has no source file: {name}")
        path = Path(module_file).resolve()
        if path.parent != package_root or path.is_symlink() or not path.is_file():
            raise ValueError(f"Installed arfpy runtime path is unsafe: {path}")
        runtime_hashes[name] = sha256_file(path)
    if runtime_hashes != ARFAdapter.runtime_file_sha256:
        raise ValueError("Installed arfpy runtime source differs from the locked PyPI source distribution")
    if model_cls.__module__ != "arfpy.arf" or model_cls.__name__ != "arf":
        raise ValueError("Installed package does not expose the official arfpy.arf.arf class")

    verified_record_files = 0
    for package_path in distribution.files or ():
        if package_path.hash is None:
            continue
        installed_path = Path(package_path.locate()).resolve()
        if not installed_path.is_file():
            raise ValueError(f"Installed arfpy RECORD path is missing: {installed_path}")
        observed = bytes.fromhex(sha256_file(installed_path))
        if package_path.hash.mode != "sha256" or observed != _decode_record_hash(package_path.hash.value):
            raise ValueError(f"Installed arfpy RECORD checksum mismatch: {package_path}")
        verified_record_files += 1
    license_paths = [
        Path(path.locate()).resolve()
        for path in distribution.files or ()
        if path.name == "LICENSE" and "dist-info" in str(path)
    ]
    if len(license_paths) != 1 or sha256_file(license_paths[0]) != LICENSE_SHA256:
        raise ValueError("Installed arfpy LICENSE differs from the locked MIT license")
    return {
        "version": distribution.version,
        "package_root": str(package_root),
        "distribution_root": str(distribution_root),
        "runtime_file_sha256": runtime_hashes,
        "record_hashes_verified": verified_record_files,
        "license": LICENSE_EXPRESSION,
        "license_sha256": LICENSE_SHA256,
        "public_class": "arfpy.arf.arf",
    }


def verify_package(sdist_path: Path) -> dict[str, Any]:
    return {
        "authority": "method-author",
        "distribution_form": "package-source-distribution",
        "reproduction_target": "method-author-official-python-package",
        "repository": UPSTREAM_REPOSITORY,
        "method_repository": METHOD_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "release_provenance": (
            "The PyPI 0.1.1 source distribution contains byte-exact release files from the locked "
            "method-author commit; the repository has no release tag."
        ),
        "license": LICENSE_EXPRESSION,
        "source_distribution": _verify_sdist(sdist_path.resolve()),
        "installed_distribution": _verify_installed_distribution(),
    }


def _fixture_frame(variant: str) -> tuple[pd.DataFrame, list[str], list[str], list[str], str]:
    row_ids = np.arange(EXPECTED_SOURCE_ROWS)
    continuous = np.round(np.sin(row_ids / 5.0) + row_ids / 20.0, 6)
    segment = np.asarray(["alpha", "beta", "gamma"])[(row_ids // 5) % 3]
    frame = pd.DataFrame(
        {
            "continuous": continuous,
            "linked_value": np.round(2.0 * continuous + (row_ids % 2) * 0.01, 6),
            "segment": segment,
        }
    )
    if variant == "binary":
        frame["target"] = np.where((segment == "alpha") | (row_ids % 7 == 0), "yes", "no")
        return frame, ["continuous", "linked_value"], ["segment"], ["target"], "classification"
    if variant == "multiclass":
        frame["target"] = np.asarray(["class-a", "class-b", "class-c"])[(row_ids // 5) % 3]
        return frame, ["continuous", "linked_value"], ["segment"], ["target"], "classification"
    if variant == "regression":
        frame["target"] = np.round(1.5 * frame["linked_value"] - 0.25 * continuous, 6)
        return frame, ["continuous", "linked_value"], ["segment"], ["target"], "regression"
    raise ValueError(f"Unknown ARF validation variant: {variant}")


def _write_fixtures(root: Path) -> list[tuple[str, pd.DataFrame, DatasetSpec, dict[str, Any]]]:
    fixtures = []
    for variant in VARIANTS:
        fixture_root = root / variant
        fixture_root.mkdir(parents=True, exist_ok=True)
        frame, numerical, categorical, targets, task_type = _fixture_frame(variant)
        train_path = fixture_root / "train.csv"
        metadata_path = fixture_root / "info.json"
        atomic_write_bytes(train_path, frame.to_csv(index=False).encode("utf-8"))
        atomic_write_bytes(
            metadata_path,
            (
                json.dumps(
                    {
                        "name": f"arf-{variant}-parity-fixture",
                        "task_type": task_type,
                        "column_names": list(frame.columns),
                        "numerical_columns": numerical,
                        "categorical_columns": categorical,
                        "target_columns": targets,
                    },
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
        dataset_spec = DatasetSpec(
            name=f"arf-{variant}-parity-fixture",
            task_type=task_type,
            column_names=list(frame.columns),
            numerical_columns=numerical,
            categorical_columns=categorical,
            target_columns=targets,
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
                    "task_type": task_type,
                    "rows": len(loaded),
                    "columns": list(loaded.columns),
                    "numerical_columns": numerical,
                    "categorical_columns": categorical,
                    "target_columns": targets,
                    "missing_values": int(loaded.isna().sum().sum()),
                    "train_csv_sha256": sha256_file(train_path),
                },
            )
        )
    return fixtures


def _typed_frame(frame: pd.DataFrame, dataset_spec: DatasetSpec) -> pd.DataFrame:
    result = frame[dataset_spec.column_names].copy()
    categorical = list(dataset_spec.categorical_columns)
    numerical = list(dataset_spec.numerical_columns)
    if dataset_spec.task_type == "classification":
        categorical.extend(dataset_spec.target_columns)
    else:
        numerical.extend(dataset_spec.target_columns)
    for column in dict.fromkeys(numerical):
        result[column] = pd.to_numeric(result[column], errors="raise")
    for column in dict.fromkeys(categorical):
        result[column] = result[column].astype("category")
    return result


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:])


@contextlib.contextmanager
def _scoped_numpy_seed(seed: int):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def _native_training_params(seed: int) -> dict[str, Any]:
    return {**TRAINING_PARAMS, "random_state": seed}


def _adapter_extra() -> dict[str, Any]:
    return {**TRAINING_PARAMS, **FORDE_PARAMS}


def _run_native(
    frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    from arfpy.arf import arf

    output_dir.mkdir(parents=True, exist_ok=True)
    typed = _typed_frame(frame, dataset_spec)
    before_train = np.random.get_state()
    with _scoped_numpy_seed(seed):
        model = arf(typed.copy(), **_native_training_params(seed))
        model.forde(**FORDE_PARAMS)
    after_train = np.random.get_state()
    before_sample = np.random.get_state()
    with _scoped_numpy_seed(seed):
        samples = model.forge(EXPECTED_SAMPLE_ROWS)
    after_sample = np.random.get_state()
    sample_path = output_dir / "samples.csv"
    atomic_write_bytes(sample_path, samples[dataset_spec.column_names].to_csv(index=False).encode("utf-8"))
    return model, pd.read_csv(sample_path), {
        "sample_path": str(sample_path),
        "sample_sha256": sha256_file(sample_path),
        "adversarial_oob_accuracy": [float(value) for value in model.acc],
        "adversarial_loop_exercised": len(model.acc) == TRAINING_PARAMS["max_iters"] + 1,
        "global_numpy_state_unchanged": _numpy_state_equal(before_train, after_train)
        and _numpy_state_equal(before_sample, after_sample),
    }


def _run_adapter(
    repo_root: Path,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    adapter = ARFAdapter(repo_root)
    common = {
        "model": "arf",
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
    checkpoint_path = output_dir / adapter.checkpoint_filename
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_metadata = json.loads(adapter._metadata_path(checkpoint_path).read_text(encoding="utf-8"))
    before_sample = np.random.get_state()
    sample_bundle = adapter.sample(RunSpec(**common, num_samples=EXPECTED_SAMPLE_ROWS))
    after_sample = np.random.get_state()
    sample_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    sample_metadata = json.loads((output_dir / "arf_sample_metadata.json").read_text(encoding="utf-8"))
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("ARF adapter did not declare a generated sample path")
    restored = adapter._restore_model(adapter._import_model_cls(), checkpoint_payload, dataset_spec)
    manifests_valid = (
        train_bundle.model == "arf"
        and train_manifest["model"] == "arf"
        and train_manifest["dataset"] == dataset_spec.name
        and sample_manifest["generated_sample_path"] == str(sample_bundle.generated_sample_path)
        and Path(train_manifest["upstream_workdir"]).resolve() == repo_root.resolve()
    )
    return restored, pd.read_csv(sample_bundle.generated_sample_path), {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_metadata": checkpoint_metadata,
        "sample_path": str(sample_bundle.generated_sample_path),
        "sample_sha256": sha256_file(sample_bundle.generated_sample_path),
        "sample_metadata": sample_metadata,
        "manifests_valid": manifests_valid,
        "global_numpy_state_unchanged": _numpy_state_equal(before_train, after_train)
        and _numpy_state_equal(before_sample, after_sample),
    }


def _frame_exact(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_exact=True)
    except AssertionError:
        return False
    return True


def _compare_forge_state(native: Any, restored: Any, checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    attributes_exact = bool(
        native.p == restored.p
        and native.num_trees == restored.num_trees
        and native.orig_colnames == restored.orig_colnames
        and native.dist == restored.dist
        and native.factor_cols.equals(restored.factor_cols)
        and native.object_cols.equals(restored.object_cols)
    )
    levels_exact = set(native.levels) == set(restored.levels) and all(
        native.levels[column].equals(restored.levels[column]) for column in native.levels
    )
    frames = {
        "bnds_exact": _frame_exact(native.bnds, restored.bnds),
        "params_exact": _frame_exact(native.params, restored.params),
        "class_probs_exact": _frame_exact(native.class_probs, restored.class_probs),
    }
    accuracy = [ARFAdapter._decode_value(value) for value in checkpoint_payload["training"]["adversarial_oob_accuracy"]]
    return {
        "attributes_exact": attributes_exact,
        "levels_exact": levels_exact,
        **frames,
        "adversarial_oob_accuracy_exact": accuracy == [float(value) for value in native.acc],
        "safe_json_checkpoint": checkpoint_payload.get("format") == "arfpy-forge-state",
        "row_level_training_data_absent": checkpoint_payload.get("privacy", {}).get(
            "contains_row_level_training_data"
        )
        is False,
        "random_forest_absent": checkpoint_payload.get("privacy", {}).get("contains_random_forest") is False,
        "privacy_not_overclaimed": checkpoint_payload.get("privacy", {}).get("privacy_guarantee") == "none",
        "artifact_access_control_required": checkpoint_payload.get("privacy", {}).get(
            "trained_artifact_access_control_required"
        )
        is True,
    }


def _compare_samples(
    native: pd.DataFrame,
    adapter: pd.DataFrame,
    source: pd.DataFrame,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    numerical = list(dataset_spec.numerical_columns)
    categorical = list(dataset_spec.categorical_columns)
    if dataset_spec.task_type == "classification":
        categorical.extend(dataset_spec.target_columns)
    else:
        numerical.extend(dataset_spec.target_columns)
    finite = bool(
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
    state = comparisons["forge_state"]
    samples = comparisons["samples"]
    return bool(
        comparisons["adapter_manifests_valid"]
        and comparisons["checkpoint_metadata_valid"]
        and comparisons["sample_metadata_valid"]
        and comparisons["sample_bytes_exact"]
        and comparisons["native_adversarial_loop_exercised"]
        and comparisons["native_global_numpy_state_unchanged"]
        and comparisons["adapter_global_numpy_state_unchanged"]
        and all(state.values())
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
            f"ARF validation environment does not match its frozen lock: expected={EXPECTED_DISTRIBUTION_VERSIONS}, "
            f"observed={observed}"
        )
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative ARF validation requires Linux and Python 3.11; "
            f"observed platform={platform.platform()!r}, python={python_version!r}"
        )
    return observed


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(repo_root: Path, output_dir: Path, evidence_path: Path, sdist_path: Path) -> dict[str, Any]:
    """Run nine exact official-package-versus-adapter ARF comparisons."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = _verify_environment()
    source_before = verify_package(sdist_path.resolve())
    fixtures = _write_fixtures(output_dir / "fixtures")
    cases: list[dict[str, Any]] = []
    for variant, frame, dataset_spec, _ in fixtures:
        for seed in SEED_CASES:
            native_model, native_samples, native_artifacts = _run_native(
                frame, dataset_spec, output_dir / variant / f"seed-{seed}" / "native", seed
            )
            adapter_dir = output_dir / variant / f"seed-{seed}" / "adapter"
            restored_model, adapter_samples, adapter_artifacts = _run_adapter(
                repo_root, dataset_spec, adapter_dir, seed
            )
            checkpoint_metadata = adapter_artifacts["checkpoint_metadata"]
            sample_metadata = adapter_artifacts["sample_metadata"]
            comparisons = {
                "adapter_manifests_valid": adapter_artifacts["manifests_valid"],
                "checkpoint_metadata_valid": (
                    checkpoint_metadata.get("safe_json_checkpoint") is True
                    and checkpoint_metadata.get("contains_row_level_training_data") is False
                    and checkpoint_metadata.get("contains_random_forest") is False
                    and checkpoint_metadata.get("privacy_guarantee") == "none"
                    and checkpoint_metadata.get("trained_artifact_access_control_required") is True
                    and checkpoint_metadata.get("checkpoint_sha256") == adapter_artifacts["checkpoint_sha256"]
                    and checkpoint_metadata.get("source_rows") == EXPECTED_SOURCE_ROWS
                ),
                "sample_metadata_valid": (
                    sample_metadata.get("requested_rows") == EXPECTED_SAMPLE_ROWS
                    and sample_metadata.get("seed") == seed
                    and sample_metadata.get("sample_sha256") == adapter_artifacts["sample_sha256"]
                    and sample_metadata.get("checkpoint_sha256") == adapter_artifacts["checkpoint_sha256"]
                ),
                "sample_bytes_exact": native_artifacts["sample_sha256"] == adapter_artifacts["sample_sha256"],
                "native_adversarial_loop_exercised": native_artifacts["adversarial_loop_exercised"],
                "native_global_numpy_state_unchanged": native_artifacts["global_numpy_state_unchanged"],
                "adapter_global_numpy_state_unchanged": adapter_artifacts["global_numpy_state_unchanged"],
                "forge_state": _compare_forge_state(
                    native_model, restored_model, adapter_artifacts["checkpoint_payload"]
                ),
                "samples": _compare_samples(native_samples, adapter_samples, frame, dataset_spec),
            }
            cases.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": "pass" if _case_passed(comparisons) else "fail",
                    "native_artifacts": native_artifacts,
                    "adapter_artifacts": {
                        key: value
                        for key, value in adapter_artifacts.items()
                        if key not in {"checkpoint_payload", "checkpoint_metadata", "sample_metadata"}
                    },
                    "comparisons": comparisons,
                }
            )
    source_after = verify_package(sdist_path.resolve())
    package_unchanged = (
        source_before["installed_distribution"]["runtime_file_sha256"]
        == source_after["installed_distribution"]["runtime_file_sha256"]
        and source_before["source_distribution"] == source_after["source_distribution"]
    )
    passed = package_unchanged and all(case["status"] == "pass" for case in cases)
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "arf",
        "status": "pass" if passed else "fail",
        "reproduction_target": "method-author-official-python-package",
        "claim_boundary": (
            "Exact parity with the official method-author arfpy 0.1.1 Python package for mixed-type, "
            "missing-free, flat single-table FORDE/FORGE generation. This does not establish cross-language "
            "equivalence with the separate R package, benchmark eligibility, Official Results, or release support."
        ),
        "repository_commit": _repository_commit(repo_root),
        "source": source_before,
        "source_unchanged_after_validation": package_unchanged,
        "environment_lock": {
            "path": "requirements-arf-validation.txt",
            "sha256": sha256_file(repo_root / "requirements-arf-validation.txt"),
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), **versions},
        "fixtures": [record for _, _, _, record in fixtures],
        "runtime_config": {
            "training_params": TRAINING_PARAMS,
            "forde_params": FORDE_PARAMS,
            "sample_rows": EXPECTED_SAMPLE_ROWS,
            "safe_checkpoint": "arfpy-forge-state JSON without forest or row-level training data",
        },
        "variants": list(VARIANTS),
        "seed_cases": list(SEED_CASES),
        "cases": cases,
    }
    atomic_write_bytes(evidence_path, (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("ARF official-package parity failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ARF official-package native-parity protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--sdist-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path, args.sdist_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            atomic_write_bytes(
                args.evidence_path,
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "protocol_id": PROTOCOL_ID,
                            "model_id": "arf",
                            "status": "fail",
                            "repository_commit": _repository_commit(args.repo_root.resolve()),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
