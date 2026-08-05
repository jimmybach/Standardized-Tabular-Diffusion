"""Executable parity protocol for the official pgmpy-backed BN recipe."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import platform
import stat
import subprocess
import traceback
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.structured_baselines import BNAdapter

PROTOCOL_ID = "pgmpy-bn-recipe-parity-v1"
PACKAGE_NAME = "pgmpy"
PACKAGE_VERSION = "1.1.2"
WHEEL_FILENAME = "pgmpy-1.1.2-py3-none-any.whl"
WHEEL_BYTES = 2_446_383
WHEEL_SHA256 = "e55c78763a4a45dd644a13b250cea86af0c7e08590cf35de489624f34a4d9a0b"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/c6/5d/"
    "d03634ed296986abad834a69b0df21510cc9b6c40fb8afaed5df1c4b6074/"
    "pgmpy-1.1.2-py3-none-any.whl"
)
UPSTREAM_REPOSITORY = "https://github.com/pgmpy/pgmpy"
UPSTREAM_TAG = "v1.1.2"
UPSTREAM_TAG_OBJECT = "ff663f9203c5075b2367707917016efafed03593"
UPSTREAM_COMMIT = BNAdapter.upstream_commit
UPSTREAM_TREE = BNAdapter.upstream_tree
LICENSE_EXPRESSION = "MIT"
LICENSE_SHA256 = "89171dcc8977530b0c101fbbb1c1d34caee998fc7def9eded629753cd2616a15"
EXPECTED_WHEEL_FILES = 649
EXPECTED_PACKAGE_FILES = 636
EXPECTED_RECORD_HASHES = 648
EXPECTED_GIT_BLOBS = {
    "pgmpy/__init__.py": "7589aa93784ec90a0cfa173caa548fbc6d7a0a59",
    "pgmpy/causal_discovery/HillClimbSearch.py": "6271ab19fdbd5f130e2917ccf112701ae9aba5e3",
    "pgmpy/causal_discovery/_base.py": "fce2baf91a37aea3e215900b5fafa0d7083215c9",
    "pgmpy/structure_score/bic.py": "a22aeff0a50be20800c5d907fb168aded93ff3a0",
    "pgmpy/structure_score/log_likelihood.py": "1acb6a3f1e9ce798dec050a3fd9c145149b7ec20",
    "pgmpy/parameter_estimator/discrete_bayesian.py": "68b18bacde998f39162c7797e1357050085ed40a",
    "pgmpy/models/DiscreteBayesianNetwork.py": "80469c7c3682f225b534416805df804e647b7825",
    "pgmpy/sampling/Sampling.py": "e11b39194668c0b800a14993439bc6305641abda",
    "pgmpy/factors/discrete/CPD.py": "e4b49b42fa08c2dba4aacf9faa8efd35fadf00fc",
    "pgmpy-1.1.2.dist-info/licenses/LICENSE": "03ecd36bef85c9ec40fbb2c7bc9cb84a755c96af",
}
EXPECTED_RUNTIME_SHA256 = {f"pgmpy/{path}": digest for path, digest in BNAdapter.runtime_file_sha256.items()}
EXPECTED_REQUIREMENTS = [
    "huggingface_hub>=0.23",
    "networkx>=3.0",
    "numpy>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "pandas>=1.5",
    "statsmodels>=0.14.5",
    "tqdm>=4.64",
    "pyparsing>=3.0",
    "joblib>=1.2",
    "opt_einsum>=3.3",
    "scikit-base>=0.12.4",
    'torch>=2.5; extra == "torch"',
    'pyro-ppl>=1.9.1; extra == "torch"',
    'pgmpy[torch]; extra == "optional"',
    'daft-pgm>=0.1.4; extra == "optional"',
    'xgboost>=2.0.3; extra == "optional"',
    'litellm>=1.61.15; extra == "optional"',
    'pygraphviz; extra == "optional"',
    'xdoctest>=0.11.0; extra == "tests"',
    'pytest>=3.3.1; extra == "tests"',
    'pytest-cov; extra == "tests"',
    'pytest-split; extra == "tests"',
    'pytest-xdist; extra == "tests"',
    'coverage>=4.3.4; extra == "tests"',
    'mock; extra == "tests"',
    'black; extra == "tests"',
    'pre-commit; extra == "tests"',
    'jsonschema; extra == "tests"',
    'sempler; extra == "tests"',
    'sphinx>=5.0; extra == "docs"',
    'ipython; extra == "docs"',
    'nbsphinx; extra == "docs"',
    'numpydoc; extra == "docs"',
    'pydata-sphinx-theme; extra == "docs"',
    'sphinx-copybutton; extra == "docs"',
    'sphinx-design; extra == "docs"',
    'sphinxext-opengraph; extra == "docs"',
    'sphinx_sitemap; extra == "docs"',
    'myst_parser; extra == "docs"',
    'pgmpy[docs,optional,tests]; extra == "all"',
]
EXPECTED_DISTRIBUTION_VERSIONS = {
    "anyio": "4.14.2",
    "certifi": "2026.7.22",
    "click": "8.4.2",
    "filelock": "3.32.2",
    "fsspec": "2026.7.0",
    "h11": "0.16.0",
    "hf-xet": "1.6.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.26.0",
    "idna": "3.18",
    "joblib": "1.5.3",
    "narwhals": "2.24.0",
    "networkx": "3.6.1",
    "numpy": "2.4.6",
    "opt-einsum": "3.4.0",
    "packaging": "26.3",
    "pandas": "3.0.5",
    "patsy": "1.0.2",
    "pgmpy": PACKAGE_VERSION,
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "PyYAML": "6.0.3",
    "scikit-base": "1.1.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.17.1",
    "setuptools": "80.10.2",
    "six": "1.17.0",
    "statsmodels": "0.14.6",
    "threadpoolctl": "3.6.0",
    "tqdm": "4.70.0",
    "typing-extensions": "4.16.0",
    "tzdata": "2026.3",
    "wheel": "0.45.1",
}
EXPECTED_SOURCE_ROWS = 60
EXPECTED_SAMPLE_ROWS = 13
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
RECIPE = {
    "num_bins": 6,
    "quantile_method": "averaged_inverted_cdf",
    "subsample": None,
    "scoring_method": "bic-d",
    "return_type": "dag",
    "max_indegree": 2,
    "max_iter": 100,
    "tabu_length": 20,
    "epsilon": 1e-4,
    "prior_type": "BDeu",
    "equivalent_sample_size": 5.0,
    "n_jobs": 1,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()  # noqa: S324 - Git object identity uses SHA-1.


def _decode_record_hash(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _verify_wheel(wheel_path: Path) -> dict[str, Any]:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(f"pgmpy wheel must be a regular non-symlinked file: {wheel_path}")
    if wheel_path.name != WHEEL_FILENAME:
        raise ValueError(f"Expected wheel {WHEEL_FILENAME!r}, observed {wheel_path.name!r}")
    observed_sha256 = sha256_file(wheel_path)
    if observed_sha256 != WHEEL_SHA256 or wheel_path.stat().st_size != WHEEL_BYTES:
        raise ValueError(
            "pgmpy wheel identity differs from the PyPI lock: "
            f"sha256={observed_sha256}, bytes={wheel_path.stat().st_size}"
        )
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != EXPECTED_WHEEL_FILES:
            raise ValueError(f"pgmpy wheel file count mismatch: expected={EXPECTED_WHEEL_FILES}, observed={len(infos)}")
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or info.flag_bits & 0x1
                or stat.S_IFMT(mode) == stat.S_IFLNK
                or info.filename in files
            ):
                raise ValueError(f"Unsafe or duplicate member in pgmpy wheel: {info.filename!r}")
            files[info.filename] = archive.read(info)
    package_files = {path for path in files if path.startswith("pgmpy/")}
    if len(package_files) != EXPECTED_PACKAGE_FILES:
        raise ValueError("pgmpy package file count differs from the audited wheel/source/tag comparison")
    for path, expected in EXPECTED_RUNTIME_SHA256.items():
        if path not in files or _sha256_bytes(files[path]) != expected:
            raise ValueError(f"pgmpy wheel runtime file differs from the lock: {path}")
    license_path = "pgmpy-1.1.2.dist-info/licenses/LICENSE"
    if _sha256_bytes(files.get(license_path, b"")) != LICENSE_SHA256:
        raise ValueError("pgmpy wheel license differs from the MIT license lock")
    git_blobs = {path: _git_blob_sha1(files[path]) for path in EXPECTED_GIT_BLOBS}
    if git_blobs != EXPECTED_GIT_BLOBS:
        raise ValueError("pgmpy wheel critical files differ from the recorded official Git blobs")
    metadata = Parser().parsestr(files["pgmpy-1.1.2.dist-info/METADATA"].decode("utf-8"))
    if (
        metadata.get("Name") != PACKAGE_NAME
        or metadata.get("Version") != PACKAGE_VERSION
        or metadata.get("Requires-Python") != "<3.15,>=3.10"
        or metadata.get_all("Requires-Dist", []) != EXPECTED_REQUIREMENTS
        or "Repository, https://github.com/pgmpy/pgmpy" not in metadata.get_all("Project-URL", [])
        or "Programming Language :: Python :: 3.11" not in metadata.get_all("Classifier", [])
    ):
        raise ValueError("pgmpy wheel metadata differs from the package lock")
    return {
        "filename": wheel_path.name,
        "bytes": wheel_path.stat().st_size,
        "sha256": observed_sha256,
        "wheel_files_verified": len(files),
        "package_files_verified": len(package_files),
        "critical_git_blob_matches": git_blobs,
        "license": LICENSE_EXPRESSION,
        "license_sha256": LICENSE_SHA256,
        "requires_python": metadata.get("Requires-Python"),
        "requirements": metadata.get_all("Requires-Dist", []),
        "source_commit": UPSTREAM_COMMIT,
        "source_tree": UPSTREAM_TREE,
        "source_tag": UPSTREAM_TAG,
        "source_tag_object": UPSTREAM_TAG_OBJECT,
        "trusted_publishing": "PyPI provenance binds both 1.1.2 distributions to the locked tag commit.",
        "audited_source_relation": (
            "All 636 pgmpy files are byte-exact across the wheel, source distribution, and locked Git tag."
        ),
    }


def _verify_installed_distribution(wheel_path: Path) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    if distribution.version != PACKAGE_VERSION or list(distribution.requires or []) != EXPECTED_REQUIREMENTS:
        raise ValueError("Installed pgmpy distribution metadata differs from the locked official wheel")
    package = __import__("pgmpy")
    package_root = Path(package.__file__).resolve().parent
    runtime_hashes: dict[str, str] = {}
    for relative_path, expected in BNAdapter.runtime_file_sha256.items():
        path = package_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Installed pgmpy runtime file is missing or unsafe: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Installed pgmpy runtime file differs from the lock: {relative_path}")
        runtime_hashes[relative_path] = observed
    # pip is required to rewrite the installed RECORD: it adds INSTALLER,
    # REQUESTED/direct_url.json and interpreter-specific ``__pycache__`` files.
    # Use the already authenticated wheel RECORD as the immutable manifest and
    # verify every payload member against its installed counterpart instead of
    # treating those standards-compliant installation additions as upstream
    # content changes.
    with zipfile.ZipFile(wheel_path) as archive:
        rows = list(csv.reader(archive.read("pgmpy-1.1.2.dist-info/RECORD").decode("utf-8").splitlines()))
    verified_hashes = 0
    for relative_path, encoded_hash, size_text in rows:
        path = Path(distribution.locate_file(relative_path))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Installed pgmpy RECORD path is missing or unsafe: {relative_path}")
        if encoded_hash:
            algorithm, digest = encoded_hash.split("=", maxsplit=1)
            if algorithm != "sha256" or hashlib.sha256(path.read_bytes()).digest() != _decode_record_hash(digest):
                raise ValueError(f"Installed pgmpy RECORD hash mismatch: {relative_path}")
            if int(size_text) != path.stat().st_size:
                raise ValueError(f"Installed pgmpy RECORD size mismatch: {relative_path}")
            verified_hashes += 1
    if len(rows) != EXPECTED_WHEEL_FILES or verified_hashes != EXPECTED_RECORD_HASHES:
        raise ValueError("Installed pgmpy RECORD coverage differs from the wheel lock")
    license_path = Path(distribution.locate_file("pgmpy-1.1.2.dist-info/licenses/LICENSE"))
    if sha256_file(license_path) != LICENSE_SHA256:
        raise ValueError("Installed pgmpy license differs from the wheel lock")
    api = BNAdapter(Path.cwd())._import_official_api()
    return {
        "version": distribution.version,
        "package_root": str(package_root),
        "distribution_root": str(Path(distribution.locate_file(".")).resolve()),
        "runtime_file_sha256": runtime_hashes,
        "wheel_record_rows_verified": len(rows),
        "wheel_record_hashes_verified": verified_hashes,
        "license": LICENSE_EXPRESSION,
        "license_sha256": LICENSE_SHA256,
        "public_classes": {name: f"{value.__module__}.{value.__name__}" for name, value in api.items()},
    }


def verify_package(wheel_path: Path) -> dict[str, Any]:
    return {
        "authority": "canonical-library",
        "repository": UPSTREAM_REPOSITORY,
        "reproduction_target": "official-pgmpy-package-plus-declared-bn-recipe",
        "distribution_form": "package-wheel",
        "license": LICENSE_EXPRESSION,
        "wheel": _verify_wheel(wheel_path),
        "installed_distribution": _verify_installed_distribution(wheel_path),
    }


def _verify_environment() -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTION_VERSIONS}
    if observed != EXPECTED_DISTRIBUTION_VERSIONS:
        changed = {
            name: {"expected": expected, "observed": observed.get(name)}
            for name, expected in EXPECTED_DISTRIBUTION_VERSIONS.items()
            if observed.get(name) != expected
        }
        raise RuntimeError(f"BN validation dependency lock mismatch: {changed}")
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative BN validation requires Linux and Python 3.11; "
            f"observed platform={platform.platform()!r}, python={python_version!r}"
        )
    return observed


def _fixture(variant: str) -> tuple[pd.DataFrame, str]:
    row = np.arange(EXPECTED_SOURCE_ROWS)
    continuous = (row % 13).astype(float) + row * 0.1
    linked_value = ((row % 13) > 5).astype(float) + (row % 3) * 0.1
    segment = np.where(row % 3 == 0, "alpha", np.where(row % 3 == 1, "beta", "gamma"))
    if variant == "binary":
        target: Any = np.where((row % 4) < 2, "yes", "no")
        task_type = "classification"
    elif variant == "multiclass":
        target = np.where(row % 3 == 0, "red", np.where(row % 3 == 1, "green", "blue"))
        task_type = "classification"
    elif variant == "regression":
        target = continuous * 0.4 + linked_value * 2.0 + (row % 5) * 0.05
        task_type = "regression"
    else:
        raise ValueError(f"Unknown BN validation variant: {variant}")
    return (
        pd.DataFrame(
            {
                "continuous": continuous,
                "linked_value": linked_value,
                "constant_value": np.full(EXPECTED_SOURCE_ROWS, 3.25),
                "segment": segment,
                "target": target,
            }
        ),
        task_type,
    )


def _write_fixtures(root: Path) -> list[tuple[str, pd.DataFrame, DatasetSpec, dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for variant in VARIANTS:
        frame, task_type = _fixture(variant)
        variant_root = root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        train_path = variant_root / "train.csv"
        metadata_path = variant_root / "info.json"
        atomic_write_bytes(train_path, frame.to_csv(index=False).encode("utf-8"))
        atomic_write_bytes(metadata_path, b"{}\n")
        # The adapter contract begins at the persisted canonical CSV. Read the
        # fixture back through that same boundary so the direct-package oracle
        # sees identical IEEE-754 values (rather than pre-serialization values
        # that can differ by one representable float after CSV parsing).
        canonical_frame = pd.read_csv(train_path)
        spec = DatasetSpec(
            name=f"bn-{variant}-fixture",
            task_type=task_type,
            column_names=list(frame.columns),
            numerical_columns=["continuous", "linked_value", "constant_value"],
            categorical_columns=["segment"],
            target_columns=["target"],
            metadata_path=metadata_path,
            train_data_path=train_path,
        )
        record = {
            "variant": variant,
            "task_type": task_type,
            "rows": len(frame),
            "columns": list(frame.columns),
            "numerical_columns": spec.numerical_columns,
            "categorical_columns": spec.categorical_columns,
            "target_columns": spec.target_columns,
            "missing_values": int(frame.isna().sum().sum()),
            "train_csv_sha256": sha256_file(train_path),
        }
        fixtures.append((variant, canonical_frame, spec, record))
    return fixtures


def _reference_preprocess(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    numerical = list(spec.numerical_columns)
    categorical = list(spec.categorical_columns)
    if spec.task_type == "regression":
        numerical.extend(spec.target_columns)
    else:
        categorical.extend(spec.target_columns)
    output = pd.DataFrame(index=frame.index)
    constants: dict[str, float] = {}
    variable_numeric = []
    states: dict[str, list[str]] = {}
    for column in numerical:
        values = frame[column].to_numpy(dtype=float)
        if np.all(values == values[0]):
            constants[column] = float(values[0])
            output[column] = "0"
            states[column] = ["0"]
        else:
            variable_numeric.append(column)
    edges: dict[str, list[float]] = {}
    if variable_numeric:
        binner = KBinsDiscretizer(
            n_bins=RECIPE["num_bins"],
            encode="ordinal",
            strategy="quantile",
            quantile_method=RECIPE["quantile_method"],
            subsample=RECIPE["subsample"],
        )
        transformed = binner.fit_transform(frame[variable_numeric])
        for index, column in enumerate(variable_numeric):
            output[column] = transformed[:, index].astype(int).astype(str)
            edges[column] = [float(value) for value in binner.bin_edges_[index]]
            states[column] = [str(value) for value in range(len(edges[column]) - 1)]
    levels: dict[str, list[str]] = {}
    for column in categorical:
        output[column] = frame[column].astype(str)
        levels[column] = sorted(output[column].unique().tolist())
        states[column] = levels[column]
    state = {
        "num_bins": RECIPE["num_bins"],
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "bin_edges": edges,
        "constant_values": constants,
        "categorical_levels": levels,
        "discrete_state_names": states,
        "quantile_method": RECIPE["quantile_method"],
        "subsample": RECIPE["subsample"],
    }
    return output[spec.column_names], state


def _reference_decode(
    discrete: pd.DataFrame, spec: DatasetSpec, preprocessing: dict[str, Any], seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = pd.DataFrame(index=discrete.index)
    for column in preprocessing["numerical_columns"]:
        if column in preprocessing["constant_values"]:
            output[column] = preprocessing["constant_values"][column]
            continue
        edges = preprocessing["bin_edges"][column]
        values = []
        for state in discrete[column].astype(int).tolist():
            lower, upper = edges[state], edges[state + 1]
            values.append(float(lower) if lower == upper else float(rng.uniform(lower, upper)))
        output[column] = values
    for column in preprocessing["categorical_columns"]:
        output[column] = discrete[column].astype(str)
    return output[spec.column_names]


def _cpd_records(model: Any, columns: list[str]) -> list[dict[str, Any]]:
    records = []
    for column in columns:
        cpd = model.get_cpds(column)
        variables = [str(value) for value in cpd.variables]
        records.append(
            {
                "variable": str(cpd.variable),
                "variable_card": int(cpd.variable_card),
                "evidence": variables[1:],
                "evidence_card": [int(value) for value in cpd.cardinality[1:]],
                "state_names": {
                    variable: [str(value) for value in cpd.state_names[variable]] for variable in variables
                },
                "values": np.asarray(cpd.get_values(), dtype=float).tolist(),
            }
        )
    return records


def _fit_reference(api: dict[str, type], discrete: pd.DataFrame, columns: list[str]) -> Any:
    search = api["HillClimbSearch"](
        scoring_method=RECIPE["scoring_method"],
        return_type=RECIPE["return_type"],
        max_indegree=RECIPE["max_indegree"],
        max_iter=RECIPE["max_iter"],
        tabu_length=RECIPE["tabu_length"],
        epsilon=RECIPE["epsilon"],
        show_progress=False,
    )
    graph = search.fit(discrete).causal_graph_
    model = api["DiscreteBayesianNetwork"]()
    model.add_nodes_from(columns)
    model.add_edges_from(sorted(graph.edges()))
    estimator = api["DiscreteBayesianEstimator"](
        prior_type=RECIPE["prior_type"],
        equivalent_sample_size=RECIPE["equivalent_sample_size"],
        n_jobs=RECIPE["n_jobs"],
    )
    model.fit(discrete, estimator=estimator)
    if model.check_model() is not True:
        raise AssertionError("Direct official pgmpy model failed check_model")
    return model


def _sample_official(api: dict[str, type], model: Any, columns: list[str], seed: int) -> tuple[pd.DataFrame, bool]:
    before = np.random.get_state()
    try:
        np.random.seed(seed)
        sample = api["BayesianModelSampling"](model).forward_sample(
            size=EXPECTED_SAMPLE_ROWS, seed=seed, show_progress=False, n_jobs=1
        )
    finally:
        np.random.set_state(before)
    unchanged = _numpy_state_equal(before, np.random.get_state())
    return sample[columns].copy(), unchanged


def _run_native(
    api: dict[str, type], frame: pd.DataFrame, spec: DatasetSpec, output_dir: Path, seed: int
) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    discrete, preprocessing = _reference_preprocess(frame, spec)
    model = _fit_reference(api, discrete, spec.column_names)
    raw_sample, rng_unchanged = _sample_official(api, model, spec.column_names, seed)
    final_sample = _reference_decode(raw_sample, spec, preprocessing, seed)
    raw_path = output_dir / "discrete-samples.csv"
    sample_path = output_dir / "samples.csv"
    atomic_write_bytes(raw_path, raw_sample.to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(sample_path, final_sample.to_csv(index=False).encode("utf-8"))
    # Compare the public CSV artifact after parsing on both paths. This keeps
    # the semantic frame check aligned with the already exact byte-level check
    # and avoids comparing an in-memory float to its CSV-round-tripped value.
    canonical_sample = pd.read_csv(sample_path)
    return (
        model,
        raw_sample,
        canonical_sample,
        preprocessing,
        {
            "edges": sorted([list(edge) for edge in model.edges()]),
            "cpds": _cpd_records(model, spec.column_names),
            "discrete_training_sha256": _sha256_bytes(discrete.to_csv(index=False).encode("utf-8")),
            "discrete_sample_sha256": sha256_file(raw_path),
            "sample_sha256": sha256_file(sample_path),
            "global_numpy_state_unchanged": rng_unchanged,
        },
    )


def _run_adapter(
    repo_root: Path, spec: DatasetSpec, output_dir: Path, seed: int
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    adapter = BNAdapter(repo_root)
    extra = {
        "dataset_spec": spec.to_dict(),
        **{key: value for key, value in RECIPE.items() if key not in {"quantile_method", "subsample", "return_type"}},
    }
    common = {
        "model": "bn",
        "dataset": spec.name,
        "output_dir": output_dir,
        "device": "cpu",
        "seed": seed,
        "extra": extra,
    }
    before = np.random.get_state()
    train_bundle = adapter.train(RunSpec(**common))
    train_manifest = read_json(output_dir / "artifacts.json")
    sample_bundle = adapter.sample(RunSpec(**common, num_samples=EXPECTED_SAMPLE_ROWS))
    unchanged = _numpy_state_equal(before, np.random.get_state())
    sample_manifest = read_json(output_dir / "artifacts.json")
    checkpoint_path = output_dir / adapter.checkpoint_filename
    checkpoint_payload = read_json(checkpoint_path)
    checkpoint_metadata = read_json(adapter._metadata_path(checkpoint_path))
    sample_metadata = read_json(output_dir / "bn_sample_metadata.json")
    api = adapter._import_official_api()
    restored_model, _ = adapter._restore_model(api, checkpoint_payload, spec)
    sample_path = output_dir / "samples.csv"
    sample = pd.read_csv(sample_path)
    manifests_valid = (
        train_bundle.model == sample_bundle.model == "bn"
        and train_manifest.get("model") == sample_manifest.get("model") == "bn"
        and train_manifest.get("generated_sample_path") is None
        and sample_manifest.get("generated_sample_path") == str(sample_path)
    )
    return (
        restored_model,
        sample,
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "sample_path": str(sample_path),
            "sample_sha256": sha256_file(sample_path),
            "global_numpy_state_unchanged": unchanged,
            "manifests_valid": manifests_valid,
            "checkpoint_payload": checkpoint_payload,
            "checkpoint_metadata": checkpoint_metadata,
            "sample_metadata": sample_metadata,
        },
    )


def _compare_samples(
    native: pd.DataFrame, adapter: pd.DataFrame, training: pd.DataFrame, spec: DatasetSpec
) -> dict[str, Any]:
    numerical = list(spec.numerical_columns)
    categorical = list(spec.categorical_columns)
    if spec.task_type == "regression":
        numerical.extend(spec.target_columns)
    else:
        categorical.extend(spec.target_columns)
    numeric_in_range = all(
        bool(adapter[column].between(training[column].min(), training[column].max(), inclusive="both").all())
        for column in numerical
    )
    categorical_valid = all(
        set(adapter[column].astype(str)) <= set(training[column].astype(str)) for column in categorical
    )
    return {
        "rows": len(adapter),
        "columns_exact": list(adapter.columns) == spec.column_names,
        "frame_exact": native.equals(adapter),
        "finite_numerical": bool(np.isfinite(adapter[numerical].to_numpy(dtype=float)).all()),
        "numeric_ranges_valid": numeric_in_range,
        "constant_numeric_exact": bool((adapter["constant_value"] == 3.25).all()),
        "categorical_domains_valid": categorical_valid,
        "missing_values": int(adapter.isna().sum().sum()),
    }


def _case_passed(comparisons: dict[str, Any]) -> bool:
    model = comparisons["model_state"]
    samples = comparisons["samples"]
    required = [
        comparisons["preprocessing_exact"],
        comparisons["discrete_training_exact"],
        comparisons["graph_edges_exact"],
        comparisons["cpds_exact"],
        comparisons["restored_model_exact"],
        comparisons["raw_discrete_sample_exact"],
        comparisons["sample_bytes_exact"],
        comparisons["adapter_manifests_valid"],
        comparisons["checkpoint_metadata_valid"],
        comparisons["sample_metadata_valid"],
        comparisons["native_global_numpy_state_unchanged"],
        comparisons["adapter_global_numpy_state_unchanged"],
        model["safe_json_checkpoint"],
        model["row_level_training_data_absent"],
        model["privacy_not_overclaimed"],
        model["artifact_access_control_required"],
        samples["columns_exact"],
        samples["frame_exact"],
        samples["finite_numerical"],
        samples["numeric_ranges_valid"],
        samples["constant_numeric_exact"],
        samples["categorical_domains_valid"],
        samples["rows"] == EXPECTED_SAMPLE_ROWS,
        samples["missing_values"] == 0,
    ]
    return all(value is True for value in required)


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(repo_root: Path, output_dir: Path, evidence_path: Path, wheel_path: Path) -> dict[str, Any]:
    """Run nine exact official-pgmpy-versus-adapter BN recipe comparisons."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = _verify_environment()
    source_before = verify_package(wheel_path.resolve())
    api = BNAdapter(repo_root)._import_official_api()
    fixtures = _write_fixtures(output_dir / "fixtures")
    cases: list[dict[str, Any]] = []
    for variant, frame, spec, _ in fixtures:
        reference_discrete, reference_preprocessing = _reference_preprocess(frame, spec)
        for seed in SEED_CASES:
            native_model, native_raw, native_samples, native_preprocessing, native_artifacts = _run_native(
                api, frame, spec, output_dir / variant / f"seed-{seed}" / "native", seed
            )
            adapter_model, adapter_samples, adapter_artifacts = _run_adapter(
                repo_root, spec, output_dir / variant / f"seed-{seed}" / "adapter", seed
            )
            payload = adapter_artifacts["checkpoint_payload"]
            checkpoint_metadata = adapter_artifacts["checkpoint_metadata"]
            sample_metadata = adapter_artifacts["sample_metadata"]
            checkpoint_bytes = Path(adapter_artifacts["checkpoint_path"]).read_bytes()
            native_cpds = _cpd_records(native_model, spec.column_names)
            restored_cpds = _cpd_records(adapter_model, spec.column_names)
            comparisons = {
                "preprocessing_exact": (
                    native_preprocessing == reference_preprocessing == payload.get("preprocessing")
                ),
                "discrete_training_exact": (
                    native_artifacts["discrete_training_sha256"]
                    == _sha256_bytes(reference_discrete.to_csv(index=False).encode("utf-8"))
                    == payload["training"].get("discrete_frame_sha256")
                ),
                "graph_edges_exact": (
                    native_artifacts["edges"]
                    == payload["model"].get("edges")
                    == sorted([list(edge) for edge in adapter_model.edges()])
                ),
                "cpds_exact": native_cpds == payload["model"].get("cpds"),
                "restored_model_exact": native_cpds == restored_cpds,
                "raw_discrete_sample_exact": (
                    native_artifacts["discrete_sample_sha256"] == sample_metadata.get("discrete_sample_sha256")
                ),
                "sample_bytes_exact": native_artifacts["sample_sha256"] == adapter_artifacts["sample_sha256"],
                "adapter_manifests_valid": adapter_artifacts["manifests_valid"],
                "checkpoint_metadata_valid": (
                    checkpoint_metadata.get("safe_json_checkpoint") is True
                    and checkpoint_metadata.get("contains_row_level_training_data") is False
                    and checkpoint_metadata.get("code_executing_checkpoint") is False
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
                "native_global_numpy_state_unchanged": native_artifacts["global_numpy_state_unchanged"],
                "adapter_global_numpy_state_unchanged": adapter_artifacts["global_numpy_state_unchanged"],
                "model_state": {
                    "safe_json_checkpoint": b"pickle" not in checkpoint_bytes.lower(),
                    "row_level_training_data_absent": payload["privacy"].get("contains_row_level_training_data")
                    is False,
                    "privacy_not_overclaimed": payload["privacy"].get("privacy_guarantee") == "none",
                    "artifact_access_control_required": payload["privacy"].get(
                        "trained_artifact_access_control_required"
                    )
                    is True,
                },
                "samples": _compare_samples(native_samples, adapter_samples, frame, spec),
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
    source_after = verify_package(wheel_path.resolve())
    package_unchanged = (
        source_before["installed_distribution"]["runtime_file_sha256"]
        == source_after["installed_distribution"]["runtime_file_sha256"]
        and source_before["wheel"] == source_after["wheel"]
    )
    passed = package_unchanged and all(case["status"] == "pass" for case in cases)
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "bn",
        "status": "pass" if passed else "fail",
        "reproduction_target": "official-pgmpy-package-plus-declared-bn-recipe",
        "claim_boundary": (
            "Exact parity with direct calls to official pgmpy 1.1.2 for the repository-declared discretized "
            "BIC hill-climb and BDeu Bayesian-network recipe. This is not a paper-native implementation, does "
            "not validate alternative BN recipes, and does not establish benchmark eligibility or release support."
        ),
        "repository_commit": _repository_commit(repo_root),
        "source": source_before,
        "source_unchanged_after_validation": package_unchanged,
        "environment_lock": {
            "path": "requirements-bn-validation.txt",
            "sha256": sha256_file(repo_root / "requirements-bn-validation.txt"),
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), **versions},
        "fixtures": [record for _, _, _, record in fixtures],
        "runtime_config": {
            "recipe": RECIPE,
            "sample_rows": EXPECTED_SAMPLE_ROWS,
            "safe_checkpoint": "JSON graph, CPD, and preprocessing state without executable pickle or rows",
        },
        "variants": list(VARIANTS),
        "seed_cases": list(SEED_CASES),
        "cases": cases,
    }
    atomic_write_bytes(evidence_path, (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("BN official-package recipe parity failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BN official-package recipe-parity protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path, args.wheel_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            atomic_write_bytes(
                args.evidence_path,
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "protocol_id": PROTOCOL_ID,
                            "model_id": "bn",
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
