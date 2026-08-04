"""Executable native-parity protocol for the official CTAB-GAN source."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import pickle
import platform
import random
import subprocess
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import isolated_module_tree
from standardized_tabular_diffusion.models.ctabgan import CTABGANAdapter
from standardized_tabular_diffusion.upstream_sources import source_manifest_path, validate_upstream_source

PROTOCOL_ID = "ctabgan-native-parity-v1"
MODEL_ID = "ctab-gan"
UPSTREAM_REPOSITORY = "https://github.com/Team-TUD/CTAB-GAN"
UPSTREAM_COMMIT = "73d4e315a2a51cf16c97ed8a00d2dad456cfce8a"
UPSTREAM_TREE = "3ef0223477193400d88344ff66b7ac6ffeefa173"
UPSTREAM_MODEL_TREE = "89ad16bce9f0f6c23f393d9b6b2959ce8ef64bf9"
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary-classification", "multiclass-classification")
SOURCE_ROWS = 40
SAMPLE_ROWS = 13
TRAINING_PARAMETERS = {
    "epochs": 1,
    "batch_size": 8,
    "random_dim": 8,
    "num_channels": 4,
    "class_dim": [8, 8],
    "l2scale": 1e-5,
    "num_threads": 1,
}
EXPECTED_VERSIONS = {
    "joblib": "1.5.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
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


def _numpy_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:])


@contextlib.contextmanager
def _seeded_runtime(seed: int, torch: Any):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    previous_threads = torch.get_num_threads()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(TRAINING_PARAMETERS["num_threads"])
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_num_threads(previous_threads)


def _verify_environment() -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in EXPECTED_VERSIONS.items()
        if observed[name] != expected and not (name == "torch" and observed[name].startswith(f"{expected}+"))
    }
    python_version = platform.python_version()
    if platform.system() != "Linux" or not python_version.startswith("3.11."):
        raise RuntimeError(
            "Authoritative CTAB-GAN validation requires Linux and Python 3.11; "
            f"observed platform={platform.platform()!r}, python={python_version!r}"
        )
    if mismatches:
        raise RuntimeError(f"CTAB-GAN validation environment differs from its frozen lock: {mismatches}")
    return observed


def _fixture_frame(variant: str) -> pd.DataFrame:
    row = np.arange(SOURCE_ROWS)
    frame = pd.DataFrame(
        {
            "continuous": np.round(np.sin(row / 4.0) + row / 20.0, 4),
            "count": ((row * 7) % 23).astype(int),
            "segment": np.asarray(["alpha", "beta", "gamma", "delta"])[row % 4],
        }
    )
    if variant == "binary-classification":
        frame["target"] = np.asarray(["no", "yes"])[row % 2]
    elif variant == "multiclass-classification":
        frame["target"] = np.asarray(["class-a", "class-b", "class-c", "class-d"])[(row * 3) % 4]
    else:
        raise ValueError(f"Unknown CTAB-GAN validation variant: {variant}")
    return frame


def _write_fixture(root: Path, variant: str) -> tuple[pd.DataFrame, DatasetSpec, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    frame = _fixture_frame(variant)
    train_path = root / "train.csv"
    metadata_path = root / "info.json"
    atomic_write_bytes(train_path, frame.to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(
        metadata_path,
        (
            json.dumps({"name": f"ctabgan-{variant}-parity", "task_type": "classification"}, indent=2)
            + "\n"
        ).encode(),
    )
    numerical = ["continuous", "count"]
    categorical = ["segment"]
    spec = DatasetSpec(
        name=f"ctabgan-{variant}-parity",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=numerical,
        categorical_columns=categorical,
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )
    return pd.read_csv(train_path), spec, {
        "variant": variant,
        "rows": len(frame),
        "columns": list(frame.columns),
        "train_csv_sha256": _sha256_file(train_path),
        "missing_values": int(frame.isna().sum().sum()),
    }


def _official_parameters(dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "test_ratio": 0.2,
        "categorical_columns": [*dataset_spec.categorical_columns, *dataset_spec.target_columns],
        "log_columns": [],
        "mixed_columns": {},
        "integer_columns": ["count"],
        "problem_type": {"Classification": "target"},
    }


def _configure_synthesizer(model: Any, torch: Any) -> None:
    for name in ("epochs", "batch_size", "random_dim", "num_channels", "class_dim", "l2scale"):
        value = TRAINING_PARAMETERS[name]
        setattr(model.synthesizer, name, tuple(value) if name == "class_dim" else value)
    model.synthesizer.device = torch.device("cpu")


def _model_signature(model: Any) -> dict[str, Any]:
    generator_state = {}
    for name, tensor in model.synthesizer.generator.state_dict().items():
        array = tensor.detach().cpu().numpy()
        generator_state[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return {
        "generator_state": generator_state,
        "raw_frame_sha256": hashlib.sha256(model.raw_df.to_csv(index=False).encode("utf-8")).hexdigest(),
        "data_prep_sha256": hashlib.sha256(
            pickle.dumps(model.data_prep, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest(),
        "transformer_sha256": hashlib.sha256(
            pickle.dumps(model.synthesizer.transformer, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest(),
        "conditional_generator_sha256": hashlib.sha256(
            pickle.dumps(model.synthesizer.cond_generator, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest(),
        "configuration": {
            "test_ratio": model.test_ratio,
            "categorical_columns": model.categorical_columns,
            "log_columns": model.log_columns,
            "mixed_columns": model.mixed_columns,
            "integer_columns": model.integer_columns,
            "problem_type": model.problem_type,
            "epochs": model.synthesizer.epochs,
            "batch_size": model.synthesizer.batch_size,
            "random_dim": model.synthesizer.random_dim,
            "num_channels": model.synthesizer.num_channels,
            "class_dim": list(model.synthesizer.class_dim),
            "l2scale": model.synthesizer.l2scale,
            "device": str(model.synthesizer.device),
        },
    }


@contextlib.contextmanager
def _official_source_runtime(source_dir: Path):
    warning_filters = list(warnings.filters)
    warning_default = warnings.defaultaction
    try:
        with isolated_module_tree(source_dir, "model"):
            CTABGAN = importlib.import_module("model.ctabgan").CTABGAN
            transformer_module = importlib.import_module("model.synthesizer.transformer")
            sklearn_mixture = importlib.import_module("sklearn.mixture")
            torch = importlib.import_module("torch")
            previous_bgm = transformer_module.BayesianGaussianMixture

            def keyword_only_bayesian_gaussian_mixture(n_components: int = 1, *args: Any, **kwargs: Any):
                return sklearn_mixture.BayesianGaussianMixture(n_components=n_components, *args, **kwargs)

            transformer_module.BayesianGaussianMixture = keyword_only_bayesian_gaussian_mixture
            try:
                yield CTABGAN, torch
            finally:
                transformer_module.BayesianGaussianMixture = previous_bgm
    finally:
        warnings.filters[:] = warning_filters
        warnings.defaultaction = warning_default


def _checkpoint_signature(source_dir: Path, checkpoint_path: Path) -> dict[str, Any]:
    with _official_source_runtime(source_dir):
        with checkpoint_path.open("rb") as handle:
            model = pickle.load(handle)
        return _model_signature(model)


def _run_native(
    source_dir: Path,
    frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ctabgan.pkl"
    sample_path = output_dir / "samples.csv"
    before = np.random.get_state()
    warning_filters_before = list(warnings.filters)
    warning_default_before = warnings.defaultaction
    with _official_source_runtime(source_dir) as (CTABGAN, torch):
        with tempfile.TemporaryDirectory(prefix="input-", dir=output_dir) as temporary:
            input_path = Path(temporary) / "train.csv"
            atomic_write_bytes(input_path, frame.to_csv(index=False).encode("utf-8"))
            with _seeded_runtime(seed, torch):
                model = CTABGAN(raw_csv_path=str(input_path), **_official_parameters(dataset_spec))
                _configure_synthesizer(model, torch)
                model.fit()
                with checkpoint_path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with _seeded_runtime(seed, torch):
                encoded = model.synthesizer.sample(SAMPLE_ROWS)
                samples = model.data_prep.inverse_prep(encoded)[dataset_spec.column_names].copy()
    after = np.random.get_state()
    atomic_write_bytes(sample_path, samples.to_csv(index=False).encode("utf-8"))
    return pd.read_csv(sample_path), {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_signature": _checkpoint_signature(source_dir, checkpoint_path),
        "sample_path": str(sample_path),
        "sample_sha256": _sha256_file(sample_path),
        "global_numpy_state_unchanged": _numpy_state_equal(before, after),
        "global_warning_state_unchanged": (
            warning_filters_before == list(warnings.filters)
            and warning_default_before == warnings.defaultaction
        ),
    }


def _adapter_extra(source_dir: Path, dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "source_dir": str(source_dir),
        "test_ratio": 0.2,
        "integer_columns": ["count"],
        **TRAINING_PARAMETERS,
        "dataset_spec": dataset_spec.to_dict(),
    }


def _run_adapter(
    repo_root: Path,
    source_dir: Path,
    dataset_spec: DatasetSpec,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = CTABGANAdapter(repo_root)
    extra = _adapter_extra(source_dir, dataset_spec)
    common = {
        "model": MODEL_ID,
        "dataset": dataset_spec.name,
        "output_dir": output_dir,
        "device": "cpu",
        "seed": seed,
        "extra": extra,
    }
    before = np.random.get_state()
    warning_filters_before = list(warnings.filters)
    warning_default_before = warnings.defaultaction
    train_bundle = adapter.train(RunSpec(**common))
    train_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    checkpoint_path = output_dir / "ctabgan.pkl"
    checkpoint_metadata = json.loads(
        (output_dir / "ctabgan.pkl.metadata.json").read_text(encoding="utf-8")
    )
    sample_bundle = adapter.sample(RunSpec(**common, num_samples=SAMPLE_ROWS))
    after = np.random.get_state()
    sample_manifest = json.loads((output_dir / "artifacts.json").read_text(encoding="utf-8"))
    sample_metadata = json.loads((output_dir / "ctabgan_sample_metadata.json").read_text(encoding="utf-8"))
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("CTAB-GAN adapter did not declare a generated sample path")
    manifests_valid = (
        train_bundle.model == MODEL_ID
        and train_manifest["model"] == MODEL_ID
        and train_manifest["dataset"] == dataset_spec.name
        and sample_manifest["generated_sample_path"] == str(sample_bundle.generated_sample_path)
        and Path(sample_manifest["upstream_workdir"]).resolve() == source_dir.resolve()
    )
    return pd.read_csv(sample_bundle.generated_sample_path), {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_signature": _checkpoint_signature(source_dir, checkpoint_path),
        "sample_path": str(sample_bundle.generated_sample_path),
        "sample_sha256": _sha256_file(sample_bundle.generated_sample_path),
        "checkpoint_metadata": checkpoint_metadata,
        "sample_metadata": sample_metadata,
        "manifests_valid": manifests_valid,
        "global_numpy_state_unchanged": _numpy_state_equal(before, after),
        "global_warning_state_unchanged": (
            warning_filters_before == list(warnings.filters)
            and warning_default_before == warnings.defaultaction
        ),
    }


def _metadata_valid(
    artifacts: dict[str, Any], dataset_spec: DatasetSpec, source: dict[str, Any], seed: int
) -> bool:
    checkpoint = artifacts["checkpoint_metadata"]
    sample = artifacts["sample_metadata"]
    return bool(
        checkpoint["model"] == MODEL_ID
        and checkpoint["dataset"] == dataset_spec.name
        and checkpoint["seed"] == seed
        and checkpoint["source_rows"] == SOURCE_ROWS
        and checkpoint["columns"] == dataset_spec.column_names
        and checkpoint["official_parameters"] == _official_parameters(dataset_spec)
        and checkpoint["training_parameters"] == TRAINING_PARAMETERS
        and checkpoint["compatibility_shims"] == [CTABGANAdapter.compatibility_shim_id]
        and checkpoint["source"]["upstream_commit"] == source["upstream_commit"] == UPSTREAM_COMMIT
        and checkpoint["source"]["manifest_sha256"] == source["manifest_sha256"]
        and checkpoint["checkpoint_sha256"] == artifacts["checkpoint_sha256"]
        and sample["requested_rows"] == SAMPLE_ROWS
        and sample["seed"] == seed
        and sample["checkpoint_sha256"] == artifacts["checkpoint_sha256"]
        and sample["sample_sha256"] == artifacts["sample_sha256"]
        and sample["compatibility_shims"] == [CTABGANAdapter.compatibility_shim_id]
    )


def _compare_samples(
    native: pd.DataFrame,
    adapter: pd.DataFrame,
    source_frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    categorical = [*dataset_spec.categorical_columns, *dataset_spec.target_columns]
    numerical = dataset_spec.numerical_columns
    finite = bool(
        np.isfinite(native[numerical].to_numpy(dtype=float)).all()
        and np.isfinite(adapter[numerical].to_numpy(dtype=float)).all()
    )
    domains_valid = all(
        set(native[column]).issubset(set(source_frame[column]))
        and set(adapter[column]).issubset(set(source_frame[column]))
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
        and comparisons["adapter_metadata_valid"]
        and comparisons["checkpoint_state_exact"]
        and comparisons["sample_bytes_exact"]
        and comparisons["native_global_numpy_state_unchanged"]
        and comparisons["adapter_global_numpy_state_unchanged"]
        and comparisons["native_global_warning_state_unchanged"]
        and comparisons["adapter_global_warning_state_unchanged"]
        and samples["rows"] == SAMPLE_ROWS
        and samples["columns_exact"]
        and samples["frame_exact"]
        and samples["finite_numerical"]
        and samples["categorical_domains_valid"]
        and samples["missing_values"] == 0
    )


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(repo_root: Path, source_dir: Path, output_dir: Path, evidence_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = _verify_environment()
    source = validate_upstream_source(MODEL_ID, source_dir)
    if source["upstream_commit"] != UPSTREAM_COMMIT or source["upstream_tree"] != UPSTREAM_TREE:
        raise RuntimeError("Validated CTAB-GAN source identity differs from the frozen protocol")

    fixtures = [_write_fixture(output_dir / "fixtures" / variant, variant) for variant in VARIANTS]
    cases: list[dict[str, Any]] = []
    for frame, dataset_spec, fixture_record in fixtures:
        for seed in SEED_CASES:
            case_root = output_dir / fixture_record["variant"] / f"seed-{seed}"
            native_samples, native_artifacts = _run_native(
                source_dir, frame, dataset_spec, case_root / "native", seed
            )
            adapter_samples, adapter_artifacts = _run_adapter(
                repo_root, source_dir, dataset_spec, case_root / "adapter", seed
            )
            comparisons = {
                "adapter_manifests_valid": adapter_artifacts["manifests_valid"],
                "adapter_metadata_valid": _metadata_valid(adapter_artifacts, dataset_spec, source, seed),
                "checkpoint_state_exact": (
                    native_artifacts["checkpoint_signature"] == adapter_artifacts["checkpoint_signature"]
                ),
                "sample_bytes_exact": native_artifacts["sample_sha256"] == adapter_artifacts["sample_sha256"],
                "native_global_numpy_state_unchanged": native_artifacts["global_numpy_state_unchanged"],
                "adapter_global_numpy_state_unchanged": adapter_artifacts["global_numpy_state_unchanged"],
                "native_global_warning_state_unchanged": native_artifacts["global_warning_state_unchanged"],
                "adapter_global_warning_state_unchanged": adapter_artifacts["global_warning_state_unchanged"],
                "samples": _compare_samples(native_samples, adapter_samples, frame, dataset_spec),
            }
            cases.append(
                {
                    "variant": fixture_record["variant"],
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
        "model_id": MODEL_ID,
        "status": "pass" if passed else "fail",
        "claim_boundary": (
            "Native adapter parity for the Apache-2.0 method-author source and the documented keyword-only API "
            "compatibility bridge. Benchmark eligibility, Official Results, and release support remain separate "
            "pending gates."
        ),
        "compatibility_shims": [
            {
                "id": CTABGANAdapter.compatibility_shim_id,
                "scope": "Maps the legacy positional n_components argument to the same keyword-only "
                "BayesianGaussianMixture parameter required by scikit-learn 1.5.2.",
                "semantic_effect": "No parameter value, estimator, source file, or training operation is changed.",
            }
        ],
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "source_manifest": {
            "path": str(source_manifest_path(MODEL_ID).relative_to(repo_root)),
            "sha256": _sha256_file(source_manifest_path(MODEL_ID)),
        },
        "environment_lock": {
            "path": "requirements-ctabgan-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-ctabgan-validation.txt"),
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), **versions},
        "fixtures": [record for _, _, record in fixtures],
        "runtime_config": {"training": TRAINING_PARAMETERS, "sample_rows": SAMPLE_ROWS},
        "variants": list(VARIANTS),
        "seed_cases": list(SEED_CASES),
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("CTAB-GAN native-parity protocol failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CTAB-GAN native-parity validation protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.source_dir, args.output_dir, args.evidence_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": PROTOCOL_ID,
                        "model_id": MODEL_ID,
                        "status": "fail",
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
