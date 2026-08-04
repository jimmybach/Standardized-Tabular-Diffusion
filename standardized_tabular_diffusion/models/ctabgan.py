from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import pickle
import random
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin, isolated_module_tree
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import validate_upstream_source


def _import_or_raise(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{module_name} is required for this adapter. Install it with `{install_hint}`.") from exc


class CTABGANAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    """Strict adapter around the checksum-locked method-author CTAB-GAN source."""

    model_name = "ctab-gan"
    upstream_dirname = "TabDDPM-main/CTAB-GAN"
    checkpoint_filename = "ctabgan.pkl"
    source_environment_variable = "STANDARDIZED_TABULAR_DIFFUSION_CTABGAN_SOURCE"
    upstream_commit = "73d4e315a2a51cf16c97ed8a00d2dad456cfce8a"
    compatibility_shim_id = "ctabgan-sklearn-keyword-only-v1"
    expected_versions = {
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "scikit-learn": "1.5.2",
        "scipy": "1.13.1",
        "torch": "2.3.0",
        "tqdm": "4.66.5",
    }

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    @staticmethod
    def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")

    def _resolve_source_root(self, spec: RunSpec) -> tuple[Path, dict[str, Any]]:
        configured = spec.extra.get("source_dir") or os.environ.get(self.source_environment_variable)
        source_root = Path(configured) if configured is not None else self.upstream_root
        try:
            source = validate_upstream_source(self.model_name, source_root)
        except (FileNotFoundError, RuntimeError) as exc:
            command = "python -m standardized_tabular_diffusion.cli materialize-model-source --model ctab-gan"
            raise RuntimeError(
                f"CTAB-GAN requires the checksum-locked official source at {source_root}. Restore the complete "
                f"repository checkout, run `{command}` and pass its destination as spec.extra['source_dir'], or set "
                f"{self.source_environment_variable}; underlying error: {exc}"
            ) from exc
        if source["upstream_commit"] != self.upstream_commit:
            raise RuntimeError("CTAB-GAN source validation returned an unexpected upstream commit")
        return source_root.resolve(), source

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("CTAB-GAN requires dataset_spec.train_data_path")
        if dataset_spec.task_type != "classification":
            raise ValueError(
                "The locked official CTAB-GAN implementation supports the standardized classification path only; "
                "its supervised preprocessing always stratifies the target."
            )
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("CTAB-GAN requires exactly one target column")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "CTAB-GAN does not accept missing values in the standardized adapter. Run the explicit "
                f"training-split-fitted preprocessing module first; observed: {observed}"
            )
        return frame

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        parsed = int(value)
        if isinstance(value, bool) or parsed < 1:
            raise ValueError(f"CTAB-GAN {name} must be a positive integer; observed {value!r}")
        return parsed

    @staticmethod
    def _column_list(name: str, value: Any, columns: list[str]) -> list[str]:
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise TypeError(f"CTAB-GAN {name} must be a list of column names")
        result = list(value)
        if len(result) != len(set(result)):
            raise ValueError(f"CTAB-GAN {name} contains duplicate columns")
        unknown = sorted(set(result) - set(columns))
        if unknown:
            raise ValueError(f"CTAB-GAN {name} contains unknown columns: {unknown}")
        return result

    def _official_parameters(
        self,
        frame: pd.DataFrame,
        dataset_spec: DatasetSpec,
        spec: RunSpec,
    ) -> dict[str, Any]:
        dataset_defaults = dataset_spec.extra.get("ctab_gan", {})
        if not isinstance(dataset_defaults, dict):
            raise TypeError("dataset_spec.extra['ctab_gan'] must be a mapping")

        def setting(name: str, default: Any) -> Any:
            return spec.extra.get(name, dataset_defaults.get(name, default))

        target = dataset_spec.target_columns[0]
        categorical_default = [*dataset_spec.categorical_columns, target]
        integer_default = [
            column
            for column in dataset_spec.numerical_columns
            if column in frame and pd.api.types.is_integer_dtype(frame[column].dtype)
        ]
        categorical = self._column_list(
            "categorical_columns", setting("categorical_columns", categorical_default), dataset_spec.column_names
        )
        if target not in categorical:
            raise ValueError("CTAB-GAN classification requires the target in categorical_columns")
        log_columns = self._column_list(
            "log_columns", setting("log_columns", []), dataset_spec.column_names
        )
        integer_columns = self._column_list(
            "integer_columns", setting("integer_columns", integer_default), dataset_spec.column_names
        )
        mixed = setting("mixed_columns", {})
        if not isinstance(mixed, dict) or any(not isinstance(key, str) for key in mixed):
            raise TypeError("CTAB-GAN mixed_columns must be a mapping from column names to modal values")
        unknown_mixed = sorted(set(mixed) - set(dataset_spec.column_names))
        if unknown_mixed:
            raise ValueError(f"CTAB-GAN mixed_columns contains unknown columns: {unknown_mixed}")
        mixed_columns: dict[str, list[Any]] = {}
        for column, modes in mixed.items():
            if not isinstance(modes, (list, tuple)) or not modes:
                raise ValueError(f"CTAB-GAN mixed column {column!r} requires at least one modal value")
            mixed_columns[column] = list(modes)
        incompatible = {
            "categorical/log": sorted(set(categorical) & set(log_columns)),
            "categorical/mixed": sorted(set(categorical) & set(mixed_columns)),
            "categorical/integer": sorted(set(categorical) & set(integer_columns)),
        }
        incompatible = {name: values for name, values in incompatible.items() if values}
        if incompatible:
            raise ValueError(f"CTAB-GAN column roles overlap incompatibly: {incompatible}")
        test_ratio = float(setting("test_ratio", 0.2))
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("CTAB-GAN test_ratio must lie strictly between 0 and 1")
        class_counts = frame[target].value_counts(dropna=False)
        class_count = len(class_counts)
        test_rows = int(np.ceil(len(frame) * test_ratio))
        train_rows = len(frame) - test_rows
        if int(class_counts.min()) < 2 or min(train_rows, test_rows) < class_count:
            raise ValueError(
                "CTAB-GAN's official stratified split requires at least two rows per target class and enough "
                f"train/test rows to contain every class; counts={class_counts.to_dict()}, "
                f"train_rows={train_rows}, test_rows={test_rows}"
            )
        return {
            "test_ratio": test_ratio,
            "categorical_columns": categorical,
            "log_columns": log_columns,
            "mixed_columns": mixed_columns,
            "integer_columns": integer_columns,
            "problem_type": {"Classification": target},
        }

    def _training_parameters(self, spec: RunSpec) -> dict[str, Any]:
        class_dim = spec.extra.get("class_dim", [256, 256, 256, 256])
        if not isinstance(class_dim, (list, tuple)) or not class_dim:
            raise TypeError("CTAB-GAN class_dim must be a non-empty sequence")
        parsed_class_dim = [self._positive_int("class_dim entry", item) for item in class_dim]
        l2scale = float(spec.extra.get("l2scale", 1e-5))
        if not np.isfinite(l2scale) or l2scale < 0:
            raise ValueError("CTAB-GAN l2scale must be finite and non-negative")
        return {
            "epochs": self._positive_int("epochs", spec.extra.get("epochs", 1)),
            "batch_size": self._positive_int("batch_size", spec.extra.get("batch_size", 500)),
            "random_dim": self._positive_int("random_dim", spec.extra.get("random_dim", 100)),
            "num_channels": self._positive_int("num_channels", spec.extra.get("num_channels", 64)),
            "class_dim": parsed_class_dim,
            "l2scale": l2scale,
            "num_threads": self._positive_int("num_threads", spec.extra.get("num_threads", 1)),
        }

    def _verify_runtime_versions(self) -> dict[str, str]:
        observed: dict[str, str] = {}
        for package, expected in self.expected_versions.items():
            try:
                installed = version(package)
            except PackageNotFoundError as exc:
                raise ImportError(
                    "CTAB-GAN requires the frozen optional runtime. Install the project with the 'ctab-gan' "
                    "extra on Linux/Python 3.11."
                ) from exc
            if installed != expected and not (package == "torch" and installed.startswith(f"{expected}+")):
                raise ImportError(f"CTAB-GAN requires validated {package}=={expected}; observed {installed}.")
            observed[package] = installed
        return observed

    @contextlib.contextmanager
    def _official_runtime(self, source_root: Path):
        versions = self._verify_runtime_versions()
        warning_filters = list(warnings.filters)
        warning_default = warnings.defaultaction
        try:
            with isolated_module_tree(source_root, "model"):
                ctab_module = _import_or_raise(
                    "model.ctabgan", "pip install 'standardized-tabular-diffusion[ctab-gan]'"
                )
                transformer_module = _import_or_raise(
                    "model.synthesizer.transformer", "pip install 'standardized-tabular-diffusion[ctab-gan]'"
                )
                sklearn_mixture = _import_or_raise(
                    "sklearn.mixture", "pip install 'standardized-tabular-diffusion[ctab-gan]'"
                )
                torch = _import_or_raise("torch", "pip install 'standardized-tabular-diffusion[ctab-gan]'")
                previous_bgm = transformer_module.BayesianGaussianMixture

                def keyword_only_bayesian_gaussian_mixture(n_components: int = 1, *args: Any, **kwargs: Any):
                    return sklearn_mixture.BayesianGaussianMixture(
                        n_components=n_components,
                        *args,
                        **kwargs,
                    )

                transformer_module.BayesianGaussianMixture = keyword_only_bayesian_gaussian_mixture
                try:
                    yield ctab_module.CTABGAN, torch, versions
                finally:
                    transformer_module.BayesianGaussianMixture = previous_bgm
        finally:
            warnings.filters[:] = warning_filters
            warnings.defaultaction = warning_default

    @contextlib.contextmanager
    def _seeded_runtime(self, seed: int, torch: Any, num_threads: int):
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
        torch.set_num_threads(num_threads)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            torch.set_num_threads(previous_threads)

    def _build_model(
        self,
        train_df: pd.DataFrame,
        spec: RunSpec,
        CTABGAN: Any,
        torch: Any,
        official_parameters: dict[str, Any],
        training_parameters: dict[str, Any],
    ) -> Any:
        with tempfile.TemporaryDirectory(prefix=".ctabgan-input-", dir=spec.output_dir) as temporary:
            input_path = Path(temporary) / "train.csv"
            self._write_dataframe_csv(train_df, input_path)
            model = CTABGAN(
                raw_csv_path=str(input_path),
                epochs=training_parameters["epochs"],
                **official_parameters,
            )
        synthesizer = model.synthesizer
        for name in ("batch_size", "random_dim", "num_channels", "class_dim", "l2scale"):
            value = training_parameters[name]
            setattr(synthesizer, name, tuple(value) if name == "class_dim" else value)
        synthesizer.device = torch.device(spec.device)
        if synthesizer.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CTAB-GAN requested unavailable device: {spec.device}")
        return model

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        source_root, source = self._resolve_source_root(spec)
        official_parameters = self._official_parameters(train_df, dataset_spec, spec)
        training_parameters = self._training_parameters(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self._official_runtime(source_root) as (CTABGAN, torch, versions):
            with self._seeded_runtime(spec.seed, torch, training_parameters["num_threads"]):
                model = self._build_model(
                    train_df,
                    spec,
                    CTABGAN,
                    torch,
                    official_parameters,
                    training_parameters,
                )
                model.fit()
                with checkpoint_path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        atomic_write_json(
            self._checkpoint_metadata_path(checkpoint_path),
            {
                "model": self.model_name,
                "dataset": dataset_spec.name,
                "seed": spec.seed,
                "source": source,
                "runtime_versions": versions,
                "compatibility_shims": [self.compatibility_shim_id],
                "source_rows": len(train_df),
                "columns": dataset_spec.column_names,
                "official_parameters": official_parameters,
                "training_parameters": training_parameters,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=source_root,
            notes=[
                f"Serialized official CTAB-GAN checkpoint written to {checkpoint_path}.",
                f"Official source commit: {self.upstream_commit}.",
                "The runtime compatibility bridge maps the legacy positional n_components argument to the same "
                "keyword-only scikit-learn parameter; it does not alter model values or training logic.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        self._load_training_frame(dataset_spec)
        source_root, source = self._resolve_source_root(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        metadata_path = self._checkpoint_metadata_path(trusted_checkpoint)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing trusted CTAB-GAN checkpoint metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        observed_checkpoint_sha256 = hashlib.sha256(trusted_checkpoint.read_bytes()).hexdigest()
        if metadata.get("checkpoint_sha256") != observed_checkpoint_sha256:
            raise RuntimeError("CTAB-GAN checkpoint checksum differs from its training metadata")
        if metadata.get("source", {}).get("upstream_commit") != self.upstream_commit:
            raise RuntimeError("CTAB-GAN checkpoint was not produced from the locked official source")
        if metadata.get("source", {}).get("manifest_sha256") != source["manifest_sha256"]:
            raise RuntimeError("CTAB-GAN checkpoint source manifest differs from the active source")
        if metadata.get("compatibility_shims") != [self.compatibility_shim_id]:
            raise RuntimeError("CTAB-GAN checkpoint compatibility-shim identity is missing or unexpected")
        num_samples = int(metadata["source_rows"]) if spec.num_samples is None else spec.num_samples
        if isinstance(num_samples, bool) or int(num_samples) < 1:
            raise ValueError("CTAB-GAN num_samples must be a positive integer")
        num_threads = self._positive_int("num_threads", spec.extra.get("num_threads", 1))
        with self._official_runtime(source_root) as (CTABGAN, torch, versions):
            with self._seeded_runtime(spec.seed, torch, num_threads):
                with trusted_checkpoint.open("rb") as handle:
                    model = pickle.load(handle)
                if model.__class__ is not CTABGAN:
                    raise RuntimeError("CTAB-GAN checkpoint class does not match the locked official source")
                encoded = model.synthesizer.sample(int(num_samples))
                sample_df = model.data_prep.inverse_prep(encoded)
        sample_df = sample_df[dataset_spec.column_names].copy()
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        atomic_write_json(
            spec.output_dir / "ctabgan_sample_metadata.json",
            {
                "model": self.model_name,
                "dataset": dataset_spec.name,
                "seed": spec.seed,
                "requested_rows": int(num_samples),
                "source": source,
                "runtime_versions": versions,
                "compatibility_shims": [self.compatibility_shim_id],
                "checkpoint_path": str(trusted_checkpoint),
                "checkpoint_sha256": observed_checkpoint_sha256,
                "sample_path": str(sample_path),
                "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=source_root,
            generated_sample_path=sample_path,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
