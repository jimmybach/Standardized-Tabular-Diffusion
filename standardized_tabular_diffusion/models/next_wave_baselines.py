from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import pickle
import random
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import (
    SampleFileEvaluatorMixin,
    disable_torchvision_for_transformers,
    isolated_module_tree,
)
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import (
    default_source_path,
    validate_upstream_source,
)


def _import_or_raise(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{module_name} is required for this adapter. Install it with `{install_hint}`.") from exc


class CTABGANPlusAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "ctab-gan-plus"
    upstream_dirname = "."
    checkpoint_filename = "ctabgan_plus.pkl"
    source_environment_variable = "STANDARDIZED_TABULAR_DIFFUSION_CTABGAN_PLUS_SOURCE"
    upstream_commit = "6a6f90188cca3dac2c533fd5e8e7f20de074365b"
    expected_versions = {
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "scikit-learn": "1.5.2",
        "scipy": "1.13.1",
        "six": "1.17.0",
        "torch": "2.3.0",
        "tqdm": "4.66.5",
    }

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _checkpoint_metadata_path(self, checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")

    def _resolve_source_root(self, spec: RunSpec) -> tuple[Path, dict[str, Any]]:
        configured = spec.extra.get("source_dir") or os.environ.get(self.source_environment_variable)
        source_root = Path(configured) if configured is not None else default_source_path(self.repo_root, self.model_name)
        try:
            source = validate_upstream_source(self.model_name, source_root)
        except (FileNotFoundError, RuntimeError) as exc:
            command = (
                "python -m standardized_tabular_diffusion.cli materialize-model-source "
                "--model ctab-gan-plus"
            )
            raise RuntimeError(
                f"CTAB-GAN+ requires the checksum-locked official source at {source_root}. Run `{command}`, "
                f"or set spec.extra['source_dir']; underlying error: {exc}"
            ) from exc
        if source["upstream_commit"] != self.upstream_commit:
            raise RuntimeError("CTAB-GAN+ source validation returned an unexpected upstream commit")
        return source_root.resolve(), source

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("CTAB-GAN+ requires dataset_spec.train_data_path")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("CTAB-GAN+ requires exactly one target column")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "CTAB-GAN+ does not accept missing values in the standardized adapter. "
                f"Run the explicit training-split-fitted preprocessing module first; observed: {observed}"
            )
        return frame

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        parsed = int(value)
        if isinstance(value, bool) or parsed < 1:
            raise ValueError(f"CTAB-GAN+ {name} must be a positive integer; observed {value!r}")
        return parsed

    @staticmethod
    def _column_list(name: str, value: Any, columns: list[str]) -> list[str]:
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise TypeError(f"CTAB-GAN+ {name} must be a list of column names")
        result = list(value)
        if len(result) != len(set(result)):
            raise ValueError(f"CTAB-GAN+ {name} contains duplicate columns")
        unknown = sorted(set(result) - set(columns))
        if unknown:
            raise ValueError(f"CTAB-GAN+ {name} contains unknown columns: {unknown}")
        return result

    def _official_parameters(
        self,
        frame: pd.DataFrame,
        dataset_spec: DatasetSpec,
        spec: RunSpec,
    ) -> dict[str, Any]:
        dataset_defaults = dataset_spec.extra.get("ctab_gan_plus", {})
        if not isinstance(dataset_defaults, dict):
            raise TypeError("dataset_spec.extra['ctab_gan_plus'] must be a mapping")

        def setting(name: str, default: Any) -> Any:
            return spec.extra.get(name, dataset_defaults.get(name, default))

        categorical_default = [*dataset_spec.categorical_columns]
        if dataset_spec.task_type == "classification":
            categorical_default.extend(dataset_spec.target_columns)
        integer_default = [
            column
            for column in dataset_spec.numerical_columns
            if column in frame and pd.api.types.is_integer_dtype(frame[column].dtype)
        ]
        categorical = self._column_list(
            "categorical_columns", setting("categorical_columns", categorical_default), dataset_spec.column_names
        )
        log_columns = self._column_list(
            "log_columns", setting("log_columns", []), dataset_spec.column_names
        )
        general_columns = self._column_list(
            "general_columns", setting("general_columns", []), dataset_spec.column_names
        )
        non_categorical = self._column_list(
            "non_categorical_columns", setting("non_categorical_columns", []), dataset_spec.column_names
        )
        integer_columns = self._column_list(
            "integer_columns", setting("integer_columns", integer_default), dataset_spec.column_names
        )
        mixed = setting("mixed_columns", {})
        if not isinstance(mixed, dict) or any(not isinstance(key, str) for key in mixed):
            raise TypeError("CTAB-GAN+ mixed_columns must be a mapping from column names to modal values")
        unknown_mixed = sorted(set(mixed) - set(dataset_spec.column_names))
        if unknown_mixed:
            raise ValueError(f"CTAB-GAN+ mixed_columns contains unknown columns: {unknown_mixed}")
        mixed_columns: dict[str, list[Any]] = {}
        for column, modes in mixed.items():
            if not isinstance(modes, (list, tuple)) or not modes:
                raise ValueError(f"CTAB-GAN+ mixed column {column!r} requires at least one modal value")
            mixed_columns[column] = list(modes)

        test_ratio = float(setting("test_ratio", 0.2))
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("CTAB-GAN+ test_ratio must lie strictly between 0 and 1 for supervised tasks")
        task_name = {"classification": "Classification", "regression": "Regression"}.get(dataset_spec.task_type)
        if task_name is None:
            raise ValueError("CTAB-GAN+ supports only classification and regression dataset specifications")
        return {
            "test_ratio": test_ratio,
            "categorical_columns": categorical,
            "log_columns": log_columns,
            "mixed_columns": mixed_columns,
            "general_columns": general_columns,
            "non_categorical_columns": non_categorical,
            "integer_columns": integer_columns,
            "problem_type": {task_name: dataset_spec.target_columns[0]},
        }

    def _training_parameters(self, spec: RunSpec) -> dict[str, Any]:
        class_dim = spec.extra.get("class_dim", [256, 256, 256, 256])
        if not isinstance(class_dim, (list, tuple)) or not class_dim:
            raise TypeError("CTAB-GAN+ class_dim must be a non-empty sequence")
        parsed_class_dim = [self._positive_int("class_dim entry", item) for item in class_dim]
        l2scale = float(spec.extra.get("l2scale", 1e-5))
        if not np.isfinite(l2scale) or l2scale < 0:
            raise ValueError("CTAB-GAN+ l2scale must be finite and non-negative")
        return {
            "epochs": self._positive_int("epochs", spec.extra.get("epochs", 150)),
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
                    "CTAB-GAN+ requires the frozen optional runtime. Install the project with the "
                    "'ctab-gan-plus' extra on Linux/Python 3.11."
                ) from exc
            if installed != expected and not (package == "torch" and installed.startswith(f"{expected}+")):
                raise ImportError(
                    f"CTAB-GAN+ requires validated {package}=={expected}; observed {installed}."
                )
            observed[package] = installed
        return observed

    @contextlib.contextmanager
    def _official_runtime(self, source_root: Path):
        versions = self._verify_runtime_versions()
        with isolated_module_tree(source_root, "model"):
            ctab_module = _import_or_raise(
                "model.ctabgan", "pip install 'standardized-tabular-diffusion[ctab-gan-plus]'"
            )
            torch = _import_or_raise("torch", "pip install 'standardized-tabular-diffusion[ctab-gan-plus]'")
            yield ctab_module.CTABGAN, torch, versions

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
        dataset_spec: DatasetSpec,
        spec: RunSpec,
        CTABGAN: Any,
        torch: Any,
        official_parameters: dict[str, Any],
        training_parameters: dict[str, Any],
    ) -> Any:
        with tempfile.TemporaryDirectory(prefix=".ctabgan-plus-input-", dir=spec.output_dir) as temporary:
            input_path = Path(temporary) / "train.csv"
            self._write_dataframe_csv(train_df, input_path)
            model = CTABGAN(raw_csv_path=str(input_path), **official_parameters)
        synthesizer = model.synthesizer
        for name in ("epochs", "batch_size", "random_dim", "num_channels", "class_dim", "l2scale"):
            value = training_parameters[name]
            setattr(synthesizer, name, tuple(value) if name == "class_dim" else value)
        synthesizer.device = torch.device(spec.device)
        if synthesizer.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CTAB-GAN+ requested unavailable device: {spec.device}")
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
                    dataset_spec,
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
                f"Serialized official CTAB-GAN+ checkpoint written to {checkpoint_path}.",
                f"Official source commit: {self.upstream_commit}.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        source_root, source = self._resolve_source_root(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        metadata_path = self._checkpoint_metadata_path(trusted_checkpoint)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing trusted CTAB-GAN+ checkpoint metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        observed_checkpoint_sha256 = hashlib.sha256(trusted_checkpoint.read_bytes()).hexdigest()
        if metadata.get("checkpoint_sha256") != observed_checkpoint_sha256:
            raise RuntimeError("CTAB-GAN+ checkpoint checksum differs from its training metadata")
        if metadata.get("source", {}).get("upstream_commit") != self.upstream_commit:
            raise RuntimeError("CTAB-GAN+ checkpoint was not produced from the locked official source")
        if metadata.get("source", {}).get("manifest_sha256") != source["manifest_sha256"]:
            raise RuntimeError("CTAB-GAN+ checkpoint source manifest differs from the active source")
        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        if isinstance(num_samples, bool) or int(num_samples) < 1:
            raise ValueError("CTAB-GAN+ num_samples must be a positive integer")
        num_threads = self._positive_int("num_threads", spec.extra.get("num_threads", 1))
        with self._official_runtime(source_root) as (CTABGAN, torch, versions):
            with self._seeded_runtime(spec.seed, torch, num_threads):
                with trusted_checkpoint.open("rb") as handle:
                    model = pickle.load(handle)
                if model.__class__ is not CTABGAN:
                    raise RuntimeError("CTAB-GAN+ checkpoint class does not match the locked official source")
                encoded = model.synthesizer.sample(int(num_samples))
                sample_df = model.data_prep.inverse_prep(encoded)
        sample_df = sample_df[dataset_spec.column_names].copy()
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        atomic_write_json(
            spec.output_dir / "ctabgan_plus_sample_metadata.json",
            {
                "model": self.model_name,
                "dataset": dataset_spec.name,
                "seed": spec.seed,
                "requested_rows": int(num_samples),
                "source": source,
                "runtime_versions": versions,
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


class REaLTabFormerAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "realtabformer"
    upstream_dirname = "TabSyn-main"

    def _model_root(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / "realtabformer_model"

    def _resolve_saved_model_dir(self, model_root: Path) -> Path:
        if model_root.is_dir() and model_root.name.startswith("id"):
            return model_root
        candidates = sorted([path for path in model_root.glob("id*") if path.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"Could not find saved REaLTabFormer model directory under {model_root}")
        return candidates[-1]

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("realtabformer requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def _limit_training_frame(self, train_df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        max_train_rows = spec.extra.get("max_train_rows")
        if max_train_rows is None or len(train_df) <= int(max_train_rows):
            return train_df
        return train_df.sample(n=int(max_train_rows), random_state=spec.seed).reset_index(drop=True)

    def _import_model_cls(self):
        with disable_torchvision_for_transformers():
            module = _import_or_raise("realtabformer", "pip install realtabformer")
            return module.REaLTabFormer

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._limit_training_frame(self._load_training_frame(dataset_spec), spec)
        REaLTabFormer = self._import_model_cls()
        model = REaLTabFormer(
            model_type="tabular",
            epochs=int(spec.extra.get("epochs", 100)),
            batch_size=int(spec.extra.get("batch_size", 64)),
            gradient_accumulation_steps=int(spec.extra.get("gradient_accumulation_steps", 4)),
            logging_steps=int(spec.extra.get("logging_steps", 100)),
            report_to=spec.extra.get("report_to", "none"),
        )
        fit_kwargs: dict[str, Any] = {
            "num_bootstrap": int(spec.extra.get("num_bootstrap", 0)),
            "n_critic": int(spec.extra.get("n_critic", 0)),
        }
        model.fit(train_df, **fit_kwargs)
        model_root = self._model_root(spec)
        model_root.mkdir(parents=True, exist_ok=True)
        full_save_dir = getattr(model, "full_save_dir", None)
        if full_save_dir is not None:
            setattr(model, "full_save_dir", str(full_save_dir))
        model.save(str(model_root))
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Saved REaLTabFormer artifacts under {model_root}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        REaLTabFormer = self._import_model_cls()
        model_dir = self._resolve_saved_model_dir(self._model_root(spec))
        trusted_model_dir = self._validate_trusted_executable_artifact(
            spec,
            model_dir,
            format_name="REaLTabFormer model directory",
            allow_directory=True,
        )
        model = REaLTabFormer.load_from_dir(path=str(trusted_model_dir))
        num_samples = spec.num_samples or len(train_df)
        sample_kwargs: dict[str, Any] = {}
        if spec.extra.get("gen_batch") is not None:
            sample_kwargs["gen_batch"] = int(spec.extra["gen_batch"])
        sample_df = model.sample(n_samples=num_samples, **sample_kwargs)
        sample_df = sample_df[dataset_spec.column_names].copy()
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


class NRGBoostAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "nrgboost"
    upstream_dirname = "."
    checkpoint_filename = "model.nrgboost"
    package_name = "nrgboost"
    package_version = "0.0.3"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("NRGBoost requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        missing_counts = frame.isna().sum()
        if bool(missing_counts.any()):
            observed = {str(column): int(count) for column, count in missing_counts.items() if count}
            raise ValueError(
                "NRGBoost does not accept missing values in this benchmark. "
                f"Run the explicit preprocessing module first; observed: {observed}"
            )
        for column in dataset_spec.categorical_columns + (
            dataset_spec.target_columns if dataset_spec.task_type == "classification" else []
        ):
            if column in frame.columns:
                frame[column] = frame[column].astype("category")
        return frame

    def _import_bits(self):
        try:
            observed_version = version(self.package_name)
        except PackageNotFoundError as exc:
            raise ImportError(
                "NRGBoost requires the optional official nrgboost package. "
                "Install the project with the 'nrgboost' extra."
            ) from exc
        if observed_version != self.package_version:
            raise ImportError(
                "NRGBoost requires the exact validated official package version "
                f"{self.package_name}=={self.package_version}; observed {observed_version}."
            )
        module = importlib.import_module("nrgboost")
        return module.Dataset, module.NRGBooster

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        parsed = int(value)
        if isinstance(value, bool) or parsed < 1:
            raise ValueError(f"NRGBoost {name} must be a positive integer; observed {value!r}.")
        return parsed

    @staticmethod
    def _nonnegative_int(name: str, value: Any) -> int:
        parsed = int(value)
        if isinstance(value, bool) or parsed < 0:
            raise ValueError(f"NRGBoost {name} must be a non-negative integer; observed {value!r}.")
        return parsed

    @staticmethod
    def _finite_float(name: str, value: Any) -> float:
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"NRGBoost {name} must be finite; observed {value!r}.")
        return parsed

    @staticmethod
    def _boolean(name: str, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"NRGBoost {name} must be a boolean; observed {value!r}.")
        return value

    def _dataset_params(self, spec: RunSpec) -> dict[str, Any]:
        num_bins = self._positive_int("num_bins", spec.extra.get("num_bins", 255))
        if num_bins > 255:
            raise ValueError(
                f"NRGBoost num_bins cannot exceed the uint8 domain limit of 255; observed {num_bins}."
            )
        discretization_types = spec.extra.get("discretization_types")
        if discretization_types is not None and not isinstance(discretization_types, dict):
            raise TypeError("NRGBoost discretization_types must be a mapping when explicitly provided.")
        return {
            "num_bins": num_bins,
            "infer_fixed_point": self._boolean("infer_fixed_point", spec.extra.get("infer_fixed_point", True)),
            "discretization_types": discretization_types,
            "infer_ordered_categoricals": self._boolean(
                "infer_ordered_categoricals", spec.extra.get("infer_ordered_categoricals", False)
            ),
            "infer_continuous_ordered_categoricals": self._boolean(
                "infer_continuous_ordered_categoricals",
                spec.extra.get("infer_continuous_ordered_categoricals", False)
            ),
        }

    def _training_params(self, spec: RunSpec) -> dict[str, Any]:
        params = {
            "num_trees": self._nonnegative_int("num_trees", spec.extra.get("num_trees", 200)),
            "shrinkage": self._finite_float("shrinkage", spec.extra.get("shrinkage", 0.15)),
            "line_search": self._boolean("line_search", spec.extra.get("line_search", True)),
            "max_leaves": self._positive_int("max_leaves", spec.extra.get("max_leaves", 256)),
            "max_ratio_in_leaf": self._finite_float(
                "max_ratio_in_leaf", spec.extra.get("max_ratio_in_leaf", 2)
            ),
            "min_data_in_leaf": self._finite_float(
                "min_data_in_leaf", spec.extra.get("min_data_in_leaf", 0)
            ),
            "initial_uniform_mixture": self._finite_float(
                "initial_uniform_mixture", spec.extra.get("initial_uniform_mixture", 0.1)
            ),
            "categorical_split_one_vs_all": self._boolean(
                "categorical_split_one_vs_all", spec.extra.get("categorical_split_one_vs_all", False)
            ),
            "feature_frac": self._finite_float("feature_frac", spec.extra.get("feature_frac", 1)),
            "splitter": str(spec.extra.get("splitter", "best")),
            "num_model_samples": self._positive_int(
                "num_model_samples", spec.extra.get("num_model_samples", 80000)
            ),
            "p_refresh": self._finite_float("p_refresh", spec.extra.get("p_refresh", 0.1)),
            "num_chains": self._positive_int("num_chains", spec.extra.get("num_chains", 16)),
            "burn_in": self._nonnegative_int("burn_in", spec.extra.get("burn_in", 100)),
            "temperature": self._finite_float("temperature", spec.extra.get("training_temperature", 1.0)),
            "initial_samples": str(spec.extra.get("initial_samples", "data")),
            "min_gain": self._finite_float("min_gain", spec.extra.get("min_gain", 0.0)),
            "jit_all": self._boolean("jit_all", spec.extra.get("jit_all", False)),
            "num_threads": self._nonnegative_int("num_threads", spec.extra.get("num_threads", 0)),
        }
        if params["shrinkage"] <= 0 or params["max_ratio_in_leaf"] <= 0 or params["temperature"] <= 0:
            raise ValueError("NRGBoost shrinkage, max_ratio_in_leaf, and training_temperature must be positive.")
        if params["min_data_in_leaf"] < 0:
            raise ValueError("NRGBoost min_data_in_leaf must be non-negative.")
        if not 0 <= params["initial_uniform_mixture"] <= 1:
            raise ValueError("NRGBoost initial_uniform_mixture must lie in [0, 1].")
        if not 0 < params["feature_frac"] <= 1:
            raise ValueError("NRGBoost feature_frac must lie in (0, 1].")
        if not 0 <= params["p_refresh"] <= 1:
            raise ValueError("NRGBoost p_refresh must lie in [0, 1].")
        if params["splitter"] not in {"best", "depth", "random"}:
            raise ValueError("NRGBoost splitter must be one of: best, depth, random.")
        if params["initial_samples"] not in {"data", "uniform", "initial"}:
            raise ValueError("NRGBoost initial_samples must be one of: data, uniform, initial.")
        if params["num_model_samples"] < params["num_chains"]:
            raise ValueError("NRGBoost num_model_samples must be at least num_chains.")
        return params

    def _sampling_params(self, spec: RunSpec) -> dict[str, Any]:
        num_rounds_value = spec.extra.get("num_rounds")
        num_rounds = None if num_rounds_value is None else self._positive_int("num_rounds", num_rounds_value)
        params = {
            "num_steps": self._positive_int("num_steps", spec.extra.get("num_steps", 100)),
            "num_rounds": num_rounds,
            "temperature": self._finite_float("temperature", spec.extra.get("temperature", 1.0)),
            "num_threads": self._nonnegative_int("num_threads", spec.extra.get("num_threads", 0)),
            "output_full_chain": False,
            "seed": spec.seed,
        }
        if params["temperature"] <= 0:
            raise ValueError("NRGBoost sampling temperature must be positive.")
        return params

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        Dataset, NRGBooster = self._import_bits()
        dataset_params = self._dataset_params(spec)
        training_params = self._training_params(spec)
        train_ds = Dataset(train_df, **dataset_params)
        model = NRGBooster.fit(train_ds, dict(training_params), seed=spec.seed)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        model.save(str(checkpoint_path))
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise RuntimeError(f"NRGBoost did not create the expected checkpoint file: {checkpoint_path}")
        atomic_write_json(
            spec.output_dir / "nrgboost_metadata.json",
            {
                "package": self.package_name,
                "package_version": self.package_version,
                "seed": spec.seed,
                "source_rows": len(train_df),
                "columns": dataset_spec.column_names,
                "categorical_columns": [
                    *dataset_spec.categorical_columns,
                    *(
                        dataset_spec.target_columns
                        if dataset_spec.task_type == "classification"
                        else []
                    ),
                ],
                "dataset_params": dataset_params,
                "training_params": training_params,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Saved NRGBoost model to {checkpoint_path}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        _, NRGBooster = self._import_bits()
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")
        trusted_checkpoint = self._validate_trusted_executable_artifact(
            spec,
            checkpoint_path,
            format_name="NRGBoost joblib model",
        )
        model = NRGBooster.load(str(trusted_checkpoint))
        num_samples = len(train_df) if spec.num_samples is None else spec.num_samples
        if num_samples < 1:
            raise ValueError(f"NRGBoost num_samples must be positive; observed {num_samples}.")
        sampling_params = self._sampling_params(spec)
        sample_df = model.sample(num_samples, **sampling_params)
        if not isinstance(sample_df, pd.DataFrame):
            sample_df = pd.DataFrame(sample_df, columns=dataset_spec.column_names)
        missing_columns = [column for column in dataset_spec.column_names if column not in sample_df.columns]
        if missing_columns:
            raise ValueError(f"NRGBoost sample output is missing canonical columns: {missing_columns}")
        sample_df = sample_df[dataset_spec.column_names].copy()
        if len(sample_df) != num_samples:
            raise ValueError(
                f"NRGBoost returned {len(sample_df)} rows for a request of {num_samples}."
            )
        if bool(sample_df.isna().any().any()):
            raise ValueError("NRGBoost produced missing values; refusing to write an invalid benchmark sample.")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        metadata_path = spec.output_dir / "nrgboost_metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        metadata["sampling"] = {
            "requested_rows": num_samples,
            **sampling_params,
        }
        metadata["sample_path"] = str(sample_path)
        atomic_write_json(metadata_path, metadata)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
