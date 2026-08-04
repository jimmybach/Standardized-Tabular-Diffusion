from __future__ import annotations

import importlib
import pickle
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.models.final_wave_baselines import _disable_torchvision_for_transformers
from standardized_tabular_diffusion.models.sample_baselines import _SampleFileEvaluatorMixin, _temporary_sys_path


def _import_or_raise(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{module_name} is required for this adapter. Install it with `{install_hint}`.") from exc


class CTABGANPlusAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "ctab-gan-plus"
    upstream_dirname = "TabDDPM-main"
    checkpoint_filename = "ctabgan_plus.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _dataset_dir(self, dataset_spec: DatasetSpec) -> Path:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("ctab-gan-plus requires dataset_spec.train_data_path")
        return dataset_spec.train_data_path.parent

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("ctab-gan-plus requires dataset_spec.train_data_path")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("ctab-gan-plus requires exactly one target column")
        ordered_features = [*dataset_spec.numerical_columns, *dataset_spec.categorical_columns]
        frame = pd.read_csv(dataset_spec.train_data_path)[[*ordered_features, *dataset_spec.target_columns]].copy()
        frame.columns = [*[str(index) for index in range(len(ordered_features))], "y"]
        return frame

    def _load_ctabgan_params(self, dataset_name: str) -> dict[str, Any]:
        columns_path = self.repo_root / "TabDDPM-main" / "CTAB-GAN-Plus" / "columns.json"
        payload = read_json(columns_path)
        if dataset_name not in payload:
            raise KeyError(f"Missing CTAB-GAN-Plus column config for dataset {dataset_name}")
        return dict(payload[dataset_name])

    def _train_kwargs(self, spec: RunSpec) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if spec.extra.get("epochs") is not None:
            kwargs["epochs"] = int(spec.extra["epochs"])
        if spec.extra.get("batch_size") is not None:
            kwargs["batch_size"] = int(spec.extra["batch_size"])
        return kwargs

    def _build_model(self, train_df: pd.DataFrame, spec: RunSpec):
        ctabgan_root = self.repo_root / "TabDDPM-main" / "CTAB-GAN-Plus"
        with _temporary_sys_path(ctabgan_root):
            ctab_module = _import_or_raise("model.ctabgan", "pip install dython")
            CTABGAN = ctab_module.CTABGAN
            params = self._load_ctabgan_params(spec.dataset)
            params.update(self._train_kwargs(spec))
            return CTABGAN(
                df=train_df,
                test_ratio=0.0,
                device=spec.device,
                **params,
            )

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        model = self._build_model(train_df, spec)
        model.fit()
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump(model, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Serialized CTAB-GAN-Plus checkpoint written to {checkpoint_path}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        with trusted_checkpoint.open("rb") as handle:
            model = pickle.load(handle)
        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        sample_df = model.generate_samples(num_samples, seed=spec.seed)
        if dataset_spec.target_columns:
            target_idx = dataset_spec.column_names.index(dataset_spec.target_columns[0])
            sample_df = sample_df.rename(columns={"y": str(target_idx)})
        sample_df = sample_df[[str(idx) for idx in range(len(dataset_spec.column_names))]].copy()
        sample_df.columns = dataset_spec.column_names
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


class REaLTabFormerAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
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
        with _disable_torchvision_for_transformers():
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


class NRGBoostAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
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
