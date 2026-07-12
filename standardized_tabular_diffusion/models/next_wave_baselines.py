from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.models.final_wave_baselines import _disable_torchvision_for_transformers
from standardized_tabular_diffusion.models.sample_baselines import _SampleFileEvaluatorMixin, _temporary_sys_path


def _import_or_raise(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{module_name} is required for this adapter. Install it with `{install_hint}`."
        ) from exc


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
        dataset_dir = self._dataset_dir(dataset_spec)
        y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
        x_num_path = dataset_dir / "X_num_train.npy"
        x_cat_path = dataset_dir / "X_cat_train.npy"
        x_num_train = np.load(x_num_path, allow_pickle=True) if x_num_path.exists() else None
        x_cat_train = np.load(x_cat_path, allow_pickle=True) if x_cat_path.exists() else None

        parts: list[pd.DataFrame] = []
        if x_num_train is not None:
            parts.append(pd.DataFrame(x_num_train, columns=list(range(x_num_train.shape[1]))))
        if x_cat_train is not None:
            offset = 0 if x_num_train is None else x_num_train.shape[1]
            parts.append(pd.DataFrame(x_cat_train, columns=list(range(offset, offset + x_cat_train.shape[1]))))
        parts.append(pd.DataFrame(y_train, columns=["y"]))
        frame = pd.concat(parts, axis=1)
        frame.columns = [str(column) for column in frame.columns]
        return frame

    def _load_ctabgan_params(self, dataset_name: str) -> dict[str, Any]:
        columns_path = self.repo_root / "TabDDPM-main" / "CTAB-GAN-Plus" / "columns.json"
        payload = json.loads(columns_path.read_text())
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
        with checkpoint_path.open("rb") as handle:
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
        sample_df.to_csv(sample_path, index=False)
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
        model = REaLTabFormer.load_from_dir(path=str(model_dir))
        num_samples = spec.num_samples or len(train_df)
        sample_kwargs: dict[str, Any] = {}
        if spec.extra.get("gen_batch") is not None:
            sample_kwargs["gen_batch"] = int(spec.extra["gen_batch"])
        sample_df = model.sample(n_samples=num_samples, **sample_kwargs)
        sample_df = sample_df[dataset_spec.column_names].copy()
        sample_path = spec.output_dir / "samples.csv"
        sample_df.to_csv(sample_path, index=False)
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
    upstream_dirname = "TabSyn-main"
    checkpoint_filename = "model.nrgboost"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("nrgboost requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        for column in dataset_spec.categorical_columns + (dataset_spec.target_columns if dataset_spec.task_type == "classification" else []):
            if column in frame.columns:
                frame[column] = frame[column].astype("category")
        return frame

    def _import_bits(self):
        module = _import_or_raise("nrgboost", "pip install nrgboost")
        return module.Dataset, module.NRGBooster

    def _training_params(self, spec: RunSpec) -> dict[str, Any]:
        return {
            "num_trees": int(spec.extra.get("num_trees", 200)),
            "shrinkage": float(spec.extra.get("shrinkage", 0.15)),
            "max_leaves": int(spec.extra.get("max_leaves", 256)),
            "max_ratio_in_leaf": float(spec.extra.get("max_ratio_in_leaf", 2)),
            "num_model_samples": int(spec.extra.get("num_model_samples", 80000)),
            "p_refresh": float(spec.extra.get("p_refresh", 0.1)),
            "num_chains": int(spec.extra.get("num_chains", 16)),
            "burn_in": int(spec.extra.get("burn_in", 100)),
        }

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        Dataset, NRGBooster = self._import_bits()
        train_ds = Dataset(train_df)
        model = NRGBooster.fit(train_ds, self._training_params(spec), seed=spec.seed)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        model.save(str(checkpoint_path))
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
        model = NRGBooster.load(str(checkpoint_path))
        num_samples = spec.num_samples or len(train_df)
        sample_df = model.sample(num_samples, num_steps=int(spec.extra.get("num_steps", 100)))
        sample_df = sample_df[dataset_spec.column_names].copy()
        sample_path = spec.output_dir / "samples.csv"
        sample_df.to_csv(sample_path, index=False)
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
