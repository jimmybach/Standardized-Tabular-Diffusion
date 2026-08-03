from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.models.sample_baselines import _SampleFileEvaluatorMixin, _temporary_sys_path


class _TabSynVendoredBaselineAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    upstream_dirname = "TabSyn-main"
    tabsyn_method_name: str

    def _build_cli_args(self, spec: RunSpec, *, mode: str) -> list[str]:
        args = [
            "main.py",
            "--method",
            self.tabsyn_method_name,
            "--mode",
            mode,
            "--dataname",
            spec.dataset,
            "--gpu",
            str(spec.extra.get("gpu", 0)),
        ]

        passthrough_map = {
            "num_epochs": "--num_epochs",
            "batch_size": "--batch_size",
            "bs": "--bs",
            "training_batch_size": "--training_batch_size",
            "eval_batch_size": "--eval_batch_size",
            "T": "--T",
            "beta_1": "--beta_1",
            "beta_T": "--beta_T",
            "lr_con": "--lr_con",
            "lr_dis": "--lr_dis",
            "total_epochs_both": "--total_epochs_both",
            "sample_step": "--sample_step",
            "lambda_con": "--lambda_con",
            "lambda_dis": "--lambda_dis",
            "nf_con": "--nf_con",
            "nf_dis": "--nf_dis",
            "encoder_dim_con": "--encoder_dim_con",
            "encoder_dim_dis": "--encoder_dim_dis",
            "steps": "--steps",
        }
        for key, flag in passthrough_map.items():
            if spec.extra.get(key) is not None:
                args.extend([flag, str(spec.extra[key])])

        if mode == "sample":
            args.extend(["--save_path", str((spec.output_dir / "samples.csv").resolve())])
            if spec.num_samples is not None:
                args.extend(["--num-samples", str(spec.num_samples)])
        return args

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        self._run_python(self._build_cli_args(spec, mode="train"), self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Training dispatched through TabSyn vendored baseline method `{self.tabsyn_method_name}`.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = (spec.output_dir / "samples.csv").resolve()
        self._run_python(self._build_cli_args(spec, mode="sample"), self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                f"Sampling dispatched through TabSyn vendored baseline method `{self.tabsyn_method_name}`.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


class STaSyAdapter(_TabSynVendoredBaselineAdapter):
    model_name = "stasy"
    tabsyn_method_name = "stasy"


class CoDiAdapter(_TabSynVendoredBaselineAdapter):
    model_name = "codi"
    tabsyn_method_name = "codi"


class CTABGANAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "ctab-gan"
    upstream_dirname = "TabDDPM-main"
    checkpoint_filename = "ctabgan.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _dataset_dir(self, dataset_spec: DatasetSpec) -> Path:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("ctab-gan requires dataset_spec.train_data_path")
        return dataset_spec.train_data_path.parent

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("ctab-gan requires dataset_spec.train_data_path")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("ctab-gan requires exactly one target column")
        ordered_features = [*dataset_spec.numerical_columns, *dataset_spec.categorical_columns]
        frame = pd.read_csv(dataset_spec.train_data_path)[[*ordered_features, *dataset_spec.target_columns]].copy()
        frame.columns = [*[str(index) for index in range(len(ordered_features))], "y"]
        return frame

    def _load_ctabgan_params(self, dataset_name: str) -> dict[str, Any]:
        columns_path = self.repo_root / "TabDDPM-main" / "CTAB-GAN" / "columns.json"
        payload = read_json(columns_path)
        if dataset_name not in payload:
            raise KeyError(f"Missing CTAB-GAN column config for dataset {dataset_name}")
        return dict(payload[dataset_name])

    def _train_kwargs(self, spec: RunSpec) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if spec.extra.get("epochs") is not None:
            kwargs["epochs"] = int(spec.extra["epochs"])
        if spec.extra.get("batch_size") is not None:
            kwargs["batch_size"] = int(spec.extra["batch_size"])
        return kwargs

    def _build_model(self, train_df: pd.DataFrame, spec: RunSpec):
        ctabgan_root = self.repo_root / "TabDDPM-main" / "CTAB-GAN"
        with _temporary_sys_path(ctabgan_root):
            ctab_module = importlib.import_module("model.ctabgan")
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
            notes=[f"Serialized CTAB-GAN checkpoint written to {checkpoint_path}."],
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
