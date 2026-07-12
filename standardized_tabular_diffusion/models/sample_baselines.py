from __future__ import annotations

import contextlib
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from standardized_tabular_diffusion.evaluation.tabstruct import normalize_tabdiff_or_tabsyn_summary
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter


@contextlib.contextmanager
def _temporary_sys_path(path: Path):
    inserted = False
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


class _SampleFileEvaluatorMixin:
    def _evaluate_from_sample_file(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = spec.extra.get("sample_path")
        if sample_path is None:
            raise ValueError(f"{self.model_name} evaluation requires spec.extra['sample_path'].")
        summary_path = spec.output_dir / "standardized_summary.json"
        normalize_tabdiff_or_tabsyn_summary(
            repo_root=self.repo_root,
            model_name=self.model_name,
            dataset=spec.dataset,
            sample_path=Path(sample_path),
            output_path=summary_path,
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=Path(sample_path),
            standardized_summary_path=summary_path,
        )
        return self._write_bundle(bundle)


class _PickleBackedGenerativeAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    checkpoint_filename = "model.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError(f"{self.model_name} requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)
        return frame[dataset_spec.column_names].copy()

    def _discrete_columns(self, dataset_spec: DatasetSpec) -> list[str]:
        discrete = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            for target in dataset_spec.target_columns:
                if target not in discrete:
                    discrete.append(target)
        return discrete

    def _device_arg(self, spec: RunSpec) -> Any:
        if spec.device == "cpu":
            return False
        if spec.device.startswith("cuda"):
            return spec.device
        return False

    def _train_kwargs(self, spec: RunSpec) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if spec.extra.get("epochs") is not None:
            kwargs["epochs"] = int(spec.extra["epochs"])
        if spec.extra.get("batch_size") is not None:
            kwargs["batch_size"] = int(spec.extra["batch_size"])
        if spec.extra.get("verbose") is not None:
            kwargs["verbose"] = bool(spec.extra["verbose"])
        return kwargs

    def _import_synthesizer_cls(self):
        vendor_root = self.repo_root / "TabDDPM-main" / "CTGAN" / "CTGAN"
        with _temporary_sys_path(vendor_root):
            from ctgan import CTGANSynthesizer, TVAESynthesizer  # pylint: disable=import-error

            return {
                "ctgan": CTGANSynthesizer,
                "tvae": TVAESynthesizer,
            }[self.model_name]

    def _build_synthesizer(self, spec: RunSpec):
        synthesizer_cls = self._import_synthesizer_cls()
        kwargs = self._train_kwargs(spec)
        if self.model_name == "ctgan":
            kwargs["cuda"] = self._device_arg(spec)
        else:
            kwargs["device"] = spec.device
        return synthesizer_cls(**kwargs)

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        model = self._build_synthesizer(spec)
        model.fit(train_df, discrete_columns=self._discrete_columns(dataset_spec))
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump(model, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Serialized {self.model_name} checkpoint written to {checkpoint_path}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")

        vendor_root = self.repo_root / "TabDDPM-main" / "CTGAN" / "CTGAN"
        with _temporary_sys_path(vendor_root):
            with checkpoint_path.open("rb") as handle:
                model = pickle.load(handle)

        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        sample_df = model.sample(num_samples)
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


class CTGANAdapter(_PickleBackedGenerativeAdapter):
    model_name = "ctgan"
    upstream_dirname = "TabDDPM-main"


class TVAEAdapter(_PickleBackedGenerativeAdapter):
    model_name = "tvae"
    upstream_dirname = "TabDDPM-main"


class SMOTEAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "smote"
    upstream_dirname = "TabSyn-main"

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("SMOTE requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)
        return frame[dataset_spec.column_names].copy()

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        if dataset_spec.task_type != "classification":
            raise ValueError("SMOTE is only supported for classification datasets.")
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                "SMOTE has no persistent fitted checkpoint in the standardized adapter.",
                "Sample generation recomputes the resampled table directly from the canonical training split.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        if dataset_spec.task_type != "classification":
            raise ValueError("SMOTE is only supported for classification datasets.")

        from imblearn.over_sampling import SMOTE  # pylint: disable=import-error

        train_df = self._load_training_frame(dataset_spec)
        target = dataset_spec.target_columns[0]
        feature_cols = [column for column in dataset_spec.column_names if column != target]
        x_train = train_df[feature_cols]
        y_train = train_df[target]

        sampler = SMOTE(random_state=spec.seed, k_neighbors=int(spec.extra.get("k_neighbors", 5)))
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        sampled_df = pd.DataFrame(x_resampled, columns=feature_cols)
        sampled_df[target] = y_resampled
        sampled_df = sampled_df[dataset_spec.column_names]

        desired_rows = spec.num_samples or len(sampled_df)
        replace = len(sampled_df) < desired_rows
        sampled_df = sampled_df.sample(n=desired_rows, replace=replace, random_state=spec.seed).reset_index(drop=True)

        sample_path = spec.output_dir / "samples.csv"
        sampled_df.to_csv(sample_path, index=False)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                "Rows are drawn from the SMOTE-resampled table.",
                "If num_samples differs from the balanced-resample size, the adapter resamples rows from the SMOTE output.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
