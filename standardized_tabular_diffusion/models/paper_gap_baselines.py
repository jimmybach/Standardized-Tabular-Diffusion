from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter


@dataclass
class TabSDSState:
    train_df: pd.DataFrame
    column_names: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    target_columns: list[str]
    task_type: str


class TabSDSAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "tabsds"
    upstream_dirname = "."
    checkpoint_filename = "tabsds.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("tabsds requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        state = TabSDSState(
            train_df=train_df,
            column_names=list(dataset_spec.column_names),
            numerical_columns=list(dataset_spec.numerical_columns),
            categorical_columns=list(dataset_spec.categorical_columns),
            target_columns=list(dataset_spec.target_columns),
            task_type=dataset_spec.task_type,
        )
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump(state, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Serialized TabSDS state written to {checkpoint_path}.",
                "This adapter uses a local non-parametric compatibility implementation inspired by the TabSDS paper.",
            ],
        )
        return self._write_bundle(bundle)

    @staticmethod
    def _sample_column_marginal(series: pd.Series, *, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        values = series.to_numpy()
        if len(values) == 0:
            raise ValueError("Cannot sample from an empty training column.")
        sample_idx = rng.integers(0, len(values), size=n_samples)
        sampled = values[sample_idx]
        return np.array(sampled, copy=True)

    @staticmethod
    def _sorted_values(values: np.ndarray, *, categorical: bool) -> np.ndarray:
        if categorical:
            order = np.argsort(values.astype(str), kind="mergesort")
            return values[order]
        return np.sort(values, kind="mergesort")

    @staticmethod
    def _rank_template(series: pd.Series) -> np.ndarray:
        return series.rank(method="first").to_numpy().astype(int) - 1

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_name}: {checkpoint_path}")
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        with trusted_checkpoint.open("rb") as handle:
            state: TabSDSState = pickle.load(handle)

        train_df = state.train_df[state.column_names].copy()
        n_samples = spec.num_samples or len(train_df)
        rng = np.random.default_rng(spec.seed)

        template_idx = rng.integers(0, len(train_df), size=n_samples)
        template_df = train_df.iloc[template_idx].reset_index(drop=True)

        sampled_columns: dict[str, np.ndarray] = {}
        for column in state.column_names:
            marginal = self._sample_column_marginal(train_df[column], n_samples=n_samples, rng=rng)
            is_categorical = column in state.categorical_columns or (
                state.task_type == "classification" and column in state.target_columns
            )
            sorted_marginal = self._sorted_values(marginal, categorical=is_categorical)
            template_ranks = self._rank_template(template_df[column])
            assigned = np.empty(n_samples, dtype=object if is_categorical else sorted_marginal.dtype)
            assigned[template_ranks] = sorted_marginal
            sampled_columns[column] = assigned

        sample_df = pd.DataFrame(sampled_columns, columns=state.column_names)
        for column in state.numerical_columns:
            sample_df[column] = pd.to_numeric(sample_df[column], errors="coerce")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                "TabSDS sampling uses a local rank-and-shuffle compatibility implementation to preserve marginal structure and approximate dependency ordering.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
