from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler

from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter


@dataclass
class BNNumericMetadata:
    columns: list[str]
    bin_edges: dict[str, list[float]]
    fallback_states: dict[str, str]


class BNPreprocessor:
    def __init__(self, dataset_spec: DatasetSpec, num_bins: int = 16):
        self.dataset_spec = dataset_spec
        self.num_bins = num_bins
        self.numeric_binner = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="quantile")
        self.numeric_columns = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            self.numeric_columns.extend(dataset_spec.target_columns)
        self.categorical_columns = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            self.categorical_columns.extend(dataset_spec.target_columns)
        self._fitted = False
        self.numeric_metadata = BNNumericMetadata(columns=self.numeric_columns, bin_edges={}, fallback_states={})

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        if self.numeric_columns:
            transformed = self.numeric_binner.fit_transform(df[self.numeric_columns])
            for idx, column in enumerate(self.numeric_columns):
                output[column] = transformed[:, idx].astype(int).astype(str)
                self.numeric_metadata.bin_edges[column] = [float(x) for x in self.numeric_binner.bin_edges_[idx]]
                self.numeric_metadata.fallback_states[column] = output[column].mode(dropna=False).iloc[0]
        for column in self.categorical_columns:
            output[column] = df[column].astype(str)
            self.numeric_metadata.fallback_states[column] = output[column].mode(dropna=False).iloc[0]
        self._fitted = True
        return output[self.dataset_spec.column_names]

    def inverse_transform(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("BNPreprocessor must be fitted before inverse_transform.")
        rng = np.random.default_rng(seed)
        completed = df.copy()
        for column in self.dataset_spec.column_names:
            if column not in completed.columns:
                completed[column] = self.numeric_metadata.fallback_states[column]
        output = pd.DataFrame(index=df.index)
        for column in self.numeric_columns:
            edges = self.numeric_metadata.bin_edges[column]
            values: list[float] = []
            for raw_value in completed[column].astype(int).tolist():
                lower = edges[max(0, min(raw_value, len(edges) - 2))]
                upper = edges[max(1, min(raw_value + 1, len(edges) - 1))]
                if lower == upper:
                    values.append(float(lower))
                else:
                    values.append(float(rng.uniform(lower, upper)))
            output[column] = values
        for column in self.categorical_columns:
            output[column] = completed[column].astype(str)
        return output[self.dataset_spec.column_names]


@dataclass
class NFlowPreprocessorState:
    column_names: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    target_columns: list[str]
    categorical_sizes: dict[str, int]


class NFlowPreprocessor:
    def __init__(self, dataset_spec: DatasetSpec):
        self.dataset_spec = dataset_spec
        self.numeric_columns = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            self.numeric_columns.extend(dataset_spec.target_columns)
        self.categorical_columns = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            self.categorical_columns.extend(dataset_spec.target_columns)
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.state = NFlowPreprocessorState(
            column_names=list(dataset_spec.column_names),
            numerical_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            target_columns=list(dataset_spec.target_columns),
            categorical_sizes={},
        )
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            blocks.append(self.scaler.fit_transform(df[self.numeric_columns].astype(float)))
        if self.categorical_columns:
            encoded = self.encoder.fit_transform(df[self.categorical_columns].astype(str))
            for idx, column in enumerate(self.categorical_columns):
                self.state.categorical_sizes[column] = int(len(self.encoder.categories_[idx]))
            blocks.append(encoded.astype(np.float32))
        self._fitted = True
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("NFlowPreprocessor must be fitted before inverse_transform.")
        output = pd.DataFrame(index=range(len(array)))
        start = 0
        if self.numeric_columns:
            stop = start + len(self.numeric_columns)
            numeric = self.scaler.inverse_transform(array[:, start:stop])
            for idx, column in enumerate(self.numeric_columns):
                output[column] = numeric[:, idx]
            start = stop
        if self.categorical_columns:
            stop = start + len(self.categorical_columns)
            categorical = array[:, start:stop]
            decoded = np.zeros_like(categorical)
            for idx, column in enumerate(self.categorical_columns):
                size = self.state.categorical_sizes[column]
                decoded[:, idx] = np.clip(np.round(categorical[:, idx]), 0, max(0, size - 1))
            recovered = self.encoder.inverse_transform(decoded)
            for idx, column in enumerate(self.categorical_columns):
                output[column] = recovered[:, idx]
        return output[self.dataset_spec.column_names]


class BNAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "bn"
    upstream_dirname = "TabSyn-main"
    checkpoint_filename = "model.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.checkpoint_filename)

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("bn requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def train(self, spec: RunSpec) -> ArtifactBundle:
        from pgmpy.estimators import BayesianEstimator, BicScore, HillClimbSearch
        from pgmpy.models import BayesianNetwork

        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        preprocessor = BNPreprocessor(dataset_spec, num_bins=int(spec.extra.get("num_bins", 16)))
        discrete_df = preprocessor.fit_transform(train_df)

        search = HillClimbSearch(discrete_df)
        model_structure = search.estimate(
            scoring_method=BicScore(discrete_df),
            max_indegree=int(spec.extra.get("max_indegree", 2)),
            max_iter=int(spec.extra.get("max_iter", 100)),
        )
        model = BayesianNetwork(model_structure.edges())
        model.fit(discrete_df, estimator=BayesianEstimator, prior_type="BDeu")

        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump({"model": model, "preprocessor": preprocessor}, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Serialized Bayesian-network checkpoint written to {checkpoint_path}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        from pgmpy.sampling import BayesianModelSampling

        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        with trusted_checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        model = payload["model"]
        preprocessor: BNPreprocessor = payload["preprocessor"]

        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        sampler = BayesianModelSampling(model)
        sampled = sampler.forward_sample(size=num_samples, seed=spec.seed, show_progress=False)
        sample_df = preprocessor.inverse_transform(sampled, seed=spec.seed)
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


class NFlowAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "nflow"
    upstream_dirname = "TabSyn-main"
    checkpoint_filename = "model.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.checkpoint_filename)

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("nflow requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def _build_flow(self, num_features: int, spec: RunSpec):
        from nflows.distributions import StandardNormal
        from nflows.flows import Flow
        from nflows.transforms import CompositeTransform, RandomPermutation
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

        num_layers = int(spec.extra.get("num_layers", 4))
        hidden_features = int(spec.extra.get("hidden_features", 64))
        transforms = []
        for _ in range(num_layers):
            transforms.append(RandomPermutation(features=num_features))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=num_features,
                    hidden_features=hidden_features,
                )
            )
        transform = CompositeTransform(transforms)
        return Flow(transform, StandardNormal([num_features]))

    def train(self, spec: RunSpec) -> ArtifactBundle:
        import torch

        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        preprocessor = NFlowPreprocessor(dataset_spec)
        train_array = preprocessor.fit_transform(train_df)
        flow = self._build_flow(train_array.shape[1], spec)
        flow.train()

        optimizer = torch.optim.Adam(flow.parameters(), lr=float(spec.extra.get("learning_rate", 1e-3)))
        batch_size = int(spec.extra.get("batch_size", 512))
        epochs = int(spec.extra.get("epochs", 10))
        train_tensor = torch.tensor(train_array, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(train_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = -flow.log_prob(batch).mean()
                loss.backward()
                optimizer.step()

        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump({"flow": flow, "preprocessor": preprocessor}, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Serialized normalizing-flow checkpoint written to {checkpoint_path}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        import torch

        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        with trusted_checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        flow = payload["flow"]
        preprocessor: NFlowPreprocessor = payload["preprocessor"]
        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        flow.eval()
        with torch.no_grad():
            samples = flow.sample(num_samples).cpu().numpy()
        sample_df = preprocessor.inverse_transform(samples)
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
