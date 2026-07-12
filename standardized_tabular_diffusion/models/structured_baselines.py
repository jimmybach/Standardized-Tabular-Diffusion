from __future__ import annotations

import contextlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler

from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.models.sample_baselines import _SampleFileEvaluatorMixin, _temporary_sys_path


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


class BNAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
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
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        model = payload["model"]
        preprocessor: BNPreprocessor = payload["preprocessor"]

        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        sampler = BayesianModelSampling(model)
        sampled = sampler.forward_sample(size=num_samples, seed=spec.seed, show_progress=False)
        sample_df = preprocessor.inverse_transform(sampled, seed=spec.seed)
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


class NFlowAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
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
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("rb") as handle:
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


class GoggleAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "goggle"
    upstream_dirname = "TabSyn-main"
    checkpoint_filename = "model.pt"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.checkpoint_filename)

    def _cache_env(self) -> dict[str, str]:
        cache_root = Path(tempfile.gettempdir()) / "standardized-tabular-diffusion" / "dgl"
        cache_root.mkdir(parents=True, exist_ok=True)
        return {
            "DGLBACKEND": "pytorch",
            "HOME": str(cache_root),
            "MPLCONFIGDIR": str(cache_root / "mpl"),
        }

    def _import_bits(self):
        compat_root = self.repo_root / "standardized_tabular_diffusion" / "vendor"
        with _temporary_env(self._cache_env()):
            with _temporary_sys_path(compat_root):
                with _temporary_sys_path(self.upstream_root):
                    from utils_train import preprocess  # pylint: disable=import-error
                    from baselines.goggle.GoggleModel import GoggleModel  # pylint: disable=import-error

                    return preprocess, GoggleModel

    def _load_info(self, dataset_spec: DatasetSpec) -> dict[str, Any]:
        return json.loads(dataset_spec.metadata_path.read_text())

    def _recover_data(self, syn_num: np.ndarray, syn_cat: np.ndarray, info: dict[str, Any]) -> pd.DataFrame:
        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]
        idx_mapping = {int(key): value for key, value in info["idx_mapping"].items()}

        if info["task_type"] == "regression":
            syn_target = syn_num[:, : len(target_col_idx)]
            syn_num = syn_num[:, len(target_col_idx) :]
        else:
            syn_target = syn_cat[:, : len(target_col_idx)]
            syn_cat = syn_cat[:, len(target_col_idx) :]

        syn_df = pd.DataFrame()
        total_columns = len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)
        for idx in range(total_columns):
            if idx in set(num_col_idx):
                syn_df[idx] = syn_num[:, idx_mapping[idx]]
            elif idx in set(cat_col_idx):
                syn_df[idx] = syn_cat[:, idx_mapping[idx] - len(num_col_idx)]
            else:
                syn_df[idx] = syn_target[:, idx_mapping[idx] - len(num_col_idx) - len(cat_col_idx)]
        idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
        syn_df.rename(columns=idx_name_mapping, inplace=True)
        return syn_df

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        preprocess, GoggleModel = self._import_bits()
        info = self._load_info(dataset_spec)
        dataset_dir = dataset_spec.metadata_path.parent

        with _temporary_env(self._cache_env()):
            dataset = preprocess(str(dataset_dir), task_type=info["task_type"], cat_encoding="one-hot")
            x_train = torch.tensor(dataset.X_num["train"]).float()
            model = GoggleModel(
                ds_name=spec.dataset,
                input_dim=x_train.shape[1],
                encoder_dim=int(spec.extra.get("encoder_dim", 256)),
                encoder_l=int(spec.extra.get("encoder_l", 2)),
                het_encoding=bool(spec.extra.get("het_encoding", True)),
                decoder_dim=int(spec.extra.get("decoder_dim", 256)),
                decoder_l=int(spec.extra.get("decoder_l", 2)),
                threshold=float(spec.extra.get("threshold", 0.1)),
                decoder_arch=spec.extra.get("decoder_arch", "gcn"),
                graph_prior=None,
                prior_mask=None,
                device=spec.device,
                beta=float(spec.extra.get("beta", 1.0)),
                learning_rate=float(spec.extra.get("learning_rate", 0.01)),
                seed=spec.seed,
                epochs=int(spec.extra.get("epochs", 10)),
                batch_size=int(spec.extra.get("batch_size", 512)),
            )
            train_loader = torch.utils.data.DataLoader(x_train, batch_size=model.batch_size, shuffle=True)
            checkpoint_path = self._resolve_checkpoint_path(spec)
            model.fit(train_loader, str(checkpoint_path))
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Serialized GOGGLE checkpoint written to {self._resolve_checkpoint_path(spec)}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        preprocess, GoggleModel = self._import_bits()
        info = self._load_info(dataset_spec)
        dataset_dir = dataset_spec.metadata_path.parent
        with _temporary_env(self._cache_env()):
            dataset = preprocess(str(dataset_dir), task_type=info["task_type"], cat_encoding="one-hot")
            x_train = torch.tensor(dataset.X_num["train"]).float()
            model = GoggleModel(
                ds_name=spec.dataset,
                input_dim=x_train.shape[1],
                encoder_dim=int(spec.extra.get("encoder_dim", 256)),
                encoder_l=int(spec.extra.get("encoder_l", 2)),
                het_encoding=bool(spec.extra.get("het_encoding", True)),
                decoder_dim=int(spec.extra.get("decoder_dim", 256)),
                decoder_l=int(spec.extra.get("decoder_l", 2)),
                threshold=float(spec.extra.get("threshold", 0.1)),
                decoder_arch=spec.extra.get("decoder_arch", "gcn"),
                graph_prior=None,
                prior_mask=None,
                device=spec.device,
                beta=float(spec.extra.get("beta", 1.0)),
                learning_rate=float(spec.extra.get("learning_rate", 0.01)),
                seed=spec.seed,
                epochs=int(spec.extra.get("epochs", 10)),
                batch_size=int(spec.extra.get("batch_size", 512)),
            )
            model.model.load_state_dict(torch.load(self._resolve_checkpoint_path(spec), map_location=spec.device))

            num_samples = spec.num_samples or len(x_train)
            x_ref = x_train[:num_samples]
            samples = model.sample(x_ref)
            n_num_feat = len(info["num_col_idx"])
            n_cat_feat = len(info["cat_col_idx"])
            if info["task_type"] == "regression":
                n_num_feat += len(info["target_col_idx"])
            else:
                n_cat_feat += len(info["target_col_idx"])
            syn_data_num = samples[:, :n_num_feat]
            cat_sample = samples[:, n_num_feat:]
            syn_num = dataset.num_transform.inverse_transform(syn_data_num)
            syn_cat = dataset.cat_transform.inverse_transform(cat_sample)
            syn_df = self._recover_data(syn_num, syn_cat, info)

        sample_path = spec.output_dir / "samples.csv"
        syn_df.to_csv(sample_path, index=False)
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
