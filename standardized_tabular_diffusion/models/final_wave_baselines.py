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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

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


@contextlib.contextmanager
def _disable_torchvision_for_transformers():
    import transformers.utils.import_utils as import_utils

    previous = import_utils._torchvision_available
    import_utils._torchvision_available = False
    try:
        yield
    finally:
        import_utils._torchvision_available = previous


def _tabpfn_cache_env() -> dict[str, str]:
    cache_root = Path(tempfile.gettempdir()) / "standardized-tabular-diffusion" / "tabpfn"
    cache_root.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(cache_root),
        "XDG_CACHE_HOME": str(cache_root),
        "MPLCONFIGDIR": str(cache_root / "mpl"),
    }


class GReaTAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "great"
    upstream_dirname = "TabSyn-main"

    def _model_root(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / "great_model"

    def _metadata_path(self, model_root: Path) -> Path:
        return model_root / "adapter_metadata.json"

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("great requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def _limit_training_frame(self, train_df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        max_train_rows = spec.extra.get("max_train_rows")
        if max_train_rows is None or len(train_df) <= int(max_train_rows):
            return train_df
        return train_df.sample(n=int(max_train_rows), random_state=spec.seed).reset_index(drop=True)

    def _import_model_cls(self):
        with _disable_torchvision_for_transformers():
            with _temporary_sys_path(self.upstream_root):
                from baselines.great.models.great import GReaT  # pylint: disable=import-error

                return GReaT

    @staticmethod
    def _row_to_text(row: pd.Series, column_order: list[str]) -> str:
        return ", ".join(f"{column} is {str(row[column]).strip()}" for column in column_order)

    def _build_training_metadata(
        self,
        train_df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        model: Any,
        *,
        preserve_column_order: bool,
    ) -> dict[str, Any]:
        sample_size = min(len(train_df), 256)
        token_lengths: list[int] = []
        if sample_size > 0:
            sampled_df = train_df.sample(n=sample_size, random_state=0).reset_index(drop=True)
            for _, row in sampled_df.iterrows():
                text = self._row_to_text(row, dataset_spec.column_names)
                token_lengths.append(len(model.tokenizer(text)["input_ids"]))
        observed_max = max(token_lengths, default=64)
        observed_p95 = int(np.percentile(token_lengths, 95)) if token_lengths else observed_max
        recommended_max_length = max(128, min(2048, observed_p95 + 32))
        return {
            "column_names": list(dataset_spec.column_names),
            "target_columns": list(dataset_spec.target_columns),
            "recommended_start_col": (
                dataset_spec.column_names[0]
                if preserve_column_order
                else (dataset_spec.target_columns[0] if dataset_spec.target_columns else dataset_spec.column_names[-1])
            ),
            "recommended_temperature": 0.5,
            "recommended_max_length": recommended_max_length,
            "observed_token_length_max": observed_max,
            "observed_token_length_p95": observed_p95,
        }

    @staticmethod
    def _load_training_metadata(model_root: Path) -> dict[str, Any]:
        metadata_path = model_root / "adapter_metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text())

    @staticmethod
    def _resolve_start_distribution(train_df: pd.DataFrame, start_col: str) -> dict[str, float] | list[Any]:
        series = train_df[start_col]
        if pd.api.types.is_numeric_dtype(series):
            return series.tolist()
        return series.astype(str).value_counts(normalize=True).to_dict()

    @staticmethod
    def _candidate_temperature_schedule(spec: RunSpec, metadata: dict[str, Any]) -> list[float]:
        explicit_schedule = spec.extra.get("temperature_schedule")
        if explicit_schedule is not None:
            return [float(value) for value in explicit_schedule]
        primary = float(spec.extra.get("temperature", metadata.get("recommended_temperature", 0.5)))
        candidates = [primary, 0.4, 0.3]
        return list(dict.fromkeys(round(value, 3) for value in candidates if value > 0))

    @staticmethod
    def _candidate_max_length_schedule(spec: RunSpec, metadata: dict[str, Any]) -> list[int]:
        explicit_schedule = spec.extra.get("max_length_schedule")
        if explicit_schedule is not None:
            return [int(value) for value in explicit_schedule]
        primary = int(spec.extra.get("max_length", metadata.get("recommended_max_length", 256)))
        candidates = [primary, int(primary * 1.25), int(primary * 1.5)]
        return list(dict.fromkeys(max(64, min(4096, int(value))) for value in candidates))

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._limit_training_frame(self._load_training_frame(dataset_spec), spec)
        GReaT = self._import_model_cls()
        model = GReaT(
            spec.extra.get("llm", "distilgpt2"),
            experiment_dir=str(spec.output_dir / "great_trainer"),
            epochs=int(spec.extra.get("epochs", 100)),
            batch_size=int(spec.extra.get("batch_size", 8)),
            save_steps=int(spec.extra.get("save_steps", 2000)),
            logging_steps=int(spec.extra.get("logging_steps", 50)),
            report_to=spec.extra.get("report_to", "none"),
        )
        preserve_column_order = bool(spec.extra.get("preserve_column_order", True))
        env_updates = {"STANDARDIZED_GREAT_PRESERVE_ORDER": "1" if preserve_column_order else "0"}
        with _temporary_env(env_updates):
            model.fit(train_df)
        model_root = self._model_root(spec)
        model.save(str(model_root))
        metadata = self._build_training_metadata(
            train_df,
            dataset_spec,
            model,
            preserve_column_order=preserve_column_order,
        )
        metadata["preserve_column_order"] = preserve_column_order
        self._metadata_path(model_root).write_text(json.dumps(metadata, indent=2))
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Saved GReaT artifacts under {model_root}.",
                f"Recorded adapter metadata with recommended max_length={metadata['recommended_max_length']}, start_col={metadata['recommended_start_col']}, preserve_column_order={preserve_column_order}.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        GReaT = self._import_model_cls()
        model_root = self._model_root(spec)
        metadata = self._load_training_metadata(model_root)
        model = GReaT.load_from_dir(str(model_root))
        model._update_column_information(train_df)  # noqa: SLF001
        model._update_conditional_information(train_df, conditional_col=None)  # noqa: SLF001
        num_samples = spec.num_samples or len(train_df)
        start_col = spec.extra.get("start_col", metadata.get("recommended_start_col"))
        start_col_dist = None if not start_col else self._resolve_start_distribution(train_df, start_col)
        k = int(spec.extra.get("k", max(8, min(100, num_samples * 4))))
        max_tries = int(spec.extra.get("max_tries", 100))
        sample_df = None
        sampling_notes: list[str] = []
        last_error: Exception | None = None
        for max_length in self._candidate_max_length_schedule(spec, metadata):
            for temperature in self._candidate_temperature_schedule(spec, metadata):
                debug_text_path = spec.output_dir / f"great_last_raw_text_t{temperature}_len{max_length}.json"
                try:
                    sample_df = model.sample(
                        n_samples=num_samples,
                        start_col=start_col,
                        start_col_dist=start_col_dist,
                        k=k,
                        max_length=max_length,
                        temperature=temperature,
                        device=spec.device,
                        max_tries=max_tries,
                        debug_text_path=str(debug_text_path),
                    )
                    sampling_notes.append(
                        f"GReaT sampling succeeded with start_col={start_col}, temperature={temperature}, max_length={max_length}, k={k}."
                    )
                    break
                except RuntimeError as exc:
                    last_error = exc
                    sampling_notes.append(
                        f"GReaT sampling retry exhausted with start_col={start_col}, temperature={temperature}, max_length={max_length}; last raw text dumped to {debug_text_path.name}."
                    )
            if sample_df is not None:
                break
        if sample_df is None:
            raise RuntimeError(
                "GReaT sampling exhausted the configured temperature/max_length fallback schedule without producing enough valid rows. "
                "Try increasing epochs, max_train_rows, or the sampling schedules in sample.extra."
            ) from last_error
        sample_df = sample_df[dataset_spec.column_names].copy()
        sample_path = spec.output_dir / "samples.csv"
        sample_df.to_csv(sample_path, index=False)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=sampling_notes,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


class ARFAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "arf"
    upstream_dirname = "TabSyn-main"
    checkpoint_filename = "model.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("arf requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    def train(self, spec: RunSpec) -> ArtifactBundle:
        from arfpy.arf import arf

        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        model = arf(
            train_df,
            num_trees=int(spec.extra.get("num_trees", 30)),
            delta=float(spec.extra.get("delta", 0.0)),
            max_iters=int(spec.extra.get("max_iters", 10)),
            early_stop=bool(spec.extra.get("early_stop", True)),
            verbose=bool(spec.extra.get("verbose", False)),
            min_node_size=int(spec.extra.get("min_node_size", 5)),
        )
        model.forde(
            dist=spec.extra.get("dist", "truncnorm"),
            oob=bool(spec.extra.get("oob", False)),
            alpha=float(spec.extra.get("alpha", 0.0)),
        )
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump(model, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[f"Serialized ARF checkpoint written to {checkpoint_path}."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("rb") as handle:
            model = pickle.load(handle)
        train_df = self._load_training_frame(dataset_spec)
        num_samples = spec.num_samples or len(train_df)
        sample_df = model.forge(num_samples)[dataset_spec.column_names].copy()
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


@dataclass
class TabEBMState:
    column_names: list[str]
    feature_columns: list[str]
    target_column: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    feature_matrix: np.ndarray
    target_values: np.ndarray
    class_labels: list[str]
    categorical_sizes: dict[str, int]
    sgld_defaults: dict[str, Any]


class TabEBMPreprocessor:
    def __init__(self, dataset_spec: DatasetSpec):
        if dataset_spec.task_type != "classification" or len(dataset_spec.target_columns) != 1:
            raise ValueError("tabebm currently supports only single-target classification datasets.")
        self.dataset_spec = dataset_spec
        self.feature_columns = [column for column in dataset_spec.column_names if column not in dataset_spec.target_columns]
        self.numeric_columns = [column for column in dataset_spec.numerical_columns if column in self.feature_columns]
        self.categorical_columns = [column for column in dataset_spec.categorical_columns if column in self.feature_columns]
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.target_encoder = LabelEncoder()
        self.categorical_sizes: dict[str, int] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            blocks.append(self.scaler.fit_transform(df[self.numeric_columns].astype(float)))
        if self.categorical_columns:
            encoded = self.encoder.fit_transform(df[self.categorical_columns].astype(str))
            for idx, column in enumerate(self.categorical_columns):
                self.categorical_sizes[column] = int(len(self.encoder.categories_[idx]))
            blocks.append(encoded.astype(np.float32))
        X = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.empty((len(df), 0), dtype=np.float32)
        y = self.target_encoder.fit_transform(df[self.dataset_spec.target_columns[0]].astype(str))
        self._fitted = True
        return X, y.astype(np.int64)

    def inverse_transform(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TabEBMPreprocessor must be fitted before inverse_transform.")
        output = pd.DataFrame(index=range(len(X)))
        start = 0
        if self.numeric_columns:
            stop = start + len(self.numeric_columns)
            numeric = self.scaler.inverse_transform(X[:, start:stop])
            for idx, column in enumerate(self.numeric_columns):
                output[column] = numeric[:, idx]
            start = stop
        if self.categorical_columns:
            stop = start + len(self.categorical_columns)
            categorical = X[:, start:stop].copy()
            decoded = np.zeros_like(categorical)
            for idx, column in enumerate(self.categorical_columns):
                size = self.categorical_sizes[column]
                decoded[:, idx] = np.clip(np.round(categorical[:, idx]), 0, max(0, size - 1))
            recovered = self.encoder.inverse_transform(decoded)
            for idx, column in enumerate(self.categorical_columns):
                output[column] = recovered[:, idx]
        output[self.dataset_spec.target_columns[0]] = self.target_encoder.inverse_transform(y.astype(int))
        return output[self.dataset_spec.column_names]


class TabEBMAdapter(BaseModelAdapter, _SampleFileEvaluatorMixin):
    model_name = "tabebm"
    upstream_dirname = "TabSyn-main"
    checkpoint_filename = "model.pkl"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("tabebm requires dataset_spec.train_data_path")
        return pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()

    @staticmethod
    def _allow_gated_sampling(spec: RunSpec) -> bool:
        explicit_flag = spec.extra.get("allow_gated_model")
        if explicit_flag is not None:
            return bool(explicit_flag)
        return os.environ.get("STANDARDIZED_TABPFN_ALLOW_GATED", "").strip().lower() in {"1", "true", "yes"}

    def _build_model(self, spec: RunSpec):
        from tabpfn.classifier import TabPFNClassifier
        from tabpfn.inference_config import InferenceConfig, PreprocessorConfig

        config = InferenceConfig(
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none", differentiable=True)],
            FINGERPRINT_FEATURE=False,
            FEATURE_SHIFT_METHOD=None,
            CLASS_SHIFT_METHOD=None,
        )
        return TabPFNClassifier(
            n_estimators=1,
            fit_mode="fit_preprocessors",
            differentiable_input=True,
            inference_config=config,
            device=spec.device,
            show_progress_bar=False,
            random_state=spec.seed,
        )

    @staticmethod
    def _add_surrogate_negative_samples(X: np.ndarray, distance_negative_class: float) -> tuple[np.ndarray, np.ndarray]:
        num_features = X.shape[1]
        if num_features == 2:
            surrogate_negatives = np.array(
                [
                    [-distance_negative_class, -distance_negative_class],
                    [distance_negative_class, distance_negative_class],
                    [-distance_negative_class, distance_negative_class],
                    [distance_negative_class, -distance_negative_class],
                ],
                dtype=X.dtype,
            )
        else:
            points: set[tuple[float, ...]] = set()
            while len(points) < 4:
                point = tuple(np.random.choice([-distance_negative_class, distance_negative_class], num_features).tolist())
                points.add(point)
                points.add(tuple((-np.array(point)).tolist()))
            surrogate_negatives = np.array(list(points), dtype=X.dtype)
        X_ebm = np.concatenate([X, surrogate_negatives], axis=0)
        y_ebm = np.concatenate(
            [
                np.zeros(X.shape[0], dtype=np.int64),
                np.ones(surrogate_negatives.shape[0], dtype=np.int64),
            ]
        )
        return X_ebm, y_ebm

    @staticmethod
    def _class_sample_counts(labels: np.ndarray, total: int) -> dict[int, int]:
        values, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        base = np.floor(probabilities * total).astype(int)
        remainder = total - int(base.sum())
        order = np.argsort(-(probabilities * total - base))
        for idx in order[:remainder]:
            base[idx] += 1
        return {int(value): int(count) for value, count in zip(values, base)}

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        preprocessor = TabEBMPreprocessor(dataset_spec)
        X, y = preprocessor.fit_transform(train_df)
        state = TabEBMState(
            column_names=list(dataset_spec.column_names),
            feature_columns=list(preprocessor.feature_columns),
            target_column=dataset_spec.target_columns[0],
            numerical_columns=list(preprocessor.numeric_columns),
            categorical_columns=list(preprocessor.categorical_columns),
            feature_matrix=X,
            target_values=y,
            class_labels=preprocessor.target_encoder.classes_.tolist(),
            categorical_sizes=dict(preprocessor.categorical_sizes),
            sgld_defaults={
                "starting_point_noise_std": float(spec.extra.get("starting_point_noise_std", 0.01)),
                "sgld_step_size": float(spec.extra.get("sgld_step_size", 0.1)),
                "sgld_noise_std": float(spec.extra.get("sgld_noise_std", 0.01)),
                "sgld_steps": int(spec.extra.get("sgld_steps", 50)),
                "distance_negative_class": float(spec.extra.get("distance_negative_class", 5.0)),
            },
        )
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("wb") as handle:
            pickle.dump({"state": state, "preprocessor": preprocessor}, handle)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Serialized TabEBM state written to {checkpoint_path}.",
                "Sampling requires a TabPFN environment with accepted Hugging Face gated-model terms.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if not self._allow_gated_sampling(spec):
            raise RuntimeError(
                "TabEBM sample is intentionally opt-in because it depends on Prior Labs' gated TabPFN model access. "
                "Set sample.extra.allow_gated_model=true or STANDARDIZED_TABPFN_ALLOW_GATED=1 after accepting the Hugging Face terms."
            )
        checkpoint_path = self._resolve_checkpoint_path(spec)
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        state: TabEBMState = payload["state"]
        preprocessor: TabEBMPreprocessor = payload["preprocessor"]
        num_samples = spec.num_samples or len(state.feature_matrix)
        class_counts = self._class_sample_counts(state.target_values, num_samples)
        rng = np.random.default_rng(spec.seed)
        sampled_blocks: list[np.ndarray] = []
        sampled_targets: list[np.ndarray] = []

        with _temporary_env(_tabpfn_cache_env()):
            try:
                for class_id, class_count in class_counts.items():
                    if class_count <= 0:
                        continue
                    class_features = state.feature_matrix[state.target_values == class_id]
                    model = self._build_model(spec)
                    X_ebm, y_ebm = self._add_surrogate_negative_samples(
                        class_features,
                        distance_negative_class=float(spec.extra.get("distance_negative_class", state.sgld_defaults["distance_negative_class"])),
                    )
                    X_ebm_tensor = torch.tensor(X_ebm, dtype=torch.float32)
                    y_ebm_tensor = torch.tensor(y_ebm, dtype=torch.long)
                    model.fit_with_differentiable_input(X_ebm_tensor, y_ebm_tensor)

                    start_idx = rng.integers(0, len(class_features), size=class_count)
                    X_sgld = torch.tensor(class_features[start_idx], dtype=torch.float32)
                    noise_std = float(spec.extra.get("starting_point_noise_std", state.sgld_defaults["starting_point_noise_std"]))
                    if noise_std > 0:
                        X_sgld = X_sgld + torch.randn_like(X_sgld) * noise_std
                    X_sgld = X_sgld.requires_grad_(True)

                    step_size = float(spec.extra.get("sgld_step_size", state.sgld_defaults["sgld_step_size"]))
                    sgld_noise_std = float(spec.extra.get("sgld_noise_std", state.sgld_defaults["sgld_noise_std"]))
                    sgld_steps = int(spec.extra.get("sgld_steps", state.sgld_defaults["sgld_steps"]))

                    for _ in range(sgld_steps):
                        if X_sgld.grad is not None:
                            X_sgld.grad.zero_()
                        logits = model.forward([X_sgld], return_logits=True)
                        if logits.ndim > 2:
                            logits = logits.reshape(-1, logits.shape[-1])
                        energy = -torch.logsumexp(logits, dim=-1).mean()
                        energy.backward()
                        with torch.no_grad():
                            X_sgld = (
                                X_sgld
                                - step_size * X_sgld.grad
                                + sgld_noise_std * torch.randn_like(X_sgld)
                            ).detach().requires_grad_(True)

                    sampled_blocks.append(X_sgld.detach().cpu().numpy())
                    sampled_targets.append(np.full(class_count, class_id, dtype=np.int64))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "TabEBM sampling requires a working TabPFN setup and accepted gated-model terms. "
                    "If you have not accepted Prior Labs' model terms on Hugging Face, do that first and "
                    "provide authentication before retrying."
                ) from exc

        X_sampled = np.concatenate(sampled_blocks, axis=0)
        y_sampled = np.concatenate(sampled_targets, axis=0)
        sample_df = preprocessor.inverse_transform(X_sampled, y_sampled)
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
