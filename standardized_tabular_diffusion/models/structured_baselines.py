from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import zipfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter


class BNPreprocessor:
    def __init__(self, dataset_spec: DatasetSpec, num_bins: int = 16):
        self.dataset_spec = dataset_spec
        self.num_bins = num_bins
        self.numeric_columns = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            self.numeric_columns.extend(dataset_spec.target_columns)
        self.categorical_columns = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            self.categorical_columns.extend(dataset_spec.target_columns)
        self.numeric_columns = list(dict.fromkeys(self.numeric_columns))
        self.categorical_columns = list(dict.fromkeys(self.categorical_columns))
        self.bin_edges: dict[str, list[float]] = {}
        self.constant_values: dict[str, float] = {}
        self.categorical_levels: dict[str, list[str]] = {}
        self.discrete_state_names: dict[str, list[str]] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        variable_numeric: list[str] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"BN numerical column {column!r} contains non-finite values.")
            if bool(np.all(values == values[0])):
                self.constant_values[column] = float(values[0])
                output[column] = "0"
                self.discrete_state_names[column] = ["0"]
            else:
                variable_numeric.append(column)
        if variable_numeric:
            binner = KBinsDiscretizer(
                n_bins=self.num_bins,
                encode="ordinal",
                strategy="quantile",
                quantile_method="averaged_inverted_cdf",
                subsample=None,
            )
            transformed = binner.fit_transform(df[variable_numeric])
            for idx, column in enumerate(variable_numeric):
                output[column] = transformed[:, idx].astype(int).astype(str)
                edges = [float(value) for value in binner.bin_edges_[idx]]
                if len(edges) < 2 or not np.isfinite(edges).all():
                    raise ValueError(f"BN learned invalid discretization edges for {column!r}.")
                self.bin_edges[column] = edges
                self.discrete_state_names[column] = [str(value) for value in range(len(edges) - 1)]
        for column in self.categorical_columns:
            output[column] = df[column].astype(str)
            levels = sorted(output[column].unique().tolist())
            if not levels:
                raise ValueError(f"BN categorical column {column!r} has no observed states.")
            self.categorical_levels[column] = levels
            self.discrete_state_names[column] = levels
        self._fitted = True
        return output[self.dataset_spec.column_names]

    def inverse_transform(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("BNPreprocessor must be fitted before inverse_transform.")
        if set(df.columns) != set(self.dataset_spec.column_names):
            raise ValueError(
                "Official pgmpy sample columns differ from the BN checkpoint schema: "
                f"observed={list(df.columns)}, expected={self.dataset_spec.column_names}"
            )
        rng = np.random.default_rng(seed)
        completed = df[self.dataset_spec.column_names].copy()
        output = pd.DataFrame(index=df.index)
        for column in self.numeric_columns:
            states = completed[column].astype(str)
            expected_states = self.discrete_state_names[column]
            unexpected = sorted(set(states) - set(expected_states))
            if unexpected:
                raise ValueError(f"Official pgmpy returned invalid states for {column!r}: {unexpected}")
            if column in self.constant_values:
                output[column] = self.constant_values[column]
                continue
            edges = self.bin_edges[column]
            values: list[float] = []
            for raw_value in states.astype(int).tolist():
                lower = edges[raw_value]
                upper = edges[raw_value + 1]
                if lower == upper:
                    values.append(float(lower))
                else:
                    values.append(float(rng.uniform(lower, upper)))
            output[column] = values
        for column in self.categorical_columns:
            states = completed[column].astype(str)
            unexpected = sorted(set(states) - set(self.categorical_levels[column]))
            if unexpected:
                raise ValueError(f"Official pgmpy returned invalid states for {column!r}: {unexpected}")
            output[column] = states
        return output[self.dataset_spec.column_names]

    def to_payload(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("BNPreprocessor must be fitted before serialization.")
        return {
            "num_bins": self.num_bins,
            "numerical_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "bin_edges": self.bin_edges,
            "constant_values": self.constant_values,
            "categorical_levels": self.categorical_levels,
            "discrete_state_names": self.discrete_state_names,
            "quantile_method": "averaged_inverted_cdf",
            "subsample": None,
        }

    @classmethod
    def from_payload(cls, dataset_spec: DatasetSpec, payload: Any) -> BNPreprocessor:
        if not isinstance(payload, dict):
            raise ValueError("BN checkpoint preprocessing state must be a JSON object.")
        if payload.get("quantile_method") != "averaged_inverted_cdf" or payload.get("subsample") is not None:
            raise ValueError("BN checkpoint requests an unsupported discretization recipe.")
        num_bins = BNAdapter._positive_int("checkpoint num_bins", payload.get("num_bins"), minimum=2)
        instance = cls(dataset_spec, num_bins=num_bins)
        if payload.get("numerical_columns") != instance.numeric_columns:
            raise ValueError("BN checkpoint numerical columns differ from the current DatasetSpec.")
        if payload.get("categorical_columns") != instance.categorical_columns:
            raise ValueError("BN checkpoint categorical columns differ from the current DatasetSpec.")
        bin_edges = payload.get("bin_edges")
        constant_values = payload.get("constant_values")
        categorical_levels = payload.get("categorical_levels")
        state_names = payload.get("discrete_state_names")
        if not all(isinstance(value, dict) for value in (bin_edges, constant_values, categorical_levels, state_names)):
            raise ValueError("BN checkpoint preprocessing maps are invalid.")
        if set(bin_edges) | set(constant_values) != set(instance.numeric_columns):
            raise ValueError("BN checkpoint does not define every numerical column exactly once.")
        if set(bin_edges) & set(constant_values):
            raise ValueError("BN checkpoint marks a numerical column as both variable and constant.")
        if set(categorical_levels) != set(instance.categorical_columns):
            raise ValueError("BN checkpoint categorical-level map is incomplete.")
        if set(state_names) != set(dataset_spec.column_names):
            raise ValueError("BN checkpoint discrete state-name map is incomplete.")
        instance.bin_edges = {}
        for column, raw_edges in bin_edges.items():
            if not isinstance(raw_edges, list) or len(raw_edges) < 2:
                raise ValueError(f"BN checkpoint bin edges are invalid for {column!r}.")
            edges = [float(value) for value in raw_edges]
            if not np.isfinite(edges).all() or any(right <= left for left, right in zip(edges, edges[1:])):
                raise ValueError(f"BN checkpoint bin edges are not finite and strictly increasing for {column!r}.")
            instance.bin_edges[column] = edges
        instance.constant_values = {column: float(value) for column, value in constant_values.items()}
        if not np.isfinite(list(instance.constant_values.values())).all():
            raise ValueError("BN checkpoint contains a non-finite constant numerical value.")
        instance.categorical_levels = {}
        for column, values in categorical_levels.items():
            if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
                raise ValueError(f"BN checkpoint categorical levels are invalid for {column!r}.")
            if values != sorted(set(values)):
                raise ValueError(f"BN checkpoint categorical levels are not unique and canonical for {column!r}.")
            instance.categorical_levels[column] = values
        instance.discrete_state_names = {}
        for column, values in state_names.items():
            if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
                raise ValueError(f"BN checkpoint discrete states are invalid for {column!r}.")
            if len(values) != len(set(values)):
                raise ValueError(f"BN checkpoint discrete states are duplicated for {column!r}.")
            instance.discrete_state_names[column] = values
        for column, edges in instance.bin_edges.items():
            if instance.discrete_state_names[column] != [str(value) for value in range(len(edges) - 1)]:
                raise ValueError(f"BN checkpoint numerical states disagree with bin edges for {column!r}.")
        for column in instance.constant_values:
            if instance.discrete_state_names[column] != ["0"]:
                raise ValueError(f"BN checkpoint constant state is invalid for {column!r}.")
        for column, levels in instance.categorical_levels.items():
            if instance.discrete_state_names[column] != levels:
                raise ValueError(f"BN checkpoint categorical states disagree with levels for {column!r}.")
        instance._fitted = True
        return instance


@dataclass
class NFlowPreprocessorState:
    column_names: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    target_columns: list[str]
    numerical_mean: list[float]
    numerical_scale: list[float]
    categorical_levels: dict[str, list[str]]


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
            numerical_mean=[],
            numerical_scale=[],
            categorical_levels={},
        )
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            numerical = self.scaler.fit_transform(df[self.numeric_columns].astype(float))
            self.state.numerical_mean = [float(value) for value in self.scaler.mean_]
            self.state.numerical_scale = [float(value) for value in self.scaler.scale_]
            blocks.append(numerical)
        if self.categorical_columns:
            encoded = self.encoder.fit_transform(df[self.categorical_columns].astype(str))
            for idx, column in enumerate(self.categorical_columns):
                self.state.categorical_levels[column] = [str(value) for value in self.encoder.categories_[idx]]
            blocks.append(encoded.astype(np.float32))
        self._fitted = True
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("NFlowPreprocessor must be fitted before inverse_transform.")
        expected_features = len(self.numeric_columns) + len(self.categorical_columns)
        if array.ndim != 2 or array.shape[1] != expected_features or not np.isfinite(array).all():
            raise ValueError("NFlow inverse transformation requires a finite 2D array with one value per column.")
        output = pd.DataFrame(index=range(len(array)))
        start = 0
        if self.numeric_columns:
            stop = start + len(self.numeric_columns)
            numeric = array[:, start:stop].astype(np.float32, copy=True)
            numeric *= np.asarray(self.state.numerical_scale, dtype=np.float64)
            numeric += np.asarray(self.state.numerical_mean, dtype=np.float64)
            for idx, column in enumerate(self.numeric_columns):
                output[column] = numeric[:, idx]
            start = stop
        if self.categorical_columns:
            stop = start + len(self.categorical_columns)
            categorical = array[:, start:stop]
            for idx, column in enumerate(self.categorical_columns):
                levels = self.state.categorical_levels[column]
                indices = np.clip(np.round(categorical[:, idx]), 0, len(levels) - 1).astype(np.int64)
                output[column] = [levels[int(value)] for value in indices]
        return output[self.dataset_spec.column_names]

    def to_payload(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("NFlowPreprocessor must be fitted before serialization.")
        return {
            "column_names": list(self.state.column_names),
            "numerical_columns": list(self.state.numerical_columns),
            "categorical_columns": list(self.state.categorical_columns),
            "target_columns": list(self.state.target_columns),
            "numerical_mean": list(self.state.numerical_mean),
            "numerical_scale": list(self.state.numerical_scale),
            "categorical_levels": {
                column: list(self.state.categorical_levels[column]) for column in self.state.categorical_columns
            },
        }

    @classmethod
    def from_payload(cls, dataset_spec: DatasetSpec, payload: Any) -> NFlowPreprocessor:
        if not isinstance(payload, dict):
            raise ValueError("NFlow checkpoint preprocessing state must be a JSON object.")
        instance = cls(dataset_spec)
        expected_roles = {
            "column_names": dataset_spec.column_names,
            "numerical_columns": instance.numeric_columns,
            "categorical_columns": instance.categorical_columns,
            "target_columns": dataset_spec.target_columns,
        }
        if any(payload.get(key) != value for key, value in expected_roles.items()):
            raise ValueError("NFlow checkpoint preprocessing roles differ from the current DatasetSpec.")
        means = payload.get("numerical_mean")
        scales = payload.get("numerical_scale")
        if (
            not isinstance(means, list)
            or not isinstance(scales, list)
            or len(means) != len(instance.numeric_columns)
            or len(scales) != len(instance.numeric_columns)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in means + scales)
        ):
            raise ValueError("NFlow checkpoint numerical preprocessing state is invalid.")
        mean_array = np.asarray(means, dtype=float)
        scale_array = np.asarray(scales, dtype=float)
        if not np.isfinite(mean_array).all() or not np.isfinite(scale_array).all() or bool((scale_array <= 0).any()):
            raise ValueError("NFlow checkpoint numerical scales must be positive and all statistics must be finite.")
        levels = payload.get("categorical_levels")
        if not isinstance(levels, dict) or set(levels) != set(instance.categorical_columns):
            raise ValueError("NFlow checkpoint categorical preprocessing state is invalid.")
        normalized_levels: dict[str, list[str]] = {}
        for column in instance.categorical_columns:
            values = levels[column]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"NFlow checkpoint category levels are invalid for {column!r}.")
            normalized_levels[column] = values
        instance.state.numerical_mean = [float(value) for value in means]
        instance.state.numerical_scale = [float(value) for value in scales]
        instance.state.categorical_levels = normalized_levels
        instance._fitted = True
        return instance


class BNAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "bn"
    upstream_dirname = "."
    checkpoint_filename = "model.bn.json"
    package_name = "pgmpy"
    package_version = "1.1.2"
    upstream_commit = "617cb48af678a7a471aad81d523ca95d2095430f"
    upstream_tree = "6c7adc00a479f540b2215889b1fac99a7b0b8a9c"
    checkpoint_schema_version = 1
    runtime_file_sha256 = {
        "__init__.py": "9101323f3d2a90053a876c27d8c59ce1c5b8aa2639f35a44069af0ef4028a093",
        "causal_discovery/HillClimbSearch.py": "5434e891f61a9c8c008555c4395f0cf3d239cb185250a905b6ad6f03278952a9",
        "causal_discovery/_base.py": "6212ae1ed122646b61e4e690518f2e5c6b97dbb6e3215223b64b17278aad6087",
        "structure_score/bic.py": "e696aef70d8c911fb8e2cfe1f1d752cb904aebcc482cbf6c9c077a1708c31ee8",
        "structure_score/log_likelihood.py": "b414f8df40b3e4ee04a55cc006f20a45c4c045ddff06b618fd343fcb5ba0cd9e",
        "parameter_estimator/discrete_bayesian.py": "c5b1ae4f69e8b755712680a352ffb9efa08831703e8fcc9d679983f09179fe98",
        "models/DiscreteBayesianNetwork.py": "3502fc66983b2b2075c0c6cddb217e598dc9845e0d494a98e10370fd421d275c",
        "sampling/Sampling.py": "55f3a9ccd20aa47a1f93ee1983ae0f222d04691fbb5a372148ffd49ebc9c2258",
        "factors/discrete/CPD.py": "b8a546710c5a5c779836a683c9dd600832508b00e5c1da4a1830c09536741812",
    }

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.checkpoint_filename)

    @staticmethod
    def _metadata_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")

    @staticmethod
    @contextlib.contextmanager
    def _scoped_numpy_seed(seed: int):
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)

    @staticmethod
    def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
            raise ValueError(f"BN {name} must be an integer >= {minimum}; observed {value!r}.")
        return int(value)

    @staticmethod
    def _nonnegative_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"BN {name} must be a non-negative integer; observed {value!r}.")
        return int(value)

    @staticmethod
    def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BN {name} must be numeric; observed {value!r}.") from exc
        if isinstance(value, bool) or not np.isfinite(parsed) or (positive and parsed <= 0):
            qualifier = "positive and finite" if positive else "finite"
            raise ValueError(f"BN {name} must be {qualifier}; observed {value!r}.")
        return parsed

    @classmethod
    def _validate_seed(cls, seed: int) -> int:
        parsed = cls._nonnegative_int("seed", seed)
        if parsed >= 2**32:
            raise ValueError("BN seed must be smaller than 2**32 for official pgmpy sampling.")
        return parsed

    def _recipe_params(self, spec: RunSpec, dataset_spec: DatasetSpec, source_rows: int) -> dict[str, Any]:
        num_bins = self._positive_int("num_bins", spec.extra.get("num_bins", 16), minimum=2)
        if num_bins > source_rows:
            raise ValueError(f"BN num_bins cannot exceed source rows ({source_rows}); observed {num_bins}.")
        max_indegree = self._nonnegative_int("max_indegree", spec.extra.get("max_indegree", 2))
        if max_indegree > max(0, len(dataset_spec.column_names) - 1):
            raise ValueError("BN max_indegree cannot exceed the number of other columns.")
        n_jobs = self._positive_int("n_jobs", spec.extra.get("n_jobs", 1))
        if n_jobs != 1:
            raise ValueError("BN requires n_jobs=1 for deterministic official-package parity.")
        scoring_method = spec.extra.get("scoring_method", "bic-d")
        prior_type = spec.extra.get("prior_type", "BDeu")
        if scoring_method != "bic-d":
            raise ValueError("BN currently supports only the validated official pgmpy scoring_method='bic-d'.")
        if prior_type != "BDeu":
            raise ValueError("BN currently supports only the validated official pgmpy prior_type='BDeu'.")
        return {
            "num_bins": num_bins,
            "quantile_method": "averaged_inverted_cdf",
            "subsample": None,
            "scoring_method": scoring_method,
            "return_type": "dag",
            "max_indegree": max_indegree,
            "max_iter": self._positive_int("max_iter", spec.extra.get("max_iter", 100)),
            "tabu_length": self._nonnegative_int("tabu_length", spec.extra.get("tabu_length", 100)),
            "epsilon": self._finite_float("epsilon", spec.extra.get("epsilon", 1e-4), positive=True),
            "prior_type": prior_type,
            "equivalent_sample_size": self._finite_float(
                "equivalent_sample_size", spec.extra.get("equivalent_sample_size", 5.0), positive=True
            ),
            "n_jobs": n_jobs,
        }

    def _import_official_api(self) -> dict[str, type]:
        try:
            observed_version = version(self.package_name)
        except PackageNotFoundError as exc:
            raise ImportError(
                "BN requires the checksum-audited official pgmpy package. Install the project with the 'bn' extra."
            ) from exc
        if observed_version != self.package_version:
            raise ImportError(f"BN requires {self.package_name}=={self.package_version}; observed {observed_version}.")
        package = importlib.import_module("pgmpy")
        package_root = Path(package.__file__).resolve().parent
        for relative_path, expected_sha256 in self.runtime_file_sha256.items():
            path = package_root / relative_path
            if path.is_symlink() or not path.is_file():
                raise ImportError(f"Official pgmpy runtime file is missing or unsafe: {path}")
            observed_sha256 = sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise ImportError(
                    "Installed pgmpy runtime source differs from the checksum-locked official 1.1.2 release: "
                    f"{relative_path} expected={expected_sha256}, observed={observed_sha256}."
                )
        api = {
            "HillClimbSearch": getattr(importlib.import_module("pgmpy.causal_discovery"), "HillClimbSearch"),
            "DiscreteBayesianNetwork": getattr(importlib.import_module("pgmpy.models"), "DiscreteBayesianNetwork"),
            "DiscreteBayesianEstimator": getattr(
                importlib.import_module("pgmpy.parameter_estimator"), "DiscreteBayesianEstimator"
            ),
            "BayesianModelSampling": getattr(importlib.import_module("pgmpy.sampling"), "BayesianModelSampling"),
            "TabularCPD": getattr(importlib.import_module("pgmpy.factors.discrete"), "TabularCPD"),
        }
        expected_modules = {
            "HillClimbSearch": "pgmpy.causal_discovery.HillClimbSearch",
            "DiscreteBayesianNetwork": "pgmpy.models.DiscreteBayesianNetwork",
            "DiscreteBayesianEstimator": "pgmpy.parameter_estimator.discrete_bayesian",
            "BayesianModelSampling": "pgmpy.sampling.Sampling",
            "TabularCPD": "pgmpy.factors.discrete.CPD",
        }
        for name, model_cls in api.items():
            if not isinstance(model_cls, type) or model_cls.__module__ != expected_modules[name]:
                raise ImportError(f"Installed pgmpy does not expose the expected official {name} class.")
        return api

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("bn requires dataset_spec.train_data_path")
        if len(dataset_spec.column_names) != len(set(dataset_spec.column_names)):
            raise ValueError("BN requires unique canonical column names.")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("BN supports exactly one target column in the standardized benchmark.")
        if dataset_spec.task_type not in {"classification", "regression"}:
            raise ValueError(f"BN does not support task type {dataset_spec.task_type!r}.")
        raw = pd.read_csv(dataset_spec.train_data_path)
        missing_columns = [column for column in dataset_spec.column_names if column not in raw.columns]
        if missing_columns:
            raise ValueError(f"BN training data is missing canonical columns: {missing_columns}")
        frame = raw[dataset_spec.column_names].copy()
        if len(frame) < 2:
            raise ValueError("BN requires at least two training rows.")
        missing_counts = frame.isna().sum()
        if bool(missing_counts.any()):
            observed = {str(column): int(count) for column, count in missing_counts.items() if count}
            raise ValueError(
                "BN does not accept missing values. Run the explicit train-fitted preprocessing module first; "
                f"observed: {observed}"
            )
        numerical_columns = list(dataset_spec.numerical_columns)
        categorical_columns = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "regression":
            numerical_columns.extend(dataset_spec.target_columns)
        else:
            categorical_columns.extend(dataset_spec.target_columns)
        numerical_columns = list(dict.fromkeys(numerical_columns))
        categorical_columns = list(dict.fromkeys(categorical_columns))
        overlap = sorted(set(numerical_columns) & set(categorical_columns))
        if overlap:
            raise ValueError(f"BN DatasetSpec assigns columns to conflicting type roles: {overlap}")
        undeclared = [
            column
            for column in dataset_spec.column_names
            if column not in numerical_columns and column not in categorical_columns
        ]
        if undeclared:
            raise ValueError(f"BN DatasetSpec leaves columns without a numerical/categorical role: {undeclared}")
        for column in numerical_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"BN numerical column {column!r} contains non-finite values.")
        return frame

    @staticmethod
    def _serialize_cpd(cpd: Any) -> dict[str, Any]:
        variables = [str(value) for value in cpd.variables]
        state_names = {variable: [str(value) for value in cpd.state_names[variable]] for variable in variables}
        values = np.asarray(cpd.get_values(), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Official pgmpy produced non-finite CPD values for {cpd.variable!r}.")
        if not np.allclose(values.sum(axis=0), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Official pgmpy produced a non-normalized CPD for {cpd.variable!r}.")
        return {
            "variable": str(cpd.variable),
            "variable_card": int(cpd.variable_card),
            "evidence": variables[1:],
            "evidence_card": [int(value) for value in cpd.cardinality[1:]],
            "state_names": state_names,
            "values": values.tolist(),
        }

    @classmethod
    def _checkpoint_payload(
        cls,
        model: Any,
        preprocessor: BNPreprocessor,
        dataset_spec: DatasetSpec,
        train_df: pd.DataFrame,
        discrete_df: pd.DataFrame,
        recipe: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        columns = list(dataset_spec.column_names)
        edges = sorted([[str(source), str(target)] for source, target in model.edges()])
        cpds = [cls._serialize_cpd(model.get_cpds(column)) for column in columns]
        return {
            "schema_version": cls.checkpoint_schema_version,
            "format": "pgmpy-discrete-bn-state",
            "package": {
                "name": cls.package_name,
                "version": cls.package_version,
                "upstream_commit": cls.upstream_commit,
                "upstream_tree": cls.upstream_tree,
            },
            "training": {
                "seed": seed,
                "source_rows": len(train_df),
                "canonical_frame_sha256": hashlib.sha256(train_df.to_csv(index=False).encode("utf-8")).hexdigest(),
                "discrete_frame_sha256": hashlib.sha256(discrete_df.to_csv(index=False).encode("utf-8")).hexdigest(),
                "task_type": dataset_spec.task_type,
                "numerical_columns": list(dataset_spec.numerical_columns),
                "categorical_columns": list(dataset_spec.categorical_columns),
                "target_columns": list(dataset_spec.target_columns),
                "recipe": recipe,
            },
            "preprocessing": preprocessor.to_payload(),
            "model": {"columns": columns, "edges": edges, "cpds": cpds},
            "privacy": {
                "contains_row_level_training_data": False,
                "code_executing_checkpoint": False,
                "retained_state": "discretization boundaries, category levels, graph edges, and conditional probabilities",
                "privacy_guarantee": "none",
                "trained_artifact_access_control_required": True,
            },
        }

    @classmethod
    def _restore_model(
        cls, api: dict[str, type], payload: Any, dataset_spec: DatasetSpec
    ) -> tuple[Any, BNPreprocessor]:
        if not isinstance(payload, dict):
            raise ValueError("BN checkpoint must be a JSON object.")
        if payload.get("schema_version") != cls.checkpoint_schema_version:
            raise ValueError("BN checkpoint schema version is not supported.")
        if payload.get("format") != "pgmpy-discrete-bn-state":
            raise ValueError("BN checkpoint format is not supported.")
        expected_package = {
            "name": cls.package_name,
            "version": cls.package_version,
            "upstream_commit": cls.upstream_commit,
            "upstream_tree": cls.upstream_tree,
        }
        if payload.get("package") != expected_package:
            raise ValueError("BN checkpoint package identity differs from the locked official release.")
        training = payload.get("training")
        model_state = payload.get("model")
        privacy = payload.get("privacy")
        if not isinstance(training, dict) or not isinstance(model_state, dict) or not isinstance(privacy, dict):
            raise ValueError("BN checkpoint is missing required state sections.")
        source_rows = cls._positive_int("checkpoint source_rows", training.get("source_rows"), minimum=2)
        training_seed = cls._validate_seed(training.get("seed"))
        expected_training_roles = {
            "task_type": dataset_spec.task_type,
            "numerical_columns": list(dataset_spec.numerical_columns),
            "categorical_columns": list(dataset_spec.categorical_columns),
            "target_columns": list(dataset_spec.target_columns),
        }
        if any(training.get(key) != value for key, value in expected_training_roles.items()):
            raise ValueError("BN checkpoint training roles differ from the current DatasetSpec.")
        for fingerprint_name in ("canonical_frame_sha256", "discrete_frame_sha256"):
            fingerprint = training.get(fingerprint_name)
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError(f"BN checkpoint {fingerprint_name} is not a canonical SHA-256 value.")
        recipe = training.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError("BN checkpoint training recipe is invalid.")
        recipe_spec = RunSpec(
            model=cls.model_name,
            dataset=dataset_spec.name,
            output_dir=Path.cwd(),
            seed=training_seed,
            extra=recipe,
        )
        expected_recipe = cls(Path.cwd())._recipe_params(recipe_spec, dataset_spec, source_rows)
        if recipe != expected_recipe:
            raise ValueError("BN checkpoint training recipe differs from the supported declared recipe.")
        if (
            privacy.get("contains_row_level_training_data") is not False
            or privacy.get("code_executing_checkpoint") is not False
            or privacy.get("privacy_guarantee") != "none"
            or privacy.get("trained_artifact_access_control_required") is not True
        ):
            raise ValueError("BN checkpoint does not satisfy the safe statistical-state contract.")
        columns = model_state.get("columns")
        if columns != dataset_spec.column_names:
            raise ValueError("BN checkpoint columns differ from the current DatasetSpec.")
        preprocessor = BNPreprocessor.from_payload(dataset_spec, payload.get("preprocessing"))
        edges = model_state.get("edges")
        cpds = model_state.get("cpds")
        if not isinstance(edges, list) or not isinstance(cpds, list) or len(cpds) != len(columns):
            raise ValueError("BN checkpoint graph or CPD collection is invalid.")
        normalized_edges: list[tuple[str, str]] = []
        for edge in edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or any(not isinstance(node, str) or node not in columns for node in edge)
                or edge[0] == edge[1]
            ):
                raise ValueError("BN checkpoint contains an invalid graph edge.")
            normalized_edges.append((edge[0], edge[1]))
        if normalized_edges != sorted(set(normalized_edges)):
            raise ValueError("BN checkpoint graph edges are duplicated or non-canonical.")
        model = api["DiscreteBayesianNetwork"]()
        model.add_nodes_from(columns)
        model.add_edges_from(normalized_edges)
        seen_variables: set[str] = set()
        restored_cpds = []
        for cpd_state in cpds:
            if not isinstance(cpd_state, dict):
                raise ValueError("BN checkpoint contains an invalid CPD entry.")
            variable = cpd_state.get("variable")
            evidence = cpd_state.get("evidence")
            evidence_card = cpd_state.get("evidence_card")
            state_names = cpd_state.get("state_names")
            values = cpd_state.get("values")
            if not isinstance(variable, str) or variable not in columns or variable in seen_variables:
                raise ValueError("BN checkpoint contains an invalid or duplicate CPD variable.")
            seen_variables.add(variable)
            if (
                not isinstance(evidence, list)
                or any(not isinstance(node, str) or node not in columns for node in evidence)
                or evidence != sorted(model.get_parents(variable))
                or not isinstance(evidence_card, list)
                or len(evidence_card) != len(evidence)
                or any(isinstance(card, bool) or not isinstance(card, int) or card < 1 for card in evidence_card)
                or not isinstance(state_names, dict)
                or set(state_names) != {variable, *evidence}
            ):
                raise ValueError(f"BN checkpoint CPD structure is invalid for {variable!r}.")
            variable_card = cls._positive_int("checkpoint variable_card", cpd_state.get("variable_card"))
            for node, states in state_names.items():
                if (
                    not isinstance(states, list)
                    or not states
                    or any(not isinstance(state, str) for state in states)
                    or len(states) != len(set(states))
                    or states != preprocessor.discrete_state_names[node]
                ):
                    raise ValueError(f"BN checkpoint state names are invalid for {node!r}.")
            if variable_card != len(state_names[variable]):
                raise ValueError(f"BN checkpoint variable cardinality is invalid for {variable!r}.")
            if evidence_card != [len(state_names[node]) for node in evidence]:
                raise ValueError(f"BN checkpoint evidence cardinalities are invalid for {variable!r}.")
            if (
                not isinstance(values, list)
                or any(not isinstance(row, list) for row in values)
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) for row in values for value in row
                )
            ):
                raise ValueError(f"BN checkpoint CPD values are not a numeric JSON matrix for {variable!r}.")
            value_array = np.asarray(values, dtype=float)
            expected_shape = (variable_card, int(np.prod(evidence_card, dtype=int)))
            if value_array.shape != expected_shape or not np.isfinite(value_array).all():
                raise ValueError(f"BN checkpoint CPD values are invalid for {variable!r}.")
            if not np.allclose(value_array.sum(axis=0), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(f"BN checkpoint CPD is not normalized for {variable!r}.")
            restored_cpds.append(
                api["TabularCPD"](
                    variable=variable,
                    variable_card=variable_card,
                    values=value_array.tolist(),
                    evidence=evidence or None,
                    evidence_card=evidence_card or None,
                    state_names=state_names,
                )
            )
        if seen_variables != set(columns):
            raise ValueError("BN checkpoint does not contain exactly one CPD per canonical column.")
        model.add_cpds(*restored_cpds)
        if model.check_model() is not True:
            raise ValueError("Restored official pgmpy model failed its consistency check.")
        return model, preprocessor

    @staticmethod
    def _regular_json_file(path: Path, description: str) -> Path:
        if path.is_symlink():
            raise PermissionError(f"Refusing to read a symlinked {description}: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"Expected a regular {description}: {resolved}")
        return resolved

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if spec.device != "cpu":
            raise ValueError("The validated pgmpy BN recipe is CPU-only; use device='cpu'.")
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        seed = self._validate_seed(spec.seed)
        recipe = self._recipe_params(spec, dataset_spec, len(train_df))
        api = self._import_official_api()
        preprocessor = BNPreprocessor(dataset_spec, num_bins=recipe["num_bins"])
        discrete_df = preprocessor.fit_transform(train_df)
        search = api["HillClimbSearch"](
            scoring_method=recipe["scoring_method"],
            return_type=recipe["return_type"],
            max_indegree=recipe["max_indegree"],
            max_iter=recipe["max_iter"],
            tabu_length=recipe["tabu_length"],
            epsilon=recipe["epsilon"],
            show_progress=False,
        )
        model_structure = search.fit(discrete_df).causal_graph_
        model = api["DiscreteBayesianNetwork"]()
        model.add_nodes_from(dataset_spec.column_names)
        model.add_edges_from(sorted(model_structure.edges()))
        estimator = api["DiscreteBayesianEstimator"](
            prior_type=recipe["prior_type"],
            equivalent_sample_size=recipe["equivalent_sample_size"],
            n_jobs=recipe["n_jobs"],
        )
        model.fit(discrete_df, estimator=estimator)
        if model.check_model() is not True:
            raise ValueError("Official pgmpy produced an inconsistent fitted Bayesian network.")

        checkpoint_path = self._resolve_checkpoint_path(spec)
        payload = self._checkpoint_payload(model, preprocessor, dataset_spec, train_df, discrete_df, recipe, seed)
        atomic_write_json(checkpoint_path, payload)
        atomic_write_json(
            self._metadata_path(checkpoint_path),
            {
                "schema_version": 1,
                "model": self.model_name,
                "package": self.package_name,
                "package_version": self.package_version,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "seed": seed,
                "source_rows": len(train_df),
                "columns": dataset_spec.column_names,
                "safe_json_checkpoint": True,
                "contains_row_level_training_data": False,
                "code_executing_checkpoint": False,
                "privacy_guarantee": "none",
                "trained_artifact_access_control_required": True,
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Saved official pgmpy {self.package_version} Bayesian-network state to {checkpoint_path}.",
                "The safe JSON checkpoint contains graph/CPD and preprocessing state, not executable pickle or rows.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if spec.device != "cpu":
            raise ValueError("The validated pgmpy BN recipe is CPU-only; use device='cpu'.")
        dataset_spec = self.resolve_dataset_spec(spec)
        seed = self._validate_seed(spec.seed)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        trusted_checkpoint = self._regular_json_file(checkpoint_path, "BN JSON checkpoint")
        metadata_path = self._regular_json_file(self._metadata_path(checkpoint_path), "BN checkpoint metadata")
        metadata = read_json(metadata_path)
        checkpoint_sha256 = sha256_file(trusted_checkpoint)
        if not isinstance(metadata, dict) or metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("BN checkpoint integrity verification failed.")
        api = self._import_official_api()
        payload = read_json(trusted_checkpoint)
        model, preprocessor = self._restore_model(api, payload, dataset_spec)
        expected_metadata = {
            "schema_version": 1,
            "model": self.model_name,
            "package": self.package_name,
            "package_version": self.package_version,
            "seed": payload["training"]["seed"],
            "source_rows": payload["training"]["source_rows"],
            "columns": dataset_spec.column_names,
            "safe_json_checkpoint": True,
            "contains_row_level_training_data": False,
            "code_executing_checkpoint": False,
            "privacy_guarantee": "none",
            "trained_artifact_access_control_required": True,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("BN checkpoint metadata differs from the safe statistical-state contract.")
        source_rows = payload["training"].get("source_rows")
        num_samples = (
            self._positive_int("checkpoint source_rows", source_rows)
            if spec.num_samples is None
            else self._positive_int("num_samples", spec.num_samples)
        )
        sampler = api["BayesianModelSampling"](model)
        with self._scoped_numpy_seed(seed):
            discrete_sample = sampler.forward_sample(size=num_samples, seed=seed, show_progress=False, n_jobs=1)
        if not isinstance(discrete_sample, pd.DataFrame):
            raise TypeError("Official pgmpy forward_sample did not return a DataFrame.")
        if set(discrete_sample.columns) != set(dataset_spec.column_names):
            raise ValueError("Official pgmpy forward_sample returned an invalid column set.")
        discrete_sample = discrete_sample[dataset_spec.column_names].copy()
        if len(discrete_sample) != num_samples or bool(discrete_sample.isna().any().any()):
            raise ValueError("Official pgmpy forward_sample returned an invalid row count or missing values.")
        sample_df = preprocessor.inverse_transform(discrete_sample, seed=seed)
        if len(sample_df) != num_samples or list(sample_df.columns) != dataset_spec.column_names:
            raise ValueError("BN inverse transformation violated the standardized row/column contract.")
        if bool(sample_df.isna().any().any()):
            raise ValueError("BN produced missing values; refusing to write an invalid sample.")
        numerical_columns = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numerical_columns.extend(dataset_spec.target_columns)
        if numerical_columns and not np.isfinite(sample_df[numerical_columns].to_numpy(dtype=float)).all():
            raise ValueError("BN produced non-finite numerical values.")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        atomic_write_json(
            spec.output_dir / "bn_sample_metadata.json",
            {
                "package": self.package_name,
                "package_version": self.package_version,
                "seed": seed,
                "requested_rows": num_samples,
                "checkpoint_path": str(trusted_checkpoint),
                "checkpoint_sha256": sha256_file(trusted_checkpoint),
                "discrete_sample_sha256": hashlib.sha256(
                    discrete_sample.to_csv(index=False).encode("utf-8")
                ).hexdigest(),
                "sample_path": str(sample_path),
                "sample_sha256": sha256_file(sample_path),
                "columns": dataset_spec.column_names,
            },
        )
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
    upstream_dirname = "."
    checkpoint_filename = "model.nflow.json"
    package_name = "nflows"
    package_version = "0.14"
    upstream_commit = "64b856c081e5f07521b32be99da262e8338fbfe8"
    upstream_tree = "83057958f8773e35044e3aa5c13ac9c06c4a3994"
    checkpoint_schema_version = 1
    max_checkpoint_bytes = 1024 * 1024
    max_weights_bytes = 1024 * 1024 * 1024
    runtime_file_sha256 = {
        "version.py": "30f8af2fc2d2e23d6dc061e36586e4843adda27833afbceff06e4a1cff0210a3",
        "flows/base.py": "f4ababbed0fac5afc4678f7df2fe804f8bc81c4a9168523e71040fda76480c62",
        "distributions/base.py": "914d98088ef44233388293a37f38260c3a7fdb40d749bfdd0725491655f86bab",
        "distributions/normal.py": "194d6e1c3b75d3e088a88b8dedbab3fb4c0c19f76fd03f994be95b420e53e62e",
        "transforms/base.py": "f302d6042a996681477cd2f9159bfb7736e2e1819d2d4369201ece37bc901330",
        "transforms/autoregressive.py": "eb52e2841c1cb5fd27c985979a98eb1e6ff8336b5bf638b82b6b651d1f1c55bf",
        "transforms/permutations.py": "774296ca27ce18bb73a501906d35166b3ebbb8b79494fe9ac24cb7a0be1f7305",
        "transforms/made.py": "5e621fa91b7fc0e8659a65f46f3fd6d00f2931cb01307de23d83e367dcf40480",
        "nn/nde/made.py": "3a91441c32f3d9e842297475822b54b7895877d79c2218f908f345fc1c06900e",
    }

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.checkpoint_filename)

    @staticmethod
    def _metadata_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")

    @staticmethod
    def _weights_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.stem}.weights.npz")

    @staticmethod
    def _positive_int(name: str, value: Any, *, maximum: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"NFlow {name} must be a positive integer; observed {value!r}.")
        parsed = int(value)
        if maximum is not None and parsed > maximum:
            raise ValueError(f"NFlow {name} must be <= {maximum}; observed {parsed}.")
        return parsed

    @staticmethod
    def _positive_float(name: str, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"NFlow {name} must be numeric; observed {value!r}.") from exc
        if isinstance(value, bool) or not np.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"NFlow {name} must be positive and finite; observed {value!r}.")
        return parsed

    @classmethod
    def _validate_seed(cls, seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0 or int(seed) >= 2**63:
            raise ValueError("NFlow seed must be an integer in [0, 2**63).")
        return int(seed)

    def _recipe_params(self, spec: RunSpec) -> dict[str, Any]:
        fixed_values = {
            "transform_order": "random-permutation-then-masked-affine-autoregressive",
            "base_distribution": "standard-normal",
            "context_features": None,
            "use_residual_blocks": True,
            "random_mask": False,
            "activation": "relu",
            "dropout_probability": 0.0,
            "use_batch_norm": False,
            "optimizer": "adam",
            "shuffle": True,
            "dtype": "float32",
        }
        for name, expected in fixed_values.items():
            if name in spec.extra and spec.extra[name] != expected:
                raise ValueError(f"NFlow supports only the validated {name}={expected!r} recipe.")
        num_threads = self._positive_int("num_threads", spec.extra.get("num_threads", 1), maximum=64)
        if num_threads != 1:
            raise ValueError("NFlow requires num_threads=1 for deterministic official-package parity.")
        return {
            "num_layers": self._positive_int("num_layers", spec.extra.get("num_layers", 4), maximum=64),
            "hidden_features": self._positive_int(
                "hidden_features", spec.extra.get("hidden_features", 64), maximum=16384
            ),
            "num_blocks": self._positive_int("num_blocks", spec.extra.get("num_blocks", 2), maximum=64),
            "learning_rate": self._positive_float("learning_rate", spec.extra.get("learning_rate", 1e-3)),
            "batch_size": self._positive_int("batch_size", spec.extra.get("batch_size", 512)),
            "epochs": self._positive_int("epochs", spec.extra.get("epochs", 10)),
            "num_threads": num_threads,
            **fixed_values,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
        }

    @staticmethod
    @contextlib.contextmanager
    def _scoped_torch_runtime(torch: Any, seed: int, num_threads: int):
        previous_state = torch.random.get_rng_state()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_threads)
            torch.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(previous_state)
            torch.set_num_threads(previous_threads)

    def _import_official_api(self) -> dict[str, Any]:
        try:
            observed_version = version(self.package_name)
        except PackageNotFoundError as exc:
            raise ImportError(
                "NFlow requires the checksum-audited official nflows package. "
                "Install the project with the 'nflow' extra."
            ) from exc
        if observed_version != self.package_version:
            raise ImportError(f"NFlow requires nflows=={self.package_version}; observed {observed_version}.")
        package = importlib.import_module("nflows")
        package_root = Path(package.__file__).resolve().parent
        for relative_path, expected_sha256 in self.runtime_file_sha256.items():
            path = package_root / relative_path
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
                raise ImportError(
                    "Installed nflows runtime differs from the checksum-locked official 0.14 release: "
                    f"{relative_path}."
                )
        torch = importlib.import_module("torch")
        api = {
            "torch": torch,
            "StandardNormal": getattr(importlib.import_module("nflows.distributions"), "StandardNormal"),
            "Flow": getattr(importlib.import_module("nflows.flows"), "Flow"),
            "CompositeTransform": getattr(importlib.import_module("nflows.transforms"), "CompositeTransform"),
            "RandomPermutation": getattr(importlib.import_module("nflows.transforms"), "RandomPermutation"),
            "MaskedAffineAutoregressiveTransform": getattr(
                importlib.import_module("nflows.transforms.autoregressive"),
                "MaskedAffineAutoregressiveTransform",
            ),
        }
        expected_modules = {
            "StandardNormal": "nflows.distributions.normal",
            "Flow": "nflows.flows.base",
            "CompositeTransform": "nflows.transforms.base",
            "RandomPermutation": "nflows.transforms.permutations",
            "MaskedAffineAutoregressiveTransform": "nflows.transforms.autoregressive",
        }
        for name, expected_module in expected_modules.items():
            if not isinstance(api[name], type) or api[name].__module__ != expected_module:
                raise ImportError(f"Installed nflows does not expose the expected official {name} class.")
        return api

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("nflow requires dataset_spec.train_data_path")
        if len(dataset_spec.column_names) != len(set(dataset_spec.column_names)):
            raise ValueError("NFlow requires unique canonical column names.")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("NFlow supports exactly one target column in the standardized benchmark.")
        if dataset_spec.task_type not in {"classification", "regression"}:
            raise ValueError(f"NFlow does not support task type {dataset_spec.task_type!r}.")
        raw = pd.read_csv(dataset_spec.train_data_path)
        missing_columns = [column for column in dataset_spec.column_names if column not in raw.columns]
        if missing_columns:
            raise ValueError(f"NFlow training data is missing canonical columns: {missing_columns}")
        frame = raw[dataset_spec.column_names].copy()
        if len(frame) < 2:
            raise ValueError("NFlow requires at least two training rows.")
        missing_counts = frame.isna().sum()
        if bool(missing_counts.any()):
            observed = {str(column): int(count) for column, count in missing_counts.items() if count}
            raise ValueError(
                "NFlow does not accept missing values. Run the explicit train-fitted preprocessing module first; "
                f"observed: {observed}"
            )
        numerical_columns = list(dataset_spec.numerical_columns)
        categorical_columns = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "regression":
            numerical_columns.extend(dataset_spec.target_columns)
        else:
            categorical_columns.extend(dataset_spec.target_columns)
        numerical_columns = list(dict.fromkeys(numerical_columns))
        categorical_columns = list(dict.fromkeys(categorical_columns))
        overlap = sorted(set(numerical_columns) & set(categorical_columns))
        if overlap:
            raise ValueError(f"NFlow DatasetSpec assigns columns to conflicting type roles: {overlap}")
        undeclared = [
            column
            for column in dataset_spec.column_names
            if column not in numerical_columns and column not in categorical_columns
        ]
        if undeclared:
            raise ValueError(f"NFlow DatasetSpec leaves columns without a numerical/categorical role: {undeclared}")
        for column in numerical_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"NFlow numerical column {column!r} contains non-finite values.")
        return frame

    def _build_flow(self, num_features: int, recipe: dict[str, Any], api: dict[str, Any]):
        transforms = []
        for _ in range(recipe["num_layers"]):
            transforms.append(api["RandomPermutation"](features=num_features, dim=1))
            transforms.append(
                api["MaskedAffineAutoregressiveTransform"](
                    features=num_features,
                    hidden_features=recipe["hidden_features"],
                    context_features=None,
                    num_blocks=recipe["num_blocks"],
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=api["torch"].nn.functional.relu,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                )
            )
        transform = api["CompositeTransform"](transforms)
        return api["Flow"](transform=transform, distribution=api["StandardNormal"](shape=[num_features]))

    @staticmethod
    def _state_arrays(flow: Any) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, tensor in sorted(flow.state_dict().items()):
            array = tensor.detach().cpu().numpy().copy()
            if array.dtype.hasobject:
                raise ValueError(f"NFlow state tensor {name!r} has an unsafe object dtype.")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"NFlow state tensor {name!r} contains non-finite values.")
            arrays[name] = array
        return arrays

    @staticmethod
    def _array_sha256(array: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    @classmethod
    def _write_weights(cls, path: Path, flow: Any) -> dict[str, Any]:
        arrays = cls._state_arrays(flow)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        payload = buffer.getvalue()
        if len(payload) > cls.max_weights_bytes:
            raise ValueError("NFlow weight archive exceeds the safe checkpoint size limit.")
        atomic_write_bytes(path, payload)
        tensors = [
            {
                "name": name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": cls._array_sha256(array),
            }
            for name, array in arrays.items()
        ]
        return {
            "filename": path.name,
            "format": "numpy-npz-no-pickle",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tensor_count": len(tensors),
            "tensors": tensors,
        }

    @classmethod
    def _regular_file(cls, path: Path, description: str, *, max_bytes: int) -> Path:
        if path.is_symlink():
            raise PermissionError(f"Refusing to read a symlinked {description}: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"Expected a regular {description}: {resolved}")
        if resolved.stat().st_size > max_bytes:
            raise ValueError(f"{description} exceeds the supported size limit.")
        return resolved

    @classmethod
    def _load_weights(cls, path: Path, weights: Any, flow: Any, torch: Any) -> dict[str, np.ndarray]:
        if not isinstance(weights, dict):
            raise ValueError("NFlow checkpoint weight manifest must be a JSON object.")
        expected_keys = {"filename", "format", "bytes", "sha256", "tensor_count", "tensors"}
        if set(weights) != expected_keys or weights.get("format") != "numpy-npz-no-pickle":
            raise ValueError("NFlow checkpoint weight manifest is not supported.")
        trusted = cls._regular_file(path, "NFlow NumPy weight archive", max_bytes=cls.max_weights_bytes)
        if (
            weights.get("filename") != trusted.name
            or weights.get("bytes") != trusted.stat().st_size
            or weights.get("sha256") != sha256_file(trusted)
        ):
            raise ValueError("NFlow weight archive identity differs from its checkpoint manifest.")
        tensor_manifest = weights.get("tensors")
        if not isinstance(tensor_manifest, list) or weights.get("tensor_count") != len(tensor_manifest):
            raise ValueError("NFlow checkpoint tensor manifest is invalid.")
        by_name: dict[str, dict[str, Any]] = {}
        for item in tensor_manifest:
            if not isinstance(item, dict) or set(item) != {"name", "dtype", "shape", "sha256"}:
                raise ValueError("NFlow checkpoint contains an invalid tensor manifest entry.")
            name = item.get("name")
            if not isinstance(name, str) or not name or name in by_name or "/" in name or "\\" in name:
                raise ValueError("NFlow checkpoint contains an unsafe or duplicate tensor name.")
            by_name[name] = item
        expected_state = flow.state_dict()
        if set(by_name) != set(expected_state):
            raise ValueError("NFlow checkpoint tensor names differ from the declared architecture.")
        with zipfile.ZipFile(trusted) as archive:
            infos = archive.infolist()
            expected_members = {f"{name}.npy" for name in by_name}
            observed_members = {info.filename for info in infos}
            if (
                len(infos) != len(observed_members)
                or observed_members != expected_members
                or any(info.is_dir() or info.flag_bits & 0x1 for info in infos)
                or sum(info.file_size for info in infos) > cls.max_weights_bytes
            ):
                raise ValueError("NFlow weight archive members are unsafe or differ from the tensor manifest.")
        restored: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        with np.load(trusted, allow_pickle=False) as archive:
            if set(archive.files) != set(by_name):
                raise ValueError("NFlow NumPy archive keys differ from the tensor manifest.")
            for name, expected_tensor in expected_state.items():
                array = np.asarray(archive[name])
                item = by_name[name]
                if (
                    array.dtype.hasobject
                    or str(array.dtype) != item.get("dtype")
                    or list(array.shape) != item.get("shape")
                    or cls._array_sha256(array) != item.get("sha256")
                    or tuple(array.shape) != tuple(expected_tensor.shape)
                    or str(array.dtype) != str(expected_tensor.detach().cpu().numpy().dtype)
                ):
                    raise ValueError(f"NFlow checkpoint tensor identity is invalid for {name!r}.")
                if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                    raise ValueError(f"NFlow checkpoint tensor {name!r} contains non-finite values.")
                arrays[name] = array.copy()
                restored[name] = torch.from_numpy(arrays[name])
        incompatible = flow.load_state_dict(restored, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("NFlow checkpoint tensors do not exactly match the declared architecture.")
        return arrays

    @classmethod
    def _checkpoint_payload(
        cls,
        dataset_spec: DatasetSpec,
        train_df: pd.DataFrame,
        preprocessor: NFlowPreprocessor,
        recipe: dict[str, Any],
        seed: int,
        losses: list[float],
        weights: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": cls.checkpoint_schema_version,
            "format": "nflows-maf-tabular-recipe-state",
            "package": {
                "name": cls.package_name,
                "version": cls.package_version,
                "upstream_commit": cls.upstream_commit,
                "upstream_tree": cls.upstream_tree,
            },
            "training": {
                "seed": seed,
                "source_rows": len(train_df),
                "canonical_frame_sha256": hashlib.sha256(train_df.to_csv(index=False).encode("utf-8")).hexdigest(),
                "task_type": dataset_spec.task_type,
                "numerical_columns": list(dataset_spec.numerical_columns),
                "categorical_columns": list(dataset_spec.categorical_columns),
                "target_columns": list(dataset_spec.target_columns),
                "recipe": recipe,
                "epoch_mean_negative_log_likelihood": losses,
            },
            "preprocessing": preprocessor.to_payload(),
            "weights": weights,
            "privacy": {
                "contains_row_level_training_data": False,
                "code_executing_checkpoint": False,
                "retained_state": "standardization statistics, category levels, and trained flow tensors",
                "privacy_guarantee": "none",
                "trained_artifact_access_control_required": True,
            },
        }

    def _restore_flow(
        self,
        payload: Any,
        checkpoint_path: Path,
        dataset_spec: DatasetSpec,
        api: dict[str, Any],
    ) -> tuple[Any, NFlowPreprocessor, dict[str, np.ndarray]]:
        if not isinstance(payload, dict):
            raise ValueError("NFlow checkpoint must be a JSON object.")
        if payload.get("schema_version") != self.checkpoint_schema_version:
            raise ValueError("NFlow checkpoint schema version is not supported.")
        if payload.get("format") != "nflows-maf-tabular-recipe-state":
            raise ValueError("NFlow checkpoint format is not supported.")
        expected_package = {
            "name": self.package_name,
            "version": self.package_version,
            "upstream_commit": self.upstream_commit,
            "upstream_tree": self.upstream_tree,
        }
        if payload.get("package") != expected_package:
            raise ValueError("NFlow checkpoint package identity differs from the locked official release.")
        training = payload.get("training")
        privacy = payload.get("privacy")
        if not isinstance(training, dict) or not isinstance(privacy, dict):
            raise ValueError("NFlow checkpoint is missing required state sections.")
        self._positive_int("checkpoint source_rows", training.get("source_rows"))
        training_seed = self._validate_seed(training.get("seed"))
        expected_roles = {
            "task_type": dataset_spec.task_type,
            "numerical_columns": list(dataset_spec.numerical_columns),
            "categorical_columns": list(dataset_spec.categorical_columns),
            "target_columns": list(dataset_spec.target_columns),
        }
        if any(training.get(key) != value for key, value in expected_roles.items()):
            raise ValueError("NFlow checkpoint training roles differ from the current DatasetSpec.")
        fingerprint = training.get("canonical_frame_sha256")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("NFlow checkpoint training-frame fingerprint is invalid.")
        recipe = training.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError("NFlow checkpoint training recipe is invalid.")
        recipe_spec = RunSpec(
            model=self.model_name,
            dataset=dataset_spec.name,
            output_dir=Path.cwd(),
            seed=training_seed,
            extra=recipe,
        )
        if recipe != self._recipe_params(recipe_spec):
            raise ValueError("NFlow checkpoint training recipe differs from the supported declared recipe.")
        losses = training.get("epoch_mean_negative_log_likelihood")
        if (
            not isinstance(losses, list)
            or len(losses) != recipe["epochs"]
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in losses)
            or not np.isfinite(np.asarray(losses, dtype=float)).all()
        ):
            raise ValueError("NFlow checkpoint training loss record is invalid.")
        if (
            privacy.get("contains_row_level_training_data") is not False
            or privacy.get("code_executing_checkpoint") is not False
            or privacy.get("privacy_guarantee") != "none"
            or privacy.get("trained_artifact_access_control_required") is not True
        ):
            raise ValueError("NFlow checkpoint does not satisfy the safe learned-state contract.")
        preprocessor = NFlowPreprocessor.from_payload(dataset_spec, payload.get("preprocessing"))
        num_features = len(preprocessor.numeric_columns) + len(preprocessor.categorical_columns)
        with self._scoped_torch_runtime(api["torch"], training_seed, recipe["num_threads"]):
            flow = self._build_flow(num_features, recipe, api)
        weights_path = self._weights_path(checkpoint_path)
        weights = payload.get("weights")
        if not isinstance(weights, dict) or weights.get("filename") != weights_path.name:
            raise ValueError("NFlow checkpoint weight filename is not canonical.")
        arrays = self._load_weights(weights_path, weights, flow, api["torch"])
        return flow, preprocessor, arrays

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if spec.device != "cpu":
            raise ValueError("The validated NFlow recipe is CPU-only; use device='cpu'.")
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        seed = self._validate_seed(spec.seed)
        recipe = self._recipe_params(spec)
        api = self._import_official_api()
        torch = api["torch"]
        preprocessor = NFlowPreprocessor(dataset_spec)
        train_array = preprocessor.fit_transform(train_df)
        losses: list[float] = []
        with self._scoped_torch_runtime(torch, seed, recipe["num_threads"]):
            flow = self._build_flow(train_array.shape[1], recipe, api)
            flow.train()
            optimizer = torch.optim.Adam(
                flow.parameters(),
                lr=recipe["learning_rate"],
                betas=tuple(recipe["adam_betas"]),
                eps=recipe["adam_eps"],
                weight_decay=recipe["weight_decay"],
                amsgrad=recipe["amsgrad"],
            )
            train_tensor = torch.tensor(train_array, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(train_tensor)
            generator = torch.Generator().manual_seed(seed)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=recipe["batch_size"],
                shuffle=recipe["shuffle"],
                generator=generator,
                num_workers=0,
                drop_last=False,
            )
            for _ in range(recipe["epochs"]):
                epoch_losses: list[float] = []
                for (batch,) in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = -flow.log_prob(batch).mean()
                    if not bool(torch.isfinite(loss).item()):
                        raise ValueError("Official nflows training produced a non-finite loss.")
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu().item()))
                losses.append(float(np.mean(epoch_losses)))

        checkpoint_path = self._resolve_checkpoint_path(spec)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = self._weights_path(checkpoint_path)
        weights = self._write_weights(weights_path, flow)
        payload = self._checkpoint_payload(dataset_spec, train_df, preprocessor, recipe, seed, losses, weights)
        atomic_write_json(checkpoint_path, payload)
        atomic_write_json(
            self._metadata_path(checkpoint_path),
            {
                "schema_version": 1,
                "model": self.model_name,
                "package": self.package_name,
                "package_version": self.package_version,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "weights_path": str(weights_path),
                "weights_sha256": weights["sha256"],
                "seed": seed,
                "source_rows": len(train_df),
                "columns": dataset_spec.column_names,
                "safe_json_and_numpy_checkpoint": True,
                "contains_row_level_training_data": False,
                "code_executing_checkpoint": False,
                "privacy_guarantee": "none",
                "trained_artifact_access_control_required": True,
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Saved official nflows {self.package_version} recipe state to {checkpoint_path}.",
                "The checkpoint uses validated JSON plus non-object NumPy tensors, never executable pickle.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if spec.device != "cpu":
            raise ValueError("The validated NFlow recipe is CPU-only; use device='cpu'.")
        dataset_spec = self.resolve_dataset_spec(spec)
        seed = self._validate_seed(spec.seed)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        trusted_checkpoint = self._regular_file(
            checkpoint_path, "NFlow JSON checkpoint", max_bytes=self.max_checkpoint_bytes
        )
        metadata_path = self._regular_file(
            self._metadata_path(checkpoint_path),
            "NFlow checkpoint metadata",
            max_bytes=self.max_checkpoint_bytes,
        )
        payload = read_json(trusted_checkpoint)
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict) or metadata.get("checkpoint_sha256") != sha256_file(trusted_checkpoint):
            raise ValueError("NFlow checkpoint integrity verification failed.")
        api = self._import_official_api()
        torch = api["torch"]
        flow, preprocessor, _ = self._restore_flow(payload, trusted_checkpoint, dataset_spec, api)
        expected_metadata = {
            "schema_version": 1,
            "model": self.model_name,
            "package": self.package_name,
            "package_version": self.package_version,
            "weights_sha256": payload["weights"]["sha256"],
            "seed": payload["training"]["seed"],
            "source_rows": payload["training"]["source_rows"],
            "columns": dataset_spec.column_names,
            "safe_json_and_numpy_checkpoint": True,
            "contains_row_level_training_data": False,
            "code_executing_checkpoint": False,
            "privacy_guarantee": "none",
            "trained_artifact_access_control_required": True,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("NFlow checkpoint metadata differs from the safe learned-state contract.")
        num_samples = (
            self._positive_int("checkpoint source_rows", payload["training"]["source_rows"])
            if spec.num_samples is None
            else self._positive_int("num_samples", spec.num_samples)
        )
        flow.eval()
        with self._scoped_torch_runtime(torch, seed, payload["training"]["recipe"]["num_threads"]):
            with torch.no_grad():
                samples = flow.sample(num_samples).detach().cpu().numpy()
        if samples.shape != (num_samples, len(dataset_spec.column_names)) or not np.isfinite(samples).all():
            raise ValueError("Official nflows sampling returned an invalid shape or non-finite values.")
        sample_df = preprocessor.inverse_transform(samples)
        if len(sample_df) != num_samples or list(sample_df.columns) != dataset_spec.column_names:
            raise ValueError("NFlow inverse transformation violated the standardized row/column contract.")
        if bool(sample_df.isna().any().any()):
            raise ValueError("NFlow produced missing values; refusing to write an invalid sample.")
        numerical_columns = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numerical_columns.extend(dataset_spec.target_columns)
        if numerical_columns and not np.isfinite(sample_df[numerical_columns].to_numpy(dtype=float)).all():
            raise ValueError("NFlow produced non-finite numerical values.")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        atomic_write_json(
            spec.output_dir / "nflow_sample_metadata.json",
            {
                "package": self.package_name,
                "package_version": self.package_version,
                "seed": seed,
                "requested_rows": num_samples,
                "checkpoint_path": str(trusted_checkpoint),
                "checkpoint_sha256": sha256_file(trusted_checkpoint),
                "weights_path": str(self._weights_path(trusted_checkpoint)),
                "weights_sha256": payload["weights"]["sha256"],
                "raw_sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                "sample_path": str(sample_path),
                "sample_sha256": sha256_file(sample_path),
                "columns": dataset_spec.column_names,
            },
        )
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
