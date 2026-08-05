from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import pickle
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import (
    SampleFileEvaluatorMixin,
    disable_torchvision_for_transformers,
    temporary_sys_path,
)
from standardized_tabular_diffusion.models.base import BaseModelAdapter


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


def _tabpfn_cache_env() -> dict[str, str]:
    cache_root = Path(tempfile.gettempdir()) / "standardized-tabular-diffusion" / "tabpfn"
    cache_root.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(cache_root),
        "XDG_CACHE_HOME": str(cache_root),
        "MPLCONFIGDIR": str(cache_root / "mpl"),
    }


class GReaTAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
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
        with disable_torchvision_for_transformers():
            with temporary_sys_path(self.upstream_root):
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
        return read_json(metadata_path)

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
        preserve_column_order = bool(spec.extra.get("preserve_column_order", False))
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
        atomic_write_json(self._metadata_path(model_root), metadata)
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
        trusted_model_root = self._validate_trusted_executable_artifact(
            spec,
            model_root,
            format_name="GReaT model directory",
            allow_directory=True,
        )
        model = GReaT.load_from_dir(str(trusted_model_root))
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
        self._write_dataframe_csv(sample_df, sample_path)
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


class ARFAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "arf"
    upstream_dirname = "."
    checkpoint_filename = "model.arf.json"
    package_name = "arfpy"
    package_version = "0.1.1"
    upstream_commit = "6f737baaaa589f7ac3ff59f0d739ce04b0f1381c"
    upstream_tree = "68b6fc5d28578a5c21bef560bd28f4c0d2d6401c"
    checkpoint_schema_version = 1
    runtime_file_sha256 = {
        "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "arf.py": "860f49dd232b78eba12d7a56ed88c9c6c814fae6938b9dbf047668874b368898",
        "utils.py": "391032b116763ed1da0a539a9ae34bed69fbb477026034bbb6f33d44c4ad56f4",
    }

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        return spec.output_dir / self.checkpoint_filename

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
    def _positive_int(name: str, value: Any) -> int:
        parsed = int(value)
        if isinstance(value, bool) or parsed < 1:
            raise ValueError(f"ARF {name} must be a positive integer; observed {value!r}.")
        return parsed

    @staticmethod
    def _nonnegative_int(name: str, value: Any) -> int:
        parsed = int(value)
        if isinstance(value, bool) or parsed < 0:
            raise ValueError(f"ARF {name} must be a non-negative integer; observed {value!r}.")
        return parsed

    @staticmethod
    def _finite_float(name: str, value: Any) -> float:
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"ARF {name} must be finite; observed {value!r}.")
        return parsed

    @staticmethod
    def _boolean(name: str, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"ARF {name} must be a boolean; observed {value!r}.")
        return value

    def _training_params(self, spec: RunSpec) -> dict[str, Any]:
        delta = self._finite_float("delta", spec.extra.get("delta", 0.0))
        if not 0 <= delta <= 0.5:
            raise ValueError(f"ARF delta must lie in [0, 0.5]; observed {delta}.")
        n_jobs = int(spec.extra.get("n_jobs", 1))
        if isinstance(spec.extra.get("n_jobs", 1), bool) or n_jobs == 0:
            raise ValueError(f"ARF n_jobs must be a non-zero integer; observed {n_jobs!r}.")
        return {
            "num_trees": self._positive_int("num_trees", spec.extra.get("num_trees", 30)),
            "delta": delta,
            "max_iters": self._nonnegative_int("max_iters", spec.extra.get("max_iters", 10)),
            "early_stop": self._boolean("early_stop", spec.extra.get("early_stop", True)),
            "verbose": self._boolean("verbose", spec.extra.get("verbose", False)),
            "min_node_size": self._positive_int(
                "min_node_size", spec.extra.get("min_node_size", 5)
            ),
            "random_state": spec.seed,
            "n_jobs": n_jobs,
        }

    def _forde_params(self, spec: RunSpec) -> dict[str, Any]:
        dist = spec.extra.get("dist", "truncnorm")
        if dist != "truncnorm":
            raise ValueError("arfpy 0.1.1 only implements dist='truncnorm'.")
        oob = self._boolean("oob", spec.extra.get("oob", False))
        if oob:
            raise ValueError(
                "arfpy 0.1.1 has a broken oob=True FORDE path that references unavailable state. "
                "This adapter refuses to patch the official algorithm silently; use oob=false."
            )
        alpha = self._finite_float("alpha", spec.extra.get("alpha", 0.0))
        if alpha < 0:
            raise ValueError(f"ARF alpha must be non-negative; observed {alpha}.")
        return {"dist": dist, "oob": oob, "alpha": alpha}

    def _import_model_cls(self):
        try:
            observed_version = version(self.package_name)
        except PackageNotFoundError as exc:
            raise ImportError(
                "ARF requires the checksum-audited official arfpy package. "
                "Install the project with the 'arf' extra."
            ) from exc
        if observed_version != self.package_version:
            raise ImportError(
                "ARF requires the exact validated official package version "
                f"{self.package_name}=={self.package_version}; observed {observed_version}."
            )
        module = importlib.import_module("arfpy.arf")
        package_root = Path(module.__file__).resolve().parent
        for filename, expected_sha256 in self.runtime_file_sha256.items():
            path = package_root / filename
            if path.is_symlink() or not path.is_file():
                raise ImportError(f"Official arfpy runtime file is missing or unsafe: {path}")
            observed_sha256 = sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise ImportError(
                    "Installed arfpy runtime source differs from the checksum-locked official 0.1.1 release: "
                    f"{filename} expected={expected_sha256}, observed={observed_sha256}."
                )
        model_cls = getattr(module, "arf", None)
        if not isinstance(model_cls, type) or model_cls.__module__ != "arfpy.arf":
            raise ImportError("Installed arfpy package does not expose the official arfpy.arf.arf class.")
        return model_cls

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("arf requires dataset_spec.train_data_path")
        if len(dataset_spec.column_names) != len(set(dataset_spec.column_names)):
            raise ValueError("ARF requires unique canonical column names.")
        raw = pd.read_csv(dataset_spec.train_data_path)
        missing_columns = [column for column in dataset_spec.column_names if column not in raw.columns]
        if missing_columns:
            raise ValueError(f"ARF training data is missing canonical columns: {missing_columns}")
        frame = raw[dataset_spec.column_names].copy()
        missing_counts = frame.isna().sum()
        if bool(missing_counts.any()):
            observed = {str(column): int(count) for column, count in missing_counts.items() if count}
            raise ValueError(
                "ARF does not accept missing values in this benchmark. Run the explicit train-fitted "
                f"preprocessing module first; observed: {observed}"
            )

        categorical_columns = list(dataset_spec.categorical_columns)
        numerical_columns = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "classification":
            categorical_columns.extend(dataset_spec.target_columns)
        elif dataset_spec.task_type == "regression":
            numerical_columns.extend(dataset_spec.target_columns)
        else:
            raise ValueError(f"ARF does not support task type {dataset_spec.task_type!r}.")
        categorical_columns = list(dict.fromkeys(categorical_columns))
        numerical_columns = list(dict.fromkeys(numerical_columns))
        overlap = sorted(set(categorical_columns) & set(numerical_columns))
        if overlap:
            raise ValueError(f"ARF DatasetSpec assigns columns to conflicting type roles: {overlap}")
        undeclared = [
            column
            for column in dataset_spec.column_names
            if column not in categorical_columns and column not in numerical_columns
        ]
        if undeclared:
            raise ValueError(f"ARF DatasetSpec leaves columns without a numerical/categorical role: {undeclared}")
        for column in numerical_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"ARF numerical column {column!r} contains non-finite values.")
        for column in categorical_columns:
            frame[column] = frame[column].astype("category")
        return frame

    @classmethod
    def _encode_value(cls, value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if np.isnan(value):
                return {"__arf_float__": "nan"}
            if np.isposinf(value):
                return {"__arf_float__": "+inf"}
            if np.isneginf(value):
                return {"__arf_float__": "-inf"}
            return value
        raise TypeError(f"ARF checkpoint cannot safely encode value of type {type(value).__name__}.")

    @classmethod
    def _decode_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) != {"__arf_float__"}:
                raise ValueError("ARF checkpoint contains an unknown tagged value.")
            labels = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}
            label = value["__arf_float__"]
            if label not in labels:
                raise ValueError(f"ARF checkpoint contains an invalid float tag: {label!r}")
            return labels[label]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError(f"ARF checkpoint contains an unsupported JSON value: {type(value).__name__}")

    @classmethod
    def _encode_frame(cls, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "data": [[cls._encode_value(value) for value in row] for row in frame.itertuples(index=False, name=None)],
        }

    @classmethod
    def _decode_frame(cls, payload: Any) -> pd.DataFrame:
        if not isinstance(payload, dict) or set(payload) != {"columns", "dtypes", "data"}:
            raise ValueError("ARF checkpoint contains an invalid DataFrame payload.")
        columns = payload["columns"]
        dtypes = payload["dtypes"]
        data = payload["data"]
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) for column in columns)
            or len(columns) != len(set(columns))
            or not isinstance(dtypes, list)
            or len(dtypes) != len(columns)
            or any(not isinstance(dtype, str) for dtype in dtypes)
            or not isinstance(data, list)
        ):
            raise ValueError("ARF checkpoint DataFrame schema is invalid.")
        decoded_rows = []
        for row in data:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError("ARF checkpoint DataFrame row width is invalid.")
            decoded_rows.append([cls._decode_value(value) for value in row])
        frame = pd.DataFrame(decoded_rows, columns=columns)
        for column, dtype in zip(columns, dtypes, strict=True):
            if dtype.startswith(("int", "uint", "float")):
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(dtype)
            elif dtype == "bool":
                frame[column] = frame[column].astype(bool)
            elif dtype not in {"object", "string"}:
                raise ValueError(f"ARF checkpoint declares unsupported DataFrame dtype {dtype!r}.")
        return frame

    @classmethod
    def _checkpoint_payload(
        cls,
        model: Any,
        dataset_spec: DatasetSpec,
        train_df: pd.DataFrame,
        training_params: dict[str, Any],
        forde_params: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        columns = list(model.orig_colnames)
        factor_columns = [column for column in columns if bool(model.factor_cols[column])]
        object_columns = [column for column in columns if bool(model.object_cols[column])]
        levels = {
            column: [cls._encode_value(value) for value in model.levels[column].tolist()]
            for column in factor_columns
        }
        return {
            "schema_version": cls.checkpoint_schema_version,
            "format": "arfpy-forge-state",
            "package": {
                "name": cls.package_name,
                "version": cls.package_version,
                "upstream_commit": cls.upstream_commit,
                "upstream_tree": cls.upstream_tree,
            },
            "training": {
                "seed": seed,
                "source_rows": len(train_df),
                "canonical_frame_sha256": hashlib.sha256(
                    train_df.to_csv(index=False).encode("utf-8")
                ).hexdigest(),
                "task_type": dataset_spec.task_type,
                "numerical_columns": list(dataset_spec.numerical_columns),
                "categorical_columns": list(dataset_spec.categorical_columns),
                "target_columns": list(dataset_spec.target_columns),
                "training_params": training_params,
                "forde_params": forde_params,
                "adversarial_oob_accuracy": [cls._encode_value(float(value)) for value in model.acc],
            },
            "model": {
                "p": int(model.p),
                "num_trees": int(model.num_trees),
                "orig_colnames": columns,
                "factor_columns": factor_columns,
                "object_columns": object_columns,
                "levels": levels,
                "dist": str(model.dist),
                "bnds": cls._encode_frame(model.bnds),
                "params": cls._encode_frame(model.params),
                "class_probs": cls._encode_frame(model.class_probs),
            },
            "privacy": {
                "contains_row_level_training_data": False,
                "contains_random_forest": False,
                "retained_state": "FORGE density parameters, category levels, and leaf coverage only",
                "privacy_guarantee": "none",
                "trained_artifact_access_control_required": True,
            },
        }

    @classmethod
    def _restore_model(cls, model_cls: type, payload: Any, dataset_spec: DatasetSpec) -> Any:
        if not isinstance(payload, dict):
            raise ValueError("ARF checkpoint must be a JSON object.")
        if payload.get("schema_version") != cls.checkpoint_schema_version or payload.get("format") != "arfpy-forge-state":
            raise ValueError("ARF checkpoint schema or format is not supported.")
        package = payload.get("package")
        expected_package = {
            "name": cls.package_name,
            "version": cls.package_version,
            "upstream_commit": cls.upstream_commit,
            "upstream_tree": cls.upstream_tree,
        }
        if package != expected_package:
            raise ValueError("ARF checkpoint package identity differs from the locked official release.")
        training = payload.get("training")
        model_state = payload.get("model")
        privacy = payload.get("privacy")
        if not isinstance(training, dict) or not isinstance(model_state, dict) or not isinstance(privacy, dict):
            raise ValueError("ARF checkpoint is missing required state sections.")
        if privacy.get("contains_row_level_training_data") is not False or privacy.get("contains_random_forest") is not False:
            raise ValueError("ARF checkpoint does not satisfy the safe sanitized-state contract.")
        columns = model_state.get("orig_colnames")
        if columns != dataset_spec.column_names:
            raise ValueError(
                "ARF checkpoint columns differ from the current DatasetSpec: "
                f"checkpoint={columns}, dataset={dataset_spec.column_names}"
            )
        factor_columns = model_state.get("factor_columns")
        object_columns = model_state.get("object_columns")
        levels = model_state.get("levels")
        if (
            not isinstance(factor_columns, list)
            or any(column not in columns for column in factor_columns)
            or not isinstance(object_columns, list)
            or any(column not in columns for column in object_columns)
            or not isinstance(levels, dict)
            or set(levels) != set(factor_columns)
        ):
            raise ValueError("ARF checkpoint categorical schema is invalid.")
        model = model_cls.__new__(model_cls)
        model.p = int(model_state.get("p"))
        if model.p != len(columns):
            raise ValueError("ARF checkpoint feature count does not match its columns.")
        model.num_trees = cls._positive_int("checkpoint num_trees", model_state.get("num_trees"))
        model.orig_colnames = list(columns)
        model.factor_cols = pd.Series(
            [column in factor_columns for column in columns], index=columns, dtype=bool
        )
        model.object_cols = pd.Series(
            [column in object_columns for column in columns], index=columns, dtype=bool
        )
        model.levels = {
            column: pd.Index([cls._decode_value(value) for value in levels[column]])
            for column in factor_columns
        }
        model.dist = model_state.get("dist")
        if model.dist != "truncnorm":
            raise ValueError("ARF checkpoint requests an unsupported density distribution.")
        model.bnds = cls._decode_frame(model_state.get("bnds"))
        model.params = cls._decode_frame(model_state.get("params"))
        model.class_probs = cls._decode_frame(model_state.get("class_probs"))
        return model

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
            raise ValueError("Official arfpy 0.1.1 is CPU-only; ARF requires device='cpu'.")
        dataset_spec = self.resolve_dataset_spec(spec)
        train_df = self._load_training_frame(dataset_spec)
        model_cls = self._import_model_cls()
        training_params = self._training_params(spec)
        forde_params = self._forde_params(spec)
        with self._scoped_numpy_seed(spec.seed):
            model = model_cls(train_df.copy(), **training_params)
            model.forde(**forde_params)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        payload = self._checkpoint_payload(
            model,
            dataset_spec,
            train_df,
            training_params,
            forde_params,
            spec.seed,
        )
        atomic_write_json(checkpoint_path, payload)
        metadata_path = self._metadata_path(checkpoint_path)
        atomic_write_json(
            metadata_path,
            {
                "schema_version": 1,
                "model": self.model_name,
                "package": self.package_name,
                "package_version": self.package_version,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "seed": spec.seed,
                "source_rows": len(train_df),
                "columns": dataset_spec.column_names,
                "safe_json_checkpoint": True,
                "contains_row_level_training_data": False,
                "contains_random_forest": False,
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
                f"Saved official arfpy {self.package_version} FORGE state to {checkpoint_path}.",
                "The safe JSON checkpoint omits the fitted forest and row-level training data.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if spec.device != "cpu":
            raise ValueError("Official arfpy 0.1.1 is CPU-only; ARF requires device='cpu'.")
        dataset_spec = self.resolve_dataset_spec(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        trusted_checkpoint = self._regular_json_file(checkpoint_path, "ARF JSON checkpoint")
        metadata_path = self._regular_json_file(
            self._metadata_path(checkpoint_path), "ARF checkpoint metadata"
        )
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict) or metadata.get("checkpoint_sha256") != sha256_file(trusted_checkpoint):
            raise ValueError("ARF checkpoint integrity verification failed.")
        model_cls = self._import_model_cls()
        payload = read_json(trusted_checkpoint)
        model = self._restore_model(model_cls, payload, dataset_spec)
        source_rows = payload["training"].get("source_rows")
        num_samples = (
            self._positive_int("checkpoint source_rows", source_rows)
            if spec.num_samples is None
            else self._positive_int("num_samples", spec.num_samples)
        )
        with self._scoped_numpy_seed(spec.seed):
            sample_df = model.forge(num_samples)
        if not isinstance(sample_df, pd.DataFrame):
            raise TypeError(f"Official ARF forge returned {type(sample_df).__name__}, expected DataFrame.")
        if list(sample_df.columns) != dataset_spec.column_names:
            raise ValueError(
                "Official ARF forge returned non-canonical columns: "
                f"observed={list(sample_df.columns)}, expected={dataset_spec.column_names}"
            )
        sample_df = sample_df[dataset_spec.column_names].copy()
        if len(sample_df) != num_samples:
            raise ValueError(f"Official ARF forge returned {len(sample_df)} rows for a request of {num_samples}.")
        if bool(sample_df.isna().any().any()):
            raise ValueError("Official ARF forge produced missing values; refusing to write an invalid sample.")
        numerical = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numerical.extend(dataset_spec.target_columns)
        if numerical and not np.isfinite(sample_df[numerical].to_numpy(dtype=float)).all():
            raise ValueError("Official ARF forge produced non-finite numerical values.")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        atomic_write_json(
            spec.output_dir / "arf_sample_metadata.json",
            {
                "package": self.package_name,
                "package_version": self.package_version,
                "seed": spec.seed,
                "requested_rows": num_samples,
                "checkpoint_path": str(trusted_checkpoint),
                "checkpoint_sha256": sha256_file(trusted_checkpoint),
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
        self.feature_columns = [
            column for column in dataset_spec.column_names if column not in dataset_spec.target_columns
        ]
        self.numeric_columns = [column for column in dataset_spec.numerical_columns if column in self.feature_columns]
        self.categorical_columns = [
            column for column in dataset_spec.categorical_columns if column in self.feature_columns
        ]
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


class TabEBMAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
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
    def _add_surrogate_negative_samples(
        X: np.ndarray,
        distance_negative_class: float,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_features = X.shape[1]
        if num_features == 0:
            raise ValueError("TabEBM requires at least one feature column.")
        if not np.isfinite(distance_negative_class) or distance_negative_class <= 0:
            raise ValueError("distance_negative_class must be a finite positive number.")
        if num_features == 1:
            surrogate_negatives = np.array(
                [[-distance_negative_class], [distance_negative_class]],
                dtype=X.dtype,
            )
        elif num_features == 2:
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
            point = rng.choice([-distance_negative_class, distance_negative_class], size=num_features)
            adjacent_point = point.copy()
            adjacent_point[int(rng.integers(0, num_features))] *= -1
            surrogate_negatives = np.stack([point, -point, adjacent_point, -adjacent_point]).astype(
                X.dtype,
                copy=False,
            )
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
        if X.shape[1] == 0:
            raise ValueError("TabEBM requires at least one feature column.")
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
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("TabEBM sampling requires the optional PyTorch runtime.") from exc
        trusted_checkpoint = self._validate_trusted_executable_artifact(spec, checkpoint_path, format_name="pickle")
        with trusted_checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        state: TabEBMState = payload["state"]
        preprocessor: TabEBMPreprocessor = payload["preprocessor"]
        num_samples = spec.num_samples or len(state.feature_matrix)
        class_counts = self._class_sample_counts(state.target_values, num_samples)
        rng = np.random.default_rng(spec.seed)
        sampled_blocks: list[np.ndarray] = []
        sampled_targets: list[np.ndarray] = []

        with _temporary_env(_tabpfn_cache_env()), torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            try:
                for class_id, class_count in class_counts.items():
                    if class_count <= 0:
                        continue
                    class_features = state.feature_matrix[state.target_values == class_id]
                    model = self._build_model(spec)
                    X_ebm, y_ebm = self._add_surrogate_negative_samples(
                        class_features,
                        distance_negative_class=float(
                            spec.extra.get("distance_negative_class", state.sgld_defaults["distance_negative_class"])
                        ),
                        rng=rng,
                    )
                    X_ebm_tensor = torch.tensor(X_ebm, dtype=torch.float32)
                    y_ebm_tensor = torch.tensor(y_ebm, dtype=torch.long)
                    model.fit_with_differentiable_input(X_ebm_tensor, y_ebm_tensor)

                    start_idx = rng.integers(0, len(class_features), size=class_count)
                    X_sgld = torch.tensor(class_features[start_idx], dtype=torch.float32)
                    noise_std = float(
                        spec.extra.get("starting_point_noise_std", state.sgld_defaults["starting_point_noise_std"])
                    )
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
                                (X_sgld - step_size * X_sgld.grad + sgld_noise_std * torch.randn_like(X_sgld))
                                .detach()
                                .requires_grad_(True)
                            )

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


# Backward-compatible imports for callers that used this historical aggregate module.
# The active implementations live in dedicated, provenance-audited modules.
from standardized_tabular_diffusion.models.great import GReaTAdapter as GReaTAdapter  # noqa: E402,F811
from standardized_tabular_diffusion.models.tabebm import TabEBMAdapter as TabEBMAdapter  # noqa: E402,F811
