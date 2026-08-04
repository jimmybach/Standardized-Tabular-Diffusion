from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import (
    default_source_path,
    validate_upstream_source,
)


class GoggleAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    """Adapter for the checksum-locked method-author Goggle implementation."""

    model_name = "goggle"
    upstream_dirname = ".cache/upstream-sources/goggle/1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0"
    upstream_commit = "1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0"
    checkpoint_filename = "model.pt"

    _STANDARD_EXTRA_KEYS = {"action_extras", "config", "dataset_spec", "evaluation", "tags"}
    _TRAIN_KEYS = {
        "alpha",
        "batch_size",
        "beta",
        "decoder_arch",
        "decoder_dim",
        "decoder_l",
        "encoder_dim",
        "encoder_l",
        "epochs",
        "graph_prior",
        "het_encoding",
        "iter_opt",
        "learning_rate",
        "logging",
        "num_threads",
        "patience",
        "prior_mask",
        "source_dir",
        "threshold",
        "weight_decay",
    }
    _SAMPLE_KEYS = {"allow_unsafe_external_checkpoint", "num_threads", "source_dir"}
    _TRAIN_DEFAULTS: dict[str, Any] = {
        "encoder_dim": 64,
        "encoder_l": 2,
        "het_encoding": True,
        "decoder_dim": 64,
        "decoder_l": 2,
        "threshold": 0.1,
        "decoder_arch": "gcn",
        "graph_prior": None,
        "prior_mask": None,
        "alpha": 0.1,
        "beta": 0.1,
        "iter_opt": True,
        "learning_rate": 0.005,
        "weight_decay": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "patience": 50,
        "logging": 100,
        "num_threads": 1,
    }

    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root)
        self.upstream_root = default_source_path(repo_root, self.model_name)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Goggle {name} must be an integer greater than or equal to {minimum}.")
        return value

    @staticmethod
    def _finite_float(name: str, value: Any, *, minimum: float | None = None) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Goggle {name} must be a finite number.")
        result = float(value)
        if not math.isfinite(result) or (minimum is not None and result < minimum):
            qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
            raise ValueError(f"Goggle {name} must be {qualifier}.")
        return result

    def _validate_extra(self, spec: RunSpec, *, action: str) -> None:
        action_keys = self._TRAIN_KEYS if action == "train" else self._SAMPLE_KEYS
        unknown = sorted(set(spec.extra) - self._STANDARD_EXTRA_KEYS - action_keys)
        if unknown:
            raise ValueError(f"Unsupported Goggle {action} controls: {', '.join(unknown)}")

    def _source_root(self, spec: RunSpec) -> Path:
        configured = spec.extra.get("source_dir") or os.environ.get("STANDARDIZED_TABULAR_DIFFUSION_GOGGLE_SOURCE")
        source = Path(configured) if configured is not None else default_source_path(self.repo_root, self.model_name)
        validate_upstream_source(self.model_name, source)
        return source.resolve(strict=True)

    def _training_config(self, spec: RunSpec) -> dict[str, Any]:
        config = {**self._TRAIN_DEFAULTS, **{key: spec.extra[key] for key in self._TRAIN_KEYS if key in spec.extra}}
        config.pop("source_dir", None)
        for key in ("encoder_dim", "encoder_l", "decoder_dim", "decoder_l", "epochs", "batch_size", "logging"):
            config[key] = self._positive_int(key, config[key])
        config["patience"] = self._positive_int("patience", config["patience"], minimum=0)
        config["num_threads"] = self._positive_int("num_threads", config["num_threads"])
        for key in ("alpha", "beta", "weight_decay"):
            config[key] = self._finite_float(key, config[key], minimum=0.0)
        config["learning_rate"] = self._finite_float("learning_rate", config["learning_rate"])
        if config["learning_rate"] <= 0:
            raise ValueError("Goggle learning_rate must be positive.")
        config["threshold"] = self._finite_float("threshold", config["threshold"])
        if not 0 <= config["threshold"] <= 1:
            raise ValueError("Goggle threshold must be between zero and one.")
        if config["decoder_arch"] not in {"gcn", "sage", "het"}:
            raise ValueError("Goggle decoder_arch must be 'gcn', 'sage', or 'het'.")
        for key in ("het_encoding", "iter_opt"):
            if not isinstance(config[key], bool):
                raise ValueError(f"Goggle {key} must be a boolean.")
        if (config["graph_prior"] is None) != (config["prior_mask"] is None):
            raise ValueError("Goggle graph_prior and prior_mask must either both be supplied or both be omitted.")
        return config

    @staticmethod
    def _scalar_record(value: Any) -> dict[str, Any]:
        if isinstance(value, (np.bool_, bool)):
            return {"type": "bool", "value": bool(value)}
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return {"type": "int", "value": int(value)}
        if isinstance(value, (np.floating, float)):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("Goggle categorical and target values must be finite.")
            return {"type": "float", "value": result}
        return {"type": "str", "value": str(value)}

    @classmethod
    def _category_key(cls, value: Any) -> str:
        return json.dumps(cls._scalar_record(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _ordered_categories(cls, series: pd.Series) -> list[dict[str, Any]]:
        records = {cls._category_key(value): cls._scalar_record(value) for value in series.tolist()}
        return [records[key] for key in sorted(records)]

    @staticmethod
    def _record_value(record: dict[str, Any]) -> Any:
        return record["value"]

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("Goggle requires a materialized real training CSV.")
        train_path = dataset_spec.train_data_path
        if train_path.is_symlink() or not train_path.is_file():
            raise FileNotFoundError(f"Goggle training data must be a regular CSV file: {train_path}")
        frame = pd.read_csv(train_path)
        if frame.empty:
            raise ValueError("Goggle does not accept an empty training dataset.")
        if frame.columns.tolist() != dataset_spec.column_names:
            raise ValueError(
                "Goggle training columns must exactly match the canonical dataset order: "
                f"observed={frame.columns.tolist()}, expected={dataset_spec.column_names}."
            )
        if len(set(frame.columns)) != len(frame.columns):
            raise ValueError("Goggle does not accept duplicate training columns.")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("Goggle supports exactly one target column.")
        declared = [
            *dataset_spec.numerical_columns,
            *dataset_spec.categorical_columns,
            *dataset_spec.target_columns,
        ]
        if len(declared) != len(set(declared)) or set(declared) != set(dataset_spec.column_names):
            raise ValueError("Goggle dataset column roles must be disjoint and cover every canonical column.")
        if bool(frame.isna().any().any()):
            raise ValueError(
                "Goggle rejects missing values; run the centralized train-split-fitted mean/mode imputer first."
            )
        numerical = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numerical.extend(dataset_spec.target_columns)
        for column in numerical:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError(f"Goggle numerical column must have a numeric dtype: {column}")
            if not bool(np.isfinite(frame[column].to_numpy(dtype=np.float64)).all()):
                raise ValueError(f"Goggle numerical column contains non-finite values: {column}")
        return frame

    def _transform_training_frame(
        self, frame: pd.DataFrame, dataset_spec: DatasetSpec
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        transformed: dict[str, np.ndarray] = {}
        numeric_records: list[dict[str, Any]] = []
        categorical_records: list[dict[str, Any]] = []
        for column in dataset_spec.numerical_columns:
            values = frame[column].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            scale = float(np.sqrt(np.mean(np.square(values - mean))))
            if scale == 0.0:
                scale = 1.0
            transformed[f"num:{column}"] = (values - mean) / scale
            numeric_records.append({"column": column, "mean": mean, "scale": scale})
        for column in dataset_spec.categorical_columns:
            categories = self._ordered_categories(frame[column])
            index = {
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")): offset
                for offset, record in enumerate(categories)
            }
            encoded = np.zeros((len(frame), len(categories)), dtype=np.float64)
            for row, value in enumerate(frame[column].tolist()):
                encoded[row, index[self._category_key(value)]] = 1.0
            for offset in range(len(categories)):
                transformed[f"cat:{column}:{offset}"] = encoded[:, offset]
            categorical_records.append({"column": column, "categories": categories})
        target = dataset_spec.target_columns[0]
        if dataset_spec.task_type == "classification":
            target_categories = self._ordered_categories(frame[target])
            target_index = {
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")): offset
                for offset, record in enumerate(target_categories)
            }
            transformed[f"target:{target}"] = np.asarray(
                [target_index[self._category_key(value)] for value in frame[target].tolist()], dtype=np.float64
            )
            target_record: dict[str, Any] = {
                "column": target,
                "kind": "classification",
                "categories": target_categories,
            }
        else:
            transformed[f"target:{target}"] = frame[target].to_numpy(dtype=np.float64)
            target_record = {"column": target, "kind": "regression"}
        transformed_frame = pd.DataFrame(transformed, index=frame.index)
        if not bool(np.isfinite(transformed_frame.to_numpy(dtype=np.float64)).all()):
            raise ValueError("Goggle preprocessing produced non-finite values.")
        metadata = {
            "schema_version": 1,
            "task_type": dataset_spec.task_type,
            "column_names": list(dataset_spec.column_names),
            "numerical": numeric_records,
            "categorical": categorical_records,
            "target": target_record,
            "transformed_columns": transformed_frame.columns.tolist(),
            "input_dim": int(transformed_frame.shape[1]),
            "training_rows": int(transformed_frame.shape[0]),
            "fit_scope": "real-training-split-only",
            "numerical_transform": "population-standardization-equivalent-to-StandardScaler",
            "categorical_transform": "deterministic-train-fitted-one-hot",
        }
        return transformed_frame, metadata

    def _validate_prior(self, config: dict[str, Any], input_dim: int) -> None:
        if config["graph_prior"] is None:
            return
        for name in ("graph_prior", "prior_mask"):
            try:
                matrix = np.asarray(config[name], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Goggle {name} must be a finite square numeric matrix.") from exc
            if matrix.shape != (input_dim, input_dim) or not bool(np.isfinite(matrix).all()):
                raise ValueError(f"Goggle {name} must have finite shape {(input_dim, input_dim)}.")
            if name == "prior_mask" and not bool(np.isin(matrix, [0.0, 1.0]).all()):
                raise ValueError("Goggle prior_mask must contain only zero and one.")

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.checkpoint_filename)

    def _metadata_path(self, spec: RunSpec) -> Path:
        return spec.output_dir / "goggle-model-metadata.json"

    def _runtime_config_path(self, spec: RunSpec) -> Path:
        return spec.output_dir / "goggle-runtime-config.json"

    def _run_goggle(self, args: list[str]) -> None:
        launcher = self.repo_root / "standardized_tabular_diffusion" / "compat" / "goggle_launcher.py"
        if not launcher.is_file():
            raise FileNotFoundError(f"Goggle compatibility launcher is missing: {launcher}")
        self._run_python(
            [str(launcher), *args],
            self.repo_root,
            env={
                "DGLBACKEND": "pytorch",
                "PYTHONPATH": os.pathsep.join(filter(None, [str(self.repo_root), os.environ.get("PYTHONPATH")])),
            },
        )

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._validate_extra(spec, action="train")
        self._ensure_output_dir(spec)
        source_root = self._source_root(spec)
        source = validate_upstream_source(self.model_name, source_root)
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._load_training_frame(dataset_spec)
        transformed, transform = self._transform_training_frame(frame, dataset_spec)
        config = self._training_config(spec)
        self._validate_prior(config, transform["input_dim"])
        execution_config = {
            **config,
            "dataset": spec.dataset,
            "seed": spec.seed,
            "input_dim": transform["input_dim"],
            "training_rows": transform["training_rows"],
        }
        runtime_config_path = self._runtime_config_path(spec)
        atomic_write_json(runtime_config_path, execution_config)
        checkpoint = spec.output_dir / self.checkpoint_filename
        if spec.checkpoint_path is not None and spec.checkpoint_path.resolve() != checkpoint.resolve():
            raise ValueError("Goggle training always writes model.pt inside output_dir; checkpoint_path is sample-only.")
        with tempfile.TemporaryDirectory(prefix="goggle-train-") as temporary:
            transformed_path = Path(temporary) / "training.csv"
            transformed.to_csv(transformed_path, index=False)
            self._run_goggle(
                [
                    "--action",
                    "train",
                    "--source-dir",
                    str(source_root),
                    "--output-dir",
                    str(spec.output_dir.resolve()),
                    "--config",
                    str(runtime_config_path.resolve()),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--device",
                    spec.device,
                    "--num-threads",
                    str(config["num_threads"]),
                    "--input-csv",
                    str(transformed_path.resolve()),
                ],
            )
        checkpoint = self._validate_trusted_executable_artifact(
            spec, checkpoint, format_name="PyTorch weights-only checkpoint"
        )
        source_after = validate_upstream_source(self.model_name, source_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("Goggle source identity changed during training.")
        atomic_write_json(
            self._metadata_path(spec),
            {
                "schema_version": 1,
                "model": self.model_name,
                "dataset": spec.dataset,
                "seed": spec.seed,
                "device": spec.device,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": self._sha256(checkpoint),
                "runtime_config_path": str(runtime_config_path.resolve()),
                "runtime_config_sha256": self._sha256(runtime_config_path),
                "execution_config": execution_config,
                "transform": transform,
                "source": source,
                "compatibility_boundary": "goggle-official-adapter-runtime-v1",
                "sampling_boundary": "official Goggle.model.sample followed by centralized inverse transformation",
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=source_root,
            notes=[
                "Training invoked checksum-locked, unmodified method-author Goggle source.",
                "Only train-fitted format adaptation, source verification, and output isolation occur outside upstream code.",
            ],
        )
        return self._write_bundle(bundle)

    def _load_metadata(self, spec: RunSpec, checkpoint: Path, source: dict[str, Any]) -> dict[str, Any]:
        path = self._metadata_path(spec)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Goggle model metadata is missing: {path}")
        metadata = read_json(path)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Goggle model metadata is malformed or unsupported.")
        if metadata.get("model") != self.model_name or metadata.get("dataset") != spec.dataset:
            raise ValueError("Goggle model metadata does not match the requested model and dataset.")
        if metadata.get("checkpoint_sha256") != self._sha256(checkpoint):
            raise ValueError("Goggle checkpoint checksum does not match its training metadata.")
        recorded_source = metadata.get("source")
        if not isinstance(recorded_source, dict) or recorded_source.get("upstream_commit") != source["upstream_commit"]:
            raise ValueError("Goggle training metadata does not match the verified official source revision.")
        runtime_config = self._runtime_config_path(spec)
        if runtime_config.is_symlink() or not runtime_config.is_file():
            raise FileNotFoundError(f"Goggle runtime configuration is missing: {runtime_config}")
        if metadata.get("runtime_config_sha256") != self._sha256(runtime_config):
            raise ValueError("Goggle runtime configuration checksum does not match its training metadata.")
        return metadata

    def _inverse_transform(self, raw: np.ndarray, transform: dict[str, Any]) -> pd.DataFrame:
        if raw.ndim != 2 or raw.shape[1] != transform["input_dim"] or not bool(np.isfinite(raw).all()):
            raise ValueError("Goggle raw samples do not match the finite transformed-data contract.")
        transformed_columns = transform["transformed_columns"]
        transformed = pd.DataFrame(raw, columns=transformed_columns)
        output: dict[str, Any] = {}
        for record in transform["numerical"]:
            output[record["column"]] = (
                transformed[f"num:{record['column']}"] * record["scale"] + record["mean"]
            ).to_numpy()
        for record in transform["categorical"]:
            columns = [f"cat:{record['column']}:{offset}" for offset in range(len(record["categories"]))]
            indices = transformed[columns].to_numpy().argmax(axis=1)
            values = [self._record_value(record["categories"][int(index)]) for index in indices]
            output[record["column"]] = values
        target = transform["target"]
        target_values = transformed[f"target:{target['column']}"]
        if target["kind"] == "classification":
            indices = np.rint(target_values.to_numpy()).astype(np.int64)
            indices = np.clip(indices, 0, len(target["categories"]) - 1)
            output[target["column"]] = [
                self._record_value(target["categories"][int(index)]) for index in indices
            ]
        else:
            output[target["column"]] = target_values.to_numpy()
        result = pd.DataFrame(output)[transform["column_names"]]
        if result.shape[0] != raw.shape[0] or bool(result.isna().any().any()):
            raise RuntimeError("Goggle inverse transformation violated the sample contract.")
        return result

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._validate_extra(spec, action="sample")
        self._ensure_output_dir(spec)
        source_root = self._source_root(spec)
        source = validate_upstream_source(self.model_name, source_root)
        checkpoint = self._validate_trusted_executable_artifact(
            spec,
            self._resolve_checkpoint_path(spec),
            format_name="PyTorch weights-only checkpoint",
        )
        metadata = self._load_metadata(spec, checkpoint, source)
        num_samples = spec.num_samples or int(metadata["transform"]["training_rows"])
        self._positive_int("num_samples", num_samples)
        num_threads = self._positive_int("num_threads", spec.extra.get("num_threads", 1))
        runtime_config_path = self._runtime_config_path(spec)
        with tempfile.TemporaryDirectory(prefix=".goggle-sample-", dir=spec.output_dir) as temporary:
            raw_path = Path(temporary) / "raw.npy"
            self._run_goggle(
                [
                    "--action",
                    "sample",
                    "--source-dir",
                    str(source_root),
                    "--output-dir",
                    str(spec.output_dir.resolve()),
                    "--config",
                    str(runtime_config_path.resolve()),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    spec.device,
                    "--num-threads",
                    str(num_threads),
                    "--num-samples",
                    str(num_samples),
                    "--raw-output",
                    str(raw_path.resolve()),
                ],
            )
            raw = np.load(raw_path, allow_pickle=False)
        sample_frame = self._inverse_transform(raw, metadata["transform"])
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_frame, sample_path)
        source_after = validate_upstream_source(self.model_name, source_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("Goggle source identity changed during sampling.")
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=source_root,
            generated_sample_path=sample_path,
            notes=[
                "Sampling loaded a checksum-verified weights-only checkpoint under the recorded official source.",
                "Requested row count and categorical reconstruction are adapter-only output-contract operations.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
