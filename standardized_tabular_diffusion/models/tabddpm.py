from __future__ import annotations

import io
import json
import math
import os
import re
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.runtime_contracts import (
    bind_native_dataset_view,
    normalize_device_request,
)


def build_tabddpm_environment(upstream_root: Path) -> dict[str, str]:
    """Expose official imports plus the narrow scikit-learn API bridge."""

    compatibility_root = Path(__file__).resolve().parents[1] / "compat" / "tabddpm_sklearn"
    entries = [str(compatibility_root), str(upstream_root.resolve())]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return {"PYTHONPATH": os.pathsep.join(entries)}


_BARE_TOML_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _toml_key(value: str) -> str:
    return value if _BARE_TOML_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TabDDPM TOML values must be finite.")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TabDDPM TOML value: {type(value).__name__}")


def _toml_bytes(payload: dict[str, Any]) -> bytes:
    lines: list[str] = []

    def emit_table(table: dict[str, Any], path: tuple[str, ...]) -> None:
        if path:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{_toml_key(key)} = {_toml_scalar(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                emit_table(value, (*path, key))

    emit_table(payload, ())
    serialized = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    if tomllib.loads(serialized.decode("utf-8")) != payload:
        raise RuntimeError("TabDDPM runtime TOML failed its semantic round-trip check.")
    return serialized


class TabDDPMAdapter(BaseModelAdapter):
    model_name = "tabddpm"
    upstream_dirname = "TabDDPM-main"

    @staticmethod
    def _runtime_root(spec: RunSpec) -> Path:
        return spec.output_dir / "tabddpm-runtime"

    @staticmethod
    def _metadata_path(spec: RunSpec) -> Path:
        return spec.output_dir / "tabddpm-model-metadata.json"

    def _require_config(self, spec: RunSpec) -> Path:
        if spec.upstream_config_path is None:
            raise ValueError("TabDDPM requires RunSpec.upstream_config_path.")
        path = spec.upstream_config_path
        if path.is_symlink():
            raise ValueError(f"TabDDPM refuses a symlinked TOML configuration: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.suffix.lower() != ".toml":
            raise ValueError(f"TabDDPM upstream_config_path must be a regular TOML file: {resolved}")
        return resolved

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = normalize_device_request(device)
        if normalized == "cpu":
            return normalized
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TabDDPM CUDA execution requires PyTorch.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for TabDDPM but is unavailable: {normalized}")
        index = 0 if normalized == "cuda" else int(normalized.split(":", maxsplit=1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"TabDDPM requested cuda:{index}, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        return f"cuda:{index}"

    @staticmethod
    def _write_numpy(path: Path, value: Any) -> None:
        import numpy as np

        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        atomic_write_bytes(path, stream.getvalue())

    def _prepare_data(self, spec: RunSpec, dataset_spec: DatasetSpec) -> dict[str, Any]:
        import numpy as np

        canonical_root = dataset_spec.metadata_path.parent
        source_binding = bind_native_dataset_view(dataset_spec, canonical_root)
        runtime_data = self._runtime_root(spec) / "data" / spec.dataset
        if runtime_data.is_symlink():
            raise ValueError(f"TabDDPM runtime data directory must not be a symlink: {runtime_data}")
        runtime_data.mkdir(parents=True, exist_ok=True)
        copied: dict[str, dict[str, Any]] = {}
        source_names = ["info.json", "y_train.npy", "y_test.npy"]
        for prefix in ("X_num", "X_cat"):
            for split in ("train", "test"):
                candidate = canonical_root / f"{prefix}_{split}.npy"
                if candidate.is_file():
                    source_names.append(candidate.name)
        for name in source_names:
            source = canonical_root / name
            if source.is_symlink() or not source.is_file():
                raise FileNotFoundError(f"TabDDPM canonical input file is missing or unsafe: {source}")
            target = runtime_data / name
            payload = source.read_bytes()
            if target.exists() and (target.is_symlink() or not target.is_file()):
                raise ValueError(f"TabDDPM runtime input path is unsafe: {target}")
            if not target.exists() or sha256_file(target) != sha256_file(source):
                atomic_write_bytes(target, payload)
            copied[name] = {
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        train_y = np.load(canonical_root / "y_train.npy", allow_pickle=False)
        if len(train_y) < 1:
            raise ValueError("TabDDPM requires a non-empty canonical training split.")
        validation_names = ["y_val.npy"]
        self._write_numpy(runtime_data / "y_val.npy", train_y[:1])
        for prefix in ("X_num", "X_cat"):
            source = canonical_root / f"{prefix}_train.npy"
            if source.is_file():
                values = np.load(source, allow_pickle=False)
                self._write_numpy(runtime_data / f"{prefix}_val.npy", values[:1])
                validation_names.append(f"{prefix}_val.npy")
        validation = {
            name: {"bytes": (runtime_data / name).stat().st_size, "sha256": sha256_file(runtime_data / name)}
            for name in validation_names
        }
        return {
            "schema_version": 1,
            "source_binding": source_binding,
            "runtime_data_path": str(runtime_data.resolve()),
            "copied_files": copied,
            "validation_compatibility_mirror": {
                "source": "canonical-training-row-0",
                "rows": 1,
                "used_for_fit": False,
                "files": validation,
            },
        }

    @staticmethod
    def _scientific_config(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(payload)
        for key in ("parent_dir", "real_data_path", "device", "seed", "sample"):
            normalized.pop(key, None)
        return normalized

    def _runtime_config(
        self,
        spec: RunSpec,
        *,
        action: str,
        data_binding: dict[str, Any],
        training_seed: int | None = None,
    ) -> tuple[Path, dict[str, Any], Path]:
        source_path = self._require_config(spec)
        with source_path.open("rb") as stream:
            payload = tomllib.load(stream)
        required = {"train", "sample", "diffusion_params", "model_params", "model_type"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"TabDDPM configuration is missing required sections: {missing}")
        runtime_root = self._runtime_root(spec)
        if runtime_root.is_symlink():
            raise ValueError(f"TabDDPM runtime root must not be a symlink: {runtime_root}")
        runtime_root.mkdir(parents=True, exist_ok=True)
        payload["parent_dir"] = str(runtime_root.resolve())
        payload["real_data_path"] = data_binding["runtime_data_path"]
        payload["device"] = self._resolve_device(spec.device)
        payload["seed"] = spec.seed
        if action == "train":
            payload["train"]["main"]["seed"] = spec.seed
            payload["train"]["T"]["seed"] = spec.seed
        elif action == "sample":
            if spec.checkpoint_path is not None:
                raise ValueError("TabDDPM sampling uses the run-owned checkpoint; checkpoint_path is unsupported.")
            payload["sample"]["seed"] = spec.seed
            if not isinstance(training_seed, int) or isinstance(training_seed, bool) or training_seed < 0:
                raise ValueError("TabDDPM sampling requires a valid recorded training seed.")
            payload["train"]["main"]["seed"] = training_seed
            payload["train"]["T"]["seed"] = training_seed
            if spec.num_samples is not None:
                if isinstance(spec.num_samples, bool) or spec.num_samples <= 0:
                    raise ValueError("TabDDPM num_samples must be a positive integer.")
                payload["sample"]["num_samples"] = spec.num_samples
        else:  # pragma: no cover - private call invariant
            raise ValueError(f"Unsupported TabDDPM action: {action}")
        runtime_path = runtime_root / f"config-{action}-seed-{spec.seed}.toml"
        atomic_write_bytes(runtime_path, _toml_bytes(payload))
        return runtime_path, payload, source_path

    def _run_pipeline(self, config_path: Path, action: str) -> None:
        self._run_python(
            ["scripts/pipeline.py", "--config", str(config_path), f"--{action}"],
            self.upstream_root,
            env=build_tabddpm_environment(self.upstream_root),
        )

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._require_config(spec)
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        data_binding = self._prepare_data(spec, dataset_spec)
        runtime_config, config, source_config = self._runtime_config(
            spec,
            action="train",
            data_binding=data_binding,
        )
        self._run_pipeline(runtime_config, "train")
        runtime_root = self._runtime_root(spec)
        checkpoints = {}
        for name in ("model.pt", "model_ema.pt"):
            path = self._validate_trusted_executable_artifact(
                spec,
                runtime_root / name,
                format_name="PyTorch checkpoint",
            )
            checkpoints[name] = {"path": str(path), "sha256": sha256_file(path)}
        metadata = {
            "schema_version": 1,
            "model": self.model_name,
            "dataset": spec.dataset,
            "training_seed": spec.seed,
            "device": config["device"],
            "source_config_path": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "runtime_config_path": str(runtime_config),
            "runtime_config_sha256": sha256_file(runtime_config),
            "scientific_config": self._scientific_config(config),
            "data_binding": data_binding,
            "checkpoints": checkpoints,
        }
        atomic_write_json(self._metadata_path(spec), metadata)
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.upstream_root,
                notes=[
                    "Official TabDDPM training received a run-owned TOML overlay binding seed, device, data, and output.",
                    "The authoritative source and input TOML were not modified.",
                ],
            )
        )

    @staticmethod
    def _target_mapping(dataset_spec: DatasetSpec, data_root: Path) -> dict[int, Any]:
        import numpy as np
        import pandas as pd

        if dataset_spec.train_data_path is None or len(dataset_spec.target_columns) != 1:
            raise ValueError("TabDDPM decoding requires one target column and a canonical training CSV.")
        target = dataset_spec.target_columns[0]
        labels = pd.read_csv(dataset_spec.train_data_path, usecols=[target])[target].tolist()
        encoded = np.load(data_root / "y_train.npy", allow_pickle=False).reshape(-1)
        if len(labels) != len(encoded):
            raise ValueError("TabDDPM target array is not row-aligned with the canonical training table.")
        mapping: dict[int, Any] = {}
        reverse: dict[str, int] = {}
        for index_value, label in zip(encoded.tolist(), labels):
            if isinstance(index_value, bool) or int(index_value) != index_value:
                raise ValueError("TabDDPM classification target indices must be integral.")
            index = int(index_value)
            label_key = json.dumps(label, ensure_ascii=False, sort_keys=True)
            if index in mapping and mapping[index] != label:
                raise ValueError("TabDDPM target index maps to multiple canonical labels.")
            if label_key in reverse and reverse[label_key] != index:
                raise ValueError("TabDDPM canonical target label maps to multiple indices.")
            mapping[index] = label
            reverse[label_key] = index
        return mapping

    def _decode_sample(
        self,
        spec: RunSpec,
        dataset_spec: DatasetSpec,
        data_binding: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        import numpy as np
        import pandas as pd

        runtime_root = self._runtime_root(spec)
        requested = spec.num_samples
        raw_paths = {"target": runtime_root / "y_train.npy"}
        if dataset_spec.numerical_columns:
            raw_paths["numerical"] = runtime_root / "X_num_train.npy"
        if dataset_spec.categorical_columns:
            raw_paths["categorical"] = runtime_root / "X_cat_train.npy"
        if any(path.is_symlink() or not path.is_file() for path in raw_paths.values()):
            raise FileNotFoundError(f"TabDDPM did not emit its complete sample arrays: {raw_paths}")
        x_num = (
            np.load(raw_paths["numerical"], allow_pickle=False)
            if "numerical" in raw_paths
            else np.empty((len(np.load(raw_paths["target"], allow_pickle=False)), 0))
        )
        x_cat = (
            np.load(raw_paths["categorical"], allow_pickle=True)
            if "categorical" in raw_paths
            else np.empty((len(x_num), 0), dtype=str)
        )
        y = np.load(raw_paths["target"], allow_pickle=False).reshape(-1)
        row_counts = {len(x_num), len(x_cat), len(y)}
        if len(row_counts) != 1 or (requested is not None and row_counts != {requested}):
            raise RuntimeError(f"TabDDPM generated inconsistent or unexpected row counts: {row_counts}")
        if x_num.ndim != 2 or x_num.shape[1] != len(dataset_spec.numerical_columns):
            raise RuntimeError(f"TabDDPM numerical output has unexpected shape: {x_num.shape}")
        if x_cat.ndim != 2 or x_cat.shape[1] != len(dataset_spec.categorical_columns):
            raise RuntimeError(f"TabDDPM categorical output has unexpected shape: {x_cat.shape}")
        if not np.isfinite(x_num).all():
            raise RuntimeError("TabDDPM generated non-finite numerical values.")
        decoded: dict[str, Any] = {}
        column_info = dataset_spec.extra.get("column_info", {})
        for index, column in enumerate(dataset_spec.numerical_columns):
            values = x_num[:, index]
            decoded[column] = np.rint(values).astype(np.int64) if column_info.get(column) == "int" else values
        canonical_train = pd.read_csv(dataset_spec.train_data_path)
        for index, column in enumerate(dataset_spec.categorical_columns):
            values = pd.Series(x_cat[:, index], dtype="string")
            allowed = set(canonical_train[column].astype(str).unique())
            observed = set(values.dropna().astype(str).unique())
            if values.isna().any() or not observed.issubset(allowed):
                raise RuntimeError(f"TabDDPM generated invalid categorical values for {column!r}.")
            decoded[column] = values.astype(str).to_numpy()
        target_column = dataset_spec.target_columns[0]
        if dataset_spec.task_type == "regression":
            target_values = y.astype(float)
            if not np.isfinite(target_values).all():
                raise RuntimeError("TabDDPM generated a non-finite regression target.")
            decoded[target_column] = (
                np.rint(target_values).astype(np.int64)
                if column_info.get(target_column) == "int"
                else target_values
            )
        else:
            mapping = self._target_mapping(
                dataset_spec,
                Path(data_binding["runtime_data_path"]),
            )
            if not np.isfinite(y.astype(float)).all() or not np.equal(y, np.rint(y)).all():
                raise RuntimeError("TabDDPM generated a non-integral classification target.")
            target_indices = np.rint(y).astype(np.int64)
            if not set(np.unique(target_indices)).issubset(mapping):
                raise RuntimeError("TabDDPM generated an unknown target index.")
            decoded[target_column] = [mapping[int(index)] for index in target_indices]
        frame = pd.DataFrame(decoded).loc[:, dataset_spec.column_names]
        if bool(frame.isna().any().any()):
            raise RuntimeError("TabDDPM decoded output contains missing values.")
        sample_path = spec.output_dir / f"samples-seed-{spec.seed}.csv"
        self._write_dataframe_csv(frame, sample_path)
        raw_dir = spec.output_dir / "raw-upstream-arrays" / f"seed-{spec.seed}"
        raw_hashes: dict[str, str] = {}
        for name in ("X_num_train.npy", "X_cat_train.npy", "y_train.npy", "X_num_unnorm.npy", "X_cat_unnorm.npy"):
            source = runtime_root / name
            if source.is_file() and not source.is_symlink():
                destination = raw_dir / name
                atomic_write_bytes(destination, source.read_bytes())
                raw_hashes[name] = sha256_file(destination)
        return sample_path, {
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": sha256_file(sample_path),
            "raw_array_sha256": raw_hashes,
            "integer_decoding": "nearest-integer-at-adapter-boundary-for-declared-integer-columns",
        }

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._require_config(spec)
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        data_binding = self._prepare_data(spec, dataset_spec)
        metadata_path = self._metadata_path(spec)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError("TabDDPM sampling requires run-owned training metadata and checkpoints.")
        metadata = read_json(metadata_path)
        runtime_config, config, source_config = self._runtime_config(
            spec,
            action="sample",
            data_binding=data_binding,
            training_seed=metadata.get("training_seed"),
        )
        if metadata.get("model") != self.model_name or metadata.get("dataset") != spec.dataset:
            raise ValueError("TabDDPM training metadata does not match the requested run.")
        if metadata.get("scientific_config") != self._scientific_config(config):
            raise ValueError("TabDDPM sampling configuration differs from the trained scientific configuration.")
        if metadata.get("data_binding") != data_binding:
            raise ValueError("TabDDPM dataset binding changed after training.")
        checkpoint_records = metadata.get("checkpoints")
        if not isinstance(checkpoint_records, dict) or set(checkpoint_records) != {"model.pt", "model_ema.pt"}:
            raise ValueError("TabDDPM training metadata has an incomplete checkpoint inventory.")
        for name, record in checkpoint_records.items():
            checkpoint = self._validate_trusted_executable_artifact(
                spec,
                self._runtime_root(spec) / name,
                format_name="PyTorch checkpoint",
            )
            if sha256_file(checkpoint) != record.get("sha256"):
                raise ValueError(f"TabDDPM checkpoint changed after training: {name}")
        self._run_pipeline(runtime_config, "sample")
        sample_path, sample_record = self._decode_sample(spec, dataset_spec, data_binding)
        atomic_write_json(
            spec.output_dir / f"tabddpm-sample-seed-{spec.seed}.json",
            {
                "schema_version": 1,
                "model": self.model_name,
                "dataset": spec.dataset,
                "seed": spec.seed,
                "device": config["device"],
                "requested_rows": spec.num_samples,
                "source_config_path": str(source_config),
                "source_config_sha256": sha256_file(source_config),
                "runtime_config_path": str(runtime_config),
                "runtime_config_sha256": sha256_file(runtime_config),
                "sample": sample_record,
            },
        )
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.upstream_root,
                generated_sample_path=sample_path,
                notes=[
                    "Official TabDDPM sampling received the requested seed, device, and row count through a run-owned TOML overlay.",
                    "The decoded canonical table and raw upstream arrays are retained under seed-specific paths.",
                ],
            )
        )

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        raise RuntimeError(
            "TabDDPM upstream-summary normalization is retired; provide the decoded sample to the central runner."
        )
