from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import validate_upstream_source


class STaSyAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "stasy"
    upstream_dirname = "TabSyn-main"

    _STANDARD_EXTRA_KEYS = {"action_extras", "config", "dataset_spec", "evaluation", "tags"}
    _TRAIN_KEYS = {
        "batch_size",
        "epochs",
        "hidden_dims",
        "nf",
        "num_scales",
        "num_threads",
        "num_workers",
        "sampler",
        "spl",
    }
    _SAMPLE_KEYS = {"num_threads", "sampler"}
    _TRAIN_DEFAULTS: dict[str, Any] = {
        "epochs": 10001,
        "batch_size": 1000,
        "nf": 64,
        "hidden_dims": [1024, 2048, 1024, 1024],
        "num_scales": 50,
        "num_workers": 4,
        "num_threads": 1,
        "sampler": "ode",
        "spl": True,
    }

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
            raise ValueError(f"STaSy {name} must be an integer greater than or equal to {minimum}.")
        return value

    def _validate_extra(self, spec: RunSpec, *, action: str) -> None:
        action_keys = self._TRAIN_KEYS if action == "train" else self._SAMPLE_KEYS
        unknown = sorted(set(spec.extra) - self._STANDARD_EXTRA_KEYS - action_keys)
        if unknown:
            raise ValueError(f"Unsupported STaSy {action} controls: {', '.join(unknown)}")

    def _training_config(self, extra: dict[str, Any]) -> dict[str, Any]:
        config = {**self._TRAIN_DEFAULTS, **{key: extra[key] for key in self._TRAIN_KEYS if key in extra}}
        config["epochs"] = self._positive_int("epochs", config["epochs"])
        config["batch_size"] = self._positive_int("batch_size", config["batch_size"])
        config["nf"] = self._positive_int("nf", config["nf"])
        config["num_scales"] = self._positive_int("num_scales", config["num_scales"], minimum=2)
        config["num_workers"] = self._positive_int("num_workers", config["num_workers"], minimum=0)
        config["num_threads"] = self._positive_int("num_threads", config["num_threads"])
        hidden_dims = config["hidden_dims"]
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError("STaSy hidden_dims must be a non-empty list of positive integers.")
        config["hidden_dims"] = [self._positive_int("hidden_dims item", value) for value in hidden_dims]
        if config["sampler"] not in {"ode", "pc"}:
            raise ValueError("STaSy sampler must be 'ode' or 'pc'.")
        if not isinstance(config["spl"], bool):
            raise ValueError("STaSy spl must be a boolean.")
        return config

    def _validate_dataset(self, dataset: str) -> Path:
        import numpy as np
        import pandas as pd

        dataset_path = Path(dataset)
        if (
            not dataset
            or "\x00" in dataset
            or any(separator in dataset for separator in ("/", "\\", ":"))
            or dataset_path.is_absolute()
            or len(dataset_path.parts) != 1
            or dataset_path.name in {".", ".."}
        ):
            raise ValueError("STaSy dataset must be a single safe dataset identifier, not a path.")
        data_root = (self.upstream_root / "data").resolve()
        data_dir = data_root / dataset
        if data_dir.is_symlink() or not data_dir.is_dir():
            raise FileNotFoundError(f"STaSy processed dataset directory is missing: {data_dir}")
        if not data_dir.resolve(strict=True).is_relative_to(data_root):
            raise ValueError(f"STaSy dataset directory escapes the expected data root: {data_dir}")
        info_path = data_dir / "info.json"
        if info_path.is_symlink() or not info_path.is_file():
            raise FileNotFoundError(f"STaSy dataset metadata is missing: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError(f"STaSy dataset metadata must be a JSON object: {info_path}")
        if info.get("task_type") not in {"binclass", "multiclass", "regression"}:
            raise ValueError(f"STaSy does not support task_type={info.get('task_type')!r}.")
        feature_kinds: list[str] = []
        for kind in ("num", "cat"):
            pair = [data_dir / f"X_{kind}_{split}.npy" for split in ("train", "test")]
            if any(path.exists() for path in pair):
                if not all(path.exists() for path in pair):
                    raise FileNotFoundError(f"STaSy requires both train and test X_{kind} arrays.")
                feature_kinds.append(kind)
        if not feature_kinds:
            raise FileNotFoundError("STaSy requires at least one numerical or categorical feature array pair.")

        split_rows: dict[str, list[int]] = {"train": [], "test": []}
        required = [
            data_dir / f"{name}_{split}.npy"
            for split in ("train", "test")
            for name in ["y", *[f"X_{kind}" for kind in feature_kinds]]
        ]
        for path in required:
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"STaSy requires a regular processed dataset file: {path}")
            try:
                array = np.load(path, allow_pickle=False)
            except ValueError as exc:
                raise ValueError(
                    f"STaSy requires non-pickle NumPy arrays; convert object values to strings or numbers: {path}"
                ) from exc
            is_target = path.name.startswith("y_")
            if array.ndim not in ({1, 2} if is_target else {2}) or (array.ndim == 2 and array.shape[1] == 0):
                expected = "one- or two-dimensional target" if is_target else "two-dimensional feature"
                raise ValueError(f"STaSy requires a non-empty {expected} array: {path}")
            if is_target and array.ndim == 2 and array.shape[1] != 1:
                raise ValueError(f"STaSy supports exactly one target column: {path}")
            if array.shape[0] == 0:
                raise ValueError(f"STaSy does not accept an empty processed dataset split: {path}")
            split = "train" if path.stem.endswith("_train") else "test"
            split_rows[split].append(int(array.shape[0]))
            if bool(pd.isna(array).any()):
                raise ValueError(
                    f"STaSy rejects missing values in {path}; run the centralized train-split-fitted imputer first."
                )
            if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).all()):
                raise ValueError(f"STaSy rejects non-finite numerical values in {path}.")
            if path.name.startswith("X_num_") and not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"STaSy numerical feature arrays must have a numerical dtype: {path}")
            if is_target and info["task_type"] == "regression" and not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"STaSy regression targets must have a numerical dtype: {path}")
        for split, row_counts in split_rows.items():
            if len(set(row_counts)) != 1:
                raise ValueError(f"STaSy {split} arrays have inconsistent row counts: {row_counts}")
            declared = info.get(f"{split}_num")
            if declared is not None and (not isinstance(declared, int) or declared != row_counts[0]):
                raise ValueError(
                    f"STaSy {split}_num metadata does not match the processed arrays: "
                    f"declared={declared!r}, observed={row_counts[0]}"
                )
        return data_dir

    def _launcher_args(self, config: dict[str, Any]) -> list[str]:
        args = [
            "--epochs",
            str(config["epochs"]),
            "--batch-size",
            str(config["batch_size"]),
            "--nf",
            str(config["nf"]),
            "--hidden-dims",
            *[str(value) for value in config["hidden_dims"]],
            "--num-scales",
            str(config["num_scales"]),
            "--num-workers",
            str(config["num_workers"]),
            "--num-threads",
            str(config["num_threads"]),
            "--sampler",
            str(config["sampler"]),
            "--spl" if config["spl"] else "--no-spl",
        ]
        return args

    def _run_stasy(self, args: list[str], *, seed: int) -> None:
        launcher = self.repo_root / "standardized_tabular_diffusion" / "compat" / "stasy_launcher.py"
        if not launcher.is_file():
            raise FileNotFoundError(f"STaSy compatibility launcher is missing: {launcher}")
        self._run_python(
            [str(launcher), *args, "--seed", str(seed)],
            self.upstream_root,
            env={"PYTHONHASHSEED": str(seed)},
        )

    def _metadata_path(self, spec: RunSpec) -> Path:
        return spec.output_dir / "stasy-model-metadata.json"

    def _checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.output_dir / "ckpt" / spec.dataset / "model.pth"

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._validate_extra(spec, action="train")
        self._ensure_output_dir(spec)
        self._validate_dataset(spec.dataset)
        source = validate_upstream_source(self.model_name, self.upstream_root)
        config = self._training_config(spec.extra)
        args = [
            "--action",
            "train",
            "--dataname",
            spec.dataset,
            "--output-dir",
            str(spec.output_dir.resolve()),
            "--device",
            spec.device,
            *self._launcher_args(config),
        ]
        self._run_stasy(args, seed=spec.seed)
        source_after = validate_upstream_source(self.model_name, self.upstream_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("STaSy source identity changed during training.")
        checkpoint = self._validate_trusted_executable_artifact(
            spec,
            self._checkpoint_path(spec),
            format_name="PyTorch pickle checkpoint",
        )
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
                "training_config": config,
                "source": source,
                "compatibility_boundary": "stasy-adapter-runtime-v1",
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                "Training used the checksum-locked TabSyn STaSy snapshot through adapter-only runtime controls.",
                "The trusted PyTorch checkpoint is confined to output_dir; tracked upstream source was not modified.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._validate_extra(spec, action="sample")
        self._ensure_output_dir(spec)
        self._validate_dataset(spec.dataset)
        source = validate_upstream_source(self.model_name, self.upstream_root)
        if spec.checkpoint_path is not None:
            raise ValueError("STaSy uses the trusted checkpoint and metadata inside output_dir; checkpoint_path is unsupported.")
        metadata_path = self._metadata_path(spec)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(f"STaSy training metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("STaSy training metadata does not satisfy schema version 1.")
        if metadata.get("model") != self.model_name or metadata.get("dataset") != spec.dataset:
            raise ValueError("STaSy checkpoint metadata does not match the requested model and dataset.")
        if metadata.get("source", {}).get("manifest_sha256") != source["manifest_sha256"]:
            raise ValueError("STaSy source manifest changed after training; refusing checkpoint reuse.")
        checkpoint = self._validate_trusted_executable_artifact(
            spec,
            self._checkpoint_path(spec),
            format_name="PyTorch pickle checkpoint",
        )
        if self._sha256(checkpoint) != metadata.get("checkpoint_sha256"):
            raise ValueError("STaSy checkpoint checksum differs from its training metadata.")
        config = self._training_config(metadata["training_config"])
        if "sampler" in spec.extra:
            config["sampler"] = spec.extra["sampler"]
        if "num_threads" in spec.extra:
            config["num_threads"] = self._positive_int("num_threads", spec.extra["num_threads"])
        config = self._training_config(config)
        sample_path = (spec.output_dir / "samples.csv").resolve()
        args = [
            "--action",
            "sample",
            "--dataname",
            spec.dataset,
            "--output-dir",
            str(spec.output_dir.resolve()),
            "--device",
            spec.device,
            "--save-path",
            str(sample_path),
            *self._launcher_args(config),
        ]
        if spec.num_samples is not None:
            args.extend(["--num-samples", str(spec.num_samples)])
        self._run_stasy(args, seed=spec.seed)
        source_after = validate_upstream_source(self.model_name, self.upstream_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("STaSy source identity changed during sampling.")
        if sample_path.is_symlink() or not sample_path.is_file():
            raise FileNotFoundError(f"STaSy did not produce the expected sample file: {sample_path}")
        import numpy as np
        import pandas as pd

        frame = pd.read_csv(sample_path)
        if spec.num_samples is not None and len(frame) != spec.num_samples:
            raise RuntimeError(f"STaSy produced {len(frame)} rows instead of the requested {spec.num_samples}.")
        if bool(frame.isna().any().any()):
            raise RuntimeError("STaSy produced missing values.")
        numerical = frame.select_dtypes(include=[np.number]).to_numpy()
        if numerical.size and not bool(np.isfinite(numerical).all()):
            raise RuntimeError("STaSy produced non-finite numerical values.")
        atomic_write_json(
            spec.output_dir / "stasy-sample-metadata.json",
            {
                "schema_version": 1,
                "model": self.model_name,
                "dataset": spec.dataset,
                "seed": spec.seed,
                "rows": len(frame),
                "columns": list(frame.columns),
                "sample_sha256": self._sha256(sample_path),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "sampling_config": {"sampler": config["sampler"], "num_scales": config["num_scales"]},
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=["Sampling verified the locked source, trusted checkpoint checksum, requested row count, and output."],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


class CoDiAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "codi"
    upstream_dirname = "TabSyn-main"

    _STANDARD_EXTRA_KEYS = {"action_extras", "config", "dataset_spec", "evaluation", "tags"}
    _TRAIN_KEYS = {
        "T",
        "activation",
        "beta_1",
        "beta_T",
        "encoder_dim_con",
        "encoder_dim_dis",
        "eval_batch_size",
        "grad_clip",
        "lambda_con",
        "lambda_dis",
        "lr_con",
        "lr_dis",
        "mean_type",
        "nf_con",
        "nf_dis",
        "num_threads",
        "sample_step",
        "total_epochs_both",
        "training_batch_size",
        "var_type",
    }
    _SAMPLE_KEYS = {"num_threads"}
    _TRAIN_DEFAULTS: dict[str, Any] = {
        "training_batch_size": 4096,
        "eval_batch_size": 2100,
        "T": 50,
        "beta_1": 0.00001,
        "beta_T": 0.02,
        "lr_con": 0.002,
        "lr_dis": 0.002,
        "total_epochs_both": 20000,
        "grad_clip": 1.0,
        "sample_step": 2000,
        "lambda_con": 0.2,
        "lambda_dis": 0.2,
        "nf_con": 16,
        "nf_dis": 64,
        "encoder_dim_con": [512, 1024, 1024, 512],
        "encoder_dim_dis": [512, 1024, 1024, 512],
        "activation": "relu",
        "mean_type": "epsilon",
        "var_type": "fixedsmall",
        "num_threads": 1,
    }

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
            raise ValueError(f"CoDi {name} must be an integer greater than or equal to {minimum}.")
        return value

    @staticmethod
    def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"CoDi {name} must be a finite number.")
        result = float(value)
        if not math.isfinite(result) or (positive and result <= 0):
            qualifier = "positive and finite" if positive else "finite"
            raise ValueError(f"CoDi {name} must be {qualifier}.")
        return result

    def _validate_extra(self, spec: RunSpec, *, action: str) -> None:
        action_keys = self._TRAIN_KEYS if action == "train" else self._SAMPLE_KEYS
        unknown = sorted(set(spec.extra) - self._STANDARD_EXTRA_KEYS - action_keys)
        if unknown:
            raise ValueError(f"Unsupported CoDi {action} controls: {', '.join(unknown)}")

    def _training_config(self, extra: dict[str, Any]) -> dict[str, Any]:
        config = {**self._TRAIN_DEFAULTS, **{key: extra[key] for key in self._TRAIN_KEYS if key in extra}}
        for key in ("training_batch_size", "eval_batch_size", "total_epochs_both", "sample_step", "num_threads"):
            config[key] = self._positive_int(key, config[key])
        config["T"] = self._positive_int("T", config["T"], minimum=2)
        for key in ("nf_con", "nf_dis"):
            config[key] = self._positive_int(key, config[key], minimum=4)
        for key in ("beta_1", "beta_T", "lr_con", "lr_dis", "grad_clip"):
            config[key] = self._finite_float(key, config[key], positive=True)
        for key in ("lambda_con", "lambda_dis"):
            config[key] = self._finite_float(key, config[key])
            if config[key] < 0:
                raise ValueError(f"CoDi {key} must be non-negative.")
        if not 0 < config["beta_1"] < config["beta_T"] < 1:
            raise ValueError("CoDi requires 0 < beta_1 < beta_T < 1.")
        for key in ("encoder_dim_con", "encoder_dim_dis"):
            values = config[key]
            if not isinstance(values, (list, tuple)) or len(values) < 2:
                raise ValueError(f"CoDi {key} must contain at least two positive integer widths.")
            config[key] = [self._positive_int(f"{key} item", value) for value in values]
        if config["activation"] not in {"elu", "relu", "lrelu", "swish", "tanh", "softplus"}:
            raise ValueError("CoDi activation is not supported by the pinned source.")
        if config["mean_type"] not in {"xprev", "xstart", "epsilon"}:
            raise ValueError("CoDi mean_type must be 'xprev', 'xstart', or 'epsilon'.")
        if config["var_type"] not in {"fixedlarge", "fixedsmall"}:
            raise ValueError("CoDi var_type must be 'fixedlarge' or 'fixedsmall'.")
        return config

    @staticmethod
    def _index_list(info: dict[str, Any], key: str) -> list[int]:
        values = info.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
        ):
            raise ValueError(f"CoDi dataset metadata {key} must be a list of non-negative integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"CoDi dataset metadata {key} contains duplicate indices.")
        return values

    def _validate_dataset(self, dataset: str) -> dict[str, Any]:
        import numpy as np
        import pandas as pd

        dataset_path = Path(dataset)
        if (
            not dataset
            or "\x00" in dataset
            or any(separator in dataset for separator in ("/", "\\", ":"))
            or dataset_path.is_absolute()
            or len(dataset_path.parts) != 1
            or dataset_path.name in {".", ".."}
        ):
            raise ValueError("CoDi dataset must be a single safe dataset identifier, not a path.")
        data_root = (self.upstream_root / "data").resolve()
        data_dir = data_root / dataset
        if data_dir.is_symlink() or not data_dir.is_dir():
            raise FileNotFoundError(f"CoDi processed dataset directory is missing: {data_dir}")
        if not data_dir.resolve(strict=True).is_relative_to(data_root):
            raise ValueError(f"CoDi dataset directory escapes the expected data root: {data_dir}")
        info_path = data_dir / "info.json"
        if info_path.is_symlink() or not info_path.is_file():
            raise FileNotFoundError(f"CoDi dataset metadata is missing: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError(f"CoDi dataset metadata must be a JSON object: {info_path}")
        task_type = info.get("task_type")
        if task_type not in {"binclass", "multiclass", "regression"}:
            raise ValueError(f"CoDi does not support task_type={task_type!r}.")
        num_idx = self._index_list(info, "num_col_idx")
        cat_idx = self._index_list(info, "cat_col_idx")
        target_idx = self._index_list(info, "target_col_idx")
        if len(target_idx) != 1:
            raise ValueError("CoDi supports exactly one target column.")

        frames: dict[str, Any] = {}
        for split in ("train", "test"):
            csv_path = data_dir / f"{split}.csv"
            if csv_path.is_symlink() or not csv_path.is_file():
                raise FileNotFoundError(f"CoDi requires a regular processed CSV: {csv_path}")
            frame = pd.read_csv(csv_path)
            if frame.empty or not frame.columns.is_unique:
                raise ValueError(f"CoDi requires a non-empty CSV with unique columns: {csv_path}")
            if bool(frame.isna().any().any()):
                raise ValueError(
                    f"CoDi rejects missing values in {csv_path}; run the centralized train-split-fitted imputer first."
                )
            frames[split] = frame
        if list(frames["train"].columns) != list(frames["test"].columns):
            raise ValueError("CoDi train and test CSV columns must match exactly and in order.")
        column_count = len(frames["train"].columns)
        all_indices = num_idx + cat_idx + target_idx
        if len(all_indices) != len(set(all_indices)) or set(all_indices) != set(range(column_count)):
            raise ValueError("CoDi numerical, categorical, and target indices must partition every CSV column.")
        idx_name_mapping = info.get("idx_name_mapping")
        if not isinstance(idx_name_mapping, dict):
            raise ValueError("CoDi dataset metadata requires idx_name_mapping.")
        try:
            expected_columns = [idx_name_mapping[str(index)] for index in range(column_count)]
        except KeyError as exc:
            raise ValueError("CoDi idx_name_mapping must name every column index.") from exc
        if any(not isinstance(name, str) for name in expected_columns) or len(set(expected_columns)) != column_count:
            raise ValueError("CoDi idx_name_mapping values must be unique strings.")
        if list(frames["train"].columns) != expected_columns:
            raise ValueError("CoDi CSV columns do not match idx_name_mapping in canonical order.")

        continuous_idx = [*num_idx, *target_idx] if task_type == "regression" else list(num_idx)
        discrete_idx = list(cat_idx) if task_type == "regression" else [*cat_idx, *target_idx]
        if not continuous_idx or not discrete_idx:
            raise ValueError(
                "CoDi requires at least one continuous diffusion column and one discrete diffusion column after "
                "task-aware target assignment."
            )
        for index in continuous_idx:
            for split, frame in frames.items():
                try:
                    values = pd.to_numeric(frame.iloc[:, index], errors="raise").to_numpy(dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"CoDi continuous column {expected_columns[index]!r} is not numerical.") from exc
                if not bool(np.isfinite(values).all()):
                    raise ValueError(f"CoDi continuous column {expected_columns[index]!r} contains non-finite values.")
            if frames["train"].iloc[:, index].nunique(dropna=False) < 2:
                raise ValueError(
                    f"CoDi continuous training column {expected_columns[index]!r} must contain at least two values."
                )
        target_values = frames["train"].iloc[:, target_idx[0]]
        target_cardinality = int(target_values.nunique(dropna=False))
        if task_type == "binclass" and target_cardinality != 2:
            raise ValueError("CoDi binary classification requires exactly two training target classes.")
        if task_type == "multiclass" and target_cardinality < 3:
            raise ValueError("CoDi multiclass classification requires at least three training target classes.")

        arrays_by_split: dict[str, dict[str, Any]] = {"train": {}, "test": {}}
        array_groups = {"X_num": num_idx, "X_cat": cat_idx, "y": target_idx}
        for split, frame in frames.items():
            for name, indices in array_groups.items():
                path = data_dir / f"{name}_{split}.npy"
                required = name == "y" or bool(indices)
                if not required:
                    if path.exists():
                        raise ValueError(f"CoDi found unexpected {name} data for an empty metadata group: {path}")
                    continue
                if path.is_symlink() or not path.is_file():
                    raise FileNotFoundError(f"CoDi requires a regular processed NumPy array: {path}")
                try:
                    array = np.load(path, allow_pickle=False)
                except ValueError as exc:
                    raise ValueError(f"CoDi requires non-pickle NumPy arrays: {path}") from exc
                if name == "y" and array.ndim == 1:
                    array = array.reshape(-1, 1)
                if array.ndim != 2 or array.shape != (len(frame), len(indices)):
                    raise ValueError(
                        f"CoDi array shape mismatch for {path}: expected {(len(frame), len(indices))}, "
                        f"observed {array.shape}."
                    )
                if bool(pd.isna(array).any()):
                    raise ValueError(
                        f"CoDi rejects missing values in {path}; run the centralized train-split-fitted imputer first."
                    )
                csv_values = frame.iloc[:, indices].to_numpy()
                numerical_group = name == "X_num" or (name == "y" and task_type == "regression")
                if numerical_group:
                    try:
                        array_numeric = array.astype(np.float64)
                        csv_numeric = csv_values.astype(np.float64)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"CoDi numerical array cannot be converted to float64: {path}") from exc
                    if not bool(np.isfinite(array_numeric).all()) or not np.allclose(
                        array_numeric, csv_numeric, rtol=1e-6, atol=1e-8
                    ):
                        raise ValueError(f"CoDi numerical array does not match its canonical CSV columns: {path}")
                elif not np.array_equal(array.astype(str), csv_values.astype(str)):
                    raise ValueError(f"CoDi categorical array does not exactly match its canonical CSV columns: {path}")
                arrays_by_split[split][name] = array
            declared_rows = info.get(f"{split}_num")
            if declared_rows is not None and (
                not isinstance(declared_rows, int) or isinstance(declared_rows, bool) or declared_rows != len(frame)
            ):
                raise ValueError(
                    f"CoDi {split}_num metadata does not match the processed CSV: "
                    f"declared={declared_rows!r}, observed={len(frame)}"
                )
        return {
            "data_dir": data_dir,
            "info": info,
            "task_type": task_type,
            "columns": expected_columns,
            "continuous_columns": [expected_columns[index] for index in continuous_idx],
            "discrete_columns": [expected_columns[index] for index in discrete_idx],
            "categorical_columns": [expected_columns[index] for index in cat_idx],
            "train_rows": len(frames["train"]),
            "categorical_domains": {
                expected_columns[index]: sorted(frames["train"].iloc[:, index].astype(str).unique().tolist())
                for index in discrete_idx
            },
        }

    def _launcher_args(self, config: dict[str, Any]) -> list[str]:
        return [
            "--training-batch-size",
            str(config["training_batch_size"]),
            "--eval-batch-size",
            str(config["eval_batch_size"]),
            "--diffusion-steps",
            str(config["T"]),
            "--beta-1",
            str(config["beta_1"]),
            "--beta-T",
            str(config["beta_T"]),
            "--lr-con",
            str(config["lr_con"]),
            "--lr-dis",
            str(config["lr_dis"]),
            "--epochs",
            str(config["total_epochs_both"]),
            "--grad-clip",
            str(config["grad_clip"]),
            "--sample-step",
            str(config["sample_step"]),
            "--lambda-con",
            str(config["lambda_con"]),
            "--lambda-dis",
            str(config["lambda_dis"]),
            "--nf-con",
            str(config["nf_con"]),
            "--nf-dis",
            str(config["nf_dis"]),
            "--encoder-dim-con",
            *[str(value) for value in config["encoder_dim_con"]],
            "--encoder-dim-dis",
            *[str(value) for value in config["encoder_dim_dis"]],
            "--activation",
            str(config["activation"]),
            "--mean-type",
            str(config["mean_type"]),
            "--var-type",
            str(config["var_type"]),
            "--num-threads",
            str(config["num_threads"]),
        ]

    def _run_codi(self, args: list[str], *, seed: int) -> None:
        launcher = self.repo_root / "standardized_tabular_diffusion" / "compat" / "codi_launcher.py"
        if not launcher.is_file():
            raise FileNotFoundError(f"CoDi compatibility launcher is missing: {launcher}")
        self._run_python(
            [str(launcher), *args, "--seed", str(seed)],
            self.upstream_root,
            env={"PYTHONHASHSEED": str(seed)},
        )

    @staticmethod
    def _metadata_path(spec: RunSpec) -> Path:
        return spec.output_dir / "codi-model-metadata.json"

    @staticmethod
    def _checkpoint_paths(spec: RunSpec) -> tuple[Path, Path]:
        root = spec.output_dir / "ckpt" / spec.dataset
        return root / "model_con.pt", root / "model_dis.pt"

    def _prepare_output(self, spec: RunSpec) -> None:
        if spec.output_dir.is_symlink():
            raise ValueError(f"CoDi output_dir must not be a symlink: {spec.output_dir}")
        self._ensure_output_dir(spec)

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._validate_extra(spec, action="train")
        self._prepare_output(spec)
        dataset = self._validate_dataset(spec.dataset)
        source = validate_upstream_source(self.model_name, self.upstream_root)
        config = self._training_config(spec.extra)
        self._run_codi(
            [
                "--action",
                "train",
                "--dataname",
                spec.dataset,
                "--output-dir",
                str(spec.output_dir.resolve()),
                "--device",
                spec.device,
                *self._launcher_args(config),
            ],
            seed=spec.seed,
        )
        source_after = validate_upstream_source(self.model_name, self.upstream_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("CoDi source identity changed during training.")
        checkpoint_con, checkpoint_dis = self._checkpoint_paths(spec)
        checkpoint_con = self._validate_trusted_executable_artifact(
            spec, checkpoint_con, format_name="PyTorch continuous-model checkpoint"
        )
        checkpoint_dis = self._validate_trusted_executable_artifact(
            spec, checkpoint_dis, format_name="PyTorch discrete-model checkpoint"
        )
        atomic_write_json(
            self._metadata_path(spec),
            {
                "schema_version": 1,
                "model": self.model_name,
                "dataset": spec.dataset,
                "seed": spec.seed,
                "device": spec.device,
                "checkpoint_con_path": str(checkpoint_con),
                "checkpoint_dis_path": str(checkpoint_dis),
                "checkpoint_con_sha256": self._sha256(checkpoint_con),
                "checkpoint_dis_sha256": self._sha256(checkpoint_dis),
                "training_config": config,
                "dataset_contract": {
                    key: dataset[key]
                    for key in ("task_type", "columns", "continuous_columns", "discrete_columns", "train_rows")
                },
                "source": source,
                "compatibility_boundary": "codi-tabsyn-adapter-runtime-v1",
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                "Training used the checksum-locked TabSyn CoDi snapshot through adapter-only runtime controls.",
                "Both trusted PyTorch state_dict checkpoints are confined to output_dir; tracked source was unchanged.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        import numpy as np
        import pandas as pd

        self._validate_extra(spec, action="sample")
        self._prepare_output(spec)
        dataset = self._validate_dataset(spec.dataset)
        source = validate_upstream_source(self.model_name, self.upstream_root)
        if spec.checkpoint_path is not None:
            raise ValueError("CoDi uses the trusted checkpoint pair inside output_dir; checkpoint_path is unsupported.")
        metadata_path = self._metadata_path(spec)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(f"CoDi training metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("CoDi training metadata does not satisfy schema version 1.")
        if metadata.get("model") != self.model_name or metadata.get("dataset") != spec.dataset:
            raise ValueError("CoDi checkpoint metadata does not match the requested model and dataset.")
        if metadata.get("source", {}).get("manifest_sha256") != source["manifest_sha256"]:
            raise ValueError("CoDi source manifest changed after training; refusing checkpoint reuse.")
        if metadata.get("dataset_contract", {}).get("columns") != dataset["columns"]:
            raise ValueError("CoDi dataset schema changed after training; refusing checkpoint reuse.")
        checkpoint_con, checkpoint_dis = self._checkpoint_paths(spec)
        checkpoint_con = self._validate_trusted_executable_artifact(
            spec, checkpoint_con, format_name="PyTorch continuous-model checkpoint"
        )
        checkpoint_dis = self._validate_trusted_executable_artifact(
            spec, checkpoint_dis, format_name="PyTorch discrete-model checkpoint"
        )
        if self._sha256(checkpoint_con) != metadata.get("checkpoint_con_sha256"):
            raise ValueError("CoDi continuous checkpoint checksum differs from its training metadata.")
        if self._sha256(checkpoint_dis) != metadata.get("checkpoint_dis_sha256"):
            raise ValueError("CoDi discrete checkpoint checksum differs from its training metadata.")
        config = self._training_config(metadata.get("training_config", {}))
        if "num_threads" in spec.extra:
            config["num_threads"] = self._positive_int("num_threads", spec.extra["num_threads"])
        sample_path = (spec.output_dir / "samples.csv").resolve()
        args = [
            "--action",
            "sample",
            "--dataname",
            spec.dataset,
            "--output-dir",
            str(spec.output_dir.resolve()),
            "--device",
            spec.device,
            "--save-path",
            str(sample_path),
            *self._launcher_args(config),
        ]
        if spec.num_samples is not None:
            if not isinstance(spec.num_samples, int) or isinstance(spec.num_samples, bool) or spec.num_samples <= 0:
                raise ValueError("CoDi num_samples must be a positive integer.")
            args.extend(["--num-samples", str(spec.num_samples)])
        self._run_codi(args, seed=spec.seed)
        source_after = validate_upstream_source(self.model_name, self.upstream_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("CoDi source identity changed during sampling.")
        if sample_path.is_symlink() or not sample_path.is_file():
            raise FileNotFoundError(f"CoDi did not produce the expected sample file: {sample_path}")
        frame = pd.read_csv(sample_path)
        expected_rows = spec.num_samples if spec.num_samples is not None else dataset["train_rows"]
        if len(frame) != expected_rows:
            raise RuntimeError(f"CoDi produced {len(frame)} rows instead of the expected {expected_rows}.")
        if list(frame.columns) != dataset["columns"]:
            raise RuntimeError("CoDi output columns do not match the canonical training schema.")
        if bool(frame.isna().any().any()):
            raise RuntimeError("CoDi produced missing values.")
        for column in dataset["continuous_columns"]:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            if not bool(np.isfinite(values).all()):
                raise RuntimeError(f"CoDi produced non-finite values in continuous column {column!r}.")
        for column, allowed in dataset["categorical_domains"].items():
            observed = set(frame[column].astype(str).unique().tolist())
            if not observed.issubset(set(allowed)):
                raise RuntimeError(f"CoDi produced values outside the fitted domain for {column!r}.")
        atomic_write_json(
            spec.output_dir / "codi-sample-metadata.json",
            {
                "schema_version": 1,
                "model": self.model_name,
                "dataset": spec.dataset,
                "seed": spec.seed,
                "rows": len(frame),
                "columns": list(frame.columns),
                "sample_sha256": self._sha256(sample_path),
                "checkpoint_con_sha256": metadata["checkpoint_con_sha256"],
                "checkpoint_dis_sha256": metadata["checkpoint_dis_sha256"],
                "sampling_config": {"T": config["T"], "num_threads": config["num_threads"]},
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                "Sampling verified source identity, both trusted checkpoint hashes, exact rows, schema, domains, and finite output."
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
