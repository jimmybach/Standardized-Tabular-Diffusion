from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import validate_upstream_source


class _TabSynVendoredBaselineAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    upstream_dirname = "TabSyn-main"
    tabsyn_method_name: str

    def _build_cli_args(self, spec: RunSpec, *, mode: str) -> list[str]:
        args = [
            "main.py",
            "--method",
            self.tabsyn_method_name,
            "--mode",
            mode,
            "--dataname",
            spec.dataset,
            "--gpu",
            str(spec.extra.get("gpu", 0)),
        ]

        passthrough_map = {
            "num_epochs": "--num_epochs",
            "batch_size": "--batch_size",
            "bs": "--bs",
            "training_batch_size": "--training_batch_size",
            "eval_batch_size": "--eval_batch_size",
            "T": "--T",
            "beta_1": "--beta_1",
            "beta_T": "--beta_T",
            "lr_con": "--lr_con",
            "lr_dis": "--lr_dis",
            "total_epochs_both": "--total_epochs_both",
            "sample_step": "--sample_step",
            "lambda_con": "--lambda_con",
            "lambda_dis": "--lambda_dis",
            "nf_con": "--nf_con",
            "nf_dis": "--nf_dis",
            "encoder_dim_con": "--encoder_dim_con",
            "encoder_dim_dis": "--encoder_dim_dis",
            "steps": "--steps",
        }
        for key, flag in passthrough_map.items():
            if spec.extra.get(key) is not None:
                args.extend([flag, str(spec.extra[key])])

        if mode == "sample":
            args.extend(["--save_path", str((spec.output_dir / "samples.csv").resolve())])
            if spec.num_samples is not None:
                args.extend(["--num-samples", str(spec.num_samples)])
        return args

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        self._run_python(self._build_cli_args(spec, mode="train"), self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Training dispatched through TabSyn vendored baseline method `{self.tabsyn_method_name}`.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = (spec.output_dir / "samples.csv").resolve()
        self._run_python(self._build_cli_args(spec, mode="sample"), self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                f"Sampling dispatched through TabSyn vendored baseline method `{self.tabsyn_method_name}`.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


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


class CoDiAdapter(_TabSynVendoredBaselineAdapter):
    model_name = "codi"
    tabsyn_method_name = "codi"
