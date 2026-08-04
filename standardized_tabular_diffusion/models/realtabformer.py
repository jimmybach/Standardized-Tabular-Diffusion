from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import random
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import (
    SampleFileEvaluatorMixin,
    disable_torchvision_for_transformers,
)
from standardized_tabular_diffusion.models.base import BaseModelAdapter

PACKAGE_NAME = "realtabformer"
PACKAGE_VERSION = "0.2.4"
UPSTREAM_REPOSITORY = "https://github.com/worldbank/REaLTabFormer"
UPSTREAM_TAG = "v0.2.4"
UPSTREAM_COMMIT = "73f239643f9ea5abc877f685ce927e986302ac2d"
UPSTREAM_TREE = "aa4431468f040fc485f82e7e15238c57eef05753"
WHEEL_MANIFEST_PATH = (
    Path("standardized_tabular_diffusion") / "resources" / "upstream" / "realtabformer-wheel-manifest.json"
)

EXPECTED_RUNTIME_VERSIONS = {
    "accelerate": "1.1.1",
    "datasets": "3.1.0",
    "joblib": "1.5.3",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "realtabformer": PACKAGE_VERSION,
    "safetensors": "0.4.5",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "sentencepiece": "0.2.0",
    "shapely": "2.0.6",
    "tokenizers": "0.20.3",
    "torch": "2.3.0",
    "tqdm": "4.66.5",
    "transformers": "4.46.3",
}

_CONSTRUCTOR_KEYS = {
    "epochs",
    "batch_size",
    "train_size",
    "early_stopping_patience",
    "early_stopping_threshold",
    "mask_rate",
    "numeric_nparts",
    "numeric_precision",
    "numeric_max_len",
}
_TRAINING_ARGUMENT_KEYS = {
    "gradient_accumulation_steps",
    "logging_steps",
    "save_steps",
    "eval_steps",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "lr_scheduler_type",
    "max_grad_norm",
    "disable_tqdm",
    "report_to",
    "save_total_limit",
}
_FIT_KEYS = {
    "num_bootstrap",
    "frac",
    "frac_max_data",
    "qt_max",
    "qt_max_default",
    "qt_interval",
    "qt_interval_unique",
    "quantile",
    "n_critic",
    "n_critic_stop",
    "gen_rounds",
    "sensitivity_max_col_nums",
    "use_ks",
    "full_sensitivity",
    "sensitivity_orig_frac_multiple",
    "orig_samples_rounds",
    "load_from_best_mean_sensitivity",
    "save_full_every_epoch",
    "target_col",
}
_COMMON_INTERNAL_KEYS = {"action_extras", "config", "dataset_spec", "evaluation", "tags"}
_TRAIN_INTERNAL_KEYS = _COMMON_INTERNAL_KEYS | {
    "max_train_rows",
    "tabular_config",
    "training_args",
    "gen_kwargs",
}
_SAMPLE_KEYS = {
    "allow_unsafe_external_checkpoint",
    "checkpoint_metadata_path",
    "constrain_tokens_gen",
    "continuous_empty_limit",
    "forced_decoder_ids",
    "gen_batch",
    "generation_kwargs",
    "seed_input",
    "suppress_tokens",
} | _COMMON_INTERNAL_KEYS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_version(package_name: str) -> str:
    observed = version(package_name)
    return observed.partition("+")[0] if package_name == "torch" else observed


def _load_wheel_manifest(repo_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / WHEEL_MANIFEST_PATH
    manifest = read_json(manifest_path)
    if manifest.get("model_id") != "realtabformer":
        raise RuntimeError("REaLTabFormer wheel manifest has an unexpected model identity")
    package = manifest.get("package", {})
    source = manifest.get("source", {})
    if (
        package.get("name") != PACKAGE_NAME
        or package.get("version") != PACKAGE_VERSION
        or source.get("repository") != UPSTREAM_REPOSITORY
        or source.get("commit") != UPSTREAM_COMMIT
        or source.get("tree") != UPSTREAM_TREE
    ):
        raise RuntimeError("REaLTabFormer wheel manifest does not match the adapter's locked release")
    return manifest, manifest_path


def verify_realtabformer_distribution(repo_root: Path) -> dict[str, Any]:
    """Verify every hashed installed file from the locked official wheel."""

    manifest, manifest_path = _load_wheel_manifest(repo_root)
    try:
        installed = distribution(PACKAGE_NAME)
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            'REaLTabFormer requires the locked official package; install "standardized-tabular-diffusion[realtabformer]".'
        ) from exc
    if installed.version != PACKAGE_VERSION:
        raise RuntimeError(
            f"REaLTabFormer package version mismatch: expected {PACKAGE_VERSION}, observed {installed.version}."
        )

    distribution_root = Path(installed.locate_file("")).resolve()
    verified: list[dict[str, Any]] = []
    for record in manifest["installed_files"]:
        relative = Path(record["path"])
        installed_path = Path(installed.locate_file(relative))
        if installed_path.is_symlink() or not installed_path.is_file():
            raise RuntimeError(f"Locked REaLTabFormer distribution file is missing or symlinked: {relative}")
        resolved = installed_path.resolve()
        if not resolved.is_relative_to(distribution_root):
            raise RuntimeError(f"REaLTabFormer distribution file escaped its installation root: {resolved}")
        observed_bytes = resolved.stat().st_size
        observed_sha256 = _sha256_file(resolved)
        if observed_bytes != record["bytes"] or observed_sha256 != record["sha256"]:
            raise RuntimeError(
                f"Installed REaLTabFormer file differs from the locked official wheel: {relative}; "
                f"expected bytes/hash={record['bytes']}/{record['sha256']}, "
                f"observed={observed_bytes}/{observed_sha256}"
            )
        verified.append({"path": relative.as_posix(), "bytes": observed_bytes, "sha256": observed_sha256})

    observed_versions: dict[str, str] = {}
    for package_name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        try:
            observed = _normalized_version(package_name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"REaLTabFormer validated runtime dependency is missing: {package_name}=={expected}"
            ) from exc
        if observed != expected:
            raise RuntimeError(
                f"REaLTabFormer runtime version mismatch for {package_name}: expected {expected}, observed {observed}"
            )
        observed_versions[package_name] = observed

    return {
        "authority": "method-author",
        "distribution_form": "package",
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "wheel_filename": manifest["package"]["filename"],
        "wheel_sha256": manifest["package"]["sha256"],
        "wheel_bytes": manifest["package"]["bytes"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "distribution_root": str(distribution_root),
        "installed_files_verified": len(verified),
        "installed_files": verified,
        "runtime_versions": observed_versions,
        "license": manifest["source"]["license"],
        "license_sha256": manifest["source"]["license_sha256"],
    }


@contextlib.contextmanager
def _weights_only_torch_load(torch_module: Any) -> Iterator[None]:
    """Force the official loader's state-dict read into PyTorch's restricted mode."""

    original_load = torch_module.load

    def safe_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = True
        return original_load(*args, **kwargs)

    torch_module.load = safe_load
    try:
        yield
    finally:
        torch_module.load = original_load


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PermissionError(f"REaLTabFormer model directory must not contain symlinks: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not records:
        raise RuntimeError(f"REaLTabFormer model directory contains no files: {root}")
    return records


class REaLTabFormerAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "realtabformer"
    upstream_dirname = "."
    package_name = PACKAGE_NAME
    package_version = PACKAGE_VERSION
    upstream_commit = UPSTREAM_COMMIT
    metadata_filename = "realtabformer-model-metadata.json"

    def _model_root(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / "realtabformer_model")

    def _metadata_path(self, spec: RunSpec) -> Path:
        configured = spec.extra.get("checkpoint_metadata_path")
        return Path(configured) if configured is not None else spec.output_dir / self.metadata_filename

    def _resolve_saved_model_dir(self, model_root: Path) -> Path:
        if model_root.is_dir() and model_root.name.startswith("id"):
            return model_root
        if not model_root.is_dir():
            raise FileNotFoundError(f"REaLTabFormer model root does not exist: {model_root}")
        candidates = sorted(path for path in model_root.glob("id*") if path.is_dir() and not path.is_symlink())
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one saved REaLTabFormer id directory under {model_root}, observed {len(candidates)}"
            )
        return candidates[0]

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("REaLTabFormer requires dataset_spec.train_data_path")
        if dataset_spec.task_type not in {"classification", "regression"}:
            raise ValueError("REaLTabFormer supports classification and regression DatasetSpecs")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("REaLTabFormer requires exactly one target column")
        if len(set(dataset_spec.column_names)) != len(dataset_spec.column_names):
            raise ValueError("REaLTabFormer requires unique column names")

        target = dataset_spec.target_columns[0]
        declared_features = set(dataset_spec.numerical_columns) | set(dataset_spec.categorical_columns)
        if set(dataset_spec.numerical_columns) & set(dataset_spec.categorical_columns):
            raise ValueError("REaLTabFormer numerical and categorical feature roles must be disjoint")
        if declared_features | {target} != set(dataset_spec.column_names):
            raise ValueError("REaLTabFormer DatasetSpec roles must cover every column exactly once")

        frame = pd.read_csv(dataset_spec.train_data_path)
        if list(frame.columns) != dataset_spec.column_names:
            raise ValueError(
                "REaLTabFormer training CSV columns must exactly match the canonical ordered DatasetSpec columns"
            )
        if frame.empty:
            raise ValueError("REaLTabFormer requires at least one training row")
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "REaLTabFormer does not accept missing values in the standardized adapter. "
                f"Run the centralized train-only mean/mode imputer first; observed: {observed}"
            )

        numeric = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numeric.append(target)
        for column in numeric:
            converted = pd.to_numeric(frame[column], errors="raise")
            values = converted.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"REaLTabFormer requires finite numerical values; invalid column: {column}")
            frame[column] = converted

        categorical = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            categorical.append(target)
        domains: dict[str, list[str]] = {}
        for column in categorical:
            frame[column] = frame[column].astype(str)
            domains[column] = sorted(frame[column].unique().tolist())
            if not domains[column]:
                raise ValueError(f"REaLTabFormer categorical column has an empty training domain: {column}")
        return frame[dataset_spec.column_names].copy(), domains

    def _limit_training_frame(self, frame: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        configured = spec.extra.get("max_train_rows")
        if configured is None:
            return frame
        max_rows = int(configured)
        if max_rows <= 0:
            raise ValueError("REaLTabFormer max_train_rows must be positive")
        if len(frame) <= max_rows:
            return frame
        return frame.sample(n=max_rows, random_state=spec.seed).reset_index(drop=True)

    def _validate_train_controls(self, spec: RunSpec) -> None:
        allowed = _CONSTRUCTOR_KEYS | _TRAINING_ARGUMENT_KEYS | _FIT_KEYS | _TRAIN_INTERNAL_KEYS
        unknown = sorted(set(spec.extra) - allowed)
        if unknown:
            raise ValueError(f"Unknown REaLTabFormer training controls: {unknown}")

    def _validate_sample_controls(self, spec: RunSpec) -> None:
        unknown = sorted(set(spec.extra) - _SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"Unknown REaLTabFormer sampling controls: {unknown}")

    def _import_runtime(self) -> tuple[type[Any], type[Any], Any, dict[str, Any]]:
        package = verify_realtabformer_distribution(self.repo_root)
        with disable_torchvision_for_transformers():
            module = importlib.import_module("realtabformer")
            implementation = importlib.import_module("realtabformer.realtabformer")
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        model_class = module.REaLTabFormer
        if model_class is not implementation.REaLTabFormer:
            raise RuntimeError("REaLTabFormer public export does not match the locked official implementation class")
        return model_class, transformers.GPT2Config, torch, package

    def _tabular_config(self, spec: RunSpec, config_class: type[Any]) -> Any | None:
        payload = spec.extra.get("tabular_config")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise TypeError("REaLTabFormer tabular_config must be a JSON object")
        forbidden = {"vocab_size", "bos_token_id", "eos_token_id"} & set(payload)
        if forbidden:
            raise ValueError(f"REaLTabFormer owns these data-derived tabular_config fields: {sorted(forbidden)}")
        config = config_class(**payload)
        if config.n_layer <= 0 or config.n_head <= 0 or config.n_embd <= 0 or config.n_positions <= 0:
            raise ValueError("REaLTabFormer GPT-2 layer, head, embedding, and position dimensions must be positive")
        if config.n_embd % config.n_head:
            raise ValueError("REaLTabFormer GPT-2 n_embd must be divisible by n_head")
        return config

    def _constructor_kwargs(self, spec: RunSpec, config_class: type[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime_root = spec.output_dir / "realtabformer_runtime"
        kwargs: dict[str, Any] = {
            "model_type": "tabular",
            "tabular_config": self._tabular_config(spec, config_class),
            "checkpoints_dir": str(runtime_root / "checkpoints"),
            "samples_save_dir": str(runtime_root / "samples"),
            "full_save_dir": str(runtime_root / "full-save"),
            "epochs": int(spec.extra.get("epochs", 1000)),
            "batch_size": int(spec.extra.get("batch_size", 8)),
            "random_state": int(spec.seed),
            "train_size": float(spec.extra.get("train_size", 1.0)),
            "early_stopping_patience": int(spec.extra.get("early_stopping_patience", 5)),
            "early_stopping_threshold": float(spec.extra.get("early_stopping_threshold", 0.0)),
            "mask_rate": float(spec.extra.get("mask_rate", 0.0)),
            "numeric_nparts": int(spec.extra.get("numeric_nparts", 1)),
            "numeric_precision": int(spec.extra.get("numeric_precision", 4)),
            "numeric_max_len": int(spec.extra.get("numeric_max_len", 10)),
        }
        if kwargs["epochs"] <= 0 or kwargs["batch_size"] <= 0:
            raise ValueError("REaLTabFormer epochs and batch_size must be positive")
        if not 0 < kwargs["train_size"] <= 1:
            raise ValueError("REaLTabFormer train_size must be in (0, 1]")
        if kwargs["early_stopping_patience"] < 0 or kwargs["early_stopping_threshold"] < 0:
            raise ValueError("REaLTabFormer early-stopping controls must be non-negative")
        if not 0 <= kwargs["mask_rate"] <= 1:
            raise ValueError("REaLTabFormer mask_rate must be in [0, 1]")
        if kwargs["numeric_nparts"] <= 0 or kwargs["numeric_precision"] < 0 or kwargs["numeric_max_len"] <= 0:
            raise ValueError("REaLTabFormer numeric transform controls are out of range")

        training_args = spec.extra.get("training_args", {})
        if not isinstance(training_args, dict):
            raise TypeError("REaLTabFormer training_args must be a JSON object")
        unknown_training_args = sorted(set(training_args) - _TRAINING_ARGUMENT_KEYS)
        if unknown_training_args:
            raise ValueError(f"Unknown REaLTabFormer TrainingArguments controls: {unknown_training_args}")
        merged_training_args = {key: spec.extra[key] for key in _TRAINING_ARGUMENT_KEYS if key in spec.extra}
        merged_training_args.update(training_args)
        merged_training_args.setdefault("gradient_accumulation_steps", 4)
        merged_training_args.setdefault("logging_steps", 100)
        merged_training_args.setdefault("report_to", "none")
        if merged_training_args["report_to"] not in {"none", None}:
            raise ValueError("REaLTabFormer external TrainingArguments reporting is disabled by default")
        merged_training_args.setdefault("seed", int(spec.seed))
        merged_training_args.setdefault("data_seed", int(spec.seed))
        merged_training_args.setdefault("dataloader_num_workers", 0)
        if not spec.device.startswith("cuda"):
            merged_training_args.setdefault("use_cpu", True)
        kwargs.update(merged_training_args)

        recorded = dict(kwargs)
        tabular_config = recorded.get("tabular_config")
        recorded["tabular_config"] = (
            None if tabular_config is None else json.loads(json.dumps(tabular_config.to_dict(), allow_nan=False))
        )
        for name in ("checkpoints_dir", "samples_save_dir", "full_save_dir"):
            recorded[name] = str(Path(recorded[name]).relative_to(spec.output_dir))
        return kwargs, recorded

    def _fit_kwargs(self, spec: RunSpec, dataset_spec: DatasetSpec) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "device": spec.device,
            "num_bootstrap": int(spec.extra.get("num_bootstrap", 500)),
            "frac": float(spec.extra.get("frac", 0.165)),
            "frac_max_data": int(spec.extra.get("frac_max_data", 10000)),
            "qt_max": spec.extra.get("qt_max", 0.05),
            "qt_max_default": float(spec.extra.get("qt_max_default", 0.05)),
            "qt_interval": int(spec.extra.get("qt_interval", 100)),
            "qt_interval_unique": int(spec.extra.get("qt_interval_unique", 100)),
            "quantile": float(spec.extra.get("quantile", 0.95)),
            "n_critic": int(spec.extra.get("n_critic", 5)),
            "n_critic_stop": int(spec.extra.get("n_critic_stop", 2)),
            "gen_rounds": int(spec.extra.get("gen_rounds", 3)),
            "sensitivity_max_col_nums": int(spec.extra.get("sensitivity_max_col_nums", 20)),
            "use_ks": bool(spec.extra.get("use_ks", False)),
            "full_sensitivity": bool(spec.extra.get("full_sensitivity", False)),
            "sensitivity_orig_frac_multiple": int(spec.extra.get("sensitivity_orig_frac_multiple", 4)),
            "orig_samples_rounds": int(spec.extra.get("orig_samples_rounds", 5)),
            "load_from_best_mean_sensitivity": bool(spec.extra.get("load_from_best_mean_sensitivity", False)),
            "save_full_every_epoch": int(spec.extra.get("save_full_every_epoch", 5)),
        }
        target_col = spec.extra.get("target_col")
        if target_col is not None:
            if target_col != dataset_spec.target_columns[0]:
                raise ValueError("REaLTabFormer target_col must equal the DatasetSpec target column")
            kwargs["target_col"] = target_col
        gen_kwargs = spec.extra.get("gen_kwargs")
        if gen_kwargs is not None:
            if not isinstance(gen_kwargs, dict):
                raise TypeError("REaLTabFormer gen_kwargs must be a JSON object")
            kwargs["gen_kwargs"] = dict(gen_kwargs)
        if kwargs["num_bootstrap"] < 0 or kwargs["frac_max_data"] <= 0:
            raise ValueError("REaLTabFormer bootstrap and data-size controls are out of range")
        if not 0 < kwargs["frac"] < 1 or not 0 < kwargs["quantile"] < 1:
            raise ValueError("REaLTabFormer frac and quantile must be in (0, 1)")
        for name in (
            "qt_interval",
            "qt_interval_unique",
            "n_critic_stop",
            "gen_rounds",
            "sensitivity_max_col_nums",
            "sensitivity_orig_frac_multiple",
            "orig_samples_rounds",
        ):
            if kwargs[name] <= 0:
                raise ValueError(f"REaLTabFormer {name} must be positive")
        if kwargs["n_critic"] > 0 and kwargs["num_bootstrap"] <= 0:
            raise ValueError("REaLTabFormer sensitivity training requires positive num_bootstrap")
        return kwargs

    def _seed_runtime(self, seed: int, torch_module: Any) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        self._validate_train_controls(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        full_frame, categorical_domains = self._load_training_frame(dataset_spec)
        train_frame = self._limit_training_frame(full_frame, spec)
        model_class, config_class, torch, package = self._import_runtime()
        constructor_kwargs, recorded_constructor = self._constructor_kwargs(spec, config_class)
        fit_kwargs = self._fit_kwargs(spec, dataset_spec)

        model_root = self._model_root(spec)
        if not model_root.resolve().is_relative_to(spec.output_dir.resolve()):
            raise PermissionError("REaLTabFormer training artifacts must remain inside output_dir")
        if model_root.exists() and any(model_root.iterdir()):
            raise FileExistsError(
                f"REaLTabFormer model root is not empty: {model_root}. Use a new output_dir for a reproducible run."
            )
        model_root.mkdir(parents=True, exist_ok=True)
        before = {path.resolve() for path in model_root.glob("id*") if path.is_dir()}

        self._seed_runtime(spec.seed, torch)
        model = model_class(**constructor_kwargs)
        model.fit(train_frame, **fit_kwargs)

        # v0.2.4 keeps full_save_dir as Path but omits it from the two Path-to-string
        # conversions in save(). Serializing the same path string restores the intended
        # official configuration write without changing model or training state.
        model.full_save_dir = str(model.full_save_dir)
        model.save(str(model_root))
        after = {path.resolve() for path in model_root.glob("id*") if path.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(f"Expected one new REaLTabFormer saved model directory, observed {len(created)}")
        saved_model_dir = created[0]
        files = _directory_manifest(saved_model_dir)

        training_record = {
            "constructor": recorded_constructor,
            "fit": fit_kwargs,
            "max_train_rows": spec.extra.get("max_train_rows"),
        }
        metadata = {
            "schema_version": "1.0",
            "model_id": self.model_name,
            "dataset": dataset_spec.name,
            "task_type": dataset_spec.task_type,
            "column_names": dataset_spec.column_names,
            "numerical_columns": dataset_spec.numerical_columns,
            "categorical_columns": dataset_spec.categorical_columns,
            "target_columns": dataset_spec.target_columns,
            "categorical_domains": categorical_domains,
            "seed": spec.seed,
            "source_rows": len(full_frame),
            "training_rows": len(train_frame),
            "training_frame_sha256": hashlib.sha256(train_frame.to_csv(index=False).encode("utf-8")).hexdigest(),
            "package": package,
            "training": training_record,
            "training_sha256": _sha256_json(training_record),
            "saved_model_dir": saved_model_dir.relative_to(spec.output_dir.resolve()).as_posix(),
            "saved_model_files": files,
            "saved_model_manifest_sha256": _sha256_json(files),
            "compatibility_boundaries": [
                "output-local-official-runtime-directories",
                "v0.2.4-full-save-dir-json-path-serialization",
                "centralized-declared-type-coercion",
                "sampling-rng-reset",
                "weights-only-official-state-dict-load",
            ],
        }
        metadata_path = self._metadata_path(spec)
        atomic_write_json(metadata_path, metadata)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=Path(package["distribution_root"]),
            notes=[
                f"Saved official REaLTabFormer {PACKAGE_VERSION} artifacts under {saved_model_dir}.",
                f"Integrity metadata written to {metadata_path}.",
            ],
        )
        return self._write_bundle(bundle)

    def _validate_saved_model(
        self,
        spec: RunSpec,
        dataset_spec: DatasetSpec,
        model_dir: Path,
        package: dict[str, Any],
    ) -> dict[str, Any]:
        metadata_path = self._metadata_path(spec)
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing non-symlinked REaLTabFormer checkpoint metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        if metadata.get("model_id") != self.model_name or metadata.get("dataset") != dataset_spec.name:
            raise RuntimeError("REaLTabFormer checkpoint metadata identity does not match the requested run")
        for name, expected in (
            ("task_type", dataset_spec.task_type),
            ("column_names", dataset_spec.column_names),
            ("numerical_columns", dataset_spec.numerical_columns),
            ("categorical_columns", dataset_spec.categorical_columns),
            ("target_columns", dataset_spec.target_columns),
        ):
            if metadata.get(name) != expected:
                raise RuntimeError(f"REaLTabFormer checkpoint metadata {name} differs from the DatasetSpec")
        recorded_package = metadata.get("package", {})
        for name in ("package_version", "wheel_sha256", "manifest_sha256", "upstream_commit"):
            if recorded_package.get(name) != package.get(name):
                raise RuntimeError(f"REaLTabFormer checkpoint package identity mismatch: {name}")
        expected_dir = metadata.get("saved_model_dir")
        if model_dir.is_relative_to(spec.output_dir.resolve()):
            observed_dir = model_dir.relative_to(spec.output_dir.resolve()).as_posix()
            if expected_dir != observed_dir:
                raise RuntimeError("REaLTabFormer checkpoint directory differs from its recorded metadata")
        observed_files = _directory_manifest(model_dir)
        if observed_files != metadata.get("saved_model_files"):
            raise RuntimeError("REaLTabFormer saved model files differ from the recorded integrity manifest")
        if _sha256_json(observed_files) != metadata.get("saved_model_manifest_sha256"):
            raise RuntimeError("REaLTabFormer saved model manifest digest mismatch")
        return metadata

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        self._validate_sample_controls(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        train_frame, categorical_domains = self._load_training_frame(dataset_spec)
        model_class, _, torch, package = self._import_runtime()
        model_dir = self._resolve_saved_model_dir(self._model_root(spec))
        trusted_model_dir = self._validate_trusted_executable_artifact(
            spec,
            model_dir,
            format_name="REaLTabFormer state-dict directory",
            allow_directory=True,
        )
        metadata = self._validate_saved_model(spec, dataset_spec, trusted_model_dir, package)

        with _weights_only_torch_load(torch):
            model = model_class.load_from_dir(path=str(trusted_model_dir))
        if model.__class__ is not model_class:
            raise RuntimeError("Loaded REaLTabFormer checkpoint class differs from the locked official class")

        num_samples = len(train_frame) if spec.num_samples is None else int(spec.num_samples)
        if num_samples <= 0:
            raise ValueError("REaLTabFormer num_samples must be positive")
        sample_kwargs: dict[str, Any] = {"device": spec.device}
        for name in (
            "gen_batch",
            "seed_input",
            "constrain_tokens_gen",
            "continuous_empty_limit",
            "suppress_tokens",
            "forced_decoder_ids",
        ):
            if name in spec.extra:
                sample_kwargs[name] = spec.extra[name]
        generation_kwargs = spec.extra.get("generation_kwargs", {})
        if not isinstance(generation_kwargs, dict):
            raise TypeError("REaLTabFormer generation_kwargs must be a JSON object")
        sample_kwargs.update(generation_kwargs)

        self._seed_runtime(spec.seed, torch)
        sample_frame = model.sample(n_samples=num_samples, **sample_kwargs)
        if len(sample_frame) != num_samples:
            raise RuntimeError(
                f"REaLTabFormer returned {len(sample_frame)} rows for a request of {num_samples}; refusing truncation/padding"
            )
        if list(sample_frame.columns) != dataset_spec.column_names:
            if set(sample_frame.columns) != set(dataset_spec.column_names):
                raise RuntimeError("REaLTabFormer sample columns differ from the canonical DatasetSpec")
            sample_frame = sample_frame[dataset_spec.column_names].copy()
        if bool(sample_frame.isna().any().any()):
            raise RuntimeError("REaLTabFormer generated missing values, which are forbidden by the benchmark contract")
        numeric = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numeric.extend(dataset_spec.target_columns)
        for column in numeric:
            values = pd.to_numeric(sample_frame[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"REaLTabFormer generated non-finite numerical values in {column}")
        for column, domain in categorical_domains.items():
            observed = set(sample_frame[column].astype(str))
            if not observed.issubset(domain):
                raise RuntimeError(
                    f"REaLTabFormer generated out-of-domain values in {column}: {sorted(observed - set(domain))}"
                )

        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_frame, sample_path)
        atomic_write_json(
            spec.output_dir / "realtabformer-sample-metadata.json",
            {
                "schema_version": "1.0",
                "model_id": self.model_name,
                "dataset": dataset_spec.name,
                "seed": spec.seed,
                "requested_rows": num_samples,
                "sample_sha256": _sha256_file(sample_path),
                "checkpoint_manifest_sha256": metadata["saved_model_manifest_sha256"],
                "package_manifest_sha256": package["manifest_sha256"],
                "sample_kwargs": sample_kwargs,
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=Path(package["distribution_root"]),
            generated_sample_path=sample_path,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


__all__ = [
    "EXPECTED_RUNTIME_VERSIONS",
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "REaLTabFormerAdapter",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_TAG",
    "UPSTREAM_TREE",
    "verify_realtabformer_distribution",
]
