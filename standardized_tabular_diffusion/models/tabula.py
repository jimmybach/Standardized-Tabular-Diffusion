from __future__ import annotations

import contextlib
import importlib
import json
import os
import random
import signal
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import (
    SampleFileEvaluatorMixin,
    disable_torchvision_for_transformers,
    isolated_module_tree,
)
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import validate_upstream_source


class TabulaAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    """Strict adapter around the checksum-locked method-author TabuLa source."""

    model_name = "tabula"
    upstream_dirname = "."
    checkpoint_dirname = "tabula_model"
    source_environment_variable = "STANDARDIZED_TABULAR_DIFFUSION_TABULA_SOURCE"
    upstream_commit = "a7d34a94adee5a269f6807395d0040d936bb0e60"
    protocol_id = "tabula-method-author-source-parity-v1"

    def _model_root(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or spec.output_dir / self.checkpoint_dirname

    @staticmethod
    def _state_path(model_root: Path) -> Path:
        return model_root / "tabula-state.json"

    @staticmethod
    def _integrity_path(model_root: Path) -> Path:
        return model_root / "tabula-integrity.json"

    def _resolve_source_root(self, spec: RunSpec) -> tuple[Path, dict[str, Any]]:
        configured = spec.extra.get("source_dir") or os.environ.get(self.source_environment_variable)
        source_root = (
            Path(configured)
            if configured is not None
            else self.repo_root / ".cache" / "upstream-sources" / self.model_name / self.upstream_commit
        )
        try:
            source = validate_upstream_source(self.model_name, source_root)
        except (FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError(
                "TabuLa requires the checksum-locked method-author source. Run "
                "`python -m standardized_tabular_diffusion.cli materialize-model-source --model tabula`, "
                f"or provide spec.extra['source_dir']; underlying error: {exc}"
            ) from exc
        if source["upstream_commit"] != self.upstream_commit:
            raise RuntimeError("TabuLa source validation returned an unexpected commit")
        return source_root.resolve(), source

    @staticmethod
    @contextlib.contextmanager
    def _official_class(source_root: Path) -> Iterator[type[Any]]:
        with disable_torchvision_for_transformers(), isolated_module_tree(source_root, "tabula"):
            module = importlib.import_module("tabula.tabula")
            yield module.Tabula

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"TabuLa {name} must be a positive integer")
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"TabuLa {name} must be a positive integer")
        return parsed

    @staticmethod
    def _positive_float(name: str, value: Any) -> float:
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"TabuLa {name} must be finite and positive")
        return parsed

    @staticmethod
    def _roles(dataset_spec: DatasetSpec) -> tuple[list[str], list[str]]:
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("TabuLa requires exactly one target column")
        numerical = list(dataset_spec.numerical_columns)
        categorical = list(dataset_spec.categorical_columns)
        target = dataset_spec.target_columns[0]
        if target not in numerical and target not in categorical:
            (categorical if dataset_spec.task_type == "classification" else numerical).append(target)
        if set(numerical) & set(categorical) or set(numerical) | set(categorical) != set(dataset_spec.column_names):
            raise ValueError("TabuLa requires disjoint and complete declared column roles")
        return numerical, categorical

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("TabuLa requires dataset_spec.train_data_path")
        if any(
            " " in column or "," in column or "\n" in column or "\r" in column
            for column in dataset_spec.column_names
        ):
            raise ValueError("TabuLa official parsing requires column names without spaces, commas, or newlines")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        if frame.empty:
            raise ValueError("TabuLa cannot train on an empty table")
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "TabuLa requires missing values to be imputed by the train-fitted preprocessing module; "
                f"observed: {observed}"
            )
        numerical, categorical = self._roles(dataset_spec)
        for column in numerical:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"TabuLa numerical column {column!r} contains non-finite values")
            frame[column] = values
        for column in categorical:
            frame[column] = frame[column].astype(str)
        return frame

    @staticmethod
    def _limit_training_frame(frame: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        limit = spec.extra.get("max_train_rows")
        if limit is None:
            return frame
        parsed = TabulaAdapter._positive_int("max_train_rows", limit)
        if len(frame) <= parsed:
            return frame
        return frame.sample(n=parsed, random_state=spec.seed).reset_index(drop=True)

    @staticmethod
    @contextlib.contextmanager
    def _scoped_randomness(seed: int, num_threads: int) -> Iterator[None]:
        import torch

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        previous_threads = torch.get_num_threads()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.set_num_threads(num_threads)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_num_threads(previous_threads)

    @staticmethod
    @contextlib.contextmanager
    def _sampling_timeout(seconds: int, *, allow_unbounded: bool) -> Iterator[None]:
        if os.name != "posix":
            if not allow_unbounded:
                raise RuntimeError(
                    "Bounded TabuLa sampling is supported in the official Linux environment. On other systems, "
                    "set allow_unbounded_sampling=true only if you accept that the upstream loop has no retry bound."
                )
            yield
            return
        previous = signal.getsignal(signal.SIGALRM)

        def timeout_handler(_signum: int, _frame: Any) -> None:
            raise TimeoutError("Official TabuLa sampling exceeded the configured timeout")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    @staticmethod
    def _training_parameters(spec: RunSpec, categorical_columns: list[str]) -> dict[str, Any]:
        allowed = {
            "learning_rate",
            "weight_decay",
            "logging_steps",
            "disable_tqdm",
            "dataloader_num_workers",
            "gradient_accumulation_steps",
            "warmup_steps",
            "seed",
            "data_seed",
            "report_to",
        }
        train_kwargs = dict(spec.extra.get("train_kwargs", {}))
        unknown = sorted(set(train_kwargs) - allowed)
        if unknown:
            raise ValueError(f"TabuLa unsupported train_kwargs: {unknown}")
        train_kwargs.setdefault("disable_tqdm", True)
        train_kwargs.setdefault("seed", spec.seed)
        train_kwargs.setdefault("data_seed", spec.seed)
        train_kwargs.setdefault("report_to", [])
        return {
            "llm": str(spec.extra.get("llm", "distilgpt2")),
            "epochs": TabulaAdapter._positive_int("epochs", spec.extra.get("epochs", 5)),
            "batch_size": TabulaAdapter._positive_int("batch_size", spec.extra.get("batch_size", 8)),
            "categorical_columns": categorical_columns,
            "conditional_col": spec.extra.get("conditional_col"),
            "num_threads": TabulaAdapter._positive_int("num_threads", spec.extra.get("num_threads", 1)),
            "train_kwargs": train_kwargs,
        }

    @staticmethod
    def _label_encoder_state(model: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in getattr(model, "label_encoder_list", []):
            result.append(
                {
                    "column": str(item["column"]),
                    "classes": [str(value) for value in item["label_encoder"].classes_.tolist()],
                }
            )
        return result

    @staticmethod
    def _write_integrity_manifest(model_root: Path) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}
        total = 0
        for path in sorted(model_root.rglob("*")):
            if path == TabulaAdapter._integrity_path(model_root):
                continue
            if path.is_symlink():
                raise RuntimeError(f"Refusing symlinked TabuLa artifact: {path}")
            if path.is_file():
                relative = path.relative_to(model_root).as_posix()
                size = path.stat().st_size
                total += size
                files[relative] = {"bytes": size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": 1,
            "format": "tabula-safe-transformers-directory",
            "files": files,
            "file_count": len(files),
            "total_bytes": total,
        }
        atomic_write_json(TabulaAdapter._integrity_path(model_root), manifest)
        return manifest

    @staticmethod
    def _validate_safe_model_root(model_root: Path) -> dict[str, Any]:
        if model_root.is_symlink() or not model_root.is_dir():
            raise FileNotFoundError(f"Missing safe TabuLa model directory: {model_root}")
        manifest = read_json(TabulaAdapter._integrity_path(model_root))
        if set(manifest) != {"schema_version", "format", "files", "file_count", "total_bytes"}:
            raise ValueError("Malformed TabuLa integrity manifest")
        if manifest["schema_version"] != 1 or manifest["format"] != "tabula-safe-transformers-directory":
            raise ValueError("Unsupported TabuLa integrity manifest")
        files = manifest["files"]
        if not isinstance(files, dict) or len(files) != manifest["file_count"] or len(files) > 64:
            raise ValueError("Invalid TabuLa artifact inventory")
        total = 0
        for name, record in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError(f"Unsafe TabuLa artifact path: {name!r}")
            if relative.suffix.lower() in {".bin", ".pt", ".pth", ".pkl", ".pickle", ".joblib"}:
                raise ValueError(f"Executable TabuLa checkpoint format is forbidden: {name}")
            path = model_root.joinpath(*relative.parts)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing or unsafe TabuLa artifact: {name}")
            if set(record) != {"bytes", "sha256"} or path.stat().st_size != record["bytes"]:
                raise ValueError(f"TabuLa artifact size mismatch: {name}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"TabuLa artifact digest mismatch: {name}")
            total += path.stat().st_size
        actual = {
            path.relative_to(model_root).as_posix()
            for path in model_root.rglob("*")
            if path.is_file() and path != TabulaAdapter._integrity_path(model_root)
        }
        if actual != set(files) or total != manifest["total_bytes"] or total > 5 * 1024**3:
            raise ValueError("TabuLa model directory differs from its integrity manifest")
        return manifest

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._limit_training_frame(self._load_training_frame(dataset_spec), spec)
        source_root, source = self._resolve_source_root(spec)
        parameters = self._training_parameters(spec, self._roles(dataset_spec)[1])
        conditional_col = parameters["conditional_col"]
        if conditional_col is not None and conditional_col not in dataset_spec.column_names:
            raise ValueError(f"TabuLa conditional_col is unknown: {conditional_col!r}")
        with self._official_class(source_root) as model_class, self._scoped_randomness(
            spec.seed, parameters["num_threads"]
        ):
            model = model_class(
                parameters["llm"],
                experiment_dir=str(spec.output_dir / "tabula_trainer"),
                epochs=parameters["epochs"],
                batch_size=parameters["batch_size"],
                categorical_columns=parameters["categorical_columns"],
                **parameters["train_kwargs"],
            )
            model.fit(frame.copy(), conditional_col=conditional_col)
        model_root = self._model_root(spec)
        if model_root.exists() and any(model_root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty TabuLa model directory: {model_root}")
        transformer_root = model_root / "transformer"
        transformer_root.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(str(transformer_root), safe_serialization=True)
        model.tokenizer.save_pretrained(str(transformer_root))
        state = {
            "schema_version": 1,
            "format": "tabula-method-author-safe-state",
            "model_id": self.model_name,
            "protocol_id": self.protocol_id,
            "upstream_commit": self.upstream_commit,
            "source_manifest_sha256": source["manifest_sha256"],
            "column_names": list(dataset_spec.column_names),
            "numerical_columns": self._roles(dataset_spec)[0],
            "categorical_columns": self._roles(dataset_spec)[1],
            "target_columns": list(dataset_spec.target_columns),
            "task_type": dataset_spec.task_type,
            "official_state": {
                "columns": list(model.columns),
                "num_cols": list(model.num_cols),
                "conditional_col": model.conditional_col,
                "conditional_col_dist": (
                    model.conditional_col_dist.tolist()
                    if isinstance(model.conditional_col_dist, np.ndarray)
                    else model.conditional_col_dist
                ),
                "categorical_columns": list(model.categorical_columns),
                "label_encoders": self._label_encoder_state(model),
            },
            "training_parameters": json.loads(json.dumps(parameters, allow_nan=False)),
            "privacy_guarantee": False,
        }
        atomic_write_json(self._state_path(model_root), state)
        integrity = self._write_integrity_manifest(model_root)
        self._validate_safe_model_root(model_root)
        validate_upstream_source(self.model_name, source_root)
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=source_root,
                notes=[
                    f"Stored {integrity['file_count']} integrity-checked TabuLa artifacts under {model_root}.",
                    "The upstream repository has no declared license; release and Official Results remain blocked.",
                ],
            )
        )

    def _restore_model(
        self,
        model_root: Path,
        dataset_spec: DatasetSpec,
        source_root: Path,
        source: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], contextlib.AbstractContextManager[type[Any]]]:
        self._validate_safe_model_root(model_root)
        state = read_json(self._state_path(model_root))
        required = {
            "schema_version",
            "format",
            "model_id",
            "protocol_id",
            "upstream_commit",
            "source_manifest_sha256",
            "column_names",
            "numerical_columns",
            "categorical_columns",
            "target_columns",
            "task_type",
            "official_state",
            "training_parameters",
            "privacy_guarantee",
        }
        if set(state) != required or state.get("format") != "tabula-method-author-safe-state":
            raise ValueError("Malformed TabuLa state")
        roles = self._roles(dataset_spec)
        if (
            state["upstream_commit"] != self.upstream_commit
            or state["source_manifest_sha256"] != source["manifest_sha256"]
            or state["column_names"] != dataset_spec.column_names
            or state["numerical_columns"] != roles[0]
            or state["categorical_columns"] != roles[1]
            or state["target_columns"] != dataset_spec.target_columns
            or state["task_type"] != dataset_spec.task_type
        ):
            raise ValueError("TabuLa checkpoint source or dataset contract mismatch")
        manager = self._official_class(source_root)
        model_class = manager.__enter__()
        try:
            transformer_root = model_root / "transformer"
            model = model_class(
                str(transformer_root),
                categorical_columns=state["official_state"]["categorical_columns"],
            )
            with disable_torchvision_for_transformers():
                transformers = importlib.import_module("transformers")
            model.model = transformers.AutoModelForCausalLM.from_pretrained(
                str(transformer_root), trust_remote_code=False
            )
            # Method-author load_from_dir() constructs a fresh module and loads
            # its state dict without switching to eval mode. Transformers'
            # from_pretrained() does switch modes, so restore the official
            # sampling semantics explicitly after safe deserialization.
            model.model.train()
            for name in ("columns", "num_cols", "conditional_col", "conditional_col_dist", "categorical_columns"):
                setattr(model, name, state["official_state"][name])
            encoders: list[dict[str, Any]] = []
            for item in state["official_state"]["label_encoders"]:
                encoder = LabelEncoder()
                encoder.classes_ = np.asarray(item["classes"], dtype=str)
                encoders.append({"column": item["column"], "label_encoder": encoder})
            model.label_encoder_list = encoders
        except Exception:
            manager.__exit__(None, None, None)
            raise
        return model, state, manager

    @staticmethod
    def _sample_exact_rows(
        model: Any,
        *,
        requested: int,
        start_col: str,
        start_dist: Any,
        temperature: float,
        k: int,
        max_length: int,
        device: str,
        max_empty_batches: int,
    ) -> pd.DataFrame:
        """Complete the public exact-row contract with unchanged official calls."""

        batches: list[pd.DataFrame] = []
        remaining = requested
        empty_batches = 0
        while remaining:
            batch = model.sample(
                n_samples=remaining,
                start_col=start_col,
                start_col_dist=start_dist,
                temperature=temperature,
                k=k,
                max_length=max_length,
                device=device,
            )
            if not isinstance(batch, pd.DataFrame):
                raise RuntimeError("Official TabuLa sampling did not return a DataFrame")
            if len(batch) > remaining:
                raise RuntimeError("Official TabuLa sampling returned more rows than requested for a batch")
            if batch.empty:
                empty_batches += 1
                if empty_batches >= max_empty_batches:
                    raise RuntimeError("Official TabuLa sampling repeatedly returned zero usable categorical rows")
                continue
            empty_batches = 0
            batches.append(batch)
            remaining -= len(batch)
        return pd.concat(batches, ignore_index=True).head(requested)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        model_root = self._model_root(spec)
        source_root, source = self._resolve_source_root(spec)
        model, state, manager = self._restore_model(model_root, dataset_spec, source_root, source)
        requested = spec.num_samples or int(spec.extra.get("num_samples", 0))
        if requested < 1:
            manager.__exit__(None, None, None)
            raise ValueError("TabuLa sample requires a positive num_samples")
        temperature = self._positive_float("temperature", spec.extra.get("temperature", 0.7))
        k = self._positive_int("k", spec.extra.get("k", min(100, max(8, requested))))
        max_length = self._positive_int("max_length", spec.extra.get("max_length", 256))
        num_threads = self._positive_int("num_threads", spec.extra.get("num_threads", 1))
        timeout_seconds = self._positive_int("timeout_seconds", spec.extra.get("timeout_seconds", 900))
        max_empty_batches = self._positive_int(
            "max_empty_batches", spec.extra.get("max_empty_batches", 8)
        )
        start_col = spec.extra.get("start_col", "")
        start_dist = spec.extra.get("start_col_dist")
        if start_col and start_col not in dataset_spec.column_names:
            manager.__exit__(None, None, None)
            raise ValueError(f"TabuLa start_col is unknown: {start_col!r}")
        try:
            with self._scoped_randomness(spec.seed, num_threads), self._sampling_timeout(
                timeout_seconds,
                allow_unbounded=bool(spec.extra.get("allow_unbounded_sampling", False)),
            ):
                sample_df = self._sample_exact_rows(
                    model,
                    requested=requested,
                    start_col=start_col,
                    start_dist=start_dist,
                    temperature=temperature,
                    k=k,
                    max_length=max_length,
                    device=spec.device,
                    max_empty_batches=max_empty_batches,
                )
        finally:
            manager.__exit__(None, None, None)
        if not isinstance(sample_df, pd.DataFrame) or len(sample_df) != requested:
            observed = None if not isinstance(sample_df, pd.DataFrame) else len(sample_df)
            raise RuntimeError(f"Official TabuLa sampling returned {observed} rows; expected exactly {requested}")
        sample_df = sample_df[dataset_spec.column_names].copy()
        if bool(sample_df.isna().any().any()):
            raise RuntimeError("Official TabuLa sampling returned missing values")
        numerical, categorical = self._roles(dataset_spec)
        for column in numerical:
            sample_df[column] = pd.to_numeric(sample_df[column], errors="raise")
            if not bool(np.isfinite(sample_df[column].to_numpy(dtype=float)).all()):
                raise RuntimeError(f"Official TabuLa sampling returned non-finite values in {column!r}")
        for column in categorical:
            sample_df[column] = sample_df[column].astype(str)
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        self._validate_safe_model_root(model_root)
        validate_upstream_source(self.model_name, source_root)
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=source_root,
                generated_sample_path=sample_path,
                notes=[
                    f"Generated exactly {requested} rows through the locked method-author TabuLa source.",
                    f"Sampling was bounded to {timeout_seconds} seconds in the supported Linux environment.",
                    "No privacy guarantee is implied; model artifacts require access control.",
                ],
            )
        )

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


__all__ = ["TabulaAdapter"]
