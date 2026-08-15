from __future__ import annotations

import contextlib
import importlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
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
from standardized_tabular_diffusion.runtime_contracts import (
    observe_torch_model_device,
    resolve_torch_training_device,
)
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
        device_contract = resolve_torch_training_device(spec.device)
        parameters["train_kwargs"]["use_cpu"] = device_contract["trainer_use_cpu"]
        parameters["device_contract"] = device_contract
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
        device_contract["observed"] = observe_torch_model_device(model, device_contract["requested"])
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

    def _load_state(
        self,
        model_root: Path,
        dataset_spec: DatasetSpec,
        source: dict[str, Any],
    ) -> dict[str, Any]:
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
        return state

    def _restore_model(
        self,
        model_root: Path,
        dataset_spec: DatasetSpec,
        source_root: Path,
        source: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], contextlib.AbstractContextManager[type[Any]]]:
        state = self._load_state(model_root, dataset_spec, source)
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
    def _default_sampling_start(state: dict[str, Any]) -> tuple[str, Any, bool]:
        """Translate the upstream pre-encoding default for a categorical start column."""

        official = state["official_state"]
        start_col = official["conditional_col"] or ""
        start_dist = official["conditional_col_dist"]
        if not start_col or not isinstance(start_dist, dict):
            return start_col, start_dist, False
        encoder = next(
            (item for item in official["label_encoders"] if item["column"] == start_col),
            None,
        )
        if encoder is None:
            return start_col, start_dist, False
        classes = list(encoder["classes"])
        if set(start_dist) - set(classes):
            raise ValueError("TabuLa categorical default distribution differs from its fitted label encoder")
        translated = {
            str(classes.index(label)): probability for label, probability in start_dist.items()
        }
        return start_col, translated, True

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

    def _sample_in_subprocess(
        self,
        *,
        dataset_spec: DatasetSpec,
        model_root: Path,
        source_root: Path,
        source: dict[str, Any],
        seed: int,
        requested: int,
        start_col: str,
        start_dist: Any,
        temperature: float,
        k: int,
        max_length: int,
        device: str,
        num_threads: int,
        max_empty_batches: int,
        timeout_seconds: int,
        output_dir: Path,
    ) -> pd.DataFrame:
        """Run the unchanged upstream retry loop behind a Windows-safe hard timeout."""

        launcher = Path(__file__).resolve().parents[1] / "compat" / "tabula_sampling_launcher.py"
        if launcher.is_symlink() or not launcher.is_file():
            raise FileNotFoundError(f"Missing trusted TabuLa sampling launcher: {launcher}")
        with tempfile.TemporaryDirectory(prefix=".tabula-sampling-", dir=output_dir) as temporary:
            temporary_root = Path(temporary)
            request_path = temporary_root / "request.json"
            response_path = temporary_root / "response.json"
            sample_path = temporary_root / "samples.csv"
            atomic_write_json(
                request_path,
                {
                    "schema_version": 1,
                    "repo_root": str(self.repo_root.resolve()),
                    "dataset_spec": dataset_spec.to_dict(),
                    "model_root": str(model_root.resolve()),
                    "source_root": str(source_root.resolve()),
                    "source_manifest_sha256": source["manifest_sha256"],
                    "seed": seed,
                    "requested": requested,
                    "start_col": start_col,
                    "start_dist": start_dist,
                    "temperature": temperature,
                    "k": k,
                    "max_length": max_length,
                    "device": device,
                    "num_threads": num_threads,
                    "max_empty_batches": max_empty_batches,
                },
            )
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "standardized_tabular_diffusion.compat.tabula_sampling_launcher",
                        "--request",
                        str(request_path),
                        "--response",
                        str(response_path),
                        "--sample",
                        str(sample_path),
                    ],
                    cwd=self.repo_root,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Official TabuLa sampling exceeded the configured {timeout_seconds}-second timeout"
                ) from exc
            if not response_path.is_file() or response_path.is_symlink():
                diagnostic = (completed.stderr or completed.stdout or "no child-process output")[-2000:]
                raise RuntimeError(
                    "Bounded TabuLa sampling subprocess did not produce a trusted response; "
                    f"exit_code={completed.returncode}, diagnostic={diagnostic!r}"
                )
            response = read_json(response_path)
            if not isinstance(response, dict):
                raise RuntimeError("Bounded TabuLa sampling subprocess returned a non-object response")
            if response.get("status") != "pass":
                raise RuntimeError(
                    "Bounded TabuLa sampling subprocess failed: "
                    f"{response.get('error_type', 'unknown')}: {response.get('error', 'unknown error')}"
                )
            expected_response_keys = {"status", "rows", "columns", "sample_sha256"}
            if set(response) != expected_response_keys or completed.returncode != 0:
                raise RuntimeError("Bounded TabuLa sampling subprocess returned an invalid response contract")
            if sample_path.is_symlink() or not sample_path.is_file():
                raise RuntimeError("Bounded TabuLa sampling subprocess did not create a safe sample file")
            if sha256_file(sample_path) != response["sample_sha256"]:
                raise RuntimeError("Bounded TabuLa sampling subprocess sample digest mismatch")
            sample_df = pd.read_csv(sample_path)
            if len(sample_df) != response["rows"] or list(sample_df.columns) != response["columns"]:
                raise RuntimeError("Bounded TabuLa sampling subprocess sample shape mismatch")
            return sample_df

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        model_root = self._model_root(spec)
        source_root, source = self._resolve_source_root(spec)
        requested = spec.num_samples or int(spec.extra.get("num_samples", 0))
        if requested < 1:
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
            raise ValueError(f"TabuLa start_col is unknown: {start_col!r}")
        state = self._load_state(model_root, dataset_spec, source)
        translated_default_start = False
        if not start_col and start_dist is None:
            start_col, start_dist, translated_default_start = self._default_sampling_start(state)
        allow_unbounded = bool(spec.extra.get("allow_unbounded_sampling", False))
        if os.name == "nt" and not allow_unbounded:
            self._validate_safe_model_root(model_root)
            sample_df = self._sample_in_subprocess(
                dataset_spec=dataset_spec,
                model_root=model_root,
                source_root=source_root,
                source=source,
                seed=spec.seed,
                requested=requested,
                start_col=start_col,
                start_dist=start_dist,
                temperature=temperature,
                k=k,
                max_length=max_length,
                device=spec.device,
                num_threads=num_threads,
                max_empty_batches=max_empty_batches,
                timeout_seconds=timeout_seconds,
                output_dir=spec.output_dir,
            )
            timeout_boundary = "an isolated subprocess on Windows"
        else:
            model, _state, manager = self._restore_model(model_root, dataset_spec, source_root, source)
            try:
                with self._scoped_randomness(spec.seed, num_threads), self._sampling_timeout(
                    timeout_seconds,
                    allow_unbounded=allow_unbounded,
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
            timeout_boundary = "SIGALRM on POSIX" if os.name == "posix" else "an explicit unbounded override"
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
                    f"Sampling was bounded to {timeout_seconds} seconds through {timeout_boundary}.",
                    (
                        "The fitted categorical default was translated to the exact label IDs used by official training."
                        if translated_default_start
                        else "The requested sampling start distribution was forwarded unchanged."
                    ),
                    "No privacy guarantee is implied; model artifacts require access control.",
                ],
            )
        )

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


__all__ = ["TabulaAdapter"]


def execute_tabula_sampling_request(request_path: Path, sample_path: Path) -> dict[str, Any]:
    """Execute one internally generated TabuLa sampling request in a disposable child process."""

    request = read_json(request_path)
    required = {
        "schema_version",
        "repo_root",
        "dataset_spec",
        "model_root",
        "source_root",
        "source_manifest_sha256",
        "seed",
        "requested",
        "start_col",
        "start_dist",
        "temperature",
        "k",
        "max_length",
        "device",
        "num_threads",
        "max_empty_batches",
    }
    if not isinstance(request, dict) or set(request) != required or request["schema_version"] != 1:
        raise ValueError("Malformed internal TabuLa sampling request")
    dataset_payload = dict(request["dataset_spec"])
    for name in ("metadata_path", "train_data_path", "val_data_path", "test_data_path"):
        value = dataset_payload.get(name)
        dataset_payload[name] = None if value is None else Path(value)
    dataset_spec = DatasetSpec(**dataset_payload)
    adapter = TabulaAdapter(Path(request["repo_root"]))
    source_root = Path(request["source_root"])
    source = validate_upstream_source(adapter.model_name, source_root)
    if source["manifest_sha256"] != request["source_manifest_sha256"]:
        raise ValueError("TabuLa child process observed a different locked source manifest")
    model_root = Path(request["model_root"])
    model, _state, manager = adapter._restore_model(model_root, dataset_spec, source_root, source)
    try:
        with adapter._scoped_randomness(int(request["seed"]), int(request["num_threads"])):
            sample_df = adapter._sample_exact_rows(
                model,
                requested=int(request["requested"]),
                start_col=str(request["start_col"]),
                start_dist=request["start_dist"],
                temperature=float(request["temperature"]),
                k=int(request["k"]),
                max_length=int(request["max_length"]),
                device=str(request["device"]),
                max_empty_batches=int(request["max_empty_batches"]),
            )
    finally:
        manager.__exit__(None, None, None)
    sample_df = sample_df[dataset_spec.column_names].copy()
    adapter._write_dataframe_csv(sample_df, sample_path)
    return {
        "status": "pass",
        "rows": len(sample_df),
        "columns": list(sample_df.columns),
        "sample_sha256": sha256_file(sample_path),
    }
