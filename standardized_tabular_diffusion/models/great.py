from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import random
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import (
    SampleFileEvaluatorMixin,
    disable_torchvision_for_transformers,
)
from standardized_tabular_diffusion.models.base import BaseModelAdapter

GREAT_PACKAGE_VERSION = "0.0.14"
GREAT_UPSTREAM_COMMIT = "b300f6123cf1d9590b76ea45cc23298df944a319"
GREAT_UPSTREAM_TREE = "82eccd92deb18d6138a4fd96a8e0eb097b38babb"
GREAT_MODEL_TREE = "b2f1799f00557ca2021f4b73c9d3962d33ada806"
GREAT_WHEEL_SHA256 = "4f6384ec4a736177ae2d1e6146951cfdfc764b1cc041ae5c2b155a99dd18cb74"
GREAT_SDIST_SHA256 = "5f84487e958349d5aaf31ecc8368779664af7be500f8f3719007c1e9c47b1045"
GREAT_RUNTIME_SHA256 = {
    "be_great/__init__.py": "646351494b2987471e18df8ee225c3e920cf96396a6a83ac674084b562c8186e",
    "be_great/great.py": "58c12342f9fff0b1811ec08ed85e71803f4a06990cd56b43e33118b2800fd536",
    "be_great/great_constrained.py": "ec747abd355fc93661c18e335c3e6223218169110ee1e9c85d5de21b77d8ecd7",
    "be_great/great_dataset.py": "5a98ad598480af392667a1dba6bf625fb6a19f39eeb8429a93d957ba2da6490a",
    "be_great/great_mock_datasets.py": "05af9ed53bbca11d67e6d66393c75d493e2e0a665a1b82798544ca311ff30fa8",
    "be_great/great_start.py": "2f8b150c3fb9b46528335ca0b3ab09a21ddef66122c9c7e852d6e22a272d68f9",
    "be_great/great_trainer.py": "b67afb0731c7161853390e0469fc9ff991cd66195c1ce0db6e29ef0f7d77deec",
    "be_great/great_utils.py": "4622e2a911575f846235ce402994f9fee9477853024a9f7342b9ef955f2f7dd2",
}


def verify_great_distribution() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("be-great")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            'GReaT requires the locked official package; install "standardized-tabular-diffusion[great]".'
        ) from exc
    if distribution.version != GREAT_PACKAGE_VERSION:
        raise RuntimeError(
            f"GReaT requires be-great=={GREAT_PACKAGE_VERSION}; observed {distribution.version!r}"
        )
    verified: dict[str, str] = {}
    for relative, expected in GREAT_RUNTIME_SHA256.items():
        path = Path(distribution.locate_file(relative))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"GReaT official package file is missing or unsafe: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"GReaT official package file differs from the 0.0.14 lock: {relative}")
        verified[relative] = observed
    return {
        "name": "be-great",
        "version": distribution.version,
        "upstream_commit": GREAT_UPSTREAM_COMMIT,
        "upstream_tree": GREAT_UPSTREAM_TREE,
        "upstream_model_tree": GREAT_MODEL_TREE,
        "runtime_files_verified": len(verified),
        "runtime_files": verified,
        "wheel_sha256": GREAT_WHEEL_SHA256,
        "sdist_sha256": GREAT_SDIST_SHA256,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class GReaTAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    """Strict adapter around the unmodified method-author be-great package."""

    model_name = "great"
    upstream_dirname = "."
    checkpoint_dirname = "great_model"
    protocol_id = "be-great-official-package-parity-v1"

    def _model_root(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or spec.output_dir / self.checkpoint_dirname

    @staticmethod
    def _state_path(model_root: Path) -> Path:
        return model_root / "great-state.json"

    @staticmethod
    def _integrity_path(model_root: Path) -> Path:
        return model_root / "great-integrity.json"

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"GReaT {name} must be a positive integer")
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"GReaT {name} must be a positive integer")
        return parsed

    @staticmethod
    def _positive_float(name: str, value: Any) -> float:
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"GReaT {name} must be finite and positive")
        return parsed

    @staticmethod
    def _roles(dataset_spec: DatasetSpec) -> tuple[list[str], list[str]]:
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("GReaT requires exactly one target column")
        numerical = list(dataset_spec.numerical_columns)
        categorical = list(dataset_spec.categorical_columns)
        target = dataset_spec.target_columns[0]
        if target not in numerical and target not in categorical:
            (categorical if dataset_spec.task_type == "classification" else numerical).append(target)
        if set(numerical) & set(categorical) or set(numerical) | set(categorical) != set(dataset_spec.column_names):
            raise ValueError("GReaT requires disjoint and complete declared column roles")
        return numerical, categorical

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("GReaT requires dataset_spec.train_data_path")
        if any("," in column or " is " in column or "\n" in column or "\r" in column for column in dataset_spec.column_names):
            raise ValueError("GReaT column names cannot contain commas, newlines, or the official ' is ' delimiter")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        if frame.empty:
            raise ValueError("GReaT cannot train on an empty table")
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "GReaT requires missing values to be imputed by the train-fitted preprocessing module; "
                f"observed: {observed}"
            )
        numerical, categorical = self._roles(dataset_spec)
        for column in numerical:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"GReaT numerical column {column!r} contains non-finite values")
            frame[column] = values
        for column in categorical:
            values = frame[column].astype(str)
            if bool(values.str.contains(r",|\r|\n", regex=True).any()):
                raise ValueError(
                    f"GReaT categorical column {column!r} contains text ambiguous under the official comma parser"
                )
            frame[column] = values
        return frame

    @staticmethod
    def _limit_training_frame(frame: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        limit = spec.extra.get("max_train_rows")
        if limit is None:
            return frame
        parsed = GReaTAdapter._positive_int("max_train_rows", limit)
        if len(frame) <= parsed:
            return frame
        return frame.sample(n=parsed, random_state=spec.seed).reset_index(drop=True)

    @staticmethod
    def _import_model_class() -> type[Any]:
        verify_great_distribution()
        with disable_torchvision_for_transformers():
            module = importlib.import_module("be_great")
        return module.GReaT

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
    def _training_parameters(spec: RunSpec) -> dict[str, Any]:
        allowed = {
            "learning_rate",
            "weight_decay",
            "logging_steps",
            "save_strategy",
            "save_steps",
            "disable_tqdm",
            "dataloader_num_workers",
            "gradient_accumulation_steps",
            "warmup_steps",
        }
        unknown = sorted(set(spec.extra.get("train_kwargs", {})) - allowed)
        if unknown:
            raise ValueError(f"GReaT unsupported train_kwargs: {unknown}")
        train_kwargs = dict(spec.extra.get("train_kwargs", {}))
        train_kwargs.setdefault("save_strategy", "no")
        train_kwargs.setdefault("disable_tqdm", True)
        train_kwargs.setdefault("seed", spec.seed)
        train_kwargs.setdefault("data_seed", spec.seed)
        return {
            "llm": str(spec.extra.get("llm", "tabularisai/Qwen3-0.3B-distil")),
            "epochs": GReaTAdapter._positive_int("epochs", spec.extra.get("epochs", 5)),
            "batch_size": GReaTAdapter._positive_int("batch_size", spec.extra.get("batch_size", 8)),
            "float_precision": spec.extra.get("float_precision"),
            "random_conditional_col": bool(spec.extra.get("random_conditional_col", False)),
            "conditional_col": spec.extra.get("conditional_col"),
            "num_threads": GReaTAdapter._positive_int("num_threads", spec.extra.get("num_threads", 1)),
            "train_kwargs": train_kwargs,
        }

    @staticmethod
    def _write_integrity_manifest(model_root: Path) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}
        total = 0
        for path in sorted(model_root.rglob("*")):
            if path == GReaTAdapter._integrity_path(model_root):
                continue
            if path.is_symlink():
                raise RuntimeError(f"Refusing symlinked GReaT artifact: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(model_root).as_posix()
            size = path.stat().st_size
            total += size
            files[relative] = {"bytes": size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": 1,
            "format": "great-safe-transformers-directory",
            "files": files,
            "file_count": len(files),
            "total_bytes": total,
        }
        atomic_write_json(GReaTAdapter._integrity_path(model_root), manifest)
        return manifest

    @staticmethod
    def _validate_safe_model_root(model_root: Path) -> dict[str, Any]:
        if model_root.is_symlink() or not model_root.is_dir():
            raise FileNotFoundError(f"Missing safe GReaT model directory: {model_root}")
        manifest = read_json(GReaTAdapter._integrity_path(model_root))
        if set(manifest) != {"schema_version", "format", "files", "file_count", "total_bytes"}:
            raise ValueError("Malformed GReaT integrity manifest")
        if manifest["schema_version"] != 1 or manifest["format"] != "great-safe-transformers-directory":
            raise ValueError("Unsupported GReaT integrity manifest")
        files = manifest["files"]
        if not isinstance(files, dict) or len(files) != manifest["file_count"] or len(files) > 64:
            raise ValueError("Invalid GReaT artifact file inventory")
        observed_total = 0
        for name, record in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError(f"Unsafe GReaT artifact path: {name!r}")
            if relative.suffix.lower() in {".bin", ".pt", ".pth", ".pkl", ".pickle", ".joblib"}:
                raise ValueError(f"Executable GReaT checkpoint format is forbidden: {name}")
            path = model_root.joinpath(*relative.parts)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing or unsafe GReaT artifact: {name}")
            size = path.stat().st_size
            if set(record) != {"bytes", "sha256"} or size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"GReaT artifact integrity mismatch: {name}")
            observed_total += size
        actual = {
            path.relative_to(model_root).as_posix()
            for path in model_root.rglob("*")
            if path.is_file() and path != GReaTAdapter._integrity_path(model_root)
        }
        if actual != set(files) or observed_total != manifest["total_bytes"] or observed_total > 5 * 1024**3:
            raise ValueError("GReaT model directory differs from its integrity manifest")
        return manifest

    @staticmethod
    def _state_payload(
        model: Any,
        dataset_spec: DatasetSpec,
        parameters: dict[str, Any],
        package: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "format": "be-great-official-safe-state",
            "model_id": "great",
            "protocol_id": GReaTAdapter.protocol_id,
            "package_version": GREAT_PACKAGE_VERSION,
            "upstream_commit": GREAT_UPSTREAM_COMMIT,
            "runtime_files": package["runtime_files"],
            "column_names": list(dataset_spec.column_names),
            "numerical_columns": GReaTAdapter._roles(dataset_spec)[0],
            "categorical_columns": GReaTAdapter._roles(dataset_spec)[1],
            "target_columns": list(dataset_spec.target_columns),
            "task_type": dataset_spec.task_type,
            "official_state": {
                "columns": _json_value(model.columns),
                "num_cols": _json_value(model.num_cols),
                "conditional_col": _json_value(model.conditional_col),
                "conditional_col_dist": _json_value(model.conditional_col_dist),
                "col_stats": _json_value(model.col_stats),
                "float_precision": _json_value(model.float_precision),
            },
            "training_parameters": _json_value(parameters),
            "privacy_guarantee": False,
        }

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._limit_training_frame(self._load_training_frame(dataset_spec), spec)
        package = verify_great_distribution()
        model_class = self._import_model_class()
        parameters = self._training_parameters(spec)
        conditional_col = parameters["conditional_col"]
        if conditional_col is not None and conditional_col not in dataset_spec.column_names:
            raise ValueError(f"GReaT conditional_col is unknown: {conditional_col!r}")
        with self._scoped_randomness(spec.seed, parameters["num_threads"]):
            model = model_class(
                parameters["llm"],
                experiment_dir=str(spec.output_dir / "great_trainer"),
                epochs=parameters["epochs"],
                batch_size=parameters["batch_size"],
                float_precision=parameters["float_precision"],
                report_to=[],
                **parameters["train_kwargs"],
            )
            model.fit(
                frame,
                conditional_col=conditional_col,
                random_conditional_col=parameters["random_conditional_col"],
            )
        model_root = self._model_root(spec)
        if model_root.exists() and any(model_root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty GReaT model directory: {model_root}")
        transformer_root = model_root / "transformer"
        transformer_root.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(str(transformer_root), safe_serialization=True)
        model.tokenizer.save_pretrained(str(transformer_root))
        atomic_write_json(self._state_path(model_root), self._state_payload(model, dataset_spec, parameters, package))
        integrity = self._write_integrity_manifest(model_root)
        self._validate_safe_model_root(model_root)
        verify_great_distribution()
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.repo_root,
                notes=[
                    f"Stored {integrity['file_count']} integrity-checked GReaT artifact files under {model_root}.",
                    "The checkpoint uses safetensors and JSON; the official package's torch.load-based model.pt format is not used.",
                ],
            )
        )

    def _restore_model(self, model_root: Path, dataset_spec: DatasetSpec) -> tuple[Any, dict[str, Any]]:
        self._validate_safe_model_root(model_root)
        state = read_json(self._state_path(model_root))
        expected = {
            "schema_version",
            "format",
            "model_id",
            "protocol_id",
            "package_version",
            "upstream_commit",
            "runtime_files",
            "column_names",
            "numerical_columns",
            "categorical_columns",
            "target_columns",
            "task_type",
            "official_state",
            "training_parameters",
            "privacy_guarantee",
        }
        if set(state) != expected or state.get("format") != "be-great-official-safe-state":
            raise ValueError("Malformed GReaT state")
        package = verify_great_distribution()
        if (
            state["package_version"] != GREAT_PACKAGE_VERSION
            or state["upstream_commit"] != GREAT_UPSTREAM_COMMIT
            or state["runtime_files"] != package["runtime_files"]
        ):
            raise ValueError("GReaT checkpoint package identity mismatch")
        roles = self._roles(dataset_spec)
        if (
            state["column_names"] != dataset_spec.column_names
            or state["numerical_columns"] != roles[0]
            or state["categorical_columns"] != roles[1]
            or state["target_columns"] != dataset_spec.target_columns
            or state["task_type"] != dataset_spec.task_type
        ):
            raise ValueError("GReaT checkpoint dataset contract mismatch")
        model_class = self._import_model_class()
        model = model_class(str(model_root / "transformer"), report_to=[])
        for name, value in state["official_state"].items():
            setattr(model, name, value)
        return model, state

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        model_root = self._model_root(spec)
        model, state = self._restore_model(model_root, dataset_spec)
        requested = spec.num_samples or int(spec.extra.get("num_samples", 0))
        if requested < 1:
            raise ValueError("GReaT sample requires a positive num_samples")
        temperature = self._positive_float("temperature", spec.extra.get("temperature", 0.7))
        k = self._positive_int("k", spec.extra.get("k", min(100, max(8, requested))))
        max_length = self._positive_int("max_length", spec.extra.get("max_length", 256))
        num_threads = self._positive_int("num_threads", spec.extra.get("num_threads", 1))
        start_col = spec.extra.get("start_col", "")
        start_dist = spec.extra.get("start_col_dist")
        if start_col and start_col not in dataset_spec.column_names:
            raise ValueError(f"GReaT start_col is unknown: {start_col!r}")
        with self._scoped_randomness(spec.seed, num_threads):
            sample_df = model.sample(
                n_samples=requested,
                start_col=start_col,
                start_col_dist=start_dist,
                temperature=temperature,
                k=k,
                max_length=max_length,
                drop_nan=True,
                device=spec.device,
                guided_sampling=bool(spec.extra.get("guided_sampling", False)),
                random_feature_order=bool(spec.extra.get("random_feature_order", True)),
                conditions=spec.extra.get("conditions"),
            )
        if not isinstance(sample_df, pd.DataFrame) or len(sample_df) != requested:
            observed = None if not isinstance(sample_df, pd.DataFrame) else len(sample_df)
            raise RuntimeError(f"Official GReaT sampling returned {observed} rows; expected exactly {requested}")
        sample_df = sample_df[dataset_spec.column_names].copy()
        if bool(sample_df.isna().any().any()):
            raise RuntimeError("Official GReaT sampling returned missing values")
        numerical, categorical = self._roles(dataset_spec)
        for column in numerical:
            sample_df[column] = pd.to_numeric(sample_df[column], errors="raise")
            if not bool(np.isfinite(sample_df[column].to_numpy(dtype=float)).all()):
                raise RuntimeError(f"Official GReaT sampling returned non-finite values in {column!r}")
        for column in categorical:
            sample_df[column] = sample_df[column].astype(str)
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        self._validate_safe_model_root(model_root)
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.repo_root,
                generated_sample_path=sample_path,
                notes=[
                    f"Generated exactly {requested} rows through unchanged be-great {state['package_version']} APIs.",
                    "No privacy guarantee is implied; model artifacts require access control.",
                ],
            )
        )

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


__all__ = ["GReaTAdapter", "verify_great_distribution"]
