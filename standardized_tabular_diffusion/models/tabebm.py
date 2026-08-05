from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter

TABEBM_PACKAGE_VERSION = "2025.8.19"
TABEBM_UPSTREAM_COMMIT = "72eb78dab896c7a8f39c4dcc288c834fd72eff2b"
TABEBM_UPSTREAM_TREE = "627af984e9447bf1a88f1d13e4c766704738ec28"
TABEBM_MODEL_TREE = "5df0517922b9b03d29c5d84e46af3423537d801f"
TABEBM_SDIST_SHA256 = "6111611326747a680f93dfadcbac1d602ce20cb722b9b6cbff1f556b9f48d503"
TABEBM_RUNTIME_SHA256 = {
    "tabebm/TabEBM.py": "e5ed68e8af8e4b4362485471931718356b6a9d2760472719d3d8245688840f4a",
    "tabebm/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def verify_tabebm_distribution() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("tabebm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            'TabEBM requires the locked official package; install "standardized-tabular-diffusion[tabebm]".'
        ) from exc
    if distribution.version != TABEBM_PACKAGE_VERSION:
        raise RuntimeError(
            f"TabEBM requires tabebm=={TABEBM_PACKAGE_VERSION}; observed {distribution.version!r}"
        )
    verified: dict[str, str] = {}
    for relative, expected in TABEBM_RUNTIME_SHA256.items():
        path = Path(distribution.locate_file(relative))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"TabEBM official package file is missing or unsafe: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"TabEBM official package file differs from the release lock: {relative}")
        verified[relative] = observed
    return {
        "name": "tabebm",
        "version": distribution.version,
        "upstream_commit": TABEBM_UPSTREAM_COMMIT,
        "upstream_tree": TABEBM_UPSTREAM_TREE,
        "upstream_model_tree": TABEBM_MODEL_TREE,
        "sdist_sha256": TABEBM_SDIST_SHA256,
        "runtime_files_verified": len(verified),
        "runtime_files": verified,
    }


class _TabEBMPreprocessor:
    def __init__(self, dataset_spec: DatasetSpec) -> None:
        if dataset_spec.task_type != "classification" or len(dataset_spec.target_columns) != 1:
            raise ValueError("TabEBM supports single-target classification only")
        self.dataset_spec = dataset_spec
        self.target = dataset_spec.target_columns[0]
        self.features = [column for column in dataset_spec.column_names if column != self.target]
        self.numerical = [column for column in dataset_spec.numerical_columns if column in self.features]
        self.categorical = [column for column in dataset_spec.categorical_columns if column in self.features]
        if set(self.numerical) & set(self.categorical) or set(self.numerical) | set(self.categorical) != set(self.features):
            raise ValueError("TabEBM requires disjoint and complete feature roles")
        self.state: dict[str, Any] | None = None

    def fit_transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        blocks: list[np.ndarray] = []
        numeric_state: dict[str, dict[str, float]] = {}
        for column in self.numerical:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(mean) or not np.isfinite(scale):
                raise ValueError(f"TabEBM numerical column {column!r} is non-finite")
            if scale == 0.0:
                scale = 1.0
            blocks.append(((values - mean) / scale).reshape(-1, 1))
            numeric_state[column] = {"mean": mean, "scale": scale}
        categorical_state: dict[str, list[str]] = {}
        for column in self.categorical:
            values = frame[column].astype(str)
            categories = sorted(values.unique().tolist())
            mapping = {value: index for index, value in enumerate(categories)}
            blocks.append(values.map(mapping).to_numpy(dtype=np.float64).reshape(-1, 1))
            categorical_state[column] = categories
        target_values = frame[self.target].astype(str)
        target_classes = sorted(target_values.unique().tolist())
        if len(target_classes) < 2:
            raise ValueError("TabEBM requires at least two target classes")
        target_mapping = {value: index for index, value in enumerate(target_classes)}
        y = target_values.map(target_mapping).to_numpy(dtype=np.int64)
        X = np.concatenate(blocks, axis=1).astype(np.float64) if blocks else np.empty((len(frame), 0))
        if X.shape[1] == 0 or not bool(np.isfinite(X).all()):
            raise ValueError("TabEBM requires at least one finite feature")
        self.state = {
            "features": self.features,
            "numerical": self.numerical,
            "categorical": self.categorical,
            "target": self.target,
            "numeric_state": numeric_state,
            "categorical_state": categorical_state,
            "target_classes": target_classes,
        }
        return X, y

    def inverse_transform(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("TabEBM preprocessor is not fitted")
        if X.ndim != 2 or X.shape[1] != len(self.features) or len(X) != len(y):
            raise ValueError("TabEBM official output has an invalid shape")
        output = pd.DataFrame(index=range(len(X)))
        for index, column in enumerate(self.features):
            values = X[:, index]
            if column in self.numerical:
                params = self.state["numeric_state"][column]
                output[column] = values * params["scale"] + params["mean"]
            else:
                categories = self.state["categorical_state"][column]
                codes = np.clip(np.rint(values).astype(int), 0, len(categories) - 1)
                output[column] = [categories[code] for code in codes]
        classes = self.state["target_classes"]
        if bool(((y < 0) | (y >= len(classes))).any()):
            raise ValueError("TabEBM official output contains an invalid class index")
        output[self.target] = [classes[int(index)] for index in y]
        return output[self.dataset_spec.column_names]


class TabEBMAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    """Official-package TabEBM adapter with an explicit gated-model boundary."""

    model_name = "tabebm"
    upstream_dirname = "."
    checkpoint_filename = "model.tabebm.json"
    protocol_id = "tabebm-official-package-core-validation-v1"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or spec.output_dir / self.checkpoint_filename

    @staticmethod
    def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")

    @staticmethod
    def _allow_gated_sampling(spec: RunSpec) -> bool:
        explicit = spec.extra.get("allow_gated_model")
        if explicit is not None:
            return bool(explicit)
        return os.environ.get("STANDARDIZED_TABPFN_ALLOW_GATED", "").strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"TabEBM {name} must be a positive integer")
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"TabEBM {name} must be a positive integer")
        return parsed

    @staticmethod
    def _nonnegative_float(name: str, value: Any) -> float:
        parsed = float(value)
        if not np.isfinite(parsed) or parsed < 0:
            raise ValueError(f"TabEBM {name} must be finite and nonnegative")
        return parsed

    @staticmethod
    def _positive_float(name: str, value: Any) -> float:
        parsed = TabEBMAdapter._nonnegative_float(name, value)
        if parsed == 0:
            raise ValueError(f"TabEBM {name} must be positive")
        return parsed

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("TabEBM requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        if frame.empty:
            raise ValueError("TabEBM cannot operate on an empty table")
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "TabEBM requires missing values to be imputed by the train-fitted preprocessing module; "
                f"observed: {observed}"
            )
        for column in dataset_spec.numerical_columns:
            if column in frame:
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
                if not bool(np.isfinite(values).all()):
                    raise ValueError(f"TabEBM numerical column {column!r} contains non-finite values")
                frame[column] = values
        return frame

    @staticmethod
    @contextlib.contextmanager
    def _scoped_randomness(seed: int, num_threads: int) -> Iterator[None]:
        import torch

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        previous_threads = torch.get_num_threads()
        previous_global_seed = os.environ.get("PL_GLOBAL_SEED")
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
            if previous_global_seed is None:
                os.environ.pop("PL_GLOBAL_SEED", None)
            else:
                os.environ["PL_GLOBAL_SEED"] = previous_global_seed

    @staticmethod
    def _parameters(spec: RunSpec) -> dict[str, Any]:
        return {
            "max_data_size": TabEBMAdapter._positive_int(
                "max_data_size", spec.extra.get("max_data_size", 10_000)
            ),
            "starting_point_noise_std": TabEBMAdapter._nonnegative_float(
                "starting_point_noise_std", spec.extra.get("starting_point_noise_std", 0.01)
            ),
            "sgld_step_size": TabEBMAdapter._positive_float(
                "sgld_step_size", spec.extra.get("sgld_step_size", 0.1)
            ),
            "sgld_noise_std": TabEBMAdapter._nonnegative_float(
                "sgld_noise_std", spec.extra.get("sgld_noise_std", 0.01)
            ),
            "sgld_steps": TabEBMAdapter._positive_int("sgld_steps", spec.extra.get("sgld_steps", 200)),
            "distance_negative_class": TabEBMAdapter._positive_float(
                "distance_negative_class", spec.extra.get("distance_negative_class", 5.0)
            ),
            "num_threads": TabEBMAdapter._positive_int("num_threads", spec.extra.get("num_threads", 1)),
        }

    @staticmethod
    def _import_official_class() -> type[Any]:
        verify_tabebm_distribution()
        module = importlib.import_module("tabebm.TabEBM")
        return module.TabEBM

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._load_training_frame(dataset_spec)
        preprocessor = _TabEBMPreprocessor(dataset_spec)
        X, y = preprocessor.fit_transform(frame)
        package = verify_tabebm_distribution()
        parameters = self._parameters(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        state = {
            "schema_version": 1,
            "format": "tabebm-official-package-state",
            "model_id": self.model_name,
            "protocol_id": self.protocol_id,
            "package_version": TABEBM_PACKAGE_VERSION,
            "upstream_commit": TABEBM_UPSTREAM_COMMIT,
            "runtime_files": package["runtime_files"],
            "column_names": list(dataset_spec.column_names),
            "target_columns": list(dataset_spec.target_columns),
            "task_type": dataset_spec.task_type,
            "train_rows": len(frame),
            "train_data_sha256": sha256_file(dataset_spec.train_data_path),
            "preprocessor": preprocessor.state,
            "encoded_shape": list(X.shape),
            "class_counts": {str(key): int(value) for key, value in zip(*np.unique(y, return_counts=True))},
            "parameters": parameters,
            "privacy_guarantee": False,
        }
        atomic_write_json(checkpoint_path, state)
        atomic_write_json(
            self._checkpoint_metadata_path(checkpoint_path),
            {
                "schema_version": 1,
                "format": "safe-json-no-training-rows",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "privacy_guarantee": False,
                "access_control_required": True,
                "notes": "The checkpoint stores preprocessing statistics and a training-file digest, not rows.",
            },
        )
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.repo_root,
                notes=[
                    f"Stored safe TabEBM state at {checkpoint_path}.",
                    "The official model has no separate fit stage; TabPFN-backed generation occurs only during sample.",
                    "Full sampling requires explicit gated-model opt-in and accepted Prior Labs terms.",
                ],
            )
        )

    def _load_state(self, checkpoint_path: Path, dataset_spec: DatasetSpec) -> dict[str, Any]:
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing safe TabEBM checkpoint: {checkpoint_path}")
        state = read_json(checkpoint_path)
        required = {
            "schema_version",
            "format",
            "model_id",
            "protocol_id",
            "package_version",
            "upstream_commit",
            "runtime_files",
            "column_names",
            "target_columns",
            "task_type",
            "train_rows",
            "train_data_sha256",
            "preprocessor",
            "encoded_shape",
            "class_counts",
            "parameters",
            "privacy_guarantee",
        }
        if set(state) != required or state.get("format") != "tabebm-official-package-state":
            raise ValueError("Malformed TabEBM checkpoint")
        package = verify_tabebm_distribution()
        if (
            state["package_version"] != TABEBM_PACKAGE_VERSION
            or state["upstream_commit"] != TABEBM_UPSTREAM_COMMIT
            or state["runtime_files"] != package["runtime_files"]
            or state["column_names"] != dataset_spec.column_names
            or state["target_columns"] != dataset_spec.target_columns
            or state["task_type"] != dataset_spec.task_type
        ):
            raise ValueError("TabEBM checkpoint package or dataset contract mismatch")
        return state

    @staticmethod
    def _round_robin(blocks: dict[int, np.ndarray], requested: int) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        targets: list[int] = []
        index = 0
        class_ids = sorted(blocks)
        while len(rows) < requested:
            progressed = False
            for class_id in class_ids:
                block = blocks[class_id]
                if index < len(block):
                    rows.append(block[index])
                    targets.append(class_id)
                    progressed = True
                    if len(rows) == requested:
                        break
            if not progressed:
                break
            index += 1
        if len(rows) != requested:
            raise RuntimeError("Official TabEBM returned too few rows for the exact-row adapter boundary")
        return np.stack(rows), np.asarray(targets, dtype=np.int64)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        if not self._allow_gated_sampling(spec):
            raise RuntimeError(
                "TabEBM sampling requires Prior Labs TabPFN-v2 model access. Accept the upstream terms and set "
                "sample.extra.allow_gated_model=true; this explicit opt-in is never inferred."
            )
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._load_training_frame(dataset_spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        state = self._load_state(checkpoint_path, dataset_spec)
        if dataset_spec.train_data_path is None or sha256_file(dataset_spec.train_data_path) != state["train_data_sha256"]:
            raise ValueError("TabEBM training data differs from the safe checkpoint identity")
        preprocessor = _TabEBMPreprocessor(dataset_spec)
        X, y = preprocessor.fit_transform(frame)
        if preprocessor.state != state["preprocessor"] or list(X.shape) != state["encoded_shape"]:
            raise ValueError("TabEBM preprocessing state differs from the safe checkpoint")
        requested = spec.num_samples or len(frame)
        if requested < 1:
            raise ValueError("TabEBM num_samples must be positive")
        class_count = len(np.unique(y))
        per_class = int(np.ceil(requested / class_count))
        parameters = dict(state["parameters"])
        parameters.update(self._parameters(spec))
        official_class = self._import_official_class()
        import torch

        if spec.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"TabEBM requested unavailable accelerator: {spec.device}")
        original_cuda_available = torch.cuda.is_available
        try:
            if spec.device == "cpu":
                torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
            with self._scoped_randomness(spec.seed, parameters["num_threads"]):
                model = official_class(max_data_size=parameters["max_data_size"])
                generated = model.generate(
                    X,
                    y,
                    num_samples=per_class,
                    starting_point_noise_std=parameters["starting_point_noise_std"],
                    sgld_step_size=parameters["sgld_step_size"],
                    sgld_noise_std=parameters["sgld_noise_std"],
                    sgld_steps=parameters["sgld_steps"],
                    distance_negative_class=parameters["distance_negative_class"],
                    seed=spec.seed,
                    debug=bool(spec.extra.get("debug", False)),
                )
        finally:
            torch.cuda.is_available = original_cuda_available  # type: ignore[method-assign]
        blocks: dict[int, np.ndarray] = {}
        for class_id in range(class_count):
            key = f"class_{class_id}"
            block = np.asarray(generated.get(key), dtype=np.float64)
            if block.ndim != 2 or block.shape != (per_class, X.shape[1]) or not bool(np.isfinite(block).all()):
                raise RuntimeError(f"Official TabEBM returned an invalid block for {key}")
            blocks[class_id] = block
        sampled_X, sampled_y = self._round_robin(blocks, requested)
        sample_df = preprocessor.inverse_transform(sampled_X, sampled_y)
        if len(sample_df) != requested or bool(sample_df.isna().any().any()):
            raise RuntimeError("TabEBM failed its exact-row or missing-value postcondition")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.repo_root,
                generated_sample_path=sample_path,
                notes=[
                    f"Generated {per_class} rows per class through official TabEBM and retained {requested} rows by deterministic round-robin truncation.",
                    "No privacy guarantee is implied; the generated output and model access credentials require appropriate controls.",
                ],
            )
        )

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


__all__ = ["TabEBMAdapter", "verify_tabebm_distribution"]
