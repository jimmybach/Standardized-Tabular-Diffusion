from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter

PACKAGE_NAME = "mostlyai-engine"
PACKAGE_VERSION = "2.6.2"
UPSTREAM_REPOSITORY = "https://github.com/mostly-ai/mostlyai-engine"
UPSTREAM_TAG = "2.6.2"
UPSTREAM_COMMIT = "0b96f02e4fad47c7c19c985fda4311230e20bbb5"
UPSTREAM_TREE = "199a4085315e601261898007b0dd4ac532d355fe"
WHEEL_MANIFEST_PATH = (
    Path("standardized_tabular_diffusion") / "resources" / "upstream" / "tabularargn-wheel-manifest.json"
)

# These are the direct runtime versions in the method-author 2.6.2 release lock,
# with the package version corrected to the released tag/wheel identity.
EXPECTED_RUNTIME_VERSIONS = {
    "accelerate": "1.12.0",
    "datasets": "4.6.1",
    "huggingface-hub": "1.12.2",
    "joblib": "1.5.3",
    "json-repair": "0.58.1",
    "mostlyai-engine": PACKAGE_VERSION,
    "numpy": "2.2.6",
    "opacus": "1.6.0",
    "pandas": "3.0.2",
    "peft": "0.19.1",
    "psutil": "5.9.8",
    "pyarrow": "23.0.1",
    "scikit-learn": "1.8.0",
    "setuptools": "80.10.2",
    "tokenizers": "0.22.2",
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.7.0",
    "xgrammar": "0.1.33",
}

_COMMON_INTERNAL_KEYS = {"action_extras", "config", "dataset_spec", "evaluation", "tags"}
_TRAIN_KEYS = {
    "batch_size",
    "enable_flexible_generation",
    "gradient_accumulation_steps",
    "max_epochs",
    "max_train_rows",
    "max_training_time",
    "model",
    "tgt_encoding_types",
    "value_protection",
    "verbose",
} | _COMMON_INTERNAL_KEYS
_SAMPLE_KEYS = {
    "allow_unsafe_external_checkpoint",
    "batch_size",
    "checkpoint_metadata_path",
    "rare_category_replacement_method",
    "sampling_temperature",
    "sampling_top_p",
} | _COMMON_INTERNAL_KEYS
_MODEL_IDS = {"MOSTLY_AI/Small", "MOSTLY_AI/Medium", "MOSTLY_AI/Large"}
_ENCODING_TYPES = {
    "AUTO",
    "TABULAR_CATEGORICAL",
    "TABULAR_NUMERIC_AUTO",
    "TABULAR_NUMERIC_DISCRETE",
    "TABULAR_NUMERIC_BINNED",
    "TABULAR_NUMERIC_DIGIT",
    "TABULAR_CHARACTER",
    "TABULAR_DATETIME",
    "TABULAR_DATETIME_RELATIVE",
    "TABULAR_LAT_LONG",
}


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
    if package_name in {"torch", "torchaudio", "torchvision"}:
        return observed.partition("+")[0]
    return observed


def _load_wheel_manifest(repo_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = repo_root / WHEEL_MANIFEST_PATH
    manifest = read_json(manifest_path)
    package = manifest.get("package", {})
    source = manifest.get("source", {})
    if manifest.get("model_id") != "tabularargn":
        raise RuntimeError("TabularARGN wheel manifest has an unexpected model identity")
    if (
        package.get("name") != PACKAGE_NAME
        or package.get("version") != PACKAGE_VERSION
        or source.get("repository") != UPSTREAM_REPOSITORY
        or source.get("tag") != UPSTREAM_TAG
        or source.get("commit") != UPSTREAM_COMMIT
        or source.get("tree") != UPSTREAM_TREE
    ):
        raise RuntimeError("TabularARGN wheel manifest does not match the adapter's locked release")
    return manifest, manifest_path


def verify_tabularargn_distribution(repo_root: Path) -> dict[str, Any]:
    """Verify every RECORD-hashed file installed from the official locked wheel."""

    manifest, manifest_path = _load_wheel_manifest(repo_root)
    try:
        installed = distribution(PACKAGE_NAME)
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            'TabularARGN requires the locked official package; install "standardized-tabular-diffusion[tabularargn]".'
        ) from exc
    if installed.version != PACKAGE_VERSION:
        raise RuntimeError(
            f"TabularARGN package version mismatch: expected {PACKAGE_VERSION}, observed {installed.version}."
        )

    distribution_root = Path(installed.locate_file("")).resolve()
    verified: list[dict[str, Any]] = []
    for record in manifest["installed_files"]:
        relative = Path(record["path"])
        installed_path = Path(installed.locate_file(relative))
        if installed_path.is_symlink() or not installed_path.is_file():
            raise RuntimeError(f"Locked TabularARGN distribution file is missing or symlinked: {relative}")
        resolved = installed_path.resolve()
        if not resolved.is_relative_to(distribution_root):
            raise RuntimeError(f"TabularARGN distribution file escaped its installation root: {resolved}")
        observed_bytes = resolved.stat().st_size
        observed_sha256 = _sha256_file(resolved)
        if observed_bytes != record["bytes"] or observed_sha256 != record["sha256"]:
            raise RuntimeError(
                f"Installed TabularARGN file differs from the locked official wheel: {relative}; "
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
                f"TabularARGN validated runtime dependency is missing: {package_name}=={expected}"
            ) from exc
        if observed != expected:
            raise RuntimeError(
                f"TabularARGN runtime version mismatch for {package_name}: expected {expected}, observed {observed}"
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


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"TabularARGN model store must be a regular directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PermissionError(f"TabularARGN model store must not contain symlinks: {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not records:
        raise RuntimeError(f"TabularARGN model store contains no files: {root}")
    return records


def _dataset_identity(dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "name": dataset_spec.name,
        "task_type": dataset_spec.task_type,
        "column_names": list(dataset_spec.column_names),
        "numerical_columns": list(dataset_spec.numerical_columns),
        "categorical_columns": list(dataset_spec.categorical_columns),
        "target_columns": list(dataset_spec.target_columns),
    }


class TabularARGNAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    model_name = "tabularargn"
    upstream_dirname = "."
    package_name = PACKAGE_NAME
    package_version = PACKAGE_VERSION
    upstream_commit = UPSTREAM_COMMIT
    workspace_name = "tabularargn_workspace"
    metadata_filename = "tabularargn-model-metadata.json"

    def _workspace_dir(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or (spec.output_dir / self.workspace_name)

    def _metadata_path(self, spec: RunSpec) -> Path:
        configured = spec.extra.get("checkpoint_metadata_path")
        return Path(configured) if configured is not None else spec.output_dir / self.metadata_filename

    def _validate_dataset_contract(self, dataset_spec: DatasetSpec) -> None:
        if dataset_spec.task_type not in {"classification", "regression"}:
            raise ValueError("TabularARGN supports classification and regression DatasetSpecs")
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("TabularARGN requires exactly one target column")
        if len(set(dataset_spec.column_names)) != len(dataset_spec.column_names):
            raise ValueError("TabularARGN requires unique column names")
        if any(not isinstance(column, str) or not column for column in dataset_spec.column_names):
            raise ValueError("TabularARGN requires non-empty string column names")
        numerical = set(dataset_spec.numerical_columns)
        categorical = set(dataset_spec.categorical_columns)
        targets = set(dataset_spec.target_columns)
        if numerical & categorical or (numerical | categorical) & targets:
            raise ValueError("TabularARGN feature and target roles must be disjoint")
        if numerical | categorical | targets != set(dataset_spec.column_names):
            raise ValueError("TabularARGN DatasetSpec roles must cover every column exactly once")

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        self._validate_dataset_contract(dataset_spec)
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("TabularARGN requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)
        if list(frame.columns) != dataset_spec.column_names:
            raise ValueError(
                "TabularARGN training CSV columns must exactly match the canonical ordered DatasetSpec columns"
            )
        if frame.empty:
            raise ValueError("TabularARGN requires at least one training row")
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "TabularARGN does not accept missing values in the standardized adapter. "
                f"Run the centralized train-only mean/mode imputer first; observed: {observed}"
            )

        target = dataset_spec.target_columns[0]
        numeric = list(dataset_spec.numerical_columns)
        if dataset_spec.task_type == "regression":
            numeric.append(target)
        for column in numeric:
            converted = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(converted.to_numpy(dtype=float)).all():
                raise ValueError(f"TabularARGN requires finite numerical values; invalid column: {column}")
            frame[column] = converted

        categorical = list(dataset_spec.categorical_columns)
        if dataset_spec.task_type == "classification":
            categorical.append(target)
        domains: dict[str, list[str]] = {}
        for column in categorical:
            frame[column] = frame[column].astype(str)
            domains[column] = sorted(frame[column].unique().tolist())
            if not domains[column]:
                raise ValueError(f"TabularARGN categorical column has an empty training domain: {column}")
        return frame[dataset_spec.column_names].copy(), domains

    def _limit_training_frame(self, frame: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        configured = spec.extra.get("max_train_rows")
        if configured is None:
            return frame
        max_rows = int(configured)
        if max_rows <= 0:
            raise ValueError("TabularARGN max_train_rows must be positive")
        if len(frame) <= max_rows:
            return frame
        return frame.sample(n=max_rows, random_state=spec.seed).reset_index(drop=True)

    def _validate_train_controls(self, spec: RunSpec, dataset_spec: DatasetSpec) -> dict[str, Any]:
        unknown = sorted(set(spec.extra) - _TRAIN_KEYS)
        if unknown:
            raise ValueError(f"Unknown TabularARGN training controls: {unknown}")
        if isinstance(spec.seed, bool) or not isinstance(spec.seed, int) or not 0 <= spec.seed <= 2**32 - 1:
            raise ValueError("TabularARGN seed must be an integer in [0, 2**32 - 1]")
        model = spec.extra.get("model")
        if model is not None and model not in _MODEL_IDS:
            raise ValueError(f"TabularARGN model must be one of {sorted(_MODEL_IDS)}")
        for name in ("max_training_time", "max_epochs"):
            value = spec.extra.get(name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"TabularARGN {name} must be positive")
        for name in ("batch_size", "gradient_accumulation_steps"):
            value = spec.extra.get(name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"TabularARGN {name} must be positive")
        for name in ("enable_flexible_generation", "value_protection"):
            value = spec.extra.get(name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"TabularARGN {name} must be a boolean")
        verbose = spec.extra.get("verbose", 0)
        if isinstance(verbose, bool) or int(verbose) not in {0, 1}:
            raise ValueError("TabularARGN verbose must be 0 or 1")
        encoding_types = spec.extra.get("tgt_encoding_types")
        if encoding_types is not None:
            if not isinstance(encoding_types, dict):
                raise TypeError("TabularARGN tgt_encoding_types must be a JSON object")
            unknown_columns = sorted(set(encoding_types) - set(dataset_spec.column_names))
            if unknown_columns:
                raise ValueError(f"TabularARGN encoding types refer to unknown columns: {unknown_columns}")
            invalid = {column: value for column, value in encoding_types.items() if value not in _ENCODING_TYPES}
            if invalid:
                raise ValueError(f"TabularARGN encoding types are invalid: {invalid}")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", spec.device):
            raise ValueError("TabularARGN device must be 'cpu', 'cuda', or 'cuda:<index>'")
        return {
            "model": model,
            "max_training_time": float(spec.extra.get("max_training_time", 14400.0)),
            "max_epochs": float(spec.extra.get("max_epochs", 100.0)),
            "batch_size": None if spec.extra.get("batch_size") is None else int(spec.extra["batch_size"]),
            "gradient_accumulation_steps": (
                None
                if spec.extra.get("gradient_accumulation_steps") is None
                else int(spec.extra["gradient_accumulation_steps"])
            ),
            "enable_flexible_generation": spec.extra.get("enable_flexible_generation", True),
            "value_protection": spec.extra.get("value_protection", True),
            "tgt_encoding_types": encoding_types,
            "device": spec.device,
            "random_state": int(spec.seed),
            "verbose": int(verbose),
        }

    def _validate_sample_controls(self, spec: RunSpec) -> dict[str, Any]:
        unknown = sorted(set(spec.extra) - _SAMPLE_KEYS)
        if unknown:
            raise ValueError(f"Unknown TabularARGN sampling controls: {unknown}")
        if isinstance(spec.seed, bool) or not isinstance(spec.seed, int) or not 0 <= spec.seed <= 2**32 - 1:
            raise ValueError("TabularARGN seed must be an integer in [0, 2**32 - 1]")
        if spec.num_samples is not None and (
            isinstance(spec.num_samples, bool) or not isinstance(spec.num_samples, int) or spec.num_samples <= 0
        ):
            raise ValueError("TabularARGN num_samples must be a positive integer")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", spec.device):
            raise ValueError("TabularARGN device must be 'cpu', 'cuda', or 'cuda:<index>'")
        batch_size = spec.extra.get("batch_size")
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError("TabularARGN sampling batch_size must be positive")
        temperature = float(spec.extra.get("sampling_temperature", 1.0))
        top_p = float(spec.extra.get("sampling_top_p", 1.0))
        if temperature <= 0:
            raise ValueError("TabularARGN sampling_temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("TabularARGN sampling_top_p must be in (0, 1]")
        rare_method = spec.extra.get("rare_category_replacement_method", "SAMPLE")
        if rare_method not in {"CONSTANT", "SAMPLE"}:
            raise ValueError("TabularARGN rare_category_replacement_method must be CONSTANT or SAMPLE")
        return {
            "sample_size": None if spec.num_samples is None else int(spec.num_samples),
            "batch_size": None if batch_size is None else int(batch_size),
            "sampling_temperature": temperature,
            "sampling_top_p": top_p,
            "device": spec.device,
            "rare_category_replacement_method": rare_method,
        }

    def _import_runtime(self) -> tuple[Any, type[Any], Any, dict[str, Any]]:
        package = verify_tabularargn_distribution(self.repo_root)
        engine = importlib.import_module("mostlyai.engine")
        implementation = importlib.import_module("mostlyai.engine._tabular.interface")
        generation = importlib.import_module("mostlyai.engine.generation")
        random_state = importlib.import_module("mostlyai.engine.random_state")
        common = importlib.import_module("mostlyai.engine._common")
        if engine.TabularARGN is not implementation.TabularARGN:
            raise RuntimeError("TabularARGN public export does not match the locked official implementation class")
        if engine.generate is not generation.generate or engine.set_random_state is not random_state.set_random_state:
            raise RuntimeError("TabularARGN public engine functions differ from the locked official implementation")
        return engine, engine.TabularARGN, common.load_generated_data, package

    @staticmethod
    def _prune_training_data(workspace: Path) -> None:
        # The official persistent workspace is the checkpoint format. OriginalData
        # contains both raw and encoded training rows and is not needed by flat-table
        # generation, so it must not be carried in the reusable artifact.
        original_data = workspace / "OriginalData"
        if original_data.exists():
            shutil.rmtree(original_data)
        synthetic_data = workspace / "SyntheticData"
        if synthetic_data.exists():
            shutil.rmtree(synthetic_data)
        for filename in ("optimizer.pt", "lr-scheduler.pt", "dp-accountant.pt"):
            (workspace / "ModelStore" / "model-data" / filename).unlink(missing_ok=True)

    def _validate_saved_workspace(
        self,
        spec: RunSpec,
        dataset_spec: DatasetSpec,
        workspace: Path,
        package: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = read_json(self._metadata_path(spec))
        if metadata.get("schema_version") != "1.0" or metadata.get("model_id") != self.model_name:
            raise RuntimeError("TabularARGN checkpoint metadata has an unexpected schema or model identity")
        if metadata.get("dataset_identity") != _dataset_identity(dataset_spec):
            raise RuntimeError("TabularARGN checkpoint DatasetSpec differs from the requested dataset contract")
        package_record = metadata.get("package", {})
        for key in ("package_version", "wheel_sha256", "manifest_sha256", "upstream_commit"):
            if package_record.get(key) != package.get(key):
                raise RuntimeError(f"TabularARGN checkpoint package identity differs at {key}")
        if (workspace / "OriginalData").exists():
            raise RuntimeError("TabularARGN checkpoint unexpectedly retains raw or encoded training data")
        observed = _directory_manifest(workspace / "ModelStore")
        if observed != metadata.get("model_store_files"):
            raise RuntimeError("TabularARGN model store differs from its recorded integrity manifest")
        if _sha256_json(observed) != metadata.get("model_store_manifest_sha256"):
            raise RuntimeError("TabularARGN model-store manifest digest mismatch")
        required = {
            "model-data/model-configs.json",
            "model-data/model-weights.pt",
            "tgt-stats/stats.json",
        }
        observed_paths = {record["path"] for record in observed}
        if not required.issubset(observed_paths):
            raise RuntimeError(
                f"TabularARGN model store is missing required inference files: {sorted(required - observed_paths)}"
            )
        return metadata

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        source_frame, categorical_domains = self._load_training_frame(dataset_spec)
        frame = self._limit_training_frame(source_frame, spec)
        constructor = self._validate_train_controls(spec, dataset_spec)
        engine, model_class, _, package = self._import_runtime()
        workspace = self._workspace_dir(spec)
        if not workspace.resolve().is_relative_to(spec.output_dir.resolve()):
            raise PermissionError("TabularARGN training workspace must remain beneath output_dir")
        if workspace.is_symlink():
            raise PermissionError(f"TabularARGN training workspace must not be symlinked: {workspace}")
        if workspace.exists() and not workspace.is_dir():
            raise FileExistsError(f"TabularARGN training workspace is not a directory: {workspace}")
        if workspace.exists() and any(workspace.iterdir()):
            raise FileExistsError(f"TabularARGN training workspace must be empty: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)

        runtime_constructor = {**constructor, "workspace_dir": str(workspace)}
        model = model_class(**runtime_constructor)
        if model.__class__ is not model_class:
            raise RuntimeError("Constructed TabularARGN object differs from the locked official class")
        try:
            model.fit(frame)
        finally:
            # A failed official fit must not leave raw or encoded training rows
            # behind in a reusable output directory.
            self._prune_training_data(workspace)
        model_store_files = _directory_manifest(workspace / "ModelStore")
        required = {
            "model-data/model-configs.json",
            "model-data/model-weights.pt",
            "tgt-stats/stats.json",
        }
        observed_paths = {record["path"] for record in model_store_files}
        if not required.issubset(observed_paths):
            raise RuntimeError(
                f"Official TabularARGN training did not produce a reusable flat-table model: {sorted(required - observed_paths)}"
            )
        metadata = {
            "schema_version": "1.0",
            "model_id": self.model_name,
            "dataset_identity": _dataset_identity(dataset_spec),
            "source_rows": len(source_frame),
            "training_rows": len(frame),
            "training_frame_sha256": hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest(),
            "categorical_domains": categorical_domains,
            "numeric_output_columns": [
                *dataset_spec.numerical_columns,
                *(dataset_spec.target_columns if dataset_spec.task_type == "regression" else []),
            ],
            "training": {"constructor": constructor},
            "package": package,
            "checkpoint_format": "official-mostlyai-engine-workspace",
            "model_store_files": model_store_files,
            "model_store_manifest_sha256": _sha256_json(model_store_files),
            "raw_or_encoded_training_rows_retained": False,
            "validated_scope": "flat-single-table-unconditional-generation",
        }
        atomic_write_json(self._metadata_path(spec), metadata)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=Path(package["distribution_root"]),
            notes=[
                f"Official TabularARGN workspace written to {workspace}.",
                "OriginalData was removed after training; ModelStore remains integrity-manifested for generation.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        self._validate_dataset_contract(dataset_spec)
        sample_kwargs = self._validate_sample_controls(spec)
        engine, _, load_generated_data, package = self._import_runtime()
        workspace = self._validate_trusted_executable_artifact(
            spec,
            self._workspace_dir(spec),
            format_name="TabularARGN workspace",
            allow_directory=True,
        )
        metadata = self._validate_saved_workspace(spec, dataset_spec, workspace, package)

        n_samples = metadata["source_rows"] if sample_kwargs["sample_size"] is None else sample_kwargs["sample_size"]
        sample_kwargs["sample_size"] = int(n_samples)
        engine.set_random_state(int(spec.seed))
        engine.generate(**sample_kwargs, workspace_dir=workspace)
        sample_frame = load_generated_data(workspace).reset_index(drop=True)
        if len(sample_frame) != n_samples:
            raise RuntimeError(
                f"TabularARGN returned {len(sample_frame)} rows for a request of {n_samples}; refusing truncation/padding"
            )
        if list(sample_frame.columns) != dataset_spec.column_names:
            if set(sample_frame.columns) != set(dataset_spec.column_names):
                raise RuntimeError("TabularARGN sample columns differ from the canonical DatasetSpec")
            sample_frame = sample_frame[dataset_spec.column_names].copy()
        if bool(sample_frame.isna().any().any()):
            raise RuntimeError("TabularARGN generated missing values, which are forbidden by the benchmark contract")
        for column in metadata["numeric_output_columns"]:
            values = pd.to_numeric(sample_frame[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"TabularARGN generated non-finite numerical values in {column}")
        for column, domain in metadata["categorical_domains"].items():
            sample_frame[column] = sample_frame[column].astype(str)
            observed = set(sample_frame[column])
            if not observed.issubset(set(domain)):
                raise RuntimeError(
                    f"TabularARGN generated out-of-domain values in {column}: {sorted(observed - set(domain))}"
                )

        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_frame, sample_path)
        atomic_write_json(
            spec.output_dir / "tabularargn-sample-metadata.json",
            {
                "schema_version": "1.0",
                "model_id": self.model_name,
                "dataset": dataset_spec.name,
                "seed": int(spec.seed),
                "requested_rows": int(n_samples),
                "sample_sha256": _sha256_file(sample_path),
                "model_store_manifest_sha256": metadata["model_store_manifest_sha256"],
                "package_manifest_sha256": package["manifest_sha256"],
                "sampling": sample_kwargs,
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
    "TabularARGNAdapter",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_TAG",
    "UPSTREAM_TREE",
    "verify_tabularargn_distribution",
]
