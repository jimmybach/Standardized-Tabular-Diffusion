from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.upstream_sources import validate_upstream_source


class TabSDSAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    """Adapter around the checksum-locked method-author TabSDS Python functions."""

    model_name = "tabsds"
    upstream_dirname = "."
    checkpoint_filename = "model.tabsds.json"
    source_environment_variable = "STANDARDIZED_TABULAR_DIFFUSION_TABSDS_SOURCE"
    upstream_commit = "866501495069c7e1300bdea91c411f1947d19f2f"
    protocol_id = "tabsds-official-source-parity-v1"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        return spec.checkpoint_path or spec.output_dir / self.checkpoint_filename

    @staticmethod
    def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")

    def _resolve_source_root(self, spec: RunSpec) -> tuple[Path, dict[str, Any]]:
        configured = spec.extra.get("source_dir") or os.environ.get(self.source_environment_variable)
        source_root = Path(configured) if configured is not None else self.repo_root / ".cache" / "upstream-sources" / self.model_name / self.upstream_commit
        try:
            source = validate_upstream_source(self.model_name, source_root)
        except (FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError(
                "TabSDS requires the checksum-locked method-author source. Run "
                "`python -m standardized_tabular_diffusion.cli materialize-model-source --model tabsds`, "
                f"or provide spec.extra['source_dir']; underlying error: {exc}"
            ) from exc
        if source["upstream_commit"] != self.upstream_commit:
            raise RuntimeError("TabSDS source validation returned an unexpected commit")
        return source_root.resolve(), source

    @staticmethod
    def _load_module(path: Path, name: str, initial_globals: dict[str, Any] | None = None) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import locked TabSDS source file: {path}")
        module = importlib.util.module_from_spec(spec)
        if initial_globals:
            module.__dict__.update(initial_globals)
        spec.loader.exec_module(module)
        return module

    def _official_functions(self, source_root: Path) -> tuple[ModuleType, ModuleType]:
        core = self._load_module(
            source_root / "utility_functions_syn_tab_sjppds_for_icml_2025.py",
            "_standardized_tabsds_core",
        )
        wrapper = self._load_module(
            source_root / "utility_functions_additional_for_icml_2025.py",
            "_standardized_tabsds_wrapper",
            {"np": np, "pd": pd},
        )
        # The official notebooks import both utility files into one global namespace.
        # Reproduce that import boundary without changing either locked source file.
        wrapper.np = np
        wrapper.pd = pd
        for name in (
            "sequential_jppds",
            "cat_sjppds",
            "categorical_to_numeric",
            "numeric_to_categorical",
        ):
            setattr(wrapper, name, getattr(core, name))
        return core, wrapper

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"TabSDS {name} must be a positive integer")
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"TabSDS {name} must be a positive integer")
        return parsed

    @staticmethod
    def _roles(dataset_spec: DatasetSpec) -> tuple[list[str], list[str]]:
        if len(dataset_spec.target_columns) != 1:
            raise ValueError("TabSDS requires exactly one target column")
        numerical = list(dataset_spec.numerical_columns)
        categorical = list(dataset_spec.categorical_columns)
        target = dataset_spec.target_columns[0]
        if target not in numerical and target not in categorical:
            (categorical if dataset_spec.task_type == "classification" else numerical).append(target)
        if set(numerical) & set(categorical):
            raise ValueError("TabSDS numerical and categorical roles must be disjoint")
        if set(numerical) | set(categorical) != set(dataset_spec.column_names):
            raise ValueError("TabSDS requires every column to have exactly one declared numerical/categorical role")
        return numerical, categorical

    def _load_training_frame(self, dataset_spec: DatasetSpec) -> pd.DataFrame:
        if dataset_spec.train_data_path is None:
            raise FileNotFoundError("TabSDS requires dataset_spec.train_data_path")
        frame = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names].copy()
        if frame.empty:
            raise ValueError("TabSDS cannot train on an empty table")
        missing = frame.isna().sum()
        if bool(missing.any()):
            observed = {str(column): int(count) for column, count in missing.items() if count}
            raise ValueError(
                "TabSDS requires missing values to be imputed by the train-fitted preprocessing module; "
                f"observed: {observed}"
            )
        numerical, categorical = self._roles(dataset_spec)
        for column in numerical:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"TabSDS numerical column {column!r} contains non-finite values")
            frame[column] = values
        for column in categorical:
            frame[column] = frame[column].astype(str)
        return frame

    @staticmethod
    @contextlib.contextmanager
    def _scoped_numpy_seed(seed: int) -> Iterator[None]:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)

    def _generate_once(
        self,
        frame: pd.DataFrame,
        dataset_spec: DatasetSpec,
        wrapper: ModuleType,
        *,
        n_levels: int,
    ) -> pd.DataFrame:
        numerical, categorical = self._roles(dataset_spec)
        num_indices = [dataset_spec.column_names.index(column) for column in numerical] or None
        cat_indices = [dataset_spec.column_names.index(column) for column in categorical] or None
        generated = wrapper.tab_sjppds(
            dat=frame.copy(),
            num_variables=num_indices,
            cat_variables=cat_indices,
            n_levels=n_levels,
            shuffle_type="simple",
            verbose=False,
        )
        if not isinstance(generated, pd.DataFrame) or generated.shape != frame.shape:
            raise RuntimeError("The locked TabSDS source returned an invalid table shape")
        generated.columns = dataset_spec.column_names
        return generated.reset_index(drop=True)

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._load_training_frame(dataset_spec)
        source_root, source = self._resolve_source_root(spec)
        self._official_functions(source_root)
        n_levels = self._positive_int("n_levels", spec.extra.get("n_levels", 30))
        checkpoint_path = self._resolve_checkpoint_path(spec)
        payload = {
            "schema_version": 1,
            "format": "tabsds-official-source-state",
            "model_id": self.model_name,
            "protocol_id": self.protocol_id,
            "upstream_commit": self.upstream_commit,
            "source_manifest_sha256": source["manifest_sha256"],
            "column_names": list(dataset_spec.column_names),
            "numerical_columns": self._roles(dataset_spec)[0],
            "categorical_columns": self._roles(dataset_spec)[1],
            "target_columns": list(dataset_spec.target_columns),
            "task_type": dataset_spec.task_type,
            "train_rows": len(frame),
            "train_data_sha256": sha256_file(dataset_spec.train_data_path),
            "n_levels": n_levels,
            "shuffle_type": "simple",
        }
        atomic_write_json(checkpoint_path, payload)
        atomic_write_json(
            self._checkpoint_metadata_path(checkpoint_path),
            {
                "schema_version": 1,
                "format": "safe-json-no-training-rows",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "privacy_guarantee": False,
                "access_control_required": True,
                "notes": "The checkpoint stores schema and a training-file digest, not row-level training data.",
            },
        )
        source_after = validate_upstream_source(self.model_name, source_root)
        if source_after["manifest_sha256"] != source["manifest_sha256"]:
            raise RuntimeError("TabSDS source changed during training")
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=source_root,
                notes=[
                    f"Stored safe TabSDS recipe state at {checkpoint_path}.",
                    "No upstream source was modified; the adapter reproduces the official notebook import boundary.",
                ],
            )
        )

    def _load_state(self, checkpoint_path: Path, dataset_spec: DatasetSpec) -> dict[str, Any]:
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing safe TabSDS checkpoint: {checkpoint_path}")
        payload = read_json(checkpoint_path)
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
            "train_rows",
            "train_data_sha256",
            "n_levels",
            "shuffle_type",
        }
        if set(payload) != required or payload.get("format") != "tabsds-official-source-state":
            raise ValueError("Malformed TabSDS checkpoint")
        if payload["model_id"] != self.model_name or payload["upstream_commit"] != self.upstream_commit:
            raise ValueError("TabSDS checkpoint source identity mismatch")
        expected_roles = self._roles(dataset_spec)
        if (
            payload["column_names"] != dataset_spec.column_names
            or payload["numerical_columns"] != expected_roles[0]
            or payload["categorical_columns"] != expected_roles[1]
            or payload["target_columns"] != dataset_spec.target_columns
            or payload["task_type"] != dataset_spec.task_type
        ):
            raise ValueError("TabSDS checkpoint dataset contract mismatch")
        return payload

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_spec = self.resolve_dataset_spec(spec)
        frame = self._load_training_frame(dataset_spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        state = self._load_state(checkpoint_path, dataset_spec)
        if dataset_spec.train_data_path is None or sha256_file(dataset_spec.train_data_path) != state["train_data_sha256"]:
            raise ValueError("TabSDS training data differs from the safe checkpoint identity")
        if len(frame) != state["train_rows"]:
            raise ValueError("TabSDS training row count differs from the checkpoint")
        source_root, source = self._resolve_source_root(spec)
        if source["manifest_sha256"] != state["source_manifest_sha256"]:
            raise ValueError("TabSDS source manifest differs from the checkpoint")
        _, wrapper = self._official_functions(source_root)
        requested = spec.num_samples or len(frame)
        if requested < 1:
            raise ValueError("TabSDS num_samples must be positive")
        blocks: list[pd.DataFrame] = []
        remaining = requested
        round_index = 0
        with self._scoped_numpy_seed(spec.seed):
            while remaining:
                block = self._generate_once(
                    frame,
                    dataset_spec,
                    wrapper,
                    n_levels=int(state["n_levels"]),
                )
                take = min(remaining, len(block))
                blocks.append(block.iloc[:take].copy())
                remaining -= take
                round_index += 1
        sample_df = pd.concat(blocks, ignore_index=True)[dataset_spec.column_names]
        if len(sample_df) != requested or bool(sample_df.isna().any().any()):
            raise RuntimeError("TabSDS failed its exact-row or missing-value postcondition")
        sample_path = spec.output_dir / "samples.csv"
        self._write_dataframe_csv(sample_df, sample_path)
        validate_upstream_source(self.model_name, source_root)
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=source_root,
                generated_sample_path=sample_path,
                notes=[
                    f"Generated {requested} rows using {round_index} unchanged official TabSDS table-shuffle call(s).",
                    "Requests larger than the training table repeat the official same-size generator and truncate only the final block.",
                ],
            )
        )

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)
