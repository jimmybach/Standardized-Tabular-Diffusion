from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from standardized_tabular_diffusion.config import ExperimentConfig
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, atomic_write_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec


class BaseModelAdapter(ABC):
    model_name: str
    upstream_dirname: str

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.upstream_root = repo_root / self.upstream_dirname

    @abstractmethod
    def train(self, spec: RunSpec) -> ArtifactBundle:
        raise NotImplementedError

    @abstractmethod
    def sample(self, spec: RunSpec) -> ArtifactBundle:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        raise NotImplementedError

    def _ensure_output_dir(self, spec: RunSpec) -> None:
        spec.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_trusted_executable_artifact(
        self,
        spec: RunSpec,
        path: Path,
        *,
        format_name: str,
        allow_directory: bool = False,
    ) -> Path:
        """Reject silent loading of code-executing artifacts from arbitrary locations."""

        if path.is_symlink():
            raise PermissionError(f"Refusing to load a symlinked {format_name} artifact: {path}")
        resolved_path = path.resolve(strict=True)
        output_root = spec.output_dir.resolve()
        if not resolved_path.is_relative_to(output_root) and not bool(
            spec.extra.get("allow_unsafe_external_checkpoint", False)
        ):
            raise PermissionError(
                f"Refusing to load external {format_name} artifact {resolved_path}. These formats can execute code. "
                "Use a checkpoint produced inside output_dir, or explicitly set "
                "allow_unsafe_external_checkpoint=true only after verifying its provenance and integrity."
            )
        if not resolved_path.is_file() and not (allow_directory and resolved_path.is_dir()):
            expected = "file or directory" if allow_directory else "regular file"
            raise FileNotFoundError(f"Expected a {expected} for the {format_name} artifact: {resolved_path}")
        return resolved_path

    def _run_python(self, args: list[str], cwd: Path, *, module: bool = False) -> None:
        command = [sys.executable]
        if module:
            command.append("-m")
        command.extend(args)
        subprocess.run(command, cwd=cwd, check=True)

    def _write_dataframe_csv(self, frame: object, path: Path) -> None:
        csv_text = frame.to_csv(index=False)  # type: ignore[attr-defined]
        atomic_write_bytes(path, csv_text.encode("utf-8"))

    def resolve_dataset_spec(self, spec: RunSpec) -> DatasetSpec:
        embedded_spec = spec.extra.get("dataset_spec")
        if embedded_spec:
            return DatasetSpec(
                name=embedded_spec["name"],
                task_type=embedded_spec["task_type"],
                column_names=list(embedded_spec["column_names"]),
                numerical_columns=list(embedded_spec["numerical_columns"]),
                categorical_columns=list(embedded_spec["categorical_columns"]),
                target_columns=list(embedded_spec["target_columns"]),
                metadata_path=Path(embedded_spec["metadata_path"]),
                train_data_path=None
                if embedded_spec.get("train_data_path") is None
                else Path(embedded_spec["train_data_path"]),
                val_data_path=None
                if embedded_spec.get("val_data_path") is None
                else Path(embedded_spec["val_data_path"]),
                test_data_path=None
                if embedded_spec.get("test_data_path") is None
                else Path(embedded_spec["test_data_path"]),
                provenance=list(embedded_spec.get("provenance", [])),
                extra=dict(embedded_spec.get("extra", {})),
            )
        return get_dataset_spec(spec.dataset, repo_root=self.repo_root)

    def build_run_spec(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
        *,
        action: str | None = None,
    ) -> RunSpec:
        dataset_spec = dataset_spec or get_dataset_spec(config.dataset, repo_root=self.repo_root)
        spec = config.to_run_spec(action=action)
        spec.extra.setdefault("dataset_spec", dataset_spec.to_dict())
        spec.extra.setdefault("config", config.to_dict())
        return spec

    def train_from_config(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
    ) -> ArtifactBundle:
        return self.train(self.build_run_spec(config, dataset_spec=dataset_spec, action="train"))

    def sample_from_config(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
    ) -> ArtifactBundle:
        return self.sample(self.build_run_spec(config, dataset_spec=dataset_spec, action="sample"))

    def evaluate_from_config(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
    ) -> ArtifactBundle:
        return self.evaluate(self.build_run_spec(config, dataset_spec=dataset_spec, action="evaluate"))

    def _write_bundle(self, bundle: ArtifactBundle) -> ArtifactBundle:
        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = bundle.output_dir / "artifacts.json"
        atomic_write_json(manifest_path, bundle.to_dict())
        return bundle
