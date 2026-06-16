from __future__ import annotations

import json
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from standardized_tabular_diffusion.config import ExperimentConfig
from standardized_tabular_diffusion.datasets import get_dataset_spec
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

    def _run_python(self, args: list[str], cwd: Path, *, module: bool = False) -> None:
        command = [sys.executable]
        if module:
            command.append("-m")
        command.extend(args)
        subprocess.run(command, cwd=cwd, check=True)

    def resolve_dataset_spec(self, spec: RunSpec) -> DatasetSpec:
        return get_dataset_spec(spec.dataset, repo_root=self.repo_root)

    def build_run_spec(self, config: ExperimentConfig, dataset_spec: DatasetSpec | None = None) -> RunSpec:
        dataset_spec = dataset_spec or get_dataset_spec(config.dataset, repo_root=self.repo_root)
        spec = config.to_run_spec()
        spec.extra.setdefault("dataset_spec", dataset_spec.to_dict())
        spec.extra.setdefault("config", config.to_dict())
        return spec

    def train_from_config(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
    ) -> ArtifactBundle:
        return self.train(self.build_run_spec(config, dataset_spec=dataset_spec))

    def sample_from_config(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
    ) -> ArtifactBundle:
        return self.sample(self.build_run_spec(config, dataset_spec=dataset_spec))

    def evaluate_from_config(
        self,
        config: ExperimentConfig,
        dataset_spec: DatasetSpec | None = None,
    ) -> ArtifactBundle:
        return self.evaluate(self.build_run_spec(config, dataset_spec=dataset_spec))

    def _write_bundle(self, bundle: ArtifactBundle) -> ArtifactBundle:
        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = bundle.output_dir / "artifacts.json"
        manifest_path.write_text(json.dumps(bundle.to_dict(), indent=2))
        return bundle
