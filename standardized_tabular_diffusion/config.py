from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.interfaces import RunSpec


@dataclass
class TrainConfig:
    enabled: bool = True
    seed: int = 0
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleConfig:
    enabled: bool = True
    num_samples: int | None = None
    checkpoint_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    enabled: bool = True
    total_time_limit_seconds: int = 900
    compute_density: bool = True
    compute_ml_efficacy: bool = True
    compute_detection: bool = True
    compute_privacy: bool = True
    compute_structural_fidelity: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    model: str
    dataset: str
    output_dir: str
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    upstream_config_path: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_run_spec(self) -> RunSpec:
        extra = {
            "evaluation": asdict(self.evaluation),
            "tags": list(self.tags),
        }
        extra.update(self.sample.extra)
        extra.update(self.evaluation.extra)
        extra.update(self.train.extra)
        return RunSpec(
            model=self.model,
            dataset=self.dataset,
            output_dir=Path(self.output_dir),
            device=self.train.device,
            seed=self.train.seed,
            num_samples=self.sample.num_samples,
            checkpoint_path=None if self.sample.checkpoint_path is None else Path(self.sample.checkpoint_path),
            upstream_config_path=None if self.upstream_config_path is None else Path(self.upstream_config_path),
            extra=extra,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text())
    return ExperimentConfig(
        model=payload["model"],
        dataset=payload["dataset"],
        output_dir=payload["output_dir"],
        train=TrainConfig(**payload.get("train", {})),
        sample=SampleConfig(**payload.get("sample", {})),
        evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        upstream_config_path=payload.get("upstream_config_path"),
        tags=payload.get("tags", []),
        notes=payload.get("notes"),
    )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))


def build_example_config(model: str, dataset: str, output_dir: str) -> ExperimentConfig:
    return ExperimentConfig(model=model, dataset=dataset, output_dir=output_dir)
