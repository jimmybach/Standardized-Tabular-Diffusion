from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.datasets import validate_dataset_name
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.interfaces import RunSpec


def _validate_extra(section: str, extra: dict[str, Any]) -> None:
    if not isinstance(extra, dict) or any(not isinstance(key, str) for key in extra):
        raise TypeError(f"{section}.extra must be a mapping with string keys")


@dataclass
class TrainConfig:
    enabled: bool = True
    seed: int = 0
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("train.seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("train.device must be a non-empty string")
        _validate_extra("train", self.extra)


@dataclass
class SampleConfig:
    enabled: bool = True
    num_samples: int | None = None
    checkpoint_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_samples is not None and (
            not isinstance(self.num_samples, int) or isinstance(self.num_samples, bool) or self.num_samples <= 0
        ):
            raise ValueError("sample.num_samples must be a positive integer when provided")
        _validate_extra("sample", self.extra)


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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.total_time_limit_seconds, int)
            or isinstance(self.total_time_limit_seconds, bool)
            or self.total_time_limit_seconds <= 0
        ):
            raise ValueError("evaluation.total_time_limit_seconds must be a positive integer")
        _validate_extra("evaluation", self.extra)


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

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        validate_dataset_name(self.dataset)
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")
        if not isinstance(self.tags, list) or any(not isinstance(tag, str) for tag in self.tags):
            raise TypeError("tags must be a list of strings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_run_spec(self, action: str | None = None) -> RunSpec:
        if action not in {None, "train", "sample", "evaluate"}:
            raise ValueError(f"Unsupported action for RunSpec: {action}")
        action_extras = {
            "train": dict(self.train.extra),
            "sample": dict(self.sample.extra),
            "evaluation": dict(self.evaluation.extra),
        }
        extra = {
            "evaluation": asdict(self.evaluation),
            "tags": list(self.tags),
            "action_extras": action_extras,
        }
        if action is not None:
            selected_extras = action_extras["evaluation" if action == "evaluate" else action]
            reserved_collisions = sorted(set(selected_extras) & set(extra))
            if reserved_collisions:
                raise ValueError(f"Action extras use reserved RunSpec keys: {reserved_collisions}")
            extra.update(selected_extras)
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
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("Experiment configuration must be a JSON object")
    allowed_keys = {
        "model",
        "dataset",
        "output_dir",
        "train",
        "sample",
        "evaluation",
        "upstream_config_path",
        "tags",
        "notes",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown experiment configuration keys: {unknown_keys}")
    for section in ("train", "sample", "evaluation"):
        if section in payload and not isinstance(payload[section], dict):
            raise TypeError(f"{section} must be a JSON object")
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
    atomic_write_json(path, config.to_dict())


def build_example_config(model: str, dataset: str, output_dir: str) -> ExperimentConfig:
    return ExperimentConfig(model=model, dataset=dataset, output_dir=output_dir)
