from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetSpec:
    name: str
    task_type: str
    column_names: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    target_columns: list[str]
    metadata_path: Path
    train_data_path: Path | None = None
    val_data_path: Path | None = None
    test_data_path: Path | None = None
    provenance: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("metadata_path", "train_data_path", "val_data_path", "test_data_path"):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        return payload


@dataclass
class RunSpec:
    model: str
    dataset: str
    output_dir: Path
    device: str = "cpu"
    seed: int = 0
    num_samples: int | None = None
    checkpoint_path: Path | None = None
    upstream_config_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("output_dir", "checkpoint_path", "upstream_config_path"):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        return payload


@dataclass
class ArtifactBundle:
    model: str
    dataset: str
    output_dir: Path
    upstream_workdir: Path
    generated_sample_path: Path | None = None
    upstream_metrics_path: Path | None = None
    standardized_summary_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "output_dir",
            "upstream_workdir",
            "generated_sample_path",
            "upstream_metrics_path",
            "standardized_summary_path",
        ):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        return payload
