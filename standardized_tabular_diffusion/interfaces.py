from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Any


def _serialize_paths(value: Any) -> Any:
    """Recursively convert path values at a public JSON boundary."""

    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_paths(item) for item in value]
    return value


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
        return _serialize_paths(asdict(self))


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
        return _serialize_paths(asdict(self))


@dataclass
class ArtifactBundle:
    model: str
    dataset: str
    output_dir: Path
    upstream_workdir: Path
    generated_sample_path: Path | None = None
    upstream_metrics_path: Path | None = None
    evaluation_bundle_path: Path | None = None
    standardized_summary_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_paths(asdict(self))
