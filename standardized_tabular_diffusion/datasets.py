from __future__ import annotations

import re
from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.interfaces import DatasetSpec

_DATASET_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}


def validate_dataset_name(dataset_name: str) -> str:
    """Return a portable dataset identifier or reject an unsafe path component."""

    if not isinstance(dataset_name, str) or not _DATASET_NAME_PATTERN.fullmatch(dataset_name):
        raise ValueError(
            "Dataset names must contain 1-128 lowercase ASCII letters, digits, dots, "
            "underscores, or hyphens, and must begin and end with a letter or digit"
        )
    if dataset_name.split(".", maxsplit=1)[0] in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Dataset name is reserved on Windows: {dataset_name}")
    return dataset_name


def _normalize_task_type(task_type: str) -> str:
    mapping = {
        "binclass": "classification",
        "multiclass": "classification",
        "classification": "classification",
        "regression": "regression",
    }
    try:
        return mapping[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset task_type: {task_type!r}") from exc


def _resolve_upstream_data_path(repo_root: Path, relative_path: str | None, upstream_root: Path) -> Path | None:
    if not relative_path:
        return None
    path = Path(relative_path)
    if path.is_absolute():
        resolved_path = path.resolve()
        return resolved_path if resolved_path.is_relative_to(repo_root.resolve()) else None
    if ".." in path.parts:
        return None
    upstream_candidate = (upstream_root / path).resolve()
    if not upstream_candidate.is_relative_to(repo_root.resolve()):
        return None
    if upstream_candidate.exists():
        return upstream_candidate
    repository_candidate = (repo_root / path).resolve()
    if not repository_candidate.is_relative_to(repo_root.resolve()):
        return None
    if repository_candidate.exists():
        return repository_candidate
    if path.parts and path.parts[0] in {"TabDDPM-main", "TabDiff-main", "TabSyn-main", "materialized_datasets"}:
        return repository_candidate
    return upstream_candidate


def _resolve_repository_reference(repo_root: Path, value: str | None) -> Path | None:
    """Resolve new relative manifests and rebase known stale repository paths."""

    if value is None:
        return None
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        if ".." in path.parts:
            return None
        candidate = (repo_root / Path(*normalized.split("/"))).resolve()
        return candidate if candidate.is_relative_to(repo_root.resolve()) else None
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(repo_root.resolve()) and resolved_path.exists():
        return resolved_path

    for marker in ("TabDiff-main/", "TabSyn-main/", "TabDDPM-main/", "materialized_datasets/", "data/"):
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            rebased = repo_root / Path(*normalized[marker_index:].split("/"))
            if rebased.exists():
                return rebased
    return None


def _spec_from_info_json(repo_root: Path, info_path: Path) -> DatasetSpec:
    info = read_json(info_path)
    if not isinstance(info, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {info_path}")
    upstream_root = info_path.parents[2]
    index_groups: dict[str, list[int]] = {}
    for key in ("num_col_idx", "cat_col_idx", "target_col_idx"):
        value = info.get(key)
        if not isinstance(value, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in value
        ):
            raise ValueError(f"{key} must be a list of non-negative integer indices in {info_path}")
        index_groups[key] = value
    if not index_groups["target_col_idx"]:
        raise ValueError(f"target_col_idx must contain at least one target in {info_path}")

    all_indices = [index for group in index_groups.values() for index in group]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError(f"Numerical, categorical, and target indices must be disjoint in {info_path}")
    column_names = info.get("column_names")
    if column_names is None:
        max_idx = max(all_indices)
        column_names = [f"column_{idx}" for idx in range(max_idx + 1)]
    if (
        not isinstance(column_names, list)
        or any(not isinstance(column, str) or not column for column in column_names)
        or len(set(column_names)) != len(column_names)
    ):
        raise ValueError(f"column_names must be a list of unique non-empty strings in {info_path}")
    if set(all_indices) != set(range(len(column_names))):
        raise ValueError(f"Column index groups must cover every column exactly once in {info_path}")
    numerical_columns = [column_names[idx] for idx in index_groups["num_col_idx"]]
    categorical_columns = [column_names[idx] for idx in index_groups["cat_col_idx"]]
    target_columns = [column_names[idx] for idx in index_groups["target_col_idx"]]

    dataset_name = validate_dataset_name(info["name"])
    return DatasetSpec(
        name=dataset_name,
        task_type=_normalize_task_type(info["task_type"]),
        column_names=column_names,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        metadata_path=info_path,
        train_data_path=_resolve_upstream_data_path(repo_root, info.get("data_path"), upstream_root),
        val_data_path=_resolve_upstream_data_path(repo_root, info.get("val_path"), upstream_root),
        test_data_path=_resolve_upstream_data_path(repo_root, info.get("test_path"), upstream_root),
        provenance=[str(info_path)],
        extra={
            "header": info.get("header"),
            "file_type": info.get("file_type"),
            "column_info": info.get("column_info", {}),
            "train_num": info.get("train_num"),
            "test_num": info.get("test_num"),
            "dataset_view": info.get("dataset_view"),
            "source_version": info.get("source_version"),
            "excluded_raw_columns": info.get("excluded_raw_columns", {}),
            "preprocessing": info.get("preprocessing", {}),
        },
    )


def _merge_specs(preferred: DatasetSpec, other: DatasetSpec) -> DatasetSpec:
    merged = DatasetSpec(
        name=preferred.name,
        task_type=preferred.task_type or other.task_type,
        column_names=preferred.column_names or other.column_names,
        numerical_columns=preferred.numerical_columns or other.numerical_columns,
        categorical_columns=preferred.categorical_columns or other.categorical_columns,
        target_columns=preferred.target_columns or other.target_columns,
        metadata_path=preferred.metadata_path,
        train_data_path=preferred.train_data_path or other.train_data_path,
        val_data_path=preferred.val_data_path or other.val_data_path,
        test_data_path=preferred.test_data_path or other.test_data_path,
        provenance=[*preferred.provenance, *other.provenance],
        extra={**other.extra, **preferred.extra},
    )
    return merged


def discover_dataset_specs(repo_root: Path | None = None) -> dict[str, DatasetSpec]:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    dataset_specs: dict[str, DatasetSpec] = {}

    preferred_info_paths = sorted((repo_root / "TabDiff-main" / "data" / "Info").glob("*.json"))
    fallback_info_paths = sorted((repo_root / "TabSyn-main" / "data" / "Info").glob("*.json"))

    for info_path in preferred_info_paths:
        spec = _spec_from_info_json(repo_root, info_path)
        dataset_specs[spec.name] = spec

    for info_path in fallback_info_paths:
        spec = _spec_from_info_json(repo_root, info_path)
        if spec.name in dataset_specs:
            dataset_specs[spec.name] = _merge_specs(dataset_specs[spec.name], spec)
        else:
            dataset_specs[spec.name] = spec

    materialized_root = repo_root / "materialized_datasets"
    if materialized_root.exists():
        for manifest_path in sorted(materialized_root.glob("*/manifest.json")):
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError(f"Materialization manifest must be a JSON object: {manifest_path}")
            dataset_name = validate_dataset_name(manifest["dataset"])
            if dataset_name not in dataset_specs:
                continue
            spec = dataset_specs[dataset_name]
            metadata_path = _resolve_repository_reference(repo_root, manifest.get("metadata_path"))
            train_data_path = _resolve_repository_reference(repo_root, manifest.get("train_data_path"))
            val_data_path = _resolve_repository_reference(repo_root, manifest.get("val_data_path"))
            test_data_path = _resolve_repository_reference(repo_root, manifest.get("test_data_path"))
            dataset_specs[dataset_name] = DatasetSpec(
                name=spec.name,
                task_type=spec.task_type,
                column_names=spec.column_names,
                numerical_columns=spec.numerical_columns,
                categorical_columns=spec.categorical_columns,
                target_columns=spec.target_columns,
                metadata_path=metadata_path
                if metadata_path is not None and metadata_path.exists()
                else spec.metadata_path,
                train_data_path=(
                    train_data_path
                    if train_data_path is not None and train_data_path.exists()
                    else spec.train_data_path
                ),
                val_data_path=val_data_path
                if val_data_path is not None and val_data_path.exists()
                else spec.val_data_path,
                test_data_path=(
                    test_data_path if test_data_path is not None and test_data_path.exists() else spec.test_data_path
                ),
                provenance=[*spec.provenance, str(manifest_path)],
                extra={
                    **spec.extra,
                    "materialized_manifest_path": str(manifest_path),
                    "synthetic_real_path": _resolve_repository_reference(
                        repo_root, manifest.get("synthetic_real_path")
                    ),
                    "synthetic_val_path": _resolve_repository_reference(repo_root, manifest.get("synthetic_val_path")),
                    "synthetic_test_path": _resolve_repository_reference(
                        repo_root, manifest.get("synthetic_test_path")
                    ),
                },
            )

    return dict(sorted(dataset_specs.items()))


def get_dataset_spec(dataset_name: str, repo_root: Path | None = None) -> DatasetSpec:
    validate_dataset_name(dataset_name)
    specs = discover_dataset_specs(repo_root=repo_root)
    if dataset_name not in specs:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    return specs[dataset_name]
