from __future__ import annotations

import json
from pathlib import Path

from standardized_tabular_diffusion.interfaces import DatasetSpec


def _normalize_task_type(task_type: str) -> str:
    mapping = {
        "binclass": "classification",
        "multiclass": "classification",
        "classification": "classification",
        "regression": "regression",
    }
    return mapping.get(task_type, task_type)


def _resolve_upstream_data_path(repo_root: Path, relative_path: str | None, upstream_root: Path) -> Path | None:
    if not relative_path:
        return None
    path = Path(relative_path)
    if path.is_absolute():
        return path
    candidate = upstream_root / path
    if candidate.exists():
        return candidate
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return candidate


def _spec_from_info_json(repo_root: Path, info_path: Path) -> DatasetSpec:
    info = json.loads(info_path.read_text())
    upstream_root = info_path.parents[2]
    column_names = info.get("column_names")
    if column_names is None:
        max_idx = max(info["num_col_idx"] + info["cat_col_idx"] + info["target_col_idx"])
        column_names = [f"column_{idx}" for idx in range(max_idx + 1)]
    numerical_columns = [column_names[idx] for idx in info["num_col_idx"]]
    categorical_columns = [column_names[idx] for idx in info["cat_col_idx"]]
    target_columns = [column_names[idx] for idx in info["target_col_idx"]]

    return DatasetSpec(
        name=info["name"],
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
            manifest = json.loads(manifest_path.read_text())
            dataset_name = manifest["dataset"]
            if dataset_name not in dataset_specs:
                continue
            spec = dataset_specs[dataset_name]
            dataset_specs[dataset_name] = DatasetSpec(
                name=spec.name,
                task_type=spec.task_type,
                column_names=spec.column_names,
                numerical_columns=spec.numerical_columns,
                categorical_columns=spec.categorical_columns,
                target_columns=spec.target_columns,
                metadata_path=Path(manifest["metadata_path"]),
                train_data_path=None if manifest.get("train_data_path") is None else Path(manifest["train_data_path"]),
                val_data_path=None if manifest.get("val_data_path") is None else Path(manifest["val_data_path"]),
                test_data_path=None if manifest.get("test_data_path") is None else Path(manifest["test_data_path"]),
                provenance=[*spec.provenance, str(manifest_path)],
                extra={
                    **spec.extra,
                    "materialized_manifest_path": str(manifest_path),
                    "synthetic_real_path": manifest.get("synthetic_real_path"),
                    "synthetic_val_path": manifest.get("synthetic_val_path"),
                    "synthetic_test_path": manifest.get("synthetic_test_path"),
                },
            )

    return dict(sorted(dataset_specs.items()))


def get_dataset_spec(dataset_name: str, repo_root: Path | None = None) -> DatasetSpec:
    specs = discover_dataset_specs(repo_root=repo_root)
    if dataset_name not in specs:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    return specs[dataset_name]
