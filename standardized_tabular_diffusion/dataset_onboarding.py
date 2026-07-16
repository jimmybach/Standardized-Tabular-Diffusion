from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import CategoricalDtype

from standardized_tabular_diffusion.materialization import _build_manifest, _sync_processed_dataset, manifest_path


def upload_root(repo_root: Path | None = None) -> Path:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    return repo_root / "data" / "uploads"


def _normalize_task_type_for_info(task_type: str) -> str:
    normalized = task_type.strip().lower()
    mapping = {
        "classification": "binclass",
        "binclass": "binclass",
        "multiclass": "multiclass",
        "regression": "regression",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return mapping[normalized]


def _read_header_arg(has_header: bool) -> int | None:
    return 0 if has_header else None


def _infer_column_partitions(
    frame: pd.DataFrame,
    *,
    target_column: str,
    numerical_columns: list[str] | None,
    categorical_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
    provided_numerical = list(numerical_columns or [])
    provided_categorical = list(categorical_columns or [])

    if provided_numerical or provided_categorical:
        overlap = set(provided_numerical) & set(provided_categorical)
        if overlap:
            raise ValueError(f"Columns cannot be both numerical and categorical: {sorted(overlap)}")
        for column in [*provided_numerical, *provided_categorical, target_column]:
            if column not in frame.columns:
                raise ValueError(f"Unknown column: {column}")
        uncovered = [column for column in frame.columns if column != target_column and column not in set(provided_numerical + provided_categorical)]
        if uncovered:
            raise ValueError(
                "Every non-target column must be assigned when explicit column groups are provided. "
                f"Missing: {uncovered}"
            )
        return provided_numerical, provided_categorical

    inferred_numerical: list[str] = []
    inferred_categorical: list[str] = []
    for column in frame.columns:
        if column == target_column:
            continue
        series = frame[column]
        if isinstance(series.dtype, CategoricalDtype):
            inferred_categorical.append(column)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.replace({"?": np.nan, " ?": np.nan, "": np.nan, " ": np.nan})
            coerced = pd.to_numeric(normalized, errors="coerce")
            if normalized.notna().sum() > 0 and coerced.notna().sum() == normalized.notna().sum():
                inferred_numerical.append(column)
                continue
        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            inferred_categorical.append(column)
        else:
            inferred_numerical.append(column)
    return inferred_numerical, inferred_categorical


def _build_column_info(
    frame: pd.DataFrame,
    *,
    numerical_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    task_type: str,
) -> dict[str, str]:
    info: dict[str, str] = {}
    for column in numerical_columns:
        info[column] = "float"
    for column in categorical_columns:
        info[column] = "str"
    info[target_column] = "float" if task_type == "regression" else "str"
    return info


def _sanitize_local_frame(
    frame: pd.DataFrame,
    *,
    numerical_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = frame.copy()
    missing_markers = {"?": np.nan, " ?": np.nan, "": np.nan, " ": np.nan}
    categorical_missing_token = "__missing__"
    cleaned = cleaned.replace(missing_markers)

    for column in numerical_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    dropped_target_rows = int(cleaned[target_column].isna().sum())
    if dropped_target_rows:
        cleaned = cleaned.loc[~cleaned[target_column].isna()].copy()

    dropped_numeric_rows = int(cleaned[numerical_columns].isna().any(axis=1).sum()) if numerical_columns else 0
    if dropped_numeric_rows:
        cleaned = cleaned.loc[~cleaned[numerical_columns].isna().any(axis=1)].copy()

    for column in categorical_columns:
        cleaned[column] = cleaned[column].fillna(categorical_missing_token).astype(str)

    report = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(cleaned)),
        "dropped_missing_target_rows": dropped_target_rows,
        "dropped_missing_numerical_rows": dropped_numeric_rows,
    }
    return cleaned, report


def _run_python(args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def register_dataset(
    *,
    dataset_name: str,
    raw_csv_path: str | Path,
    task_type: str,
    target_column: str,
    numerical_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    has_header: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    source_path = Path(raw_csv_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {source_path}")

    header = _read_header_arg(has_header)
    frame = pd.read_csv(source_path, header=header)
    if not has_header:
        frame.columns = [f"column_{idx}" for idx in range(len(frame.columns))]

    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    numerical_columns, categorical_columns = _infer_column_partitions(
        frame,
        target_column=target_column,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    column_names = list(frame.columns)
    target_idx = column_names.index(target_column)
    numerical_indices = [column_names.index(column) for column in numerical_columns]
    categorical_indices = [column_names.index(column) for column in categorical_columns]

    tabdiff_data_dir = repo_root / "TabDiff-main" / "data" / dataset_name
    tabdiff_info_dir = repo_root / "TabDiff-main" / "data" / "Info"
    uploaded_dir = upload_root(repo_root=repo_root) / dataset_name

    tabdiff_data_dir.mkdir(parents=True, exist_ok=True)
    tabdiff_info_dir.mkdir(parents=True, exist_ok=True)
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    upload_copy_path = uploaded_dir / source_path.name
    shutil.copy2(source_path, upload_copy_path)
    raw_copy_path = tabdiff_data_dir / "raw.csv"
    cleaned_frame, cleaning_report = _sanitize_local_frame(
        frame,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
    )
    cleaned_frame.to_csv(raw_copy_path, index=False)

    normalized_task_type = _normalize_task_type_for_info(task_type)
    info_payload = {
        "name": dataset_name,
        "task_type": normalized_task_type,
        "header": header,
        "column_names": column_names,
        "num_col_idx": numerical_indices,
        "cat_col_idx": categorical_indices,
        "target_col_idx": [target_idx],
        "file_type": "csv",
        "data_path": f"data/{dataset_name}/raw.csv",
        "val_path": None,
        "test_path": None,
        "column_info": _build_column_info(
            frame,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target_column=target_column,
            task_type=task_type.strip().lower(),
        ),
    }

    info_path = tabdiff_info_dir / f"{dataset_name}.json"
    info_path.write_text(json.dumps(info_payload, indent=2))

    return {
        "dataset": dataset_name,
        "info_path": str(info_path),
        "raw_data_path": str(raw_copy_path),
        "upload_copy_path": str(upload_copy_path),
        "task_type": normalized_task_type,
        "column_names": column_names,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_column": target_column,
        "cleaning_report": cleaning_report,
    }


def process_registered_dataset(
    dataset_name: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    info_path = repo_root / "TabDiff-main" / "data" / "Info" / f"{dataset_name}.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_name} is not registered. Expected metadata at {info_path}"
        )

    upstream_root = repo_root / "TabDiff-main"
    _run_python(["process_dataset.py", "--dataname", dataset_name], upstream_root)

    manifest = _build_manifest(dataset_name, repo_root)
    manifest["materialized_by"] = "local-registration"
    manifest["synced_roots"] = _sync_processed_dataset(dataset_name, repo_root)
    out_path = manifest_path(dataset_name, repo_root=repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest
