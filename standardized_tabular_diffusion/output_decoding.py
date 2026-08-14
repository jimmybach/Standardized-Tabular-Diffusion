from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.interfaces import DatasetSpec


def declared_integer_columns(dataset_spec: DatasetSpec, *, model_name: str) -> list[str]:
    """Resolve explicit integer roles without inferring them from generated values."""

    declared: list[str] = []
    explicit = dataset_spec.extra.get("integer_columns")
    if explicit is not None:
        if not isinstance(explicit, list) or any(not isinstance(column, str) for column in explicit):
            raise TypeError(f"{model_name} DatasetSpec integer_columns must be a list of column names.")
        declared.extend(explicit)
    column_info = dataset_spec.extra.get("column_info", {})
    if isinstance(column_info, dict):
        for column, description in column_info.items():
            if isinstance(description, str) and description.lower() in {"int", "integer"}:
                declared.append(column)
            elif isinstance(description, dict) and (
                description.get("semantic_type") == "integer"
                or description.get("type") == "integer"
                or description.get("integer") is True
            ):
                declared.append(column)
    metadata = read_json(dataset_spec.metadata_path)
    if isinstance(metadata, dict) and metadata.get("int_columns") is not None:
        metadata_columns = metadata["int_columns"]
        if not isinstance(metadata_columns, list) or any(
            not isinstance(column, str) for column in metadata_columns
        ):
            raise TypeError(f"{model_name} dataset metadata int_columns must be a list of column names.")
        declared.extend(metadata_columns)
    declared = list(dict.fromkeys(declared))
    invalid = [column for column in declared if column not in dataset_spec.column_names]
    numerical_roles = list(dataset_spec.numerical_columns)
    if dataset_spec.task_type == "regression":
        numerical_roles.extend(dataset_spec.target_columns)
    non_numerical = [column for column in declared if column not in numerical_roles]
    if invalid or non_numerical:
        raise ValueError(
            f"{model_name} integer declarations must name canonical numerical columns: "
            f"unknown={invalid}, non_numerical={non_numerical}."
        )
    return declared


def decode_declared_integer_columns(
    frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    *,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Apply an explicit nearest-integer adapter decoding contract without clipping."""

    decoded = frame.copy()
    report: dict[str, dict[str, Any]] = {}
    int64 = np.iinfo(np.int64)
    for column in declared_integer_columns(dataset_spec, model_name=model_name):
        numeric = pd.to_numeric(decoded[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{model_name} integer column {column!r} contains non-finite native samples.")
        rounded = np.rint(numeric)
        if bool((rounded < int64.min).any() or (rounded > int64.max).any()):
            raise OverflowError(f"{model_name} integer column {column!r} exceeds the signed 64-bit range.")
        changed_rows = int(np.count_nonzero(numeric != rounded))
        decoded[column] = rounded.astype(np.int64)
        report[column] = {
            "policy": "numpy-rint-ties-to-even-at-adapter-decoding-boundary",
            "changed_rows": changed_rows,
            "clipped_rows": 0,
        }
    return decoded, report
