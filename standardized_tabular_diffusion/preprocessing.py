"""Leakage-safe preprocessing for incomplete tabular dataset splits.

This module is deliberately separate from model adapters.  It fits every learned
statistic on the real training split, then applies the frozen state to validation
and test data without refitting.  It must not be used to repair generated samples.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_bytes,
    atomic_write_json,
    content_fingerprint,
    sha256_file,
)

PREPROCESSING_IMPLEMENTATION_VERSION = "1.0.0"
DEFAULT_MISSING_MARKERS = ("?", " ?", "", " ")


class PreprocessingError(ValueError):
    """Base class for deterministic preprocessing contract failures."""


class SplitSchemaError(PreprocessingError):
    """Raised when a split does not match the schema fitted on training data."""


class MissingTargetError(PreprocessingError):
    """Raised when a target value is missing; labels are never imputed."""


class UndefinedImputationStatisticError(PreprocessingError):
    """Raised when a training column has no observed value to learn from."""


@dataclass(frozen=True)
class MissingValuePolicy:
    """Versioned missing-value policy for real dataset features."""

    numerical_strategy: str = "mean"
    categorical_strategy: str = "most_frequent"
    missing_markers: tuple[str, ...] = DEFAULT_MISSING_MARKERS
    add_missing_indicators: bool = False
    target_strategy: str = "error"
    synthetic_strategy: str = "reject"

    def __post_init__(self) -> None:
        if self.numerical_strategy != "mean":
            raise PreprocessingError("The v1 numerical strategy must be 'mean'")
        if self.categorical_strategy != "most_frequent":
            raise PreprocessingError("The v1 categorical strategy must be 'most_frequent'")
        if self.target_strategy != "error":
            raise PreprocessingError("Target imputation is prohibited; target_strategy must be 'error'")
        if self.synthetic_strategy != "reject":
            raise PreprocessingError("Generated samples may not be repaired; synthetic_strategy must be 'reject'")
        if len(set(self.missing_markers)) != len(self.missing_markers):
            raise PreprocessingError("missing_markers contains duplicate values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImputationState:
    """Frozen state learned exclusively from one real training split."""

    state_schema_version: str
    implementation_version: str
    fitted_on_split: str
    feature_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    input_column_order: tuple[str, ...]
    output_column_order: tuple[str, ...]
    numerical_fill_values: dict[str, float]
    categorical_fill_values: dict[str, str]
    training_rows: int
    training_missing_counts: dict[str, int]
    policy: MissingValuePolicy

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy"] = self.policy.to_dict()
        return payload

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())


@dataclass(frozen=True)
class PreprocessedSplits:
    """In-memory result and its immutable learned state."""

    train: pd.DataFrame
    validation: pd.DataFrame | None
    test: pd.DataFrame | None
    state: ImputationState
    reports: dict[str, dict[str, Any]]


def _validate_declared_columns(
    frame: pd.DataFrame,
    *,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
) -> None:
    if frame.columns.has_duplicates:
        duplicates = sorted({str(column) for column in frame.columns[frame.columns.duplicated()]})
        raise SplitSchemaError(f"Duplicate column names are prohibited: {duplicates}")

    groups = {
        "numerical_columns": tuple(numerical_columns),
        "categorical_columns": tuple(categorical_columns),
        "target_columns": tuple(target_columns),
    }
    for name, columns in groups.items():
        if not columns and name == "target_columns":
            raise SplitSchemaError("At least one target column is required")
        if len(columns) != len(set(columns)):
            raise SplitSchemaError(f"{name} contains duplicate column names")

    numerical = set(numerical_columns)
    categorical = set(categorical_columns)
    targets = set(target_columns)
    overlaps = (numerical & categorical) | (numerical & targets) | (categorical & targets)
    if overlaps:
        raise SplitSchemaError(f"Column roles must be disjoint: {sorted(overlaps)}")

    declared = numerical | categorical | targets
    actual = {str(column) for column in frame.columns}
    if declared != actual:
        missing = sorted(declared - actual)
        undeclared = sorted(actual - declared)
        raise SplitSchemaError(
            f"Every column must have exactly one declared role; missing={missing}, undeclared={undeclared}"
        )


def _normalize_missing_markers(frame: pd.DataFrame, markers: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    marker_set = set(markers)
    for column in normalized.columns:
        series = normalized[column]
        marker_mask = series.map(lambda value: isinstance(value, str) and value in marker_set)
        if bool(marker_mask.any()):
            normalized.loc[marker_mask, column] = pd.NA
    return normalized


def _prepare_features(
    frame: pd.DataFrame,
    *,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    policy: MissingValuePolicy,
    split_name: str,
) -> pd.DataFrame:
    prepared = _normalize_missing_markers(frame, policy.missing_markers)

    missing_targets = {column: int(prepared[column].isna().sum()) for column in target_columns}
    if any(missing_targets.values()):
        raise MissingTargetError(
            f"Split {split_name!r} contains missing targets and labels are never imputed: {missing_targets}"
        )

    for column in numerical_columns:
        original = prepared[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_mask = original.notna() & converted.isna()
        if bool(invalid_mask.any()):
            examples = [str(value) for value in original.loc[invalid_mask].head(5).tolist()]
            raise PreprocessingError(
                f"Numerical column {column!r} in split {split_name!r} contains non-numeric values: {examples}"
            )
        finite_mask = converted.notna() & ~converted.map(math.isfinite)
        if bool(finite_mask.any()):
            raise PreprocessingError(f"Numerical column {column!r} in split {split_name!r} contains non-finite values")
        prepared[column] = converted.astype("float64")

    for column in categorical_columns:
        prepared[column] = prepared[column].astype("string")
    return prepared


def _most_frequent_value(series: pd.Series, column: str) -> str:
    observed = series.dropna().astype("string")
    if observed.empty:
        raise UndefinedImputationStatisticError(
            f"Categorical training column {column!r} is entirely missing; a mode cannot be learned"
        )
    counts = observed.value_counts(sort=False)
    maximum = int(counts.max())
    candidates = [str(value) for value, count in counts.items() if int(count) == maximum]
    return min(candidates, key=lambda value: unicodedata.normalize("NFC", value))


def fit_imputation_state(
    train: pd.DataFrame,
    *,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    policy: MissingValuePolicy | None = None,
) -> ImputationState:
    """Fit means and modes on the real training split only."""

    policy = policy or MissingValuePolicy()
    numerical_columns = tuple(numerical_columns)
    categorical_columns = tuple(categorical_columns)
    target_columns = tuple(target_columns)
    _validate_declared_columns(
        train,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
    )
    prepared = _prepare_features(
        train,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        policy=policy,
        split_name="train",
    )

    numerical_fill_values: dict[str, float] = {}
    for column in numerical_columns:
        observed = [float(value) for value in prepared[column].dropna().tolist()]
        if not observed:
            raise UndefinedImputationStatisticError(
                f"Numerical training column {column!r} is entirely missing; a mean cannot be learned"
            )
        numerical_fill_values[column] = math.fsum(observed) / len(observed)

    categorical_fill_values = {column: _most_frequent_value(prepared[column], column) for column in categorical_columns}
    feature_columns = (*numerical_columns, *categorical_columns)
    output_column_order = list(map(str, train.columns))
    if policy.add_missing_indicators:
        indicator_columns = [f"{column}__missing" for column in feature_columns]
        collisions = sorted(set(output_column_order) & set(indicator_columns))
        if collisions:
            raise SplitSchemaError(f"Missing-indicator names collide with input columns: {collisions}")
        output_column_order.extend(indicator_columns)

    return ImputationState(
        state_schema_version="1.0.0",
        implementation_version=PREPROCESSING_IMPLEMENTATION_VERSION,
        fitted_on_split="train",
        feature_columns=feature_columns,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        input_column_order=tuple(map(str, train.columns)),
        output_column_order=tuple(output_column_order),
        numerical_fill_values=numerical_fill_values,
        categorical_fill_values=categorical_fill_values,
        training_rows=int(len(train)),
        training_missing_counts={column: int(prepared[column].isna().sum()) for column in prepared.columns},
        policy=policy,
    )


def transform_with_imputation_state(
    frame: pd.DataFrame,
    state: ImputationState,
    *,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply frozen training state to one real split without refitting."""

    if tuple(map(str, frame.columns)) != state.input_column_order:
        raise SplitSchemaError(
            f"Split {split_name!r} column order differs from training: "
            f"expected={list(state.input_column_order)}, observed={list(map(str, frame.columns))}"
        )
    prepared = _prepare_features(
        frame,
        numerical_columns=state.numerical_columns,
        categorical_columns=state.categorical_columns,
        target_columns=state.target_columns,
        policy=state.policy,
        split_name=split_name,
    )
    missing_counts = {column: int(prepared[column].isna().sum()) for column in prepared.columns}

    if state.policy.add_missing_indicators:
        for column in state.feature_columns:
            prepared[f"{column}__missing"] = prepared[column].isna().astype("int8")

    for column, value in state.numerical_fill_values.items():
        prepared[column] = prepared[column].fillna(value).astype("float64")
    for column, value in state.categorical_fill_values.items():
        prepared[column] = prepared[column].fillna(value).astype("string")

    prepared = prepared.loc[:, list(state.output_column_order)]
    remaining = {column: int(prepared[column].isna().sum()) for column in prepared.columns}
    if any(remaining.values()):
        raise PreprocessingError(f"Split {split_name!r} still contains missing values after preprocessing: {remaining}")

    values_imputed = sum(missing_counts[column] for column in state.feature_columns)
    report = {
        "split": split_name,
        "rows": int(len(prepared)),
        "missing_values_by_column_before": missing_counts,
        "values_imputed": int(values_imputed),
        "missing_values_after": 0,
        "state_fingerprint": state.fingerprint,
    }
    return prepared, report


def preprocess_splits(
    train: pd.DataFrame,
    *,
    validation: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    policy: MissingValuePolicy | None = None,
) -> PreprocessedSplits:
    """Fit on train and transform all supplied real-data splits."""

    state = fit_imputation_state(
        train,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        policy=policy,
    )
    transformed: dict[str, pd.DataFrame | None] = {"validation": None, "test": None}
    reports: dict[str, dict[str, Any]] = {}
    transformed_train, reports["train"] = transform_with_imputation_state(train, state, split_name="train")
    for name, frame in (("validation", validation), ("test", test)):
        if frame is not None:
            transformed[name], reports[name] = transform_with_imputation_state(frame, state, split_name=name)
    return PreprocessedSplits(
        train=transformed_train,
        validation=transformed["validation"],
        test=transformed["test"],
        state=state,
        reports=reports,
    )


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_frame_bytes(frame)).hexdigest()


def preprocess_split_files(
    *,
    train_path: str | Path,
    output_dir: str | Path,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    validation_path: str | Path | None = None,
    test_path: str | Path | None = None,
    policy: MissingValuePolicy | None = None,
) -> dict[str, Any]:
    """Preprocess CSV splits and emit portable data, learned state, and audit manifest."""

    paths: dict[str, Path] = {"train": Path(train_path)}
    if validation_path is not None:
        paths["validation"] = Path(validation_path)
    if test_path is not None:
        paths["test"] = Path(test_path)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} CSV does not exist: {path}")

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    result = preprocess_splits(
        frames["train"],
        validation=frames.get("validation"),
        test=frames.get("test"),
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        policy=policy,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "imputation-state.json"
    atomic_write_json(state_path, result.state.to_dict())

    output_frames: Mapping[str, pd.DataFrame | None] = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    output_records: dict[str, dict[str, Any]] = {}
    for name, frame in output_frames.items():
        if frame is None:
            continue
        filename = "val.csv" if name == "validation" else f"{name}.csv"
        output_path = destination / filename
        payload = _frame_bytes(frame)
        atomic_write_bytes(output_path, payload)
        output_records[name] = {
            "path": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": int(len(frame)),
        }

    manifest = {
        "manifest_schema_version": "1.0.0",
        "implementation_version": PREPROCESSING_IMPLEMENTATION_VERSION,
        "fitted_on_split": "train",
        "train_only_fitting": True,
        "state": {
            "path": state_path.name,
            "sha256": sha256_file(state_path),
            "fingerprint": result.state.fingerprint,
        },
        "policy": result.state.policy.to_dict(),
        "inputs": {
            name: {
                "filename": path.name,
                "sha256": sha256_file(path),
                "rows": int(len(frames[name])),
                "canonical_frame_sha256": _frame_sha256(frames[name]),
            }
            for name, path in paths.items()
        },
        "outputs": output_records,
        "reports": result.reports,
    }
    manifest_path = destination / "preprocessing-manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}
