"""Canonical table loading and protocol-specific structural validation gates."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, NoReturn

import pandas as pd


class TableValidationError(ValueError):
    """Raised before metric execution when a table violates its Dataset Profile."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class ValidatedTables:
    """Canonical, reordered inputs plus a serializable validation report."""

    reference: pd.DataFrame
    synthetic: pd.DataFrame
    metadata: dict[str, Any]
    column_ids: dict[str, str]
    report: dict[str, Any]


def _fail(reason_code: str, detail: str) -> NoReturn:
    raise TableValidationError(reason_code, detail)


def _validate_csv_header(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
    except StopIteration:
        _fail("empty_input_file", f"CSV input is empty: {path}")
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail("input_read_failure", f"Cannot read CSV header for {path}: {exc}")
    if len(header) != len(set(header)):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        _fail("duplicate_columns", f"CSV contains duplicate column names: {duplicates}")


def load_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV/Parquet file or defensively copy a DataFrame.

    The suffix is authoritative for file inputs. CSV headers are checked before
    pandas can mangle duplicate names.
    """

    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    path = Path(source)
    if not path.is_file():
        _fail("input_not_found", f"Table input is not a regular file: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            _validate_csv_header(path)
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except TableValidationError:
        raise
    except Exception as exc:
        _fail("input_read_failure", f"Cannot read table {path}: {type(exc).__name__}: {exc}")
    _fail("unsupported_input_format", f"Expected .csv, .parquet, or .pq input, got: {path}")


def _canonical_boolean(series: pd.Series, column: str, *, preserve_content_violations: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if preserve_content_violations:
            return series.astype("boolean")
        return series.astype(bool)
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }
    converted = series.map(mapping)
    invalid = series.notna() & converted.isna()
    if invalid.any():
        bad = series[invalid].iloc[0]
        _fail("type_mismatch", f"Column {column!r} contains a non-boolean value: {bad!r}")
    if preserve_content_violations and series.isna().any():
        return converted.astype("boolean")
    return converted.astype(bool)


def _content_preserving_numeric(series: pd.Series, table_name: str, column: str) -> pd.Series:
    """Parse scalar numerics without forcing nullable integers through float64."""

    parsed: list[Any] = []
    for value in series.tolist():
        if pd.isna(value):
            parsed.append(pd.NA)
            continue
        if isinstance(value, bool):
            _fail("type_mismatch", f"{table_name} column {column!r} contains a Boolean in a numerical field")
        scalar = value.item() if hasattr(value, "item") else value
        if isinstance(scalar, int):
            parsed.append(scalar)
            continue
        if isinstance(scalar, float):
            parsed.append(scalar)
            continue
        try:
            decimal = Decimal(str(scalar).strip())
        except (InvalidOperation, ValueError) as exc:
            _fail("type_mismatch", f"{table_name} column {column!r} contains a non-numeric value: {scalar!r}: {exc}")
        if decimal.is_finite() and decimal == decimal.to_integral_value():
            parsed.append(int(decimal))
        else:
            parsed.append(float(decimal))
    return pd.Series(parsed, index=series.index, dtype="object", name=series.name)


def _canonicalize_column(
    series: pd.Series,
    spec: dict[str, Any],
    table_name: str,
    *,
    preserve_content_violations: bool,
) -> pd.Series:
    name = spec["name"]
    semantic_type = spec["semantic_type"]
    if not preserve_content_violations and series.isna().any():
        _fail("missing_values_prohibited", f"{table_name} column {name!r} contains missing values")

    if semantic_type in {"continuous", "integer"}:
        if preserve_content_violations:
            return _content_preserving_numeric(series, table_name, name)
        try:
            converted = pd.to_numeric(series, errors="raise")
        except (TypeError, ValueError, OverflowError) as exc:
            _fail("type_mismatch", f"{table_name} column {name!r} is not losslessly numerical: {exc}")
        try:
            finite = converted.dropna().map(lambda value: math.isfinite(float(value))).all()
        except (TypeError, ValueError, OverflowError) as exc:
            _fail("type_mismatch", f"{table_name} column {name!r} is not losslessly numerical: {exc}")
        if not preserve_content_violations and not finite:
            _fail("nonfinite_values", f"{table_name} column {name!r} contains NaN or infinity")
        if semantic_type == "integer":
            non_missing = converted.dropna()
            integral = non_missing.map(lambda value: math.isfinite(float(value)) and int(value) == value)
            if not preserve_content_violations and not integral.all():
                _fail("type_mismatch", f"{table_name} column {name!r} contains a non-integer value")
            in_int64 = non_missing.map(
                lambda value: math.isfinite(float(value)) and -(2**63) <= int(value) <= 2**63 - 1
            )
            if not preserve_content_violations and not in_int64.all():
                _fail("type_mismatch", f"{table_name} column {name!r} exceeds signed int64")
            if preserve_content_violations and (series.isna().any() or not integral.all() or not in_int64.all()):
                return converted.astype("float64")
            try:
                # Preserve the logical integer representation. Converting all
                # integers through float64 would silently round values above
                # 2**53 before scientific evaluation.
                return converted.astype("int64")
            except (TypeError, ValueError, OverflowError) as exc:
                _fail("type_mismatch", f"{table_name} column {name!r} exceeds signed int64: {exc}")
        try:
            return converted.astype("float64")
        except (TypeError, ValueError, OverflowError) as exc:
            _fail("type_mismatch", f"{table_name} column {name!r} cannot be represented as float64: {exc}")

    if semantic_type == "boolean":
        return _canonical_boolean(series, name, preserve_content_violations=preserve_content_violations)

    if semantic_type == "datetime":
        try:
            converted = pd.to_datetime(series, errors="raise")
        except (TypeError, ValueError, OverflowError) as exc:
            _fail("type_mismatch", f"{table_name} column {name!r} is not a valid datetime: {exc}")
        if not preserve_content_violations and converted.isna().any():
            _fail("missing_values_prohibited", f"{table_name} column {name!r} contains invalid datetimes")
        return converted

    if semantic_type in {"categorical", "string"}:
        try:
            return series.astype("string")
        except (TypeError, ValueError) as exc:
            _fail("type_mismatch", f"{table_name} column {name!r} cannot be represented as strings: {exc}")

    _fail("unsupported_semantic_type", f"Unsupported semantic type {semantic_type!r} for {name!r}")


def _validate_columns(frame: pd.DataFrame, expected: list[str], table_name: str) -> None:
    if not all(isinstance(name, str) and name for name in frame.columns):
        _fail("invalid_column_names", f"{table_name} column names must be non-empty strings")
    names = list(frame.columns)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        _fail("duplicate_columns", f"{table_name} contains duplicate column names: {duplicates}")
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    if missing or extra:
        _fail(
            "schema_mismatch", f"{table_name} columns differ from the Dataset Profile; missing={missing}, extra={extra}"
        )


def _sdtype(semantic_type: str) -> str:
    return {
        "continuous": "numerical",
        "integer": "numerical",
        "categorical": "categorical",
        "boolean": "boolean",
        "datetime": "datetime",
        "string": "id",
    }[semantic_type]


def model_view_column_specs(dataset_profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the ordered model-view column definitions from a Dataset Profile.

    Profiles may retain raw audit-only or explicitly ignored fields outside the
    canonical model view. Those fields are never requested from a decoded model
    output and therefore never become metric denominators.
    """

    columns = dataset_profile.get("columns")
    contract = dataset_profile.get("table_contract")
    if not isinstance(columns, list) or not columns or not isinstance(contract, dict):
        _fail("invalid_dataset_profile", "Dataset Profile lacks a usable table contract or columns")
    expected = contract.get("canonical_column_order")
    if not isinstance(expected, list) or not expected or len(expected) != len(set(expected)):
        _fail("invalid_dataset_profile", "Dataset Profile canonical order must be non-empty and unique")
    by_name = {column.get("name"): column for column in columns if isinstance(column, dict)}
    if len(by_name) != len(columns) or any(name not in by_name for name in expected):
        _fail("invalid_dataset_profile", "Dataset Profile canonical order references missing or duplicate columns")
    excluded = [column for name, column in by_name.items() if name not in expected]
    if any(not set(column.get("roles", ())).intersection({"ignored", "audit_only"}) for column in excluded):
        _fail(
            "invalid_dataset_profile",
            "Columns outside the canonical model view must be explicitly ignored or audit_only",
        )
    return [by_name[name] for name in expected]


def validate_tables(
    reference: str | Path | pd.DataFrame,
    synthetic: str | Path | pd.DataFrame,
    dataset_profile: dict[str, Any],
    *,
    expected_synthetic_rows: int | None = None,
    content_mode: Literal["strict", "preserve"] = "strict",
) -> ValidatedTables:
    """Resolve both inputs to the exact Dataset Profile table contract.

    Structural validation is intentionally narrower than P3 validity scoring:
    it checks shape, order, lossless logical typing, non-missing inputs, and the
    requested row count, but it does not score learned domains or constraints.
    """

    if content_mode not in {"strict", "preserve"}:
        _fail("invalid_validation_mode", f"Unsupported content validation mode: {content_mode!r}")
    preserve_content_violations = content_mode == "preserve"
    columns = model_view_column_specs(dataset_profile)
    expected = [column["name"] for column in columns]

    real = load_table(reference)
    synth = load_table(synthetic)
    _validate_columns(real, expected, "reference")
    _validate_columns(synth, expected, "synthetic")
    if len(real) == 0 or len(synth) == 0:
        _fail("empty_table", "Reference and synthetic tables must each contain at least one row")
    requested_rows = len(real) if expected_synthetic_rows is None else expected_synthetic_rows
    if isinstance(requested_rows, bool) or not isinstance(requested_rows, int) or requested_rows <= 0:
        _fail("invalid_row_count", "expected_synthetic_rows must be a positive integer")
    if len(synth) != requested_rows:
        _fail("row_count_mismatch", f"Synthetic row count is {len(synth)}, expected {requested_rows}")

    real = real.loc[:, expected].copy()
    synth = synth.loc[:, expected].copy()
    for spec in columns:
        real[spec["name"]] = _canonicalize_column(
            real[spec["name"]],
            spec,
            "reference",
            preserve_content_violations=preserve_content_violations,
        )
        synth[spec["name"]] = _canonicalize_column(
            synth[spec["name"]],
            spec,
            "synthetic",
            preserve_content_violations=preserve_content_violations,
        )

    metadata_columns: dict[str, dict[str, Any]] = {}
    for spec in columns:
        meta: dict[str, Any] = {"sdtype": _sdtype(spec["semantic_type"])}
        metadata_columns[spec["name"]] = meta
    metadata = {"columns": metadata_columns}
    report = {
        "validation_schema_version": "1.0.0",
        "status": "passed",
        "dataset_id": dataset_profile["dataset_id"],
        "dataset_profile_version": dataset_profile["dataset_profile_version"],
        "canonical_column_order": expected,
        "column_count": len(expected),
        "reference_rows": len(real),
        "synthetic_rows": len(synth),
        "expected_synthetic_rows": requested_rows,
        "missing_values": {
            "reference": int(real.isna().sum().sum()),
            "synthetic": int(synth.isna().sum().sum()),
        },
        "content_mode": content_mode,
        "checks": [
            "unique_columns",
            "exact_column_set",
            "canonical_column_order",
            "logical_types",
            *(("nonfinite_values", "missing_values") if content_mode == "strict" else ()),
            "synthetic_row_count",
        ],
    }
    return ValidatedTables(
        reference=real,
        synthetic=synth,
        metadata=metadata,
        column_ids={spec["name"]: spec["column_id"] for spec in columns},
        report=report,
    )
