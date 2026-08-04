"""Reproducible builders for reviewed public dataset sources."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.dataset_sources import DatasetSource, fetch_dataset_source, get_dataset_source
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_bytes,
    atomic_write_json,
    content_fingerprint,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.preprocessing import MissingValuePolicy, preprocess_splits

SICK_BUILD_SPEC_NAME = "sick-uci-102-v1.json"
ADULT_BUILD_SPEC_NAME = "adult-uci-2-v1.json"


class OfficialDatasetError(ValueError):
    """Raised when an official source does not satisfy its frozen build contract."""


@dataclass(frozen=True)
class ParsedSickSplit:
    frame: pd.DataFrame
    record_ids: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ParsedAdultSplit:
    frame: pd.DataFrame
    summary: dict[str, Any]


def _build_spec_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "resources" / "datasets" / filename


def load_sick_build_spec(path: str | Path | None = None) -> dict[str, Any]:
    payload = read_json(path or _build_spec_path(SICK_BUILD_SPEC_NAME))
    required = {
        "build_schema_version",
        "dataset_id",
        "dataset_view",
        "source_version",
        "source_members",
        "raw_columns",
        "model_view",
        "split_suffix_pattern",
        "duplicate_audit",
        "splits",
        "preprocessing",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise OfficialDatasetError("The Sick build specification does not satisfy the v1 field contract")
    if payload["build_schema_version"] != "1.0.0" or payload["dataset_id"] != "sick":
        raise OfficialDatasetError("Unsupported Sick build specification")
    raw_columns = payload["raw_columns"]
    if not isinstance(raw_columns, list) or len(raw_columns) != 30 or len(set(raw_columns)) != 30:
        raise OfficialDatasetError("The official Sick raw schema must contain 30 unique columns")
    model_view = payload["model_view"]
    if not isinstance(model_view, dict) or set(model_view) != {
        "numerical_columns",
        "categorical_columns",
        "target_columns",
        "excluded_columns",
    }:
        raise OfficialDatasetError("Invalid Sick model-view contract")
    role_columns = [
        *model_view["numerical_columns"],
        *model_view["categorical_columns"],
        *model_view["target_columns"],
    ]
    excluded = model_view["excluded_columns"]
    if (
        not isinstance(excluded, dict)
        or len(role_columns) != len(set(role_columns))
        or set(role_columns) | set(excluded) != set(raw_columns)
        or set(role_columns) & set(excluded)
    ):
        raise OfficialDatasetError("Sick model roles and exclusions must partition the raw schema")
    if model_view["target_columns"] != ["Class"] or set(excluded) != {"TBG"}:
        raise OfficialDatasetError("The v1 Sick view must target Class and exclude only all-missing TBG")
    if set(payload["splits"]) != {"train", "test"}:
        raise OfficialDatasetError("The official Sick split contract must contain train and test")
    re.compile(payload["split_suffix_pattern"])
    return payload


def load_adult_build_spec(path: str | Path | None = None) -> dict[str, Any]:
    payload = read_json(path or _build_spec_path(ADULT_BUILD_SPEC_NAME))
    required = {
        "build_schema_version",
        "dataset_id",
        "dataset_view",
        "source_version",
        "source_members",
        "raw_columns",
        "model_view",
        "source_format",
        "valid_domains",
        "duplicate_audit",
        "splits",
        "preprocessing",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise OfficialDatasetError("The Adult build specification does not satisfy the v1 field contract")
    if (
        payload["build_schema_version"] != "1.0.0"
        or payload["dataset_id"] != "adult"
        or payload["dataset_view"] != "adult-uci-2-model-v1"
    ):
        raise OfficialDatasetError("Unsupported Adult build specification")
    expected_members = {"Index", "adult.data", "adult.names", "adult.test", "old.adult.names"}
    source_members = payload["source_members"]
    if (
        not isinstance(source_members, dict)
        or set(source_members) != expected_members
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in source_members.values()
        )
    ):
        raise OfficialDatasetError("Adult source members must contain the five checksum-pinned UCI files")
    raw_columns = payload["raw_columns"]
    if not isinstance(raw_columns, list) or len(raw_columns) != 15 or len(set(raw_columns)) != 15:
        raise OfficialDatasetError("The official Adult raw schema must contain 15 unique columns")
    model_view = payload["model_view"]
    if not isinstance(model_view, dict) or set(model_view) != {
        "numerical_columns",
        "categorical_columns",
        "target_columns",
        "excluded_columns",
        "integer_columns",
    }:
        raise OfficialDatasetError("Invalid Adult model-view contract")
    role_columns = [
        *model_view["numerical_columns"],
        *model_view["categorical_columns"],
        *model_view["target_columns"],
    ]
    if (
        len(role_columns) != len(set(role_columns))
        or set(role_columns) != set(raw_columns)
        or model_view["excluded_columns"] != {}
        or model_view["target_columns"] != ["income"]
        or set(model_view["integer_columns"]) != set(model_view["numerical_columns"])
    ):
        raise OfficialDatasetError("Adult model roles must partition all 15 columns and target income")
    source_format = payload["source_format"]
    if source_format != {
        "encoding": "ascii",
        "delimiter": ", ",
        "terminal_newlines": 2,
        "test_header": "|1x3 Cross validator",
        "train_target_suffix": "",
        "test_target_suffix": ".",
    }:
        raise OfficialDatasetError("The Adult source-format contract has changed")
    domains = payload["valid_domains"]
    if not isinstance(domains, dict) or set(domains) != {
        "numerical",
        "categorical",
        "nullable_raw_columns",
    }:
        raise OfficialDatasetError("Invalid Adult domain contract")
    if set(domains["numerical"]) != set(model_view["numerical_columns"]):
        raise OfficialDatasetError("Adult numerical domains do not match numerical columns")
    if set(domains["categorical"]) != {
        *model_view["categorical_columns"],
        *model_view["target_columns"],
    }:
        raise OfficialDatasetError("Adult categorical domains do not match categorical and target columns")
    if set(domains["nullable_raw_columns"]) != {"workclass", "occupation", "native.country"}:
        raise OfficialDatasetError("Only the three official Adult unknown-value columns may be nullable")
    for column, domain in domains["numerical"].items():
        if (
            not isinstance(domain, dict)
            or set(domain) != {"minimum", "maximum", "integer"}
            or domain["integer"] is not True
            or not isinstance(domain["minimum"], int)
            or isinstance(domain["minimum"], bool)
            or not isinstance(domain["maximum"], int)
            or isinstance(domain["maximum"], bool)
            or domain["minimum"] > domain["maximum"]
        ):
            raise OfficialDatasetError(f"Invalid Adult numerical domain for {column}")
    for column, domain in domains["categorical"].items():
        if (
            not isinstance(domain, list)
            or not domain
            or any(not isinstance(value, str) or not value or value == "?" for value in domain)
            or len(domain) != len(set(domain))
        ):
            raise OfficialDatasetError(f"Invalid Adult categorical domain for {column}")
    if set(payload["splits"]) != {"train", "test"}:
        raise OfficialDatasetError("The official Adult split contract must contain train and test")
    preprocessing = payload["preprocessing"]
    if (
        not isinstance(preprocessing, dict)
        or set(preprocessing)
        != {
            "missing_markers",
            "fitted_on_split",
            "numerical_strategy",
            "categorical_strategy",
            "target_strategy",
            "expected_categorical_fill_values",
        }
        or preprocessing["missing_markers"] != ["?"]
        or preprocessing["fitted_on_split"] != "train"
        or preprocessing["numerical_strategy"] != "mean"
        or preprocessing["categorical_strategy"] != "most_frequent"
        or preprocessing["target_strategy"] != "error"
        or set(preprocessing["expected_categorical_fill_values"]) != {"workclass", "occupation", "native.country"}
    ):
        raise OfficialDatasetError("The Adult train-only preprocessing contract has changed")
    expected_split_members = {"train": "adult.data", "test": "adult.test"}
    for split_name, split in payload["splits"].items():
        if (
            not isinstance(split, dict)
            or set(split) != {"member", "rows", "class_counts", "canonical_rows_sha256", "missing_counts"}
            or split["member"] != expected_split_members[split_name]
            or not isinstance(split["rows"], int)
            or isinstance(split["rows"], bool)
            or split["rows"] <= 0
            or set(split["class_counts"]) != {"<=50K", ">50K"}
            or sum(split["class_counts"].values()) != split["rows"]
            or not re.fullmatch(r"[0-9a-f]{64}", split["canonical_rows_sha256"])
            or set(split["missing_counts"]) != {"workclass", "occupation", "native.country"}
        ):
            raise OfficialDatasetError(f"Invalid Adult {split_name} identity contract")
    audit_fields = {
        "train_duplicate_rows",
        "test_duplicate_rows",
        "cross_split_unique_rows",
        "train_rows_in_cross_split_overlap",
        "test_rows_in_cross_split_overlap",
    }
    duplicate_audit = payload["duplicate_audit"]
    if not isinstance(duplicate_audit, dict) or set(duplicate_audit) != {
        "raw_model_view",
        "processed_model_view",
    }:
        raise OfficialDatasetError("Invalid Adult duplicate-audit contract")
    for audit in duplicate_audit.values():
        if (
            not isinstance(audit, dict)
            or set(audit) != audit_fields
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in audit.values())
        ):
            raise OfficialDatasetError("Invalid Adult duplicate-audit counters")
    return payload


def _ordered_values_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _ordered_rows_sha256(frame: pd.DataFrame) -> str:
    rows = ("\x1f".join(map(str, row)) for row in frame.itertuples(index=False, name=None))
    return hashlib.sha256(("\n".join(rows) + "\n").encode("ascii")).hexdigest()


def _duplicate_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    train_rows = list(map(tuple, train.to_numpy()))
    test_rows = list(map(tuple, test.to_numpy()))
    overlap = set(train_rows) & set(test_rows)
    return {
        "train_duplicate_rows": len(train_rows) - len(set(train_rows)),
        "test_duplicate_rows": len(test_rows) - len(set(test_rows)),
        "cross_split_unique_rows": len(overlap),
        "train_rows_in_cross_split_overlap": sum(row in overlap for row in train_rows),
        "test_rows_in_cross_split_overlap": sum(row in overlap for row in test_rows),
    }


def _validate_raw_values(frame: pd.DataFrame, build_spec: dict[str, Any], split_name: str) -> None:
    model_view = build_spec["model_view"]
    numerical_columns = [*model_view["numerical_columns"], "TBG"]
    for column in numerical_columns:
        for value in frame[column]:
            if value == "?":
                continue
            try:
                converted = float(value)
            except ValueError as exc:
                raise OfficialDatasetError(
                    f"Official Sick {split_name} has a non-numeric {column} value: {value!r}"
                ) from exc
            if not math.isfinite(converted):
                raise OfficialDatasetError(f"Official Sick {split_name} has a non-finite {column} value")

    categorical_columns = set(model_view["categorical_columns"])
    binary_columns = categorical_columns - {"sex", "referral_source"}
    allowed = {
        "sex": {"M", "F", "?"},
        "referral_source": {"WEST", "STMW", "SVHC", "SVI", "SVHD", "other"},
        "Class": {"sick", "negative"},
    }
    for column in binary_columns:
        allowed[column] = {"f", "t"}
    for column, domain in allowed.items():
        unexpected = sorted(set(map(str, frame[column])) - domain)
        if unexpected:
            raise OfficialDatasetError(
                f"Official Sick {split_name} contains values outside the {column} domain: {unexpected}"
            )


def _parse_sick_payload(payload: bytes, build_spec: dict[str, Any], split_name: str) -> ParsedSickSplit:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OfficialDatasetError(f"Official Sick {split_name} is not ASCII") from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise OfficialDatasetError(f"Official Sick {split_name} contains an empty input line")

    raw_columns = build_spec["raw_columns"]
    suffix_pattern = re.compile(build_spec["split_suffix_pattern"])
    rows: list[list[str]] = []
    record_ids: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        row = next(csv.reader([line], strict=True))
        if len(row) != len(raw_columns):
            raise OfficialDatasetError(
                f"Official Sick {split_name} line {line_number} has {len(row)} fields; expected {len(raw_columns)}"
            )
        suffix = suffix_pattern.fullmatch(row[-1])
        if suffix is None:
            raise OfficialDatasetError(
                f"Official Sick {split_name} line {line_number} has an invalid class/record suffix"
            )
        row[-1] = suffix.group(1)
        record_ids.append(suffix.group(2))
        rows.append(row)
    if len(record_ids) != len(set(record_ids)):
        raise OfficialDatasetError(f"Official Sick {split_name} contains duplicate record identifiers")

    frame = pd.DataFrame(rows, columns=raw_columns)
    _validate_raw_values(frame, build_spec, split_name)
    missing_counts = {column: int((frame[column] == "?").sum()) for column in raw_columns}
    summary = {
        "rows": len(frame),
        "class_counts": dict(sorted(Counter(map(str, frame["Class"])).items())),
        "record_ids_sha256": _ordered_values_sha256(record_ids),
        "missing_counts": {column: count for column, count in missing_counts.items() if count},
    }
    return ParsedSickSplit(frame=frame, record_ids=tuple(record_ids), summary=summary)


def parse_uci_sick_file(
    path: str | Path,
    *,
    split_name: str,
    build_spec: dict[str, Any] | None = None,
) -> ParsedSickSplit:
    """Parse one official UCI Sick split and enforce its locked identity."""

    spec = build_spec or load_sick_build_spec()
    if split_name not in {"train", "test"}:
        raise OfficialDatasetError(f"Unknown Sick split: {split_name!r}")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Official Sick source member does not exist: {source_path}")
    expected = spec["splits"][split_name]
    expected_member_sha = spec["source_members"][expected["member"]]
    observed_member_sha = sha256_file(source_path)
    if observed_member_sha != expected_member_sha:
        raise OfficialDatasetError(
            f"Official Sick {split_name} member checksum mismatch: "
            f"expected={expected_member_sha}, observed={observed_member_sha}"
        )
    parsed = _parse_sick_payload(source_path.read_bytes(), spec, split_name)
    expected_summary = {
        "rows": expected["rows"],
        "class_counts": expected["class_counts"],
        "record_ids_sha256": expected["record_ids_sha256"],
        "missing_counts": expected["missing_counts"],
    }
    if parsed.summary != expected_summary:
        raise OfficialDatasetError(
            f"Official Sick {split_name} content differs from its locked split contract: "
            f"expected={expected_summary}, observed={parsed.summary}"
        )
    return parsed


def _validate_adult_values(frame: pd.DataFrame, build_spec: dict[str, Any], split_name: str) -> None:
    domains = build_spec["valid_domains"]
    for column, domain in domains["numerical"].items():
        for value in frame[column]:
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
                raise OfficialDatasetError(
                    f"Official Adult {split_name} has a non-canonical integer {column} value: {value!r}"
                )
            converted = int(value)
            if not domain["minimum"] <= converted <= domain["maximum"]:
                raise OfficialDatasetError(f"Official Adult {split_name} has an out-of-range {column} value: {value!r}")

    nullable = set(domains["nullable_raw_columns"])
    for column, declared_values in domains["categorical"].items():
        allowed = set(declared_values)
        if column in nullable:
            allowed.add("?")
        unexpected = sorted(set(map(str, frame[column])) - allowed)
        if unexpected:
            raise OfficialDatasetError(
                f"Official Adult {split_name} contains values outside the {column} domain: {unexpected}"
            )
    unexpected_missing = {
        column: int((frame[column] == "?").sum())
        for column in frame.columns
        if column not in nullable and bool((frame[column] == "?").any())
    }
    if unexpected_missing:
        raise OfficialDatasetError(
            f"Official Adult {split_name} contains unknown markers in prohibited columns: {unexpected_missing}"
        )


def _parse_adult_payload(payload: bytes, build_spec: dict[str, Any], split_name: str) -> ParsedAdultSplit:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OfficialDatasetError(f"Official Adult {split_name} is not ASCII") from exc
    terminal = "\n" * build_spec["source_format"]["terminal_newlines"]
    if "\r" in text or not text.endswith(terminal):
        raise OfficialDatasetError(
            f"Official Adult {split_name} must use LF endings and exactly the registered terminal blank line"
        )
    body = text[: -len(terminal)]
    lines = body.split("\n")
    if not lines or any(not line for line in lines):
        raise OfficialDatasetError(f"Official Adult {split_name} contains an unexpected empty input line")
    if split_name == "test":
        expected_header = build_spec["source_format"]["test_header"]
        if lines[0] != expected_header:
            raise OfficialDatasetError("Official Adult test header differs from the registered source contract")
        lines = lines[1:]

    raw_columns = build_spec["raw_columns"]
    target_suffix = build_spec["source_format"][f"{split_name}_target_suffix"]
    rows: list[list[str]] = []
    for line_number, line in enumerate(lines, start=2 if split_name == "test" else 1):
        raw_fields = line.split(",")
        if len(raw_fields) != len(raw_columns):
            raise OfficialDatasetError(
                f"Official Adult {split_name} line {line_number} has {len(raw_fields)} fields; "
                f"expected {len(raw_columns)}"
            )
        if raw_fields[0] != raw_fields[0].strip() or any(
            not field.startswith(" ") or field.startswith("  ") or field[1:] != field[1:].strip()
            for field in raw_fields[1:]
        ):
            raise OfficialDatasetError(
                f"Official Adult {split_name} line {line_number} does not use the exact comma-space format"
            )
        row = [raw_fields[0], *(field[1:] for field in raw_fields[1:])]
        if target_suffix:
            if not row[-1].endswith(target_suffix) or row[-1][: -len(target_suffix)].endswith(target_suffix):
                raise OfficialDatasetError(
                    f"Official Adult {split_name} line {line_number} has an invalid target suffix"
                )
            row[-1] = row[-1][: -len(target_suffix)]
        rows.append(row)

    frame = pd.DataFrame(rows, columns=raw_columns)
    _validate_adult_values(frame, build_spec, split_name)
    missing_counts = {column: int((frame[column] == "?").sum()) for column in raw_columns}
    summary = {
        "rows": len(frame),
        "class_counts": dict(sorted(Counter(map(str, frame["income"])).items())),
        "canonical_rows_sha256": _ordered_rows_sha256(frame),
        "missing_counts": {column: count for column, count in missing_counts.items() if count},
    }
    return ParsedAdultSplit(frame=frame, summary=summary)


def parse_uci_adult_file(
    path: str | Path,
    *,
    split_name: str,
    build_spec: dict[str, Any] | None = None,
) -> ParsedAdultSplit:
    """Parse one official UCI Adult split and enforce its locked identity."""

    spec = build_spec or load_adult_build_spec()
    if split_name not in {"train", "test"}:
        raise OfficialDatasetError(f"Unknown Adult split: {split_name!r}")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Official Adult source member does not exist: {source_path}")
    expected = spec["splits"][split_name]
    expected_member_sha = spec["source_members"][expected["member"]]
    observed_member_sha = sha256_file(source_path)
    if observed_member_sha != expected_member_sha:
        raise OfficialDatasetError(
            f"Official Adult {split_name} member checksum mismatch: "
            f"expected={expected_member_sha}, observed={observed_member_sha}"
        )
    parsed = _parse_adult_payload(source_path.read_bytes(), spec, split_name)
    expected_summary = {
        "rows": expected["rows"],
        "class_counts": expected["class_counts"],
        "canonical_rows_sha256": expected["canonical_rows_sha256"],
        "missing_counts": expected["missing_counts"],
    }
    if parsed.summary != expected_summary:
        raise OfficialDatasetError(
            f"Official Adult {split_name} content differs from its locked split contract: "
            f"expected={expected_summary}, observed={parsed.summary}"
        )
    return parsed


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _column_mappings(
    columns: list[str], numerical_columns: list[str], categorical_columns: list[str], target_columns: list[str]
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    grouped = [*numerical_columns, *categorical_columns, *target_columns]
    grouped_index = {column: index for index, column in enumerate(grouped)}
    index_mapping = {str(index): grouped_index[column] for index, column in enumerate(columns)}
    inverse_mapping = {str(grouped_index[column]): index for index, column in enumerate(columns)}
    name_mapping = {str(index): column for index, column in enumerate(columns)}
    return index_mapping, inverse_mapping, name_mapping


def _upstream_info(
    dataset_id: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    build_spec: dict[str, Any],
    state_fingerprint: str,
) -> dict[str, Any]:
    model_view = build_spec["model_view"]
    numerical_columns = list(model_view["numerical_columns"])
    categorical_columns = list(model_view["categorical_columns"])
    target_columns = list(model_view["target_columns"])
    columns = list(train.columns)
    numerical_indices = [columns.index(column) for column in numerical_columns]
    categorical_indices = [columns.index(column) for column in categorical_columns]
    target_indices = [columns.index(column) for column in target_columns]
    index_mapping, inverse_mapping, name_mapping = _column_mappings(
        columns, numerical_columns, categorical_columns, target_columns
    )
    integer_columns = list(model_view.get("integer_columns", []))
    integer_indices = [columns.index(column) for column in integer_columns]
    integer_indices_wrt_numerical = [numerical_columns.index(column) for column in integer_columns]

    column_info: dict[str, Any] = {}
    metadata_columns: dict[str, Any] = {}
    for index, column in enumerate(columns):
        if column in numerical_columns:
            column_info[column] = {
                "type": "numerical",
                "min": float(train[column].min()),
                "max": float(train[column].max()),
            }
            metadata_columns[str(index)] = {"sdtype": "numerical", "computer_representation": "Float"}
        else:
            categories = sorted(set(map(str, train[column])))
            column_info[column] = {"type": "categorical", "categories": categories}
            metadata_columns[str(index)] = {"sdtype": "categorical"}

    return {
        "name": dataset_id,
        "task_type": "binclass",
        "header": 0,
        "column_names": columns,
        "num_col_idx": numerical_indices,
        "cat_col_idx": categorical_indices,
        "target_col_idx": target_indices,
        "file_type": "csv",
        "data_path": f"data/{dataset_id}/train.csv",
        "val_path": None,
        "test_path": f"data/{dataset_id}/test.csv",
        "column_info": column_info,
        "int_col_idx": integer_indices,
        "int_columns": integer_columns,
        "int_col_idx_wrt_num": integer_indices_wrt_numerical,
        "train_num": len(train),
        "test_num": len(test),
        "val_num": 0,
        "n_classes": 2,
        "idx_mapping": index_mapping,
        "inverse_idx_mapping": inverse_mapping,
        "idx_name_mapping": name_mapping,
        "metadata": {"columns": metadata_columns},
        "dataset_view": build_spec["dataset_view"],
        "source_version": build_spec["source_version"],
        "preprocessing_state_fingerprint": state_fingerprint,
        "excluded_raw_columns": model_view["excluded_columns"],
    }


def _assert_scoped_directory(path: Path, repo_root: Path) -> None:
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root) or resolved_path == resolved_root:
        raise OfficialDatasetError(f"Refusing an out-of-scope dataset directory: {resolved_path}")


def _remove_scoped_directory(path: Path, repo_root: Path) -> None:
    _assert_scoped_directory(path, repo_root)
    if path.is_symlink():
        raise OfficialDatasetError(f"Refusing a symlinked dataset directory: {path}")
    if path.exists():
        shutil.rmtree(path)


def _atomic_replace_directory(staging: Path, destination: Path, repo_root: Path) -> None:
    _assert_scoped_directory(staging, repo_root)
    _assert_scoped_directory(destination, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise OfficialDatasetError(f"Refusing a symlinked dataset destination: {destination}")
    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    installed = False
    try:
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
        installed = True
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if installed and backup.exists():
            _remove_scoped_directory(backup, repo_root)


def _copy_directory_atomically(source: Path, destination: Path, repo_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        _atomic_replace_directory(staging, destination, repo_root)
    finally:
        if staging.exists():
            _remove_scoped_directory(staging, repo_root)


def _write_synthetic_view(source_data: Path, destination: Path, repo_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copy2(source_data / "train.csv", staging / "real.csv")
        shutil.copy2(source_data / "test.csv", staging / "test.csv")
        _atomic_replace_directory(staging, destination, repo_root)
    finally:
        if staging.exists():
            _remove_scoped_directory(staging, repo_root)


def _write_primary_dataset_data(
    dataset_id: str,
    destination: Path,
    repo_root: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    build_spec: dict[str, Any],
    source_manifest: dict[str, Any],
    split_summaries: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-build-", dir=destination.parent))
    try:
        model_view = build_spec["model_view"]
        policy = MissingValuePolicy(missing_markers=tuple(build_spec["preprocessing"]["missing_markers"]))
        transformed = preprocess_splits(
            train,
            test=test,
            numerical_columns=model_view["numerical_columns"],
            categorical_columns=model_view["categorical_columns"],
            target_columns=model_view["target_columns"],
            policy=policy,
        )
        if transformed.test is None:
            raise OfficialDatasetError(f"The official {dataset_id} test split was not transformed")
        train_output = transformed.train
        test_output = transformed.test
        expected_fill_values = build_spec["preprocessing"].get("expected_categorical_fill_values")
        if expected_fill_values is not None:
            observed_fill_values = {
                column: transformed.state.categorical_fill_values.get(column) for column in expected_fill_values
            }
            if observed_fill_values != expected_fill_values:
                raise OfficialDatasetError(
                    f"Official {dataset_id} train-fitted categorical modes differ from the locked build contract: "
                    f"expected={expected_fill_values}, observed={observed_fill_values}"
                )
        processed_duplicate_audit = _duplicate_audit(train_output, test_output)
        expected_processed_audit = build_spec["duplicate_audit"]["processed_model_view"]
        if processed_duplicate_audit != expected_processed_audit:
            raise OfficialDatasetError(
                f"Processed {dataset_id} duplicate audit differs from the locked build contract: "
                f"expected={expected_processed_audit}, observed={processed_duplicate_audit}"
            )

        atomic_write_bytes(staging / "train.csv", _csv_bytes(train_output))
        atomic_write_bytes(staging / "test.csv", _csv_bytes(test_output))
        atomic_write_json(staging / "imputation-state.json", transformed.state.to_dict())
        preprocessing_manifest = {
            "manifest_schema_version": "1.0.0",
            "dataset": dataset_id,
            "dataset_view": build_spec["dataset_view"],
            "source_version": build_spec["source_version"],
            "train_only_fitting": True,
            "fitted_on_split": "train",
            "state_fingerprint": transformed.state.fingerprint,
            "policy": transformed.state.policy.to_dict(),
            "excluded_raw_columns": model_view["excluded_columns"],
            "duplicate_audit": build_spec["duplicate_audit"],
            "reports": transformed.reports,
        }
        atomic_write_json(staging / "preprocessing-manifest.json", preprocessing_manifest)
        atomic_write_json(
            staging / "split-identity.json",
            {
                "identity_schema_version": "1.0.0",
                "dataset": dataset_id,
                "dataset_view": build_spec["dataset_view"],
                "splits": split_summaries,
                "duplicate_audit": build_spec["duplicate_audit"],
            },
        )
        atomic_write_json(staging / "source-manifest.json", source_manifest)
        atomic_write_json(staging / "dataset-build.json", build_spec)

        info = _upstream_info(dataset_id, train_output, test_output, build_spec, transformed.state.fingerprint)
        atomic_write_json(staging / "info.json", info)
        numerical_columns = model_view["numerical_columns"]
        categorical_columns = model_view["categorical_columns"]
        target_columns = model_view["target_columns"]
        for split_name, frame in (("train", train_output), ("test", test_output)):
            atomic_write_bytes(
                staging / f"X_num_{split_name}.npy",
                _npy_bytes(frame[numerical_columns].to_numpy(dtype=np.float32)),
            )
            atomic_write_bytes(
                staging / f"X_cat_{split_name}.npy",
                _npy_bytes(frame[categorical_columns].to_numpy(dtype=str)),
            )
            atomic_write_bytes(
                staging / f"y_{split_name}.npy",
                _npy_bytes(frame[target_columns].to_numpy(dtype=str)),
            )
        _atomic_replace_directory(staging, destination, repo_root)
    finally:
        if staging.exists():
            _remove_scoped_directory(staging, repo_root)


def _install_adapter_views_and_manifest(
    *,
    root: Path,
    dataset_id: str,
    build_spec: dict[str, Any],
    source: DatasetSource,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    primary_data = root / "TabDiff-main" / "data" / dataset_id
    tabdiff_synthetic = root / "TabDiff-main" / "synthetic" / dataset_id
    tabsyn_data = root / "TabSyn-main" / "data" / dataset_id
    tabsyn_synthetic = root / "TabSyn-main" / "synthetic" / dataset_id
    _write_synthetic_view(primary_data, tabdiff_synthetic, root)
    _copy_directory_atomically(primary_data, tabsyn_data, root)
    _write_synthetic_view(primary_data, tabsyn_synthetic, root)

    def repository_path(path: Path) -> str:
        return path.relative_to(root).as_posix()

    artifact_paths = [
        primary_data / "train.csv",
        primary_data / "test.csv",
        primary_data / "info.json",
        primary_data / "imputation-state.json",
        primary_data / "preprocessing-manifest.json",
        primary_data / "split-identity.json",
    ]
    manifest = {
        "manifest_schema_version": "1.0.0",
        "path_base": "repository-root",
        "dataset": dataset_id,
        "dataset_view": build_spec["dataset_view"],
        "materialized_by": "official-uci-builder",
        "source": source.to_dict(),
        "source_manifest_fingerprint": content_fingerprint(source_manifest),
        "build_spec_fingerprint": content_fingerprint(build_spec),
        "metadata_path": repository_path(primary_data / "info.json"),
        "train_data_path": repository_path(primary_data / "train.csv"),
        "val_data_path": None,
        "test_data_path": repository_path(primary_data / "test.csv"),
        "synthetic_real_path": repository_path(tabdiff_synthetic / "real.csv"),
        "synthetic_val_path": None,
        "synthetic_test_path": repository_path(tabdiff_synthetic / "test.csv"),
        "artifacts": {
            repository_path(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
        "synced_roots": {
            "TabSyn-main": {
                "data_dir": repository_path(tabsyn_data),
                "synthetic_dir": repository_path(tabsyn_synthetic),
            }
        },
    }
    output_path = root / "materialized_datasets" / dataset_id / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, manifest)
    return manifest


def materialize_official_sick(
    *,
    repo_root: str | Path,
    cache_root: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Build the canonical Sick model view from the checksum-pinned UCI source."""

    root = Path(repo_root).resolve()
    build_spec = load_sick_build_spec()
    source = get_dataset_source("sick")
    if source.source_version != build_spec["source_version"]:
        raise OfficialDatasetError("Sick source registry and build specification versions disagree")
    fetched = fetch_dataset_source(
        "sick",
        cache_root=cache_root,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )
    extracted = Path(fetched["extraction"]["extracted_path"])
    for member_name, expected_sha in build_spec["source_members"].items():
        observed_sha = sha256_file(extracted / member_name)
        if observed_sha != expected_sha:
            raise OfficialDatasetError(
                f"Official Sick member checksum mismatch for {member_name}: "
                f"expected={expected_sha}, observed={observed_sha}"
            )

    parsed = {
        split_name: parse_uci_sick_file(
            extracted / split_spec["member"],
            split_name=split_name,
            build_spec=build_spec,
        )
        for split_name, split_spec in build_spec["splits"].items()
    }
    overlap = set(parsed["train"].record_ids) & set(parsed["test"].record_ids)
    if overlap:
        raise OfficialDatasetError(f"Official Sick train/test record identifiers overlap: {len(overlap)}")

    model_columns = [
        column for column in build_spec["raw_columns"] if column not in build_spec["model_view"]["excluded_columns"]
    ]
    train = parsed["train"].frame.loc[:, model_columns]
    test = parsed["test"].frame.loc[:, model_columns]
    raw_duplicate_audit = _duplicate_audit(train, test)
    expected_raw_audit = build_spec["duplicate_audit"]["raw_model_view"]
    if raw_duplicate_audit != expected_raw_audit:
        raise OfficialDatasetError(
            "Raw Sick duplicate audit differs from the locked build contract: "
            f"expected={expected_raw_audit}, observed={raw_duplicate_audit}"
        )
    primary_data = root / "TabDiff-main" / "data" / "sick"
    _write_primary_dataset_data(
        "sick",
        primary_data,
        root,
        train,
        test,
        build_spec,
        fetched["extraction"]["manifest"],
        {split_name: value.summary for split_name, value in parsed.items()},
    )

    return _install_adapter_views_and_manifest(
        root=root,
        dataset_id="sick",
        build_spec=build_spec,
        source=source,
        source_manifest=fetched["extraction"]["manifest"],
    )


def materialize_official_adult(
    *,
    repo_root: str | Path,
    cache_root: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Build the canonical Adult model view from the checksum-pinned UCI source."""

    root = Path(repo_root).resolve()
    build_spec = load_adult_build_spec()
    source = get_dataset_source("adult")
    if source.source_version != build_spec["source_version"]:
        raise OfficialDatasetError("Adult source registry and build specification versions disagree")
    fetched = fetch_dataset_source(
        "adult",
        cache_root=cache_root,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )
    extracted = Path(fetched["extraction"]["extracted_path"])
    for member_name, expected_sha in build_spec["source_members"].items():
        observed_sha = sha256_file(extracted / member_name)
        if observed_sha != expected_sha:
            raise OfficialDatasetError(
                f"Official Adult member checksum mismatch for {member_name}: "
                f"expected={expected_sha}, observed={observed_sha}"
            )

    parsed = {
        split_name: parse_uci_adult_file(
            extracted / split_spec["member"],
            split_name=split_name,
            build_spec=build_spec,
        )
        for split_name, split_spec in build_spec["splits"].items()
    }
    train = parsed["train"].frame
    test = parsed["test"].frame
    raw_duplicate_audit = _duplicate_audit(train, test)
    expected_raw_audit = build_spec["duplicate_audit"]["raw_model_view"]
    if raw_duplicate_audit != expected_raw_audit:
        raise OfficialDatasetError(
            "Raw Adult duplicate audit differs from the locked build contract: "
            f"expected={expected_raw_audit}, observed={raw_duplicate_audit}"
        )

    primary_data = root / "TabDiff-main" / "data" / "adult"
    _write_primary_dataset_data(
        "adult",
        primary_data,
        root,
        train,
        test,
        build_spec,
        fetched["extraction"]["manifest"],
        {split_name: value.summary for split_name, value in parsed.items()},
    )
    return _install_adapter_views_and_manifest(
        root=root,
        dataset_id="adult",
        build_spec=build_spec,
        source=source,
        source_manifest=fetched["extraction"]["manifest"],
    )


__all__ = [
    "OfficialDatasetError",
    "ParsedAdultSplit",
    "ParsedSickSplit",
    "load_adult_build_spec",
    "load_sick_build_spec",
    "materialize_official_adult",
    "materialize_official_sick",
    "parse_uci_adult_file",
    "parse_uci_sick_file",
]
