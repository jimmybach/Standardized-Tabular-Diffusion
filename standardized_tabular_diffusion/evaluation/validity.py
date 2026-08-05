"""Benchmark-native P3 validity rules for original decoded synthetic tables."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

import pandas as pd

from standardized_tabular_diffusion.evaluation.contracts import (
    AtomicResult,
    EvaluationRequest,
    MetricState,
    RawDirection,
    utc_timestamp,
)
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint
from standardized_tabular_diffusion.evaluation.table import ValidatedTables, model_view_column_specs

COLUMN_VALIDITY_METRIC_ID = "std-tabular-column-validity"
CONSTRAINT_VALIDITY_METRIC_ID = "std-tabular-constraint-validity"
P3_METRIC_VERSION = "1.0.0"
P3_METRICS = (
    {"metric_id": COLUMN_VALIDITY_METRIC_ID, "metric_version": P3_METRIC_VERSION},
    {"metric_id": CONSTRAINT_VALIDITY_METRIC_ID, "metric_version": P3_METRIC_VERSION},
)
VALIDITY_DETAILS_ARTIFACT_PATH = "artifacts/validity-details.json"
VALIDITY_IMPLEMENTATION_VERSION = "1.0.0"

_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_SEMANTIC_TYPES = {"continuous", "integer", "categorical", "boolean", "datetime", "string"}
_RULE_TYPES = {
    "not_null",
    "finite",
    "integer",
    "allowed_values",
    "bounds",
    "regex",
    "length",
    "unique",
    "datetime_range",
}
_CONSTRAINT_TYPES = {
    "comparison",
    "conditional_domain",
    "mutual_exclusion",
    "sum_equals",
    "allowed_combinations",
    "functional_dependency",
}
_EVIDENCE_TYPES = {
    "authoritative-data-dictionary",
    "dataset-documentation",
    "legal-or-policy-requirement",
    "recorded-human-review",
}


class ValidityError(RuntimeError):
    """Base class for deterministic P3 validity failures."""


class ValidityProfileError(ValidityError):
    """Raised when a Dataset Profile has an invalid or incomplete rule contract."""


@dataclass(frozen=True)
class ValidityOutcome:
    """Atomic validity records and exactly reproducible report-level summaries."""

    atomic_results: tuple[AtomicResult, ...]
    property_scores: dict[str, float | None]
    denominator_counts: dict[str, int]
    details: dict[str, Any]
    source: dict[str, Any]


@dataclass(frozen=True)
class _ConstraintEvaluation:
    definition: dict[str, Any]
    applicable: pd.Series
    satisfied: pd.Series
    score: float | None


def _fail(detail: str) -> NoReturn:
    raise ValidityProfileError(detail)


def _require_exact(name: str, payload: Any, required: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) != required:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        _fail(f"{name} fields differ from the P3 contract; expected={sorted(required)}, observed={observed}")


def _require_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{name} must be a portable lowercase identifier")
    return value


def _require_string_list(name: str, value: Any, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value) or not all(isinstance(item, str) for item in value):
        _fail(f"{name} must be {'a non-empty' if nonempty else 'an'} array of strings")
    if len(value) != len(set(value)):
        _fail(f"{name} must not contain duplicates")
    return value


def _validate_evidence(name: str, payload: Any) -> None:
    _require_exact(name, payload, {"source_type", "reference"})
    if payload["source_type"] not in _EVIDENCE_TYPES:
        _fail(f"{name}.source_type is not an approved hard-rule evidence class")
    if not isinstance(payload["reference"], str) or not payload["reference"].strip():
        _fail(f"{name}.reference must be non-empty")


def _validate_selector(name: str, payload: Any, known_columns: set[str]) -> None:
    allowed = {"column_ids", "semantic_types", "nullable_model_input", "roles_any", "valid_domain_keys"}
    if not isinstance(payload, dict) or not payload or not set(payload).issubset(allowed):
        _fail(f"{name} must be a non-empty selector using only {sorted(allowed)}")
    if "column_ids" in payload:
        columns = set(_require_string_list(f"{name}.column_ids", payload["column_ids"], nonempty=True))
        if not columns.issubset(known_columns):
            _fail(f"{name}.column_ids contains unknown identifiers: {sorted(columns - known_columns)}")
    if "semantic_types" in payload:
        values = set(_require_string_list(f"{name}.semantic_types", payload["semantic_types"], nonempty=True))
        if not values.issubset(_SEMANTIC_TYPES):
            _fail(f"{name}.semantic_types contains unsupported values: {sorted(values - _SEMANTIC_TYPES)}")
    if "nullable_model_input" in payload and not isinstance(payload["nullable_model_input"], bool):
        _fail(f"{name}.nullable_model_input must be Boolean")
    for key in ("roles_any", "valid_domain_keys"):
        if key in payload:
            _require_string_list(f"{name}.{key}", payload[key], nonempty=True)


def _finite_number(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail(f"{name} must be a finite number")
    if not math.isfinite(result):
        _fail(f"{name} must be a finite number")
    if nonnegative and result < 0:
        _fail(f"{name} must be non-negative")
    return result


def _validate_scalar_values(name: str, values: Any, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(values, list) or (nonempty and not values):
        _fail(f"{name} must be {'a non-empty' if nonempty else 'an'} array")
    for index, value in enumerate(values):
        if value is None or isinstance(value, (dict, list)):
            _fail(f"{name}[{index}] must be a non-null scalar JSON value")
        if isinstance(value, float) and not math.isfinite(value):
            _fail(f"{name}[{index}] must be finite")
    fingerprints = [content_fingerprint(value) for value in values]
    if len(fingerprints) != len(set(fingerprints)):
        _fail(f"{name} must not contain duplicate values")
    return values


def _validate_rule_parameters(rule_type: str, parameters: Any, name: str) -> None:
    if not isinstance(parameters, dict):
        _fail(f"{name}.parameters must be an object")
    if rule_type in {"not_null", "finite", "integer", "unique"}:
        _require_exact(f"{name}.parameters", parameters, set())
    elif rule_type == "allowed_values":
        if set(parameters) == {"source"}:
            if parameters["source"] != "valid_domain.values":
                _fail(f"{name}.parameters.source must be valid_domain.values")
        elif set(parameters) == {"values"}:
            _validate_scalar_values(f"{name}.parameters.values", parameters["values"])
        else:
            _fail(f"{name}.parameters must contain exactly source or values")
    elif rule_type == "bounds":
        if set(parameters) == {"source"}:
            if parameters["source"] != "valid_domain":
                _fail(f"{name}.parameters.source must be valid_domain")
        elif not set(parameters).issubset({"minimum", "maximum"}) or not parameters:
            _fail(f"{name}.parameters must contain minimum and/or maximum, or source")
        else:
            for key, value in parameters.items():
                _finite_number(f"{name}.parameters.{key}", value)
            if {"minimum", "maximum"}.issubset(parameters) and parameters["minimum"] > parameters["maximum"]:
                _fail(f"{name}.parameters.minimum must not exceed maximum")
    elif rule_type == "regex":
        _require_exact(f"{name}.parameters", parameters, {"pattern"})
        try:
            re.compile(parameters["pattern"])
        except (TypeError, re.error) as exc:
            _fail(f"{name}.parameters.pattern is invalid: {exc}")
    elif rule_type == "length":
        if not set(parameters).issubset({"minimum", "maximum"}) or not parameters:
            _fail(f"{name}.parameters must contain minimum and/or maximum")
        for key, value in parameters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{name}.parameters.{key} must be a non-negative integer")
        if {"minimum", "maximum"}.issubset(parameters) and parameters["minimum"] > parameters["maximum"]:
            _fail(f"{name}.parameters.minimum must not exceed maximum")
    elif rule_type == "datetime_range":
        if not set(parameters).issubset({"minimum", "maximum"}) or not parameters:
            _fail(f"{name}.parameters must contain minimum and/or maximum")
        for key, value in parameters.items():
            try:
                pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError) as exc:
                _fail(f"{name}.parameters.{key} is not a valid datetime: {exc}")
        if {"minimum", "maximum"}.issubset(parameters) and pd.Timestamp(parameters["minimum"]) > pd.Timestamp(
            parameters["maximum"]
        ):
            _fail(f"{name}.parameters.minimum must not exceed maximum")


def _validate_applicability(name: str, payload: Any, known_columns: set[str]) -> None:
    if not isinstance(payload, dict) or "type" not in payload:
        _fail(f"{name} must declare an applicability type")
    kind = payload["type"]
    if kind in {"all_rows", "all_non_missing"}:
        _require_exact(name, payload, {"type"})
    elif kind == "column_equals":
        _require_exact(name, payload, {"type", "column_id", "value"})
        _validate_scalar_values(f"{name}.value", [payload["value"]])
    elif kind == "column_in":
        _require_exact(name, payload, {"type", "column_id", "values"})
        _validate_scalar_values(f"{name}.values", payload["values"])
    else:
        _fail(f"{name}.type is unsupported: {kind!r}")
    if "column_id" in payload and payload["column_id"] not in known_columns:
        _fail(f"{name}.column_id is unknown: {payload['column_id']!r}")


def _validate_constraint_parameters(constraint_type: str, parameters: Any, columns: list[str], name: str) -> None:
    if not isinstance(parameters, dict):
        _fail(f"{name}.parameters must be an object")
    if constraint_type == "comparison":
        _require_exact(f"{name}.parameters", parameters, {"operator"})
        if len(columns) != 2 or parameters["operator"] not in {"lt", "le", "eq", "ne", "ge", "gt"}:
            _fail(f"{name} comparison requires two columns and a supported operator")
    elif constraint_type == "conditional_domain":
        _require_exact(f"{name}.parameters", parameters, {"if_values", "then_allowed_values"})
        if len(columns) != 2 or not all(isinstance(parameters[key], list) and parameters[key] for key in parameters):
            _fail(f"{name} conditional_domain requires two non-empty value arrays")
        _validate_scalar_values(f"{name}.parameters.if_values", parameters["if_values"])
        _validate_scalar_values(f"{name}.parameters.then_allowed_values", parameters["then_allowed_values"])
    elif constraint_type == "mutual_exclusion":
        _require_exact(f"{name}.parameters", parameters, {"truthy_values", "maximum_true"})
        if len(columns) < 2 or not isinstance(parameters["truthy_values"], list) or not parameters["truthy_values"]:
            _fail(f"{name} mutual_exclusion requires at least two columns and truthy_values")
        _validate_scalar_values(f"{name}.parameters.truthy_values", parameters["truthy_values"])
        maximum = parameters["maximum_true"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum < len(columns):
            _fail(f"{name}.parameters.maximum_true must be in [0, number of columns)")
    elif constraint_type == "sum_equals":
        _require_exact(f"{name}.parameters", parameters, {"absolute_tolerance", "relative_tolerance"})
        if len(columns) < 2:
            _fail(f"{name} sum_equals requires inputs followed by a total column")
        _finite_number(f"{name}.parameters.absolute_tolerance", parameters["absolute_tolerance"], nonnegative=True)
        _finite_number(f"{name}.parameters.relative_tolerance", parameters["relative_tolerance"], nonnegative=True)
    elif constraint_type == "allowed_combinations":
        _require_exact(f"{name}.parameters", parameters, {"allowed"})
        allowed = parameters["allowed"]
        if len(columns) < 2 or not isinstance(allowed, list) or not allowed:
            _fail(f"{name} allowed_combinations requires at least two columns and rows")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in allowed):
            _fail(f"{name}.parameters.allowed rows must match the declared column width")
        for index, row in enumerate(allowed):
            _validate_scalar_values(f"{name}.parameters.allowed[{index}]", row, nonempty=False)
    elif constraint_type == "functional_dependency":
        _require_exact(f"{name}.parameters", parameters, {"mapping"})
        mapping = parameters["mapping"]
        if len(columns) < 2 or not isinstance(mapping, list) or not mapping:
            _fail(f"{name} functional_dependency requires determinant columns, a dependent column, and mapping")
        determinant_width = len(columns) - 1
        for row in mapping:
            if not isinstance(row, dict) or set(row) != {"determinant", "dependent"}:
                _fail(f"{name}.parameters.mapping rows must contain determinant and dependent")
            if not isinstance(row["determinant"], list) or len(row["determinant"]) != determinant_width:
                _fail(f"{name}.parameters.mapping determinant width is invalid")
            _validate_scalar_values(f"{name}.parameters.mapping.determinant", row["determinant"])
            _validate_scalar_values(f"{name}.parameters.mapping.dependent", [row["dependent"]])


def validate_validity_profile(dataset_profile: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the strict machine-readable P3 validity contract."""

    columns = model_view_column_specs(dataset_profile)
    if any(not isinstance(column.get("column_id"), str) for column in columns):
        _fail("Dataset Profile column_id values must be present and unique")
    known_columns: set[str] = {column["column_id"] for column in columns}
    if len(known_columns) != len(columns):
        _fail("Dataset Profile column_id values must be present and unique")

    contract = dataset_profile.get("validity")
    if not isinstance(contract, dict):
        _fail("validity must be an object")
    required = {
        "contract_schema_version",
        "status",
        "hard_column_rules",
        "cross_column_constraints",
        "soft_diagnostics",
        "unresolved_reviews",
    }
    _require_exact("validity", contract, required)
    if contract["contract_schema_version"] != "1.0.0":
        _fail("validity.contract_schema_version must be 1.0.0")
    if contract["status"] not in {"reviewed-diagnostic", "frozen"}:
        _fail("validity.status must be reviewed-diagnostic or frozen")
    _require_string_list("validity.soft_diagnostics", contract["soft_diagnostics"])
    _require_string_list("validity.unresolved_reviews", contract["unresolved_reviews"])

    rules = contract["hard_column_rules"]
    if not isinstance(rules, list) or not rules:
        _fail("validity.hard_column_rules must be a non-empty array")
    seen_rules: set[str] = set()
    for index, rule in enumerate(rules):
        name = f"validity.hard_column_rules[{index}]"
        if not isinstance(rule, dict):
            _fail(f"{name} must be an object")
        _require_exact(
            name,
            rule,
            {"rule_id", "rule_type", "selector", "parameters", "evidence", "severity", "version"},
        )
        rule_id = _require_identifier(f"{name}.rule_id", rule["rule_id"])
        if rule_id in seen_rules:
            _fail(f"Duplicate hard column rule_id: {rule_id}")
        seen_rules.add(rule_id)
        if rule["rule_type"] not in _RULE_TYPES:
            _fail(f"{name}.rule_type is unsupported: {rule['rule_type']!r}")
        if rule["severity"] != "hard" or rule["version"] != "1.0.0":
            _fail(f"{name} must be a hard version 1.0.0 rule")
        _validate_selector(f"{name}.selector", rule["selector"], known_columns)
        _validate_rule_parameters(rule["rule_type"], rule["parameters"], name)
        _validate_evidence(f"{name}.evidence", rule["evidence"])
        matched = [column for column in columns if _selector_matches(rule["selector"], column)]
        if not matched:
            _fail(f"{name}.selector does not match a canonical model-view column")
        compatible_types = {
            "not_null": _SEMANTIC_TYPES,
            "finite": {"continuous", "integer"},
            "integer": {"integer"},
            "allowed_values": {"categorical", "boolean", "string"},
            "bounds": {"continuous", "integer"},
            "regex": {"categorical", "string"},
            "length": {"categorical", "string"},
            "unique": _SEMANTIC_TYPES,
            "datetime_range": {"datetime"},
        }[rule["rule_type"]]
        incompatible = [column["column_id"] for column in matched if column["semantic_type"] not in compatible_types]
        if incompatible:
            _fail(f"{name} is incompatible with matched columns: {incompatible}")
        if rule["parameters"].get("source") == "valid_domain.values":
            missing_domain = [
                column["column_id"]
                for column in matched
                if not isinstance(column.get("valid_domain"), dict)
                or not isinstance(column["valid_domain"].get("values"), list)
                or not column["valid_domain"]["values"]
            ]
            if missing_domain:
                _fail(f"{name} requires non-empty valid_domain.values for columns: {missing_domain}")
            for column in matched:
                _validate_scalar_values(
                    f"{name}.source[{column['column_id']}].values",
                    column["valid_domain"]["values"],
                )
        if rule["parameters"].get("source") == "valid_domain":
            missing_bounds = [
                column["column_id"]
                for column in matched
                if not isinstance(column.get("valid_domain"), dict)
                or not {"minimum", "maximum"}.intersection(column["valid_domain"])
            ]
            if missing_bounds:
                _fail(f"{name} requires valid_domain minimum or maximum for columns: {missing_bounds}")
            for column in matched:
                domain = column["valid_domain"]
                for key in ("minimum", "maximum"):
                    if key in domain:
                        _finite_number(f"{name}.source[{column['column_id']}].{key}", domain[key])
                if {"minimum", "maximum"}.issubset(domain) and domain["minimum"] > domain["maximum"]:
                    _fail(f"{name}.source[{column['column_id']}].minimum must not exceed maximum")

    constraints = contract["cross_column_constraints"]
    if not isinstance(constraints, list):
        _fail("validity.cross_column_constraints must be an array")
    seen_constraints: set[str] = set()
    for index, constraint in enumerate(constraints):
        name = f"validity.cross_column_constraints[{index}]"
        if not isinstance(constraint, dict):
            _fail(f"{name} must be an object")
        _require_exact(
            name,
            constraint,
            {
                "constraint_id",
                "constraint_type",
                "columns",
                "parameters",
                "applicability",
                "missing_behavior",
                "evidence",
                "severity",
                "version",
            },
        )
        constraint_id = _require_identifier(f"{name}.constraint_id", constraint["constraint_id"])
        if constraint_id in seen_constraints:
            _fail(f"Duplicate cross-column constraint_id: {constraint_id}")
        seen_constraints.add(constraint_id)
        if constraint["constraint_type"] not in _CONSTRAINT_TYPES:
            _fail(f"{name}.constraint_type is unsupported: {constraint['constraint_type']!r}")
        selected = _require_string_list(f"{name}.columns", constraint["columns"], nonempty=True)
        unknown = set(selected) - known_columns
        if unknown:
            _fail(f"{name}.columns contains unknown identifiers: {sorted(unknown)}")
        if constraint["missing_behavior"] not in {"violation", "not_applicable"}:
            _fail(f"{name}.missing_behavior must be violation or not_applicable")
        if constraint["severity"] != "hard" or constraint["version"] != "1.0.0":
            _fail(f"{name} must be a hard version 1.0.0 constraint")
        _validate_constraint_parameters(constraint["constraint_type"], constraint["parameters"], selected, name)
        columns_by_id = {column["column_id"]: column for column in columns}
        selected_types = [columns_by_id[column_id]["semantic_type"] for column_id in selected]
        if constraint["constraint_type"] == "sum_equals" and any(
            semantic_type not in {"continuous", "integer"} for semantic_type in selected_types
        ):
            _fail(f"{name} sum_equals columns must all be numerical")
        if constraint["constraint_type"] == "comparison" and constraint["parameters"]["operator"] not in {
            "eq",
            "ne",
        }:
            comparable_families = [
                "numeric"
                if semantic_type in {"continuous", "integer"}
                else "datetime"
                if semantic_type == "datetime"
                else "string"
                for semantic_type in selected_types
            ]
            if len(set(comparable_families)) != 1:
                _fail(f"{name} ordered comparison columns must have compatible semantic types")
        if constraint["constraint_type"] == "allowed_combinations":
            rows = constraint["parameters"]["allowed"]
            fingerprints = [content_fingerprint(row) for row in rows]
            if len(fingerprints) != len(set(fingerprints)):
                _fail(f"{name}.parameters.allowed must not contain duplicate combinations")
        if constraint["constraint_type"] == "functional_dependency":
            determinants = [content_fingerprint(row["determinant"]) for row in constraint["parameters"]["mapping"]]
            if len(determinants) != len(set(determinants)):
                _fail(f"{name}.parameters.mapping must define each determinant exactly once")
        _validate_applicability(f"{name}.applicability", constraint["applicability"], known_columns)
        _validate_evidence(f"{name}.evidence", constraint["evidence"])

    uncovered = [
        column["column_id"]
        for column in columns
        if not any(_selector_matches(rule["selector"], column) for rule in rules)
    ]
    if uncovered:
        _fail(f"Every model-view column requires at least one hard validity rule; uncovered={uncovered}")
    return contract


def _selector_matches(selector: dict[str, Any], column: dict[str, Any]) -> bool:
    if "column_ids" in selector and column["column_id"] not in selector["column_ids"]:
        return False
    if "semantic_types" in selector and column["semantic_type"] not in selector["semantic_types"]:
        return False
    if "nullable_model_input" in selector and column["nullable_model_input"] != selector["nullable_model_input"]:
        return False
    if "roles_any" in selector and not set(column["roles"]).intersection(selector["roles_any"]):
        return False
    domain = column.get("valid_domain")
    if "valid_domain_keys" in selector and (
        not isinstance(domain, dict) or not set(selector["valid_domain_keys"]).issubset(domain)
    ):
        return False
    return True


def _missing_pass(series: pd.Series, predicate: Callable[[Any], bool]) -> pd.Series:
    return series.map(lambda value: True if pd.isna(value) else bool(predicate(value))).astype(bool)


def _rule_mask(series: pd.Series, column: dict[str, Any], rule: dict[str, Any]) -> pd.Series:
    kind = rule["rule_type"]
    parameters = rule["parameters"]
    if kind == "not_null":
        return series.notna()
    if kind == "finite":
        return _missing_pass(series, lambda value: math.isfinite(float(value)))
    if kind == "integer":
        return _missing_pass(
            series,
            lambda value: math.isfinite(float(value)) and -(2**63) <= int(value) <= 2**63 - 1 and int(value) == value,
        )
    if kind == "allowed_values":
        values = parameters.get("values")
        if values is None:
            values = column["valid_domain"]["values"]
        return series.isna() | series.isin(values)
    if kind == "bounds":
        domain = column.get("valid_domain") if parameters.get("source") == "valid_domain" else parameters
        if not isinstance(domain, dict):
            raise ValidityProfileError(f"Column {column['column_id']} lacks the configured bounds source")
        return _missing_pass(
            series,
            lambda value: ("minimum" not in domain or value >= domain["minimum"])
            and ("maximum" not in domain or value <= domain["maximum"]),
        )
    if kind == "regex":
        pattern = re.compile(parameters["pattern"])
        return _missing_pass(series, lambda value: pattern.fullmatch(str(value)) is not None)
    if kind == "length":
        return _missing_pass(
            series,
            lambda value: ("minimum" not in parameters or len(str(value)) >= parameters["minimum"])
            and ("maximum" not in parameters or len(str(value)) <= parameters["maximum"]),
        )
    if kind == "unique":
        return series.isna() | ~series.duplicated(keep=False)
    if kind == "datetime_range":

        def in_range(value: Any) -> bool:
            timestamp = pd.Timestamp(value)
            return ("minimum" not in parameters or timestamp >= pd.Timestamp(parameters["minimum"])) and (
                "maximum" not in parameters or timestamp <= pd.Timestamp(parameters["maximum"])
            )

        return _missing_pass(series, in_range)
    raise ValidityProfileError(f"Unsupported rule type at execution: {kind}")


def _applicability_mask(
    frame: pd.DataFrame,
    definition: dict[str, Any],
    names_by_id: dict[str, str],
) -> pd.Series:
    payload = definition["applicability"]
    kind = payload["type"]
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if kind == "column_equals":
        mask &= frame[names_by_id[payload["column_id"]]] == payload["value"]
    elif kind == "column_in":
        mask &= frame[names_by_id[payload["column_id"]]].isin(payload["values"])
    relevant = [names_by_id[column_id] for column_id in definition["columns"]]
    complete = frame[relevant].notna().all(axis=1)
    if kind == "all_non_missing" or definition["missing_behavior"] == "not_applicable":
        mask &= complete
    return mask.fillna(False).astype(bool)


def _constraint_satisfaction(
    frame: pd.DataFrame,
    definition: dict[str, Any],
    names_by_id: dict[str, str],
) -> pd.Series:
    names = [names_by_id[column_id] for column_id in definition["columns"]]
    parameters = definition["parameters"]
    kind = definition["constraint_type"]
    if kind == "comparison":
        left, right = frame[names[0]], frame[names[1]]
        operators = {
            "lt": left < right,
            "le": left <= right,
            "eq": left == right,
            "ne": left != right,
            "ge": left >= right,
            "gt": left > right,
        }
        result = operators[parameters["operator"]]
    elif kind == "conditional_domain":
        condition = frame[names[0]].isin(parameters["if_values"])
        result = ~condition | frame[names[1]].isin(parameters["then_allowed_values"])
    elif kind == "mutual_exclusion":
        truthy = frame[names].isin(parameters["truthy_values"]).sum(axis=1)
        result = truthy <= parameters["maximum_true"]
    elif kind == "sum_equals":
        observed_sum = frame[names[:-1]].sum(axis=1, skipna=False)
        expected_total = frame[names[-1]]
        result = pd.Series(
            [
                math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=parameters["relative_tolerance"],
                    abs_tol=parameters["absolute_tolerance"],
                )
                if not pd.isna(observed) and not pd.isna(expected)
                else False
                for observed, expected in zip(observed_sum, expected_total, strict=True)
            ],
            index=frame.index,
        )
    elif kind == "allowed_combinations":
        allowed = {tuple(row) for row in parameters["allowed"]}
        result = frame[names].apply(lambda row: tuple(row.tolist()) in allowed, axis=1)
    elif kind == "functional_dependency":
        mapping = {tuple(row["determinant"]): row["dependent"] for row in parameters["mapping"]}
        result = frame[names].apply(
            lambda row: mapping.get(tuple(row.iloc[:-1].tolist()), object()) == row.iloc[-1],
            axis=1,
        )
    else:  # pragma: no cover - guarded by contract validation
        raise ValidityProfileError(f"Unsupported constraint type at execution: {kind}")
    complete = frame[names].notna().all(axis=1)
    if definition["missing_behavior"] == "violation":
        result &= complete
    return result.fillna(False).astype(bool)


def _identity(request: EvaluationRequest, profile: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "protocol_version": request.protocol["protocol_version"],
        "dataset_id": profile["dataset_id"],
        "dataset_version": profile["dataset_version"],
        "dataset_view": profile["dataset_view"],
        "split_id": profile.get("split", {}).get("split_id", "external-evaluation"),
        "model_id": (request.model or {}).get("model_id", "external"),
        "comparison_track": request.comparison_track,
        "generation_seed": request.generation_seed,
    }


def _atomic(
    *,
    identity: dict[str, Any],
    metric_id: str,
    scope_type: str,
    scope_id: str,
    state: MetricState,
    score: float | None,
    weight: float,
    n_reference: int,
    n_synthetic: int,
    n_valid: int,
    n_excluded: int,
    computed_at: str,
    reason_code: str | None = None,
    reason_detail: str | None = None,
) -> AtomicResult:
    return AtomicResult(
        **identity,
        metric_id=metric_id,
        metric_version=P3_METRIC_VERSION,
        dimension="validity",
        scope_type=scope_type,
        scope_id=scope_id,
        state=state,
        raw_direction=RawDirection.MAXIMIZE,
        weight=weight,
        n_reference=n_reference,
        n_synthetic=n_synthetic,
        n_valid=n_valid,
        n_excluded=n_excluded,
        computed_at=computed_at,
        raw_value=score if state is MetricState.COMPUTED else None,
        normalized_value=score if state is MetricState.COMPUTED else None,
        aggregate_contribution=score * weight if score is not None and state is MetricState.COMPUTED else None,
        unit="satisfaction-rate",
        evaluator_id="std-tabular-validity",
        evaluator_version=VALIDITY_IMPLEMENTATION_VERSION,
        reason_code=reason_code,
        reason_detail=reason_detail,
        artifact_ref=VALIDITY_DETAILS_ARTIFACT_PATH,
    )


def evaluate_validity(
    request: EvaluationRequest,
    dataset_profile: dict[str, Any],
    tables: ValidatedTables,
    *,
    run_id: str,
) -> ValidityOutcome:
    """Evaluate the immutable decoded synthetic table without repair or learned rules."""

    requested = {(item["metric_id"], item["metric_version"]) for item in request.metrics}
    supported = {(item["metric_id"], item["metric_version"]) for item in P3_METRICS}
    if requested != supported:
        raise ValidityError(f"P3 requires exactly the two validity metric identities, got {sorted(requested)}")
    if len(request.evaluator_seeds) != 1:
        raise ValidityError("P3 requires exactly one recorded evaluator seed even though its rules are deterministic")
    contract = validate_validity_profile(dataset_profile)
    synthetic = tables.synthetic
    model_columns = model_view_column_specs(dataset_profile)
    names_by_id = {column["column_id"]: column["name"] for column in model_columns}
    identity = _identity(request, dataset_profile, run_id)
    computed_at = utc_timestamp()

    atoms: list[AtomicResult] = []
    column_details: list[dict[str, Any]] = []
    fully_valid = pd.Series(True, index=synthetic.index, dtype=bool)
    column_weight = 1.0 / len(model_columns)
    for column in model_columns:
        series = synthetic[column["name"]]
        combined = pd.Series(True, index=synthetic.index, dtype=bool)
        rule_details: list[dict[str, Any]] = []
        for rule in contract["hard_column_rules"]:
            if not _selector_matches(rule["selector"], column):
                continue
            mask = _rule_mask(series, column, rule)
            violations = int((~mask).sum())
            combined &= mask
            rule_details.append(
                {
                    "rule_id": rule["rule_id"],
                    "rule_type": rule["rule_type"],
                    "valid_cells": int(mask.sum()),
                    "invalid_cells": violations,
                    "evidence": rule["evidence"],
                    "version": rule["version"],
                }
            )
        valid_cells = int(combined.sum())
        invalid_cells = len(synthetic) - valid_cells
        column_rate = valid_cells / len(synthetic)
        fully_valid &= combined
        atoms.append(
            _atomic(
                identity=identity,
                metric_id=COLUMN_VALIDITY_METRIC_ID,
                scope_type="column",
                scope_id=column["column_id"],
                state=MetricState.COMPUTED,
                score=column_rate,
                weight=column_weight,
                n_reference=len(tables.reference),
                n_synthetic=len(synthetic),
                n_valid=len(synthetic),
                n_excluded=0,
                computed_at=computed_at,
            )
        )
        column_details.append(
            {
                "column_id": column["column_id"],
                "column_name": column["name"],
                "valid_cell_rate": column_rate,
                "valid_cells": valid_cells,
                "invalid_cells": invalid_cells,
                "rules": rule_details,
            }
        )

    constraint_evaluations: list[_ConstraintEvaluation] = []
    for definition in contract["cross_column_constraints"]:
        applicable = _applicability_mask(synthetic, definition, names_by_id)
        satisfied = _constraint_satisfaction(synthetic, definition, names_by_id)
        applicable_count = int(applicable.sum())
        constraint_rate = float(satisfied[applicable].mean()) if applicable_count else None
        fully_valid &= ~applicable | satisfied
        constraint_evaluations.append(
            _ConstraintEvaluation(
                definition=definition,
                applicable=applicable,
                satisfied=satisfied,
                score=constraint_rate,
            )
        )

    computed_constraints = sum(item.score is not None for item in constraint_evaluations)
    constraint_weight = 1.0 / computed_constraints if computed_constraints else 0.0
    constraint_details: list[dict[str, Any]] = []
    for item in constraint_evaluations:
        definition = item.definition
        applicable_count = int(item.applicable.sum())
        satisfied_count = int((item.applicable & item.satisfied).sum())
        if item.score is None:
            state = MetricState.NOT_APPLICABLE
            reason_code = "no_applicable_rows"
            reason_detail = "No synthetic rows satisfy the reviewed constraint applicability rule."
            weight = 0.0
        else:
            state = MetricState.COMPUTED
            reason_code = None
            reason_detail = None
            weight = constraint_weight
        atoms.append(
            _atomic(
                identity=identity,
                metric_id=CONSTRAINT_VALIDITY_METRIC_ID,
                scope_type="dataset",
                scope_id=definition["constraint_id"],
                state=state,
                score=item.score,
                weight=weight,
                n_reference=len(tables.reference),
                n_synthetic=len(synthetic),
                n_valid=applicable_count,
                n_excluded=len(synthetic) - applicable_count,
                computed_at=computed_at,
                reason_code=reason_code,
                reason_detail=reason_detail,
            )
        )
        constraint_details.append(
            {
                "constraint_id": definition["constraint_id"],
                "constraint_type": definition["constraint_type"],
                "applicable_rows": applicable_count,
                "satisfied_rows": satisfied_count,
                "violating_rows": applicable_count - satisfied_count,
                "satisfaction_rate": item.score,
                "evidence": definition["evidence"],
                "version": definition["version"],
            }
        )

    if not constraint_evaluations:
        atoms.append(
            _atomic(
                identity=identity,
                metric_id=CONSTRAINT_VALIDITY_METRIC_ID,
                scope_type="dataset",
                scope_id="no-reviewed-constraints",
                state=MetricState.NOT_APPLICABLE,
                score=None,
                weight=0.0,
                n_reference=len(tables.reference),
                n_synthetic=len(synthetic),
                n_valid=0,
                n_excluded=len(synthetic),
                computed_at=computed_at,
                reason_code="no_reviewed_constraints",
                reason_detail="The Dataset Profile declares no reviewed hard cross-column constraints.",
            )
        )

    column_score = sum(
        atom.aggregate_contribution or 0.0 for atom in atoms if atom.metric_id == COLUMN_VALIDITY_METRIC_ID
    )
    constraint_score = (
        sum(atom.aggregate_contribution or 0.0 for atom in atoms if atom.metric_id == CONSTRAINT_VALIDITY_METRIC_ID)
        if computed_constraints
        else None
    )
    validity_score = 0.5 * column_score + 0.5 * constraint_score if constraint_score is not None else column_score
    fully_valid_rows = int(fully_valid.sum())
    source = {
        "definition_origin": "benchmark-native",
        "implementation": "standardized_tabular_diffusion.evaluation.validity.evaluate_validity",
        "implementation_version": VALIDITY_IMPLEMENTATION_VERSION,
        "dataset_validity_contract_sha256": content_fingerprint(contract),
        "learned_from_data": False,
        "synthetic_repair_applied": False,
    }
    details = {
        "details_schema_version": "1.0.0",
        "source": source,
        "input_view": "original-decoded-synthetic-output",
        "input_mutated": False,
        "synthetic_repair_applied": False,
        "rows": len(synthetic),
        "columns": column_details,
        "cross_column_constraints": constraint_details,
        "fully_valid_rows": fully_valid_rows,
        "fully_valid_row_rate": fully_valid_rows / len(synthetic),
        "fully_valid_row_rate_aggregation_role": "reported-only-width-sensitive",
        "property_scores": {
            "column_validity_score": column_score,
            "constraint_validity_score": constraint_score,
            "validity_score": validity_score,
        },
    }
    return ValidityOutcome(
        atomic_results=tuple(atoms),
        property_scores={
            "column_validity_score": column_score,
            "constraint_validity_score": constraint_score,
            "validity_score": validity_score,
            "fully_valid_row_rate": fully_valid_rows / len(synthetic),
        },
        denominator_counts={
            "requested_columns": len(model_columns),
            "evaluated_columns": len(model_columns),
            "reviewed_cross_column_constraints": len(constraint_evaluations),
            "computed_cross_column_constraints": computed_constraints,
            "synthetic_rows": len(synthetic),
            "fully_valid_rows": fully_valid_rows,
        },
        details=details,
        source=source,
    )
