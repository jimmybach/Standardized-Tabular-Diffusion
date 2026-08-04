"""JSON Schema discovery and validation for evaluation wire contracts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import SerializationError, canonical_json_bytes, read_json

SCHEMA_PACKAGE = "standardized_tabular_diffusion.schemas.evaluation"

SCHEMA_FILES = {
    "artifact-index": "artifact-index.schema.json",
    "atomic-result": "atomic-result.schema.json",
    "dataset-profile": "dataset-profile.schema.json",
    "evaluation-request": "evaluation-request.schema.json",
    "manifest": "manifest.schema.json",
    "metadata": "metadata.schema.json",
    "metric-registry-entry": "metric-registry-entry.schema.json",
    "protocol-profile": "protocol-profile.schema.json",
    "stage-record": "stage-record.schema.json",
    "summary": "summary.schema.json",
}


@dataclass(frozen=True)
class SchemaViolation:
    path: str
    message: str


class SchemaValidationError(ValueError):
    def __init__(self, schema_name: str, violations: list[SchemaViolation]) -> None:
        self.schema_name = schema_name
        self.violations = violations
        details = "; ".join(f"{item.path}: {item.message}" for item in violations)
        super().__init__(f"{schema_name} schema validation failed: {details}")


def list_schemas() -> tuple[str, ...]:
    return tuple(sorted(SCHEMA_FILES))


def load_schema(schema_name: str) -> dict[str, Any]:
    try:
        filename = SCHEMA_FILES[schema_name]
    except KeyError as exc:
        raise KeyError(f"Unknown evaluation schema: {schema_name}") from exc
    resource = resources.files(SCHEMA_PACKAGE).joinpath(filename)
    with resources.as_file(resource) as path:
        payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Packaged schema {schema_name} is not a JSON object")
    return payload


def validate_instance(schema_name: str, instance: Any) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - minimal installation behavior
        raise RuntimeError("Schema validation requires the optional 'contracts' dependency") from exc

    try:
        canonical_json_bytes(instance)
    except SerializationError as exc:
        raise SchemaValidationError(schema_name, [SchemaViolation("$", str(exc))]) from exc

    schema = load_schema(schema_name)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    violations = [
        SchemaViolation(
            path="$" + "".join(f"[{item}]" if isinstance(item, int) else f".{item}" for item in error.absolute_path),
            message=error.message,
        )
        for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
    ]
    if violations:
        raise SchemaValidationError(schema_name, violations)


def validate_file(schema_name: str, path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SchemaValidationError(schema_name, [SchemaViolation("$", "document must be a JSON object")])
    validate_instance(schema_name, payload)
    return payload
