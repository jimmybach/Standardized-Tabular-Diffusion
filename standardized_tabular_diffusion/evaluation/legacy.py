"""Read-only import boundary for frozen pre-P2 evaluation summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.schema import validate_file, validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    SerializationError,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)

LEGACY_SOURCE_PATH = "source/standardized_summary.json"
LEGACY_RECORD_PATH = "legacy_import_record.json"
LEGACY_CHECKSUM_PATH = "checksums.sha256"
LEGACY_MIGRATION_WINDOW = {"supported_release_line": "0.1.x", "removal_not_before": "0.2.0"}


class LegacyImportError(ValueError):
    """Raised when legacy evidence cannot be imported without ambiguity."""


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise LegacyImportError(f"{label} must be a regular, non-symlinked file: {path}")


def _record(summary: dict[str, Any], source_bytes: bytes) -> dict[str, Any]:
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    record: dict[str, Any] = {
        "legacy_import_schema_version": "1.0.0",
        "bundle_type": "legacy-summary-import",
        "classification": "legacy-diagnostic",
        "source": {
            "format": "standardized_summary.json",
            "schema_version": "1.0",
            "protocol_name": "tabstruct-aligned-v1",
            "sha256": source_sha256,
            "byte_size": len(source_bytes),
        },
        "identity": {"model": summary["model"], "dataset": summary["dataset"]},
        "conversion": {
            "status": "not-converted",
            "atomic_evidence_available": False,
            "official_results_allowed": False,
            "lossy_fields": [],
            "reason_code": "legacy-summary-has-no-atomic-evidence",
        },
        "migration_window": dict(LEGACY_MIGRATION_WINDOW),
        "files": [
            {"path": LEGACY_SOURCE_PATH, "sha256": source_sha256, "byte_size": len(source_bytes)},
        ],
    }
    return record


def import_legacy_summary(source: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Preserve a frozen legacy source verbatim without fabricating Atomic Results."""

    source_path = Path(source)
    output = Path(output_dir)
    _regular_file(source_path, "Legacy summary")
    if output.exists():
        raise LegacyImportError(f"Refusing to overwrite an existing legacy import path: {output}")
    summary = validate_file("legacy-standardized-summary", source_path)
    source_bytes = source_path.read_bytes()
    output.mkdir(parents=True)
    (output / "source").mkdir()
    atomic_write_bytes(output / LEGACY_SOURCE_PATH, source_bytes)
    record = _record(summary, source_bytes)
    validate_instance("legacy-import-record", record)
    atomic_write_json(output / LEGACY_RECORD_PATH, record)
    checksums = "".join(
        f"{sha256_file(output / relative)}  {relative}\n"
        for relative in (LEGACY_RECORD_PATH, LEGACY_SOURCE_PATH)
    ).encode("utf-8")
    atomic_write_bytes(output / LEGACY_CHECKSUM_PATH, checksums)
    return validate_legacy_import(output)


def validate_legacy_import(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise LegacyImportError(f"Legacy import must be a regular directory: {root}")
    observed = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    expected = [LEGACY_CHECKSUM_PATH, LEGACY_RECORD_PATH, LEGACY_SOURCE_PATH]
    if observed != expected:
        raise LegacyImportError(f"Legacy import file set differs from the frozen contract: {observed}")
    for relative in expected:
        _regular_file(root / relative, relative)
    record = validate_file("legacy-import-record", root / LEGACY_RECORD_PATH)
    summary = validate_file("legacy-standardized-summary", root / LEGACY_SOURCE_PATH)
    if sha256_file(root / LEGACY_SOURCE_PATH) != record["source"]["sha256"]:
        raise LegacyImportError("Preserved source checksum differs from the import record")
    if (root / LEGACY_SOURCE_PATH).stat().st_size != record["source"]["byte_size"]:
        raise LegacyImportError("Preserved source byte size differs from the import record")
    if record["files"] != [
        {
            "path": LEGACY_SOURCE_PATH,
            "sha256": record["source"]["sha256"],
            "byte_size": record["source"]["byte_size"],
        }
    ]:
        raise LegacyImportError("Legacy import file inventory differs from the frozen contract")
    if record["identity"] != {"model": summary["model"], "dataset": summary["dataset"]}:
        raise LegacyImportError("Legacy import identity differs from the preserved source")
    expected_checksums = "".join(
        f"{sha256_file(root / relative)}  {relative}\n"
        for relative in (LEGACY_RECORD_PATH, LEGACY_SOURCE_PATH)
    )
    try:
        observed_checksums = (root / LEGACY_CHECKSUM_PATH).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SerializationError(f"Cannot read legacy checksum manifest: {exc}") from exc
    if observed_checksums != expected_checksums:
        raise LegacyImportError("Legacy import checksum manifest is invalid")
    return {
        "valid": True,
        "bundle_type": "legacy-summary-import",
        "classification": "legacy-diagnostic",
        "model": summary["model"],
        "dataset": summary["dataset"],
        "official_results_allowed": False,
        "atomic_evidence_available": False,
        "source_sha256": record["source"]["sha256"],
    }
