"""Deterministic, safe serialization primitives for evaluation contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any


class SerializationError(ValueError):
    """Raised when a value cannot be represented by the wire contract."""


def _canonicalize(value: Any, location: str = "$") -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError(f"Non-finite number at {location}")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(f"Object key at {location} is not a string: {key!r}")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise SerializationError(f"Object at {location} has duplicate keys after Unicode normalization")
            normalized[normalized_key] = _canonicalize(item, f"{location}.{normalized_key}")
        return normalized
    raise SerializationError(f"Unsupported value at {location}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return NFC-normalized RFC 8259 JSON with stable key ordering."""

    normalized = _canonicalize(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def content_fingerprint(value: Any) -> str:
    """Hash a scientific identity payload using canonical JSON and SHA-256."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_bundle_relative_path(value: str) -> str:
    """Validate a portable POSIX path that cannot escape a result bundle."""

    if not isinstance(value, str) or not value:
        raise SerializationError("Bundle-relative path must be a non-empty string")
    if value != unicodedata.normalize("NFC", value):
        raise SerializationError("Bundle-relative path must use NFC Unicode normalization")
    if "\\" in value:
        raise SerializationError(f"Bundle-relative path must use POSIX separators: {value!r}")
    if re.match(r"^[A-Za-z]:/", value) or any(
        any(character in '<>:"|?*' for character in part) for part in value.split("/")
    ):
        raise SerializationError(f"Bundle-relative path is not portable: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.drive or value.startswith("/"):
        raise SerializationError(f"Absolute bundle path is prohibited: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SerializationError(f"Unsafe bundle-relative path: {value!r}")
    if str(path) != value:
        raise SerializationError(f"Non-canonical bundle-relative path: {value!r}")
    return value


def read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SerializationError(f"Cannot read valid UTF-8 JSON from {source}: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise SerializationError(f"Non-finite JSON number is prohibited: {value}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting literal and NFC-equivalent keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        normalized_key = unicodedata.normalize("NFC", key)
        if normalized_key in result:
            raise SerializationError(f"Duplicate JSON object key after Unicode normalization: {normalized_key!r}")
        result[normalized_key] = value
    return result


def parse_json_text(text: str, *, source: str = "JSON text") -> Any:
    """Parse strict RFC 8259 JSON while rejecting duplicate and non-finite values."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Cannot read valid JSON from {source}: {exc}") from exc


def read_yaml_safe(path: str | Path) -> Any:
    """Load JSON or YAML without permitting Python/object constructors."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SerializationError(f"Cannot read UTF-8 configuration from {source}: {exc}") from exc

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised in minimal installations
        raise SerializationError(
            "YAML input requires the optional 'contracts' dependency; JSON remains supported without PyYAML"
        ) from exc
    class UniqueKeySafeLoader(yaml.SafeLoader):
        """Safe YAML loader whose mappings have JSON-compatible unique string keys."""

    def construct_unique_mapping(loader: UniqueKeySafeLoader, node: Any, deep: bool = False) -> dict[str, Any]:
        pairs = loader.construct_pairs(node, deep=deep)
        result: dict[str, Any] = {}
        for key, item in pairs:
            if not isinstance(key, str):
                raise SerializationError(f"YAML object key is not a string: {key!r}")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in result:
                raise SerializationError(
                    f"Duplicate YAML object key after Unicode normalization: {normalized_key!r}"
                )
            result[normalized_key] = item
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    try:
        value = yaml.load(text, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise SerializationError(f"Cannot safely load YAML from {source}: {exc}") from exc
    _canonicalize(value)
    return value


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Replace one regular file atomically, never exposing a partial write."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_symlink():
        raise SerializationError(f"Refusing to replace symlink: {destination}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        if os.name == "posix":  # fsync the directory entry on the official Linux platform
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    normalized = _canonicalize(value)
    if pretty:
        payload = (json.dumps(normalized, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    else:
        payload = canonical_json_bytes(normalized) + b"\n"
    atomic_write_bytes(path, payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
