"""Checksum-locked acquisition of non-redistributable upstream model sources."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, atomic_write_json

_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources" / "upstream"
_MANIFESTS = {
    "ctab-gan": _RESOURCE_ROOT / "ctabgan-source-manifest.json",
    "ctab-gan-plus": _RESOURCE_ROOT / "ctabgan-plus-source-manifest.json",
}
_INSTALL_RECORD = ".standardized-source.json"
_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


class UpstreamSourceError(RuntimeError):
    """Base error for locked upstream source operations."""


class UpstreamSourceIntegrityError(UpstreamSourceError):
    """Raised when source bytes or paths differ from the immutable lock."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_source_bytes(payload: bytes, manifest: dict[str, Any]) -> bytes:
    normalization = manifest.get("source_hash_normalization")
    if normalization is None:
        return payload
    if normalization != "lf-one-final-newline":
        raise UpstreamSourceIntegrityError(f"Unsupported source hash normalization: {normalization!r}")
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"


def load_source_manifest(model_id: str) -> dict[str, Any]:
    try:
        path = _MANIFESTS[model_id]
    except KeyError as exc:
        raise UpstreamSourceError(f"No locked upstream source is registered for {model_id!r}") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_id") != model_id or payload.get("manifest_schema_version") != "1.0.0":
        raise UpstreamSourceIntegrityError(f"Malformed source manifest for {model_id!r}")
    return payload


def source_manifest_path(model_id: str) -> Path:
    load_source_manifest(model_id)
    return _MANIFESTS[model_id]


def default_source_path(repo_root: str | Path, model_id: str) -> Path:
    manifest = load_source_manifest(model_id)
    return Path(repo_root).resolve() / ".cache" / "upstream-sources" / model_id / manifest["upstream_commit"]


def _validated_runtime_files(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise UpstreamSourceIntegrityError(f"Upstream source root must be a regular directory: {root}")
    resolved_root = root.resolve(strict=True)
    verified: list[dict[str, Any]] = []
    for record in manifest["runtime_files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"]:
            raise UpstreamSourceIntegrityError(f"Unsafe locked source path: {record['path']!r}")
        path = resolved_root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise UpstreamSourceIntegrityError(f"Required official source file is missing: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise UpstreamSourceIntegrityError(f"Official source file escapes its root: {path}")
        payload = _normalized_source_bytes(path.read_bytes(), manifest)
        observed_bytes = len(payload)
        observed_sha256 = _sha256_bytes(payload)
        if observed_bytes != record["bytes"] or observed_sha256 != record["sha256"]:
            raise UpstreamSourceIntegrityError(
                f"Official source mismatch for {record['path']}: "
                f"expected sha256={record['sha256']} bytes={record['bytes']}, "
                f"observed sha256={observed_sha256} bytes={observed_bytes}"
            )
        verified.append({"path": record["path"], "bytes": observed_bytes, "sha256": observed_sha256})
    return verified


def validate_upstream_source(model_id: str, source_dir: str | Path) -> dict[str, Any]:
    manifest = load_source_manifest(model_id)
    root = Path(source_dir)
    files = _validated_runtime_files(root, manifest)
    return {
        "model_id": model_id,
        "source_dir": str(root.resolve()),
        "repository": manifest["repository"],
        "upstream_commit": manifest["upstream_commit"],
        "upstream_tree": manifest["upstream_tree"],
        "upstream_model_tree": manifest["upstream_model_tree"],
        "runtime_files_verified": len(files),
        "runtime_files": files,
        "license": manifest["license"],
        "manifest_sha256": _sha256_file(source_manifest_path(model_id)),
    }


def source_status(model_id: str, *, repo_root: str | Path, source_dir: str | Path | None = None) -> dict[str, Any]:
    destination = Path(source_dir) if source_dir is not None else default_source_path(repo_root, model_id)
    if not destination.exists():
        manifest = load_source_manifest(model_id)
        return {
            "model_id": model_id,
            "status": "missing",
            "source_dir": str(destination.resolve()),
            "upstream_commit": manifest["upstream_commit"],
        }
    try:
        verified = validate_upstream_source(model_id, destination)
    except UpstreamSourceIntegrityError as exc:
        return {"model_id": model_id, "status": "invalid", "source_dir": str(destination.resolve()), "error": str(exc)}
    return {"status": "ready", **verified}


def _download_archive(manifest: dict[str, Any], timeout_seconds: float) -> bytes:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = urllib.request.Request(
        manifest["archive"]["url"],
        headers={"User-Agent": "standardized-tabular-diffusion-source-materializer/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > _MAX_ARCHIVE_BYTES:
            raise UpstreamSourceIntegrityError(f"Upstream source archive exceeds {_MAX_ARCHIVE_BYTES} bytes")
        payload = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise UpstreamSourceIntegrityError(f"Upstream source archive exceeds {_MAX_ARCHIVE_BYTES} bytes")
    archive = manifest["archive"]
    if len(payload) != archive["bytes"] or _sha256_bytes(payload) != archive["sha256"]:
        raise UpstreamSourceIntegrityError("Downloaded upstream source archive does not match the immutable lock")
    return payload


def _extract_runtime_files(payload: bytes, staging: Path, manifest: dict[str, Any]) -> None:
    root_prefix = manifest["archive"]["root_prefix"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise UpstreamSourceIntegrityError("Upstream source archive contains duplicate paths")
        for record in manifest["runtime_files"]:
            member_name = root_prefix + record["path"]
            try:
                info = archive.getinfo(member_name)
            except KeyError as exc:
                raise UpstreamSourceIntegrityError(f"Locked source archive is missing {record['path']}") from exc
            member_path = PurePosixPath(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts or "\\" in info.filename:
                raise UpstreamSourceIntegrityError(f"Unsafe upstream archive path: {info.filename!r}")
            if info.is_dir():
                raise UpstreamSourceIntegrityError(f"Unexpected archive member metadata for {record['path']}")
            data = _normalized_source_bytes(archive.read(info), manifest)
            if len(data) != record["bytes"] or _sha256_bytes(data) != record["sha256"]:
                raise UpstreamSourceIntegrityError(f"Archive member checksum mismatch for {record['path']}")
            output = staging.joinpath(*PurePosixPath(record["path"]).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(output, data)


def materialize_upstream_source(
    model_id: str,
    *,
    repo_root: str | Path,
    destination: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    manifest = load_source_manifest(model_id)
    target = Path(destination) if destination is not None else default_source_path(repo_root, model_id)
    target = target.resolve()
    if target.is_symlink():
        raise UpstreamSourceIntegrityError(f"Refusing a symlinked upstream source destination: {target}")
    if target.exists() and not refresh:
        return {"status": "ready", "cached": True, **validate_upstream_source(model_id, target)}

    payload = _download_archive(manifest, timeout_seconds)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{model_id}-", dir=target.parent) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        _extract_runtime_files(payload, staging, manifest)
        atomic_write_json(
            staging / _INSTALL_RECORD,
            {
                "model_id": model_id,
                "repository": manifest["repository"],
                "upstream_commit": manifest["upstream_commit"],
                "archive_sha256": manifest["archive"]["sha256"],
                "manifest_sha256": _sha256_file(source_manifest_path(model_id)),
                "license": manifest["license"],
            },
        )
        validate_upstream_source(model_id, staging)
        backup: Path | None = None
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{os.getpid()}"
            if backup.exists():
                raise UpstreamSourceError(f"Refusing to overwrite an existing source backup: {backup}")
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup is not None and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)

    return {"status": "ready", "cached": False, **validate_upstream_source(model_id, target)}
