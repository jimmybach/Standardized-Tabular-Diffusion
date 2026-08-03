"""Authenticated, checksum-pinned retrieval of public benchmark datasets."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from standardized_tabular_diffusion.datasets import validate_dataset_name
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_READ_CHUNK_SIZE = 1024 * 1024


class DatasetSourceError(ValueError):
    """Base class for source registry and retrieval failures."""


class DownloadIntegrityError(DatasetSourceError):
    """Raised when downloaded or cached bytes do not match the source lock."""


class UnsafeArchiveError(DatasetSourceError):
    """Raised when an archive member could escape the extraction boundary."""


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    dataset_view: str
    source_version: str
    publisher: str
    canonical_page: str
    retrieval_url: str
    retrieved_date: str
    archive_name: str
    archive_format: str
    sha256: str
    required_members: tuple[str, ...]
    max_download_bytes: int
    max_extracted_bytes: int
    license: str
    license_url: str
    citation: str
    redistribution_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _registry_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "datasets" / "sources.json"


def _validate_https_url(value: str, field: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise DatasetSourceError(f"{field} must be an HTTPS URL without credentials or fragments: {value!r}")
    return value


def _validate_member_name(value: str) -> str:
    if not value or "\\" in value:
        raise DatasetSourceError(f"Archive member must be a non-empty POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or str(path) != value:
        raise DatasetSourceError(f"Unsafe archive member in source registry: {value!r}")
    if any(":" in part for part in path.parts):
        raise DatasetSourceError(f"Non-portable archive member in source registry: {value!r}")
    return value


def _parse_source(dataset_id: str, payload: Any) -> DatasetSource:
    validate_dataset_name(dataset_id)
    if not isinstance(payload, dict):
        raise DatasetSourceError(f"Dataset source {dataset_id!r} must be a JSON object")
    required = {
        "dataset_view",
        "source_version",
        "publisher",
        "canonical_page",
        "retrieval_url",
        "retrieved_date",
        "archive_name",
        "archive_format",
        "sha256",
        "required_members",
        "max_download_bytes",
        "max_extracted_bytes",
        "license",
        "license_url",
        "citation",
        "redistribution_status",
    }
    if set(payload) != required:
        raise DatasetSourceError(
            f"Dataset source {dataset_id!r} fields differ from the v1 contract; "
            f"missing={sorted(required - set(payload))}, unknown={sorted(set(payload) - required)}"
        )
    if payload["archive_format"] != "zip":
        raise DatasetSourceError(f"Only ZIP sources are supported in v1: {dataset_id}")
    archive_name = payload["archive_name"]
    if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
        raise DatasetSourceError(f"archive_name must be a basename: {archive_name!r}")
    sha256 = payload["sha256"]
    if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
        raise DatasetSourceError(f"Invalid SHA-256 lock for {dataset_id}")
    members = payload["required_members"]
    if not isinstance(members, list) or not members or not all(isinstance(value, str) for value in members):
        raise DatasetSourceError(f"required_members must be a non-empty string list: {dataset_id}")
    if len(members) != len(set(members)):
        raise DatasetSourceError(f"required_members contains duplicates: {dataset_id}")
    for size_field in ("max_download_bytes", "max_extracted_bytes"):
        if (
            not isinstance(payload[size_field], int)
            or isinstance(payload[size_field], bool)
            or payload[size_field] <= 0
        ):
            raise DatasetSourceError(f"{size_field} must be a positive integer: {dataset_id}")
    if payload["redistribution_status"] not in {"permitted", "download-script-only", "metadata-only"}:
        raise DatasetSourceError(f"Unsupported public retrieval status for {dataset_id}")

    return DatasetSource(
        dataset_id=dataset_id,
        dataset_view=payload["dataset_view"],
        source_version=payload["source_version"],
        publisher=payload["publisher"],
        canonical_page=_validate_https_url(payload["canonical_page"], "canonical_page"),
        retrieval_url=_validate_https_url(payload["retrieval_url"], "retrieval_url"),
        retrieved_date=payload["retrieved_date"],
        archive_name=archive_name,
        archive_format=payload["archive_format"],
        sha256=sha256,
        required_members=tuple(_validate_member_name(value) for value in members),
        max_download_bytes=payload["max_download_bytes"],
        max_extracted_bytes=payload["max_extracted_bytes"],
        license=payload["license"],
        license_url=_validate_https_url(payload["license_url"], "license_url"),
        citation=payload["citation"],
        redistribution_status=payload["redistribution_status"],
    )


def load_dataset_source_registry(path: str | Path | None = None) -> dict[str, DatasetSource]:
    payload = read_json(path or _registry_path())
    if not isinstance(payload, dict) or set(payload) != {"registry_schema_version", "sources"}:
        raise DatasetSourceError("Dataset source registry must contain registry_schema_version and sources")
    if payload["registry_schema_version"] != "1.0.0" or not isinstance(payload["sources"], dict):
        raise DatasetSourceError("Unsupported dataset source registry schema")
    return {
        dataset_id: _parse_source(dataset_id, source_payload)
        for dataset_id, source_payload in sorted(payload["sources"].items())
    }


def list_dataset_sources() -> list[dict[str, Any]]:
    return [source.to_dict() for source in load_dataset_source_registry().values()]


def get_dataset_source(dataset_id: str) -> DatasetSource:
    validate_dataset_name(dataset_id)
    registry = load_dataset_source_registry()
    try:
        return registry[dataset_id]
    except KeyError as exc:
        raise KeyError(f"No pinned dataset source is registered for {dataset_id!r}") from exc


def dataset_cache_root() -> Path:
    override = os.environ.get("STD_TABULAR_DIFFUSION_CACHE")
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "standardized-tabular-diffusion" / "datasets"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "standardized-tabular-diffusion" / "datasets"


def _archive_path(source: DatasetSource, cache_root: Path) -> Path:
    root = cache_root.expanduser().resolve()
    path = (root / "archives" / source.dataset_id / source.source_version / source.archive_name).resolve()
    if not path.is_relative_to(root):
        raise DatasetSourceError(f"Resolved cache path escapes cache root: {path}")
    return path


def _open_request(request: Request, timeout: float):
    return urlopen(request, timeout=timeout)


def download_dataset_archive(
    dataset_id: str,
    *,
    cache_root: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
    _urlopen: Callable[[Request, float], Any] | None = None,
) -> dict[str, Any]:
    """Download one immutable source archive and verify its expected SHA-256."""

    if timeout_seconds <= 0:
        raise DatasetSourceError("timeout_seconds must be positive")
    source = get_dataset_source(dataset_id)
    root = Path(cache_root) if cache_root is not None else dataset_cache_root()
    archive_path = _archive_path(source, root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and archive_path.is_symlink():
        raise DatasetSourceError(f"Refusing a symlinked dataset archive: {archive_path}")
    if archive_path.is_file() and not refresh:
        observed = sha256_file(archive_path)
        if observed != source.sha256:
            raise DownloadIntegrityError(
                f"Cached archive checksum mismatch for {dataset_id}: expected={source.sha256}, observed={observed}; "
                "use --refresh only after checking the cache location"
            )
        return {"dataset": dataset_id, "archive_path": str(archive_path), "sha256": observed, "cached": True}

    opener = _urlopen or _open_request
    request = Request(source.retrieval_url, headers={"User-Agent": "standardized-tabular-diffusion/0.1"})
    temporary_path: Path | None = None
    try:
        with opener(request, timeout_seconds) as response:
            final_url = response.geturl()
            _validate_https_url(final_url, "redirected retrieval URL")
            header_length = response.headers.get("Content-Length") if response.headers is not None else None
            if header_length is not None and int(header_length) > source.max_download_bytes:
                raise DatasetSourceError(
                    f"Source archive exceeds the registered byte limit: {header_length} > {source.max_download_bytes}"
                )
            digest = hashlib.sha256()
            total = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=archive_path.parent,
                prefix=f".{source.archive_name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                while True:
                    block = response.read(_READ_CHUNK_SIZE)
                    if not block:
                        break
                    total += len(block)
                    if total > source.max_download_bytes:
                        raise DatasetSourceError(
                            f"Source archive exceeded the registered byte limit while downloading: {total}"
                        )
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        observed = digest.hexdigest()
        if observed != source.sha256:
            raise DownloadIntegrityError(
                f"Downloaded archive checksum mismatch for {dataset_id}: expected={source.sha256}, observed={observed}"
            )
        os.replace(temporary_path, archive_path)
        temporary_path = None
        atomic_write_json(
            archive_path.with_suffix(f"{archive_path.suffix}.source.json"),
            {"source": source.to_dict(), "archive_sha256": observed, "archive_bytes": total},
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {"dataset": dataset_id, "archive_path": str(archive_path), "sha256": source.sha256, "cached": False}


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK


def _validate_cached_extraction(destination: Path, source: DatasetSource) -> dict[str, Any] | None:
    manifest_path = destination / "source-manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("archive_sha256") != source.sha256:
        return None
    records = manifest.get("members")
    if not isinstance(records, dict) or set(records) != set(source.required_members):
        return None
    for member_name, record in records.items():
        member_path = destination / Path(*PurePosixPath(member_name).parts)
        if (
            not member_path.is_file()
            or not isinstance(record, dict)
            or sha256_file(member_path) != record.get("sha256")
        ):
            return None
    return manifest


def extract_dataset_archive(
    dataset_id: str,
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Safely extract only registry-approved members from a verified archive."""

    source = get_dataset_source(dataset_id)
    root = (Path(cache_root) if cache_root is not None else dataset_cache_root()).expanduser().resolve()
    archive_path = _archive_path(source, root)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive has not been downloaded: {archive_path}")
    observed_archive_sha = sha256_file(archive_path)
    if observed_archive_sha != source.sha256:
        raise DownloadIntegrityError(
            f"Archive checksum mismatch before extraction: expected={source.sha256}, observed={observed_archive_sha}"
        )

    destination = (root / "extracted" / source.dataset_id / source.source_version).resolve()
    if not destination.is_relative_to(root):
        raise DatasetSourceError(f"Resolved extraction path escapes cache root: {destination}")
    if destination.exists():
        manifest = _validate_cached_extraction(destination, source)
        if manifest is None:
            raise DownloadIntegrityError(
                f"Existing extraction is incomplete or modified: {destination}; remove this content-addressed cache directory"
            )
        return {"dataset": dataset_id, "extracted_path": str(destination), "cached": True, "manifest": manifest}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{source.dataset_id}-", dir=destination.parent)).resolve()
    if not temporary_root.is_relative_to(root):
        raise DatasetSourceError(f"Temporary extraction path escapes cache root: {temporary_root}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members_by_name: dict[str, zipfile.ZipInfo] = {}
            for member in archive.infolist():
                _validate_member_name(member.filename.rstrip("/"))
                if member.filename in members_by_name:
                    raise UnsafeArchiveError(f"Archive contains a duplicate member: {member.filename!r}")
                members_by_name[member.filename] = member
            missing = sorted(set(source.required_members) - set(members_by_name))
            if missing:
                raise DatasetSourceError(f"Source archive is missing required members: {missing}")

            total = 0
            records: dict[str, dict[str, Any]] = {}
            for member_name in source.required_members:
                member = members_by_name[member_name]
                if member.is_dir() or _zip_member_is_symlink(member):
                    raise UnsafeArchiveError(f"Required archive member is not a regular file: {member_name!r}")
                total += member.file_size
                if total > source.max_extracted_bytes:
                    raise UnsafeArchiveError(
                        f"Required archive members exceed the registered extraction limit: {total}"
                    )
                output_path = (temporary_root / Path(*PurePosixPath(member_name).parts)).resolve()
                if not output_path.is_relative_to(temporary_root):
                    raise UnsafeArchiveError(f"Archive member escapes extraction root: {member_name!r}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with archive.open(member) as source_handle, output_path.open("wb") as destination_handle:
                    while True:
                        block = source_handle.read(_READ_CHUNK_SIZE)
                        if not block:
                            break
                        written += len(block)
                        if written > member.file_size:
                            raise UnsafeArchiveError(f"Archive member expanded beyond declared size: {member_name!r}")
                        digest.update(block)
                        destination_handle.write(block)
                if written != member.file_size:
                    raise UnsafeArchiveError(f"Archive member size mismatch: {member_name!r}")
                records[member_name] = {"bytes": written, "sha256": digest.hexdigest()}

        manifest = {
            "manifest_schema_version": "1.0.0",
            "dataset": source.dataset_id,
            "dataset_view": source.dataset_view,
            "source_version": source.source_version,
            "archive_sha256": source.sha256,
            "license": source.license,
            "citation": source.citation,
            "members": records,
        }
        atomic_write_json(temporary_root / "source-manifest.json", manifest)
        os.replace(temporary_root, destination)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return {"dataset": dataset_id, "extracted_path": str(destination), "cached": False, "manifest": manifest}


def fetch_dataset_source(
    dataset_id: str,
    *,
    cache_root: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    download = download_dataset_archive(
        dataset_id,
        cache_root=cache_root,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )
    extraction = extract_dataset_archive(dataset_id, cache_root=cache_root)
    return {"source": get_dataset_source(dataset_id).to_dict(), "download": download, "extraction": extraction}
