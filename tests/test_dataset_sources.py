from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import standardized_tabular_diffusion.dataset_sources as dataset_sources
from standardized_tabular_diffusion.dataset_sources import (
    DatasetSource,
    DownloadIntegrityError,
    UnsafeArchiveError,
    download_dataset_archive,
    extract_dataset_archive,
    get_dataset_source,
    list_dataset_sources,
)

pytestmark = pytest.mark.integration


class _FakeResponse:
    def __init__(self, payload: bytes, *, url: str = "https://example.test/source.zip") -> None:
        self._buffer = io.BytesIO(payload)
        self._url = url
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def geturl(self) -> str:
        return self._url


def _zip_bytes(*, unsafe_member: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.csv", "feature,target\n1,0\n")
        archive.writestr("README.txt", "test fixture\n")
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "escape")
    return buffer.getvalue()


def _fixture_source(payload: bytes) -> DatasetSource:
    return DatasetSource(
        dataset_id="fixture",
        dataset_view="fixture",
        source_version="fixture-v1",
        publisher="Test Publisher",
        canonical_page="https://example.test/fixture",
        retrieval_url="https://example.test/source.zip",
        retrieved_date="2026-08-03",
        archive_name="source.zip",
        archive_format="zip",
        sha256=hashlib.sha256(payload).hexdigest(),
        required_members=("data.csv", "README.txt"),
        max_download_bytes=100_000,
        max_extracted_bytes=100_000,
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        citation="Test fixture.",
        redistribution_status="permitted",
    )


def test_public_source_registry_pins_adult_and_sick() -> None:
    records = {record["dataset_id"]: record for record in list_dataset_sources()}

    assert set(records) == {"adult", "sick"}
    assert records["adult"]["sha256"] == "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
    assert records["sick"]["sha256"] == "a0982569a7442c03a20815db58f271245e7a111b10ac46f6c6b5fa6feee4c1f4"
    assert records["adult"]["license"] == "CC-BY-4.0"
    assert get_dataset_source("sick").required_members == ("sick.data", "sick.names", "sick.test")


def test_download_verifies_checksum_and_reuses_valid_cache(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes()
    source = _fixture_source(payload)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)
    calls = 0

    def fake_open(_request, _timeout):
        nonlocal calls
        calls += 1
        return _FakeResponse(payload)

    first = download_dataset_archive("fixture", cache_root=tmp_path, _urlopen=fake_open)
    second = download_dataset_archive("fixture", cache_root=tmp_path, _urlopen=fake_open)

    archive_path = Path(first["archive_path"])
    sidecar = archive_path.with_suffix(".zip.source.json")
    assert first["cached"] is False
    assert second["cached"] is True
    assert calls == 1
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == source.sha256
    assert json.loads(sidecar.read_text(encoding="utf-8"))["source"]["dataset_id"] == "fixture"


def test_download_mismatch_leaves_no_unverified_archive(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes()
    source = replace(_fixture_source(payload), sha256="0" * 64)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)

    with pytest.raises(DownloadIntegrityError, match="checksum mismatch"):
        download_dataset_archive(
            "fixture",
            cache_root=tmp_path,
            _urlopen=lambda _request, _timeout: _FakeResponse(payload),
        )

    assert not list(tmp_path.rglob("source.zip"))
    assert not list(tmp_path.rglob("*.part"))


def test_download_rejects_redirect_downgrade(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes()
    source = _fixture_source(payload)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)

    with pytest.raises(dataset_sources.DatasetSourceError, match="HTTPS URL"):
        download_dataset_archive(
            "fixture",
            cache_root=tmp_path,
            _urlopen=lambda _request, _timeout: _FakeResponse(payload, url="http://example.test/source.zip"),
        )


def test_extraction_is_content_addressed_and_member_limited(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes()
    source = _fixture_source(payload)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)
    download_dataset_archive(
        "fixture",
        cache_root=tmp_path,
        _urlopen=lambda _request, _timeout: _FakeResponse(payload),
    )

    first = extract_dataset_archive("fixture", cache_root=tmp_path)
    second = extract_dataset_archive("fixture", cache_root=tmp_path)
    destination = Path(first["extracted_path"])

    assert first["cached"] is False
    assert second["cached"] is True
    assert (destination / "data.csv").is_file()
    assert (destination / "README.txt").is_file()
    assert set(first["manifest"]["members"]) == {"data.csv", "README.txt"}


def test_extraction_rejects_any_archive_traversal_member(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes(unsafe_member="../escape.txt")
    source = _fixture_source(payload)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)
    download_dataset_archive(
        "fixture",
        cache_root=tmp_path,
        _urlopen=lambda _request, _timeout: _FakeResponse(payload),
    )

    with pytest.raises(dataset_sources.DatasetSourceError, match="Unsafe archive member"):
        extract_dataset_archive("fixture", cache_root=tmp_path)

    assert not (tmp_path / "escape.txt").exists()


def test_modified_extraction_fails_closed(tmp_path: Path, monkeypatch) -> None:
    payload = _zip_bytes()
    source = _fixture_source(payload)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)
    download_dataset_archive(
        "fixture",
        cache_root=tmp_path,
        _urlopen=lambda _request, _timeout: _FakeResponse(payload),
    )
    first = extract_dataset_archive("fixture", cache_root=tmp_path)
    (Path(first["extracted_path"]) / "data.csv").write_text("modified", encoding="utf-8")

    with pytest.raises(DownloadIntegrityError, match="incomplete or modified"):
        extract_dataset_archive("fixture", cache_root=tmp_path)


def test_required_symlink_member_is_rejected(tmp_path: Path, monkeypatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        link = zipfile.ZipInfo("data.csv")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "README.txt")
        archive.writestr("README.txt", "fixture")
    payload = buffer.getvalue()
    source = _fixture_source(payload)
    monkeypatch.setattr(dataset_sources, "get_dataset_source", lambda _dataset_id: source)
    download_dataset_archive(
        "fixture",
        cache_root=tmp_path,
        _urlopen=lambda _request, _timeout: _FakeResponse(payload),
    )

    with pytest.raises(UnsafeArchiveError, match="not a regular file"):
        extract_dataset_archive("fixture", cache_root=tmp_path)
