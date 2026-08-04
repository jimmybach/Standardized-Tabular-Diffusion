from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

import standardized_tabular_diffusion.upstream_sources as upstream_sources

pytestmark = pytest.mark.core


def _locked_fixture(tmp_path: Path) -> tuple[dict[str, object], bytes, Path]:
    runtime = b"print('official runtime')\n"
    readme = b"not selected\n"
    prefix = "Official-deadbeef/"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(prefix + "model/runtime.py", runtime)
        archive.writestr(prefix + "Real_Datasets/private.csv", b"not materialized")
        archive.writestr(prefix + "README.md", readme)
    payload = buffer.getvalue()
    manifest = {
        "manifest_schema_version": "1.0.0",
        "model_id": "fixture",
        "repository": "https://github.com/example/Official",
        "upstream_commit": "deadbeef",
        "upstream_tree": "tree",
        "upstream_model_tree": "model-tree",
        "license": {
            "declared_expression": None,
            "license_file_present": False,
            "redistribution_status": "not-authorized",
            "review_status": "upstream-public-no-license-declared",
        },
        "archive": {
            "url": "https://example.invalid/archive.zip",
            "format": "zip",
            "root_prefix": prefix,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "runtime_files": [
            {
                "path": "model/runtime.py",
                "bytes": len(runtime),
                "sha256": hashlib.sha256(runtime).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "fixture-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, payload, manifest_path


def test_materializer_extracts_only_locked_runtime_files(tmp_path: Path, monkeypatch) -> None:
    manifest, payload, manifest_path = _locked_fixture(tmp_path)
    monkeypatch.setattr(upstream_sources, "load_source_manifest", lambda model_id: manifest)
    monkeypatch.setattr(upstream_sources, "source_manifest_path", lambda model_id: manifest_path)
    monkeypatch.setattr(upstream_sources, "_download_archive", lambda locked, timeout: payload)

    destination = tmp_path / "cache" / "fixture"
    result = upstream_sources.materialize_upstream_source(
        "fixture", repo_root=tmp_path, destination=destination
    )

    assert result["status"] == "ready"
    assert result["cached"] is False
    assert result["runtime_files_verified"] == 1
    assert (destination / "model" / "runtime.py").read_bytes() == b"print('official runtime')\n"
    assert not (destination / "Real_Datasets").exists()
    assert (destination / ".standardized-source.json").is_file()

    cached = upstream_sources.materialize_upstream_source(
        "fixture", repo_root=tmp_path, destination=destination
    )
    assert cached["cached"] is True


def test_source_validation_fails_closed_after_tampering(tmp_path: Path, monkeypatch) -> None:
    manifest, payload, manifest_path = _locked_fixture(tmp_path)
    monkeypatch.setattr(upstream_sources, "load_source_manifest", lambda model_id: manifest)
    monkeypatch.setattr(upstream_sources, "source_manifest_path", lambda model_id: manifest_path)
    monkeypatch.setattr(upstream_sources, "_download_archive", lambda locked, timeout: payload)
    destination = tmp_path / "source"
    upstream_sources.materialize_upstream_source("fixture", repo_root=tmp_path, destination=destination)
    (destination / "model" / "runtime.py").write_text("tampered", encoding="utf-8")

    with pytest.raises(upstream_sources.UpstreamSourceIntegrityError, match="Official source mismatch"):
        upstream_sources.validate_upstream_source("fixture", destination)


def test_ctabgan_plus_manifest_forbids_redistribution_and_locks_runtime() -> None:
    manifest = upstream_sources.load_source_manifest("ctab-gan-plus")

    assert manifest["upstream_commit"] == "6a6f90188cca3dac2c533fd5e8e7f20de074365b"
    assert manifest["upstream_tree"] == "f5a08d81b0309d6635bf1c7a646965a34913fa93"
    assert manifest["license"]["license_file_present"] is False
    assert manifest["license"]["redistribution_status"] == "not-authorized"
    assert len(manifest["runtime_files"]) == 5
