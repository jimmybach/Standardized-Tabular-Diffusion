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


def test_materializer_applies_declared_text_normalization_before_hashing(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_archive = b"line-one\r\nline-two\r\n\r\n"
    runtime_normalized = b"line-one\nline-two\n"
    prefix = "Official-deadbeef/"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(prefix + "model/runtime.py", runtime_archive)
    payload = buffer.getvalue()
    manifest = {
        "manifest_schema_version": "1.0.0",
        "model_id": "fixture",
        "repository": "https://github.com/example/Official",
        "upstream_commit": "deadbeef",
        "upstream_tree": "tree",
        "upstream_model_tree": "model-tree",
        "source_hash_normalization": "lf-one-final-newline",
        "license": {},
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
                "bytes": len(runtime_normalized),
                "sha256": hashlib.sha256(runtime_normalized).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "fixture-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(upstream_sources, "load_source_manifest", lambda model_id: manifest)
    monkeypatch.setattr(upstream_sources, "source_manifest_path", lambda model_id: manifest_path)
    monkeypatch.setattr(upstream_sources, "_download_archive", lambda locked, timeout: payload)

    destination = tmp_path / "normalized-source"
    result = upstream_sources.materialize_upstream_source(
        "fixture", repo_root=tmp_path, destination=destination
    )

    assert result["runtime_files_verified"] == 1
    assert (destination / "model" / "runtime.py").read_bytes() == runtime_normalized


def test_ctabgan_plus_manifest_forbids_redistribution_and_locks_runtime() -> None:
    manifest = upstream_sources.load_source_manifest("ctab-gan-plus")

    assert manifest["upstream_commit"] == "6a6f90188cca3dac2c533fd5e8e7f20de074365b"
    assert manifest["upstream_tree"] == "f5a08d81b0309d6635bf1c7a646965a34913fa93"
    assert manifest["license"]["license_file_present"] is False
    assert manifest["license"]["redistribution_status"] == "not-authorized"
    assert len(manifest["runtime_files"]) == 5


def test_ctabgan_manifest_authorizes_redistribution_and_locks_selected_source() -> None:
    manifest = upstream_sources.load_source_manifest("ctab-gan")

    assert manifest["upstream_commit"] == "73d4e315a2a51cf16c97ed8a00d2dad456cfce8a"
    assert manifest["upstream_tree"] == "3ef0223477193400d88344ff66b7ac6ffeefa173"
    assert manifest["source_hash_normalization"] == "lf-one-final-newline"
    assert manifest["license"]["declared_expression"] == "Apache-2.0"
    assert manifest["license"]["redistribution_status"] == "authorized"
    assert len(manifest["runtime_files"]) == 7


def test_distributed_ctabgan_source_matches_the_checksum_lock() -> None:
    source_root = Path(__file__).resolve().parents[1] / "TabDDPM-main" / "CTAB-GAN"
    result = upstream_sources.validate_upstream_source("ctab-gan", source_root)

    assert result["upstream_commit"] == "73d4e315a2a51cf16c97ed8a00d2dad456cfce8a"
    assert result["runtime_files_verified"] == 7


def test_stasy_manifest_locks_the_exact_tabsyn_benchmark_snapshot() -> None:
    manifest = upstream_sources.load_source_manifest("stasy")

    assert manifest["upstream_commit"] == "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"
    assert manifest["upstream_model_tree"] == "4f56a7223d71d6b75c1698824c5d0245bf716bc6"
    assert manifest["license"]["declared_expression"] == "Apache-2.0"
    assert manifest["authority"] == "benchmark-vendored"
    assert manifest["distributed_source_path"] == "TabSyn-main"
    assert manifest["method_author_repository"]["license_file_present"] is False
    assert manifest["dependencies"]["libzero"]["wheel_sha256"] == (
        "f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d"
    )
    assert len(manifest["runtime_files"]) == 30


def test_distributed_stasy_source_matches_the_checksum_lock() -> None:
    source_root = Path(__file__).resolve().parents[1] / "TabSyn-main"
    result = upstream_sources.validate_upstream_source("stasy", source_root)

    assert result["upstream_commit"] == "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"
    assert result["runtime_files_verified"] == 30


def test_stasy_source_status_defaults_to_the_distributed_snapshot() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = upstream_sources.source_status("stasy", repo_root=repo_root)

    assert result["status"] == "ready"
    assert Path(result["source_dir"]) == (repo_root / "TabSyn-main").resolve()


def test_codi_manifest_locks_the_exact_tabsyn_benchmark_snapshot() -> None:
    manifest = upstream_sources.load_source_manifest("codi")

    assert manifest["upstream_commit"] == "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"
    assert manifest["upstream_model_tree"] == "85c16ccfb76fbf00db6b30450ca47e9928efa8d3"
    assert manifest["license"]["declared_expression"] == "Apache-2.0"
    assert manifest["authority"] == "benchmark-vendored"
    assert manifest["distributed_source_path"] == "TabSyn-main"
    assert manifest["method_author_repository"]["license_file_present"] is False
    assert manifest["dependencies"]["libzero"]["wheel_sha256"] == (
        "f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d"
    )
    assert len(manifest["runtime_files"]) == 24


def test_distributed_codi_source_matches_the_checksum_lock() -> None:
    source_root = Path(__file__).resolve().parents[1] / "TabSyn-main"
    result = upstream_sources.validate_upstream_source("codi", source_root)

    assert result["upstream_commit"] == "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"
    assert result["runtime_files_verified"] == 24


def test_codi_source_status_defaults_to_the_distributed_snapshot() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = upstream_sources.source_status("codi", repo_root=repo_root)

    assert result["status"] == "ready"
    assert Path(result["source_dir"]) == (repo_root / "TabSyn-main").resolve()
