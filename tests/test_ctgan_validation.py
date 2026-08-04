from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import standardized_tabular_diffusion.validation.ctgan as ctgan_validation
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "ctgan" / "native-parity-run-30910275922.json"
EVIDENCE_SHA256 = "748501c8671c272a1e5d54c85fdb6550182d0e5578d550a3ca7681cc712f4570"


def _write_test_wheel(path: Path, *, unsafe_member: str | None = None) -> str:
    metadata_path = "ctgan-0.12.1.dist-info/METADATA"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            metadata_path,
            "Metadata-Version: 2.4\nName: ctgan\nVersion: 0.12.1\nLicense-Expression: BUSL-1.1\n",
        )
        archive.writestr("ctgan/__init__.py", "__version__ = '0.12.1'\n")
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ctgan_package_lock_matches_registry_and_protocol() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["ctgan"]
    spec = get_adapter_spec("ctgan")

    assert spec.upstream_repository == ctgan_validation.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == ctgan_validation.UPSTREAM_COMMIT
    assert spec.install_extra == "ctgan"
    assert spec.validation_level.value == "native-parity-validated"
    assert source_lock["package_lock"]["version"] == ctgan_validation.PACKAGE_VERSION
    assert source_lock["package_lock"]["sha256"] == ctgan_validation.WHEEL_SHA256
    assert source_lock["upstream_tree"] == ctgan_validation.UPSTREAM_TREE
    assert source_lock["license"] == ctgan_validation.LICENSE_EXPRESSION


def test_retained_ctgan_evidence_is_immutable_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == ctgan_validation.PROTOCOL_ID
    assert evidence["repository_commit"] == "18528f7f28ec2d8aa1a3f2b7d94c6d2cf8163d0e"
    assert evidence["environment"]["platform"].startswith("Linux-")
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["torch"] == "2.3.0+cpu"
    assert evidence["source"]["installed_distribution"]["record_files_verified"] == 20
    assert evidence["source"]["wheel"]["sha256"] == ctgan_validation.WHEEL_SHA256
    assert evidence["seed_cases"] == [0, 19, 73]
    assert [case["seed"] for case in evidence["cases"]] == [0, 19, 73]
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(ctgan_validation._case_passed(case["comparisons"]) for case in evidence["cases"])
    assert all(
        case["adapter_artifacts"]["sample_sha256"] == case["native_artifacts"]["sample_sha256"]
        for case in evidence["cases"]
    )


def test_ctgan_wheel_validation_checks_identity_and_license(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / ctgan_validation.WHEEL_FILENAME
    wheel_sha256 = _write_test_wheel(wheel_path)
    monkeypatch.setattr(ctgan_validation, "WHEEL_SHA256", wheel_sha256)

    record = ctgan_validation._verify_wheel(wheel_path)

    assert record["sha256"] == wheel_sha256
    assert record["license_expression"] == "BUSL-1.1"
    assert record["archive_members"] == 2


def test_ctgan_wheel_validation_rejects_archive_traversal(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / ctgan_validation.WHEEL_FILENAME
    wheel_sha256 = _write_test_wheel(wheel_path, unsafe_member="../escape.py")
    monkeypatch.setattr(ctgan_validation, "WHEEL_SHA256", wheel_sha256)

    with pytest.raises(ValueError, match="Unsafe path"):
        ctgan_validation._verify_wheel(wheel_path)


def test_ctgan_parity_gate_requires_every_comparison() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "sample_bytes_exact": True,
        "model": {
            "constructor_exact": True,
            "generator": {"keys_exact": True, "tensor_values_exact": True, "finite": True},
            "transformer_exact": True,
            "data_sampler": {"arrays_exact": True, "row_ids_exact": True, "scalars_exact": True},
            "random_state": {"numpy_exact": True, "torch_exact": True},
            "loss_values_exact": True,
        },
        "samples": {
            "rows": ctgan_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }

    assert ctgan_validation._case_passed(comparisons) is True
    comparisons["samples"]["frame_exact"] = False
    assert ctgan_validation._case_passed(comparisons) is False
