from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import standardized_tabular_diffusion.validation.smote as smote_validation
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "smote" / "native-parity-run-30918785254.json"
EVIDENCE_SHA256 = "1b375b93c332327dd2118c2aad9420497008be1390078e1e48e79f8270f74863"


def _write_test_wheel(path: Path, *, unsafe_member: str | None = None) -> tuple[str, str, int]:
    dist_info = "imbalanced_learn-0.14.2.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: imbalanced-learn\n"
        "Version: 0.14.2\n"
        "Requires-Python: >=3.10\n"
        "License-File: LICENSE\n"
    )
    license_bytes = b"MIT test fixture\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/licenses/LICENSE", license_bytes)
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(license_bytes).hexdigest(), 2


def test_smote_package_lock_matches_registry_and_protocol() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["smote"]
    spec = get_adapter_spec("smote")

    assert spec.upstream_repository == smote_validation.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == smote_validation.UPSTREAM_COMMIT
    assert spec.install_extra == "smote"
    assert spec.reproduction_target == "classical-oversampling-reference"
    assert spec.validation_level.value == "native-parity-validated"
    assert source_lock["benchmark_role"] == "classification-only-classical-oversampling-reference"
    assert source_lock["package_lock"]["version"] == smote_validation.PACKAGE_VERSION
    assert source_lock["package_lock"]["sha256"] == smote_validation.WHEEL_SHA256
    assert source_lock["upstream_tree"] == smote_validation.UPSTREAM_TREE
    assert source_lock["license"] == smote_validation.LICENSE_EXPRESSION
    assert source_lock["validation"]["status"] == "pass"
    assert source_lock["validation"]["workflow_run_id"] == 30918785254


def test_retained_smote_evidence_is_immutable_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == smote_validation.PROTOCOL_ID
    assert evidence["repository_commit"] == "283c8987a28ef3c3871d8d819325a035f957a045"
    assert evidence["claim_boundary"].startswith("Classification-only classical oversampling")
    assert evidence["environment"]["platform"].startswith("Linux-")
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["imbalanced-learn"] == "0.14.2"
    assert evidence["source"]["installed_distribution"]["record_files_verified"] == 123
    assert evidence["source"]["wheel"]["sha256"] == smote_validation.WHEEL_SHA256
    assert evidence["variants"] == ["smote", "smotenc", "smoten"]
    assert evidence["seed_cases"] == [0, 19, 73]
    assert [(case["variant"], case["seed"]) for case in evidence["cases"]] == [
        (variant, seed)
        for variant in ("smote", "smotenc", "smoten")
        for seed in (0, 19, 73)
    ]
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(smote_validation._case_passed(case["comparisons"]) for case in evidence["cases"])
    assert all(
        case["adapter_artifacts"]["sample_sha256"] == case["native_artifacts"]["sample_sha256"]
        for case in evidence["cases"]
    )


def test_smote_wheel_validation_checks_identity_and_license(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / smote_validation.WHEEL_FILENAME
    wheel_sha256, license_sha256, member_count = _write_test_wheel(wheel_path)
    monkeypatch.setattr(smote_validation, "WHEEL_SHA256", wheel_sha256)
    monkeypatch.setattr(smote_validation, "WHEEL_LICENSE_FILE_SHA256", license_sha256)
    monkeypatch.setattr(smote_validation, "EXPECTED_ARCHIVE_MEMBERS", member_count)

    record = smote_validation._verify_wheel(wheel_path)

    assert record["sha256"] == wheel_sha256
    assert record["license"] == "MIT"
    assert record["license_file_sha256"] == license_sha256
    assert record["archive_members"] == member_count


def test_smote_wheel_validation_rejects_archive_traversal(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / smote_validation.WHEEL_FILENAME
    wheel_sha256, _, _ = _write_test_wheel(wheel_path, unsafe_member="../escape.py")
    monkeypatch.setattr(smote_validation, "WHEEL_SHA256", wheel_sha256)

    with pytest.raises(ValueError, match="Unsafe path"):
        smote_validation._verify_wheel(wheel_path)


def test_smote_parity_gate_requires_every_comparison() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "adapter_metadata_exact": True,
        "sample_bytes_exact": True,
        "native_balanced_rows_exact": True,
        "native_balanced_classes_exact": True,
        "native_global_numpy_state_unchanged": True,
        "adapter_global_numpy_state_unchanged": True,
        "sampler": {
            "class_exact": True,
            "module_exact": True,
            "params_exact": True,
            "sampling_strategy_exact": True,
            "n_features_exact": True,
            "feature_names_exact": True,
            "neighbor_state_exact": True,
            "categorical_features_exact": True,
            "continuous_features_exact": True,
            "encoder_categories_exact": True,
            "median_std_exact": True,
        },
        "samples": {
            "rows": smote_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "target_classes_present": True,
            "missing_values": 0,
        },
    }

    assert smote_validation._case_passed(comparisons) is True
    comparisons["sampler"]["neighbor_state_exact"] = False
    assert smote_validation._case_passed(comparisons) is False
