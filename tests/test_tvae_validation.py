from __future__ import annotations

import hashlib
import json
from pathlib import Path

import standardized_tabular_diffusion.validation.tvae as tvae_validation
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tvae" / "native-parity-run-30913867621.json"
EVIDENCE_SHA256 = "ad539ffdb637084a25dc3ab4ec5d54374ff6831525ca63adca2cfa48c3ef95f7"


def test_tvae_package_lock_matches_registry_and_protocol() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tvae"]
    spec = get_adapter_spec("tvae")

    assert spec.upstream_repository == tvae_validation.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == tvae_validation.UPSTREAM_COMMIT
    assert spec.install_extra == "tvae"
    assert spec.validation_level.value == "native-parity-validated"
    assert source_lock["package_lock"]["version"] == tvae_validation.PACKAGE_VERSION
    assert source_lock["package_lock"]["sha256"] == tvae_validation.WHEEL_SHA256
    assert source_lock["upstream_tree"] == tvae_validation.UPSTREAM_TREE
    assert source_lock["license"] == tvae_validation.LICENSE_EXPRESSION
    assert source_lock["validation"]["status"] == "pass"
    assert EVIDENCE_PATH.as_posix().endswith(source_lock["validation"]["artifact"]["permanent_evidence_path"])


def test_retained_tvae_evidence_is_immutable_and_complete() -> None:
    raw_evidence = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(raw_evidence)

    assert hashlib.sha256(raw_evidence).hexdigest() == EVIDENCE_SHA256
    assert evidence["status"] == "pass"
    assert evidence["model_id"] == "tvae"
    assert evidence["protocol_id"] == "tvae-native-parity-v1"
    assert evidence["repository_commit"] == "64f3c23c617f2fc4cbebfba9d36845a812892355"
    assert evidence["seed_cases"] == [0, 19, 73]
    assert [case["seed"] for case in evidence["cases"]] == [0, 19, 73]
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(tvae_validation._case_passed(case["comparisons"]) for case in evidence["cases"])
    assert evidence["source"]["installed_distribution"]["record_files_verified"] == 20
    assert evidence["source"]["synthesizer"] == {
        "class": "TVAE",
        "legacy_snapshot_absent": True,
        "module": "ctgan.synthesizers.tvae",
    }


def test_legacy_tvae_snapshot_is_recorded_and_removed() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tvae"]
    disposition = source_lock["legacy_snapshot_disposition"]

    assert disposition["declared_version"] == "0.5.2.dev0"
    assert disposition["closest_reviewed_upstream_commit"] == (
        "ace3dbc4bd3ef7f4ddc027a1b47e8eb916378893"
    )
    assert disposition["shared_files_at_closest_commit"] == 44
    assert disposition["exact_shared_files_at_closest_commit"] == 39
    assert disposition["files_removed"] == 47
    assert disposition["removed_bytes"] == 168098
    assert disposition["tvae_source_sha256"] == (
        "0b0bc0ed424f295084a395a212eb40f1464df0e6474c313e488e8ad43226689f"
    )
    assert not (REPO_ROOT / "TabDDPM-main" / "CTGAN" / "CTGAN" / "ctgan" / "__init__.py").exists()
    assert not (REPO_ROOT / "TabDDPM-main" / "CTGAN" / "train_sample_tvae.py").exists()


def test_tvae_parity_gate_requires_every_comparison() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "sample_bytes_exact": True,
        "model": {
            "constructor_exact": True,
            "device_exact": True,
            "decoder": {
                "keys_exact": True,
                "tensor_values_exact": True,
                "finite": True,
                "sigma_finite": True,
            },
            "transformer_exact": True,
            "random_state": {"numpy_exact": True, "torch_exact": True},
            "loss_values_exact": True,
        },
        "samples": {
            "rows": tvae_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }

    assert tvae_validation._case_passed(comparisons) is True
    comparisons["model"]["decoder"]["sigma_finite"] = False
    assert tvae_validation._case_passed(comparisons) is False
