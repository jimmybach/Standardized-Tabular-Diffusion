from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.validation import nrgboost as nrgboost_validation

pytestmark = pytest.mark.adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
WINDOWS_EVIDENCE_PATH = (
    REPO_ROOT / "docs" / "evidence" / "nrgboost" / "windows-v2-real-function-64eec7d.json"
)
WINDOWS_EVIDENCE_SHA256 = "1c258f7b05775d252aa4c2a960cdfcc71f715be15755261551dd464cff95631c"
WINDOWS_FAILURE_PATH = (
    REPO_ROOT / "docs" / "evidence" / "nrgboost" / "windows-v2-probe-dependency-failure-3271298.json"
)
WINDOWS_FAILURE_SHA256 = "5d95f7a752ea0747b4809b26d960b9b31688c0f537d7588aea4226bd864bf1ed"
WINDOWS_PROVENANCE_PATH = (
    REPO_ROOT / "docs" / "evidence" / "nrgboost" / "windows-source-build-provenance-20260820.json"
)
WINDOWS_PROVENANCE_SHA256 = "65b5e7b2a21ed39fe621207065fc3120d6bed6ee5a8a64f4575344be11e802f0"


def test_nrgboost_protocol_constants_lock_the_official_release() -> None:
    assert nrgboost_validation.PACKAGE_NAME == "nrgboost"
    assert nrgboost_validation.PACKAGE_VERSION == "0.0.3"
    assert nrgboost_validation.UPSTREAM_TAG == "v0.0.3"
    assert nrgboost_validation.UPSTREAM_COMMIT == "feef73a3edb20b911c2f7214b13f810909ef20ad"
    assert nrgboost_validation.UPSTREAM_TREE == "e3e84bacc7236a36af93c3d214de14bd308d2767"
    assert nrgboost_validation.WHEEL_SHA256 == (
        "dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb"
    )
    assert nrgboost_validation.LICENSE_EXPRESSION == "MIT"
    assert nrgboost_validation.SEED_CASES == (0, 19, 73)
    assert nrgboost_validation.VARIANTS == ("classification", "regression")


def test_nrgboost_validation_runtime_is_bounded_and_deterministic() -> None:
    assert nrgboost_validation.TRAINING_PARAMS["num_trees"] == 3
    assert nrgboost_validation.TRAINING_PARAMS["num_threads"] == 1
    assert nrgboost_validation.SAMPLING_PARAMS["num_threads"] == 1
    assert nrgboost_validation.SAMPLING_PARAMS["output_full_chain"] is False
    assert nrgboost_validation.EXPECTED_SAMPLE_ROWS == 16
    extras = nrgboost_validation._adapter_extra()
    assert extras["training_temperature"] == nrgboost_validation.TRAINING_PARAMS["temperature"]
    assert extras["num_steps"] == nrgboost_validation.SAMPLING_PARAMS["num_steps"]


def test_nrgboost_windows_runtime_lock_is_separate_and_declares_pipeline_dependency() -> None:
    lock_lines = {
        line.strip()
        for line in (REPO_ROOT / "requirements-nrgboost-windows-v2.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "packaging==26.3" in lock_lines
    authoritative_lines = (REPO_ROOT / "requirements-nrgboost-validation.txt").read_text(encoding="utf-8")
    assert "packaging==" not in authoritative_lines


def test_nrgboost_retained_windows_v2_evidence_is_exact_complete_and_attempt_preserving() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "64eec7d590f3610e74795036a1fb188e554741a0"
    assert evidence["train"]["status"] == "pass"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["hardware"] == {
        "cuda_available": False,
        "cuda_runtime": None,
        "gpu": None,
        "torch": None,
    }
    assert evidence["environment_lock"]["sha256"] == (
        "d27ceee1ed87e1eb9d7a70b5ed0c1171c81c4f430425806f20b037aca547a11e"
    )
    assert evidence["environment_lock"]["packages"]["packaging"]["observed"] == "26.3"
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [16, 16]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True
    assert evidence["central_evaluation"]["status"] == "pass"
    assert evidence["central_evaluation"]["validation"]["pending_files"] == 0

    failure_bytes = WINDOWS_FAILURE_PATH.read_bytes()
    assert hashlib.sha256(failure_bytes).hexdigest() == WINDOWS_FAILURE_SHA256
    assert failure_bytes.endswith(b"\n")
    failure = json.loads(failure_bytes)
    assert failure["status"] == "fail"
    assert failure["finding_id"] == "RF-CORE-008"
    assert failure["model_execution_started"] is False
    assert failure["error"]["message"] == "No module named 'packaging'"
    assert failure["resolution"]["passing_evidence_sha256"] == WINDOWS_EVIDENCE_SHA256

    provenance_bytes = WINDOWS_PROVENANCE_PATH.read_bytes()
    assert hashlib.sha256(provenance_bytes).hexdigest() == WINDOWS_PROVENANCE_SHA256
    assert provenance_bytes.endswith(b"\n")
    provenance = json.loads(provenance_bytes)
    assert provenance["status"] == "pass"
    assert provenance["source"]["source_code_modified"] is False
    assert provenance["output"]["committed_or_redistributed"] is False
    assert provenance["output"]["sha256"] == (
        "24d852ebad1687bb4598ac0c739922f1f4cc4a1496ce37b06630c4439622f739"
    )
    assert provenance["claim"] == "diagnostic-windows-source-build-not-authoritative-native-parity"

    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["nrgboost"]
    windows = source_lock["windows_real_function"]
    assert windows["level"] == "minimal-real-passed"
    assert windows["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert windows["failure_evidence_file_sha256"] == WINDOWS_FAILURE_SHA256
    assert windows["provenance_evidence_file_sha256"] == WINDOWS_PROVENANCE_SHA256
    assert windows["source_code_modified"] is False
    assert windows["attempt_chain"] == [
        WINDOWS_FAILURE_PATH.relative_to(REPO_ROOT).as_posix(),
        WINDOWS_PROVENANCE_PATH.relative_to(REPO_ROOT).as_posix(),
        WINDOWS_EVIDENCE_PATH.relative_to(REPO_ROOT).as_posix(),
    ]

    spec = get_adapter_spec("nrgboost")
    for path in (WINDOWS_FAILURE_PATH, WINDOWS_PROVENANCE_PATH, WINDOWS_EVIDENCE_PATH):
        assert path.relative_to(REPO_ROOT).as_posix() in spec.evidence_records


def test_nrgboost_case_gate_fails_closed() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "adapter_metadata_exact": True,
        "checkpoint_bytes_exact": True,
        "sample_bytes_exact": True,
        "checkpoint_structure_exact": True,
        "native_global_numpy_state_unchanged": True,
        "adapter_global_numpy_state_unchanged": True,
        "samples": {
            "rows": nrgboost_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }
    assert nrgboost_validation._case_passed(comparisons) is True
    comparisons["checkpoint_bytes_exact"] = False
    assert nrgboost_validation._case_passed(comparisons) is False


def test_nrgboost_authoritative_environment_rejects_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(
        nrgboost_validation.importlib.metadata,
        "version",
        lambda name: nrgboost_validation.EXPECTED_DISTRIBUTION_VERSIONS[name],
    )
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "platform", lambda: "Windows-test")
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    with pytest.raises(RuntimeError, match="requires Linux"):
        nrgboost_validation._verify_environment()


def test_nrgboost_adapter_module_does_not_import_unrelated_sklearn_baselines() -> None:
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise AssertionError(f'unexpected optional import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import standardized_tabular_diffusion.models.next_wave_baselines
"""
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
