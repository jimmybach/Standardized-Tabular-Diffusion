from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.config import (
    EvaluationConfig,
    ExperimentConfig,
    SampleConfig,
    TrainConfig,
)
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.final_wave_baselines import ARFAdapter
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.runner import validate_action_inputs
from standardized_tabular_diffusion.validation import arf as arf_validation

pytestmark = pytest.mark.adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "arf" / "native-parity-run-30964711614.json"
EVIDENCE_SHA256 = "959753701a3a615afe841c32a37bb2f2610be3a6ad421ac6476ab6f50573783f"
WINDOWS_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "arf" / "windows-v2-real-function-eb37290.json"
WINDOWS_EVIDENCE_SHA256 = "056bb96d381374caa41cbb54e304b3f84bd96a3e87f310ad0262c993cc29baca"
WINDOWS_FAILURE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "arf"
    / "windows-v2-finalization-adapter-import-failure-c84a869.json"
)
WINDOWS_FAILURE_SHA256 = "01318e65bd5c735d058198ca4232a1ed4c44a1449a67a6bde768b5ed51eeed4a"
SOURCE_LOCK_PATH = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def _dataset_spec(tmp_path: Path) -> DatasetSpec:
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
            "segment": ["a", "b", "a", "b"],
            "target": ["no", "yes", "no", "yes"],
        }
    )
    train_path = tmp_path / "train.csv"
    metadata_path = tmp_path / "info.json"
    frame.to_csv(train_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")
    return DatasetSpec(
        name="arf-test",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=["value"],
        categorical_columns=["segment"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )


def test_arf_protocol_constants_lock_the_official_release() -> None:
    assert arf_validation.PACKAGE_NAME == "arfpy"
    assert arf_validation.PACKAGE_VERSION == "0.1.1"
    assert arf_validation.SDIST_FILENAME == "arfpy-0.1.1.tar.gz"
    assert arf_validation.SDIST_SHA256 == (
        "88170d5e72638b0dbfec28cfbdfee02e97bd6a06d5a636e960acd5d90d480707"
    )
    assert arf_validation.UPSTREAM_COMMIT == "6f737baaaa589f7ac3ff59f0d739ce04b0f1381c"
    assert arf_validation.UPSTREAM_TREE == "68b6fc5d28578a5c21bef560bd28f4c0d2d6401c"
    assert arf_validation.LICENSE_EXPRESSION == "MIT"
    assert arf_validation.SEED_CASES == (0, 19, 73)
    assert arf_validation.VARIANTS == ("binary", "multiclass", "regression")
    assert len(arf_validation.EXPECTED_ARCHIVE_FILES) == 16
    assert len(arf_validation.EXPECTED_GIT_BLOBS) == 6


def test_arf_retained_evidence_is_exact_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")

    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "arfpy-official-package-parity-v1"
    assert evidence["reproduction_target"] == "method-author-official-python-package"
    assert evidence["repository_commit"] == "1a6bb669796f1f5a01e6599b9373b1d4e0c33b4a"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["arfpy"] == "0.1.1"
    assert evidence["environment"]["platform"].startswith("Linux-")
    assert evidence["environment_lock"]["sha256"] == (
        "5ed951e5aee424e46a27329e826af5863a2e79e648bd46398b4e6d8c3f497fe8"
    )
    assert evidence["source"]["authority"] == "method-author"
    assert evidence["source"]["source_distribution"]["regular_files_verified"] == 16
    assert len(evidence["source"]["source_distribution"]["git_blob_matches"]) == 6
    assert evidence["source"]["installed_distribution"]["record_hashes_verified"] == 10
    assert evidence["source_unchanged_after_validation"] is True
    assert len(evidence["cases"]) == 9
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(case["comparisons"]["sample_bytes_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["samples"]["frame_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["attributes_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["bnds_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["params_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["class_probs_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["safe_json_checkpoint"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["row_level_training_data_absent"] for case in evidence["cases"])
    assert all(case["comparisons"]["forge_state"]["random_forest_absent"] for case in evidence["cases"])


def test_arf_retained_windows_v2_evidence_is_exact_complete_and_attempt_preserving() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "eb3729031189ce6b06b1b4201e1028f1c8258d73"
    assert evidence["train"]["status"] == "pass"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["hardware"] == {
        "cuda_available": False,
        "cuda_runtime": None,
        "gpu": None,
        "torch": None,
    }
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [16, 16]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True
    assert evidence["tracked_repository_unchanged"] is True

    central = evidence["central_evaluation"]
    assert central["status"] == "pass"
    assert central["protocol"] == "p3-validity"
    assert central["finalization_status"] == "finalized"
    assert central["validation"]["pending_files"] == 0
    assert central["environment"]["hardware"]["torch"] is None
    assert central["environment_lock"]["sha256"] == (
        "df78903678a6a8de185bfbc1bf7e1cca74ba01f8f5229f35ee2c495ebf6a02ca"
    )

    failure_bytes = WINDOWS_FAILURE_PATH.read_bytes()
    assert hashlib.sha256(failure_bytes).hexdigest() == WINDOWS_FAILURE_SHA256
    assert failure_bytes.endswith(b"\n")
    failure = json.loads(failure_bytes)
    assert failure["status"] == "fail"
    assert failure["finding_id"] == "RF-CORE-007"
    assert failure["stage"] == "central-evaluation-finalization"
    assert failure["model_probe"]["training_status"] == "pass"
    assert failure["error"]["cause"] == "ModuleNotFoundError: No module named 'sklearn'"
    assert failure["resolution"]["model_runtime_imported_for_central_evaluation"] is False
    assert failure["resolution"]["passing_evidence_sha256"] == WINDOWS_EVIDENCE_SHA256

    spec = get_adapter_spec("arf")
    for path in (WINDOWS_FAILURE_PATH, WINDOWS_EVIDENCE_PATH):
        assert path.relative_to(REPO_ROOT).as_posix() in spec.evidence_records

    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    windows_validation = source_lock["components"]["arf"]["windows_real_function"]
    assert windows_validation["status"] == "pass"
    assert windows_validation["level"] == "minimal-real-passed"
    assert windows_validation["repository_commit"] == evidence["repository_commit"]
    assert windows_validation["attempt_chain"] == [
        WINDOWS_FAILURE_PATH.relative_to(REPO_ROOT).as_posix(),
        WINDOWS_EVIDENCE_PATH.relative_to(REPO_ROOT).as_posix(),
    ]
    assert windows_validation["failure_evidence_file_sha256"] == WINDOWS_FAILURE_SHA256
    assert windows_validation["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256


def test_arf_checkpoint_codec_round_trips_nonfinite_bounds_without_pickle() -> None:
    source = pd.DataFrame(
        {
            "tree": pd.Series([0, 0], dtype="int64"),
            "variable": ["x", "x"],
            "min": [float("-inf"), 0.5],
            "max": [0.5, float("inf")],
            "sd": [0.0, float("nan")],
        }
    )
    payload = ARFAdapter._encode_frame(source)
    encoded = json.dumps(payload, allow_nan=False)
    restored = ARFAdapter._decode_frame(json.loads(encoded))

    pd.testing.assert_frame_equal(source, restored, check_exact=True)
    assert "pickle" not in encoded.lower()
    assert '"-inf"' in encoded and '"+inf"' in encoded and '"nan"' in encoded


def test_arf_adapter_loads_declared_types_and_rejects_missing_values(tmp_path: Path) -> None:
    adapter = ARFAdapter(tmp_path)
    dataset_spec = _dataset_spec(tmp_path)
    frame = adapter._load_training_frame(dataset_spec)

    assert str(frame["value"].dtype) == "float64"
    assert str(frame["segment"].dtype) == "category"
    assert str(frame["target"].dtype) == "category"

    damaged = pd.read_csv(dataset_spec.train_data_path)
    damaged.loc[0, "value"] = np.nan
    damaged.to_csv(dataset_spec.train_data_path, index=False)
    with pytest.raises(ValueError, match="does not accept missing values"):
        adapter._load_training_frame(dataset_spec)


def test_arf_declared_integer_decoding_is_explicit_and_retains_no_clipping(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    dataset_spec.extra["integer_columns"] = ["value"]
    native = pd.DataFrame(
        {
            "value": [1.2, 2.5, 3.8],
            "segment": ["a", "b", "a"],
            "target": ["no", "yes", "no"],
        }
    )

    decoded, report = ARFAdapter._decode_declared_integer_columns(native, dataset_spec)

    assert decoded["value"].tolist() == [1, 2, 4]
    assert str(decoded["value"].dtype) == "int64"
    assert native["value"].tolist() == [1.2, 2.5, 3.8]
    assert report["value"] == {
        "policy": "numpy-rint-ties-to-even-at-adapter-decoding-boundary",
        "changed_rows": 3,
        "clipped_rows": 0,
    }

    dataset_spec.extra["integer_columns"] = ["segment"]
    with pytest.raises(ValueError, match="canonical numerical columns"):
        ARFAdapter._decode_declared_integer_columns(native, dataset_spec)


def test_arf_parameter_contract_fails_closed(tmp_path: Path) -> None:
    adapter = ARFAdapter(tmp_path)
    base = {
        "model": "arf",
        "dataset": "arf-test",
        "output_dir": tmp_path / "run",
        "device": "cpu",
        "seed": 0,
    }
    assert adapter._training_params(RunSpec(**base))["random_state"] == 0
    assert adapter._forde_params(RunSpec(**base)) == {
        "dist": "truncnorm",
        "oob": False,
        "alpha": 0.0,
    }

    with pytest.raises(ValueError, match="delta must lie"):
        adapter._training_params(RunSpec(**base, extra={"delta": 0.6}))
    with pytest.raises(TypeError, match="early_stop must be a boolean"):
        adapter._training_params(RunSpec(**base, extra={"early_stop": "false"}))
    with pytest.raises(ValueError, match="broken oob=True"):
        adapter._forde_params(RunSpec(**base, extra={"oob": True}))


def test_arf_safe_json_checkpoint_can_be_external_without_unsafe_override(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    checkpoint = tmp_path / "reviewed" / "model.arf.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}\n", encoding="utf-8")
    checkpoint.with_name("model.arf.json.metadata.json").write_text("{}\n", encoding="utf-8")
    config = ExperimentConfig(
        model="arf",
        dataset=dataset_spec.name,
        output_dir=str(tmp_path / "run"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=True, checkpoint_path=str(checkpoint)),
        evaluation=EvaluationConfig(enabled=False),
    )

    readiness = validate_action_inputs(config, "sample", dataset_spec=dataset_spec, repo_root=tmp_path)

    assert readiness["ready"] is True
    assert readiness["checked"]["checkpoint_code_executing"] is False
    assert readiness["checked"]["allow_unsafe_external_checkpoint"] is False


def test_arf_case_gate_requires_every_exact_comparison() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "checkpoint_metadata_valid": True,
        "sample_metadata_valid": True,
        "sample_bytes_exact": True,
        "native_adversarial_loop_exercised": True,
        "native_global_numpy_state_unchanged": True,
        "adapter_global_numpy_state_unchanged": True,
        "forge_state": {
            "attributes_exact": True,
            "levels_exact": True,
            "bnds_exact": True,
            "params_exact": True,
            "class_probs_exact": True,
            "adversarial_oob_accuracy_exact": True,
            "safe_json_checkpoint": True,
            "row_level_training_data_absent": True,
            "random_forest_absent": True,
            "privacy_not_overclaimed": True,
            "artifact_access_control_required": True,
        },
        "samples": {
            "rows": arf_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }
    assert arf_validation._case_passed(comparisons) is True
    comparisons["forge_state"]["params_exact"] = False
    assert arf_validation._case_passed(comparisons) is False


def test_arf_authoritative_environment_rejects_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        arf_validation.importlib.metadata,
        "version",
        lambda name: arf_validation.EXPECTED_DISTRIBUTION_VERSIONS[name],
    )
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "platform", lambda: "Windows-test")
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    with pytest.raises(RuntimeError, match="requires Linux"):
        arf_validation._verify_environment()
