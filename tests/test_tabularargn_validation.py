from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.validation.tabularargn import (
    PROTOCOL_ID,
    SAMPLE_ROWS,
    SEED_CASES,
    SOURCE_ARCHIVE_FILENAME,
    TRAIN_ROWS,
    VARIANTS,
    WHEEL_FILENAME,
    _load_manifest,
    _normalize_sample_for_adapter_contract,
    _verify_source_archive,
    _verify_wheel,
    _write_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabularargn" / "native-parity-run-30961590047.json"
EVIDENCE_SHA256 = "411d24cd5b06090ea0d2d96e22232198fc83d0731b3371c14e9b4c50165850ec"
WINDOWS_EVIDENCE_PATH = (
    REPO_ROOT / "docs" / "evidence" / "tabularargn" / "windows-v2-real-function-6f9e065.json"
)
WINDOWS_EVIDENCE_SHA256 = "bfd636f15c51bfee3a94a8041dbe928a235f95e77ea8204b1cb3f8fc9ed0c45c"
WINDOWS_FAILURE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "tabularargn"
    / "windows-v2-finalization-dependency-failure-ec53f98.json"
)
WINDOWS_FAILURE_SHA256 = "8a19d8245ac6c1611b27b420f8a99879a2b6f402f22aa52478d6b356bae67529"


def test_locked_release_manifest_and_protocol_scope_are_exact() -> None:
    manifest = _load_manifest(REPO_ROOT)
    assert manifest["model_id"] == "tabularargn"
    assert manifest["package"]["filename"] == WHEEL_FILENAME
    assert manifest["package"]["record_hashed_files"] == len(manifest["installed_files"]) == 53
    assert manifest["wheel_source_comparison"]["exact_shared_source_files"] == 50
    assert PROTOCOL_ID == "tabularargn-official-package-parity-v2"
    assert SEED_CASES == (0, 19, 73)
    assert VARIANTS == ("binary", "multiclass", "regression")
    assert SAMPLE_ROWS == 7


def test_retained_native_parity_evidence_is_exact_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == PROTOCOL_ID
    assert evidence["repository_commit"] == "e5129b8799beb1b792c882a9387dbcc1f13a39ce"
    assert evidence["seed_cases"] == list(SEED_CASES)
    assert evidence["variants"] == list(VARIANTS)
    assert evidence["package_unchanged_after_validation"] is True
    assert evidence["compatibility_boundary"]["source_patches"] == []
    assert len(evidence["cases"]) == 9

    for case in evidence["cases"]:
        comparisons = case["comparisons"]
        assert case["status"] == "pass"
        assert comparisons["checkpoint"]["keys_exact"] is True
        assert comparisons["checkpoint"]["tensors_exact"] is True
        assert comparisons["checkpoint"]["file_bytes_exact"] is True
        assert comparisons["model_config_semantics_exact"] is True
        assert comparisons["target_stats_semantics_exact"] is True
        assert comparisons["contract_normalized_samples_exact"] is True
        assert comparisons["raw_samples_exact"] is False
        assert comparisons["raw_sample_dtypes"]["native"]["group"] == "string"
        assert comparisons["raw_sample_dtypes"]["adapter"]["group"] == "str"
        assert comparisons["sample_bytes_exact"] is True
        assert comparisons["sample_rows"] == SAMPLE_ROWS
        assert comparisons["missing_values"] == 0
        assert comparisons["finite_numerical_output"] is True
        assert comparisons["categorical_domains_valid"] is True
        assert comparisons["adapter_metadata_valid"] is True
        assert comparisons["raw_and_encoded_training_data_pruned"] is True
        assert comparisons["checkpoint_output_local"] is True

    spec = get_adapter_spec("tabularargn")
    assert spec.validation_level.value == "native-parity-validated"
    assert str(EVIDENCE_PATH.relative_to(REPO_ROOT)).replace("\\", "/") in spec.evidence_records


def test_retained_windows_v2_evidence_is_exact_complete_and_attempt_preserving() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "6f9e065fd69f094806794fab69b4e03874699a82"
    assert evidence["train"]["status"] == "pass"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["hardware"] == {
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "gpu": "NVIDIA GeForce RTX 5080",
        "gpu_count": 1,
        "torch": "2.11.0+cu128",
    }
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [8, 8]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True

    central = evidence["central_evaluation"]
    assert central["status"] == "pass"
    assert central["finalization_status"] == "finalized"
    assert central["validation"]["pending_files"] == 0
    assert central["environment_lock"]["sha256"] == (
        "df78903678a6a8de185bfbc1bf7e1cca74ba01f8f5229f35ee2c495ebf6a02ca"
    )
    assert central["environment_lock"]["packages"]["jsonschema"]["observed"] == "4.23.0"

    failure_bytes = WINDOWS_FAILURE_PATH.read_bytes()
    assert hashlib.sha256(failure_bytes).hexdigest() == WINDOWS_FAILURE_SHA256
    assert failure_bytes.endswith(b"\n")
    failure = json.loads(failure_bytes)
    assert failure["status"] == "fail"
    assert failure["stage"] == "central-evaluation-finalization"
    assert failure["model_probe"]["training_status"] == "pass"
    assert failure["error"]["cause"] == "ModuleNotFoundError: No module named 'jsonschema'"
    assert failure["resolution"]["passing_evidence_sha256"] == WINDOWS_EVIDENCE_SHA256

    spec = get_adapter_spec("tabularargn")
    for path in (WINDOWS_FAILURE_PATH, WINDOWS_EVIDENCE_PATH):
        assert path.relative_to(REPO_ROOT).as_posix() in spec.evidence_records


@pytest.mark.parametrize("variant", VARIANTS)
def test_parity_fixture_covers_declared_types_without_missing_values(tmp_path: Path, variant: str) -> None:
    dataset_spec, record = _write_fixture(tmp_path, variant)
    assert dataset_spec.train_data_path is not None and dataset_spec.train_data_path.is_file()
    assert dataset_spec.column_names == ["first", "second", "group", "target"]
    assert dataset_spec.numerical_columns == ["first", "second"]
    assert dataset_spec.categorical_columns == ["group"]
    assert dataset_spec.target_columns == ["target"]
    assert record["training_rows"] == TRAIN_ROWS
    assert record["missing_values"] == 0
    assert dataset_spec.task_type == ("regression" if variant == "regression" else "classification")


@pytest.mark.parametrize("variant", VARIANTS)
def test_sample_contract_normalizes_only_categorical_outputs(tmp_path: Path, variant: str) -> None:
    dataset_spec, _ = _write_fixture(tmp_path, variant)
    native = pd.DataFrame(
        {
            "first": [1.25],
            "second": [2.5],
            "group": [1],
            "target": [2 if variant != "regression" else 2.75],
        }
    )
    adapter = native.copy()
    adapter["group"] = adapter["group"].astype(str)
    if variant != "regression":
        adapter["target"] = adapter["target"].astype(str)

    normalized_native = _normalize_sample_for_adapter_contract(native, dataset_spec)
    normalized_adapter = _normalize_sample_for_adapter_contract(adapter, dataset_spec)

    assert not native.equals(adapter)
    assert normalized_native.equals(normalized_adapter)
    assert normalized_native["group"].map(lambda value: isinstance(value, str)).all()
    if variant == "regression":
        assert pd.api.types.is_float_dtype(normalized_native["target"])
    else:
        assert normalized_native["target"].map(lambda value: isinstance(value, str)).all()


def test_sample_contract_does_not_hide_numerical_dtype_or_value_changes(tmp_path: Path) -> None:
    dataset_spec, _ = _write_fixture(tmp_path, "binary")
    native = pd.DataFrame({"first": [1.25], "second": [2.5], "group": [1], "target": [0]})
    numerical_dtype_changed = native.copy()
    numerical_dtype_changed["first"] = numerical_dtype_changed["first"].astype(str)
    value_changed = native.copy()
    value_changed.loc[0, "second"] = 9.5

    normalized_native = _normalize_sample_for_adapter_contract(native, dataset_spec)

    assert not normalized_native.equals(
        _normalize_sample_for_adapter_contract(numerical_dtype_changed, dataset_spec)
    )
    assert not normalized_native.equals(_normalize_sample_for_adapter_contract(value_changed, dataset_spec))


def test_downloaded_official_wheel_and_source_archive_match_every_locked_member() -> None:
    configured_wheel = os.environ.get("TABULARARGN_WHEEL_PATH")
    configured_source = os.environ.get("TABULARARGN_SOURCE_ARCHIVE_PATH")
    audit = REPO_ROOT / ".cache" / "tabularargn-audit-2.6.2"
    wheel_path = Path(configured_wheel) if configured_wheel else audit / WHEEL_FILENAME
    source_path = Path(configured_source) if configured_source else audit / SOURCE_ARCHIVE_FILENAME
    if not wheel_path.is_file() or not source_path.is_file():
        pytest.skip("locked official release files are supplied only in the formal validation job")
    wheel = _verify_wheel(REPO_ROOT, wheel_path)
    source = _verify_source_archive(REPO_ROOT, source_path, wheel_path)
    assert wheel["record_hashed_files"] == 53
    assert source["exact_shared_package_source_files"] == 50
