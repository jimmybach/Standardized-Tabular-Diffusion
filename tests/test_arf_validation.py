from __future__ import annotations

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
from standardized_tabular_diffusion.runner import validate_action_inputs
from standardized_tabular_diffusion.validation import arf as arf_validation

pytestmark = pytest.mark.adapter


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
