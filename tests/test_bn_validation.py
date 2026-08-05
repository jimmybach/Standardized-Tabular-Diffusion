from __future__ import annotations

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
from standardized_tabular_diffusion.models.structured_baselines import BNAdapter, BNPreprocessor
from standardized_tabular_diffusion.runner import validate_action_inputs
from standardized_tabular_diffusion.validation import bn as bn_validation

pytestmark = pytest.mark.adapter


def _dataset_spec(tmp_path: Path, *, task_type: str = "classification") -> DatasetSpec:
    target: list[object]
    if task_type == "regression":
        target = [0.2, 0.8, 1.4, 2.0]
    else:
        target = ["no", "yes", "no", "yes"]
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
            "constant": [3.25, 3.25, 3.25, 3.25],
            "segment": ["a", "b", "a", "b"],
            "target": target,
        }
    )
    train_path = tmp_path / "train.csv"
    metadata_path = tmp_path / "info.json"
    frame.to_csv(train_path, index=False)
    metadata_path.write_text("{}\n", encoding="utf-8")
    return DatasetSpec(
        name=f"bn-{task_type}-test",
        task_type=task_type,
        column_names=list(frame.columns),
        numerical_columns=["value", "constant"],
        categorical_columns=["segment"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )


def test_bn_protocol_constants_lock_the_official_release() -> None:
    assert bn_validation.PACKAGE_NAME == "pgmpy"
    assert bn_validation.PACKAGE_VERSION == "1.1.2"
    assert bn_validation.WHEEL_FILENAME == "pgmpy-1.1.2-py3-none-any.whl"
    assert bn_validation.WHEEL_SHA256 == ("e55c78763a4a45dd644a13b250cea86af0c7e08590cf35de489624f34a4d9a0b")
    assert bn_validation.UPSTREAM_COMMIT == "617cb48af678a7a471aad81d523ca95d2095430f"
    assert bn_validation.UPSTREAM_TREE == "6c7adc00a479f540b2215889b1fac99a7b0b8a9c"
    assert bn_validation.UPSTREAM_TAG == "v1.1.2"
    assert bn_validation.LICENSE_EXPRESSION == "MIT"
    assert bn_validation.EXPECTED_WHEEL_FILES == 649
    assert bn_validation.EXPECTED_PACKAGE_FILES == 636
    assert bn_validation.EXPECTED_RECORD_HASHES == 648
    # Nine runtime files plus the license are tied to official Git blobs.
    assert len(bn_validation.EXPECTED_GIT_BLOBS) == 10
    assert bn_validation.SEED_CASES == (0, 19, 73)
    assert bn_validation.VARIANTS == ("binary", "multiclass", "regression")


def test_bn_adapter_validates_declared_types_and_missing_values(tmp_path: Path) -> None:
    adapter = BNAdapter(tmp_path)
    dataset_spec = _dataset_spec(tmp_path)
    frame = adapter._load_training_frame(dataset_spec)

    assert str(frame["value"].dtype) == "float64"
    assert frame.isna().sum().sum() == 0

    damaged = pd.read_csv(dataset_spec.train_data_path)
    damaged.loc[0, "value"] = np.nan
    damaged.to_csv(dataset_spec.train_data_path, index=False)
    with pytest.raises(ValueError, match="does not accept missing values"):
        adapter._load_training_frame(dataset_spec)


def test_bn_recipe_contract_fails_closed(tmp_path: Path) -> None:
    adapter = BNAdapter(tmp_path)
    dataset_spec = _dataset_spec(tmp_path)
    base = RunSpec(
        model="bn",
        dataset=dataset_spec.name,
        output_dir=tmp_path / "run",
        device="cpu",
        seed=0,
    )
    recipe = adapter._recipe_params(base, dataset_spec, source_rows=20)
    assert recipe == {
        "num_bins": 16,
        "quantile_method": "averaged_inverted_cdf",
        "subsample": None,
        "scoring_method": "bic-d",
        "return_type": "dag",
        "max_indegree": 2,
        "max_iter": 100,
        "tabu_length": 100,
        "epsilon": 1e-4,
        "prior_type": "BDeu",
        "equivalent_sample_size": 5.0,
        "n_jobs": 1,
    }

    with pytest.raises(ValueError, match="only the validated.*bic-d"):
        adapter._recipe_params(
            RunSpec(**{**base.__dict__, "extra": {"scoring_method": "k2"}}),
            dataset_spec,
            source_rows=20,
        )
    with pytest.raises(ValueError, match="requires n_jobs=1"):
        adapter._recipe_params(
            RunSpec(**{**base.__dict__, "extra": {"n_jobs": 2}}),
            dataset_spec,
            source_rows=20,
        )
    with pytest.raises(ValueError, match="num_bins must be an integer"):
        adapter._recipe_params(
            RunSpec(**{**base.__dict__, "extra": {"num_bins": 2.5}}),
            dataset_spec,
            source_rows=20,
        )
    with pytest.raises(ValueError, match="equivalent_sample_size must be numeric"):
        adapter._recipe_params(
            RunSpec(**{**base.__dict__, "extra": {"equivalent_sample_size": "invalid"}}),
            dataset_spec,
            source_rows=20,
        )


def test_bn_safe_preprocessing_state_restores_without_fitting(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    payload = {
        "num_bins": 3,
        "numerical_columns": ["value", "constant"],
        "categorical_columns": ["segment", "target"],
        "bin_edges": {"value": [1.0, 2.5, 4.0]},
        "constant_values": {"constant": 3.25},
        "categorical_levels": {"segment": ["a", "b"], "target": ["no", "yes"]},
        "discrete_state_names": {
            "value": ["0", "1"],
            "constant": ["0"],
            "segment": ["a", "b"],
            "target": ["no", "yes"],
        },
        "quantile_method": "averaged_inverted_cdf",
        "subsample": None,
    }
    preprocessor = BNPreprocessor.from_payload(dataset_spec, payload)
    restored = preprocessor.inverse_transform(
        pd.DataFrame(
            {
                "value": ["0", "1"],
                "constant": ["0", "0"],
                "segment": ["a", "b"],
                "target": ["no", "yes"],
            }
        ),
        seed=19,
    )

    assert restored.columns.tolist() == dataset_spec.column_names
    assert restored["value"].between(1.0, 4.0).all()
    assert (restored["constant"] == 3.25).all()
    assert restored[["segment", "target"]].astype(str).to_dict("list") == {
        "segment": ["a", "b"],
        "target": ["no", "yes"],
    }


def test_bn_safe_json_checkpoint_can_be_external_without_unsafe_override(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    checkpoint = tmp_path / "reviewed" / "model.bn.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}\n", encoding="utf-8")
    checkpoint.with_name("model.bn.json.metadata.json").write_text("{}\n", encoding="utf-8")
    config = ExperimentConfig(
        model="bn",
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


def test_bn_case_gate_requires_every_exact_comparison() -> None:
    comparisons = {
        "preprocessing_exact": True,
        "discrete_training_exact": True,
        "graph_edges_exact": True,
        "cpds_exact": True,
        "restored_model_exact": True,
        "raw_discrete_sample_exact": True,
        "sample_bytes_exact": True,
        "adapter_manifests_valid": True,
        "checkpoint_metadata_valid": True,
        "sample_metadata_valid": True,
        "native_global_numpy_state_unchanged": True,
        "adapter_global_numpy_state_unchanged": True,
        "model_state": {
            "safe_json_checkpoint": True,
            "row_level_training_data_absent": True,
            "privacy_not_overclaimed": True,
            "artifact_access_control_required": True,
        },
        "samples": {
            "rows": bn_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "numeric_ranges_valid": True,
            "constant_numeric_exact": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }
    assert bn_validation._case_passed(comparisons) is True
    comparisons["cpds_exact"] = False
    assert bn_validation._case_passed(comparisons) is False


def test_bn_authoritative_environment_rejects_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bn_validation.importlib.metadata,
        "version",
        lambda name: bn_validation.EXPECTED_DISTRIBUTION_VERSIONS[name],
    )
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "platform", lambda: "Windows-test")
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    with pytest.raises(RuntimeError, match="requires Linux"):
        bn_validation._verify_environment()
