from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, SampleConfig, TrainConfig
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.structured_baselines import NFlowAdapter, NFlowPreprocessor
from standardized_tabular_diffusion.runner import validate_action_inputs
from standardized_tabular_diffusion.validation import nflow as nflow_validation

pytestmark = pytest.mark.adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "nflow" / "native-parity-run-30970260840.json"
EVIDENCE_SHA256 = "940be2b0668baf990d640040544a4f16c7cccd9e9f6df7d0f7a582e8d2999923"


def _dataset_spec(tmp_path: Path, *, task_type: str = "classification") -> DatasetSpec:
    target = ["no", "yes", "no", "yes"] if task_type == "classification" else [1.0, 2.0, 3.0, 4.0]
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
        name="nflow-test",
        task_type=task_type,
        column_names=list(frame.columns),
        numerical_columns=["value", "constant"],
        categorical_columns=["segment"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )


def test_nflow_protocol_constants_lock_the_official_release_and_declared_recipe() -> None:
    assert nflow_validation.PROTOCOL_ID == "nflows-maf-tabular-recipe-parity-v1"
    assert nflow_validation.PACKAGE_VERSION == "0.14"
    assert nflow_validation.SDIST_FILENAME == "nflows-0.14.tar.gz"
    assert nflow_validation.SDIST_SHA256 == (
        "6299844a62f9999fcdf2d95cb2d01c091a50136bd17826e303aba646b2d11b55"
    )
    assert nflow_validation.UPSTREAM_COMMIT == "64b856c081e5f07521b32be99da262e8338fbfe8"
    assert nflow_validation.UPSTREAM_TREE == "83057958f8773e35044e3aa5c13ac9c06c4a3994"
    assert nflow_validation.LICENSE_EXPRESSION == "MIT"
    assert nflow_validation.EXPECTED_PACKAGE_FILES == 42
    assert nflow_validation.SEED_CASES == (0, 19, 73)
    assert nflow_validation.VARIANTS == ("binary", "multiclass", "regression")
    assert nflow_validation.RECIPE["transform_order"] == (
        "random-permutation-then-masked-affine-autoregressive"
    )


def test_nflow_retained_evidence_is_exact_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")

    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "nflows-maf-tabular-recipe-parity-v1"
    assert evidence["reproduction_target"] == "official-nflows-package-plus-declared-tabular-maf-recipe"
    assert evidence["repository_commit"] == "37e9234f7160cd4c9f2f6c1ae2beb7eef96aaf35"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["distributions"]["nflows"] == "0.14"
    assert evidence["environment"]["distributions"]["torch"] == "2.3.0+cpu"
    assert evidence["environment"]["platform"].startswith("Linux-")
    assert evidence["environment_lock"]["sha256"] == (
        "a05b0c1a8c3f14ec2c285038ede6c9d61c327400ce0986aa05361c970fbbc319"
    )
    assert evidence["source"]["authority"] == "canonical-library"
    assert evidence["source"]["source_distribution"]["archive_members_verified"] == 96
    assert evidence["source"]["source_distribution"]["package_files_verified"] == 42
    assert evidence["source"]["source_distribution"]["license_file_in_sdist"] is False
    assert evidence["source"]["installed_distribution"]["record_hashes_verified"] == 48
    assert evidence["source_unchanged_after_validation"] is True
    assert len(evidence["cases"]) == 9
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(case["comparisons"]["preprocessing_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["losses_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["state_tensors_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["raw_samples_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["sample_bytes_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["safe_json_numpy_checkpoint"] for case in evidence["cases"])
    assert all(case["comparisons"]["row_level_training_data_absent"] for case in evidence["cases"])
    assert all(case["comparisons"]["privacy_not_overclaimed"] for case in evidence["cases"])


def test_nflow_preprocessor_safe_payload_round_trip_is_exact(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    frame = pd.read_csv(dataset_spec.train_data_path)
    preprocessor = NFlowPreprocessor(dataset_spec)
    transformed = preprocessor.fit_transform(frame)
    payload = json.loads(json.dumps(preprocessor.to_payload(), allow_nan=False))
    restored = NFlowPreprocessor.from_payload(dataset_spec, payload)

    assert transformed.dtype == np.float32
    assert np.array_equal(preprocessor.inverse_transform(transformed), restored.inverse_transform(transformed))
    assert payload["numerical_scale"][1] == 1.0
    assert payload["categorical_levels"] == {"segment": ["a", "b"], "target": ["no", "yes"]}


def test_nflow_adapter_loads_declared_types_and_rejects_missing_or_nonfinite_values(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    adapter = NFlowAdapter(tmp_path)
    frame = adapter._load_training_frame(dataset_spec)
    assert list(frame.columns) == dataset_spec.column_names

    damaged = frame.copy()
    damaged.loc[0, "value"] = np.nan
    damaged.to_csv(dataset_spec.train_data_path, index=False)
    with pytest.raises(ValueError, match="does not accept missing values"):
        adapter._load_training_frame(dataset_spec)

    damaged.loc[0, "value"] = np.inf
    damaged.to_csv(dataset_spec.train_data_path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        adapter._load_training_frame(dataset_spec)


def test_nflow_recipe_contract_fails_closed(tmp_path: Path) -> None:
    adapter = NFlowAdapter(tmp_path)
    base = {"model": "nflow", "dataset": "nflow-test", "output_dir": tmp_path, "seed": 0}
    recipe = adapter._recipe_params(RunSpec(**base))
    assert recipe["num_layers"] == 4
    assert recipe["hidden_features"] == 64
    assert recipe["num_blocks"] == 2
    assert recipe["num_threads"] == 1

    with pytest.raises(ValueError, match="num_layers"):
        adapter._recipe_params(RunSpec(**base, extra={"num_layers": 0}))
    with pytest.raises(ValueError, match="positive and finite"):
        adapter._recipe_params(RunSpec(**base, extra={"learning_rate": float("nan")}))
    with pytest.raises(ValueError, match="num_threads=1"):
        adapter._recipe_params(RunSpec(**base, extra={"num_threads": 2}))
    with pytest.raises(ValueError, match="transform_order"):
        adapter._recipe_params(RunSpec(**base, extra={"transform_order": "maf-then-permutation"}))
    with pytest.raises(ValueError, match="seed"):
        adapter._validate_seed(-1)


def test_nflow_safe_checkpoint_can_be_external_without_unsafe_override(tmp_path: Path) -> None:
    dataset_spec = _dataset_spec(tmp_path)
    checkpoint = tmp_path / "reviewed" / "model.nflow.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}\n", encoding="utf-8")
    checkpoint.with_name("model.nflow.json.metadata.json").write_text("{}\n", encoding="utf-8")
    checkpoint.with_name("model.nflow.weights.npz").write_bytes(b"stub")
    config = ExperimentConfig(
        model="nflow",
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
    assert readiness["checked"]["checkpoint_weights_path"].endswith("model.nflow.weights.npz")


def test_nflow_case_gate_requires_every_exact_comparison() -> None:
    comparisons = {
        "preprocessing_exact": True,
        "losses_exact": True,
        "state_tensor_names_exact": True,
        "state_tensors_exact": True,
        "raw_samples_exact": True,
        "sample_frame_exact": True,
        "sample_bytes_exact": True,
        "adapter_manifests_valid": True,
        "checkpoint_metadata_valid": True,
        "safe_json_numpy_checkpoint": True,
        "row_level_training_data_absent": True,
        "privacy_not_overclaimed": True,
        "artifact_access_control_required": True,
        "direct_global_torch_state_unchanged": True,
        "adapter_global_torch_state_unchanged": True,
        "thread_count_unchanged": True,
        "rows": nflow_validation.EXPECTED_SAMPLE_ROWS,
        "columns_exact": True,
        "missing_values": 0,
        "finite_numerical": True,
        "categorical_domains_valid": True,
    }
    assert nflow_validation._case_passed(comparisons) is True
    comparisons["state_tensors_exact"] = False
    assert nflow_validation._case_passed(comparisons) is False


def test_nflow_authoritative_environment_rejects_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="requires Linux"):
        nflow_validation._verify_environment()
