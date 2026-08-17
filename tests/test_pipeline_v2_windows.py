from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.config import (
    EvaluationConfig,
    ExperimentConfig,
    SampleConfig,
    TrainConfig,
    load_experiment_config,
)
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.registry import get_adapter_spec, list_adapter_specs
from standardized_tabular_diffusion.runner import _action_identity, build_run_context
from standardized_tabular_diffusion.runtime_contracts import validate_action_controls
from standardized_tabular_diffusion.validation.pipeline_v2_windows import (
    COPY_EXCLUSIONS,
    PipelineV2Error,
    _assert_manifest_unchanged,
    _copy_training_artifacts,
    _load_plan,
    _review_pip_check,
    _verify_environment_lock,
    materialize_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "configs/validation/pipeline-v2-windows-v1.json"


def test_v2_plan_covers_the_complete_runtime_inventory() -> None:
    plan = _load_plan(PLAN_PATH)
    central = plan["central_evaluation"]
    assert central["protocol"] == "p3-validity"
    assert (REPO_ROOT / central["environment_lock"]).is_file()
    assert central["required_packages"] == {
        "jsonschema": "4.23.0",
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "pyarrow": "18.1.0",
        "PyYAML": "6.0.2",
    }
    planned = {row["model_id"] for row in plan["models"]}
    passed = {row["model_id"] for row in plan["already_passed"]}
    blocked = {row["model_id"] for row in plan["external_blocks"]}
    assert len(planned) == 18
    assert planned | passed | blocked == set(list_adapter_specs())
    assert not (planned & passed or planned & blocked or passed & blocked)
    for row in plan["models"]:
        config_path = REPO_ROOT / row["config_path"]
        assert config_path.is_file()
        assert (REPO_ROOT / row["environment_lock"]).is_file()
        assert row["device"] in {"cpu", "cuda"}
        assert isinstance(row["required_packages"], dict)
        assert isinstance(row.get("pip_check_waivers", []), list)
        config = load_experiment_config(config_path)
        assert config.model == row["model_id"]
        validate_action_controls(config.model, "train", config.train.extra)
        validate_action_controls(config.model, "sample", config.sample.extra)
    by_model = {row["model_id"]: row for row in plan["models"]}
    for model_id in ("nflow", "ctab-gan", "ctab-gan-plus", "realtabformer"):
        assert by_model[model_id]["batch"] == "legacy-cpu"
        assert by_model[model_id]["device"] == "cpu"
        assert by_model[model_id]["required_packages"]["torch"] == "2.3.0"
    assert by_model["goggle"]["batch"] == "neural-gpu"
    assert by_model["goggle"]["device"] == "cuda"
    assert by_model["goggle"]["required_packages"] == {"torch": "2.8.0"}
    for model_id in ("codi", "stasy", "tabsyn"):
        assert by_model[model_id]["required_packages"]["libzero"] == "0.0.8"
        assert len(by_model[model_id]["pip_check_waivers"]) == 1


def test_v2_adapter_metadata_is_strict_json_serializable() -> None:
    payload = get_adapter_spec("arf").to_dict("arf")
    assert json.loads(json.dumps(payload, allow_nan=False))["model_id"] == "arf"


def _write_source_fixture(repo_root: Path) -> None:
    root = repo_root / "TabSyn-main" / "data" / "adult"
    root.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "number": list(range(20)),
            "category": ["a", "b", "c", "d"] * 5,
            "label": ["no", "yes"] * 10,
        }
    )
    test = pd.DataFrame(
        {
            "number": list(range(100, 112)),
            "category": ["a", "b", "c", "d"] * 3,
            "label": ["yes", "no"] * 6,
        }
    )
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    for split, frame in (("train", train), ("test", test)):
        np.save(root / f"X_num_{split}.npy", frame[["number"]].to_numpy(dtype=np.float32), allow_pickle=False)
        np.save(root / f"X_cat_{split}.npy", frame[["category"]].to_numpy(dtype=str), allow_pickle=False)
        np.save(root / f"y_{split}.npy", frame[["label"]].to_numpy(dtype=str), allow_pickle=False)
    (root / "info.json").write_text(
        json.dumps(
            {
                "name": "adult",
                "dataset_view": "adult",
                "column_names": ["number", "category", "label"],
                "num_col_idx": [0],
                "cat_col_idx": [1],
                "target_col_idx": [2],
                "int_col_idx": [0],
                "int_columns": ["number"],
                "task_type": "binclass",
                "n_classes": 2,
                "train_num": 20,
                "test_num": 12,
                "val_num": 0,
                "data_path": "data/adult/train.csv",
                "test_path": "data/adult/test.csv",
                "val_path": None,
                "column_info": {
                    "number": {"type": "numerical", "min": 0.0, "max": 19.0},
                    "category": {"type": "categorical", "categories": ["a", "b", "c", "d"]},
                    "label": {"type": "categorical", "categories": ["no", "yes"]},
                },
            }
        ),
        encoding="utf-8",
    )


def test_v2_fixture_is_deterministic_bound_and_reusable(tmp_path: Path) -> None:
    _write_source_fixture(tmp_path)
    plan = {
        "fixture": {
            "fixture_id": "pipeline-v2-adult-test",
            "source_dataset": "adult",
            "train_rows": 12,
            "test_rows": 8,
            "selection_seed": 7,
            "selection_policy": "test",
        }
    }
    spec = materialize_fixture(tmp_path, plan, tmp_path / "work")
    reused = materialize_fixture(tmp_path, plan, tmp_path / "work")
    assert spec.to_dict() == reused.to_dict()
    assert len(pd.read_csv(spec.train_data_path)) == 12
    assert len(pd.read_csv(spec.test_data_path)) == 8
    assert set(pd.read_csv(spec.train_data_path)["category"]) == {"a", "b", "c", "d"}
    native = tmp_path / "TabSyn-main" / "data" / spec.name
    assert (native / "info.json").read_bytes() == spec.metadata_path.read_bytes()
    assert np.array_equal(
        np.load(native / "X_num_train.npy", allow_pickle=False),
        np.load(spec.metadata_path.parent / "X_num_train.npy", allow_pickle=False),
    )


def test_v2_training_artifact_copy_preserves_bytes_and_rejects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "train"
    destination = tmp_path / "sample"
    source.mkdir()
    destination.mkdir()
    (source / "checkpoint").mkdir()
    (source / "checkpoint" / "model.bin").write_bytes(b"authoritative-checkpoint")
    for root in (source, destination):
        (root / ".standardized-run-identity.json").write_text("{}", encoding="utf-8")
        (root / "artifacts.json").write_text("{}", encoding="utf-8")
        (root / "run_context.json").write_text("{}", encoding="utf-8")
    manifest = _copy_training_artifacts(source, destination)
    assert set(manifest) == {"checkpoint/model.bin"}
    assert not (destination / "artifact_bundle.json").exists()
    assert COPY_EXCLUSIONS >= {".standardized-run-identity.json", "artifacts.json", "run_context.json"}
    _assert_manifest_unchanged(destination, manifest)
    (destination / "checkpoint" / "model.bin").write_bytes(b"mutated")
    with pytest.raises(PipelineV2Error, match="mutated"):
        _assert_manifest_unchanged(destination, manifest)


def test_v2_environment_lock_checks_versions_and_cuda_local_suffix(tmp_path: Path) -> None:
    lock = tmp_path / "requirements-validation.txt"
    lock.write_text(
        "numpy==1.26.4\nignored==9.9.9; python_version < '3.0'\n",
        encoding="utf-8",
    )
    environment = {
        "packages": {
            "NumPy": "1.26.4",
            "torch": "2.8.0+cu128",
        }
    }
    result = _verify_environment_lock(lock, environment, {"torch": "2.8.0"})
    assert result["status"] == "pass"
    assert result["packages"]["torch"]["observed"] == "2.8.0+cu128"

    with pytest.raises(PipelineV2Error, match="does not satisfy"):
        _verify_environment_lock(lock, {"packages": {"numpy": "2.0.0"}}, {})

    with pytest.raises(PipelineV2Error, match="runtime package mismatch"):
        _verify_environment_lock(lock, environment, {"torch": "2.7.0"})


def test_v2_pip_check_only_accepts_an_exact_exercised_reviewed_waiver() -> None:
    line = "libzero 0.0.8 has requirement torch<2,>=1.7, but you have torch 2.8.0+cu128.\n"
    waiver = {
        "distribution": "libzero",
        "version": "0.0.8",
        "requirement": "torch<2,>=1.7",
        "installed_dependency": "torch",
        "installed_dependency_version": "2.8.0+cu128",
        "reason": "Reviewed stale metadata bound.",
    }
    result = _review_pip_check(
        returncode=1,
        stdout=line,
        stderr="",
        packages={"libzero": "0.0.8", "torch": "2.8.0+cu128"},
        waivers=[waiver],
    )
    assert result["status"] == "pass-with-reviewed-waiver"
    assert result["conflicts"][0]["requirement"] == "torch<2,>=1.7"

    modern_success = _review_pip_check(
        returncode=0,
        stdout="No broken requirements found.\n",
        stderr="",
        packages={},
        waivers=[],
    )
    assert modern_success == {
        "status": "pass",
        "output": ["No broken requirements found."],
        "conflicts": [],
        "reviewed_waivers": [],
    }

    with pytest.raises(PipelineV2Error, match="lacks one exact reviewed waiver"):
        _review_pip_check(
            returncode=1,
            stdout=line.replace("2.8.0+cu128", "2.8.1+cu128"),
            stderr="",
            packages={"libzero": "0.0.8", "torch": "2.8.1+cu128"},
            waivers=[waiver],
        )
    with pytest.raises(PipelineV2Error, match="not exercised"):
        _review_pip_check(
            returncode=0,
            stdout="",
            stderr="",
            packages={"libzero": "0.0.8", "torch": "2.8.0+cu128"},
            waivers=[waiver],
        )


def test_runner_accepts_a_name_matched_explicit_validation_dataset(tmp_path: Path) -> None:
    metadata = tmp_path / "info.json"
    train = tmp_path / "train.csv"
    metadata.write_text("{}", encoding="utf-8")
    train.write_text("x,label\n1,no\n2,yes\n", encoding="utf-8")
    dataset = DatasetSpec(
        name="pipeline-v2-test",
        task_type="classification",
        column_names=["x", "label"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["label"],
        metadata_path=metadata,
        train_data_path=train,
        extra={"column_info": {"x": "int", "label": "str"}},
    )
    config = ExperimentConfig(
        model="ctgan",
        dataset=dataset.name,
        output_dir=str(tmp_path / "output"),
        train=TrainConfig(enabled=False),
        sample=SampleConfig(enabled=False),
        evaluation=EvaluationConfig(enabled=False),
    )
    context = build_run_context(config, repo_root=tmp_path, dataset_spec=dataset)
    assert context["dataset_spec"]["name"] == dataset.name
    mismatched = DatasetSpec(**{**dataset.__dict__, "name": "different"})
    with pytest.raises(ValueError, match="does not match"):
        build_run_context(config, repo_root=tmp_path, dataset_spec=mismatched)


def test_runner_action_identity_uses_the_experiment_upstream_config(tmp_path: Path) -> None:
    metadata = tmp_path / "info.json"
    train = tmp_path / "train.csv"
    upstream = tmp_path / "upstream.toml"
    metadata.write_text("{}", encoding="utf-8")
    train.write_text("x,label\n1,no\n2,yes\n", encoding="utf-8")
    upstream.write_text("seed = 13\n", encoding="utf-8")
    dataset = DatasetSpec(
        name="pipeline-v2-test",
        task_type="classification",
        column_names=["x", "label"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["label"],
        metadata_path=metadata,
        train_data_path=train,
    )
    config = ExperimentConfig(
        model="ctgan",
        dataset=dataset.name,
        output_dir=str(tmp_path / "output"),
        upstream_config_path=str(upstream),
        train=TrainConfig(enabled=True),
        sample=SampleConfig(enabled=True, num_samples=2),
        evaluation=EvaluationConfig(enabled=False),
    )
    for action in ("train", "sample"):
        identity = _action_identity(config, action, dataset)
        assert identity["upstream_config"]["path"] == str(upstream.resolve())
        assert identity["upstream_config"]["sha256"]
