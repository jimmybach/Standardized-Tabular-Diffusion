from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from standardized_tabular_diffusion.config import EvaluationConfig, ExperimentConfig, SampleConfig, TrainConfig
from standardized_tabular_diffusion.evaluation.bundle import BundleValidationReport
from standardized_tabular_diffusion.evaluation.service import EvaluationOutcome
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.runner import run_action, run_pipeline

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_public_adapter_evaluation_routes_to_central_bundle_without_legacy_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "sample.csv"
    train = tmp_path / "train.csv"
    profile = tmp_path / "profile.json"
    for path in (sample, train):
        path.write_text("x,target\n1,a\n", encoding="utf-8")
    profile.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    config = ExperimentConfig(
        model="tabdiff",
        dataset="fixture",
        output_dir=str(output),
        train=TrainConfig(enabled=False, seed=9),
        sample=SampleConfig(enabled=False, seed=17, num_samples=1),
        evaluation=EvaluationConfig(
            enabled=True,
            protocol="p3-validity",
            dataset_profile_path=str(profile),
            extra={"sample_path": str(sample)},
        ),
    )
    dataset = DatasetSpec(
        name="fixture",
        task_type="classification",
        column_names=["x", "target"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["target"],
        metadata_path=tmp_path / "metadata.json",
        train_data_path=train,
    )
    observed = {}

    def fake_evaluate_adapter_output(**kwargs):
        observed.update(kwargs)
        return EvaluationOutcome(
            request=SimpleNamespace(fingerprint="f" * 64),
            report=BundleValidationReport(
                root=Path(kwargs["output_dir"]),
                bundle_id="run-central-route",
                finalization_status="finalized",
                present_files=18,
                pending_files=0,
                not_applicable_files=2,
            ),
        )

    monkeypatch.setattr(
        "standardized_tabular_diffusion.evaluation.service.evaluate_adapter_output",
        fake_evaluate_adapter_output,
    )
    def reject_adapter_import(*args, **kwargs):
        raise AssertionError("Central evaluation must not import a model adapter")

    monkeypatch.setattr("standardized_tabular_diffusion.runner.get_adapter", reject_adapter_import)

    bundle = run_action(config, "evaluate", dataset_spec=dataset, repo_root=tmp_path)

    assert observed["model_id"] == "tabdiff"
    assert observed["synthetic_path"] == str(sample)
    assert observed["reference_path"] == train
    assert observed["protocol_id"] == "p3-validity"
    assert observed["generation_seed"] == 17
    assert observed["output_dir"] == output / "evaluation-result"
    assert bundle.evaluation_bundle_path == output / "evaluation-result"
    assert bundle.standardized_summary_path is None
    assert bundle.upstream_workdir == tmp_path.resolve() / "TabDiff-main"
    assert (output / "artifacts.json").is_file()
    assert not (output / "standardized_summary.json").exists()


def test_evaluate_only_pipeline_does_not_import_a_model_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "sample.csv"
    train = tmp_path / "train.csv"
    profile = tmp_path / "profile.json"
    for path in (sample, train):
        path.write_text("x,target\n1,a\n", encoding="utf-8")
    profile.write_text("{}", encoding="utf-8")
    config = ExperimentConfig(
        model="arf",
        dataset="fixture",
        output_dir=str(tmp_path / "run"),
        train=TrainConfig(enabled=False, seed=9),
        sample=SampleConfig(enabled=False, seed=17, num_samples=1),
        evaluation=EvaluationConfig(
            enabled=True,
            protocol="p3-validity",
            dataset_profile_path=str(profile),
            extra={"sample_path": str(sample)},
        ),
    )
    dataset = DatasetSpec(
        name="fixture",
        task_type="classification",
        column_names=["x", "target"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["target"],
        metadata_path=tmp_path / "metadata.json",
        train_data_path=train,
    )

    monkeypatch.setattr(
        "standardized_tabular_diffusion.runner.get_dataset_spec",
        lambda *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        "standardized_tabular_diffusion.runner.get_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Evaluate-only pipelines must not import a model adapter")
        ),
    )
    monkeypatch.setattr(
        "standardized_tabular_diffusion.runner.run_central_evaluation",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {"status": "pass"}),
    )

    result = run_pipeline(config, repo_root=tmp_path)

    assert result["phases"]["evaluate"] == {"status": "pass"}
