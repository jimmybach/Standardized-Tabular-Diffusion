from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.tabularargn import TabularARGNAdapter

pytestmark = pytest.mark.adapter


class _FakeTabularARGN:
    instances: list[_FakeTabularARGN] = []
    fitted_frames: list[pd.DataFrame] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.workspace = Path(kwargs["workspace_dir"])
        self.__class__.instances.append(self)

    def fit(self, frame: pd.DataFrame) -> _FakeTabularARGN:
        self.__class__.fitted_frames.append(frame.copy())
        model_data = self.workspace / "ModelStore" / "model-data"
        target_stats = self.workspace / "ModelStore" / "tgt-stats"
        model_data.mkdir(parents=True)
        target_stats.mkdir(parents=True)
        (model_data / "model-configs.json").write_text('{"model_units":"S"}', encoding="utf-8")
        (model_data / "model-weights.pt").write_bytes(b"weights-only-state-dict")
        (model_data / "optimizer.pt").write_bytes(b"training-only")
        (target_stats / "stats.json").write_text('{"is_sequential":false}', encoding="utf-8")
        original = self.workspace / "OriginalData" / "tgt-data"
        original.mkdir(parents=True)
        (original / "part.000000.parquet").write_bytes(b"raw-training-rows")
        return self


class _FakeEngine:
    seeds: list[int] = []
    generation_calls: list[dict[str, Any]] = []
    generated = pd.DataFrame()

    @classmethod
    def set_random_state(cls, seed: int) -> None:
        cls.seeds.append(seed)

    @classmethod
    def generate(cls, **kwargs: Any) -> None:
        cls.generation_calls.append(dict(kwargs))
        rows = int(kwargs["sample_size"])
        cls.generated = pd.DataFrame(
            {
                "value": [float(index) for index in range(rows)],
                "group": ["a" if index % 2 == 0 else "b" for index in range(rows)],
                "target": ["yes" if index % 2 == 0 else "no" for index in range(rows)],
            }
        )


def _load_generated_data(_: Path) -> pd.DataFrame:
    return _FakeEngine.generated.copy()


def _package_record(tmp_path: Path) -> dict[str, Any]:
    return {
        "distribution_root": str(tmp_path / "site-packages"),
        "package_version": "2.6.2",
        "wheel_sha256": "wheel-sha",
        "manifest_sha256": "manifest-sha",
        "upstream_commit": "0b96f02e4fad47c7c19c985fda4311230e20bbb5",
    }


def _dataset(tmp_path: Path, *, missing: bool = False, name: str = "fixture") -> DatasetSpec:
    train_path = tmp_path / f"{name}-train.csv"
    rows = "value,group,target\n1.0,a,yes\n2.0,b,no\n3.0,a,yes\n4.0,b,no\n"
    if missing:
        rows = "value,group,target\n1.0,a,yes\n,b,no\n"
    train_path.write_text(rows, encoding="utf-8")
    metadata_path = tmp_path / f"{name}-dataset.json"
    metadata_path.write_text("{}", encoding="utf-8")
    return DatasetSpec(
        name=name,
        task_type="classification",
        column_names=["value", "group", "target"],
        numerical_columns=["value"],
        categorical_columns=["group"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )


@pytest.fixture(autouse=True)
def _reset_fake_runtime() -> None:
    _FakeTabularARGN.instances.clear()
    _FakeTabularARGN.fitted_frames.clear()
    _FakeEngine.seeds.clear()
    _FakeEngine.generation_calls.clear()
    _FakeEngine.generated = pd.DataFrame()


def _patch_runtime(adapter: TabularARGNAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "_import_runtime",
        lambda: (_FakeEngine, _FakeTabularARGN, _load_generated_data, _package_record(tmp_path)),
    )


def test_tabularargn_train_and_sample_use_integrity_checked_official_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    output = tmp_path / "output"
    adapter.train(
        RunSpec(
            model="tabularargn",
            dataset=dataset.name,
            output_dir=output,
            device="cpu",
            seed=19,
            extra={
                "dataset_spec": dataset.to_dict(),
                "model": "MOSTLY_AI/Small",
                "max_epochs": 2,
                "batch_size": 4,
                "max_train_rows": 3,
            },
        )
    )

    model = _FakeTabularARGN.instances[0]
    assert model.kwargs["random_state"] == 19
    assert model.kwargs["model"] == "MOSTLY_AI/Small"
    assert model.kwargs["max_epochs"] == 2.0
    assert model.workspace.resolve().is_relative_to(output.resolve())
    assert len(_FakeTabularARGN.fitted_frames[0]) == 3
    workspace = output / adapter.workspace_name
    assert not (workspace / "OriginalData").exists()
    assert not (workspace / "ModelStore" / "model-data" / "optimizer.pt").exists()
    assert not (output / "tabularargn.pkl").exists()

    metadata = json.loads((output / adapter.metadata_filename).read_text(encoding="utf-8"))
    assert metadata["raw_or_encoded_training_rows_retained"] is False
    assert metadata["source_rows"] == 4
    assert metadata["training_rows"] == 3
    assert metadata["package"]["wheel_sha256"] == "wheel-sha"

    bundle = adapter.sample(
        RunSpec(
            model="tabularargn",
            dataset=dataset.name,
            output_dir=output,
            device="cpu",
            seed=73,
            num_samples=3,
            extra={"dataset_spec": dataset.to_dict(), "batch_size": 2},
        )
    )
    assert bundle.generated_sample_path == output / "samples.csv"
    assert pd.read_csv(bundle.generated_sample_path).shape == (3, 3)
    assert _FakeEngine.seeds == [73]
    call = _FakeEngine.generation_calls[-1]
    assert call["rare_category_replacement_method"] == "SAMPLE"
    assert call["sample_size"] == 3
    assert Path(call["workspace_dir"]) == workspace.resolve()


def test_tabularargn_defaults_match_official_flat_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    adapter.train(
        RunSpec(
            model="tabularargn",
            dataset=dataset.name,
            output_dir=tmp_path / "output",
            extra={"dataset_spec": dataset.to_dict()},
        )
    )
    constructor = _FakeTabularARGN.instances[0].kwargs
    assert constructor["model"] is None
    assert constructor["max_training_time"] == 14400.0
    assert constructor["max_epochs"] == 100.0
    assert constructor["batch_size"] is None
    assert constructor["gradient_accumulation_steps"] is None
    assert constructor["enable_flexible_generation"] is True
    assert constructor["value_protection"] is True
    assert constructor["tgt_encoding_types"] is None
    assert constructor["verbose"] == 0


def test_tabularargn_rejects_missing_values_and_invalid_controls(tmp_path: Path) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    missing = _dataset(tmp_path, missing=True)
    with pytest.raises(ValueError, match="centralized train-only mean/mode imputer"):
        adapter.train(
            RunSpec(
                model="tabularargn",
                dataset=missing.name,
                output_dir=tmp_path / "missing-output",
                extra={"dataset_spec": missing.to_dict()},
            )
        )

    dataset = _dataset(tmp_path, name="controls")
    with pytest.raises(ValueError, match="Unknown TabularARGN training controls"):
        adapter.train(
            RunSpec(
                model="tabularargn",
                dataset=dataset.name,
                output_dir=tmp_path / "control-output",
                extra={"dataset_spec": dataset.to_dict(), "max_epohs": 1},
            )
        )
    with pytest.raises(ValueError, match="encoding types are invalid"):
        adapter.train(
            RunSpec(
                model="tabularargn",
                dataset=dataset.name,
                output_dir=tmp_path / "encoding-output",
                extra={"dataset_spec": dataset.to_dict(), "tgt_encoding_types": {"group": "LANGUAGE_TEXT"}},
            )
        )


def test_tabularargn_prunes_original_rows_when_official_fit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingTabularARGN(_FakeTabularARGN):
        def fit(self, frame: pd.DataFrame) -> _FailingTabularARGN:
            super().fit(frame)
            raise RuntimeError("official training failure")

    adapter = TabularARGNAdapter(tmp_path)
    dataset = _dataset(tmp_path)
    output = tmp_path / "failed-output"
    monkeypatch.setattr(
        adapter,
        "_import_runtime",
        lambda: (_FakeEngine, _FailingTabularARGN, _load_generated_data, _package_record(tmp_path)),
    )
    with pytest.raises(RuntimeError, match="official training failure"):
        adapter.train(
            RunSpec(
                model="tabularargn",
                dataset=dataset.name,
                output_dir=output,
                extra={"dataset_spec": dataset.to_dict()},
            )
        )
    assert not (output / adapter.workspace_name / "OriginalData").exists()
    assert not (output / adapter.metadata_filename).exists()


def test_tabularargn_rejects_external_training_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    with pytest.raises(PermissionError, match="beneath output_dir"):
        adapter.train(
            RunSpec(
                model="tabularargn",
                dataset=dataset.name,
                output_dir=tmp_path / "output",
                checkpoint_path=tmp_path / "external-workspace",
                extra={"dataset_spec": dataset.to_dict()},
            )
        )


def test_tabularargn_detects_checkpoint_tampering_and_dataset_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    output = tmp_path / "output"
    common = {"dataset_spec": dataset.to_dict()}
    adapter.train(RunSpec(model="tabularargn", dataset=dataset.name, output_dir=output, extra=common))
    other = _dataset(tmp_path, name="other")
    with pytest.raises(RuntimeError, match="DatasetSpec differs"):
        adapter.sample(
            RunSpec(
                model="tabularargn",
                dataset=other.name,
                output_dir=output,
                num_samples=2,
                extra={"dataset_spec": other.to_dict()},
            )
        )

    weights = output / adapter.workspace_name / "ModelStore" / "model-data" / "model-weights.pt"
    weights.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="integrity manifest"):
        adapter.sample(
            RunSpec(
                model="tabularargn",
                dataset=dataset.name,
                output_dir=output,
                num_samples=2,
                extra=common,
            )
        )


def test_tabularargn_rejects_out_of_domain_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = TabularARGNAdapter(tmp_path)
    _patch_runtime(adapter, tmp_path, monkeypatch)
    dataset = _dataset(tmp_path)
    output = tmp_path / "output"
    common = {"dataset_spec": dataset.to_dict()}
    adapter.train(RunSpec(model="tabularargn", dataset=dataset.name, output_dir=output, extra=common))

    def generate_out_of_domain(**_: Any) -> None:
        _FakeEngine.generated = pd.DataFrame({"value": [1.0], "group": ["unseen"], "target": ["yes"]})

    monkeypatch.setattr(_FakeEngine, "generate", generate_out_of_domain)
    with pytest.raises(RuntimeError, match="out-of-domain values"):
        adapter.sample(
            RunSpec(
                model="tabularargn",
                dataset=dataset.name,
                output_dir=output,
                num_samples=1,
                extra=common,
            )
        )


def test_tabularargn_manifest_records_exact_official_release() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            repo_root / "standardized_tabular_diffusion" / "resources" / "upstream" / "tabularargn-wheel-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["package"]["version"] == "2.6.2"
    assert manifest["package"]["sha256"] == ("3ead3770c936919f8fce4e1f9fffd271ffdd490f0292c2ab9a42cb4bafe3caea")
    assert manifest["source"]["commit"] == "0b96f02e4fad47c7c19c985fda4311230e20bbb5"
    assert manifest["source"]["license"] == "Apache-2.0"
    assert len(manifest["installed_files"]) == manifest["package"]["record_hashed_files"] == 53
    assert manifest["wheel_source_comparison"]["exact_shared_source_files"] == 50
