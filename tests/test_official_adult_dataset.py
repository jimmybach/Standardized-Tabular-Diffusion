from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import standardized_tabular_diffusion.official_datasets as official_datasets
from standardized_tabular_diffusion import materialization
from standardized_tabular_diffusion.dataset_sources import DatasetSource
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile
from standardized_tabular_diffusion.official_datasets import (
    OfficialDatasetError,
    load_adult_build_spec,
    materialize_official_adult,
    parse_uci_adult_file,
)
from standardized_tabular_diffusion.preprocessing import MissingValuePolicy, preprocess_splits

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_ROW_1 = (
    "25, Private, 226802, Some-college, 10, Never-married, Prof-specialty, Not-in-family, "
    "White, Female, 0, 0, 40, United-States, <=50K"
)
TRAIN_ROW_2 = (
    "45, Private, 83311, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, "
    "Asian-Pac-Islander, Female, 0, 0, 50, United-States, >50K"
)
TRAIN_ROW_MISSING = "39, ?, 77516, HS-grad, 9, Never-married, ?, Own-child, Black, Male, 0, 0, 20, ?, <=50K"
TEST_ROW_MISSING = "38, ?, 89814, HS-grad, 9, Married-civ-spouse, ?, Husband, White, Male, 0, 0, 50, ?, <=50K."


def _payload(*rows: str, test: bool = False) -> bytes:
    header = "|1x3 Cross validator\n" if test else ""
    return (header + "\n".join(rows) + "\n\n").encode("ascii")


def _fixture_build_spec(
    train_payload: bytes,
    test_payload: bytes,
    auxiliary_payloads: dict[str, bytes],
) -> dict[str, object]:
    build_spec = copy.deepcopy(load_adult_build_spec())
    build_spec["source_version"] = "fixture-adult-v1"
    all_payloads = {
        "adult.data": train_payload,
        "adult.test": test_payload,
        **auxiliary_payloads,
    }
    build_spec["source_members"] = {name: hashlib.sha256(payload).hexdigest() for name, payload in all_payloads.items()}
    parsed_train = official_datasets._parse_adult_payload(train_payload, build_spec, "train")
    parsed_test = official_datasets._parse_adult_payload(test_payload, build_spec, "test")
    build_spec["splits"] = {
        "train": {"member": "adult.data", **parsed_train.summary},
        "test": {"member": "adult.test", **parsed_test.summary},
    }
    raw_audit = official_datasets._duplicate_audit(parsed_train.frame, parsed_test.frame)
    model_view = build_spec["model_view"]
    transformed = preprocess_splits(
        parsed_train.frame,
        test=parsed_test.frame,
        numerical_columns=model_view["numerical_columns"],
        categorical_columns=model_view["categorical_columns"],
        target_columns=model_view["target_columns"],
        policy=MissingValuePolicy(missing_markers=("?",)),
    )
    assert transformed.test is not None
    processed_audit = official_datasets._duplicate_audit(transformed.train, transformed.test)
    build_spec["duplicate_audit"] = {
        "raw_model_view": raw_audit,
        "processed_model_view": processed_audit,
    }
    return build_spec


def _fixture_source() -> DatasetSource:
    return DatasetSource(
        dataset_id="adult",
        dataset_view="adult",
        source_version="fixture-adult-v1",
        publisher="Fixture",
        canonical_page="https://example.test/adult",
        retrieval_url="https://example.test/adult.zip",
        retrieved_date="2026-08-03",
        archive_name="adult.zip",
        archive_format="zip",
        sha256="0" * 64,
        required_members=("Index", "adult.data", "adult.names", "adult.test", "old.adult.names"),
        max_download_bytes=100_000,
        max_extracted_bytes=100_000,
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Fixture.",
        redistribution_status="permitted",
    )


def test_adult_build_spec_locks_official_split_schema_missingness_and_overlap() -> None:
    build_spec = load_adult_build_spec()

    assert len(build_spec["raw_columns"]) == 15
    assert build_spec["model_view"]["target_columns"] == ["income"]
    assert build_spec["model_view"]["excluded_columns"] == {}
    assert build_spec["splits"]["train"]["rows"] == 32561
    assert build_spec["splits"]["test"]["rows"] == 16281
    assert build_spec["splits"]["train"]["missing_counts"] == {
        "native.country": 583,
        "occupation": 1843,
        "workclass": 1836,
    }
    assert build_spec["preprocessing"]["expected_categorical_fill_values"] == {
        "native.country": "United-States",
        "occupation": "Prof-specialty",
        "workclass": "Private",
    }
    assert build_spec["duplicate_audit"]["raw_model_view"]["cross_split_unique_rows"] == 23
    assert build_spec["duplicate_audit"]["processed_model_view"]["cross_split_unique_rows"] == 24


def test_reviewed_adult_profile_does_not_overclaim_release_eligibility() -> None:
    profile = load_dataset_profile(REPO_ROOT / "configs" / "datasets" / "adult-uci-2-v1.json")

    assert profile.dataset_id == "adult"
    assert profile.payload["status"] == "reviewed"
    assert profile.payload["official_eligible"] is False
    assert profile.payload["split"]["train"]["rows"] == 32561
    assert profile.payload["split"]["test"]["rows"] == 16281
    assert profile.payload["privacy"]["status"] == "review-required"


def test_repository_adult_metadata_points_only_to_the_official_model_view() -> None:
    spec = get_dataset_spec("adult", repo_root=REPO_ROOT)

    assert spec.extra["dataset_view"] == "adult-uci-2-model-v1"
    assert spec.extra["source_version"] == "uci-2-sha256-7537312dd56c"
    assert spec.extra["train_num"] == 32561
    assert spec.extra["test_num"] == 16281
    assert spec.extra["excluded_raw_columns"] == {}
    assert spec.train_data_path == REPO_ROOT / "TabDiff-main" / "data" / "adult" / "train.csv"
    assert spec.test_data_path == REPO_ROOT / "TabDiff-main" / "data" / "adult" / "test.csv"


def test_parse_adult_test_normalizes_only_the_registered_target_suffix(tmp_path: Path) -> None:
    train_payload = _payload(TRAIN_ROW_1, TRAIN_ROW_2, TRAIN_ROW_MISSING)
    test_payload = _payload(TEST_ROW_MISSING, test=True)
    auxiliary = {"Index": b"index\n", "adult.names": b"names\n", "old.adult.names": b"old\n"}
    build_spec = _fixture_build_spec(train_payload, test_payload, auxiliary)
    test_path = tmp_path / "adult.test"
    test_path.write_bytes(test_payload)

    parsed = parse_uci_adult_file(test_path, split_name="test", build_spec=build_spec)

    assert parsed.frame["income"].tolist() == ["<=50K"]
    assert parsed.frame["workclass"].tolist() == ["?"]
    assert parsed.summary == {key: value for key, value in build_spec["splits"]["test"].items() if key != "member"}


def test_parse_adult_file_rejects_noncanonical_delimiter(tmp_path: Path) -> None:
    train_payload = _payload(TRAIN_ROW_1, TRAIN_ROW_2, TRAIN_ROW_MISSING)
    malformed = train_payload.replace(b"25, Private", b"25,Private", 1)
    test_payload = _payload(TEST_ROW_MISSING, test=True)
    auxiliary = {"Index": b"index\n", "adult.names": b"names\n", "old.adult.names": b"old\n"}
    build_spec = _fixture_build_spec(train_payload, test_payload, auxiliary)
    build_spec["source_members"]["adult.data"] = hashlib.sha256(malformed).hexdigest()
    path = tmp_path / "adult.data"
    path.write_bytes(malformed)

    with pytest.raises(OfficialDatasetError, match="comma-space format"):
        parse_uci_adult_file(path, split_name="train", build_spec=build_spec)


def test_generic_materializer_routes_adult_to_the_official_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"dataset": "adult", "materialized_by": "official-uci-builder"}

    monkeypatch.setattr(official_datasets, "materialize_official_adult", fake_builder)

    result = materialization.materialize_dataset(
        "adult",
        repo_root=tmp_path,
        cache_root=tmp_path / "cache",
        refresh=True,
        timeout_seconds=12,
    )

    assert result["materialized_by"] == "official-uci-builder"
    assert observed == {
        "repo_root": tmp_path,
        "cache_root": tmp_path / "cache",
        "refresh": True,
        "timeout_seconds": 12,
    }


def test_materialize_official_adult_builds_train_fitted_compatible_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_payload = _payload(TRAIN_ROW_1, TRAIN_ROW_2, TRAIN_ROW_MISSING)
    test_payload = _payload(TEST_ROW_MISSING, test=True)
    auxiliary = {"Index": b"index\n", "adult.names": b"names\n", "old.adult.names": b"old\n"}
    build_spec = _fixture_build_spec(train_payload, test_payload, auxiliary)
    extracted = tmp_path / "cache" / "extracted"
    extracted.mkdir(parents=True)
    payloads = {"adult.data": train_payload, "adult.test": test_payload, **auxiliary}
    for name, payload in payloads.items():
        (extracted / name).write_bytes(payload)
    source = _fixture_source()
    source_manifest = {"dataset": "adult", "source_version": source.source_version}

    monkeypatch.setattr(official_datasets, "load_adult_build_spec", lambda: build_spec)
    monkeypatch.setattr(official_datasets, "get_dataset_source", lambda _dataset_id: source)
    monkeypatch.setattr(
        official_datasets,
        "fetch_dataset_source",
        lambda *_args, **_kwargs: {"extraction": {"extracted_path": str(extracted), "manifest": source_manifest}},
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    manifest = materialize_official_adult(repo_root=repo_root)

    primary = repo_root / "TabDiff-main" / "data" / "adult"
    train = pd.read_csv(primary / "train.csv")
    test = pd.read_csv(primary / "test.csv")
    x_num = np.load(primary / "X_num_train.npy", allow_pickle=False)
    x_cat = np.load(primary / "X_cat_train.npy", allow_pickle=False)
    targets = np.load(primary / "y_train.npy", allow_pickle=False)
    state = official_datasets.read_json(primary / "imputation-state.json")
    info = official_datasets.read_json(primary / "info.json")
    assert manifest["materialized_by"] == "official-uci-builder"
    assert manifest["dataset_view"] == "adult-uci-2-model-v1"
    assert len(train) == 3 and len(test) == 1
    assert not train.isna().any().any()
    assert not test.isna().any().any()
    assert test.loc[0, ["workclass", "occupation", "native.country"]].tolist() == [
        "Private",
        "Prof-specialty",
        "United-States",
    ]
    expected_fill_values = build_spec["preprocessing"]["expected_categorical_fill_values"]
    assert {column: state["categorical_fill_values"][column] for column in expected_fill_values} == expected_fill_values
    assert info["int_columns"] == build_spec["model_view"]["integer_columns"]
    assert x_num.shape == (3, 6)
    assert x_cat.shape == (3, 8)
    assert targets.shape == (3, 1)
    assert (repo_root / "TabSyn-main" / "data" / "adult" / "train.csv").is_file()
    assert (repo_root / "TabDiff-main" / "synthetic" / "adult" / "real.csv").is_file()
    assert (repo_root / "materialized_datasets" / "adult" / "manifest.json").is_file()
