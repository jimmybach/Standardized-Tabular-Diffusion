from __future__ import annotations

import copy
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import standardized_tabular_diffusion.official_datasets as official_datasets
from standardized_tabular_diffusion.dataset_sources import DatasetSource
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile
from standardized_tabular_diffusion.official_datasets import (
    OfficialDatasetError,
    load_sick_build_spec,
    materialize_official_sick,
    parse_uci_sick_file,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]

NEGATIVE_ROW = "41,F,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,1.3,t,2.5,t,125,t,1.14,t,109,f,?,SVHC,negative.|3733"
SICK_ROW = "80,F,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,2.2,t,0.6,t,80,t,0.7,t,115,f,?,SVI,sick.|1367"
TEST_ROW = "63,M,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,3.5,t,2.5,t,108,t,0.96,t,113,f,?,SVI,negative.|2059"


def _payload(*rows: str) -> bytes:
    return ("\n".join(rows) + "\n").encode("ascii")


def _summary(payload: bytes, build_spec: dict[str, object]) -> dict[str, object]:
    lines = payload.decode("ascii").splitlines()
    raw_columns = build_spec["raw_columns"]
    assert isinstance(raw_columns, list)
    pattern = build_spec["split_suffix_pattern"]
    assert isinstance(pattern, str)
    labels: list[str] = []
    identifiers: list[str] = []
    missing = Counter()
    for line in lines:
        row = next(csv.reader([line]))
        suffix = re.fullmatch(pattern, row[-1])
        assert suffix is not None
        labels.append(suffix.group(1))
        identifiers.append(suffix.group(2))
        for column, value in zip(raw_columns, row, strict=True):
            if value == "?":
                missing[column] += 1
    return {
        "rows": len(lines),
        "class_counts": dict(sorted(Counter(labels).items())),
        "record_ids_sha256": hashlib.sha256(("\n".join(identifiers) + "\n").encode("ascii")).hexdigest(),
        "missing_counts": dict(missing),
    }


def _fixture_build_spec(train_payload: bytes, test_payload: bytes, names_payload: bytes) -> dict[str, object]:
    build_spec = copy.deepcopy(load_sick_build_spec())
    build_spec["source_version"] = "fixture-sick-v1"
    build_spec["source_members"] = {
        "sick.data": hashlib.sha256(train_payload).hexdigest(),
        "sick.names": hashlib.sha256(names_payload).hexdigest(),
        "sick.test": hashlib.sha256(test_payload).hexdigest(),
    }
    build_spec["splits"] = {
        "train": {"member": "sick.data", **_summary(train_payload, build_spec)},
        "test": {"member": "sick.test", **_summary(test_payload, build_spec)},
    }
    parsed_train = official_datasets._parse_sick_payload(train_payload, build_spec, "train").frame
    parsed_test = official_datasets._parse_sick_payload(test_payload, build_spec, "test").frame
    model_columns = [column for column in build_spec["raw_columns"] if column != "TBG"]
    audit = official_datasets._duplicate_audit(parsed_train[model_columns], parsed_test[model_columns])
    build_spec["duplicate_audit"] = {"raw_model_view": audit, "processed_model_view": audit}
    return build_spec


def _fixture_source() -> DatasetSource:
    return DatasetSource(
        dataset_id="sick",
        dataset_view="sick",
        source_version="fixture-sick-v1",
        publisher="Fixture",
        canonical_page="https://example.test/sick",
        retrieval_url="https://example.test/sick.zip",
        retrieved_date="2026-08-03",
        archive_name="sick.zip",
        archive_format="zip",
        sha256="0" * 64,
        required_members=("sick.data", "sick.names", "sick.test"),
        max_download_bytes=10_000,
        max_extracted_bytes=10_000,
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        citation="Fixture.",
        redistribution_status="permitted",
    )


def test_sick_build_spec_preserves_official_schema_and_excludes_only_all_missing_tbg() -> None:
    build_spec = load_sick_build_spec()
    model_view = build_spec["model_view"]

    assert len(build_spec["raw_columns"]) == 30
    assert model_view["target_columns"] == ["Class"]
    assert set(model_view["excluded_columns"]) == {"TBG"}
    assert build_spec["splits"]["train"]["rows"] == 2800
    assert build_spec["splits"]["test"]["rows"] == 972
    assert build_spec["splits"]["train"]["missing_counts"]["TBG"] == 2800
    assert build_spec["splits"]["test"]["missing_counts"]["TBG"] == 972
    assert build_spec["duplicate_audit"]["raw_model_view"] == {
        "train_duplicate_rows": 47,
        "test_duplicate_rows": 3,
        "cross_split_unique_rows": 11,
        "train_rows_in_cross_split_overlap": 20,
        "test_rows_in_cross_split_overlap": 13,
    }


def test_reviewed_sick_profile_binds_official_source_but_does_not_overclaim_release_eligibility() -> None:
    profile = load_dataset_profile(REPO_ROOT / "configs" / "datasets" / "sick-uci-102-v1.json")

    assert profile.dataset_id == "sick"
    assert profile.payload["status"] == "reviewed"
    assert profile.payload["official_eligible"] is False
    assert profile.payload["split"]["train"]["rows"] == 2800
    assert profile.payload["split"]["test"]["rows"] == 972
    assert profile.payload["privacy"]["status"] == "review-required"


def test_repository_sick_metadata_points_only_to_the_official_model_view() -> None:
    spec = get_dataset_spec("sick", repo_root=REPO_ROOT)

    assert spec.extra["dataset_view"] == "sick-uci-102-model-v1"
    assert spec.extra["source_version"] == "uci-102-sha256-a0982569a744"
    assert spec.extra["train_num"] == 2800
    assert spec.extra["test_num"] == 972
    assert spec.extra["excluded_raw_columns"] == {
        "TBG": "The official train and test files contain no observed TBG value; no train-fitted statistic exists."
    }
    assert "TBG" not in spec.column_names
    assert spec.train_data_path == REPO_ROOT / "TabDiff-main" / "data" / "sick" / "train.csv"
    assert spec.test_data_path == REPO_ROOT / "TabDiff-main" / "data" / "sick" / "test.csv"


def test_parse_sick_file_separates_class_and_record_id(tmp_path: Path) -> None:
    train_payload = _payload(NEGATIVE_ROW, SICK_ROW)
    test_payload = _payload(TEST_ROW)
    names_payload = b"; fixture\n"
    build_spec = _fixture_build_spec(train_payload, test_payload, names_payload)
    train_path = tmp_path / "sick.data"
    train_path.write_bytes(train_payload)

    parsed = parse_uci_sick_file(train_path, split_name="train", build_spec=build_spec)

    assert parsed.record_ids == ("3733", "1367")
    assert parsed.frame["Class"].tolist() == ["negative", "sick"]
    assert parsed.frame["TBG"].tolist() == ["?", "?"]
    expected_summary = dict(build_spec["splits"]["train"])
    expected_summary.pop("member")
    assert parsed.summary == expected_summary


def test_parse_sick_file_rejects_malformed_record_suffix(tmp_path: Path) -> None:
    malformed_payload = _payload(NEGATIVE_ROW.replace("negative.|3733", "negative"))
    test_payload = _payload(TEST_ROW)
    names_payload = b"; fixture\n"
    build_spec = _fixture_build_spec(_payload(NEGATIVE_ROW), test_payload, names_payload)
    build_spec["source_members"]["sick.data"] = hashlib.sha256(malformed_payload).hexdigest()
    path = tmp_path / "sick.data"
    path.write_bytes(malformed_payload)

    with pytest.raises(OfficialDatasetError, match="invalid class/record suffix"):
        parse_uci_sick_file(path, split_name="train", build_spec=build_spec)


def test_materialize_official_sick_builds_compatible_outputs_without_tbg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_payload = _payload(NEGATIVE_ROW, SICK_ROW)
    test_payload = _payload(TEST_ROW)
    names_payload = b"; fixture\n"
    build_spec = _fixture_build_spec(train_payload, test_payload, names_payload)
    extracted = tmp_path / "cache" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "sick.data").write_bytes(train_payload)
    (extracted / "sick.test").write_bytes(test_payload)
    (extracted / "sick.names").write_bytes(names_payload)
    source = _fixture_source()
    source_manifest = {"dataset": "sick", "source_version": source.source_version}

    monkeypatch.setattr(official_datasets, "load_sick_build_spec", lambda: build_spec)
    monkeypatch.setattr(official_datasets, "get_dataset_source", lambda _dataset_id: source)
    monkeypatch.setattr(
        official_datasets,
        "fetch_dataset_source",
        lambda *_args, **_kwargs: {"extraction": {"extracted_path": str(extracted), "manifest": source_manifest}},
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    manifest = materialize_official_sick(repo_root=repo_root)

    primary = repo_root / "TabDiff-main" / "data" / "sick"
    train = pd.read_csv(primary / "train.csv")
    test = pd.read_csv(primary / "test.csv")
    x_num = np.load(primary / "X_num_train.npy", allow_pickle=False)
    x_cat = np.load(primary / "X_cat_train.npy", allow_pickle=False)
    targets = np.load(primary / "y_train.npy", allow_pickle=False)
    assert manifest["materialized_by"] == "official-uci-builder"
    assert manifest["dataset_view"] == "sick-uci-102-model-v1"
    assert len(train) == 2 and len(test) == 1
    assert "TBG" not in train.columns
    assert not train.isna().any().any()
    assert not test.isna().any().any()
    assert x_num.shape == (2, 6)
    assert x_cat.shape == (2, 22)
    assert targets.shape == (2, 1)
    assert (repo_root / "TabSyn-main" / "data" / "sick" / "train.csv").is_file()
    assert (repo_root / "TabDiff-main" / "synthetic" / "sick" / "real.csv").is_file()
    assert (repo_root / "materialized_datasets" / "sick" / "manifest.json").is_file()
