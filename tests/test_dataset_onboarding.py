from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standardized_tabular_diffusion.dataset_onboarding import process_registered_dataset, register_dataset
from standardized_tabular_diffusion.datasets import get_dataset_spec


def _init_repo_layout(repo_root: Path) -> None:
    (repo_root / "TabDiff-main" / "data" / "Info").mkdir(parents=True)
    (repo_root / "TabSyn-main").mkdir(parents=True)


def test_register_dataset_writes_registry_files_and_copies_raw_csv(tmp_path: Path) -> None:
    _init_repo_layout(tmp_path)
    raw_csv_path = tmp_path / "customer_churn.csv"
    pd.DataFrame(
        {
            "age": [22, 35, 48],
            "state": ["NY", "CA", "NY"],
            "churned": [0, 1, 0],
        }
    ).to_csv(raw_csv_path, index=False)

    payload = register_dataset(
        dataset_name="customer_churn",
        raw_csv_path=raw_csv_path,
        task_type="classification",
        target_column="churned",
        categorical_columns=["state"],
        numerical_columns=["age"],
        repo_root=tmp_path,
    )

    info_path = tmp_path / "TabDiff-main" / "data" / "Info" / "customer_churn.json"
    saved_info = json.loads(info_path.read_text())

    assert payload["dataset"] == "customer_churn"
    assert Path(payload["raw_data_path"]).exists()
    assert Path(payload["upload_copy_path"]).exists()
    assert saved_info["task_type"] == "binclass"
    assert saved_info["column_names"] == ["age", "state", "churned"]
    assert saved_info["num_col_idx"] == [0]
    assert saved_info["cat_col_idx"] == [1]
    assert saved_info["target_col_idx"] == [2]
    assert saved_info["data_path"] == "data/customer_churn/raw.csv"


def test_register_dataset_infers_pandas_string_dtype_as_categorical(tmp_path: Path) -> None:
    _init_repo_layout(tmp_path)
    raw_csv_path = tmp_path / "stringy.csv"
    frame = pd.DataFrame(
        {
            "age": pd.Series([22, 35, 48], dtype="int64"),
            "state": pd.Series(["NY", "CA", "NY"], dtype="string"),
            "churned": pd.Series([0, 1, 0], dtype="int64"),
        }
    )
    frame.to_csv(raw_csv_path, index=False)

    payload = register_dataset(
        dataset_name="stringy",
        raw_csv_path=raw_csv_path,
        task_type="classification",
        target_column="churned",
        repo_root=tmp_path,
    )

    assert payload["numerical_columns"] == ["age"]
    assert payload["categorical_columns"] == ["state"]


def test_process_registered_dataset_builds_manifest_and_syncs_tabsyn(tmp_path: Path, monkeypatch) -> None:
    _init_repo_layout(tmp_path)
    raw_csv_path = tmp_path / "housing.csv"
    frame = pd.DataFrame(
        {
            "sqft": [800, 900, 1100, 1300],
            "zip_code": ["10001", "10001", "94105", "94105"],
            "price": [400000, 450000, 900000, 1100000],
        }
    )
    frame.to_csv(raw_csv_path, index=False)

    register_dataset(
        dataset_name="housing",
        raw_csv_path=raw_csv_path,
        task_type="regression",
        target_column="price",
        categorical_columns=["zip_code"],
        numerical_columns=["sqft"],
        repo_root=tmp_path,
    )

    def fake_run_python(args: list[str], cwd: Path) -> None:
        assert args == ["process_dataset.py", "--dataname", "housing"]
        assert cwd == tmp_path / "TabDiff-main"
        data_dir = tmp_path / "TabDiff-main" / "data" / "housing"
        synth_dir = tmp_path / "TabDiff-main" / "synthetic" / "housing"
        data_dir.mkdir(parents=True, exist_ok=True)
        synth_dir.mkdir(parents=True, exist_ok=True)

        train_df = frame.iloc[:3].copy()
        test_df = frame.iloc[3:].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)
        train_df.to_csv(synth_dir / "real.csv", index=False)
        test_df.to_csv(synth_dir / "test.csv", index=False)

        info = json.loads((tmp_path / "TabDiff-main" / "data" / "Info" / "housing.json").read_text())
        info["train_num"] = len(train_df)
        info["test_num"] = len(test_df)
        (data_dir / "info.json").write_text(json.dumps(info, indent=2))

    monkeypatch.setattr("standardized_tabular_diffusion.dataset_onboarding._run_python", fake_run_python)

    manifest = process_registered_dataset("housing", repo_root=tmp_path)
    manifest_file = tmp_path / "materialized_datasets" / "housing" / "manifest.json"

    assert manifest["dataset"] == "housing"
    assert manifest["materialized_by"] == "local-registration"
    assert manifest_file.exists()
    assert (tmp_path / "TabSyn-main" / "data" / "housing" / "train.csv").exists()
    assert (tmp_path / "TabSyn-main" / "synthetic" / "housing" / "real.csv").exists()

    dataset_spec = get_dataset_spec("housing", repo_root=tmp_path)
    assert dataset_spec.train_data_path == tmp_path / "TabDiff-main" / "data" / "housing" / "train.csv"
    assert dataset_spec.test_data_path == tmp_path / "TabDiff-main" / "data" / "housing" / "test.csv"


def test_register_dataset_sanitizes_missing_numeric_markers(tmp_path: Path) -> None:
    _init_repo_layout(tmp_path)
    raw_csv_path = tmp_path / "sick_like.csv"
    pd.DataFrame(
        {
            "age": ["21", "?", "42"],
            "sex": ["F", "M", None],
            "TSH": ["1.2", " ?", "3.4"],
            "Class": ["negative", "negative", "positive"],
        }
    ).to_csv(raw_csv_path, index=False)

    payload = register_dataset(
        dataset_name="sick_like",
        raw_csv_path=raw_csv_path,
        task_type="classification",
        target_column="Class",
        repo_root=tmp_path,
    )

    cleaned_frame = pd.read_csv(tmp_path / "TabDiff-main" / "data" / "sick_like" / "raw.csv")

    assert payload["cleaning_report"]["input_rows"] == 3
    assert payload["cleaning_report"]["output_rows"] == 2
    assert payload["cleaning_report"]["dropped_missing_numerical_rows"] == 1
    assert cleaned_frame.shape[0] == 2
    assert cleaned_frame["TSH"].dtype.kind in {"f", "i"}
    assert "__missing__" in set(cleaned_frame["sex"].astype(str))
