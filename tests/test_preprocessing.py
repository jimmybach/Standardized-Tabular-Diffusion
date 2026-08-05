from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.preprocessing import (
    MissingTargetError,
    MissingValuePolicy,
    PreprocessingError,
    SplitSchemaError,
    UndefinedImputationStatisticError,
    fit_imputation_state,
    load_imputation_state,
    preprocess_split_files,
    preprocess_splits,
    transform_with_imputation_state,
)

pytestmark = pytest.mark.integration


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 40, "?"],
            "city": ["NY", "CA", "NY"],
            "target": [0, 1, 0],
        }
    )


def test_imputation_statistics_are_fitted_on_train_only() -> None:
    train = _train_frame()
    validation = pd.DataFrame({"age": [1000, None], "city": [None, "TX"], "target": [1, 0]})
    test = pd.DataFrame({"age": [None], "city": [None], "target": [1]})

    result = preprocess_splits(
        train,
        validation=validation,
        test=test,
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
    )

    assert result.state.fitted_on_split == "train"
    assert result.state.numerical_fill_values == {"age": 30.0}
    assert result.state.categorical_fill_values == {"city": "NY"}
    assert result.train["age"].tolist() == [20.0, 40.0, 30.0]
    assert result.validation is not None
    assert result.validation["age"].tolist() == [1000.0, 30.0]
    assert result.test is not None
    assert result.test.loc[0, "age"] == 30.0
    assert result.test.loc[0, "city"] == "NY"
    assert result.reports["test"]["values_imputed"] == 2


def test_categorical_mode_ties_have_a_deterministic_unicode_order() -> None:
    train = pd.DataFrame({"x": [1, 2], "category": ["z", "a"], "target": [0, 1]})

    state = fit_imputation_state(
        train,
        numerical_columns=["x"],
        categorical_columns=["category"],
        target_columns=["target"],
    )

    assert state.categorical_fill_values == {"category": "a"}

    canonically_equivalent = pd.DataFrame({"x": [1, 2], "category": ["é", "e\u0301"], "target": [0, 1]})
    reversed_state = fit_imputation_state(
        canonically_equivalent.iloc[::-1].reset_index(drop=True),
        numerical_columns=["x"],
        categorical_columns=["category"],
        target_columns=["target"],
    )
    forward_state = fit_imputation_state(
        canonically_equivalent,
        numerical_columns=["x"],
        categorical_columns=["category"],
        target_columns=["target"],
    )
    assert forward_state.categorical_fill_values == reversed_state.categorical_fill_values


@pytest.mark.parametrize("split_name", ["train", "validation", "test"])
def test_missing_targets_are_never_imputed(split_name: str) -> None:
    train = _train_frame()
    validation = train.copy()
    test = train.copy()
    frame = {"train": train, "validation": validation, "test": test}[split_name]
    frame.loc[0, "target"] = None

    with pytest.raises(MissingTargetError, match="never imputed"):
        preprocess_splits(
            train,
            validation=validation,
            test=test,
            numerical_columns=["age"],
            categorical_columns=["city"],
            target_columns=["target"],
        )


@pytest.mark.parametrize("column,column_type", [("age", "numerical"), ("city", "categorical")])
def test_entirely_missing_training_features_fail_closed(column: str, column_type: str) -> None:
    train = _train_frame()
    train[column] = None

    with pytest.raises(UndefinedImputationStatisticError, match=f"{column!r} is entirely missing"):
        preprocess_splits(
            train,
            numerical_columns=["age"],
            categorical_columns=["city"],
            target_columns=["target"],
        )


def test_invalid_numerical_values_are_not_misclassified_as_missing() -> None:
    train = _train_frame()
    train.loc[0, "age"] = "not-a-number"

    with pytest.raises(PreprocessingError, match="non-numeric values"):
        preprocess_splits(
            train,
            numerical_columns=["age"],
            categorical_columns=["city"],
            target_columns=["target"],
        )


def test_split_schema_and_column_order_must_match_training() -> None:
    train = _train_frame()
    test = train[["city", "age", "target"]]

    with pytest.raises(SplitSchemaError, match="column order differs"):
        preprocess_splits(
            train,
            test=test,
            numerical_columns=["age"],
            categorical_columns=["city"],
            target_columns=["target"],
        )


def test_missing_indicators_are_stable_for_every_feature() -> None:
    test = pd.DataFrame({"age": [None], "city": ["NY"], "target": [1]})
    result = preprocess_splits(
        _train_frame(),
        test=test,
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
        policy=MissingValuePolicy(add_missing_indicators=True),
    )

    assert result.test is not None
    assert list(result.test.columns) == ["age", "city", "target", "age__missing", "city__missing"]
    assert result.test.loc[0, "age__missing"] == 1
    assert result.test.loc[0, "city__missing"] == 0


def test_file_workflow_writes_portable_audit_artifacts(tmp_path: Path) -> None:
    train_path = tmp_path / "source" / "train.csv"
    test_path = tmp_path / "source" / "test.csv"
    train_path.parent.mkdir()
    _train_frame().to_csv(train_path, index=False)
    pd.DataFrame({"age": [None], "city": [None], "target": [1]}).to_csv(test_path, index=False)

    result = preprocess_split_files(
        train_path=train_path,
        test_path=test_path,
        output_dir=tmp_path / "processed",
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
    )

    output_dir = tmp_path / "processed"
    manifest = json.loads((output_dir / "preprocessing-manifest.json").read_text(encoding="utf-8"))
    state = json.loads((output_dir / "imputation-state.json").read_text(encoding="utf-8"))
    transformed_test = pd.read_csv(output_dir / "test.csv")

    assert result["train_only_fitting"] is True
    assert manifest["fitted_on_split"] == "train"
    assert manifest["inputs"]["train"]["filename"] == "train.csv"
    assert "source" not in json.dumps(manifest["inputs"])
    assert len(manifest["state"]["sha256"]) == 64
    assert len(manifest["transformed_schema"]["fingerprint"]) == 64
    assert manifest["identity"]["dataset_view_token"].startswith("imputed-")
    assert manifest["identity"]["policy_change_requires_new_dataset_view"] is True
    assert state["numerical_fill_values"] == {"age": 30.0}
    assert transformed_test.loc[0, "age"] == 30.0
    assert transformed_test.loc[0, "city"] == "NY"


def test_policy_rejects_target_and_synthetic_repair_modes() -> None:
    with pytest.raises(PreprocessingError, match="Target imputation"):
        MissingValuePolicy(target_strategy="mean")
    with pytest.raises(PreprocessingError, match="Generated samples"):
        MissingValuePolicy(synthetic_strategy="impute")


def test_policy_or_transformed_schema_change_creates_a_new_dataset_view_identity(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _train_frame().to_csv(train_path, index=False)

    plain = preprocess_split_files(
        train_path=train_path,
        output_dir=tmp_path / "plain",
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
    )
    indicators = preprocess_split_files(
        train_path=train_path,
        output_dir=tmp_path / "indicators",
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
        policy=MissingValuePolicy(add_missing_indicators=True),
    )

    assert plain["identity"]["dataset_view_token"] != indicators["identity"]["dataset_view_token"]
    assert plain["transformed_schema"]["fingerprint"] != indicators["transformed_schema"]["fingerprint"]


def test_portable_imputation_state_loads_safely_and_reproduces_transform(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _train_frame().to_csv(train_path, index=False)
    manifest = preprocess_split_files(
        train_path=train_path,
        output_dir=tmp_path / "processed",
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
    )
    state_path = tmp_path / "processed" / "imputation-state.json"

    restored = load_imputation_state(state_path, expected_sha256=manifest["state"]["sha256"])
    transformed, report = transform_with_imputation_state(
        pd.DataFrame({"age": [None], "city": [None], "target": [1]}),
        restored,
        split_name="future-test",
    )

    assert restored.fingerprint == manifest["state"]["fingerprint"]
    assert transformed.loc[0, "age"] == 30.0
    assert transformed.loc[0, "city"] == "NY"
    assert report["values_imputed"] == 2
    with pytest.raises(PreprocessingError, match="checksum"):
        load_imputation_state(state_path, expected_sha256="0" * 64)


def test_imputation_state_rejects_tampered_repair_policy(tmp_path: Path) -> None:
    state = fit_imputation_state(
        _train_frame(),
        numerical_columns=["age"],
        categorical_columns=["city"],
        target_columns=["target"],
    ).to_dict()
    state["policy"]["synthetic_strategy"] = "impute"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(PreprocessingError, match="Generated samples"):
        load_imputation_state(path)


def test_preprocessing_refuses_to_overwrite_a_nonempty_output_directory(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _train_frame().to_csv(train_path, index=False)
    output = tmp_path / "processed"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new or empty"):
        preprocess_split_files(
            train_path=train_path,
            output_dir=output,
            numerical_columns=["age"],
            categorical_columns=["city"],
            target_columns=["target"],
        )

    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"
