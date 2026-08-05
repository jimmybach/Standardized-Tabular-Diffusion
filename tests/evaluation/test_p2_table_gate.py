from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.table import (
    TableValidationError,
    load_table,
    validate_tables,
)

pytestmark = [pytest.mark.evaluation]


def test_gate_reorders_to_profile_and_canonicalizes_types(adult_profile, adult_frames) -> None:
    reference, synthetic = adult_frames
    reversed_columns = list(reversed(reference.columns))
    result = validate_tables(
        reference[reversed_columns],
        synthetic[reversed_columns],
        adult_profile.payload,
        expected_synthetic_rows=20,
    )
    expected = adult_profile.payload["table_contract"]["canonical_column_order"]
    assert list(result.reference.columns) == expected
    assert result.reference["age"].dtype == "int64"
    assert str(result.reference["workclass"].dtype) == "string"
    assert result.report["status"] == "passed"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda frame: frame.drop(columns=["age"]), "schema_mismatch"),
        (lambda frame: frame.assign(unknown=1), "schema_mismatch"),
        (lambda frame: frame.assign(age=17.5), "type_mismatch"),
        (lambda frame: frame.assign(age=float("inf")), "nonfinite_values"),
        (lambda frame: frame.assign(workclass=pd.NA), "missing_values_prohibited"),
    ],
)
def test_gate_fails_closed_with_stable_reason(adult_profile, adult_frames, mutation, reason) -> None:
    reference, synthetic = adult_frames
    with pytest.raises(TableValidationError) as caught:
        validate_tables(reference, mutation(synthetic), adult_profile.payload, expected_synthetic_rows=20)
    assert caught.value.reason_code == reason


def test_gate_rejects_duplicate_dataframe_columns(adult_profile, adult_frames) -> None:
    reference, synthetic = adult_frames
    names = list(synthetic.columns)
    names[-1] = names[0]
    synthetic.columns = names
    with pytest.raises(TableValidationError) as caught:
        validate_tables(reference, synthetic, adult_profile.payload, expected_synthetic_rows=20)
    assert caught.value.reason_code == "duplicate_columns"


def test_gate_rejects_wrong_row_count(adult_profile, adult_frames) -> None:
    reference, synthetic = adult_frames
    with pytest.raises(TableValidationError) as caught:
        validate_tables(reference, synthetic.iloc[:-1], adult_profile.payload, expected_synthetic_rows=20)
    assert caught.value.reason_code == "row_count_mismatch"


def test_gate_preserves_large_exact_integers(adult_profile, adult_frames) -> None:
    reference, synthetic = adult_frames
    exact = 2**53 + 1
    reference = reference.assign(age=exact)
    synthetic = synthetic.assign(age=exact)
    result = validate_tables(reference, synthetic, adult_profile.payload, expected_synthetic_rows=20)
    assert result.reference["age"].iloc[0] == exact
    assert result.reference["age"].dtype == "int64"


def test_gate_rejects_integer_outside_canonical_int64(adult_profile, adult_frames) -> None:
    reference, synthetic = adult_frames
    synthetic = synthetic.assign(age=2**63)
    with pytest.raises(TableValidationError) as caught:
        validate_tables(reference, synthetic, adult_profile.payload, expected_synthetic_rows=20)
    assert caught.value.reason_code == "type_mismatch"


def test_csv_duplicate_header_is_detected_before_pandas_mangling(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(TableValidationError) as caught:
        load_table(path)
    assert caught.value.reason_code == "duplicate_columns"


def test_parquet_input_is_supported(tmp_path: Path, adult_profile, adult_frames) -> None:
    pytest.importorskip("pyarrow")
    reference, synthetic = adult_frames
    real_path = tmp_path / "real.parquet"
    synthetic_path = tmp_path / "synthetic.parquet"
    reference.to_parquet(real_path, index=False)
    synthetic.to_parquet(synthetic_path, index=False)
    result = validate_tables(real_path, synthetic_path, adult_profile.payload, expected_synthetic_rows=20)
    assert len(result.reference) == len(result.synthetic) == 20
