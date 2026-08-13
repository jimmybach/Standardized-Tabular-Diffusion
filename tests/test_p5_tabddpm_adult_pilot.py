from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standardized_tabular_diffusion.validation import p5_tabddpm_adult_pilot as pilot

REPO_ROOT = Path(__file__).resolve().parents[1]


def _profile() -> dict:
    return json.loads((REPO_ROOT / pilot.DATASET_PROFILE).read_text(encoding="utf-8"))


def test_pilot_identity_is_exactly_one_model_one_dataset_three_seeds() -> None:
    assert pilot.MODEL_ID == "tabddpm"
    assert pilot.DATASET_ID == "adult"
    assert pilot._require_exact_seed_panel([0, 1, 2]) == (0, 1, 2)
    with pytest.raises(pilot.PilotError, match="frozen"):
        pilot._require_exact_seed_panel([0, 1])


def test_evidence_artifact_paths_are_portable_and_confined(tmp_path: Path) -> None:
    output_root = tmp_path / "pilot-output"

    assert pilot._artifact_relative_path(output_root / "seed-0" / "synthetic.csv", output_root) == (
        "seed-0/synthetic.csv"
    )
    with pytest.raises(pilot.PilotError, match="escaped"):
        pilot._artifact_relative_path(tmp_path / "outside.csv", output_root)


def test_declared_official_adult_config_and_profile_are_locked() -> None:
    base, profile = pilot._load_declared_inputs(REPO_ROOT)

    assert base["train"]["main"]["steps"] == 30000
    assert base["diffusion_params"]["num_timesteps"] == 100
    assert base["model_params"]["rtdl_params"]["d_layers"] == [256, 1024, 1024, 1024, 1024, 256]
    assert profile["split"]["train"]["rows"] == 32561
    assert profile["split"]["test"]["rows"] == 16281


def test_runtime_config_changes_only_declared_bindings(tmp_path: Path) -> None:
    base, _ = pilot._load_declared_inputs(REPO_ROOT)
    runtime = pilot._runtime_config(
        base,
        parent_dir=tmp_path / "checkpoint",
        data_dir=tmp_path / "data",
        sample_seed=2,
        num_samples=32561,
    )

    assert runtime["parent_dir"] == str(tmp_path / "checkpoint")
    assert runtime["real_data_path"] == str(tmp_path / "data")
    assert runtime["sample"]["seed"] == 2
    assert runtime["sample"]["num_samples"] == 32561
    assert runtime["train"] == base["train"]
    assert runtime["model_params"] == base["model_params"]
    assert runtime["diffusion_params"] == base["diffusion_params"]


def test_real_pilot_still_fails_closed_when_adult_data_is_absent(tmp_path: Path) -> None:
    with pytest.raises(pilot.PilotError, match="Required pilot data is missing"):
        pilot._prepare_upstream_data(tmp_path, tmp_path / "output", _profile())


def test_decode_sample_is_explicit_and_canonical(tmp_path: Path) -> None:
    profile = _profile()
    parent = tmp_path / "upstream"
    parent.mkdir()
    rows = 32561
    numerical = np.tile(np.array([[39.4, 77516.0, 13.0, 2174.0, 0.0, 40.6]]), (rows, 1))
    categorical = np.tile(
        np.array(
            [
                [
                    "State-gov",
                    "Bachelors",
                    "Never-married",
                    "Adm-clerical",
                    "Not-in-family",
                    "White",
                    "Male",
                    "United-States",
                ]
            ],
            dtype=str,
        ),
        (rows, 1),
    )
    np.save(parent / "X_num_train.npy", numerical, allow_pickle=False)
    np.save(parent / "X_cat_train.npy", categorical, allow_pickle=False)
    np.save(parent / "y_train.npy", np.zeros(rows, dtype=np.int64), allow_pickle=False)

    decoded = pilot._decode_sample(parent, tmp_path / "seed-0", profile, {"<=50K": 0, ">50K": 1})
    frame = pd.read_csv(decoded["synthetic_path"])

    assert list(frame.columns) == profile["table_contract"]["canonical_column_order"]
    assert len(frame) == rows
    assert frame.loc[0, "age"] == 39
    assert frame.loc[0, "hours.per.week"] == 41
    assert frame.loc[0, "income"] == "<=50K"
    assert decoded["integer_decoding"]["age"]["changed_rows"] == rows
    assert decoded["synthetic_repair_applied_by_evaluator"] is False
