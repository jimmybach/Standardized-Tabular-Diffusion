from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    content_fingerprint,
)
from standardized_tabular_diffusion.evaluation.utility import _task_type
from standardized_tabular_diffusion.validation import p4_dataset_scale as validation
from standardized_tabular_diffusion.validation import p4_global_source

pytestmark = [pytest.mark.evaluation, pytest.mark.source_parity]


def test_preregistered_successor_binds_full_datasets_seeds_and_safety_limits() -> None:
    manifest = validation.validate_pilot_manifest()

    assert manifest["status"] == "preregistered-diagnostic"
    assert manifest["pilot_id"] == "p4-dataset-scale-windows-gpu-stable-candidate"
    assert manifest["pilot_version"] == "0.3.0"
    assert manifest["official_results_allowed"] is False
    assert manifest["amendments"] == []
    assert (manifest["evaluator_profile_id"], manifest["evaluator_profile_version"]) == (
        "p4-utility-stable",
        "0.2.0",
    )
    assert manifest["coverage"]["seed"] == 0
    assert manifest["stability"]["seeds"] == [0, 1, 2, 3, 4]
    assert manifest["surrogate"] == {
        "type": "full-row-multiset-preserving-permutation",
        "purpose": "Verify that equal row multisets produce equal fit boundaries and Global Utility results regardless of input row order.",
        "preserves_all_rows": True,
        "preserves_all_per-column_support": True,
        "generation_seed_source": "evaluator seed",
        "quality_result_publishable": False,
    }
    assert manifest["resources"]["autogluon_fit_time_limit_per_arm_seconds"] == 300
    assert manifest["resources"]["maximum_observed_process_tree_peak_rss_gib"] == 14.0
    assert manifest["environment"]["runtime_profile"] == "windows-rtx5080"
    assert manifest["environment"]["tabpfn_cpu_large_dataset_opt_in"] is None


def test_successor_keeps_the_fixed_scientific_gates_and_complete_schedule() -> None:
    manifest = validation.validate_pilot_manifest()

    assert manifest["stability"]["maximum_absolute_identity_ratio_deviation"] == 0.05
    assert manifest["stability"]["maximum_seed_ratio_range"] == 0.05
    assert manifest["resources"]["maximum_observed_cuda_peak_allocated_gib"] == 15.0
    assert len(validation._expected_task_keys(manifest)) == 67


def test_cuda_resource_gate_applies_only_when_tabpfn_is_protocol_applicable() -> None:
    limits = {"maximum_observed_cuda_peak_allocated_gib": 15.0}

    assert (
        validation._cuda_resource_failures(
            "trtr",
            {"cuda_peak_allocation_increase_bytes": 0},
            limits,
            cuda_required=False,
        )
        == []
    )
    assert validation._cuda_resource_failures(
        "trtr",
        {"cuda_peak_allocation_increase_bytes": 0},
        limits,
        cuda_required=True,
    ) == ["trtr-cuda-execution-not-proven"]
    assert validation._cuda_resource_failures(
        "tstr",
        {"cuda_peak_allocation_increase_bytes": 16 * 1024**3},
        limits,
        cuda_required=True,
    ) == ["tstr-cuda-peak-allocated"]


def test_schedule_covers_every_nonconstant_target_once_and_adds_only_preregistered_stability() -> None:
    manifest = validation.validate_pilot_manifest()
    expected_counts = {"adult": (15, 12), "sick": (28, 12)}

    all_keys: set[tuple[str, str, int]] = set()
    for dataset, (coverage_count, stability_count) in expected_counts.items():
        shard_count = manifest["coverage"]["datasets"][dataset]["shards"]
        coverage = [
            task
            for shard_index in range(shard_count)
            for task in validation.task_schedule(dataset, "coverage", shard_index, shard_count)
        ]
        stability = validation.task_schedule(dataset, "stability", 0, 1)
        assert len(coverage) == coverage_count
        assert len(stability) == stability_count
        assert {task["seed"] for task in coverage} == {0}
        assert {task["seed"] for task in stability} == {1, 2, 3, 4}
        for task in [*coverage, *stability]:
            key = (task["dataset"], task["target_column_id"], task["seed"])
            assert key not in all_keys
            all_keys.add(key)

    assert len(all_keys) == 67


def test_sick_constant_tbg_measured_is_retained_but_reasoned_out_of_global_utility() -> None:
    manifest = validation.validate_pilot_manifest()
    profile = validation._profile_for_dataset("sick", manifest).payload
    global_profile = profile["utility"]["global"]

    assert profile["dataset_profile_version"] == "1.5.0-reviewed"
    assert profile["compatibility"]["p4_dataset_scale"]["compatible_predecessor_versions"] == ["1.4.0-reviewed"]
    assert "TBG_measured" in profile["table_contract"]["canonical_column_order"]
    assert "tbg-measured" not in global_profile["included_target_column_ids"]
    assert global_profile["excluded_targets"] == [
        {
            "column_id": "tbg-measured",
            "reason_code": "constant-real-target",
            "reason_detail": (
                "TBG_measured is False for every row in both checksum-pinned official splits and therefore "
                "cannot define a predictive classification task."
            ),
        }
    ]


def test_full_row_permutation_is_deterministic_and_preserves_exact_multiset_and_support() -> None:
    frame = pd.DataFrame({"value": [1, 1, 2, 3, 5], "label": ["a", "b", "a", "c", "a"]})

    first = validation.full_row_permutation(frame, 7)
    repeated = validation.full_row_permutation(frame, 7)
    other = validation.full_row_permutation(frame, 11)

    pd.testing.assert_frame_equal(first, repeated)
    assert list(first.index) != list(other.index)
    assert sorted(first.index) == list(frame.index)
    for column in frame:
        assert first[column].value_counts(dropna=False).to_dict() == frame[column].value_counts(dropna=False).to_dict()


def _fake_result(task: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    dataset = str(task["dataset"])
    target_id = str(task["target_column_id"])
    profile = validation._profile_for_dataset(dataset, manifest)
    column = next(column for column in profile.payload["columns"] if column["column_id"] == target_id)
    task_type = _task_type(column)
    domain = column.get("valid_domain") or {}
    high_cardinality = task_type == "classification" and len(domain.get("values", [])) > 10
    predictors = (
        ["KNeighbors", "XGBoost"]
        if high_cardinality
        else [
            "CustomTabPFNModel",
            "KNeighbors",
            "XGBoost",
        ]
    )
    families = {"xgb": True, "knn": True, "tabpfn": not high_cardinality}
    arm = {
        "status": "pass",
        "score": 0.75 if task_type == "classification" else 2.0,
        "predictors": predictors,
        "predictor_scores": {name: 0.75 for name in predictors},
        "predictor_failures": [],
        "fit_evidence": {
            "training_row_order": "lexicographic-all-model-columns-v1",
            "split_implementation": "autogluon.core.utils.utils.generate_train_test_split",
            "seed": int(task["seed"]),
            "task_type": task_type,
            "problem_type": "regression" if task_type == "regression" else "binary",
            "holdout_fraction": 0.1,
            "input_rows": manifest["coverage"]["datasets"][dataset]["train_rows"],
            "fit_train_rows": int(manifest["coverage"]["datasets"][dataset]["train_rows"] * 0.9),
            "tuning_rows": (
                manifest["coverage"]["datasets"][dataset]["train_rows"]
                - int(manifest["coverage"]["datasets"][dataset]["train_rows"] * 0.9)
            ),
            "input_multiset_fingerprint": "e" * 64,
            "fit_train_fingerprint": "f" * 64,
            "tuning_fingerprint": "0" * 64,
            "real_test_used_for_fit": False,
        },
        "families": families,
        "wall_seconds": 2.0,
        "baseline_rss_bytes": 256 * 1024**2,
        "peak_rss_bytes": 512 * 1024**2,
        "cuda_peak_allocation_increase_bytes": 0 if high_cardinality else 1024**3,
    }
    seed = int(task["seed"])
    return {
        **task,
        "task_key": f"{dataset}:{target_id}:seed-{seed}",
        "target_name": column["name"],
        "task_type": task_type,
        "target_real_train_cardinality": len(domain.get("values", [])) or 20,
        "high_cardinality_tabpfn_omission_expected": high_cardinality,
        "surrogate": {
            "type": "full-row-multiset-preserving-permutation",
            "rows": manifest["coverage"]["datasets"][dataset]["train_rows"],
            "row_order_fingerprint": "a" * 64,
            "table_fingerprint": "b" * 64,
            "per_column_support_preserved": True,
        },
        "arms": {"trtr": copy.deepcopy(arm), "tstr": copy.deepcopy(arm)},
        "ratio": 1.0,
        "resource_gate": "pass",
        "failures": [],
        "status": "pass",
    }


def _fake_shards(tmp_path: Path) -> list[Path]:
    manifest = validation.validate_pilot_manifest()
    commit = p4_global_source._repository_commit()
    fingerprint = content_fingerprint(manifest)
    paths: list[Path] = []
    for dataset, record in manifest["coverage"]["datasets"].items():
        specs = [("coverage", shard_index, record["shards"]) for shard_index in range(record["shards"])] + [
            ("stability", 0, 1)
        ]
        for mode, shard_index, shard_count in specs:
            schedule = validation.task_schedule(dataset, mode, shard_index, shard_count, manifest)
            payload = {
                "evidence_schema_version": "1.0.0",
                "protocol_id": validation.PROTOCOL_ID,
                "evidence_type": "dataset-scale-shard",
                "status": "pass",
                "repository_commit": commit,
                "pilot_manifest_fingerprint": fingerprint,
                "shard": {
                    "dataset": dataset,
                    "mode": mode,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                },
                "environment": {"platform": "Windows / AMD64", "python": "3.11.9"},
                "runtime": {"status": "benchmark-approved-not-upstream-official"},
                "source": {"revision": "dba19a4ee7aa391621cbeb464609285fd515dece"},
                "checkpoints": {"classifier": {"sha256": "c" * 64}, "regressor": {"sha256": "d" * 64}},
                "dataset": {
                    "dataset_id": dataset,
                    "dataset_profile_version": record["dataset_profile_version"],
                    "train_rows": record["train_rows"],
                    "test_rows": record["test_rows"],
                },
                "installed_distributions": ["synthetic-test-runtime==1"],
                "results": [_fake_result(task, manifest) for task in schedule],
            }
            path = tmp_path / f"{dataset}-{mode}-{shard_index}.json"
            atomic_write_json(path, payload)
            paths.append(path)
    return paths


def test_finalizer_reconstructs_complete_coverage_stability_resources_and_high_cardinality(
    tmp_path: Path,
) -> None:
    paths = _fake_shards(tmp_path)
    output = tmp_path / "final.json"

    evidence = validation.finalize_shards(paths, output)

    assert evidence["status"] == "pass"
    assert evidence["schedule"] == {
        "coverage_seed": 0,
        "stability_seeds": [0, 1, 2, 3, 4],
        "expected_tasks": 67,
        "completed_tasks": 67,
        "completed_arms": 134,
    }
    assert evidence["resource_summary"]["arm_count"] == 134
    assert evidence["resource_summary"]["wall_seconds"]["maximum"] == 2.0
    assert all(record["gate"] == "pass" for dataset in evidence["stability"].values() for record in dataset.values())
    assert {record["target_column_id"] for record in evidence["high_cardinality_targets"]} == {
        "education",
        "occupation",
        "native-country",
    }
    assert set(evidence["exit_gates"].values()) == {"pass", "not-assessed"}


def test_finalizer_fails_closed_when_one_preregistered_task_is_missing(tmp_path: Path) -> None:
    paths = _fake_shards(tmp_path)
    first = paths[0]
    payload = copy.deepcopy(validation.read_json(first))
    payload["results"].pop()
    atomic_write_json(first, payload)

    evidence = validation.finalize_shards(paths, tmp_path / "failed.json")

    assert evidence["status"] == "fail"
    assert evidence["error_type"] == "P4DatasetScaleValidationError"
    assert "Task coverage differs" in evidence["error"]
    assert "traceback" not in evidence
    assert evidence["observations"]["observed_task_count"] == 66
    assert len(evidence["observations"]["missing_task_keys"]) == 1


def test_finalizer_retains_partial_observations_and_stability_failures(tmp_path: Path) -> None:
    paths = [path for path in _fake_shards(tmp_path) if path.name.startswith("sick-")]
    for path in paths:
        payload = copy.deepcopy(validation.read_json(path))
        for result in payload["results"]:
            if result["target_column_id"] == "referral-source" and result["seed"] == 4:
                result["ratio"] = 1.06
            if result["target_column_id"] == "tsh" and result["seed"] == 3:
                result["ratio"] = 0.94
        atomic_write_json(path, payload)

    evidence = validation.finalize_shards(paths, tmp_path / "partial.json")
    observations = evidence["observations"]

    assert evidence["status"] == "fail"
    assert evidence["repository_commit"] == p4_global_source._repository_commit()
    assert observations["expected_shard_count"] == 9
    assert observations["observed_shard_count"] == 5
    assert observations["missing_shards"] == [
        "adult:coverage:0-of-3",
        "adult:coverage:1-of-3",
        "adult:coverage:2-of-3",
        "adult:stability:0-of-1",
    ]
    assert observations["observed_task_count"] == 40
    assert observations["task_status_counts"] == {"pass": 40}
    assert observations["resources"]["observed_arm_count"] == 80
    assert observations["stability"]["sick"]["class"]["gate"] == "pass"
    assert observations["stability"]["sick"]["referral-source"]["gate"] == "fail"
    assert observations["stability"]["sick"]["tsh"]["gate"] == "fail"
    assert "missing-shards:4" in observations["issues"]
    assert "stability-gate:sick:referral-source" in observations["issues"]
    assert "stability-gate:sick:tsh" in observations["issues"]


def test_finalizer_writes_failure_evidence_for_an_unreadable_shard(tmp_path: Path) -> None:
    paths = _fake_shards(tmp_path)
    paths[0].write_text("not-json", encoding="utf-8")
    output = tmp_path / "unreadable-final.json"

    evidence = validation.finalize_shards(paths, output)

    assert output.is_file()
    assert evidence["status"] == "fail"
    assert evidence["shard_read_errors"][0]["path"] == paths[0].name
    assert evidence["observations"]["observed_shard_count"] == 8
    assert any(issue.startswith("malformed-shard-identity:") for issue in evidence["observations"]["issues"])
