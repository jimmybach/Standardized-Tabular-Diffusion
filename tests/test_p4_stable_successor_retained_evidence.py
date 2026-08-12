from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs/evidence/evaluation/p4-dataset-scale-windows-gpu-stable-6dc485f.json"
)
EVIDENCE_SHA256 = "19d55b260eaa5a3d1e522d1a1cabeadf4e952698a833e529576fd64b71797721"


def test_retained_p4_stable_successor_passes_complete_schedule_and_fixed_gates() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "p4-dataset-scale-windows-gpu-stable-candidate-v1"
    assert evidence["repository_commit"] == "6dc485f0f0f7071fcdf38a7c48c50012f3d21627"
    assert evidence["pilot_manifest_fingerprint"] == (
        "24b0a05f5d7d5b313ff5c458eb9dea35a056dc23807303d68a79fd03f8166633"
    )
    assert evidence["official_results_allowed"] is False
    assert evidence["schedule"] == {
        "completed_arms": 134,
        "completed_tasks": 67,
        "coverage_seed": 0,
        "expected_tasks": 67,
        "stability_seeds": [0, 1, 2, 3, 4],
    }
    assert set(evidence["exit_gates"].values()) == {"pass", "not-assessed"}
    assert evidence["exit_gates"]["official_results_admission"] == "not-assessed"
    assert evidence["exit_gates"]["generator_quality_assessed"] == "not-assessed"


def test_retained_p4_stable_successor_proves_identity_and_resource_stability() -> None:
    evidence = read_json(EVIDENCE_PATH)

    results = evidence["results"]
    assert len(results) == 67
    assert len({result["task_key"] for result in results}) == 67
    assert all(result["status"] == "pass" and result["failures"] == [] for result in results)
    assert all(result["ratio"] == 1.0 for result in results)
    assert all(
        result["arms"]["trtr"]["fit_evidence"]
        == result["arms"]["tstr"]["fit_evidence"]
        for result in results
    )
    assert all(
        arm["fit_evidence"]["real_test_used_for_fit"] is False
        for result in results
        for arm in result["arms"].values()
    )

    for dataset in evidence["stability"].values():
        for record in dataset.values():
            assert record["ratios"] == [1.0] * 5
            assert record["range"] == 0.0
            assert record["maximum_absolute_deviation_from_identity"] == 0.0
            assert record["gate"] == "pass"

    resources = evidence["resource_summary"]
    limits = resources["preregistered_limits"]
    assert resources["arm_count"] == 134
    assert resources["wall_seconds"]["maximum"] < limits[
        "maximum_observed_arm_wall_seconds"
    ]
    assert resources["process_tree_peak_rss_gib"]["maximum"] < limits[
        "maximum_observed_process_tree_peak_rss_gib"
    ]
    assert resources["cuda_peak_allocation_increase_gib"]["maximum"] < limits[
        "maximum_observed_cuda_peak_allocated_gib"
    ]


def test_retained_p4_stable_successor_is_bound_to_declared_windows_gpu_runtime() -> None:
    evidence = read_json(EVIDENCE_PATH)
    runtime = evidence["runtime"]

    assert evidence["environment"]["platform"] == "Windows / AMD64"
    assert runtime["versions"]["autogluon.tabular"] == "1.4.0"
    assert runtime["versions"]["tabpfn"] == "2.1.2"
    assert runtime["versions"]["xgboost"] == "3.0.3"
    assert runtime["versions"]["torch"] == "2.8.0+cu128"
    assert runtime["cuda"]["device_name"] == "NVIDIA GeForce RTX 5080"
    assert runtime["cuda"]["runtime"] == "12.8"
    assert runtime["torch_cuda_available"] is True
    assert {
        (record["dataset"], record["target_column_id"])
        for record in evidence["high_cardinality_targets"]
    } == {
        ("adult", "education"),
        ("adult", "occupation"),
        ("adult", "native-country"),
    }
    assert all(
        record["tabpfn_omitted_in_all_arms"] is True
        for record in evidence["high_cardinality_targets"]
    )
