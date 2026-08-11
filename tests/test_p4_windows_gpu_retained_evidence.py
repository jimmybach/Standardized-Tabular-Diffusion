from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_EVIDENCE = REPO_ROOT / "docs/evidence/evaluation/p4-global-source-windows-gpu-c0e6e72.json"
DATASET_EVIDENCE = REPO_ROOT / "docs/evidence/evaluation/p4-dataset-scale-windows-gpu-a754ca1.json"
SOURCE_SHA256 = "3f3033348c075a7b2f2584eaad2bd50d419c7aa12391af320c6bec75ecab1306"
DATASET_SHA256 = "2c7274a924b5e0673ba30878eb15de3c0e5d7cc78f930d8869054fc0177f4478"


def test_retained_windows_gpu_source_runtime_passed_exact_parity_and_cuda_gates() -> None:
    assert sha256_file(SOURCE_EVIDENCE) == SOURCE_SHA256
    evidence = read_json(SOURCE_EVIDENCE)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "p4-global-source-windows-gpu-pilot-v1"
    assert evidence["repository_commit"] == "c0e6e7232ea435948dcf3b41eeee9f7f47e1debd"
    assert evidence["environment"]["platform"] == "Windows / AMD64"
    assert evidence["runtime"]["versions"]["torch"] == "2.8.0+cu128"
    assert evidence["runtime"]["cuda"]["device_name"] == "NVIDIA GeForce RTX 5080"
    assert evidence["execution"]["absolute_differences"] == {
        "binary_target": 0.0,
        "numeric_target": 0.0,
    }
    assert evidence["execution"]["gpu_execution"]["source"][
        "peak_allocation_increase_bytes"
    ] > 0
    assert evidence["execution"]["gpu_execution"]["adapter"][
        "peak_allocation_increase_bytes"
    ] > 0
    assert evidence["exit_gates"]["official_results_admission"] == "not-assessed"


def test_retained_windows_gpu_dataset_run_is_complete_but_fails_stability() -> None:
    assert sha256_file(DATASET_EVIDENCE) == DATASET_SHA256
    evidence = read_json(DATASET_EVIDENCE)
    observations = evidence["observations"]

    assert evidence["status"] == "fail"
    assert evidence["repository_commit"] == "a754ca1950aca5714070dc86b466a556be4545f2"
    assert evidence["error"] == "Preregistered sentinel stability bound failed"
    assert observations["observed_shard_count"] == observations["expected_shard_count"] == 9
    assert observations["observed_task_count"] == observations["expected_task_count"] == 67
    assert observations["task_status_counts"] == {"pass": 67}
    assert observations["missing_shards"] == []
    assert observations["missing_task_keys"] == []
    assert observations["resources"]["observed_arm_count"] == 134
    assert observations["resources"]["wall_seconds"]["maximum"] < 600
    assert observations["resources"]["process_tree_peak_rss_gib"]["maximum"] < 14
    assert observations["resources"]["cuda_peak_allocation_increase_gib"]["maximum"] < 15
    assert observations["stability"]["adult"]["income"]["gate"] == "pass"
    assert observations["stability"]["adult"]["fnlwgt"]["gate"] == "pass"
    assert observations["stability"]["adult"]["native-country"]["gate"] == "fail"
    assert observations["stability"]["sick"]["class"]["gate"] == "pass"
    assert observations["stability"]["sick"]["referral-source"]["gate"] == "fail"
    assert observations["stability"]["sick"]["tsh"]["gate"] == "fail"
    assert observations["issues"] == [
        "stability-gate:adult:native-country",
        "stability-gate:sick:referral-source",
        "stability-gate:sick:tsh",
    ]
