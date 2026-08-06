from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = REPO_ROOT / "docs/evidence/evaluation/p4-dataset-scale-run-31060416318.json"
OBSERVATIONS_PATH = (
    REPO_ROOT
    / "docs/evidence/evaluation/p4-dataset-scale-observations-run-31060416318.json"
)
RUNNER_FAILURES_PATH = (
    REPO_ROOT
    / "docs/evidence/evaluation/p4-dataset-scale-runner-failures-run-31060416318.json"
)
DECISION_PATH = (
    REPO_ROOT
    / "docs/evidence/evaluation/p4-dataset-scale-admission-decision-run-31060416318.json"
)

FINAL_SHA256 = "cf3a53395f50af49600cb9ab190978ee45b875286b0e15c63215db6a26c90ae8"
OBSERVATIONS_SHA256 = "a7022a64e1a279f20e3090ccb00c0f445e1168d803efb06e5ae685280804057f"
RUNNER_FAILURES_SHA256 = "135381b05c11eb4887a8563b29e4ccf0b9818679e913cea4fb53895f22d3fa35"
DECISION_SHA256 = "8d6555c586f5b1a2c9a8024d6e151cb9559742ef023bfec2c3ea36b7b578d85e"


def test_retained_p4_dataset_scale_evidence_preserves_failed_admission_history() -> None:
    assert sha256_file(FINAL_PATH) == FINAL_SHA256
    evidence = read_json(FINAL_PATH)

    assert evidence["status"] == "fail"
    assert evidence["official_results_allowed"] is False
    assert evidence["repository_commit"] == "ce53739340ebf2afe4fdc836203fcd8f2b7696e7"
    assert evidence["pilot_manifest_fingerprint"] == (
        "0314a80f0d9b274092aca83fd4407186ec505755682a1fac3f50ea3321f5f45e"
    )
    assert evidence["error"] == "Expected 9 shard artifacts, observed 5"


def test_retained_partial_adjudication_is_bound_to_original_shards_and_gates() -> None:
    assert sha256_file(OBSERVATIONS_PATH) == OBSERVATIONS_SHA256
    evidence = read_json(OBSERVATIONS_PATH)
    observations = evidence["observations"]

    assert evidence["status"] == "fail"
    assert evidence["repository_commit"] == "ce53739340ebf2afe4fdc836203fcd8f2b7696e7"
    assert evidence["adjudicator_repository_commit"] == (
        "c4aafebed24862a44f193d6ba4f3fe9cfa30941b"
    )
    assert "traceback" not in evidence
    assert observations["observed_shard_count"] == 5
    assert observations["expected_shard_count"] == 9
    assert observations["observed_task_count"] == 40
    assert observations["expected_task_count"] == 67
    assert observations["task_status_counts"] == {"pass": 40}
    assert observations["resources"]["observed_arm_count"] == 80
    assert observations["resources"]["wall_seconds"]["maximum"] < 600
    assert observations["resources"]["process_tree_peak_rss_gib"]["maximum"] < 14
    assert observations["stability"]["sick"]["class"]["gate"] == "pass"
    assert observations["stability"]["sick"]["referral-source"]["gate"] == "fail"
    assert observations["stability"]["sick"]["tsh"]["gate"] == "fail"
    assert observations["issues"] == [
        "missing-shards:4",
        "missing-tasks:27",
        "stability-gate:sick:referral-source",
        "stability-gate:sick:tsh",
    ]
    assert len(observations["observed_shards"]) == 5
    assert all(len(shard["sha256"]) == 64 for shard in observations["observed_shards"])


def test_retained_runner_failures_separate_observation_from_causal_inference() -> None:
    assert sha256_file(RUNNER_FAILURES_PATH) == RUNNER_FAILURES_SHA256
    evidence = read_json(RUNNER_FAILURES_PATH)

    assert evidence["run_id"] == 31060416318
    assert len(evidence["jobs"]) == 4
    assert {job["execute_step_conclusion"] for job in evidence["jobs"]} == {"cancelled"}
    assert {job["upload_step_conclusion"] for job in evidence["jobs"]} == {"skipped"}
    assert evidence["adjudication"]["direct_cause_proven"] is False
    assert evidence["adjudication"]["resource_exhaustion_interpretation"] == (
        "consistent-but-not-proven"
    )


def test_retained_admission_decision_fails_closed_without_changing_preregistered_gates() -> None:
    assert sha256_file(DECISION_PATH) == DECISION_SHA256
    decision = read_json(DECISION_PATH)

    assert decision["decision"] == "fail"
    assert decision["lifecycle_after_decision"] == "diagnostic"
    assert decision["profile_freeze_allowed"] is False
    assert decision["official_results_allowed"] is False
    assert decision["protocol_version"] == "0.1.1"
    assert decision["gates"]["adult_full_target_execution"] == "fail"
    assert decision["gates"]["sick_full_target_execution"] == "pass"
    assert decision["gates"]["sick_five_seed_stability"] == "fail"
    assert decision["gates"]["sick_resource_envelope"] == "pass"
    assert all(item["sha256"] for item in decision["evidence"])
    assert "do not relax" in decision["next_review_requirements"][1]
