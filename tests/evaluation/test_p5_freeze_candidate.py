from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.high_order_privacy import load_p5_evaluator_profile
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, read_json, sha256_file
from standardized_tabular_diffusion.validation.p5_tabddpm_adult_confirmatory import (
    GENERATION_SEEDS,
    PREREGISTRATION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION = REPO_ROOT / "docs/evidence/evaluation/p5-tabddpm-adult-confirmatory-windows-py311-8351b93.json"
CONFIRMATION_SHA256 = "ff57502eb691d6ccccef85f40e14564c3f99cc3be40a282dbdef09762ea40e36"
AUDIT = REPO_ROOT / "docs/evidence/evaluation/p5-confirmatory-execution-audit-2026-08-13.json"
AUDIT_SHA256 = "9fe5976e99a4f26a5811f7f682374a3db014269f3a080a2ab0293803ca2b1b6c"
DECISION = REPO_ROOT / "docs/evidence/evaluation/p5-protocol-freeze-decision-2026-08-13.json"
DECISION_SHA256 = "986ab7cf05120d467ffeb81c5998a1d254c9f219715753652e46a2687823b843"


def test_p5_scientific_identity_was_preregistered_and_is_now_frozen_unchanged() -> None:
    preregistration = read_json(REPO_ROOT / PREREGISTRATION)
    protocol = resolve_protocol("p5-high-order-privacy", "1.0.0")
    evaluator = load_p5_evaluator_profile()

    assert preregistration["status"] == "preregistered"
    assert preregistration["generation_seeds"] == list(GENERATION_SEEDS) == [3, 4, 5]
    assert preregistration["evaluation"]["protocol_version"] == protocol.protocol_version
    assert preregistration["evaluation"]["evaluator_profile_version"] == evaluator["profile_version"]
    assert preregistration["decision_rule"]["metric_values_used_for_decision"] is False
    scientific_parameters = {
        key: evaluator[key] for key in ("default_evaluator_seeds", "c2st", "dcr", "domias", "excluded_metrics")
    }
    decision = read_json(DECISION)
    assert content_fingerprint(scientific_parameters) == decision["scientific_identity"]["scientific_parameters_sha256"]
    assert decision["scientific_identity"]["changed_after_preregistration"] is False
    assert protocol.payload["status"] == "frozen"
    assert protocol.payload["official_results_allowed"] is True
    assert evaluator["status"] == "frozen"
    assert evaluator["official_results_allowed"] is True


def test_p5_confirmation_and_freeze_decision_are_immutable_complete_and_bounded() -> None:
    assert sha256_file(CONFIRMATION) == CONFIRMATION_SHA256
    assert sha256_file(AUDIT) == AUDIT_SHA256
    assert sha256_file(DECISION) == DECISION_SHA256
    confirmation = read_json(CONFIRMATION)
    audit = read_json(AUDIT)
    decision = read_json(DECISION)

    assert confirmation["repository"]["head"] == "8351b931a6b2f18cbaf273c8762344d9f2fe6756"
    assert confirmation["repository"]["working_tree_dirty"] is False
    assert confirmation["experiment_kind"] == "preregistered-confirmatory"
    assert confirmation["model"]["training_runs"] == 1
    assert confirmation["model"]["training_seed"] == 0
    assert confirmation["model"]["generation_seeds"] == [3, 4, 5]
    assert [item["generation_seed"] for item in confirmation["seed_results"]] == [3, 4, 5]
    for result in confirmation["seed_results"]:
        assert result["decoded_sample"]["rows"] == 32561
        assert result["metric_state_counts"] == {"computed": 34}
        assert result["denominator_counts"]["computed_c2st_seeds"] == 5
        assert result["denominator_counts"]["computed_domias_seeds"] == 5
        assert result["high_order_fidelity"]["overall_fidelity_score"] is None
        assert result["privacy_risk"]["overall_privacy_score"] is None
        assert result["privacy_risk"]["formal_privacy_guarantee"] is False
    assert all(item["metric_value_created"] is False for item in audit["preflight_failures"])
    assert audit["complete_run"]["fresh_training_from_step_zero"] is True
    assert decision["gates"]["numerical_metric_threshold_used"] is False
    assert decision["admission"]["metric_components_allowed"] is True
    assert decision["admission"]["adult_or_sick_admitted_by_this_decision"] is False
    assert decision["admission"]["tabddpm_admitted_by_this_decision"] is False
    assert decision["admission"]["formal_privacy_guarantee_established"] is False
    for evidence in decision["evidence"]:
        assert sha256_file(REPO_ROOT / evidence["path"]) == evidence["sha256"]


def test_dataset_privacy_roles_are_reviewed_without_admitting_attribute_inference() -> None:
    expectations = {
        "adult": {
            "path": "configs/datasets/adult-uci-2-v1.json",
            "direct": [],
            "quasi": {"age", "sex", "native-country"},
            "sensitive": {"race", "sex", "income"},
        },
        "sick": {
            "path": "configs/datasets/sick-uci-102-v1.json",
            "direct": ["record-id"],
            "quasi": {"age", "sex", "referral-source"},
            "sensitive": {"sex", "tsh", "class"},
        },
    }
    for expected in expectations.values():
        profile = load_dataset_profile(REPO_ROOT / expected["path"]).payload
        privacy = profile["privacy"]
        assert privacy["sensitive_roles_reviewed"] is True
        assert privacy["direct_identifier_column_ids"] == expected["direct"]
        assert expected["quasi"] <= set(privacy["quasi_identifier_column_ids"])
        assert expected["sensitive"] <= set(privacy["sensitive_attribute_column_ids"])
        assert privacy["threat_models"][0]["status"] == "approved-for-p5-v1"
        assert "attribute inference" in privacy["excluded_threat_models"]
        assert "not a legal de-identification" in privacy["legal_claim_boundary"]
        assert all(column["sensitivity"]["status"] == "reviewed" for column in profile["columns"])
        assert profile["official_eligible"] is False
