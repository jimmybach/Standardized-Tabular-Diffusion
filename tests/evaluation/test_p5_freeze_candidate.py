from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.high_order_privacy import load_p5_evaluator_profile
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.validation.p5_tabddpm_adult_confirmatory import (
    GENERATION_SEEDS,
    PREREGISTRATION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_p5_scientific_identity_is_preregistered_before_confirmatory_execution() -> None:
    preregistration = read_json(REPO_ROOT / PREREGISTRATION)
    protocol = resolve_protocol("p5-high-order-privacy", "1.0.0")
    evaluator = load_p5_evaluator_profile()

    assert preregistration["status"] == "preregistered"
    assert preregistration["generation_seeds"] == list(GENERATION_SEEDS) == [3, 4, 5]
    assert preregistration["evaluation"]["protocol_version"] == protocol.protocol_version
    assert preregistration["evaluation"]["evaluator_profile_version"] == evaluator["profile_version"]
    assert preregistration["decision_rule"]["metric_values_used_for_decision"] is False
    assert protocol.payload["status"] == "draft"
    assert protocol.payload["official_results_allowed"] is False
    assert evaluator["status"] == "freeze-candidate"
    assert evaluator["official_results_allowed"] is False


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
