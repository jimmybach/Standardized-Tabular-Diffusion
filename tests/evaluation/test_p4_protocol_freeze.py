from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol
from standardized_tabular_diffusion.evaluation.registry import load_metric_registry
from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file
from standardized_tabular_diffusion.evaluation.utility import load_p4_evaluator_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION = REPO_ROOT / "docs/evidence/evaluation/p4-protocol-freeze-decision-2026-08-12.json"
DECISION_SHA256 = "ac6854726abe51aa7e1bfdc440d7a32edebeb596d364ced59cfdf601efd3e3fc"


def test_p4_freeze_decision_is_immutable_bounded_and_evidence_attested() -> None:
    assert sha256_file(DECISION) == DECISION_SHA256
    decision = read_json(DECISION)
    assert decision["decision"] == "approve-protocol-freeze"
    assert decision["scientific_identity"]["changed_from_validated_successor"] is False
    assert decision["validated_environment"] == {
        "operating_system": "Windows 11 x86-64",
        "python": "3.11",
        "gpu": "NVIDIA GeForce RTX 5080",
    }
    assert decision["admission"]["metric_components_allowed"] is True
    assert decision["admission"]["adult_or_sick_admitted_by_this_decision"] is False
    assert decision["admission"]["generator_quality_assessed"] is False
    for evidence in decision["evidence"]:
        assert sha256_file(REPO_ROOT / evidence["path"]) == evidence["sha256"]


def test_p4_protocol_and_registry_are_frozen_without_dataset_overclaim() -> None:
    protocol = resolve_protocol("p4-utility", "1.0.0")
    evaluator = load_p4_evaluator_profile()
    records = [
        record
        for record in load_metric_registry()
        if record.payload["admission"]["compatibility_version"] == "p4-utility-1.0.0"
    ]
    assert protocol.payload["status"] == "frozen"
    assert protocol.payload["official_results_allowed"] is True
    assert evaluator["status"] == "frozen"
    assert evaluator["official_results_allowed"] is True
    assert len(records) == 11
    assert all(record.payload["lifecycle_status"] == "protocol-frozen" for record in records)
    assert all(record.payload["validation"]["release_decision"] == "pending" for record in records)
    for dataset in ("configs/datasets/adult-uci-2-v1.json", "configs/datasets/sick-uci-102-v1.json"):
        profile = load_dataset_profile(REPO_ROOT / dataset)
        assert profile.payload["official_eligible"] is False
        assert profile.payload["suite_membership"] == ["diagnostic"]
