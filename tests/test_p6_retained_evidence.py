from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs/evidence/evaluation/p6-windows-py311-0fb5d07.json"
EVIDENCE_SHA256 = "d1ef8213d3885f7c6cccf9bb159768d96790f3defbb69f66d52a7f3fa3150525"


def test_retained_p6_evidence_is_immutable_and_bound_to_implementation() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P6"
    assert evidence["protocol_id"] == "p6-orchestration-exit-gate-v1"
    assert evidence["repository_commit"] == "0fb5d07794790479eb3297f83e62832158053c81"
    assert evidence["environment"] == {
        "hardware_comparison_key": "ef653fc54b104a6e3a0e1573f0708b1cb847965cd67b73806ed210c110a8d20c",
        "hardware_profile_id": "p6-validation-observed-host",
        "platform": "Windows / AMD64",
        "primary_family_environment_required": True,
        "python": "3.11.15",
    }
    assert set(evidence["exit_gates"].values()) == {"pass"}
    assert len(evidence["locked_files"]) >= 25
    for relative, digest in evidence["locked_files"].items():
        assert sha256_file(REPO_ROOT / relative) == digest


def test_retained_p6_evidence_preserves_failure_and_efficiency_boundaries() -> None:
    evidence = read_json(EVIDENCE_PATH)
    summary = evidence["result_summary"]

    assert summary["cache_first_decision"] == "miss"
    assert summary["cache_repeat_decision"] == "hit"
    assert summary["changed_identity_decision"] == "miss"
    assert summary["stale_cache_decision"] == "invalid"
    assert summary["retry_attempt_count"] == 2
    assert summary["timeout_category"] == "timeout"
    assert summary["memory_category"] == "out-of-memory"
    assert summary["interruption_status"] == "cancelled"
    assert summary["partial_status"] == "partial"
    assert summary["completed_atomic_result_preserved"] is True
    assert summary["cross_profile_efficiency_rejected"] is True

    wall = summary["efficiency_wall_seconds"]
    rss = summary["efficiency_peak_rss_bytes"]
    tolerance = summary["efficiency_tolerance"]
    assert len(wall) == len(rss) == summary["efficiency_observation_count"] == 3
    assert max(wall) - min(wall) <= tolerance["wall_absolute_seconds"]
    assert max(rss) / min(rss) <= tolerance["peak_rss_ratio"]
    assert "no model-quality assessment" in evidence["claim_boundary"].casefold()
    assert "or official results admission." in evidence["claim_boundary"].casefold()
