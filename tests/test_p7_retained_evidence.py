from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs/evidence/evaluation/p7-windows-py311-4c8da76.json"
EVIDENCE_SHA256 = "52ddd40c108e47d1e9fbd1f5ebd8862b1a935c3758bb867e50700d87ca91f835"


def test_retained_p7_evidence_is_immutable_and_bound_to_implementation_commit() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P7"
    assert evidence["protocol_id"] == "p7-leaderboard-exit-gate-v1"
    assert evidence["repository_commit"] == "4c8da76c6e4e4d5ed1bd55c8d1d6825be6da05cc"
    assert evidence["environment"] == {
        "platform": "Windows / AMD64",
        "primary_family_environment_required": True,
        "python": "3.11.15",
    }
    assert set(evidence["exit_gates"].values()) == {"pass"}
    assert len(evidence["locked_files"]) >= 15
    assert all(len(digest) == 64 for digest in evidence["locked_files"].values())
    assert {
        ".github/workflows/p7-leaderboard-validation.yml",
        "standardized_tabular_diffusion/evaluation/leaderboard.py",
        "standardized_tabular_diffusion/validation/p7_leaderboard.py",
        "tests/evaluation/test_p7_leaderboard.py",
        "tests/evaluation/test_p7_run_bundle_integration.py",
    } <= set(evidence["locked_files"])


def test_retained_p7_evidence_preserves_publication_boundaries() -> None:
    evidence = read_json(EVIDENCE_PATH)
    summary = evidence["result_summary"]

    assert summary["failed_seed_denominator_preserved"] is True
    assert summary["missing_seed_denominator_preserved"] is True
    assert summary["official_without_admissions_rejected"] is True
    assert summary["publication_assets"] == [
        "leaderboard.json",
        "leaderboard.csv",
        "leaderboard.html",
        "leaderboard.md",
    ]
    assert len(summary["snapshot_fingerprint"]) == 64
    assert len(summary["input_fingerprint"]) == 64
    assert "does not admit" in evidence["claim_boundary"].casefold()
    assert "official results" in evidence["claim_boundary"].casefold()
