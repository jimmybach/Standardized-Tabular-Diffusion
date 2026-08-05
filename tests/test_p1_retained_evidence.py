from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "evaluation" / "p1-foundation-run-31018595264.json"
EVIDENCE_SHA256 = "3013e913c58adf0c03c6ec30118879c522a87f4682d1cceb99f8778115c7da5a"


def test_retained_p1_evidence_is_immutable_historical_evidence() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "86f4251e0142484ef7379772a178d10d88c5d9bf"
    assert evidence["environment"]["platform"] == "Linux / x86_64"
    assert evidence["environment"]["python"].startswith("3.11.")
    assert evidence["environment"]["primary_environment_required"] is True
    assert evidence["result_summary"]["scientific_metrics_executed"] is False
    assert evidence["result_summary"]["finalized_bundle_created"] is False

    locked_files = evidence["locked_files"]
    assert "standardized_tabular_diffusion/evaluation/tabstruct.py" not in locked_files
    assert len(locked_files) >= 30
    assert all(len(expected_hash) == 64 for expected_hash in locked_files.values())

    # The hashes attest the exact P1 commit above. Later phases intentionally
    # evolve these files, so comparing historical hashes to the current worktree
    # would turn valid P2 development into a false P1 evidence failure.
