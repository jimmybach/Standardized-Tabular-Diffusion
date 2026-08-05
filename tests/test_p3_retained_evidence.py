from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "evaluation" / "p3-validity-run-31036844043.json"
EVIDENCE_SHA256 = "bc63a2df553036ee7e161ce81c6f264dace950f3fe414ba2f8195d8e557e401d"


def test_retained_p3_evidence_is_immutable_authoritative_history() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P3"
    assert evidence["protocol_id"] == "p3-validity-and-preprocessing-v1"
    assert evidence["repository_commit"] == "681cb2cc7a3360d89bafa24c4b5595309869a913"
    assert evidence["environment"] == {
        "pandas": "2.2.3",
        "platform": "Linux / x86_64",
        "primary_environment_required": True,
        "pyarrow": "18.1.0",
        "python": "3.11.15",
    }
    assert set(evidence["exit_gates"].values()) == {"pass"}

    result = evidence["result_summary"]
    assert result["scientific_metrics_executed"] is True
    assert result["atomic_results"] == result["expected_atomic_results"] == 16
    assert result["finalized_bundles_created"] == 2
    assert result["column_validity_score"] == pytest.approx(0.99)
    assert result["constraint_validity_score"] is None
    assert result["fully_valid_row_rate"] == pytest.approx(0.95)
    assert result["synthetic_repair_applied"] is False
    assert result["official_results_allowed"] is False

    preprocessing = evidence["preprocessing_boundary"]
    assert preprocessing["train_only_state_equal_under_test_changes"] is True
    assert preprocessing["numerical_fill"] == pytest.approx(15.0)
    assert preprocessing["categorical_fill"] == "b"
    assert preprocessing["synthetic_repair_policy"] == "reject"

    # These hashes attest the exact historical commit above. Later phases may
    # intentionally evolve the locked files without invalidating this record.
    assert len(evidence["locked_files"]) >= 20
    assert all(len(digest) == 64 for digest in evidence["locked_files"].values())
