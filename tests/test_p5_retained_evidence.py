from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs/evidence/evaluation/p5-windows-py311-identity-c66fa23.json"
EVIDENCE_SHA256 = "7c680f47a9a592b0cefcc8d5a37c96ddbfc9dff0fd0285a2f56bf6b5b5dbfd8c"


def test_retained_p5_identity_evidence_is_immutable_and_bounded() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P5"
    assert evidence["repository_commit"] == "c66fa23b7f4541205d004d701bb92ad2d35709b5"
    assert evidence["protocol"] == {
        "official_results_allowed": False,
        "protocol_id": "p5-high-order-privacy",
        "protocol_version": "0.1.0",
        "sha256": "8341db05a1166958f7dd609a9c757ff94bcbf55c202f56d3e0638d25be87eb83",
        "status": "draft",
    }
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["operating_system"]["caption"] == "Microsoft Windows 11 Pro"
    assert evidence["environment"]["accelerator"]["evaluation_used_accelerator"] is False
    assert evidence["identity_surrogate"] == {
        "expected_exact_train_collision_rate": 1.0,
        "purpose": "End-to-end implementation and boundary validation only",
        "synthetic_artifact_equals_real_train_artifact": True,
    }

    for dataset_id, expected_rows in {"adult": 32561, "sick": 2800}.items():
        run = evidence["runs"][dataset_id]
        assert run["finalization_status"] == "finalized"
        assert run["computed_atomic_results"] == 34
        assert run["seed_coverage"] == {"c2st": "5/5", "domias": "5/5"}
        assert run["details_reproduced_across_independent_attempts"] is True
        assert run["real_train_and_identity_synthetic"]["rows"] == expected_rows
        assert run["metrics"]["exact_train_collision_rate"] == 1.0
        assert run["metrics"]["formal_privacy_guarantee"] is False
        assert run["metrics"]["overall_fidelity_score"] is None
        assert run["metrics"]["overall_privacy_score"] is None
        assert all(len(digest) == 64 for digest in run["artifact_hashes"].values())

    assert evidence["runs"]["adult"]["metrics"]["c2st_raw_auroc_mean"] == pytest.approx(0.49691568231486355)
    assert evidence["runs"]["sick"]["metrics"]["domias_attack_auroc_mean"] == pytest.approx(0.5635611949397957)
    assert evidence["source_attestation"]["sdmetrics_dcr"]["python_source_file_count"] == 121
    assert all(len(digest) == 64 for digest in evidence["locked_files"].values())
