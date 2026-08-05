from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPO_ROOT / "docs" / "evidence" / "evaluation" / "p2-shape-trend-run-31025796906.json"
)
EVIDENCE_SHA256 = "f9d34b64c10b97ec40cec88ea26b769220b099364cf9e9e60e6c19c0c4f6e69a"


def test_retained_p2_evidence_is_immutable_authoritative_history() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P2"
    assert evidence["repository_commit"] == "27ecd237c771c745dbdc1e288bb4927d7b08e930"
    assert evidence["environment"] == {
        "pandas": "2.2.3",
        "platform": "Linux / x86_64",
        "primary_environment_required": True,
        "pyarrow": "18.1.0",
        "python": "3.11.15",
        "sdmetrics": "0.28.3.dev0",
    }
    assert set(evidence["exit_gates"].values()) == {"pass"}
    assert evidence["source"]["revision"] == "ba8842f2ba04ce914f698cc1cf746ca12338ab0e"
    assert evidence["source"]["python_source_file_count"] == 121
    assert evidence["source"]["python_source_tree_sha256"] == (
        "784beda5c7a63d5ebb5fe74f98d00db3a2e018a29b2f32f643bf857750a6c2a9"
    )
    assert evidence["source"]["license_spdx"] == "MIT"
    assert evidence["result_summary"]["atomic_results"] == 120
    assert evidence["result_summary"]["finalized_bundles_created"] == 2
    assert evidence["result_summary"]["subsampling_reproducibility"]["rows"] == 50_001
    assert evidence["result_summary"]["overall_fidelity_score_emitted"] is False
    assert evidence["result_summary"]["official_results_allowed"] is False

    # These hashes attest the exact historical commit above. Later phases may
    # intentionally evolve the locked files without changing this evidence.
    assert len(evidence["locked_files"]) >= 30
    assert all(len(digest) == 64 for digest in evidence["locked_files"].values())
