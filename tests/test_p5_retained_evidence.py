from __future__ import annotations

import json
from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs/evidence/evaluation/p5-windows-py311-identity-c66fa23.json"
EVIDENCE_SHA256 = "7c680f47a9a592b0cefcc8d5a37c96ddbfc9dff0fd0285a2f56bf6b5b5dbfd8c"
GENERATOR_EVIDENCE_PATH = REPO_ROOT / "docs/evidence/evaluation/p5-tabddpm-adult-windows-py311-a2e4f27.json"
GENERATOR_EVIDENCE_SHA256 = "a855b7e246eb4d6cc1a4942488ddd96eb7733e0dd0781e5a75c46acc034b9874"


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


def test_retained_tabddpm_adult_generator_pilot_is_immutable_and_bounded() -> None:
    assert sha256_file(GENERATOR_EVIDENCE_PATH) == GENERATOR_EVIDENCE_SHA256
    evidence = read_json(GENERATOR_EVIDENCE_PATH)

    assert evidence["status"] == "passed"
    assert evidence["protocol_id"] == "p5-tabddpm-adult-generator-pilot-v1"
    assert evidence["model"]["model_id"] == "tabddpm"
    assert evidence["model"]["upstream_commit"] == "b476257dd460b778ba09eb97f7a51d6490fa17f8"
    assert evidence["model"]["training_runs"] == 1
    assert evidence["model"]["training_seed"] == 0
    assert evidence["model"]["generation_seeds"] == [0, 1, 2]
    assert evidence["dataset"]["dataset_id"] == "adult"
    assert evidence["dataset"]["path"] == "runtime-input/adult"
    assert evidence["dataset"]["train_rows"] == 32561
    assert evidence["dataset"]["test_rows"] == 16281
    assert evidence["environment"]["hardware"]["gpu"] == "NVIDIA GeForce RTX 5080"
    assert evidence["evaluation"]["evaluator_seeds_per_generated_table"] == [0, 1, 2, 3, 4]
    assert evidence["evaluation"]["overall_fidelity_score"] is None
    assert evidence["evaluation"]["overall_privacy_score"] is None
    assert evidence["evaluation"]["formal_privacy_guarantee"] is False
    assert evidence["assertions"]["official_results_admitted"] is False
    assert evidence["assertions"]["protocol_frozen_by_this_pilot"] is False
    assert evidence["publication_transform"] == {
        "applied": True,
        "execution_source_sha256_preserved": True,
        "scientific_values_changed": False,
        "scope": "host-path-redaction-only",
        "transformed_fields": [
            "dataset.path",
            "seed_results[*].bundle_path",
            "seed_results[*].decoded_sample.synthetic_path",
        ],
    }
    assert "C:\\\\Users\\\\" not in json.dumps(evidence)
    assert all(len(digest) == 64 for digest in evidence["repository"]["execution_source_sha256"].values())

    assert [item["generation_seed"] for item in evidence["seed_results"]] == [0, 1, 2]
    for result in evidence["seed_results"]:
        seed = result["generation_seed"]
        assert result["bundle_path"] == f"seed-{seed}/p5-bundle"
        assert result["decoded_sample"]["synthetic_path"] == f"seed-{seed}/synthetic.csv"
        assert result["metric_state_counts"] == {"computed": 34}
        assert result["denominator_counts"]["computed_c2st_seeds"] == 5
        assert result["denominator_counts"]["computed_domias_seeds"] == 5
        assert result["decoded_sample"]["rows"] == 32561
        assert result["privacy_risk"]["formal_privacy_guarantee"] is False
        assert result["high_order_fidelity"]["overall_fidelity_score"] is None
        assert result["privacy_risk"]["overall_privacy_score"] is None
        assert all(len(digest) == 64 for digest in result["decoded_sample"]["raw_array_sha256"].values())
