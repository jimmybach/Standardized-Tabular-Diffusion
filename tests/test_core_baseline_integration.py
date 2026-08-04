from __future__ import annotations

import hashlib
import json
from pathlib import Path

from standardized_tabular_diffusion.model_inventory import MODEL_INVENTORY
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "core-baselines"
    / "native-parity-integration-9da6e55.json"
)
PRIMARY_MODELS = ("tabddpm", "tabdiff", "tabsyn")


def test_integrated_native_parity_evidence_is_immutable_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == (
        "08aebe5409ffb88c980f24e0d36b12b589930717f6c7dd6b473749e79a36c860"
    )
    assert evidence["status"] == "pass"
    assert evidence["candidate_commit"] == "9da6e556a091f2501af4cd80ed938feaccb34055"
    assert evidence["core_ci"]["status"] == "pass"
    assert set(evidence["models"]) == set(PRIMARY_MODELS)
    assert all(record["status"] == "pass" for record in evidence["models"].values())
    assert {
        model_id: record["source_files_verified"]
        for model_id, record in evidence["models"].items()
    } == {"tabddpm": 64, "tabdiff": 27, "tabsyn": 20}


def test_integrated_promotions_do_not_change_release_or_leaderboard_status() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    boundary = evidence["claim_boundary"]

    assert boundary == {
        "benchmark_eligible": False,
        "benchmark_track": "experimental",
        "official_results_eligible": False,
        "release_supported": False,
        "support_level": "unsupported",
        "validation_level": "native-parity-validated",
    }
    for model_id in PRIMARY_MODELS:
        spec = get_adapter_spec(model_id)
        assert spec.validation_level.value == boundary["validation_level"]
        assert spec.benchmark_track == boundary["benchmark_track"]
        assert spec.support_level == boundary["support_level"]
        assert MODEL_INVENTORY[model_id].validation_level == boundary["validation_level"]
