from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs/evidence/evaluation/p4-global-source-runtime-run-31057073762.json"
)
EVIDENCE_SHA256 = "1ca205c3fbad6c6e80cd275330dae00edda5dfcad7efe229f4fcb285b9d63596"


def test_retained_p4_global_source_evidence_is_immutable_authoritative_history() -> None:
    assert sha256_file(EVIDENCE_PATH) == EVIDENCE_SHA256
    evidence = read_json(EVIDENCE_PATH)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "p4-global-source-runtime-pilot-v1"
    assert evidence["repository_commit"] == "d710950c24cbe11700f877975b05c90937273113"
    assert evidence["environment"] == {
        "platform": "Linux / x86_64",
        "primary_environment_required": True,
        "python": "3.11.15",
    }
    assert evidence["exit_gates"]["official_results_admission"] == "not-assessed"
    assert {
        value
        for gate, value in evidence["exit_gates"].items()
        if gate != "official_results_admission"
    } == {"pass"}

    source = evidence["source"]
    assert source["revision"] == "dba19a4ee7aa391621cbeb464609285fd515dece"
    assert source["source_sha256"] == (
        "1861a7573949e50b360c722f4e73110f2c3d014c412693b66c704d070df62743"
    )
    assert source["license_spdx"] == "Apache-2.0"
    assert source["license_sha256"] == (
        "11d68d4f6040e19f06cfce52c85f1f64d9a2d2d8cc67567daf53377f7b8358a7"
    )

    runtime = evidence["runtime"]
    assert runtime["status"] == "benchmark-approved-not-upstream-official"
    assert runtime["upstream_official_environment_claimed"] is False
    assert runtime["torch_cuda_available"] is False
    assert runtime["versions"] == {
        "autogluon.common": "1.4.0",
        "autogluon.core": "1.4.0",
        "autogluon.features": "1.4.0",
        "autogluon.tabular": "1.4.0",
        "tabpfn": "2.1.2",
        "torch": "2.3.0+cpu",
        "xgboost": "3.0.3",
        "xgboost_distribution": "xgboost-cpu",
        "xgboost_import": "3.0.3",
    }

    checkpoints = evidence["checkpoints"]
    assert checkpoints["classifier"]["sha256"] == (
        "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
    )
    assert checkpoints["regressor"]["sha256"] == (
        "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
    )

    execution = evidence["execution"]
    assert execution["exact_source_evaluate_completed"] is True
    assert execution["absolute_differences"] == {
        "binary_target": 0.0,
        "numeric_target": 0.0,
    }
    assert execution["absolute_tolerance"] == 1e-8
    assert execution["high_cardinality_source_guard"]["rejected_before_model_fit"] is True
    expected_models = ["CustomTabPFNModel", "KNeighbors", "XGBoost"]
    assert execution["adapter_results"]["binary_target"]["predictors"] == expected_models
    assert execution["adapter_results"]["numeric_target"]["predictors"] == expected_models
    assert len(execution["source_predictor_traces"]) == 3
    assert all(
        trace["leaderboard"]["models"] == expected_models
        for trace in execution["source_predictor_traces"]
    )
    assert len(evidence["installed_distributions"]) == 66

    # The locked-file hashes describe the exact historical source commit above.
    # Later documentation and admission work may evolve those files without
    # changing this retained artifact.
    assert len(evidence["locked_files"]) == 9
    assert all(len(digest) == 64 for digest in evidence["locked_files"].values())
