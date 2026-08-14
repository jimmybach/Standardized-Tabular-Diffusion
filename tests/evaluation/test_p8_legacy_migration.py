from __future__ import annotations

import json
from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.legacy import (
    LegacyImportError,
    import_legacy_summary,
    validate_legacy_import,
)
from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file

pytestmark = [pytest.mark.core, pytest.mark.evaluation, pytest.mark.legacy]


def _legacy_summary() -> dict:
    return {
        "schema_version": "1.0",
        "protocol_name": "tabstruct-aligned-v1",
        "model": "tabddpm",
        "dataset": "adult",
        "metrics": {
            "density": {
                "shape_score": None,
                "trend_score": None,
                "overall_score": None,
                "status": "not_emitted_by_upstream_tabddpm_pipeline",
            },
            "ml_efficacy": {
                "primary_metric_name": "catboost_auroc",
                "primary_metric_value": 0.7,
                "task_type": "classification",
                "details": {},
            },
            "detection": {
                "logistic_detection": None,
                "status": "not_emitted_by_upstream_tabddpm_pipeline",
            },
            "privacy": {"dcr_score": None, "details": None},
            "structural_fidelity": {
                "global_utility": None,
                "status": "not_available_without_real_and_synthetic_tables",
            },
        },
        "tabstruct_alignment": {
            "density_fidelity": "missing in current upstream outputs",
            "ml_efficacy": "normalized from upstream evaluator outputs",
            "detection": "missing in current upstream outputs",
            "privacy": "normalized from upstream privacy output when provided",
            "structural_fidelity": "requires table-level artifacts",
        },
    }


def test_legacy_import_preserves_bytes_and_cannot_claim_official_status(tmp_path: Path) -> None:
    source = tmp_path / "standardized_summary.json"
    source.write_text(json.dumps(_legacy_summary(), ensure_ascii=False, indent=3) + "\n", encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "legacy-import"

    report = import_legacy_summary(source, output)

    assert report["valid"] is True
    assert report["classification"] == "legacy-diagnostic"
    assert report["official_results_allowed"] is False
    assert report["atomic_evidence_available"] is False
    assert (output / "source" / "standardized_summary.json").read_bytes() == original
    record = read_json(output / "legacy_import_record.json")
    assert record["conversion"] == {
        "atomic_evidence_available": False,
        "lossy_fields": [],
        "official_results_allowed": False,
        "reason_code": "legacy-summary-has-no-atomic-evidence",
        "status": "not-converted",
    }
    assert record["source"]["sha256"] == sha256_file(source)
    assert validate_legacy_import(output) == report


def test_legacy_import_fails_closed_on_unknown_schema_and_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "standardized_summary.json"
    payload = _legacy_summary()
    payload["official_results_allowed"] = True
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Additional properties"):
        import_legacy_summary(source, tmp_path / "invalid")

    payload.pop("official_results_allowed")
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(LegacyImportError, match="overwrite"):
        import_legacy_summary(source, output)


def test_legacy_import_rejects_unknown_metric_fields(tmp_path: Path) -> None:
    source = tmp_path / "standardized_summary.json"
    payload = _legacy_summary()
    payload["metrics"]["structural_fidelity"]["invented_score"] = 1.0
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Additional properties"):
        import_legacy_summary(source, tmp_path / "invalid")


def test_legacy_import_rejects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "standardized_summary.json"
    source.write_text(json.dumps(_legacy_summary()), encoding="utf-8")
    output = tmp_path / "legacy-import"
    import_legacy_summary(source, output)
    with (output / "source" / "standardized_summary.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(LegacyImportError, match="checksum"):
        validate_legacy_import(output)
