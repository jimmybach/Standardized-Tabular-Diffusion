from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.validation import p1_foundation

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_p1_evidence_is_machine_readable_and_excludes_tabstruct_runtime(tmp_path: Path) -> None:
    output = tmp_path / "p1-evidence.json"
    evidence = p1_foundation.run_validation(output)

    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P1"
    assert evidence["result_summary"]["scientific_metrics_executed"] is False
    assert evidence["result_summary"]["finalized_bundle_created"] is False
    assert "standardized_tabular_diffusion/evaluation/tabstruct.py" not in evidence["locked_files"]
    assert read_json(output) == evidence


def test_authoritative_p1_evidence_fails_outside_primary_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p1_foundation, "is_primary_release_environment", lambda: False)

    with pytest.raises(AssertionError, match="Windows with Python 3.11"):
        p1_foundation._assert_primary_environment()


def test_authoritative_p1_evidence_accepts_primary_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p1_foundation, "is_primary_release_environment", lambda: True)
    p1_foundation._assert_primary_environment()
