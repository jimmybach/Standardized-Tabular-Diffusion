from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.leaderboard import validate_snapshot_bundle
from standardized_tabular_diffusion.evaluation.serialization import read_json
from standardized_tabular_diffusion.quickstart import run_quickstart

pytestmark = [pytest.mark.integration, pytest.mark.evaluation, pytest.mark.adapter]


def test_packaged_quickstart_runs_official_adapter_to_diagnostic_snapshot(tmp_path: Path) -> None:
    try:
        observed = version("imbalanced-learn")
    except PackageNotFoundError:
        pytest.skip("quickstart extra is not installed")
    if observed != "0.14.2":
        pytest.skip("exact quickstart adapter dependency is not installed")

    output = tmp_path / "quickstart"
    report = run_quickstart(output, seed=17)

    assert report["valid"] is True
    assert report["classification"] == "partial-diagnostic"
    assert report["official_results_allowed"] is False
    assert validate_result_bundle(output / "evaluation-result").finalization_status == "finalized"
    snapshot = validate_snapshot_bundle(output / "diagnostic-snapshot")
    assert snapshot["publication_class"] == "partial-diagnostic"
    assert len(snapshot["leaderboard"]) == 1
    assert snapshot["leaderboard"][0]["rank"] is None
    request = read_json(output / "evaluation-result" / "config.yaml")
    assert request["subject_type"] == "adapter-run"
    assert request["model"] == {"model_id": "smote"}
    assert not (output / "adapter" / "standardized_summary.json").exists()
