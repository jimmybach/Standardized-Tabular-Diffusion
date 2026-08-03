from __future__ import annotations

import hashlib
import json
from pathlib import Path

from standardized_tabular_diffusion.validation.tabddpm import MANIFEST_RELATIVE_PATH, verify_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabddpm" / "native-parity-run-30863212268.json"


def test_tabddpm_source_manifest_matches_pinned_sources() -> None:
    result = verify_sources(REPO_ROOT)

    assert result["upstream_commit"] == "b476257dd460b778ba09eb97f7a51d6490fa17f8"
    assert result["upstream_files_verified"] == 64
    assert result["libzero_modules_verified"] == 7


def test_tabddpm_source_manifest_has_unique_complete_paths() -> None:
    payload = json.loads((REPO_ROOT / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    paths = [record["path"] for record in payload["files"]]
    zero_paths = [record["path"] for record in payload["dependencies"]["libzero"]["vendored_modules"]]

    assert len(paths) == len(set(paths)) == 64
    assert len(zero_paths) == len(set(zero_paths)) == 7
    assert all(len(record["sha256_lf"]) == 64 for record in payload["files"])
    assert all(len(record["sha256"]) == 64 for record in payload["dependencies"]["libzero"]["vendored_modules"])


def test_tabddpm_native_parity_evidence_is_complete_and_immutable() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == "8fd277aef64a2e7225626a95379ecf67462ac686a4d688a56748f9ef965dd29e"
    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "3339af2603bac7a4736e68d7f369194b6b095653"
    assert len(evidence["cases"]) == 3
    for case in evidence["cases"]:
        assert case["status"] == "pass"
        comparisons = case["comparisons"]
        assert comparisons["config_exact"] is True
        assert comparisons["model"]["tensor_values_exact"] is True
        assert comparisons["ema_model"]["tensor_values_exact"] is True
        assert comparisons["loss_csv_exact"] is True
        assert comparisons["sample_rows"] == 12
        assert all(record["exact"] and record["finite"] for record in comparisons["generated_arrays"].values())
