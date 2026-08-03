from __future__ import annotations

import json
from pathlib import Path

from standardized_tabular_diffusion.validation.tabddpm import MANIFEST_RELATIVE_PATH, verify_sources

REPO_ROOT = Path(__file__).resolve().parents[1]


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
