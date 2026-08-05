from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from standardized_tabular_diffusion.models.tabebm import (
    TABEBM_PACKAGE_VERSION,
    TABEBM_SDIST_SHA256,
    TABEBM_UPSTREAM_COMMIT,
    TabEBMAdapter,
    verify_tabebm_distribution,
)
from standardized_tabular_diffusion.registry import get_adapter_spec
from standardized_tabular_diffusion.validation import tabebm as tabebm_validation

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabebm" / "smoke-validation-run-30974574544.json"
EVIDENCE_SHA256 = "8d461e440440d73213f31efe1b8086e9c78fed299822da2fe203ea62af3c21dc"
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def test_tabebm_package_lock_and_claim_boundary() -> None:
    spec = get_adapter_spec("tabebm")
    assert spec.source_authority == "method-author"
    assert spec.distribution_form == "package"
    assert spec.modification_status == "adapter-only"
    assert spec.task_types == ("classification",)
    assert spec.install_extra == "tabebm"
    assert spec.upstream_revision == TABEBM_UPSTREAM_COMMIT
    assert TABEBM_PACKAGE_VERSION == "2025.8.19"
    assert TABEBM_SDIST_SHA256 == "6111611326747a680f93dfadcbac1d602ce20cb722b9b6cbff1f556b9f48d503"
    assert tabebm_validation.PROTOCOL_ID == TabEBMAdapter.protocol_id


def test_tabebm_locked_sdist_when_provided() -> None:
    sdist = os.environ.get("TABEBM_SDIST_PATH")
    if sdist is None:
        pytest.skip("TABEBM_SDIST_PATH is provided by the authoritative validation workflow")
    result = tabebm_validation._verify_sdist(Path(sdist))
    assert result["regular_files_verified"] == 12
    assert verify_tabebm_distribution()["version"] == TABEBM_PACKAGE_VERSION


def test_tabebm_round_robin_requires_exact_capacity() -> None:
    rows, labels = TabEBMAdapter._round_robin(
        {0: np.asarray([[0.0], [1.0]]), 1: np.asarray([[10.0], [11.0]])},
        requested=3,
    )
    np.testing.assert_array_equal(rows[:, 0], np.asarray([0.0, 10.0, 1.0]))
    assert labels.tolist() == [0, 1, 0]
    with pytest.raises(RuntimeError, match="too few rows"):
        TabEBMAdapter._round_robin({0: np.asarray([[0.0]])}, requested=2)


def test_tabebm_retained_smoke_evidence_preserves_claim_boundary() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    evidence = json.loads(evidence_bytes)
    assert evidence["status"] == "pass"
    assert evidence["validation_level"] == "smoke-validated"
    assert evidence["result_summary"]["full_tabpfn_generation_executed"] is False
    validation = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tabebm"]["validation"]
    assert validation["level"] == "smoke-validated"
    assert validation["workflow_run_id"] == 30974574544
    assert validation["artifact"]["evidence_file_sha256"] == EVIDENCE_SHA256
    spec = get_adapter_spec("tabebm")
    assert spec.validation_level.value == "smoke-validated"
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"
