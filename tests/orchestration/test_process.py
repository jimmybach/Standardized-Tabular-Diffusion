from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion.orchestration.process import ProcessLimits, run_isolated_process

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_process_timeout_is_explicit_and_terminates_the_stage(tmp_path: Path) -> None:
    outcome = run_isolated_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        log_path=tmp_path / "timeout.jsonl",
        stage_id="timeout-fixture",
        limits=ProcessLimits(timeout_seconds=0.2, poll_interval_seconds=0.02),
    )
    assert outcome.timed_out is True
    assert outcome.memory_limit_exceeded is False
    assert outcome.wall_seconds < 5


def test_process_tree_memory_limit_is_explicit(tmp_path: Path) -> None:
    pytest.importorskip("psutil")
    outcome = run_isolated_process(
        [sys.executable, "-c", "import time; payload = bytearray(32 * 1024 * 1024); time.sleep(5)"],
        cwd=tmp_path,
        log_path=tmp_path / "memory.jsonl",
        stage_id="memory-fixture",
        limits=ProcessLimits(memory_limit_bytes=8 * 1024 * 1024, poll_interval_seconds=0.02),
    )
    assert outcome.memory_limit_exceeded is True
    assert outcome.peak_rss_bytes is not None
    assert outcome.peak_rss_bytes > 8 * 1024 * 1024


def test_structured_logs_redact_credentials_and_declared_paths(tmp_path: Path) -> None:
    unsafe = str(tmp_path.resolve())
    outcome = run_isolated_process(
        [sys.executable, "-c", "print('api_key=super-secret-value'); print('PATH_VALUE')"],
        cwd=tmp_path,
        log_path=tmp_path / "redacted.jsonl",
        stage_id="redaction-fixture",
        limits=ProcessLimits(timeout_seconds=5),
        environment={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"},
        path_aliases={unsafe: "<fixture-root>", "PATH_VALUE": "<unsafe-path>"},
    )
    assert outcome.exit_code == 0
    text = (tmp_path / "redacted.jsonl").read_text(encoding="utf-8")
    assert "super-secret-value" not in text
    assert "<redacted>" in text
    assert "<unsafe-path>" in text
    for line in text.splitlines():
        assert isinstance(json.loads(line), dict)
