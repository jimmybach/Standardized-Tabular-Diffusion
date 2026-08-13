from __future__ import annotations

import json
from pathlib import Path

from standardized_tabular_diffusion.validation.workflow_evidence import (
    preserve_pre_protocol_failure,
)


def test_preserve_pre_protocol_failure_is_non_destructive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    output = tmp_path / "nested" / "evidence.json"

    assert preserve_pre_protocol_failure(
        output=output,
        protocol_id="example-v1",
        phase="P0",
        message="earlier step failed",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["repository_commit"] == "abc123"

    original = output.read_bytes()
    assert not preserve_pre_protocol_failure(
        output=output,
        protocol_id="changed",
        phase="changed",
        message="must not overwrite",
    )
    assert output.read_bytes() == original
