from __future__ import annotations

from pathlib import Path

import pytest

from standardized_tabular_diffusion.orchestration import environment

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_non_git_install_uses_stable_installed_tree_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(environment, "_git", lambda *args, **kwargs: None)

    first = environment.capture_repository_state(tmp_path)
    second = environment.capture_repository_state(tmp_path)

    assert first == second
    assert first["commit"].startswith("installed-tree-")
    assert first["dirty"] is False
    assert first["untracked_file_count"] == 0
    assert len(first["fingerprint"]) == 64
