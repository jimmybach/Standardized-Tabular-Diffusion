from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from standardized_tabular_diffusion.validation.core_ci import REQUIRED_WHEEL_FILES, inspect_wheel_contents


def _write_wheel(path: Path, names: set[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "fixture")


def test_wheel_inspection_accepts_required_portable_surface(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "fixture.whl", REQUIRED_WHEEL_FILES)
    inspect_wheel_contents(tmp_path)


def test_wheel_inspection_rejects_reference_tree_files(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "fixture.whl", REQUIRED_WHEEL_FILES | {"TabSyn-main/forbidden.py"})
    with pytest.raises(RuntimeError, match="excluded reference-tree"):
        inspect_wheel_contents(tmp_path)
