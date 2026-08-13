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


def test_required_wheel_surface_includes_complete_p5_identity_resources() -> None:
    assert {
        "standardized_tabular_diffusion/resources/evaluation/evaluators/p5-high-order-privacy-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/metrics/p5-excluded-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/metrics/p5-high-order-privacy-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/protocols/p5-high-order-privacy.json",
        "standardized_tabular_diffusion/resources/evaluation/upstream/p5-sources.json",
    } <= REQUIRED_WHEEL_FILES


def test_wheel_inspection_rejects_reference_tree_files(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "fixture.whl", REQUIRED_WHEEL_FILES | {"TabSyn-main/forbidden.py"})
    with pytest.raises(RuntimeError, match="excluded reference-tree"):
        inspect_wheel_contents(tmp_path)
