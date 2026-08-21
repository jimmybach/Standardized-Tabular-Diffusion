from __future__ import annotations

import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from standardized_tabular_diffusion.validation.core_ci import (
    REQUIRED_SDIST_FILES,
    REQUIRED_WHEEL_FILES,
    inspect_sdist_contents,
    inspect_wheel_contents,
    verify_clean_distribution_install,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_wheel(path: Path, names: set[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "fixture")


def _write_sdist(path: Path, names: set[str]) -> None:
    fixture_root = path.parent / "fixture-0.1.0"
    fixture_root.mkdir()
    for name in names:
        target = fixture_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture", encoding="utf-8")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(fixture_root, arcname=fixture_root.name)


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


def test_sdist_inspection_accepts_documented_release_surface(tmp_path: Path) -> None:
    _write_sdist(tmp_path / "fixture.tar.gz", REQUIRED_SDIST_FILES)
    inspect_sdist_contents(tmp_path)


def test_sdist_inspection_rejects_materialized_upstream_tree(tmp_path: Path) -> None:
    _write_sdist(tmp_path / "fixture.tar.gz", REQUIRED_SDIST_FILES | {"TabDDPM-main/forbidden.py"})
    with pytest.raises(RuntimeError, match="excluded reference-tree"):
        inspect_sdist_contents(tmp_path)


def test_sdist_inspection_rejects_local_machine_paths(tmp_path: Path) -> None:
    names = REQUIRED_SDIST_FILES | {"docs/private.json"}
    _write_sdist(tmp_path / "fixture.tar.gz", names)
    private = tmp_path / "fixture-0.1.0" / "docs" / "private.json"
    private.write_text(r'{"path":"C:\\Users\\alice\\private\\run.json"}', encoding="utf-8")
    with tarfile.open(tmp_path / "fixture.tar.gz", "w:gz") as archive:
        archive.add(tmp_path / "fixture-0.1.0", arcname="fixture-0.1.0")
    with pytest.raises(RuntimeError, match="developer-local"):
        inspect_sdist_contents(tmp_path)


def test_sdist_inspection_rejects_credential_like_members(tmp_path: Path) -> None:
    _write_sdist(tmp_path / "fixture.tar.gz", REQUIRED_SDIST_FILES | {"config/private.pem"})
    with pytest.raises(RuntimeError, match="credential-like"):
        inspect_sdist_contents(tmp_path)


def test_psutil_profiles_preserve_p6_exactness_and_p4_compatibility() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        extras = tomllib.load(stream)["project"]["optional-dependencies"]

    assert "psutil==5.9.8" in extras["orchestration"]
    assert [item for item in extras["test"] if item.startswith("psutil")] == ["psutil>=5.9.8,<7"]
    assert [item for item in extras["dev"] if item.startswith("psutil")] == ["psutil>=5.9.8,<7"]


def test_wheel_inspection_rejects_reference_tree_files(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "fixture.whl", REQUIRED_WHEEL_FILES | {"TabSyn-main/forbidden.py"})
    with pytest.raises(RuntimeError, match="excluded reference-tree"):
        inspect_wheel_contents(tmp_path)


@pytest.mark.parametrize(
    "member, message",
    [
        ("C:/Users/alice/private.txt", "unsafe path"),
        ("docs/CON.txt", "non-portable path"),
    ],
)
def test_wheel_inspection_rejects_cross_platform_unsafe_member_names(
    tmp_path: Path, member: str, message: str
) -> None:
    _write_wheel(tmp_path / "fixture.whl", REQUIRED_WHEEL_FILES | {member})
    with pytest.raises(RuntimeError, match=message):
        inspect_wheel_contents(tmp_path)


def test_wheel_inspection_rejects_case_insensitive_member_collisions(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "fixture.whl", REQUIRED_WHEEL_FILES | {"docs/Guide.md", "docs/guide.md"})
    with pytest.raises(RuntimeError, match="non-portable-colliding"):
        inspect_wheel_contents(tmp_path)


def test_clean_install_rejects_unknown_artifact_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported artifact kind"):
        verify_clean_distribution_install(tmp_path, "archive")


@pytest.mark.parametrize("artifact_kind", ["wheel", "sdist"])
def test_clean_install_requires_exactly_one_artifact(tmp_path: Path, artifact_kind: str) -> None:
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        verify_clean_distribution_install(tmp_path, artifact_kind)
