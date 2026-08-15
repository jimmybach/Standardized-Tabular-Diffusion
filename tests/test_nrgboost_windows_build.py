from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_nrgboost_windows.ps1"
LOCK = ROOT / "tools" / "nrgboost-windows-toolchain.explicit.txt"
RUNTIME_REQUIREMENTS = ROOT / "requirements-nrgboost-validation.txt"


def test_nrgboost_windows_build_locks_official_source_and_claim_boundary() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "nrgboost-0.0.3.tar.gz" in text
    assert "7b9e6a2a951755a75f34f1ec1185e82c4038938de6d126b046d46ce0624bbda0" in text
    assert 'source_code_modified = $false' in text
    assert "diagnostic-windows-source-build-not-authoritative-native-parity" in text
    assert "native-parity-run-30922326384.json" in text


def test_nrgboost_windows_build_is_fail_closed_and_verifies_clean_install() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'WorkRoot must be absent or empty' in text
    assert "WorkRoot must be at most 60 characters" in text
    assert 'build_ext", "--compiler=mingw32", "bdist_wheel' in text
    assert '"-m", "delvewheel", "repair"' in text
    assert '"-m", "pip", "check"' in text
    assert "NRGBoost extension smoke check passed" in text
    assert "Remove-Item" not in text
    assert not re.search(r"\b(patch|git apply)\b", text, flags=re.IGNORECASE)


def test_nrgboost_windows_toolchain_is_an_exact_hashed_conda_lock() -> None:
    lines = [
        line.strip()
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines[0] == "@EXPLICIT"
    packages = lines[1:]
    assert len(packages) == 26
    assert all(line.startswith("https://conda.anaconda.org/conda-forge/win-64/") for line in packages)
    assert all(re.search(r"#[0-9a-f]{32}$", line) for line in packages)
    assert any("m2w64-gcc-5.3.0-6" in line for line in packages)
    assert any("m2w64-toolchain-5.3.0-7" in line for line in packages)


def test_nrgboost_windows_only_runtime_dependency_is_pinned() -> None:
    requirements = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines()

    assert 'colorama==0.4.6; sys_platform == "win32"' in requirements
