from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APACHE_2_LICENSE_SHA256_LF = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"


def _sha256_lf(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def test_root_license_is_complete_apache_2_text() -> None:
    license_path = REPO_ROOT / "LICENSE"

    assert license_path.is_file()
    assert _sha256_lf(license_path) == APACHE_2_LICENSE_SHA256_LF


def test_package_metadata_declares_and_distributes_legal_files() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert all(not item.startswith("License ::") for item in project["classifiers"])


def test_notice_preserves_root_scope_and_third_party_boundaries() -> None:
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    inventory = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "The Standardized Tabular Diffusion Authors" in notice
    assert "Third-party source trees, packages, datasets, papers, and model weights" in notice
    assert "does not relicense any component listed here" in inventory


def test_contribution_policy_uses_dco_and_preserves_credit() -> None:
    policy = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    contributors = (REPO_ROOT / "CONTRIBUTORS.md").read_text(encoding="utf-8")

    assert "Developer Certificate of Origin 1.1" in policy
    assert "Signed-off-by" in policy
    assert "Contributor License Agreement" in policy
    assert "principal contributor to the initial version" in contributors
    assert "@jimmybach" in contributors
