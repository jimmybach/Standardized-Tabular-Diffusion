from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from standardized_tabular_diffusion.evaluation.serialization import read_json, sha256_file
from standardized_tabular_diffusion.model_inventory import MODEL_INVENTORY, get_inventory_entry
from standardized_tabular_diffusion.validation.core_ci import REQUIRED_WHEEL_FILES

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
P8_EVIDENCE = REPO_ROOT / "docs/evidence/evaluation/p8-native-windows11-py311-aae531b.json"
P8_EVIDENCE_SHA256 = "6a5d34c4f1845cb2600791a4e266905ffd9c508eb44be3a731f91c898000e8e7"


def test_release_version_and_citation_are_synchronized() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation_text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation = yaml.safe_load(citation_text)
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert project["version"] == "0.1.0rc1"
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == project["version"]
    assert citation["license"] == "Apache-2.0"
    assert citation["repository-code"] == project["urls"]["Repository"]
    assert "## 0.1.0rc1 - Unreleased" in changelog


def test_release_assets_and_bilingual_p8_guides_are_present() -> None:
    for relative in (
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "MANIFEST.in",
        "RELEASE_CHECKLIST.md",
        "docs/QUICKSTART.md",
        "docs/QUICKSTART.zh-CN.md",
        "docs/ARCHITECTURE.md",
        "docs/ARCHITECTURE.zh-CN.md",
        "docs/TROUBLESHOOTING.md",
        "docs/TROUBLESHOOTING.zh-CN.md",
        "docs/METRIC_CARDS.md",
        "docs/METRIC_CARDS.zh-CN.md",
        "docs/DATASET_CARDS.md",
        "docs/DATASET_CARDS.zh-CN.md",
        "docs/evaluation/P8_MIGRATION_AND_RELEASE.md",
        "docs/evaluation/P8_MIGRATION_AND_RELEASE.zh-CN.md",
    ):
        assert (REPO_ROOT / relative).is_file(), relative


def test_inventory_serializes_normative_statuses_separately_from_research_assessment() -> None:
    payload = get_inventory_entry("tabddpm").to_dict()
    assert payload["validation_level"] == "native-parity-validated"
    assert payload["benchmark_track"] == "experimental"
    assert payload["support_level"] == "unsupported"
    assert "implemented" not in payload
    assert payload["research_assessment"]["status_authority"] == "non-normative-landscape-review"


def test_every_inventory_entry_uses_exact_normative_status_dimensions() -> None:
    for entry in MODEL_INVENTORY.values():
        payload = entry.to_dict()
        assert "implemented" not in payload
        assert payload["validation_level"] in {
            "registered",
            "adapter-complete",
            "smoke-validated",
            "native-parity-validated",
        }
        assert payload["benchmark_track"] in {"experimental", "excluded"}
        assert payload["support_level"] == "unsupported"
        assert payload["research_assessment"]["status_authority"] == "non-normative-landscape-review"


def test_wheel_allowlist_contains_p8_migration_and_quickstart_contracts() -> None:
    assert {
        "standardized_tabular_diffusion/schemas/evaluation/legacy-import-record.schema.json",
        "standardized_tabular_diffusion/schemas/evaluation/legacy-standardized-summary.schema.json",
        "standardized_tabular_diffusion/resources/quickstart/train.csv",
        "standardized_tabular_diffusion/resources/quickstart/metadata.json",
    } <= REQUIRED_WHEEL_FILES


def test_release_document_local_links_resolve() -> None:
    documents = {
        REPO_ROOT / "README.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "RELEASE_CHECKLIST.md",
        REPO_ROOT / "SECURITY.md",
        *(REPO_ROOT / "docs" / name for name in (
            "ARCHITECTURE.md",
            "ARCHITECTURE.zh-CN.md",
            "DATASET_CARDS.md",
            "DATASET_CARDS.zh-CN.md",
            "METRIC_CARDS.md",
            "METRIC_CARDS.zh-CN.md",
            "QUICKSTART.md",
            "QUICKSTART.zh-CN.md",
            "TROUBLESHOOTING.md",
            "TROUBLESHOOTING.zh-CN.md",
        )),
        REPO_ROOT / "docs" / "evaluation" / "P8_MIGRATION_AND_RELEASE.md",
        REPO_ROOT / "docs" / "evaluation" / "P8_MIGRATION_AND_RELEASE.zh-CN.md",
    }
    broken: list[str] = []
    for document in sorted(documents):
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            if relative and not (document.parent / relative).resolve().exists():
                broken.append(f"{document.relative_to(REPO_ROOT).as_posix()} -> {target}")
    assert not broken, "Broken release-document links:\n" + "\n".join(broken)


def test_native_windows_11_gate_rejects_server_and_old_builds() -> None:
    from standardized_tabular_diffusion.validation.p8_release import _is_exact_native_windows_11

    expected = {"build": "26200", "product_type": "WinNT", "machine": "AMD64", "python_version": (3, 11)}
    assert _is_exact_native_windows_11(**expected)
    assert not _is_exact_native_windows_11(**(expected | {"product_type": "ServerNT"}))
    assert not _is_exact_native_windows_11(**(expected | {"build": "19045"}))
    assert not _is_exact_native_windows_11(**(expected | {"python_version": (3, 12)}))


def test_evaluation_config_rejects_negative_evaluator_seeds() -> None:
    from standardized_tabular_diffusion.config import EvaluationConfig

    with pytest.raises(ValueError, match="non-negative"):
        EvaluationConfig(evaluator_seeds=[-1])


def test_native_p8_evidence_is_immutable_and_binds_the_implementation_commit() -> None:
    assert sha256_file(P8_EVIDENCE) == P8_EVIDENCE_SHA256
    evidence = read_json(P8_EVIDENCE)
    assert evidence["status"] == "pass"
    assert evidence["phase"] == "P8"
    assert evidence["protocol_id"] == "p8-migration-release-exit-gate-v1"
    assert evidence["repository_commit"] == "aae531bf0345ac7c46db2e1d94193c97a402c4eb"
    assert set(evidence["exit_gates"].values()) == {"pass"}
    environment = evidence["environment"]
    assert environment["python"] == "3.11.15"
    assert environment["native_windows_release"]["build"] == "26200"
    assert environment["native_windows_release"]["product_type"] == "WinNT"
    assert environment["native_windows_release"]["is_exact_native_windows_11_python_311_x86_64"] is True
    assert "does not admit a model" in evidence["claim_boundary"]
    for relative, digest in evidence["locked_files"].items():
        assert (REPO_ROOT / relative).is_file()
        assert len(digest) == 64
