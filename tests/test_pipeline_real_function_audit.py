from __future__ import annotations

import json
import re
from pathlib import Path

from standardized_tabular_diffusion.registry import list_adapter_specs

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = REPO_ROOT / "configs/validation/pipeline-real-function-audit-v1.json"
AUDIT_DOC = REPO_ROOT / "docs/audits/PIPELINE_REAL_FUNCTION_AUDIT.md"
AUDIT_DOC_ZH = REPO_ROOT / "docs/audits/PIPELINE_REAL_FUNCTION_AUDIT.zh-CN.md"
FINDINGS_DOC = REPO_ROOT / "docs/audits/PIPELINE_FINDINGS.md"
FINDINGS_DOC_ZH = REPO_ROOT / "docs/audits/PIPELINE_FINDINGS.zh-CN.md"
LOCAL_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
FINDING_ID = re.compile(r"RF-[A-Z0-9-]+")


def _load_audit() -> dict[str, object]:
    with AUDIT_CONFIG.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_audit_inventory_matches_runtime_registry() -> None:
    audit = _load_audit()
    rows = audit["baselines"]
    assert isinstance(rows, list)
    registry = list_adapter_specs()
    by_model = {row["model_id"]: row for row in rows}
    assert len(rows) == len(by_model) == 21
    assert set(by_model) == set(registry)
    for model_id, spec in registry.items():
        assert by_model[model_id]["registry_validation_level"] == spec["validation_level"]


def test_audit_summary_is_derived_from_declared_rows() -> None:
    audit = _load_audit()
    rows = audit["baselines"]
    summary = audit["inventory_summary"]
    assert isinstance(rows, list)
    assert isinstance(summary, dict)
    registry_levels = [row["registry_validation_level"] for row in rows]
    audit_states = [row["windows_real_function_state"] for row in rows]
    assert summary == {
        "registered": len(rows),
        "registry_native_parity_validated": registry_levels.count("native-parity-validated"),
        "registry_smoke_validated": registry_levels.count("smoke-validated"),
        "representative_real_passed": audit_states.count("representative-real-passed"),
        "minimal_real_pending": audit_states.count("minimal-real-pending"),
        "externally_blocked": audit_states.count("externally-blocked"),
    }


def test_audit_assets_and_declared_evidence_exist() -> None:
    audit = _load_audit()
    required = [AUDIT_DOC, AUDIT_DOC_ZH, FINDINGS_DOC, FINDINGS_DOC_ZH]
    for relative in audit["task_files"]:
        task = REPO_ROOT / relative
        required.extend((task, task.with_name(f"{task.stem}.zh-CN{task.suffix}")))
    for path in required:
        assert path.is_file(), path
    for row in audit["baselines"]:
        for relative in row["evidence"]:
            assert (REPO_ROOT / relative).is_file(), relative


def test_audit_document_local_links_resolve() -> None:
    documents = {AUDIT_DOC, AUDIT_DOC_ZH, FINDINGS_DOC, FINDINGS_DOC_ZH}
    audit = _load_audit()
    for relative in audit["task_files"]:
        task = REPO_ROOT / relative
        documents.update((task, task.with_name(f"{task.stem}.zh-CN{task.suffix}")))
    broken: list[str] = []
    for document in sorted(documents):
        for raw_target in LOCAL_LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            if relative and not (document.parent / relative).resolve().exists():
                broken.append(f"{document.relative_to(REPO_ROOT).as_posix()} -> {target}")
    assert not broken, "Broken audit-document links:\n" + "\n".join(broken)


def test_findings_ledger_translations_contain_the_same_issue_ids() -> None:
    english = set(FINDING_ID.findall(FINDINGS_DOC.read_text(encoding="utf-8")))
    chinese = set(FINDING_ID.findall(FINDINGS_DOC_ZH.read_text(encoding="utf-8")))
    assert english
    assert english == chinese
