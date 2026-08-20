from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from standardized_tabular_diffusion.registry import list_adapter_specs

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = REPO_ROOT / "configs/validation/pipeline-real-function-audit-v1.json"
AUDIT_DOC = REPO_ROOT / "docs/audits/PIPELINE_REAL_FUNCTION_AUDIT.md"
AUDIT_DOC_ZH = REPO_ROOT / "docs/audits/PIPELINE_REAL_FUNCTION_AUDIT.zh-CN.md"
FINDINGS_DOC = REPO_ROOT / "docs/audits/PIPELINE_FINDINGS.md"
FINDINGS_DOC_ZH = REPO_ROOT / "docs/audits/PIPELINE_FINDINGS.zh-CN.md"
PHASE_1_REPORT = REPO_ROOT / "docs/audits/PHASE_1_LOGIC_AUDIT_REPORT.md"
PHASE_1_REPORT_ZH = REPO_ROOT / "docs/audits/PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md"
PHASE_1_EVIDENCE = REPO_ROOT / "docs/evidence/audits/pipeline-phase1-logic-audit-20260814.json"
PHASE_2_REPORT = REPO_ROOT / "docs/audits/PHASE_2_REMEDIATION_REPORT.md"
PHASE_2_REPORT_ZH = REPO_ROOT / "docs/audits/PHASE_2_REMEDIATION_REPORT.zh-CN.md"
PHASE_2_EVIDENCE = REPO_ROOT / "docs/evidence/audits/pipeline-phase2-remediation-20260814.json"
PHASE_3_REPORT = REPO_ROOT / "docs/audits/PHASE_3_V2_WINDOWS_PROTOCOL.md"
PHASE_3_REPORT_ZH = REPO_ROOT / "docs/audits/PHASE_3_V2_WINDOWS_PROTOCOL.zh-CN.md"
LOCAL_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
FINDING_ID = re.compile(r"RF-[A-Z0-9-]+")
SHA256 = re.compile(r"[0-9a-f]{64}")


def _load_audit() -> dict[str, object]:
    with AUDIT_CONFIG.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_phase_1_evidence() -> dict[str, object]:
    with PHASE_1_EVIDENCE.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_phase_2_evidence() -> dict[str, object]:
    with PHASE_2_EVIDENCE.open(encoding="utf-8") as stream:
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
        "minimal_real_passed": audit_states.count("minimal-real-passed"),
        "minimal_real_pending": audit_states.count("minimal-real-pending"),
        "externally_blocked": audit_states.count("externally-blocked"),
    }


def test_audit_assets_and_declared_evidence_exist() -> None:
    audit = _load_audit()
    required = [
        AUDIT_DOC,
        AUDIT_DOC_ZH,
        FINDINGS_DOC,
        FINDINGS_DOC_ZH,
        PHASE_1_REPORT,
        PHASE_1_REPORT_ZH,
        PHASE_1_EVIDENCE,
        PHASE_2_REPORT,
        PHASE_2_REPORT_ZH,
        PHASE_2_EVIDENCE,
        PHASE_3_REPORT,
        PHASE_3_REPORT_ZH,
    ]
    for relative in audit["task_files"]:
        task = REPO_ROOT / relative
        required.extend((task, task.with_name(f"{task.stem}.zh-CN{task.suffix}")))
    for path in required:
        assert path.is_file(), path
    for row in audit["baselines"]:
        for relative in row["evidence"]:
            assert (REPO_ROOT / relative).is_file(), relative


def test_audit_document_local_links_resolve() -> None:
    documents = {
        AUDIT_DOC,
        AUDIT_DOC_ZH,
        FINDINGS_DOC,
        FINDINGS_DOC_ZH,
        PHASE_1_REPORT,
        PHASE_1_REPORT_ZH,
        PHASE_2_REPORT,
        PHASE_2_REPORT_ZH,
        PHASE_3_REPORT,
        PHASE_3_REPORT_ZH,
    }
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


def test_phase_1_evidence_covers_every_registered_adapter_and_task() -> None:
    evidence = _load_phase_1_evidence()
    baselines = evidence["baselines"]
    summary = evidence["summary"]
    assert evidence["status"] == "passed-with-confirmed-findings"
    assert isinstance(baselines, list)
    by_model = {row["model_id"]: row for row in baselines}
    assert len(baselines) == len(by_model) == 21
    assert set(by_model) == set(list_adapter_specs())
    for row in baselines:
        assert set(row) == {"model_id", "source_file", "t01", "t02", "t03", "t04"}
        assert (REPO_ROOT / row["source_file"]).is_file()
    assert summary["models_audited"] == len(baselines)
    assert summary["task_model_cells_completed"] == len(baselines) * 4 == 84
    assert summary["configuration_projection_passed"] == len(baselines)
    assert summary["v2_minimal_real_execution_started"] is False


def test_phase_1_finding_references_are_complete_and_reciprocal() -> None:
    evidence = _load_phase_1_evidence()
    findings = evidence["findings"]
    baselines = evidence["baselines"]
    summary = evidence["summary"]
    by_finding = {finding["id"]: finding for finding in findings}
    by_model = {row["model_id"]: row for row in baselines}
    assert len(findings) == len(by_finding) == summary["confirmed_findings"] == 10
    assert summary["severity_counts"] == {
        severity: Counter(finding["severity"] for finding in findings).get(severity, 0)
        for severity in ("S0", "S1", "S2", "S3")
    }

    for model_id, row in by_model.items():
        for task in ("T01", "T02", "T03", "T04"):
            for finding_id in row[task.lower()]:
                finding = by_finding[finding_id]
                assert finding["task"] == task
                assert model_id in finding["affected_models"]

    for finding in findings:
        assert finding["state"] == "confirmed"
        for model_id in finding["affected_models"]:
            assert finding["id"] in by_model[model_id][finding["task"].lower()]


def test_phase_1_snapshot_agrees_with_plan_and_has_bound_source_inventory() -> None:
    audit = _load_audit()
    phase = audit["phase_1_logic_audit"]
    evidence = _load_phase_1_evidence()
    summary = evidence["summary"]
    assert phase["state"] == "complete-with-confirmed-findings"
    for field in (
        "models_audited",
        "task_model_cells_completed",
        "configuration_projection_passed",
        "confirmed_findings",
        "severity_counts",
        "v2_minimal_real_execution_started",
    ):
        assert phase[field] == summary[field]
    for field in ("report", "report_zh_cn", "evidence"):
        assert (REPO_ROOT / phase[field]).is_file()
    assert re.fullmatch(r"[0-9a-f]{40}", evidence["audited_repository_commit"])
    source_hashes = evidence["audited_files_sha256"]
    assert len(source_hashes) >= 20
    for relative, digest in source_hashes.items():
        assert (REPO_ROOT / relative).is_file(), relative
        assert SHA256.fullmatch(digest), relative


def test_phase_2_evidence_records_all_fixes_without_overclaiming_v2() -> None:
    audit = _load_audit()
    phase = audit["phase_2_remediation"]
    evidence = _load_phase_2_evidence()
    findings = evidence["findings"]
    assert evidence["status"] == "remediation-regression-passed-v2-pending"
    assert phase["state"] == "complete-v2-pending"
    assert len(findings) == phase["findings_fixed"] == 10
    assert phase["findings_verified"] == 0
    assert {row["state"] for row in findings} == {"fixed"}
    assert phase["v2_minimal_real_execution_started"] is False
    targeted = evidence["test_results"]["targeted_phase2"]
    full = evidence["test_results"]["full_repository"]
    assert phase["targeted_tests_passed"] == targeted["passed"]
    assert phase["full_repository_tests_passed"] == full["passed"]
    assert phase["full_repository_tests_skipped"] == full["skipped"]
    remediation_commit = evidence["remediation_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", remediation_commit)
    assert (
        subprocess.check_output(
            ["git", "rev-parse", remediation_commit],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        == remediation_commit
    )
    for field in ("report", "report_zh_cn", "evidence"):
        assert (REPO_ROOT / phase[field]).is_file()
    for relative, digest in evidence["retained_files_sha256"].items():
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        assert SHA256.fullmatch(digest), relative
        retained_bytes = subprocess.check_output(
            ["git", "show", f"{remediation_commit}:{relative}"],
            cwd=REPO_ROOT,
        )
        assert hashlib.sha256(retained_bytes).hexdigest() == digest, relative


def test_phase_3_snapshot_records_completed_windows_execution() -> None:
    audit = _load_audit()
    phase = audit["phase_3_v2_windows"]
    summary = audit["inventory_summary"]

    assert audit["overall_state"] == "complete-with-one-external-block"
    assert phase["state"] == "complete"
    assert phase["scheduled_minimal_real"] == phase["minimal_real_passed"] == 18
    assert phase["minimal_real_passed"] == summary["minimal_real_passed"]
    assert phase["representative_real_retained"] == summary["representative_real_passed"] == 2
    assert phase["externally_blocked"] == summary["externally_blocked"] == 1
    assert phase["minimal_real_pending"] == summary["minimal_real_pending"] == 0
    assert phase["registered_identities_accounted_for"] == summary["registered"] == 21
    for field in ("report", "report_zh_cn"):
        assert (REPO_ROOT / phase[field]).is_file()


def test_current_audit_task_documents_do_not_remain_planned() -> None:
    audit = _load_audit()
    for relative in audit["task_files"]:
        task = REPO_ROOT / relative
        translated = task.with_name(f"{task.stem}.zh-CN{task.suffix}")
        assert "planned across all 21 adapters" not in task.read_text(encoding="utf-8")
        assert "计划覆盖全部 21 个适配器" not in translated.read_text(encoding="utf-8")
