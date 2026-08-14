# Pipeline Real-Function Findings Ledger

Chinese translation: [PIPELINE_FINDINGS.zh-CN.md](PIPELINE_FINDINGS.zh-CN.md)

- Status: active ledger
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Last synchronized: 2026-08-14

## Rules

This file is the human-readable index of cross-baseline findings. It records confirmed facts, not suspicions. Large logs and generated artifacts remain outside Git; each row links to a compact reproduction, test, or retained evidence record when one exists.

Severity is independent of implementation-roadmap phases:

- **S0 Critical:** credential/data exposure, destructive cross-run behavior, or invalid published scientific evidence.
- **S1 High:** silent scientific error, leakage, wrong seed/config/data, stale result reuse, or false success.
- **S2 Medium:** explicit runtime failure, unsupported platform edge, or recoverability defect without silent scientific corruption.
- **S3 Low:** diagnostics, documentation, or usability issue with a safe failure mode.

Allowed states are `confirmed`, `fix-in-progress`, `fixed`, `verified`, `deferred`, and `accepted-blocker`. A finding becomes `verified` only after its regression test and the applicable real-function probe pass.

Every new row must include: finding ID, task, severity, affected adapters, observed behavior, expected behavior, reproduction/evidence, root cause, resolution, regression coverage, state, and claim boundary.

## Current findings

| ID | Task | Severity | Affected adapters | Finding and resolution | Evidence / regression | State |
|---|---|---:|---|---|---|---|
| RF-TABDIFF-001 | T01 | S1 | `tabdiff` | The upstream entry path did not expose the repository-requested config path and train/sample seeds. Checksum-guarded runtime overlays now bind both without changing tracked upstream files. | [`TABDIFF_VALIDATION.md`](../TABDIFF_VALIDATION.md); adapter/validation tests | verified |
| RF-TABDIFF-002 | T03 | S1 | `tabdiff` | Multiple generation seeds previously relied on the upstream default result location. The adapter now copies each decoded sample into a seed-specific run artifact and records its hash. | [Adult real-function evidence](../evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json) | verified |
| RF-TABDIFF-003 | T02 | S1 | `tabdiff` | The official Adult configuration could emit fractional values in declared integer columns when inverse dequantization was `none`. The standardized output contract now requires the official `round` path or rejects the sample. | [Adult real-function evidence](../evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json); integer-restoration tests | verified |
| RF-TABDIFF-004 | T04 | S2 | `tabdiff` | Current PyTorch and Windows execution required narrow scheduler, diagnostic-plot, encoding, and non-ASCII-path runtime handling. Each bridge is fail-closed and source-checksum guarded. | [`TABDIFF_VALIDATION.md`](../TABDIFF_VALIDATION.md); runtime overlay manifests | verified |
| RF-TABDIFF-005 | T04 | S2 | `tabdiff` | Run-metadata construction imported optional PyTorch unconditionally in the parent process, breaking dependency-light adapter contract tests. Metadata now records an explicit unavailable inspection state when PyTorch is absent while real model execution still requires its declared dependency. | Optional-dependency regression test; full core suite | verified |

These resolved TabDiff rows demonstrate why V1 simulation alone is insufficient. They do not imply that another adapter has the same defect, and they do not replace the planned cross-baseline audit.

## New finding template

~~~text
ID: RF-<MODEL-OR-CORE>-NNN
Task: T01-T06
Severity: S0-S3
Affected adapters:
Observed behavior:
Expected behavior:
Reproduction/evidence:
Root cause:
Resolution:
Regression coverage:
State:
Claim boundary:
~~~
