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
| RF-CORE-001 | T01 | S1 | `arf`, `bn`, `ctab-gan`, `ctab-gan-plus`, `ctgan`, `great`, `nflow`, `nrgboost`, `smote`, `tabddpm`, `tabdiff`, `tabebm`, `tabsds`, `tabsyn`, `tabula`, `tvae` | Unknown top-level action controls are not uniformly rejected, so a misspelled option can silently use a default. Phase 2 must add strict allowlists without changing model mathematics. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CORE-002 | T02 | S1 | all 21 adapters through direct `run`/`run-action` | Shared preflight records dataset paths and existence, but not size or SHA-256, so it does not content-bind the direct run to the registered dataset identity. Phase 2 must add common content binding while retaining stronger model-specific/P6 checks. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CORE-003 | T03 | S1 | all 21 adapters through direct `run`/`run-action` | The shared output boundary accepts an existing directory without a universal run/seed identity or non-empty policy; a later sample can replace `samples.csv`. Phase 2 must isolate or reject conflicting output ownership. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CORE-004 | T04 | S2 | `great` training, `tabula` training, `nrgboost`, `smote`, `tabddpm`, `tabsds` | These paths ignore the requested device or delegate automatic selection without a fail-closed observation. Phase 2 must make CPU-only rejection and CPU/CUDA selection explicit and observable. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-INTERNAL-DATA-001 | T02 | S1 | `codi`, `stasy`, `tabddpm`, `tabdiff`, `tabsyn` | Native runtimes consume an internal layout or TOML path, while the shared boundary does not prove that content matches the embedded canonical `DatasetSpec`. Phase 2 must checksum-bind or deterministically materialize the native view. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-TABDDPM-001 | T01 | S1 | `tabddpm` | The ordinary adapter passes only the TOML path and action flag. Controlled simulation showed that changing seed, device, requested rows, and output directory leaves the upstream command unchanged. Phase 2 must bind these controls explicitly. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-TABDDPM-002 | T03 | S1 | `tabddpm` | Output ownership remains in TOML and the sample bundle does not expose `generated_sample_path`, so the normal top-level route cannot automatically pass the table to central evaluation. Phase 2 must make sample ownership explicit. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-GOGGLE-001 | T01 | S1 | `goggle` | Sampling reseeds from the training runtime configuration rather than the independent sample `RunSpec.seed`. Phase 2 must pass and apply the sample seed. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-CTGAN-FAMILY-001 | T01 | S1 | `ctgan`, `tvae` | Training sets the official model random state, but sampling reloads the checkpoint and calls `sample()` without resetting it from the requested sample seed. Phase 2 must restore official-package random-state control before generation. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |
| RF-UPSTREAM-WORKSPACE-001 | T03 | S1 | `tabddpm`, `tabdiff`, `tabsyn` | Mutable checkpoints or results can remain in upstream worktrees unless model-specific redirection/copying occurs. Phase 2 must place mutable artifacts under declared run ownership without editing authoritative mathematics. | [Phase 1 report](PHASE_1_LOGIC_AUDIT_REPORT.md); [immutable evidence](../evidence/audits/pipeline-phase1-logic-audit-20260814.json) | confirmed |

The resolved TabDiff rows demonstrate why V1 simulation alone is insufficient. The ten confirmed Phase 1 rows are bounded V0/V1 findings; they do not claim that V2 failed, and they remain open until both regression coverage and the applicable real-function probe pass.

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
