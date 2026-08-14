# Phase 1 Cross-Baseline Logic Audit Report

Chinese translation: [PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md](PHASE_1_LOGIC_AUDIT_REPORT.zh-CN.md)

- Status: complete with confirmed findings
- Audit date: 2026-08-14
- Audited commit: `e80296340f6a083eead55fded24a5ea6027d5c93`
- Scope: T01-T04, V0 static review plus V1 controlled simulation, all 21 registered adapters
- Evidence: [`pipeline-phase1-logic-audit-20260814.json`](../evidence/audits/pipeline-phase1-logic-audit-20260814.json)

## 1. Outcome

Phase 1 completed all **84 task/model cells**: four dependency-light audit tasks for each of 21 adapters. Configuration projection passed for every adapter: distinct train and sample seeds, the requested device, row count, action-specific controls, and the embedded `DatasetSpec` all reach `RunSpec` correctly.

The downstream review confirmed **10 open findings**: nine S1 high-severity contract gaps and one S2 device-boundary gap. No S0 credential, destructive deletion, or sensitive-data exposure was found. No finding changes a retained upstream-parity result because this audit examines the broader public pipeline contract on a separate axis.

Phase 1 did not train a real model and did not modify adapter behavior. Remediation belongs to Phase 2; V2 native-Windows minimal-real execution starts only after the S1 shared boundaries are corrected.

## 2. What was checked

For every adapter, the audit reviewed:

- configuration precedence and action-specific train/sample controls;
- downstream consumption of train and sample seeds;
- canonical dataset paths versus upstream-native dataset layouts;
- decoded row, column, missing-value, and type postconditions;
- checkpoint, sample, metadata, and result ownership;
- non-empty, reused, and symlinked output behavior;
- requested CPU/CUDA device behavior and unintended fallback; and
- Windows-safe subprocess arguments, paths, and environment handling.

The review bound all relevant repository-owned adapter files, launchers, shared configuration/runner files, and the directly involved upstream entry points by SHA-256. Controlled simulations additionally proved the TabDDPM argument-invariance defect and the shared absence of direct-preflight data digests.

## 3. Confirmed findings

### 3.1 T01: seed and configuration

`RF-TABDDPM-001` is the highest-priority model-specific defect. The ordinary TabDDPM adapter passes only `--config <path> --train/--sample`; changing `RunSpec.seed`, `RunSpec.device`, `RunSpec.num_samples`, or `RunSpec.output_dir` does not change the upstream invocation. A controlled simulation varied all four and observed the same command.

`RF-GOGGLE-001` shows that Goggle sampling reseeds from the training runtime configuration. The independent sample seed reaches `RunSpec` but not the launcher configuration.

`RF-CTGAN-FAMILY-001` affects CTGAN and TVAE. Training applies `spec.seed`, but sampling loads the checkpoint and calls `model.sample()` without resetting the official package random state from the requested sample seed.

`RF-CORE-001` affects 16 adapters. Their action-control parsing accepts unknown top-level extras instead of failing closed, so a misspelled tuning option can silently fall back to a default. CoDi, Goggle, REaLTabFormer, STaSy, and TabularARGN already have strict top-level control allowlists.

### 3.2 T02: data and output contract

`RF-CORE-002` affects the direct public runner for all 21 adapters. Preflight records dataset paths and existence, but not file size or SHA-256. Some individual adapters later bind data, and P6 has stronger content-addressed orchestration, but the common direct `run`/`run-action` boundary is not content-bound.

`RF-INTERNAL-DATA-001` affects CoDi, STaSy, TabDDPM, TabDiff, and TabSyn. Their authoritative runtimes read native internal layouts or a TOML path rather than the embedded canonical `DatasetSpec` paths. The current preflight does not prove that those internal files have the same content as the registered dataset identity, so a stale native-layout copy can be selected.

### 3.3 T03: checkpoints and artifacts

`RF-CORE-003` affects direct runs of all 21 adapters. The common output boundary creates an existing directory without a run/seed identity or a universal non-empty policy. Sampling normally writes `samples.csv`; a later seed in the same output directory can replace it. P6 content-addressed execution mitigates this only when the P6 path is used.

`RF-TABDDPM-002` shows that TabDDPM artifact ownership remains inside TOML. Its sample bundle does not expose `generated_sample_path`, so the normal top-level pipeline cannot automatically hand the generated table to central evaluation.

`RF-UPSTREAM-WORKSPACE-001` affects TabDDPM, TabDiff, and TabSyn. Their native training/checkpoint or result layouts remain inside upstream worktrees unless model-specific runtime configuration redirects or copies them. This conflicts with the new audit policy that tracked source trees are read-only runtimes and every mutable artifact belongs to the declared run directory.

### 3.4 T04: Windows, GPU, and dependencies

`RF-CORE-004` affects GReaT training, Tabula training, NRGBoost, SMOTE, TabDDPM, and TabSDS. These paths either ignore `RunSpec.device` or delegate automatic device selection without recording a fail-closed observation. CPU-only algorithms must explicitly reject a CUDA request; automatic trainer paths must honor an explicit CPU request and record the observed device.

No additional static non-ASCII quoting defect was confirmed. That is a V1 result only: real Windows qualification still requires each non-blocked adapter's V2 run from the existing repository path.

## 4. Per-model result matrix

An empty cell means no model-specific or shared finding was assigned for that task at V0/V1. It does not mean V2 passed.

| Model | T01 | T02 | T03 | T04 |
|---|---|---|---|---|
| ARF | CORE-001 | CORE-002 | CORE-003 | — |
| BN | CORE-001 | CORE-002 | CORE-003 | — |
| CoDi | — | CORE-002, INTERNAL-DATA-001 | CORE-003 | — |
| CTAB-GAN | CORE-001 | CORE-002 | CORE-003 | — |
| CTAB-GAN+ | CORE-001 | CORE-002 | CORE-003 | — |
| CTGAN | CORE-001, CTGAN-FAMILY-001 | CORE-002 | CORE-003 | — |
| Goggle | GOGGLE-001 | CORE-002 | CORE-003 | — |
| GReaT | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| NFlow | CORE-001 | CORE-002 | CORE-003 | — |
| NRGBoost | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| REaLTabFormer | — | CORE-002 | CORE-003 | — |
| SMOTE | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| STaSy | — | CORE-002, INTERNAL-DATA-001 | CORE-003 | — |
| TabDDPM | CORE-001, TABDDPM-001 | CORE-002, INTERNAL-DATA-001 | CORE-003, TABDDPM-002, UPSTREAM-WORKSPACE-001 | CORE-004 |
| TabDiff | CORE-001 | CORE-002, INTERNAL-DATA-001 | CORE-003, UPSTREAM-WORKSPACE-001 | — |
| TabEBM | CORE-001 | CORE-002 | CORE-003 | — |
| TabSDS | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| TabSyn | CORE-001 | CORE-002, INTERNAL-DATA-001 | CORE-003, UPSTREAM-WORKSPACE-001 | — |
| Tabula | CORE-001 | CORE-002 | CORE-003 | CORE-004 |
| TabularARGN | — | CORE-002 | CORE-003 | — |
| TVAE | CORE-001, CTGAN-FAMILY-001 | CORE-002 | CORE-003 | — |

## 5. Phase 2 remediation order

The recommended order is:

1. Add common dataset-content binding and run/seed output isolation in shared code.
2. Repair the ordinary TabDDPM adapter so seed, device, row count, dataset identity, output ownership, and generated sample path are explicit.
3. Correct independent sampling seed propagation for CTGAN, TVAE, and Goggle.
4. Move or safely redirect TabDiff/TabSyn mutable native workspaces without changing upstream mathematics; discuss any unavoidable upstream patch before implementation.
5. Add strict action-control allowlists and fail-closed device policies.
6. Re-run all Phase 1 tests, then start V2 in the approved fast-to-heavy model batches.

Phase 2 must preserve the existing source locks and native-parity evidence. A fix to a repository adapter boundary does not authorize editing an authoritative implementation.
