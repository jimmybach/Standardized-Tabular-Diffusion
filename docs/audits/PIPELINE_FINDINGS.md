# Pipeline Real-Function Findings Ledger

Chinese translation: [PIPELINE_FINDINGS.zh-CN.md](PIPELINE_FINDINGS.zh-CN.md)

- Status: active ledger
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Last synchronized: 2026-08-20

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
| RF-CORE-001 | T01 | S1 | `arf`, `bn`, `ctab-gan`, `ctab-gan-plus`, `ctgan`, `great`, `nflow`, `nrgboost`, `smote`, `tabddpm`, `tabdiff`, `tabebm`, `tabsds`, `tabsyn`, `tabula`, `tvae` | Explicit per-action allowlists now reject unknown top-level controls before execution; stale GReaT presets were migrated to the current interface rather than grandfathered. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); [regression evidence](../evidence/audits/pipeline-phase2-remediation-20260814.json) | fixed |
| RF-CORE-002 | T02 | S1 | all 21 adapters through direct `run`/`run-action` | Shared preflight and public `RunSpec` construction now bind regular-file status, byte size, and SHA-256 and reject content changed after construction. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); content-mutation regression | fixed |
| RF-CORE-003 | T03 | S1 | all 21 adapters through direct `run`/`run-action` | Direct output directories now carry a shared immutable dataset identity plus per-action run identities. Conflicting data, seeds, or configurations are rejected while compatible actions merge into one declared run. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); output-identity regression | fixed |
| RF-CORE-004 | T04 | S2 | `great` training, `tabula` training, `nrgboost`, `smote`, `tabddpm`, `tabsds` | CPU-only adapters reject CUDA; trainer adapters explicitly select and observe the device; TabDDPM binds the device in runtime TOML and rejects unavailable CUDA. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); device regressions | fixed |
| RF-CORE-005 | T02 | S2 | public `DatasetSpec` consumers; observed in `tabula` Windows sampling and `show-dataset` | Nested `Path` values in `DatasetSpec.extra` crossed the public JSON boundary unchanged, so canonical Adult sampling failed before the isolated child process started. Public interface payloads now recursively serialize paths in mappings and sequences. | [TabuLa Windows V2 evidence](../evidence/tabula/windows-v2-real-function-8d72ee8.json); recursive interface serialization regression | verified |
| RF-CORE-006 | T05 | S2 | Pipeline V2 finalization for every model-specific runtime; observed with `tabularargn` | The first TabularARGN model probe passed, but central P3 finalization failed explicitly because the model runtime did not declare `jsonschema`. Model execution and central evaluation now use separate frozen environments, and finalization verifies the central lock before reading the sample. | [Retained failure](../evidence/tabularargn/windows-v2-finalization-dependency-failure-ec53f98.json); [passing clean rerun](../evidence/tabularargn/windows-v2-real-function-6f9e065.json); V2 environment-lock regression | verified |
| RF-CORE-007 | T05 | S2 | Evaluate-only actions for all adapters; observed with `arf` in the independent central runtime | ARF training and both samples passed, but central P3 finalization imported the ARF adapter and failed because the evaluator intentionally does not install model-only `sklearn`. Evaluate-only action, context, and pipeline routes now derive provenance from the registry without importing a model runtime; the independent evaluator then finalized the clean ARF rerun. | [Retained failure](../evidence/arf/windows-v2-finalization-adapter-import-failure-c84a869.json); [passing clean rerun](../evidence/arf/windows-v2-real-function-eb37290.json); no-model-import regressions | verified |
| RF-CORE-008 | T04 | S2 | Clean Pipeline V2 model runtimes; observed with the Windows `nrgboost` verification environment | The first NRGBoost probe stopped before training because its frozen runtime omitted `packaging`, which Pipeline V2 uses to validate requirement identities. The exact dependency is now declared by the NRGBoost extra and a separate Windows V2 lock, leaving the Linux parity lock unchanged; the clean rerun passed model execution and independent central finalization. | [Retained failure](../evidence/nrgboost/windows-v2-probe-dependency-failure-3271298.json); [source-build provenance](../evidence/nrgboost/windows-source-build-provenance-20260820.json); [passing clean rerun](../evidence/nrgboost/windows-v2-real-function-64eec7d.json); lock regression | verified |
| RF-INTERNAL-DATA-001 | T02 | S1 | `codi`, `stasy`, `tabddpm`, `tabdiff`, `tabsyn` | Model-native views are now checksum-bound to the canonical `DatasetSpec` or deterministically materialized beneath run ownership. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); binding regressions | fixed |
| RF-TABDDPM-001 | T01 | S1 | `tabddpm` | A semantically round-tripped runtime TOML now binds training/transformation/sample seeds, device, requested rows, data, and output without editing the source TOML. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); controlled effective-TOML simulation | fixed |
| RF-TABDDPM-002 | T03 | S1 | `tabddpm` | The adapter now validates run-owned checkpoints, decodes a seed-specific canonical table, retains raw arrays, and exposes `generated_sample_path`. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); TabDDPM decode regression | fixed |
| RF-GOGGLE-001 | T01 | S1 | `goggle` | The independent sample seed is passed to the launcher and reapplied to Python, NumPy, and PyTorch immediately before sampling, after the upstream constructor's training-seed reset. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); launcher RNG regression; Windows V2 real-function probe | fixed |
| RF-CTGAN-FAMILY-001 | T01 | S1 | `ctgan`, `tvae` | The loaded official synthesizer now resets its random state from the requested sample seed immediately before generation. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); CTGAN-family random-state regression | fixed |
| RF-UPSTREAM-WORKSPACE-001 | T03 | S1 | `tabddpm`, `tabdiff`, `tabsyn` | All mutable checkpoints and results are redirected below model-specific run-owned runtime directories; authoritative source trees remain unchanged. | [Phase 2 report](PHASE_2_REMEDIATION_REPORT.md); runtime-path regressions | fixed |

The ten Phase 1 findings have completed remediation and V1 regression, and every non-blocked affected adapter has now passed V2. Rows whose declared scope also includes externally blocked TabEBM remain conservatively `fixed`; the block is not treated as a passing probe.

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
