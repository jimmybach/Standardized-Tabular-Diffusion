# Phase 2 Cross-Baseline Remediation Report

Chinese translation: [PHASE_2_REMEDIATION_REPORT.zh-CN.md](PHASE_2_REMEDIATION_REPORT.zh-CN.md)

- Status: remediation and dependency-light regression complete; V2 real-function execution pending
- Date: 2026-08-14
- Scope: the ten findings retained by the Phase 1 cross-baseline logic audit
- Machine-readable evidence: [`pipeline-phase2-remediation-20260814.json`](../evidence/audits/pipeline-phase2-remediation-20260814.json)

## 1. Outcome

All ten Phase 1 findings now have root-cause code changes and regression coverage. Their ledger state is `fixed`, not `verified`: this phase used static checks and controlled V1 simulations, including simulated official-process outputs, but did not execute a new authoritative V2 train/sample probe for every affected model.

No authoritative model equation, loss, optimizer step, or sampling equation was edited. The TabDDPM, TabDiff, and TabSyn changes are adapter/runtime-boundary changes: runtime configuration, dataset binding, checkpoint placement, and decoded-output ownership. The checksum-locked upstream source trees remain unmodified.

## 2. Remediated findings

| Finding | Resolution | Regression boundary |
|---|---|---|
| `RF-CORE-001` | The 16 affected adapters now use explicit action-control allowlists. Misspelled top-level controls fail before execution. Four stale GReaT presets were migrated instead of allowing ignored legacy controls. | Parameterized rejection across all 16 affected model IDs plus configuration-inventory validation. |
| `RF-CORE-002` | Shared preflight and every public `RunSpec` now record regular-file status, byte size, and SHA-256 for metadata and data splits. Adapters reject content changed after `RunSpec` construction. | Content mutation after construction is rejected. |
| `RF-CORE-003` | Direct runs claim an output directory with a shared immutable dataset identity and per-action identities. Conflicting data, seeds, or configurations are rejected; compatible train/sample/evaluate actions may merge into the same declared run. P6 retains its stronger content-addressed ownership. | Conflicting dataset/sample identity rejection, safe context adoption, compatible action merge, and strict non-empty-directory rejection. |
| `RF-CORE-004` | NRGBoost, SMOTE, and TabSDS explicitly reject CUDA. GReaT and Tabula pass an explicit `use_cpu` trainer control and record/validate the observed trained-model device. TabDDPM writes the requested device into its runtime TOML and rejects unavailable CUDA. | CPU-only rejection, device-resolution tests, full adapter suite. |
| `RF-INTERNAL-DATA-001` | CoDi, STaSy, and TabSyn checksum-bind their model-native views to the embedded canonical dataset. TabDiff materializes a byte-verified run-owned native view. TabDDPM creates a run-owned input view and records its one-row, non-fit validation compatibility mirror. | Native/canonical byte binding, late-mutation rejection, TabDDPM controlled execution. |
| `RF-TABDDPM-001` | TabDDPM now derives a semantically round-tripped runtime TOML that binds training seed, transformation seed, sampling seed, requested rows, device, canonical data, and run-owned output. | Controlled train/sample simulation inspects both effective TOMLs. |
| `RF-TABDDPM-002` | TabDDPM validates run-owned checkpoints, decodes raw arrays into canonical column order, retains seed-specific raw arrays, and exposes `generated_sample_path`. | Two-row classification decode with integer restoration and target-label mapping. |
| `RF-GOGGLE-001` | Goggle passes the sample seed to its compatibility launcher and reapplies it immediately before the official sampler, after the upstream constructor resets PyTorch to the training seed. | Adapter command assertion, launcher RNG regression, and distinct-output Windows V2 probe. |
| `RF-CTGAN-FAMILY-001` | CTGAN and TVAE reset the loaded official synthesizer with the requested sample seed before calling `sample`. | Loaded-model random-state assertion. |
| `RF-UPSTREAM-WORKSPACE-001` | TabDDPM uses `output_dir/tabddpm-runtime`, TabDiff uses `output_dir/tabdiff-runtime`, and TabSyn uses `output_dir/tabsyn-runtime`. Runtime overlays redirect official path derivation without changing upstream source bytes. | Run-owned path assertions and controlled no-upstream-checkpoint check. |

## 3. Safety and compatibility decisions

- Dataset symlinks, non-regular inputs, unsafe checkpoints, and symlinked output/runtime roots fail closed.
- TabDDPM runtime TOML serialization is dependency-free and is parsed back with `tomllib`; execution stops if the semantic round trip differs.
- TabDDPM sampling must reuse the recorded run-owned checkpoint inventory and the trained scientific configuration. Only sampling seed, requested row count, device, and path bindings may vary at the sampling boundary.
- TabDiff and TabSyn keep their official imports in the pinned source checkout while redirecting only mutable runtime paths.
- Direct low-level `adapter.train(RunSpec(...))` test fixtures without an embedded public dataset identity retain a diagnostic compatibility path. Public `train_from_config`, `sample_from_config`, `run-action`, and `run` always embed and enforce the content identity.

## 4. Verification boundary and next step

The full dependency-light repository suite passes after remediation. This establishes the V0/V1 contract behavior of the bound repository snapshot; it does not establish new native-Windows real model functionality, output quality, native parity, benchmark eligibility, Official Results admission, or release support.

The next phase is V2: run the smallest real authoritative train/sample probe for each non-blocked adapter in the approved fast-to-heavy batches. A finding moves from `fixed` to `verified` only after its applicable V2 probe passes.
