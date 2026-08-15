# Phase 3 Native-Windows V2 Protocol

Chinese translation: [PHASE_3_V2_WINDOWS_PROTOCOL.zh-CN.md](PHASE_3_V2_WINDOWS_PROTOCOL.zh-CN.md)

- Protocol: `pipeline-v2-native-windows-v1`
- Plan: [`pipeline-v2-windows-v1.json`](../../configs/validation/pipeline-v2-windows-v1.json)
- Primary environment: native Windows 11 x86-64 and Python 3.11
- Status: execution protocol implemented; retained model execution evidence is produced only by passing probes

## 1. Purpose and claim boundary

V2 answers one narrow question: can the current adapter execute a real fit and two real generations through the authoritative algorithm on the primary Windows family after the Phase 2 fixes? It does not assess full-data quality, native parity, benchmark eligibility, Official Results admission, or release support.

TabDDPM and TabDiff already possess stronger representative-real Windows evidence. TabEBM remains externally blocked because real generation requires gated TabPFN-v2 access. This phase therefore schedules the other 18 adapters.

## 2. Real-data validation fixture

The protocol deterministically selects 256 training rows and 128 test rows from the reviewed, checksum-bound UCI Adult model view. Selection first covers every categorical value observed in the source split and then performs a seeded fill. CSV and native NumPy views use identical selected row indices. A manifest records the source hashes, selection-index hashes, output hashes, and plan identity.

This is an Adult-derived real-data validation fixture, not a new benchmark dataset and not a substitute for full Adult execution. The bounded fixture exists solely to make one indispensable authoritative execution feasible for every usable adapter.

## 3. Execution design

For each scheduled model, the protocol:

1. requires native Windows, Python 3.11, a clean tracked worktree, a passing `pip check` (or only an exact, plan-declared stale-metadata conflict reviewed below), the declared dependency lock, and the requested CPU/CUDA device;
2. performs one real fit with training seed `13`;
3. byte-copies the resulting training artifacts into two separately claimed sample workspaces;
4. performs real generation at seeds `17` and `29` from identical checkpoint bytes;
5. requires exact row count, canonical column order, no missing/non-finite values, valid categorical domains, integral declared integer columns, distinct seed outputs, and no checkpoint mutation;
6. routes the first valid table through central `p3-validity` evaluation and validates the finalized Result Bundle; and
7. retains the command inputs, configuration and lock hashes, adapter/source identity, environment, elapsed time, artifact manifests, output hashes, evaluation bundle identity, failures, and claim boundary.

CPU-only algorithms remain on CPU. Models locked to the validated PyTorch 2.3 runtime also remain on CPU because that runtime predates RTX 5080 support. CUDA-capable adapters with a compatible validated runtime request the declared RTX 5080. Device choice is part of the retained evidence and is derived from the adapter/runtime contract, not inferred from hardware availability alone.

The frozen CoDi, STaSy, and TabSyn snapshot imports `libzero==0.0.8`. That distribution's stale metadata declares `torch<2`, while the exact newer runtime has already passed the retained native-parity workflow. The plan therefore permits only that fully specified `libzero 0.0.8` versus `torch 2.8.0+cu128` diagnostic. The harness checks the installed versions, requires the waiver to be exercised exactly once, records it in evidence, and rejects every changed, additional, unrecognized, or unused waiver.

GReaT uses the official `distilgpt2` backbone, five epochs over all 256 fixture rows, and official guided sampling at temperature `0.2`. The earlier `sshleifer/tiny-gpt2` mechanical preset trained successfully but produced zero parseable legacy rows; guided sampling still returned missing numerical fields. That small fixture remains useful for adapter mechanics, but it is not treated as a minimally functional Adult generator. The V2 replacement changes only declared model hyperparameters and uses unchanged `be-great==0.0.14` APIs.

## 4. Cost-controlled batches

| Batch | Models |
|---|---|
| Fast CPU | ARF, BN, SMOTE, NRGBoost, TabSDS |
| Legacy/runtime-locked CPU | NFlow, CTAB-GAN, CTAB-GAN+, Goggle, REaLTabFormer |
| Neural GPU | CTGAN, TVAE, CoDi, STaSy, TabSyn |
| LLM GPU | GReaT, TabuLa |
| Heavy GPU | TabularARGN |

A failed attempt is retained and diagnosed; it is never silently deleted or replaced because its result was unfavorable. Environment repair may be retried with explicit ancestry. Any algorithm/configuration change requires a new protocol identity or reviewed plan update.

## 5. Acceptance rule

An adapter becomes `minimal-real-passed` only when its fit, both seed-specific generations, structural checks, immutable-checkpoint check, central evaluation, and Result Bundle validation all pass on the same committed adapter revision. A model-specific V2 result may verify an applicable Phase 2 finding, but it does not automatically promote the registry lifecycle level.

The phase is complete only when all 18 scheduled adapters pass, the two prior representative-real records remain valid, TabEBM's external block remains independently actionable, all 21 identities appear exactly once in the aggregate evidence, and the English/Chinese audit documents agree.
