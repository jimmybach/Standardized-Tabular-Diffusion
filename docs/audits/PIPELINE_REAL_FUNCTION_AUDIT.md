# Cross-Baseline Pipeline Real-Function Audit

Chinese translation: [PIPELINE_REAL_FUNCTION_AUDIT.zh-CN.md](PIPELINE_REAL_FUNCTION_AUDIT.zh-CN.md)

- Status: approved plan; cross-baseline execution not yet complete
- Plan version: 1.0
- Snapshot date: 2026-08-14
- Primary target: native Windows 11 x86-64, Python 3.11, and the requested CUDA device
- Machine-readable snapshot: [`pipeline-real-function-audit-v1.json`](../../configs/validation/pipeline-real-function-audit-v1.json)

## 1. Purpose

This plan looks for failures that model-specific parity tests can miss at the shared pipeline boundary. It tests whether a registered adapter can receive the user's configuration, execute the authoritative algorithm, preserve run isolation, emit a valid decoded table, and route that table through central evaluation on the primary Windows environment.

This is an operational quality-assurance plan. Its audit states do not replace the registry's validation level, benchmark track, or support level, and they do not admit any model or result to Official Results.

## 2. Current inventory

The runtime registry contains **21 baselines**:

- `arf`, `bn`, `codi`, `ctab-gan`, `ctab-gan-plus`, `ctgan`, `goggle`, `great`, `nflow`, `nrgboost`, `realtabformer`, `smote`, `stasy`, `tabddpm`, `tabdiff`, `tabebm`, `tabsds`, `tabsyn`, `tabula`, `tabularargn`, and `tvae`.
- Twenty have retained Linux/Python 3.11 `native-parity-validated` evidence.
- TabEBM is `smoke-validated`; its full generation path requires externally gated TabPFN-v2 access.
- TabDDPM has a representative native-Windows Adult train/sample run with three generation seeds and finalized P5 bundles.
- TabDiff has a representative native-Windows Adult train/sample run with three generation seeds and finalized central P2/P3 bundles.
- The other 18 baselines have not yet passed this new native-Windows cross-cutting real-function audit. `pending` means untested by this plan, not failed.
- SMOTE remains a classification-only classical reference and is excluded from generative-model ranking, but its adapter still receives the same pipeline contract audit.

## 3. Validation layers and claim boundaries

| Layer | What actually runs | What it can establish | What it cannot establish |
|---|---|---|---|
| V0: inventory/static | Registry, source locks, configuration parsing, and path inspection | The baseline is identifiable and its declared assets are internally consistent | Runtime functionality |
| V1: contract simulation | Repository-owned adapter and pipeline logic with controlled test doubles | Configuration, paths, errors, artifacts, and routing obey the repository contract | That the authoritative model can really train or sample |
| V2: minimal real | The authoritative algorithm performs at least one real fit and sample on a small valid dataset/configuration | Native-Windows smoke functionality for the tested path | Full-dataset quality, native parity, benchmark eligibility, or release support |
| V3: representative real | A reviewed representative dataset/configuration, three declared generation seeds, decoded outputs, and central evaluation | Strong real-function evidence for exactly that model/data/config/environment identity | Automatic Official Results or generalization to other identities |
| V4: admission/release | Frozen protocols plus independent model, dataset, metric, run, environment, legal, and governance decisions | Only the claims explicitly admitted by those decisions | Any undeclared broader claim |

V1 is useful and inexpensive, but it is never called a real run. V2 is the minimum required by this plan before native-Windows runtime functionality is claimed. Native parity is a separate question: it requires exact comparison with the pinned upstream implementation and is not inferred from V2 or V3.

## 4. Cost-controlled execution policy

1. Run the six cross-cutting V0/V1 tasks against all 21 adapters first. These checks are read-only or use isolated temporary outputs.
2. Record every observation in [PIPELINE_FINDINGS.md](PIPELINE_FINDINGS.md) before changing shared code.
3. Group confirmed findings by root cause. Fix a shared pipeline defect once and add regression coverage for every affected adapter class.
4. Run V2 for every adapter that is not externally blocked. Use the smallest valid real dataset and a model-specific fast configuration without replacing the authoritative algorithm. Require one real fit and one real sample; when sampling is separable, run a second declared sampling seed from the same checkpoint. Otherwise run the smallest second seeded execution needed to verify seed propagation.
5. Reserve V3 for release candidates, leaderboard candidates, high-risk adapters, or models whose V2 result reveals platform-sensitive behavior. A V3 failure may also justify expanding another model's coverage.
6. Never select only favorable seeds or discard a failed attempt. Retain the declared attempt chain and distinguish environment remediation from scientific reruns.

This policy avoids full three-seed Adult-scale training for every baseline while retaining one indispensable real-algorithm check per usable adapter.

## 5. Audit tasks

| Task | Primary question | File |
|---|---|---|
| T01 | Do seeds and user configuration reach the authoritative runtime unchanged? | [Seed and configuration](TASK_01_SEED_AND_CONFIG.md) |
| T02 | Are input splits, types, missingness, decoding, and output rows correct? | [Data and output contract](TASK_02_DATA_AND_OUTPUT_CONTRACT.md) |
| T03 | Are checkpoints and generated artifacts isolated, complete, and non-stale? | [Checkpoint and artifact isolation](TASK_03_CHECKPOINT_AND_ARTIFACT_ISOLATION.md) |
| T04 | Does the adapter behave correctly on Windows, CUDA, Python 3.11, and non-ASCII paths? | [Windows, GPU, and dependencies](TASK_04_WINDOWS_GPU_AND_DEPENDENCIES.md) |
| T05 | Does every decoded sample use the same central evaluation route and finalize correctly? | [Central evaluation integration](TASK_05_CENTRAL_EVALUATION_INTEGRATION.md) |
| T06 | Are failures explicit, recoverable, redacted, and safe to retry? | [Failure and recovery](TASK_06_FAILURE_AND_RECOVERY.md) |

## 6. Scheduling and mutation rule

The tasks may inspect different adapters in parallel, but they must not concurrently edit shared runner, configuration, preprocessing, orchestration, or evaluation modules. Inspection produces findings first; fixes are then ordered by dependency and severity. This prevents one task from invalidating another task's observations and avoids conflicting partial fixes.

Suggested order:

1. T01 and T02, because silent scientific errors outrank convenience failures.
2. T03 and T04, because stale artifacts and platform/device drift can invalidate real runs.
3. T05, after a trustworthy decoded sample exists.
4. T06 across every failure boundary discovered by T01-T05.

## 7. Required evidence for each adapter

Each completed adapter row must identify:

- model ID, pinned upstream revision/package, adapter revision, and modification status;
- exact command/configuration, dataset/profile checksum, requested device, and seeds;
- observed Python, Windows, CUDA, GPU, and dependency identities;
- process exit state, elapsed time, checkpoint/output paths, row/column counts, schema result, and file hashes;
- whether outputs from distinct declared seeds are distinct, or why deterministic equality is expected;
- finalized central bundle path and validation result when the task reaches central evaluation;
- every failed attempt, remediation, and regression test; and
- an explicit claim boundary limited to the tested identity.

Large run artifacts remain ignored unless a separate evidence-retention decision approves a compact manifest or result record.

## 8. Completion rule

This audit is complete only when:

- all 21 adapters have V0/V1 results;
- every non-blocked adapter has a passing V2 result on the primary Windows family;
- every external block is explicit and independently actionable;
- every S0/S1 finding is fixed and regression-tested or formally accepted as a release blocker;
- central evaluation rejects malformed output and finalizes valid output without using the legacy evaluator; and
- the machine-readable snapshot, English documents, and Chinese translations agree.

Completion does not change a registry lifecycle status automatically. Any promotion remains a separate reviewed change with its own evidence.
