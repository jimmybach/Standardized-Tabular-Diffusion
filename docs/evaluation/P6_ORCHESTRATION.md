# P6 Resource-Aware Orchestration

## 1. Scope and claim boundary

P6 is the benchmark execution layer. It makes a configured experiment resumable, resource-bounded, content-addressed, and auditable. It does not add a scientific metric, recompute an Atomic Result, aggregate a leaderboard, normalize performance across hardware, or admit any model, dataset, run, or metric to Official Results.

The existing `run` command remains the legacy direct adapter pipeline during the declared P8 migration window. New P6 execution uses the explicit `benchmark` command group.

## 2. Seven-stage plan

`benchmark run` resolves this ordered directed acyclic graph:

1. `prepare`: resolve the experiment configuration, adapter, Dataset Spec, source identities, and readiness evidence;
2. `train`: run the selected adapter training action when requested;
3. `sample`: generate the requested decoded synthetic table when requested;
4. `validate`: verify that the generated artifact is a regular, readable file and record its checksum and row count without repairing it;
5. `evaluate`: run the configured adapter evaluation action when requested;
6. `aggregate`: index completed operational stage artifacts without performing P7 scientific or leaderboard aggregation; and
7. `report`: emit an operational report without recomputing scientific values.

Disabled stages are recorded as `skipped` with `not_requested`. A failed dependency skips its consumers with `dependency_failed`. `aggregate` and `report` may run after an earlier failure so that completed artifacts and the failure record remain inspectable.

Every enabled action runs in a separate process group. The private worker reports action-only time separately from process startup and adapter import overhead. Accelerator timing boundaries are synchronized when a requested CUDA runtime is available.

## 3. Command surface

Install the execution dependencies:

```powershell
python -m pip install ".[orchestration]"
```

Run one configuration:

```powershell
std-tabular-diffusion benchmark run `
  --config configs/my-experiment.json `
  --hardware-profile-id local-windows-rtx5080
```

Optional boundaries and retry policy are explicit:

```powershell
std-tabular-diffusion benchmark run `
  --config configs/my-experiment.json `
  --timeout-seconds 3600 `
  --memory-gib 48 `
  --max-retries 1 `
  --cache-dir D:/std-tabular-cache
```

Inspect and validate the latest invocation:

```powershell
std-tabular-diffusion benchmark status --output-dir artifacts/model/dataset/run-001
std-tabular-diffusion benchmark validate --output-dir artifacts/model/dataset/run-001
std-tabular-diffusion benchmark hardware-profile --profile-id local-windows-rtx5080
```

`--no-cache` forces fresh execution. `--no-resume` fails if the output directory already contains orchestration state; it never silently hides prior attempts.

## 4. State and artifact separation

Execution diagnostics are isolated under `<output_dir>/.orchestration/`:

```text
.orchestration/
  run.json
  profiles/hardware-<fingerprint>.json
  profiles/software-<fingerprint>.json
  invocations/<invocation-id>.json
  attempts/<stage-id>.json
  logs/<stage-id>.jsonl
  worker-results/<stage-id>.json
  cache/
```

Scientific and adapter outputs retain their normal locations. Operational stage summaries are written under `artifacts/execution/`, while `pipeline_result.json` and `benchmark_report.json` are explicitly labelled operational and are not leaderboard sources of truth.

Timestamps, host diagnostics, process identifiers, and resource observations do not enter scientific artifacts. Logs are structured JSON Lines. Common credentials, bearer tokens, GitHub tokens, the user-home prefix, repository root, run root, and cache root are redacted. Experiment configuration fields that look like embedded credentials fail before execution; credentials must be supplied through an external environment-backed mechanism.

## 5. Identity and content-addressed cache

Each stage identity binds:

- stage name and version;
- canonical scientific configuration identity;
- the normalized scientific configuration fingerprint plus checksums of dataset inputs, external synthetic input, upstream configuration, input checkpoint, and source lock when present;
- dependency stage status, identity, and output fingerprint;
- repository commit plus dirty-patch identity;
- installed-package and material environment identity; and
- hardware comparison key.

A cache entry is addressable only by that complete identity. Every declared output is stored as a SHA-256 blob and is checked by digest and byte size before reuse. Missing, malformed, or corrupt entries are marked `invalid`, visibly rejected, and executed afresh. A regular corrupt blob may be repaired only from the newly verified stage output; symlinked or non-regular cache objects fail closed.

A cache hit creates a new stage attempt linked to the source attempt. It records `measurement_mode=cache-reuse`, and `efficiency_eligible=false`; cached wall time is cache-validation cost, not model efficiency. Stages whose complete material outputs cannot be captured under the run root are not cached. This prevents a metadata-only success record from substituting for a missing checkpoint, generated table, or evaluation artifact.

## 6. Resume, retry, and failure semantics

Every invocation and stage attempt has a new identifier. A resumed invocation names its previous invocation; a retried stage names all same-identity ancestor attempts. Attempt files and logs are append-only by identity: a successful retry does not delete or rewrite its failed ancestor.

Resuming the same output directory with a different configuration fingerprint fails closed. `--max-retries` applies only to declared timeout and implementation categories by default. Timeout, process-tree memory exhaustion, dependency/import failure, implementation failure, and user interruption remain distinct structured categories.

An optional stage failure produces a `partial` run when all required stages succeed. Completed outputs remain checksum-bound and available to downstream stages that explicitly allow failed dependencies. Missing and failed observations are never converted to zeros.

## 7. Resource measurements

Fresh-process records contain, when reliable:

- complete process wall time;
- worker-reported action wall time;
- excluded setup time;
- process-tree CPU time;
- peak process-tree resident memory;
- peak accelerator process memory and worker-reported framework peak;
- requested and actual row counts;
- rows per action second;
- warm-up policy; and
- measurement reliability codes.

Process-tree RAM and CPU enforcement uses `psutil`. A requested memory limit fails before launch if that dependency is unavailable. NVIDIA memory sampling uses `nvidia-smi` compute-process observations; a requested CUDA worker also reports framework peak allocation when PyTorch is available. Unsupported measurements remain `null` with a reason instead of being invented.

Dependency installation and dataset download are outside timed stage execution. The full child-process wall time and action-only time remain separate. The current standard adapter plan labels sampling as `cold-single-run-diagnostic`; it does not claim the protocol's future three-repeat warm-generation median.

## 8. Hardware profiles and comparability

Each invocation captures operating system, Python, CPU model, logical and permitted thread counts, RAM, accelerator inventory, driver/CUDA runtime, and material runtime settings. These fields form a SHA-256 hardware comparison key.

Efficiency records with different comparison keys cannot be combined by the provided compatibility guard. No FLOPS conversion or cross-device normalization exists. A user-supplied profile label identifies an observed profile only. Capturing or naming a profile always leaves `official_efficiency_eligible=false`; formal admission remains a separate P7/release decision.

## 9. Optional backends and partial success

The public `StageSpec` and `execute_plan` API supports separately named optional evaluator stages such as `evaluate.metric-a`. Each receives its own process, time/memory boundary, log, outputs, cache identity, and failure state. Downstream operational aggregation may opt into failed dependencies. The CLI's standard seven-stage adapter plan remains one evaluation stage until the P8 migration routes central P2-P5 table evaluation through this engine.

## 10. Validation and current status

P6 provides schema validation for hardware profiles, software profiles, orchestration stage attempts, and run manifests. `benchmark validate` recomputes profile identities and current output checksums, validates dependency references, checks cache-hit evidence, rejects efficiency claims from cache reuse, verifies log identity and secret redaction, and recomputes terminal status from current attempts.

The P6 exit-gate validator and CI exercise:

- forced timeout and process-tree memory exhaustion;
- simulated interruption and dependent-stage cancellation;
- failed-attempt retry with ancestry;
- exact cache hit, changed-identity miss, corrupt-cache rejection, and cache repair;
- optional-backend partial success with preservation of a completed result;
- repeated fixed-workload timing/RAM observations under one named observed profile; and
- rejection of cross-profile efficiency comparison.

These are engineering and operational claims only. A real model run, Official Results admission, leaderboard aggregation, and release support remain independent gates.
