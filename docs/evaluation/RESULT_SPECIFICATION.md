# Result Specification

Chinese translation: [RESULT_SPECIFICATION.zh-CN.md](RESULT_SPECIFICATION.zh-CN.md)

- Status: design baseline
- Result schema version: 0.1.0
- Last updated: 2026-08-03

## 1. Purpose

This specification defines the durable, machine-readable representation of benchmark execution, atomic metrics, aggregates, provenance, failures, and publication artifacts.

Its goals are:

- complete scientific traceability;
- explicit failure semantics;
- safe resume and cache reuse;
- independent validation;
- static leaderboard generation; and
- long-term compatibility without silent history rewriting.

## 2. Result hierarchy

Results exist at four levels.

### 2.1 Atomic Result

One metric observation for one declared scope, such as a column, column pair, target, evaluator, attack, timing phase, group, or seed.

### 2.2 Run Result

One model, one dataset and view, one split, one comparison track, one resolved configuration, and one generation seed.

A Run Result contains Atomic Results and phase evidence. It remains valid as a failure bundle when training, sampling, validation, or evaluation fails.

### 2.3 Dataset Summary

A versioned aggregate over all required Run Results for one model-dataset combination, normally five generation seeds. It references immutable Run Results rather than copying or replacing them.

### 2.4 Leaderboard Summary

A versioned aggregate over a declared dataset suite. It references Dataset Summaries and contains the compatibility, coverage, uncertainty, and rank evidence required by the leaderboard snapshot.

Aggregation at a higher level MUST be reproducible from referenced lower-level results.

## 3. Identity model

### 3.1 Run identity

Every Run Result MUST include:

- run_id: immutable opaque identifier;
- run_fingerprint: deterministic hash of compatibility-defining inputs;
- repository_commit;
- protocol_version;
- result_schema_version;
- metric_registry_version;
- dataset identity;
- split identity;
- model and adapter identity;
- comparison track;
- resolved configuration hash;
- generation seed;
- evaluator profile;
- environment profile; and
- hardware profile.

run_id distinguishes execution attempts. run_fingerprint identifies executions that claim the same scientific inputs. Multiple run IDs MAY share a fingerprint, but duplicate attempts MUST NOT be silently averaged unless the protocol explicitly selects them.

### 3.2 Deterministic fingerprint

The fingerprint input MUST use a documented canonical serialization and include every field that can change scientific meaning. It MUST exclude volatile fields such as wall-clock start time, host name, and log path.

The exact fingerprint algorithm and canonicalization are schema-versioned.

### 3.3 Identifiers

Stable identifiers SHOULD be lowercase, ASCII, and safe in portable paths. Display names and localized labels are stored separately.

Paths inside a bundle MUST be relative POSIX-style paths. Absolute developer paths, parent traversal, and environment-specific home paths are prohibited.

## 4. Bundle types

bundle_type is one of:

- run;
- dataset_summary; or
- leaderboard_snapshot.

Every bundle is immutable after finalization. A correction creates a new bundle with a new identifier and a supersedes reference.

## 5. Run bundle layout

A finalized Run Result uses:

~~~text
result_bundle/
├── manifest.json
├── metadata.json
├── config.yaml
├── environment.json
├── metrics.parquet
├── summary.json
├── checksums.sha256
├── logs/
│   ├── events.jsonl
│   ├── stdout.log
│   └── stderr.log
├── stages/
│   ├── prepare.json
│   ├── train.json
│   ├── sample.json
│   ├── validate.json
│   ├── evaluate.json
│   ├── aggregate.json
│   └── report.json
└── artifacts/
    └── index.json
~~~

An optional file is omitted only when the manifest states that it is not applicable. A failed run still writes metadata, config, environment, stage records, logs, checksums, and all Atomic Results produced before failure.

## 6. Serialization rules

- JSON and JSON Lines use UTF-8.
- YAML is loaded through a safe loader and MUST NOT contain executable tags.
- Timestamps use UTC RFC 3339 format.
- Numeric metric columns use 64-bit floating point.
- Serialized JSON MUST NOT contain NaN, positive infinity, or negative infinity.
- Non-computed numeric values are null.
- Enum values use the exact lowercase identifiers defined by the schema.
- File paths are bundle-relative and use forward slashes.
- Human-readable messages do not replace stable reason codes.
- Structured files declare their schema version.

Canonical JSON used for hashes has sorted object keys, a defined Unicode normalization form, and no insignificant whitespace.

## 7. manifest.json

The manifest is the bundle index. It MUST include:

- bundle_id;
- bundle_type;
- bundle_schema_version;
- created_at;
- finalized_at;
- finalization_status;
- run or aggregate identity;
- required and optional file inventory;
- file media types;
- referenced external artifacts;
- supersedes and invalidates relationships;
- producer repository commit; and
- checksum algorithm.

finalization_status is one of:

- incomplete;
- finalized;
- invalidated; or
- withdrawn.

An incomplete bundle MUST NOT be admitted to a leaderboard.

## 8. metadata.json

metadata.json records scientific identity and execution outcome.

Required sections are:

- identity;
- protocol;
- dataset;
- model;
- implementation;
- comparison_track;
- seeds;
- evaluator;
- execution;
- coverage;
- provenance;
- review; and
- status.

### 8.1 Dataset identity

Dataset metadata includes:

- dataset_id;
- dataset_version;
- dataset_view;
- dataset_profile_version;
- split_id;
- raw, canonical, split, and preprocessing checksums;
- row and column counts by partition; and
- feature, target, identifier, sensitive, and quasi-identifier roles.

### 8.2 Model identity

Model metadata includes:

- model_id;
- model_family;
- upstream repository or package;
- immutable revision or exact version;
- reproduction target;
- modification status;
- patch identifiers;
- adapter identifier and version;
- validation level;
- eligibility track; and
- support level.

### 8.3 Execution outcome

Execution metadata includes:

- requested action;
- start and end timestamps;
- terminal phase;
- run status;
- requested and actual synthetic row counts;
- timeout and resource limits;
- interruption and resume ancestry;
- warnings;
- failure category and reason code; and
- artifact references.

run status is one of:

- success;
- partial;
- failed;
- cancelled; or
- invalidated.

## 9. config.yaml

config.yaml is the fully resolved configuration used for execution, not merely the user input.

It MUST include:

- protocol and comparison track;
- model and dataset selections;
- all model parameters after defaults and overrides;
- preprocessing and postprocessing configuration;
- requested sample count;
- generation and evaluator seeds;
- metric selections and versions;
- tuning profile where applicable;
- resource limits;
- hardware request;
- output and retention policy; and
- configuration source and precedence evidence.

Unknown configuration fields MUST fail validation unless the schema explicitly permits extensions.

Secrets MUST NOT be embedded. A redacted secret reference MAY record that an external credential was required, but not its value.

## 10. environment.json

environment.json records the execution environment:

- operating system, distribution, kernel, and architecture;
- Python implementation and version;
- installed dependency lock identifier and package inventory;
- accelerator frameworks;
- CPU model, logical and permitted thread counts, and RAM;
- GPU model, device count, VRAM, driver, CUDA, and relevant libraries;
- locale and timezone;
- deterministic settings;
- container or environment image identity;
- environment variables that materially affect computation, with secrets removed;
- repository dirty-state indicator; and
- official hardware-profile identifier.

An official result SHOULD be produced from a clean repository state. If a controlled exception permits a dirty state, the exact patch hash and review evidence are mandatory.

## 11. metrics.parquet

metrics.parquet is the canonical table of Atomic Results. One row represents one scalar observation for one scope. Large distributions or matrices are stored as separate content-addressed artifacts, with summary scalars and references in the table.

### 11.1 Required columns

| Column | Type | Description |
|---|---|---|
| result_schema_version | string | Atomic-result schema version |
| run_id | string | Parent Run Result |
| protocol_version | string | Evaluation protocol |
| dataset_id | string | Stable dataset identifier |
| dataset_version | string | Dataset version |
| dataset_view | string | Versioned view identifier |
| split_id | string | Official split |
| model_id | string | Stable model identifier |
| comparison_track | string | native or standardized-tuning |
| generation_seed | int64 | Generator seed |
| metric_id | string | Metric Registry identifier |
| metric_version | string | Exact metric version |
| dimension | string | Evaluation dimension |
| scope_type | string | column, pair, target, evaluator, attack, group, phase, or dataset |
| scope_id | string | Stable identifier within scope type |
| evaluator_id | string | Nullable evaluator identifier |
| evaluator_version | string | Nullable evaluator version |
| task_type | string | Nullable classification or regression task |
| state | string | Structured metric state |
| raw_value | float64 | Nullable raw value |
| normalized_value | float64 | Nullable normalized value |
| aggregate_contribution | float64 | Nullable predeclared contribution |
| reference_value | float64 | Nullable baseline or target |
| unit | string | Nullable unit |
| raw_direction | string | maximize, minimize, target, distributional, or descriptive |
| weight | float64 | Predeclared aggregation weight |
| n_reference | int64 | Reference sample count |
| n_synthetic | int64 | Synthetic sample count |
| n_valid | int64 | Valid observations |
| n_excluded | int64 | Excluded observations |
| reason_code | string | Nullable stable reason |
| reason_detail | string | Nullable diagnostic detail |
| warning_codes | list[string] | Structured warnings |
| artifact_ref | string | Nullable bundle-relative artifact |
| computed_at | timestamp | UTC completion time |

Schema evolution MAY add nullable columns under a compatible minor schema update. Removing or changing meaning requires a new incompatible schema version.

### 11.2 Value invariants

When state is computed:

- required raw or structured output MUST be present;
- every scalar MUST be finite;
- counts MUST be non-negative;
- weight MUST match the metric contract; and
- direction and unit MUST match the Metric Registry.

When state is not computed:

- raw_value, normalized_value, and aggregate_contribution MUST be null;
- reason_code MUST be present; and
- denominator counts MUST still be reported when knowable.

## 12. summary.json

summary.json is a reproducible view, not the source of truth.

It contains:

- run identity and terminal status;
- structural and content validity summary;
- dimension component scores;
- Local and Global Utility summaries;
- privacy-risk diagnostics;
- efficiency measurements;
- metric-state and denominator counts;
- warnings and failure summaries;
- links to atomic rows or artifacts;
- aggregation implementation and version; and
- a declaration of whether the run is eligible for dataset aggregation.

Every summary value MUST be reproducible from metrics.parquet, stage evidence, and the versioned aggregation contract. Manual edits are prohibited.

## 13. Stage records

Each stage record includes:

- stage name and version;
- status;
- dependency stage identifiers;
- input fingerprints;
- resolved action;
- start and end timestamps;
- elapsed time;
- process exit code where applicable;
- log references;
- output inventory and checksums;
- warnings;
- failure category and reason;
- cache decision;
- retry count; and
- resume ancestry.

Stage status is one of:

- pending;
- running;
- succeeded;
- failed;
- skipped;
- cancelled; or
- invalidated.

A skipped stage requires a stable reason. It MUST NOT be treated as succeeded without compatible cached evidence.

## 14. Logs

events.jsonl is the canonical structured event log. stdout.log and stderr.log preserve external-process streams where applicable.

Logs MUST:

- use timestamps and severity;
- identify stage and component;
- preserve upstream exit status and working-directory context without exposing unsafe absolute paths;
- redact credentials and tokens;
- avoid embedding complete restricted datasets or synthetic rows;
- record truncation; and
- remain useful after relocation of the bundle.

Human-readable logs do not replace structured failure metadata.

## 15. Artifacts

artifacts/index.json records every material artifact:

- artifact_id;
- role;
- media type;
- relative path or approved external URI;
- byte size;
- checksum;
- producer stage;
- retention class;
- publication class;
- license or data-rights classification; and
- encryption or access requirements where applicable.

Large checkpoints, real data, synthetic row-level data, and attack traces are not included by default. An external artifact reference MUST be immutable or content-addressed and MUST state access expectations.

Unsafe object deserialization formats MUST NOT be required merely to inspect result identity, metrics, or summary.

## 16. Checksums and finalization

checksums.sha256 lists every finalized regular file except itself, using bundle-relative paths and a documented deterministic order.

Finalization requires:

1. all required files exist;
2. schemas validate;
3. cross-file identities agree;
4. every referenced local artifact exists;
5. the immutable bundle identifier, finalized timestamp, and finalized status are written to the manifest;
6. checksums are generated for every finalized regular file except the checksum file itself;
7. checksums verify; and
8. no finalized file changes after hashing.

If finalization fails, the bundle remains incomplete and is ineligible for publication.

## 17. Cache and resume

A stage output MAY be reused only when all declared input fingerprints, code and dependency identities, configuration, dataset, seeds, and upstream identities match.

Cache entries MUST record:

- producer run and stage;
- complete compatibility key;
- artifact checksums;
- creation time;
- validation time;
- retention policy; and
- invalidation reason when revoked.

Resume creates a new execution attempt linked to its ancestor. It MUST NOT rewrite the ancestor's logs or metadata.

## 18. Dataset Summary bundle

A Dataset Summary references all required Run Result bundle IDs and fingerprints. It contains:

- model-dataset identity;
- expected and observed seeds;
- accepted and rejected runs with reasons;
- per-seed values;
- dataset-level aggregates;
- uncertainty;
- state and coverage counts;
- compatibility validation;
- aggregation version; and
- official-admission status.

Duplicate attempts with the same fingerprint require an explicit selection rule. Failed attempts remain referenced even if a valid retry is selected.

## 19. Leaderboard Snapshot bundle

A Leaderboard Snapshot references Dataset Summaries and records:

- dataset-suite version;
- eligible model set;
- protocol and comparison track;
- metric and result-schema versions;
- hardware profile for efficiency views;
- aggregation and bootstrap configuration;
- coverage gates;
- ranks and tie groups;
- invalidated or excluded result references;
- publication assets; and
- reviewer approval.

The structured snapshot is the source for static HTML, Markdown, CSV, JSON, and Parquet publication views.

## 20. Validation and compatibility

The project MUST provide a result-bundle validator that checks:

- file and schema completeness;
- enum and type correctness;
- finite-value rules;
- checksums;
- cross-file identity;
- Metric Registry compatibility;
- Dataset Profile compatibility;
- protocol compatibility;
- aggregation recomputation;
- prohibited absolute paths;
- secret patterns;
- unsafe artifact types;
- coverage gates; and
- publication classification.

Validation output itself is versioned evidence and is linked from admission review.

## 21. Retention, publication, and privacy

Retention classes include:

- permanent evidence;
- release evidence;
- reproducible cache;
- temporary diagnostic; and
- restricted.

Publication defaults to metadata, aggregate metrics, configuration, environment, checksums, and approved logs. Real data, row-level synthetic data, sensitive attack outputs, and large model artifacts require separate rights and privacy review.

Deletion or expiration of an external artifact MUST NOT make published numerical provenance misleading. The bundle records whether reproduction requires an artifact that is no longer retained.

## 22. Schema evolution

Compatible additions use a new minor schema version. Incompatible field meaning, required-field removal, identity changes, or serialization changes use a new major schema version.

Migration tools MUST:

- preserve the original bundle;
- record source and target schema versions;
- be deterministic;
- report lossy fields;
- write a new bundle;
- recompute checksums; and
- never invent missing scientific evidence.

## 23. Minimal metadata example

~~~json
{
  "identity": {
    "run_id": "run-example",
    "run_fingerprint": "sha256:pending",
    "result_schema_version": "0.1.0"
  },
  "protocol": {
    "protocol_version": "benchmark-v1",
    "metric_registry_version": "pending"
  },
  "dataset": {
    "dataset_id": "example",
    "dataset_version": "1",
    "dataset_view": "canonical",
    "split_id": "official-v1"
  },
  "model": {
    "model_id": "example-model",
    "adapter_version": "pending"
  },
  "comparison_track": "native",
  "seeds": {
    "generation_seed": 0
  },
  "status": {
    "run_status": "partial",
    "terminal_phase": "evaluate"
  }
}
~~~

This example is illustrative and not a complete valid bundle.

## 24. Related specifications

- [Evaluation Protocol](EVALUATION_PROTOCOL.md)
- [Leaderboard Policy](LEADERBOARD_POLICY.md)
- [Metric Governance](METRIC_GOVERNANCE.md)
- [Dataset Profile Specification](DATASET_PROFILE_SPEC.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [Repository Quality Standard](../QUALITY_STANDARD.md)
