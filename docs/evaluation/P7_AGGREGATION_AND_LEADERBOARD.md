# P7 Aggregation and Leaderboard Publication

Status: implemented engineering baseline. This document specifies the P7 code contract; the broader scientific policy remains [LEADERBOARD_POLICY.md](LEADERBOARD_POLICY.md).

## 1. Scope and claim boundary

P7 turns validated, finalized Run Result bundles into immutable Leaderboard Snapshot bundles. It implements:

- fail-closed input and compatibility validation;
- the declared Atomic Result → run → generation seed → dataset → suite hierarchy;
- deterministic percentile uncertainty with dataset/seed structure;
- coverage, failed-seed, missing-seed, and pairwise-completeness denominators;
- separate Native and Standardized Tuning tracks;
- Official, Partial/Diagnostic, and Community publication classes;
- independent admission and reviewed correction records; and
- JSON, CSV, HTML, and Markdown assets generated from one structured snapshot.

P7 does not make a result Official by itself. It does not combine Fidelity, Utility, Validity, Privacy, and Efficiency into an overall score. It does not issue pairwise superiority claims, normalize results across hardware/software profiles, or infer missing evidence.

## 2. Frozen aggregation contract

One Snapshot Request selects exactly one metric identity, protocol identity, dataset-suite identity, comparison track, publication class, seed denominator, and aggregation procedure.

The P7 v1 hierarchy is:

1. Validate each finalized Run Result bundle and load the requested Atomic Results.
2. Within a run, sum only the metric's declared `aggregate_contribution` values. Positive weights must total exactly one. A required failed state, absent contribution, non-finite value, or incomplete weight makes the run non-computed.
3. Within a dataset, average computed generation-seed scores with equal seed weight. Failed and missing seeds remain visible in the fixed expected-seed denominator; they never improve coverage.
4. Across a suite, macro-average available dataset summaries with equal dataset weight. Row count, predictor count, target count, and seed count cannot give a dataset additional weight.
5. Publish the structured snapshot and derive every display asset from its stored leaderboard rows.

Partial/Diagnostic output may expose an aggregate over available contributions, but its incomplete coverage and absent Official rank remain explicit. Official output requires complete coverage.

## 3. Compatibility boundaries

P7 never silently pools heterogeneous results. Active inputs must agree exactly on the applicable identities:

- protocol ID, version, and checksum;
- metric ID, version, dimension, and direction;
- comparison track;
- evaluator profile;
- hardware profile;
- software/environment checksum;
- Atomic Result schema version; and
- within each dataset, Dataset Profile, dataset view, split, and dataset version.

Native and Standardized Tuning results therefore require different snapshots. Likewise, results produced under different evaluator, hardware, software, preprocessing-view, or split identities cannot be merged merely because their displayed metric names match.

A reviewed invalidation may keep an incompatible bundle in snapshot input history while removing it from the active aggregation set. The invalidated bundle, reason, reviewer, evidence, and replacement identity remain auditable.

## 4. Publication classes

### 4.1 Official Results

An Official Snapshot Request must declare exactly five generation seeds and a frozen tie procedure. Every model/dataset cell must have all five computed runs. In addition, active approved admission records are required for:

- repository release;
- dataset suite;
- model;
- dataset;
- metric;
- protocol;
- comparison track; and
- every accepted Run Result bundle.

The metric registry entry must also be `release-supported`, allow Official Results, and carry an approved release decision. A missing, rejected, withdrawn, ambiguous, or superseded admission fails closed. P7 currently does not promote any existing result merely because this mechanism is implemented.

Official ranks use unrounded scores. Entries satisfying the frozen practical-margin and interval-overlap rule receive one tie group and share the first rank occupied by that group. Point ordering is retained only as `diagnostic_order` for navigation inside a tie.

### 4.2 Partial/Diagnostic Results

This class accepts useful but incomplete evidence. It publishes scores, uncertainty, coverage, failure counts, and diagnostic ordering, but every `rank` is null. It is the default class for engineering checks, pilots, incomplete suites, and results awaiting admission.

### 4.3 Community Results

Community publication has the same no-Official-rank boundary and additionally requires submitter, source repository, and immutable revision provenance in the Snapshot Request. Community results cannot become Official without the independent Official workflow.

## 5. Uncertainty, ties, and pairwise records

The v1 Snapshot Request records the bootstrap method, replicate count, confidence level, and random seed. P7 derives a deterministic salt from scientific identities so repeated reconstruction from the same declared inputs produces byte-identical structured and display assets.

- Dataset intervals bootstrap generation seeds.
- Suite intervals resample datasets and then seeds within sampled datasets.
- Empty computed sets retain the complete interval configuration and report null bounds.

Pairwise records expose the fixed expected dataset-seed cell count, observed paired-cell count, completeness fraction, and mean paired raw difference. `superiority_claim` is deliberately `not-issued`; no visual ordering is evidence of general superiority, and no multiple-comparison procedure is implied.

## 6. Admission and correction records

Admission records are independent review artifacts, not flags embedded by a model run. Their subject identity uses an exact field set for the selected subject type. Each record contains a decision, applicable publication classes, reviewer, UTC review time, evidence references, and any superseded decision IDs.

Evidence references must be safe repository-relative paths or credential-free HTTPS URLs. Absolute paths, parent traversal, Windows paths, non-HTTPS URLs, and embedded credentials are rejected.

Correction records support:

- `invalidate`: remove an affected bundle from active aggregation; or
- `supersede`: replace it with one reviewed bundle for the same model/dataset/seed slot and scientific compatibility group.

Duplicate active attempts are prohibited. P7 never chooses the newest, highest-scoring, or otherwise favorable attempt automatically.

## 7. Snapshot Request

A request is a versioned JSON object validated by `snapshot-request.schema.json`. Important fields are:

- exact repository release, protocol, dataset suite, and metric identities;
- one publication class and comparison track;
- sorted, unique expected generation seeds;
- the frozen v1 run/seed/dataset aggregation rules;
- deterministic hierarchical-percentile bootstrap configuration;
- tie method and equivalence margin;
- exact evaluator/hardware compatibility policy; and
- Community submission provenance or null for non-Community classes.

Unknown fields are rejected. Official requests with fewer or more than five seeds or a disabled tie method are rejected before any bundle is aggregated.

## 8. Command-line workflow

Install the P7 dependencies:

```powershell
python -m pip install ".[leaderboard]"
```

Build a new immutable snapshot. Repeat `--bundle`, `--admission`, and `--correction` as needed:

```powershell
std-tabular-diffusion build-leaderboard `
  --request path/to/snapshot-request.json `
  --bundle path/to/run-seed-1 `
  --bundle path/to/run-seed-2 `
  --admission path/to/admission.json `
  --correction path/to/correction.json `
  --output artifacts/leaderboard-snapshot
```

Validate a published snapshot independently:

```powershell
std-tabular-diffusion validate-leaderboard `
  --snapshot artifacts/leaderboard-snapshot
```

The builder refuses to overwrite a non-empty output directory.

## 9. Immutable snapshot layout

```text
leaderboard-snapshot/
├── manifest.json
├── checksums.sha256
├── leaderboard.json
├── leaderboard.csv
├── leaderboard.html
└── leaderboard.md
```

`leaderboard.json` is the canonical structured publication record. CSV, HTML, and Markdown are deterministic renderings of its stored rows; they do not recalculate scores, uncertainty, coverage, ranks, or ordering.

The snapshot records both:

- `input_fingerprint`, binding the Snapshot Request, input bundle references, admissions, and corrections; and
- `snapshot_fingerprint`, binding the complete scientific snapshot content.

The finalized manifest inventories every publication asset with SHA-256 and byte size. Validation enforces an exact file allowlist, manifest/schema validity, both fingerprints, asset checksums, request/snapshot identity agreement, coverage denominators, model/dataset cell completeness, pairwise record completeness, and byte-for-byte regeneration of every display asset.

## 10. Validation and CI

Run the focused suite:

```powershell
python -m pytest `
  tests/evaluation/test_p7_leaderboard.py `
  tests/evaluation/test_p7_run_bundle_integration.py

python -m standardized_tabular_diffusion.validation.p7_leaderboard `
  --output artifacts/p7-local.json `
  --require-primary-family-environment
```

`.github/workflows/p7-leaderboard-validation.yml` runs the contract, negative, finalized-bundle integration, lint, type, package, and exit-gate checks on hosted Windows/Python 3.11 as the primary family and Linux/Python 3.11 as the secondary compatibility family.

## 11. Current limitations and next boundary

- P7 publishes one fixed metric/protocol/suite/track/class snapshot at a time. A multi-snapshot website and interactive cross-snapshot filtering are P8 publication work.
- Pairwise completeness and raw differences are descriptive only; superiority testing and a reviewed multiple-comparison policy are not enabled.
- P7 can enforce approved admissions but cannot create scientific approvals. Those decisions remain independent review artifacts.
- Historical legacy summaries lacking Atomic Results cannot be promoted through P7. Their read-only migration boundary belongs to P8.
