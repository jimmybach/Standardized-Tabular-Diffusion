# Evaluation and Leaderboard Implementation Roadmap

Chinese translation: [IMPLEMENTATION_ROADMAP.zh-CN.md](IMPLEMENTATION_ROADMAP.zh-CN.md)

- Status: implementation baseline
- Roadmap version: 0.1.0
- Last updated: 2026-08-03
- Primary release environment: Linux and Python 3.11

## 1. Purpose

This roadmap converts the approved evaluation specifications into an executable engineering sequence for this repository. It is deliberately tied to the current codebase: it records what exists, what may be reused, what must be replaced, how compatibility is preserved, and what evidence is required before a metric or leaderboard is described as official.

This document does not itself admit a metric, model, dataset, result, or release. The normative semantics remain in the linked specifications. A phase is complete only when its exit gate and evidence requirements pass.

## 2. Locked implementation principles

The implementation MUST follow these decisions:

1. Evaluation is a standalone subsystem. The same engine evaluates adapter-produced samples and externally supplied synthetic tables.
2. Model adapters own train, sample, and decoding behavior. They do not own metric formulas, result aggregation, or leaderboard eligibility.
3. Source-defined metrics wrap an immutable official package or pinned official source whenever possible. A local reimplementation never silently substitutes for a source-parity metric.
4. Upstream model and metric source trees are read-only evidence or runtimes. Any unavoidable source edit requires prior discussion, an isolated patch, provenance, licensing review, and parity validation.
5. Machine-readable Dataset Profiles, Metric Registry records, protocol profiles, and result schemas are required. Prose defaults are not sufficient.
6. JSON Schema files are the canonical wire-contract validators. Internal Python models may change without changing the wire contract, but checked-in schemas and versioning rules may not be bypassed.
7. Metric calls return structured states. Exceptions, `NaN`, infinity, omitted targets, and silently reduced denominators are not valid result representations.
8. Raw source values, benchmark-derived transformations, and aggregate contributions are separate fields.
9. Missing values never reach a generative model under the initial official protocol. A dataset with raw missingness must invoke an explicit, versioned, train-fitted preprocessing module; registration must not silently drop or impute rows.
10. No overall leaderboard score is introduced in the initial release. Fidelity, Local Utility, Global Utility, Validity, Privacy Risk, and Efficiency remain visible dimensions or sub-leaderboards.
11. Only metrics that are both `protocol-frozen` and `release-supported` may affect Official Results.
12. The first end-to-end implementation slice is structural validation plus source-parity Column Shapes and Column Pair Trends. Breadth is added only after this slice produces a valid finalized result bundle.

## 3. Current implementation audit

### 3.1 Current execution path

The current path is:

~~~text
ExperimentConfig
    -> runner.run_action / runner.run_pipeline
    -> model_adapter.evaluate
    -> evaluation.tabstruct.normalize_*
    -> standardized_summary.json
    -> comparison.compare_summaries
~~~

This path can produce a compact comparison table, but it does not implement the approved prepare/train/sample/validate/evaluate/aggregate/report lifecycle, metric registry, Dataset Profile contract, Atomic Results, compatibility groups, finalized bundles, or leaderboard admission checks.

### 3.2 Code disposition map

| Current location | Current role | Disposition |
|---|---|---|
| [`evaluation/tabstruct.py`](../../standardized_tabular_diffusion/evaluation/tabstruct.py) | Static metric descriptions, upstream metric invocation, structural utility, exception handling, and old summary writing in one module | Freeze as legacy behavior, split responsibilities, and remove it from the new official path after parity-preserving migration |
| [`models/base.py`](../../standardized_tabular_diffusion/models/base.py) | Adapter interface and artifact manifest writing | Retain train/sample adapter boundary; route evaluation through the standalone engine and replace ad hoc artifact manifests with bundle references |
| [`models/tabdiff.py`](../../standardized_tabular_diffusion/models/tabdiff.py), [`models/tabsyn.py`](../../standardized_tabular_diffusion/models/tabsyn.py), and [`models/sample_baselines.py`](../../standardized_tabular_diffusion/models/sample_baselines.py) | Call the same sample-file normalizer | Reuse sample-path discovery; replace direct metric calls with an `EvaluationRequest` |
| [`models/tabddpm.py`](../../standardized_tabular_diffusion/models/tabddpm.py) | Normalizes pre-existing upstream JSON metric files | Keep a clearly labeled legacy-import path; official evaluation must use canonical decoded samples and the common engine |
| [`config.py`](../../standardized_tabular_diffusion/config.py) | Dataclasses with evaluation Boolean flags and an untyped `extra` map | Preserve a compatibility loader; introduce versioned protocol, metric-profile, dataset-profile, seed, failure-policy, and hardware-profile identities |
| [`interfaces.py`](../../standardized_tabular_diffusion/interfaces.py) | Minimal `DatasetSpec`, `RunSpec`, and `ArtifactBundle` | Retain temporary adapter compatibility types; add strict evaluation contracts instead of expanding `extra` |
| [`datasets.py`](../../standardized_tabular_diffusion/datasets.py) | Discovers dataset metadata from upstream `info.json` files | Use only as a legacy importer into reviewed Dataset Profiles; do not treat upstream metadata as official eligibility evidence |
| [`dataset_onboarding.py`](../../standardized_tabular_diffusion/dataset_onboarding.py) | Registers CSV data and silently sanitizes missing values | Separate registration, profiling, preprocessing, and materialization; deprecate silent row deletion and implicit missing-token insertion |
| [`materialization.py`](../../standardized_tabular_diffusion/materialization.py) | Runs upstream processing, copies data, and writes path-based manifests | Add source/view/split identities, checksums, preprocessing lineage, and atomic publication; avoid copied directories as identity |
| [`runner.py`](../../standardized_tabular_diffusion/runner.py) | Runs train, sample, and evaluate phases | Retain as a high-level model workflow; delegate evaluation lifecycle, resume, and result finalization to the evaluation orchestrator |
| [`comparison.py`](../../standardized_tabular_diffusion/comparison.py) | Flattens old summaries into rows | Replace official use with schema validation, compatibility grouping, coverage accounting, uncertainty, and policy-aware aggregation |
| [`cli.py`](../../standardized_tabular_diffusion/cli.py) | Exposes metadata, three run actions, and old summary comparison | Add profile/registry/result validation and standalone evaluation commands; keep deprecated commands through the migration window |
| [`requirements-benchmark-stack.txt`](../../requirements-benchmark-stack.txt) | One broad environment for models and metrics | Split core, evaluation, model, and development dependency groups; lock the official Linux/Python 3.11 evaluation environment |
| [`tests/`](../../tests) | 51 repository test functions, mainly adapter, CLI, onboarding, and byte-stability checks | Preserve useful adapter tests; add an evaluation-specific test pyramid and stop treating mocked or byte-stable old summaries as scientific validation |

### 3.3 Confirmed gaps and hazards

- `METRIC_DEFINITIONS` is a Python constant, not a validated registry, and it uses the old `tabstruct-aligned-v1` identity.
- Per-metric `EvaluationConfig` flags are not carried as a complete, enforceable metric request into the normalizers.
- Broad exception handling can convert implementation defects into missing values without a stable state or reason code.
- Structural Utility may silently omit undefined targets from its mean, and constant synthetic targets currently receive behavior that is incompatible with the approved failure policy.
- Predictor composition can change with installed packages or an environment variable, so current structural-utility outputs do not have one stable metric identity.
- The old summary schema lacks atomic per-column, per-pair, per-target, and per-seed records, checksums, compatibility identity, coverage, and finalization state.
- Dataset registration currently changes data while registering it. This prevents a clean audit of raw input, preprocessing, and resulting dataset view.
- Package import eagerly loads model modules and optional heavy dependencies such as PyTorch, preventing core metadata and data tests from running in a minimal environment.
- No root `pyproject.toml`, root pytest discovery boundary, or repository CI workflow is present.
- On the 2026-08-03 audit machine, unrestricted `pytest -q` collected upstream and research-reference test suites and stopped with 105 collection errors. Even a repository-only onboarding test could not collect because PyTorch was imported eagerly. This is baseline evidence, not an accepted test state.

## 4. Target architecture

The exact filenames may change through review, but responsibility boundaries must remain stable:

~~~text
standardized_tabular_diffusion/evaluation/
  contracts.py          # requests, contexts, atomic results, enums
  registry.py           # metric definitions and lifecycle records
  profiles.py           # protocol and evaluator-profile loading
  engine.py             # dependency-aware metric execution
  validation.py         # table and contract validation
  bundle.py             # atomic writes, checksums, finalization, resume
  aggregation.py        # compatible run/dataset aggregation only
  metrics/
    fidelity/
    utility/
    validity/
    privacy/
    efficiency/
  backends/             # isolated wrappers for SDMetrics, TabStruct, etc.

schemas/evaluation/     # checked-in JSON Schemas
configs/evaluation/
  protocols/
  metrics/
  evaluators/
  hardware/
configs/datasets/       # reviewed Dataset Profiles
tests/evaluation/
  fixtures/             # small synthetic, redistributable tables
  golden/               # versioned authoritative expected outputs
~~~

The dependency direction is one-way: adapters and CLI may call public evaluation APIs; metric modules may depend on contracts and isolated backends; contracts, registry validation, and bundle validation must not import model adapters or heavy optional metric packages.

### 4.1 Public evaluation request

An evaluation request must identify at least:

- decoded synthetic table or immutable sample artifact;
- Dataset Profile identifier and checksum;
- protocol profile and version;
- requested metric identifiers and versions;
- evaluator and hardware profiles where applicable;
- model/run provenance when available;
- subject type: adapter run or external synthetic table;
- seed set;
- output bundle location; and
- resource and failure policy.

The engine resolves all identities before computation. Unknown fields, missing required identities, incompatible metric/profile combinations, and unreviewed official claims fail validation before expensive work begins.

### 4.2 Execution graph

Metric dependencies are explicit. For example, structural validation precedes every official metric; reusable encoded views may feed several metrics; raw TRTR/TSTR/Dummy results precede Local Utility retention; per-target ratios precede Global Utility; atomic records precede aggregation; and only a validated incomplete bundle may be finalized.

A node records its content-addressed inputs, outputs, implementation version, seed, state, elapsed time, and resource observations. Resume reuses a node only when all identity inputs match.

## 5. Delivery sequence and dependency gates

| Phase | Deliverable | Depends on | Exit gate |
|---|---|---|---|
| P0 | Trustworthy development baseline | None | Core tests collect in a minimal environment; repository and reference tests are isolated |
| P1 | Contracts, registries, profiles, and incomplete bundle writer | P0 | Invalid contracts fail deterministically; round-trip and schema tests pass |
| P2 | First vertical slice: external table -> structural gate -> Shape/Trend -> finalized bundle | P1 | Direct pinned-source parity and bundle validation pass on Linux/Python 3.11 |
| P3 | Full Validity subsystem and explicit preprocessing boundary | P2 | No hidden repair or missing-value mutation; rule and failure tests pass |
| P4 | Local and Global Utility | P1, P3 | Raw arms, state semantics, profile identity, and source/formula validation pass |
| P5 | High-order fidelity and empirical privacy work packages | P2, P3 | Only resolved and approved metrics advance; blocked metrics remain excluded |
| P6 | Resource-aware orchestration, efficiency, cache, and resume | P2 | Phase accounting and reuse integrity pass under declared hardware profiles |
| P7 | Dataset aggregation, uncertainty, compatibility groups, and leaderboard snapshots | P2-P6 as applicable | Incompatible results cannot be merged; coverage and publication gates pass |
| P8 | Legacy migration, documentation, packaging, CI, and release evidence | P0-P7 | Public-preview or official-release gate passes for the claimed release class |

P3, P4, P5, and parts of P6 may proceed concurrently after their dependencies are stable. P7 must not be used to publish rankings before each contributing metric and dataset independently passes its admission gate.

## 6. Phase work packages

### 6.1 P0 — trustworthy baseline

Tasks:

- Add root packaging metadata for Python 3.11, explicit core/evaluation/model/development extras, and a console entry point.
- Configure pytest to collect only repository-owned tests by default. Upstream and `research_inputs/` tests run only through explicit provenance/parity jobs.
- Make top-level imports lightweight; move optional model and metric imports behind factories with actionable missing-extra messages.
- Establish formatting, linting, static typing, schema validation, unit-test, and documentation-link commands.
- Record the current 51 repository tests as a migration baseline; classify each as unit, integration, smoke, or legacy-regression.
- Add a Linux/Python 3.11 CI job that installs only core dependencies and runs metadata, schema, CLI-help, and core tests.
- Treat `research_inputs/` as immutable review input and exclude it from packaging, ordinary test discovery, and runtime import paths.

Exit evidence:

- a clean checkout can install the core package;
- importing contracts, dataset metadata, and CLI help does not require PyTorch, AutoGluon, SDMetrics, or a model runtime;
- default pytest discovery contains only declared repository tests; and
- the previous collection failures are represented by regression tests or CI configuration checks.

### 6.2 P1 — contracts and identity foundation

Tasks:

- Implement versioned schemas for Dataset Profile, Metric Registry entry, protocol profile, Evaluation Request, Atomic Result, stage record, manifest, metadata, summary, and artifact index.
- Implement the six metric result states and stable reason-code validation.
- Implement finite-value, raw/derived separation, direction, support-count, and aggregation-effect invariants.
- Create a data-driven Metric Registry loader; move the old static descriptions into explicitly legacy records.
- Create a Dataset Profile loader and a non-official importer for existing upstream `info.json` metadata.
- Define protocol resolution, immutable identity hashing, canonical JSON, safe YAML loading, and bundle-relative path validation.
- Implement an incomplete Run Result bundle writer with atomic file replacement, event logs, and deterministic content fingerprints.
- Add CLI commands to list and validate metric records, Dataset Profiles, protocol profiles, and result bundles.

Exit evidence:

- schema positive, negative, unknown-field, version, path-traversal, non-finite-value, and round-trip tests pass;
- equivalent requests have identical fingerprints and scientifically different requests do not;
- an interrupted writer leaves an auditable incomplete bundle, never a false finalized bundle; and
- registry lifecycle status cannot be advanced without its required evidence fields.

### 6.3 P2 — first end-to-end vertical slice

Scope is intentionally limited to an external or adapter-produced single table, the structural gate, source-parity Shape, source-parity Trend, Atomic Results, summary views, and bundle finalization.

Tasks:

- Resolve CSV, Parquet, and DataFrame inputs to one canonical semantic table without lossy coercion.
- Implement row-count, column-set, uniqueness, serialization, ordering, and safe-conversion validation.
- Wrap the pinned SDMetrics Column Shapes and Column Pair Trends report behavior in an isolated evaluation dependency profile.
- Preserve one Atomic Result per evaluable column and per evaluable pair, including input counts and source raw output.
- Compute only the source-defined equal-column/equal-pair summaries plus separately named benchmark report views.
- Implement synthetic-vs-test, synthetic-vs-train, and a versioned real-vs-real reference interface; keep any unresolved reference construction diagnostic.
- Produce all required Run Result files, validate them, write checksums last, and finalize immutably.
- Add `evaluate-table` and `validate-result` CLI paths independent of model training.

Parity suite:

- numerical, categorical, Boolean, datetime, and mixed-pair normal cases;
- constant, empty, one-row, missing, unseen-category, zero-range, and unequal-row-count cases where supported;
- report-level preprocessing and aggregation versus direct pinned source calls;
- exact source revision, resolved parameters, warnings, tolerances, and states; and
- intentional comparison against lower-level alternatives to prevent accidental use of Spearman or common-bin mixed-pair behavior under the source-parity identifiers.

Exit evidence:

- direct authoritative calls and benchmark-path results agree under the approved parity protocol;
- two identical requests produce semantically identical bundles and fingerprints;
- structural failures prevent official downstream metrics but still produce a valid incomplete/failed bundle;
- every requested column and pair has a denominator-accounted record; and
- no overall Fidelity score is emitted until the high-order component is protocol-frozen.

### 6.4 P3 — Validity and preprocessing boundary

Tasks:

- Implement per-column hard rules for nullability, finiteness, integer semantics, reviewed bounds, categories, string formats, and datetime ranges.
- Implement reviewed cross-column constraints with stable identifiers and applicability rules.
- Preserve original decoded output and record separate evaluator-normalized or repaired views, if permitted.
- Split dataset onboarding into raw registration, rights/provenance review, schema profiling, explicit preprocessing, split generation, and materialization.
- Replace silent deletion and implicit imputation with a required preprocessing request and a complete cleaning report.
- Fit every learned preprocessor on train only; checksum learned state and transformed schema; create a new dataset-view identity for policy changes.
- Do not choose imputation strategies until the affected datasets and strategies have been discussed and approved.

Exit evidence:

- hand-computable rule, malformed-schema, lossy-conversion, no-constraint, and width-sensitivity tests pass;
- tests prove test data cannot affect fitted preprocessing state;
- original validity never improves because an evaluation-only view was repaired; and
- registration alone is byte-preserving apart from an explicitly declared serialization conversion.

### 6.5 P4 — Utility

Local Utility tasks:

- Implement identical Dummy, TRTR, and TSTR arms over one held-out real test set.
- Freeze separate classification and regression evaluator profiles after pilot review.
- Preserve Macro-F1/RMSE primary raw values and applicable secondary values for every evaluator, target, and seed.
- Implement baseline-adjusted retention as a separately identified benchmark-derived result.
- Encode omitted classes, constant targets, weak TRTR-vs-Dummy baselines, predictor failures, and resource failures without dropping tasks.

Global Utility tasks:

- Implement TabStruct Equation 4 exactly at the aggregation layer: Balanced Accuracy ratio for categorical targets, inverse RMSE ratio for numerical targets, and equal-target mean.
- Keep Full-tuned, Tiny-default, and any pinned TabEval predictor profiles as different metric identities and compatibility groups.
- Exclude identifiers by default and require a Dataset Profile reason for every other target exclusion.
- Never clip ratios above one and never omit zero/non-finite denominators silently.
- Prefer wrapping the approved authoritative predictor implementation. Any required source patch pauses implementation for review.

Exit evidence:

- hand calculations, raw-arm invariants, class-support failures, constant-target cases, multiclass cases, regression edge cases, and all-target denominator tests pass;
- stochastic reproducibility and tolerance policies are predeclared;
- the selected profiles pass source parity where source parity is claimed; and
- Local Utility and Global Utility remain distinct outputs and sub-leaderboards.

### 6.6 P5 — high-order fidelity and empirical privacy

High-order tasks:

- Pilot and freeze C2ST preprocessing, discriminator, split, balancing, seed, uncertainty, and AUROC-complement transformation before it contributes to Fidelity.
- Preserve GReaT RF discriminator accuracy under a source-specific diagnostic identifier.
- Keep integrated Alpha-Precision/Beta-Recall experimental until the mixed-table embedding and integration behavior are resolved.

Privacy tasks:

- Implement exact train collision and synthetic internal duplication as separate diagnostics.
- Wrap the pinned SDMetrics DCR distance; add separately identified held-out calibration and distribution summaries.
- Define and validate at least one membership-inference threat model before freezing the privacy suite.
- Add attribute inference only for Dataset Profiles with reviewed sensitive/quasi-identifier roles and an approved threat model.
- Keep Authenticity excluded until the paper/code discrepancy is adjudicated.
- Keep Delta Presence excluded from official scoring until its semantics and failure behavior are scientifically resolved.

Exit evidence:

- every privacy output identifies attacker knowledge, member/non-member construction, representation, model, and direction;
- collision and DCR boundary/parity suites pass, including null and zero-range behavior;
- no privacy diagnostic is described as a formal privacy guarantee; and
- unresolved metrics remain present only as explicitly experimental or excluded records.

### 6.7 P6 — orchestration, efficiency, cache, and resume

Tasks:

- Extend execution to prepare, train, sample, validate, evaluate, aggregate, and report stage records.
- Record wall time, CPU time where reliable, peak RAM, peak accelerator memory, row throughput, requested/actual sample counts, warm-up policy, and excluded setup time.
- Define hardware profiles and prohibit cross-profile efficiency ranking.
- Run optional metric backends in isolated processes or environments with explicit time, memory, and failure boundaries.
- Implement content-addressed stage caching and resume without changing result identity or hiding prior failures.
- Make logs structured, redact secrets and unsafe paths, and keep deterministic scientific outputs separate from timestamps and host-specific diagnostics.

Exit evidence:

- forced timeout, out-of-memory simulation, interruption, retry, stale-cache, and partial-success tests pass;
- cache reuse proves every identity input matches and is visible in stage metadata;
- efficiency measures are reproducible within declared tolerances on a named hardware profile; and
- failures in one optional metric do not erase completed Atomic Results.

### 6.8 P7 — aggregation and leaderboard publication

Tasks:

- Validate every input bundle and construct compatibility groups before aggregation.
- Aggregate Atomic Results to runs, seeds, datasets, suites, and snapshots in the order defined by the policy.
- Implement uncertainty intervals, pairwise completeness, coverage, failed/undefined denominator accounting, ties, and correction/supersession records.
- Keep Native and Standardized Tuning comparison tracks separate.
- Enforce Official, Partial/Diagnostic, and Community publication classes and their evidence requirements.
- Generate immutable Leaderboard Snapshot bundles, human-readable tables, and machine-readable exports from the same validated records.
- Prevent display code from recomputing scientific values or changing ordering rules.

Exit evidence:

- deliberately incompatible protocol, split, preprocessing, metric, seed, hardware, and tuning records cannot be merged;
- missing or failed contributions cannot improve coverage or disappear from denominators;
- snapshot reconstruction is deterministic from its declared bundles; and
- no model, dataset, or metric can enter Official Results without its independent admission record.

### 6.9 P8 — migration and release

Tasks:

- Mark `standardized_summary.json` and `tabstruct-aligned-v1` as legacy, freeze their schema, and keep a read-only importer for a declared migration window.
- Never convert a legacy summary into an Official Result when required atomic evidence is unavailable.
- Route adapter evaluation through the new engine while preserving train/sample behavior and existing artifact locations where safe.
- Replace old `implemented` inventory language with the approved model status dimensions and evidence records.
- Update README, tutorials, examples, architecture, metric cards, dataset cards, troubleshooting, and contributor guidance in English; provide Chinese review translations where planned.
- Add license, third-party notice, citation, contributor acknowledgement, security policy, code of conduct, and release checklist after their separate audits.
- Test clean installation, table-only evaluation, one adapter smoke run, result validation, and diagnostic comparison on Linux/Python 3.11.

Exit evidence:

- legacy and new outputs cannot be confused by filename, schema, CLI label, or documentation;
- clean-checkout quickstarts pass without developer-local paths or undeclared data;
- published claims match actual lifecycle, eligibility, and support records; and
- every applicable release gate in the Repository Quality Standard has an evidence-backed decision.

## 7. Verification strategy

### 7.1 Required test layers

| Layer | Purpose | Required examples |
|---|---|---|
| Contract | Enforce schemas and invariants | versions, enums, unknown fields, finite values, paths, hashes |
| Formula unit | Verify benchmark math | hand-computable normal and boundary cases |
| Source parity | Verify authoritative behavior | direct pinned call versus wrapper on shared fixtures |
| State and negative | Prevent favorable silent failure | empty, constant, missing class, timeout, dependency failure |
| Integration | Verify subsystem boundaries | profile -> table -> metric -> bundle -> validator |
| End-to-end | Verify user workflows | external table and one real adapter on Linux/Python 3.11 |
| Determinism | Verify scientific identity | repeated seeds, process isolation, cache reuse, canonical serialization |
| Migration | Preserve intentional compatibility | legacy reader, deprecation warnings, no official promotion |
| Security and publication | Protect release artifacts | traversal, unsafe YAML, secret/path redaction, manifest allowlist |

Mocked tests are useful for control flow but cannot satisfy source parity, real smoke, scientific validation, or release-support gates.

### 7.2 Golden-fixture policy

Golden fixtures must be small, synthetic, redistributable, human-inspectable, and versioned. Each records the authoritative source revision, dependency lock, invocation parameters, raw expected output, allowed tolerance, and reason for any platform tolerance. Updating a golden value requires a review record; a dependency upgrade does not automatically authorize regeneration.

### 7.3 CI partitioning

- `core`: no model or heavy metric dependencies; contracts, profiles, CLI metadata, and bundle validation.
- `evaluation-unit`: deterministic metric formula and state tests.
- `source-parity`: isolated locked environments for each authoritative backend.
- `adapter-smoke`: selected real adapters, scheduled or hardware-tagged where necessary.
- `release`: clean installation, documentation links, artifact allowlist, security/licensing evidence, and quickstart.

Network access is disabled in ordinary tests. Tests needing downloads use pre-approved cached inputs and verify checksums.

## 8. Evidence and review control

Each work package must produce:

- implementation commit and changed public-contract inventory;
- tests and immutable fixtures;
- resolved dependency lock and license record;
- metric or model lifecycle update;
- scientific reviewer decision where formulas or semantics change;
- compatibility impact and migration note;
- documentation and limitation update; and
- machine-readable assessment against its exit gate.

Source edits, new imputation policies, hard dataset constraints, metric transformations, predictor profiles, threat models, aggregation weights, and official thresholds are review checkpoints. Implementation pauses for discussion before one of these choices is made or changed.

## 9. Release-blocking decisions

The roadmap can progress around these items, but affected outputs cannot become official until they are resolved:

- dataset-specific imputation strategies and missing-indicator policy;
- official real-vs-real reference construction;
- Trend source-faithful versus separately named common-bin variant role;
- C2ST evaluator and uncertainty profile;
- Local Utility evaluator profile and any clipped retention view;
- Global Utility predictor profile;
- integrated Alpha-Precision/Beta-Recall embedding;
- membership- and attribute-inference threat models;
- Authenticity paper/code discrepancy;
- Delta Presence scientific role;
- pilot thresholds for support, tolerances, wide-table pair sampling, and resource limits; and
- the Core Dataset Suite, Core Model Set, and hardware profiles.

These are not reasons to invent temporary official defaults. Until approved, the corresponding registry records remain at the appropriate earlier lifecycle stage and their values are diagnostic or excluded.

## 10. Milestones and definition of done

### M1 — evaluation foundation

P0 and P1 pass. Contracts and tooling are usable, but no metric is yet advertised as source-parity validated.

### M2 — trustworthy first report

P2 passes. A user can evaluate a compatible synthetic table and receive a finalized, validated bundle containing structural validation, Shape, and Trend evidence. This is the first public-preview-capable evaluation slice, subject to repository-wide release gates.

### M3 — benchmark dimensions

P3 through P6 pass for an approved subset. Validity, Utility, selected Privacy diagnostics, high-order Fidelity where frozen, and Efficiency have explicit lifecycle records and failure semantics.

### M4 — publishable benchmark

P7 and P8 pass for the declared release class. Official rankings are possible only if all contributing models, datasets, metrics, protocol records, and bundles also pass their independent gates.

The implementation is not done because code exists, a mocked test passes, or one machine produces a table. It is done when the applicable phase exit gate, lifecycle evidence, compatibility checks, documentation, and release assessment are complete.

## 11. Immediate next implementation increment

The first implementation increment should contain only:

1. root packaging and pytest discovery boundaries;
2. lightweight optional-dependency imports;
3. schema and contract skeletons for metric records, Atomic Results, Evaluation Requests, and incomplete bundles;
4. legacy identity and deprecation markers without changing metric values; and
5. core CI for Linux/Python 3.11.

It should not yet modify upstream algorithms, choose imputation strategies, publish a leaderboard, or rewrite metric formulas. Its purpose is to create a trustworthy surface on which the Shape/Trend vertical slice can be implemented and reviewed.

## 12. Related specifications

- [Evaluation Protocol](EVALUATION_PROTOCOL.md)
- [Metric Governance](METRIC_GOVERNANCE.md)
- [Metric Source Review](METRIC_SOURCE_REVIEW.md)
- [Dataset Profile Specification](DATASET_PROFILE_SPEC.md)
- [Result Specification](RESULT_SPECIFICATION.md)
- [Leaderboard Policy](LEADERBOARD_POLICY.md)
- [Repository Quality Standard](../QUALITY_STANDARD.md)
