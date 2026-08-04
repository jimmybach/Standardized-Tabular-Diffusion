# Metric Governance

Chinese translation: [METRIC_GOVERNANCE.zh-CN.md](METRIC_GOVERNANCE.zh-CN.md)

- Status: design baseline
- Governance version: 0.1.0
- Last updated: 2026-08-03

## 1. Purpose

This specification governs how evaluation metrics are identified, sourced, implemented, validated, versioned, and admitted to official benchmark scoring.

Its central rule is that a paper definition, an upstream software behavior, and this benchmark's reporting contract are three separate layers. Agreement among them MUST be demonstrated rather than assumed.

## 2. Scope

This policy applies to every computation that can affect:

- an official or diagnostic score;
- metric applicability or result state;
- normalization or direction;
- aggregation or leaderboard ordering;
- uncertainty or significance;
- a privacy-risk statement; or
- a performance measurement.

Simple display formatting is outside scope only when it cannot change stored numerical meaning.

## 3. Three-layer source model

### 3.1 Method layer

The method layer records the scientific definition:

- paper or technical report;
- equation, algorithm, or prose definition;
- stated direction and interpretation;
- assumptions and threat model;
- known limitations; and
- official supplementary material.

A citation alone is insufficient when the source is ambiguous. The registry MUST identify the relevant section, equation, algorithm, or source artifact where practical.

### 3.2 Implementation layer

The implementation layer records executable behavior:

- official package or repository;
- exact package version or immutable commit;
- function or source path;
- default and explicitly overridden parameters;
- random behavior;
- edge-case behavior;
- dependency stack; and
- license and redistribution requirements.

Software behavior MUST NOT be attributed to a paper when the paper does not define it.

### 3.3 Benchmark contract layer

The benchmark contract defines:

- canonical input representation;
- applicability;
- raw output capture;
- state mapping;
- normalization;
- aggregation;
- uncertainty;
- reporting;
- admission status; and
- compatibility version.

A benchmark transformation MUST preserve the upstream raw result. It MUST NOT be presented as the original metric if it changes scientific meaning.

## 4. Metric Registry

Every metric MUST have a machine-readable registry entry before implementation work is considered complete.

### 4.1 Required identity fields

- metric_id;
- metric_version;
- display_name;
- dimension;
- subdimension;
- description;
- definition origin;
- planned leaderboard role;
- owner;
- lifecycle status; and
- review record.

`definition_origin` is one of `source-defined`, `source-parameterized`, `benchmark-derived`, or `benchmark-native`. A source-parameterized metric preserves the source formula while freezing source-permitted choices. A benchmark-derived metric transforms or extends a source output. A benchmark-native metric has no claimed upstream formula.

Metric identifiers MUST identify non-equivalent variants. For example, implementations with different Authenticity indexing rules, Column Pair Trends discretization, Utility normalization, or predictor profiles require different identifiers.

### 4.2 Required source fields

- method references;
- upstream repository or package;
- exact revision or version;
- implementation symbol or source path;
- source authority;
- implementation mode;
- retrieved date;
- integrity hash where applicable;
- license;
- required attribution; and
- known deviations.

### 4.3 Required semantic fields

- supported table and column types;
- required dataset-profile metadata;
- input view;
- fitted-on partition;
- output shape and unit;
- raw range;
- raw direction;
- target value when applicable;
- undefined conditions;
- minimum support;
- randomness;
- default parameters and overrides;
- normalization function and version;
- aggregation rule;
- uncertainty procedure; and
- failure contribution policy.

### 4.4 Required validation fields

- unit-test evidence;
- boundary-test evidence;
- source-parity evidence;
- numerical tolerance;
- reference fixtures;
- supported environment;
- unresolved limitations; and
- release decision.

### 4.5 Direction vocabulary

raw_direction uses one of:

- maximize;
- minimize;
- target;
- distributional; or
- descriptive.

Target metrics MUST declare the target value and distance rule. Distributional metrics, such as calibrated DCR, MUST declare the reference distribution and comparison outputs. They MUST NOT be coerced into a maximize or minimize direction without a separately reviewed transformation.

### 4.6 Illustrative entry

~~~yaml
metric_id: calibrated_mixed_dcr
metric_version: 0.1.0
dimension: empirical_privacy_risk
subdimension: direct_disclosure
definition_origin: benchmark-derived
planned_leaderboard_role: diagnostic
lifecycle_status: registered

method:
  references:
    - identifier: pending-citation-record

implementation:
  authority: official-upstream
  mode: package-wrapper
  package: pending
  package_version: pending
  source_symbol: pending
  license: pending-review

semantics:
  input_view: decoded_semantic_table
  fitted_on: train
  raw_direction: distributional
  outputs:
    - synthetic_to_train_dcr_distribution
    - test_to_train_dcr_distribution
  requires_reference_calibration: true
  lower_tail_quantiles: [0.01, 0.05]

validation:
  status: pending
  source_parity: pending
~~~

An illustrative entry is not an implementation or admission claim.

## 5. Implementation source hierarchy

The preferred implementation order is:

1. wrapper around a locked official package;
2. wrapper around a locked official source revision;
3. reviewed patch to locked official source;
4. local compatibility reimplementation.

### 5.1 Package wrapper

A package wrapper MUST verify the installed version at runtime. Version mismatch MUST fail clearly. It MUST NOT silently import any available package version.

The wrapper SHOULD call the public upstream API where it preserves the required semantics. If private symbols are necessary, the dependency and compatibility risk MUST be recorded.

### 5.2 Source wrapper

Official source MAY be pinned when no suitable package exists or a required commit is unavailable through package distribution. The project SHOULD avoid copying an entire repository when a minimal, licensed, integrity-checked dependency mechanism is sufficient.

### 5.3 Approved source patch

A patch is a last resort. It MUST:

- identify the exact upstream revision;
- explain why wrapping is insufficient;
- be minimal and isolated;
- classify compatibility versus semantic effect;
- preserve license and notices;
- include before-and-after tests; and
- pass source-parity review for every claim of equivalent behavior.

A semantic patch produces a distinct metric identity unless and until it becomes part of an authoritative upstream release.

### 5.4 Local reimplementation

A local reimplementation is experimental by default. Matching a formula on ordinary inputs is insufficient. Official admission requires:

- independent definition review;
- comprehensive boundary tests;
- direct comparison with authoritative executable behavior;
- agreement within declared tolerances across representative fixtures and datasets;
- equivalent result-state behavior; and
- explicit approval.

If authoritative code cannot be run, the implementation MUST remain clearly identified as a local method and MUST NOT claim source parity.

## 6. Metric lifecycle

Lifecycle status is cumulative:

1. registered: identity, purpose, sources, license status, intended role, and known open questions are recorded.

2. definition-reviewed: paper, upstream behavior, direction, assumptions, discrepancies, and proposed benchmark contract have been reviewed.

3. implementation-complete: the wrapper or implementation and structured result contract exist.

4. unit-validated: deterministic unit, boundary, negative, and state tests pass.

5. source-parity-validated: direct authoritative calls and benchmark-path calls agree under the approved parity protocol.

6. protocol-frozen: formula, source version, input representation, direction, normalization, aggregation, uncertainty, applicability, and failure policy are fixed for a protocol version.

7. release-supported: the metric has an owner, supported environment, dependency lock, documentation, tests, compatibility policy, and release approval.

Only protocol-frozen and release-supported metrics MAY affect Official Results. Earlier lifecycle stages MAY appear only in clearly labeled diagnostic output.

## 7. Raw and derived values

Every transformation preserves the upstream or method-level raw output.

A metric result MAY contain:

- raw_value;
- normalized_value;
- aggregate_contribution;
- reference_value or distribution;
- direction;
- transformation identifier and version; and
- clipping indicator.

Raw values MUST NOT be overwritten by normalized values. Clipping, inversion, scaling, calibration, baseline adjustment, or thresholding MUST be explicit and independently tested.

Examples include:

- raw GReaT-style discriminator accuracy as a source-defined diagnostic, versus raw C2ST AUROC and its benchmark-derived fidelity complement;
- raw Dummy, TSTR, and TRTR values, the benchmark-derived Local Utility retention, and the distinct TabStruct performance ratio;
- raw DCR distributions and calibrated low-tail diagnostics; and
- raw wall-clock measurements and derived rows-per-second throughput.

## 8. Result states

Every metric call returns exactly one state:

- computed;
- mathematically_undefined;
- insufficient_support;
- not_applicable;
- implementation_failure; or
- resource_failure.

### 8.1 State rules

- computed requires a finite value or a valid structured distribution result.
- mathematically_undefined applies when the requested quantity has no mathematical value under the observed inputs.
- insufficient_support applies when the quantity is meaningful but sample, class, group, or neighborhood support is below its declared requirement.
- not_applicable applies when the dataset profile or task does not satisfy the metric's declared domain.
- implementation_failure applies to defects, unexpected exceptions, incompatible dependency behavior, or invalid internal output.
- resource_failure applies to timeout, out-of-memory, disk exhaustion, or another declared resource limit.

A non-computed state MUST have a null numeric value, reason code, human-readable detail, and applicable counts. Bare NaN and infinity are prohibited in serialized official results.

### 8.2 Aggregation effect

Every registry entry MUST predeclare the aggregation effect of each state. A state MUST NOT be reclassified after observing whether the result helps or harms a model.

## 9. Validation requirements

### 9.1 Core boundary suite

Every official metric MUST test, where applicable:

- empty input;
- one row;
- one column;
- constant numerical columns;
- zero numerical range;
- single-class targets;
- missing target classes in synthetic data;
- unseen synthetic categories;
- missing values;
- positive and negative infinity;
- insufficient group support;
- identical real and synthetic tables;
- deliberately separated distributions;
- changed column order;
- unequal row counts;
- deterministic repeated execution; and
- malformed metadata.

Expected behavior includes both numerical results and structured states.

### 9.2 Golden fixtures

Golden fixtures MUST be small, licensed for repository inclusion, deterministic, and explainable by hand or authoritative output. They MUST store:

- input checksum;
- authoritative source version;
- resolved parameters;
- expected raw outputs;
- expected states;
- numerical tolerance; and
- fixture-generation evidence.

Large real datasets are not substitutes for understandable boundary fixtures.

### 9.3 Source parity

Parity tests execute:

~~~text
direct authoritative implementation
benchmark wrapper or adapter
~~~

on identical inputs and resolved parameters.

The parity protocol MUST compare:

- raw values;
- shapes and labels;
- random seeds;
- state behavior;
- warnings that affect interpretation;
- deterministic preprocessing;
- aggregation inputs; and
- supported data-type combinations.

Exact equality is required for deterministic discrete outputs where expected. Floating-point and stochastic outputs use predeclared numerical or statistical tolerances.

### 9.4 Scientific review

Source parity alone does not make a metric scientifically appropriate. Reviewers MUST also confirm that:

- the threat model or quality construct matches the benchmark claim;
- direction is correctly interpreted;
- required reference data are available without leakage;
- the aggregation does not hide unsupported cases;
- the metric is not redundant in a misleading way; and
- limitations are visible to users.

## 10. Discrepancy policy

When paper prose, equations, authors' code, or third-party code disagree:

1. record each behavior separately;
2. identify the authoritative reproduction target;
3. do not silently repair or merge definitions;
4. assign distinct metric identifiers to material variants;
5. state which variant is official, diagnostic, or excluded; and
6. preserve evidence for the decision.

Examples that require this treatment include contradictory DCR direction statements, paper-versus-repository Authenticity indexing, report-versus-direct-metric mixed-pair discretization, and favorable constant-target fallbacks.

## 11. Dependency and execution isolation

Metric dependencies MUST be version-locked. The evaluator MUST:

- verify dependency versions during preflight;
- isolate incompatible metric environments when necessary;
- record runtime backend and optional acceleration;
- control random seeds and thread settings;
- avoid mutable global import state where concurrent evaluation is supported; and
- fail explicitly when the required implementation is unavailable.

Fallback behavior is permitted only when the registry declares a separately identified fallback metric. A fallback result is not source parity with the unavailable implementation.

## 12. Licensing and attribution

Before code or assets are copied, the project MUST review source license, file-level notices, redistribution terms, and citation obligations.

Where practical, a locked dependency plus wrapper is preferred over vendoring. Vendored code MUST retain copyright, license, NOTICE, modification annotations, and immutable source identity.

A top-level repository license does not relicense third-party metric code.

## 13. Metric versioning

A new metric version is required when any of the following changes numerical meaning or result state:

- paper or method target;
- upstream revision or package version;
- input encoding or distance representation;
- parameter defaults or overrides;
- randomness;
- formula;
- direction;
- normalization or clipping;
- aggregation;
- applicability;
- undefined or failure handling;
- numerical tolerance; or
- bug fix.

Documentation-only changes MAY retain the version when they cannot alter interpretation.

Historical results remain bound to their original metric version. Recalculation produces new immutable results rather than overwriting history.

## 14. Planned initial metric roles

This table records the source audit decision and design intent, not current implementation status.

| Metric family | Definition origin and locked behavior | Initial role |
|---|---|---|
| SDMetrics Column Shapes | Source-defined at pinned commit: KSComplement for numerical/datetime, TVComplement for categorical/Boolean, equal-column mean | Official Shape candidate |
| SDMetrics Column Pair Trends | Source-defined at pinned commit: Pearson for continuous pairs; contingency similarity otherwise; report-specific independent real/synthetic discretization for mixed pairs | Official Trend candidate |
| Spearman or common train-fitted-bin Trend | Benchmark-derived variant with a distinct identifier | Experimental pending scientific comparison |
| GReaT RF discriminator accuracy | Source-defined accuracy with target 0.5 | Source-parity diagnostic |
| C2ST AUROC fidelity complement | Benchmark-derived AUROC transformation | Official high-order candidate after pilot freeze |
| Integrated Alpha-Precision and Beta-Recall | Source-defined formulas; pinned code uses a 30-point grid and Euclidean embedding distances | Experimental until embedding and integration validation |
| Raw TRTR and TSTR | Source-parameterized TSTR evaluation; task metrics and predictors are profile-specific | Official Utility evidence |
| Baseline-adjusted Local Utility retention | Benchmark-derived from Dummy, TSTR, and TRTR; not a TabStruct formula | Official Local Utility candidate after pilot freeze |
| TabStruct Local and Global Utility | Source-defined Balanced-Accuracy TSTR/TRTR ratio for categorical targets, RMSE TRTR/TSTR ratio for numerical targets, and equal-target Global mean | Official Global Utility candidate; source-comparable Local diagnostic |
| Column and cross-column validity | Benchmark-native, motivated but not defined by PAFT | Official Validity candidates |
| Exact train collision | Benchmark-native | Official privacy-risk diagnostic candidate |
| SDMetrics mixed DCR plus held-out calibration | Source-defined distance implementation with benchmark-derived distribution and low-tail comparisons following GReaT's reference principle | Official distributional diagnostic candidate |
| Alaa Authenticity | Paper-defined and repository-executed variants disagree and require separate identifiers | Excluded from Official Results pending adjudication and parity validation |
| Membership inference | Source implementation not yet selected | Required before privacy suite freeze |
| Attribute inference | Source implementation and dataset threat model not yet selected | Dataset-profile-dependent candidate |
| SynthCity Delta Presence | Source-defined executable behavior returns a maximum cluster count ratio, not a bounded probability | Excluded from official scoring pending scientific resolution |
| Timing, throughput, RAM, and VRAM | Benchmark-native operational definitions | Official Efficiency candidates within hardware profiles |

No row in this table bypasses the lifecycle or release gate.

## 15. Review and ownership

Every release-supported metric requires:

- an implementation owner;
- a scientific reviewer;
- a compatibility and dependency reviewer where different;
- a dated admission decision;
- linked evidence;
- a periodic upstream review cadence; and
- a deprecation or replacement plan.

A reviewer MUST disclose a material conflict when they authored the local implementation or the benchmark claim being validated. Independent review is required for semantic patches and privacy claims.

## 16. Official admission checklist

A metric MUST NOT affect Official Results until:

- registry fields are complete;
- licensing is cleared;
- definition review is approved;
- implementation identity is immutable;
- raw and derived values are separated;
- structured states are implemented;
- boundary and negative tests pass;
- source parity passes where claimed;
- scientific appropriateness is approved;
- protocol semantics are frozen;
- result-schema compatibility passes;
- documentation states limitations; and
- a release-supported decision is recorded.

## 17. Related specifications

- [Evaluation Protocol](EVALUATION_PROTOCOL.md)
- [Metric Source Review](METRIC_SOURCE_REVIEW.md)
- [Leaderboard Policy](LEADERBOARD_POLICY.md)
- [Result Specification](RESULT_SPECIFICATION.md)
- [Dataset Profile Specification](DATASET_PROFILE_SPEC.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [Repository Quality Standard](../QUALITY_STANDARD.md)
