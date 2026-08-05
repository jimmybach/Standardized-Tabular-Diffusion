# Dataset Profile Specification

Chinese translation: [DATASET_PROFILE_SPEC.zh-CN.md](DATASET_PROFILE_SPEC.zh-CN.md)

- Status: design baseline
- Profile schema version: 0.1.0
- Last updated: 2026-08-03

## 1. Purpose

This specification defines the machine-readable Dataset Profile that binds dataset identity, rights, schema, views, splits, preprocessing, prediction tasks, validity rules, privacy threat models, and metric applicability.

A Dataset Profile is part of the evaluation protocol. It prevents target selection, constraints, sensitive roles, and metric exclusions from being improvised after model results are observed.

## 2. Governing principles

Every official dataset MUST be:

- traceable to a specific source and version;
- legally reviewed for the project's intended use and distribution;
- content-addressed through checksums;
- represented by an explicit semantic schema;
- assigned a frozen split;
- processed through train-fitted transformations;
- evaluated through predeclared tasks and constraints; and
- admitted to a versioned dataset suite through recorded review.

The existence of a local file or an upstream script does not establish redistribution rights, scientific suitability, or official eligibility.

## 3. Profile identity

Every profile declares:

- profile_schema_version;
- dataset_profile_version;
- dataset_id;
- display_name;
- dataset_version;
- dataset_view;
- dataset_family;
- status;
- owners;
- review record; and
- change log.

dataset_id identifies the conceptual dataset. dataset_version identifies source content. dataset_view identifies a deterministic representation of that version.

Stable identifiers SHOULD be lowercase ASCII and portable-path safe. Display names and translations are separate.

## 4. Dataset suites

suite_membership uses one or more of:

- universal-core;
- extended-catalog; or
- diagnostic.

### 4.1 Universal Core Suite

The Universal Core Suite is the mandatory dataset set for complete Official Results. Membership requires:

- cleared access and publication rights for the intended workflow;
- stable retrieval and checksums;
- a complete schema and split;
- representative task and data-type coverage;
- successful capability tests across the intended Core Model Set;
- acceptable runtime under the official resource profile; and
- complete required evaluation metadata.

Core membership is frozen by protocol version.

### 4.2 Extended Catalog

The Extended Catalog includes valid benchmark targets that are not mandatory for complete Core coverage. Reasons may include algorithm capability, runtime, data rights, unusual schema, scale, or pending validation.

### 4.3 Diagnostic Suite

Diagnostic datasets or fixtures target specific edge cases, regressions, or failure modes. CI uses only small, synthetic, redistributable fixtures. Diagnostic results do not alter Core rankings.

## 5. Source, rights, and provenance

Required source fields include:

- canonical source URL or access mechanism;
- source publisher;
- source version or retrieval date;
- immutable archive identifier where available;
- raw-file inventory and checksums;
- dataset citation;
- license or terms;
- access restrictions;
- permitted uses;
- redistribution status;
- modification and attribution requirements; and
- rights-review decision with reviewer and date.

redistribution_status is one of:

- permitted;
- metadata-only;
- download-script-only;
- restricted;
- prohibited;
- unknown.

Unknown or prohibited data MUST NOT be committed, bundled in releases, or published as benchmark artifacts. A reproducible registration or download procedure MAY be provided only when its terms permit.

Source-code licensing does not determine dataset rights.

## 6. Canonical table contract

The profile declares:

- canonical serialization;
- encoding;
- delimiter where applicable;
- row-count expectations;
- unique column names;
- canonical column order;
- duplicate-row policy;
- primary-key or identifier policy;
- target presence;
- missing-value inventory;
- supported input forms; and
- canonical-table checksum.

The public interface MAY accept CSV, Parquet, or DataFrame. Official evaluation resolves all forms to the same canonical logical table and checksum.

## 7. Column schema

Every column declares:

- name;
- stable column_id;
- semantic_type;
- storage_type;
- role or roles;
- nullable in raw data;
- nullable in canonical model input;
- valid domain;
- category vocabulary source;
- transformation policy;
- inverse-transformation policy;
- description and units;
- sensitivity metadata; and
- constraint references.

### 7.1 Semantic types

Initial semantic types are:

- continuous;
- integer;
- categorical;
- boolean;
- datetime; and
- string.

Free text and complex nested values are outside the initial static single-table benchmark unless a future protocol explicitly admits them.

### 7.2 Column roles

A column may declare applicable roles:

- feature;
- primary_target;
- secondary_target;
- identifier;
- sensitive_attribute;
- quasi_identifier;
- group_attribute;
- ignored; or
- audit_only.

Identifiers and pure row indices are excluded from model features and Global Utility by default. An exception requires an explicit scientific rationale.

Sensitive and quasi-identifier roles MUST be based on dataset documentation and review, not guessed from column names.

### 7.3 Category vocabularies

Allowed categories come from an authoritative data dictionary or a reviewed schema declaration. If the vocabulary is learned from data, it MUST be learned from train only and this limitation MUST be explicit.

A category absent from train but valid according to an authoritative domain MAY remain schema-valid. The preprocessing and unknown-category behavior MUST be declared separately.

### 7.4 Numerical domains

Hard numerical bounds require an authoritative domain source or reviewed scientific rationale. Observed train minima and maxima are soft support diagnostics by default.

Integer columns MUST define lossless conversion rules. Values with fractional components cannot be silently truncated.

## 8. Missing values and preprocessing

Raw datasets MAY contain missing values. The initial model-input contract does not pass missing values to generative models.

Every profile with raw missingness MUST invoke the centralized preprocessing module and declare:

- missing markers;
- affected columns and rates by split;
- imputation strategy per semantic type;
- whether missing indicators are added;
- fitting partition;
- random seed where applicable;
- preprocessing configuration version;
- preprocessing implementation version;
- learned-artifact checksum;
- transformed schema checksum; and
- inverse or decoded representation.

All learned imputation and transformation state is fitted on train only. Validation and test are transformed without refitting.

Imputation policy is scientifically material. A policy change produces a new preprocessing and dataset-view identity and requires result compatibility review.

## 9. Dataset views

A dataset view is a deterministic, named, versioned transformation of a source dataset version.

Examples include:

- canonical mixed-type view;
- a view required by an authoritative upstream method;
- a no-categorical view;
- a privacy-distance analysis view; or
- an imputed view.

Suffix-like forms such as nocat or dcr are views, not independent source datasets.

Each view MUST declare:

- parent dataset and version;
- transformation graph;
- implementation and configuration versions;
- fitting boundary;
- output schema;
- checksums;
- information loss;
- intended models and metrics;
- comparability limitations; and
- inverse transformation where available.

Official cross-model comparison SHOULD use the same canonical semantic view. Method-specific internal encodings belong inside adapters and MUST decode back to the canonical semantic view for evaluation.

## 10. Split contract

Every official profile declares one frozen split identifier and:

- split-generation method;
- split seed;
- train, validation, and test membership artifacts or deterministic rule;
- row counts and class or target summaries;
- stratification or grouping rule;
- temporal ordering where applicable;
- checksums;
- leakage checks; and
- limitations.

Train is used for fitting. Validation is used only for permitted model selection. Test is used only for final evaluation.

The split validator MUST check:

- disjoint row identities where identities exist;
- exact and near-duplicate leakage according to the dataset policy;
- group leakage for grouped data;
- temporal leakage for temporal data;
- target distribution pathologies;
- minimum class support; and
- deterministic reconstruction.

Changing split membership or logic creates a new split and protocol compatibility group.

## 11. Local Utility profile

Every official predictive dataset declares exactly one primary Local Utility target:

- target column;
- task type;
- positive-class definition for binary classification;
- label mapping;
- primary metric;
- secondary metrics;
- evaluator profile;
- minimum support;
- Dummy strategy when the benchmark-derived retention is applicable;
- normalization metric identifier and version; and
- known limitations.

Optional secondary targets use the same complete declaration. They are reported separately and do not alter the primary Local Utility leaderboard weight.

The target MUST reflect a documented dataset task or a reviewed scientific purpose. It MUST NOT be chosen after comparing generator results.

The reviewed Adult and Sick diagnostic profiles now bind their primary classification targets, positive classes, label mappings, Macro-F1 and secondary metrics, support declarations, Dummy strategy, and `p4-utility-pilot@0.1.0` identity. This records the P4 pilot contract without making either dataset official-eligible.

## 12. Global Utility profile

Global Utility includes every evaluable non-identifier column as a rotated target by default.

The profile declares:

- included targets;
- excluded targets and stable reason codes;
- task type for each target;
- evaluator profile;
- Global Utility metric and predictor-profile identifiers;
- target-specific support thresholds;
- high-cardinality handling;
- datetime handling; and
- aggregation weights.

For the TabStruct-compatible profile, each categorical target uses the Balanced-Accuracy TSTR/TRTR ratio, each numerical target uses the RMSE TRTR/TSTR ratio, and targets are equally weighted. A different normalization or predictor profile creates a distinct metric identity and compatibility group. Exclusion because a model performs poorly is prohibited.

Adult and Sick enumerate every canonical model-view column in their P4 diagnostic Global Utility declaration. Audit-only and identifier columns remain outside the canonical view; no target is silently omitted.

## 13. Validity profile

### 13.1 Hard column rules

Hard rules may define:

- nullability;
- finite numerical values;
- integer requirements;
- authoritative bounds;
- allowed categories;
- string format or length;
- datetime parse and range;
- uniqueness; and
- identifier format.

Every hard rule requires a source:

- authoritative-data-dictionary;
- dataset-documentation;
- legal-or-policy-requirement; or
- recorded-human-review.

### 13.2 Cross-column constraints

Cross-column rules may define:

- temporal order;
- conditional domains;
- mutual exclusion;
- sum or total relationships;
- functional dependencies;
- logically impossible combinations; and
- dataset-specific business rules.

Each constraint declares an identifier, expression or executable implementation, applicability, missing-value behavior, evidence source, severity, test fixtures, and version.

### 13.3 Soft diagnostics

Soft diagnostics MAY include:

- outside-train-support rate;
- unseen category combinations;
- rare-group behavior;
- observed-range exceedance;
- inferred dependencies; and
- anomaly scores.

Soft diagnostics do not determine hard validity unless promoted through reviewed evidence and a new profile version.

## 14. Privacy-risk profile

Privacy evaluation requires predeclared roles and threat models.

The profile may declare:

- sensitive attributes;
- quasi-identifiers;
- permitted attacker knowledge;
- member and non-member populations;
- group definitions;
- exact-collision normalization;
- DCR feature representation;
- membership-inference applicability;
- attribute-inference target and known features;
- minimum group and class support;
- formal differential-privacy metadata expectations; and
- restrictions on row-level artifact publication.

If a defensible sensitive attribute or quasi-identifier set is unavailable, the corresponding attack is not_applicable. It is not reported as zero risk.

Fairness or subgroup analysis runs only for declared group attributes with sufficient support. Thresholds are protocol-frozen and group-level results MUST NOT expose sensitive individuals.

## 15. Metric applicability

The profile maps each metric or metric family to:

- required;
- optional;
- experimental;
- not_applicable; or
- prohibited.

Every non-required decision needs a stable reason and evidence. Applicability is fixed before model evaluation.

Examples include:

- regression metrics prohibited for categorical targets;
- attribute inference not applicable without a reviewed sensitive target;
- Global Utility excluded for pure identifiers;
- datetime diagnostics optional until their protocol is frozen; and
- a privacy attack prohibited when its artifact would violate data terms.

## 16. Predeclared evaluation repair

The profile defines whether evaluator-only repair is permitted for content-invalid synthetic outputs.

Allowed repair MUST:

- leave original synthetic data immutable;
- use train-fitted parameters only;
- be deterministic under a recorded seed;
- record every affected column and row count;
- map unknown categories to an explicit unknown level;
- preserve the original Validity penalty; and
- set evaluation_repair_applied.

Structural failures cannot be repaired into official runs.

## 17. Example profile

~~~yaml
profile_schema_version: 0.1.0
dataset_profile_version: 0.1.0

identity:
  dataset_id: example-income
  display_name: Example Income Dataset
  dataset_version: source-version
  dataset_view: canonical-imputed-v1
  status: registered

suite_membership:
  - extended-catalog

source:
  publisher: pending
  url: pending
  retrieved_at: 2026-08-03
  raw_checksums: {}
  citation: pending
  license_or_terms: pending-review
  redistribution_status: unknown

canonical_table:
  formats: [csv, parquet, dataframe]
  expected_columns: [age, occupation, income]
  canonical_checksum: pending

columns:
  - column_id: age
    name: age
    semantic_type: integer
    storage_type: int64
    roles: [feature]
    raw_nullable: true
    model_input_nullable: false
    hard_domain:
      minimum: 0
      source: recorded-human-review

  - column_id: occupation
    name: occupation
    semantic_type: categorical
    storage_type: string
    roles: [feature, quasi_identifier]
    raw_nullable: true
    model_input_nullable: false
    categories:
      source: pending
      values: []

  - column_id: income
    name: income
    semantic_type: boolean
    storage_type: string
    roles: [primary_target, sensitive_attribute]
    raw_nullable: false
    model_input_nullable: false
    categories:
      source: pending
      values: [low, high]

preprocessing:
  fitted_on: train
  missing_policy:
    implementation: centralized
    configuration_version: pending
  learned_artifact_checksum: pending

split:
  split_id: official-v1
  method: pending
  seed: pending
  checksums: {}

local_utility:
  primary_target: income
  task_type: binary_classification
  primary_metric: macro_f1
  secondary_metrics: [balanced_accuracy, roc_auc, pr_auc]
  evaluator_profile: pending
  dummy_strategy: most_frequent
  normalization_metric: baseline_adjusted_local_utility_retention@pending

global_utility:
  default_include_non_identifiers: true
  exclusions: []
  metric: tabstruct_global_utility@pending
  predictor_profile: pending

validity:
  hard_constraints: []
  soft_diagnostics:
    - outside_train_support_rate

privacy:
  sensitive_attributes: [income]
  quasi_identifiers: [occupation]
  threat_models:
    membership_inference: pending-review
    attribute_inference: pending-review

metric_applicability: {}

review:
  owners: []
  reviewers: []
  decision: pending
~~~

This example is intentionally incomplete and MUST NOT be used as a real dataset profile.

## 18. Profile lifecycle

Recommended profile lifecycle:

1. registered: identity and candidate source are recorded.

2. source-and-rights-reviewed: source, citation, access, license, and redistribution status are reviewed.

3. schema-reviewed: columns, semantic types, roles, domains, and missingness are reviewed.

4. split-and-preprocessing-validated: split boundaries, train-only fitting, checksums, and transformed schema pass validation.

5. evaluation-profile-complete: tasks, constraints, privacy roles, applicability, and evaluator requirements are complete.

6. benchmark-eligible: capability, runtime, legal, scientific, and protocol gates pass for a declared suite.

7. release-supported: retrieval, validation, documentation, tests, ownership, and compatibility are maintained for release.

Only benchmark-eligible profiles enter official suites. Universal Core membership additionally requires protocol-level approval.

## 19. Onboarding and review

Dataset onboarding MUST include:

1. source and rights inventory;
2. raw integrity verification;
3. schema and semantic-role review;
4. missingness and preprocessing proposal;
5. split design and leakage audit;
6. target and evaluator review;
7. hard-constraint evidence review;
8. privacy-role and threat-model review;
9. metric applicability review;
10. representative model capability and runtime tests;
11. profile validation;
12. suite-admission decision.

The reviewer approving rights SHOULD be independent from the person who merely downloaded the data. Sensitive-role and domain-constraint decisions require subject-matter review when the dataset demands it.

## 20. Change control

A new dataset profile or view version is required when a change affects:

- source content;
- data rights;
- rows or columns;
- semantic types or roles;
- target definition;
- category vocabulary;
- hard constraints;
- split;
- preprocessing or imputation;
- privacy roles or threat models;
- metric applicability;
- evaluator profile; or
- aggregation weight.

Historical results remain bound to their original profile, split, and preprocessing checksums.

## 21. Initial catalog work

The intended 21-dataset catalog requires individual review. No dataset is admitted merely because it appears in an existing script or research package.

For each candidate, the project will determine:

- exact source and rights;
- canonical version;
- task and target;
- schema and missingness;
- Core, Extended, or Diagnostic role;
- constraints and privacy metadata;
- compatibility across candidate models; and
- pilot runtime.

Universal Core membership is selected only after model capability tests across the intended Core Model Set.

## 22. Related specifications

- [Evaluation Protocol](EVALUATION_PROTOCOL.md)
- [Leaderboard Policy](LEADERBOARD_POLICY.md)
- [Metric Governance](METRIC_GOVERNANCE.md)
- [Result Specification](RESULT_SPECIFICATION.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [Repository Quality Standard](../QUALITY_STANDARD.md)
