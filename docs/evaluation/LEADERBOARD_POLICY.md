# Leaderboard Policy

Chinese translation: [LEADERBOARD_POLICY.zh-CN.md](LEADERBOARD_POLICY.zh-CN.md)

- Status: design baseline
- Policy version: 0.1.0
- Last updated: 2026-08-03

## 1. Purpose

This policy defines how results qualify for publication, which results may be compared, how scores are aggregated, and how official, partial, diagnostic, and community results are presented.

It is subordinate to the [Repository Quality Standard](../QUALITY_STANDARD.md) and uses the metric semantics in [Evaluation Protocol](EVALUATION_PROTOCOL.md). Publication on a page called a leaderboard does not by itself make a result official.

## 2. Terminology and independent classifications

The benchmark deliberately separates properties that are often conflated.

### 2.1 Eligibility track

Eligibility track describes scientific and provenance eligibility:

- official;
- experimental; or
- excluded.

These values are defined by the Repository Quality Standard. A model is not official merely because it runs successfully or is maintained by this repository.

### 2.2 Comparison track

Comparison track describes configuration and tuning conditions:

- native; or
- standardized-tuning.

A model MAY appear in both comparison tracks through separate runs. Results from different comparison tracks MUST NOT be mixed.

### 2.3 Support level

Support level describes maintenance commitment:

- unsupported;
- experimental; or
- release-supported.

Support level is independent of eligibility and comparison track. A release-supported experimental implementation remains ineligible for official rankings.

### 2.4 Publication class

Published results use one of:

- Official Results;
- Partial/Diagnostic Results; or
- Community Results.

Publication class does not weaken the source, validation, or support requirements of the classifications above.

## 3. Native comparison track

The native track answers:

> How does the authoritative implementation perform when used according to its official or documented intended configuration?

### 3.1 Configuration precedence

Native configuration is selected in this order:

1. an official repository configuration explicitly recommended for comparable benchmarking;
2. a configuration published in the method paper;
3. official package defaults; or
4. a minimal compatibility configuration when the preceding options cannot run.

The selected source MUST be cited and the resolved configuration MUST be frozen before test evaluation. A minimal compatibility configuration requires a recorded rationale and MUST NOT be chosen by inspecting test results.

### 3.2 Permitted adaptation

The native track permits:

- approved adapter-level input and output conversion;
- configuration mapping that preserves upstream semantics;
- deterministic artifact discovery and metadata capture; and
- approved compatibility patches that have passed native parity validation.

It does not permit unapproved semantic patches, local algorithm substitution, test-driven parameter changes, or silent fallback to a different implementation.

### 3.3 Early stopping

An authoritative early-stopping procedure MAY inspect validation only. Its monitor, patience, checkpoint rule, and selected epoch MUST be recorded. Test metrics MUST NOT affect checkpoint selection.

## 4. Standardized Tuning comparison track

The standardized-tuning track answers:

> How does each authoritative implementation perform after receiving a declared and comparable opportunity for hyperparameter search?

### 4.1 Search space

Every model has a reviewed, model-specific search space. Different algorithms need not expose the same parameter names or ranges, but all spaces MUST:

- be justified by upstream documentation or method behavior;
- be frozen before search begins;
- exclude parameters that change the declared method identity;
- state parameter distributions and conditional dependencies; and
- carry a version and integrity hash.

Search spaces MUST NOT be enlarged after observing test performance.

### 4.2 Search method and budgets

The initial search method is seeded random search. A tuning profile imposes both:

- a maximum trial count; and
- a maximum compute-time or resource budget.

Search stops when either limit is reached. Dataset-scale profiles MAY define different small, medium, and large budgets, but the scale rule and all thresholds MUST be frozen before execution.

Bayesian optimization or another search method requires a separately versioned tuning profile and cannot be silently mixed with random-search results.

### 4.3 Selection data and objective

Tuning trials train on train and select on validation. Test remains inaccessible until the best configuration is frozen.

The initial selection principle is:

1. satisfy a pilot-frozen minimum validity gate; then
2. maximize the harmonic mean of validation fidelity and the benchmark-derived, baseline-adjusted validation Local Utility retention.

This objective prevents one strong component from fully masking a collapsed component. Privacy attacks are not a tuning target. Efficiency is enforced through the resource budget rather than added to the quality objective.

The precise component set, numerical treatment of undefined retention, clipping transformation if any, validity threshold, and tie rule are pilot-frozen items. This tuning objective MUST NOT be described as the TabStruct utility formula.

### 4.4 Final runs and accounting

After selection:

1. the best resolved configuration and hash are frozen;
2. the model is retrained on the same train partition;
3. five official generation seeds are executed;
4. test evaluation is run once per final seed; and
5. every trial, failure, prune decision, and resource measurement is retained.

The published result MUST report both total search cost and final retraining cost. Reporting only the fastest or best trial is prohibited.

## 5. Official Results admission

A result enters Official Results only when all of the following hold:

- the model has an official eligibility decision under the Repository Quality Standard;
- its source authority, reproduction target, upstream revision, modification status, and parity evidence are complete;
- the model is benchmark-eligible for the declared protocol and dataset suite;
- all scoring metrics are protocol-frozen and release-supported;
- dataset profiles, data rights, splits, preprocessing, and checksums are approved;
- every required Universal Core Suite dataset and five-seed run is complete;
- no prohibited test access occurred;
- the result bundle passes schema, integrity, provenance, and compatibility validation;
- mandatory failures or missing denominators are absent; and
- reviewers approve the admission record.

Release support is required for a model advertised as part of the public Core Model Set. A protocol MAY publish official evidence for an otherwise maintained model only when the release plan and Repository Quality Standard explicitly permit it.

## 6. Partial and Diagnostic Results

A run belongs to Partial/Diagnostic Results when it is scientifically useful but fails complete official coverage, for example:

- only some Universal Core datasets are complete;
- fewer than five generation seeds are available;
- a required metric remains experimental;
- a resource limit prevented completion;
- the implementation is still in parity review; or
- the run uses a non-default diagnostic sample-size profile.

Partial results MUST:

- state the missing requirements;
- show completed and failed coverage;
- remain visually separate from Official Results; and
- avoid an official aggregate rank.

Partial publication is not a mechanism for hiding difficult datasets.

## 7. Community Results

Community Results accept valid result bundles produced outside the official execution environment.

They MUST identify:

- submitter and repository revision;
- model and implementation provenance;
- protocol and metric versions;
- dataset checksums;
- resolved configuration and seeds;
- environment and hardware; and
- all validation warnings.

Community Results MUST NOT be labeled official even if their values are numerically reproducible. Promotion requires an official admission review and, where required, rerun by maintainers in the official environment.

External synthetic tables without verifiable generation provenance remain Community or Diagnostic Results.

## 8. Dataset suites and coverage

Datasets belong to versioned suites:

- Universal Core Suite;
- Extended Catalog; or
- Diagnostic Suite.

Suite membership is fixed by protocol version. Special forms such as suffix-based, no-categorical, or DCR-oriented forms are versioned dataset views rather than silently independent datasets.

Complete Official Results require one hundred percent of the Universal Core Suite and all mandatory five-seed runs. Extended and Diagnostic results are displayed separately and do not alter Core coverage.

Coverage reports MUST distinguish:

- applicable and computed;
- mathematically undefined;
- insufficient support;
- not applicable;
- implementation failure; and
- resource failure.

## 9. Aggregation

### 9.1 Aggregation order

The required hierarchy is:

1. preserve atomic values;
2. aggregate evaluator repetitions, columns, pairs, predictors, or targets according to the metric definition;
3. aggregate generation seeds within each dataset;
4. compute the dataset summary; and
5. macro-average dataset summaries with equal dataset weight.

Rows, targets, predictor count, or seed count MUST NOT cause one dataset to receive greater cross-dataset weight.

### 9.2 Dimension outputs

The leaderboard publishes:

- Fidelity components and the equal-component Fidelity score;
- raw Local Utility results and the separately identified benchmark-derived Local Utility retention;
- source-defined TabStruct Global Utility ratios, including every per-target ratio;
- Validity components and Validity score;
- separate empirical privacy-risk diagnostics; and
- separate Training Time, Sampling Throughput, and Peak Resource Usage views.

There is no cross-dimension overall score. There is no unified Privacy Score or Efficiency Score. Local Utility and Global Utility are not combined in the initial protocol. The Local retention and TabStruct Global Utility ratio are non-equivalent normalizations and MUST NOT be relabeled as one another.

### 9.3 Undefined and failed contributions

The metric definition MUST declare whether an undefined atomic value:

- makes the parent aggregate undefined;
- is excluded as genuinely not applicable with an explicit denominator; or
- receives a protocol-defined worst contribution because it represents model failure.

These choices MUST be made before result inspection. A constant synthetic target that prevents training is a support failure and receives the defined lowest utility contribution; it is not silently excluded.

Official aggregate results require all mandatory contributions. Results that cannot meet the requirement move to Partial/Diagnostic Results rather than altering the denominator.

## 10. Ranking and uncertainty

### 10.1 Ordering

Ranks use full-precision unrounded aggregate values. Display rounding MUST NOT change rank order. Every rank is shown with score, uncertainty interval, dataset coverage, seed coverage, and protocol identity.

### 10.2 Confidence intervals

The official protocol uses a versioned hierarchical or paired bootstrap that respects dataset and seed structure. The resampling unit, number of replicates, interval method, and random seed MUST be recorded.

### 10.3 Ties

Models that are practically and statistically indistinguishable receive a tie annotation. The equivalence margin and statistical decision rule are frozen by pilot evidence.

Point estimates MAY remain sortable for navigation, but the interface MUST NOT imply a scientifically meaningful ordering inside a tie group.

### 10.4 Pairwise claims

Pairwise superiority claims SHOULD use paired dataset-seed comparisons and an appropriate multiple-comparison policy when many models are tested. A visual rank alone is not evidence of general superiority.

## 11. Efficiency comparison

Efficiency results are comparable only within the same hardware profile, software profile, thread limit, device count, sample-size profile, and timing definition.

Official efficiency views MUST expose:

- total suite time;
- per-dataset time;
- throughput;
- peak RAM and VRAM;
- checkpoint size;
- timeout and out-of-memory coverage; and
- Native or Standardized Tuning cost scope.

The project does not normalize wall-clock time across different hardware using theoretical performance ratios.

## 12. Leaderboard views

The initial release produces static HTML and Markdown views backed by downloadable structured data.

Users MUST be able to filter by:

- comparison track;
- Official, Partial/Diagnostic, or Community publication class;
- dataset or suite;
- task type;
- model family;
- protocol and metric version;
- hardware profile; and
- support and validation status.

The interface MUST support drill-down from model summary to dataset, seed, and applicable atomic results. Charts MUST link to or ship with the data used to render them.

Recommended views include dataset heatmaps, coverage matrices, paired comparisons, uncertainty plots, Fidelity-Utility views, Utility-Privacy Pareto views, and Quality-Cost Pareto views. Radar charts MAY summarize dimensions but MUST NOT be described as an overall score.

## 13. Submission and publication workflow

The initial official workflow is:

~~~text
submit adapter or configuration
→ validate identity, provenance, and eligibility
→ run interface and smoke checks
→ execute in the official environment
→ validate the result bundle
→ review admission evidence
→ publish through a pull request
~~~

A leaderboard update MUST be reviewable as a repository change. It MUST identify added, removed, superseded, and invalidated results.

The first release does not require an online evaluation service or database. A static publication pipeline is preferred until submission volume justifies additional operational complexity.

## 14. Versioning, correction, and invalidation

A leaderboard snapshot binds:

- repository release;
- protocol version;
- metric-registry version;
- dataset-suite version;
- result-schema version; and
- official hardware profile where relevant.

Changes that affect numerical meaning produce a new snapshot. Historical results MUST NOT be overwritten silently.

If an error, leakage event, license problem, provenance defect, or invalid metric is discovered:

1. affected results are marked invalid or withdrawn;
2. the reason and scope are recorded;
3. downstream summaries are regenerated;
4. a corrected result receives a new immutable identifier; and
5. the public change log explains whether scientific conclusions changed.

## 15. Publication safety

Leaderboard publication includes metrics, resolved configurations, environment metadata, provenance, aggregate logs, checksums, and review evidence by default.

It does not automatically publish:

- restricted real data;
- synthetic records that may reveal training records;
- row-level attack outputs containing sensitive content;
- large checkpoints;
- secrets, local absolute paths, or private infrastructure identifiers; or
- artifacts without redistribution clearance.

Data and artifact publication is decided separately from result admission.

## 16. Related specifications

- [Evaluation Protocol](EVALUATION_PROTOCOL.md)
- [Metric Governance](METRIC_GOVERNANCE.md)
- [Result Specification](RESULT_SPECIFICATION.md)
- [Dataset Profile Specification](DATASET_PROFILE_SPEC.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [Repository Quality Standard](../QUALITY_STANDARD.md)
