# Evaluation Protocol

Chinese translation: [EVALUATION_PROTOCOL.zh-CN.md](EVALUATION_PROTOCOL.zh-CN.md)

- Status: design baseline
- Protocol family: Standardized Tabular Diffusion Benchmark
- Document version: 0.1.0
- Last updated: 2026-08-03

## 1. Purpose

This specification defines what the benchmark evaluates and how comparable evaluation results are produced. It covers statistical fidelity, downstream utility, data validity, empirical privacy risk, and efficiency.

This document is normative for evaluation semantics. [Repository Quality Standard](../QUALITY_STANDARD.md) remains the higher-level authority for repository quality, implementation identity, legal review, security, and release gates. The presence of this specification does not imply that every requirement is already implemented.

## 2. Normative language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY have the meanings defined in the Repository Quality Standard.

Requirements marked as pilot-frozen are mandatory after their threshold or implementation is fixed by a recorded pilot decision. Until then, results that depend on the unresolved item MUST NOT be admitted to a frozen official protocol.

## 3. Evaluation scope

### 3.1 Supported problem class

The initial protocol targets static, single-table synthetic data. The canonical public interface MAY accept CSV, Parquet, or an in-memory DataFrame, but every official run MUST resolve the input to the same versioned canonical table and schema.

The primary release environment is Linux with Python 3.11. Hardware-dependent results MUST additionally identify a compatible hardware profile.

### 3.2 Evaluation subjects

The protocol supports two evaluation subjects:

1. Model runs that train and sample through a registered model adapter.
2. External synthetic tables supplied without executing a model.

An external table MAY receive a complete diagnostic report. It MUST NOT enter an official model leaderboard unless its training inputs, model identity, implementation provenance, resolved configuration, seeds, and generation procedure are independently verifiable.

### 3.3 Data boundary

Every dataset MUST provide a frozen train, validation, and test split.

- Train is used to fit data preprocessing and the generative model.
- Validation is used only for permitted early stopping, checkpoint selection, and hyperparameter selection.
- Test is used only for final evaluation.

Validation is not merged back into the final training data. The official synthetic sample count equals the number of training rows unless a versioned protocol explicitly defines another sample-size profile.

Preprocessing parameters, encoders, imputers, scalers, discretizers, and category vocabularies that are learned from data MUST be fitted on train only. Test-dependent schema inference, tuning, metric configuration, or repair is prohibited.

### 3.4 Evaluation representation

Models MAY use method-specific internal representations. Evaluation MUST operate on the adapter-decoded table in the original semantic column space.

The original decoded synthetic output MUST be immutable. If an evaluator requires a repaired or normalized view to continue, that view MUST be stored separately, derived only with train-fitted rules, and marked with evaluation_repair_applied. Repair MUST NOT erase or alter the validity result for the original output.

## 4. Protocol identity and comparability

An official result MUST identify:

- protocol version;
- dataset identifier, version, view, split identifier, and checksums;
- model identifier, upstream version, adapter version, and configuration hash;
- comparison track;
- generation seeds;
- evaluator and metric versions;
- requested and actual synthetic row counts; and
- environment and hardware profile.

Results are comparable only when all protocol-defined compatibility fields match. Changes to a split, preprocessing contract, metric implementation, evaluator, sample count, tuning rule, resource regime, aggregation rule, or failure policy require a new compatible protocol version or an explicitly separate result group.

## 5. Execution lifecycle

The evaluation workflow consists of independently auditable phases:

1. prepare;
2. train;
3. sample;
4. validate;
5. evaluate;
6. aggregate; and
7. report.

Each phase MUST record its inputs, outputs, status, elapsed time, and integrity hashes. A later-phase failure MUST NOT force an earlier successful phase to be rerun when its content-addressed inputs remain unchanged. Cache reuse and resume behavior are governed by [Result Specification](RESULT_SPECIFICATION.md).

## 6. Statistical fidelity

Statistical fidelity evaluates whether synthetic data represents the distributional structure of the target population. It does not establish utility, validity, privacy, or causal correctness.

### 6.1 Shape

The initial source-parity candidate is the pinned SDMetrics `ColumnShapes` behavior. Each evaluable column receives an atomic shape score.

- Numerical columns use KSComplement, equal to one minus the two-sample Kolmogorov-Smirnov statistic.
- Categorical and Boolean columns use TVComplement, equal to one minus total variation distance.
- Datetime columns are converted to the implementation's numeric timestamp representation and use KSComplement.

The source implementation drops null values before KSComplement and TVComplement and returns an undefined value for an empty KS input. Official datasets contain no missing values after their declared preprocessing, but parity tests MUST still cover these source edge cases. Shape scores MUST be retained per column. The source-defined property score is the equal-column mean. Per-type means, worst-column tables, and support counts are benchmark-derived report views and MUST NOT be attributed to SDMetrics.

Calendar components such as month, weekday, and hour MAY be added only as separately identified benchmark diagnostics; they are not part of source-parity `ColumnShapes`.

### 6.2 Trend

The initial source-parity candidate is the pinned SDMetrics `ColumnPairTrends` behavior. In that implementation:

- numerical-numerical pairs use Pearson correlation similarity by default, equal to `1 - abs(correlation_real - correlation_synthetic) / 2`;
- categorical-categorical pairs use joint contingency-table similarity;
- numerical-categorical pairs use contingency similarity after the report discretizes the real and synthetic numerical columns independently with NumPy histogram edges; and
- Boolean columns are treated as categorical.

Contingency similarity is `1 - 0.5 × L1(P_real, P_synthetic)` over the union of observed joint cells. The source report evaluates all eligible pairs when both association thresholds are explicitly fixed to zero.

Spearman correlation is supported by the lower-level SDMetrics metric but is not the default `ColumnPairTrends` property behavior. Likewise, direct `ContingencySimilarity` can apply ten common bins fitted from real data, which is not the report's independent-bin mixed-pair behavior. Spearman, common train-fitted bins, or another scientifically preferred adaptation MUST use a distinct metric identifier and MUST NOT claim source parity with the report.

A wide-table pair-sampling policy MAY be introduced only as a deterministic, versioned benchmark adaptation with a recorded activation threshold and sampling seed.

Every pair result MUST be retained. Reports MUST include overall and pair-type summaries and identify the worst-performing pairs.

### 6.3 High-order structure

GReaT's source-defined discriminator experiment uses a tuned Random Forest, balanced real and synthetic evaluation samples, test accuracy, and a target of 0.5. This raw behavior is retained under a source-specific identifier such as `great_rf_discriminator_accuracy`; it is not an AUROC metric.

The initial benchmark high-order candidate is a classifier two-sample test, or C2ST, based on AUROC. It follows the same distinguishability principle but is a benchmark-derived definition rather than the GReaT formula.

The evaluator MUST:

- use balanced real and synthetic classes;
- use a train-fitted, versioned feature transformation;
- train a fixed, versioned discriminator;
- evaluate on data not used to fit the discriminator; and
- report the raw AUROC and uncertainty.

Let adjusted AUROC be the larger of AUROC and one minus AUROC. The fidelity complement is:

~~~text
c2st_fidelity = 2 × (1 - adjusted_AUROC)
~~~

It is one at chance-level distinguishability and zero at perfect distinguishability.

The raw AUROC, adjusted AUROC, and derived complement MUST use distinct fields. Only the derived value contributes to `high_order_score` after its discriminator and transformation are protocol-frozen. The GReaT accuracy diagnostic and the AUROC candidate MUST NOT be averaged together as if they were independent constructs.

Alaa et al. define integrated Alpha-Precision and Beta-Recall as:

~~~text
integrated_alpha_precision = 1 - 2 × integral_0^1 |P_alpha - alpha| d alpha
integrated_beta_recall     = 1 - 2 × integral_0^1 |R_beta - beta| d beta
~~~

The pinned authors' code approximates both integrals on a 30-point grid over `[0, 1]` using Euclidean distances in its supplied embedding. These metrics remain experimental until the mixed-table embedding, numerical integration rule, edge cases, and cross-dataset behavior pass source and protocol validation. DCR and Authenticity are not fidelity metrics in this benchmark.

### 6.4 Reference comparisons

The official fidelity comparison is synthetic versus held-out real test data. The report MUST also include:

- synthetic versus real train as an overfitting diagnostic; and
- a versioned real-versus-real reference that estimates finite-sample variation.

The real-versus-real construction, balancing, and resampling policy MUST be fixed before the protocol is frozen. It MUST NOT expose test information to the generator.

### 6.5 Fidelity aggregation

The dataset-level components are:

~~~text
shape_score
trend_score
high_order_score
~~~

When all three are computed, the official dataset fidelity score is:

~~~text
fidelity_score =
    (shape_score + trend_score + high_order_score) / 3
~~~

Atomic results, component scores, worst cases, uncertainty, and reference comparisons MUST remain visible. Experimental metrics do not affect the official fidelity score.

## 7. Downstream utility

Downstream utility evaluates whether synthetic data can support predictive relationships that generalize to held-out real data.

### 7.1 Evaluation arms

Every applicable task uses:

- TRTR: train the predictor on real train and test on real test;
- TSTR: train the same predictor on synthetic train and test on the same real test; and
- Dummy: train a task-appropriate naive predictor and test on real test.

The feature definition, task definition, evaluator family, evaluator hyperparameters, transformation rules, and test set MUST match across arms.

### 7.2 Local Utility

Each dataset declares one primary target in its dataset profile. Optional secondary targets are reported separately and do not change the primary leaderboard weight.

The primary task metrics are:

- Macro-F1 for binary and multiclass classification; and
- RMSE for regression.

Balanced Accuracy, ROC-AUC, PR-AUC, MAE, and R-squared are secondary metrics where mathematically applicable.

The evaluator suite represents at least a linear model, a random forest, and a gradient-boosted tree. Exact implementations and frozen hyperparameters are pilot-frozen. Evaluator selection MUST use train and permitted validation data only and MUST be independent of the synthetic-data method being scored.

This primary Macro-F1/RMSE panel is a benchmark contract. It MUST NOT be called an exact reproduction of GReaT, which uses classification accuracy and regression MSE with linear or logistic regression, decision tree, and random forest predictors, or TabStruct, whose classification utility uses Balanced Accuracy.

### 7.3 Global Utility

Global Utility follows TabStruct Equation 4 and rotates every evaluable non-identifier column as the prediction target:

- numerical targets are regression tasks;
- categorical and Boolean targets are classification tasks; and
- exclusions require an explicit dataset-profile reason.

For target `j`, let `D` be the synthetic training table and `D_ref` the real training reference. Both predictor arms are evaluated on the same held-out real test data. The source-defined target utility is:

~~~text
categorical target: utility_j(D) = balanced_accuracy_j(D) / balanced_accuracy_j(D_ref)
numerical target:   utility_j(D) = RMSE_j(D_ref) / RMSE_j(D)
global_utility(D) = mean_j utility_j(D)
~~~

Targets are weighted equally, and values above one are valid and MUST NOT be clipped. A zero or non-finite denominator is mathematically undefined unless a future metric version declares and validates a different rule.

Predictor configuration is part of the metric identity. TabStruct's Full-tuned profile uses nine tuned predictors, while its Tiny-default profile uses three untuned predictors and is empirically recommended as a lower-cost Global Utility option. The pinned TabEval snapshot implements a three-predictor form. These profiles MUST have distinct identifiers and MUST NOT be mixed within one leaderboard compatibility group.

Local Utility and Global Utility are separate official sub-leaderboards. They are not combined into a single Utility score in the initial protocol.

### 7.4 Benchmark-derived Local Utility retention

Raw Dummy, TSTR, and TRTR values MUST be retained.

The following baseline-adjusted Local Utility transformation is defined by this benchmark. It is not the GReaT or TabStruct utility formula.

For a metric where larger is better:

~~~text
retention = (TSTR - Dummy) / (TRTR - Dummy)
~~~

For an error where smaller is better:

~~~text
retention = (Dummy - TSTR) / (Dummy - TRTR)
~~~

The raw retention MAY be below zero or above one. If a clipped `[0, 1]` contribution is admitted after pilot review, it MUST use a distinct transformation identifier; the raw value MUST remain the primary auditable result. A clipped value MUST NOT be labeled as TabStruct utility.

If TRTR does not improve on Dummy by a pilot-frozen numerical tolerance, retention is mathematically undefined and MUST NOT be fabricated. The atomic task remains visible with its raw results and status.

### 7.5 Target-support failure

If synthetic data omits a real target class, the result state is insufficient_support with reason code insufficient_target_support. Computable predictors continue to be evaluated against all real test classes. If a predictor cannot train because the synthetic target is constant, the task receives the protocol-defined lowest aggregate utility contribution and remains visible; it is not dropped or awarded a favorable default.

## 8. Data validity

Data validity evaluates whether decoded synthetic records satisfy the declared table semantics.

PAFT motivates explicit domain-rule and functional-dependency checks, but the formulas and aggregation rules in this section are benchmark-native. They MUST be cited and versioned as this repository's contract rather than attributed to PAFT.

### 8.1 Structural gate

The evaluator MUST verify requested row count, required and extra columns, unique column names, readable serialization, and safe type convertibility.

Column reordering by name and demonstrably lossless conversion MAY be normalized. Missing columns, unexpected columns, unreadable output, incorrect row count, or lossy conversion are structural failures. A structural failure prevents the run from entering subsequent official metric computation.

### 8.2 Column validity

The versioned schema may define nullability, finiteness, integer constraints, domain bounds, category vocabularies, string formats, and datetime ranges.

For each column:

~~~text
valid_cell_rate =
    cells satisfying all hard column rules / total cells
~~~

The column_validity_score is the equal-column mean. Per-column rates and violation reasons MUST be retained.

### 8.3 Cross-column constraints

Only constraints supported by an authoritative data dictionary, dataset documentation, or recorded human review are hard constraints. Examples include temporal order, mutual exclusion, conditional domain rules, and total relationships.

Each constraint receives an equal-weight satisfaction rate. Their mean is constraint_validity_score. If no reviewed cross-column constraint applies, the component is not_applicable rather than a fabricated perfect score.

Observed train minima, maxima, rare combinations, and inferred dependencies are soft diagnostics unless independently established as domain rules.

### 8.4 Validity aggregation

If reviewed cross-column constraints apply:

~~~text
validity_score =
    0.5 × column_validity_score
    + 0.5 × constraint_validity_score
~~~

Otherwise, validity_score equals column_validity_score.

The fully_valid_row_rate is reported separately and does not enter the aggregate because its severity grows mechanically with table width.

Content violations do not automatically prevent other diagnostic dimensions from running. Any evaluator-only repair MUST remain explicit and must not improve or overwrite the original validity score.

## 9. Empirical privacy risk

Privacy evaluation measures evidence under declared threat models. It does not establish a formal privacy guarantee.

### 9.1 Threat-model families

The protocol distinguishes:

1. direct disclosure and copying;
2. membership inference; and
3. attribute inference.

The report MUST identify attacker knowledge, accessible artifacts, member and non-member sampling, distance or feature representation, attack model, and evaluation metric.

### 9.2 Exact collision

The benchmark reports:

- the fraction of synthetic rows exactly matching a train row;
- the number of distinct train rows matched;
- the maximum multiplicity of any matched train row; and
- synthetic internal duplicate rate as a separate diversity diagnostic.

Equality normalization MAY standardize lossless semantic representations, but MUST NOT use rounding or lossy coercion that creates collisions.

### 9.3 Calibrated DCR

The locked mixed-table distance implementation is the pinned SDMetrics `calculate_dcr` behavior. For every candidate-reference row pair, numerical and datetime distance is `min(abs(x - r) / reference_range, 1)`, categorical and Boolean distance is zero for a match and one for a mismatch, and the row distance is the equal-column mean. A zero-range numerical column contributes zero for equality and one otherwise. Both-null contributes zero and exactly-one-null contributes one. DCR is the exact minimum row distance over all reference rows.

Using real train as the reference dataset, the benchmark reports both:

~~~text
DCR(synthetic, train)
DCR(test, train)
~~~

The comparison of these two distributions follows GReaT's principle that synthetic-to-train DCR should be non-zero and resemble held-out-real-to-train DCR. The one-percent and five-percent reference thresholds are benchmark-derived low-tail diagnostics, not SDMetrics' `DCRBaselineProtection` score and not a formula stated by GReaT.

DCR is distributional, not monotonic. Larger values MUST NOT automatically be labeled safer, because extreme distance may indicate unusable synthetic data. Exact nearest-neighbor computation is required for official results unless a separately validated approximation is declared.

### 9.4 Authenticity

For each synthetic sample `g_j`, Alaa et al. define `d_g,j` as its distance to the nearest real training sample `r_i*`, and `d_r,i*` as the distance from `r_i*` to its nearest other real training sample. The paper's classifier marks the sample authentic when `d_g,j > d_r,i*`; authenticity is the mean of these sample indicators.

The pinned authors' repository does not execute that indexing literally: it finds the nearest synthetic sample for each real row and then indexes the real-to-real distance vector with synthetic indices. This can differ from the paper and can fail when sample counts differ. Therefore `authenticity_alaa2022_paper` and `authenticity_alaa2022_repo` are separate variants. Neither may enter Official Results until the discrepancy is adjudicated, its reproduction target is declared, and parity tests pass.

Any mixed-table representation is a benchmark adaptation and MUST be train-fitted, versioned, and identified separately from the paper's embedding-space method. Authenticity is a copying/generalization diagnostic, not a privacy proof.

### 9.5 Membership and attribute inference

At least one source-validated membership-inference attack is required before the official privacy-risk suite is frozen. It reports attack AUROC, attack advantage, low-false-positive-rate behavior, and uncertainty.

Attribute inference runs only when a dataset profile declares suitable sensitive attributes and quasi-identifiers. It MUST compare against an attack baseline that does not use the synthetic data.

Unsupported threat models are not_applicable. Weak sample support is insufficient_support. Neither state is zero risk.

### 9.6 Privacy presentation

There is no unified Privacy Score and no privacy champion. Results are presented with Fidelity and Utility as quality-risk trade-offs and Pareto views.

A differentially private model MUST additionally report epsilon, delta, accounting method, clipping, noise, and composition assumptions. Empirical attacks do not replace formal differential-privacy accounting.

The pinned SynthCity `DeltaPresence` implementation fits KMeans with `k` in `{2, 5, 10, 15}` on real non-sensitive features, skips infeasible `k`, ignores real clusters absent from synthetic data, computes `real_count / (synthetic_count + 1e-8)` for shared clusters, and returns the maximum. This raw value is unbounded and is not literally a probability. It remains excluded from official scoring until its threat model, direction, empty-list behavior, and scientific suitability are resolved.

## 10. Efficiency

Efficiency is measured only within a compatible hardware profile.

The timing, throughput, RAM, and VRAM definitions below are benchmark-native operational measurements. They are not formulas taken from the six reviewed metric papers.

### 10.1 Timed phases

The benchmark separately records:

- data preparation;
- model fitting;
- checkpoint save and load;
- adapter preprocessing and postprocessing;
- cold-start generation;
- warm generation; and
- end-to-end execution.

Dependency installation and dataset download are excluded from timed execution. Model-required data transformation is included in end-to-end time and also reported separately. GPU timing MUST synchronize the device at timing boundaries.

### 10.2 Resources and throughput

Every official run records wall-clock time, requested and actual epochs, early stopping, peak RAM, peak VRAM where applicable, checkpoint size, requested and actual rows, rows per second, timeout, out-of-memory status, and numerical failure status.

Warm generation is repeated three times and summarized by the median. The primary workload generates the same number of rows as train. Optional fixed-size scaling profiles MAY be published separately.

### 10.3 Hardware profiles

An official hardware profile fixes operating system, Python, CPU model, CPU thread limit, RAM, GPU model and count, CUDA stack, and material runtime settings.

Results from incompatible hardware profiles MUST NOT be mixed. CPU-only, official single-GPU, and community hardware results are separately labeled. The benchmark does not convert observed time between devices using theoretical FLOPS.

### 10.4 Efficiency presentation

Training Time, Sampling Throughput, and Peak Resource Usage are separate sub-leaderboards. No unified Efficiency Score is produced.

Native comparisons report the final official training cost. Standardized Tuning comparisons additionally report all search trials, failures, total compute time, and final retraining cost. Timeout and out-of-memory outcomes remain visible.

## 11. Statistical protocol

The initial protocol uses one frozen official data split and five generation seeds. Evaluator randomness is frozen and recorded. A later split-robustness study is a separate protocol and MUST NOT be silently merged with the initial leaderboard.

Aggregation order is:

1. retain atomic metric values;
2. aggregate evaluator repetitions or targets within a run as specified;
3. aggregate the five generation seeds within a dataset; and
4. macro-average dataset summaries with equal dataset weight.

Datasets are never weighted by row count. Confidence intervals use a versioned hierarchical or paired bootstrap procedure. Rankings use unrounded values. Practically and statistically indistinguishable results receive a tie annotation under pilot-frozen equivalence tolerances.

## 12. Result states and denominator integrity

Every metric invocation returns exactly one state:

- computed;
- mathematically_undefined;
- insufficient_support;
- not_applicable;
- implementation_failure; or
- resource_failure.

A non-computed result has no numeric value and MUST include a reason and applicable counts. Bare NaN, silent row dropping, and favorable fallback scores are prohibited.

Failed, skipped, inapplicable, and unsupported results MUST remain distinguishable in aggregation denominators. Complete official leaderboards require the coverage defined in [Leaderboard Policy](LEADERBOARD_POLICY.md).

## 13. Pilot-frozen items

The following details require empirical pilot evidence before the first official protocol is frozen:

- exact Local and Global Utility predictor implementations and hyperparameters;
- real-versus-real fidelity resampling;
- mixed-table C2ST transformation and discriminator;
- source-faithful versus adapted Column Pair Trends discretization;
- Alpha-Precision and Beta-Recall admission;
- the paper-versus-repository Authenticity discrepancy;
- membership-inference attack selection;
- numerical tolerances for utility normalization and statistical ties;
- validity admission thresholds;
- CPU, GPU, time, memory, and tuning budgets; and
- optional scale profiles.

Every decision MUST be recorded with the tested alternatives, representative datasets, representative models, evidence, and resulting protocol-version effect.

## 14. Related specifications

- [Leaderboard Policy](LEADERBOARD_POLICY.md)
- [Metric Governance](METRIC_GOVERNANCE.md)
- [Metric Source Review](METRIC_SOURCE_REVIEW.md)
- [Result Specification](RESULT_SPECIFICATION.md)
- [Dataset Profile Specification](DATASET_PROFILE_SPEC.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
- [Repository Quality Standard](../QUALITY_STANDARD.md)
