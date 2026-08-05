# Metric Source Review

Chinese translation: [指标来源审阅](METRIC_SOURCE_REVIEW.zh-CN.md)

## 1. Status and purpose

Review date: 2026-08-03

Review status: definition reviewed; implementation parity not yet certified

This record reconciles the evaluation definitions in the reviewed papers with the executable behavior of the pinned upstream source trees. It prevents a paper formula, upstream implementation, and benchmark-derived transformation from being presented as the same metric.

This review establishes the implementation target for future work. A metric still requires an implementation, golden fixtures, direct source-parity tests, protocol freeze, and release approval before it can affect Official Results.

## 2. Reviewed evidence

### 2.1 Papers

| Identifier | Relevant evidence |
|---|---|
| [P01 GReaT](../../research_inputs/evaluation/papers/P01_GReaT_ICLR2023.pdf) | TSTR efficacy, Random-Forest discriminator, and DCR in Sections 4.1-4.2 and pages 7-9 |
| [P02 PAFT](../../research_inputs/evaluation/papers/P02_PAFT_Why_LLMs_Are_Bad_2025.pdf) | SDMetrics Shape/Trend usage and domain-rule violations |
| [P03 GraDe](../../research_inputs/evaluation/papers/P03_GraDe_2025.pdf) | TSTR, correlation error, and a directionally inconsistent DCR interpretation |
| [P04 TabStruct](../../research_inputs/evaluation/papers/P04_TabStruct_ICLR2026.pdf) | Utility Equation 4, Global Utility, predictor profiles, experimental setup, and metric comparisons |
| [P05 Alaa et al.](../../research_inputs/evaluation/papers/P05_Alaa_Faithful_Synthetic_Data_ICML2022.pdf) | Alpha-Precision, Beta-Recall, integrated scores, and the Authenticity classifier |
| [P06 SynthCity](../../research_inputs/evaluation/papers/P06_Synthcity_2023.pdf) | Framework context for SynthCity evaluators; executable details come from the pinned source |

### 2.2 Pinned executable sources

The immutable revisions are recorded in [source_lock.json](../../research_inputs/evaluation/provenance/source_lock.json). The relevant snapshots are:

| Source | Revision | Review target |
|---|---|---|
| SDMetrics | `ba8842f2ba04ce914f698cc1cf746ca12338ab0e` | Column Shapes, Column Pair Trends, and mixed-table DCR |
| TabStruct | `501f48942f890c92a796edf236e670cdc270ad9f` | Paper pipeline and evaluation configuration |
| TabEval | `dba19a4ee7aa391621cbeb464609285fd515dece` | `UtilityPerFeature` executable behavior |
| evaluating-generative-models | `093910e487d07959db7d87b54698da60aaeb50c0` | Authors' Alpha-Precision, Beta-Recall, and Authenticity code |
| SynthCity | `23f322fe381326ed01c41b13d469a06e38cce545` | `DeltaPresence` executable behavior |
| GReaT / be_great | `7d30fcd4b811b49173e54c04e3d68a6c5cc2d483` | GReaT-related evaluation utilities |

The package integrity and license checks are recorded in [Package QA](../../research_inputs/evaluation/provenance/PACKAGE_QA.md).

## 3. Source findings and benchmark decisions

### 3.1 Column Shapes

The pinned SDMetrics `ColumnShapes` property uses:

~~~text
numerical or datetime: KSComplement = 1 - two-sample KS statistic
categorical or Boolean: TVComplement = 1 - 0.5 × sum_k |p_real(k) - p_synthetic(k)|
property score: equal mean over computed columns
~~~

The implementation drops nulls before the atomic metrics; an empty KS input returns an undefined value. Datetime values are converted to the implementation's numeric representation. Per-type summaries and worst-column lists are benchmark report views, not source-defined metrics.

Decision: use the exact pinned behavior as the initial Shape source-parity target.

### 3.2 Column Pair Trends

The pinned `ColumnPairTrends` property uses Pearson correlation by default for continuous-continuous pairs:

~~~text
correlation_similarity = 1 - |correlation_real - correlation_synthetic| / 2
~~~

Categorical-categorical and mixed pairs use contingency similarity:

~~~text
contingency_similarity = 1 - 0.5 × L1(P_real_joint, P_synthetic_joint)
~~~

The report preprocesses a mixed pair by deriving histogram edges independently for the real and synthetic continuous columns. This differs from a direct lower-level `ContingencySimilarity` call, which can derive ten bins from real data and apply the same edges to both datasets. Spearman is supported by the lower-level correlation metric but is not the report default.

At the pinned revision, `QualityReport` also applies an absolute real Pearson-correlation threshold of `0.5`, a real Cramér's-V association threshold of `0.3`, and `num_rows_subsample=50000`. A pair at or below its applicable real-data threshold returns a non-finite score and does not enter the property mean. P2 preserves such a pair as a denominator-visible `not_applicable` Atomic Result with reason `below_source_threshold`; it does not silently omit it.

The pinned source delegates subsampling above 50,000 rows to pandas without passing `random_state`. P2 controls that exact source path with the single recorded evaluator seed, serializes source calls, and restores the caller's NumPy random state.

Decision: the report behavior is the initial source-parity target. Spearman and common train-fitted bins are distinct experimental variants, even if later review finds them scientifically preferable.

### 3.3 Discriminator and C2ST

GReaT trains a tuned Random-Forest discriminator, evaluates accuracy on equal real and generated shares, and treats 50 percent as ideal indistinguishability. It does not define an AUROC complement.

The benchmark C2ST candidate records AUROC and defines:

~~~text
adjusted_AUROC = max(AUROC, 1 - AUROC)
c2st_fidelity = 2 × (1 - adjusted_AUROC)
~~~

Decision: retain GReaT discriminator accuracy as a source-specific diagnostic. Treat the AUROC complement as benchmark-derived and freeze it under a different identifier. Do not average both into one high-order component.

### 3.4 Integrated Alpha-Precision and Beta-Recall

Alaa et al. define:

~~~text
integrated_alpha_precision = 1 - 2 × integral_0^1 |P_alpha - alpha| d alpha
integrated_beta_recall     = 1 - 2 × integral_0^1 |R_beta - beta| d beta
~~~

The authors' code evaluates a 30-point grid over `[0, 1]` and uses a discrete sum in a Euclidean embedding. Its support construction and embedding are essential parts of the method.

Decision: keep both metrics experimental until a mixed-table embedding and its fit boundary reproduce the intended authors' method and the discrete integration passes parity fixtures.

### 3.5 TSTR and Local Utility

GReaT evaluates predictors trained on synthetic data and tested on held-out real data. Its main panel uses linear or logistic regression, decision tree, and random forest; classification reports accuracy and regression reports MSE, averaged over five seeds.

This benchmark's Local Utility panel instead selects Macro-F1 for classification and RMSE for regression, with additional model families and secondary endpoints. Its baseline-adjusted transformation is:

~~~text
higher-is-better: (TSTR - Dummy) / (TRTR - Dummy)
lower-is-better:  (Dummy - TSTR) / (Dummy - TRTR)
~~~

Decision: raw TSTR/TRTR results remain source-comparable when their evaluator profile matches a source. The baseline-adjusted retention is benchmark-derived, must retain its raw terms, and must not be called GReaT or TabStruct utility.

### 3.6 TabStruct Local and Global Utility

TabStruct Equation 4 uses Balanced Accuracy for categorical targets and RMSE for numerical targets. With `D` as the synthetic training table and `D_ref` as the real reference training table:

~~~text
categorical target: Utility_j(D) = BalancedAccuracy_j(D) / BalancedAccuracy_j(D_ref)
numerical target:   Utility_j(D) = RMSE_j(D_ref) / RMSE_j(D)
GlobalUtility(D) = mean_j Utility_j(D)
~~~

All targets are equally weighted. Values above one indicate performance on par with or better than the real-reference arm and must not be clipped.

The paper's Full-tuned profile ensembles nine tuned predictors: Logistic Regression, KNN, MLP, Random Forest, Extra Trees, LightGBM, CatBoost, XGBoost, and TabPFN. Its Tiny-default profile uses three untuned predictors and is supported by the paper as a lower-cost Global Utility profile. The pinned TabEval `UtilityPerFeature` snapshot implements a three-predictor configuration and assigns a favorable `[1]` classification value when a synthetic target is constant.

Decision: predictor profiles have distinct metric identities. The favorable constant-target fallback is not accepted by this benchmark; it becomes an explicit support failure and therefore prevents exact code-parity claims for that edge case. The TabStruct formula remains the Global Utility target. A bounded Linux/Python 3.11 pilot directly executed the locked TabEval source and real XGB/KNN/TabPFN models and matched adapter classification/regression aggregates exactly. Because upstream published no dependency lock, that runtime remains explicitly benchmark-approved rather than upstream-official; dataset-scale admission is still pending.

### 3.7 DCR

GReaT defines nearest-training-record distance using an L1-style mixed distance and interprets a good result as non-zero synthetic DCR whose distribution resembles held-out-real-to-train DCR. GraDe's formula and prose imply ordinary nearest distance, but its reported direction and interpretation contradict that formula; the contradictory direction is rejected.

The pinned SDMetrics mixed distance computes, for each column:

~~~text
numerical or datetime: min(|x - r| / reference_range, 1)
categorical or Boolean: 0 if equal, otherwise 1
row distance: equal-column mean
DCR(x): exact minimum row distance over the reference rows
~~~

For a zero-range numerical column, equality contributes zero and inequality one. Both-null contributes zero; exactly-one-null contributes one.

Decision: use the exact SDMetrics distance as the implementation identity, with real train as the reference. Compare the complete synthetic-to-train and held-out-real-to-train distributions. Low-tail reference thresholds are benchmark-derived diagnostics. Do not use or relabel SDMetrics `DCRBaselineProtection`, and do not rank raw DCR monotonically.

### 3.8 Authenticity

For each synthetic sample `g_j`, the Alaa paper finds its nearest real sample `r_i*`, compares `d_g,j = d(g_j, r_i*)` with the nearest-other-real distance `d_r,i*`, and declares `g_j` authentic when `d_g,j > d_r,i*`.

The pinned authors' code reverses the nearest-neighbor query: it finds the nearest synthetic sample for each real row and indexes a real-to-real distance vector using the resulting synthetic indices. This is not the same sample-level computation and may fail when real and synthetic sample counts differ.

Decision: register `authenticity_alaa2022_paper` and `authenticity_alaa2022_repo` separately. Neither is admitted to Official Results until the discrepancy is adjudicated. Any mixed-table encoder is another benchmark adaptation and must be separately identified.

### 3.9 Delta Presence

The pinned SynthCity implementation removes declared sensitive features, fits KMeans on real data for `k` in `{2, 5, 10, 15}`, predicts synthetic clusters, skips real clusters absent from synthetic data, computes `real_count / (synthetic_count + 1e-8)` for shared clusters, and returns the maximum ratio. The returned value is unbounded and therefore is not literally a probability despite the source docstring.

Decision: preserve the exact raw behavior only as a named diagnostic. Exclude it from official scoring until the threat model, direction, missing-cluster treatment, and empty-result behavior receive a scientific decision.

### 3.10 Validity and Efficiency

PAFT supports the scientific need for rule-violation reporting, but it does not define this benchmark's column-validity, cross-column-constraint, or aggregation formulas. Likewise, the reviewed papers do not define this benchmark's timing phases, throughput, RAM, or VRAM contracts.

Decision: register Validity and Efficiency metrics as benchmark-native and validate their formulas directly, without claiming paper parity.

## 4. Admission matrix after review

| Metric or family | Current definition decision | Earliest permitted publication role |
|---|---|---|
| SDMetrics Column Shapes | Exact pinned behavior | Official candidate after parity and release gates |
| SDMetrics Column Pair Trends | Exact pinned report behavior | Official candidate after parity and release gates |
| C2ST AUROC complement | Benchmark-derived | Diagnostic until pilot freeze and validation |
| Integrated Alpha-Precision/Beta-Recall | Source formula; mixed-table adaptation unresolved | Experimental diagnostic |
| Baseline-adjusted Local Utility retention | Benchmark-derived | Diagnostic until pilot freeze and validation |
| TabStruct Global Utility | Paper Equation 4; predictor profile remains part of identity | Official candidate after profile freeze and parity validation |
| SDMetrics DCR with held-out calibration | Source distance plus benchmark-derived calibration | Privacy-risk diagnostic candidate |
| Alaa Authenticity | Paper and repository conflict | Excluded from Official Results |
| SynthCity Delta Presence | Exact raw code behavior has unresolved semantics | Excluded from official scoring |
| Validity and Efficiency | Benchmark-native | Official candidates after direct validation and release gates |

## 5. Required parity fixtures

Before a source-based metric becomes `source-parity-validated`, tests MUST cover at least:

- hand-computable normal cases and exact upstream outputs;
- constant, empty, single-row, missing-value, unseen-category, and zero-range cases where applicable;
- numerical, categorical, Boolean, datetime, and mixed pairs supported by the source;
- report-level versus atomic-metric aggregation and preprocessing;
- real and synthetic sample counts that differ;
- raw outputs before benchmark normalization or clipping;
- deterministic seeds, tolerances, warnings, and result states; and
- the pinned source revision and resolved parameter set.

For stochastic predictors and discriminators, parity uses predeclared statistical tolerances or a deterministic small fixture, not post-hoc tolerance selection.

## 6. Release-blocking source questions

The following issues remain intentionally unresolved and block the affected metric from Official Results:

- whether official Trend keeps source-faithful independent mixed-pair bins or adopts a separately named common train-fitted-bin variant;
- the exact C2ST discriminator, preprocessing, split, and uncertainty procedure;
- the mixed-table embedding for Alpha-Precision and Beta-Recall;
- the Local Utility predictor profile and any clipped retention contribution;
- the Global Utility predictor profile;
- the Authenticity paper-versus-repository discrepancy;
- the membership-inference implementation; and
- the scientific suitability of SynthCity Delta Presence.

## 7. Related specifications

- [Evaluation Protocol](EVALUATION_PROTOCOL.md)
- [Metric Governance](METRIC_GOVERNANCE.md)
- [Leaderboard Policy](LEADERBOARD_POLICY.md)
- [Dataset Profile Specification](DATASET_PROFILE_SPEC.md)
- [Result Specification](RESULT_SPECIFICATION.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)
