# P4 Local and Global Utility

## Status and claim boundary

P4 is implemented and its bounded diagnostic gates passed on [Linux/Python 3.11](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31053624769), with [machine-readable engineering evidence](../evidence/evaluation/p4-utility-run-31053624769.json) retained at SHA-256 `bb2b5f3d48647122b1036f8ce010eeecee948a0dfb4a0bfc247ab7100439cd59`. The separate real Global source-runtime pilot passed in [run 31057073762](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31057073762); its [retained evidence](../evidence/evaluation/p4-global-source-runtime-run-31057073762.json) has SHA-256 `1ca205c3fbad6c6e80cd275330dae00edda5dfcad7efe229f4fcb285b9d63596`. P4 remains a **diagnostic pilot**: it is not protocol-frozen, release-supported, or eligible for Official Results.

The implementation establishes the complete auditable path from three immutable decoded tables to Local and Global Utility Atomic Results and a finalized result bundle. The bounded pilot now establishes exact aggregate parity between the locked TabEval source and the adapter for one classification target and one regression target with real XGB, KNN, and TabPFN training. It does not establish full-dataset, multi-seed stability or Official Results admission.

## Required inputs and leakage boundary

P4 requires three checksum-bound tables:

- real train: the only real table allowed to fit feature transformations and TRTR predictors;
- synthetic train: the only table allowed to fit TSTR predictors; and
- held-out real test: evaluation only, never a fit input.

All three tables must pass the strict canonical model-view gate. The gate rejects missing model-input values, non-finite numerical values, lossy logical types, schema differences, and an unexpected synthetic row count. Generated values are never repaired.

The Evaluation Request stores the real-train, real-test, and synthetic checksums separately. Bundle validation verifies these identities and requires every Local run to attest the same test view.

## Local Utility

Each reviewed dataset declares one primary task. Adult predicts `income`; Sick predicts `Class`. The pilot profile uses:

- classification primary metric: Macro-F1;
- regression primary metric: RMSE;
- classification secondary metrics: Balanced Accuracy, ROC-AUC, and PR-AUC when defined;
- regression secondary metrics: MAE and R-squared when defined; and
- three evaluator families: linear, random forest, and histogram gradient boosting.

The exact package classes and parameters are stored in `p4-utility-pilot-v1.json`. Classification uses Logistic Regression, Random Forest, and Histogram Gradient Boosting. Regression uses Ridge, Random Forest, and Histogram Gradient Boosting. Randomized estimators receive the Evaluation Request seed and use one worker where supported.

One-hot category state and numerical scaling state are fitted once on real-train features. The same frozen transformation is applied to real train, synthetic train, and real test. The target and held-out test values never fit the transformation.

For every evaluator and seed, P4 retains the primary and applicable secondary values for:

- Dummy: a most-frequent classifier or mean regressor fitted on real train;
- TRTR: predictor fitted on real train and evaluated on real test; and
- TSTR: the same predictor configuration fitted on synthetic train and evaluated on the same real test.

The separately identified, benchmark-derived Local Utility retention is:

~~~text
higher is better: (TSTR - Dummy) / (TRTR - Dummy)
lower is better:  (Dummy - TSTR) / (Dummy - TRTR)
~~~

Retention is not clipped. If TRTR fails to improve on Dummy by more than `1e-12`, retention is `mathematically_undefined`. The strict summary is null unless every requested evaluator/seed retention computes.

## Global Utility

Global Utility rotates every included canonical model-view column as the target. Dataset Profiles must account for every model-view column: include it, or record a stable exclusion reason. Identifiers, ignored fields, and audit-only fields cannot be included. Datetime and string targets require an explicit exclusion until their target policies are frozen.

The reviewed source formula is TabStruct Equation 4:

~~~text
categorical target: balanced_accuracy(TSTR) / balanced_accuracy(TRTR)
numerical target:   RMSE(TRTR) / RMSE(TSTR)
global utility:     equal mean over targets, then equal mean over seeds
~~~

Ratios above one are valid and are never clipped. A zero or non-finite denominator is explicit `mathematically_undefined`. The published diagnostic `global_utility` is null if any requested target/seed ratio is unavailable; computed targets are never silently reweighted.

The pinned low-cost predictor identity is TabEval `UtilityPerFeature` at revision `dba19a4ee7aa391621cbeb464609285fd515dece`, timestamp `2025-08-09`, configured with XGB, KNN, and TabPFN through AutoGluon. The LF-normalized source and Apache-2.0 license are checksum-locked. Upstream did not declare AutoGluon and left XGBoost and TabPFN unbounded, so no reproducible upstream-official environment exists. The pilot therefore labels its Linux/Python 3.11 CPU tuple—AutoGluon 1.4.0, `xgboost-cpu` 3.0.3, TabPFN 2.1.2, and PyTorch 2.3.0+cpu—as a benchmark-approved reconstruction rather than an upstream-official lock.

In the retained run, both source and adapter trained `CustomTabPFNModel`, `KNeighbors`, and `XGBoost`. Their aggregate Balanced Accuracy was exactly `0.5416666666666666`; their aggregate RMSE was exactly `8.979373060535432`; both absolute source/adapter differences were zero under the declared `1e-8` gate. P4 records exact trained model names and per-model scores and requires matching TRTR/TSTR predictor sets for a target. No fallback model or unrecorded reduced profile is permitted.

The pinned TabPFN implementation supports at most ten classes. The pilot directly executed the locked source guard and confirmed that an eleven-class target is rejected before TabPFN model fitting. AutoGluon may then source-faithfully omit the failed family; P4 records `source-predictor-set-reduced` and accepts a ratio only when both arms expose the same predictor set. End-to-end high-cardinality behavior on reviewed dataset targets remains an admission item.

TabEval's favorable value of one for a constant synthetic target is rejected. Missing synthetic target classes produce `insufficient_support`, remain visible, and prevent a strict Global Utility summary.

## Atomic Results and bundle evidence

P4 writes:

- one raw Local Atomic Result per metric, arm, evaluator, and seed;
- one Local retention Atomic Result per evaluator and seed;
- one raw Global Atomic Result per target, arm, and seed;
- one Global target-ratio Atomic Result per target and seed;
- `artifacts/utility-details.json` with raw-arm mappings, target support, exact predictor sets, per-predictor scores, and the test boundary;
- `metrics.parquet`, `summary.json`, `metadata.json`, stage records, artifact inventory, and final checksums.

Raw arms have zero aggregation weight. Local retention uses equal evaluator/seed weights. Global target ratios use equal target/seed weights. Bundle finalization independently reconstructs both formulas from Atomic Results and rejects changed summaries, missing raw arms, unequal weights, mismatched predictor sets, altered denominators, or a missing real-test identity.

## Failure states

- `insufficient_support`: synthetic target classes are missing, or the real train cannot support the real-test labels;
- `mathematically_undefined`: a weak Local denominator, zero Global denominator, or undefined secondary metric;
- `implementation_failure`: a declared predictor cannot satisfy its result contract;
- `resource_failure`: the authoritative optional Global backend, weights, memory, or time budget is unavailable;
- `not_applicable`: reserved for an explicit reviewed applicability decision.

Failures are never dropped. A bundle may finalize as `partial`, but partial summaries are not leaderboard scores.

## Command

Install the frozen Local Utility, contract, table, and bundle dependencies with:

~~~bash
python -m pip install -e ".[utility]"
~~~

This extra deliberately excludes the pending AutoGluon/XGBoost/TabPFN Global source runtime. Without that separately reviewed environment, the command still evaluates Local Utility and records explicit Global `resource_failure` states; it does not substitute another predictor.

~~~bash
std-tabular-diffusion evaluate-table \
  --protocol p4-utility \
  --reference real_train.csv \
  --real-test real_test.csv \
  --synthetic synthetic_train.csv \
  --dataset-profile configs/datasets/adult-uci-2-v1.json \
  --output artifacts/p4/adult/run-001
~~~

P4 defaults to evaluator seeds `0,1,2,3,4`. A diagnostic run may provide `--evaluator-seeds 23` or another comma-separated list. A different seed set is recorded and is not automatically leaderboard-compatible.

## Dataset-scale admission protocol

`p4-dataset-scale-admission-pilot@0.1.1` is the preregistered, non-official admission protocol for the real Global Utility runtime. It runs on Linux x86-64, Python 3.11, and CPU with the exact dependency and checkpoint identities used by the bounded source-runtime pilot. The first execution established that TabPFN 2.1.2 applies its official 1,000-row CPU guard. Version `0.1.1` therefore records and requires the official `TABPFN_ALLOW_CPU_LARGE_DATASET=1` opt-in and retains AutoGluon per-model failure metadata. It does not change the predictor panel, targets, seeds, surrogate, or thresholds.

The schedule contains 67 unique target/seed tasks and 134 TRTR/TSTR arms:

- seed-zero coverage of all 15 reviewed Adult targets at the official 32,561/16,281 split;
- seed-zero coverage of all 28 reviewed non-constant Sick targets at the official 2,800/972 split; and
- seeds zero through four for one binary, one high-cardinality or multiclass, and one numerical sentinel per dataset: Adult `income`, `native-country`, and `fnlwgt`; Sick `class`, `referral-source`, and `tsh`.

The TSTR input is a deterministic full-row permutation of real train. It preserves every row and every column's support, exercises both evaluator arms, and cannot be published as generator-quality evidence. Targets with more than ten real-train classes may omit TabPFN exactly as the locked source does; XGB and KNN remain mandatory, and both arms must expose the same trained model set.

The preregistered gates require every scheduled task exactly once, every applicable predictor family, an absolute identity-ratio deviation no greater than `0.05`, a five-seed ratio range no greater than `0.05`, no arm above 600 observed seconds, and no observed Python process-tree peak above 14 GiB. Shards write failure-first JSON after every target. The finalizer rejects missing or duplicate shards, changed commits or manifests, incomplete target coverage, model-set drift, resource failures, and threshold failures before emitting one retained result.

TabPFN's version-matched official documentation recommends GPU execution and states that only datasets of approximately 1,000 rows or fewer are feasible on CPU; it describes the large-dataset CPU opt-in as very slow. The admission run therefore measures a deliberately strict source-faithful CPU envelope rather than assuming it will pass. See the [TabPFN v2.1.1 documentation](https://github.com/PriorLabs/TabPFN/tree/v2.1.1#-quick-start); PyPI 2.1.2 contains no source change from that release according to the upstream changelog.

## Remaining P4 exit work

Before P4 can advance beyond diagnostic use:

1. run representative Adult and Sick Global pilots across their reviewed target sets;
2. measure multi-seed stability, wall time, peak memory, and target-level failure behavior;
3. validate full AutoGluon omission and equal-arm handling for reviewed high-cardinality targets;
4. freeze predictor versions, parameters, checkpoint identities, seed policy, and runtime budgets; and
5. issue a separate protocol-freeze and Official Results admission decision.
