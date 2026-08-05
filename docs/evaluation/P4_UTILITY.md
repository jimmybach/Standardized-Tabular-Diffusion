# P4 Local and Global Utility

## Status and claim boundary

P4 is implemented and its bounded diagnostic gates passed on [Linux/Python 3.11](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31053624769), with [machine-readable evidence](../evidence/evaluation/p4-utility-run-31053624769.json) retained at SHA-256 `bb2b5f3d48647122b1036f8ce010eeecee948a0dfb4a0bfc247ab7100439cd59`. It remains a **diagnostic pilot**: it is not protocol-frozen, release-supported, or eligible for Official Results.

The implementation establishes the complete auditable path from three immutable decoded tables to Local and Global Utility Atomic Results and a finalized result bundle. It does not claim executable parity with the full TabEval predictor runtime. That claim requires a separate Linux/Python 3.11 run with the pinned AutoGluon, XGBoost, KNN, and TabPFN stack, reviewed runtime budgets, and retained evidence.

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

The pinned low-cost predictor identity is TabEval `UtilityPerFeature` at revision `dba19a4ee7aa391621cbeb464609285fd515dece`, timestamp `2025-08-09`, configured with XGB, KNN, and TabPFN through AutoGluon. P4 records the exact trained model names and per-model scores returned by the backend. It requires matching TRTR and TSTR predictor sets for a target. No fallback model or reduced unrecorded profile is permitted.

The pinned TabPFN implementation supports at most ten classes. For a higher-cardinality categorical target, AutoGluon may source-faithfully omit TabPFN; P4 records `source-predictor-set-reduced` and accepts the ratio only when both arms expose the same predictor set. This behavior remains part of the pending source-runtime pilot.

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

## Remaining P4 exit work

Before P4 can advance beyond diagnostic use:

1. run the exact optional Global predictor stack on Linux/Python 3.11;
2. retain source-runtime, dependency, model-set, numerical, stability, and resource evidence;
3. review Adult and Sick pilot behavior, including high-cardinality targets;
4. freeze predictor versions, parameters, seed policy, and runtime budgets; and
5. issue a separate protocol-freeze and Official Results admission decision.
