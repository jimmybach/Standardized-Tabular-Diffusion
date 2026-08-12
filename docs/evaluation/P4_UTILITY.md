# P4 Local and Global Utility

## Status and claim boundary

P4 has one result-producing implementation: `p4-utility@0.5.0`, bound to evaluator profile `p4-utility-stable@0.2.0`. It is implemented but remains diagnostic. It is not yet protocol-frozen, release-supported, or eligible for Official Results.

The exact TabEval source at revision `dba19a4ee7aa391621cbeb464609285fd515dece` remains checksum-locked for provenance and internal source-parity tests. It is not a second CLI option, metric profile, or leaderboard implementation. User-facing evaluation always uses the stable adapter described below.

## Required inputs and leakage boundary

P4 requires three checksum-bound canonical model-view tables:

- real train, used to fit transformations, Dummy models, and TRTR predictors;
- synthetic train, used to fit TSTR predictors; and
- held-out real test, used only for scoring and never supplied to a fit operation.

The structural gate rejects schema drift, missing model inputs, non-finite numerical values, lossy logical types, and unexpected synthetic row counts. Generated values are not repaired. Bundle finalization verifies the three table identities separately and records `real_test_used_for_fit: false`.

## Local Utility

Each dataset declares one reviewed primary predictive task. Adult predicts `income`; Sick predicts `Class`. Classification uses Macro-F1 as the primary metric and Balanced Accuracy, ROC-AUC, and PR-AUC when defined. Regression uses RMSE as the primary metric and MAE and R-squared when defined.

The frozen local panel contains three families:

- Logistic Regression or Ridge;
- Random Forest; and
- Histogram Gradient Boosting.

The exact official scikit-learn classes and parameters are stored in `p4-utility-stable-v1.json`. One-hot encoding and numerical scaling are fitted only on real-train features and then frozen for real train, synthetic train, and real test.

For every evaluator seed, P4 retains Dummy, TRTR, and TSTR raw results. The benchmark-derived retention is:

~~~text
higher is better: (TSTR - Dummy) / (TRTR - Dummy)
lower is better:  (Dummy - TSTR) / (Dummy - TRTR)
~~~

Retention is never clipped. It is `mathematically_undefined` when TRTR does not improve on Dummy beyond the declared `1e-12` tolerance.

## Global Utility

Global Utility rotates every included canonical model-view column as the target. Every column must be included or excluded with a stable reason. The formula follows TabStruct Equation 4:

~~~text
categorical target: balanced_accuracy(TSTR) / balanced_accuracy(TRTR)
numerical target:   RMSE(TRTR) / RMSE(TSTR)
global utility:     equal mean over targets, then equal mean over seeds
~~~

Ratios are not clipped. Missing class support, zero denominators, predictor failures, and resource failures remain explicit; unavailable targets are never silently removed from the denominator.

The predictor panel follows the locked TabEval `UtilityPerFeature` configuration: XGB, KNN, and TabPFN through AutoGluon, with weighted ensembling disabled. TabPFN may be omitted only for classification targets exceeding the locked ten-class limit. XGB and KNN remain mandatory, and TRTR/TSTR must expose the same trained model set.

### Stable fit protocol

TabEval passes `tuning_data=None` to AutoGluon. AutoGluon then creates a hidden validation split from the input row positions. The historical identity-surrogate runs showed that this made equal row multisets produce different fit boundaries after a harmless row permutation.

The sole public adapter corrects that protocol-level instability without patching any upstream model source:

1. convert all three inputs to the reviewed canonical model view;
2. sort each training arm lexicographically by every model-view column, including the target, using `lexicographic-all-model-columns-v1`;
3. compute AutoGluon's declared default holdout fraction;
4. call `autogluon.core.utils.utils.generate_train_test_split` with the Evaluation Request evaluator seed and classification stratification when applicable;
5. pass the resulting fit and tuning tables explicitly to AutoGluon; and
6. pass the seed to AutoGluon's learner while keeping the held-out real test outside both tables.

For each arm, `utility-details.json` records the ordering identity, split implementation, seed, task/problem type, holdout fraction, row counts, content fingerprints for the canonical input/fit/tuning tables, and `real_test_used_for_fit: false`. No row content is embedded in this evidence.

This preserves the official XGB/KNN/TabPFN implementations, TabStruct formula, target coverage, and five-seed design. It changes only the benchmark-owned invocation boundary that previously depended on arbitrary input order.

## Atomic Results and bundle evidence

P4 writes raw Local results per metric/arm/evaluator/seed, Local retention per evaluator/seed, raw Global results per target/arm/seed, and Global target ratios per target/seed. The finalized bundle includes `metrics.parquet`, `summary.json`, `metadata.json`, `artifacts/utility-details.json`, stage records, an artifact inventory, and final checksums.

Bundle validation reconstructs derived values from raw arms and rejects missing arms, changed summaries, unequal weights, mismatched predictor sets, altered denominators, or missing real-test provenance. A partial bundle may be retained for diagnosis but is not a leaderboard score.

## Source parity versus result production

The internal `_source_exact_global_scorer` reconstructs TabEval's original implicit-split call only inside the locked-source validation harness. It establishes that the pinned source, predictor wrapper, dependencies, and checkpoints were understood correctly. It is deliberately unregistered and unavailable from `evaluate-table`.

The stable adapter does not claim line-for-line or numerical source parity because its explicit split is an intentional, documented invocation change. Its scientific lineage is instead established by unchanged formulas, predictors, metrics, and direct use of official packages.

## Historical evidence and successor identity

Historical results are immutable:

- the bounded Linux source-runtime evidence remains at [p4-global-source-runtime-run-31057073762.json](../evidence/evaluation/p4-global-source-runtime-run-31057073762.json);
- the exact Windows GPU source-runtime evidence remains at [p4-global-source-windows-gpu-c0e6e72.json](../evidence/evaluation/p4-global-source-windows-gpu-c0e6e72.json); and
- the complete historical Windows dataset-scale result remains at [p4-dataset-scale-windows-gpu-a754ca1.json](../evidence/evaluation/p4-dataset-scale-windows-gpu-a754ca1.json).

The last result executed all 67 tasks and 134 arms and passed its execution/resource gates, but failed the unchanged `0.05` stability gates for Adult `native-country`, Sick `referral-source`, and Sick `tsh`. That result remains failed. It is not reinterpreted under the new implementation.

The successor validation has a new identity, `p4-dataset-scale-windows-gpu-stable-candidate@0.3.0`. It retains the same datasets, targets, five seeds, full-row-permutation identity surrogate, predictor policy, `0.05` gates, and resource limits. It binds the new evaluator and dataset-profile versions and runs only on the declared native Windows 11/Python 3.11/RTX 5080 environment. GitHub-hosted runners validate its contracts but cannot substitute for the required GPU run.

## Command

Install the standard utility dependencies with:

~~~powershell
python -m pip install -e ".[utility]"
~~~

The optional Global runtime must additionally satisfy `requirements-p4-windows-gpu-validation.txt`. Without it, Local Utility is computed and Global failures remain explicit; no fallback predictor is substituted.

~~~powershell
std-tabular-diffusion evaluate-table `
  --protocol p4-utility `
  --reference real_train.csv `
  --real-test real_test.csv `
  --synthetic synthetic_train.csv `
  --dataset-profile configs/datasets/adult-uci-2-v1.json `
  --output artifacts/p4/adult/run-001
~~~

P4 defaults to evaluator seeds `0,1,2,3,4`. Overrides are recorded and are not automatically leaderboard-compatible.

## Remaining admission work

Before P4 can be frozen or admitted to Official Results:

1. run the successor's former-failure sentinels on the declared Windows GPU runtime;
2. run and finalize the complete preregistered 67-task schedule;
3. require every scientific, execution, resource, fit-boundary, and evidence gate to pass without changing thresholds after observation; and
4. perform a separate profile-freeze and release-admission review.
