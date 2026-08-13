# P5 High-order Fidelity and Empirical Privacy

## Status and claim boundary

`p5-high-order-privacy@1.0.0` is the preregistered freeze candidate. Its scientific identity has unit, bundle, retained Windows 11/Python 3.11 Adult/Sick identity-surrogate validation, and an exploratory real-generator pilot. Adult and Sick field roles plus the bounded black-box membership threat model are reviewed. The candidate is not yet protocol-frozen, release-supported, or eligible for Official Results; unresolved metrics remain explicit exclusions.

P5 reports high-order fidelity and empirical privacy risk as separate dimensions. It emits neither an overall Fidelity score nor an overall Privacy score. Empirical attacks and distances do not establish differential privacy or any other formal privacy guarantee.

## Required tables and fit boundaries

P5 requires checksum-bound real train, held-out real test, and synthetic tables in the reviewed canonical model view. The common structural gate rejects missing values, non-finite numerics, schema drift, lossy logical types, and unexpected synthetic row counts. Synthetic output is never repaired.

- C2ST fits its mixed-table transform on real train only. Held-out real test and synthetic rows never fit the transform.
- DCR uses real train only as the closest-record reference.
- DOMIAS splits held-out real test into disjoint nonmembers and an auxiliary reference. Its mixed-table transform, PCA, and reference density fit only the auxiliary reference.

## Classifier two-sample test

The primary comparison is held-out real test versus synthetic. Each of five evaluator seeds uses equal class sizes, deterministic canonical subsampling, a stratified 70/30 discriminator split, and a fixed scikit-learn Random Forest with 200 trees, `min_samples_leaf=2`, and one execution thread. Numerical features use real-train-fitted standardization; categorical and Boolean features use real-train-fitted one-hot encoding with unknown categories ignored.

The protocol records:

- `std-c2st-rf-auroc`: raw discriminator AUROC with target `0.5`;
- adjusted AUROC: `max(AUROC, 1 - AUROC)` in evidence details; and
- `std-c2st-fidelity`: `2 * (1 - adjusted_AUROC)`, where one means chance distinguishability and zero means perfect separability.

Raw AUROC and the fidelity complement remain distinct. Five-seed values and deterministic 1,000-replicate stratified-bootstrap 95% AUROC intervals are retained. AUROC and its complement are benchmark-derived extensions of the classifier two-sample-test principle; they are not presented as the original paper's primary accuracy statistic.

GReaT Random-Forest discriminator accuracy is an explicit source-specific exclusion from this P5 request. It is not averaged with C2ST.

## Exact copying and duplication

Rows are converted to type-aware, lossless semantic tokens without rounding. P5 separately reports:

- the fraction and count of synthetic rows exactly matching any real-train row;
- the number of distinct real-train rows matched;
- the maximum synthetic multiplicity of a matched real-train row; and
- the internal synthetic duplicate rate and duplicate count.

A duplicated synthetic row is not automatically a training collision, and a training collision is not hidden inside the diversity diagnostic.

## SDMetrics distance to closest record

P5 calls `sdmetrics.single_table.privacy.dcr_utils.calculate_dcr` directly from the checksum-attested SDMetrics `0.28.3.dev0` source tree at commit `ba8842f2ba04ce914f698cc1cf746ca12338ab0e`. It retains every value for:

- `DCR(synthetic, real_train)`; and
- `DCR(heldout_real_test, real_train)`.

The detail artifact includes minimum, quantiles, mean, population standard deviation, zero rate, heldout one/five-percent thresholds, and the fraction of synthetic distances at or below those thresholds. `std-dcr-heldout-wasserstein` compares the two complete sampled distributions.

Raw DCR is distributional. It is not ranked as “larger is always safer,” because an extremely distant but useless table is not a privacy success.

## DOMIAS membership inference

The declared threat model is black-box: the attacker receives the released synthetic table and a disjoint auxiliary real reference sample, but no generator source, parameters, gradients, or internal state.

- members are a seeded sample from real train;
- nonmembers are a disjoint seeded sample from held-out real test;
- the auxiliary reference is the other disjoint held-out sample; and
- candidate groups are balanced.

P5 reproduces the DOMIAS density-ratio score `p_G(x) / (p_R(x) + 1e-10)`, the official pooled-median strict-threshold attack accuracy, and attack AUROC. It additionally reports maximum attack advantage and TPR at FPR no greater than one percent. AUROC uncertainty uses the same deterministic stratified-bootstrap policy as C2ST.

The official DOMIAS formula is preserved, but the mixed-table representation is a benchmark adaptation: auxiliary-reference-fitted standardization/one-hot encoding, followed by auxiliary-reference-fitted PCA and Gaussian KDE. PCA is not invertible, so this adapter does not claim exact end-to-end source parity. Singular or weakly supported density estimation returns an explicit non-computed state; no silent estimator substitution occurs.

## Explicit exclusions

The Metric Registry records these metrics as excluded, and they cannot be selected by the P5 protocol:

- GReaT RF discriminator accuracy: distinct source diagnostic, not C2ST AUROC;
- integrated Alpha-Precision and Beta-Recall: mixed-table support embedding unresolved;
- Alaa paper and repository Authenticity variants: paper/code semantics unresolved;
- SynthCity Delta Presence: threat model, direction, and failure semantics unresolved; and
- attribute inference: Adult and Sick roles are reviewed, but P5 v1 has no separately approved attribute-inference implementation and threat model.

## Bundle and command

`evaluate-table` writes Atomic Results, raw DCR arrays, C2ST runs and intervals, exact-row evidence, the complete DOMIAS threat model and attacks, source provenance, stage records, artifact inventory, and checksums. Bundle finalization rejects altered summaries, missing metric/seed scopes, nonzero overall-score weights, incomplete threat models, broken fit boundaries, and inconsistent exact-row or DCR evidence.

~~~powershell
std-tabular-diffusion evaluate-table `
  --protocol p5-high-order-privacy `
  --reference real_train.csv `
  --real-test real_test.csv `
  --synthetic synthetic_train.csv `
  --dataset-profile configs/datasets/adult-uci-2-v1.json `
  --output artifacts/p5/adult/run-001
~~~

P5 requires evaluator seeds `0,1,2,3,4`. The retained [Windows Adult/Sick identity-surrogate evidence](../evidence/evaluation/p5-windows-py311-identity-c66fa23.json) validates the complete execution and result boundary but deliberately does not assess generator quality. The retained [TabDDPM/Adult three-generation-seed evidence](../evidence/evaluation/p5-tabddpm-adult-windows-py311-a2e4f27.json) closes the first exploratory non-identity pilot. The independent seeds `3,4,5` confirmation and its pass/fail gates are fixed in the [confirmatory preregistration](../evidence/evaluation/p5-confirmatory-preregistration-2026-08-13.json). None of these records independently admits Official Results.

## Completed first generator pilot

The first non-identity pilot passed in the deliberately narrow declared scope: one checksum-pinned
TabDDPM model on the reviewed Adult dataset. It trains one checkpoint with the
official `ddpm_cb_best` Adult configuration and samples the same checkpoint at
generation seeds `0,1,2`, producing exactly 32,561 rows per table. Every table
is independently evaluated with the unchanged P5 evaluator seeds
`0,1,2,3,4`.

TabDDPM's unchanged loader requires a validation array although its generator
training does not use one. The pilot therefore supplies a disclosed one-row
mirror of real train solely at the loader boundary while fitting on the full
32,561-row official training split. The adapter's decoded CSV maps the upstream
classification indices to reviewed labels and converts declared integer fields
to nearest integers (ties to even); raw upstream arrays, hashes, and conversion
counts remain in the ignored experiment artifacts. P5 applies no repair.

The pilot is exploratory. Its passing result does not freeze P5, admit TabDDPM or Adult
to Official Results, or establish a formal privacy guarantee. Attribute
inference remains excluded from P5 v1. Reviewed field roles do not substitute
for a separately approved implementation and attribute-inference threat model.

The pilot also exercises a dated dependency boundary absent from the small
TabDDPM parity fixture: upstream passes the mathematically integral value `1e9`
as a float to `QuantileTransformer.subsample`. Supported Python 3.11
scikit-learn rejects the type before fitting, so an adapter-only startup bridge
converts integral floats to their exactly equal integers. It does not modify
upstream source or any non-integral estimator value.

## Preregistered confirmation

The freeze-candidate run uses a fresh training run at training seed `0`, then
samples the checksum-identical checkpoint at new generation seeds `3,4,5`.
Each 32,561-row table is evaluated with the unchanged five-seed panel. The
decision uses only preregistered completeness, structural, environment, and
provenance gates; numerical metric values cannot cause acceptance, rejection,
seed replacement, or selective reruns. Row-level materializations remain in
ignored local artifacts. A successful confirmation may support a separate
protocol-freeze decision, but cannot admit a dataset, model, track, run, or
environment and cannot establish privacy or regulatory compliance.
