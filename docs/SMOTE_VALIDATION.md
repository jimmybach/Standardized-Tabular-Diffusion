# SMOTE Validation Protocol

Status: passed on Linux/Python 3.11; adapter is `native-parity-validated`

Protocol ID: `smote-native-parity-v1`

Supported validation platform: Linux, Python 3.11

## Scope and claim boundary

This protocol tests whether the standardized `smote` adapter invokes the checksum-pinned official `imbalanced-learn==0.14.2` package with the same input table, target, sampler selection, constructor arguments, seed, resampling request, and output selection as a direct native call. It covers official `SMOTE`, `SMOTENC`, and `SMOTEN`, package identity, deterministic output, fitted sampler state, artifact metadata, and exact native-versus-adapter comparisons.

SMOTE is a classification-only classical oversampling reference. It is not a model of the joint table distribution and must not be ranked as a generative-model peer. Its output contains the original resampled training rows together with interpolated minority-class rows; requesting a different output size subsequently samples from that combined table. Consequently, this adapter is suitable for downstream classification-utility experiments, but its output must not be used as if it were a standalone synthetic dataset in generative fidelity, privacy, or memorization rankings.

A passing mandatory run may promote the adapter to `native-parity-validated`. It does not make SMOTE `benchmark-eligible`, admit it to Official Results, or make it `release-supported`. A separate classical-reference track, admitted datasets, frozen evaluation rules, leakage controls, runtime thresholds, and release ownership remain independent gates.

## Authority, package identity, and license

- Canonical repository: `scikit-learn-contrib/imbalanced-learn`.
- Release tag: `0.14.2`.
- Git commit: `8504e95f0160f61d1b617ca66f779646d2ee609e`.
- Git tree: `af452de62e0f5c3d7e65fdc44a32dc97078152f2`.
- PyPI wheel: `imbalanced_learn-0.14.2-py3-none-any.whl`.
- Wheel SHA-256: `f9b81c47231aa1e3a71a1e4b3cc85b42e3b14f85e3a36922f3323c4da23605ef`.
- License: MIT.

Before model execution, the protocol verifies the wheel filename and hash, safe archive paths, package metadata, Python requirement, source and wheel license hashes, archive member count, installed distribution version, public class identity, all 123 hash-bearing files from the wheel `RECORD`, and the imported distribution location. No imbalanced-learn source is vendored or patched by this repository.

Primary references are the [0.14.2 release](https://github.com/scikit-learn-contrib/imbalanced-learn/releases/tag/0.14.2), [PyPI distribution](https://pypi.org/project/imbalanced-learn/0.14.2/), and official API documentation for [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), [SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html), and [SMOTEN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTEN.html).

## Adapter contract

The repository-owned adapter:

- accepts classification datasets with exactly one target and at least one feature;
- rejects missing values and instructs callers to run explicit train-split-fitted preprocessing first;
- uses official `SMOTE` when every feature is numerical;
- uses official `SMOTENC` with categorical column names when features are mixed;
- uses official `SMOTEN` when every feature is categorical;
- forwards `random_state`, `k_neighbors`, and a JSON-serializable `sampling_strategy` without changing upstream source;
- rejects an unavailable or non-0.14.2 package, invalid neighbor count, one-class data, and classes too small for the requested neighbors;
- returns canonical dataset column order and records the source, balanced, and requested output row counts; and
- creates no persistent model checkpoint because resampling is recomputed from the canonical training split.

The adapter deliberately removed the earlier repository-side ordinal encoding, rounding, clipping, and inverse transformation. Official `SMOTENC` accepts DataFrames and categorical column names directly; adding a second encoding layer would create local semantics and would no longer be a strict native wrapper.

## Input and evaluation policy

The input must be the canonical training split only. Validation and test rows must never participate in fitting, neighbor search, resampling, preprocessing statistics, or output-size selection. If missing-value preprocessing is required, numerical means and categorical modes are fitted on the training split and then applied to the other splits.

`sampling_strategy="auto"` is the default. `k_neighbors=5` is the public adapter default; the frozen parity fixture uses `k_neighbors=3` so that a bounded minority class can exercise the real algorithm. The smallest class must contain at least `k_neighbors + 1` rows.

SMOTE results belong only in a clearly labeled classical oversampling reference section. Downstream classifiers must be trained on the resampled training table and evaluated on the untouched real test split. The combined SMOTE table must not be scored as a generated table against the training data, because it intentionally retains original records.

## Frozen environment

The workflow installs CPython 3.11, the exact packages in `requirements-smote-validation.txt`, and the official wheel only after verifying its SHA-256. The repository package is installed without optional dependency resolution. `pip check` must pass before validation starts. The protocol rejects any non-Linux platform, non-3.11 interpreter, or distribution-version mismatch.

Equivalent Linux setup commands are:

```bash
python -m pip install -r requirements-smote-validation.txt
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest /tmp/smote-wheel "imbalanced-learn==0.14.2"
echo "f9b81c47231aa1e3a71a1e4b3cc85b42e3b14f85e3a36922f3323c4da23605ef  /tmp/smote-wheel/imbalanced_learn-0.14.2-py3-none-any.whl" | sha256sum --check
python -m pip install --no-deps /tmp/smote-wheel/imbalanced_learn-0.14.2-py3-none-any.whl
python -m pip install --no-deps .
python -m pip check
```

## Frozen comparison

The protocol uses three deterministic, missing-free binary-classification fixtures:

1. two numerical features, dispatched to `SMOTE`;
2. two numerical plus one categorical feature, dispatched to `SMOTENC`; and
3. two categorical features, dispatched to `SMOTEN`.

Each fixture has 18 source rows with class counts 12 and 6. `sampling_strategy="auto"` and `k_neighbors=3` produce a 24-row balanced table, from which 20 rows are selected deterministically. For seeds `0`, `19`, and `73`, both the direct native path and adapter path must satisfy all of the following:

1. wheel, release, installed package, license, and installed file identities match the lock;
2. official sampler class, module, and constructor parameters are identical;
3. fitted sampling strategy, feature metadata, categorical feature selection, encoder categories, neighbor matrix, and SMOTENC median state are exactly equal when applicable;
4. the global NumPy random state is not mutated;
5. balanced row counts and target class counts are exact;
6. output CSV bytes and reloaded DataFrames are exactly equal;
7. numerical outputs are finite, categorical outputs remain in observed domains, and no value is missing; and
8. artifact manifests and `smote_metadata.json` exactly describe the execution.

There is no numerical tolerance: deterministic parity must be exact across all nine variant/seed cases.

## Retained result

GitHub Actions run [`30918785254`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30918785254) passed all nine variant/seed cases on Linux with Python 3.11.15. It verified 123 hash-bearing installed files against the locked wheel, executed official SMOTE, SMOTENC, and SMOTEN for seeds 0, 19, and 73, and produced byte-exact native/adapter CSV output for every case.

The inspected JSON is permanently retained at `docs/evidence/smote/native-parity-run-30918785254.json` with SHA-256 `1b375b93c332327dd2118c2aad9420497008be1390078e1e48e79f8270f74863`. The corresponding GitHub artifact is `8896180932`, with artifact digest `sha256:ecc8167acd739762bdaca258d5dcb6a5f23648b45e6adba9c6900c14063d1aa6`. The repository copy remains authoritative after the temporary artifact expires.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.smote \
  --repo-root . \
  --output-dir /tmp/smote-validation \
  --evidence-path /tmp/smote-evidence.json \
  --wheel-path /tmp/smote-wheel/imbalanced_learn-0.14.2-py3-none-any.whl
```

`.github/workflows/smote-validation.yml` runs this command and retains the evidence artifact for 90 days. Any package, dependency, adapter, fixture, or protocol change requires a new run. The inspected passing evidence promotes the adapter to `native-parity-validated`; it remains `experimental`, `unsupported`, excluded from Official Results, and excluded from joint generative-model ranking pending the separate gates described above.
