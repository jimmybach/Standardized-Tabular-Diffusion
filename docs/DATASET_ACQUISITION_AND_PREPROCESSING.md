# Dataset Acquisition and Missing-Value Preprocessing

This document defines the executable v1 data boundary. Dataset Profiles remain the authority for official-suite admission; the source registry and preprocessing artifacts implement the security and leakage controls required by those profiles.

## 1. Source acquisition

Public sources are locked in `standardized_tabular_diffusion/resources/datasets/sources.json`. Each record fixes:

- the canonical publisher page and HTTPS retrieval URL;
- the source version and retrieval date;
- the archive SHA-256 and byte limits;
- the exact archive members used by this project;
- the license, citation, and redistribution status.

List the source locks:

```bash
python -m standardized_tabular_diffusion.cli list-dataset-sources
```

Download and safely extract one source:

```bash
python -m standardized_tabular_diffusion.cli download-dataset --dataset adult
```

For either canonical official model view, acquisition, schema validation, fixed-split verification, train-only preprocessing, and adapter-layout generation are one command:

```bash
pip install "standardized-tabular-diffusion[data]"
python -m standardized_tabular_diffusion.cli materialize-dataset --dataset adult
python -m standardized_tabular_diffusion.cli materialize-dataset --dataset sick
```

The downloader writes to the user cache by default. Set `STD_TABULAR_DIFFUSION_CACHE` or pass `--cache-dir` to choose a different local cache. A cached archive is reused only when its SHA-256 still matches the registry. Downloads use HTTPS, enforce a byte limit, write through a temporary file, and become visible only after checksum verification.

ZIP extraction rejects absolute paths, parent traversal, duplicate members, symbolic links, undeclared expansion, and modified content-addressed caches. Only registry-approved members are extracted. Raw archives and extracted data are not Python package contents.

`--refresh` replaces a managed archive only after a fresh download has passed the same checksum. Updating a source to new bytes requires a reviewed registry change; `--refresh` never accepts an unregistered checksum.

## 2. Split before fitting preprocessing

Missing-value statistics are learned only after the real-data split has been fixed:

```text
verified source -> canonical parsing -> frozen real splits
                -> fit preprocessing on train
                -> transform train / validation / test without refitting
                -> model adapter
```

For cross-validation metrics, the same rule applies inside every fold: fit on that fold's training partition and transform its held-out partition.

The generic CSV interface consumes already frozen splits:

```bash
python -m standardized_tabular_diffusion.cli preprocess-missing-values \
  --train-csv path/to/train.csv \
  --validation-csv path/to/val.csv \
  --test-csv path/to/test.csv \
  --output-dir local-data/example/imputed-v1 \
  --numerical-column age \
  --categorical-column workclass \
  --target-column income
```

Every input column must have exactly one role: numerical feature, categorical feature, or target. Column names and order must agree across splits. Preprocessing never silently drops a row or column.

## 3. v1 missing-value contract

- Numerical features use the arithmetic mean of observed training values.
- Categorical features use the most frequent observed training value.
- A mode tie is resolved by deterministic Unicode string order.
- Validation and test values never influence a learned statistic.
- A feature that is entirely missing in training fails because its statistic is undefined.
- A non-missing, non-numeric value in a numerical column fails instead of being reclassified as missing.
- Missing targets fail. Labels are not imputed.
- Missing generated values fail model-output validation. Generated samples are not repaired after the fact.
- Missing indicators are optional and, when enabled, are created for every feature so the output schema is stable across splits.

The default raw markers are `?`, ` ?`, the empty string, and one space. Dataset Profiles may override the exact marker set. Marker matching is exact: preprocessing does not trim or otherwise rewrite valid category values implicitly.

## 4. Audit artifacts

The file workflow emits:

- `train.csv`, optional `val.csv`, and optional `test.csv`;
- `imputation-state.json`, containing the learned means, modes, policy, training missing counts, schema, and implementation version;
- `preprocessing-manifest.json`, containing input/output checksums, row counts, per-split imputation reports, the state checksum, and the state fingerprint.

Artifacts record that fitting occurred on `train`. They use portable output-relative paths and do not record local absolute input paths. A policy, implementation, split, schema, or learned-state change creates a different preprocessing identity and must not be mixed with older official results.

## 5. Current source status

The registry initially locks the UCI Adult archive and the UCI Thyroid Disease archive used by the `sick` view. Both canonical UCI pages declare CC BY 4.0 and provide the citations recorded in the registry.

Adult now uses only the official UCI files. Its build contract freezes the 32,561 rows in `adult.data`, 16,281 rows in `adult.test`, exact ASCII comma-space syntax, the test header and target suffix normalization, member and ordered canonical-row checksums, class counts, raw missing counts, categorical domains, integer source ranges, and duplicate-row audits. Only `workclass`, `occupation`, and `native.country` contain `?`; their modes are fitted on `adult.data` as `Private`, `Prof-specialty`, and `United-States`, then applied unchanged to the official test split. No row is dropped.

The fixed Adult split contains 23 unique raw rows on both sides, covering 25 training rows and 23 test rows. Train-fitted imputation increases this to 24 unique processed rows, covering 26 training rows and 24 test rows. These are disclosed properties of the official source split, not silently removed leakage corrections. The reviewed Dataset Profile is `configs/datasets/adult-uci-2-v1.json`.

Sick also uses only the official UCI files. The build contract freezes 2,800 training rows from `sick.data`, 972 test rows from `sick.test`, class counts, ordered record-ID hashes, zero cross-split record-ID overlap, member checksums, raw missing counts, categorical domains, and duplicate-row audits before and after preprocessing. The source field `TBG` is present in the raw audit schema but is missing in every official row. It is therefore explicitly excluded from the 29-column model view instead of inventing an undefined mean. Record IDs are audit-only and never enter model inputs.

Record-ID disjointness does not imply row disjointness after identifiers are removed. The official split contains 11 unique model rows on both sides of the split, covering 20 training rows and 13 test rows. The canonical builder preserves this official split and reports the overlap; it does not silently drop records or invent a replacement split. Formal leaderboard treatment remains a Dataset Profile and protocol decision.

The reviewed Sick Dataset Profile is `configs/datasets/sick-uci-102-v1.json`. Both reviewed profiles deliberately remain non-eligible for Official Results until their person-level-data privacy roles, threat models, domain constraints, metric applicability, ethical considerations, and suite admission receive the required review. "Official source" does not automatically mean "release-supported benchmark dataset."

The vendored `TabDiff-main/download_dataset.py` and dataset-specific branches in `TabDiff-main/process_dataset.py` are legacy upstream behavior. They do not provide this source-lock, safe-extraction, or centralized imputation contract and are not evidence for official dataset admission.
