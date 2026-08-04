# TabularARGN Validation Protocol

Status: protocol implemented; mandatory Linux run pending

Protocol: `tabularargn-official-package-parity-v2`

Target: method-author official `mostlyai-engine==2.6.2` flat TabularARGN package

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This protocol tests whether the standardized `tabularargn` adapter preserves the selected official flat-table execution. It compares direct calls to the checksum-pinned method-author package with adapter calls using the same typed training table, constructor controls, official persistent workspace, sampling controls, and random seed.

A passing mandatory run may promote this path to `native-parity-validated`. It does not make the model `benchmark-eligible`, admit it to Official Results, establish paper-scale generation quality, or make it `release-supported`. Differential privacy, sequential and relational data, conditional generation, prediction, probability estimation, likelihood, imputation, central evaluation, dataset admission, and resource budgets remain independent scopes or gates.

## Audited Authority and Distribution

The authority is the [MOSTLY AI method-author repository](https://github.com/mostly-ai/mostlyai-engine) at tag `2.6.2`, commit `0b96f02e4fad47c7c19c985fda4311230e20bbb5`, and tree `199a4085315e601261898007b0dd4ac532d355fe`. The commit is verified by GitHub. The selected artifact is the official [PyPI 2.6.2 release](https://pypi.org/project/mostlyai-engine/2.6.2/):

- wheel: `mostlyai_engine-2.6.2-py3-none-any.whl`;
- size: 185,077 bytes;
- SHA-256: `3ead3770c936919f8fce4e1f9fffd271ffdd490f0292c2ab9a42cb4bafe3caea`;
- source distribution SHA-256: `a75ce62fc4e91adf1f5fae7166431aec8e1505f26ef260d078e1607fcfb44d82`;
- license: Apache-2.0;
- Python requirement: `>=3.11,<3.14`;
- distribution form: optional external package; no MOSTLY AI source is vendored here.

All 50 package source files in the wheel are byte-exact with the tagged method-author source archive. The source and wheel license files are also byte-exact. The protocol locks the GitHub archive, wheel, source distribution, metadata, license, and every one of the wheel's 53 hash-bearing `RECORD` files.

The protocol rejects renamed, resized, symlinked, altered, or path-traversing artifacts. It verifies package identity, Python range, declared dependencies, wheel tag, license, every `RECORD` path/size/hash, all installed copies, the source archive, and the 50 source-to-wheel comparisons. Direct runtime dependencies must match the frozen validation versions.

## Adapter Semantics

The adapter:

- accepts one missing-free single table with exactly one classification or regression target;
- requires unique columns in exact `DatasetSpec` order and disjoint roles that cover every column once;
- checks numerical features and regression targets for finite numeric values;
- casts declared categorical features and classification targets to strings before the unchanged official fit path;
- maps the standardized device and seed to official constructor arguments;
- exposes bounded official flat-table constructor controls and rejects unknown controls;
- places the official persistent workspace beneath `output_dir`;
- removes `OriginalData` after fit because it contains raw and encoded training rows not required by flat-table generation;
- retains the official `ModelStore`, removes training-only optimizer state, and records a file-level integrity manifest;
- verifies package, dataset, and every retained model-store file before generation;
- uses the official public `set_random_state` and `generate` functions plus the official generated-data loader; and
- enforces exact output rows, canonical columns, no missing/non-finite values, and training-domain categorical values.

Missing values fail closed. A dataset with missing values must first use the centralized mean/mode imputer fitted only on the training split.

The package default for rare categories may emit its `_RARE_` sentinel. The benchmark contract forbids categories outside the observed training domain, so the adapter's standardized default is the official `rare_category_replacement_method=SAMPLE` option. Users can request `CONSTANT`, but the output-domain check will still fail if `_RARE_` is generated.

## Persistent Artifact and Security

The old adapter pickled the full estimator without package identity or checkpoint integrity checks. The current adapter stores no estimator pickle. The official workspace is the package's persistent checkpoint format: generation reconstructs the model from `ModelStore/model-data/model-configs.json`, `model-weights.pt`, and `ModelStore/tgt-stats/stats.json`. Official 2.6.2 already calls `torch.load(..., weights_only=True)` for these weights.

The adapter removes the official `OriginalData` tree after training, including raw and encoded row files. This limits accidental row-level disclosure but does not make the model or statistics non-sensitive. Model artifacts still require normal access control and privacy review.

## Supported Controls

Training exposes the official model identifiers (`MOSTLY_AI/Small`, `MOSTLY_AI/Medium`, and `MOSTLY_AI/Large`), maximum time and epochs, batch size, gradient accumulation, flexible generation, value protection, target encoding types, and verbosity. `RunSpec.device`, `RunSpec.seed`, and the output workspace are owned by the standardized contract. A deterministic `max_train_rows` control exists only for bounded smoke and parity jobs.

Sampling exposes official batch size, temperature, top-p, and rare-category replacement controls. Seed data, context tables, rebalancing, imputation, fairness, differential privacy, sequential training, and relational training are not silently mapped into this single-table unconditional-generation adapter.

## Frozen Parity Cases

The mandatory protocol uses three deterministic, mixed-type, missing-free fixtures with 48 rows each:

1. binary classification;
2. multiclass classification; and
3. regression.

Each fixture has two numerical features, one categorical feature, and one target. Every variant runs with seeds 0, 19, and 73, for nine independent cases. Each case trains the real official `MOSTLY_AI/Small` flat model for one bounded epoch, with value protection and flexible generation enabled, then requests seven rows through the official `SAMPLE` rare-category option.

The independent paths are:

- native: official `TabularARGN` constructor → `fit` → official `set_random_state` → `sample`;
- adapter: standardized `train` → pruned and integrity-verified official workspace → standardized `sample`.

## Mandatory Pass Criteria

All nine cases must pass. The gate requires:

1. exact Linux/Python 3.11 environment, release artifacts, source/wheel equivalence, license, installed files, and direct dependency identity;
2. identical official checkpoint key order and tensor values;
3. semantically identical official model configuration and target statistics;
4. value-identical samples after applying only the adapter's documented categorical string normalization, with the unnormalized native and adapter dtypes retained separately in evidence;
5. byte-identical native and adapter CSV files;
6. exact requested row count and canonical column order;
7. no missing or non-finite numerical output and no out-of-domain categorical output;
8. valid package, dataset, training, and checkpoint-integrity metadata;
9. confirmed removal of raw and encoded training rows from the adapter artifact; and
10. unchanged locked package files after every case.

Any mismatch or provenance, dependency, platform, path, metadata, or integrity failure fails closed and retains a diagnostic JSON artifact.

Protocol v2 makes the distinction in criterion 4 explicit. The official package may return a categorical value such as integer `1`, while the public adapter contract represents that same category as string `"1"`. This interface-only dtype normalization is audited separately and cannot hide value, row, order, numerical-dtype, checkpoint, or serialized-output differences.

## Known Boundaries

- The protocol establishes adapter parity, not generation quality at research-scale budgets.
- Only flat, single-table, unconditional generation is in scope.
- Differential privacy is an official package capability but is not validated by this protocol.
- Sequential and two-table context modes require a future relational dataset contract.
- Official prediction, probability, likelihood, conditional generation, and imputation APIs are not claimed by this adapter.
- The checkpoint no longer contains raw row files, but learned weights and aggregate statistics may still be sensitive.
- Benchmark eligibility still requires central metrics, dataset admission, resource profiles, and release review.

## Evidence

The formal GitHub Actions job is defined in `.github/workflows/tabularargn-validation.yml`. Until its Linux/Python 3.11 artifact is downloaded, inspected, and retained under `docs/evidence/tabularargn/`, the registry remains `adapter-complete`.
