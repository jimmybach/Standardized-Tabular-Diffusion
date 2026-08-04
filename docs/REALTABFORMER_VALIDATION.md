# REaLTabFormer Validation Protocol

Status: passed and permanently retained

Protocol: `realtabformer-official-package-parity-v1`

Target: method-author official `realtabformer==0.2.4` tabular package

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This protocol tests whether the standardized `realtabformer` adapter preserves the selected official tabular execution. It compares a direct call to the checksum-pinned official package with the adapter on the same typed training table, GPT-2 configuration, training controls, checkpoint reload, sampling controls, and random seed.

A passing mandatory run may promote the adapter to `native-parity-validated` for the tested tabular path. It does not make REaLTabFormer `benchmark-eligible`, admit it to Official Results, establish full-paper statistical quality, or make it `release-supported`. The official sensitivity-based stopping path, relational model, dataset admission, central evaluation, resource budgets, and release ownership remain separate gates.

## Audited Authority and Distribution

The authority is the [World Bank method-author repository](https://github.com/worldbank/REaLTabFormer) at tag `v0.2.4`, commit `73f239643f9ea5abc877f685ce927e986302ac2d`, tree `aa4431468f040fc485f82e7e15238c57eef05753`. The selected artifact is the official [PyPI 0.2.4 release](https://pypi.org/project/realtabformer/0.2.4/):

- filename: `realtabformer-0.2.4-py3-none-any.whl`;
- size: 49,890 bytes;
- SHA-256: `852436c5c82a0bf470ca7e9063e5a4f3e250b3ff5b9c8f6c50113c1e9ba76486`;
- license: MIT, Copyright 2022 Aivin V. Solatorio;
- distribution form: optional package dependency; no REaLTabFormer source is vendored here.

The tag and repository default branch currently resolve to the same commit. Eleven source files shared by the wheel and the tagged source archive are byte-exact. The wheel adds only an empty `rtf_tokenizer.py`. The source archive, wheel, source license, package metadata, and all 16 hash-bearing installed files are independently checksum-locked.

The protocol rejects a renamed, resized, symlinked, altered, or path-traversing wheel. It checks the package identity, Python requirement, declared dependencies, pure-Python tag, MIT license, all `RECORD` paths, sizes, and hashes, and the installed copies of every locked file. Runtime direct dependencies must match the frozen validation versions.

## Adapter Semantics

The adapter:

- accepts one missing-free single table with exactly one classification or regression target;
- reads columns in the exact `DatasetSpec` order and requires declared roles to cover them once;
- checks numerical features and regression targets for finite numeric values;
- casts declared categorical features and classification targets to strings for the unchanged official tokenizer;
- passes `RunSpec.seed` to the official constructor and Hugging Face training arguments;
- places official checkpoint, sensitivity-sample, periodic-save, and final-model directories under `output_dir`;
- saves the official `rtf_config.json`, `rtf_model.pt`, and applicable official artifacts;
- records package, transformed-frame, effective-control, and file-level checkpoint integrity metadata;
- verifies that metadata before loading and requires one unambiguous saved model directory;
- runs the official loader with PyTorch `weights_only=True`; and
- resets Python, NumPy, and PyTorch generators immediately before the unchanged official `sample()` call.

Missing values fail closed. A dataset with missing values must first use the centralized imputer fitted only on the training split; the adapter never learns replacement statistics from validation or test data.

## Recorded Compatibility Boundaries

No official source file is edited. Six explicit boundaries are recorded:

1. output isolation supplies official directory arguments beneath `output_dir`;
2. declared categorical roles are converted to strings before both native and adapter calls;
3. sampling generators are reset from the requested seed before both calls;
4. unused torchvision probing is disabled only while Transformers imports;
5. the official state-dict loader is constrained to `weights_only=True`; and
6. immediately before the unchanged official `save()`, `full_save_dir` is represented by its identical string path.

The last item works around a v0.2.4 serialization defect: official `save()` converts two sibling `Path` attributes to strings but omits `full_save_dir`, so a newly constructed model otherwise fails in `json.dumps`. This changes no model, tensor, optimizer, preprocessing, or sampling state. Because this and the restricted loader alter runtime statements at the call boundary, the integration is classified conservatively as `compatibility-patched`, not `adapter-only`.

## Supported Controls

The adapter restores official defaults of 1,000 epochs and batch size 8. It exposes bounded constructor controls for train fraction, early stopping, masking, and numerical tokenization; a validated subset of Hugging Face `TrainingArguments`; all official tabular fit sensitivity controls; a custom `GPT2Config` mapping whose data-derived vocabulary and special-token fields remain owned by REaLTabFormer; deterministic row limiting for smoke tests; and official generation arguments for sampling.

Unknown controls fail closed. External training reporters are disabled unless explicitly represented by the supported no-reporting value. The tiny validation configuration uses one GPT-2 layer only to bound CI cost; it is not a recommended benchmark configuration.

## Frozen Parity Cases

The mandatory protocol uses three deterministic mixed-type, missing-free fixtures with 24 rows each:

1. binary classification;
2. multiclass classification; and
3. regression.

Each fixture has two numerical features, one categorical feature, and one target. Every variant runs with seeds 0, 19, and 73, for nine independent cases. Each case trains the real official GPT-2 implementation for one bounded epoch with sensitivity stopping disabled (`n_critic=0`), saves and reloads the model, and requests seven rows.

The two independent paths are:

- native: direct official constructor → `fit` → `save` → `load_from_dir` → `sample`;
- adapter: standardized `train` → integrity-verified official checkpoint → standardized `sample`.

## Mandatory Pass Criteria

All nine cases must pass. The gate requires:

1. exact Linux/Python 3.11 environment, wheel, installed-file, license, and dependency identity;
2. identical checkpoint key order and tensor values;
3. semantically identical saved official configuration after replacing output-root and time-based experiment identifiers with placeholders;
4. DataFrame-identical raw outputs before standardized serialization;
5. byte-identical native and adapter sample CSV files;
6. exact requested row count and canonical column order;
7. no missing or non-finite numerical output and no out-of-domain categorical output;
8. valid adapter package, training-frame, effective-control, and checkpoint-integrity metadata; and
9. unchanged locked package files after all cases.

Any mismatch, dependency drift, wrong platform, unsafe artifact, ambiguous model directory, or integrity failure fails closed and retains a diagnostic JSON artifact.

## Known Boundaries

- The validation fixtures establish wrapper parity, not generation quality at paper-scale training budgets.
- The official sensitivity-based stopping path is exposed but is not promoted by this protocol; it requires a separate, resource-bounded validation study.
- The official relational model accepts linked parent and child tables. The repository's current canonical contract is single-table, so relational mode is outside this adapter's validated scope.
- Transformer training and autoregressive sampling are substantially heavier than classical baselines. Dataset-specific sequence-length, runtime, and memory budgets remain pending.
- Conditional seed inputs and advanced generation controls are official expert interfaces; benchmark profiles must freeze them before Official Results.
- The official package supports Python versions older than 3.11, but this repository's supported release environment is Linux/Python 3.11 only.

## Evidence

[GitHub Actions run `30950369908`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30950369908) passed all nine task/seed cases on Linux with Python 3.11.15. The run verified the official wheel and all 16 hash-bearing installed files, then produced exact native/adapter checkpoint tensors and files, semantically exact saved configurations, identical raw samples, and byte-identical final CSV files in every case. All requested outputs had seven canonical rows, finite numerical values, valid categorical domains, and no missing values.

The permanent evidence record is `docs/evidence/realtabformer/native-parity-run-30950369908.json`, SHA-256 `0c6047efc3463aa21fa4b2e6aeed66858cbc29bfd5a9e836f330d975ec0cfa07`. It is retained byte-for-byte from artifact `8908863813`, whose archive digest is `sha256:03ae72ed21ea357c466a9c7f9ee3b29a1c2e5e29ec8fcc2305c9dc7a7f2f8147`. The PR head was `7db46e00452ce5cc25d28d8b484c9d6ee14de5b3`, and the checked-out PR merge commit recorded by the evidence was `fb2f03dd579bb4d1847fa18395696ed698c8ce58`.

REaLTabFormer is therefore `native-parity-validated` for the official tabular `n_critic=0` path. It remains `experimental` and `unsupported`; sensitivity stopping, relational mode, central benchmark evaluation, dataset admission, resource budgets, and release support are not promoted by this evidence.
