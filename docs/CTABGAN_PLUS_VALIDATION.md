# CTAB-GAN+ Native-Parity Validation

Status: passed on Linux/Python 3.11; adapter is `native-parity-validated`

## Claim Boundary

This protocol asks one narrow question: does the standardized adapter preserve the behavior of the pinned, unmodified method-author CTAB-GAN+ source under the frozen validation environment?

A pass may promote the adapter to `native-parity-validated`. It cannot make CTAB-GAN+ `benchmark-eligible`, admit it to Official Results, or make it `release-supported`. The method-author repository declares no license file or license expression. Consequently, this repository does not redistribute its source, and all public-release uses remain blocked pending explicit rights clarification in addition to the independent evaluation, dataset, runtime, and ownership gates.

## Authoritative Source

- repository: `https://github.com/Team-TUD/CTAB-GAN-Plus`
- commit: `6a6f90188cca3dac2c533fd5e8e7f20de074365b`
- root tree: `f5a08d81b0309d6635bf1c7a646965a34913fa93`
- `model/` tree: `645e6a9d5129346f5d4e29085f1bafd5de4531fd`
- commit purpose upstream: correct the critic iteration for the WGAN-GP framework

The codeload archive URL, byte length, SHA-256, and the byte length and SHA-256 of all five required runtime files are frozen in `standardized_tabular_diffusion/resources/upstream/ctabgan-plus-source-manifest.json`.

The official repository contains no packaging metadata. Users obtain the locked source directly from the method-author repository with:

```bash
python -m standardized_tabular_diffusion.cli materialize-model-source --model ctab-gan-plus
python -m standardized_tabular_diffusion.cli model-source-status --model ctab-gan-plus
```

The materializer verifies the complete archive before extracting only the five required runtime files into the ignored `.cache/upstream-sources/` directory. It does not copy the upstream Adult/King datasets, notebooks, generated data, `.DS_Store`, or bytecode into this repository.

## Retired Embedded Snapshot

The initial repository import contained 18 CTAB-GAN+ files under `TabDDPM-main/CTAB-GAN-Plus/` (146,977 bytes). Comparison with the current official commit found substantive differences, including:

- a DataFrame constructor in place of the official CSV constructor;
- removal of the official supervised train/test split;
- a different critic-iteration count and training-loop structure;
- configurable optimizer/device behavior inserted into upstream classes;
- different sampling batch size, seed handling, and timeout behavior; and
- separate duplicate `model copy/` source.

These are not merely import-path adaptations. The snapshot was therefore removed from the active working tree rather than labelled official or parity validated. It remains recoverable through Git history; history was not rewritten.

## Adapter Boundary

The adapter does not patch any official source file. It:

1. verifies all five official runtime file hashes before import;
2. isolates the generic upstream `model` namespace so it cannot collide with another embedded baseline;
3. writes a temporary CSV because the official constructor accepts a CSV path;
4. derives categorical and task roles from the canonical Dataset Specification;
5. applies documented model controls to the official synthesizer object before `fit()` builds its networks;
6. seeds and restores Python, NumPy, and PyTorch random state around official calls;
7. calls the official synthesizer and inverse preprocessor for an explicit standardized sample count; and
8. binds every pickle checkpoint to its source-manifest and checkpoint hashes before loading.

The official internal train split defaults to `0.2` and is retained. Setting it to zero would reproduce the former semantic patch, not the official method. Dataset-specific `mixed_columns`, `log_columns`, `general_columns`, `non_categorical_columns`, and `integer_columns` can be supplied through Dataset Specification metadata or action extras, but the adapter validates every referenced column.

Missing values are outside this adapter contract. Mean/mode imputation must be fitted on the training split through the explicit preprocessing layer before CTAB-GAN+ is called.

## Frozen Environment

The authoritative environment is Linux with Python 3.11 and:

- PyTorch 2.3.0 CPU;
- NumPy 1.26.4;
- pandas 2.2.3;
- scikit-learn 1.5.2;
- SciPy 1.13.1;
- six 1.17.0; and
- tqdm 4.66.5.

The upstream README lists Python-era dependencies such as PyTorch 1.9.1 and scikit-learn 0.24.1 that do not provide a supported Python 3.11 environment. The pinned official commit was executed without source modification against the versions above before this protocol was implemented. The frozen CI run is the authoritative compatibility claim.

`dython` is intentionally absent: it is imported only by the upstream standalone evaluation module, which is neither needed for generation nor used by the central benchmark evaluator.

## Frozen Cases

The mandatory protocol covers classification and regression. Each missing-free fixture has 40 rows, two numerical features, one categorical feature, and one target. Seeds 0, 19, and 73 yield six independent cases.

Each case performs two full one-epoch CPU training paths:

- native: direct construction, configuration, `fit`, pickle, synthesizer sampling, and inverse preprocessing with the official classes;
- adapter: standardized `train` and `sample` operations against the same verified source.

The bounded model uses batch size 8, latent dimension 8, four generator/discriminator channels, two eight-unit classifier layers, one thread, and 13 generated rows. These settings exercise the real WGAN-GP, preprocessing, downstream loss, serialization, and sampling code. They are validation fixtures, not benchmark-quality hyperparameters.

## Mandatory Pass Criteria

All six cases must pass. The protocol requires:

1. Linux/Python 3.11 and every frozen distribution version are exact;
2. all five official runtime files match the source manifest;
3. native and adapter generator tensors, preprocessing state, conditional generator, source frame, and configuration signatures are identical (raw pickle bytes are not compared because PyTorch serialization embeds object-specific storage identifiers);
4. native and adapter CSV sample bytes and DataFrames are identical;
5. artifact manifests and checkpoint/sample metadata are exact;
6. requested row count, column order, finite numerical values, categorical domains, and missing-value constraints are exact; and
7. adapter and native executions restore global NumPy state.

Any source, environment, metadata, checkpoint, sample, or state mismatch fails closed and retains a diagnostic JSON artifact.

## Known Boundaries

- One-epoch parity fixtures do not establish paper-level model quality.
- CPU parity does not claim byte-identical GPU training.
- The official preprocessing performs its own supervised split; benchmark dataset profiles must document this effective training subset.
- Arbitrary mixed/general/log column roles require dataset-specific review.
- Pickle is code-executing. The adapter rejects external or hash-mismatched checkpoints unless the existing explicit unsafe override and provenance review are used.
- Lack of an upstream license is an independent release blocker even when technical parity passes.

## Evidence Procedure

`.github/workflows/ctabgan-plus-validation.yml` downloads the locked source into a temporary runner directory, installs the frozen Linux/Python 3.11 CPU environment, runs the real six-case protocol, and retains the JSON artifact for 90 days. Any source, adapter, protocol, or frozen-environment change requires a new mandatory run and a new inspected evidence record.

## Retained Evidence

GitHub Actions [run `30926267432`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30926267432), job [`92049288002`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30926267432/job/92049288002), passed the protocol on Linux with Python 3.11.15 and PyTorch 2.3.0 CPU. It evaluated pull-request head `473af6334d6f367b75b35736370c4dfa6adf85bf` through GitHub's test merge commit `48837271b693b8af396f4f35cb68707b5c52e5bc`.

All six classification/regression and seed cases passed. Every case had an exact native/adapter checkpoint-state signature and byte-exact sample CSV, valid manifests and metadata, restored random state, 13 correctly ordered rows, no missing values, finite numerical values, and valid categorical domains. The run also verified all five official runtime files against source-manifest SHA-256 `a76cb5e64fec6d99aae2df2d66a51598bd72ae26bc7f2e0e3104bf5a1dc1652a`.

The inspected JSON is permanently retained at `docs/evidence/ctabgan-plus/native-parity-run-30926267432.json` with SHA-256 `df3bbf0dd46d34e8d57551048c7b7abe60340eddb3738e31d400e44344c5e5f2`. The corresponding GitHub artifact is ID `8899232990`, archive digest `sha256:f3abfc1e2bbd69d7858e2ce1e5b1ab0099e9b8fa06b95711a762dd16a06a2729`, expiring 2026-11-02. This retained result promotes only the adapter's validation level; CTAB-GAN+ remains `experimental`, `unsupported`, excluded from Official Results, and blocked from release while the upstream license and every independent admission gate remain unresolved.
