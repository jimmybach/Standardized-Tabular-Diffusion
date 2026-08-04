# Goggle Source and Validation Record

Status: native parity validated; benchmark and release gates remain pending<br>
Protocol: `goggle-method-author-native-parity-v1`<br>
Official environment: Linux and Python 3.11

## Scope

This record defines what the repository means by `goggle`, which source is authoritative, what the adapter is allowed to change, and what evidence is required before the status can advance. It does not make the adapter benchmark-eligible or release-supported.

The reproduction target is the method-author implementation for the ICLR 2023 paper *GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure*:

- repository: `https://github.com/vanderschaarlab/GOGGLE`;
- commit: `1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0`;
- repository tree: `2d6a54f6d6f4d156890bf4e035119dbb483a46d0`;
- `src/goggle` tree: `6dcaae801859f63e173537445548a50cd1f8625b`;
- license: MIT, copyright 2023 Tennison Liu; and
- locked archive SHA-256: `62dc6c98a2067d950513b4fe6343715f03a6a096990241fc6143b18fb56aaf65`.

The source manifest freezes 18 files: the license, authorship, README, declared environment and packaging files, and all runtime Python modules. The materializer verifies the archive size and digest, extracts only those paths, normalizes text under the declared rule, and then verifies every file again. Source is stored in an ignored cache, not committed as a mutable local copy.

## Retired Snapshot

The former `TabSyn-main/baselines/goggle` directory was not the method-author original. It contained 11 files; all nine paths shared with the official package differed after text normalization. Material changes included:

- a different `fit` signature and externally supplied data loader;
- different encoder/decoder widths and batch-size defaults;
- removal of the official validation split and early-stopping behavior;
- changed checkpoint placement and sampling constraints; and
- changes to graph-decoder and RGCN imports.

That snapshot has been removed from the current tree without rewriting Git history. It is not used as native evidence and is not described as an official implementation.

## Adapter Boundary

The adapter calls the untouched official `GoggleModel.fit` method. The official model, graph learner, encoder, decoder, loss, optimizer alternation, seeded train/validation split, validation selection, early stopping, and state-dict serialization remain in upstream code.

Five operations remain outside upstream source:

1. **Source and artifact safety.** Source bytes are verified before and after execution. Training is run with `output_dir` as the working directory so the official `tmp/<dataset>.pt` write cannot touch tracked source. The resulting state dict is moved to `output_dir/model.pt` without changing tensor content. Sampling uses `torch.load(..., weights_only=True)` and rejects unapproved external checkpoints.
2. **Numeric table contract.** Official experiments pass a finite numeric `DataFrame`. Numerical feature transforms are fitted on the real training split only using population standardization, equivalent to `StandardScaler`. Categorical feature transforms are fitted on the same split and one-hot encoded deterministically. The single target remains part of the joint modeled vector.
3. **Output contract.** The requested positive row count is passed directly to the unchanged `Goggle.model.sample` core. Numerical values are inverse-standardized, one-hot blocks use argmax over training categories, and classification targets are mapped to the nearest recorded class code. This replaces the method's dependency on an arbitrary reference-frame row count with an explicit interface contract.
4. **Legacy Synthcity import boundary.** The official module eagerly imports Synthcity 0.2.2 metrics and `Schema`, although neither `fit` nor core sampling executes them. The Python 3.11 adapter supplies only those import names and fails if `Schema` is instantiated. Formal benchmark evaluation always uses the central versioned evaluator.
5. **Unused RGCN import boundary.** Official `GraphDecoder.py` imports `RGCNConv` even when the default homogeneous GCN/SAGE decoder is used. If `torch-sparse` is absent, the unused symbol becomes a fail-on-instantiation placeholder. The validated GCN path never constructs it. `decoder_arch="het"` still requires the official compiled extension stack and remains outside the validated claim.

There are no upstream patch files and no modified official executable statements.

## Data Contract

The adapter accepts classification and regression tables with:

- any non-empty combination of numerical and categorical feature columns;
- exactly one target column;
- a non-empty real training CSV whose columns exactly match the canonical order;
- finite numerical values; and
- no missing values.

Missing values fail closed. Users must first run the centralized imputer, which fits numerical means and categorical modes only on the real training split. The adapter does not silently impute.

The model jointly synthesizes features and the target. The preprocessing metadata, training configuration, source identity, checkpoint digest, and transform fit scope are written to `goggle-model-metadata.json`. Sampling requires that metadata and the hashed runtime configuration; checkpoint or configuration tampering is rejected.

## Supported Controls

The public controls expose official constructor and fit parameters: encoder/decoder widths and layers, heterogeneous node encoding, GCN/SAGE/heterogeneous decoder selection, graph threshold, graph prior and mask, KL and graph-loss weights, iterative optimizer selection, learning rate, weight decay, epochs, batch size, patience, logging interval, seed, device, and thread count.

Defaults match the method-author source: 64-wide encoder and decoder, two layers, GCN, threshold 0.1, `alpha=beta=0.1`, iterative optimization, learning rate 0.005, weight decay 0.001, 1,000 epochs, batch size 32, patience 50, and logging interval 100. Unknown controls, invalid ranges, non-square priors, and non-binary masks fail before execution.

## Formal Parity Protocol

The mandatory workflow installs a frozen CPU environment with PyTorch 2.3.0, DGL 1.1.3, torch-geometric 2.5.3, NumPy 1.26.4, pandas 2.2.3, and scikit-learn 1.5.2. It then materializes and verifies two isolated copies of the locked official source for each case.

Nine cases cover:

- binary classification, multiclass classification, and regression; and
- seeds 0, 19, and 73.

Each case uses 12 mixed-type training rows and requests seven samples. The independent native path calls official `GoggleModel.fit` and `Goggle.model.sample` directly. The standardized path invokes the public adapter with identical transformed input and effective model configuration. A case passes only when all of the following hold:

- every checkpoint key, shape, dtype, and tensor value is exact;
- raw core sample arrays are exact;
- final sample frames and CSV bytes are exact;
- requested row count and canonical column order are exact;
- numerical output is finite and output contains no missing values;
- adapter metadata describes the locked source and effective configuration exactly;
- checkpoints remain outside source; and
- all 18 source files still match the manifest after execution.

All nine cases passed exactly in GitHub Actions run [`30945676747`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30945676747). The inspected JSON was downloaded and committed byte-for-byte as `docs/evidence/goggle/native-parity-run-30945676747.json` with SHA-256 `1dbcf50194505820cac0650ba72d519f4f331008bbcaac635f8eb846bec7da59`. The workflow artifact remains available for 90 days; the permanent repository copy is the long-term evidence record.

## Usage

Materialize the locked source once:

```bash
python -m standardized_tabular_diffusion.cli materialize-model-source --model goggle
python -m standardized_tabular_diffusion.cli model-source-status --model goggle
```

Install the Linux/Python 3.11 runtime and run the smoke preset:

```bash
python -m pip install "standardized-tabular-diffusion[goggle]"
python -m standardized_tabular_diffusion.cli run --config configs/smoke/goggle-adult-smoke.json
```

Run the formal protocol:

```bash
python -m standardized_tabular_diffusion.validation.goggle \
  --repo-root . \
  --output-dir /tmp/goggle-validation \
  --evidence-path /tmp/goggle-evidence.json
```

## Remaining Gates

Even after exact native parity passes, Goggle remains `experimental` and `unsupported`. Benchmark eligibility separately requires the frozen central evaluation protocol, approved dataset profiles, resource-policy qualification, representative-scale runs, and an explicit decision about the unvalidated SAGE/heterogeneous decoder paths. Release support additionally requires packaging, installation, security, documentation, and long-term maintenance review.
