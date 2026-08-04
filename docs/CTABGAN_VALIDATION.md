# CTAB-GAN Validation

Status: protocol implemented; authoritative Linux/Python 3.11 run pending

## Claim boundary

This record validates the standardized `ctab-gan` adapter against the method-author CTAB-GAN implementation. It does not admit CTAB-GAN to Official Results and does not make it release-supported. Dataset admission, the central evaluation protocol, full-scale runtime qualification, governance review, and release testing remain independent gates.

The supported path is classification only. The official `DataPrep` implementation always performs a stratified supervised train/test split, so presenting its regression label as a verified regression interface would overstate the source behavior.

## Official source identity

- Repository: `https://github.com/Team-TUD/CTAB-GAN`
- Commit: `73d4e315a2a51cf16c97ed8a00d2dad456cfce8a`
- Commit tree: `3ef0223477193400d88344ff66b7ac6ffeefa173`
- `model/` tree: `89ad16bce9f0f6c23f393d9b6b2959ce8ef64bf9`
- License: Apache-2.0

The upstream repository has no tag or release. The latest method-author commit is therefore pinned by immutable commit and tree identities. The codeload archive length and SHA-256, every selected file's Git blob, normalized byte length, and normalized SHA-256 are frozen in `standardized_tabular_diffusion/resources/upstream/ctabgan-source-manifest.json`.

Seven selected files are distributed under `TabDDPM-main/CTAB-GAN/`: the two upstream license/attribution files, upstream README, and four files required for generation. Upstream datasets, generated CSVs, notebooks, and the upstream evaluation module are intentionally excluded. Formal benchmark metrics come only from the central evaluator.

The selected source is exact after the repository's declared text normalization: CRLF and LF are normalized to LF and the file is canonicalized to one final newline before hashing. No executable statement is changed.

## Legacy snapshot disposition

The initial repository import contained 15 CTAB-GAN files (78,185 Git-blob bytes). Against the pinned official tree, only two of nine shared paths matched after line-ending normalization; seven differed. The fork changed the public constructor, bypassed the official stratified split, exposed different optimization/device controls, and implemented different sample-count and seed behavior.

That semantic fork has been removed from the active tree without rewriting Git history. Dataset-specific `columns.json`, local train/tune/pipeline wrappers, the modified evaluator, and the local `model/__init__.py` are not part of the supported implementation. The remaining selected files come from the pinned method-author revision and retain its licenses and attribution.

## Python 3.11 compatibility boundary

The official dependency list targets scikit-learn 0.24.1, which is not a supported Python 3.11 environment. The official transformer calls:

```python
BayesianGaussianMixture(self.n_clusters, ...)
```

In scikit-learn 1.5.2, `n_components` is the same parameter but is keyword-only. The adapter therefore installs the runtime bridge `ctabgan-sklearn-keyword-only-v1`, which forwards the unchanged integer as `n_components=self.n_clusters`. The bridge changes no estimator, parameter value, source file, or training operation. Both native and adapter sides of the protocol install this bridge independently. Any future compatibility issue that changes algorithm semantics requires separate review and a new protocol version.

## Adapter contract

The adapter:

1. validates all seven selected source files before import;
2. isolates the generic upstream `model` namespace;
3. preserves the official default `test_ratio=0.2` unless an explicit valid ratio is supplied;
4. derives categorical and integer roles from the Dataset Profile while permitting explicit reviewed overrides;
5. requires exactly one classification target and includes it in the categorical role;
6. rejects missing values until the training-split-fitted preprocessing module has run;
7. seeds Python, NumPy, PyTorch, and CUDA where available, then restores the caller's RNG and thread state;
8. serializes the official model class and binds its checkpoint to source, environment, configuration, compatibility-shim, and checksum metadata;
9. supports an explicit sample count by calling the official synthesizer and official inverse preprocessing; and
10. accepts pickle checkpoints only through the repository's trusted-artifact boundary.

## Frozen parity protocol

Protocol `ctabgan-native-parity-v1` contains six real training-and-sampling cases:

- variants: balanced binary classification and four-class classification;
- seeds: `0`, `19`, and `73`;
- source rows: `40`;
- columns: continuous, integer, categorical, and target;
- generated rows: `13`;
- training: one epoch, batch size 8, random dimension 8, four channels, classifier dimensions `[8, 8]`, and one CPU thread.

For each case, the native side constructs and trains the official class directly. The adapter side independently enters the standardized train/sample contract. A case passes only when all of the following hold:

- the selected source identity and seven file checksums validate;
- adapter manifests and checkpoint/sample metadata are complete;
- generator tensor names, shapes, dtypes, and bytes match exactly;
- serialized preprocessing, transformer, and conditional-generator signatures match exactly;
- source frame and effective configuration signatures match exactly;
- native and adapter sample CSV bytes and parsed frames match exactly;
- requested row count and column order are exact;
- numerical values are finite, categorical values remain in source domains, and no missing values appear; and
- caller NumPy state is unchanged after both paths.

Pickle file bytes are recorded but are not compared because Python/PyTorch serialization can include non-semantic object identity details. The model-state comparison is semantic and byte-exact at the tensor and fitted-component level.

## Frozen environment

The authoritative workflow uses Linux, Python 3.11, CPU PyTorch 2.3.0, and `requirements-ctabgan-validation.txt`. It rejects any listed version drift. Local Windows execution is useful diagnostic evidence only and cannot promote the validation state.

Run the integrity and protocol checks with:

```bash
python -m standardized_tabular_diffusion.cli model-source-status \
  --model ctab-gan \
  --source-dir TabDDPM-main/CTAB-GAN

python -m standardized_tabular_diffusion.validation.ctabgan \
  --source-dir TabDDPM-main/CTAB-GAN \
  --output-dir /tmp/ctabgan-validation \
  --evidence-path /tmp/ctabgan-evidence.json
```

`.github/workflows/ctabgan-validation.yml` runs the authoritative protocol and retains its JSON evidence for 90 days. Source, adapter, compatibility, protocol, or environment changes invalidate prior evidence and require a new inspected run.

## Current decision

The complete local six-case protocol passes under Python 3.11 and the frozen dependency versions. This is implementation evidence, not the authoritative platform claim. The registry remains `adapter-complete`, `experimental`, and `unsupported` until the Linux artifact is inspected and permanently retained. Even after native-parity promotion, Official Results and release support remain separately blocked.
