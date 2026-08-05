# NFlow Validation Protocol

Status: protocol implemented; authoritative Linux/Python 3.11 evidence pending

Protocol ID: `nflows-maf-tabular-recipe-parity-v1`

Supported validation platform: Linux, Python 3.11, CPU

## Scope and claim boundary

`nflows` is a general normalizing-flow library, not a paper-native tabular synthesizer. This repository's `nflow` model is therefore a declared benchmark recipe built from unchanged official `nflows==0.14` classes. It standardizes numerical variables, ordinal-encodes categorical variables, trains a masked affine autoregressive flow (MAF), samples continuous latent outputs, and rounds/clips categorical coordinates back to their train-fitted domains.

A passing authoritative run may promote `nflow` to `native-parity-validated` only for the exact official-package-plus-declared-recipe target. It does not reproduce a paper table, establish equivalence with Neural Spline Flows or another tabular-flow method, validate alternative flow architectures or categorical representations, make the model `benchmark-eligible`, admit it to Official Results, or make it `release-supported`.

## Source authority, package identity, and license

- Canonical library repository: `bayesiains/nflows`.
- PyPI package: `nflows==0.14`.
- Official source distribution: `nflows-0.14.tar.gz`.
- Source-distribution SHA-256: `6299844a62f9999fcdf2d95cb2d01c091a50136bd17826e303aba646b2d11b55`.
- Source-distribution size: 45,784 bytes.
- Git tag: lightweight tag `v0.14`.
- Locked commit: `64b856c081e5f07521b32be99da262e8338fbfe8`.
- Locked tree: `83057958f8773e35044e3aa5c13ac9c06c4a3994`.
- License: MIT; source-tag `LICENSE.md` SHA-256 `74a24abd8e13ac55286f5a8396a88c20da9f67a64cbc5daa8999f31843a8b948`.

Version 0.14 was uploaded to PyPI on 2020-12-02 without Trusted Publishing and without a detached distribution signature. The lightweight tag and its commit are also unsigned. The immutable PyPI digest, commit/tree lock, and byte comparison are recorded instead of claiming cryptographic publisher provenance.

The protocol verifies all 96 archive members, all 80 regular files, package metadata, dependency declarations, nine critical runtime files, and a deterministic aggregate over all 42 `nflows/` package files before execution. Those 42 package files are byte-exact with the locked Git tree and are verified again in the installed distribution and after all model cases.

The official Git tag contains an MIT `LICENSE.md`, but the PyPI source distribution omits that file and only declares `License: MIT` in metadata. The repository does not vendor or modify the package. This packaging omission is explicitly recorded and must not be misrepresented as an in-package license-file verification.

## Declared preprocessing and model recipe

The adapter assigns one flow coordinate to every canonical table column:

- declared numerical features and a regression target are standardized using train-fitted `StandardScaler` means and scales;
- declared categorical features and a classification target are converted to strings and ordinal-encoded with train-fitted, lexicographically ordered categories; and
- sampled categorical coordinates are rounded to the nearest integer, clipped to the observed category range, and decoded.

The default production recipe consists of four repetitions of:

1. official `RandomPermutation`; then
2. official `MaskedAffineAutoregressiveTransform` with 64 hidden features, two residual blocks, ReLU, deterministic masks, no context, no dropout, and no batch normalization.

The base distribution is official `StandardNormal`. Training uses float32, Adam (`lr=1e-3`, betas 0.9/0.999, epsilon `1e-8`, no weight decay, no AMSGrad), shuffled deterministic batches of 512, ten epochs, no workers, one CPU thread, and the standardized seed. The validation fixture uses the same declared family with two layers, 16 hidden features, one residual block, batch size 16, and three epochs to keep the nine authoritative comparisons bounded. Unsupported architectural substitutions fail closed and require a separately identified protocol.

This ordinal continuous representation is a benchmark recipe choice, not a native discrete-flow mechanism. Its statistical quality must be assessed by the central evaluator before leaderboard use.

## Adapter boundary and input contract

The adapter does not modify official package source. It supplies only the standardized boundaries needed by this repository:

- exactly one classification or regression target;
- unique canonical columns with complete numerical/categorical role assignment and no overlaps;
- strict rejection of missing and non-finite values until the explicit train-fitted benchmark preprocessing module has run;
- CPU-only, single-threaded deterministic execution for the validated path;
- exact requested row count and canonical column order;
- finite numerical output and categorical-domain checks; and
- restoration of the caller's process-global PyTorch random state and thread count.

## Safe persistence boundary

The former `model.pkl` checkpoint has been removed. The adapter now writes:

- `model.nflow.json`, containing package identity, recipe, training-frame fingerprint, preprocessing statistics, category levels, training losses, tensor manifest, and conservative privacy declarations; and
- `model.nflow.weights.npz`, containing only non-object NumPy arrays reconstructed into the declared official architecture.

Loading rejects symlinks, oversized files, path traversal, encrypted or duplicate archive entries, unexpected tensor names, object dtypes, wrong shapes/dtypes, non-finite tensors, and any file/tensor hash mismatch. NumPy is always invoked with `allow_pickle=False`; official `Flow` objects are reconstructed from the locked recipe before strict state loading. Reviewed external checkpoints can therefore be loaded without the unsafe-executable-checkpoint override.

The files omit row-level training data and executable Python objects. This is data minimization, not a privacy guarantee: scaling statistics, category domains, losses, and trained weights may disclose information or memorize records. Trained artifacts still require access control and retention review.

## Frozen environment

The workflow uses CPython 3.11, the CPU-only PyTorch 2.3.0 build, and the exact dependency set in `requirements-nflow-validation.txt`. It downloads the official source distribution over HTTPS, verifies its SHA-256 before building, installs it without dependency resolution or build isolation, installs this repository without dependency resolution, and requires `pip check` to pass.

Equivalent Linux commands are:

```bash
python -m pip install --upgrade "pip==25.1.1" "setuptools==80.10.2"
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-nflow-validation.txt
curl --fail --location --proto '=https' --tlsv1.2 \
  --output /tmp/nflows-0.14.tar.gz \
  "https://files.pythonhosted.org/packages/bd/16/a484db41aab28332f42080435c9342fa87cfc9a4fce5495521ea1e80ca27/nflows-0.14.tar.gz"
echo "6299844a62f9999fcdf2d95cb2d01c091a50136bd17826e303aba646b2d11b55  /tmp/nflows-0.14.tar.gz" | sha256sum --check
python -m pip install --no-deps --no-build-isolation --use-pep517 /tmp/nflows-0.14.tar.gz
python -m pip install --no-deps .
python -m pip check
```

## Frozen comparison

The protocol covers binary classification, multiclass classification, and regression. Each missing-free fixture has 48 rows, a correlated numerical column, a constant numerical column, one categorical column, and one target. Each variant runs with seeds `0`, `19`, and `73` and requests 13 rows, producing nine cases.

For every case, the direct path independently preprocesses the persisted canonical CSV and directly constructs, trains, and samples the official classes. The adapter path trains, serializes, reloads, and samples through `NFlowAdapter`. Every case must establish all of the following exactly:

1. Train-fitted preprocessing state and epoch losses match.
2. Every official model state tensor name and byte value matches.
3. Reloaded raw continuous samples, final frames, and CSV bytes match.
4. Artifact manifests and checkpoint/sample hashes are valid.
5. JSON/NumPy persistence is non-executable, omits row-level training data, makes no privacy overclaim, and requires trained-artifact access control.
6. Both paths restore the caller's global PyTorch state and thread count.
7. Output has the requested rows and canonical columns, finite numerical values, valid categorical domains, and no missing values.
8. The installed official package remains unchanged after validation.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.nflow \
  --repo-root . \
  --output-dir /tmp/nflow-validation \
  --evidence-path /tmp/nflow-evidence.json \
  --sdist-path /tmp/nflows-0.14.tar.gz
```

`.github/workflows/nflow-validation.yml` runs this command and retains its JSON artifact for 90 days. Any package, dependency, adapter, checkpoint schema, preprocessing, architecture, optimizer, or protocol change requires a new run. Promotion is permitted only after a passing Linux/Python 3.11 artifact is inspected and retained unchanged under `docs/evidence/nflow/`.

## Current result

The protocol and workflow are implemented, but no authoritative artifact has yet been retained. `nflow` therefore remains `adapter-complete`, `experimental`, `unsupported`, and excluded from Official Results until the required run passes and its evidence is reviewed and committed.
