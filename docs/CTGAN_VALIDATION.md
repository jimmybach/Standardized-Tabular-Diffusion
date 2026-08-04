# CTGAN Validation Protocol

Status: passed on Linux/Python 3.11; adapter is `native-parity-validated`

Protocol ID: `ctgan-native-parity-v1`

Supported validation platform: Linux, Python 3.11, PyTorch 2.3 CPU

## Scope and claim boundary

This protocol tests whether the standardized CTGAN adapter invokes the checksum-pinned official `ctgan==0.12.1` package with the same data, constructor arguments, seed, persistence API, and sample request as a direct native call. It covers package identity, real mixed-type training, save/load behavior, three deterministic seed cases, artifact manifests, and exact native-versus-adapter comparisons.

A passing mandatory run may promote CTGAN to `native-parity-validated`. It does not make CTGAN `benchmark-eligible`, admit it to Official Results, or make it `release-supported`. Dataset admission, model-quality evaluation, privacy and fairness review, runtime thresholds, dependency maintenance, and release ownership remain separate gates.

## Authority, package identity, and license

- Method-author repository: `sdv-dev/CTGAN`.
- Release tag: `v0.12.1`.
- Git commit: `826da23f8f9385ad15fd206ecad691e04cb0ccdc`.
- Git tree: `164a4e877a6db2ca51b3cd7dbb22cbc18af536cb`.
- PyPI wheel: `ctgan-0.12.1-py3-none-any.whl`.
- Wheel SHA-256: `38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18`.
- License expression: `BUSL-1.1`.

PyPI records that the wheel was produced through trusted publishing from the same release commit. Before model execution, the protocol verifies the downloaded wheel hash, wheel metadata, installed distribution version, installed license metadata, every hash-bearing installed `RECORD` entry, and the imported module location. Importing repository-local CTGAN source fails the protocol.

BUSL-1.1 is not an OSI open-source license. Version 0.12.1 permits non-production use and grants additional production use except for the restricted Synthetic Data Creation Service described by the upstream license. This repository does not vendor version 0.12.1; it installs it as an optional package. The protocol is research validation, not legal advice or permission to offer a service. Official-track and release decisions remain blocked pending an explicit license review.

## Legacy snapshot disposition

The repository previously contained an older, locally modified CTGAN `0.5.2.dev0` snapshot under `TabDDPM-main/CTGAN/`. After TVAE was also migrated to the official package, the 47-file legacy subtree and its obsolete wrappers were removed from the active working tree. Its history remains recoverable through Git, and the source lock records its version, closest reviewed upstream relation, modification status, size, and hashes.

## Adapter contract

The repository-owned adapter:

- imports `CTGAN` from the exact official package version and rejects other versions;
- maps the benchmark CPU/GPU request to the official `enable_gpu` and `set_device` interfaces;
- forwards documented constructor controls without changing upstream source;
- calls the official `set_random_state` before fitting;
- marks categorical features and a classification target as discrete columns;
- uses the official `save` and `load` APIs;
- refuses to load a symlinked or external code-executing checkpoint unless the user explicitly accepts the existing unsafe-external-checkpoint override; and
- returns the requested row count in canonical dataset column order.

Official CTGAN requires input with no missing values. Datasets with missing values must first use the repository preprocessing module, which fits numerical means and categorical modes on the training split only.

## Frozen environment

The workflow installs CPython 3.11, the official PyTorch 2.3.0 CPU wheel, the exact packages in `requirements-ctgan-validation.txt`, and the CTGAN wheel only after verifying its SHA-256. The repository package is installed without optional dependency resolution. `pip check` must pass before validation starts. The protocol rejects any non-Linux platform, non-3.11 interpreter, or distribution-version mismatch.

Equivalent Linux setup commands are:

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ctgan-validation.txt
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest /tmp/ctgan-wheel "ctgan==0.12.1"
echo "38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18  /tmp/ctgan-wheel/ctgan-0.12.1-py3-none-any.whl" | sha256sum --check
python -m pip install --no-deps /tmp/ctgan-wheel/ctgan-0.12.1-py3-none-any.whl
python -m pip install --no-deps .
python -m pip check
```

## Frozen comparison

For seeds `0`, `19`, and `73`, both paths receive the same 40-row, no-missing-value mixed-type binary-classification fixture with two numerical features, one categorical feature, and one categorical target. Both use one real training epoch, batch size 20, PacGAN group size 10, 16-dimensional embeddings, 16-wide generator/discriminator layers, CPU execution, and 12 requested samples. This bounded fixture tests execution and parity; it is not a quality benchmark.

The native path directly constructs, seeds, fits, saves, loads, and samples the official `CTGAN` class. The adapter path performs the same operations through `CTGANAdapter`. Every seed must satisfy all of the following:

1. the wheel, installed package, license metadata, and installed file hashes match the lock;
2. constructor parameters are identical;
3. all retained generator keys and tensor values are exactly equal and finite;
4. transformed fixture arrays and data-sampler state are exactly equal;
5. NumPy and PyTorch model random states are exactly equal after sampling;
6. recorded generator/discriminator loss values are exactly equal;
7. generated CSV bytes and reloaded DataFrames are exactly equal with 12 rows and canonical columns;
8. generated numerical values are finite, categorical values stay in the observed domains, and no value is missing; and
9. standardized artifact manifests identify CTGAN and its fixture correctly.

There is no numerical tolerance: deterministic parity must be exact.

## Retained result

GitHub Actions run [`30910275922`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30910275922) passed this protocol for all three seed cases on Linux with Python 3.11.15 and PyTorch 2.3.0 CPU. The run verified 20 hash-bearing installed-package records and produced byte-exact native/adapter sample CSV files for every seed. The inspected JSON is permanently retained at `docs/evidence/ctgan/native-parity-run-30910275922.json` with SHA-256 `748501c8671c272a1e5d54c85fdb6550182d0e5578d550a3ca7681cc712f4570`.

The corresponding GitHub artifact is `8892774473`, with artifact digest `sha256:96ba5b9dde6eed95fa1972990c6a4231e43f204bdcb99a4a0d7837099ff3b71a`. The permanent repository copy remains authoritative after the temporary artifact expires.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.ctgan \
  --repo-root . \
  --output-dir /tmp/ctgan-validation \
  --evidence-path /tmp/ctgan-evidence.json \
  --wheel-path /tmp/ctgan-wheel/ctgan-0.12.1-py3-none-any.whl
```

`.github/workflows/ctgan-validation.yml` runs this command and retains the workflow artifact for 90 days. Any package, dependency, adapter, or protocol change requires a new run. The inspected passing evidence promotes CTGAN to `native-parity-validated`; it remains `experimental`, `unsupported`, and excluded from Official Results pending every separate gate described above.
