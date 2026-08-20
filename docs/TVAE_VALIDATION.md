# TVAE Validation Protocol

Status: passed; adapter is `native-parity-validated`; native-Windows GPU V2 functionality passed

Protocol ID: `tvae-native-parity-v2`

Supported validation platforms: Linux/Python 3.11/PyTorch 2.3 CPU for parity; native Windows/Python 3.11/PyTorch 2.8 CUDA for bounded V2 functionality

## Scope and claim boundary

This protocol tests whether the standardized TVAE adapter invokes `TVAE` from the checksum-pinned official `ctgan==0.12.1` package with the same data, constructor arguments, independently controlled training and sampling seeds, persistence API, and sample request as a direct native call. It covers package identity, real mixed-type training, save/load behavior, three deterministic seed pairs, artifact manifests, and exact native-versus-adapter comparisons.

A passing mandatory run may promote TVAE to `native-parity-validated`. It does not make TVAE `benchmark-eligible`, admit it to Official Results, or make it `release-supported`. Dataset admission, model-quality evaluation, privacy and fairness review, runtime thresholds, dependency maintenance, and release ownership remain separate gates.

## Authority, package identity, and license

- Method-author repository: `sdv-dev/CTGAN`.
- Exported synthesizer: `ctgan.TVAE` from `ctgan.synthesizers.tvae`.
- Release tag: `v0.12.1`.
- Git commit: `826da23f8f9385ad15fd206ecad691e04cb0ccdc`.
- Git tree: `164a4e877a6db2ca51b3cd7dbb22cbc18af536cb`.
- PyPI wheel: `ctgan-0.12.1-py3-none-any.whl`.
- Wheel SHA-256: `38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18`.
- License expression: `BUSL-1.1`.

PyPI records that the wheel was produced through trusted publishing from the same release commit. Before model execution, the protocol verifies the wheel hash and metadata, installed distribution version and license metadata, every hash-bearing installed `RECORD` entry, imported module location, exported TVAE class identity, and absence of the retired repository-local source entry points.

BUSL-1.1 is not an OSI open-source license. Version 0.12.1 permits non-production use and grants additional production use except for the restricted Synthetic Data Creation Service described by the upstream license. This repository installs the package optionally and does not vendor its 0.12.1 source. The protocol is research validation, not legal advice or permission to offer a service. Official-track and release decisions remain blocked pending an explicit license review.

## Retired legacy snapshot

The former `TabDDPM-main/CTGAN/` subtree contained 47 tracked files totaling 168,098 bytes. Its nested package declared version `0.5.2.dev0`, Python `<3.10`, PyTorch `<2`, and MIT. Comparison against the closest reviewed upstream history point, commit `ace3dbc4bd3ef7f4ddc027a1b47e8eb916378893`, found 39 exact files among 44 shared paths. TVAE and four other source files carried local changes; the TVAE source itself had SHA-256 `0b0bc0ed424f295084a395a212eb40f1464df0e6474c313e488e8ad43226689f`.

That snapshot was neither a strict official implementation nor suitable for the supported runtime. The entire subtree, including three obsolete TabDDPM wrapper scripts, has therefore been removed from the active working tree. Its history remains recoverable through Git, while the source lock permanently records its identity, comparison, and disposition.

## Adapter contract

The repository-owned adapter:

- imports `TVAE` from the exact official package version and rejects other versions;
- forwards the official `embedding_dim`, `compress_dims`, `decompress_dims`, `l2scale`, `batch_size`, `epochs`, `loss_factor`, and `verbose` controls;
- maps CPU or default-visible-GPU training through the official `enable_gpu` parameter without modifying upstream source;
- rejects non-default indexed CUDA requests because the official constructor does not expose an exact pre-fit device-index control;
- calls the official `set_random_state` before fitting;
- calls the official `set_random_state` again after loading and immediately before sampling, so generation is reproducible independently of the training seed;
- marks categorical features and a classification target as discrete columns;
- uses the official `save` and `load` APIs and sets the requested sampling device after loading;
- refuses to load a symlinked or external code-executing checkpoint unless the user explicitly accepts the existing unsafe-external-checkpoint override; and
- returns the requested row count in canonical dataset column order.

Official TVAE requires input with no missing values. Datasets with missing values must first use the repository preprocessing module, which fits numerical means and categorical modes on the training split only.

## Frozen environment

The workflow installs CPython 3.11, the official PyTorch 2.3.0 CPU wheel, the exact packages in `requirements-tvae-validation.txt`, and the CTGAN wheel only after verifying its SHA-256. The repository package is installed without optional dependency resolution. `pip check` must pass before validation starts. The protocol rejects any non-Linux platform, non-3.11 interpreter, or distribution-version mismatch.

Equivalent Linux setup commands are:

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tvae-validation.txt
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --dest /tmp/tvae-wheel "ctgan==0.12.1"
echo "38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18  /tmp/tvae-wheel/ctgan-0.12.1-py3-none-any.whl" | sha256sum --check
python -m pip install --no-deps /tmp/tvae-wheel/ctgan-0.12.1-py3-none-any.whl
python -m pip install --no-deps .
python -m pip check
```

## Frozen comparison

For training/sampling seed pairs `(0, 101)`, `(19, 7)`, and `(73, 29)`, both paths receive the same 40-row, no-missing-value mixed-type binary-classification fixture with two numerical features, one categorical feature, and one categorical target. Deliberately different seeds prove that generation does not silently reuse the post-training random stream. Both paths use one real training epoch, batch size 20, 16-dimensional embeddings, one 16-unit compression layer, one 16-unit decompression layer, CPU execution, and 12 requested samples. This bounded fixture tests execution and parity; it is not a model-quality benchmark.

The native path directly constructs the official `TVAE` class, applies the training seed, fits, saves, loads, applies the independent sampling seed through the official API, and samples. The adapter path performs the same operations through `TVAEAdapter`. Every seed pair must satisfy all of the following:

1. the wheel, installed package, license metadata, installed file hashes, and TVAE class identity match the lock;
2. constructor parameters and resolved CPU devices are identical;
3. all retained decoder keys and tensor values, including learned sigma values, are exactly equal and finite;
4. transformed fixture arrays are exactly equal;
5. NumPy and PyTorch model random states are exactly equal after sampling;
6. recorded per-batch loss values are exactly equal;
7. generated CSV bytes and reloaded DataFrames are exactly equal with 12 rows and canonical columns;
8. numerical values are finite, categorical values stay in observed domains, and no value is missing; and
9. standardized artifact manifests identify TVAE and its fixture correctly.

There is no numerical tolerance: deterministic parity must be exact.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.tvae \
  --repo-root . \
  --output-dir /tmp/tvae-validation \
  --evidence-path /tmp/tvae-evidence.json \
  --wheel-path /tmp/tvae-wheel/ctgan-0.12.1-py3-none-any.whl
```

`.github/workflows/tvae-validation.yml` runs this command and retains evidence for 90 days. Any package, dependency, adapter, or protocol change requires a new run.

## Retained result

[GitHub Actions run `32052308431`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32052308431) passed on Linux with Python 3.11.15 and PyTorch 2.3.0 CPU. All required comparisons passed for the independent training/sampling seed pairs `(0, 101)`, `(19, 7)`, and `(73, 29)`: official wheel and installed-file identity, constructor and device settings, all five retained decoder tensors including finite sigma, transformed fixture data, recorded losses, NumPy and PyTorch random states, manifest integrity, generated DataFrames, and CSV bytes.

The raw workflow artifact was inspected and retained unchanged at `docs/evidence/tvae/native-parity-run-32052308431.json` with SHA-256 `5c1a050af546b1b4fa0c7a7bd354430f34c130ca4d0f4c1875d42ae0ebd5fd7e`. Its GitHub artifact ID is `9295157218` and its artifact digest is `sha256:17fc9a204c3b3c4d59a5e34514d57a1b90cceb59a80df52d51d4836870c26462`. TVAE is therefore `native-parity-validated`, while remaining `experimental`, `unsupported`, and excluded from Official Results until the separate license, evaluation, dataset, and release gates are satisfied.

## Native-Windows V2 result

Repository commit `5bf59b0effa29a0c2694cdbe05b1a8f40443c481` passed `pipeline-v2-native-windows-v1` with `TVAE` from the unmodified official `ctgan==0.12.1` package on Windows 11, Python 3.11.15, PyTorch 2.8.0+cu128, CUDA 12.8, and NVIDIA GeForce RTX 5080. One bounded real epoch on the deterministic 256-row Adult-derived missing-free fixture was reused for generation seeds `17` and `29`. Each output contained 32 canonical, missing-free rows; both schemas were valid, their hashes differed, and sampling left the copied training artifacts unchanged. The seed-17 output finalized and validated a central `p3-validity` Result Bundle with zero pending files in the independently locked evaluation environment.

Evidence is retained at `docs/evidence/tvae/windows-v2-real-function-5bf59b0.json` with SHA-256 `a6bcb8eefd81141d5e0f485b9aecb50666d3490e97eb18ce11c6c2a170250301`. This bounded functionality result does not establish representative quality, privacy, Official Results admission, or release support; BUSL-1.1 review remains independent.
