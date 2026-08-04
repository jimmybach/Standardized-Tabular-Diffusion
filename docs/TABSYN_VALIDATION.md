# TabSyn Validation Protocol

Status: passed on Linux/Python 3.11; native parity validated

Protocol ID: `tabsyn-native-parity-v1`

Supported validation platform: Linux, Python 3.11, PyTorch 2.3 CPU

## Scope and claim boundary

This protocol tests whether the standardized TabSyn adapter invokes the pinned method-author implementation without changing the tracked VAE, latent-diffusion, decoding, or EDM sampler source. It covers fail-closed source integrity, real mixed-type VAE and diffusion training, sampling, three deterministic seed cases, artifact integrity, and exact native-versus-adapter comparisons.

A passing run promotes the adapter through `smoke-validated` to `native-parity-validated`. It does not make TabSyn `benchmark-eligible`, admit it to the Official Results track, or make it `release-supported`. Full-dataset quality, the central evaluation protocol, privacy and fairness review, runtime thresholds, broader task coverage, dependency maintenance, and release ownership remain independent gates.

## Source authority and patch disposition

- Method source: `amazon-science/tabsyn` at `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`.
- Method tree: `cb10c6da6e4b5c6f27261dfa0e4c593df9cc19ca`.
- Integrity manifest: `standardized_tabular_diffusion/resources/upstream/tabsyn-source-manifest.json`.

The repository previously carried local changes in six official files plus a local `zero` substitute. All six files were restored, and the substitute was removed. The frozen manifest now covers 20 files: the primary TabSyn entrypoint, VAE, diffusion, decoding and sampling implementation, shared data utilities, dependency declaration, and Apache license/NOTICE/readme. A mismatch fails before model execution. Bundled baseline implementations, upstream evaluation scripts, data, images, checkpoints, and generated outputs are outside this TabSyn-primary scope and require separate audits.

The upstream requirements file names `zero`, but TabSyn imports the research utility API distributed as `libzero`. The `zero` distribution is an unrelated circuit-analysis package. The official `libzero==0.0.8` wheel is checksum-recorded and installed without dependencies because its old metadata requires `torch<2`; NumPy, pynvml, PyTorch, and tqdm are instead locked explicitly. The parity workflow proves this resolution on PyTorch 2.3.

## Adapter contract

The repository-owned launcher imports and calls the official implementation and provides only invocation-level controls:

- `device="cpu"` maps to CPU, while `cuda` and `cuda:<index>` map to the requested CUDA device and fail clearly if it is unavailable;
- Python, NumPy, and PyTorch are seeded before the official modules execute;
- the official sampler receives the requested device, sampling steps, and sample-row count;
- VAE and diffusion training still use the official hard-coded training schedules and architecture;
- former local VAE/diffusion epoch controls are rejected because the official source does not expose them;
- explicit external checkpoint paths are rejected because official TabSyn uses a coupled fixed VAE/diffusion layout; and
- internal latent and PyTorch checkpoint paths must be regular, non-symlinked files inside the TabSyn worktree.

Sampling uses PyTorch serialization files, which can execute code during loading. Only checkpoints produced or deliberately placed inside the audited TabSyn worktree should be used, and their provenance must be verified before execution.

## Frozen environment

The workflow installs CPython 3.11, PyTorch 2.3.0 CPU, and the exact packages in `requirements-tabsyn-validation.txt`. The protocol fails closed if the observed core versions differ.

Equivalent local installation commands are:

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tabsyn-validation.txt
python -m pip install --no-deps "libzero==0.0.8"
python -m pip install --no-deps .
```

## Frozen comparison

For each seed in `0`, `19`, and `73`, the protocol creates two isolated copies from the verified manifest. Both receive the same no-missing-value mixed-type binary-classification fixture: 24 training rows, 12 test rows, two numerical features, one categorical feature, and one categorical target.

Official TabSyn hard-codes 4,000 VAE epochs, 10,001 diffusion epochs, four data-loader workers, a 1,024-wide diffusion MLP, and a CUDA-default sampler call. After source integrity succeeds, both disposable copies receive identical predeclared execution overrides: two VAE epochs, two diffusion epochs, zero workers, diffusion width 64, four sampling steps, 12 output rows, and explicit CPU device propagation. These bounded controls are never written to tracked upstream source. They make real CI execution feasible while preserving identical official functions and mathematics on both comparison paths. The fixture is an execution/parity case, not a model-quality benchmark.

The native path calls the official root `main.py` for VAE training, diffusion training, and sampling. A validation-only `sitecustomize.py` initializes the selected seed before the official entrypoint runs. The adapter path calls the repository launcher with the same seed and runtime controls. For every seed, all of the following must pass:

1. all 20 scoped source hashes match the pinned manifest;
2. the native and adapter runtime overrides are identical;
3. every tensor and key in the VAE model, encoder, decoder, best diffusion model, and epoch-zero diffusion model is exactly equal;
4. the complete latent embedding array is element-for-element equal and finite;
5. the final generated CSV is byte-for-byte equal;
6. exactly 12 rows with the expected four-column schema are generated;
7. all generated numerical values are finite; and
8. both standardized artifact manifests identify TabSyn correctly.

There is no numerical tolerance: deterministic parity is exact.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.tabsyn \
  --repo-root . \
  --output-dir /tmp/tabsyn-validation \
  --evidence-path /tmp/tabsyn-evidence.json
```

`.github/workflows/tabsyn-validation.yml` executes this command and retains the JSON evidence for 90 days. Any later source, dependency, adapter-command, or protocol change invalidates the record and requires another run.

The protocol passed in [GitHub Actions run 30871758645](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30871758645) at repository commit `54d419642842d7146d6afa4aa1b3d5167301c51c`. The retained artifact ID is `8878140935`, with digest `sha256:72e6488aa48357f03b685e101e0c218ef73fbefc40229963ca4eee80b9dca57c`. An exact permanent copy is stored at `docs/evidence/tabsyn/native-parity-run-30871758645.json` with file SHA-256 `3b74600a9c6d5e4e841cf56bd128ac7d17b70a6d186b48a3de78d8ca476d8089`.

Accordingly, TabSyn is `native-parity-validated` while remaining `experimental`, `unsupported`, and ineligible for Official Results until the separate dataset, central-evaluation, governance, runtime, and release gates pass.
