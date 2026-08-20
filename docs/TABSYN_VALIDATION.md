# TabSyn Validation Protocol

Status: passed on Linux/Python 3.11 with native parity validated; native-Windows minimal-real functionality passed

Protocol ID: `tabsyn-native-parity-v2`

Supported validation platform: Linux, Python 3.11, PyTorch 2.3 CPU

## Scope and claim boundary

This protocol tests whether the standardized TabSyn adapter invokes the pinned method-author implementation without changing the tracked VAE, latent-diffusion, decoding, or EDM sampler source. It covers fail-closed source integrity, real mixed-type VAE and diffusion training, sampling, three deterministic seed cases, artifact integrity, and exact native-versus-adapter comparisons.

A passing run promotes the adapter through `smoke-validated` to `native-parity-validated`. It does not make TabSyn `benchmark-eligible`, admit it to the Official Results track, or make it `release-supported`. Full-dataset quality, privacy and fairness review, runtime thresholds, broader task coverage, dependency maintenance, and release ownership remain independent gates.

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
- internal latent and PyTorch checkpoint paths must be regular, non-symlinked files inside the declared run-owned `tabsyn-runtime` directory.

PyTorch 2.8 removed the logging-only `verbose` parameter from `ReduceLROnPlateau`, while the frozen VAE and diffusion entrypoints still pass it. The isolated launcher drops only that keyword on the V2 runtime, forwards every mathematical scheduler argument unchanged, restores the original class after each official call, and never edits the checksum-locked upstream files.

The official inverse transform may return fractional values for columns that the admitted DatasetSpec explicitly declares as integers. At the standardized output boundary, TabSyn uses the shared `numpy.rint` nearest-integer policy without clipping. If any value changes, the byte-exact official CSV is retained as `tabsyn-native-samples.csv`, while the canonical table and `tabsyn-sample-metadata.json` record every changed-row count and both file digests.

The official VAE and diffusion entrypoints hard-code four DataLoader workers. Recreating those workers in every one of 4,000 VAE epochs causes extreme process-spawn overhead on Windows without changing model mathematics. The V2 preset therefore selects `num_workers=0` through a scoped DataLoader-construction bridge already used by the native-parity protocol. Batch size, shuffling, all 4,000/10,001 epoch limits, losses, optimizer, scheduler, and checkpoints remain official and unchanged; the original DataLoader class is restored after each stage.

Sampling uses PyTorch serialization files, which can execute code during loading. Only checkpoints produced inside the declared run-owned workspace should be used, and their provenance and recorded digests must be verified before execution.

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

Protocol V2 runs adapter training and sampling in one shared run-owned `output_dir`, exactly as the public pipeline does. The training artifact manifest is snapshotted before the sample action updates the run manifest. All adapter checkpoints are then compared from `output_dir/tabsyn-runtime`; no mutable model artifact is read from or written to the verified upstream source checkout.

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

Protocol V2 passed in [GitHub Actions run 32055783087](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32055783087) at repository commit `4668853b5d0acf7bff779453fb1a6e67f5838384`. The retained artifact ID is `9296369638`, with digest `sha256:1e5a8a6563a4ee05cbd77ad5567ff07f9dd50e5503d3d8198a4583705f6a8920`. An exact permanent copy is stored at `docs/evidence/tabsyn/native-parity-run-32055783087.json` with file SHA-256 `8cbfa66a57b99e5f9fdb0381b21b02eb9f5b062a4f8e4f1ef13de24be3862e48`.

Accordingly, TabSyn is `native-parity-validated` while remaining `experimental`, `unsupported`, and ineligible for Official Results until the separate dataset, governance, runtime, and release gates pass.

## Native-Windows Minimal-Real Evidence

The separate `pipeline-v2-native-windows-v1` run at repository commit `6b3f2bca50d79d5e59bb22b798eb8cb0a6a9f8f7` executed the checksum-exact official VAE, latent-diffusion, sampler, and decoding chain on the deterministic 256-row Adult-derived fixture. It used the source's native training schedules, changing only the data-loader worker count to zero at the adapter-only Windows boundary. The observed environment was native Windows 11, Python 3.11.15, PyTorch 2.8.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 5080.

Seeds 17 and 29 each generated 32 valid, missing-free rows; the outputs differed, training artifacts remained immutable, and the independently locked central environment finalized and validated `p3-validity`. The environment retained the already reviewed `libzero==0.0.8` stale `torch<2` metadata waiver; no official source was modified. This proves bounded Windows functionality, not representative-scale quality or Official Results eligibility. The permanent record is `docs/evidence/tabsyn/windows-v2-real-function-6b3f2bc.json`, SHA-256 `8f2c21d38c64484b019d43995d79c7a5d9cc837a1ae3c22d061db3b361758ccf`.
