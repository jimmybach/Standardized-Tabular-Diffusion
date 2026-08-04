# TabDiff Validation Protocol

Status: candidate protocol; Linux/Python 3.11 execution pending

Protocol ID: `tabdiff-native-parity-v1`

Supported validation platform: Linux, Python 3.11, CPU

## Scope and claim boundary

This protocol tests whether the standardized TabDiff adapter invokes the pinned method-author implementation without changing training, sampling, or deterministic outputs. It covers source integrity, a real mixed-type train-and-sample smoke run, native-versus-adapter parity, artifact integrity, and explicit handling of the upstream seed limitation.

A passing run may promote the adapter through `smoke-validated` to `native-parity-validated`. It does not make TabDiff `benchmark-eligible`, admit it to the Official Results track, or make it `release-supported`. Dataset admission, central metric validation, privacy and fairness review, dependency maintenance, and release ownership remain independent gates.

## Source authority and evaluator disposition

- Method source: `MinkaiXu/TabDiff` at `5ecdb3356261aea72716cc9a779f31d7ad083bf4`.
- Method tree: `052a505cb1fbee5cbc705eeb0717d90d706ffb91`.
- Integrity manifest: `standardized_tabular_diffusion/resources/upstream/tabdiff-source-manifest.json`.

The repository previously carried a semantic modification to `eval/mle/mle.py`. That patch changed estimator settings, device behavior, objectives, randomness, error handling, and edge-case metric semantics. It has been removed, and the pinned official file has been restored exactly after canonical line-ending normalization. All 27 files in the frozen source scope now match the pinned method-author source. A mismatch fails before model execution.

The restored upstream evaluator is retained for source fidelity and for exercising the native runtime. It is not the authority for formal leaderboard metrics. Official benchmark results must use the repository's separately versioned and reviewed central evaluation engine.

## Adapter contract

The adapter makes only invocation-level mappings:

- `device="cpu"` maps to the official `--gpu -1` option;
- `cuda` and `cuda:<index>` map to the corresponding official GPU index;
- Weights & Biases is disabled by default for controlled runs;
- the official `--deterministic` option is enabled by default; and
- explicit PyTorch checkpoints outside the artifact directory require an affirmative trust override because loading them can execute code.

The pinned CLI does not expose a configurable seed. Its deterministic mode fixes Python, NumPy, and PyTorch to seed 0. The adapter therefore rejects nonzero `RunSpec.seed` values instead of silently ignoring them. This protocol validates seed 0 and makes no multi-seed or configurable-seed claim. Adding configurable seeds would require a separately reviewed upstream-source change.

## Frozen environment

The workflow installs CPython 3.11, PyTorch 2.3.0 CPU, and the exact packages in `requirements-tabdiff-validation.txt`. This is a supported benchmark validation environment, not a claim that the upstream project's original Python 3.10/CUDA 11.7 environment was byte-identically recreated.

Equivalent local installation commands are:

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tabdiff-validation.txt
python -m pip install --no-deps .
```

## Frozen comparison

The protocol creates two isolated copies from the verified source manifest so that validation never writes data, checkpoints, or results into the working source tree. Each copy receives the same deterministic mixed-type binary-classification fixture: 30 training rows, 14 test rows, two numerical features, one categorical feature, one categorical target, and no missing values.

Both paths use the official debug configuration: four optimizer epochs, four diffusion timesteps, CPU execution, disabled online logging, and deterministic seed 0. The numeric-looking fixture column names intentionally exercise the upstream plotting path without modifying its source. The fixture is an execution/parity case, not a model-quality benchmark.

The native path calls `main.py` directly for training and testing. The adapter path performs the same train and sample operations. The following must all pass:

1. all 27 scoped source hashes match the pinned manifest;
2. cached runtime configs are semantically exact;
3. every tensor in the epoch-4 checkpoint is exactly equal;
4. training-time samples and density metrics are exact;
5. final generated CSV files are byte-for-byte exact;
6. final upstream DCR metrics are exact;
7. exactly 12 rows with the expected four-column schema are generated;
8. all generated numerical values are finite; and
9. both standardized artifact manifests identify TabDiff correctly.

There is no numerical tolerance: deterministic parity is exact.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.tabdiff \
  --repo-root . \
  --output-dir /tmp/tabdiff-validation \
  --evidence-path /tmp/tabdiff-evidence.json
```

`.github/workflows/tabdiff-validation.yml` executes this command on Linux/Python 3.11 and retains the JSON evidence for 90 days. The registry remains `adapter-complete` until a successful retained run is copied into `docs/evidence/tabdiff/` and linked from the source lock. Any later source, dependency, adapter-command, or protocol change invalidates that evidence and requires another run.
