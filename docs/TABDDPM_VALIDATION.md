# TabDDPM Validation Protocol

Status: executable protocol awaiting its first retained Linux result

Protocol ID: `tabddpm-native-parity-v1`

Supported validation platform: Linux, Python 3.11, CPU

## Scope and claim boundary

This protocol validates that the standardized TabDDPM train and sample adapter invokes the pinned method-author implementation without changing its configuration or deterministic outputs. It covers source integrity, a real end-to-end smoke run, adapter/native parity, generated-artifact integrity, and reproducibility across three predeclared seed pairs.

Passing this protocol is sufficient to promote the adapter validation level through `smoke-validated` to `native-parity-validated`. It is not sufficient to make TabDDPM `benchmark-eligible`, assign it to the Official Results track, or declare it `release-supported`. Dataset admission, the central evaluation protocol, privacy/fairness review, dependency maintenance, and release ownership remain separate gates.

## Authorities and source integrity

- Method source: `yandex-research/tab-ddpm` at `b476257dd460b778ba09eb97f7a51d6490fa17f8`.
- Method tree: `b0b380892ae2fdcedadaac52a6334ad36a5d60ce`.
- Runtime dependency: official PyPI wheel `libzero==0.0.8`, SHA-256 `f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d`.
- Integrity manifest: `standardized_tabular_diffusion/resources/upstream/tabddpm-source-manifest.json`.

The initial repository import contained 58 exact scoped upstream files but omitted the six-file official `lib/` package, so the native pipeline could not run. Those files were restored from the pinned checkout. The former three-file `zero` compatibility shim was also rejected: unlike official libzero, it assigned the same seed to Python, NumPy, and PyTorch and therefore changed randomness semantics. It was replaced with the seven byte-exact runtime modules from the official wheel, with its MIT license retained.

All 64 scoped TabDDPM files are checked using SHA-256 after canonical line-ending normalization. All seven libzero modules and its license are checked byte-for-byte. A mismatch fails before model execution.

## Frozen environment

The workflow installs:

- CPython 3.11;
- PyTorch 2.3.0 CPU;
- the pinned packages in `requirements-tabddpm-validation.txt`; and
- `rtdl==0.0.9` without dependency resolution.

The last step is intentional. `rtdl==0.0.9` declares `torch<2`, while the limited API used by the pinned TabDDPM code is exercised here against the supported PyTorch 2.3 environment. This is an environment compatibility decision, not a source patch. Official libzero source is vendored for the same legacy-metadata reason.

The official entrypoint is a script under `scripts/`, while `lib` and `zero` are sibling packages at the upstream repository root. The adapter therefore prepends that root to `PYTHONPATH`; the native comparison command uses the identical environment. This is an invocation-only adaptation and does not modify upstream source or runtime semantics.

Equivalent local installation commands are:

```bash
python -m pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps "rtdl==0.0.9"
python -m pip install -r requirements-tabddpm-validation.txt
python -m pip install --no-deps .
```

## Predeclared comparison

The protocol creates a deterministic, numeric-only binary-classification fixture with 24 training, 8 validation, and 8 test rows. The tiny execution configuration uses three numerical features, an MLP with layers `[16, 16]`, three optimizer steps, four diffusion timesteps, and twelve generated rows. It is an execution/parity fixture, not a quality benchmark.

The three `(training seed, sampling seed)` pairs are `(0, 23)`, `(17, 47)`, and `(101, 89)`. For every pair:

1. the native path runs `scripts/pipeline.py --train` and `scripts/pipeline.py --sample` directly;
2. the standardized adapter runs its train and sample operations using an otherwise identical config;
3. configs must be exactly equal after removing only the output-specific `parent_dir`;
4. raw and EMA model state dictionaries must have identical keys and tensor values;
5. every generated NumPy array must have an identical inventory, dtype, shape, and element values;
6. generated numeric values must all be finite;
7. the loss CSV must match exactly;
8. exactly twelve label rows must be generated; and
9. both adapter artifact manifests must be valid and identify the expected model and fixture.

There is no numerical tolerance in this protocol: all deterministic comparisons are exact.

## Execution and evidence

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.tabddpm \
  --repo-root . \
  --output-dir /tmp/tabddpm-validation \
  --evidence-path /tmp/tabddpm-evidence.json
```

`.github/workflows/tabddpm-validation.yml` executes this command on Linux/Python 3.11 and retains the JSON evidence artifact for 90 days. The registry must remain `adapter-complete` until a successful retained run is linked in the source lock. Any later source, dependency, adapter-command, or protocol change invalidates the evidence and requires rerunning the workflow.
