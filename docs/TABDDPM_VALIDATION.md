# TabDDPM Validation Protocol

Status: passed on Linux/Python 3.11; native parity validated

Protocol ID: `tabddpm-native-parity-v2`

Supported validation platform: Linux, Python 3.11, CPU

## Scope and claim boundary

This protocol validates that the standardized TabDDPM train and sample adapter invokes the pinned method-author implementation without changing its effective configuration or deterministic outputs. Version 2 also exercises the current public adapter contract: an identity-checked embedded `DatasetSpec`, one shared run-owned train/sample workspace, the actual generated runtime TOMLs, the decoded canonical table, and source immutability. It covers source integrity, a real end-to-end smoke run, adapter/native parity, generated-artifact integrity, and reproducibility across three predeclared seed pairs.

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

Full Adult configurations use upstream quantile normalization. The pinned
source passes `subsample=1e9`, an integral float accepted by its original
`scikit-learn==1.0.2` environment but rejected by supported Python 3.11
scikit-learn releases before fitting. A repository-owned startup bridge converts
only integral float `subsample` values to the exactly equal integer before
calling the unchanged official `QuantileTransformer`; all other values and
arguments are forwarded. This is an adapter-only API compatibility boundary,
not an upstream source or algorithm patch. The earlier tiny parity fixture used
min-max normalization and therefore did not exercise this branch; the real
Adult P5 pilot adds that coverage.

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
3. the fixture's complete `DatasetSpec` and content identity must be embedded rather than looked up as a registered dataset;
4. training and sampling must use the same run-owned workspace, with checkpoints and mutable artifacts outside the upstream source tree;
5. the effective train and sample configurations actually executed by the adapter must be exactly equal to the native configuration after excluding only the output/data paths and the global evaluation seed that the official sampling path does not consume;
6. raw and EMA model state dictionaries must have identical keys and tensor values;
7. every generated NumPy array must have an identical inventory, dtype, shape, and element values;
8. generated numeric values must all be finite and the loss CSV must match exactly;
9. the decoded table must contain exactly twelve rows, the declared columns, finite numerical values, no missing values, and only known target labels;
10. both adapter artifact manifests must be valid; and
11. the checksum-locked upstream source must remain exact after execution.

There is no numerical tolerance in this protocol: all deterministic comparisons are exact.

## Execution and evidence

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.tabddpm \
  --repo-root . \
  --output-dir /tmp/tabddpm-validation \
  --evidence-path /tmp/tabddpm-evidence.json
```

`.github/workflows/tabddpm-validation.yml` executes this command on Linux/Python 3.11 and retains the JSON evidence artifact for 90 days. The registry remained `adapter-complete` until the successful retained run was linked in the source lock. Any later source, dependency, adapter-command, or protocol change invalidates the evidence and requires rerunning the workflow.

The current protocol passed in [GitHub Actions run 32045685956](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32045685956) at repository commit `ebe706fe64c1601a0d3f02c6ef43c0754468ea57`. All three seed cases passed every exact comparison, including both runtime configurations, raw and EMA checkpoints, all generated arrays, loss files, decoded tables, manifests, output isolation, and post-run source integrity. The retained artifact digest is `sha256:424821b320390a0d2ccb96d15a3f8a37d6b040f8706d2a86a87a65f5e4e104c3`, and an exact permanent copy of the evidence is stored at `docs/evidence/tabddpm/native-parity-run-32045685956.json` with file SHA-256 `cad319c6141a3c2c91bab43cb1159322bdfcd844765d62293c53ca156c9a7c65`.

Accordingly, TabDDPM is `native-parity-validated` while remaining `experimental`, `unsupported`, and pending a separate official-track eligibility decision.
