# STaSy Validation Protocol

Status: implementation complete; mandatory Linux/Python 3.11 evidence pending

## Claim Boundary

This integration targets the STaSy baseline snapshot distributed by the official Amazon TabSyn repository at commit `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`. It does **not** claim byte or behavioral identity with the separate method-author repository.

The method-author repository is `JayoungKim408/STaSy` at commit `3dcc660db26e31cc1bbb00cfc14bf7687fde448d`. It has no detected license file or GitHub license expression. The TabSyn snapshot has a different execution layout and differs in 12 of 14 shared paths after line-ending normalization. Consequently:

- redistributed source comes only from the Apache-2.0 TabSyn snapshot;
- validation is named **TabSyn snapshot parity**, not original-method parity;
- the model remains in the Experimental track; and
- Official Results and release support remain blocked independently of technical parity.

## Source Integrity

The runtime manifest freezes 30 files:

- all 17 selected Python files under `TabSyn-main/baselines/stasy/`;
- the TabSyn dispatcher and shared preprocessing modules;
- the Apache-2.0 license, NOTICE, README, and dependency declaration.

All 17 local STaSy Python files match the pinned TabSyn tree after the declared `lf-one-final-newline` normalization. Tracked bytecode, datasets, checkpoints, generated data, unrelated baselines, and the primary TabSyn model are outside this STaSy execution scope. Every train and sample action validates the 30 files before importing upstream code.

## Adapter Boundary

Tracked upstream source remains unchanged. `standardized_tabular_diffusion/compat/stasy_launcher.py` supplies the controls missing or broken in the snapshot:

- explicit CPU or validated CUDA selection;
- deterministic Python, NumPy, and PyTorch seeding;
- effective epoch count, batch size, score-network width, number of SDE scales, worker count, and thread count;
- choice of the snapshot's ODE or predictor-corrector sampler;
- exact requested sample-row count;
- checkpoint redirection from the tracked source tree into `output_dir`; and
- preservation of self-paced learning unless explicitly disabled.

Python 3.11 uses scikit-learn 1.5.2, where `OneHotEncoder` renamed the snapshot's `sparse` keyword to `sparse_output`. The adapter-only `stasy-sklearn-onehot-keyword-v1` bridge forwards the unchanged `False` value to the renamed keyword. It changes neither the encoder nor its dense output.

These controls configure objects returned by the snapshot's own `get_config` and call its own preprocessing, score model, SDE loss, EMA, checkpoint, sampling, and inverse-transform functions. No repository-side substitute model is used.

The original snapshot's root CLI accepted epoch and row-count flags that STaSy did not consume, selected CUDA unconditionally inside the STaSy modules, and wrote checkpoints under tracked source. The dedicated boundary makes those behaviors explicit and testable instead of silently reporting unsupported controls as effective.

## Runtime Environment

Install the STaSy extra in Python 3.11, then install the official `libzero==0.0.8` wheel without dependencies:

```bash
python -m pip install ".[stasy]"
python -m pip install --no-deps "libzero==0.0.8"
```

The shared TabSyn runtime imports the research utility package as `zero`, but that API is distributed by `libzero`; the PyPI distribution actually named `zero` is unrelated. The official `libzero` wheel is locked as `libzero-0.0.8-py3-none-any.whl` with SHA-256 `f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d`. Its stale metadata requires `torch<2`, so the validated environment ignores that metadata and separately locks its actual dependencies, including PyTorch 2.3.0.

## Data Contract

The adapter supports binary classification, multiclass classification, and regression through the TabSyn processed-data layout. Numerical and categorical feature columns are supported. Targets follow the snapshot's task-dependent concatenation behavior.

Missing or non-finite values are rejected before execution. A dataset with missing values must first use the centralized preprocessor, whose numerical means and categorical modes are fitted on the real training split only. This avoids silently relying on a different model-local missing-value policy and prevents test leakage.

Dataset identifiers cannot contain path separators or traversal components. Train/test arrays must be non-pickle NumPy files with consistent, non-zero row counts, at least one feature group, exactly one target column, and metadata row counts that agree with the arrays.

## Checkpoint Security and Reproducibility

STaSy checkpoints use PyTorch's pickle-capable format. The adapter therefore:

- writes checkpoints only below the configured `output_dir`;
- rejects symlinked or externally supplied checkpoints;
- records the checkpoint SHA-256, source-manifest identity, effective training configuration, seed, and device;
- verifies the checksum and source identity before sampling; and
- records sample SHA-256, row count, columns, and sampler configuration.

The smoke preset uses a deliberately small architecture and predictor-corrector schedule. It proves integration behavior only and is not a quality benchmark configuration.

## Mandatory Parity Protocol

The Linux/Python 3.11 workflow executes nine real cases:

- binary classification, multiclass classification, and regression;
- seeds `0`, `19`, and `73`;
- mixed numerical and categorical features;
- one real training epoch using the snapshot loss, optimizer, EMA, and checkpoint functions; and
- exact 12-row sampling using the snapshot predictor-corrector implementation.

For each case, the protocol creates two isolated source roots. The native reference receives transparent temporary source overrides for the bounded CPU configuration. The adapter root remains checksum-exact and receives the equivalent values through the public compatibility boundary. The protocol requires:

- exact model, optimizer, EMA, step, and epoch checkpoint state;
- byte-exact and frame-exact generated CSV output;
- exact requested row and column counts;
- no missing or non-finite numerical output;
- valid adapter manifests and metadata;
- all 30 runtime files still exact after adapter execution; and
- no checkpoint or generated artifact written into the upstream source tree.

The environment is frozen in `requirements-stasy-validation.txt` with PyTorch `2.3.0` on CPU. Evidence is uploaded even when validation fails. The registry cannot be promoted beyond `adapter-complete` until a successful artifact has been inspected and retained in the repository.

## Remaining Gates

Successful snapshot parity will not make STaSy benchmark-eligible or release-supported. Remaining gates include central metric execution, dataset-profile admission, full-scale runtime characterization, configuration approval, and release review. Original-method claims additionally require a licensed method-author source and a separate equivalence decision.
