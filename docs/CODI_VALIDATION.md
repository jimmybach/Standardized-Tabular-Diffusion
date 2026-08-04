# CoDi Validation Protocol

Status: implementation complete; mandatory Linux/Python 3.11 evidence pending

## Claim Boundary

This integration targets the CoDi benchmark snapshot distributed by `amazon-science/tabsyn` at commit `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`. It does not claim to execute an unmodified copy of the separate method-author repository.

The method-author repository, `ChaejeongLee/CoDi`, is pinned for provenance at commit `8da2af242e7c43cba86b9ff5a86d05d3411b4ed5`. It has no detected license file. Of ten shared source paths, five are byte-exact and five differ; TabSyn also supplies a separate sampling entry point. Consequently, successful validation establishes parity only with the licensed TabSyn benchmark snapshot. It does not establish original-method parity, reproduce paper tables, approve Official Results, or provide release support.

## Source and Dependency Integrity

The local CoDi subtree contains 11 files and is byte-exact with TabSyn tree `85c16ccfb76fbf00db6b30450ca47e9928efa8d3`. The fail-closed execution manifest covers 24 files:

- the 11 CoDi files;
- TabSyn's dispatcher and shared data runtime;
- the root requirements file, README, Apache-2.0 license, and NOTICE.

Every file is checked after LF normalization before training and again after training or sampling. Missing, additional execution assumptions do not weaken this check: the adapter refuses to run if any locked file differs.

The shared runtime imports `zero`, but the required distribution is `libzero==0.0.8`, not the unrelated PyPI project named `zero`. The official wheel is locked to SHA-256 `f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d`. The validation workflow installs that wheel with `--no-deps` because its historical `torch<2` metadata does not describe the validated PyTorch 2.3 runtime. Its import-time fallback requires `tqdm`, which is separately frozen at `4.66.5` and checked by the protocol.

For the supported Python 3.11 environment, install the CoDi extra and then the locked research utility wheel explicitly:

```bash
python -m pip install ".[codi]"
python -m pip install --no-deps "libzero==0.0.8"
```

## Supported Data Contract

CoDi accepts processed single-table datasets with one target and task type `binclass`, `multiclass`, or `regression`. Before execution, the adapter requires:

- safe single-component dataset identifiers, never paths;
- matching, non-empty train and test CSV files with unique columns;
- numerical, categorical, and target indices that form an exact partition of the schema;
- `idx_name_mapping` that matches the canonical CSV column order;
- non-pickle NumPy arrays whose shapes and values agree with the CSV representation;
- no missing or non-finite numerical values;
- exactly two target classes for binary classification and at least three for multiclass classification;
- at least one continuous and one discrete diffusion component after task-aware target assignment; and
- non-constant continuous training columns, because the pinned transformer performs min-max scaling without a constant-column special case.

For classification, the target belongs to the discrete component. For regression, it belongs to the continuous component. Therefore, support depends on the effective diffusion components, not merely on whether the feature list is described as mixed type. Missing values must be handled first by the centralized train-split-fitted imputer.

## Adapter-Only Runtime Boundary

Tracked upstream files are not patched. `standardized_tabular_diffusion/compat/codi_launcher.py` applies three explicit invocation bridges:

1. `codi-cpu-device-count-v1` reports one logical execution device to the pinned loader's batch-divisibility guard. This prevents division by zero on CPU-only hosts without changing data, batch size, model, loss, or optimizer.
2. `codi-output-checkpoint-root-v1` changes only the module-local file anchor used to derive the checkpoint directory. The official `model_con.pt` and `model_dis.pt` state dictionaries are written under `output_dir` instead of tracked source.
3. `codi-exact-sample-count-v1` fits the official transformers on the complete training data, then changes only the row dimension of transformed sampling placeholders by deterministic repetition or truncation. Learned transforms, categories, weights, and reverse-diffusion equations are unchanged.

The adapter also exposes deterministic seed, CPU or validated CUDA selection, thread count, and the official architecture/training controls. Unknown controls fail closed.

## Checkpoint and Output Safety

CoDi produces two PyTorch state-dictionary files. Because PyTorch serialization is executable/trusted content, the adapter accepts only the checkpoint pair created inside the current `output_dir`. External paths and symlinks are rejected. Training metadata records both SHA-256 values, source identity, dataset schema, seed, device, and effective configuration. Sampling refuses source, schema, or checkpoint drift.

Generated CSV output must have the exact requested row count, canonical column order, no missing values, finite continuous columns, and discrete values inside the fitted training domains. A separate sample metadata record binds the output hash to both checkpoint hashes.

## Native-Parity Protocol

The Linux/Python 3.11 protocol runs nine real cases:

- binary classification, multiclass classification, and regression;
- seeds 0, 19, and 73 for every task variant;
- mixed numerical and categorical features plus one task-aware target;
- bounded but real training of both continuous and discrete diffusion models; and
- seven requested sample rows from twelve training rows, so the exact-row bridge is exercised.

For every case, the protocol creates two isolated source roots. The native root contains checksum-verified source plus transparent temporary validation overrides for CPU execution, small network widths, two diffusion steps, one epoch multiplier, deterministic seeding, and seven sampling rows. The adapter root remains checksum-exact and receives the equivalent public controls through the compatibility launcher.

The protocol requires exact continuous-checkpoint state, exact discrete-checkpoint state, byte-for-byte and frame-for-frame identical CSV output, exact row and schema checks, finite values, zero missing values, valid manifests and metadata, source integrity after execution, and absence of checkpoints under the adapter source tree.

The environment is frozen in `requirements-codi-validation.txt` with CPU PyTorch `2.3.0`. Evidence is uploaded even on failure. Registry promotion beyond `adapter-complete` is forbidden until a successful artifact has been inspected and retained in the repository.

## Remaining Gates

Snapshot parity alone will not make CoDi `benchmark-eligible` or `release-supported`. Remaining gates include central metric execution, dataset-profile admission, full-scale runtime characterization, approved benchmark configurations, license/governance review, and release review. Any original-method claim additionally requires a licensable method-author source and a separate equivalence decision.
