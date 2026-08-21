# Troubleshooting

## Python or platform mismatch

The release target is 64-bit CPython 3.11. If a locked package has no wheel or attempts a local compiler build, confirm `python --version` and `python -c "import platform; print(platform.architecture())"`, then recreate the environment with Python 3.11.

## Optional dependency missing or wrong version

Install the narrow extra named by the error, such as `.[quickstart]`, `.[evaluation]`, `.[validity]`, or the model-specific extra. Adapters intentionally reject a different upstream package version because an importable package is not necessarily the validated implementation.

## Central evaluation says a profile or table is missing

Adapter evaluation requires a decoded synthetic table, a reviewed `dataset_profile_path`, and a real training reference. P4/P5 also require a held-out real test table. Paths may be explicit in `evaluation` config or resolved from a materialized DatasetSpec.

`import-legacy-dataset-profile` preserves legacy metadata and is intentionally not P3-ready. For a non-Official functionality check of a materialized complete-data view, use `create-diagnostic-dataset-profile`. A scientifically admitted run still needs a fully reviewed Dataset Profile.

## A materialized dataset is reported as unknown

Materialized datasets belong to a user workspace, not the installed package. Pass the same `--workspace` to `materialize-dataset`, `list-datasets`, profile creation, `run`, and `benchmark run`. If the command was launched from another directory, do not rely on the default `.` workspace.

## Structural validation fails

Do not repair generated output inside the evaluator. Check exact column names/order, supported CSV/Parquet format, row count, missing values, types, and the selected dataset view. If real inputs contain missing values, run the centralized train-only mean/mode preprocessor before training and regenerate the sample.

## Result directory already exists

Result Bundles, legacy imports, snapshots, and quickstart roots are immutable. Choose a new directory. Do not delete or edit retained evidence to make a run pass.

## Legacy summary rejected

Only frozen `schema_version: 1.0` / `protocol_name: tabstruct-aligned-v1` documents are accepted. Use `import-legacy-summary`; do not pass the file to `validate-result` or `build-leaderboard`. Unknown fields, including an Official claim, fail schema validation.

## Checkpoint blocked

Pickle and many PyTorch formats can execute code. Move a checkpoint produced by the same run beneath `output_dir`. Use `allow_unsafe_external_checkpoint=true` only after separately verifying provenance and integrity.

## CUDA unavailable or out of memory

Confirm that the configured adapter supports Windows/CUDA, that PyTorch sees the GPU, and that the selected official dependency environment matches its validation record. Reduce only parameters that are declared tunable; such a change may move the result to `standardized-tuning`. Never silently fall back to CPU in evidence intended for a declared GPU profile.

The default Python package index may select a CPU-only PyTorch build. A model extra cannot by itself guarantee a compatible NVIDIA build. For the currently accepted Windows CTGAN/RTX 5080 profile, install the narrow extras, replace PyTorch from the official CUDA 12.8 index, and verify the resolved environment before training:

```powershell
python -m pip install ".[ctgan,validity]"
python -m pip install --force-reinstall "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA DEVICE')"
```

Stop if CUDA is false, the device is not the intended device, or `pip check` fails. Other adapters may require a different pinned environment; use that adapter's validation document rather than applying this CTGAN profile universally.

## A diagnostic snapshot has no rank

That is expected. `partial-diagnostic` and `community` snapshots use diagnostic ordering but never Official ranks. Official ranking needs every independent admission and coverage gate.

When filing an issue, include the command, Python/platform versions, stable reason code, and redacted logs. Do not attach private rows, credentials, or unreviewed checkpoints.
