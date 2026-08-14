# Troubleshooting

## Python or platform mismatch

The release target is 64-bit CPython 3.11. If a locked package has no wheel or attempts a local compiler build, confirm `python --version` and `python -c "import platform; print(platform.architecture())"`, then recreate the environment with Python 3.11.

## Optional dependency missing or wrong version

Install the narrow extra named by the error, such as `.[quickstart]`, `.[evaluation]`, `.[validity]`, or the model-specific extra. Adapters intentionally reject a different upstream package version because an importable package is not necessarily the validated implementation.

## Central evaluation says a profile or table is missing

Adapter evaluation requires a decoded synthetic table, a reviewed `dataset_profile_path`, and a real training reference. P4/P5 also require a held-out real test table. Paths may be explicit in `evaluation` config or resolved from a materialized DatasetSpec.

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

## A diagnostic snapshot has no rank

That is expected. `partial-diagnostic` and `community` snapshots use diagnostic ordering but never Official ranks. Official ranking needs every independent admission and coverage gate.

When filing an issue, include the command, Python/platform versions, stable reason code, and redacted logs. Do not attach private rows, credentials, or unreviewed checkpoints.
