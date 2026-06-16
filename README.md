# Standardized Tabular Diffusion Benchmark

This repository now includes a shared benchmarking layer on top of the upstream model code in:

- `TabDiff-main`
- `TabSyn-main`
- `TabDDPM-main`

The goal is to keep the original implementations intact while giving every diffusion-based model the same external contract for:

- training
- sample generation
- evaluation
- result comparison

The standardized layer is designed to be the integration boundary for a future single-repository benchmark setup, while the upstream projects remain close to their original research code.

## Shared Layout

The new root package is `standardized_tabular_diffusion/`.

- `interfaces.py`: common run and artifact schemas
- `models/`: adapters for each upstream model family
- `evaluation/`: TabStruct-aligned metric definitions and normalization
- `comparison.py`: run aggregation utilities
- `cli.py`: a single entrypoint for listing models, describing metrics, running evaluations, and building comparisons

This organization is meant to make eventual migration into a single GitHub repository much easier: the upstream projects remain vendor-like sources, while the root package acts as the stable integration boundary.

## Standardized Interface

Every adapter exposes the same high-level operations:

- `train(spec)`
- `sample(spec)`
- `evaluate(spec)`

Each operation accepts a shared `RunSpec` object and returns standardized artifact metadata. Model-specific arguments still exist, but they are isolated inside `spec.extra`.

The currently supported adapters are:

- `tabdiff`
- `tabsyn`
- `tabddpm`

Each standardized run writes canonical metadata such as:

- `artifacts.json`
- `pipeline_result.json`
- `standardized_summary.json` when evaluation is enabled

## Evaluation Protocol

The normalized evaluation schema is `tabstruct-aligned-v1`.

It aligns outputs to the evaluation dimensions emphasized by the TabStruct paper:

- density fidelity
- ML efficacy
- distinguishability / detection
- privacy
- structural fidelity

Where this repo already has exact implementations, the benchmark layer records normalized values directly. Where an upstream project does not yet expose the full TabStruct dimension, the summary records that gap explicitly instead of silently inventing a score.

The main normalized summary file is:

- `standardized_summary.json`

It is designed to be the only file a future benchmark table needs to read.

## Benchmark Policy

The benchmark layer now makes a few explicit policy choices so results are easier to compare and reproduce:

- Dataset-level inputs are resolved through a canonical dataset registry in `standardized_tabular_diffusion/datasets.py`.
- Materialized datasets override raw upstream paths when available.
- `TabDiff` and `TabSyn` use a shared normalized evaluator.
- `TabDDPM` is normalized from the metrics already emitted by its upstream evaluation stack.
- Structural fidelity defaults to a reproducible local predictor set of `XGB + KNN`.
- `TabPFN` is disabled by default and only enabled when `STANDARDIZED_TABULAR_DIFFUSION_ENABLE_TABPFN=1` is set.
- When `TabPFN` is enabled, it is still treated as optional if it is unavailable because of gated-model access, unsupported class counts, or missing dependencies.

## Reproducibility

The standardized evaluation path now uses a benchmark-oriented deterministic configuration:

- fixed benchmark seeds in the MLE evaluator and structural-fidelity layer
- deterministic train/validation splitting in the upstream MLE path
- seeded row ordering for feature construction
- single-threaded XGBoost for the normalized MLE benchmark path
- stable structural-fidelity predictor policy emitted into the summary metadata

For the current smoke benchmark setup, repeated standardized evaluation runs now produce identical summary hashes.

## Dataset Materialization

For datasets that need a shared canonical processed layout, use:

```bash
python -m standardized_tabular_diffusion.cli materialize-dataset --dataset adult
```

Check the resolved materialization state with:

```bash
python -m standardized_tabular_diffusion.cli materialization-status --dataset adult
```

This is especially important for `TabDiff` and `TabSyn`, which are standardized around a shared processed dataset layout.

## CLI

List available models:

```bash
python -m standardized_tabular_diffusion.cli list-models
```

Describe the shared metric schema:

```bash
python -m standardized_tabular_diffusion.cli describe-metrics
```

Print the shared config schema:

```bash
python -m standardized_tabular_diffusion.cli describe-config
```

Generate an example config:

```bash
python -m standardized_tabular_diffusion.cli example-config \
  --model tabsyn \
  --dataset adult \
  --output-dir artifacts/tabsyn/adult/run-001
```

Resolve a config into the canonical run context:

```bash
python -m standardized_tabular_diffusion.cli build-context --config tmp/example.json
```

Run one standardized action:

```bash
python -m standardized_tabular_diffusion.cli run-action \
  --config tmp/example.json \
  --action evaluate
```

Run the full standardized pipeline:

```bash
python -m standardized_tabular_diffusion.cli run --config tmp/example.json
```

Compare previously normalized run summaries:

```bash
python -m standardized_tabular_diffusion.cli compare \
  --summary artifacts/tabdiff/adult/run-1/standardized_summary.json \
  --summary artifacts/tabsyn/adult/run-1/standardized_summary.json
```

## Tests

The standardized layer now has lightweight regression coverage for:

- reproducibility of dataset splitting and normalized summary generation
- adapter-level contracts for `tabdiff` and `tabddpm`

Run the current standardized test set with:

```bash
pytest tests/test_reproducibility.py tests/test_adapters.py
```

## Notes

- `TabDiff` and `TabSyn` can share the same normalized evaluator because both repos use the same `info.json`-style tabular metadata.
- `TabDDPM` currently has a partially different evaluation stack, so the adapter normalizes the metrics that are already available and marks unavailable TabStruct dimensions explicitly.
- `TabSyn` required a few upstream entrypoint fixes so the standardized runner can execute train/sample stages reliably.
- Some upstream code has been patched locally to support standardization and reproducibility; these changes should be treated as part of the benchmark integration layer unless they are later upstreamed.
- This layer still tries to minimize changes to the original research code unless standardization or reproducibility requires them.
