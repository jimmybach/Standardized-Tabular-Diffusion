# Runtime Status

This document separates "integrated in code" from "operationally easy to run".

## Fully Smoke-Validated End to End

- `tabddpm`
- `tabsyn`
- `tabdiff`
- `ctgan`
- `tvae`
- `smote`
- `ctab-gan-plus`
- `nrgboost`
- `bn`
- `nflow`
- `goggle`
- `arf`

These models have a standardized adapter and at least one successful local smoke path through the shared CLI in this environment.

## Train + Sample Works, but Environment Is Fragile

- `goggle`
- `realtabformer`
- `tabsds`
- `tabularargn`
- `tabula`
- `ctab-gan`
- `stasy`
- `codi`

These are integrated and runnable, but they depend on more brittle stacks:

- `goggle`: DGL, torch-geometric, and binary extension compatibility.
- `realtabformer`: Hugging Face imports that currently need the adapter-side torchvision disable shim.
- `tabsds`: local lightweight compatibility implementation inspired by the TabSDS method, not yet smoke-validated against the official upstream code.
- `tabularargn`: optional-package adapter around `mostlyai-engine`; integrated in code, but not yet smoke-validated in this repository.
- `tabula`: local Transformers-based compatibility adapter; integrated into the shared CLI, but not yet smoke-validated against the original upstream workflow.
- `ctab-gan`: legacy research-code path with weaker packaging than `ctgan` or `tvae`.
- `stasy`: vendored baseline path under `TabSyn-main/baselines`, not yet smoke-validated through the shared presets.
- `codi`: vendored baseline path under `TabSyn-main/baselines`, not yet smoke-validated through the shared presets.

## Train-Validated, Sampling-Guarded

- `great`

`great` now imports and trains correctly through the standardized adapter. For the standardized path, ordered-column training plus a first-column start prompt materially improves parseability, and stronger `distilgpt2` runs can now complete sampling. The tiny CPU smoke presets still do not reliably emit parseable rows, so use the train-only preset for deterministic adapter validation and treat sampled tiny runs as stress tests rather than pass/fail quality checks.

## Runtime-Gated by External Model Access

- `tabebm`

`tabebm` is standardized in code and its train action completes, but sample generation depends on Prior Labs' gated TabPFN model access via Hugging Face. The standardized runner now treats sampling as an explicit opt-in path through `sample.extra.allow_gated_model=true`; without that plus accepted terms and authentication, it exits with a clear runtime error.

## Current Environment Caveats

- `torch==2.3.0` is the current pinned runtime in the benchmark stack requirements.
- `transformers==4.46.3` and `tokenizers==0.20.3` are pinned because the vendored `great` code is happier on that surface than on the newer 4.57 series.
- `dgl` still wants a writable home/cache path; the adapter works around this by redirecting cache-related environment variables.
- `torch-scatter` and `torch-sparse` currently emit load warnings under this torch stack, but the `goggle` smoke path still completes in this environment.
- Some vendored baselines expect legacy helper modules; this repository now vendors a minimal local `zero` compatibility shim for the `TabSyn-main` and `TabDDPM-main` paths to avoid pulling the wrong PyPI package.

## Recommended Interpretation

- Use `arf`, `ctgan`, `tvae`, `smote`, `bn`, `nflow`, `nrgboost`, and the diffusion models for the least surprising benchmark runs.
- Treat `goggle`, `great`, `realtabformer`, `tabsds`, `tabularargn`, `tabula`, `ctab-gan`, `stasy`, `codi`, and `tabebm` as integrated but higher-maintenance baselines.
- Use the train-only `tabebm` smoke preset for routine integration checks and the gated-sample preset only on machines that already have TabPFN access configured.
