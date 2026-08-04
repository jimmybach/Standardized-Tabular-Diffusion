# Runtime Observations (Non-Normative)

This document preserves historical local execution observations. It is not the model status source of truth and must not be used to claim benchmark eligibility or release support. Machine-readable current status is available through `std-tabular-diffusion list-models --details`.

At the current release-preparation baseline, every registered runnable adapter is conservatively recorded as `adapter-complete`, `experimental`, and `unsupported`. Earlier local executions were not accompanied by the complete evidence required for `smoke-validated`: a supported Linux/Python 3.11 environment, immutable dependency/source identity, artifact integrity checks, and a retained evidence record.

## Previously Reported Local End-to-End Executions

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

These models were reported to have completed at least one local path through the shared CLI. The runs are useful engineering history but do not currently satisfy the formal smoke-validation gate.

## Previously Reported Train and Sample Paths with Fragile Environments

- `goggle`
- `realtabformer`
- `tabsds`
- `tabularargn`
- `tabula`
- `ctab-gan`
- `stasy`
- `codi`

These adapters were reported as runnable in a prior environment, but they depend on brittle stacks and require fresh supported-environment evidence:

- `goggle`: DGL, torch-geometric, and binary extension compatibility.
- `realtabformer`: Hugging Face imports that currently need the adapter-side torchvision disable shim.
- `tabsds`: local lightweight compatibility implementation inspired by the TabSDS method, not yet smoke-validated against the official upstream code.
- `tabularargn`: optional-package adapter around `mostlyai-engine`; integrated in code, but not yet smoke-validated in this repository.
- `tabula`: local Transformers-based compatibility adapter; integrated into the shared CLI, but not yet smoke-validated against the original upstream workflow.
- `ctab-gan`: legacy research-code path with weaker packaging than `ctgan` or `tvae`.
- `stasy`: vendored baseline path under `TabSyn-main/baselines`, not yet smoke-validated through the shared presets.
- `codi`: vendored baseline path under `TabSyn-main/baselines`, not yet smoke-validated through the shared presets.

## Previously Reported Training Path, Sampling Guarded

- `great`

`great` was reported to import and train through the standardized adapter. Ordered-column training plus a first-column start prompt improved parseability in those experiments, while tiny CPU presets did not reliably emit parseable rows. These observations require fresh evidence and native-parity review before any stronger claim.

## Runtime-Gated by External Model Access

- `tabebm`

`tabebm` is standardized in code and its train action completes, but sample generation depends on Prior Labs' gated TabPFN model access via Hugging Face. The standardized runner now treats sampling as an explicit opt-in path through `sample.extra.allow_gated_model=true`; without that plus accepted terms and authentication, it exits with a clear runtime error.

## Historical Environment Caveats

- `torch==2.3.0` is the current pinned runtime in the benchmark stack requirements.
- `transformers==4.46.3` and `tokenizers==0.20.3` are pinned because the vendored `great` code is happier on that surface than on the newer 4.57 series.
- `dgl` still wants a writable home/cache path; the adapter works around this by redirecting cache-related environment variables.
- `torch-scatter` and `torch-sparse` currently emit load warnings under this torch stack, but the `goggle` smoke path still completes in this environment.
- The TabSyn primary path resolves its imported `zero` research-utility API to the frozen `libzero==0.0.8` distribution; the similarly named `zero` distribution is unrelated. The separate TabDDPM shim remains tracked by its own audit record.

## Current Interpretation

- Treat every current adapter as an experimental engineering integration until its evidence record says otherwise.
- Local compatibility implementations (`tabsds`, `tabula`, and the current `tabebm` path) are excluded from the future official track unless replaced by or validated against an approved authoritative implementation.
- The `tabebm` gated-sample preset may be used only on machines whose users have accepted the applicable model terms and configured access.
