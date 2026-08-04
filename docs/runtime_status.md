# Runtime Observations (Non-Normative)

This document preserves historical local execution observations. It is not the model status source of truth and must not be used to claim benchmark eligibility or release support. Machine-readable current status is available through `std-tabular-diffusion list-models --details`.

At the current release-preparation baseline, CTAB-GAN+, CTGAN, TVAE, SMOTE, NRGBoost, TabDDPM, TabDiff, and TabSyn are `native-parity-validated`, `experimental`, and `unsupported` based on retained Linux/Python 3.11 evidence. Other runnable adapters on this branch remain conservatively recorded as `adapter-complete`, `experimental`, and `unsupported`. Earlier local executions were not accompanied by the complete evidence required for `smoke-validated`: a supported Linux/Python 3.11 environment, immutable dependency/source identity, artifact integrity checks, and a retained evidence record.

CTAB-GAN+ no longer imports the semantically modified snapshot formerly embedded in the TabDDPM tree. The adapter now acquires five byte-exact files from locked method-author commit `6a6f901` into an ignored cache and does not patch them. Its mandatory six-case Linux/Python 3.11 protocol passed in run `30926267432`, and the inspected evidence is permanently retained, so its adapter status is `native-parity-validated`. The absent upstream license independently blocks redistribution, Official Results, and release claims.

CTGAN has been moved from the legacy embedded `0.5.2.dev0` import path to the checksum-pinned official `ctgan==0.12.1` wheel. Its first mandatory parity run passed and is retained. The package is BUSL-1.1, so validation cannot by itself grant Official Results or release eligibility.

TVAE has now been moved to `ctgan.TVAE` from the same official wheel. The locally modified legacy subtree and its wrappers were removed. Its first mandatory Linux/Python 3.11 parity run passed and is permanently retained, so TVAE is `native-parity-validated`; the shared package-license gate still applies.

## Current Retained Validation

- `ctab-gan-plus`: passed all six classification/regression and seed cases in `ctabgan-plus-native-parity-v1` GitHub Actions run `30926267432`; see `docs/CTABGAN_PLUS_VALIDATION.md` and the permanent JSON evidence record. The absent upstream license remains an independent redistribution and release gate.
- `tabddpm`: passed `tabddpm-native-parity-v1` for the `(training, sampling)` seed pairs `(0, 23)`, `(17, 47)`, and `(101, 89)` in GitHub Actions run `30863212268`; see `docs/TABDDPM_VALIDATION.md` and the permanent JSON evidence record. This is not an Official Results or release-support claim.
- `tabdiff`: passed `tabdiff-native-parity-v1` in GitHub Actions run `30866879879`; see `docs/TABDIFF_VALIDATION.md` and the permanent JSON evidence record. This is not an Official Results or release-support claim.
- `tabsyn`: passed `tabsyn-native-parity-v1` for seeds 0, 19, and 73 in GitHub Actions run `30871758645`; see `docs/TABSYN_VALIDATION.md` and the permanent JSON evidence record. This is not an Official Results or release-support claim.
- `ctgan`: passed `ctgan-native-parity-v1` for seeds 0, 19, and 73 in GitHub Actions run `30910275922`; see `docs/CTGAN_VALIDATION.md` and the permanent JSON evidence record. The official package's BUSL-1.1 terms remain a separate release gate.
- `tvae`: passed `tvae-native-parity-v1` for seeds 0, 19, and 73 in GitHub Actions run `30913867621`; see `docs/TVAE_VALIDATION.md` and the permanent JSON evidence record. The official package's BUSL-1.1 terms remain a separate release gate.
- `smote`: passed all nine `smote-native-parity-v1` sampler/seed cases in GitHub Actions run `30918785254`; see `docs/SMOTE_VALIDATION.md` and the permanent JSON evidence record. SMOTE remains a classification-only classical reference excluded from the joint generative-model ranking.
- `nrgboost`: passed all six classification/regression and seed cases in `nrgboost-native-parity-v1` GitHub Actions run `30922326384`; see `docs/NRGBOOST_VALIDATION.md` and the permanent JSON evidence record. Benchmark eligibility, runtime policy, and release support remain separate gates.

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

These models were reported to have completed at least one local path through the shared CLI. Except for the separately revalidated CTAB-GAN+, CTGAN, TVAE, SMOTE, NRGBoost, TabDDPM, TabDiff, and TabSyn paths, these runs are useful engineering history but do not currently satisfy the formal smoke-validation gate.

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
- TabDDPM vendors the byte-exact runtime modules from the official `libzero==0.0.8` wheel; TabSyn installs that same distribution without its stale dependency metadata. The similarly named `zero` distribution is unrelated, and neither path uses the former local compatibility substitutes.

## Current Interpretation

- Treat every current adapter as an experimental engineering integration until its evidence record says otherwise.
- Local compatibility implementations (`tabsds`, `tabula`, and the current `tabebm` path) are excluded from the future official track unless replaced by or validated against an approved authoritative implementation.
- The `tabebm` gated-sample preset may be used only on machines whose users have accepted the applicable model terms and configured access.
