# Runtime Observations (Non-Normative)

This document preserves historical local execution observations. It is not the model status source of truth and must not be used to claim benchmark eligibility or release support. Machine-readable current status is available through `std-tabular-diffusion list-models --details`.

At the current release-preparation baseline, ARF, CoDi, CTAB-GAN, CTAB-GAN+, CTGAN, Goggle, TVAE, SMOTE, NRGBoost, REaLTabFormer, TabularARGN, TabDDPM, TabDiff, TabSyn, and STaSy are `native-parity-validated`, `experimental`, and `unsupported` based on retained Linux/Python 3.11 evidence. ARF's claim is limited to the official Python package and does not establish R/Python equivalence; CoDi and STaSy claims are limited to their TabSyn benchmark snapshots; Goggle's claim is limited to the method-author GCN core; REaLTabFormer's claim is limited to official tabular training with `n_critic=0`; TabularARGN's claim is limited to flat single-table unconditional generation. Other runnable adapters on this branch remain conservatively recorded as `adapter-complete`, `experimental`, and `unsupported`. Earlier local executions were not accompanied by the complete evidence required for `smoke-validated`: a supported Linux/Python 3.11 environment, immutable dependency/source identity, artifact integrity checks, and a retained evidence record.

CoDi now has a checksum-locked 24-file TabSyn-snapshot execution scope, strict dual-checkpoint handling, an exact-row compatibility boundary, and a dedicated CPU smoke preset. Its mandatory nine-case Linux/Python 3.11 protocol passed exactly in run `30941940893`, and the inspected evidence is retained, so its adapter status is `native-parity-validated`. It remains `experimental` and `unsupported`; the separate method-author repository has no declared license and materially differs from the TabSyn adaptation.

CTAB-GAN+ no longer imports the semantically modified snapshot formerly embedded in the TabDDPM tree. The adapter now acquires five byte-exact files from locked method-author commit `6a6f901` into an ignored cache and does not patch them. Its mandatory six-case Linux/Python 3.11 protocol passed in run `30926267432`, and the inspected evidence is permanently retained, so its adapter status is `native-parity-validated`. The absent upstream license independently blocks redistribution, Official Results, and release claims.

CTGAN has been moved from the legacy embedded `0.5.2.dev0` import path to the checksum-pinned official `ctgan==0.12.1` wheel. Its first mandatory parity run passed and is retained. The package is BUSL-1.1, so validation cannot by itself grant Official Results or release eligibility.

TVAE has now been moved to `ctgan.TVAE` from the same official wheel. The locally modified legacy subtree and its wrappers were removed. Its first mandatory Linux/Python 3.11 parity run passed and is permanently retained, so TVAE is `native-parity-validated`; the shared package-license gate still applies.

## Current Retained Validation

- `arf`: passed all nine binary, multiclass, regression, and seed cases in `arfpy-official-package-parity-v1` GitHub Actions run `30964711614`; restored FORGE state and generated CSV bytes matched exactly. The claim covers only the method-author official Python package, not R/Python cross-language equivalence. Official Results and release gates remain pending.
- `ctab-gan-plus`: passed all six classification/regression and seed cases in `ctabgan-plus-native-parity-v1` GitHub Actions run `30926267432`; see `docs/CTABGAN_PLUS_VALIDATION.md` and the permanent JSON evidence record. The absent upstream license remains an independent redistribution and release gate.
- `ctab-gan`: the former semantic fork has been replaced by checksum-locked Apache-2.0 official source. All six binary/multiclass and seed parity cases passed exactly in retained Linux/Python 3.11 run `30930939961`; the registry is `native-parity-validated` while Official Results and release gates remain pending.
- `tabddpm`: passed `tabddpm-native-parity-v1` for the `(training, sampling)` seed pairs `(0, 23)`, `(17, 47)`, and `(101, 89)` in GitHub Actions run `30863212268`; see `docs/TABDDPM_VALIDATION.md` and the permanent JSON evidence record. This is not an Official Results or release-support claim.
- `tabdiff`: passed `tabdiff-native-parity-v1` in GitHub Actions run `30866879879`; see `docs/TABDIFF_VALIDATION.md` and the permanent JSON evidence record. This is not an Official Results or release-support claim.
- `tabsyn`: passed `tabsyn-native-parity-v1` for seeds 0, 19, and 73 in GitHub Actions run `30871758645`; see `docs/TABSYN_VALIDATION.md` and the permanent JSON evidence record. This is not an Official Results or release-support claim.
- `ctgan`: passed `ctgan-native-parity-v1` for seeds 0, 19, and 73 in GitHub Actions run `30910275922`; see `docs/CTGAN_VALIDATION.md` and the permanent JSON evidence record. The official package's BUSL-1.1 terms remain a separate release gate.
- `tvae`: passed `tvae-native-parity-v1` for seeds 0, 19, and 73 in GitHub Actions run `30913867621`; see `docs/TVAE_VALIDATION.md` and the permanent JSON evidence record. The official package's BUSL-1.1 terms remain a separate release gate.
- `smote`: passed all nine `smote-native-parity-v1` sampler/seed cases in GitHub Actions run `30918785254`; see `docs/SMOTE_VALIDATION.md` and the permanent JSON evidence record. SMOTE remains a classification-only classical reference excluded from the joint generative-model ranking.
- `nrgboost`: passed all six classification/regression and seed cases in `nrgboost-native-parity-v1` GitHub Actions run `30922326384`; see `docs/NRGBOOST_VALIDATION.md` and the permanent JSON evidence record. Benchmark eligibility, runtime policy, and release support remain separate gates.
- `realtabformer`: passed all nine binary/multiclass/regression and seed cases in `realtabformer-official-package-parity-v1` GitHub Actions run `30950369908`; checkpoint tensors/files, saved configuration semantics, raw samples, and final CSV bytes were exact. The retained claim covers official tabular training with sensitivity stopping disabled. Sensitivity stopping, relational mode, benchmark eligibility, runtime policy, and release support remain separate gates.
- `tabularargn`: passed all nine binary, multiclass, regression, and seed cases in `tabularargn-official-package-parity-v2` GitHub Actions run `30961590047`; contract-normalized samples and generated CSV bytes matched exactly. The claim covers flat single-table unconditional generation only; sequential, relational, privacy, prediction, likelihood, imputation, Official Results, and release gates remain outside scope.
- `stasy`: passed all nine binary, multiclass, regression, and seed cases in `stasy-tabsyn-snapshot-parity-v1` GitHub Actions run `30936275831`; checkpoint state and generated CSV bytes matched exactly. The permanent evidence validates the TabSyn snapshot only, not the distinct method-author source.
- `codi`: passed all nine binary, multiclass, regression, and seed cases in `codi-tabsyn-snapshot-parity-v1` GitHub Actions run `30941940893`; both checkpoint states and generated CSV bytes matched exactly. The permanent evidence validates the TabSyn snapshot only, not the distinct method-author source.
- `goggle`: passed all nine binary, multiclass, regression, and seed cases in `goggle-method-author-native-parity-v1` GitHub Actions run `30945676747`; checkpoint tensors and files, raw core samples, and final frames/CSV bytes matched exactly. The permanent evidence validates the unmodified method-author GCN core only; SAGE, heterogeneous decoding, Official Results, and release gates remain pending.

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
- `stasy`
- `codi`

These models were reported to have completed at least one local path through the shared CLI. ARF, CoDi, CTAB-GAN, CTAB-GAN+, CTGAN, Goggle, TVAE, SMOTE, NRGBoost, REaLTabFormer, TabularARGN, TabDDPM, TabDiff, TabSyn, and STaSy now have separate retained native-parity evidence; all other entries in this historical list remain useful engineering observations but do not currently satisfy the formal smoke-validation gate.

## Previously Reported Train and Sample Paths with Fragile Environments

- `tabsds`
- `tabula`

These adapters were reported as runnable in a prior environment, but they depend on brittle stacks and require fresh supported-environment evidence:

- `tabsds`: local lightweight compatibility implementation inspired by the TabSDS method, not yet smoke-validated against the official upstream code.
- `tabula`: local Transformers-based compatibility adapter; integrated into the shared CLI, but not yet smoke-validated against the original upstream workflow.

## Previously Reported Training Path, Sampling Guarded

- `great`

`great` was reported to import and train through the standardized adapter. Ordered-column training plus a first-column start prompt improved parseability in those experiments, while tiny CPU presets did not reliably emit parseable rows. These observations require fresh evidence and native-parity review before any stronger claim.

## Runtime-Gated by External Model Access

- `tabebm`

`tabebm` is standardized in code and its train action completes, but sample generation depends on Prior Labs' gated TabPFN model access via Hugging Face. The standardized runner now treats sampling as an explicit opt-in path through `sample.extra.allow_gated_model=true`; without that plus accepted terms and authentication, it exits with a clear runtime error.

## Historical Environment Caveats

- `torch==2.3.0` is the current pinned runtime in the benchmark stack requirements.
- `transformers==4.46.3` and `tokenizers==0.20.3` are pinned because the vendored `great` code is happier on that surface than on the newer 4.57 series.
- Goggle's validated GCN path uses `dgl==1.1.3` and `torch-geometric==2.5.3` without requiring the unused heterogeneous `torch-sparse` path. Requesting the heterogeneous decoder still fails closed unless its official extension stack is available; that path has no parity claim.
- TabDDPM vendors the byte-exact runtime modules from the official `libzero==0.0.8` wheel; TabSyn installs that same distribution without its stale dependency metadata. The similarly named `zero` distribution is unrelated, and neither path uses the former local compatibility substitutes.

## Current Interpretation

- Treat every current adapter as an experimental engineering integration until its evidence record says otherwise.
- Local compatibility implementations (`tabsds`, `tabula`, and the current `tabebm` path) are excluded from the future official track unless replaced by or validated against an approved authoritative implementation.
- The `tabebm` gated-sample preset may be used only on machines whose users have accepted the applicable model terms and configured access.
