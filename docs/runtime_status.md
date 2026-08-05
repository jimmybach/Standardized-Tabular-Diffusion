# Runtime Observations (Non-Normative)

This document preserves historical local execution observations. It is not the model status source of truth and must not be used to claim benchmark eligibility or release support. Machine-readable current status is available through `std-tabular-diffusion list-models --details`.

At the current release-preparation baseline, 20 adapters are `native-parity-validated`, `experimental`, and `unsupported` based on retained Linux/Python 3.11 evidence. GReaT, TabuLa, and TabSDS now target checksum-locked official packages or method-author source rather than local compatibility implementations, and their authoritative parity runs have passed. TabEBM is `smoke-validated` against the official package but cannot claim native parity from public CI because full generation requires explicit acceptance of gated TabPFN-v2 terms and model access. Benchmark eligibility and release support remain independent gates for every adapter.

BN targets the checksum-locked official `pgmpy==1.1.2` wheel plus an explicit quantile/BIC/BDeu recipe. Its adapter includes every canonical node, fails closed on missing data and unsupported recipe changes, and stores safe JSON graph/CPD state rather than pickle. All nine exact cases passed in retained Linux/Python 3.11 run `30967779298`, so its adapter status is `native-parity-validated`. This target is recipe parity with a canonical library, not a paper-native implementation claim; Official Results and release support remain separate gates.

NFlow now targets the checksum-locked canonical `nflows==0.14` source distribution plus an explicit mixed-type MAF recipe. The adapter fails closed on invalid data and unsupported recipe changes, restores PyTorch RNG/thread state, and replaces executable pickle with integrity-checked JSON plus `allow_pickle=False` NumPy tensors. All nine exact cases passed in retained Linux/Python 3.11 run `30970260840`, so the status is `native-parity-validated`, `experimental`, and `unsupported`. This is a declared-recipe target, not a paper-native tabular synthesizer claim; Official Results and release support remain separate gates.

CoDi now has a checksum-locked 24-file TabSyn-snapshot execution scope, strict dual-checkpoint handling, an exact-row compatibility boundary, and a dedicated CPU smoke preset. Its mandatory nine-case Linux/Python 3.11 protocol passed exactly in run `30941940893`, and the inspected evidence is retained, so its adapter status is `native-parity-validated`. It remains `experimental` and `unsupported`; the separate method-author repository has no declared license and materially differs from the TabSyn adaptation.

CTAB-GAN+ no longer imports the semantically modified snapshot formerly embedded in the TabDDPM tree. The adapter now acquires five byte-exact files from locked method-author commit `6a6f901` into an ignored cache and does not patch them. Its mandatory six-case Linux/Python 3.11 protocol passed in run `30926267432`, and the inspected evidence is permanently retained, so its adapter status is `native-parity-validated`. The absent upstream license independently blocks redistribution, Official Results, and release claims.

CTGAN has been moved from the legacy embedded `0.5.2.dev0` import path to the checksum-pinned official `ctgan==0.12.1` wheel. Its first mandatory parity run passed and is retained. The package is BUSL-1.1, so validation cannot by itself grant Official Results or release eligibility.

TVAE has now been moved to `ctgan.TVAE` from the same official wheel. The locally modified legacy subtree and its wrappers were removed. Its first mandatory Linux/Python 3.11 parity run passed and is permanently retained, so TVAE is `native-parity-validated`; the shared package-license gate still applies.

## Current Retained Validation

- `arf`: passed all nine binary, multiclass, regression, and seed cases in `arfpy-official-package-parity-v1` GitHub Actions run `30964711614`; restored FORGE state and generated CSV bytes matched exactly. The claim covers only the method-author official Python package, not R/Python cross-language equivalence. Official Results and release gates remain pending.
- `bn`: passed all nine binary, multiclass, regression, and seed cases in `pgmpy-bn-recipe-parity-v1` GitHub Actions run `30967779298`; preprocessing, graph/CPD state, the restored official model, discrete samples, and final CSV bytes matched exactly. The claim covers only the canonical package plus declared quantile/BIC/BDeu recipe, not a paper-native implementation. Official Results and release gates remain pending.
- `nflow`: passed all nine binary, multiclass, regression, and seed cases in `nflows-maf-tabular-recipe-parity-v1` GitHub Actions run `30970260840`; preprocessing, losses, every official model tensor, reloaded raw samples, and final CSV bytes matched exactly. The claim covers only the canonical package plus declared mixed-type MAF recipe, not a paper-native implementation. Official Results and release gates remain pending.
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
- `great`: passed all three seed cases in `be-great-official-package-parity-v1` GitHub Actions run `30974574472`; trained tensors, guided samples, CSV bytes, safe restoration, and caller-state restoration matched exactly. Full-scale pretrained-model quality, Official Results, and release gates remain outside the claim.
- `tabula`: passed all three seed cases in `tabula-method-author-source-parity-v1` GitHub Actions run `30974574505`; trained tensors, exact-row samples, CSV bytes, safe restoration, and caller-state restoration matched exactly. The absent upstream license independently blocks redistribution and release.
- `tabsds`: passed all nine binary, multiclass, regression, and seed cases in `tabsds-official-source-parity-v1` GitHub Actions run `30974574593`; direct-source and adapter DataFrames and CSV bytes matched exactly, including the repeat/truncate boundary. The absent upstream license independently blocks redistribution and release.
- `tabebm`: passed `tabebm-official-package-core-validation-v1` GitHub Actions run `30974574544`; package identity, deterministic official core helpers, safe state, and the adapter delegation boundary were validated. Full gated TabPFN generation did not execute, so the claim stops at `smoke-validated`.

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

These models were reported to have completed at least one local path through the shared CLI. All 20 adapters except TabEBM now have separate retained native-parity evidence. TabEBM has retained smoke evidence with an explicit gated-generation limitation. Historical local executions remain engineering observations rather than independent release claims.

## Final Four Integration Status

- `great`: official `be-great==0.0.14` wheel, installed runtime hashes, safe safetensors/JSON artifact, and three-seed direct-package parity passed in retained run `30974574472`.
- `tabula`: six method-author files are acquired on demand and never patched. Safe persistence and a bounded, exact-row Linux sampling boundary replace the former local reimplementation; three-seed parity passed in retained run `30974574505`. The absent upstream license blocks release independently.
- `tabsds`: the former approximation is replaced by the two official notebook helper files. All nine binary/multiclass/regression and seed cases passed exact source-versus-adapter comparison in retained run `30974574593`. The absent upstream license blocks release independently.
- `tabebm`: official package identity, safe preprocessing state, explicit gated opt-in, and exact-row assembly are implemented. Retained run `30974574544` validates deterministic official helpers and delegation only; real TabPFN-backed generation requires a separately authorized run.

## Historical Environment Caveats

- `torch==2.3.0` is the current pinned runtime in the benchmark stack requirements.
- `transformers==4.46.3` and `tokenizers==0.20.3` are frozen for the official GReaT and TabuLa validation environments.
- Goggle's validated GCN path uses `dgl==1.1.3` and `torch-geometric==2.5.3` without requiring the unused heterogeneous `torch-sparse` path. Requesting the heterogeneous decoder still fails closed unless its official extension stack is available; that path has no parity claim.
- TabDDPM vendors the byte-exact runtime modules from the official `libzero==0.0.8` wheel; TabSyn installs that same distribution without its stale dependency metadata. The similarly named `zero` distribution is unrelated, and neither path uses the former local compatibility substitutes.

## Current Interpretation

- Treat every current adapter as an experimental engineering integration until its evidence record says otherwise.
- GReaT, TabuLa, and TabSDS are `native-parity-validated` for their precisely scoped official targets; TabEBM cannot exceed `smoke-validated` without a separately authorized real TabPFN run.
- The `tabebm` gated-sample preset may be used only on machines whose users have accepted the applicable model terms and configured access.
