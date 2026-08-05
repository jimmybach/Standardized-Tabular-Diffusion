# Upstream Source Audit

Status: release-preparation record
Audit date: 2026-08-04
Scope: the primary TabDDPM, TabDiff, and TabSyn source trees, the TabSyn STaSy and CoDi baseline snapshots, plus official integrations for ARF, CTGAN, TVAE, SMOTE, NRGBoost, REaLTabFormer, TabularARGN, CTAB-GAN, CTAB-GAN+, and Goggle

## Purpose

This audit establishes the immutable upstream revision behind each primary source snapshot, separates adapter changes from upstream-source patches, and prevents patched implementations from being presented as strict native reproductions. The machine-readable source of truth is `standardized_tabular_diffusion/resources/upstream/source-lock.json`.

An exact upstream revision does not by itself make an adapter benchmark-eligible or release-supported. Those states additionally require approved patch treatment, smoke evidence, native-parity evidence, dependency locks, dataset-profile approval, and a frozen evaluation protocol.

## Method

The authoritative repositories were cloned into an ignored audit workspace. Blob identities were compared across repository history rather than only against the latest default branch. Data, generated outputs, checkpoints, and explicitly nested projects were excluded from core-code comparisons. For every primary component, the audit records:

1. authoritative repository, commit, and Git tree;
2. root-repository commit at which the snapshot first appeared;
3. exact and modified shared-path counts;
4. local patch sets, affected files, classification, and review status; and
5. consequences for official-result eligibility.

The comparison counts are scoped evidence, not a claim about nested baselines or transitive packages.

## Findings

| Component | Pinned upstream revision | Snapshot relation | Local source treatment | Current official eligibility |
|---|---|---|---|---|
| ARF (official Python package) | `6f737baaaa589f7ac3ff59f0d739ce04b0f1381c`, tree `68b6fc5d28578a5c21bef560bd28f4c0d2d6401c` (`arfpy==0.1.1`) | The checksum-pinned PyPI source distribution contains six byte-exact release files from the method-author commit. The repository has no 0.1.1 tag; the distinct R package is recorded separately and is outside this cross-language claim. | No package source is vendored or patched. Typed input, deterministic seed scopes, strict OOB rejection, and a safe JSON FORGE-state checkpoint are adapter-only. The checkpoint omits the forest and row-level training data while sampling still calls the official `forge()` method. All nine exact cases passed in retained Linux/Python 3.11 run `30964711614`. | Native parity validated against the exact official Python package; central evaluation, dataset admission, runtime, and release gates remain pending. No R/Python equivalence is claimed. |
| CTGAN | `826da23f8f9385ad15fd206ecad691e04cb0ccdc` (`v0.12.1`) | The adapter previously loaded a nested `0.5.2.dev0` snapshot. It now requires the official PyPI wheel whose SHA-256 and trusted-publishing source commit are locked. | Adapter-only package integration; no 0.12.1 source is vendored. Exact native parity passed in run `30910275922`. | Blocked pending BUSL-1.1 review, central evaluation, dataset admission, and release gates. |
| TVAE | `826da23f8f9385ad15fd206ecad691e04cb0ccdc` (`v0.12.1`) | The former `0.5.2.dev0` subtree matched 39 of 44 shared paths at its closest reviewed history point; TVAE and four other source files were locally modified. | The 47-file legacy subtree and obsolete wrappers were removed. The adapter now uses the unmodified official package API; exact native parity passed in run `30913867621`. | Blocked pending BUSL-1.1 review, central evaluation, dataset admission, and release gates. |
| SMOTE | `8504e95f0160f61d1b617ca66f779646d2ee609e` (`0.14.2`) | The adapter now requires the checksum-pinned official imbalanced-learn wheel. No source snapshot is vendored. | Adapter-only package integration. Direct DataFrame dispatch selects official SMOTE, SMOTENC, or SMOTEN; all nine exact cases passed in run `30918785254`. | Native parity validated but excluded from joint generative-model ranking; classical-reference admission and release gates remain pending. |
| NRGBoost | `feef73a3edb20b911c2f7214b13f810909ef20ad` (`v0.0.3`) | The adapter requires the checksum-pinned method-author Linux/Python 3.11 wheel. No source snapshot is vendored. | Adapter-only package integration. All six classification/regression and seed cases passed exact native parity in run `30922326384`. | Native parity validated; central evaluation, dataset, runtime, and release gates remain blocked. |
| REaLTabFormer | `73f239643f9ea5abc877f685ce927e986302ac2d` (`v0.2.4`) | The adapter requires the checksum-pinned method-author PyPI wheel. Its 11 source files shared with the tagged source archive are byte-exact; the wheel adds only one empty module. No package source is vendored. | Official tabular package integration with recorded output-path, declared-type, seed, safe-load, and v0.2.4 save-serialization boundaries. No upstream file is patched. All nine binary/multiclass/regression and seed cases passed exact Linux/Python 3.11 parity in run `30950369908`. | Native parity validated for official tabular training with `n_critic=0`; blocked pending sensitivity/relational scope decisions, central evaluation, dataset admission, runtime budgets, and release gates. |
| CTAB-GAN | `73d4e315a2a51cf16c97ed8a00d2dad456cfce8a` | The initial 15-file snapshot matched only two of nine latest-official shared paths after line-ending normalization; it changed construction, the stratified split, training controls, and sampling. | The semantic fork was removed. Seven selected official files are checksum-locked and distributed with Apache-2.0 and the original attribution. A documented adapter-only bridge maps the legacy positional `n_components` call to the same keyword-only scikit-learn parameter. All six exact cases passed in retained Linux run `30930939961`. | Native parity validated; central evaluation, dataset, runtime, and release gates remain pending. |
| CTAB-GAN+ | `6a6f90188cca3dac2c533fd5e8e7f20de074365b` | The former TabDDPM-embedded snapshot changed the constructor, train split, critic loop, device/optimizer controls, and sampling behavior; its duplicate source made 18 files and 146,977 bytes. | The snapshot was removed without history rewriting. Five byte-exact official runtime files are downloaded on demand and used without source patches; all six exact native-parity cases passed in run `30926267432`. | Native parity validated, but blocked because the official repository declares no license and because central evaluation, dataset, runtime, and release gates remain pending. |
| Goggle | `1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0`, subtree `6dcaae801859f63e173537445548a50cd1f8625b` | The former TabSyn copy contained 11 files; all nine shared method-author paths differed after normalization, including the fit contract, defaults, validation/early stopping, sampling constraints, and graph imports. | The copy was removed without history rewriting. Eighteen official source, attribution, and environment files are checksum-locked and acquired on demand under MIT. No upstream statement is patched. All nine exact binary/multiclass/regression and seed cases passed in retained Linux/Python 3.11 run `30945676747`. | Native parity validated for the method-author GCN core; central evaluation, dataset admission, heterogeneous decoder runtime, and release gates remain pending. |
| TabDDPM | `b476257dd460b778ba09eb97f7a51d6490fa17f8` | The initial import had 58 exact scoped files but omitted all six official `lib/` files. The missing files have now been restored; all 64 scoped files match the integrity manifest after declared text normalization. | Adapter-only. The former local `zero` shim was removed and replaced by the seven byte-exact modules from the official `libzero==0.0.8` wheel. | Native parity validated in run `30863212268`; official-track eligibility remains a separate pending decision. |
| TabDiff | `5ecdb3356261aea72716cc9a779f31d7ad083bf4` | All 27 files in the frozen validation scope match the pinned source after line-ending normalization. | Adapter-only. The former local evaluator patch was removed and the official file restored. | Native parity validated in run `30866879879`; central-evaluation and other official-track gates remain pending. |
| TabSyn | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` | Of 101 shared source paths, 96 matched and five carried local changes at import. The 20-file primary execution scope has now been restored exactly. | Official source is unmodified; compatibility controls are outside the upstream tree. | Native parity passed; Official Results remain blocked by central-evaluation, dataset, runtime, governance, and release gates. |
| STaSy (TabSyn baseline snapshot) | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`, subtree `4f56a7223d71d6b75c1698824c5d0245bf716bc6` | All 17 local STaSy Python files match the TabSyn snapshot. The separate method-author commit `3dcc660` has no declared license; only 2 of 14 shared paths match after text normalization. | Thirty execution files are checksum-locked; device, seed, effective training controls, row count, and output-local checkpoints are adapter-only. All nine exact snapshot-parity cases passed in retained Linux/Python 3.11 run `30936275831`. | Native parity validated only against the TabSyn snapshot. Original-method, Official Results, and release claims remain blocked. |
| CoDi (TabSyn baseline snapshot) | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`, subtree `85c16ccfb76fbf00db6b30450ca47e9928efa8d3` | All 11 local CoDi files match the TabSyn snapshot byte-for-byte. The separate method-author commit `8da2af2` has no declared license; 5 of 10 shared paths match and 5 differ. | Twenty-four execution files are checksum-locked; device-count compatibility, deterministic controls, exact rows, and output-local checkpoint roots are adapter-only. All nine exact snapshot-parity cases passed in retained Linux/Python 3.11 run `30941940893`. | Native parity validated only against the TabSyn snapshot. Original-method, Official Results, and release claims remain blocked. |

## Patch Classification

### Resolved TabDDPM import and dependency defects

- The initial repository import omitted `lib/__init__.py`, `data.py`, `deep.py`, `env.py`, `metrics.py`, and `util.py`, so the real upstream pipeline could not import `lib`.
- Those six files are now restored from the pinned TabDDPM checkout and covered by the 64-file integrity manifest.
- `tabddpm-libzero-compat-v1` was removed because its `improve_reproducibility` implementation used the same seed for Python, NumPy, and PyTorch, whereas official `libzero==0.0.8` deliberately uses offset seeds. It therefore was not behaviorally equivalent.
- `TabDDPM-main/zero/` now contains the seven byte-exact Python modules from the official `libzero==0.0.8` wheel, along with its MIT license. Vendoring avoids the wheel's legacy `torch<2` dependency metadata while retaining official runtime behavior in the supported Python 3.11 / PyTorch 2.3 validation environment.
- This source repair changes the treatment from `compatibility-patched` to `adapter-only`. The mandatory real protocol subsequently passed all exact comparisons across three seed cases in run `30863212268`; evidence is permanently retained under `docs/evidence/tabddpm/`.

### `tabdiff-mle-evaluator-v1`

- Classification: semantic-patched.
- File: `TabDiff-main/eval/mle/mle.py`.
- Changes include estimator configuration, compute backend, objectives, seeded splitting, failure handling, and edge-case metric semantics.
- Disposition: removed. The pinned method-author file was restored exactly modulo repository line endings. This upstream evaluator remains outside the benchmark's formal leaderboard path; official results must use the separately reviewed central versioned evaluator.

### TabSyn patch disposition

- The six modified official files were restored to commit `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`.
- The local `TabSyn-main/zero/` substitute was removed. The frozen validation environment uses `libzero==0.0.8`, which supplies the `zero` research-utility import used by the source. The PyPI distribution named `zero` is an unrelated circuit-analysis package. Because the 2021 `libzero` wheel carries stale `torch<2` metadata, it is installed without dependencies; its actual dependencies are locked separately and parity-tested with the repository's PyTorch 2.3 stack.
- Device selection, deterministic seeding, requested sample rows, and sampling steps are implemented in `standardized_tabular_diffusion/compat/tabsyn_launcher.py`. This is an adapter-only invocation boundary: it imports and calls the official VAE, latent diffusion, decoding, and EDM sampler implementations without changing files under `TabSyn-main/`.
- The official source does not expose VAE or diffusion epoch counts. The public adapter therefore rejects those former local controls instead of silently relying on patched source.
- A 20-file manifest freezes the primary TabSyn execution, shared data utilities, dependency declaration, and attribution files. Bundled baselines and upstream evaluation scripts remain outside this TabSyn-primary validation scope.

### STaSy snapshot disposition

- The active baseline is the STaSy adaptation distributed by the Apache-2.0 TabSyn repository, not a copy of the separate method-author repository.
- The 17 local STaSy Python files exactly match the pinned TabSyn subtree after declared text normalization. Thirteen shared dispatcher, preprocessing, attribution, and dependency files bring the fail-closed execution scope to 30 files.
- The method-author repository is pinned separately for provenance. It has no detected license and differs in 12 of 14 shared paths, so the TabSyn adaptation cannot be described as an unmodified original-method implementation.
- `standardized_tabular_diffusion/compat/stasy_launcher.py` fixes the invocation contract without editing upstream source: CPU/CUDA selection, deterministic seeding, effective training configuration, exact requested rows, and checkpoint isolation under `output_dir`.
- Its `stasy-sklearn-onehot-keyword-v1` bridge forwards the snapshot's unchanged `sparse=False` value to the renamed `sparse_output` parameter required by scikit-learn 1.5.2; encoder type and dense-output semantics are unchanged.
- The nine-case Linux/Python 3.11 snapshot-parity protocol passed in run `30936275831`. Its inspected evidence is retained under `docs/evidence/stasy/`, so the adapter is `native-parity-validated` against the TabSyn snapshot only.

### CoDi snapshot disposition

- The active baseline is the CoDi adaptation distributed by the Apache-2.0 TabSyn repository, not an unmodified copy of the separate method-author repository.
- All 11 local CoDi files match the pinned TabSyn subtree byte-for-byte. Thirteen shared dispatcher, preprocessing, dependency, and attribution files bring the fail-closed execution scope to 24 files.
- The method-author repository is pinned separately for provenance. It has no detected license; 5 of 10 shared paths differ, and TabSyn adds a separate sampling entry point. Original-method equivalence is therefore blocked.
- `standardized_tabular_diffusion/compat/codi_launcher.py` changes no tracked upstream file. Its module-local bridges prevent the CPU device-count division by zero, isolate both official state-dict checkpoints under `output_dir`, and resize only post-transform sampling placeholders to honor exact requested rows.
- The nine-case Linux/Python 3.11 snapshot-parity protocol passed in run `30941940893`. Both checkpoint states and generated CSV bytes matched exactly in every case. Its inspected evidence is retained under `docs/evidence/codi/`, so the adapter is `native-parity-validated` against the TabSyn snapshot only.

### Goggle snapshot and compatibility disposition

- The former `TabSyn-main/baselines/goggle` copy is removed. Its changed fit interface, defaults, validation selection, checkpoint behavior, and graph imports prevented it from serving as method-author evidence.
- The active source is acquired from `vanderschaarlab/GOGGLE` commit `1a3d87ad`. The archive plus 18 selected runtime, packaging, and attribution files are checksum-locked; the source has an MIT license and is never patched.
- `standardized_tabular_diffusion/compat/goggle_launcher.py` confines the official relative checkpoint write to `output_dir`, passes an explicit requested row count to the unchanged core sampler, and supplies import-only boundaries for legacy Synthcity evaluator names and the unused RGCN symbol on the GCN path.
- Numerical standardization and categorical one-hot encoding are fitted only on real training rows. Missing values fail closed pending the centralized train-only mean/mode imputer.
- The nine-case Linux/Python 3.11 protocol passed in run `30945676747`: complete checkpoint state, raw core samples, final frames, and CSV bytes were exact in all cases. The claim covers the method-author GCN core; SAGE and heterogeneous RGCN paths are not promoted by this evidence.

### CTAB-GAN snapshot and compatibility disposition

- The initial CTAB-GAN subtree contained 15 files and 78,185 Git-blob bytes. Nine paths were shared with the pinned latest official tree; only `LICENSE` and `License.txt` matched after line-ending normalization, while seven shared paths differed.
- The active source is now limited to the official license, attribution notice, README, and four generation runtime files from commit `73d4e315`. Dataset artifacts, notebooks, generated outputs, local wrappers, and upstream evaluation code are excluded.
- The selected files are frozen after the same documented LF/one-final-newline normalization used by the source audit. No executable statement is edited.
- Python 3.11 cannot use the official scikit-learn 0.24.1 dependency. The adapter-only `ctabgan-sklearn-keyword-only-v1` bridge forwards the unchanged positional mixture-component count as the same `n_components` keyword required by scikit-learn 1.5.2. It is applied independently to native and adapter paths and changes no algorithm parameter.
- The official `DataPrep` always stratifies its target. The registry therefore exposes CTAB-GAN as classification-only instead of claiming an unsupported regression path.

### TabularARGN package and artifact disposition

- The active adapter targets method-author `mostly-ai/mostlyai-engine` tag `2.6.2`, commit `0b96f02e`, distributed as the official `mostlyai-engine==2.6.2` wheel under Apache-2.0. No package source is vendored.
- The GitHub archive, PyPI wheel, PyPI source distribution, source/wheel license, package metadata, and all 53 hash-bearing installed files are checksum-locked. All 50 wheel package source files match the tagged source byte-for-byte.
- No official source statement is patched. Output placement, DatasetSpec-directed categorical coercion, explicit sampling seeding, and the official rare-category `SAMPLE` option are adapter arguments or pre/postconditions.
- The former adapter's unrestricted estimator pickle has been removed. The persistent artifact is the official `ModelStore`; official 2.6.2 uses `weights_only=True` for model weights. The adapter verifies all retained files and removes `OriginalData`, including raw and encoded row files, after fit.
- All nine binary-classification, multiclass-classification, regression, and seed cases passed the retained Linux/Python 3.11 official-package parity protocol in run `30961590047`. The flat single-table unconditional-generation path is therefore `native-parity-validated`. Sequential, relational, differential-privacy, prediction, likelihood, and imputation APIs remain outside this validated scope.

## Artifact Disposition

Six Adult TabSyn checkpoint artifacts and one Sick VAE checkpoint, totaling approximately 93.5 MB, were present in the root repository import but not in the pinned official upstream tree. They have been removed from the working tree because provenance, redistribution permission, training configuration, and checksum evidence were absent. Their original paths, byte sizes, and SHA-256 values are recorded in the source lock. Recoverable local copies remain under ignored `tmp/` quarantine directories during review. Git history has not been rewritten.

Legacy Adult data directories contained duplicated raw archives, processed tables, NumPy arrays, and synthetic-layout mirrors that were not bound to the reviewed build identity. They have been removed from active tracked paths and replaced by a checksum-pinned local builder. Generated data are ignored and reproducible from UCI; a recoverable local quarantine remains during review. This working-tree cleanup does not rewrite historical commits.

The legacy `TabDDPM-main/CTGAN/` subtree contained a locally modified `ctgan` `0.5.2.dev0` snapshot plus TVAE wrapper scripts. Its 47 tracked files (168,098 bytes) were removed after both CTGAN and TVAE adapters were migrated to the checksum-pinned official package. Git history was not rewritten; detailed comparison counts and the removed TVAE source hash are retained in the source lock.

## License Notes

- The official CTGAN package used by both CTGAN and TVAE declares BUSL-1.1, not an OSI open-source license. It is installed optionally and is not vendored. Validation is permitted research work, but Official Results and release support remain blocked pending an explicit review of the upstream use restrictions.
- The official imbalanced-learn 0.14.2 package used by SMOTE declares MIT. The wheel and source license hashes are locked separately because packaging normalizes the license file; no package source is vendored.
- The official NRGBoost 0.0.3 package declares MIT. Its method-author tag, Trusted Publishing source commit, Linux/Python 3.11 wheel, source and wheel license hashes, compiled extension, and bundled OpenMP runtime are locked; no package source is vendored.
- The official REaLTabFormer 0.2.4 package declares MIT. Its method-author tag, source archive, universal wheel, source/wheel license, package metadata, all 16 hash-bearing installed files, and frozen Linux/Python 3.11 direct runtime are locked; no package source is vendored.
- The official mostlyai-engine 2.6.2 package used for TabularARGN declares Apache-2.0. Its method-author tag, verified commit, source archive, universal wheel, source distribution, byte-exact source/wheel license, all 53 hash-bearing installed files, and frozen Linux/Python 3.11 direct runtime are locked; no package source is vendored.
- The official CTAB-GAN+ repository declares no license file or expression. Public visibility does not grant redistribution rights. Its source is therefore fetched directly from the method-author repository into an ignored user cache, never committed here, and excluded from release support and Official Results pending explicit clarification.
- The official CTAB-GAN repository declares Apache-2.0 and includes a separate attribution notice. Both files remain beside the selected source. Source redistribution is authorized, but release support remains blocked by non-license gates.
- The official Goggle repository declares MIT and identifies Copyright 2023 Tennison Liu. The license, authorship file, archive, and selected source files are checksum-locked. Source is acquired on demand; the license is not the remaining release blocker.
- The TabDDPM snapshot carries its upstream MIT license; the vendored official `libzero==0.0.8` modules carry their separate upstream MIT license.
- The TabDiff license file is byte-for-byte identical to the pinned upstream file. Its malformed quote characters are therefore an upstream defect, not local corruption; the original attribution is preserved.
- The TabSyn snapshot carries Apache-2.0 license and NOTICE files. Its bundled baseline directories require their own source, license, and patch audit.

## Release Gates

Before any of these adapters can enter the Official Results track:

1. each retained patch set must be approved and represented as a reviewable patch;
2. exact dependencies and executable environments must be locked;
3. native commands and standardized adapters must be compared on frozen smoke cases;
4. outputs and key metrics must satisfy documented parity tolerances;
5. no unproven checkpoint may be treated as an official pretrained artifact; and
6. the model evidence record must be promoted independently through `registered`, `adapter-complete`, `smoke-validated`, `native-parity-validated`, `benchmark-eligible`, and `release-supported` states.

Until all applicable gates are met, registry records remain conservative. ARF, CTAB-GAN, CTAB-GAN+, CTGAN, TVAE, NRGBoost, REaLTabFormer, Goggle, TabDDPM, TabDiff, TabularARGN, TabSyn, STaSy, and SMOTE have retained native-parity claims, but remain experimental, unsupported, and excluded from Official Results. ARF's claim is limited to the official Python package and does not establish R/Python equivalence; TabularARGN's claim is limited to flat single-table unconditional generation; Goggle's claim is limited to its method-author GCN core; REaLTabFormer's claim is limited to official tabular training with `n_critic=0`; STaSy's claim is limited to the TabSyn benchmark snapshot; SMOTE is additionally and categorically excluded from the joint generative-model ranking.
