# Upstream Source Audit

Status: release-preparation record
Audit date: 2026-08-03
Scope: the primary TabDDPM, TabDiff, and TabSyn source trees and the official CTGAN package

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
| CTGAN | `826da23f8f9385ad15fd206ecad691e04cb0ccdc` (`v0.12.1`) | The adapter previously loaded a nested `0.5.2.dev0` snapshot. It now requires the official PyPI wheel whose SHA-256 and trusted-publishing source commit are locked. | Adapter-only package integration; no 0.12.1 source is vendored. Exact native parity passed in run `30910275922`. | Blocked pending BUSL-1.1 review, central evaluation, dataset admission, and release gates. |
| TabDDPM | `b476257dd460b778ba09eb97f7a51d6490fa17f8` | The initial import had 58 exact scoped files but omitted all six official `lib/` files. The missing files have now been restored; all 64 scoped files match the integrity manifest after declared text normalization. | Adapter-only. The former local `zero` shim was removed and replaced by the seven byte-exact modules from the official `libzero==0.0.8` wheel. | Native parity validated in run `30863212268`; official-track eligibility remains a separate pending decision. |
| TabDiff | `5ecdb3356261aea72716cc9a779f31d7ad083bf4` | All 27 files in the frozen validation scope match the pinned source after line-ending normalization. | Adapter-only. The former local evaluator patch was removed and the official file restored. | Native parity validated in run `30866879879`; central-evaluation and other official-track gates remain pending. |
| TabSyn | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` | Of 101 shared source paths, 96 matched and five carried local changes at import. The 20-file primary execution scope has now been restored exactly. | Official source is unmodified; compatibility controls are outside the upstream tree. | Native parity passed; Official Results remain blocked by central-evaluation, dataset, runtime, governance, and release gates. |

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

## Artifact Disposition

Six Adult TabSyn checkpoint artifacts and one Sick VAE checkpoint, totaling approximately 93.5 MB, were present in the root repository import but not in the pinned official upstream tree. They have been removed from the working tree because provenance, redistribution permission, training configuration, and checksum evidence were absent. Their original paths, byte sizes, and SHA-256 values are recorded in the source lock. Recoverable local copies remain under ignored `tmp/` quarantine directories during review. Git history has not been rewritten.

Legacy Adult data directories contained duplicated raw archives, processed tables, NumPy arrays, and synthetic-layout mirrors that were not bound to the reviewed build identity. They have been removed from active tracked paths and replaced by a checksum-pinned local builder. Generated data are ignored and reproducible from UCI; a recoverable local quarantine remains during review. This working-tree cleanup does not rewrite historical commits.

## License Notes

- CTGAN 0.12.1 declares BUSL-1.1, not an OSI open-source license. The official package is installed optionally and is not vendored. Validation is permitted research work, but Official Results and release support remain blocked pending an explicit review of the upstream use restrictions.
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

Until all applicable gates are met, registry records remain conservative. CTGAN, TabDDPM, TabDiff, and TabSyn have retained native-parity claims, but remain experimental, unsupported, and excluded from Official Results.
