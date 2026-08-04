# Upstream Source Audit

Status: release-preparation record
Audit date: 2026-08-03
Scope: the primary TabDDPM, TabDiff, and TabSyn source trees

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
| TabDDPM | `b476257dd460b778ba09eb97f7a51d6490fa17f8` | The imported core matched all 58 compared upstream blobs. | Local `zero` compatibility substitute; no generative-algorithm source change found. | Blocked pending shim removal or approval plus native-parity validation. |
| TabDiff | `5ecdb3356261aea72716cc9a779f31d7ad083bf4` | The initial non-data snapshot matched all 30 compared upstream blobs. | `eval/mle/mle.py` was changed later and is a semantic evaluator patch; the generative core remains upstream-exact within the audited scope. | Patched upstream evaluator is excluded from official metrics; central reviewed evaluation must be used. |
| TabSyn | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` | Of 101 shared source paths, 96 matched and five carried local changes at import. The 20-file primary execution scope has now been restored exactly. | Official source is unmodified; compatibility controls are outside the upstream tree. | Blocked only pending a passing retained native-parity record. |

## Patch Classification

### `tabddpm-libzero-compat-v1`

- Classification: compatibility-patched.
- Files: `TabDDPM-main/zero/__init__.py`, `hardware.py`, and `random.py`.
- Reason: substitutes locally for the upstream `libzero` dependency.
- Decision: unapproved and unvalidated. It cannot support a native-parity claim until behavior is compared with the upstream dependency or the shim is removed.

### `tabdiff-mle-evaluator-v1`

- Classification: semantic-patched.
- File: `TabDiff-main/eval/mle/mle.py`.
- Changes include estimator configuration, compute backend, objectives, seeded splitting, failure handling, and edge-case metric semantics.
- Decision: experimental only. This file is not part of the generative algorithm, but using it would change benchmark measurements. Official results must use the central versioned evaluator after metric validation.

### TabSyn patch disposition

- The six modified official files were restored to commit `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`.
- The local `TabSyn-main/zero/` substitute was removed. The frozen validation environment uses `libzero==0.0.8`, which supplies the `zero` research-utility import used by the source. The PyPI distribution named `zero` is an unrelated circuit-analysis package.
- Device selection, deterministic seeding, requested sample rows, and sampling steps are implemented in `standardized_tabular_diffusion/compat/tabsyn.py`. This is an adapter-only invocation boundary: it imports and calls the official VAE, latent diffusion, decoding, and EDM sampler implementations without changing files under `TabSyn-main/`.
- The official source does not expose VAE or diffusion epoch counts. The public adapter therefore rejects those former local controls instead of silently relying on patched source.
- A 20-file manifest freezes the primary TabSyn execution, shared data utilities, dependency declaration, and attribution files. Bundled baselines and upstream evaluation scripts remain outside this TabSyn-primary validation scope.

## Artifact Disposition

Six Adult TabSyn checkpoint artifacts and one Sick VAE checkpoint, totaling approximately 93.5 MB, were present in the root repository import but not in the pinned official upstream tree. They have been removed from the working tree because provenance, redistribution permission, training configuration, and checksum evidence were absent. Their original paths, byte sizes, and SHA-256 values are recorded in the source lock. Recoverable local copies remain under ignored `tmp/` quarantine directories during review. Git history has not been rewritten.

Legacy Adult data directories contained duplicated raw archives, processed tables, NumPy arrays, and synthetic-layout mirrors that were not bound to the reviewed build identity. They have been removed from active tracked paths and replaced by a checksum-pinned local builder. Generated data are ignored and reproducible from UCI; a recoverable local quarantine remains during review. This working-tree cleanup does not rewrite historical commits.

## License Notes

- The TabDDPM snapshot carries its upstream MIT license.
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

Until those gates are met, registry records remain conservative: experimental track, unsupported, and no native-parity claim.
