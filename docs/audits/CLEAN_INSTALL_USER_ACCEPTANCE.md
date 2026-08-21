# Clean-Install and User-Journey Acceptance

Status: passed locally on 2026-08-20 for an uncommitted candidate based on repository commit `6b1669e415e7b6589e5fb04ba880405efbd5a788`.

Whole-project placement: this is local acceptance evidence for [Project Roadmap Phase 9](../PROJECT_ROADMAP.md). Phase 9 remains integration-pending until the exact candidate is committed, reviewed, merged, and revalidated by the applicable hosted gates.

## Claim boundary

This acceptance establishes that the candidate can be built, installed, and used from outside a source checkout on the primary native Windows/Python 3.11 target. It covers distribution hygiene, user-owned workspaces, official dataset acquisition and preprocessing, representative CPU/GPU pipelines, central evaluation, and Result Bundle validation.

It does not admit a dataset, model, run, metric, or result to Official Results. It does not promote an adapter to `release-supported`, assess generator quality, generalize the recorded GPU result to other hardware, or replace the per-model upstream-parity evidence. The candidate contains uncommitted changes, so the release checklist remains open until these gates are repeated for the exact committed release candidate.

## Accepted environment

- Native Windows 11 x86-64
- CPython 3.11.15
- CPU path: official `imbalanced-learn==0.14.2` SMOTE
- GPU path: official `ctgan==0.12.1`, PyTorch 2.8.0+cu128, CUDA 12.8
- GPU: NVIDIA GeForce RTX 5080
- Dataset: checksum-locked official UCI Adult train/test files

## Acceptance matrix

| Surface | Acceptance action | Result |
|---|---|---|
| Build | Build one wheel and one source archive with Python 3.11 | Pass |
| Archive safety | Inspect required members, unsafe paths, links/devices, duplicates, runtime outputs, credentials, retained evidence, and developer-local paths | Pass |
| Wheel install | Install the wheel plus `quickstart` dependencies in a new virtual environment; run `pip check` | Pass |
| Source install | Install the source archive plus `quickstart` dependencies in a separate new virtual environment; run `pip check` | Pass |
| Installed-package execution | Run package imports, metadata commands, quickstart, Result Bundle validation, and snapshot validation outside the checkout | Pass |
| Path portability | Repeat installed workflows below directories containing spaces and Chinese characters | Pass |
| Dataset acquisition | Download the official Adult archive over HTTPS and verify its registered SHA-256 before extraction | Pass |
| Workspace isolation | Materialize Adult into an explicit user workspace, rediscover it in later CLI processes, and verify its manifest | Pass |
| Missing values | Fit numerical means and categorical modes on training data only, then transform an independent test split without refitting | Pass |
| CPU pipeline | Execute Adult SMOTE train, sample, P3 evaluation, bundle finalization, and independent validation | Pass |
| GPU pipeline | Execute bounded Adult CTGAN CUDA train, sample, P3 evaluation, bundle finalization, and independent validation | Pass |
| CLI and docs | Validate the documented workspace, profile, configuration, run, error-boundary, and output-directory behavior | Pass |

## Representative results

The installed CPU journey materialized the exact 32,561-row Adult training split and 16,281-row test split. Its train-only preprocessing check learned numerical mean `20.0` and categorical mode `NY`; those training statistics filled the independent test fixture. The final-wheel SMOTE run generated 128 rows and finalized Result Bundle `run-c2a5d562b1b542e4ba73a3897d23f50a` with 18 present metric scopes, zero pending scopes, and two declared not-applicable scopes.

The installed GPU journey first demonstrated the expected failure boundary: installing ordinary CTGAN extras from the default package index resolved a CPU-only PyTorch build. After explicitly installing the official CUDA 12.8 PyTorch build, the environment reported PyTorch 2.8.0+cu128, CUDA availability, and the intended RTX 5080. The final wheel then completed a real one-epoch CTGAN run, generated 32 rows, and finalized Result Bundle `run-c72ffaa7091c40d4b891a747e59942a1` with 18 present metric scopes, zero pending scopes, and two declared not-applicable scopes.

These counts establish pipeline completeness, not favorable metric values or scientific quality.

## Defects found and corrected

1. Installed dataset commands defaulted to the package location instead of a user-owned workspace.
2. A materialized dataset manifest could not create a discoverable DatasetSpec when the upstream source checkout was absent.
3. The explicit workspace was lost by top-level runs and P6 subprocess workers.
4. `example-config` emitted evaluation settings that were not directly runnable and did not expose essential seed, device, sample-count, or profile controls.
5. The legacy-profile importer was correctly non-Official but offered no safe installed-package route to a P3 functionality profile; a deliberately diagnostic, non-eligible generator was added.
6. Source archives could include retained evidence containing machine-local paths; evidence is now excluded, and both archive types are inspected for sensitive or non-portable content.
7. The release workflow installed wheel and source archive sequentially in one environment; it now creates a fresh environment for each artifact.
8. Documentation could imply that installing a model extra guaranteed CUDA. It now requires an explicit official PyTorch CUDA installation and device verification for a GPU claim.
9. Archive path validation could depend on the CI host's path rules. It now rejects POSIX/Windows absolute paths, Windows-reserved or invalid names, and case-insensitive path collisions identically on Windows and Linux.

The 2026-08-21 integration review then passed the full available suite on Python 3.11.15 (`638 passed`, `11` optional-dependency skips) and Python 3.13.5 (`640 passed`, `9` optional-dependency skips), plus repository-wide Ruff, Mypy over 36 source files, and fresh Python 3.11 wheel/source builds with archive inspection. Exact-commit clean installation remains assigned to the hosted Windows/Linux release gate so that it is performed after commit rather than retroactively attributed to the earlier local artifacts.

## Retained evidence and remaining release gates

The portable machine-readable acceptance record is [clean-install-user-acceptance-windows-py311-6b1669e.json](../evidence/audits/clean-install-user-acceptance-windows-py311-6b1669e.json). Distribution archives deliberately exclude `docs/evidence/`; evidence remains in repository history and is not shipped as package payload.

Before a tagged release, repeat the same acceptance against the exact committed candidate, pass hosted Windows-family and Linux/Python 3.11 release CI, re-run the security/history review, and install the published artifacts by checksum. Those steps are release gates, not corrections to this local engineering result.
