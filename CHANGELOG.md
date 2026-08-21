# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning for the software interface; scientific protocol versions are governed independently.

## 0.1.0rc1 - Unreleased

### Added

- Twenty-one registered model adapters with explicit source, modification, validation, benchmark-track, support, licensing, and evidence fields.
- Versioned P1–P5 evaluation contracts, Atomic Results, finalized Result Bundles, structural gates, and diagnostic metrics.
- P6 resource-aware orchestration and P7 deterministic diagnostic/leaderboard snapshot publication.
- Train-only mean/mode missing-value preprocessing, dataset acquisition/materialization, and checksum-locked upstream source handling.
- A packaged artificial SMOTE quickstart that reaches a validated, unranked diagnostic snapshot.
- An explicit user-workspace boundary and a non-Official P3 diagnostic Dataset Profile generator for installed-package workflows.
- Read-only legacy summary import, citation metadata, contributor attribution, security guidance, code of conduct, and release checklist.
- A bilingual authoritative whole-project roadmap that separates project Phases 1-10 from the P0-P8 implementation packages and V0-V3 validation levels.

### Changed

- Public adapter evaluation now uses the central versioned evaluation engine.
- Windows 11 with Python 3.11 is the primary release target; Linux with Python 3.11 is the required secondary compatibility family.
- Model inventory claims use independent validation, eligibility, and support dimensions instead of a generic “implemented” label.
- Dataset discovery, orchestration workers, example configuration, and documentation now preserve the same explicit workspace from download through Result Bundle finalization.
- Distribution inspection now rejects unsafe archive members, credentials, runtime/development outputs, retained evidence, and developer-local paths.

### Removed

- Supported generation of `standardized_summary.json` and adapter-local `tabstruct-aligned-v1` evaluation.
- Several non-authoritative or modified baseline copies in favor of pinned official packages or source.

### Important boundaries

- This software release candidate does not publish Official Results.
- No registered adapter is automatically benchmark-eligible or release-supported.
- Third-party code, packages, datasets, papers, and weights retain their own terms.
