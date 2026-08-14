# Release Checklist

This checklist governs a software release. “Pass” here never admits a metric, dataset, model, run, or result to Official Results.

## Identity and legal

- [x] Root project license is Apache-2.0 and scoped by `NOTICE`.
- [x] Third-party components and unresolved rights are recorded in `THIRD_PARTY_NOTICES.md`.
- [x] Contributor credit and Git history preservation are documented.
- [x] Citation, contribution terms, security policy, and code of conduct exist.
- [ ] Re-run repository-history secret and restricted-data audit immediately before tagging.
- [ ] Confirm every file distributed in the sdist/wheel belongs to the reviewed allowlist.

## Engineering

- [ ] Clean Windows/Python 3.11 install, build, core tests, lint, and typing pass at the release commit.
- [ ] Native Windows 11/Python 3.11 quickstart and validation pass at the release commit.
- [ ] Linux/Python 3.11 portable release surfaces pass at the release commit.
- [ ] Wheel and sdist install in clean environments and contain the packaged quickstart inputs and schemas.
- [ ] Documentation links and commands pass without developer-local paths.

## Scientific and publication boundaries

- [x] Legacy and Result Bundle outputs have different schemas, filenames, CLI labels, and documentation.
- [x] Legacy import cannot create Atomic Results or Official eligibility.
- [x] Diagnostic snapshots contain no Official ranks.
- [x] Adapter validation, benchmark eligibility, release support, and result admission remain independent.
- [ ] Any release-note scientific claim is traceable to retained evidence at the release commit.

## Tagging

- [ ] Replace “Unreleased” in `CHANGELOG.md` with the UTC release date.
- [ ] Confirm `pyproject.toml`, `CITATION.cff`, tag, and release title use the same version.
- [ ] Sign and push the tag only after all applicable items above pass.
- [ ] Attach checksums and independently install the published artifacts before marking the release final.
