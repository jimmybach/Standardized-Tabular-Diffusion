# Security Policy

This repository is a pre-alpha research workspace and does not yet have a supported security release. Report suspected vulnerabilities privately to the laboratory maintainer; do not include credentials, private data, or exploitable artifacts in a public issue.

## Trust boundaries

- Treat experiment configuration, dataset metadata, CSV files, result bundles, and downloaded model files as untrusted input until their provenance and schema have been verified.
- Pickle and many PyTorch/model checkpoint formats can execute code while loading. The standardized adapters refuse symlinked artifacts and, by default, refuse code-executing checkpoints outside the configured `output_dir`. `allow_unsafe_external_checkpoint=true` is an explicit trust decision, not a compatibility setting.
- NumPy object arrays require pickle-backed loading. Repository-owned adapters must not load such arrays when the canonical CSV representation is sufficient.
- Dataset identifiers are portable identifiers, not paths. Registration rejects separators, traversal, uppercase/non-ASCII identifiers, and Windows-reserved names.
- Dataset registration never drops or imputes missing values silently. A dataset containing missing values must pass through the separately reviewed preprocessing module before registration.
- Never commit API keys, tokens, private dataset rows, raw uploads, generated checkpoints, or local experiment artifacts. Use secret scanning before every public release.

## Before a public release

The maintainers must complete the license and third-party review, pin upstream revisions and package versions, audit repository history for secrets and restricted data, publish checksums for supported artifacts, and run the supported Linux/Python 3.11 CI and security test suite.
