# Core Baseline Integration Validation

Status: passed on the combined Linux/Python 3.11 candidate

Candidate commit: `9da6e556a091f2501af4cd80ed938feaccb34055`

## Purpose

TabDDPM, TabDiff, and TabSyn first passed independent native-parity protocols on branches derived from the repository-audit baseline. This integration validation establishes that all three audited implementations, adapters, dependency resolutions, manifests, evidence records, and conservative status declarations can coexist in one repository state. It does not replace the model-specific protocols or broaden their claims.

The cumulative branch preserves the complete commit histories of the three independent validation branches. Its root remains `codex/repository-audit`; no validation PR or audit PR was merged or closed as part of this work.

## Integration review

The integration resolved shared changes in the registry, model inventory, source lock, upstream audit, runtime status, third-party notices, validation package, and tests. The review also found and corrected two cross-branch defects:

- TabDDPM had been promoted in the adapter registry but not in the separate model inventory. Both now report `native-parity-validated`.
- Restoring the primary TabSyn source must not imply that its separately vendored CoDi baseline was audited. CoDi remains conservatively classified as `compatibility-patched`.

A Windows checkout also exposed that the TabDDPM libzero license used a raw-byte hash while Git could normalize its line endings. The manifest now explicitly records `license_sha256_lf`, and the validator proves that LF and CRLF checkouts produce the same canonical hash. Official Python modules remain byte-exact; primary source files remain hash-checked using their declared canonical line-ending policy.

## Combined source and evidence checks

All source checks passed in the same working tree:

| Model | Pinned source | Scoped files | Validation cases |
|---|---|---:|---|
| TabDDPM | `b476257dd460b778ba09eb97f7a51d6490fa17f8` | 64 official files plus 7 exact libzero modules | `(training, sampling)` seeds `(0, 23)`, `(17, 47)`, `(101, 89)` |
| TabDiff | `5ecdb3356261aea72716cc9a779f31d7ad083bf4` | 27 | official deterministic seed `0` |
| TabSyn | `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7` | 20 | seeds `0`, `19`, `73` |

The three previously retained evidence files still match their declared SHA-256 values. Fresh evidence was then produced from the single combined candidate:

| Check | GitHub Actions run | Artifact ID | Artifact digest |
|---|---:|---:|---|
| Core CI | [30873942339](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942339) | — | — |
| TabDDPM native parity | [30873942377](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942377) | `8878855123` | `sha256:72da27768abb9c92a1e4b04932f80e2d18691d304baf65853674d8ce00e90f5d` |
| TabDiff native parity | [30873942340](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942340) | `8878859972` | `sha256:dc4e3b9d5a2426a1354451d334af6fb82a65a1be8b3b9c909b619950912244c1` |
| TabSyn native parity | [30873942394](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30873942394) | `8878857161` | `sha256:5ee1339c00528bdc3a514d868ae1198e5e579be95766a443437ca83a3e187473` |

Each downloaded evidence file identified the combined candidate commit and passed the same exact comparison assertions as its permanent model-specific record. The machine-readable integration index is `docs/evidence/core-baselines/native-parity-integration-9da6e55.json`, with SHA-256 `08aebe5409ffb88c980f24e0d36b12b589930717f6c7dd6b473749e79a36c860`.

## Local quality gates

- Ruff: passed.
- mypy: passed for all 14 configured source files.
- pytest: 161 passed, 3 skipped; the skips are documented optional-runtime/platform cases.
- source distribution and wheel build: passed.
- all three source-integrity validators: passed together.

Trailing whitespace reported by a repository-wide Git diff is confined to byte- or canonical-hash-verified official upstream files. It is intentionally retained because rewriting those files would invalidate the authoritative source relation.

## Claim boundary and invalidation

All three models are `native-parity-validated`, `experimental`, and `unsupported`. They are not `benchmark-eligible`, are excluded from Official Results, and are not `release-supported`. Dataset admission, central evaluation, model-quality benchmarking, runtime thresholds, privacy/fairness review, dependency maintenance, and release ownership remain independent gates.

Any change to a primary source scope, source manifest, locked validation dependency, adapter command mapping, or model-specific protocol invalidates the affected parity record and requires a new run. Documentation-only integration updates do not alter the validated candidate behavior.
