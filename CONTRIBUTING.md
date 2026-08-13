# Contributing

Thank you for improving Standardized Tabular Diffusion. The repository is a
research benchmark, so reproducibility, provenance, and faithful treatment of
upstream implementations are part of correctness.

## Before contributing

- Open an issue or discussion before changing an algorithm implementation,
  evaluation definition, benchmark protocol, dataset split, or preprocessing
  rule.
- Do not copy code, data, papers, checkpoints, or model weights without a
  documented source, compatible license or terms, and retained attribution.
- Prefer an unchanged official package or checksum-pinned upstream source plus
  a repository-owned adapter. Any unavoidable source patch needs prior review,
  a patch record, and equivalence validation.
- Never commit credentials, private data, raw uploads, generated checkpoints,
  or local experiment artifacts.

## Development checks

Use Python 3.11 on Windows 11 for the primary release path. Run the focused
tests for your change, then the complete test suite, lint, and type checks that
apply to the affected surface. New behavior should include deterministic tests
and machine-readable evidence where the repository's validation policy requires
it.

## Developer Certificate of Origin

This project uses the Developer Certificate of Origin 1.1 rather than a
Contributor License Agreement. By adding a `Signed-off-by` trailer, you certify
that you have the right to submit the contribution under this repository's
license and contribution terms. The certificate is published at
<https://developercertificate.org/>.

Sign each commit with:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Git can add this trailer with `git commit -s`. Contributions without a valid
sign-off may be returned for correction before merge.

## Attribution

Git history is the authoritative fine-grained contribution record. Material
contributors should also be added to `CONTRIBUTORS.md`; repository transfer or
maintenance changes must not erase existing credit.
