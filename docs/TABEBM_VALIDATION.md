# TabEBM Validation Protocol

Status: `smoke-validated`; full gated TabPFN generation was not executed

Protocol: `tabebm-official-package-core-validation-v1`

Target: method-author `tabebm==2025.8.19` package with `tabpfn==2.1.2`

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This is deliberately a `smoke-validated` gate, not native parity. Full official `generate()` constructs TabPFN-v2 estimators and may download weights that require accepted external terms and credentials. Public CI must not bypass or pretend to accept those terms. The protocol therefore verifies package/source identity, deterministic official core helpers, safe preprocessing/checkpoint behavior, and the exact official-call boundary with a test double. It records `full_tabpfn_generation_executed=false` explicitly.

## Audited Distribution

The authority is [andreimargeloiu/TabEBM](https://github.com/andreimargeloiu/TabEBM) at commit and release tag `72eb78dab896c7a8f39c4dcc288c834fd72eff2b`, tree `627af984e9447bf1a88f1d13e4c766704738ec28`. The PyPI source distribution `tabebm-2025.8.19.tar.gz` is 19,178 bytes, SHA-256 `6111611326747a680f93dfadcbac1d602ce20cb722b9b6cbff1f556b9f48d503`, under Apache-2.0. Both installed package files are byte-exact with the locked tag.

## Adapter Boundaries

TabEBM supports classification only. The adapter requires one target, at least two target classes, at least one feature, complete feature roles, finite numerical data, and no missing values. Numerical features are standardized with training-split mean and population standard deviation; categorical features and the target use deterministic sorted mappings. Typed JSON stores this state and the training-file digest without retaining rows.

Sampling fails closed unless `allow_gated_model=true` or the equivalent explicit environment opt-in is supplied. After opt-in, the adapter calls the official `TabEBM.generate()` with encoded `X`, `y`, the declared SGLD controls, and the standardized seed. Official output is per class, so the adapter requests `ceil(N/classes)` rows per class and uses deterministic class-sorted round-robin truncation to exactly `N`. CPU requests are enforced even though upstream selects its device from `torch.cuda.is_available()`.

## Mandatory Smoke Checks

The workflow validates every regular source-distribution member, critical source hashes, metadata, installed runtime hashes, and installed `RECORD`. It directly executes official energy computation, seeded surrogate-negative construction, and the full-train split helper. Binary and multiclass adapter cases verify safe JSON state, exact argument delegation, class-block validation, inverse transformation, and exact row count through a deterministic stand-in for the gated `generate()` body.

## Promotion Requirements

Promotion beyond `smoke-validated` requires a separately authorized Linux run with accepted TabPFN terms and available credentials. That run must execute the real official generation path for multiple seeds and classification fixtures; compare direct and adapter inputs, per-class outputs, final rows, and serialized CSV; retain dependency/model identities; and pass the central evaluation and resource gates independently.

## Known Boundaries

- No regression support is claimed by the official method or adapter.
- Smoke validation does not establish equality of real TabPFN-backed samples.
- TabPFN model terms, credentials, cache handling, and artifact access controls remain the operator's responsibility.
- No differential-privacy guarantee is implied.

## Evidence

GitHub Actions run [`30974574544`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30974574544) passed on Linux with Python 3.11.15. It verified the locked distribution and installed package, executed deterministic official core helpers, and passed binary and multiclass safe-state/delegation boundary cases. The inspected JSON is retained byte-for-byte at `docs/evidence/tabebm/smoke-validation-run-30974574544.json` with SHA-256 `8d461e440440d73213f31efe1b8086e9c78fed299822da2fe203ea62af3c21dc` and explicitly records `full_tabpfn_generation_executed=false`. This evidence supports `smoke-validated`, not native parity.
