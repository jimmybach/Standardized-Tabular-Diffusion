# TabSDS Validation Protocol

Status: `native-parity-validated` by a retained authoritative Linux run

Protocol: `tabsds-official-source-parity-v1`

Target: method-author Python `simple` shuffle path

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This protocol compares the adapter directly with the checksum-locked method-author Python source. A passing retained run may establish `native-parity-validated` for the selected simple-shuffle path and the declared exact-row boundary. It does not authorize redistribution or release: the upstream repository declares no license.

## Audited Source

The authority is [echaibub/TabSDS](https://github.com/echaibub/TabSDS) at its single locked commit `866501495069c7e1300bdea91c411f1947d19f2f`, tree `0f237c7b4fa02e06c525f29d1d83ff5c460816ee`. The archive is 119,407 bytes with SHA-256 `292011aab0153ca8f7cc90c21dd4acbcbd2a22da557ab080883555b7ab0cf82a`. Only the two required notebook helper files are acquired on demand; both are checked by path, normalized byte count, and SHA-256 before use.

## Adapter Boundaries

The official Python implementation is supplied as notebook helpers. One file assumes `numpy` and `pandas` already exist in its notebook namespace. The adapter reconstructs that namespace before executing the byte-exact file; it does not edit or translate an upstream statement.

Input must be one missing-free, finite, mixed-type table with one classification or regression target and complete declared roles. Training stores only typed JSON recipe state, schema, source identity, row count, and a training-file digest. Sampling calls official `tab_sjppds(..., shuffle_type="simple")`. Because one call always returns the input row count, requests larger than the training table repeat the unchanged call and truncate only the last block.

## Mandatory Parity Cases

Binary classification, multiclass classification, and regression fixtures are each executed with seeds 0, 19, and 73: nine cases total. Every case requests 53 rows from a 37-row training table, thereby testing both the official first block and the explicit repeat/truncate boundary. Direct-source and adapter CSV files must be byte-identical. The source tree and caller NumPy state must remain unchanged.

## Known Boundaries

- Only the Python simple-shuffle path is validated; no claim is made for R code or other shuffle modes.
- Exact parity is not a quality or privacy result.
- The upstream repository has no declared license. Source is fetched into an ignored cache and is not redistributed here.
- Official Results, release support, central evaluation, and dataset admission remain blocked or pending independently.

## Evidence

GitHub Actions run [`30974574593`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30974574593) passed on Linux with Python 3.11.15. All nine binary, multiclass, regression, and seed cases produced identical DataFrames and CSV bytes across direct-source and adapter paths while exercising the 53-from-37 repeat/truncate boundary. The inspected JSON is retained byte-for-byte at `docs/evidence/tabsds/native-parity-run-30974574593.json` with SHA-256 `11cfa96a3221944ebb6d423fdddf8660f278e7f6b108dff500fe39a1f9b07b66` and is cross-linked from the source lock. The absent upstream license still blocks redistribution and release.
