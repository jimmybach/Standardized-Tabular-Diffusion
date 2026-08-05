# GReaT Validation Protocol

Status: authoritative Linux run pending

Protocol: `be-great-official-package-parity-v1`

Target: method-author `be-great==0.0.14` package

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This protocol determines whether the standardized `great` adapter preserves a selected execution of the unchanged official package. A passing retained run may promote that path to `native-parity-validated`; it does not establish paper-scale quality, privacy, benchmark eligibility, Official Results, or release support.

## Audited Distribution

The authority is [tabularis-ai/be_great](https://github.com/tabularis-ai/be_great) at commit `b300f6123cf1d9590b76ea45cc23298df944a319`, tree `82eccd92deb18d6138a4fd96a8e0eb097b38babb`. The selected PyPI wheel is `be_great-0.0.14-py3-none-any.whl`, 54,019 bytes, SHA-256 `4f6384ec4a736177ae2d1e6146951cfdfc764b1cc041ae5c2b155a99dd18cb74`, under MIT.

All 14 package files are byte-exact with the tagged source distribution; their deterministic aggregate SHA-256 is `3e98e1e0e68ce614b62d425c8fcf559ab0387cbf85dd0e27166c22b360fcfaca`. Eight imported runtime files are checked again from the installed distribution before every adapter operation.

## Adapter Boundaries

The adapter accepts one missing-free table with exactly one classification or regression target. It enforces complete, disjoint declared roles, finite numerical values, and strings that are unambiguous under the official row parser. Missing values must first pass through the central imputer fitted only on the training split.

Construction, `fit`, and `sample` remain official calls. The adapter supplies output isolation, bounded controls, deterministic row limiting, and scoped Python/NumPy/PyTorch randomness. Caller RNG and thread state is restored. The official executable `model.pt` format is not used: model tensors are saved as safetensors, non-tensor state as typed JSON, and every artifact file is covered by an integrity manifest. Unsafe checkpoint extensions, symlinks, added files, and altered bytes fail closed.

## Mandatory Parity Cases

The workflow builds an offline one-layer GPT-2 fixture so validation requires no remote model checkpoint. It trains the real official package and adapter independently on the same 24-row categorical table for seeds 0, 19, and 73. Each case compares every model tensor exactly, reloads the safe adapter artifact, executes official guided sampling for seven rows, and requires DataFrame-identical output plus byte-identical CSV.

The gate also requires exact wheel identity, unchanged installed package files, exact requested row count and column order, no missing values, and restoration of caller RNG/thread state. Any mismatch retains a diagnostic JSON artifact and fails the workflow.

## Known Boundaries

- The tiny offline checkpoint establishes wrapper parity, not useful synthesis quality.
- Production profiles still need a selected pretrained model, resource limits, and dataset-specific sequence-length controls.
- Generated rows and trained artifacts have no differential-privacy guarantee and require normal access controls.
- Central evaluation, dataset admission, benchmark eligibility, and release ownership remain separate gates.

## Evidence

The mandatory GitHub Actions run and permanent byte-for-byte evidence record will be added only after the authoritative workflow passes.
