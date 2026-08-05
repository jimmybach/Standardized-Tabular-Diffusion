# TabuLa Validation Protocol

Status: `native-parity-validated` by a retained authoritative Linux run

Protocol: `tabula-method-author-source-parity-v1`

Target: method-author original source

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This protocol tests whether the standardized `tabula` adapter preserves the locked method-author execution. A passing retained run may promote the tested path to `native-parity-validated`. It does not establish paper-scale generation quality, benchmark eligibility, privacy, or release support. The upstream repository has no declared license, so redistribution and formal release remain blocked regardless of technical parity.

## Audited Source

The authority is [zhao-zilong/Tabula](https://github.com/zhao-zilong/Tabula) at commit `a7d34a94adee5a269f6807395d0040d936bb0e60`, tree `dd7d61fbbb071c240f13b6ba3bd05fdf69d25a13`, package subtree `6ee7614b3fb60930a41440e242fdade3145ab5b2`. The codeload archive is 59,173 bytes with SHA-256 `dfb69d55cf4e669f979325bf10b118a084dde49d680680dcebf7dcdaed024a26`. Six runtime files are acquired on demand and validated before every train or sample action. No source file is vendored or patched.

## Adapter Boundaries

The adapter accepts one missing-free table with one target, complete disjoint roles, finite numerical values, and parser-safe column names. Categorical features and classification targets are converted to strings before the unchanged official label encoders. Missing values must first use the central train-fitted imputer.

Official construction, label encoding, row-text training, and sampling remain unchanged calls. The official sampler checks categorical domains only after its internal row-count loop and can consequently return fewer usable rows than requested. The adapter therefore repeats unchanged official calls for the remaining count, preserves their order and RNG stream, and truncates only the final boundary. Repeated zero-row batches fail explicitly. The adapter also scopes Python, NumPy, and PyTorch randomness, isolates official imports, places trainer files below `output_dir`, and applies an external Linux alarm because the official sample loop has no retry bound. Non-Linux unbounded sampling requires explicit opt-in.

The official `torch.save` checkpoint is replaced at the persistence boundary with safetensors, typed JSON label-encoder/state data, and a complete file-integrity manifest. The adapter reconstructs the official class, restores the training mode retained by the method-author `load_from_dir()` path, and continues through its unchanged `sample()` method; executable checkpoint extensions, symlinks, unlisted files, or altered bytes are rejected.

## Mandatory Parity Cases

The workflow builds an offline one-layer GPT-2 fixture and trains direct official source and adapter paths independently on the same 24-row categorical table for seeds 0, 19, and 73. The tiny model disables early EOS termination so the source's hard-coded full-GPT-2 padding ID is never fed into its deliberately small embedding table; both paths receive the identical fixture. Each case compares all trained tensors exactly, reloads the safe artifact, independently applies the same exact-row boundary around unchanged official calls with the same encoded target distribution, and requires five-row DataFrame-identical output plus byte-identical CSV.

The gate also verifies archive and six-file source identity, unchanged source after execution, exact row count and column order, no missing values, safe artifacts, and restoration of caller RNG/thread state. Sampling exceeding the declared time limit fails with retained diagnostics.

## Known Boundaries

- The tiny offline checkpoint establishes wrapper parity, not useful synthesis quality.
- The official parser cannot safely represent arbitrary spaces or commas in column names and may reject malformed generated rows.
- Dataset-specific model selection, training budgets, evaluation, and resource limits remain pending.
- No upstream license is declared; source is fetched to an ignored cache and is not redistributed by this repository.

## Evidence

GitHub Actions run [`30974574505`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30974574505) passed on Linux with Python 3.11.15. All three seeds matched every trained tensor, exact-row sample DataFrame, and CSV byte; source identity and immutability, safe persistence, bounded sampling, and caller-state restoration also passed. The inspected JSON is retained byte-for-byte at `docs/evidence/tabula/native-parity-run-30974574505.json` with SHA-256 `35b9c8bdab2828763a72fe3fa55aa6c9fa6308dc36740217d6479c296da3ca1c` and is cross-linked from the source lock. The absent upstream license still blocks redistribution and release.
