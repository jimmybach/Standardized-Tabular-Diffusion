# T04: Windows, GPU, and Dependency Boundaries

Chinese translation: [TASK_04_WINDOWS_GPU_AND_DEPENDENCIES.zh-CN.md](TASK_04_WINDOWS_GPU_AND_DEPENDENCIES.zh-CN.md)

- Status: complete for all 18 non-blocked minimal-real baselines; GReaT and TabSyn completed the RTX 5080 path, REaLTabFormer completed its declared CPU path, and TabEBM retains its independent external-access block
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Risk class: platform-only failure or unintended device/runtime drift

## Objective

Prove that each supported adapter can start and finish on the primary Windows/Python 3.11 family with its requested device and declared dependencies. Linux parity evidence remains valuable but is not substituted for this platform check.

## V0/V1 checks for every adapter

- Resolve and record Python, package, CUDA runtime, PyTorch/TensorFlow, GPU, driver, and authoritative source/package identities before work starts.
- Test path handling with spaces and non-ASCII characters without changing the repository location or copying source to a simplified path.
- Verify subprocess argument quoting, UTF-8 input/output, path separators, temporary-directory use, and process-tree termination on Windows.
- Verify that headless plotting and optional diagnostics cannot abort successful scientific computation.
- Request devices explicitly. If CUDA is required or requested, an unintended CPU fallback must fail closed rather than silently changing cost or behavior.
- Distinguish packages required to import the adapter, train/sample, and evaluate. Missing extras must produce an actionable error before expensive work.
- Verify compatibility bridges are narrow, checksum-bound where source-sensitive, recorded in run metadata, and do not change model mathematics.
- Check that environment variables used for determinism, encoding, CUDA, or module resolution are scoped to the child run and do not leak globally.

## V2 minimal-real probe

Run the authoritative algorithm on the declared Windows/Python 3.11 environment and record observed hardware/software identity. The probe must:

- confirm that model tensors and the principal computation use the requested device;
- finish from the existing repository path containing non-ASCII characters;
- produce readable UTF-8 logs and a valid decoded output;
- exercise any approved runtime bridge and retain its identifier; and
- terminate the full process tree on an intentionally bounded failure test.

CPU-only algorithms may declare CPU intentionally. They must not be marked as CUDA failures merely because no meaningful GPU path exists.

## Failure conditions

- Silent CPU fallback when CUDA was requested.
- A Windows or non-ASCII path failure hidden by tests in a temporary ASCII-only checkout.
- An optional import or plot prevents core generation without an explicit protocol requirement.
- A compatibility patch changes scientific behavior or is applied without an exact precondition.
- The recorded environment describes requested rather than observed hardware/software.

## Exit gate

Every non-blocked adapter has one observed native-Windows V2 environment record, device behavior matches its declared policy, and all platform bridges have narrow regression coverage and explicit claim boundaries.
