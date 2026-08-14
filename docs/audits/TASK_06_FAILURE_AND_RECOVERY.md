# T06: Failure and Recovery

Chinese translation: [TASK_06_FAILURE_AND_RECOVERY.zh-CN.md](TASK_06_FAILURE_AND_RECOVERY.zh-CN.md)

- Status: planned across all 21 adapters
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Risk class: false success, irrecoverable partial state, or leaked sensitive context

## Objective

Prove that expected failures are explicit and classified, preserve completed work safely, and support a deterministic retry without pretending that failed scientific work succeeded.

## V0/V1 checks for every adapter

- Enumerate failures before start, during fit, during checkpoint save/load, during sampling/decoding, and during central evaluation.
- Exercise invalid configuration, negative seed, missing dependency/source, missing dataset, schema mismatch, impossible row count, unsupported task type, unavailable requested device, timeout, and child-process failure.
- Require non-zero process exit, a structured stage state, an actionable public error, and a redacted diagnostic log.
- Preserve attempt ancestry. A retry creates a new attempt linked to the failure; it does not rewrite the earlier record.
- Verify that cleanup never deletes a successful artifact from another run and that partial files cannot be selected as complete.
- Verify that cache/resume accepts only checksum-complete outputs with an exact identity match. Damaged cache entries execute afresh.
- Redact tokens, credentials, sensitive environment values, and unintended raw-row content from commands, logs, errors, and retained evidence.
- Distinguish an environment remediation from a scientific configuration or seed change.

## V2 bounded-failure probes

Use the adapter's smallest real setup to test only safe boundaries:

1. Request an unavailable or invalid device before expensive computation and confirm fail-closed behavior.
2. Interrupt or time-bound a disposable run and verify that the process tree terminates and no artifact is finalized.
3. Corrupt a copied checkpoint or cache manifest and confirm rejection without touching the original.
4. Correct only the declared environment fault, retry as a linked attempt, and confirm that completed independent stages are reused only when identity checks pass.

Do not inject failures that risk the user's primary checkpoints or uncommitted data. All destructive probes use isolated disposable directories.

## Failure conditions

- A child failure returns overall success or leaves a finalized bundle.
- Retrying hides, replaces, or disconnects the failed attempt.
- A damaged or identity-mismatched artifact is reused.
- Logs or evidence contain credentials or unintended raw sensitive values.
- Cleanup crosses the isolated run boundary.

## Exit gate

Each shared failure class has deterministic tests, all adapter-specific exceptions map to structured states, safe retry is demonstrated, and no S0/S1 failure-handling finding remains unresolved except an explicit release blocker.
