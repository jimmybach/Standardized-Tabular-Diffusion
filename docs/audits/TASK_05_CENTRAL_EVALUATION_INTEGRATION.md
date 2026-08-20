# T05: Central Evaluation Integration

Chinese translation: [TASK_05_CENTRAL_EVALUATION_INTEGRATION.zh-CN.md](TASK_05_CENTRAL_EVALUATION_INTEGRATION.zh-CN.md)

- Status: in progress; Goggle, TabuLa, TabularARGN, ARF, BN, SMOTE, NRGBoost, TabSDS, CTGAN, TVAE, and NFlow V2 finalization retained
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Risk class: model-specific evaluation drift or invalid result finalization

## Objective

Prove that every public adapter path hands its untouched decoded sample to the same central evaluation subsystem. Adapters may own training, sampling, and authoritative decoding; they may not compute private metric formulas, bypass the structural gate, or publish a legacy summary as a current result.

## V0/V1 checks for every adapter

- Trace `run-action evaluate`, top-level `run`, and `benchmark run` from the selected sample artifact to the public central evaluation service.
- Verify that the exact selected generation seed, sample path, sample hash, model identity, dataset-profile identity, and resolved protocol are preserved.
- Verify that adapter-local `evaluate` behavior is only a routing boundary and does not select a model-specific formula or silently fall back to `standardized_summary.json`.
- Verify protocol selection is explicit: P2, P3, P4, and P5 have different data-role and dependency requirements and cannot be guessed from available files.
- Verify malformed output fails at the protocol's structural gate before metric computation.
- Verify per-scope Atomic Results retain complete denominators, states, source values, derived values, and failure records.
- Verify bundle finalization occurs last, is checksum-complete, and is rejected if any declared file or identity is inconsistent.
- Verify evaluation can also accept an external synthetic table without importing a model adapter.
- Verify model-specific training dependencies and central-evaluation dependencies are independently frozen; finalization must validate the central lock rather than rely on packages incidentally present in a model environment.

## V2 minimal-real probe

Take the exact sample produced by the adapter's V2 run and invoke the least expensive compatible central structural/evaluation route. At minimum:

1. Bind the sample file hash before evaluation.
2. Confirm that the request identifies the actual generation seed and model run.
3. Finalize and validate a bundle for a structurally valid sample.
4. Mutate a copy into one known structural defect and confirm fail-closed behavior without changing the original sample.
5. Confirm that no legacy evaluator or adapter-private metric path was imported or written.

V3 candidates use the reviewed P2/P3 route and any separately required P4/P5 protocols. Passing V2 does not imply that every metric is applicable or admitted.

## Failure conditions

- Evaluation reads a different seed or a stale sample.
- An adapter computes or selects its own score outside the central registry.
- Structural failure is hidden by repair or by dropping invalid scopes.
- A partial or checksum-inconsistent bundle is marked finalized.
- The same decoded table receives different formulas solely because it came from a different adapter.

## Exit gate

All adapter entry points converge on the same protocol-resolved service, valid V2 outputs finalize a validated central bundle, malformed copies fail closed, and regression tests prohibit legacy/private evaluator fallback.
