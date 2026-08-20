# T01: Seed and Configuration Propagation

Chinese translation: [TASK_01_SEED_AND_CONFIG.zh-CN.md](TASK_01_SEED_AND_CONFIG.zh-CN.md)

- Status: complete for all 21 registered identities at V0/V1; all 20 non-blocked identities retain passing Windows V2 or stronger evidence, and TabEBM retains its explicit external-access block
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Risk class: silent scientific error

## Objective

Prove that repository configuration reaches the authoritative runtime with the intended values. Detect hard-coded upstream defaults, dropped fields, incorrect precedence, train/sample seed confusion, and configuration mutation after validation.

## V0/V1 checks for every adapter

- Trace each public configuration field from CLI/config model through `RunSpec`, adapter invocation, subprocess arguments/environment, and the authoritative API or source entry point.
- Record the exact precedence of user values, dataset-profile values, adapter defaults, and upstream defaults. Unknown or contradictory fields must fail before training.
- Verify that training seed, sampling seed, evaluator seed, and `PYTHONHASHSEED` are not silently conflated.
- Verify non-negative seed validation, deterministic-mode handling, and explicit reporting of unsupported reproducibility requests.
- Verify that a requested config path is the path actually opened by the authoritative runtime; the evidence must bind its hash.
- Verify that adapters do not mutate checked-in upstream configuration or source files in place.
- Verify that the resolved configuration stored with the run contains no credentials and is sufficient to reconstruct the invocation.

Controlled test doubles may observe arguments and environment variables, but the result is V1 only and must be labelled `contract-simulated`.

## V2 minimal-real probe

Use a small valid dataset and the fastest model-specific configuration that still invokes the authoritative training and sampling algorithms.

1. Use a declared non-default, non-zero seed for the first real execution.
2. Confirm that the recorded resolved configuration and authoritative runtime observation agree.
3. When sampling is separable, reuse the checkpoint with a second sampling seed and retain both outputs. Otherwise run the smallest second seeded execution supported by the model.
4. Require distinct output hashes for stochastic generation unless the model protocol documents why equality is expected.
5. Do not require bitwise reproducibility unless the upstream method declares it for the tested platform. If declared, repeat the same seed and compare the protocol-defined artifacts.

## Failure conditions

- A user value is accepted but ignored.
- Seed roles are swapped, reset, or forced to a hidden constant.
- A second seed overwrites or silently reuses the first sample.
- The actual runtime config cannot be reconstructed from retained metadata.
- Simulation succeeds but the real authoritative entry point receives different values.

## Exit gate

Every non-blocked adapter has an argument/configuration trace plus a V2 observation. Each confirmed defect has an entry in [PIPELINE_FINDINGS.md](PIPELINE_FINDINGS.md), a root-cause fix, and a regression test at the narrowest shared boundary.

## Completion record

Phase 1 completed the configuration-projection audit for all 21 adapters. Phase 2 fixed the confirmed shared and model-specific propagation defects. Phase 3 then retained a passing native-Windows observation for every non-blocked identity: 18 minimal-real probes plus the stronger representative-real TabDDPM and TabDiff records. TabEBM is recorded separately as externally blocked rather than passed or failed. This closes the audit task without promoting any model to Official Results or release support.
