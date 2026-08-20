# T03: Checkpoint and Artifact Isolation

Chinese translation: [TASK_03_CHECKPOINT_AND_ARTIFACT_ISOLATION.zh-CN.md](TASK_03_CHECKPOINT_AND_ARTIFACT_ISOLATION.zh-CN.md)

- Status: complete at the shared pipeline boundary; all 20 non-blocked identities retain isolated, immutable Windows V2 or stronger artifacts, and TabEBM retains its explicit external-access block
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Risk class: stale-result reuse, cross-run contamination, or destructive overwrite

## Objective

Prove that every run owns its checkpoint, logs, decoded samples, metadata, and evaluation bundle. No adapter may accidentally read an upstream example checkpoint, a previous seed's output, or a partially written artifact and report a fresh success.

## V0/V1 checks for every adapter

- Trace every read/write path used by upstream code, including hard-coded working-directory paths, temporary files, caches, checkpoints, plots, and default result folders.
- Bind all mutable outputs to the requested `output_dir` or an isolated run workspace. Checked-in upstream trees remain read-only.
- Require collision-resistant run and seed subdirectories. Distinct sampling seeds must not share the same final sample path.
- Reject a non-empty destination when overwrite semantics are not explicit. Never infer success from the presence of an old file.
- Record checkpoint and sample hashes after atomic completion. Validate size, format, and required companion files before consumption.
- Verify that resume/cache keys include every scientific and runtime identity that can affect the artifact.
- Verify that failed or interrupted writes cannot replace the last known-complete artifact.
- Keep large artifacts ignored by Git; retain compact evidence manifests only when approved.

## V2 minimal-real probe

1. Run a real fit into a fresh isolated directory and record the complete file manifest.
2. Sample with the first declared seed, then with a second seed when supported; verify separate paths and hashes.
3. Place an intentionally stale or incomplete sentinel in a controlled destination and prove that the adapter rejects it or creates a new isolated attempt.
4. Re-run the exact request only through the declared resume/cache mechanism and prove that identity validation occurs before reuse.
5. Confirm that no tracked upstream file changed.

## Failure conditions

- A new run succeeds by discovering an unrelated pre-existing checkpoint or sample.
- Two seeds overwrite one another or central evaluation reads the wrong seed.
- Partial output is considered complete.
- Cache reuse ignores a configuration, dataset, source, seed, platform, or dependency identity required by the protocol.
- The authoritative source tree is modified as a runtime workspace.

## Exit gate

Every adapter has a complete path map and isolation test. All real probes create reproducible manifests, stale-artifact sentinels fail safely, and retry/resume behavior is covered by regression tests.

## Completion record

Run identity, dataset identity, seed-specific output ownership, atomic finalization, cache validation, and retry ancestry are enforced at shared boundaries and covered by regressions. Every retained Windows real-function pass records immutable training artifacts and separate hashes for the two declared generation seeds; the representative TabDDPM and TabDiff records satisfy the stronger V3 boundary. No tracked upstream source is used as a mutable runtime workspace. This does not claim that every upstream implementation offers native resume semantics.
