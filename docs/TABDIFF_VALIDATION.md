# TabDiff validation

Status: native parity validated; configurable-seed and Windows/Adult real-function validation passed

Protocols: `tabdiff-native-parity-v2`, `tabdiff-adult-real-function-windows-v1`

## Claim boundary

The retained evidence establishes that the checksum-locked method-author implementation can be invoked through the standardized adapter, that seed 0 preserves the official deterministic path, and that a real TabDiff model can train and generate three complete Adult tables on the primary Windows/Python 3.11/GPU environment.

This does not make TabDiff `benchmark-eligible` or `release-supported`, and it does not admit Adult, the diagnostic scores, or any concrete run to Official Results. Dataset, model, result, full central-evaluation, governance, and release admission remain separate gates.

## Source authority

- Repository: `MinkaiXu/TabDiff`.
- Commit: `5ecdb3356261aea72716cc9a779f31d7ad083bf4`.
- Tree: `052a505cb1fbee5cbc705eeb0717d90d706ffb91`.
- Manifest: `standardized_tabular_diffusion/resources/upstream/tabdiff-source-manifest.json`.
- License: MIT.

All 27 frozen files match the method-author source after canonical line-ending normalization. The former local semantic modification to `eval/mle/mle.py` remains removed. Upstream evaluation code is retained for source fidelity and runtime diagnostics; formal repository evaluation uses the separately versioned central engine.

## Audited adapter boundaries

The tracked `TabDiff-main` source remains unchanged. The adapter applies only fail-closed, checksum-verified runtime boundaries:

1. `tabdiff-configurable-seed-overlay-v1` replaces the six official seed-0 constants in memory with the requested non-negative seed. Model, loss, optimizer, schedule, preprocessing, and sampling equations are unchanged.
2. `tabdiff-config-path-overlay-v1` exposes the common `RunSpec.upstream_config_path`. The complete supplied TOML is passed unchanged; omitting it retains the official default.
3. `tabdiff-pytorch-reduce-lr-verbose-bridge-v1` accepts and discards the deprecated logging-only `verbose` argument removed in PyTorch 2.8.
4. `tabdiff-diagnostic-plot-bypass-v1` disables only optional `density_plots.png` rendering that fails on some Windows non-ASCII paths. Synthetic tables and all serialized metrics remain enabled.

The adapter also maps CPU/CUDA devices, enables deterministic execution, disables online logging by default, isolates each seed's copied `samples.csv`, records run identities, and requires explicit trust before loading external PyTorch checkpoints.

The current adapter treats the verified upstream checkout as read-only. It materializes a byte-verified model data tree and the official metrics view (`real.csv`, `test.csv`, and optional `val.csv`) beneath `output_dir/tabdiff-runtime`. Training and sampling share that run-owned workspace and its internal checkpoint. Reusing a data view succeeds only when every file still matches its registered canonical source; links, unexpected entries, and changed bytes fail closed.

## Seed and native-parity validation

The current V2 Linux/Python 3.11/PyTorch 2.3 CPU protocol compares isolated native and adapter executions using the same mixed-type fixture and bounded runtime TOML. The adapter path uses the same registered-data and shared run-owned train/sample workspace as the public pipeline. It snapshots both action manifests and requires exact cached configuration, checkpoint tensors, training samples, generated CSV bytes, and upstream metrics.

The extended protocol preserves seed-0 exactness and adds configurable-seed checks: the same nonzero seed must reproduce exact bytes, a different seed must produce different bytes, and every run record must retain the effective seed. Native-parity diagnostics may explicitly bypass standardized integer-output enforcement because they compare the official default configuration, whose `dequant_dist="none"` intentionally does not round integer columns.

V2 passed in [GitHub Actions run 32058517599](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32058517599) at repository commit `bf3869776fbc975052426732dbd6a167566124f4`. The retained artifact ID is `9297294246`, with digest `sha256:45a6d26f32a18d1b79281ac57873e517c62ebb15d71a8d6d0a6754702dbf4b32`. Its permanent evidence copy is `docs/evidence/tabdiff/native-parity-run-32058517599.json`, SHA-256 `d4630b50924e345a112fc4ff717e27dd15f930e1a6069db7b43dadf5f0479a19`.

## Integer restoration finding

The first real Adult run exposed a genuine output-contract problem. With the official default `dequant_dist="none"`, TabDiff successfully generated 32,561 rows, but six Adult integer columns contained fractional values. The validation failed rather than repairing or accepting that table.

The official TabDiff preprocessing code already supports `dequant_dist="round"`: its training transform is a no-op, while its official inverse transform applies `numpy.rint` to declared integer columns. The retained real-function preset therefore selects this native mode. A repeated training run produced the exact same checkpoint SHA-256 as the failed-default run (`4319e6938a1ae4619cdd17a995d71f5de0d50c450ff096754e6ef6ab2e0a26f0`), confirming that the change affected only inverse output restoration. The adapter now rejects fractional generated integer columns and instructs standardized runs to use the upstream `round` mode; no silent synthetic-data repair is performed.

## Windows/Adult real-function protocol

The retained bounded preset keeps the official 10,622,977-parameter architecture, optimizer, training batch size, learned schedules, and 50-step diffusion horizon. It changes only:

- `data.dequant_dist`: `none` to the official `round` mode for schema-valid inverse restoration;
- training epochs: 8,000 to 20;
- periodic validation: every 2,000 epochs to the final epoch; and
- sampling batch size: 10,000 to 4,096 for the 16 GB GPU.

This is a real end-to-end functionality run over the complete reviewed Adult split, not a final-quality training experiment. It ran on Windows, Python 3.11.15, PyTorch 2.8.0+cu128, and an NVIDIA GeForce RTX 5080:

- one training run with seed 0;
- one checkpoint reused for generation seeds 3, 4, and 5;
- 32,561 rows and 15 columns per table;
- no missing or non-finite values;
- all integer columns integral and within reviewed ranges;
- all categorical values within reviewed domains; and
- three distinct sample hashes.

The model run evidence is `docs/evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json`.

## Central evaluation integration

Seed 3 was then passed through the repository's public `evaluate-table` route without synthetic repair:

- P3 finalized successfully, the structural gate passed, and the fully valid row rate was 1.0.
- P2 finalized successfully with 27 computed Atomic Results; diagnostic Column Shapes and Column Pair Trends values were produced under the frozen central implementation.

The exact fingerprints and bundle checksums are retained in `docs/evidence/tabdiff/adult-central-route-windows-rtx5080-20260814.json`. These numerical values are diagnostic and are not Official Results.

## Reproduction

Run source/seed parity in its locked validation environment:

```bash
python -m standardized_tabular_diffusion.validation.tabdiff \
  --repo-root . \
  --output-dir /tmp/tabdiff-validation \
  --evidence-path /tmp/tabdiff-evidence.json
```

Run the bounded Windows real-function protocol in the TabDiff GPU environment:

```powershell
python -m standardized_tabular_diffusion.validation.tabdiff_adult_real_function `
  --repo-root . `
  --output-root artifacts/tabdiff-real-function/adult-windows-rtx5080-real-function-v2 `
  --evidence-path docs/evidence/tabdiff/adult-real-function-windows-rtx5080-20260814.json
```

The output root and upstream experiment directory must not already exist. Data, samples, and checkpoints remain ignored local artifacts.
