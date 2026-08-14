# Architecture

## Public flow

The repository separates upstream algorithms, repository adapters, central evaluation, and publication:

~~~text
Dataset + reviewed profile
        |
        v
official package/source <- adapter train/sample -> original decoded table
                                                   |
                                                   v
                                      central P2-P5 evaluator
                                                   |
                                                   v
                                      finalized Result Bundle
                                                   |
                                                   v
                               P7 diagnostic/Official publication gate
~~~

Adapters may translate configuration, data layout, invocation, and artifacts. They do not own metric formulas. Public `run-action evaluate`, top-level `run`, and P6 `benchmark run` route decoded samples to the central evaluator.

## Main components

- `registry.py` records source authority, distribution, reproduction target, modification status, adapter validation, benchmark track, support, licensing, and evidence independently.
- `models/` contains train/sample adapters around official packages or checksum-locked source.
- `evaluation/` owns versioned requests, profiles, structural gates, metrics, Atomic Results, bundle finalization, validation, legacy import, and snapshots.
- `orchestration/` runs prepare/train/sample/validate/evaluate/aggregate/report as isolated, resource-bounded stages.
- `resources/` and `schemas/` contain immutable machine-readable contracts and small redistributable fixtures.
- `validation/` produces engineering evidence; it cannot grant scientific admission by itself.

## Evidence types

`artifacts.json` is adapter metadata. `evaluation-result/manifest.json` identifies a new Result Bundle containing Atomic Results. A P7 snapshot is a separate publication artifact. These objects are not interchangeable.

The frozen legacy `standardized_summary.json` contains neither Atomic Results nor the evidence required for Official admission. P8 accepts it only through `import-legacy-summary`, which preserves the bytes under `source/standardized_summary.json` and emits a `legacy_import_record.json` stating that conversion did not occur.

## Trust boundaries

Input tables, profiles, configs, downloaded source, checkpoints, and result bundles are untrusted until validated. Code-executing checkpoints are restricted to the run output by default. Writers reject symlinks, traversal, duplicate keys, non-finite JSON, overwrite of finalized evidence, and embedded credentials.

## Status boundaries

Adapter validation (`registered` through `native-parity-validated`), benchmark track, release support, metric lifecycle, dataset admission, run admission, and publication class are separate decisions. A successful quickstart or CI job cannot promote another status implicitly.
