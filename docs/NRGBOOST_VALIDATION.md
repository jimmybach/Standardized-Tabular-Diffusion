# NRGBoost Validation Protocol

Status: passed and permanently retained

Protocol: `nrgboost-native-parity-v1`

Target: official method-author `nrgboost==0.0.3` package

Supported validation environment: Linux, Python 3.11

## Claim Boundary

This protocol tests whether the standardized `nrgboost` adapter constructs the same typed input table and official `Dataset`, forwards the same dataset, training, sampling, and random-seed controls, saves and reloads the same model, and produces the same output as a direct call to the checksum-pinned official package.

A passing mandatory run may promote the adapter to `native-parity-validated`. It does not make NRGBoost `benchmark-eligible`, admit it to Official Results, establish statistical benchmark quality, or make it `release-supported`. Dataset admission, frozen central evaluation, runtime limits, and release ownership remain separate gates.

## Audited Authority and Distribution

The audited implementation is the [method-author repository](https://github.com/Ajoo/nrgboost) at tag `v0.0.3`, commit `feef73a3edb20b911c2f7214b13f810909ef20ad`, tree `e3e84bacc7236a36af93c3d214de14bd308d2767`. The supported artifact is the official [PyPI 0.0.3 release](https://pypi.org/project/nrgboost/0.0.3/) for CPython 3.11 and manylinux 2.28 x86-64:

- filename: `nrgboost-0.0.3-cp311-cp311-manylinux_2_28_x86_64.whl`;
- SHA-256: `dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb`;
- license: MIT;
- distribution form: optional package dependency; no NRGBoost source is vendored or patched here.

The protocol checks the wheel name and digest, safe archive paths, package metadata, Python and ABI tag, declared dependencies, source and wheel license hashes, compiled extension, bundled OpenMP runtime, every hash-bearing wheel `RECORD` entry, installed distribution root, and public class exports. PyPI Trusted Publishing provenance binds the release artifact to the locked tag commit.

NRGBoost 0.0.3 publishes Linux and macOS wheels and does not support Windows. Source builds require a C compiler and OpenMP. The repository therefore treats Linux/Python 3.11 as authoritative and does not reinterpret a Windows source build as equivalent evidence.

## Adapter Semantics

The adapter remains a thin package wrapper:

- reads the canonical training CSV in declared column order;
- casts declared categorical features, and classification targets, to pandas `category`, as required by the official package;
- rejects missing values and requires the explicit benchmark preprocessing module to run first;
- creates official `nrgboost.Dataset` and `nrgboost.NRGBooster` objects;
- passes a fresh copy of the training parameter mapping because official `fit` consumes selected entries from that mapping;
- passes `RunSpec.seed` to both `NRGBooster.fit` and `NRGBooster.sample`;
- saves and loads the official joblib-based checkpoint format; and
- writes only the requested final-chain samples in canonical column order.

The checkpoint format can execute Python during loading. By default, the adapter therefore loads only a regular non-symlinked file inside the run output directory. Loading an external checkpoint requires an explicit unsafe override after provenance and integrity review.

## Supported Controls

The adapter exposes the official scalar dataset and training controls without changing the algorithm. Dataset controls are `num_bins`, fixed-point inference, optional explicit discretization types, and the two ordered-categorical inference flags. Training controls include tree count, shrinkage and line search, tree size and splitting, data/model leaf constraints, initial mixture, feature fraction, model-sampling chains, refresh rate, burn-in, temperature, minimum gain, JIT selection, and thread count. Values are range-checked before the official call.

Sampling exposes output rows, Gibbs steps, optional boosting round, temperature, thread count, and seed. `output_full_chain` is deliberately fixed to `false`: returning `num_samples × num_steps` chain states would violate the standardized `num_samples` row contract.

## Frozen Parity Cases

The mandatory protocol uses two deterministic, missing-free, mixed-type tables with 36 rows each:

1. classification: two numerical features, one categorical feature, and one categorical target;
2. regression: two numerical features, one categorical feature, and one numerical target.

Each table is executed with seeds 0, 19, and 73, producing six independent native/adapter cases. The bounded CI configuration fits three trees with one thread, 256 model samples, four chains, and small tree limits; sampling requests 16 rows, 12 Gibbs steps, and one thread. These settings test the real compiled implementation while keeping CI cost bounded. They are validation fixtures, not recommended benchmark hyperparameters.

For each case the protocol runs two independent paths:

- native: direct official `Dataset` → `NRGBooster.fit` → `save` → `load` → `sample`;
- adapter: standardized `train` → official checkpoint → standardized `sample`.

## Mandatory Pass Criteria

All six cases must pass. The gate requires:

1. the frozen Linux/Python 3.11 environment and official wheel identity are exact;
2. adapter artifact manifests and structured metadata are exact;
3. native and adapter checkpoint bytes are identical;
4. both checkpoints declare serialization version `0.0`, contain exactly three trees, and retain canonical transform columns;
5. native and adapter sample CSV files are byte-identical and DataFrame-identical;
6. output row count and column order are exact, numerical values are finite, categorical values stay inside learned domains, and no missing values appear; and
7. neither training nor sampling mutates legacy global NumPy random state.

Any mismatch, missing artifact, wrong platform, dependency drift, unsafe wheel path, unverified installed file, or failed comparison fails closed and retains a diagnostic JSON artifact.

## Known Boundaries

- Exact adapter parity says that the wrapper preserves the selected official execution; it does not prove that three-tree smoke fixtures approximate full-paper quality.
- The official sampling method is approximate and its cost scales with Gibbs steps. Benchmark hyperparameters and runtime budgets require a later dataset-level study.
- Missing data are outside the model adapter contract. Imputation must be fitted on the training split only, recorded by the preprocessing layer, and applied before NRGBoost.
- Advanced explicit `discretization_types` remain an official expert API. Dataset profiles must document any non-default mapping before Official Results.
- This protocol validates generation for classification and regression tables. Predictive use of `NRGBooster.predict` is not part of the standardized generation adapter.

## Evidence

[GitHub Actions run `30922326384`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30922326384) passed all six task/seed cases on Linux with Python 3.11.15. It verified all 22 hash-bearing installed files against the locked wheel, ran classification and regression fixtures with seeds 0, 19, and 73, and produced byte-exact native/adapter checkpoints and sample CSV files in every case.

The permanent evidence record is `docs/evidence/nrgboost/native-parity-run-30922326384.json`, SHA-256 `5958c67261e8c25e60d58891efd5d27f8e8bb6439852862064e831f630cbe56c`. The run is bound to repository commit `4cd32c8beedd116c6385463d41cf9cba8b1d5438`; the downloaded GitHub artifact is additionally recorded by artifact ID and digest in the source lock. NRGBoost is therefore `native-parity-validated`, while benchmark eligibility and release support remain pending.
