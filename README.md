# Standardized Tabular Diffusion Benchmark

> **Development status:** this repository is currently a pre-alpha engineering workspace, not an official benchmark release. The existing `tabstruct-aligned-v1` output is a legacy compatibility path while the reviewed evaluation protocol is implemented. See the [development baseline](docs/DEVELOPMENT.md), [evaluation implementation roadmap](docs/evaluation/IMPLEMENTATION_ROADMAP.md), and [repository quality standard](docs/QUALITY_STANDARD.md).

This repository now includes a shared benchmarking layer on top of the upstream model code in:

- `TabDiff-main`
- `TabSyn-main`
- `TabDDPM-main`

The goal is to preserve authoritative implementations whenever possible, record any reviewed source patch explicitly, and give every registered adapter the same external contract for:

- training
- sample generation
- evaluation
- result comparison

The standardized layer is the preferred integration boundary. The vendored source trees are not assumed to be pristine until their revisions and local diffs have been audited.

Adapter presence is not a release claim. Run `python -m standardized_tabular_diffusion.cli list-models --details` to inspect source authority, modification status, validation level, benchmark track, and support level separately. At this stage, adapters remain experimental and unsupported. CoDi, CTAB-GAN, CTAB-GAN+, CTGAN, Goggle, TVAE, NRGBoost, REaLTabFormer, TabDDPM, TabDiff, TabSyn, STaSy, and SMOTE have passed separate Linux/Python 3.11 native-parity protocols, but that evidence does not make them benchmark-eligible or release-supported.

STaSy now targets the checksum-locked baseline snapshot distributed by TabSyn at commit `cb5ac0f`, not the materially different method-author repository. All nine binary, multiclass, regression, and seed parity cases passed exactly in [run `30936275831`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30936275831). This establishes `native-parity-validated` status only for the TabSyn snapshot; original-method, Official Results, benchmark-eligibility, and release-support claims remain blocked. See the [STaSy validation protocol](docs/STASY_VALIDATION.md) and [Chinese translation](docs/STASY_VALIDATION.zh-CN.md).

CoDi now targets the checksum-locked baseline snapshot distributed by the same TabSyn commit. All 11 CoDi files and the complete 24-file execution scope are fail-closed; checkpoints are isolated under `output_dir`, and exact requested sample rows are supported without changing tracked upstream source. All nine binary, multiclass, regression, and seed cases passed exactly in retained Linux/Python 3.11 [run `30941940893`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30941940893), so the adapter is `native-parity-validated` against the TabSyn snapshot. The separate method-author repository has no declared license and differs in five of ten shared paths, so original-method, Official Results, benchmark-eligibility, and release-support claims remain blocked; see the [CoDi validation protocol](docs/CODI_VALIDATION.md) and [Chinese translation](docs/CODI_VALIDATION.zh-CN.md).

Goggle now targets checksum-locked, unpatched method-author source from `vanderschaarlab/GOGGLE` commit `1a3d87ad`, acquired on demand under MIT. The former materially modified TabSyn-local copy has been removed. All nine binary, multiclass, regression, and seed cases matched the official GCN core exactly at checkpoint, raw-sample, and final CSV levels in retained Linux/Python 3.11 [run `30945676747`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30945676747), so this path is `native-parity-validated`. SAGE and heterogeneous decoding, Official Results, benchmark eligibility, and release support remain separate gates; see the [Goggle validation protocol](docs/GOGGLE_VALIDATION.md) and [Chinese translation](docs/GOGGLE_VALIDATION.zh-CN.md).

REaLTabFormer now targets the checksum-pinned official `realtabformer==0.2.4` wheel from World Bank method-author commit `73f23964`; no package source is vendored. All nine binary, multiclass, regression, and seed cases produced exact checkpoint tensors/files, saved configuration semantics, raw samples, and CSV bytes in retained Linux/Python 3.11 [run `30950369908`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30950369908), so the official tabular `n_critic=0` path is `native-parity-validated`. The integration remains conservatively `compatibility-patched` because of documented v0.2.4 save serialization and restricted state-dict loading boundaries. Sensitivity stopping, relational mode, Official Results, benchmark eligibility, and release support remain separate gates; see the [REaLTabFormer validation protocol](docs/REALTABFORMER_VALIDATION.md) and [Chinese translation](docs/REALTABFORMER_VALIDATION.zh-CN.md).

TabularARGN now targets the checksum-pinned official `mostlyai-engine==2.6.2` package from MOSTLY AI commit `0b96f02e`; no package source is vendored. All 50 wheel source files match the tag byte-for-byte, and every one of the 53 installed files is locked. The adapter replaces the former estimator pickle with an integrity-manifested official `ModelStore` and removes raw/encoded `OriginalData` after fitting. It remains `adapter-complete` until the mandatory Linux/Python 3.11 parity artifact is retained; see the [TabularARGN validation protocol](docs/TABULARARGN_VALIDATION.md) and [Chinese translation](docs/TABULARARGN_VALIDATION.zh-CN.md).

CTAB-GAN now targets method-author commit `73d4e315` under Apache-2.0. The former 15-file semantic fork has been replaced by seven checksum-locked official files; the adapter preserves the official stratified split, exposes classification only, and uses a documented compatibility bridge that forwards the unchanged mixture-component count to scikit-learn's keyword-only API. All six exact cases passed in retained Linux/Python 3.11 [run `30930939961`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30930939961), so the adapter is `native-parity-validated`; see the [CTAB-GAN validation protocol](docs/CTABGAN_VALIDATION.md) and [Chinese translation](docs/CTABGAN_VALIDATION.zh-CN.md).

CTAB-GAN+ now targets the method-author commit `6a6f901` instead of the semantically modified snapshot formerly embedded under `TabDDPM-main/`. Because the official repository declares no license, source is not redistributed: `materialize-model-source --model ctab-gan-plus` downloads and verifies five runtime files into an ignored local cache. Its mandatory Linux/Python 3.11 protocol passed all six classification/regression and seed cases in [run `30926267432`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30926267432), so its adapter status is `native-parity-validated`. This technical result does not remove the license, Official Results, or release-support blocks; see the [CTAB-GAN+ validation protocol](docs/CTABGAN_PLUS_VALIDATION.md) and [Chinese translation](docs/CTABGAN_PLUS_VALIDATION.zh-CN.md).

CTGAN now targets the checksum-pinned official `ctgan==0.12.1` package instead of the legacy embedded source snapshot. Its mandatory Linux/Python 3.11 protocol passed all exact comparisons for three fixed seeds in [run `30910275922`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30910275922), so its status is `native-parity-validated`. The official package uses BUSL-1.1; see the [CTGAN validation protocol](docs/CTGAN_VALIDATION.md) before installation or use.

TVAE now targets `ctgan.TVAE` from that same checksum-pinned official package. The locally modified `0.5.2.dev0` snapshot and its obsolete wrappers have been removed. Its mandatory Linux/Python 3.11 protocol passed all exact comparisons for seeds 0, 19, and 73 in [run `30913867621`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30913867621), and the immutable evidence is retained, so its status is `native-parity-validated`; see the [TVAE validation protocol](docs/TVAE_VALIDATION.md).

SMOTE now targets the checksum-pinned official `imbalanced-learn==0.14.2` package. Numerical, mixed-type, and all-categorical inputs use official `SMOTE`, `SMOTENC`, and `SMOTEN` respectively, without repository-side categorical encoding. All nine exact comparisons passed in [run `30918785254`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/30918785254), so its status is `native-parity-validated`. It remains a classification-only classical oversampling reference—not a joint tabular generator—and is excluded from generative-model ranking; see the [SMOTE validation protocol](docs/SMOTE_VALIDATION.md) and [Chinese translation](docs/SMOTE_VALIDATION.zh-CN.md).

The coexistence of the three vendored core baselines on one cumulative candidate is documented in the [core baseline integration validation](docs/CORE_BASELINES_INTEGRATION.md), with a corresponding [Chinese translation](docs/CORE_BASELINES_INTEGRATION.zh-CN.md) and machine-readable evidence index. CTGAN and TVAE are package-backed and have separate retained protocol evidence; SMOTE is package-backed and follows its separate classical-reference validation policy.

Project attribution and release review records live in [CONTRIBUTORS.md](CONTRIBUTORS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [SECURITY.md](SECURITY.md), [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md), and the [upstream source audit](docs/UPSTREAM_SOURCE_AUDIT.md). The repository-level license and dataset redistribution decisions remain release blockers until the project owners approve them.

## Shared Layout

The new root package is `standardized_tabular_diffusion/`.

- `interfaces.py`: common run and artifact schemas
- `models/`: adapters for each upstream model family
- `evaluation/`: versioned contracts, profile and registry loaders, result-bundle validation, and the isolated legacy path
- `comparison.py`: run aggregation utilities
- `cli.py`: a single entrypoint for listing models, describing metrics, running evaluations, and building comparisons

This organization is meant to make eventual migration into a single GitHub repository much easier: the upstream projects remain vendor-like sources, while the root package acts as the stable integration boundary.

## Standardized Interface

Every adapter exposes the same high-level operations:

- `train(spec)`
- `sample(spec)`
- `evaluate(spec)`

Each operation accepts a shared `RunSpec` object and returns standardized artifact metadata. Model-specific arguments still exist, but they are isolated inside `spec.extra`.

The currently supported adapters are:

- `arf`
- `bn`
- `codi`
- `ctab-gan`
- `ctab-gan-plus`
- `ctgan`
- `goggle`
- `great`
- `nflow`
- `nrgboost`
- `realtabformer`
- `smote`
- `stasy`
- `tabebm`
- `tabdiff`
- `tabddpm`
- `tabsds`
- `tabularargn`
- `tabula`
- `tabsyn`
- `tvae`

The repo now also includes a structured baseline inventory for the broader tabular-generation landscape, including both standardized adapters and known high-value methods that are not yet integrated.

The inventory also tracks non-generative foundation-model references separately from the runnable generator registry. Those entries are intentionally documented through the inventory/CLI without being forced into the `train / sample / evaluate` execution path.

Each standardized run writes canonical metadata such as:

- `artifacts.json`
- `pipeline_result.json`
- `standardized_summary.json` when evaluation is enabled

## Evaluation Protocol

The P1 evaluation foundation now provides versioned JSON Schemas and strict Python contracts for Evaluation Requests, Dataset Profiles, protocol profiles, Metric Registry entries, Atomic Results, stage records, manifests, metadata, summaries, and artifact indexes.

The pre-existing `tabstruct-aligned-v1` path is retained only for compatibility. Its fields have been migrated into explicit `legacy-diagnostic` Metric Registry records at lifecycle status `registered`; they are not source-parity validated, protocol frozen, release supported, or eligible for Official Results. `standardized_summary.json` is therefore not a leaderboard source of truth.

The approved result design uses one structured Atomic Result per metric scope, preserves raw and derived values separately, represents failures through six explicit result states, and stores finalized observations in `metrics.parquet`. P1 can create and validate auditable `incomplete` bundles, but deliberately cannot produce a finalized result. Metric execution and bundle finalization begin with the P2 vertical slice described in the [implementation roadmap](docs/evaluation/IMPLEMENTATION_ROADMAP.md).

Useful contract commands include:

~~~bash
std-tabular-diffusion list-metrics
std-tabular-diffusion validate-metric-registry
std-tabular-diffusion list-protocols
std-tabular-diffusion validate-dataset-profile --profile path/to/profile.json
std-tabular-diffusion validate-result --bundle path/to/result_bundle
~~~

## Benchmark Policy

The benchmark layer now makes a few explicit policy choices so results are easier to compare and reproduce:

- Dataset-level inputs are resolved through a canonical dataset registry in `standardized_tabular_diffusion/datasets.py`.
- Materialized datasets override raw upstream paths when available.
- `TabDiff` and `TabSyn` use a shared normalized evaluator.
- `TabDDPM` is normalized from the metrics already emitted by its upstream evaluation stack.
- Structural fidelity defaults to a reproducible local predictor set of `XGB + KNN`.
- `TabPFN` is disabled by default and only enabled when `STANDARDIZED_TABULAR_DIFFUSION_ENABLE_TABPFN=1` is set.
- When `TabPFN` is enabled, it is still treated as optional if it is unavailable because of gated-model access, unsupported class counts, or missing dependencies.

## Reproducibility

The standardized evaluation path now uses a benchmark-oriented deterministic configuration:

- fixed benchmark seeds in the MLE evaluator and structural-fidelity layer
- deterministic train/validation splitting in the upstream MLE path
- seeded row ordering for feature construction
- single-threaded XGBoost for the normalized MLE benchmark path
- stable structural-fidelity predictor policy emitted into the summary metadata

For the current smoke benchmark setup, repeated standardized evaluation runs now produce identical summary hashes.

## Environment

The repo now spans several dependency families that do not all evolve in lockstep:

- diffusion / CTGAN-style baselines
- Hugging Face transformer baselines
- DGL / graph baselines
- TabPFN-backed energy-based baselines

The current known-good reference stack is pinned in:

- `requirements-benchmark-stack.txt`

Important current caveats:

- `torch==2.3.0` is the current pinned benchmark runtime.
- `transformers==4.46.3` is intentionally pinned because the vendored `great` code is more compatible there than on the newer 4.57 line.
- `realtabformer` uses the checksum-pinned official 0.2.4 package. Its recorded adapter boundary disables unused torchvision probing during import, isolates official output paths, constrains state-dict loading, and works around the official `full_save_dir` JSON serialization defect without changing model state.
- `tabula` uses a local Transformers-based compatibility adapter rather than the original notebook-driven upstream workflow.
- `goggle` uses a frozen Python 3.11 validation stack and checksum-locked method-author source acquired on demand; only the GCN core has retained native-parity evidence, while SAGE and heterogeneous decoding remain unvalidated.
- `tabebm` sample generation is intentionally opt-in and requires accepted Prior Labs TabPFN model terms plus authentication before it can run.

For the operational status of each baseline, see:

- `docs/runtime_status.md`

## Dataset Acquisition

List the checksum-pinned public sources:

```bash
python -m standardized_tabular_diffusion.cli list-dataset-sources
```

Download and safely extract one source into the local cache:

```bash
python -m standardized_tabular_diffusion.cli download-dataset --dataset adult
```

Build Adult or Sick exclusively from the official checksum-pinned UCI train/test files:

```bash
pip install "standardized-tabular-diffusion[data]"
python -m standardized_tabular_diffusion.cli materialize-dataset --dataset adult
python -m standardized_tabular_diffusion.cli materialize-dataset --dataset sick
```

The Adult builder validates the official 32,561/16,281 split, exact source syntax, member and ordered-row hashes, class and missing counts, declared domains, and duplicate-row audits. It removes the test-only period from `income` labels and fits categorical modes on `adult.data` only. The official split's repeated and cross-split-identical rows are preserved and disclosed; old unbound tracked derivatives and unverified checkpoints have been removed.

The Sick builder validates the 2,800/972 official split, source-member hashes, class counts, record-ID identity and disjointness, missing counts, categorical domains, and duplicate-row audits. It fits numerical means and categorical modes on `sick.data` only. The official `TBG` field is preserved in the audit schema but excluded from model input because every source value is missing; record IDs are also audit-only. The fixed official split contains 11 cross-split duplicate model rows, which are preserved and disclosed rather than silently removed. The former unverified 2,205-row derivative has been removed.

The source registry fixes HTTPS URLs, licenses, citations, byte limits, selected archive members, and SHA-256 checksums. Downloading does not silently make a dataset official-eligible; its Dataset Profile must still freeze parsing, schema, splits, preprocessing, and rights review. See `docs/DATASET_ACQUISITION_AND_PREPROCESSING.md`.

For incomplete data, split first and run the centralized train-only preprocessor before registration:

```bash
python -m standardized_tabular_diffusion.cli preprocess-missing-values \
  --train-csv local-data/my_dataset/train.csv \
  --test-csv local-data/my_dataset/test.csv \
  --output-dir local-data/my_dataset/imputed-v1 \
  --numerical-column age \
  --categorical-column state \
  --target-column label
```

Numerical means and categorical modes are fitted on the real training split only. Validation and test reuse the frozen state; missing targets and missing generated values are rejected.

## Register A Local Dataset

For a complete local dataset, or a dataset already transformed by the centralized preprocessor, register its CSV into the legacy adapter layout:

```bash
python -m standardized_tabular_diffusion.cli register-dataset \
  --dataset my_dataset \
  --raw-csv data/uploads/my_dataset/raw.csv \
  --task-type classification \
  --target-column label \
  --numerical-column age \
  --categorical-column state
```

This writes metadata into `TabDiff-main/data/Info/`, copies the CSV into the adapter data roots, and makes the dataset visible to `list-datasets`. Registration fails closed when any missing value remains.

Then process it into the shared train/test layout:

```bash
python -m standardized_tabular_diffusion.cli process-dataset --dataset my_dataset
```

After that, the dataset can be used with the normal standardized model workflow:

```bash
python -m standardized_tabular_diffusion.cli example-config \
  --model tabsyn \
  --dataset my_dataset \
  --output-dir artifacts/tabsyn/my_dataset/run-001
```

## CLI

List available models:

```bash
python -m standardized_tabular_diffusion.cli list-models
```

Describe the shared metric schema:

```bash
python -m standardized_tabular_diffusion.cli describe-metrics
```

List the researched baseline inventory:

```bash
python -m standardized_tabular_diffusion.cli list-model-inventory
```

Filter the inventory to one benchmark paper:

```bash
python -m standardized_tabular_diffusion.cli list-model-inventory --benchmark tabstruct-2026
```

Filter the inventory to foundation-model references that are currently identity-registered but not adapter-complete:

```bash
python -m standardized_tabular_diffusion.cli list-model-inventory --family foundation --status registered
```

Inspect one model entry:

```bash
python -m standardized_tabular_diffusion.cli show-model-inventory --model ctab-gan-plus
```

Print the shared config schema:

```bash
python -m standardized_tabular_diffusion.cli describe-config
```

Generate an example config:

```bash
python -m standardized_tabular_diffusion.cli example-config \
  --model tabsyn \
  --dataset adult \
  --output-dir artifacts/tabsyn/adult/run-001
```

Resolve a config into the canonical run context:

```bash
python -m standardized_tabular_diffusion.cli build-context --config tmp/example.json
```

Run one standardized action:

```bash
python -m standardized_tabular_diffusion.cli run-action \
  --config tmp/example.json \
  --action evaluate
```

Run the full standardized pipeline:

```bash
python -m standardized_tabular_diffusion.cli run --config tmp/example.json
```

Compare previously normalized run summaries:

```bash
python -m standardized_tabular_diffusion.cli compare \
  --summary artifacts/tabdiff/adult/run-1/standardized_summary.json \
  --summary artifacts/tabsyn/adult/run-1/standardized_summary.json
```

Smoke presets for newly added baseline families live in:

- `configs/smoke/nrgboost-adult-smoke.json`
- `configs/smoke/ctab-gan-plus-adult-smoke.json`
- `configs/smoke/realtabformer-adult-tiny.json`
- `configs/smoke/arf-adult-smoke.json`
- `configs/smoke/stasy-adult-smoke.json`
- `configs/smoke/great-adult-train-smoke.json`
- `configs/smoke/great-adult-tiny.json`
- `configs/smoke/great-adult-distilgpt2-strong.json`
- `configs/smoke/tabebm-adult-smoke.json`
- `configs/smoke/tabebm-adult-gated-sample.json`

Additional usage notes are in `docs/smoke_presets.md`.

## Test One Model

To test one model on one dataset, the typical workflow is:

1. Materialize the dataset if the model depends on the shared processed layout.
2. Generate an example config.
3. Save the example config JSON to a file and edit it for the phases you want to run.
4. Run the standardized pipeline from that config.

Example for `tabsyn` on `adult`:

```bash
python -m standardized_tabular_diffusion.cli materialize-dataset --dataset adult
python -m standardized_tabular_diffusion.cli example-config \
  --model tabsyn \
  --dataset adult \
  --output-dir artifacts/tabsyn/adult/run-001
```

Save the emitted JSON to a file such as `tmp/tabsyn-adult.json`, then run:

```bash
python -m standardized_tabular_diffusion.cli run --config tmp/tabsyn-adult.json
```

If you only want one phase instead of the full pipeline:

```bash
python -m standardized_tabular_diffusion.cli run-action \
  --config tmp/tabsyn-adult.json \
  --action train
```

The main files to inspect afterward are:

- `run_context.json`
- `pipeline_result.json`
- `artifacts.json`
- `standardized_summary.json` when evaluation is enabled

For `tabddpm`, also set `upstream_config_path` in the experiment config so the standardized adapter can call the upstream TOML-based pipeline.

TabDDPM's adapter is `native-parity-validated` on Linux/Python 3.11 against the pinned author implementation. The protocol, three-seed exact comparisons, environment lock, workflow link, and permanent evidence are documented in `docs/TABDDPM_VALIDATION.md`. This validation does not yet make TabDDPM eligible for the Official Results track or release-supported.

## Tests

The standardized layer now has lightweight regression coverage for:

- reproducibility of dataset splitting and normalized summary generation
- adapter-level contracts for `tabdiff` and `tabddpm`

Run the current standardized test set with:

```bash
pytest tests/test_reproducibility.py tests/test_adapters.py
```

## Notes

- `TabDiff` and `TabSyn` can share the same normalized evaluator because both repos use the same `info.json`-style tabular metadata.
- `TabDDPM` currently has a partially different evaluation stack, so the adapter normalizes the metrics that are already available and marks unavailable TabStruct dimensions explicitly.
- `TabSyn` uses an unmodified, checksum-frozen official source scope. Device, seed, row-count, and sampling-step controls are isolated in the repository-owned invocation boundary, and three exact seed cases passed the retained native-parity protocol.
- Some upstream code has been patched locally to support standardization and reproducibility; these changes should be treated as part of the benchmark integration layer unless they are later upstreamed.
- This layer still tries to minimize changes to the original research code unless standardization or reproducibility requires them.
- The broader baseline roadmap and literature map now live in `docs/tabular_generation_landscape.md`.
