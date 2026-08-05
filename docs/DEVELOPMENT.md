# Development Baseline

- Status: P0, P1, and P2 passed; authoritative Linux/Python 3.11 evidence is retained
- Primary environment: Linux and Python 3.11
- Last updated: 2026-08-05

## Installation surfaces

The package metadata intentionally separates dependency-light metadata and configuration APIs from optional data, evaluation, and model runtimes.

~~~bash
# Metadata, configuration, registry listing, and CLI help
python -m pip install -e .

# CSV/dataframe onboarding and comparison support
python -m pip install -e ".[data]"

# P2 source-parity evaluation path; still diagnostic until later release gates
python -m pip install -e ".[evaluation]"

# Dependency-light schema, registry, profile, and bundle validation
python -m pip install -e ".[contracts]"

# Common model runtime foundations; individual adapters may need more packages
python -m pip install -e ".[models]"

# Repository-owned tests that do not require model runtimes
python -m pip install -e ".[test]"

# Test, build, lint, and type-check tools
python -m pip install -e ".[dev]"
~~~

The broad [`requirements-benchmark-stack.txt`](../requirements-benchmark-stack.txt) remains a legacy convenience surface for existing adapters. It is not the core package contract and is not the future source-parity evaluation lock. Pinned backend-specific evaluation environments will be added with the metrics they validate.

## Import boundary

Importing `standardized_tabular_diffusion`, listing model names, loading configuration types, or displaying CLI help must not import NumPy, pandas, scikit-learn, PyTorch, AutoGluon, SDMetrics, or a model-specific package.

`get_adapter` imports only the selected adapter. If its optional runtime is missing, it raises `AdapterDependencyError` with the missing module and the nearest installation extra. Installing an extra does not by itself establish benchmark eligibility or release support.

## Test boundaries

Default discovery is restricted to [`tests/`](../tests). Vendored upstream suites and [`research_inputs/`](../research_inputs) are never collected by the default command.

~~~bash
# All available repository-owned tests; unavailable adapter suites are skipped
python -m pytest

# Dependency-light P0 contract
python -m pytest -m core

# Exclude model adapters explicitly
python -m pytest -m "not adapter"

# Run adapter tests after installing their runtime dependencies
python -m pytest -m adapter
~~~

Upstream tests are executed only in an explicit, isolated source-parity job with that source's dependency lock. They must not share import paths with ordinary repository tests.

## P1 evaluation contracts

P1 establishes identity and validation infrastructure; it does not calculate an official metric or finalize a result bundle. Built-in pre-P1 metrics are registered as `legacy-diagnostic`, remain at lifecycle status `registered`, and are ineligible for Official Results.

~~~bash
# Inspect and validate the packaged registry and protocol profiles
std-tabular-diffusion list-metrics
std-tabular-diffusion validate-metric-registry
std-tabular-diffusion list-protocols

# Validate reviewed or imported profiles and result bundles
std-tabular-diffusion validate-dataset-profile --profile path/to/dataset-profile.json
std-tabular-diffusion validate-protocol-profile --profile path/to/protocol-profile.json
std-tabular-diffusion validate-result --bundle path/to/result_bundle

# Convert existing upstream info.json metadata into a diagnostic-only profile
std-tabular-diffusion import-legacy-dataset-profile --dataset adult --output adult.legacy-profile.json
~~~

JSON Schema files under `standardized_tabular_diffusion/schemas/evaluation/` are the canonical wire validators. Python contracts enforce additional finite-number, result-state, lifecycle-evidence, path-safety, and cross-file invariants. JSON is canonical for hashing; configuration YAML is accepted only through a safe loader that rejects duplicate keys. TabStruct is research reference material only: the P1 foundation does not import it, and its pre-P1 evaluator remains a legacy diagnostic compatibility path.

## Quality commands

~~~bash
python -m ruff check standardized_tabular_diffusion tests
python -m mypy
python -m build
~~~

Mocked tests demonstrate control flow only. They do not satisfy source-parity, native-parity, real-smoke, scientific-validation, or release-support gates.

The dedicated `.github/workflows/p1-foundation-validation.yml` workflow runs the P1 contract suite, lint, typing, build, and a machine-readable exit-gate assessment on Linux/Python 3.11. A local or mocked pass is not a scientific metric validation.

P1 passed in [GitHub Actions run 31018595264](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31018595264). The exact [Linux/Python 3.11 evidence](evidence/evaluation/p1-foundation-run-31018595264.json) is retained in the repository; its scope explicitly excludes metric computation, source parity, and bundle finalization.

## P2 Shape and Trend evaluation

P2 adds the standalone `evaluate-table` command, the Dataset Profile structural gate, exact source-attested SDMetrics Column Shapes and Column Pair Trends, denominator-complete Atomic Results, Parquet metric storage, terminal stage records, and interruption-safe finalized bundles. It passed in [GitHub Actions run 31025796906](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31025796906), and the exact [Linux/Python 3.11 evidence](evidence/evaluation/p2-shape-trend-run-31025796906.json) is retained. See the [P2 specification and usage guide](evaluation/P2_SHAPE_TREND_EVALUATION.md). These metrics remain diagnostic and cannot affect Official Results until later protocol-freeze and release-support gates pass.

## CI baseline

The core workflow runs on Linux and Python 3.11 with read-only repository permissions. It verifies:

1. a dependency-free package installation and lazy-import smoke test;
2. repository-owned non-adapter tests;
3. lint and the scoped type-check baseline; and
4. wheel and source-distribution creation plus wheel-content inspection.

This workflow is a P0 gate, not evidence that every model or metric works in the official environment.
