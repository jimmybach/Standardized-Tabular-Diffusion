# BN Validation Protocol

Status: protocol implemented; authoritative Linux/Python 3.11 evidence pending

Protocol ID: `pgmpy-bn-recipe-parity-v1`

Supported validation platform: Linux, Python 3.11, CPU

## Scope and claim boundary

This protocol validates the repository's declared discrete Bayesian-network recipe against direct calls to the unmodified official `pgmpy==1.1.2` package. The recipe applies deterministic quantile discretization to numerical columns, learns a DAG with pgmpy's BIC-scored hill-climb search, estimates conditional probability distributions with the BDeu prior, and samples with pgmpy's forward sampler.

pgmpy is a canonical Bayesian-network library, not the original implementation of a single BN synthesis paper. A passing authoritative run may therefore promote `bn` to `native-parity-validated` only for the exact official-package-plus-declared-recipe target. It does not reproduce a paper table, validate other discretizers, scores, priors, search procedures, or samplers, make BN `benchmark-eligible`, admit it to Official Results, or make it `release-supported`.

## Source authority, package identity, and license

- Official repository: `pgmpy/pgmpy`.
- PyPI package: `pgmpy==1.1.2`.
- Official wheel: `pgmpy-1.1.2-py3-none-any.whl`.
- Wheel SHA-256: `e55c78763a4a45dd644a13b250cea86af0c7e08590cf35de489624f34a4d9a0b`.
- Wheel size: 2,446,383 bytes.
- Git tag: `v1.1.2`; annotated tag object `ff663f9203c5075b2367707917016efafed03593`.
- Locked commit: `617cb48af678a7a471aad81d523ca95d2095430f`.
- Locked tree: `6c7adc00a479f540b2215889b1fac99a7b0b8a9c`.
- License: MIT; locked license SHA-256 `89171dcc8977530b0c101fbbb1c1d34caee998fc7def9eded629753cd2616a15`.

PyPI Trusted Publishing provenance binds both version 1.1.2 distributions to the locked GitHub tag commit. All 636 `pgmpy/` files in the wheel are byte-exact with the corresponding files in both the source distribution and locked tag; the tag contains twelve additional repository-only files. Before execution, the protocol verifies the wheel digest, all 649 wheel members, all 648 hashed `RECORD` entries, package metadata and dependency declarations, license, nine critical runtime files, their official Git blob identities, the installed distribution, and the public class locations. The same source checks run again after all model cases.

The annotated tag is not cryptographically signed. The trusted PyPI provenance, immutable package digest, tag commit/tree, byte-level comparisons, and Git blob locks are recorded explicitly instead of claiming a signed release.

## Declared recipe and adapter boundary

The validated recipe uses:

- `KBinsDiscretizer` with six quantile bins, ordinal encoding, `quantile_method="averaged_inverted_cdf"`, and `subsample=None`;
- `HillClimbSearch` with `scoring_method="bic-d"`, DAG output, maximum indegree two, 100 iterations, tabu length 20, and epsilon `1e-4`;
- `DiscreteBayesianEstimator` with `prior_type="BDeu"`, equivalent sample size five, and one job; and
- `BayesianModelSampling.forward_sample` with one job and the standardized seed.

The repository adapter does not modify pgmpy source. It supplies only the boundaries needed by the standardized benchmark:

- declared numerical/categorical roles and exactly one classification or regression target;
- strict rejection of missing and non-finite values until the explicit train-fitted preprocessing module has been run;
- deterministic treatment of constant numerical columns as a single state;
- inclusion of all canonical columns as graph nodes, including isolated variables;
- strict CPU and single-job execution for the validated deterministic path;
- restoration of the caller's process-global NumPy random state; and
- exact checks for requested row count, canonical column order, state domains, missing values, and numerical finiteness.

The adapter accepts only the validated score, prior, sampler, quantile method, and DAG return type. Changing those choices requires a new, separately identified recipe and parity protocol.

## Safe persistence boundary

The official objects are not stored with pickle. The adapter writes a typed `pgmpy-discrete-bn-state` JSON checkpoint containing the package identity, recipe, discretization edges, constant values, category/state names, graph edges, and CPDs. Loading validates the complete schema, reconstructs an official `DiscreteBayesianNetwork` with official `TabularCPD` objects, runs `check_model()`, and then calls the unchanged official sampler. A sidecar records the checkpoint SHA-256 and data-handling declarations. Reviewed external BN checkpoints can therefore be loaded without enabling the unsafe-pickle override.

The checkpoint intentionally omits row-level training data and executable objects. This is data minimization, not a privacy guarantee: bin edges, category levels, graph structure, CPDs, and training-frame fingerprints can disclose information about the fitted dataset. Trained checkpoints still require access control and retention review.

## Frozen environment

The workflow uses CPython 3.11 and the exact dependency set in `requirements-bn-validation.txt`. It downloads the official wheel over HTTPS, verifies its SHA-256 before installation, installs it and this repository without dependency resolution, and requires `pip check` to pass. The protocol rejects non-Linux hosts, non-3.11 interpreters, or any frozen distribution-version mismatch.

Equivalent Linux commands are:

```bash
python -m pip install --upgrade "pip==25.1.1"
python -m pip install -r requirements-bn-validation.txt
curl --fail --location --proto '=https' --tlsv1.2 \
  --output /tmp/pgmpy-wheel/pgmpy-1.1.2-py3-none-any.whl \
  "https://files.pythonhosted.org/packages/c6/5d/d03634ed296986abad834a69b0df21510cc9b6c40fb8afaed5df1c4b6074/pgmpy-1.1.2-py3-none-any.whl"
echo "e55c78763a4a45dd644a13b250cea86af0c7e08590cf35de489624f34a4d9a0b  /tmp/pgmpy-wheel/pgmpy-1.1.2-py3-none-any.whl" | sha256sum --check
python -m pip install --no-deps /tmp/pgmpy-wheel/pgmpy-1.1.2-py3-none-any.whl
python -m pip install --no-deps .
python -m pip check
```

## Frozen comparison

The protocol covers binary classification, multiclass classification, and regression. Each deterministic fixture has 60 missing-free rows, correlated and constant numerical columns, one categorical column, and one target. Each variant runs with seeds `0`, `19`, and `73` and requests 13 rows, producing nine cases.

For every case, the direct path independently preprocesses the persisted canonical CSV and invokes the official classes. The adapter path trains, saves, reloads, and samples through `BNAdapter`. Every case must establish all of the following without numerical tolerance:

1. Preprocessing state and the discrete training-frame hash are exact.
2. Learned graph edges and every CPD variable, evidence order, cardinality, state name, and probability are exact.
3. The JSON-restored official model is exact and passes official model validation.
4. Raw discrete samples, final CSV bytes, and reloaded frames are exact.
5. Both paths restore the caller's NumPy state.
6. Artifact manifests and checkpoint/sample hash sidecars are valid.
7. Output has the requested rows and canonical columns, valid domains and ranges, finite numerical values, and no missing values.
8. The checkpoint is non-executable JSON, omits row-level training data, makes no privacy overclaim, and declares that trained-artifact access control remains required.
9. The installed official package remains unchanged after validation.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.bn \
  --repo-root . \
  --output-dir /tmp/bn-validation \
  --evidence-path /tmp/bn-evidence.json \
  --wheel-path /tmp/pgmpy-wheel/pgmpy-1.1.2-py3-none-any.whl
```

`.github/workflows/bn-validation.yml` runs this command and retains its JSON artifact for 90 days. Any package, dependency, adapter, checkpoint schema, or protocol change requires a new run. Promotion is permitted only after a passing Linux/Python 3.11 artifact is inspected and retained unchanged under `docs/evidence/bn/`.

The protocol has passed a non-authoritative local Python 3.11 dry run covering all nine cases. The repository status remains `adapter-complete` until the authoritative Linux artifact is retained.
