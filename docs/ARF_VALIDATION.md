# ARF Validation Protocol

Status: protocol implemented; authoritative Linux/Python 3.11 result pending

Protocol ID: `arfpy-official-package-parity-v1`

Supported validation platform: Linux, Python 3.11, CPU

## Scope and claim boundary

This protocol tests whether the standardized ARF adapter performs mixed-type FORDE/FORGE generation with the same retained density state and generated output as direct calls to the unmodified, method-author official `arfpy==0.1.1` Python package. The package is maintained under the same `bips-hb` organization as the original R package and lists ARF authors Kristin Blesch and Marvin N. Wright as its authors. It is therefore an official method-author Python implementation, not an unrelated local reimplementation.

A passing run may promote the adapter to `native-parity-validated` against this exact Python package. It does not establish numerical or behavioral equivalence with the separate R package, reproduce a paper table, make ARF `benchmark-eligible`, admit it to Official Results, or make it `release-supported`. Those claims have independent evaluation, dataset, runtime, governance, and release gates.

## Source authority, package identity, and license

- Official Python repository: `bips-hb/arfpy`.
- Related original R repository: `bips-hb/arf`.
- PyPI package: `arfpy==0.1.1`.
- PyPI source distribution: `arfpy-0.1.1.tar.gz`.
- Source-distribution SHA-256: `88170d5e72638b0dbfec28cfbdfee02e97bd6a06d5a636e960acd5d90d480707`.
- Source-distribution size: 11,841 bytes.
- Locked Git commit: `6f737baaaa589f7ac3ff59f0d739ce04b0f1381c`.
- Locked Git tree: `68b6fc5d28578a5c21bef560bd28f4c0d2d6401c`.
- License: MIT, copyright 2023 Kristin Blesch and Marvin Wright.

The repository has no tag or GitHub release for version 0.1.1. Provenance is instead established at file level: all six release files shared with the locked commit—`LICENSE`, `README.md`, `setup.py`, and the three `arfpy` Python files—are byte-exact Git blobs from that commit. PyPI published the source distribution approximately fourteen minutes after the commit. The protocol verifies the archive name, size, SHA-256, all 20 members, all 16 regular-file hashes, metadata, dependency declarations, MIT license, recorded Git-blob identities, installed package metadata, installed runtime hashes, self-consistency of installed `RECORD` hashes, import location, and the exported `arfpy.arf.arf` class before model execution.

## Adapter contract

The repository-owned adapter does not modify upstream source. It:

- requires the exact `arfpy==0.1.1` distribution and verifies all three installed runtime source files on every train and sample action;
- supports flat, single-table classification and regression datasets with numerical and categorical columns;
- converts declared categorical features and classification targets to pandas categorical dtype and checks all declared numerical values for finiteness;
- rejects missing values until the explicit benchmark preprocessing module has fitted numerical means and categorical modes on training data only;
- supports the official ARF controls `num_trees`, `delta`, `max_iters`, `early_stop`, and `min_node_size`, plus scikit-learn's execution control `n_jobs`;
- supports official FORDE `dist="truncnorm"`, `oob=false`, and non-negative `alpha`;
- requires CPU execution because this official implementation is based on scikit-learn random forests;
- applies the standardized seed to NumPy/pandas/SciPy operations and to `RandomForestClassifier.random_state`, then restores the caller's process-global NumPy state; and
- verifies requested row count, canonical column order, absence of missing values, and finite numerical output before writing samples.

`arfpy==0.1.1` exposes `forde(oob=True)` but that path references an attribute the constructor never creates. The adapter fails closed for `oob=true` instead of silently patching official source. The validated and upstream-default path is `oob=false`.

## Safe persistence boundary

Version 0.1.1 provides no save/load API. Pickling the complete official object would allow code execution during loading and would retain its fitted forest and encoded row-level training frame. Neither object is used by `forge()` after `forde()` has completed.

The adapter therefore stores a typed `arfpy-forge-state` JSON checkpoint containing only the exact attributes read by the unchanged official `forge()` method: column/type metadata, category levels, leaf bounds and coverage, continuous density parameters, and categorical probabilities. Non-finite distribution bounds are represented by explicit JSON tags and recovered exactly. Loading creates an uninitialized official `arfpy.arf.arf` instance, restores those attributes, and invokes the official method. A sidecar SHA-256 record detects accidental or partial checkpoint modification. Because parsing the checkpoint cannot execute Python code, a reviewed checkpoint may be loaded from outside `output_dir` without the unsafe-pickle override.

This persistence transformation is an adapter boundary, not a new ARF algorithm. The formal protocol compares every restored FORGE attribute against the live native object and requires exact generated CSV equality.

Omitting row-level data is a data-minimization measure, not a differential-privacy or non-disclosure guarantee. Leaf bounds, density parameters, category levels, and the canonical training-frame fingerprint can still reveal information about the fitted dataset. Checkpoints must therefore receive the same access control and retention review as other trained model artifacts.

## Frozen environment

The workflow installs CPython 3.11 and every dependency in `requirements-arf-validation.txt`. It downloads only the official PyPI source distribution, verifies its SHA-256 before building, builds without an isolated or floating build environment, installs the repository without dependency resolution, and requires `pip check` to pass. The protocol rejects non-Linux hosts, non-3.11 interpreters, and any dependency-version mismatch.

Equivalent Linux commands are:

```bash
python -m pip install --upgrade "pip==25.1.1"
python -m pip install -r requirements-arf-validation.txt
curl --fail --location --proto '=https' --tlsv1.2 --output /tmp/arf-sdist/arfpy-0.1.1.tar.gz "https://files.pythonhosted.org/packages/95/6f/a61794959d3860e23f5f2de5886b61154d40c246b38eedebf19d22e4cc35/arfpy-0.1.1.tar.gz"
echo "88170d5e72638b0dbfec28cfbdfee02e97bd6a06d5a636e960acd5d90d480707  /tmp/arf-sdist/arfpy-0.1.1.tar.gz" | sha256sum --check
python -m pip install --no-deps --no-build-isolation --use-pep517 /tmp/arf-sdist/arfpy-0.1.1.tar.gz
python -m pip install --no-deps .
python -m pip check
```

## Frozen comparison

The protocol covers binary classification, multiclass classification, and regression. Each deterministic fixture contains 60 missing-free rows, two correlated numerical columns, one categorical column, and one target. Strong dependencies ensure that every case executes one real adversarial refinement iteration rather than stopping after the initial discriminator.

For each fixture and each seed `0`, `19`, and `73`, both paths use 20 trees, `delta=0`, one maximum adversarial iteration, disabled early stopping, minimum leaf size two, single-threaded execution, truncated-normal FORDE, `oob=false`, `alpha=0`, and 13 requested rows. The native path directly constructs `arfpy.arf.arf`, calls `forde()`, and calls `forge()`. The adapter path performs the same operations through `ARFAdapter`, including its safe train/sample persistence boundary.

All nine cases must satisfy every condition below:

1. The PyPI archive, installed distribution, runtime files, class identity, dependency versions, and MIT license match their locks.
2. The native adversarial loop executes and both native and adapter calls restore the caller's NumPy state.
3. Original columns, factor/object masks, category levels, tree count, distribution type, leaf bounds, continuous parameters, categorical probabilities, and recorded OOB-accuracy sequence are exact after JSON restoration.
4. The checkpoint is non-executable JSON and declares that row-level training data and the random forest are absent.
5. Checkpoint and sample metadata hashes are valid and standardized artifact manifests identify the correct model and fixture.
6. Native and adapter CSV bytes and reloaded DataFrames are exact, with 13 canonical rows, finite numerical values, categorical values inside observed domains, and no missing values.
7. The installed official package remains byte-identical after validation.

There is no numerical tolerance: deterministic parity must be exact.

## Execution and promotion rule

The authoritative command is:

```bash
python -m standardized_tabular_diffusion.validation.arf \
  --repo-root . \
  --output-dir /tmp/arf-validation \
  --evidence-path /tmp/arf-evidence.json \
  --sdist-path /tmp/arf-sdist/arfpy-0.1.1.tar.gz
```

`.github/workflows/arf-validation.yml` runs the command and retains its JSON evidence for 90 days. Any package, dependency, adapter, checkpoint schema, or protocol change requires a new run. Promotion is allowed only after a passing Linux/Python 3.11 artifact has been inspected and retained unchanged under `docs/evidence/arf/`.

## Retained result

Pending. Until a passing workflow artifact is retained and cross-linked from the source lock, ARF remains `adapter-complete`, `experimental`, `unsupported`, and excluded from Official Results.
