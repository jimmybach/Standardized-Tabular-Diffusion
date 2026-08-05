# P2 Shape and Trend Evaluation

- Status: implementation complete; authoritative Linux/Python 3.11 validation pending
- Protocol: `p2-shape-trend@0.2.0` (draft, diagnostic)
- Metric identities: `sdmetrics-column-shapes@1.0.0` and `sdmetrics-column-pair-trends@1.0.0`
- Official Results admission: no
- Last updated: 2026-08-05

## Scope and trust boundary

P2 is the first complete evaluation path in this repository. It accepts a decoded reference table and decoded synthetic table, applies the Dataset Profile structural gate, executes two exact upstream SDMetrics properties, preserves one Atomic Result per canonical column and unordered column pair, and atomically publishes a checksum-complete Run Result bundle.

The runtime path never imports `evaluation/tabstruct.py`. TabStruct remains research reference material, and the pre-P1 evaluator remains a legacy diagnostic compatibility path.

The selected source is the official [SDMetrics repository](https://github.com/sdv-dev/SDMetrics/tree/ba8842f2ba04ce914f698cc1cf746ca12338ab0e) at commit `ba8842f2ba04ce914f698cc1cf746ca12338ab0e`, distribution version `0.28.3.dev0`, under MIT. The backend hashes the LF-normalized bytes of all 121 installed Python source files and requires tree digest `784beda5c7a63d5ebb5fe74f98d00db3a2e018a29b2f32f643bf857750a6c2a9`. It also requires the installed MIT license digest `bd0e9ac3a8d0343ea371d392c6c13ff43a3b032ef75a5c9ec76ba93cec0d0b98`. Line-ending normalization makes source attestation independent of checkout policy without ignoring executable differences. A version, source-tree, or license mismatch stops evaluation before any metric is reported.

## Structural gate

File inputs may be UTF-8 CSV or Parquet; the Python gate also accepts a pandas DataFrame. Before metric execution, both inputs must satisfy all of the following:

1. the Dataset Profile is valid and its canonical order exactly matches its column records;
2. column names are unique, non-empty strings, and the column set exactly matches the profile;
3. columns are reordered to the canonical profile order;
4. numerical and integer columns convert without invalid or infinite values; integers remain exact in the canonical signed-int64 range rather than passing through float64;
5. Boolean and datetime values convert under explicit, fail-closed rules;
6. no reference or synthetic value is missing; and
7. the synthetic row count equals the request, which defaults in the CLI to the reference row count.

The gate does not silently impute, repair, clip, deduplicate, or enforce P3 domain and cross-column validity rules. A failure writes a structured failed `validate` stage with a stable reason code, leaves an auditable incomplete bundle, and creates no `metrics.parquet`.

## Exact source semantics

### Column Shapes

The upstream property selects the metric by SDMetrics semantic type:

- numerical and datetime: `KSComplement`, equal to one minus the two-sample Kolmogorov-Smirnov statistic;
- categorical and Boolean: `TVComplement`, equal to one minus the total-variation distance between empirical category frequencies; and
- unsupported types: not evaluated upstream and retained as `not_applicable` Atomic Results.

The source property score is the arithmetic mean of finite evaluated column scores. P2 copies the source score as `raw_value`, records identity normalization separately, assigns equal weight among finite source results, and verifies that Atomic Result contributions reconstruct the source property score within absolute and relative tolerance `1e-12`.

### Column Pair Trends

The upstream property evaluates every supported unordered pair:

- continuous-continuous pairs use Pearson `CorrelationSimilarity`, with score `1 - |r_real - r_synthetic| / 2`;
- discrete-discrete pairs use joint-frequency `ContingencySimilarity`;
- mixed pairs independently discretize the continuous column in the real and synthetic tables using the pinned source behavior, then use `ContingencySimilarity`; and
- unsupported semantic-type pairs remain explicit `not_applicable` Atomic Results.

The pinned Quality Report defaults are part of the metric identity: `num_rows_subsample=50000`, real absolute-correlation threshold `0.5`, and real Cramér's-V association threshold `0.3`. The source returns a non-finite score when the real relationship does not exceed the applicable threshold. P2 records that decision as `not_applicable` with reason and warning code `below_source_threshold`; it never silently removes the pair. The property score is the source arithmetic mean over finite contributing pair scores, and P2 independently reconstructs it from Atomic Results.

The locked source uses pandas sampling without an explicit random state when an input exceeds 50,000 rows. P2 therefore requires exactly one evaluator seed, applies it to the source's legacy NumPy random state while serializing the source call, and restores the caller's prior state afterward. This preserves the official computation while making repeated evaluations reproducible; the seed remains part of the request even when no subsampling occurs.

## Atomic Results and summaries

Every canonical column and unordered pair receives a stable scope based on Dataset Profile `column_id` values. Each row records the exact metric/version, dataset/view/split/model identity, state, raw and normalized values, aggregation contribution, source evaluator identity, reference and synthetic counts, valid/excluded counts, warnings, reason details, and a link to the retained source-detail artifact.

Supported states are `computed`, `mathematically_undefined`, `insufficient_support`, `not_applicable`, `implementation_failure`, and `resource_failure`. Only source-eligible finite results contribute. Coverage and state counts are visible in `summary.json`; failures and undefined outcomes cannot disappear from denominators.

P2 reports the two source property scores separately. It intentionally emits no combined or overall Fidelity score, performs no dataset aggregation, and does not create a leaderboard-eligible result.

## Finalized bundle

A successful run contains the request, environment, metadata, Atomic Results Parquet, summary, all seven stage records, event log, structural evidence, verbatim normalized SDMetrics detail tables, artifact index, manifest, and `checksums.sha256`. External reference and synthetic tables are content-addressed but are not copied into the bundle.

Every stage output checksum must match the manifest. `checksums.sha256` covers every finalized regular file except itself. Finalized status is written as the final atomic commit marker, so interruption before that replacement leaves the manifest `incomplete`; finalized bundles reject further writer mutation.

`validate-result` also parses every Parquet row back through the Atomic Result contract, checks scientific identity and unique scopes, recomputes state/warning coverage and both P2 property scores, verifies local artifact sizes/checksums/media types and producer stages, and reconciles external artifact provenance with the Evaluation Request.

## CLI

Install the isolated evaluation surface and run:

~~~bash
python -m pip install ".[evaluation]"

std-tabular-diffusion evaluate-table \
  --reference path/to/reference.csv \
  --synthetic path/to/synthetic.csv \
  --dataset-profile configs/datasets/adult-uci-2-v1.json \
  --output artifacts/evaluation/run-001 \
  --model-id my-model \
  --comparison-track native

std-tabular-diffusion validate-result --bundle artifacts/evaluation/run-001
~~~

`--expected-rows` may override the default reference row count. `--generation-seed` records the supplied synthetic artifact's generation identity. `--evaluator-seed` controls the locked source's subsampling above 50,000 rows and is recorded even when the input is smaller.

## Validation boundary

The dedicated workflow installs the exact official source on Linux/Python 3.11, attests the complete source tree, compares wrapper scores and detail DataFrames against direct authoritative calls, exercises numerical/categorical/Boolean/datetime/mixed and boundary inputs, repeats a seeded 50,001-row source-subsampling case exactly, checks denominator-complete Atomic Results, builds two semantically equivalent finalized bundles for one request, tests failure short-circuiting and interruption safety, lints, type-checks, and generates machine-readable evidence.

Passing P2 advances these two records only to `source-parity-validated`. Protocol freeze, Dataset Suite admission, Official Results eligibility, overall Fidelity definition, and repository release support remain later, independent gates.
