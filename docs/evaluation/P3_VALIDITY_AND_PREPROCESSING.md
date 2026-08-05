# P3 Validity and Explicit Preprocessing Boundary

Chinese translation: [P3_VALIDITY_AND_PREPROCESSING.zh-CN.md](P3_VALIDITY_AND_PREPROCESSING.zh-CN.md)

- Status: Linux/Python 3.11 validated diagnostic implementation
- Protocol: `p3-validity@0.3.0`
- Metric versions: `1.0.0`
- Primary environment: Linux and Python 3.11
- Official Results allowed: no

## 1. Scope

P3 implements two benchmark-native metrics over an immutable decoded synthetic table:

- `std-tabular-column-validity@1.0.0`; and
- `std-tabular-constraint-validity@1.0.0`.

P3 also freezes the repository's explicit missing-value preprocessing boundary. Learned imputation state is fitted on the real training split only. Validation and test splits use that frozen state without refitting. Generated data is never imputed or silently repaired by this path.

P3 does not publish a leaderboard, define an overall benchmark score, approve pending dataset constraints, or modify any upstream generative-model implementation. Its formulas are defined by this repository and must not be attributed to PAFT, TabStruct, or SDMetrics.

## 2. Structural validation versus content validity

The P3 structural gate checks conditions required to interpret the table:

- readable CSV or Parquet serialization;
- unique column names;
- exact canonical model-view column set and row count;
- exclusion of audit-only and ignored source fields from model-output denominators; and
- safe base-type conversion.

P3 then preserves content violations for scoring. A missing cell, non-finite number, fractional value in an integer field, unknown category, out-of-domain value, or reviewed constraint violation is not silently corrected. This differs deliberately from the P2 protocol's stricter source-compatibility gate. Selecting `p2-shape-trend` continues to use the unchanged P2 behavior.

## 3. Dataset Profile validity contract

A P3-capable Dataset Profile declares `validity.contract_schema_version: 1.0.0`, reviewed hard column rules, reviewed hard cross-column constraints, soft diagnostics, and unresolved reviews.

Every hard rule or constraint carries:

- a stable identifier and version;
- an approved evidence class;
- an evidence reference;
- hard severity; and
- a closed, declarative rule type and parameters.

Approved evidence classes are authoritative data dictionaries, dataset documentation, legal or policy requirements, and recorded human review. Arbitrary Python expressions and dynamic code loading are prohibited.

### 3.1 Column rule types

The implementation supports:

| Rule | Meaning |
|---|---|
| `not_null` | Missing cells violate the declared model-input nullability rule. |
| `finite` | Non-missing numerical cells must be finite. |
| `integer` | Values must be finite mathematical integers representable as signed int64. |
| `allowed_values` | Values must belong to an explicit or Dataset Profile vocabulary. |
| `bounds` | Values must satisfy reviewed inclusive minimum and/or maximum bounds. |
| `regex` | Strings must fully match a reviewed regular expression. |
| `length` | String lengths must satisfy reviewed inclusive limits. |
| `unique` | Every non-missing value in the column must occur exactly once. |
| `datetime_range` | Parsed datetimes must lie inside reviewed inclusive bounds. |

Selectors may target column identifiers, semantic types, model-input nullability, roles, and required domain keys. Every canonical model-view column must match at least one hard rule; otherwise P3 fails closed before scoring.

### 3.2 Cross-column constraint types

The closed constraint language supports comparisons, conditional domains, mutual exclusion, numeric sum equality with explicit tolerances, allowed combinations, and functional dependencies. Each constraint also declares applicability and missing-value behavior.

Applicability may cover all rows, all complete rows, rows where one column equals a declared value, or rows where one column belongs to a declared set. A reviewed constraint with zero applicable rows is retained as `not_applicable`; it is never assigned a fabricated score of 1.

## 4. Metric definitions

For canonical column (j):

```text
valid_cell_rate_j =
    cells satisfying every applicable hard rule for column j
    / synthetic rows

column_validity_score = equal mean of valid_cell_rate_j over canonical columns
```

For reviewed constraint (k):

```text
constraint_satisfaction_rate_k =
    applicable synthetic rows satisfying constraint k
    / applicable synthetic rows

constraint_validity_score =
    equal mean over reviewed constraints with at least one applicable row
```

The overall dimension score is:

```text
if constraint_validity_score is available:
    validity_score = 0.5 * column_validity_score
                   + 0.5 * constraint_validity_score
otherwise:
    validity_score = column_validity_score
```

`fully_valid_row_rate` is reported separately. It never contributes to `validity_score`, because the probability that a row violates at least one rule grows mechanically with table width. Tests explicitly verify that identical column-level validity can produce different fully-valid-row rates.

## 5. Atomic Results and evidence

P3 emits:

- one computed Atomic Result per canonical model-view column;
- one Atomic Result per reviewed cross-column constraint;
- or one explicit `no-reviewed-constraints` not-applicable record when the profile declares none.

`artifacts/validity-details.json` retains rule identifiers, evidence references, per-rule violation counts, per-column valid and invalid counts, per-constraint applicable/satisfied/violating counts, component scores, and the fully-valid-row diagnostic. It records that the input was not mutated and no synthetic repair was applied. It does not copy source or generated row values into the result bundle.

Final bundle validation parses every Parquet Atomic Result and reconstructs both component scores and `validity_score`. It also cross-checks the detailed rule evidence, counts, scopes, immutable-input declaration, summary, metadata, artifact inventory, manifest, and checksums.

## 6. Current reviewed dataset behavior

The Adult and Sick diagnostic Dataset Profiles currently activate only rules supported by their reviewed model-input contracts and checksum-pinned source documentation:

- model-input non-nullability;
- numerical finiteness;
- integer semantics where declared; and
- declared categorical or Boolean vocabularies.

Observed training minima and maxima are soft diagnostics, not hard generative domains. Adult education-to-education-number consistency, authoritative numerical bounds, Sick medical ranges, and medical cross-column constraints remain unresolved. P3 does not promote them automatically. Adding or changing a hard constraint requires evidence review and a new Dataset Profile version.

## 7. Explicit missing-value preprocessing

The centralized preprocessing implementation uses the approved v1 policy:

- numerical features: arithmetic mean of observed real training values;
- categorical features: most frequent observed real training value;
- deterministic category-mode ties: Unicode-normalized lexical order;
- targets: missing values are errors and are never imputed;
- all-missing training features: fail closed;
- validation and test: transform with frozen training state only;
- generated samples: reject or score as invalid; never impute; and
- missing indicators: disabled by default and versioned when explicitly enabled.

The file workflow writes the transformed splits, `imputation-state.json`, and `preprocessing-manifest.json`. The manifest records input and output hashes, reports, learned-state hash and fingerprint, transformed-schema fingerprint, policy/configuration fingerprint, and a derived dataset-view token. A policy or transformed-schema change therefore creates a different dataset-view identity. A non-empty output directory is never overwritten.

Dataset registration retains a byte-identical source copy and records both hashes. Any separately materialized canonical CSV is labeled as an explicit serialization conversion. Registration does not drop rows or impute values.

## 8. CLI usage

P2 remains the backward-compatible default. Select P3 explicitly:

```powershell
std-tabular-diffusion evaluate-table `
  --protocol p3-validity `
  --reference real.csv `
  --synthetic synthetic.csv `
  --dataset-profile configs/datasets/adult-uci-2-v1.json `
  --expected-rows 32561 `
  --output results/adult-validity
```

Validate the finalized evidence again with:

```powershell
std-tabular-diffusion validate-result --bundle results/adult-validity
```

For raw real splits with missing values, run the separate preprocessing command before model training:

```powershell
std-tabular-diffusion preprocess-missing-values `
  --train-csv raw/train.csv `
  --test-csv raw/test.csv `
  --numerical-column age --numerical-column hours `
  --categorical-column workclass --categorical-column occupation `
  --target-column income `
  --output-dir processed/adult-imputed-v1
```

## 9. Admission status and limitations

Both P3 metrics are `unit-validated`, benchmark-native, and diagnostic. They are not source-parity metrics because no upstream implementation defines these repository formulas. They are not `protocol-frozen`, `release-supported`, or eligible for Official Results.

The dedicated P3 CI gate covers hand-computable rules, malformed profiles, no-constraint behavior, applicability, width sensitivity, no hidden repair, train-only preprocessing, schema and policy identity, registration preservation, bundle reconstruction, CLI execution, Linux/Python 3.11 packaging, lint, and typing. It passed in [GitHub Actions run 31036844043](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/31036844043); the exact [machine-readable evidence](../evidence/evaluation/p3-validity-run-31036844043.json) is retained with SHA-256 `bc63a2df553036ee7e161ce81c6f264dace950f3fe414ba2f8195d8e557e401d`.
