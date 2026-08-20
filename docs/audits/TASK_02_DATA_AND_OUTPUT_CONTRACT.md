# T02: Data and Output Contract

Chinese translation: [TASK_02_DATA_AND_OUTPUT_CONTRACT.zh-CN.md](TASK_02_DATA_AND_OUTPUT_CONTRACT.zh-CN.md)

- Status: complete for all 21 registered identities at V0/V1; all 20 non-blocked identities retain a passing decoded-table observation and TabEBM retains its explicit external-access block
- Parent plan: [Cross-Baseline Pipeline Real-Function Audit](PIPELINE_REAL_FUNCTION_AUDIT.md)
- Risk class: leakage, schema corruption, or silent output repair

## Objective

Prove that each adapter consumes only the declared real-data roles and emits the requested decoded table without changing its scientific meaning. The check covers column order, logical types, missingness, row count, targets, categories, integers, and train-fitted preprocessing.

## V0/V1 checks for every adapter

- Resolve train, validation, test, and metadata paths to exact files and hashes before model execution.
- Verify that training never reads held-out test values, including through preprocessing, vocabulary construction, normalization, imputation, or model selection.
- Require the centralized preprocessor for admitted raw missingness. Means and modes are fitted on real train only; missing targets and synthetic-output imputation are rejected.
- Verify mapping from dataset-profile logical types to the upstream representation and back to the original decoded schema.
- Preserve declared column names and order. Reject duplicate names, missing columns, extra columns, and unsupported types before evaluation.
- Check Boolean, categorical, integer, floating, and target columns separately. Integer columns must be integral after decoding; categorical outputs must satisfy the declared policy for unseen values.
- Verify requested sample row count exactly, including batching remainders and counts smaller than a batch.
- Verify that output validation reports defects and does not silently clip, round, drop, reorder, impute, or repair data unless the reviewed adapter decoding contract explicitly requires that transformation.
- Exercise binary classification, multiclass classification, and regression where the adapter declares those task types.

## V2 minimal-real probe

Retain the input/profile checksums and generate a small decoded table through the authoritative algorithm. Run the central structural gate and record:

- exact row and column counts;
- ordered column names;
- decoded logical dtypes;
- missing, non-finite, fractional-integer, and unknown-category counts;
- target presence and role; and
- output file hash before evaluation.

The structural gate must inspect the generated table as produced. Any separate diagnostic repair must create a new explicitly labelled artifact and is ineligible for official evaluation.

## Failure conditions

- Any held-out test information influences training or train-fitted preprocessing.
- A generated value violates the declared schema but the run is reported as successful.
- Requested and emitted row counts differ.
- Target or feature columns are silently removed, renamed, reordered, or coerced.
- A malformed generated table is repaired in place before central evaluation.

## Exit gate

Every supported task type has contract coverage, every non-blocked adapter has a V2 decoded-table observation, and all silent mutations have either been removed or documented as an authoritative decoding operation with parity evidence.

## Completion record

Shared input identity, train-only preprocessing, schema, row-count, and immutable-output gates are regression-tested. The 18 minimal-real Windows probes and the two representative-real records all produced tables that passed the declared structural contract and central P3 finalization without synthetic-data repair. TabEBM remains externally blocked before full generation. This record establishes functionality only for the retained identities; it does not admit a dataset, model, or result to an official leaderboard.
