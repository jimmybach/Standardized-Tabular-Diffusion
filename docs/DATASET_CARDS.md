# Dataset Cards

Dataset Profile JSON is authoritative. A source-build manifest, a model-input Dataset Profile, and materialized table files are different artifacts.

## Adult (UCI dataset 2)

- Fixed official train/test split; mixed numerical and categorical classification table.
- Missing markers occur in selected categorical fields and must be imputed from training modes only before model use.
- Dataset Profile: `configs/datasets/adult-uci-2-v1.json`.
- Source and rights: UCI, CC BY 4.0; attribution and exact checksums are recorded in the profile and source manifest.
- Current status: diagnostic suite; not automatically admitted to Official Results.

## Sick / Thyroid Disease (UCI dataset 102)

- Reviewed mixed-type classification view with its own frozen acquisition, split, and preprocessing declarations.
- Dataset Profile: `configs/datasets/sick-uci-102-v1.json`.
- Source and rights decisions are recorded in the profile; do not infer permission from public accessibility alone.
- Current status: diagnostic suite; not automatically admitted to Official Results.

## Quickstart artificial dataset

- Twenty-four artificial rows with no real people or events.
- Apache-2.0 and included only to validate install → official SMOTE adapter → P3 Result Bundle → diagnostic snapshot.
- Not a benchmark-quality dataset and excluded from scientific claims.

For any new dataset, record canonical source/version/checksums, rights, view and excluded fields, column semantics, target/task, frozen splits, missingness and train-only preprocessing, constraints, privacy roles, metric applicability, and independent admission. Use [Data Governance](DATA_GOVERNANCE.md) and the Dataset Profile schema.
