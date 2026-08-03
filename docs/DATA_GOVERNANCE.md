# Data Governance and Release Inventory

Status: release-blocking review in progress. Last working-tree review: 2026-08-03.

The unverified Sick derivative and the unbound Adult materializations have been removed from active repository paths. Adult and Sick are now obtained only from checksum-pinned official UCI archives through repository-owned builders. Generated model-input tables remain local artifacts rather than source-controlled benchmark data.

| Dataset | Current evidence | Sensitivity observation | Release status |
|---|---|---|---|
| `adult` | The registry and reviewed profile `configs/datasets/adult-uci-2-v1.json` lock UCI dataset 2, all five selected member hashes, the official 32,561/16,281 split, strict parsing, class and missing counts, train-only modes, and duplicate audits. | The table contains person-level demographic and socioeconomic attributes. No declared direct identifier is present, but age, race, sex, native country, occupation, and income require privacy, fairness, and ethical-use review. | Official parsing and preprocessing are validated. The source split's 23 raw and 24 processed cross-split-identical rows are preserved and disclosed. Privacy/threat-model, fairness, semantic-constraint, metric-applicability, and suite-admission review remain release gates. |
| `sick` | The source registry locks the `sick.data`, `sick.test`, and `sick.names` members of UCI dataset 102, its archive and member SHA-256 values, DOI citation, and the CC BY 4.0 license declared by UCI. The reviewed profile is `configs/datasets/sick-uci-102-v1.json`. | The schema contains person-level health measurements and treatment/referral attributes. Age and sex require privacy-role review. The source-only record suffix is used for split auditing and excluded from model input. | Official parsing, the fixed 2,800/972 source split, record-ID disjointness, and train-only preprocessing are validated. `TBG` is retained in the raw audit schema but excluded from the model view because it is entirely missing. The official split has 11 cross-split duplicate model rows, which are preserved and disclosed. Privacy/threat-model, duplicate-treatment, and suite-admission review remain release gates. |

Generated Adult and Sick CSV/NumPy files, synthetic-directory mirrors, checkpoints, upload copies, and materialization manifests are ignored local outputs. They are rebuilt with `materialize-dataset --dataset adult` or `--dataset sick` and are not source-controlled benchmark assets. Historical active-path removals remain recoverable in ignored local quarantine during review; Git history has not been rewritten.

## Release requirements

Before public release, every dataset must have a versioned Dataset Profile that records:

- canonical publisher and retrieval URL;
- dataset version or retrieval date and file checksum;
- license or terms of use and required attribution;
- whether redistribution is permitted or scripted local retrieval is required;
- schema, target, split identity, missing-value policy, and sensitive attributes;
- duplicate/leakage checks and an owner-approved inclusion decision.

Until those records pass review, CI examples should use tiny generated fixtures and the public repository should not publish a dataset archive. Removing historical dataset blobs from Git history is a separate, destructive release-cleanup decision and requires maintainer approval after a clean archive is retained.

The executable source and missing-value controls are documented in `docs/DATASET_ACQUISITION_AND_PREPROCESSING.md`.
