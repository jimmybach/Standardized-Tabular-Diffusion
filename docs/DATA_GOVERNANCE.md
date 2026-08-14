# Data Governance and Release Inventory

Status: dataset privacy-role and bounded P5 threat-model review complete; independent release admission remains pending. Last working-tree review: 2026-08-13.

The unverified Sick derivative and the unbound Adult materializations have been removed from active repository paths. Adult and Sick are now obtained only from checksum-pinned official UCI archives through repository-owned builders. Generated model-input tables remain local artifacts rather than source-controlled benchmark data.

| Dataset | Current evidence | Sensitivity observation | Release status |
|---|---|---|---|
| `adult` | The registry and reviewed profile `configs/datasets/adult-uci-2-v1.json` lock UCI dataset 2, all five selected member hashes, the official 32,561/16,281 split, strict parsing, class and missing counts, train-only modes, and duplicate audits. | The person-level demographic, socioeconomic, quasi-identifier, and sensitive-attribute roles are reviewed. No direct identifier exists in the model view; `fnlwgt` is treated as a sampling weight rather than an identifier. | Official parsing/preprocessing and the bounded released-table black-box membership threat model are reviewed. Cross-split-identical rows remain preserved and disclosed. Fairness, semantic constraints, release rights approval, and dataset/suite admission remain independent gates. |
| `sick` | The source registry locks the `sick.data`, `sick.test`, and `sick.names` members of UCI dataset 102, its archive and member SHA-256 values, DOI citation, and the CC BY 4.0 license declared by UCI. The reviewed profile is `configs/datasets/sick-uci-102-v1.json`. | Health, treatment, laboratory, quasi-identifier, sensitive-attribute, and direct-identifier roles are reviewed. The source record suffix is a direct audit-only identifier excluded from model input and output. | Official parsing/preprocessing and the bounded released-table black-box membership threat model are reviewed. Medical bounds/constraints, duplicate treatment, release rights approval, and dataset/suite admission remain independent gates. |

Generated Adult and Sick CSV/NumPy files, synthetic-directory mirrors, checkpoints, upload copies, and materialization manifests are ignored local outputs. They are rebuilt with `materialize-dataset --dataset adult` or `--dataset sick` and are not source-controlled benchmark assets. On 2026-08-14, the maintainers approved a one-time branch-history rewrite after retaining a verified private archive; all public branch histories now exclude these obsolete objects. GitHub's read-only closed-pull-request references still require platform-side purging, as recorded in the [history sanitization notice](HISTORY_REWRITE_2026-08-14.md) and its [Chinese translation](HISTORY_REWRITE_2026-08-14.zh-CN.md).

## Release requirements

Before public release, every dataset must have a versioned Dataset Profile that records:

- canonical publisher and retrieval URL;
- dataset version or retrieval date and file checksum;
- license or terms of use and required attribution;
- whether redistribution is permitted or scripted local retrieval is required;
- schema, target, split identity, missing-value policy, and sensitive attributes;
- duplicate/leakage checks and an owner-approved inclusion decision.

Until those records pass review, CI examples should use tiny generated fixtures and the public repository should not publish a dataset archive. Any future history rewrite is a separate destructive release-cleanup decision and requires maintainer approval after a verified private archive is retained.

The executable source and missing-value controls are documented in `docs/DATASET_ACQUISITION_AND_PREPROCESSING.md`.
