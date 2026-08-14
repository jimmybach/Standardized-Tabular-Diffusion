# Metric Cards

Metric Registry JSON is authoritative; this page is a reader-oriented index. Lifecycle and Official admission remain separate.

| Protocol | Primary dimensions | Input tables | Current publication boundary |
|---|---|---:|---|
| `p2-shape-trend@0.2.0` | SDMetrics Column Shapes and Column Pair Trends | real train + synthetic | Diagnostic; source-attested backend, no automatic Official admission |
| `p3-validity@0.3.0` | per-column and reviewed cross-column hard-rule validity | real train + synthetic | Diagnostic; no evaluator-side repair |
| `p4-utility@1.0.0` | Local Utility retention and TabStruct Eq. 4 Global Utility | real train + real test + synthetic | Protocol-frozen conditionally; dataset/model/run/release gates remain |
| `p5-high-order-privacy@1.0.0` | C2ST fidelity, duplicates/collisions, DCR, bounded DOMIAS attack | real train + real test + synthetic | Protocol-frozen conditionally; no overall score or formal privacy guarantee |

Every metric emits Atomic Results with exact metric identity, scope, state, direction, denominators, raw value, normalized value where defined, contribution, reason code, and evidence references. Missing, mathematically undefined, insufficient-support, implementation-failure, resource-failure, and not-applicable states are never silently converted to zero or dropped.

Legacy `tabstruct-aligned-v1` records are `legacy-diagnostic` only. They do not have the Atomic Results needed for the current aggregator and cannot enter Official Results.

See [Metric Governance](evaluation/METRIC_GOVERNANCE.md), [Evaluation Protocol](evaluation/EVALUATION_PROTOCOL.md), and packaged records under `standardized_tabular_diffusion/resources/evaluation/metrics/` for formulas, source identities, states, and admission fields.
