# Project Roadmap and Overall Progress

Status date: 2026-08-21

Scope: the single source of truth for whole-project phase status

Current working focus: **Phase 7 — V3 representative scientific validation (`⬜ Pending`; discussion required before execution)**

## 1. Purpose

This document is the authoritative progress index for the complete Standardized Tabular Diffusion project. It answers three questions:

1. Which whole-project phase are we in?
2. What evidence is required before a phase changes state?
3. How do subsystem labels such as P1-P8 and V0-V3 map to the whole-project phases?

Subsystem roadmaps remain authoritative for their own technical contracts, but they MUST NOT be interpreted as the overall project schedule. In particular:

- **P0-P8** describe evaluation and release-engineering implementation packages.
- **V0-V3** describe levels of baseline runtime validation.
- **Phases 1-10 in this document** describe the whole project.

Completing P8 does not mean that whole-project Phases 7-10 are complete. Likewise, a local installation acceptance is whole-project Phase 9 work, not a new P9 evaluation protocol.

## 2. Status vocabulary

| Status | Meaning |
|---|---|
| ✅ Complete | The declared phase scope and exit evidence are complete. Later release-specific rechecks do not reopen the foundational phase. |
| ✅ Complete within declared scope | Every target that the repository can lawfully and technically execute has passed; any named external exception is retained and excluded from the pass denominator. |
| ✅ Substantially complete | All work possible within the declared public-access scope is complete; a named external dependency remains blocked. |
| 🟡 Integration pending | Implementation and local acceptance pass, but the exact changes are not yet committed, reviewed, merged, and revalidated by hosted CI. |
| ⬜ Pending | The phase charter or required execution has not been completed. |
| ⏸ Deferred/optional | The phase is deliberately postponed until the project needs that publication level. |
| 🚫 Externally blocked | Progress requires access, rights, or infrastructure that repository maintainers cannot provide locally. |

Whole-project phases are gates, not a strict calendar. Engineering work from Phase 9 may be performed before scientific Phases 7-8 so that later experiments run on a reliable package. The status table records what has actually passed rather than forcing numerical execution order.

## 3. Overall progress

| Phase | Current status | Scope and current result |
|---|---|---|
| 1. Repository foundations and governance | ✅ Complete | Apache-2.0 project license, contributor credit, security and contribution policies, third-party notices, dependency policy, and Windows/Python 3.11 positioning are established. |
| 2. Baseline integration | ✅ Complete | Twenty-one baselines are registered behind the shared `train`, `sample`, and central `evaluate` interfaces. |
| 3. Upstream provenance and parity validation | ✅ Substantially complete | Twenty adapters are `native-parity-validated`. TabEBM remains `smoke-validated` because full generation requires externally gated TabPFN-v2 access. |
| 4. P1-P8 evaluation and release-engineering infrastructure | ✅ Complete | Contracts, metrics, Validity, Utility, empirical privacy, orchestration, aggregation, immutable snapshots, migration, packaging, and release checks are implemented. |
| 5. Native Windows real-function validation | ✅ Complete within declared scope | Eighteen planned minimal-real V2 runs passed; TabDDPM and TabDiff retain stronger representative-real runs. TabEBM is the sole external block and is not counted as a runtime failure. |
| 6. V2 audit closure | ✅ Complete | All 84 Phase-1 logic-audit cells completed, ten cross-cutting findings were repaired and regression-tested, all accessible runtime evidence was consolidated, and the repository-wide regression gate passed. |
| 7. V3 representative scientific validation | ⬜ Pending | The representative model set, datasets, full-scale configurations, scientific decision criteria, and three-seed matrix remain pending discussion and have not been approved as one V3 campaign. Existing protocol pilots do not automatically satisfy this phase. |
| 8. Formal benchmark experiments | ⬜ Pending | The complete admitted model-by-dataset-by-seed experiment matrix, quality interpretation, and formal leaderboard have not been executed. |
| 9. Internal usable-version closure | ✅ Complete | Clean wheel and source installs, official Adult acquisition, preprocessing, CLI journey, spaces/Chinese paths, representative CPU/GPU pipelines, and distribution hygiene passed locally. PR #34 was merged, and all 13 exact-merge-commit workflows passed on `main`, including Windows/Linux Core and P8 release gates. |
| 10. Formal public release | ⏸ Deferred/optional | A final version, GitHub Release, laboratory Organization transfer, public site, and paper/submission package are postponed until they are needed. |

## 4. Phase definitions and exit evidence

### Phase 1: Repository foundations and governance

Exit condition: repository-owned code has a declared license and attribution boundary; contribution, security, data-governance, platform, and third-party policies exist; unsupported claims fail closed.

Primary records: [LICENSE](../LICENSE), [NOTICE](../NOTICE), [third-party notices](../THIRD_PARTY_NOTICES.md), [contribution guide](../CONTRIBUTING.md), [security policy](../SECURITY.md), and [quality standard](QUALITY_STANDARD.md).

Release-time secret/history rechecks remain Phase 10 gates and do not make this foundational phase incomplete.

### Phase 2: Baseline integration

Exit condition: every selected baseline is registered, exposes the common external lifecycle, declares supported task/data/device boundaries, and routes evaluation through the central engine.

Current result: 21 registered adapters. Runtime status is maintained in [runtime status](runtime_status.md).

### Phase 3: Upstream provenance and parity validation

Exit condition: each accessible adapter identifies its exact authority, revision/package, redistribution status, modification class, parity protocol, and retained evidence. External access limitations are recorded rather than replaced by an unverified implementation.

Current result: 20 `native-parity-validated`; TabEBM is the declared external exception. See the [upstream source audit](UPSTREAM_SOURCE_AUDIT.md).

### Phase 4: P1-P8 evaluation and release-engineering infrastructure

Exit condition: the evaluation contracts and registries, Atomic Results, Result Bundles, P2-P5 metrics, P6 orchestration, P7 aggregation/publication, and P8 migration/packaging surfaces pass their own engineering gates.

Current result: P1-P8 engineering gates passed. This status does not admit any dataset, model, metric, run, or result to Official Results. See the [P1-P8 implementation roadmap](evaluation/IMPLEMENTATION_ROADMAP.md).

### Phase 5: Native Windows real-function validation

Exit condition: every non-blocked adapter performs a bounded real authoritative fit/generate workflow on native Windows/Python 3.11, preserves artifact/seed/output contracts, and reaches independent central bundle validation. Previously completed stronger runs may satisfy the same functional gate without being repeated.

Current result: all publicly accessible adapters satisfy the declared Windows functional scope; TabEBM remains externally blocked. See the [cross-baseline real-function audit](audits/PIPELINE_REAL_FUNCTION_AUDIT.md).

### Phase 6: V2 audit closure

Exit condition: cross-cutting logic findings are closed by tests, runtime evidence is indexed, no accessible V2 target remains pending, unresolved external blocks are explicit, and a full repository regression check passes.

Current result: the logic audit, remediation wave, V2 execution, and repository-wide regression are complete. This is operational evidence, not generator-quality evidence.

### Phase 7: V3 representative scientific validation

Exit condition: before execution, approve a bounded representative matrix containing:

- the models and justification for their inclusion;
- admitted dataset views and reviewed Dataset Profiles;
- native or approved standardized-tuning configurations;
- exactly three declared generation seeds unless a protocol requires more;
- compute/resource limits and failure policy;
- P2-P5 metric applicability and scientific decision rules; and
- a rule preventing exploratory outcomes from being relabeled as confirmatory evidence.

Current result: individual three-seed protocol confirmations and scientific pilots exist, but no approved whole-project V3 campaign exists. This phase therefore remains pending discussion.

### Phase 8: Formal benchmark experiments

Exit condition: execute the complete approved model-by-dataset-by-seed matrix; validate every Result Bundle; record missing/failed denominators; create independent admission records; and publish an immutable leaderboard snapshot without mixing incompatible tracks or environments.

Current result: infrastructure exists, but formal experiment execution and a rank-bearing Official snapshot have not begun. See the [P7 aggregation and leaderboard contract](evaluation/P7_AGGREGATION_AND_LEADERBOARD.md).

### Phase 9: Internal usable-version closure

Exit condition: for the exact integrated candidate, independently install wheel and source archive in clean Python 3.11 environments; follow the documented path from acquisition through Result Bundle; validate representative CPU/GPU workflows; test spaces and non-ASCII paths; audit CLI/error/output behavior; and prove that distribution archives contain no datasets, run outputs, credentials, retained machine evidence, or developer-local paths.

Current result: all local acceptance actions passed, including the recorded 646-test acceptance environment, repository-wide Ruff and Mypy, separate wheel/source installations, Adult download and preprocessing, CPU SMOTE, and RTX 5080 CTGAN. A subsequent Python 3.11/3.13 integration review and fresh distribution inspection also passed. [PR #34](https://github.com/jimmybach/Standardized-Tabular-Diffusion/pull/34) merged as [`1d500e1`](https://github.com/jimmybach/Standardized-Tabular-Diffusion/commit/1d500e18c00dcb866b1d3c1e7e81931a025baf98); all 13 workflows for that exact `main` commit passed, including [Core CI](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32511293539) and the [Windows/Linux P8 release gate](https://github.com/jimmybach/Standardized-Tabular-Diffusion/actions/runs/32511293507). See the [clean-install acceptance report](audits/CLEAN_INSTALL_USER_ACCEPTANCE.md).

### Phase 10: Formal public release

Exit condition, when activated: select and freeze the version; repeat release-candidate security, legal, history, packaging, Windows, and Linux gates on the exact commit; publish checksums and release notes; install published artifacts independently; and complete any chosen Organization transfer, website, DOI, or paper assets.

Current result: deliberately deferred. See the [release checklist](../RELEASE_CHECKLIST.md).

## 5. Mapping of naming systems

| Local naming system | Meaning | Whole-project placement |
|---|---|---|
| P0 | Trustworthy development baseline | Primarily Phase 1 |
| P1-P5 | Evaluation contracts and scientific metric subsystems | Phase 4; later consumed by Phases 7-8 |
| P6 | Resource-aware execution/orchestration | Phase 4 infrastructure; used in Phases 5 and 7-9 |
| P7 | Aggregation and immutable leaderboard publication | Phase 4 infrastructure; formal use belongs to Phase 8 |
| P8 | Legacy migration, packaging, documentation, and release-engineering surfaces | Phase 4 implementation; acceptance/release decisions belong to Phases 9-10 |
| V0/V1 | Mechanical and logic validation without expensive real training | Phases 3 and 6 |
| V2 | Bounded real authoritative functionality | Phases 5 and 6 |
| V3 | Representative-scale scientific validation | Phase 7 |
| Official Results | Fully admitted formal experiments and immutable ranking | Phase 8, followed optionally by Phase 10 publication |

## 6. Immediate next actions

1. Decide whether to begin Phase 7 now or keep the repository at an internally usable, non-Official state.
2. If Phase 7 begins, approve the representative model set, datasets, configurations, compute limits, three-seed matrix, and scientific decision rules before running experiments.
3. Keep Phase 8 and Official Results closed until the Phase 7 evidence and all independent admission gates justify formal benchmark execution.

## 7. Update rules

1. This file and [the Chinese translation](PROJECT_ROADMAP.zh-CN.md) MUST change together whenever a whole-project phase status changes.
2. A subsystem document may report its own completion but MUST NOT silently change a whole-project phase.
3. Every transition to `Complete` MUST link to retained evidence or an auditable merged implementation state.
4. Local passes MUST be labeled local until the exact changes are committed and required hosted gates pass.
5. External blocks MUST name the inaccessible dependency or right and MUST NOT be counted as repository runtime failures.
6. Scientific protocol completion, model parity, benchmark eligibility, release support, and Official Results admission remain independent claims.
7. The status date at the top MUST be updated whenever the table changes.
