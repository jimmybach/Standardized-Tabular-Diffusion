# Repository Quality Standard

## 1. Purpose

This document defines the quality standard for developing, evaluating, and releasing the Standardized Tabular Diffusion Benchmark. It is the normative source for deciding:

- whether an implementation faithfully represents an upstream method;
- whether an adapter is sufficiently validated;
- whether results may appear in the official benchmark;
- whether a model is supported in a public release; and
- whether the repository as a whole is ready for release.

This standard is an acceptance specification, not a roadmap or a claim that the current repository already meets every requirement.

## 2. Scope and Non-Goals

### 2.1 Product scope

The project is a research benchmark, Python library, and command-line workflow for training, sampling, evaluating, and comparing tabular data generators.

The primary supported environment is:

- Linux;
- Python 3.11; and
- documented CPU and GPU configurations.

Windows support may be provided on a best-effort basis, but it is not a release-blocking platform unless a release explicitly states otherwise.

### 2.2 Non-goals

The project is not an online production service. The following are therefore outside the default release scope:

- public multi-tenant APIs;
- user authentication and authorization;
- request rate limiting;
- service-level availability guarantees;
- high-availability deployment; and
- Kubernetes or equivalent service orchestration.

This exclusion does not reduce the requirements for local security, dependency safety, data governance, safe checkpoint handling, or reliable research execution.

### 2.3 Language policy

Normative repository files, public documentation, code comments, issue templates, and release materials MUST be written in English. Internal discussion and review MAY be conducted in Chinese.

## 3. Normative Language

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** express requirement strength:

- **MUST/MUST NOT**: mandatory for the applicable release or eligibility decision;
- **SHOULD/SHOULD NOT**: expected unless a documented exception is approved; and
- **MAY**: optional.

An exception to a MUST requirement requires a written decision that identifies the owner, rationale, risk, scope, and expiration or review date. An exception MUST NOT be used to make an ineligible implementation appear eligible for the official benchmark.

## 4. Implementation Identity, Validation, Eligibility, and Support

Source authority, distribution form, reproduction target, modification status, validation level, benchmark track, and support level are independent properties. A model MUST NOT be described only as `implemented`, and one property MUST NOT be used as evidence for another.

### 4.1 Source authority

Every component that materially affects the implemented algorithm MUST declare one source authority:

| Source authority | Definition | Default benchmark treatment |
|---|---|---|
| `method-author` | Maintained or explicitly endorsed by the authors of the method being represented. | Candidate for the official track. |
| `benchmark-vendored` | Distributed as a baseline snapshot inside an official benchmark or paper repository, but not necessarily maintained by the original method authors. | Candidate only for a declared `benchmark-snapshot` reproduction target. |
| `third-party` | Maintained by parties other than the method authors or the maintainers of the declared benchmark snapshot. | Experimental track. |
| `local` | Written or substantially reconstructed in this repository. | Experimental track. |

A repository or package is not authoritative merely because it has the same method name. Evidence of authorship, maintenance, or endorsement MUST be recorded. A model composed of multiple material components MUST list each component and MUST NOT claim stronger authority than its core algorithm component supports.

### 4.2 Distribution form and reproduction target

Each source component MUST record whether it is consumed as `source`, `package`, or `hybrid`, together with its immutable revision or exact version.

Each adapter MUST declare one reproduction target:

- `original-method`: the declared target is the method-author implementation; or
- `benchmark-snapshot`: the declared target is the exact baseline version shipped by a named benchmark repository.

A `benchmark-vendored` implementation MUST NOT be described as the method-author implementation. Results targeting a benchmark snapshot MUST name that benchmark and revision.

### 4.3 Modification status

Each implementation MUST declare one modification status:

| Modification status | Definition | Official-track treatment |
|---|---|---|
| `unmodified` | The pinned upstream source or package is used without repository-specific behavioral adaptation. | Candidate after validation. |
| `adapter-only` | Project code adapts configuration, data layout, invocation, or artifacts without changing upstream source or monkeypatching upstream behavior. | Candidate after validation. |
| `compatibility-patched` | A reviewed upstream-source patch addresses entrypoint, platform, dependency-API, or equivalent compatibility without changing scientific semantics. | Candidate only after patch approval and native parity validation. |
| `semantic-patched` | A local change may affect the model, objective, preprocessing, randomness, sampling, or evaluation meaning. | Experimental track only. |

Using a non-default option exposed by an upstream public interface is `adapter-only`, not a source patch, but the option MUST still be reported. Monkeypatching upstream internals counts as a patch.

A semantic patch MAY be reclassified only after it is accepted into a pinned authoritative upstream release or is treated as a separately named method. Local approval alone does not make a semantic patch an official implementation of the original method.

### 4.4 Validation level

Validation level is cumulative. The recorded value is the highest level for which all preceding validation requirements have been met:

1. `registered`
   - Model identity, paper, source components, license status, reproduction target, and intended benchmark role are recorded.
   - A runnable adapter is not implied.

2. `adapter-complete`
   - Required train, sample, and evaluate operations are connected or explicitly declared not applicable.
   - Configuration and artifact contracts are implemented.
   - Successful real execution is not implied.

3. `smoke-validated`
   - A small end-to-end run succeeds in a supported Linux and Python 3.11 environment.
   - Produced samples and artifacts pass schema and integrity checks.
   - Scientific equivalence to the reproduction target is not implied.

4. `native-parity-validated`
   - The standardized path is compared with the declared reproduction target using a predeclared parity protocol.
   - Deterministic transformations and configuration mappings match exactly where exact matching is expected.
   - Stochastic outputs and metrics meet documented tolerances across enough seeds to support the equivalence claim.
   - Any compatibility patch is included in the validation.

Passing a smoke test MUST NOT be presented as evidence of native parity, benchmark eligibility, scientific correctness, or result quality.

### 4.5 Benchmark track

Each model MUST declare one benchmark track:

- `official`: the model has passed the per-model official-track gate in Section 22;
- `experimental`: results may be reported only under the experimental policy in Section 5; or
- `excluded`: the model is registered for provenance or planning but its results MUST NOT be published by this benchmark.

Native parity validation is necessary but not sufficient for the official track. Benchmark-track assignment is an eligibility decision, not a validation level.

### 4.6 Support level

Each model MUST declare one support level:

- `unsupported`: no installation, compatibility, or maintenance commitment;
- `experimental`: best-effort integration with limited support; or
- `release-supported`: included in the public support matrix with a maintenance owner, documentation, tests, compatibility record, and deprecation commitment.

Support level is independent of benchmark track. A release-supported experimental implementation remains experimental and MUST NOT be presented as an official result.

### 4.7 Required status record

Each model MUST expose machine-readable metadata equivalent to:

```yaml
model_id: example-model
source_components:
  - role: core-algorithm
    authority: method-author
    distribution_form: source
    repository: https://example.org/owner/repository
    revision: full-commit-sha
    package: null
    package_version: null
    retrieved_at: 2026-01-01
    integrity_hash: sha256:example
reproduction_target:
  kind: original-method
  identifier: owner/repository@full-commit-sha
modification_status: adapter-only
patch_set: null
validation:
  level: smoke-validated
  parity_protocol: null
  evidence:
    - path-or-artifact-identifier
  repository_commit: full-commit-sha
  environment_lock: lockfile-hash-or-identifier
  validated_at: 2026-01-01
benchmark:
  track: experimental
  decision: pending
  reasons:
    - Native parity validation has not been completed.
support:
  level: experimental
  owner: null
decision:
  reviewed_by: []
  reviewed_at: null
```

Status claims MUST reference inspectable or reproducible evidence. Historical success on an unrecorded developer environment is insufficient. Derived fields MUST be generated or consistency-checked so that validation, benchmark, and support records cannot contradict one another silently.

## 5. Benchmark Result Tracks

### 5.1 Official track

An implementation MAY enter the official track only when:

- an `original-method` target uses `method-author` authority;
- a `benchmark-snapshot` target uses the exact declared `benchmark-vendored` snapshot or an authoritative source proven equivalent to it;
- its modification status is `unmodified`, `adapter-only`, or approved `compatibility-patched`;
- it has passed native parity validation against its declared reproduction target; and
- it has passed the complete per-model official-track gate in Section 22.

A `semantic-patched`, `third-party`, or `local` implementation MUST NOT enter the official track. A benchmark-vendored implementation MUST NOT be used to claim reproduction of the original method unless it is separately validated against an authoritative method-author target.

Every official result MUST identify its source components, reproduction target, upstream revisions or package versions, modification status, patch-set identifier, parity evidence, benchmark protocol version, dataset version, environment, configuration, seeds, and immutable run identifier.

### 5.2 Experimental track

The experimental track MAY contain third-party ports, local reimplementations, semantic patches, prototypes, ablations, and otherwise authoritative implementations that have not completed the official-track gate.

Experimental results MUST:

- be visually and semantically separated from official results;
- state why the implementation is experimental;
- identify the actual source authority and modification status;
- avoid claims that the result represents the original method unless that claim has been validated; and
- remain excluded from official rankings and official aggregate claims.

Clearly labeled experimental summaries MAY aggregate experimental results. Experimental implementations SHOULD be replaced by authoritative implementations when suitable upstream code exists. If retained, they SHOULD live behind an explicit experimental namespace or opt-in mechanism.

### 5.3 Excluded track

An implementation MUST be `excluded` when a known legal, data-rights, security, identity, or scientific-integrity problem makes result publication inappropriate. Exclusion reasons and the conditions for reconsideration MUST be recorded.

### 5.4 Comparability

Results MUST be compared only within compatible protocol versions and resource regimes. A result produced under a different data split, tuning budget, sample count, evaluator, metric definition, or reproduction target MUST NOT be silently combined with another result. Protocols MAY define separately labeled native-default and controlled-budget comparisons.

## 6. Upstream Fidelity and Change Control

### 6.1 Adapter-first rule

Integration MUST be implemented at the adapter boundary whenever technically feasible. Without prior approval under Section 6.3, project code MUST NOT modify, monkeypatch, duplicate, or rewrite an upstream training objective, sampling algorithm, model architecture, optimizer schedule, preprocessing algorithm, randomness policy, or evaluation algorithm.

The standardized layer MAY adapt:

- validated configuration mapping;
- input and output data layout;
- process invocation;
- output discovery;
- artifact metadata; and
- representation-only metric normalization.

Metric normalization MUST preserve the upstream metric's definition, direction, scale, and missing-value semantics. Recomputed or newly defined metrics belong to the versioned evaluation layer and MUST NOT be represented as normalized upstream outputs.

### 6.2 Upstream provenance

For every material implementation component, the repository MUST record:

- source authority and component role;
- distribution form and reproduction target;
- canonical upstream repository URL or package name;
- immutable commit SHA or exact package version;
- retrieval date;
- upstream license and required notices;
- integrity hash of the imported source or distribution;
- import method, such as package dependency, submodule, subtree, or vendored snapshot; and
- known deviations from upstream.

Copying a source tree without its revision metadata is not acceptable for official-track or release-supported use.

### 6.3 Source modification gate

Modification or monkeypatching of upstream source is a last resort and MUST be proposed and approved before implementation by the designated model owner and a reviewer responsible for scientific integrity. The decision MUST receive a durable identifier and include:

1. the upstream file and revision;
2. the observed incompatibility or defect;
3. why an adapter-only solution is insufficient;
4. the smallest proposed change;
5. classification as compatibility or semantic;
6. the possible effects on algorithm, preprocessing, randomness, sampling, and evaluation;
7. the validation and rollback plan;
8. license and attribution implications; and
9. the reviewers, owner, and decision date.

Emergency experimentation MAY begin on a separate experimental branch before approval, but the change MUST NOT enter a supported branch, official result, or release artifact until the gate is complete.

### 6.4 Patch isolation, verification, and eligibility

Approved changes MUST be isolated as a reviewable patch set and linked from the model status record. Silent edits inside vendored source are prohibited. Formatting or cleanup changes MUST NOT be mixed with behavioral patches.

Every compatibility patch MUST have tests that demonstrate the unpatched incompatibility and the patched behavior. It MUST pass native parity validation before official-track consideration and MUST be identified in published result provenance.

A semantic patch MUST remain experimental even when it has regression or parity-comparison evidence. If the change is later accepted by the authoritative upstream project, the repository MUST update to the accepted immutable revision and repeat provenance, parity, and eligibility review.

### 6.5 Upstream synchronization

The project MUST check upstream releases and security fixes before every official release and at a documented periodic cadence. Updating an upstream revision MUST trigger provenance, license, dependency, parity, and result-compatibility review. A support record MUST identify who owns this review.

## 7. Scientific Correctness

### Objective

Ensure that each reported result corresponds to the declared method and that experimental conclusions are supported by valid computations.

### Requirements

- Data preprocessing, feature typing, missing-value handling, target handling, and inverse transformations MUST be specified and tested.
- Train, validation, and test boundaries MUST be explicit and free from leakage.
- Training and sampling configuration mappings MUST be complete; unsupported parameters MUST fail clearly rather than be silently ignored.
- Model outputs MUST preserve the declared table schema, column order, data types, valid domains, and requested row count, or fail explicitly.
- Numerical failures, invalid samples, empty outputs, and partial metrics MUST NOT be silently converted into successful runs.
- Scientific claims MUST distinguish exact reproduction, parity within tolerance, adaptation, approximation, and new implementation.
- Stochastic comparisons MUST use enough seeds and report uncertainty appropriate to the claim.
- Metric direction, aggregation, weighting, and handling of undefined values MUST be explicit.
- Any deviation from a paper or upstream default MUST be recorded.

### Required evidence

- preprocessing and inverse-transform tests;
- reference configurations;
- schema validation reports;
- parity or golden-result reports;
- multi-seed summaries where stochastic claims are made; and
- a documented limitations section.

## 8. Evaluation Integrity and Fairness

### Objective

Ensure that model comparisons measure the intended qualities under transparent and compatible conditions.

### Requirements

- Every benchmark protocol MUST have a versioned specification.
- Dataset versions, splits, target definitions, feature roles, and checksums MUST be fixed for a protocol version.
- Sample counts, evaluation seeds, evaluator versions, tuning rules, and resource budgets MUST be recorded.
- The same metric implementation and post-processing policy MUST be used for comparable results unless the difference is explicit.
- Evaluation MUST validate sample schema and row integrity before computing metrics.
- Missing, failed, skipped, gated, or unsupported metrics MUST remain distinguishable; they MUST NOT be replaced with favorable default values.
- Metrics imported from an upstream evaluator MUST identify that evaluator and MUST NOT be mixed with locally recomputed metrics without provenance.
- Hyperparameter selection MUST use only permitted data and MUST follow a declared tuning budget.
- Benchmark summaries MUST preserve per-seed results in addition to aggregates.
- Statistical comparisons SHOULD report confidence intervals or an appropriate alternative.
- Privacy, fidelity, utility, distinguishability, structural, and efficiency metrics MUST state their definitions and limitations.

### Required evidence

- protocol specification and version;
- machine-readable metric definitions;
- evaluator tests;
- raw per-run and per-seed results;
- comparison compatibility checks; and
- a leaderboard eligibility report.

## 9. Reproducibility and Provenance

### Objective

Allow another qualified researcher to recreate a run and understand every material input.

### Requirements

Every run MUST record:

- repository commit;
- upstream revisions or package versions;
- local patch identifiers;
- Python and operating-system versions;
- dependency lock identifier;
- CPU, GPU, CUDA, and relevant driver information;
- dataset and split checksums;
- full resolved configuration;
- all random seeds;
- requested and actual sample counts;
- start and end times;
- success, failure, or partial status; and
- hashes of material outputs.

Randomness MUST be controlled across all applicable libraries. Deterministic operation MUST be described as one of:

- bitwise deterministic;
- numerically reproducible within tolerance; or
- statistically reproducible across repeated runs.

Absolute developer paths MUST NOT be required to reproduce a run. Manifests SHOULD use repository-relative paths, content identifiers, or relocatable references.

### Required evidence

- a locked environment;
- a complete run manifest;
- repeat-run verification on the supported platform; and
- documented reproducibility tolerances.

## 10. Data Governance and Privacy

### Objective

Use only data that the project is permitted to process and redistribute, while preventing leakage and inappropriate disclosure.

### Requirements

- Every dataset MUST declare its source, version, retrieval method, checksum, citation, license or terms, and redistribution status.
- Data with unknown or incompatible redistribution rights MUST NOT be committed or included in a release artifact.
- Private, confidential, personal, regulated, or institution-restricted data MUST NOT be committed to the public repository.
- Raw data SHOULD be downloaded or registered through reproducible scripts rather than duplicated across upstream directories.
- Dataset names and paths MUST be validated against traversal and unsafe filesystem operations.
- Dataset onboarding MUST validate schema uniqueness, target presence, encodings, missing values, sizes, and split integrity.
- Real and synthetic data MUST be clearly distinguished in storage and metadata.
- Privacy metrics MUST state the threat model and MUST NOT be described as a privacy guarantee unless the method provides one.
- Retention and cleanup rules MUST exist for local datasets, caches, checkpoints, and generated samples.

### Required evidence

- dataset cards or equivalent manifests;
- license and redistribution review;
- checksum verification;
- leakage tests; and
- privacy-risk documentation.

## 11. Security and Supply Chain

### Objective

Protect users and maintainers from unsafe dependencies, files, downloads, commands, and repository history.

### Requirements

- Secrets and access credentials MUST NOT be committed. Automated secret scanning MUST cover the working tree and Git history.
- Untrusted pickle, joblib, NumPy object arrays, and PyTorch checkpoints MUST be treated as executable or unsafe content.
- Documentation MUST state that serialized model artifacts may be loaded only from trusted sources unless a safe format is used.
- Remote downloads MUST use authenticated transport where available and SHOULD verify an expected checksum.
- Subprocess execution MUST avoid shell interpolation and MUST validate externally influenced paths and arguments.
- Archive extraction MUST prevent path traversal.
- Dependency versions MUST be resolved through a reproducible lock and scanned for known vulnerabilities.
- A software bill of materials SHOULD be generated for public releases.
- Optional model dependencies MUST be isolated so that installing or importing the core package does not execute or import unrelated heavy frameworks.
- Security-sensitive failures MUST be visible and MUST NOT be suppressed as successful benchmark results.

### Required evidence

- secret-scan report;
- dependency and vulnerability report;
- unsafe-deserialization inventory;
- download and checksum policy;
- security tests; and
- `SECURITY.md` with a reporting process.

## 12. Architecture and Extensibility

### Objective

Keep the integration layer understandable, testable, and able to support multiple incompatible upstream stacks.

### Requirements

- Core metadata and help commands MUST work without importing every optional model dependency.
- Model registration MUST be lazy or otherwise dependency-isolated.
- Each adapter MUST implement a documented contract for configuration, training, sampling, evaluation, artifacts, and failures.
- Model-specific options MUST use a validated schema; arbitrary unvalidated dictionaries SHOULD NOT be the primary public configuration interface.
- Configuration precedence MUST be explicit and duplicate keys MUST NOT be silently overwritten.
- Adapters MUST NOT permanently mutate global environment variables, import paths, random state, or process-wide library settings.
- Upstream processes SHOULD run in isolated working and artifact directories.
- Dataset, model, evaluator, and artifact interfaces SHOULD remain independent.
- Public interfaces MUST follow a documented compatibility and deprecation policy.
- Adding one model SHOULD NOT require editing unrelated model implementations.

### Required evidence

- architecture documentation;
- adapter contract tests;
- configuration schema tests;
- optional-dependency import tests; and
- an extension guide.

## 13. Code Quality and Maintainability

### Objective

Keep original project code consistent, reviewable, testable, and economical to change without imposing repository-wide stylistic rewrites on vendored upstream source.

### Requirements

- Original project code MUST follow an automated formatting and linting policy.
- Supported public interfaces SHOULD have type annotations and MUST document their contracts and failure behavior.
- Static analysis MUST run on original project code; exclusions for vendored source MUST be explicit.
- Dead code, generated code, exploratory scripts, and abandoned compatibility shims MUST NOT remain on supported execution paths without a documented reason.
- Broad exception handling, dynamic evaluation, unsafe casts, and warning suppression MUST be minimized and justified where retained.
- Functions and modules SHOULD have focused responsibilities, and excessive complexity SHOULD trigger refactoring or an approved exception.
- Duplicated preprocessing, configuration, and artifact logic SHOULD be consolidated behind tested interfaces.
- Compatibility workarounds MUST identify the dependency and version range that require them, together with a removal condition.
- Code review MUST distinguish changes to project code, adapter code, tests, generated artifacts, and vendored upstream source.
- Formatting or cleanup changes to vendored upstream source MUST NOT be mixed with semantic patches.

### Required evidence

- formatter, linter, and static-analysis configuration;
- clean automated check reports;
- documented vendored-source exclusions;
- complexity or maintainability checks for critical modules; and
- code-review and compatibility-workaround records.

## 14. Reliability and Failure Semantics

### Objective

Make failures diagnosable and prevent incomplete or invalid runs from appearing successful.

### Requirements

- Every action MUST perform preflight validation for required files, packages, devices, permissions, and configuration.
- External processes MUST expose their command, working directory, exit status, captured logs, and elapsed time.
- Long-running actions SHOULD support configurable timeouts, interruption, checkpointing, and resume where the upstream method permits.
- Artifact writes SHOULD be atomic or use an explicit incomplete state until finalized.
- A completed artifact bundle MUST reference files that exist and pass integrity checks.
- Exceptions MUST retain actionable context; broad exception handling MUST NOT hide invalid metrics or failed training.
- Sampling loops MUST have finite termination conditions and report shortfalls explicitly.
- Temporary files and redirected environment settings MUST be cleaned up after success, failure, or interruption.
- Re-running a completed action SHOULD be idempotent or require an explicit overwrite policy.

### Required evidence

- negative-path tests;
- interruption and resume tests where supported;
- artifact-integrity tests;
- structured logs; and
- documented failure codes or categories.

## 15. Efficiency and Scalability

### Objective

Measure and control computational, memory, storage, and operational costs without misrepresenting intrinsically expensive research methods.

### Requirements

- Benchmark runs MUST record wall-clock time and SHOULD record CPU time, peak RAM, peak VRAM, device utilization, sample throughput, and artifact size.
- Performance comparisons MUST use documented hardware and compatible resource settings.
- Efficiency targets SHOULD be defined per model class and dataset scale rather than imposing one arbitrary limit on every method.
- Repeated preprocessing and downloads SHOULD be safely cached using versioned, content-addressed inputs.
- The repository MUST NOT track routine runtime outputs, caches, full checkpoints, or materialized datasets unless they are intentional, licensed, documented release fixtures.
- Sampling and evaluation MUST avoid unbounded loops and unnecessary full-table copies where practical.
- Performance regressions in supported workflows SHOULD be detected against recorded baselines.

### Required evidence

- benchmark timing and resource reports;
- scaling tests on representative dataset sizes;
- storage inventory; and
- regression thresholds for release-supported workflows.

## 16. Testing and Verification

### Objective

Provide evidence for interface correctness, scientific fidelity, robustness, and release compatibility.

### Required test layers

1. **Unit tests** for deterministic utilities, validation, schemas, preprocessing, and metrics.
2. **Contract tests** for every adapter and artifact interface.
3. **Negative tests** for invalid configuration, missing dependencies, malformed samples, unsafe paths, and failed upstream processes.
4. **Smoke tests** that execute real model code on small fixtures in Linux and Python 3.11.
5. **Native parity tests** for official and approved-patched implementations.
6. **Evaluation tests** using reference datasets and known metric behavior.
7. **Reproducibility tests** across repeated runs and supported deterministic modes.
8. **Security tests** for secret handling, unsafe archives, untrusted serialization boundaries, and dependency risks.
9. **Performance tests** for supported representative workflows.
10. **Release tests** that install the built package in a clean environment and run the documented quickstart.

Mock-only tests MAY verify orchestration details but MUST NOT be counted as evidence that an algorithm runs, reproduces upstream, or qualifies for the official track.

CI MUST clearly separate fast core checks, model-specific checks, CPU integration tests, and GPU tests. A skipped test MUST report a reason and MUST NOT silently count as validation evidence.

## 17. Environment and Portability

### Objective

Provide a reproducible primary environment while isolating model families with conflicting dependencies.

### Requirements

- Linux and Python 3.11 MUST be continuously tested as the primary release environment.
- The supported Linux distribution, architecture, PyTorch version, CUDA version, and GPU families MUST be stated for each release.
- The core package MUST have a minimal dependency set.
- Model families with incompatible stacks SHOULD use optional extras, lock profiles, or isolated environments.
- Dependency resolution MUST be reproducible from clean machines.
- CPU support and GPU requirements MUST be stated per model.
- Platform-specific behavior MUST be isolated and tested; path manipulation MUST use portable APIs.
- Windows limitations MUST be documented but are non-blocking by default.

### Required evidence

- clean-environment installation logs;
- CI environment matrix;
- dependency locks;
- `pip check` or equivalent verification; and
- per-model compatibility records.

## 18. Usability and Documentation

### Objective

Allow a new researcher to install the project, run a supported example, interpret results, and diagnose common failures without private knowledge.

### Requirements

- The README MUST state project scope, support level, installation, quickstart, result tracks, and limitations accurately.
- Documentation MUST distinguish validation levels, benchmark tracks, and support levels, and MUST use their exact defined values.
- Every release-supported model MUST have an example configuration and expected artifact description.
- CLI help and metadata commands MUST work in the minimal core environment.
- Configuration options MUST document types, defaults, valid ranges, precedence, and scientific implications.
- Error messages MUST identify the failed requirement and a corrective action where one is known.
- The repository MUST document architecture, dataset onboarding, evaluation protocols, upstream provenance, patch policy, reproducibility, security, and contributing.
- Scientific limitations and known deviations MUST be prominent, not confined to internal notes.
- Public documentation MUST use portable relative links.

### Required evidence

- a clean-user quickstart test;
- link and example validation;
- CLI documentation tests; and
- per-model support and limitation pages.

## 19. Legal, Licensing, and Attribution

### Objective

Ensure that the project may be distributed and that all contributors and upstream works receive accurate credit.

### Requirements

- The repository MUST have a root license covering original project code and documentation.
- Third-party source MUST retain its original license, copyright notices, attribution, and required NOTICE files.
- The root license MUST NOT be presented as relicensing third-party components when that is not permitted.
- A third-party inventory MUST map each vendored source, dependency, dataset, model weight, and substantial asset to its origin and terms.
- Dataset and pretrained-weight redistribution rights MUST be reviewed separately from source-code licenses.
- Required citations MUST be documented in human- and machine-readable forms.
- Git history and contributor records MUST be preserved during repository transfer when technically possible.
- `AUTHORS` or `CONTRIBUTORS` documentation MUST recognize Jimmy's initial architecture, model integration, and code implementation as foundational contributions to the project.
- Later maintainers and contributors MUST be credited according to a documented policy.

### Required evidence

- root license;
- third-party and asset inventory;
- retained upstream licenses and notices;
- dataset and model-weight review;
- citation metadata; and
- contributor acknowledgements.

## 20. Packaging and Release Engineering

### Objective

Produce installable, traceable, testable, and maintainable public releases.

### Requirements

- The Python project MUST use standard package metadata and declare Python 3.11 support.
- The core package and optional model dependencies MUST be installable through documented commands.
- Releases MUST use a documented versioning policy and maintain a changelog.
- A release MUST be built and tested from a clean checkout.
- Source distributions and wheels MUST exclude private data, local paths, runtime artifacts, caches, and unintended large binaries.
- Release artifacts MUST record their source commit and dependency lock.
- Git tags and release notes MUST identify supported models, protocol versions, known limitations, and breaking changes.
- A release checklist MUST verify all repository-wide and per-model gates.
- Containers MAY be provided as convenience artifacts, but they MUST be built from the same locked and reviewed inputs.

### Required evidence

- package metadata;
- clean build and install logs;
- artifact-content inspection;
- changelog and release notes;
- versioned dependency locks; and
- completed release checklist.

## 21. Governance and Maintenance

### Objective

Make ownership, review authority, compatibility decisions, and long-term maintenance explicit.

### Requirements

- Repository, evaluation protocol, security, data, and model-adapter ownership MUST be assigned.
- Changes to upstream source, algorithm semantics, preprocessing, metric definitions, leaderboard rules, or protocol versions MUST receive designated review.
- Contributor and pull-request guidance MUST define test, documentation, provenance, and attribution expectations.
- Supported interfaces and models MUST follow a deprecation policy.
- Model status MUST be reviewed when upstream code, dependencies, licenses, or required external services change.
- Release-supported models without an active maintenance owner MUST be downgraded or removed from the support matrix.
- Decisions that affect scientific interpretation MUST be recorded in durable decision documents.
- Repository transfer to a laboratory organization MUST preserve contribution history and acknowledgements.

### Required evidence

- ownership records;
- contribution and review policy;
- decision log;
- deprecation policy;
- periodic support-status review; and
- release approval record.

## 22. Release Gates

### 22.1 Release classes

Public availability is not equivalent to release readiness. The repository uses two release classes:

- `public-preview`: an explicitly prerelease version intended for inspection and early use; and
- `official-benchmark-release`: a versioned release, such as v1.0 or later, that publishes an approved protocol and may publish official-track results.

A public preview MUST NOT be described as an official benchmark release. A repository that is already publicly visible MAY remain in development, but its documentation MUST state its actual release class and limitations.

### 22.2 Per-model official-track gate

A model MUST NOT enter the official track unless all of the following are true:

- its source components, authority, distribution form, reproduction target, and modification status are verified;
- an `original-method` target uses `method-author` authority, or a `benchmark-snapshot` target identifies the exact benchmark-vendored revision;
- every upstream revision or official package version is immutable and recorded;
- applicable source, package, dataset, weight, and redistribution conditions are cleared;
- its modification status is `unmodified`, `adapter-only`, or approved `compatibility-patched`, and no semantic patch applies;
- every compatibility patch is approved, isolated, documented, tested, and included in parity evidence;
- native parity validation against the declared reproduction target passes;
- real smoke tests pass in the primary environment;
- dataset, configuration, artifacts, and metrics are reproducible and traceable;
- evaluation protocol checks pass;
- no failed mandatory scientific, legal, security, or data-governance requirement remains without an approved exception; and
- an eligibility record assigns the `official` track and links the evidence, reviewers, decision date, repository commit, and protocol version.

### 22.3 Per-model release-support gate

A model MUST NOT be called `release-supported` unless it has:

- a maintenance owner;
- documented installation and usage;
- a tested example;
- smoke validation on the primary platform through model-specific CI or a documented scheduled process;
- a compatibility record;
- troubleshooting and limitations documentation; and
- a deprecation path.

Release support is independent of benchmark track. The public support matrix MUST show both properties and MUST NOT imply that a release-supported experimental model is official.

### 22.4 Public-preview gate

The repository MUST NOT publish a version labeled `public-preview` unless:

- core installation, metadata commands, and CLI help pass on Linux and Python 3.11;
- the README identifies the release as a preview and does not make unsupported official-result claims;
- every model is assigned a validation level, benchmark track, and support level consistent with recorded evidence;
- official, experimental, and excluded tracks are clearly separated;
- root licensing, third-party attribution, contributor acknowledgement, and the rights for every distributed dataset, weight, and asset are reviewed;
- secrets, unsafe artifacts, absolute developer paths, caches, and unintended generated outputs are absent from release artifacts;
- relevant Git history has been reviewed and any exposed credential or impermissible material has been remediated;
- dependency locks and vulnerability review are current; and
- preview installation and quickstart tests pass from a clean checkout.

A preview MAY contain models below native parity validation, but they MUST remain experimental or excluded. It MUST NOT publish an official ranking unless every included result independently passes the official-track gate.

### 22.5 Official v1.0-or-later release gate

The repository MUST NOT publish an `official-benchmark-release` unless:

- all public-preview gates pass;
- an approved release plan declares a non-empty Core Model Set;
- every model in the Core Model Set is both `official` and `release-supported`;
- the official evaluation protocol is versioned and tested;
- official datasets, splits, metrics, tuning rules, resource regimes, and result schemas are immutable within the protocol version;
- official results have complete run manifests and pass comparison-compatibility checks;
- documentation and quickstart tests pass;
- CI protects required quality gates; and
- no failed mandatory requirement remains without a documented exception permitted by this standard; and
- the versioned release checklist is approved by the designated maintainers and scientific reviewers.

The Core Model Set MUST be selected and recorded before the v1.0 release candidate. Changing that set after release-candidate evaluation begins requires a recorded release-plan revision and renewed approval.

## 23. Assessment and Evidence Policy

Every requirement assessment MUST use one of:

- `pass`: requirement and evidence are complete;
- `partial`: some requirements or evidence remain incomplete;
- `fail`: requirement is not met;
- `not-assessed`: no sufficient review has been completed; or
- `not-applicable`: the requirement does not apply, with a rationale.

`not-assessed` MUST NOT be interpreted as pass. Evidence SHOULD be machine-readable where practical and MUST identify the repository revision to which it applies.

Quality assessments SHOULD be rerun:

- before every public release;
- after material upstream or dependency updates;
- after evaluation protocol changes;
- after security or data-governance incidents; and
- when a model's official benchmark or release-support status changes.

## 24. Definition of Done

Work is complete only when the requested implementation and all applicable quality evidence are complete. Code presence, a successful mocked test, a single developer-machine run, or a manually produced output is not by itself a definition of done.

For an official-track adapter, done means `native-parity-validated` plus a passed official-track gate and an `official` benchmark-track decision. For a publicly supported adapter, done also requires `release-supported`. Validation, benchmark, and support properties MUST be reported separately and MUST NOT inherit stronger claims from repository documentation.
