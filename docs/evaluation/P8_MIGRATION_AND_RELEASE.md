# P8 Migration and Release

Status: implementation and exact native Windows 11/Python 3.11 exit gate passed at commit `bb09085`; hosted release evidence is pending.

## 1. Purpose

P8 makes one public path from adapter output to versioned evaluation evidence, isolates pre-P2 summaries, and defines the software release gate. It changes no metric formula, aggregation weight, dataset split, model algorithm, or scientific admission.

## 2. Legacy boundary

`standardized_summary.json` with `protocol_name: tabstruct-aligned-v1` is frozen at schema `1.0`. Supported generation is retired. During the `0.1.x` migration line, `import-legacy-summary` may copy an existing document verbatim into:

~~~text
legacy-import/
  checksums.sha256
  legacy_import_record.json
  source/standardized_summary.json
~~~

The importer rejects unknown fields, duplicate keys, non-finite values, symlinks, overwrite, tampering, and non-legacy protocol identities. The record always states `classification: legacy-diagnostic`, `status: not-converted`, `atomic_evidence_available: false`, and `official_results_allowed: false`. Support will not be removed before `0.2.0`; any extension requires a documented release decision.

## 3. Central evaluation route

Public adapter evaluation resolves the original decoded sample, real reference table, reviewed Dataset Profile, exact protocol, track, generation seed, and evaluator seeds. It writes a new `evaluation-result/` Run Result bundle. Adapter-local metric engines are not called. Train/sample behavior and safe adapter artifacts remain unchanged.

P6 now treats a finalized Result Bundle directory as the required evaluation-stage output. The operational aggregate/report stages continue to index execution evidence only; P7 remains the sole scientific aggregation/publication layer.

## 4. Release quickstart

The packaged quickstart uses declared artificial Apache-2.0 inputs and exact official `imbalanced-learn==0.14.2`. It produces a decoded SMOTE table, a finalized P3 bundle, and a `partial-diagnostic` snapshot. It performs no network acquisition and emits no Official rank.

## 5. Release assets

The root package includes Apache-2.0 licensing and scope notice, third-party inventory, citation metadata, contributor acknowledgement, DCO contribution terms, security policy, code of conduct, changelog, and release checklist. Public English documentation has Chinese review translations for the P8 guides.

## 6. Required validation

Before the final `0.1.0` tag:

- hosted Windows/Python 3.11 must pass clean build/install, core gates, legacy migration, table-only evaluation, quickstart, Result Bundle validation, and snapshot validation;
- Linux/Python 3.11 must pass the portable subset;
- native Windows 11/Python 3.11 must repeat the quickstart and complete release checklist;
- package contents, links, legal inventory, secrets, and restricted data must be re-audited at the release commit.

These gates establish software release readiness only. Model support, benchmark eligibility, dataset admission, metric admission, run admission, and Official publication remain independent.

The retained [native Windows 11/Python 3.11 evidence](../evidence/evaluation/p8-native-windows11-py311-bb09085.json) passed all eight P8 software exit gates and checksum-locks the implementation surfaces, including the canonical-LF packaged quickstart inputs. The Windows registry continues to expose the NT `10.0` compatibility version on Windows 11; the gate therefore binds workstation product type `WinNT` and build `26200`, rather than trusting `platform.release()`.
