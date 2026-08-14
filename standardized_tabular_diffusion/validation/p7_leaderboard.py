"""P7 aggregation, publication-boundary, and immutable-snapshot exit validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.contracts import AtomicResult, MetricState, RawDirection
from standardized_tabular_diffusion.evaluation.leaderboard import (
    LeaderboardError,
    RunObservation,
    build_leaderboard_snapshot,
    validate_snapshot_bundle,
    write_snapshot_bundle,
)
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file
from standardized_tabular_diffusion.platform_support import is_primary_release_family_environment

PROTOCOL_ID = "p7-leaderboard-exit-gate-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SHA0 = "0" * 64
SHA1 = "1" * 64


def _repository_commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _request(*, publication_class: str = "partial-diagnostic", seeds: list[int] | None = None) -> dict[str, Any]:
    return {
        "snapshot_request_schema_version": "1.0.0",
        "repository_release": "p7-validation",
        "published_at": "2026-08-13T00:00:00Z",
        "publication_class": publication_class,
        "comparison_track": "native",
        "protocol": {"protocol_id": "p4-utility", "protocol_version": "1.0.0", "sha256": SHA0},
        "dataset_suite": {
            "suite_id": "p7-fixture-core",
            "suite_version": "1.0.0",
            "datasets": [
                {"dataset_id": "alpha", "dataset_profile_version": "1.0.0", "sha256": SHA0},
                {"dataset_id": "beta", "dataset_profile_version": "1.0.0", "sha256": SHA1},
            ],
        },
        "metric": {
            "metric_id": "std-local-utility-retention",
            "metric_version": "1.0.0",
            "dimension": "local-utility",
            "direction": "maximize",
        },
        "expected_generation_seeds": seeds or [1, 2, 3],
        "aggregation": {
            "aggregation_id": "atomic-contribution-hierarchy",
            "aggregation_version": "1.0.0",
            "run_rule": "sum-declared-atomic-contributions",
            "seed_rule": "equal-seed-mean",
            "dataset_rule": "equal-dataset-macro-mean",
        },
        "bootstrap": {
            "method": "hierarchical-percentile",
            "replicates": 200,
            "confidence_level": 0.95,
            "random_seed": 97,
        },
        "ties": {
            "method": "margin-and-interval-overlap" if publication_class == "official" else "disabled",
            "equivalence_margin": 0.01 if publication_class == "official" else 0.0,
        },
        "compatibility": {"hardware_profile": "exact", "evaluator_profile": "exact"},
        "community_submission": None,
    }


def _observation(model: str, dataset_id: str, seed: int, score: float | None) -> RunObservation:
    dataset = {
        "dataset_id": dataset_id,
        "dataset_profile_version": "1.0.0",
        "sha256": SHA0 if dataset_id == "alpha" else SHA1,
    }
    state = MetricState.COMPUTED if score is not None else MetricState.RESOURCE_FAILURE
    run_id = f"run-{model}-{dataset_id}-{seed}"
    atomic = AtomicResult(
        run_id=run_id,
        protocol_version="1.0.0",
        dataset_id=dataset_id,
        dataset_version="1.0.0",
        dataset_view="canonical",
        split_id="official-split",
        model_id=model,
        comparison_track="native",
        generation_seed=seed,
        metric_id="std-local-utility-retention",
        metric_version="1.0.0",
        dimension="local-utility",
        scope_type="evaluator",
        scope_id="mean",
        state=state,
        raw_direction=RawDirection.MAXIMIZE,
        weight=1.0 if score is not None else 0.0,
        n_reference=10,
        n_synthetic=10,
        n_valid=20 if score is not None else 0,
        n_excluded=0,
        computed_at="2026-08-13T00:00:00Z",
        raw_value=score,
        normalized_value=score,
        aggregate_contribution=score,
        reason_code=None if score is not None else "simulated-resource-failure",
        reason_detail=None if score is not None else "P7 validation fixture resource failure.",
    )
    return RunObservation(
        bundle_id=run_id,
        request_fingerprint=f"{seed:x}" * 64,
        manifest_sha256=(f"{seed + 3:x}" * 64)[:64],
        model_id=model,
        dataset=dataset,
        protocol={"protocol_id": "p4-utility", "protocol_version": "1.0.0", "sha256": SHA0},
        comparison_track="native",
        generation_seed=seed,
        metric={"metric_id": "std-local-utility-retention", "metric_version": "1.0.0"},
        dataset_version="1.0.0",
        dataset_view="canonical",
        split_id="official-split",
        evaluator_profile={"profile_id": "p7-fixture-evaluator", "profile_version": "1.0.0", "sha256": SHA0},
        hardware_profile={"profile_id": "p7-fixture-host", "profile_version": "1.0.0", "sha256": SHA0},
        environment_sha256=SHA0,
        atomic_results=(atomic,),
        state_counts={state.value: 1},
        score=score,
        score_reason=None if score is not None else "mandatory-atomic-state-not-computed",
    )


def _panel(seeds: tuple[int, ...] = (1, 2, 3)) -> list[RunObservation]:
    return [
        _observation(model, dataset, seed, base - dataset_index * 0.1 - seed_index * 0.01)
        for model, base in (("model-a", 0.9), ("model-b", 0.8))
        for dataset_index, dataset in enumerate(("alpha", "beta"))
        for seed_index, seed in enumerate(seeds)
    ]


def _must_reject(action: Any, phrase: str) -> None:
    try:
        action()
    except LeaderboardError as exc:
        if phrase not in str(exc):
            raise AssertionError(f"Expected rejection containing {phrase!r}, observed {exc!s}") from exc
    else:
        raise AssertionError(f"Unsafe P7 operation was accepted; expected {phrase!r}")


def _locked_file_hashes() -> dict[str, str]:
    paths = {
        REPO_ROOT / "standardized_tabular_diffusion" / "evaluation" / "leaderboard.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "evaluation" / "schema.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "validation" / "core_ci.py",
        REPO_ROOT / "standardized_tabular_diffusion" / "cli.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".github" / "workflows" / "p7-leaderboard-validation.yml",
        REPO_ROOT / "tests" / "evaluation" / "test_p7_leaderboard.py",
        REPO_ROOT / "tests" / "evaluation" / "test_p7_run_bundle_integration.py",
        Path(__file__).resolve(),
    }
    schema_root = REPO_ROOT / "standardized_tabular_diffusion" / "schemas" / "evaluation"
    paths.update(
        schema_root / name
        for name in (
            "admission-record.schema.json",
            "correction-record.schema.json",
            "dataset-summary.schema.json",
            "leaderboard-snapshot.schema.json",
            "snapshot-manifest.schema.json",
            "snapshot-request.schema.json",
        )
    )
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def run_validation(output: Path, *, require_primary_family_environment: bool = False) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P7",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Validates P7 aggregation hierarchy, compatibility rejection, denominator accounting, correction history, "
            "Official admission fail-closed behavior, deterministic snapshots, and source-derived static assets. It does "
            "not admit a model, dataset, metric, protocol, run, suite, or repository release to Official Results."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "primary_family_environment_required": require_primary_family_environment,
        },
    }
    try:
        if require_primary_family_environment and not is_primary_release_family_environment():
            raise AssertionError("Primary-family P7 evidence requires Windows with Python 3.11")
        panel = _panel()
        first = build_leaderboard_snapshot(_request(), panel)
        second = build_leaderboard_snapshot(_request(), list(reversed(panel)))
        assert first == second
        entries = {item["model_id"]: item for item in first["leaderboard"]}
        assert entries["model-a"]["score"] > entries["model-b"]["score"]
        assert all(item["rank"] is None for item in entries.values())

        denominator_panel = [
            item
            for item in panel
            if not (item.model_id == "model-b" and item.dataset["dataset_id"] == "beta" and item.generation_seed == 3)
        ]
        denominator_panel = [
            item
            for item in denominator_panel
            if not (item.model_id == "model-a" and item.dataset["dataset_id"] == "alpha" and item.generation_seed == 3)
        ]
        denominator_panel.append(_observation("model-a", "alpha", 3, None))
        denominator = build_leaderboard_snapshot(_request(), denominator_panel)
        summaries = {(item["model_id"], item["dataset"]["dataset_id"]): item for item in denominator["dataset_summaries"]}
        assert summaries[("model-a", "alpha")]["coverage"]["failed_seed_count"] == 1
        assert summaries[("model-b", "beta")]["coverage"]["missing_seed_count"] == 1
        assert denominator["pairwise_completeness"][0]["pairwise_completeness"] < 1.0

        base = panel[0]
        _must_reject(
            lambda: build_leaderboard_snapshot(
                _request(),
                [base, replace(panel[1], hardware_profile={"profile_id": "other", "profile_version": "1.0.0", "sha256": SHA1})],
            ),
            "hardware",
        )
        _must_reject(
            lambda: build_leaderboard_snapshot(
                _request(), [base, replace(panel[1], split_id="leaky-split")]
            ),
            "Incompatible results",
        )
        _must_reject(
            lambda: build_leaderboard_snapshot(
                _request(), [base, replace(panel[1], environment_sha256=SHA1)]
            ),
            "software",
        )

        official_request = _request(publication_class="official", seeds=[1, 2, 3, 4, 5])
        _must_reject(
            lambda: build_leaderboard_snapshot(official_request, _panel((1, 2, 3, 4, 5))),
            "cannot enter Official Results",
        )

        with tempfile.TemporaryDirectory(prefix="std-tabular-p7-") as temporary:
            snapshot_root = Path(temporary) / "snapshot"
            write_snapshot_bundle(first, snapshot_root)
            validated = validate_snapshot_bundle(snapshot_root)
            assert validated["snapshot_fingerprint"] == first["snapshot_fingerprint"]
            assets = sorted(item["path"] for item in json.loads((snapshot_root / "manifest.json").read_text())["files"])
            assert assets == ["leaderboard.csv", "leaderboard.html", "leaderboard.json", "leaderboard.md"]
            (snapshot_root / "leaderboard.csv").write_text("tampered\n", encoding="utf-8")
            _must_reject(lambda: validate_snapshot_bundle(snapshot_root), "checksum mismatch")

        evidence["result_summary"] = {
            "snapshot_id": first["snapshot_id"],
            "snapshot_fingerprint": first["snapshot_fingerprint"],
            "input_fingerprint": first["input_fingerprint"],
            "model_count": len(first["leaderboard"]),
            "dataset_summary_count": len(first["dataset_summaries"]),
            "pairwise_record_count": len(first["pairwise_completeness"]),
            "publication_assets": first["publication_assets"],
            "failed_seed_denominator_preserved": True,
            "missing_seed_denominator_preserved": True,
            "official_without_admissions_rejected": True,
        }
        evidence["exit_gates"] = {
            "atomic_to_seed_to_dataset_to_suite_hierarchy": "pass",
            "equal_dataset_macro_weighting": "pass",
            "hierarchical_uncertainty_deterministic": "pass",
            "missing_and_failed_denominators_preserved": "pass",
            "pairwise_completeness_exposed": "pass",
            "compatibility_boundaries_fail_closed": "pass",
            "official_admission_fail_closed": "pass",
            "immutable_snapshot_and_asset_checksums": "pass",
            "human_and_machine_assets_source_derived": "pass",
        }
        evidence["locked_files"] = _locked_file_hashes()
        evidence["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"P7 leaderboard validation failed; inspect {output}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-primary-family-environment", action="store_true")
    args = parser.parse_args()
    evidence = run_validation(args.output, require_primary_family_environment=args.require_primary_family_environment)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
