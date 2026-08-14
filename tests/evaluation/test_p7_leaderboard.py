from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from standardized_tabular_diffusion.evaluation import leaderboard as leaderboard_module
from standardized_tabular_diffusion.evaluation.contracts import AtomicResult, MetricState, RawDirection
from standardized_tabular_diffusion.evaluation.leaderboard import (
    LeaderboardError,
    RunObservation,
    build_leaderboard_snapshot,
    construct_compatibility_groups,
    validate_admission_records,
    validate_snapshot_bundle,
    validate_snapshot_request,
    write_snapshot_bundle,
)

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

SHA0 = "0" * 64
SHA1 = "1" * 64


def request(*, publication_class: str = "partial-diagnostic", direction: str = "maximize") -> dict:
    return {
        "snapshot_request_schema_version": "1.0.0",
        "repository_release": "p7-test-release",
        "published_at": "2026-08-13T00:00:00Z",
        "publication_class": publication_class,
        "comparison_track": "native",
        "protocol": {"protocol_id": "p4-utility", "protocol_version": "1.0.0", "sha256": SHA0},
        "dataset_suite": {
            "suite_id": "fixture-core",
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
            "direction": direction,
        },
        "expected_generation_seeds": [1, 2, 3],
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
        "ties": {"method": "disabled", "equivalence_margin": 0.0},
        "compatibility": {"hardware_profile": "exact", "evaluator_profile": "exact"},
        "community_submission": None,
    }


def observation(
    model: str,
    dataset_id: str,
    seed: int,
    score: float | None,
    *,
    bundle_suffix: str = "a",
    hardware_sha: str = SHA0,
    comparison_track: str = "native",
) -> RunObservation:
    dataset = {
        "dataset_id": dataset_id,
        "dataset_profile_version": "1.0.0",
        "sha256": SHA0 if dataset_id == "alpha" else SHA1,
    }
    state = MetricState.COMPUTED if score is not None else MetricState.RESOURCE_FAILURE
    atomic = AtomicResult(
        run_id=f"run-{model}-{dataset_id}-{seed}-{bundle_suffix}",
        protocol_version="1.0.0",
        dataset_id=dataset_id,
        dataset_version="1.0.0",
        dataset_view="canonical",
        split_id="official-split",
        model_id=model,
        comparison_track=comparison_track,
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
        reason_detail=None if score is not None else "Fixture resource failure.",
    )
    return RunObservation(
        bundle_id=atomic.run_id,
        request_fingerprint=(f"{seed:x}" * 64)[:64],
        manifest_sha256=(f"{seed + 3:x}" * 64)[:64],
        model_id=model,
        dataset=dataset,
        protocol={"protocol_id": "p4-utility", "protocol_version": "1.0.0", "sha256": SHA0},
        comparison_track=comparison_track,
        generation_seed=seed,
        metric={"metric_id": "std-local-utility-retention", "metric_version": "1.0.0"},
        dataset_version="1.0.0",
        dataset_view="canonical",
        split_id="official-split",
        evaluator_profile={"profile_id": "p4-evaluator", "profile_version": "1.0.0", "sha256": SHA0},
        hardware_profile={"profile_id": "fixture-host", "profile_version": "1.0.0", "sha256": hardware_sha},
        environment_sha256=SHA0,
        atomic_results=(atomic,),
        state_counts={state.value: 1},
        score=score,
        score_reason=None if score is not None else "mandatory-atomic-state-not-computed",
    )


def complete_panel() -> list[RunObservation]:
    values = {
        "model-a": {"alpha": [0.9, 0.8, 0.7], "beta": [0.6, 0.5, 0.4]},
        "model-b": {"alpha": [0.7, 0.7, 0.7], "beta": [0.7, 0.7, 0.7]},
    }
    return [
        observation(model, dataset, seed, score)
        for model, datasets in values.items()
        for dataset, scores in datasets.items()
        for seed, score in zip((1, 2, 3), scores, strict=True)
    ]


def test_p7_equal_seed_then_equal_dataset_aggregation_is_deterministic() -> None:
    first = build_leaderboard_snapshot(request(), complete_panel())
    second = build_leaderboard_snapshot(request(), complete_panel())

    assert first == second
    entries = {item["model_id"]: item for item in first["leaderboard"]}
    assert entries["model-a"]["score"] == pytest.approx(0.65)
    assert entries["model-b"]["score"] == pytest.approx(0.7)
    assert entries["model-b"]["diagnostic_order"] == 1
    assert entries["model-b"]["rank"] is None
    assert all(item["rank"] is None for item in first["leaderboard"])
    assert len(first["dataset_summaries"]) == 4
    assert first["snapshot_fingerprint"] == second["snapshot_fingerprint"]


def test_missing_and_failed_seeds_remain_in_denominators_and_pairwise_coverage() -> None:
    panel = complete_panel()
    panel = [
        item
        for item in panel
        if not (item.model_id == "model-b" and item.dataset["dataset_id"] == "beta" and item.generation_seed == 3)
    ]
    panel.append(observation("model-a", "alpha", 3, None, bundle_suffix="failed"))
    panel = [
        item
        for item in panel
        if not (
            item.model_id == "model-a"
            and item.dataset["dataset_id"] == "alpha"
            and item.generation_seed == 3
            and item.score is not None
        )
    ]
    snapshot = build_leaderboard_snapshot(request(), panel)
    summaries = {(item["model_id"], item["dataset"]["dataset_id"]): item for item in snapshot["dataset_summaries"]}

    failed = summaries[("model-a", "alpha")]
    missing = summaries[("model-b", "beta")]
    assert failed["coverage"] == {
        "expected_seed_count": 3,
        "computed_seed_count": 2,
        "failed_seed_count": 1,
        "missing_seed_count": 0,
        "fraction": 2 / 3,
        "complete": False,
    }
    assert failed["state_counts"]["resource_failure"] == 1
    assert missing["coverage"]["missing_seed_count"] == 1
    pair = snapshot["pairwise_completeness"][0]
    assert pair["expected_paired_cells"] == 6
    assert pair["observed_paired_cells"] == 4
    assert pair["pairwise_completeness"] == pytest.approx(4 / 6)
    assert pair["superiority_claim"] == "not-issued"


def test_fully_failed_model_remains_visible_with_zero_pairwise_completeness() -> None:
    panel = [
        observation("model-a", dataset, seed, 0.8)
        for dataset in ("alpha", "beta")
        for seed in (1, 2, 3)
    ] + [
        observation("model-b", dataset, seed, None)
        for dataset in ("alpha", "beta")
        for seed in (1, 2, 3)
    ]
    snapshot = build_leaderboard_snapshot(request(), panel)
    entries = {item["model_id"]: item for item in snapshot["leaderboard"]}
    assert entries["model-b"]["score"] is None
    assert entries["model-b"]["seed_coverage"]["fraction"] == 0.0
    assert snapshot["pairwise_completeness"][0]["observed_paired_cells"] == 0
    assert snapshot["pairwise_completeness"][0]["pairwise_completeness"] == 0.0


def test_incompatible_track_hardware_and_dataset_split_cannot_merge() -> None:
    base = complete_panel()[:2]
    incompatible_track = copy.copy(base[0])
    object.__setattr__(incompatible_track, "comparison_track", "standardized-tuning")
    object.__setattr__(incompatible_track, "bundle_id", f"{base[0].bundle_id}-other-track")
    assert len(construct_compatibility_groups([base[0], incompatible_track])) == 2
    with pytest.raises(LeaderboardError, match="protocol, metric, track, evaluator, hardware, software, or schema"):
        build_leaderboard_snapshot(request(), [base[0], incompatible_track])

    incompatible_hardware = observation("model-b", "alpha", 1, 0.5, hardware_sha=SHA1)
    with pytest.raises(LeaderboardError, match="hardware"):
        build_leaderboard_snapshot(request(), [base[0], incompatible_hardware])

    incompatible_split = copy.copy(base[1])
    object.__setattr__(incompatible_split, "split_id", "different-split")
    with pytest.raises(LeaderboardError, match="Incompatible results"):
        build_leaderboard_snapshot(request(), [base[0], incompatible_split])


def test_duplicate_attempt_requires_reviewed_supersession_and_preserves_history() -> None:
    original = observation("model-a", "alpha", 1, 0.4, bundle_suffix="old")
    replacement = observation("model-a", "alpha", 1, 0.8, bundle_suffix="new")
    with pytest.raises(LeaderboardError, match="correction records"):
        build_leaderboard_snapshot(request(), [original, replacement])

    correction = {
        "correction_schema_version": "1.0.0",
        "correction_id": "replace-bad-run",
        "action": "supersede",
        "affected_bundle_id": original.bundle_id,
        "replacement_bundle_id": replacement.bundle_id,
        "reason_code": "validated-rerun",
        "reason_detail": "The initial attempt used a corrupted sample artifact.",
        "reviewer": "fixture-reviewer",
        "reviewed_at": "2026-08-13T00:00:00Z",
        "evidence_refs": ["tests/evaluation/test_p7_leaderboard.py"],
    }
    snapshot = build_leaderboard_snapshot(request(), [original, replacement], corrections=[correction])
    summary = next(
        item
        for item in snapshot["dataset_summaries"]
        if item["model_id"] == "model-a" and item["dataset"]["dataset_id"] == "alpha"
    )
    assert summary["accepted_runs"][0]["bundle_id"] == replacement.bundle_id
    assert summary["rejected_runs"][0]["bundle_id"] == original.bundle_id
    assert summary["rejected_runs"][0]["replacement_bundle_id"] == replacement.bundle_id


def test_official_publication_fails_closed_without_independent_admissions() -> None:
    official = request(publication_class="official")
    official["expected_generation_seeds"] = [1, 2, 3, 4, 5]
    official["ties"] = {"method": "margin-and-interval-overlap", "equivalence_margin": 0.01}
    panel = [
        observation("model-a", dataset, seed, 0.8)
        for dataset in ("alpha", "beta")
        for seed in (1, 2, 3, 4, 5)
    ]
    with pytest.raises(LeaderboardError, match="cannot enter Official Results"):
        build_leaderboard_snapshot(official, panel)


def test_official_publication_requires_and_consumes_complete_exact_admissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official = request(publication_class="official")
    official["expected_generation_seeds"] = [1, 2, 3, 4, 5]
    official["ties"] = {"method": "margin-and-interval-overlap", "equivalence_margin": 0.01}
    panel = [
        observation(model, dataset, seed, 0.7)
        for model in ("model-a", "model-b")
        for dataset in ("alpha", "beta")
        for seed in (1, 2, 3, 4, 5)
    ]

    def admission(admission_id: str, subject_type: str, identity: dict) -> dict:
        return {
            "admission_schema_version": "1.0.0",
            "admission_id": admission_id,
            "subject_type": subject_type,
            "subject_identity": identity,
            "decision": "approved",
            "publication_classes": ["official"],
            "reviewer": "independent-fixture-reviewer",
            "reviewed_at": "2026-08-13T00:00:00Z",
            "evidence_refs": ["tests/evaluation/test_p7_leaderboard.py"],
            "supersedes": [],
        }

    admissions = [
        admission("release", "repository-release", {"repository_release": official["repository_release"]}),
        admission(
            "suite",
            "dataset-suite",
            {
                "suite_id": official["dataset_suite"]["suite_id"],
                "suite_version": official["dataset_suite"]["suite_version"],
            },
        ),
        admission("track", "comparison-track", {"comparison_track": "native"}),
        admission("protocol", "protocol", official["protocol"]),
        admission(
            "metric",
            "metric",
            {key: official["metric"][key] for key in ("metric_id", "metric_version")},
        ),
        *[
            admission(f"model-{model}", "model", {"model_id": model})
            for model in ("model-a", "model-b")
        ],
        *[
            admission(f"dataset-{item['dataset_id']}", "dataset", item)
            for item in official["dataset_suite"]["datasets"]
        ],
        *[
            admission(
                f"admit-{item.bundle_id}",
                "run",
                {
                    "bundle_id": item.bundle_id,
                    "request_fingerprint": item.request_fingerprint,
                    "manifest_sha256": item.manifest_sha256,
                },
            )
            for item in panel
        ],
    ]
    monkeypatch.setattr(
        leaderboard_module,
        "get_metric_record",
        lambda *args: SimpleNamespace(
            payload={
                "admission": {"official_results_allowed": True},
                "lifecycle_status": "release-supported",
                "validation": {"release_decision": "approved"},
            }
        ),
    )

    snapshot = build_leaderboard_snapshot(official, panel, admissions=admissions)
    assert all(item["official_eligible"] for item in snapshot["leaderboard"])
    assert {item["rank"] for item in snapshot["leaderboard"]} == {1}
    assert len({item["tie_group"] for item in snapshot["leaderboard"]}) == 1


def test_snapshot_assets_are_structured_source_derived_and_tamper_evident(tmp_path: Path) -> None:
    snapshot = build_leaderboard_snapshot(request(), complete_panel())
    first = write_snapshot_bundle(snapshot, tmp_path / "first")
    second = write_snapshot_bundle(snapshot, tmp_path / "second")

    assert (first / "leaderboard.json").read_bytes() == (second / "leaderboard.json").read_bytes()
    assert (first / "leaderboard.csv").read_bytes() == (second / "leaderboard.csv").read_bytes()
    assert (first / "leaderboard.html").read_bytes() == (second / "leaderboard.html").read_bytes()
    assert (first / "leaderboard.md").read_bytes() == (second / "leaderboard.md").read_bytes()
    assert validate_snapshot_bundle(first)["snapshot_id"] == snapshot["snapshot_id"]

    with pytest.raises(LeaderboardError, match="overwrite"):
        write_snapshot_bundle(snapshot, first)
    (first / "leaderboard.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(LeaderboardError, match="checksum mismatch"):
        validate_snapshot_bundle(first)


def test_invalidated_incompatible_input_is_auditable_but_not_merged() -> None:
    accepted = observation("model-a", "alpha", 1, 0.8)
    invalid = observation("model-b", "alpha", 1, 0.6, hardware_sha=SHA1)
    correction = {
        "correction_schema_version": "1.0.0",
        "correction_id": "invalidate-wrong-host",
        "action": "invalidate",
        "affected_bundle_id": invalid.bundle_id,
        "replacement_bundle_id": None,
        "reason_code": "incompatible-hardware",
        "reason_detail": "The result was produced outside the declared exact hardware profile.",
        "reviewer": "fixture-reviewer",
        "reviewed_at": "2026-08-13T00:00:00Z",
        "evidence_refs": ["tests/evaluation/test_p7_leaderboard.py"],
    }
    snapshot = build_leaderboard_snapshot(request(), [accepted, invalid], corrections=[correction])
    assert [item["model_id"] for item in snapshot["leaderboard"]] == ["model-a"]
    assert {item["bundle_id"] for item in snapshot["input_bundles"]} == {accepted.bundle_id, invalid.bundle_id}


def test_publication_contracts_reject_unreviewable_provenance() -> None:
    community = request(publication_class="community")
    with pytest.raises(LeaderboardError, match="Community publication"):
        validate_snapshot_request(community)

    admission = {
        "admission_schema_version": "1.0.0",
        "admission_id": "unsafe-admission",
        "subject_type": "model",
        "subject_identity": {"model_id": "model-a"},
        "decision": "approved",
        "publication_classes": ["official"],
        "reviewer": "fixture-reviewer",
        "reviewed_at": "2026-08-13T00:00:00Z",
        "evidence_refs": ["https://user:secret@example.test/evidence"],
        "supersedes": [],
    }
    with pytest.raises(LeaderboardError, match="credential-free HTTPS"):
        validate_admission_records([admission])


def test_snapshot_fingerprint_binds_derived_scientific_content(tmp_path: Path) -> None:
    snapshot = build_leaderboard_snapshot(request(), complete_panel())
    snapshot["leaderboard"][0]["score"] = 123.0
    with pytest.raises(LeaderboardError, match="scientific content"):
        write_snapshot_bundle(snapshot, tmp_path / "tampered")
