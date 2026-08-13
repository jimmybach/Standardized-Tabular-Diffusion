"""Fail-closed P7 aggregation and immutable leaderboard snapshot publication."""

from __future__ import annotations

import csv
import html
import io
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle
from standardized_tabular_diffusion.evaluation.contracts import AtomicResult, MetricState
from standardized_tabular_diffusion.evaluation.registry import get_metric_record
from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    content_fingerprint,
    read_json,
    sha256_file,
)

SNAPSHOT_REQUEST_SCHEMA_VERSION = "1.0.0"
DATASET_SUMMARY_SCHEMA_VERSION = "1.0.0"
LEADERBOARD_SNAPSHOT_SCHEMA_VERSION = "1.0.0"
CORRECTION_SCHEMA_VERSION = "1.0.0"
SNAPSHOT_MANIFEST_SCHEMA_VERSION = "1.0.0"
_FAILURE_STATES = {
    MetricState.MATHEMATICALLY_UNDEFINED,
    MetricState.INSUFFICIENT_SUPPORT,
    MetricState.IMPLEMENTATION_FAILURE,
    MetricState.RESOURCE_FAILURE,
}


class LeaderboardError(ValueError):
    """Raised when results cannot be aggregated or published without overclaiming."""


@dataclass(frozen=True)
class RunObservation:
    bundle_id: str
    request_fingerprint: str
    manifest_sha256: str
    model_id: str
    dataset: dict[str, str]
    protocol: dict[str, str]
    comparison_track: str
    generation_seed: int
    metric: dict[str, str]
    dataset_version: str
    dataset_view: str
    split_id: str
    evaluator_profile: dict[str, str] | None
    hardware_profile: dict[str, str] | None
    environment_sha256: str
    atomic_results: tuple[AtomicResult, ...]
    state_counts: dict[str, int]
    score: float | None
    score_reason: str | None

    @property
    def slot(self) -> tuple[str, str, int]:
        return self.model_id, self.dataset["dataset_id"], self.generation_seed

    @property
    def compatibility_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "comparison_track": self.comparison_track,
            "metric": self.metric,
            "dataset_version": self.dataset_version,
            "dataset_view": self.dataset_view,
            "split_id": self.split_id,
            "evaluator_profile": self.evaluator_profile,
            "hardware_profile": self.hardware_profile,
            "environment_sha256": self.environment_sha256,
            "result_schema_version": sorted({item.result_schema_version for item in self.atomic_results}),
        }

    @property
    def compatibility_key(self) -> str:
        return content_fingerprint(self.compatibility_payload)

    @property
    def cross_dataset_compatibility_payload(self) -> dict[str, Any]:
        payload = dict(self.compatibility_payload)
        for field in ("dataset", "dataset_version", "dataset_view", "split_id"):
            payload.pop(field)
        return payload

    @property
    def cross_dataset_compatibility_key(self) -> str:
        return content_fingerprint(self.cross_dataset_compatibility_payload)

    def reference(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "request_fingerprint": self.request_fingerprint,
            "manifest_sha256": self.manifest_sha256,
            "model_id": self.model_id,
            "dataset_id": self.dataset["dataset_id"],
            "generation_seed": self.generation_seed,
            "compatibility_key": self.compatibility_key,
            "state_counts": self.state_counts,
            "score": self.score,
            "score_reason": self.score_reason,
        }


def validate_snapshot_request(payload: dict[str, Any]) -> dict[str, Any]:
    validate_instance("snapshot-request", payload)
    datasets = payload["dataset_suite"]["datasets"]
    identities = [item["dataset_id"] for item in datasets]
    if len(set(identities)) != len(identities):
        raise LeaderboardError("Snapshot dataset suite contains duplicate dataset_id values")
    seeds = payload["expected_generation_seeds"]
    if seeds != sorted(seeds):
        raise LeaderboardError("expected_generation_seeds must be stored in ascending order")
    if payload["publication_class"] == "community" and payload["community_submission"] is None:
        raise LeaderboardError("Community publication requires submitter, repository, and revision provenance")
    if payload["publication_class"] != "community" and payload["community_submission"] is not None:
        raise LeaderboardError("community_submission is only valid for Community Results")
    if payload["publication_class"] == "official":
        if payload["ties"]["method"] == "disabled":
            raise LeaderboardError("Official publication requires an explicitly frozen tie procedure")
        if len(seeds) != 5:
            raise LeaderboardError("Official Results require exactly five declared generation seeds")
    return payload


def load_snapshot_request(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise LeaderboardError("Snapshot Request must be a JSON object")
    return validate_snapshot_request(payload)


def validate_admission_records(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    identity_fields = {
        "model": {"model_id"},
        "dataset": {"dataset_id", "dataset_profile_version", "sha256"},
        "metric": {"metric_id", "metric_version"},
        "protocol": {"protocol_id", "protocol_version", "sha256"},
        "comparison-track": {"comparison_track"},
        "run": {"bundle_id", "request_fingerprint", "manifest_sha256"},
        "repository-release": {"repository_release"},
        "dataset-suite": {"suite_id", "suite_version"},
    }
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_instance("admission-record", record)
        if set(record["subject_identity"]) != identity_fields[record["subject_type"]]:
            raise LeaderboardError(
                f"Admission {record['admission_id']} does not use the exact identity fields for {record['subject_type']}"
            )
        for evidence_ref in record["evidence_refs"]:
            _validate_evidence_ref(evidence_ref)
        admission_id = record["admission_id"]
        if admission_id in indexed:
            raise LeaderboardError(f"Duplicate admission_id: {admission_id}")
        indexed[admission_id] = record
    for record in indexed.values():
        unknown = sorted(set(record["supersedes"]) - set(indexed))
        if unknown:
            raise LeaderboardError(f"Admission {record['admission_id']} supersedes unknown records: {unknown}")
    active = {
        admission_id: record
        for admission_id, record in indexed.items()
        if not any(admission_id in candidate["supersedes"] for candidate in indexed.values())
    }
    identities: set[tuple[str, str]] = set()
    for record in active.values():
        key = (record["subject_type"], content_fingerprint(record["subject_identity"]))
        if key in identities:
            raise LeaderboardError("Multiple active admission decisions target the same exact subject identity")
        identities.add(key)
    return tuple(active[key] for key in sorted(active))


def validate_corrections(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_instance("correction-record", record)
        if record["action"] not in {"invalidate", "supersede"}:
            raise LeaderboardError("Correction action must be invalidate or supersede")
        replacement = record["replacement_bundle_id"]
        if (record["action"] == "supersede") != (isinstance(replacement, str) and bool(replacement)):
            raise LeaderboardError("Supersession requires exactly one replacement_bundle_id")
        if record["action"] == "invalidate" and replacement is not None:
            raise LeaderboardError("Invalidation cannot carry replacement_bundle_id")
        for evidence_ref in record["evidence_refs"]:
            _validate_evidence_ref(evidence_ref)
        correction_id = record["correction_id"]
        if correction_id in indexed:
            raise LeaderboardError(f"Duplicate correction_id: {correction_id}")
        indexed[correction_id] = record
    affected = [record["affected_bundle_id"] for record in indexed.values()]
    if len(set(affected)) != len(affected):
        raise LeaderboardError("A bundle cannot have multiple active correction records")
    return tuple(indexed[key] for key in sorted(indexed))


def _validate_evidence_ref(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise LeaderboardError("Evidence URLs must use credential-free HTTPS")
        return
    if value.startswith(("/", "\\")) or "\\" in value or ":" in value or ".." in Path(value).parts:
        raise LeaderboardError("Evidence references must be safe repository-relative paths or HTTPS URLs")


def _atomic_score(results: Sequence[AtomicResult]) -> tuple[float | None, str | None]:
    failures = [item for item in results if item.state in _FAILURE_STATES]
    if failures:
        return None, "mandatory-atomic-state-not-computed"
    contributors = [item for item in results if item.weight > 0]
    if not contributors:
        return None, "no-declared-aggregate-contributions"
    contributions: list[float] = []
    for item in contributors:
        if item.state is not MetricState.COMPUTED or item.aggregate_contribution is None:
            return None, "declared-contribution-not-computed"
        contributions.append(item.aggregate_contribution)
    total_weight = math.fsum(item.weight for item in contributors)
    if not math.isclose(total_weight, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return None, "incomplete-aggregation-weight"
    score = math.fsum(contributions)
    if not math.isfinite(score):
        return None, "non-finite-run-aggregate"
    return score, None


def load_run_observation(bundle: str | Path, request: dict[str, Any]) -> RunObservation:
    bundle_root = Path(bundle)
    report = validate_result_bundle(bundle_root)
    if report.finalization_status != "finalized":
        raise LeaderboardError(f"Leaderboard inputs must be finalized Run Result bundles: {bundle_root}")
    manifest = read_json(bundle_root / "manifest.json")
    config = read_json(bundle_root / "config.yaml")
    metadata = read_json(bundle_root / "metadata.json")
    if not all(isinstance(item, dict) for item in (manifest, config, metadata)):
        raise LeaderboardError(f"Run Result bundle has invalid structured records: {bundle_root}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise LeaderboardError("P7 Run Result loading requires the leaderboard extra with pyarrow") from exc
    try:
        atomic = tuple(AtomicResult.from_dict(item) for item in parquet.read_table(bundle_root / "metrics.parquet").to_pylist())
    except (OSError, TypeError, ValueError) as exc:
        raise LeaderboardError(f"Cannot read Atomic Results from {bundle_root}: {exc}") from exc

    expected_metric = request["metric"]
    selected = tuple(
        item
        for item in atomic
        if (item.metric_id, item.metric_version)
        == (expected_metric["metric_id"], expected_metric["metric_version"])
    )
    if not selected:
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} has no Atomic Results for the requested metric")
    if any(item.dimension != expected_metric["dimension"] for item in selected):
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} uses an incompatible metric dimension")
    directions = {item.raw_direction.value for item in selected if item.weight > 0}
    if directions and directions != {expected_metric["direction"]}:
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} uses an incompatible ranking direction")
    identities = {
        "protocol": config["protocol"],
        "comparison_track": config["comparison_track"],
        "dataset": config["dataset_profile"],
    }
    if identities["protocol"] != request["protocol"]:
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} is outside the requested protocol compatibility group")
    if identities["comparison_track"] != request["comparison_track"]:
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} is in a different comparison track")
    expected_datasets = {item["dataset_id"]: item for item in request["dataset_suite"]["datasets"]}
    dataset_id = identities["dataset"]["dataset_id"]
    if expected_datasets.get(dataset_id) != identities["dataset"]:
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} is outside the requested dataset suite identity")
    generation_seed = config["generation_seed"]
    if generation_seed not in request["expected_generation_seeds"]:
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} uses an undeclared generation seed")

    fields = {
        "dataset_version": {item.dataset_version for item in selected},
        "dataset_view": {item.dataset_view for item in selected},
        "split_id": {item.split_id for item in selected},
        "model_id": {item.model_id for item in selected},
    }
    if any(len(values) != 1 for values in fields.values()):
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} contains mixed Atomic Result identities")
    model_id = next(iter(fields["model_id"]))
    if model_id != (config.get("model") or {}).get("model_id", "external"):
        raise LeaderboardError(f"Bundle {manifest['bundle_id']} model identity differs across config and Atomic Results")
    score, score_reason = _atomic_score(selected)
    states = dict(sorted(Counter(item.state.value for item in selected).items()))
    return RunObservation(
        bundle_id=manifest["bundle_id"],
        request_fingerprint=manifest["identity"]["request_fingerprint"],
        manifest_sha256=sha256_file(bundle_root / "manifest.json"),
        model_id=model_id,
        dataset=dict(identities["dataset"]),
        protocol=dict(identities["protocol"]),
        comparison_track=identities["comparison_track"],
        generation_seed=generation_seed,
        metric={"metric_id": expected_metric["metric_id"], "metric_version": expected_metric["metric_version"]},
        dataset_version=next(iter(fields["dataset_version"])),
        dataset_view=next(iter(fields["dataset_view"])),
        split_id=next(iter(fields["split_id"])),
        evaluator_profile=metadata["evaluator"].get("profile"),
        hardware_profile=metadata["evaluator"].get("hardware_profile"),
        environment_sha256=sha256_file(bundle_root / "environment.json"),
        atomic_results=selected,
        state_counts=states,
        score=score,
        score_reason=score_reason,
    )


def construct_compatibility_groups(observations: Iterable[RunObservation]) -> dict[str, tuple[RunObservation, ...]]:
    groups: dict[str, list[RunObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.compatibility_key].append(observation)
    return {
        key: tuple(sorted(items, key=lambda item: (item.model_id, item.generation_seed, item.bundle_id)))
        for key, items in sorted(groups.items())
    }


def _apply_corrections(
    observations: Sequence[RunObservation], corrections: Sequence[dict[str, Any]]
) -> tuple[tuple[RunObservation, ...], dict[str, dict[str, Any]]]:
    by_id = {item.bundle_id: item for item in observations}
    if len(by_id) != len(observations):
        raise LeaderboardError("Input bundle_id values must be globally unique")
    rejected: dict[str, dict[str, Any]] = {}
    for correction in corrections:
        affected_id = correction["affected_bundle_id"]
        if affected_id not in by_id:
            raise LeaderboardError(f"Correction references unknown affected bundle: {affected_id}")
        affected = by_id[affected_id]
        replacement_id = correction["replacement_bundle_id"]
        if replacement_id is not None:
            if replacement_id not in by_id:
                raise LeaderboardError(f"Correction references unknown replacement bundle: {replacement_id}")
            replacement = by_id[replacement_id]
            if affected.slot != replacement.slot:
                raise LeaderboardError("A supersession replacement must target the same model/dataset/seed slot")
            if affected.compatibility_payload != replacement.compatibility_payload:
                raise LeaderboardError("A supersession cannot cross a scientific compatibility boundary")
        rejected[affected_id] = {
            "bundle_id": affected_id,
            "model_id": affected.model_id,
            "dataset_id": affected.dataset["dataset_id"],
            "generation_seed": affected.generation_seed,
            "reason_code": f"correction-{correction['action']}",
            "correction_id": correction["correction_id"],
            "replacement_bundle_id": replacement_id,
        }
    return tuple(item for item in observations if item.bundle_id not in rejected), rejected


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise LeaderboardError("Cannot calculate a percentile from an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def _seed_interval(values: Sequence[float], config: dict[str, Any], salt: str) -> dict[str, Any]:
    if not values:
        return {
            "method": config["method"],
            "confidence_level": config["confidence_level"],
            "lower": None,
            "upper": None,
            "replicates": config["replicates"],
            "random_seed": config["random_seed"],
        }
    rng = random.Random(config["random_seed"] ^ int(content_fingerprint(salt)[:16], 16))
    replicates = [math.fsum(rng.choice(values) for _ in values) / len(values) for _ in range(config["replicates"])]
    replicates.sort()
    alpha = (1.0 - config["confidence_level"]) / 2.0
    return {
        "method": config["method"],
        "confidence_level": config["confidence_level"],
        "lower": _percentile(replicates, alpha),
        "upper": _percentile(replicates, 1.0 - alpha),
        "replicates": config["replicates"],
        "random_seed": config["random_seed"],
    }


def _hierarchical_interval(
    dataset_values: Sequence[Sequence[float]], config: dict[str, Any], salt: str
) -> dict[str, Any]:
    if not dataset_values:
        return {
            "method": config["method"],
            "confidence_level": config["confidence_level"],
            "lower": None,
            "upper": None,
            "replicates": config["replicates"],
            "random_seed": config["random_seed"],
        }
    rng = random.Random(config["random_seed"] ^ int(content_fingerprint(salt)[:16], 16))
    replicates: list[float] = []
    for _ in range(config["replicates"]):
        sampled_datasets = [rng.choice(dataset_values) for _ in dataset_values]
        dataset_means = [math.fsum(rng.choice(values) for _ in values) / len(values) for values in sampled_datasets]
        replicates.append(math.fsum(dataset_means) / len(dataset_means))
    replicates.sort()
    alpha = (1.0 - config["confidence_level"]) / 2.0
    return {
        "method": config["method"],
        "confidence_level": config["confidence_level"],
        "lower": _percentile(replicates, alpha),
        "upper": _percentile(replicates, 1.0 - alpha),
        "replicates": config["replicates"],
        "random_seed": config["random_seed"],
    }


def _approved(
    admissions: Sequence[dict[str, Any]], subject_type: str, identity: dict[str, Any]
) -> tuple[bool, str | None]:
    matches = [
        item
        for item in admissions
        if item["subject_type"] == subject_type and item["subject_identity"] == identity
    ]
    if not matches:
        return False, f"missing-{subject_type}-admission"
    record = matches[0]
    if record["decision"] != "approved" or "official" not in record["publication_classes"]:
        return False, f"{subject_type}-not-approved-for-official"
    return True, None


def _official_admission(
    *,
    request: dict[str, Any],
    model_id: str,
    dataset: dict[str, str],
    observations: Sequence[RunObservation],
    coverage_complete: bool,
    admissions: Sequence[dict[str, Any]],
    metric_registry_official: bool,
) -> dict[str, Any]:
    checks: list[tuple[str, dict[str, Any]]] = [
        ("model", {"model_id": model_id}),
        ("dataset", dataset),
        ("metric", {key: request["metric"][key] for key in ("metric_id", "metric_version")}),
        ("protocol", request["protocol"]),
        ("comparison-track", {"comparison_track": request["comparison_track"]}),
        ("repository-release", {"repository_release": request["repository_release"]}),
        (
            "dataset-suite",
            {
                "suite_id": request["dataset_suite"]["suite_id"],
                "suite_version": request["dataset_suite"]["suite_version"],
            },
        ),
    ]
    reasons: list[str] = []
    passed: dict[str, bool] = {}
    for subject_type, identity in checks:
        approved, reason = _approved(admissions, subject_type, identity)
        passed[subject_type] = approved
        if reason:
            reasons.append(reason)
    for observation in observations:
        identity = {
            "bundle_id": observation.bundle_id,
            "request_fingerprint": observation.request_fingerprint,
            "manifest_sha256": observation.manifest_sha256,
        }
        approved, reason = _approved(admissions, "run", identity)
        passed[f"run:{observation.bundle_id}"] = approved
        if reason:
            reasons.append(reason)
    passed["metric-registry"] = metric_registry_official
    if not metric_registry_official:
        reasons.append("metric-registry-not-release-supported")
    passed["complete-coverage"] = coverage_complete
    if not coverage_complete:
        reasons.append("incomplete-dataset-seed-coverage")
    return {"eligible": all(passed.values()), "checks": passed, "reason_codes": sorted(set(reasons))}


def _dataset_summary(
    *,
    request: dict[str, Any],
    model_id: str,
    dataset: dict[str, str],
    active: Sequence[RunObservation],
    rejected_by_correction: dict[str, dict[str, Any]],
    admissions: Sequence[dict[str, Any]],
    metric_registry_official: bool,
) -> dict[str, Any]:
    expected_seeds = request["expected_generation_seeds"]
    candidates = [item for item in active if item.model_id == model_id and item.dataset == dataset]
    slot_index: dict[int, RunObservation] = {}
    for item in candidates:
        if item.generation_seed in slot_index:
            raise LeaderboardError(
                f"Duplicate active Run Result attempts for {model_id}/{dataset['dataset_id']}/seed-{item.generation_seed}; "
                "an explicit correction record is required"
            )
        slot_index[item.generation_seed] = item
    seed_results: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    computed_values: list[float] = []
    for seed in expected_seeds:
        observation = slot_index.get(seed)
        if observation is None:
            seed_results.append({"generation_seed": seed, "state": "missing", "score": None, "bundle_id": None})
            state_counts["missing"] += 1
        elif observation.score is None:
            seed_results.append(
                {
                    "generation_seed": seed,
                    "state": "failed",
                    "score": None,
                    "bundle_id": observation.bundle_id,
                    "reason_code": observation.score_reason,
                }
            )
            state_counts.update(observation.state_counts)
        else:
            seed_results.append(
                {
                    "generation_seed": seed,
                    "state": "computed",
                    "score": observation.score,
                    "bundle_id": observation.bundle_id,
                }
            )
            computed_values.append(observation.score)
            state_counts.update(observation.state_counts)
    complete = len(computed_values) == len(expected_seeds)
    aggregate = math.fsum(computed_values) / len(computed_values) if computed_values else None
    compatibility_keys = {item.compatibility_key for item in candidates}
    if len(compatibility_keys) > 1:
        raise LeaderboardError(f"Incompatible results cannot be merged for {model_id}/{dataset['dataset_id']}")
    compatibility_key = next(iter(compatibility_keys), content_fingerprint({"dataset": dataset, "empty": True}))
    corrected = [
        record
        for _, record in sorted(rejected_by_correction.items())
        if record["model_id"] == model_id and record["dataset_id"] == dataset["dataset_id"]
    ]
    admission = _official_admission(
        request=request,
        model_id=model_id,
        dataset=dataset,
        observations=candidates,
        coverage_complete=complete,
        admissions=admissions,
        metric_registry_official=metric_registry_official,
    )
    summary_seed = {
        "model_id": model_id,
        "dataset": dataset,
        "metric": request["metric"],
        "comparison_track": request["comparison_track"],
        "expected_seeds": expected_seeds,
        "accepted_bundle_ids": [item.bundle_id for item in sorted(candidates, key=lambda item: item.generation_seed)],
    }
    payload = {
        "dataset_summary_schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
        "summary_id": f"dataset-summary-{content_fingerprint(summary_seed)[:20]}",
        "model_id": model_id,
        "dataset": dataset,
        "metric": request["metric"],
        "comparison_track": request["comparison_track"],
        "compatibility_key": compatibility_key,
        "expected_seeds": expected_seeds,
        "accepted_runs": [item.reference() for item in sorted(candidates, key=lambda item: item.generation_seed)],
        "rejected_runs": corrected,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "uncertainty": _seed_interval(computed_values, request["bootstrap"], content_fingerprint(summary_seed)),
        "coverage": {
            "expected_seed_count": len(expected_seeds),
            "computed_seed_count": len(computed_values),
            "failed_seed_count": sum(item["state"] == "failed" for item in seed_results),
            "missing_seed_count": sum(item["state"] == "missing" for item in seed_results),
            "fraction": len(computed_values) / len(expected_seeds),
            "complete": complete,
        },
        "state_counts": dict(sorted(state_counts.items())),
        "official_admission": admission,
    }
    validate_instance("dataset-summary", payload)
    return payload


def _suite_entries(request: dict[str, Any], summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_model[summary["model_id"]].append(summary)
    expected_dataset_count = len(request["dataset_suite"]["datasets"])
    expected_cells = expected_dataset_count * len(request["expected_generation_seeds"])
    entries: list[dict[str, Any]] = []
    for model_id, items in sorted(by_model.items()):
        available = [item for item in items if item["aggregate"] is not None]
        score = math.fsum(float(item["aggregate"]) for item in available) / len(available) if available else None
        dataset_seed_values = [
            [float(seed["score"]) for seed in item["seed_results"] if seed["score"] is not None]
            for item in items
            if any(seed["score"] is not None for seed in item["seed_results"])
        ]
        computed_cells = sum(item["coverage"]["computed_seed_count"] for item in items)
        official = len(items) == expected_dataset_count and all(item["official_admission"]["eligible"] for item in items)
        entries.append(
            {
                "model_id": model_id,
                "score": score,
                "uncertainty": _hierarchical_interval(
                    dataset_seed_values, request["bootstrap"], f"suite:{model_id}:{request['repository_release']}"
                ),
                "dataset_coverage": {
                    "expected": expected_dataset_count,
                    "computed": len(available),
                    "fraction": len(available) / expected_dataset_count,
                },
                "seed_coverage": {
                    "expected": expected_cells,
                    "computed": computed_cells,
                    "fraction": computed_cells / expected_cells,
                },
                "official_eligible": official,
                "official_reason_codes": sorted(
                    {
                        reason
                        for item in items
                        for reason in item["official_admission"]["reason_codes"]
                    }
                ),
                "rank": None,
                "diagnostic_order": None,
                "tie_group": None,
            }
        )
    reverse = request["metric"]["direction"] == "maximize"
    sortable = [item for item in entries if item["score"] is not None]
    sortable.sort(key=lambda item: ((-float(item["score"])) if reverse else float(item["score"]), item["model_id"]))
    for index, item in enumerate(sortable, start=1):
        item["diagnostic_order"] = index
        if request["publication_class"] == "official":
            if not item["official_eligible"]:
                raise LeaderboardError(f"Model {item['model_id']} cannot enter Official Results: admission gates failed")
    _assign_ties(sortable, request["ties"])
    if request["publication_class"] == "official":
        first_rank_by_tie: dict[str, int] = {}
        for index, item in enumerate(sortable, start=1):
            tie_group = item["tie_group"]
            item["rank"] = first_rank_by_tie.setdefault(tie_group, index)
    return entries


def _assign_ties(entries: Sequence[dict[str, Any]], tie_config: dict[str, Any]) -> None:
    if tie_config["method"] == "disabled":
        for index, item in enumerate(entries, start=1):
            item["tie_group"] = f"not-asserted-{index}"
        return
    group_index = 0
    previous: dict[str, Any] | None = None
    for item in entries:
        interval = item["uncertainty"]
        tied = False
        if previous is not None:
            previous_interval = previous["uncertainty"]
            overlap = (
                interval["lower"] is not None
                and previous_interval["lower"] is not None
                and max(interval["lower"], previous_interval["lower"])
                <= min(interval["upper"], previous_interval["upper"])
            )
            tied = abs(float(item["score"]) - float(previous["score"])) <= tie_config["equivalence_margin"] and overlap
        if not tied:
            group_index += 1
        item["tie_group"] = f"tie-{group_index}"
        previous = item


def _pairwise(request: dict[str, Any], summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for summary in summaries:
        dataset_id = summary["dataset"]["dataset_id"]
        for seed in summary["seed_results"]:
            if seed["score"] is not None:
                values[summary["model_id"]][(dataset_id, seed["generation_seed"])] = float(seed["score"])
    models = sorted({summary["model_id"] for summary in summaries})
    expected = len(request["dataset_suite"]["datasets"]) * len(request["expected_generation_seeds"])
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            shared = sorted(set(values[left]) & set(values[right]))
            differences = [values[left][key] - values[right][key] for key in shared]
            pairs.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "expected_paired_cells": expected,
                    "observed_paired_cells": len(shared),
                    "pairwise_completeness": len(shared) / expected,
                    "mean_raw_difference_a_minus_b": math.fsum(differences) / len(differences) if differences else None,
                    "superiority_claim": "not-issued",
                    "multiple_comparison_correction": "not-applicable-without-superiority-claims",
                }
            )
    return pairs


def _snapshot_input_fingerprint(snapshot: dict[str, Any]) -> str:
    return content_fingerprint(
        {
            "request": snapshot["request"],
            "input_bundles": snapshot["input_bundles"],
            "admissions": snapshot["admissions"],
            "corrections": snapshot["corrections"],
        }
    )


def _snapshot_content_fingerprint(snapshot: dict[str, Any]) -> str:
    return content_fingerprint(
        {
            key: value
            for key, value in snapshot.items()
            if key not in {"snapshot_id", "snapshot_fingerprint"}
        }
    )


def validate_leaderboard_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    validate_instance("leaderboard-snapshot", snapshot)
    request = validate_snapshot_request(snapshot["request"])
    validate_admission_records(snapshot["admissions"])
    validate_corrections(snapshot["corrections"])
    bound_fields = {
        "published_at": "published_at",
        "repository_release": "repository_release",
        "publication_class": "publication_class",
        "comparison_track": "comparison_track",
        "protocol": "protocol",
        "dataset_suite": "dataset_suite",
        "metric": "metric",
        "aggregation": "aggregation",
        "bootstrap": "bootstrap",
        "ties": "ties",
    }
    for snapshot_field, request_field in bound_fields.items():
        if snapshot[snapshot_field] != request[request_field]:
            raise LeaderboardError(f"Snapshot {snapshot_field} differs from its Snapshot Request")
    if snapshot["publication_assets"] != [
        "leaderboard.json",
        "leaderboard.csv",
        "leaderboard.html",
        "leaderboard.md",
    ]:
        raise LeaderboardError("Snapshot publication asset inventory is not the frozen P7 inventory")

    expected_datasets = {item["dataset_id"]: item for item in request["dataset_suite"]["datasets"]}
    summary_slots: set[tuple[str, str]] = set()
    summary_models: set[str] = set()
    for summary in snapshot["dataset_summaries"]:
        validate_instance("dataset-summary", summary)
        slot = (summary["model_id"], summary["dataset"]["dataset_id"])
        if slot in summary_slots:
            raise LeaderboardError(f"Snapshot contains duplicate dataset summary slot: {slot}")
        summary_slots.add(slot)
        summary_models.add(summary["model_id"])
        if expected_datasets.get(summary["dataset"]["dataset_id"]) != summary["dataset"]:
            raise LeaderboardError("Dataset Summary is outside the Snapshot Request suite")
        if summary["metric"] != request["metric"] or summary["comparison_track"] != request["comparison_track"]:
            raise LeaderboardError("Dataset Summary crosses the requested metric or comparison track")
        if summary["expected_seeds"] != request["expected_generation_seeds"]:
            raise LeaderboardError("Dataset Summary seed denominator differs from the Snapshot Request")
        states = [item["state"] for item in summary["seed_results"]]
        observed_seeds = [item["generation_seed"] for item in summary["seed_results"]]
        if observed_seeds != request["expected_generation_seeds"]:
            raise LeaderboardError("Dataset Summary seed results do not exactly cover the declared seed denominator")
        computed = states.count("computed")
        failed = states.count("failed")
        missing = states.count("missing")
        coverage = summary["coverage"]
        if coverage != {
            "expected_seed_count": len(states),
            "computed_seed_count": computed,
            "failed_seed_count": failed,
            "missing_seed_count": missing,
            "fraction": computed / len(states),
            "complete": computed == len(states),
        }:
            raise LeaderboardError("Dataset Summary coverage does not match its explicit seed denominator")

    expected_summary_slots = {
        (model_id, dataset_id) for model_id in summary_models for dataset_id in expected_datasets
    }
    if summary_slots != expected_summary_slots:
        raise LeaderboardError("Snapshot does not expose every model/dataset summary slot")
    leaderboard_models = [item["model_id"] for item in snapshot["leaderboard"]]
    if len(set(leaderboard_models)) != len(leaderboard_models) or set(leaderboard_models) != summary_models:
        raise LeaderboardError("Leaderboard model identities do not exactly match Dataset Summaries")
    ordered = sorted(
        (item for item in snapshot["leaderboard"] if item["diagnostic_order"] is not None),
        key=lambda item: item["diagnostic_order"],
    )
    if [item["diagnostic_order"] for item in ordered] != list(range(1, len(ordered) + 1)):
        raise LeaderboardError("Diagnostic ordering must be unique and contiguous")
    if request["publication_class"] != "official" and any(item["rank"] is not None for item in snapshot["leaderboard"]):
        raise LeaderboardError("Non-Official publication cannot emit ranks")
    if request["publication_class"] == "official" and any(item["rank"] is None for item in ordered):
        raise LeaderboardError("Official publication must rank every scored entry")

    expected_pairs = {
        (left, right)
        for index, left in enumerate(sorted(summary_models))
        for right in sorted(summary_models)[index + 1 :]
    }
    actual_pairs = {(item["model_a"], item["model_b"]) for item in snapshot["pairwise_completeness"]}
    if actual_pairs != expected_pairs:
        raise LeaderboardError("Pairwise completeness records do not cover every model pair exactly once")
    if snapshot["input_fingerprint"] != _snapshot_input_fingerprint(snapshot):
        raise LeaderboardError("Leaderboard Snapshot input fingerprint does not bind its declared inputs")
    if snapshot["snapshot_fingerprint"] != _snapshot_content_fingerprint(snapshot):
        raise LeaderboardError("Leaderboard Snapshot fingerprint does not bind its scientific content")
    return snapshot


def build_leaderboard_snapshot(
    request: dict[str, Any],
    observations: Sequence[RunObservation],
    *,
    admissions: Sequence[dict[str, Any]] = (),
    corrections: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    request = validate_snapshot_request(request)
    admissions = validate_admission_records(admissions)
    corrections = validate_corrections(corrections)
    if not observations:
        raise LeaderboardError("At least one finalized Run Result bundle is required")
    active, rejected = _apply_corrections(observations, corrections)
    if not active:
        raise LeaderboardError("No active Run Result bundle remains after applying corrections")
    cross_keys = {item.cross_dataset_compatibility_key for item in active}
    if len(cross_keys) != 1:
        raise LeaderboardError(
            "Active input bundles cross a protocol, metric, track, evaluator, hardware, software, or schema boundary"
        )
    slots: dict[tuple[str, str, int], list[RunObservation]] = defaultdict(list)
    for item in active:
        slots[item.slot].append(item)
    duplicates = [slot for slot, items in slots.items() if len(items) > 1]
    if duplicates:
        raise LeaderboardError(f"Duplicate active scientific slots require correction records: {duplicates}")

    metric_record = get_metric_record(request["metric"]["metric_id"], request["metric"]["metric_version"])
    metric_registry_official = bool(
        metric_record.payload["admission"]["official_results_allowed"]
        and metric_record.payload["lifecycle_status"] == "release-supported"
        and metric_record.payload["validation"]["release_decision"] == "approved"
    )
    models = sorted({item.model_id for item in active})
    summaries = [
        _dataset_summary(
            request=request,
            model_id=model_id,
            dataset=dataset,
            active=active,
            rejected_by_correction=rejected,
            admissions=admissions,
            metric_registry_official=metric_registry_official,
        )
        for model_id in models
        for dataset in request["dataset_suite"]["datasets"]
    ]
    entries = _suite_entries(request, summaries)
    if request["publication_class"] == "official" and any(not item["official_eligible"] for item in entries):
        raise LeaderboardError("Official publication is prohibited until every independent admission and coverage gate passes")
    input_refs = [item.reference() for item in sorted(observations, key=lambda item: item.bundle_id)]
    input_seed = {
        "request": request,
        "input_bundles": input_refs,
        "admissions": list(admissions),
        "corrections": list(corrections),
    }
    material = {
        "leaderboard_snapshot_schema_version": LEADERBOARD_SNAPSHOT_SCHEMA_VERSION,
        "input_fingerprint": content_fingerprint(input_seed),
        "request": request,
        "published_at": request["published_at"],
        "repository_release": request["repository_release"],
        "publication_class": request["publication_class"],
        "comparison_track": request["comparison_track"],
        "protocol": request["protocol"],
        "dataset_suite": request["dataset_suite"],
        "metric": request["metric"],
        "aggregation": request["aggregation"],
        "bootstrap": request["bootstrap"],
        "ties": request["ties"],
        "input_bundles": input_refs,
        "dataset_summaries": summaries,
        "leaderboard": entries,
        "pairwise_completeness": _pairwise(request, summaries),
        "corrections": list(corrections),
        "admissions": list(admissions),
        "publication_assets": ["leaderboard.json", "leaderboard.csv", "leaderboard.html", "leaderboard.md"],
        "claim_boundary": (
            "Ranks are emitted only for Official Results after all independent gates pass. Partial/Diagnostic and "
            "Community entries expose diagnostic_order, coverage, uncertainty, and failure denominators without an official rank."
        ),
    }
    fingerprint = _snapshot_content_fingerprint(material)
    payload = {
        "leaderboard_snapshot_schema_version": LEADERBOARD_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": f"leaderboard-{fingerprint[:20]}",
        "snapshot_fingerprint": fingerprint,
        **{key: value for key, value in material.items() if key != "leaderboard_snapshot_schema_version"},
    }
    return validate_leaderboard_snapshot(payload)


def _publication_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(
        snapshot["leaderboard"],
        key=lambda entry: (entry["diagnostic_order"] is None, entry["diagnostic_order"] or 0, entry["model_id"]),
    ):
        rows.append(
            {
                "rank": item["rank"],
                "diagnostic_order": item["diagnostic_order"],
                "tie_group": item["tie_group"],
                "model_id": item["model_id"],
                "score": item["score"],
                "ci_lower": item["uncertainty"]["lower"],
                "ci_upper": item["uncertainty"]["upper"],
                "dataset_coverage": item["dataset_coverage"]["fraction"],
                "seed_coverage": item["seed_coverage"]["fraction"],
                "official_eligible": item["official_eligible"],
            }
        )
    return rows


def render_csv(snapshot: dict[str, Any]) -> bytes:
    rows = _publication_rows(snapshot)
    output = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["rank", "diagnostic_order", "model_id", "score"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render_markdown(snapshot: dict[str, Any]) -> bytes:
    lines = [
        f"# Leaderboard Snapshot `{snapshot['snapshot_id']}`",
        "",
        f"Publication class: `{snapshot['publication_class']}`  ",
        f"Comparison track: `{snapshot['comparison_track']}`  ",
        f"Metric: `{snapshot['metric']['metric_id']}@{snapshot['metric']['metric_version']}`",
        "",
        "| Rank | Diagnostic order | Model | Score | Interval | Dataset coverage | Seed coverage |",
        "|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in _publication_rows(snapshot):
        score = "—" if row["score"] is None else format(float(row["score"]), ".10g")
        interval = (
            "—"
            if row["ci_lower"] is None
            else f"[{format(float(row['ci_lower']), '.10g')}, {format(float(row['ci_upper']), '.10g')}]"
        )
        lines.append(
            f"| {row['rank'] or '—'} | {row['diagnostic_order'] or '—'} | `{row['model_id']}` | {score} | "
            f"{interval} | {row['dataset_coverage']:.1%} | {row['seed_coverage']:.1%} |"
        )
    lines.extend(["", snapshot["claim_boundary"], ""])
    return "\n".join(lines).encode("utf-8")


def render_html(snapshot: dict[str, Any]) -> bytes:
    rows: list[str] = []
    for row in _publication_rows(snapshot):
        score = "—" if row["score"] is None else format(float(row["score"]), ".10g")
        interval = (
            "—"
            if row["ci_lower"] is None
            else f"[{format(float(row['ci_lower']), '.10g')}, {format(float(row['ci_upper']), '.10g')}]"
        )
        cells = (
            row["rank"] or "—",
            row["diagnostic_order"] or "—",
            row["model_id"],
            score,
            interval,
            f"{row['dataset_coverage']:.1%}",
            f"{row['seed_coverage']:.1%}",
        )
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>")
    metric = snapshot["metric"]
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Leaderboard Snapshot {html.escape(snapshot['snapshot_id'])}</title>
  <style>body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;text-align:right}}th:nth-child(3),td:nth-child(3){{text-align:left}}code{{word-break:break-all}}</style>
</head>
<body>
  <h1>Leaderboard Snapshot <code>{html.escape(snapshot['snapshot_id'])}</code></h1>
  <p>Publication class: <code>{html.escape(snapshot['publication_class'])}</code><br>
  Comparison track: <code>{html.escape(snapshot['comparison_track'])}</code><br>
  Metric: <code>{html.escape(metric['metric_id'])}@{html.escape(metric['metric_version'])}</code></p>
  <table>
    <thead><tr><th>Rank</th><th>Diagnostic order</th><th>Model</th><th>Score</th><th>Interval</th><th>Dataset coverage</th><th>Seed coverage</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>{html.escape(snapshot['claim_boundary'])}</p>
</body>
</html>
"""
    return document.encode("utf-8")


def write_snapshot_bundle(snapshot: dict[str, Any], output: str | Path) -> Path:
    validate_leaderboard_snapshot(snapshot)
    root = Path(output)
    if root.is_symlink():
        raise LeaderboardError(f"Refusing to publish through a symbolic-link snapshot directory: {root}")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise LeaderboardError(f"Refusing to overwrite non-empty snapshot directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    incomplete = {
        "snapshot_manifest_schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "finalization_status": "incomplete",
        "files": [],
        "checksum_algorithm": "sha256",
    }
    validate_instance("snapshot-manifest", incomplete)
    atomic_write_json(root / "manifest.json", incomplete)
    json_payload = canonical_json_bytes(snapshot) + b"\n"
    assets = {
        "leaderboard.json": json_payload,
        "leaderboard.csv": render_csv(snapshot),
        "leaderboard.html": render_html(snapshot),
        "leaderboard.md": render_markdown(snapshot),
    }
    for name, payload in assets.items():
        atomic_write_bytes(root / name, payload)
    files = [
        {"path": name, "sha256": sha256_file(root / name), "byte_size": (root / name).stat().st_size}
        for name in sorted(assets)
    ]
    checksums = "".join(f"{item['sha256']}  {item['path']}\n" for item in files).encode("utf-8")
    atomic_write_bytes(root / "checksums.sha256", checksums)
    manifest = {
        "snapshot_manifest_schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "finalization_status": "finalized",
        "files": files,
        "checksum_algorithm": "sha256",
    }
    validate_instance("snapshot-manifest", manifest)
    atomic_write_json(root / "manifest.json", manifest)
    validate_snapshot_bundle(root)
    return root


def validate_snapshot_bundle(root: str | Path) -> dict[str, Any]:
    bundle = Path(root)
    if bundle.is_symlink() or any(path.is_symlink() for path in bundle.rglob("*")):
        raise LeaderboardError("Leaderboard Snapshot cannot contain symbolic links")
    manifest = read_json(bundle / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("finalization_status") != "finalized":
        raise LeaderboardError("Leaderboard Snapshot is not finalized")
    validate_instance("snapshot-manifest", manifest)
    expected_files = {
        "leaderboard.json",
        "leaderboard.csv",
        "leaderboard.html",
        "leaderboard.md",
        "checksums.sha256",
        "manifest.json",
    }
    actual_files = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise LeaderboardError("Leaderboard Snapshot contains missing or unmanifested files")
    inventory = {item["path"]: item for item in manifest.get("files", [])}
    if set(inventory) != {"leaderboard.json", "leaderboard.csv", "leaderboard.html", "leaderboard.md"}:
        raise LeaderboardError("Snapshot manifest inventory is incomplete")
    for relative, item in inventory.items():
        path = bundle / relative
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["byte_size"]:
            raise LeaderboardError(f"Snapshot asset checksum mismatch: {relative}")
    expected_checksums = "".join(
        f"{inventory[name]['sha256']}  {name}\n" for name in sorted(inventory)
    ).encode("utf-8")
    if (bundle / "checksums.sha256").read_bytes() != expected_checksums:
        raise LeaderboardError("checksums.sha256 does not exactly match the snapshot inventory")
    snapshot = read_json(bundle / "leaderboard.json")
    if not isinstance(snapshot, dict):
        raise LeaderboardError("leaderboard.json must be an object")
    validate_leaderboard_snapshot(snapshot)
    if (
        snapshot["snapshot_id"] != manifest["snapshot_id"]
        or snapshot["snapshot_fingerprint"] != manifest["snapshot_fingerprint"]
    ):
        raise LeaderboardError("Snapshot identity differs between manifest and leaderboard.json")
    if render_csv(snapshot) != (bundle / "leaderboard.csv").read_bytes():
        raise LeaderboardError("CSV publication asset differs from the structured leaderboard records")
    if render_html(snapshot) != (bundle / "leaderboard.html").read_bytes():
        raise LeaderboardError("HTML publication asset differs from the structured leaderboard records")
    if render_markdown(snapshot) != (bundle / "leaderboard.md").read_bytes():
        raise LeaderboardError("Markdown publication asset differs from the structured leaderboard records")
    return snapshot


def build_snapshot_from_paths(
    *,
    request_path: str | Path,
    bundle_paths: Sequence[str | Path],
    output: str | Path,
    admission_paths: Sequence[str | Path] = (),
    correction_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    request = load_snapshot_request(request_path)
    admissions = [read_json(path) for path in admission_paths]
    corrections = [read_json(path) for path in correction_paths]
    if not all(isinstance(item, dict) for item in admissions + corrections):
        raise LeaderboardError("Admission and correction files must each contain one JSON object")
    observations = [load_run_observation(path, request) for path in bundle_paths]
    snapshot = build_leaderboard_snapshot(request, observations, admissions=admissions, corrections=corrections)
    write_snapshot_bundle(snapshot, output)
    return snapshot
