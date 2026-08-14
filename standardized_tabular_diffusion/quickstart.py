"""Self-contained adapter-to-diagnostic-snapshot release quickstart."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.contracts import utc_timestamp
from standardized_tabular_diffusion.evaluation.leaderboard import build_snapshot_from_paths
from standardized_tabular_diffusion.evaluation.profiles import import_legacy_dataset_spec
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, atomic_write_json
from standardized_tabular_diffusion.evaluation.service import evaluate_adapter_output
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.registry import get_adapter

QUICKSTART_PACKAGE = "standardized_tabular_diffusion.resources.quickstart"


def _copy_resource(name: str, destination: Path) -> None:
    resource = resources.files(QUICKSTART_PACKAGE).joinpath(name)
    atomic_write_bytes(destination, resource.read_bytes())


def run_quickstart(output_dir: str | Path, *, seed: int = 17) -> dict[str, Any]:
    """Run official SMOTE, P3 validation, and one unranked diagnostic snapshot."""

    root = Path(output_dir)
    if root.exists():
        raise ValueError(f"Quickstart output must not already exist: {root}")
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    train_path = inputs / "train.csv"
    metadata_path = inputs / "metadata.json"
    _copy_resource("train.csv", train_path)
    _copy_resource("metadata.json", metadata_path)

    dataset_spec = DatasetSpec(
        name="quickstart-demo",
        task_type="classification",
        column_names=["age", "group", "score", "outcome"],
        numerical_columns=["age", "score"],
        categorical_columns=["group"],
        target_columns=["outcome"],
        metadata_path=metadata_path,
        train_data_path=train_path,
        provenance=["packaged-artificial-quickstart-v1"],
        extra={"source_rights": "Apache-2.0; artificial data"},
    )
    profile = import_legacy_dataset_spec(dataset_spec)
    profile.payload["validity"] = {
        "contract_schema_version": "1.0.0",
        "status": "reviewed-diagnostic",
        "hard_column_rules": [
            {
                "rule_id": "quickstart-model-input-not-null",
                "rule_type": "not_null",
                "selector": {"nullable_model_input": False},
                "parameters": {},
                "evidence": {
                    "source_type": "recorded-human-review",
                    "reference": "P8 artificial quickstart contract: decoded model inputs may not be null.",
                },
                "severity": "hard",
                "version": "1.0.0",
            }
        ],
        "cross_column_constraints": [],
        "soft_diagnostics": [],
        "unresolved_reviews": [],
    }
    profile_path = inputs / "dataset-profile.json"
    atomic_write_json(profile_path, profile.payload)

    adapter_root = root / "adapter"
    adapter = get_adapter("smote", repo_root=Path.cwd())
    embedded = dataset_spec.to_dict()
    train_spec = RunSpec(
        model="smote",
        dataset=dataset_spec.name,
        output_dir=adapter_root,
        seed=seed,
        num_samples=24,
        extra={"dataset_spec": embedded, "k_neighbors": 3},
    )
    adapter.train(train_spec)
    sample_bundle = adapter.sample(train_spec)
    if sample_bundle.generated_sample_path is None:
        raise RuntimeError("SMOTE quickstart did not produce a decoded table")

    evaluation_root = root / "evaluation-result"
    outcome = evaluate_adapter_output(
        model_id="smote",
        generation_seed=seed,
        synthetic_path=sample_bundle.generated_sample_path,
        output_dir=evaluation_root,
        protocol_id="p3-validity",
        dataset_profile_path=profile_path,
        reference_path=train_path,
        real_test_path=None,
        comparison_track="native",
        evaluator_seeds=(0,),
        expected_rows=24,
    )
    snapshot_request = {
        "snapshot_request_schema_version": "1.0.0",
        "repository_release": "p8-quickstart-diagnostic",
        "published_at": utc_timestamp(),
        "publication_class": "partial-diagnostic",
        "comparison_track": "native",
        "protocol": outcome.request.protocol,
        "dataset_suite": {
            "suite_id": "quickstart-artificial",
            "suite_version": "1.0.0",
            "datasets": [outcome.request.dataset_profile],
        },
        "metric": {
            "metric_id": "std-tabular-column-validity",
            "metric_version": "1.0.0",
            "dimension": "validity",
            "direction": "maximize",
        },
        "expected_generation_seeds": [seed],
        "aggregation": {
            "aggregation_id": "atomic-contribution-hierarchy",
            "aggregation_version": "1.0.0",
            "run_rule": "sum-declared-atomic-contributions",
            "seed_rule": "equal-seed-mean",
            "dataset_rule": "equal-dataset-macro-mean",
        },
        "bootstrap": {
            "method": "hierarchical-percentile",
            "replicates": 100,
            "confidence_level": 0.95,
            "random_seed": 11,
        },
        "ties": {"method": "disabled", "equivalence_margin": 0.0},
        "compatibility": {"hardware_profile": "exact", "evaluator_profile": "exact"},
        "community_submission": None,
    }
    snapshot_request_path = root / "snapshot-request.json"
    atomic_write_json(snapshot_request_path, snapshot_request)
    snapshot_root = root / "diagnostic-snapshot"
    snapshot = build_snapshot_from_paths(
        request_path=snapshot_request_path,
        bundle_paths=[evaluation_root],
        output=snapshot_root,
    )
    return {
        "valid": True,
        "classification": "partial-diagnostic",
        "official_results_allowed": False,
        "model": "smote",
        "generation_seed": seed,
        "result_bundle": str(evaluation_root),
        "result_bundle_id": outcome.report.bundle_id,
        "diagnostic_snapshot": str(snapshot_root),
        "snapshot_id": snapshot["snapshot_id"],
    }
