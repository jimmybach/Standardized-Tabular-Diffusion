from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import build_example_config, load_experiment_config, save_experiment_config
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes
from standardized_tabular_diffusion.materialization import materialization_status, materialize_dataset
from standardized_tabular_diffusion.model_inventory import get_inventory_entry, list_inventory
from standardized_tabular_diffusion.registry import list_adapter_specs, list_datasets, list_models
from standardized_tabular_diffusion.runner import (
    build_run_context,
    run_action,
    run_pipeline,
    save_pipeline_result,
    save_run_context,
)


def compare_summaries(summary_paths: list[Path]):
    """Load the optional dataframe comparison path only when requested."""

    from standardized_tabular_diffusion.comparison import compare_summaries as implementation

    return implementation(summary_paths)


def register_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load pandas-backed dataset registration only when requested."""

    from standardized_tabular_diffusion.dataset_onboarding import register_dataset as implementation

    return implementation(*args, **kwargs)


def process_registered_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load pandas-backed dataset processing only when requested."""

    from standardized_tabular_diffusion.dataset_onboarding import process_registered_dataset as implementation

    return implementation(*args, **kwargs)


def preprocess_split_files(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load pandas-backed leakage-safe preprocessing only when requested."""

    from standardized_tabular_diffusion.preprocessing import preprocess_split_files as implementation

    return implementation(*args, **kwargs)


def _list_dataset_sources() -> list[dict[str, Any]]:
    from standardized_tabular_diffusion.dataset_sources import list_dataset_sources

    return list_dataset_sources()


def _fetch_dataset_source(
    dataset: str,
    *,
    cache_dir: str | None,
    refresh: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    from standardized_tabular_diffusion.dataset_sources import fetch_dataset_source

    return fetch_dataset_source(
        dataset,
        cache_root=cache_dir,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


def _metric_definitions() -> dict[str, Any]:
    from standardized_tabular_diffusion.evaluation import METRIC_DEFINITIONS

    return METRIC_DEFINITIONS


def _list_metric_records(registry_dir: str | None) -> list[dict[str, Any]]:
    from standardized_tabular_diffusion.evaluation.registry import load_metric_registry

    return [record.to_dict() for record in load_metric_registry(registry_dir)]


def _list_protocol_profiles(profile_dir: str | None) -> list[dict[str, Any]]:
    from standardized_tabular_diffusion.evaluation.profiles import list_protocol_profiles

    return [profile.to_dict() for profile in list_protocol_profiles(profile_dir)]


def _validate_protocol_profile(path: str) -> dict[str, Any]:
    from standardized_tabular_diffusion.evaluation.profiles import load_protocol_profile

    profile = load_protocol_profile(path)
    return {
        "valid": True,
        "protocol_id": profile.protocol_id,
        "protocol_version": profile.protocol_version,
        "sha256": profile.fingerprint,
    }


def _validate_dataset_profile(path: str) -> dict[str, Any]:
    from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile

    profile = load_dataset_profile(path)
    return {
        "valid": True,
        "dataset_id": profile.dataset_id,
        "dataset_profile_version": profile.dataset_profile_version,
        "sha256": profile.fingerprint,
        "official_eligible": profile.payload["official_eligible"],
    }


def _list_dataset_profiles(profile_dir: str) -> list[dict[str, Any]]:
    from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile

    root = Path(profile_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset Profile directory does not exist: {root}")
    paths = sorted((*root.glob("*.json"), *root.glob("*.yaml"), *root.glob("*.yml")))
    return [
        {
            "dataset_id": profile.dataset_id,
            "dataset_profile_version": profile.dataset_profile_version,
            "sha256": profile.fingerprint,
            "official_eligible": profile.payload["official_eligible"],
            "source": str(path),
        }
        for path in paths
        for profile in [load_dataset_profile(path)]
    ]


def _import_legacy_dataset_profile(dataset: str, output: str) -> dict[str, Any]:
    from standardized_tabular_diffusion.evaluation.profiles import import_legacy_dataset_spec, write_dataset_profile

    profile = import_legacy_dataset_spec(get_dataset_spec(dataset))
    write_dataset_profile(profile, output)
    return {
        "dataset_id": profile.dataset_id,
        "dataset_profile_version": profile.dataset_profile_version,
        "sha256": profile.fingerprint,
        "official_eligible": False,
        "output": str(Path(output)),
    }


def _validate_result_bundle(path: str) -> dict[str, Any]:
    from standardized_tabular_diffusion.evaluation.bundle import validate_result_bundle

    return {"valid": True, **validate_result_bundle(path).to_dict()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardized interface for tabular diffusion benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_models_parser = subparsers.add_parser("list-models", help="List standardized model adapters")
    list_models_parser.add_argument(
        "--details",
        action="store_true",
        help="Include source, validation, benchmark-track, and support metadata",
    )
    inventory_parser = subparsers.add_parser(
        "list-model-inventory", help="List researched baseline models and their integration status"
    )
    inventory_parser.add_argument(
        "--benchmark",
        choices=["tabstruct-2026", "tabula-2025", "tabforge-2026"],
        default=None,
        help="Optionally filter the inventory by benchmark paper",
    )
    inventory_parser.add_argument(
        "--family",
        choices=[
            "diffusion",
            "llm",
            "foundation",
            "traditional",
            "vae",
            "gan",
            "graph",
            "flow",
            "tree",
            "energy-based",
            "autoregressive",
        ],
        default=None,
        help="Optionally filter the inventory by model family",
    )
    inventory_parser.add_argument(
        "--status",
        choices=["registered", "adapter-complete", "smoke-validated", "native-parity-validated"],
        default=None,
        help="Optionally filter the inventory by cumulative adapter validation level",
    )
    subparsers.add_parser("list-datasets", help="List canonical datasets from the root registry")
    subparsers.add_parser("list-dataset-sources", help="List checksum-pinned public dataset sources")
    subparsers.add_parser("describe-metrics", help="Print the shared TabStruct-aligned metric schema")
    subparsers.add_parser("describe-config", help="Print the shared experiment config schema as an example JSON")

    list_metrics_parser = subparsers.add_parser("list-metrics", help="List versioned Metric Registry records")
    list_metrics_parser.add_argument(
        "--registry-dir", default=None, help="Optional directory containing registry JSON files"
    )
    validate_registry_parser = subparsers.add_parser(
        "validate-metric-registry", help="Validate every Metric Registry record"
    )
    validate_registry_parser.add_argument(
        "--registry-dir", default=None, help="Optional directory containing registry JSON files"
    )

    list_protocols_parser = subparsers.add_parser("list-protocols", help="List versioned protocol profiles")
    list_protocols_parser.add_argument(
        "--profile-dir", default=None, help="Optional directory containing protocol profiles"
    )
    validate_protocol_parser = subparsers.add_parser("validate-protocol-profile", help="Validate one protocol profile")
    validate_protocol_parser.add_argument("--profile", required=True, help="Protocol Profile JSON or safe YAML path")

    list_dataset_profiles_parser = subparsers.add_parser(
        "list-dataset-profiles", help="List Dataset Profiles in a directory"
    )
    list_dataset_profiles_parser.add_argument(
        "--profile-dir", required=True, help="Directory containing Dataset Profiles"
    )
    validate_dataset_profile_parser = subparsers.add_parser(
        "validate-dataset-profile", help="Validate one Dataset Profile"
    )
    validate_dataset_profile_parser.add_argument(
        "--profile", required=True, help="Dataset Profile JSON or safe YAML path"
    )
    import_dataset_profile_parser = subparsers.add_parser(
        "import-legacy-dataset-profile",
        help="Import upstream info.json metadata as a non-official Dataset Profile",
    )
    import_dataset_profile_parser.add_argument("--dataset", required=True, help="Legacy dataset name")
    import_dataset_profile_parser.add_argument("--output", required=True, help="Output JSON path")

    validate_result_parser = subparsers.add_parser(
        "validate-result", help="Validate an incomplete or finalized result bundle"
    )
    validate_result_parser.add_argument("--bundle", required=True, help="Result bundle directory")

    model_parser = subparsers.add_parser("show-model-inventory", help="Show one researched baseline model entry")
    model_parser.add_argument("--model", required=True, help="Model name")

    dataset_parser = subparsers.add_parser("show-dataset", help="Show one canonical dataset spec")
    dataset_parser.add_argument("--dataset", required=True, help="Dataset name")

    example_parser = subparsers.add_parser("example-config", help="Generate an example experiment config")
    example_parser.add_argument("--model", required=True, help="Model name")
    example_parser.add_argument("--dataset", required=True, help="Dataset name")
    example_parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    example_parser.add_argument(
        "--save-config",
        default=None,
        help="Optional path to save the generated config JSON",
    )

    context_parser = subparsers.add_parser(
        "build-context", help="Resolve a config into canonical dataset and run context"
    )
    context_parser.add_argument("--config", required=True, help="Path to experiment config JSON")

    run_parser = subparsers.add_parser("run-action", help="Run one standardized adapter action from a shared config")
    run_parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    run_parser.add_argument("--action", choices=["train", "sample", "evaluate"], required=True, help="Action to run")

    pipeline_parser = subparsers.add_parser("run", help="Run the full standardized pipeline for one config")
    pipeline_parser.add_argument("--config", required=True, help="Path to experiment config JSON")

    materialize_parser = subparsers.add_parser(
        "materialize-dataset", help="Download/process one dataset into the canonical materialized layout"
    )
    materialize_parser.add_argument("--dataset", required=True, help="Dataset name")
    materialize_parser.add_argument("--cache-dir", default=None, help="Optional local dataset cache root")
    materialize_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download a fresh source archive and require it to match the registered checksum",
    )
    materialize_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Positive network timeout in seconds",
    )

    materialize_status_parser = subparsers.add_parser(
        "materialization-status", help="Show materialization status for one dataset"
    )
    materialize_status_parser.add_argument("--dataset", required=True, help="Dataset name")

    download_dataset_parser = subparsers.add_parser(
        "download-dataset",
        help="Download and safely extract a checksum-pinned public dataset source",
    )
    download_dataset_parser.add_argument("--dataset", required=True, help="Registered source name")
    download_dataset_parser.add_argument("--cache-dir", default=None, help="Optional local dataset cache root")
    download_dataset_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace the managed archive cache after downloading and verifying a fresh copy",
    )
    download_dataset_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Positive network timeout in seconds",
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess-missing-values",
        help="Fit mean/mode imputation on a real training CSV and transform frozen validation/test splits",
    )
    preprocess_parser.add_argument("--train-csv", required=True, help="Real training split CSV")
    preprocess_parser.add_argument("--validation-csv", default=None, help="Optional real validation split CSV")
    preprocess_parser.add_argument("--test-csv", default=None, help="Optional real test split CSV")
    preprocess_parser.add_argument("--output-dir", required=True, help="New directory for transformed splits")
    preprocess_parser.add_argument(
        "--numerical-column",
        dest="numerical_columns",
        action="append",
        default=[],
        help="Numerical feature name; repeat for multiple columns",
    )
    preprocess_parser.add_argument(
        "--categorical-column",
        dest="categorical_columns",
        action="append",
        default=[],
        help="Categorical feature name; repeat for multiple columns",
    )
    preprocess_parser.add_argument(
        "--target-column",
        dest="target_columns",
        action="append",
        required=True,
        help="Target name; repeat only for a declared multi-target dataset",
    )
    preprocess_parser.add_argument(
        "--missing-marker",
        dest="missing_markers",
        action="append",
        default=None,
        help="Exact raw missing marker; repeat to override the default marker set",
    )
    preprocess_parser.add_argument(
        "--add-missing-indicators",
        action="store_true",
        help="Append a stable binary missingness indicator for every feature",
    )

    register_dataset_parser = subparsers.add_parser(
        "register-dataset", help="Register a local CSV as a new canonical dataset"
    )
    register_dataset_parser.add_argument("--dataset", required=True, help="Dataset name")
    register_dataset_parser.add_argument("--raw-csv", required=True, help="Path to the local CSV file")
    register_dataset_parser.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "regression", "binclass", "multiclass"],
        help="Task type for the target column",
    )
    register_dataset_parser.add_argument("--target-column", required=True, help="Name of the target column")
    register_dataset_parser.add_argument(
        "--numerical-column",
        dest="numerical_columns",
        action="append",
        default=None,
        help="Name of a numerical feature column; repeat to provide more than one",
    )
    register_dataset_parser.add_argument(
        "--categorical-column",
        dest="categorical_columns",
        action="append",
        default=None,
        help="Name of a categorical feature column; repeat to provide more than one",
    )
    register_dataset_parser.add_argument(
        "--has-header",
        choices=["true", "false"],
        default="true",
        help="Whether the CSV includes a header row",
    )

    process_dataset_parser = subparsers.add_parser(
        "process-dataset", help="Process a registered dataset into the canonical materialized layout"
    )
    process_dataset_parser.add_argument("--dataset", required=True, help="Dataset name")

    compare_parser = subparsers.add_parser("compare", help="Aggregate standardized summaries")
    compare_parser.add_argument(
        "--summary", dest="summaries", action="append", required=True, help="Path to a standardized_summary.json file"
    )
    compare_parser.add_argument(
        "--csv", dest="csv_path", default=None, help="Optional path to save the comparison as CSV"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        models: Any = list_adapter_specs() if args.details else list_models()
        print(json.dumps({"models": models}, indent=2))
        return

    if args.command == "list-model-inventory":
        entries = [
            entry.to_dict()
            for entry in list_inventory(
                benchmark=args.benchmark,
                family=args.family,
                validation_level=args.status,
            )
        ]
        print(json.dumps({"models": entries}, indent=2))
        return

    if args.command == "list-datasets":
        print(json.dumps({"datasets": list_datasets()}, indent=2))
        return

    if args.command == "list-dataset-sources":
        print(json.dumps({"dataset_sources": _list_dataset_sources()}, indent=2))
        return

    if args.command == "describe-metrics":
        print(json.dumps(_metric_definitions(), indent=2))
        return

    if args.command == "list-metrics":
        print(json.dumps({"metrics": _list_metric_records(args.registry_dir)}, indent=2))
        return

    if args.command == "validate-metric-registry":
        metrics = _list_metric_records(args.registry_dir)
        print(json.dumps({"valid": True, "record_count": len(metrics)}, indent=2))
        return

    if args.command == "list-protocols":
        print(json.dumps({"protocols": _list_protocol_profiles(args.profile_dir)}, indent=2))
        return

    if args.command == "validate-protocol-profile":
        print(json.dumps(_validate_protocol_profile(args.profile), indent=2))
        return

    if args.command == "list-dataset-profiles":
        print(json.dumps({"dataset_profiles": _list_dataset_profiles(args.profile_dir)}, indent=2))
        return

    if args.command == "validate-dataset-profile":
        print(json.dumps(_validate_dataset_profile(args.profile), indent=2))
        return

    if args.command == "import-legacy-dataset-profile":
        print(json.dumps(_import_legacy_dataset_profile(args.dataset, args.output), indent=2))
        return

    if args.command == "validate-result":
        print(json.dumps(_validate_result_bundle(args.bundle), indent=2))
        return

    if args.command == "describe-config":
        print(
            json.dumps(
                build_example_config(
                    model="tabdiff",
                    dataset="adult",
                    output_dir="artifacts/tabdiff/adult/run-001",
                ).to_dict(),
                indent=2,
            )
        )
        return

    if args.command == "show-dataset":
        print(json.dumps(get_dataset_spec(args.dataset).to_dict(), indent=2))
        return

    if args.command == "show-model-inventory":
        print(json.dumps(get_inventory_entry(args.model).to_dict(), indent=2))
        return

    if args.command == "example-config":
        config = build_example_config(
            model=args.model,
            dataset=args.dataset,
            output_dir=args.output_dir,
        )
        payload = config.to_dict()
        if args.save_config:
            save_experiment_config(config, args.save_config)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "build-context":
        config = load_experiment_config(args.config)
        context = build_run_context(config)
        save_run_context(context, config.output_dir)
        print(json.dumps(context, indent=2))
        return

    if args.command == "run-action":
        config = load_experiment_config(args.config)
        context = build_run_context(config)
        save_run_context(context, config.output_dir)
        bundle = run_action(config, action=args.action)
        print(json.dumps(bundle.to_dict(), indent=2))
        return

    if args.command == "run":
        config = load_experiment_config(args.config)
        result = run_pipeline(config)
        save_pipeline_result(result, config.output_dir)
        print(json.dumps(result, indent=2))
        return

    if args.command == "materialize-dataset":
        manifest = materialize_dataset(
            args.dataset,
            cache_root=args.cache_dir,
            refresh=args.refresh,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "materialization-status":
        print(json.dumps(materialization_status(args.dataset), indent=2))
        return

    if args.command == "download-dataset":
        print(
            json.dumps(
                _fetch_dataset_source(
                    args.dataset,
                    cache_dir=args.cache_dir,
                    refresh=args.refresh,
                    timeout_seconds=args.timeout_seconds,
                ),
                indent=2,
            )
        )
        return

    if args.command == "preprocess-missing-values":
        from standardized_tabular_diffusion.preprocessing import MissingValuePolicy

        policy_kwargs: dict[str, Any] = {"add_missing_indicators": args.add_missing_indicators}
        if args.missing_markers is not None:
            policy_kwargs["missing_markers"] = tuple(args.missing_markers)
        policy = MissingValuePolicy(**policy_kwargs)
        print(
            json.dumps(
                preprocess_split_files(
                    train_path=args.train_csv,
                    validation_path=args.validation_csv,
                    test_path=args.test_csv,
                    output_dir=args.output_dir,
                    numerical_columns=args.numerical_columns,
                    categorical_columns=args.categorical_columns,
                    target_columns=args.target_columns,
                    policy=policy,
                ),
                indent=2,
            )
        )
        return

    if args.command == "register-dataset":
        payload = register_dataset(
            dataset_name=args.dataset,
            raw_csv_path=args.raw_csv,
            task_type=args.task_type,
            target_column=args.target_column,
            numerical_columns=args.numerical_columns,
            categorical_columns=args.categorical_columns,
            has_header=args.has_header == "true",
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "process-dataset":
        manifest = process_registered_dataset(args.dataset)
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "compare":
        summaries = [Path(path) for path in args.summaries]
        frame = compare_summaries(summaries)
        if args.csv_path:
            atomic_write_bytes(args.csv_path, frame.to_csv(index=False).encode("utf-8"))
        print(frame.to_string(index=False))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
