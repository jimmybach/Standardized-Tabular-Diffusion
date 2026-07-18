from __future__ import annotations

import argparse
import json
from pathlib import Path

from standardized_tabular_diffusion.comparison import compare_summaries
from standardized_tabular_diffusion.config import build_example_config, load_experiment_config, save_experiment_config
from standardized_tabular_diffusion.dataset_onboarding import process_registered_dataset, register_dataset
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.tabstruct import METRIC_DEFINITIONS
from standardized_tabular_diffusion.materialization import materialization_status, materialize_dataset
from standardized_tabular_diffusion.model_inventory import get_inventory_entry, list_inventory
from standardized_tabular_diffusion.registry import list_datasets, list_models
from standardized_tabular_diffusion.runner import (
    build_run_context,
    run_action,
    run_pipeline,
    save_pipeline_result,
    save_run_context,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardized interface for tabular diffusion benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List standardized model adapters")
    inventory_parser = subparsers.add_parser("list-model-inventory", help="List researched baseline models and their integration status")
    inventory_parser.add_argument(
        "--benchmark",
        choices=["tabstruct-2026", "tabula-2025", "tabforge-2026"],
        default=None,
        help="Optionally filter the inventory by benchmark paper",
    )
    inventory_parser.add_argument(
        "--family",
        choices=["diffusion", "llm", "foundation", "traditional", "vae", "gan", "graph", "flow", "tree", "energy-based", "autoregressive"],
        default=None,
        help="Optionally filter the inventory by model family",
    )
    inventory_parser.add_argument(
        "--status",
        choices=["implemented", "not implemented"],
        default=None,
        help="Optionally filter the inventory by standardized integration status",
    )
    subparsers.add_parser("list-datasets", help="List canonical datasets from the root registry")
    subparsers.add_parser("describe-metrics", help="Print the shared TabStruct-aligned metric schema")
    subparsers.add_parser("describe-config", help="Print the shared experiment config schema as an example JSON")

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

    context_parser = subparsers.add_parser("build-context", help="Resolve a config into canonical dataset and run context")
    context_parser.add_argument("--config", required=True, help="Path to experiment config JSON")

    run_parser = subparsers.add_parser("run-action", help="Run one standardized adapter action from a shared config")
    run_parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    run_parser.add_argument("--action", choices=["train", "sample", "evaluate"], required=True, help="Action to run")

    pipeline_parser = subparsers.add_parser("run", help="Run the full standardized pipeline for one config")
    pipeline_parser.add_argument("--config", required=True, help="Path to experiment config JSON")

    materialize_parser = subparsers.add_parser("materialize-dataset", help="Download/process one dataset into the canonical materialized layout")
    materialize_parser.add_argument("--dataset", required=True, help="Dataset name")

    materialize_status_parser = subparsers.add_parser("materialization-status", help="Show materialization status for one dataset")
    materialize_status_parser.add_argument("--dataset", required=True, help="Dataset name")

    register_dataset_parser = subparsers.add_parser("register-dataset", help="Register a local CSV as a new canonical dataset")
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

    process_dataset_parser = subparsers.add_parser("process-dataset", help="Process a registered dataset into the canonical materialized layout")
    process_dataset_parser.add_argument("--dataset", required=True, help="Dataset name")

    compare_parser = subparsers.add_parser("compare", help="Aggregate standardized summaries")
    compare_parser.add_argument("--summary", dest="summaries", action="append", required=True, help="Path to a standardized_summary.json file")
    compare_parser.add_argument("--csv", dest="csv_path", default=None, help="Optional path to save the comparison as CSV")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-models":
        print(json.dumps({"models": list_models()}, indent=2))
        return

    if args.command == "list-model-inventory":
        entries = [
            entry.to_dict()
            for entry in list_inventory(
                benchmark=args.benchmark,
                family=args.family,
                standardized_status=args.status,
            )
        ]
        print(json.dumps({"models": entries}, indent=2))
        return

    if args.command == "list-datasets":
        print(json.dumps({"datasets": list_datasets()}, indent=2))
        return

    if args.command == "describe-metrics":
        print(json.dumps(METRIC_DEFINITIONS, indent=2))
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
        manifest = materialize_dataset(args.dataset)
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "materialization-status":
        print(json.dumps(materialization_status(args.dataset), indent=2))
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
            Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.csv_path, index=False)
        print(frame.to_string(index=False))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
