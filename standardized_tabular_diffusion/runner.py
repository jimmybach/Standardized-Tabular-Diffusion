from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import ExperimentConfig
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.registry import get_adapter


def validate_dataset_spec(dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "name": dataset_spec.name,
        "task_type": dataset_spec.task_type,
        "paths": {
            "metadata_path": {
                "path": str(dataset_spec.metadata_path),
                "exists": dataset_spec.metadata_path.exists(),
            },
            "train_data_path": {
                "path": None if dataset_spec.train_data_path is None else str(dataset_spec.train_data_path),
                "exists": None if dataset_spec.train_data_path is None else dataset_spec.train_data_path.exists(),
            },
            "val_data_path": {
                "path": None if dataset_spec.val_data_path is None else str(dataset_spec.val_data_path),
                "exists": None if dataset_spec.val_data_path is None else dataset_spec.val_data_path.exists(),
            },
            "test_data_path": {
                "path": None if dataset_spec.test_data_path is None else str(dataset_spec.test_data_path),
                "exists": None if dataset_spec.test_data_path is None else dataset_spec.test_data_path.exists(),
            },
        },
    }


def build_run_context(config: ExperimentConfig, repo_root: Path | None = None) -> dict[str, Any]:
    adapter = get_adapter(config.model, repo_root=repo_root)
    dataset_spec = get_dataset_spec(config.dataset, repo_root=repo_root)
    run_spec = adapter.build_run_spec(config, dataset_spec=dataset_spec)
    return {
        "config": config.to_dict(),
        "dataset_spec": dataset_spec.to_dict(),
        "dataset_validation": validate_dataset_spec(dataset_spec),
        "action_readiness": {
            "train": validate_action_inputs(config, "train", dataset_spec=dataset_spec),
            "sample": validate_action_inputs(config, "sample", dataset_spec=dataset_spec),
            "evaluate": validate_action_inputs(config, "evaluate", dataset_spec=dataset_spec),
        },
        "run_spec": run_spec.to_dict(),
        "adapter": {
            "model_name": adapter.model_name,
            "upstream_root": str(adapter.upstream_root),
        },
    }


def save_run_context(context: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_context.json"
    path.write_text(json.dumps(context, indent=2))
    return path


def run_action(
    config: ExperimentConfig,
    action: str,
    repo_root: Path | None = None,
):
    adapter = get_adapter(config.model, repo_root=repo_root)
    dataset_spec = get_dataset_spec(config.dataset, repo_root=repo_root)
    readiness = validate_action_inputs(config, action, dataset_spec=dataset_spec)
    if not readiness["ready"]:
        raise FileNotFoundError(
            f"Cannot run {action} for model={config.model}, dataset={config.dataset}. Missing inputs: "
            + "; ".join(readiness["missing"])
        )

    if action == "train":
        return adapter.train_from_config(config, dataset_spec=dataset_spec)
    if action == "sample":
        return adapter.sample_from_config(config, dataset_spec=dataset_spec)
    if action == "evaluate":
        return adapter.evaluate_from_config(config, dataset_spec=dataset_spec)

    raise ValueError(f"Unsupported action: {action}")


def run_pipeline(
    config: ExperimentConfig,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    adapter = get_adapter(config.model, repo_root=repo_root)
    dataset_spec = get_dataset_spec(config.dataset, repo_root=repo_root)

    context = build_run_context(config, repo_root=repo_root)
    phase_results: dict[str, Any] = {"context": context, "phases": {}}

    sample_path: str | None = config.evaluation.extra.get("sample_path")

    if config.train.enabled:
        readiness = validate_action_inputs(config, "train", dataset_spec=dataset_spec)
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"Cannot run train for model={config.model}, dataset={config.dataset}. Missing inputs: "
                + "; ".join(readiness["missing"])
            )
        bundle = adapter.train_from_config(config, dataset_spec=dataset_spec)
        phase_results["phases"]["train"] = bundle.to_dict()

    if config.sample.enabled:
        readiness = validate_action_inputs(config, "sample", dataset_spec=dataset_spec)
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"Cannot run sample for model={config.model}, dataset={config.dataset}. Missing inputs: "
                + "; ".join(readiness["missing"])
            )
        bundle = adapter.sample_from_config(config, dataset_spec=dataset_spec)
        phase_results["phases"]["sample"] = bundle.to_dict()
        if bundle.generated_sample_path is not None:
            sample_path = str(bundle.generated_sample_path)

    if config.evaluation.enabled:
        eval_config = deepcopy(config)
        if sample_path is not None:
            eval_config.evaluation.extra["sample_path"] = sample_path
            eval_config.sample.extra["sample_path"] = sample_path
        readiness = validate_action_inputs(eval_config, "evaluate", dataset_spec=dataset_spec)
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"Cannot run evaluate for model={config.model}, dataset={config.dataset}. Missing inputs: "
                + "; ".join(readiness["missing"])
            )
        bundle = adapter.evaluate_from_config(eval_config, dataset_spec=dataset_spec)
        phase_results["phases"]["evaluate"] = bundle.to_dict()

    return phase_results


def save_pipeline_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pipeline_result.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def validate_action_inputs(
    config: ExperimentConfig,
    action: str,
    dataset_spec: DatasetSpec | None = None,
) -> dict[str, Any]:
    dataset_spec = dataset_spec or get_dataset_spec(config.dataset)
    missing: list[str] = []
    checked: dict[str, Any] = {}

    if action in {"train", "sample"} and config.model in {"tabdiff", "tabsyn"}:
        checked["metadata_path"] = str(dataset_spec.metadata_path)
        if not dataset_spec.metadata_path.exists():
            missing.append(f"metadata_path missing: {dataset_spec.metadata_path}")
        checked["train_data_path"] = None if dataset_spec.train_data_path is None else str(dataset_spec.train_data_path)
        if dataset_spec.train_data_path is None or not dataset_spec.train_data_path.exists():
            missing.append(f"train_data_path missing: {dataset_spec.train_data_path}")
        checked["test_data_path"] = None if dataset_spec.test_data_path is None else str(dataset_spec.test_data_path)
        if dataset_spec.test_data_path is not None and not dataset_spec.test_data_path.exists():
            missing.append(f"test_data_path missing: {dataset_spec.test_data_path}")

    if config.model == "tabsyn":
        method = config.sample.extra.get("method") or config.train.extra.get("method") or "tabsyn"
        repo_root = Path(__file__).resolve().parents[1]
        vae_ckpt_dir = repo_root / "TabSyn-main" / "tabsyn" / "vae" / "ckpt" / config.dataset
        diffusion_ckpt_dir = repo_root / "TabSyn-main" / "tabsyn" / "ckpt" / config.dataset
        if method == "tabsyn":
            if action == "train":
                checked["tabsyn_stage_model"] = "vae_then_diffusion"
            elif action == "sample":
                checked["tabsyn_train_z"] = str(vae_ckpt_dir / "train_z.npy")
                checked["tabsyn_decoder"] = str(vae_ckpt_dir / "decoder.pt")
                checked["tabsyn_diffusion_model"] = str(diffusion_ckpt_dir / "model.pt")
                if not (vae_ckpt_dir / "train_z.npy").exists():
                    missing.append(f"tabsyn prerequisite missing: {vae_ckpt_dir / 'train_z.npy'}")
                if not (vae_ckpt_dir / "decoder.pt").exists():
                    missing.append(f"tabsyn prerequisite missing: {vae_ckpt_dir / 'decoder.pt'}")
                if not (diffusion_ckpt_dir / "model.pt").exists():
                    missing.append(f"tabsyn prerequisite missing: {diffusion_ckpt_dir / 'model.pt'}")

    if action == "sample" and config.model == "tabddpm":
        if config.upstream_config_path is None:
            missing.append("upstream_config_path missing for tabddpm sample")
        else:
            checked["upstream_config_path"] = config.upstream_config_path
            if not Path(config.upstream_config_path).exists():
                missing.append(f"upstream_config_path missing: {config.upstream_config_path}")

    if action == "train" and config.model == "tabddpm":
        if config.upstream_config_path is None:
            missing.append("upstream_config_path missing for tabddpm train")
        else:
            checked["upstream_config_path"] = config.upstream_config_path
            if not Path(config.upstream_config_path).exists():
                missing.append(f"upstream_config_path missing: {config.upstream_config_path}")

    if action == "evaluate":
        if config.model in {"tabdiff", "tabsyn"}:
            sample_path = config.evaluation.extra.get("sample_path")
            checked["sample_path"] = sample_path
            if sample_path is None or not Path(sample_path).exists():
                missing.append(f"sample_path missing: {sample_path}")
        elif config.model == "tabddpm":
            required_any = [
                "results_catboost_path",
                "results_mlp_path",
                "privacy_path",
                "simple_path",
            ]
            existing_any = False
            for key in required_any:
                path = config.evaluation.extra.get(key)
                checked[key] = path
                if path is not None and Path(path).exists():
                    existing_any = True
            if not existing_any:
                missing.append("at least one TabDDPM evaluation artifact path is required and must exist")

    return {
        "action": action,
        "ready": not missing,
        "checked": checked,
        "missing": missing,
    }
