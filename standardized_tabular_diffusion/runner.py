from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import ExperimentConfig
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json
from standardized_tabular_diffusion.interfaces import DatasetSpec
from standardized_tabular_diffusion.registry import get_adapter, get_adapter_spec


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
            "train": validate_action_inputs(config, "train", dataset_spec=dataset_spec, repo_root=repo_root),
            "sample": validate_action_inputs(config, "sample", dataset_spec=dataset_spec, repo_root=repo_root),
            "evaluate": validate_action_inputs(config, "evaluate", dataset_spec=dataset_spec, repo_root=repo_root),
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
    atomic_write_json(path, context)
    return path


def run_action(
    config: ExperimentConfig,
    action: str,
    repo_root: Path | None = None,
):
    adapter = get_adapter(config.model, repo_root=repo_root)
    dataset_spec = get_dataset_spec(config.dataset, repo_root=repo_root)
    readiness = validate_action_inputs(config, action, dataset_spec=dataset_spec, repo_root=repo_root)
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
        readiness = validate_action_inputs(config, "train", dataset_spec=dataset_spec, repo_root=repo_root)
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"Cannot run train for model={config.model}, dataset={config.dataset}. Missing inputs: "
                + "; ".join(readiness["missing"])
            )
        bundle = adapter.train_from_config(config, dataset_spec=dataset_spec)
        phase_results["phases"]["train"] = bundle.to_dict()

    if config.sample.enabled:
        readiness = validate_action_inputs(config, "sample", dataset_spec=dataset_spec, repo_root=repo_root)
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
        readiness = validate_action_inputs(eval_config, "evaluate", dataset_spec=dataset_spec, repo_root=repo_root)
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
    atomic_write_json(path, result)
    return path


def validate_action_inputs(
    config: ExperimentConfig,
    action: str,
    dataset_spec: DatasetSpec | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if action not in {"train", "sample", "evaluate"}:
        raise ValueError(f"Unsupported action: {action}")
    resolved_root = repo_root or Path(__file__).resolve().parents[1]
    dataset_spec = dataset_spec or get_dataset_spec(config.dataset, repo_root=resolved_root)
    adapter_spec = get_adapter_spec(config.model)
    missing: list[str] = []
    checked: dict[str, Any] = {}

    def record_user_checkpoint(checkpoint_path: str) -> None:
        checked["checkpoint_path"] = checkpoint_path
        path = Path(checkpoint_path)
        if not path.exists():
            missing.append(f"checkpoint_path missing: {checkpoint_path}")
            return
        resolved_checkpoint = path.resolve()
        output_root = Path(config.output_dir).resolve()
        allow_external = bool(config.sample.extra.get("allow_unsafe_external_checkpoint", False))
        checked["allow_unsafe_external_checkpoint"] = allow_external
        if not resolved_checkpoint.is_relative_to(output_root) and not allow_external:
            missing.append(
                "external checkpoint loading is blocked for code-executing model formats; "
                "move the checkpoint under output_dir or explicitly set "
                "sample.extra.allow_unsafe_external_checkpoint=true after provenance review"
            )

    if action not in adapter_spec.actions:
        missing.append(f"action {action!r} is not supported by adapter {config.model!r}")

    if action in {"train", "sample"} and adapter_spec.requires_dataset_paths:
        checked["metadata_path"] = str(dataset_spec.metadata_path)
        if not dataset_spec.metadata_path.exists():
            missing.append(f"metadata_path missing: {dataset_spec.metadata_path}")
        checked["train_data_path"] = None if dataset_spec.train_data_path is None else str(dataset_spec.train_data_path)
        if dataset_spec.train_data_path is None or not dataset_spec.train_data_path.exists():
            missing.append(f"train_data_path missing: {dataset_spec.train_data_path}")
        checked["test_data_path"] = None if dataset_spec.test_data_path is None else str(dataset_spec.test_data_path)
        if dataset_spec.test_data_path is not None and not dataset_spec.test_data_path.exists():
            missing.append(f"test_data_path missing: {dataset_spec.test_data_path}")

    if dataset_spec.task_type not in adapter_spec.task_types:
        checked["task_type"] = dataset_spec.task_type
        missing.append(
            f"{config.model} supports task types {list(adapter_spec.task_types)}, got: {dataset_spec.task_type}"
        )

    if action == "sample" and config.model in {
        "ctgan",
        "tvae",
        "ctab-gan",
        "ctab-gan-plus",
        "nrgboost",
        "bn",
        "nflow",
        "goggle",
        "arf",
        "tabebm",
        "tabsds",
        "tabularargn",
    }:
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.pkl")
        if config.model == "ctab-gan-plus":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "ctabgan_plus.pkl")
        if config.model == "ctab-gan":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "ctabgan.pkl")
        if config.model == "nrgboost":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.nrgboost")
        if config.model == "goggle":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.pt")
        if config.model == "tabsds":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "tabsds.pkl")
        if config.model == "tabularargn":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "tabularargn.pkl")
        record_user_checkpoint(checkpoint_path)

    if action == "sample" and config.model in {"stasy", "codi"}:
        if config.model == "stasy":
            stasy_checkpoint_path = (
                resolved_root / "TabSyn-main" / "baselines" / "stasy" / "ckpt" / config.dataset / "model.pth"
            )
            checked["checkpoint_path"] = str(stasy_checkpoint_path)
            if not stasy_checkpoint_path.exists():
                missing.append(f"checkpoint_path missing: {stasy_checkpoint_path}")
        if config.model == "codi":
            checkpoint_con = (
                resolved_root / "TabSyn-main" / "baselines" / "codi" / "ckpt" / config.dataset / "model_con.pt"
            )
            checkpoint_dis = (
                resolved_root / "TabSyn-main" / "baselines" / "codi" / "ckpt" / config.dataset / "model_dis.pt"
            )
            checked["checkpoint_con_path"] = str(checkpoint_con)
            checked["checkpoint_dis_path"] = str(checkpoint_dis)
            if not checkpoint_con.exists():
                missing.append(f"checkpoint_path missing: {checkpoint_con}")
            if not checkpoint_dis.exists():
                missing.append(f"checkpoint_path missing: {checkpoint_dis}")

    if action == "sample" and config.model == "tabebm":
        allow_gated_model = bool(config.sample.extra.get("allow_gated_model", False))
        checked["allow_gated_model"] = allow_gated_model
        if not allow_gated_model:
            missing.append(
                "sample.extra.allow_gated_model must be true for tabebm sample because TabPFN access is gated"
            )

    if action == "sample" and config.model == "great":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "great_model")
        record_user_checkpoint(checkpoint_path)

    if action == "sample" and config.model == "tabula":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "tabula_model")
        record_user_checkpoint(checkpoint_path)

    if action == "sample" and config.model == "realtabformer":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "realtabformer_model")
        record_user_checkpoint(checkpoint_path)

    if config.model == "tabsyn":
        method = config.sample.extra.get("method") or config.train.extra.get("method") or "tabsyn"
        vae_ckpt_dir = resolved_root / "TabSyn-main" / "tabsyn" / "vae" / "ckpt" / config.dataset
        diffusion_ckpt_dir = resolved_root / "TabSyn-main" / "tabsyn" / "ckpt" / config.dataset
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
        if adapter_spec.evaluation_input == "sample-file":
            sample_path = config.evaluation.extra.get("sample_path")
            checked["sample_path"] = sample_path
            if sample_path is None or not Path(sample_path).exists():
                missing.append(f"sample_path missing: {sample_path}")
        elif adapter_spec.evaluation_input == "upstream-artifacts":
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
