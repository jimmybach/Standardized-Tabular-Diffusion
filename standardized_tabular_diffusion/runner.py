from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import ExperimentConfig
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.registry import get_adapter, get_adapter_spec
from standardized_tabular_diffusion.runtime_contracts import (
    claim_output_identity,
    dataset_content_identity,
    inspect_regular_file,
    validate_action_controls,
)
from standardized_tabular_diffusion.upstream_sources import source_status


def validate_dataset_spec(dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "name": dataset_spec.name,
        "task_type": dataset_spec.task_type,
        "paths": {
            "metadata_path": inspect_regular_file(dataset_spec.metadata_path),
            "train_data_path": inspect_regular_file(dataset_spec.train_data_path),
            "val_data_path": inspect_regular_file(dataset_spec.val_data_path),
            "test_data_path": inspect_regular_file(dataset_spec.test_data_path),
        },
    }


def _action_identity(
    config: ExperimentConfig,
    action: str,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    if action == "train":
        controls = config.train.extra
        seed = config.train.seed
        device = config.train.device
        num_samples = None
        checkpoint_path = None
        upstream_config = config.upstream_config_path
    elif action == "sample":
        controls = config.sample.extra
        seed = config.sample.seed if config.sample.seed is not None else config.train.seed
        device = config.train.device
        num_samples = config.sample.num_samples
        checkpoint_path = config.sample.checkpoint_path
        upstream_config = config.upstream_config_path
    elif action == "evaluate":
        controls = config.evaluation.extra
        seed = config.sample.seed if config.sample.seed is not None else config.train.seed
        device = "cpu"
        num_samples = config.sample.num_samples
        checkpoint_path = None
        upstream_config = None
    else:
        raise ValueError(f"Unsupported action: {action}")
    return {
        "schema_version": 1,
        "action": action,
        "model": config.model,
        "dataset": config.dataset,
        "seed": seed,
        "device": device,
        "num_samples": num_samples,
        "checkpoint_path": checkpoint_path,
        "upstream_config": inspect_regular_file(None if upstream_config is None else Path(upstream_config)),
        "controls": controls,
        "dataset_identity": dataset_content_identity(dataset_spec),
    }


def _claim_action_output(config: ExperimentConfig, action: str, dataset_spec: DatasetSpec) -> Path:
    action_identity = _action_identity(config, action, dataset_spec)
    return claim_output_identity(
        Path(config.output_dir),
        model=config.model,
        dataset=config.dataset,
        actions={
            "dataset": {
                "schema_version": 1,
                "dataset_identity": action_identity["dataset_identity"],
            },
            action: action_identity,
        },
    )


def _declared_upstream_root(model: str, repo_root: Path | None) -> Path:
    """Resolve provenance metadata without importing a model runtime."""

    resolved_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    source_root = get_adapter_spec(model).source_root
    return resolved_root if source_root is None else resolved_root / source_root


def _build_run_spec(config: ExperimentConfig, dataset_spec: DatasetSpec) -> RunSpec:
    """Build the shared public RunSpec without loading optional model dependencies."""

    spec = config.to_run_spec()
    spec.extra.setdefault("dataset_spec", dataset_spec.to_dict())
    spec.extra.setdefault("dataset_identity", dataset_content_identity(dataset_spec))
    spec.extra.setdefault("config", config.to_dict())
    return spec


def _write_artifact_bundle(bundle: ArtifactBundle) -> ArtifactBundle:
    """Persist a central-evaluation bundle without constructing a model adapter."""

    bundle.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(bundle.output_dir / "artifacts.json", bundle.to_dict())
    return bundle


def build_run_context(
    config: ExperimentConfig,
    repo_root: Path | None = None,
    *,
    dataset_spec: DatasetSpec | None = None,
) -> dict[str, Any]:
    dataset_spec = dataset_spec or get_dataset_spec(config.dataset, repo_root=repo_root)
    if dataset_spec.name != config.dataset:
        raise ValueError(
            f"Explicit DatasetSpec name {dataset_spec.name!r} does not match config dataset {config.dataset!r}."
        )
    run_spec = _build_run_spec(config, dataset_spec)
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
            "model_name": config.model,
            "upstream_root": str(_declared_upstream_root(config.model, repo_root)),
        },
    }


def save_run_context(context: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    config = context.get("config")
    run_spec = context.get("run_spec")
    if not isinstance(config, dict) or not isinstance(run_spec, dict):
        raise ValueError("Run context must contain object-valued config and run_spec records.")
    model = config.get("model")
    dataset = config.get("dataset")
    if not isinstance(model, str) or not isinstance(dataset, str):
        raise ValueError("Run context config must contain string-valued model and dataset identities.")
    run_extra = run_spec.get("extra")
    dataset_identity = run_extra.get("dataset_identity") if isinstance(run_extra, dict) else None
    if not isinstance(dataset_identity, dict):
        raise ValueError("Run context run_spec must contain an object-valued dataset_identity record.")
    claim_output_identity(
        output_dir,
        model=model,
        dataset=dataset,
        actions={
            "dataset": {
                "schema_version": 1,
                "dataset_identity": dataset_identity,
            }
        },
    )
    path = output_dir / "run_context.json"
    atomic_write_json(path, context)
    return path


def run_action(
    config: ExperimentConfig,
    action: str,
    repo_root: Path | None = None,
    *,
    dataset_spec: DatasetSpec | None = None,
):
    dataset_spec = dataset_spec or get_dataset_spec(config.dataset, repo_root=repo_root)
    if dataset_spec.name != config.dataset:
        raise ValueError(
            f"Explicit DatasetSpec name {dataset_spec.name!r} does not match config dataset {config.dataset!r}."
        )
    readiness = validate_action_inputs(config, action, dataset_spec=dataset_spec, repo_root=repo_root)
    if not readiness["ready"]:
        raise FileNotFoundError(
            f"Cannot run {action} for model={config.model}, dataset={config.dataset}. Missing inputs: "
            + "; ".join(readiness["missing"])
        )
    _claim_action_output(config, action, dataset_spec)

    if action == "evaluate":
        return run_central_evaluation(config, dataset_spec=dataset_spec, repo_root=repo_root)

    adapter = get_adapter(config.model, repo_root=repo_root)
    if action == "train":
        return adapter.train_from_config(config, dataset_spec=dataset_spec)
    if action == "sample":
        return adapter.sample_from_config(config, dataset_spec=dataset_spec)

    raise ValueError(f"Unsupported action: {action}")


def run_central_evaluation(
    config: ExperimentConfig,
    *,
    dataset_spec: DatasetSpec,
    repo_root: Path | None = None,
) -> ArtifactBundle:
    """Route every public adapter evaluation through the versioned central engine."""

    from standardized_tabular_diffusion.evaluation.service import evaluate_adapter_output

    sample_value = config.evaluation.extra.get("sample_path") or config.sample.extra.get("sample_path")
    if not isinstance(sample_value, str) or not sample_value:
        raise ValueError("Central adapter evaluation requires evaluation.extra.sample_path")
    if config.evaluation.dataset_profile_path is None:
        raise ValueError("Central adapter evaluation requires evaluation.dataset_profile_path")
    reference = (
        Path(config.evaluation.reference_path)
        if config.evaluation.reference_path is not None
        else dataset_spec.train_data_path
    )
    if reference is None:
        raise ValueError("Central adapter evaluation requires a real training reference table")
    real_test = (
        Path(config.evaluation.real_test_path)
        if config.evaluation.real_test_path is not None
        else dataset_spec.test_data_path
    )
    bundle_root = Path(config.output_dir) / "evaluation-result"
    generation_seed = config.sample.seed if config.sample.seed is not None else config.train.seed
    outcome = evaluate_adapter_output(
        model_id=config.model,
        generation_seed=generation_seed,
        synthetic_path=sample_value,
        output_dir=bundle_root,
        protocol_id=config.evaluation.protocol,
        dataset_profile_path=config.evaluation.dataset_profile_path,
        reference_path=reference,
        real_test_path=real_test if config.evaluation.protocol in {"p4-utility", "p5-high-order-privacy"} else None,
        comparison_track=config.evaluation.comparison_track,
        evaluator_seeds=(
            None if config.evaluation.evaluator_seeds is None else tuple(config.evaluation.evaluator_seeds)
        ),
        expected_rows=config.sample.num_samples,
    )
    result = ArtifactBundle(
        model=config.model,
        dataset=config.dataset,
        output_dir=Path(config.output_dir),
        upstream_workdir=_declared_upstream_root(config.model, repo_root),
        generated_sample_path=Path(sample_value),
        evaluation_bundle_path=bundle_root,
        notes=[
            f"Central evaluation protocol: {config.evaluation.protocol}",
            f"Finalized Result Bundle: {outcome.report.bundle_id}",
            "Legacy standardized_summary.json generation is disabled.",
        ],
    )
    return _write_artifact_bundle(result)


def run_pipeline(
    config: ExperimentConfig,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    dataset_spec = get_dataset_spec(config.dataset, repo_root=repo_root)
    adapter = (
        get_adapter(config.model, repo_root=repo_root)
        if config.train.enabled or config.sample.enabled
        else None
    )

    context = build_run_context(config, repo_root=repo_root)
    phase_results: dict[str, Any] = {"context": context, "phases": {}}

    sample_path: str | None = config.evaluation.extra.get("sample_path")

    if config.train.enabled:
        if adapter is None:  # pragma: no cover - guarded by adapter construction above
            raise RuntimeError("Training requires a model adapter")
        readiness = validate_action_inputs(config, "train", dataset_spec=dataset_spec, repo_root=repo_root)
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"Cannot run train for model={config.model}, dataset={config.dataset}. Missing inputs: "
                + "; ".join(readiness["missing"])
            )
        _claim_action_output(config, "train", dataset_spec)
        bundle = adapter.train_from_config(config, dataset_spec=dataset_spec)
        phase_results["phases"]["train"] = bundle.to_dict()

    if config.sample.enabled:
        if adapter is None:  # pragma: no cover - guarded by adapter construction above
            raise RuntimeError("Sampling requires a model adapter")
        readiness = validate_action_inputs(config, "sample", dataset_spec=dataset_spec, repo_root=repo_root)
        if not readiness["ready"]:
            raise FileNotFoundError(
                f"Cannot run sample for model={config.model}, dataset={config.dataset}. Missing inputs: "
                + "; ".join(readiness["missing"])
            )
        _claim_action_output(config, "sample", dataset_spec)
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
        _claim_action_output(eval_config, "evaluate", dataset_spec)
        bundle = run_central_evaluation(eval_config, dataset_spec=dataset_spec, repo_root=repo_root)
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
    action_controls = (
        config.train.extra
        if action == "train"
        else config.sample.extra
        if action == "sample"
        else config.evaluation.extra
    )
    validate_action_controls(config.model, action, action_controls)

    def record_user_checkpoint(
        checkpoint_path: str,
        *,
        code_executing: bool = True,
        allow_directory: bool = False,
    ) -> None:
        checked["checkpoint_path"] = checkpoint_path
        path = Path(checkpoint_path)
        if not path.exists():
            missing.append(f"checkpoint_path missing: {checkpoint_path}")
            return
        if path.is_symlink():
            missing.append(f"checkpoint_path is a prohibited symlink: {checkpoint_path}")
            return
        if not path.is_file() and not (allow_directory and path.is_dir()):
            expected = "file or directory" if allow_directory else "regular file"
            missing.append(f"checkpoint_path is not a {expected}: {checkpoint_path}")
            return
        resolved_checkpoint = path.resolve()
        output_root = Path(config.output_dir).resolve()
        allow_external = bool(config.sample.extra.get("allow_unsafe_external_checkpoint", False))
        checked["allow_unsafe_external_checkpoint"] = allow_external
        checked["checkpoint_code_executing"] = code_executing
        if code_executing and not resolved_checkpoint.is_relative_to(output_root) and not allow_external:
            missing.append(
                "external checkpoint loading is blocked for code-executing model formats; "
                "move the checkpoint under output_dir or explicitly set "
                "sample.extra.allow_unsafe_external_checkpoint=true after provenance review"
            )

    if action not in adapter_spec.actions:
        missing.append(f"action {action!r} is not supported by adapter {config.model!r}")

    if action in {"train", "sample"} and adapter_spec.requires_dataset_paths:
        identity = dataset_content_identity(dataset_spec)
        checked["dataset_identity"] = identity
        checked["metadata_path"] = str(dataset_spec.metadata_path)
        metadata_record = identity["files"]["metadata"]
        if not metadata_record["exists"] or metadata_record.get("unsafe"):
            missing.append(f"metadata_path missing or unsafe: {dataset_spec.metadata_path}")
        checked["train_data_path"] = None if dataset_spec.train_data_path is None else str(dataset_spec.train_data_path)
        train_record = identity["files"]["train"]
        if dataset_spec.train_data_path is None or not train_record["exists"] or train_record.get("unsafe"):
            missing.append(f"train_data_path missing or unsafe: {dataset_spec.train_data_path}")
        checked["test_data_path"] = None if dataset_spec.test_data_path is None else str(dataset_spec.test_data_path)
        test_record = identity["files"]["test"]
        if dataset_spec.test_data_path is not None and (
            not test_record["exists"] or test_record.get("unsafe")
        ):
            missing.append(f"test_data_path missing or unsafe: {dataset_spec.test_data_path}")

    if dataset_spec.task_type not in adapter_spec.task_types:
        checked["task_type"] = dataset_spec.task_type
        missing.append(
            f"{config.model} supports task types {list(adapter_spec.task_types)}, got: {dataset_spec.task_type}"
        )

    if config.model == "ctab-gan-plus" and action in {"train", "sample"}:
        action_extra = config.train.extra if action == "train" else config.sample.extra
        status = source_status(
            "ctab-gan-plus",
            repo_root=resolved_root,
            source_dir=action_extra.get("source_dir")
            or os.environ.get("STANDARDIZED_TABULAR_DIFFUSION_CTABGAN_PLUS_SOURCE"),
        )
        checked["model_source"] = status
        if status["status"] != "ready":
            missing.append(
                "checksum-locked CTAB-GAN+ source is not ready; run "
                "`python -m standardized_tabular_diffusion.cli materialize-model-source "
                "--model ctab-gan-plus` or provide the same verified source_dir for train and sample"
            )

    if config.model == "ctab-gan" and action in {"train", "sample"}:
        action_extra = config.train.extra if action == "train" else config.sample.extra
        source_dir = (
            action_extra.get("source_dir")
            or os.environ.get("STANDARDIZED_TABULAR_DIFFUSION_CTABGAN_SOURCE")
            or resolved_root / "TabDDPM-main" / "CTAB-GAN"
        )
        status = source_status("ctab-gan", repo_root=resolved_root, source_dir=source_dir)
        checked["model_source"] = status
        if status["status"] != "ready":
            missing.append(
                "checksum-locked CTAB-GAN source is not ready; restore the complete repository checkout, run "
                "`python -m standardized_tabular_diffusion.cli materialize-model-source --model ctab-gan`, "
                "or provide the same verified source_dir for train and sample"
            )

    if config.model == "goggle" and action in {"train", "sample"}:
        action_extra = config.train.extra if action == "train" else config.sample.extra
        status = source_status(
            "goggle",
            repo_root=resolved_root,
            source_dir=action_extra.get("source_dir") or os.environ.get("STANDARDIZED_TABULAR_DIFFUSION_GOGGLE_SOURCE"),
        )
        checked["model_source"] = status
        if status["status"] != "ready":
            missing.append(
                "checksum-locked method-author Goggle source is not ready; run "
                "`python -m standardized_tabular_diffusion.cli materialize-model-source --model goggle` "
                "or provide the same verified source_dir for train and sample"
            )

    if config.model in {"tabsds", "tabula"} and action in {"train", "sample"}:
        action_extra = config.train.extra if action == "train" else config.sample.extra
        environment_name = (
            "STANDARDIZED_TABULAR_DIFFUSION_TABSDS_SOURCE"
            if config.model == "tabsds"
            else "STANDARDIZED_TABULAR_DIFFUSION_TABULA_SOURCE"
        )
        status = source_status(
            config.model,
            repo_root=resolved_root,
            source_dir=action_extra.get("source_dir") or os.environ.get(environment_name),
        )
        checked["model_source"] = status
        if status["status"] != "ready":
            missing.append(
                f"checksum-locked method-author {config.model} source is not ready; run "
                f"`python -m standardized_tabular_diffusion.cli materialize-model-source --model {config.model}` "
                "or provide the same verified source_dir for train and sample"
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
        if config.model == "arf":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.arf.json")
        if config.model == "bn":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.bn.json")
        if config.model == "nflow":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.nflow.json")
        if config.model == "tabsds":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.tabsds.json")
        if config.model == "tabebm":
            checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "model.tabebm.json")
        record_user_checkpoint(
            checkpoint_path,
            code_executing=config.model not in {"arf", "bn", "nflow", "tabebm", "tabsds"},
        )

    if action == "sample" and config.model == "arf":
        arf_checkpoint_path = Path(
            config.sample.checkpoint_path or str(Path(config.output_dir) / "model.arf.json")
        )
        metadata_path = arf_checkpoint_path.with_name(f"{arf_checkpoint_path.name}.metadata.json")
        checked["checkpoint_metadata_path"] = str(metadata_path)
        if not metadata_path.exists():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")

    if action == "sample" and config.model == "bn":
        bn_checkpoint_path = Path(
            config.sample.checkpoint_path or str(Path(config.output_dir) / "model.bn.json")
        )
        metadata_path = bn_checkpoint_path.with_name(f"{bn_checkpoint_path.name}.metadata.json")
        checked["checkpoint_metadata_path"] = str(metadata_path)
        if not metadata_path.exists():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")

    if action == "sample" and config.model == "nflow":
        nflow_checkpoint_path = Path(
            config.sample.checkpoint_path or str(Path(config.output_dir) / "model.nflow.json")
        )
        metadata_path = nflow_checkpoint_path.with_name(f"{nflow_checkpoint_path.name}.metadata.json")
        weights_path = nflow_checkpoint_path.with_name(f"{nflow_checkpoint_path.stem}.weights.npz")
        checked["checkpoint_metadata_path"] = str(metadata_path)
        checked["checkpoint_weights_path"] = str(weights_path)
        if not metadata_path.exists():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")
        if not weights_path.exists():
            missing.append(f"checkpoint_weights_path missing: {weights_path}")

    if action == "sample" and config.model == "goggle":
        metadata_path = Path(config.output_dir) / "goggle-model-metadata.json"
        runtime_config_path = Path(config.output_dir) / "goggle-runtime-config.json"
        checked["checkpoint_metadata_path"] = str(metadata_path)
        checked["runtime_config_path"] = str(runtime_config_path)
        if not metadata_path.exists():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")
        if not runtime_config_path.exists():
            missing.append(f"runtime_config_path missing: {runtime_config_path}")

    if action == "sample" and config.model in {"stasy", "codi"}:
        if config.model == "stasy":
            stasy_checkpoint_path = Path(config.output_dir) / "ckpt" / config.dataset / "model.pth"
            stasy_metadata_path = Path(config.output_dir) / "stasy-model-metadata.json"
            checked["checkpoint_path"] = str(stasy_checkpoint_path)
            checked["checkpoint_metadata_path"] = str(stasy_metadata_path)
            if not stasy_checkpoint_path.exists():
                missing.append(f"checkpoint_path missing: {stasy_checkpoint_path}")
            if not stasy_metadata_path.exists():
                missing.append(f"checkpoint_metadata_path missing: {stasy_metadata_path}")
        if config.model == "codi":
            checkpoint_root = Path(config.output_dir) / "ckpt" / config.dataset
            checkpoint_con = checkpoint_root / "model_con.pt"
            checkpoint_dis = checkpoint_root / "model_dis.pt"
            checkpoint_metadata = Path(config.output_dir) / "codi-model-metadata.json"
            checked["checkpoint_con_path"] = str(checkpoint_con)
            checked["checkpoint_dis_path"] = str(checkpoint_dis)
            checked["checkpoint_metadata_path"] = str(checkpoint_metadata)
            if not checkpoint_con.exists():
                missing.append(f"checkpoint_path missing: {checkpoint_con}")
            if not checkpoint_dis.exists():
                missing.append(f"checkpoint_path missing: {checkpoint_dis}")
            if not checkpoint_metadata.exists():
                missing.append(f"checkpoint_metadata_path missing: {checkpoint_metadata}")

    if action == "sample" and config.model == "tabebm":
        allow_gated_model = bool(config.sample.extra.get("allow_gated_model", False))
        checked["allow_gated_model"] = allow_gated_model
        if not allow_gated_model:
            missing.append(
                "sample.extra.allow_gated_model must be true for tabebm sample because TabPFN access is gated"
            )

    if action == "sample" and config.model == "great":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "great_model")
        record_user_checkpoint(checkpoint_path, code_executing=False, allow_directory=True)
        checked["checkpoint_integrity_path"] = str(Path(checkpoint_path) / "great-integrity.json")
        if not (Path(checkpoint_path) / "great-integrity.json").is_file():
            missing.append(f"checkpoint integrity manifest missing: {Path(checkpoint_path) / 'great-integrity.json'}")

    if action == "sample" and config.model == "tabula":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "tabula_model")
        record_user_checkpoint(checkpoint_path, code_executing=False, allow_directory=True)
        checked["checkpoint_integrity_path"] = str(Path(checkpoint_path) / "tabula-integrity.json")
        if not (Path(checkpoint_path) / "tabula-integrity.json").is_file():
            missing.append(f"checkpoint integrity manifest missing: {Path(checkpoint_path) / 'tabula-integrity.json'}")

    if action == "sample" and config.model == "realtabformer":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "realtabformer_model")
        record_user_checkpoint(checkpoint_path, allow_directory=True)
        metadata_path = Path(
            config.sample.extra.get(
                "checkpoint_metadata_path",
                Path(config.output_dir) / "realtabformer-model-metadata.json",
            )
        )
        checked["checkpoint_metadata_path"] = str(metadata_path)
        if not metadata_path.exists():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")

    if action == "sample" and config.model == "tabularargn":
        checkpoint_path = config.sample.checkpoint_path or str(Path(config.output_dir) / "tabularargn_workspace")
        record_user_checkpoint(checkpoint_path, allow_directory=True)
        metadata_path = Path(
            config.sample.extra.get(
                "checkpoint_metadata_path",
                Path(config.output_dir) / "tabularargn-model-metadata.json",
            )
        )
        checked["checkpoint_metadata_path"] = str(metadata_path)
        if not metadata_path.exists():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")

    if config.model == "tabsyn":
        runtime_root = Path(config.output_dir) / "tabsyn-runtime" / "tabsyn"
        vae_ckpt_dir = runtime_root / "vae" / "ckpt" / config.dataset
        diffusion_ckpt_dir = runtime_root / "ckpt" / config.dataset
        if action == "train":
            checked["tabsyn_stage_model"] = "vae_then_diffusion"
        elif action == "sample":
            metadata_path = Path(config.output_dir) / "tabsyn-model-metadata.json"
            checked["tabsyn_metadata"] = str(metadata_path)
            checked["tabsyn_train_z"] = str(vae_ckpt_dir / "train_z.npy")
            checked["tabsyn_decoder"] = str(vae_ckpt_dir / "decoder.pt")
            checked["tabsyn_diffusion_model"] = str(diffusion_ckpt_dir / "model.pt")
            if not (vae_ckpt_dir / "train_z.npy").is_file():
                missing.append(f"tabsyn prerequisite missing: {vae_ckpt_dir / 'train_z.npy'}")
            if not (vae_ckpt_dir / "decoder.pt").is_file():
                missing.append(f"tabsyn prerequisite missing: {vae_ckpt_dir / 'decoder.pt'}")
            if not (diffusion_ckpt_dir / "model.pt").is_file():
                missing.append(f"tabsyn prerequisite missing: {diffusion_ckpt_dir / 'model.pt'}")
            if not metadata_path.is_file():
                missing.append(f"tabsyn training metadata missing: {metadata_path}")

    if action == "sample" and config.model == "tabdiff":
        if config.sample.checkpoint_path is not None:
            record_user_checkpoint(config.sample.checkpoint_path)
        else:
            exp_name = config.sample.extra.get("exp_name", Path(config.output_dir).name)
            checkpoint_root = (
                Path(config.output_dir)
                / "tabdiff-runtime"
                / "tabdiff"
                / "ckpt"
                / config.dataset
                / exp_name
            )
            inferred = sorted(checkpoint_root.glob("best_ema_model*"))
            checked["checkpoint_root"] = str(checkpoint_root)
            if not inferred:
                missing.append(f"tabdiff checkpoint missing under: {checkpoint_root}")

    if action == "sample" and config.model == "tabddpm":
        if config.upstream_config_path is None:
            missing.append("upstream_config_path missing for tabddpm sample")
        else:
            checked["upstream_config_path"] = config.upstream_config_path
            if not Path(config.upstream_config_path).is_file() or Path(config.upstream_config_path).is_symlink():
                missing.append(f"upstream_config_path missing or unsafe: {config.upstream_config_path}")
        metadata_path = Path(config.output_dir) / "tabddpm-model-metadata.json"
        checked["checkpoint_metadata_path"] = str(metadata_path)
        if not metadata_path.is_file():
            missing.append(f"checkpoint_metadata_path missing: {metadata_path}")
        for name in ("model.pt", "model_ema.pt"):
            checkpoint = Path(config.output_dir) / "tabddpm-runtime" / name
            checked[f"tabddpm_{name}_path"] = str(checkpoint)
            if not checkpoint.is_file():
                missing.append(f"tabddpm checkpoint missing: {checkpoint}")

    if action == "train" and config.model == "tabddpm":
        if config.upstream_config_path is None:
            missing.append("upstream_config_path missing for tabddpm train")
        else:
            checked["upstream_config_path"] = config.upstream_config_path
            if not Path(config.upstream_config_path).is_file() or Path(config.upstream_config_path).is_symlink():
                missing.append(f"upstream_config_path missing or unsafe: {config.upstream_config_path}")

    if action == "evaluate":
        sample_path = config.evaluation.extra.get("sample_path") or config.sample.extra.get("sample_path")
        checked["sample_path"] = sample_path
        if not isinstance(sample_path, str) or not Path(sample_path).is_file():
            missing.append(f"sample_path missing: {sample_path}")
        profile_path = config.evaluation.dataset_profile_path
        checked["dataset_profile_path"] = profile_path
        if profile_path is None or not Path(profile_path).is_file():
            missing.append(f"dataset_profile_path missing: {profile_path}")
        reference_path = config.evaluation.reference_path or (
            None if dataset_spec.train_data_path is None else str(dataset_spec.train_data_path)
        )
        checked["reference_path"] = reference_path
        if reference_path is None or not Path(reference_path).is_file():
            missing.append(f"reference_path missing: {reference_path}")
        if config.evaluation.protocol in {"p4-utility", "p5-high-order-privacy"}:
            test_path = config.evaluation.real_test_path or (
                None if dataset_spec.test_data_path is None else str(dataset_spec.test_data_path)
            )
            checked["real_test_path"] = test_path
            if test_path is None or not Path(test_path).is_file():
                missing.append(f"real_test_path missing: {test_path}")

    return {
        "action": action,
        "ready": not missing,
        "checked": checked,
        "missing": missing,
    }
