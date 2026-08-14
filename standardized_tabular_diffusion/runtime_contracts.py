from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    read_json,
    sha256_file,
)
from standardized_tabular_diffusion.interfaces import DatasetSpec

RUN_IDENTITY_FILENAME = ".standardized-run-identity.json"
_DEVICE_PATTERN = re.compile(r"cpu|cuda(?::(\d+))?")


def inspect_regular_file(path: Path | None) -> dict[str, Any]:
    """Return a stable file identity without following unsafe symbolic links."""

    if path is None:
        return {"path": None, "exists": None, "bytes": None, "sha256": None}
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": None,
        "sha256": None,
    }
    if path.is_symlink():
        result["unsafe"] = "symlink"
        return result
    if not path.exists():
        return result
    if not path.is_file():
        result["unsafe"] = "not-a-regular-file"
        return result
    result["bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    return result


def dataset_content_identity(dataset_spec: DatasetSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": dataset_spec.name,
        "task_type": dataset_spec.task_type,
        "column_names": list(getattr(dataset_spec, "column_names", [])),
        "files": {
            "metadata": inspect_regular_file(dataset_spec.metadata_path),
            "train": inspect_regular_file(dataset_spec.train_data_path),
            "validation": inspect_regular_file(dataset_spec.val_data_path),
            "test": inspect_regular_file(dataset_spec.test_data_path),
        },
    }


def require_dataset_content_identity(
    dataset_spec: DatasetSpec,
    expected: dict[str, Any],
) -> dict[str, Any]:
    observed = dataset_content_identity(dataset_spec)
    if observed != expected:
        raise ValueError(
            "Dataset content changed after RunSpec construction; refusing a run whose registered data identity "
            "no longer matches its preflight record."
        )
    unsafe = {
        name: record.get("unsafe")
        for name, record in observed["files"].items()
        if record.get("unsafe") is not None
    }
    if unsafe:
        raise ValueError(f"Dataset identity contains unsafe paths: {unsafe}")
    return observed


def bind_native_dataset_view(dataset_spec: DatasetSpec, native_data_dir: Path) -> dict[str, Any]:
    """Prove that a model-native data directory is byte-identical to the canonical view."""

    if native_data_dir.is_symlink() or not native_data_dir.is_dir():
        raise FileNotFoundError(f"Model-native dataset directory is missing or unsafe: {native_data_dir}")
    native_data_dir = native_data_dir.resolve(strict=True)
    canonical_root = dataset_spec.metadata_path.parent.resolve(strict=True)
    canonical_files = {
        "info.json": dataset_spec.metadata_path,
        "train.csv": dataset_spec.train_data_path,
        "test.csv": dataset_spec.test_data_path,
    }
    if dataset_spec.val_data_path is not None:
        canonical_files["val.csv"] = dataset_spec.val_data_path

    native_array_names = {
        path.name
        for root in (canonical_root, native_data_dir)
        for path in root.glob("*.npy")
        if path.is_file()
    }
    for name in sorted(native_array_names):
        canonical_files[name] = canonical_root / name

    records: dict[str, Any] = {}
    for name, canonical_path in canonical_files.items():
        if canonical_path is None:
            continue
        native_path = native_data_dir / name
        if canonical_path.is_symlink() or native_path.is_symlink():
            raise ValueError(f"Dataset binding refuses symlinked files: {name}")
        if not canonical_path.is_file() or not native_path.is_file():
            raise FileNotFoundError(
                f"Cannot bind model-native dataset file {name!r}: canonical={canonical_path}, native={native_path}"
            )
        canonical_hash = sha256_file(canonical_path)
        native_hash = sha256_file(native_path)
        if canonical_path.stat().st_size != native_path.stat().st_size or canonical_hash != native_hash:
            raise ValueError(
                f"Model-native dataset file differs from the canonical DatasetSpec content: {native_path}"
            )
        records[name] = {
            "canonical_path": str(canonical_path.resolve(strict=True)),
            "native_path": str(native_path.resolve(strict=True)),
            "bytes": native_path.stat().st_size,
            "sha256": native_hash,
        }
    return {
        "schema_version": 1,
        "dataset": dataset_spec.name,
        "canonical_root": str(canonical_root),
        "native_root": str(native_data_dir),
        "files": records,
    }


def _regular_tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Dataset tree is missing or unsafe: {root}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Dataset tree contains a prohibited symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Dataset tree contains a non-regular entry: {path}")
        records[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def materialize_bound_dataset_view(
    dataset_spec: DatasetSpec,
    native_data_dir: Path,
    destination: Path,
) -> dict[str, Any]:
    """Copy a verified native view into run ownership without following links."""

    source_binding = bind_native_dataset_view(dataset_spec, native_data_dir)
    source_manifest = _regular_tree_manifest(native_data_dir)
    if destination.is_symlink():
        raise ValueError(f"Runtime dataset destination must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_dir() or _regular_tree_manifest(destination) != source_manifest:
            raise FileExistsError(f"Runtime dataset destination differs from its bound source: {destination}")
    else:
        destination.mkdir(parents=True)
        for relative in source_manifest:
            source = native_data_dir / Path(relative)
            target = destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    runtime_manifest = _regular_tree_manifest(destination)
    if runtime_manifest != source_manifest:
        raise RuntimeError("Runtime dataset materialization failed its content-identity check.")
    return {
        "schema_version": 1,
        "source_binding": source_binding,
        "source_tree": source_manifest,
        "runtime_binding": bind_native_dataset_view(dataset_spec, destination),
    }


_ACTION_CONTROLS: dict[str, dict[str, set[str]]] = {
    "arf": {
        "train": {
            "alpha",
            "delta",
            "dist",
            "early_stop",
            "max_iters",
            "min_node_size",
            "n_jobs",
            "num_trees",
            "oob",
            "verbose",
        },
        "sample": set(),
    },
    "bn": {
        "train": {
            "epsilon",
            "equivalent_sample_size",
            "max_indegree",
            "max_iter",
            "n_jobs",
            "num_bins",
            "prior_type",
            "scoring_method",
            "tabu_length",
        },
        "sample": set(),
    },
    "ctab-gan": {
        "train": {
            "batch_size",
            "categorical_columns",
            "class_dim",
            "epochs",
            "integer_columns",
            "l2scale",
            "log_columns",
            "mixed_columns",
            "num_channels",
            "num_threads",
            "random_dim",
            "source_dir",
            "test_ratio",
        },
        "sample": {"allow_unsafe_external_checkpoint", "num_threads", "source_dir"},
    },
    "ctab-gan-plus": {
        "train": {
            "batch_size",
            "categorical_columns",
            "class_dim",
            "epochs",
            "general_columns",
            "integer_columns",
            "l2scale",
            "log_columns",
            "mixed_columns",
            "non_categorical_columns",
            "num_channels",
            "num_threads",
            "random_dim",
            "source_dir",
            "test_ratio",
        },
        "sample": {"allow_unsafe_external_checkpoint", "num_threads", "source_dir"},
    },
    "ctgan": {
        "train": {
            "batch_size",
            "discriminator_decay",
            "discriminator_dim",
            "discriminator_lr",
            "discriminator_steps",
            "embedding_dim",
            "epochs",
            "generator_decay",
            "generator_dim",
            "generator_lr",
            "log_frequency",
            "pac",
            "verbose",
        },
        "sample": {"allow_unsafe_external_checkpoint"},
    },
    "great": {
        "train": {
            "batch_size",
            "conditional_col",
            "epochs",
            "float_precision",
            "llm",
            "max_train_rows",
            "num_threads",
            "random_conditional_col",
            "train_kwargs",
        },
        "sample": {
            "conditions",
            "guided_sampling",
            "k",
            "max_length",
            "num_samples",
            "num_threads",
            "random_feature_order",
            "start_col",
            "start_col_dist",
            "temperature",
        },
    },
    "nflow": {
        "train": {
            "batch_size",
            "epochs",
            "hidden_features",
            "learning_rate",
            "num_blocks",
            "num_layers",
            "num_threads",
        },
        "sample": set(),
    },
    "nrgboost": {
        "train": {
            "burn_in",
            "categorical_split_one_vs_all",
            "discretization_types",
            "feature_frac",
            "infer_continuous_ordered_categoricals",
            "infer_fixed_point",
            "infer_ordered_categoricals",
            "initial_samples",
            "initial_uniform_mixture",
            "jit_all",
            "line_search",
            "max_leaves",
            "max_ratio_in_leaf",
            "min_data_in_leaf",
            "min_gain",
            "num_bins",
            "num_chains",
            "num_model_samples",
            "num_threads",
            "num_trees",
            "p_refresh",
            "shrinkage",
            "splitter",
            "training_temperature",
        },
        "sample": {
            "allow_unsafe_external_checkpoint",
            "num_rounds",
            "num_steps",
            "num_threads",
            "temperature",
        },
    },
    "smote": {"train": set(), "sample": {"k_neighbors", "sampling_strategy"}},
    "tabddpm": {"train": set(), "sample": set()},
    "tabdiff": {
        "train": {
            "allow_unstandardized_integer_output",
            "debug",
            "exp_name",
            "gpu",
            "no_wandb",
            "non_learnable_schedule",
            "y_only",
            "deterministic",
        },
        "sample": {
            "allow_unstandardized_integer_output",
            "allow_unsafe_external_checkpoint",
            "debug",
            "exp_name",
            "gpu",
            "no_wandb",
            "non_learnable_schedule",
            "num_runs",
            "report",
            "y_only",
            "deterministic",
        },
    },
    "tabebm": {
        "train": {
            "distance_negative_class",
            "max_data_size",
            "num_threads",
            "sgld_noise_std",
            "sgld_step_size",
            "sgld_steps",
            "starting_point_noise_std",
        },
        "sample": {
            "allow_gated_model",
            "debug",
            "distance_negative_class",
            "max_data_size",
            "num_threads",
            "sgld_noise_std",
            "sgld_step_size",
            "sgld_steps",
            "starting_point_noise_std",
        },
    },
    "tabsds": {"train": {"n_levels", "source_dir"}, "sample": {"source_dir"}},
    "tabsyn": {
        "train": {
            "diffusion_num_epochs",
            "gpu",
            "lambd",
            "max_beta",
            "min_beta",
            "skip_vae_if_present",
            "vae_num_epochs",
        },
        "sample": {"gpu", "steps"},
    },
    "tabula": {
        "train": {
            "batch_size",
            "conditional_col",
            "epochs",
            "llm",
            "max_train_rows",
            "num_threads",
            "source_dir",
            "train_kwargs",
        },
        "sample": {
            "allow_unbounded_sampling",
            "k",
            "max_empty_batches",
            "max_length",
            "num_samples",
            "num_threads",
            "source_dir",
            "start_col",
            "start_col_dist",
            "temperature",
            "timeout_seconds",
        },
    },
    "tvae": {
        "train": {"batch_size", "compress_dims", "decompress_dims", "embedding_dim", "epochs", "l2scale", "loss_factor", "verbose"},
        "sample": {"allow_unsafe_external_checkpoint"},
    },
}


def validate_action_controls(model: str, action: str, controls: dict[str, Any]) -> None:
    """Reject misspelled public controls before an adapter can silently use defaults."""

    model_controls = _ACTION_CONTROLS.get(model)
    if model_controls is None or action == "evaluate":
        return
    allowed = model_controls[action]
    unknown = sorted(set(controls) - allowed)
    if unknown:
        raise ValueError(f"Unsupported {model} {action} controls: {', '.join(unknown)}")


def normalize_device_request(device: str) -> str:
    normalized = device.strip().lower()
    match = _DEVICE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<non-negative index>'")
    return normalized


def require_cpu_device(model: str, device: str) -> None:
    if normalize_device_request(device) != "cpu":
        raise ValueError(f"{model} is CPU-only; use device='cpu'.")


def resolve_torch_training_device(device: str) -> dict[str, Any]:
    """Resolve a Transformers-style trainer request without implicit fallback."""

    requested = normalize_device_request(device)
    if requested == "cpu":
        return {"requested": requested, "trainer_use_cpu": True}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("A CUDA training request requires PyTorch to be installed.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    index = 0 if requested == "cuda" else int(requested.split(":", maxsplit=1)[1])
    if index != 0:
        raise ValueError(
            "In-process trainer adapters currently support only 'cuda' or 'cuda:0'; "
            "select another GPU through CUDA_VISIBLE_DEVICES before starting the process."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA was requested but PyTorch reports no visible device.")
    return {"requested": requested, "trainer_use_cpu": False}


def observe_torch_model_device(model: Any, expected: str) -> str:
    core = getattr(model, "model", model)
    observed = getattr(core, "device", None)
    if observed is None:
        try:
            observed = next(core.parameters()).device
        except (AttributeError, StopIteration, TypeError) as exc:
            raise RuntimeError("Could not observe the trained model device; refusing an unverified run.") from exc
    normalized = str(observed).lower()
    expected_kind = "cpu" if expected == "cpu" else "cuda"
    if not normalized.startswith(expected_kind):
        raise RuntimeError(
            f"Trainer device mismatch: requested {expected!r}, observed {normalized!r}."
        )
    return normalized


def _canonical_json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def claim_output_identity(output_dir: Path, *, model: str, dataset: str, actions: dict[str, Any]) -> Path:
    """Claim an output directory for one immutable action/configuration identity."""

    if output_dir.is_symlink():
        raise ValueError(f"output_dir must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = output_dir.resolve(strict=True)
    identity_path = resolved / RUN_IDENTITY_FILENAME
    requested = {
        "schema_version": 1,
        "model": model,
        "dataset": dataset,
        "output_dir": str(resolved),
        "actions": {
            action: {"identity": identity, "sha256": _canonical_json_digest(identity)}
            for action, identity in sorted(actions.items())
        },
    }
    if identity_path.exists():
        if identity_path.is_symlink() or not identity_path.is_file():
            raise ValueError(f"Run identity path is unsafe: {identity_path}")
        current = read_json(identity_path)
        if (
            not isinstance(current, dict)
            or current.get("schema_version") != 1
            or current.get("model") != model
            or current.get("dataset") != dataset
            or current.get("output_dir") != str(resolved)
            or not isinstance(current.get("actions"), dict)
        ):
            raise FileExistsError(f"output_dir belongs to a different or malformed run identity: {resolved}")
        merged = dict(current["actions"])
        for action, record in requested["actions"].items():
            previous = merged.get(action)
            if previous is not None and previous != record:
                raise FileExistsError(
                    f"Refusing to reuse output_dir for a different {action} identity: {resolved}. "
                    "Use a new output directory for each seed/configuration."
                )
            merged[action] = record
        requested["actions"] = dict(sorted(merged.items()))
    else:
        existing_entries = {path.name for path in resolved.iterdir()}
        orchestration_only = existing_entries == {".orchestration"}
        orchestration_path = resolved / ".orchestration"
        safe_orchestration = (
            orchestration_only
            and orchestration_path.is_dir()
            and not orchestration_path.is_symlink()
        )
        if existing_entries and not safe_orchestration:
            raise FileExistsError(
                f"Refusing to adopt a non-empty output_dir without {RUN_IDENTITY_FILENAME}: {resolved}"
            )
    atomic_write_json(identity_path, requested)
    return identity_path
