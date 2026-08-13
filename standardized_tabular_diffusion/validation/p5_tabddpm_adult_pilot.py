"""Run the declared TabDDPM/Adult non-identity P5 pilot on Windows.

The pilot trains one official-configuration TabDDPM checkpoint and samples it
with three generation seeds. P5 then evaluates each decoded table with its
independent, immutable five-seed evaluator panel.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tomli_w

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file
from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.tabddpm import TabDDPMAdapter
from standardized_tabular_diffusion.validation.tabddpm import verify_sources

PROTOCOL_ID = "p5-tabddpm-adult-generator-pilot-v1"
MODEL_ID = "tabddpm"
DATASET_ID = "adult"
GENERATION_SEEDS = (0, 1, 2)
EVALUATOR_SEEDS = (0, 1, 2, 3, 4)
BASE_CONFIG = Path("TabDDPM-main/exp/adult/ddpm_cb_best/config.toml")
DATASET_PROFILE = Path("configs/datasets/adult-uci-2-v1.json")
REAL_TRAIN = Path("TabDiff-main/data/adult/train.csv")
REAL_TEST = Path("TabDiff-main/data/adult/test.csv")


class PilotError(RuntimeError):
    """Raised when the pilot's predeclared identity or execution fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_text(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _git(repo_root: Path, *args: str) -> str:
    return _run_text(["git", *args], cwd=repo_root)


def _require_exact_seed_panel(seeds: Iterable[int]) -> tuple[int, ...]:
    resolved = tuple(seeds)
    if resolved != GENERATION_SEEDS:
        raise PilotError(f"Pilot generation seeds are frozen as {GENERATION_SEEDS}, got {resolved}")
    return resolved


def _artifact_relative_path(path: Path, output_root: Path) -> str:
    """Return a portable artifact path without retaining host-specific prefixes."""

    resolved_root = output_root.resolve()
    try:
        relative = path.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise PilotError(f"Pilot artifact escaped the declared output root: {path}") from error
    if relative == Path("."):
        raise PilotError("Pilot evidence must identify a concrete artifact below the output root")
    return relative.as_posix()


def _load_declared_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_path = repo_root / BASE_CONFIG
    profile_path = repo_root / DATASET_PROFILE
    for path in (base_path, profile_path, repo_root / REAL_TRAIN, repo_root / REAL_TEST):
        if not path.is_file():
            raise PilotError(f"Required pilot input is missing: {path}")
    with base_path.open("rb") as stream:
        base = tomllib.load(stream)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected_base = {
        "model_type": "mlp",
        "num_numerical_features": 6,
        "device": "cuda:0",
        "num_timesteps": 100,
        "steps": 30000,
        "batch_size": 4096,
        "sample_batch_size": 10000,
        "normalization": "quantile",
    }
    observed_base = {
        "model_type": base.get("model_type"),
        "num_numerical_features": base.get("num_numerical_features"),
        "device": base.get("device"),
        "num_timesteps": base.get("diffusion_params", {}).get("num_timesteps"),
        "steps": base.get("train", {}).get("main", {}).get("steps"),
        "batch_size": base.get("train", {}).get("main", {}).get("batch_size"),
        "sample_batch_size": base.get("sample", {}).get("batch_size"),
        "normalization": base.get("train", {}).get("T", {}).get("normalization"),
    }
    if observed_base != expected_base:
        raise PilotError(f"Official Adult base configuration drifted: {observed_base}")
    if "seed" in base["train"]["main"]:
        raise PilotError("Official Adult training config unexpectedly overrides the source-default seed")
    if profile.get("dataset_id") != DATASET_ID or profile.get("status") != "reviewed":
        raise PilotError("Adult Dataset Profile is not the reviewed pilot identity")
    return base, profile


def _column_groups(profile: dict[str, Any]) -> tuple[list[str], list[str], str]:
    expected = profile["table_contract"]["canonical_column_order"]
    by_name = {item["name"]: item for item in profile["columns"]}
    numerical = [name for name in expected if by_name[name]["semantic_type"] in {"integer", "continuous"}]
    categorical = [
        name
        for name in expected
        if by_name[name]["semantic_type"] in {"categorical", "boolean", "string"}
        and "primary_target" not in by_name[name]["roles"]
    ]
    targets = [name for name in expected if "primary_target" in by_name[name]["roles"]]
    if numerical != ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]:
        raise PilotError(f"Adult numerical model view drifted: {numerical}")
    if len(targets) != 1 or len(categorical) != 8:
        raise PilotError("Adult categorical or target model view drifted")
    return numerical, categorical, targets[0]


def _prepare_upstream_data(
    repo_root: Path,
    output_root: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Create an ignored numeric/categorical view without changing source tables.

    TabDDPM's unchanged loader requires a validation array even though training
    and sampling do not consume it. A one-row mirror of real train satisfies
    that interface; the complete 32,561-row benchmark training split remains
    the only fit input.
    """

    train_path = repo_root / REAL_TRAIN
    test_path = repo_root / REAL_TEST
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    numerical, categorical, target = _column_groups(profile)
    expected = profile["table_contract"]["canonical_column_order"]
    if list(train.columns) != expected or list(test.columns) != expected:
        raise PilotError("Adult canonical CSV column order differs from the reviewed profile")
    if len(train) != 32561 or len(test) != 16281:
        raise PilotError(f"Adult split sizes drifted: train={len(train)}, test={len(test)}")
    label_values = profile["columns"][-1]["valid_domain"]["values"]
    label_to_index = {label: index for index, label in enumerate(label_values)}
    if set(train[target]) != set(label_to_index) or set(test[target]) != set(label_to_index):
        raise PilotError("Adult target labels differ from the reviewed profile")

    data_dir = output_root / "runtime-input" / "adult"
    data_dir.mkdir(parents=True, exist_ok=False)
    frames = {"train": train, "val": train.iloc[[0]].copy(), "test": test}
    for split, frame in frames.items():
        np.save(data_dir / f"X_num_{split}.npy", frame[numerical].to_numpy(dtype=np.float64), allow_pickle=False)
        np.save(data_dir / f"X_cat_{split}.npy", frame[categorical].astype(str).to_numpy(dtype=str), allow_pickle=False)
        labels = frame[target].map(label_to_index)
        if labels.isna().any():
            raise PilotError(f"Adult {split} contains an unmapped target")
        np.save(data_dir / f"y_{split}.npy", labels.to_numpy(dtype=np.int64), allow_pickle=False)
    info = {
        "name": DATASET_ID,
        "task_type": "binclass",
        "n_classes": len(label_values),
        "train_size": len(train),
        "val_size": 1,
        "test_size": len(test),
        "validation_role": "one-row-unused-loader-compatibility-mirror",
    }
    atomic_write_json(data_dir / "info.json", info)
    return {
        "path": str(data_dir),
        "train_rows": len(train),
        "test_rows": len(test),
        "validation_rows": 1,
        "validation_used_for_fit": False,
        "train_csv_sha256": sha256_file(train_path),
        "test_csv_sha256": sha256_file(test_path),
        "label_mapping": label_to_index,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "target_column": target,
    }


def _runtime_config(
    base: dict[str, Any],
    *,
    parent_dir: Path,
    data_dir: Path,
    sample_seed: int,
    num_samples: int,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["parent_dir"] = str(parent_dir)
    config["real_data_path"] = str(data_dir)
    config["device"] = "cuda:0"
    config["sample"]["num_samples"] = num_samples
    config["sample"]["seed"] = sample_seed
    return config


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(payload), encoding="utf-8", newline="\n")


def _decode_sample(
    parent_dir: Path,
    seed_dir: Path,
    profile: dict[str, Any],
    label_mapping: dict[str, int],
) -> dict[str, Any]:
    numerical, categorical, target = _column_groups(profile)
    raw_paths = {
        "numerical": parent_dir / "X_num_train.npy",
        "categorical": parent_dir / "X_cat_train.npy",
        "target": parent_dir / "y_train.npy",
    }
    if any(not path.is_file() for path in raw_paths.values()):
        raise PilotError(f"TabDDPM did not emit its complete sample arrays: {raw_paths}")
    x_num = np.load(raw_paths["numerical"], allow_pickle=False)
    x_cat = np.load(raw_paths["categorical"], allow_pickle=True)
    y = np.load(raw_paths["target"], allow_pickle=False).reshape(-1)
    row_counts = {len(x_num), len(x_cat), len(y)}
    if row_counts != {32561} or x_num.shape[1] != len(numerical) or x_cat.shape[1] != len(categorical):
        raise PilotError(f"Unexpected TabDDPM sample shapes: {x_num.shape}, {x_cat.shape}, {y.shape}")
    if not np.isfinite(x_num).all():
        raise PilotError("TabDDPM emitted non-finite numerical values")

    by_name = {item["name"]: item for item in profile["columns"]}
    decoded: dict[str, Any] = {}
    integer_decoding: dict[str, Any] = {}
    for index, name in enumerate(numerical):
        values = np.asarray(x_num[:, index], dtype=np.float64)
        if by_name[name]["semantic_type"] == "integer":
            rounded = np.rint(values)
            delta = np.abs(values - rounded)
            decoded[name] = rounded.astype(np.int64)
            integer_decoding[name] = {
                "policy": "nearest-integer-ties-to-even-at-adapter-decoding-boundary",
                "changed_rows": int(np.count_nonzero(delta)),
                "maximum_absolute_delta": float(delta.max(initial=0.0)),
            }
        else:
            decoded[name] = values
    for index, name in enumerate(categorical):
        values = pd.Series(x_cat[:, index], dtype="string")
        allowed = set(by_name[name]["valid_domain"]["values"])
        unexpected = sorted(set(values.dropna().astype(str)) - allowed)
        if values.isna().any() or unexpected:
            raise PilotError(f"TabDDPM emitted invalid categorical values for {name}: {unexpected[:5]}")
        decoded[name] = values.astype(str).to_numpy()
    index_to_label = {index: label for label, index in label_mapping.items()}
    if not np.isfinite(y.astype(float)).all() or not np.equal(y, np.rint(y)).all():
        raise PilotError("TabDDPM emitted a non-integral classification target")
    target_indices = np.rint(y).astype(np.int64)
    if not set(np.unique(target_indices)).issubset(index_to_label):
        raise PilotError(f"TabDDPM emitted unknown target indices: {sorted(set(target_indices) - set(index_to_label))}")
    decoded[target] = [index_to_label[index] for index in target_indices]

    columns = profile["table_contract"]["canonical_column_order"]
    frame = pd.DataFrame(decoded).loc[:, columns]
    synthetic_path = seed_dir / "synthetic.csv"
    synthetic_path.parent.mkdir(parents=True, exist_ok=False)
    frame.to_csv(synthetic_path, index=False, lineterminator="\n")

    raw_dir = seed_dir / "raw-upstream-arrays"
    raw_dir.mkdir()
    raw_hashes: dict[str, str] = {}
    for key, source in raw_paths.items():
        destination = raw_dir / source.name
        shutil.copy2(source, destination)
        raw_hashes[key] = sha256_file(destination)
    for name in ("X_num_unnorm.npy", "X_cat_unnorm.npy"):
        source = parent_dir / name
        if source.is_file():
            destination = raw_dir / source.name
            shutil.copy2(source, destination)
            raw_hashes[name] = sha256_file(destination)
    return {
        "synthetic_path": str(synthetic_path),
        "synthetic_sha256": sha256_file(synthetic_path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "raw_array_sha256": raw_hashes,
        "integer_decoding": integer_decoding,
        "synthetic_repair_applied_by_evaluator": False,
    }


def _evaluate_p5(
    repo_root: Path,
    *,
    seed: int,
    synthetic_path: Path,
    bundle_dir: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "standardized_tabular_diffusion.cli",
        "evaluate-table",
        "--protocol",
        "p5-high-order-privacy",
        "--reference",
        str(repo_root / REAL_TRAIN),
        "--real-test",
        str(repo_root / REAL_TEST),
        "--synthetic",
        str(synthetic_path),
        "--dataset-profile",
        str(repo_root / DATASET_PROFILE),
        "--output",
        str(bundle_dir),
        "--expected-rows",
        "32561",
        "--comparison-track",
        "native",
        "--model-id",
        MODEL_ID,
        "--generation-seed",
        str(seed),
        "--evaluator-seeds",
        ",".join(map(str, EVALUATOR_SEEDS)),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _dependency_versions() -> dict[str, str]:
    distributions = (
        "standardized-tabular-diffusion",
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "sdmetrics",
        "rtdl",
        "category-encoders",
        "tomli-w",
    )
    return {name: importlib.metadata.version(name) for name in distributions}


def _hardware() -> dict[str, Any]:
    import torch

    if platform.system() != "Windows" or sys.version_info[:2] != (3, 11):
        raise PilotError("This pilot requires native Windows and Python 3.11")
    if not torch.cuda.is_available():
        raise PilotError("This pilot requires a CUDA GPU")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "nvidia_smi": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            cwd=Path.cwd(),
        ).splitlines(),
    }


def run_pilot(repo_root: Path, output_root: Path, evidence_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    evidence_path = evidence_path.resolve()
    _require_exact_seed_panel(GENERATION_SEEDS)
    if output_root.exists():
        raise PilotError(f"Refusing to overwrite pilot output: {output_root}")
    output_root.mkdir(parents=True)
    started_at = _utc_now()
    hardware = _hardware()
    source_integrity = verify_sources(repo_root)
    base, profile = _load_declared_inputs(repo_root)
    dataset = _prepare_upstream_data(repo_root, output_root, profile)
    parent_dir = output_root / "upstream-checkpoint"
    parent_dir.mkdir()
    config_dir = output_root / "configs"
    manifest_dir = output_root / "adapter-manifests"
    base_config_path = repo_root / BASE_CONFIG

    print(f"[{_utc_now()}] Training TabDDPM once with the official Adult configuration", flush=True)
    training_config = _runtime_config(
        base,
        parent_dir=parent_dir,
        data_dir=Path(dataset["path"]),
        sample_seed=GENERATION_SEEDS[0],
        num_samples=dataset["train_rows"],
    )
    training_config_path = config_dir / "train.toml"
    _write_toml(training_config_path, training_config)
    adapter = TabDDPMAdapter(repo_root)
    adapter.train(
        RunSpec(
            model=MODEL_ID,
            dataset=DATASET_ID,
            output_dir=manifest_dir / "train",
            device="cuda:0",
            seed=0,
            upstream_config_path=training_config_path,
        )
    )
    checkpoint_paths = (parent_dir / "model.pt", parent_dir / "model_ema.pt")
    if any(not path.is_file() for path in checkpoint_paths):
        raise PilotError("TabDDPM training did not emit both declared checkpoints")
    checkpoint_hashes = {path.name: sha256_file(path) for path in checkpoint_paths}

    seed_results: list[dict[str, Any]] = []
    for seed in GENERATION_SEEDS:
        print(f"[{_utc_now()}] Sampling and evaluating generation seed {seed}", flush=True)
        seed_dir = output_root / f"seed-{seed}"
        sample_config = _runtime_config(
            base,
            parent_dir=parent_dir,
            data_dir=Path(dataset["path"]),
            sample_seed=seed,
            num_samples=dataset["train_rows"],
        )
        sample_config_path = config_dir / f"sample-seed-{seed}.toml"
        _write_toml(sample_config_path, sample_config)
        adapter.sample(
            RunSpec(
                model=MODEL_ID,
                dataset=DATASET_ID,
                output_dir=manifest_dir / f"sample-seed-{seed}",
                device="cuda:0",
                seed=seed,
                num_samples=dataset["train_rows"],
                upstream_config_path=sample_config_path,
            )
        )
        decoded = _decode_sample(parent_dir, seed_dir, profile, dataset["label_mapping"])
        bundle_dir = seed_dir / "p5-bundle"
        _evaluate_p5(
            repo_root,
            seed=seed,
            synthetic_path=Path(decoded["synthetic_path"]),
            bundle_dir=bundle_dir,
        )
        summary = json.loads((bundle_dir / "summary.json").read_text(encoding="utf-8"))
        details = json.loads((bundle_dir / "artifacts" / "p5-details.json").read_text(encoding="utf-8"))
        if (
            summary.get("terminal_status") != "success"
            or summary.get("validity", {}).get("structural_gate") != "passed"
        ):
            raise PilotError(f"P5 bundle for generation seed {seed} did not finalize successfully")
        decoded_evidence = dict(decoded)
        decoded_evidence["synthetic_path"] = _artifact_relative_path(Path(decoded["synthetic_path"]), output_root)
        seed_results.append(
            {
                "generation_seed": seed,
                "training_checkpoint_reused": True,
                "sample_config_sha256": sha256_file(sample_config_path),
                "decoded_sample": decoded_evidence,
                "bundle_path": _artifact_relative_path(bundle_dir, output_root),
                "bundle_checksums_sha256": sha256_file(bundle_dir / "checksums.sha256"),
                "request_fingerprint": summary["identity"]["request_fingerprint"],
                "metric_state_counts": summary["metric_state_counts"],
                "denominator_counts": summary["denominator_counts"],
                "high_order_fidelity": summary["dimensions"]["high-order-fidelity"],
                "privacy_risk": summary["dimensions"]["privacy-risk"],
                "claim_boundary": details["claim_boundary"],
                "excluded_metrics": details["excluded_metrics"],
            }
        )

    git_status = _git(repo_root, "status", "--short")
    retained_sources = [
        Path("standardized_tabular_diffusion/validation/p5_tabddpm_adult_pilot.py"),
        Path("standardized_tabular_diffusion/compat/tabddpm_sklearn/sitecustomize.py"),
        Path("standardized_tabular_diffusion/models/tabddpm.py"),
        Path("standardized_tabular_diffusion/evaluation/high_order_privacy.py"),
        Path("standardized_tabular_diffusion/resources/evaluation/evaluators/p5-high-order-privacy-v1.json"),
        DATASET_PROFILE,
        BASE_CONFIG,
    ]
    evidence = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "repository": {
            "head": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "working_tree_dirty": bool(git_status),
            "execution_source_sha256": {
                str(path).replace("\\", "/"): sha256_file(repo_root / path) for path in retained_sources
            },
        },
        "model": {
            "model_id": MODEL_ID,
            "upstream_commit": source_integrity["upstream_commit"],
            "base_config": str(BASE_CONFIG).replace("\\", "/"),
            "base_config_sha256": sha256_file(base_config_path),
            "training_runs": 1,
            "training_seed": 0,
            "generation_seeds": list(GENERATION_SEEDS),
            "checkpoint_reused_across_generation_seeds": True,
            "checkpoint_sha256": checkpoint_hashes,
            "configuration_policy": "official-ddpm-cb-best-values-with-only-path-device-row-count-and-generation-seed-runtime-bindings",
            "runtime_compatibility": (
                "adapter-only integral-float-to-equal-integer QuantileTransformer.subsample bridge; "
                "upstream source and estimator behavior otherwise unchanged"
            ),
            "source_integrity": source_integrity,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "profile": str(DATASET_PROFILE).replace("\\", "/"),
            "profile_sha256": sha256_file(repo_root / DATASET_PROFILE),
            **dataset,
            "path": _artifact_relative_path(Path(dataset["path"]), output_root),
        },
        "evaluation": {
            "protocol_id": "p5-high-order-privacy",
            "protocol_version": "0.1.0",
            "comparison_track": "native",
            "evaluator_seeds_per_generated_table": list(EVALUATOR_SEEDS),
            "overall_fidelity_score": None,
            "overall_privacy_score": None,
            "formal_privacy_guarantee": False,
            "attribute_inference": "excluded-pending-approved-dataset-specific-roles-and-threat-model",
        },
        "environment": {"hardware": hardware, "dependencies": _dependency_versions()},
        "seed_results": seed_results,
        "assertions": {
            "one_model": True,
            "one_dataset": True,
            "three_generation_seeds": len(seed_results) == 3,
            "all_structural_gates_passed": True,
            "all_p5_bundles_finalized": True,
            "five_evaluator_seeds_preserved_per_table": True,
            "no_overall_score_emitted": True,
            "no_formal_privacy_guarantee_claimed": True,
            "official_results_admitted": False,
            "protocol_frozen_by_this_pilot": False,
        },
        "claim_boundary": (
            "Exploratory non-identity generator pilot for TabDDPM on the reviewed Adult split. "
            "It validates real generator execution and diagnostic P5 behavior only; it does not freeze P5, "
            "admit Official Results, establish a formal privacy guarantee, or generalize to other models, "
            "datasets, platforms, configurations, or random seeds."
        ),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(evidence_path, evidence)
    print(f"[{_utc_now()}] Evidence written to {evidence_path}", flush=True)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence = run_pilot(args.repo_root, args.output_root, args.evidence_path)
    print(json.dumps({"status": evidence["status"], "protocol_id": evidence["protocol_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
