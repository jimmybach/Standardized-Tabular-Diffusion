"""Executable native-parity protocol for the official REaLTabFormer package."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
import traceback
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models._runtime import disable_torchvision_for_transformers
from standardized_tabular_diffusion.models.realtabformer import (
    EXPECTED_RUNTIME_VERSIONS,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    UPSTREAM_TAG,
    UPSTREAM_TREE,
    REaLTabFormerAdapter,
    verify_realtabformer_distribution,
)

PROTOCOL_ID = "realtabformer-official-package-parity-v1"
WHEEL_FILENAME = "realtabformer-0.2.4-py3-none-any.whl"
WHEEL_SHA256 = "852436c5c82a0bf470ca7e9063e5a4f3e250b3ff5b9c8f6c50113c1e9ba76486"
WHEEL_BYTES = 49_890
LICENSE_SHA256 = "fb11fe9573168aa7f96555f0e7cafa8c9a8b44089646692626cf12fb95bf9db4"
EXPECTED_ARCHIVE_MEMBERS = 17
EXPECTED_RECORD_ROWS = 17
EXPECTED_RECORD_HASHES = 16
TRAIN_ROWS = 24
SAMPLE_ROWS = 7
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
TINY_TABULAR_CONFIG = {
    "n_layer": 1,
    "n_head": 1,
    "n_embd": 8,
    "n_inner": 32,
    "n_positions": 128,
    "n_ctx": 128,
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
}
TRAINING_CONTROLS = {
    "epochs": 1,
    "batch_size": 4,
    "training_args": {
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "learning_rate": 0.0005,
        "disable_tqdm": True,
        "report_to": "none",
    },
    "num_bootstrap": 0,
    "n_critic": 0,
    "tabular_config": TINY_TABULAR_CONFIG,
}
FIT_CONTROLS = {
    "device": "cpu",
    "num_bootstrap": 0,
    "frac": 0.165,
    "frac_max_data": 10_000,
    "qt_max": 0.05,
    "qt_max_default": 0.05,
    "qt_interval": 100,
    "qt_interval_unique": 100,
    "quantile": 0.95,
    "n_critic": 0,
    "n_critic_stop": 2,
    "gen_rounds": 3,
    "sensitivity_max_col_nums": 20,
    "use_ks": False,
    "full_sensitivity": False,
    "sensitivity_orig_frac_multiple": 4,
    "orig_samples_rounds": 5,
    "load_from_best_mean_sensitivity": False,
    "save_full_every_epoch": 5,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_record_hash(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _load_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "standardized_tabular_diffusion" / "resources" / "upstream" / "realtabformer-wheel-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_wheel(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(f"REaLTabFormer wheel must be a regular non-symlinked file: {wheel_path}")
    if wheel_path.name != WHEEL_FILENAME:
        raise ValueError(f"Expected wheel {WHEEL_FILENAME!r}, observed {wheel_path.name!r}")
    if wheel_path.stat().st_size != WHEEL_BYTES or _sha256_file(wheel_path) != WHEEL_SHA256:
        raise ValueError("REaLTabFormer wheel bytes or SHA-256 differ from the locked official release")

    manifest = _load_manifest(repo_root)
    expected_files = {record["path"]: record for record in manifest["installed_files"]}
    dist_info = f"realtabformer-{PACKAGE_VERSION}.dist-info"
    record_name = f"{dist_info}/RECORD"
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        for name in names:
            archive_path = PurePosixPath(name)
            if archive_path.is_absolute() or ".." in archive_path.parts or "\\" in name:
                raise ValueError(f"Unsafe path in REaLTabFormer wheel: {name!r}")
        metadata = Parser().parsestr(archive.read(f"{dist_info}/METADATA").decode("utf-8"))
        if metadata.get("Name") != PACKAGE_NAME or metadata.get("Version") != PACKAGE_VERSION:
            raise ValueError("REaLTabFormer wheel metadata does not match the locked package identity")
        if metadata.get("Requires-Python") != ">=3.8":
            raise ValueError("REaLTabFormer wheel Python requirement differs from the audited release")
        if metadata.get_all("Requires-Dist", []) != [
            "datasets (>=2.6.1)",
            "numpy (>=1.21.6)",
            "pandas (>=1.3.5)",
            "scikit-learn (>=1.0.2)",
            "shapely (>=1.8.5.post1)",
            "tqdm (>=4.64.1)",
            "transformers[sentencepiece,torch] (>=4.46.0)",
        ]:
            raise ValueError("REaLTabFormer wheel dependencies differ from the audited release")
        wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
        if "Root-Is-Purelib: true" not in wheel_metadata or "Tag: py3-none-any" not in wheel_metadata:
            raise ValueError("REaLTabFormer wheel type or compatibility tag differs from the release lock")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        hashed_rows = [row for row in rows if row[1]]
        if len(names) != EXPECTED_ARCHIVE_MEMBERS or len(rows) != EXPECTED_RECORD_ROWS:
            raise ValueError("REaLTabFormer wheel archive or RECORD member count differs from the lock")
        if len(hashed_rows) != EXPECTED_RECORD_HASHES:
            raise ValueError("REaLTabFormer wheel RECORD hash count differs from the lock")
        if {row[0] for row in hashed_rows} != set(expected_files):
            raise ValueError("REaLTabFormer wheel RECORD paths differ from the audited manifest")
        for path, encoded_hash, encoded_size in hashed_rows:
            payload = archive.read(path)
            expected = expected_files[path]
            if int(encoded_size) != expected["bytes"] or len(payload) != expected["bytes"]:
                raise ValueError(f"REaLTabFormer wheel member size mismatch: {path}")
            observed_digest = hashlib.sha256(payload).digest()
            if observed_digest != _decode_record_hash(encoded_hash.partition("=")[2]):
                raise ValueError(f"REaLTabFormer RECORD checksum mismatch: {path}")
            if observed_digest.hex() != expected["sha256"]:
                raise ValueError(f"REaLTabFormer manifest checksum mismatch: {path}")
        license_hash = _sha256_bytes(archive.read(f"{dist_info}/LICENSE"))
    if license_hash != LICENSE_SHA256:
        raise ValueError("REaLTabFormer wheel LICENSE differs from the locked MIT license")
    return {
        "filename": wheel_path.name,
        "bytes": wheel_path.stat().st_size,
        "sha256": _sha256_file(wheel_path),
        "archive_members": len(names),
        "record_rows": len(rows),
        "record_hashed_files": len(hashed_rows),
        "license": "MIT",
        "license_sha256": license_hash,
    }


def _verify_package(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    return {
        "authority": "method-author",
        "distribution_form": "package",
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "wheel": _verify_wheel(repo_root, wheel_path.resolve()),
        "installed_distribution": verify_realtabformer_distribution(repo_root),
    }


def _environment(repo_root: Path) -> dict[str, str]:
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("REaLTabFormer formal validation requires Linux and Python 3.11")
    observed = verify_realtabformer_distribution(repo_root)["runtime_versions"]
    if observed != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(f"REaLTabFormer runtime differs from the frozen validation environment: {observed}")
    return {"platform": platform.platform(), "python": platform.python_version(), **observed}


def _write_fixture(case_root: Path, variant: str) -> tuple[DatasetSpec, dict[str, Any]]:
    row = np.arange(TRAIN_ROWS)
    first = np.round(np.sin(row / 3.0) + row * 0.07, 4)
    second = ((row * 7) % 19).astype(int)
    group = np.asarray([f"g{value % 3}" for value in row])
    if variant == "binary":
        target: Any = ((row + row // 3) % 2).astype(int)
    elif variant == "multiclass":
        target = ((row * 2 + row // 2) % 3).astype(int)
    elif variant == "regression":
        target = np.round(first * 0.4 - second * 0.03 + (row % 4) * 0.08, 4)
    else:
        raise ValueError(f"Unknown REaLTabFormer validation variant: {variant}")
    frame = pd.DataFrame({"first": first, "second": second, "group": group, "target": target})
    fixture_root = case_root / "fixture"
    fixture_root.mkdir(parents=True)
    train_path = fixture_root / "train.csv"
    metadata_path = fixture_root / "info.json"
    frame.to_csv(train_path, index=False)
    dataset_name = f"realtabformer_parity_{variant}"
    metadata_path.write_text(
        json.dumps(
            {
                "name": dataset_name,
                "task_type": "regression" if variant == "regression" else "classification",
                "column_names": frame.columns.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_spec = DatasetSpec(
        name=dataset_name,
        task_type="regression" if variant == "regression" else "classification",
        column_names=frame.columns.tolist(),
        numerical_columns=["first", "second"],
        categorical_columns=["group"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
    )
    return dataset_spec, {
        "variant": variant,
        "task_type": dataset_spec.task_type,
        "training_rows": len(frame),
        "requested_sample_rows": SAMPLE_ROWS,
        "columns": frame.columns.tolist(),
        "numerical_features": 2,
        "categorical_features": 1,
        "target_features": 1,
        "missing_values": 0,
    }


def _seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.set_num_threads(1)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _coerced_training_frame(dataset_spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(dataset_spec.train_data_path)
    frame["group"] = frame["group"].astype(str)
    if dataset_spec.task_type == "classification":
        frame["target"] = frame["target"].astype(str)
    return frame[dataset_spec.column_names].copy()


def _constructor_kwargs(root: Path, seed: int, config_class: type[Any]) -> dict[str, Any]:
    runtime = root / "realtabformer_runtime"
    return {
        "model_type": "tabular",
        "tabular_config": config_class(**TINY_TABULAR_CONFIG),
        "checkpoints_dir": str(runtime / "checkpoints"),
        "samples_save_dir": str(runtime / "samples"),
        "full_save_dir": str(runtime / "full-save"),
        "epochs": 1,
        "batch_size": 4,
        "random_state": seed,
        "train_size": 1.0,
        "early_stopping_patience": 5,
        "early_stopping_threshold": 0.0,
        "mask_rate": 0.0,
        "numeric_nparts": 1,
        "numeric_precision": 4,
        "numeric_max_len": 10,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "learning_rate": 0.0005,
        "disable_tqdm": True,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 0,
        "use_cpu": True,
    }


def _one_saved_model(root: Path) -> Path:
    candidates = sorted(path for path in root.glob("id*") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one REaLTabFormer saved model directory, observed {len(candidates)}")
    return candidates[0]


def _run_native(
    output_dir: Path,
    dataset_spec: DatasetSpec,
    seed: int,
) -> tuple[Path, pd.DataFrame]:
    import torch

    with disable_torchvision_for_transformers():
        from realtabformer import REaLTabFormer
        from transformers import GPT2Config

    output_dir.mkdir(parents=True)
    frame = _coerced_training_frame(dataset_spec)
    _seed(seed, torch)
    model = REaLTabFormer(**_constructor_kwargs(output_dir, seed, GPT2Config))
    model.fit(frame, **FIT_CONTROLS)
    model.full_save_dir = str(model.full_save_dir)
    model_root = output_dir / "realtabformer_model"
    model.save(str(model_root))
    saved_model = _one_saved_model(model_root)

    loaded = REaLTabFormer.load_from_dir(path=str(saved_model))
    _seed(seed, torch)
    raw_sample = loaded.sample(n_samples=SAMPLE_ROWS, device="cpu")
    return saved_model, raw_sample


class _CapturingAdapter(REaLTabFormerAdapter):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root)
        self.captured_sample: pd.DataFrame | None = None

    def _write_dataframe_csv(self, frame: object, path: Path) -> None:
        if path.name == "samples.csv":
            self.captured_sample = frame.copy()  # type: ignore[attr-defined]
        super()._write_dataframe_csv(frame, path)


def _run_adapter(
    repo_root: Path,
    output_dir: Path,
    dataset_spec: DatasetSpec,
    seed: int,
) -> tuple[Path, pd.DataFrame, Path, dict[str, Any]]:
    adapter = _CapturingAdapter(repo_root)
    common = {"dataset_spec": dataset_spec.to_dict()}
    adapter.train(
        RunSpec(
            model="realtabformer",
            dataset=dataset_spec.name,
            output_dir=output_dir,
            device="cpu",
            seed=seed,
            extra={**common, **TRAINING_CONTROLS},
        )
    )
    sample_bundle = adapter.sample(
        RunSpec(
            model="realtabformer",
            dataset=dataset_spec.name,
            output_dir=output_dir,
            device="cpu",
            seed=seed,
            num_samples=SAMPLE_ROWS,
            extra=common,
        )
    )
    if adapter.captured_sample is None or sample_bundle.generated_sample_path is None:
        raise RuntimeError("REaLTabFormer adapter did not expose its generated sample")
    model_dir = _one_saved_model(output_dir / "realtabformer_model")
    metadata = json.loads((output_dir / REaLTabFormerAdapter.metadata_filename).read_text(encoding="utf-8"))
    return model_dir, adapter.captured_sample, sample_bundle.generated_sample_path, metadata


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_comparison(native_dir: Path, adapter_dir: Path) -> dict[str, Any]:
    import torch

    native_path = native_dir / "rtf_model.pt"
    adapter_path = adapter_dir / "rtf_model.pt"
    native = torch.load(native_path, map_location="cpu", weights_only=True)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    keys_exact = list(native) == list(adapter)
    tensors_exact = keys_exact and all(torch.equal(native[key], adapter[key]) for key in native)
    return {
        "keys_exact": keys_exact,
        "tensors_exact": tensors_exact,
        "tensor_count": len(native),
        "native_state_sha256": _state_digest(native),
        "adapter_state_sha256": _state_digest(adapter),
        "native_file_sha256": _sha256_file(native_path),
        "adapter_file_sha256": _sha256_file(adapter_path),
        "file_bytes_exact": native_path.read_bytes() == adapter_path.read_bytes(),
    }


def _normalized_config(model_dir: Path, output_root: Path) -> dict[str, Any]:
    payload = json.loads((model_dir / "rtf_config.json").read_text(encoding="utf-8"))

    def normalize(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(str(output_root), "<OUTPUT>")
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    normalized = normalize(payload)
    normalized["experiment_id"] = "<EXPERIMENT_ID>"
    return normalized


def run_validation(
    repo_root: Path,
    output_dir: Path,
    evidence_path: Path,
    wheel_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"REaLTabFormer validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_before = _verify_package(repo_root, wheel_path)
    environment = _environment(repo_root)
    cases: list[dict[str, Any]] = []
    case_number = 0
    for variant in VARIANTS:
        for seed in SEED_CASES:
            case_number += 1
            case_root = output_dir / f"case-{case_number:02d}-{variant}-seed-{seed}"
            dataset_spec, fixture = _write_fixture(case_root, variant)
            native_root = case_root / "native-output"
            adapter_root = case_root / "adapter-output"
            native_dir, native_raw = _run_native(native_root, dataset_spec, seed)
            adapter_dir, adapter_raw, adapter_csv, metadata = _run_adapter(repo_root, adapter_root, dataset_spec, seed)
            native_csv = native_root / "samples.csv"
            REaLTabFormerAdapter(repo_root)._write_dataframe_csv(native_raw, native_csv)
            checkpoints = _checkpoint_comparison(native_dir, adapter_dir)
            configs_exact = _normalized_config(native_dir, native_root) == _normalized_config(adapter_dir, adapter_root)
            raw_exact = native_raw.equals(adapter_raw)
            csv_exact = native_csv.read_bytes() == adapter_csv.read_bytes()
            frame = pd.read_csv(adapter_csv)
            metadata_valid = (
                metadata["package"]["upstream_commit"] == UPSTREAM_COMMIT
                and metadata["package"]["wheel_sha256"] == WHEEL_SHA256
                and metadata["source_rows"] == TRAIN_ROWS
                and metadata["training_rows"] == TRAIN_ROWS
                and metadata["training"]["fit"]["n_critic"] == 0
                and metadata["training"]["constructor"]["random_state"] == seed
            )
            categorical_valid = set(frame["group"].astype(str)).issubset({"g0", "g1", "g2"})
            if variant == "binary":
                categorical_valid = categorical_valid and set(frame["target"].astype(str)).issubset({"0", "1"})
            elif variant == "multiclass":
                categorical_valid = categorical_valid and set(frame["target"].astype(str)).issubset({"0", "1", "2"})
            numerical = ["first", "second", *(["target"] if variant == "regression" else [])]
            finite = bool(np.isfinite(frame[numerical].to_numpy(dtype=float)).all())
            passed = (
                checkpoints["tensors_exact"]
                and configs_exact
                and raw_exact
                and csv_exact
                and len(frame) == SAMPLE_ROWS
                and frame.columns.tolist() == dataset_spec.column_names
                and not bool(frame.isna().any().any())
                and finite
                and categorical_valid
                and metadata_valid
                and adapter_dir.resolve().is_relative_to(adapter_root.resolve())
            )
            cases.append(
                {
                    "case": case_number,
                    "variant": variant,
                    "seed": seed,
                    "status": "pass" if passed else "fail",
                    "fixture": fixture,
                    "comparisons": {
                        "checkpoint": checkpoints,
                        "saved_config_semantics_exact": configs_exact,
                        "raw_samples_exact": raw_exact,
                        "sample_bytes_exact": csv_exact,
                        "sample_sha256": _sha256_file(adapter_csv),
                        "sample_rows": len(frame),
                        "sample_columns": frame.columns.tolist(),
                        "missing_values": int(frame.isna().sum().sum()),
                        "finite_numerical_output": finite,
                        "categorical_domains_valid": categorical_valid,
                        "adapter_metadata_valid": metadata_valid,
                        "checkpoint_output_local": adapter_dir.resolve().is_relative_to(adapter_root.resolve()),
                    },
                }
            )

    package_after = _verify_package(repo_root, wheel_path)
    package_unchanged = package_before == package_after
    passed = package_unchanged and all(case["status"] == "pass" for case in cases)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "realtabformer",
        "reproduction_target": "method-author-official-tabular-package",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "package": package_before,
        "package_unchanged_after_validation": package_unchanged,
        "environment_lock": {
            "path": "requirements-realtabformer-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-realtabformer-validation.txt"),
        },
        "environment": environment,
        "compatibility_boundary": {
            "source_patches": [],
            "full_save_dir_path_serialization": (
                "v0.2.4 Path is converted to the string the official save configuration expects"
            ),
            "secure_checkpoint_load": (
                "adapter forces weights_only=True for the official state-dict loader; native control loads "
                "only the just-created audited checkpoint"
            ),
            "sampling_rng_reset": "adapter and native control reset identical Python/NumPy/PyTorch seeds",
            "declared_type_coercion": (
                "adapter and native control use the same DatasetSpec-directed categorical string coercion"
            ),
        },
        "validated_scope": {
            "model_type": "tabular",
            "training_path": "official fit with n_critic=0",
            "tasks": ["binary-classification", "multiclass-classification", "regression"],
            "sensitivity_stopping": "not-native-parity-validated",
            "relational_mode": "outside-the-current-single-table-contract",
        },
        "seed_cases": list(SEED_CASES),
        "variants": list(VARIANTS),
        "cases": cases,
    }
    atomic_write_bytes(
        evidence_path,
        (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("REaLTabFormer official-package parity failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the REaLTabFormer official-package parity protocol")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path, args.wheel_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            atomic_write_bytes(
                args.evidence_path,
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "protocol_id": PROTOCOL_ID,
                            "model_id": "realtabformer",
                            "status": "fail",
                            "repository_commit": _repository_commit(args.repo_root.resolve()),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
