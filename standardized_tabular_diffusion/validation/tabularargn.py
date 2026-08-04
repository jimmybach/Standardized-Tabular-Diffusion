"""Executable native-parity protocol for the official MOSTLY AI TabularARGN package."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import platform
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
from standardized_tabular_diffusion.models.tabularargn import (
    EXPECTED_RUNTIME_VERSIONS,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    UPSTREAM_TAG,
    UPSTREAM_TREE,
    TabularARGNAdapter,
    verify_tabularargn_distribution,
)

PROTOCOL_ID = "tabularargn-official-package-parity-v1"
WHEEL_FILENAME = "mostlyai_engine-2.6.2-py3-none-any.whl"
WHEEL_SHA256 = "3ead3770c936919f8fce4e1f9fffd271ffdd490f0292c2ab9a42cb4bafe3caea"
WHEEL_BYTES = 185_077
SOURCE_ARCHIVE_FILENAME = "mostlyai-engine-0b96f02.zip"
SOURCE_ARCHIVE_SHA256 = "d80cd89b14cdf793e07a2e5b53abe36b5824376d6aa73ae63c56052cf9043c66"
SOURCE_ARCHIVE_BYTES = 671_861
SOURCE_ARCHIVE_MEMBERS = 176
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
EXPECTED_ARCHIVE_MEMBERS = 54
EXPECTED_RECORD_ROWS = 54
EXPECTED_RECORD_HASHES = 53
TRAIN_ROWS = 48
SAMPLE_ROWS = 7
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
TRAINING_CONTROLS = {
    "model": "MOSTLY_AI/Small",
    "max_training_time": 10.0,
    "max_epochs": 1.0,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "enable_flexible_generation": True,
    "value_protection": True,
    "verbose": 0,
}
SAMPLING_CONTROLS = {
    "batch_size": 7,
    "sampling_temperature": 1.0,
    "sampling_top_p": 1.0,
    "rare_category_replacement_method": "SAMPLE",
}
EXPECTED_REQUIRES_DIST = [
    "accelerate>=1.5.0",
    "datasets>=3.0.0",
    "huggingface-hub[hf-xet]>=0.30.2",
    "joblib>=1.4.2",
    "json-repair>=0.47.0",
    "numpy>=2.0.0",
    "opacus>=1.6.0",
    "pandas>=2.2.0",
    "peft>=0.18.2",
    "psutil<6,>=5.9.5",
    "pyarrow>=16.0.0",
    "scikit-learn>=1.4.0",
    "setuptools<81.0.0,>=77.0.3",
    "tokenizers>=0.21.1",
    "torch<2.12.0,>=2.11.0",
    "torchaudio<2.12.0,>=2.11.0",
    "torchvision<0.27.0,>=0.26.0",
    "transformers>=5.5.1",
    "xgrammar<1.0.0,>=0.1.32",
    "bitsandbytes==0.42.0; (sys_platform == 'darwin') and extra == 'gpu'",
    "bitsandbytes>=0.45.5; (sys_platform == 'linux') and extra == 'gpu'",
    "vllm==0.20.0; (sys_platform == 'linux' or sys_platform == 'darwin') and extra == 'gpu'",
]


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
    path = repo_root / "standardized_tabular_diffusion" / "resources" / "upstream" / "tabularargn-wheel-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_wheel(repo_root: Path, wheel_path: Path) -> dict[str, Any]:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(f"TabularARGN wheel must be a regular non-symlinked file: {wheel_path}")
    if wheel_path.name != WHEEL_FILENAME:
        raise ValueError(f"Expected wheel {WHEEL_FILENAME!r}, observed {wheel_path.name!r}")
    if wheel_path.stat().st_size != WHEEL_BYTES or _sha256_file(wheel_path) != WHEEL_SHA256:
        raise ValueError("TabularARGN wheel bytes or SHA-256 differ from the locked official release")

    manifest = _load_manifest(repo_root)
    expected_files = {record["path"]: record for record in manifest["installed_files"]}
    dist_info = f"mostlyai_engine-{PACKAGE_VERSION}.dist-info"
    record_name = f"{dist_info}/RECORD"
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        for name in names:
            archive_path = PurePosixPath(name)
            if archive_path.is_absolute() or ".." in archive_path.parts or "\\" in name:
                raise ValueError(f"Unsafe path in TabularARGN wheel: {name!r}")
        metadata = Parser().parsestr(archive.read(f"{dist_info}/METADATA").decode("utf-8"))
        if metadata.get("Name") != PACKAGE_NAME or metadata.get("Version") != PACKAGE_VERSION:
            raise ValueError("TabularARGN wheel metadata does not match the locked package identity")
        if metadata.get("Requires-Python") != "<3.14,>=3.11":
            raise ValueError("TabularARGN wheel Python requirement differs from the audited release")
        if metadata.get_all("Requires-Dist", []) != EXPECTED_REQUIRES_DIST:
            raise ValueError("TabularARGN wheel dependencies differ from the audited release")
        wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
        if "Root-Is-Purelib: true" not in wheel_metadata or "Tag: py3-none-any" not in wheel_metadata:
            raise ValueError("TabularARGN wheel type or compatibility tag differs from the release lock")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        hashed_rows = [row for row in rows if row[1]]
        if len(names) != EXPECTED_ARCHIVE_MEMBERS or len(rows) != EXPECTED_RECORD_ROWS:
            raise ValueError("TabularARGN wheel archive or RECORD member count differs from the lock")
        if len(hashed_rows) != EXPECTED_RECORD_HASHES:
            raise ValueError("TabularARGN wheel RECORD hash count differs from the lock")
        if {row[0] for row in hashed_rows} != set(expected_files):
            raise ValueError("TabularARGN wheel RECORD paths differ from the audited manifest")
        for path, encoded_hash, encoded_size in hashed_rows:
            payload = archive.read(path)
            expected = expected_files[path]
            observed_digest = hashlib.sha256(payload).digest()
            if int(encoded_size) != expected["bytes"] or len(payload) != expected["bytes"]:
                raise ValueError(f"TabularARGN wheel member size mismatch: {path}")
            if observed_digest != _decode_record_hash(encoded_hash.partition("=")[2]):
                raise ValueError(f"TabularARGN RECORD checksum mismatch: {path}")
            if observed_digest.hex() != expected["sha256"]:
                raise ValueError(f"TabularARGN manifest checksum mismatch: {path}")
        license_hash = _sha256_bytes(archive.read(f"{dist_info}/licenses/LICENSE"))
    if license_hash != LICENSE_SHA256:
        raise ValueError("TabularARGN wheel LICENSE differs from the locked Apache-2.0 license")
    return {
        "filename": wheel_path.name,
        "bytes": wheel_path.stat().st_size,
        "sha256": _sha256_file(wheel_path),
        "archive_members": len(names),
        "record_rows": len(rows),
        "record_hashed_files": len(hashed_rows),
        "license": "Apache-2.0",
        "license_sha256": license_hash,
    }


def _verify_source_archive(repo_root: Path, source_archive_path: Path, wheel_path: Path) -> dict[str, Any]:
    if source_archive_path.is_symlink() or not source_archive_path.is_file():
        raise ValueError(f"TabularARGN source archive must be a regular file: {source_archive_path}")
    if (
        source_archive_path.name != SOURCE_ARCHIVE_FILENAME
        or source_archive_path.stat().st_size != SOURCE_ARCHIVE_BYTES
        or _sha256_file(source_archive_path) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("TabularARGN source archive identity differs from the locked official commit")
    manifest = _load_manifest(repo_root)
    package_records = [
        record for record in manifest["installed_files"] if record["path"].startswith("mostlyai/engine/")
    ]
    prefix = f"mostlyai-engine-{UPSTREAM_COMMIT}"
    with zipfile.ZipFile(source_archive_path) as source, zipfile.ZipFile(wheel_path) as wheel:
        names = source.namelist()
        if len(names) != SOURCE_ARCHIVE_MEMBERS:
            raise ValueError("TabularARGN source archive member count differs from the release lock")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe path in TabularARGN source archive: {name!r}")
        for record in package_records:
            source_payload = source.read(f"{prefix}/{record['path']}")
            wheel_payload = wheel.read(record["path"])
            if source_payload != wheel_payload or _sha256_bytes(source_payload) != record["sha256"]:
                raise ValueError(f"TabularARGN wheel/source mismatch: {record['path']}")
        source_license_hash = _sha256_bytes(source.read(f"{prefix}/LICENSE"))
    if source_license_hash != LICENSE_SHA256:
        raise ValueError("TabularARGN source LICENSE differs from the wheel and manifest")
    return {
        "filename": source_archive_path.name,
        "bytes": source_archive_path.stat().st_size,
        "sha256": _sha256_file(source_archive_path),
        "archive_members": len(names),
        "shared_package_source_files": len(package_records),
        "exact_shared_package_source_files": len(package_records),
        "license_sha256": source_license_hash,
    }


def _verify_package(
    repo_root: Path,
    wheel_path: Path,
    source_archive_path: Path,
) -> dict[str, Any]:
    return {
        "authority": "method-author",
        "distribution_form": "package",
        "repository": UPSTREAM_REPOSITORY,
        "tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "wheel": _verify_wheel(repo_root, wheel_path),
        "source_archive": _verify_source_archive(repo_root, source_archive_path, wheel_path),
        "installed_distribution": verify_tabularargn_distribution(repo_root),
    }


def _environment(repo_root: Path) -> dict[str, str]:
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("TabularARGN formal validation requires Linux and Python 3.11")
    observed = verify_tabularargn_distribution(repo_root)["runtime_versions"]
    if observed != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError(f"TabularARGN runtime differs from the frozen validation environment: {observed}")
    return {"platform": platform.platform(), "python": platform.python_version(), **observed}


def _write_fixture(case_root: Path, variant: str) -> tuple[DatasetSpec, dict[str, Any]]:
    row = np.arange(TRAIN_ROWS)
    first = np.round(np.sin(row / 4.0) + row * 0.05, 4)
    second = ((row * 7) % 23).astype(int)
    group = np.asarray([f"g{value % 3}" for value in row])
    if variant == "binary":
        target: Any = ((row + row // 3) % 2).astype(int)
    elif variant == "multiclass":
        target = ((row * 2 + row // 2) % 3).astype(int)
    elif variant == "regression":
        target = np.round(first * 0.4 - second * 0.03 + (row % 4) * 0.08, 4)
    else:
        raise ValueError(f"Unknown TabularARGN validation variant: {variant}")
    frame = pd.DataFrame({"first": first, "second": second, "group": group, "target": target})
    fixture_root = case_root / "fixture"
    fixture_root.mkdir(parents=True)
    train_path = fixture_root / "train.csv"
    metadata_path = fixture_root / "info.json"
    frame.to_csv(train_path, index=False)
    dataset_name = f"tabularargn_parity_{variant}"
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


def _seed(seed: int, torch_module: Any, engine: Any) -> None:
    torch_module.set_num_threads(1)
    engine.set_random_state(seed)


def _coerced_training_frame(dataset_spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(dataset_spec.train_data_path)
    frame["group"] = frame["group"].astype(str)
    if dataset_spec.task_type == "classification":
        frame["target"] = frame["target"].astype(str)
    return frame[dataset_spec.column_names].copy()


def _constructor_kwargs(workspace: Path, seed: int) -> dict[str, Any]:
    return {
        **TRAINING_CONTROLS,
        "tgt_encoding_types": None,
        "device": "cpu",
        "workspace_dir": str(workspace),
        "random_state": seed,
    }


def _run_native(output_dir: Path, dataset_spec: DatasetSpec, seed: int) -> tuple[Path, pd.DataFrame]:
    import torch
    from mostlyai import engine

    workspace = output_dir / "tabularargn_workspace"
    output_dir.mkdir(parents=True)
    frame = _coerced_training_frame(dataset_spec)
    _seed(seed, torch, engine)
    model = engine.TabularARGN(**_constructor_kwargs(workspace, seed))
    model.fit(frame)
    _seed(seed, torch, engine)
    raw_sample = model.sample(n_samples=SAMPLE_ROWS, device="cpu", **SAMPLING_CONTROLS)
    return workspace, raw_sample


class _CapturingAdapter(TabularARGNAdapter):
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
            model="tabularargn",
            dataset=dataset_spec.name,
            output_dir=output_dir,
            device="cpu",
            seed=seed,
            extra={**common, **TRAINING_CONTROLS},
        )
    )
    sample_bundle = adapter.sample(
        RunSpec(
            model="tabularargn",
            dataset=dataset_spec.name,
            output_dir=output_dir,
            device="cpu",
            seed=seed,
            num_samples=SAMPLE_ROWS,
            extra={**common, **SAMPLING_CONTROLS},
        )
    )
    if adapter.captured_sample is None or sample_bundle.generated_sample_path is None:
        raise RuntimeError("TabularARGN adapter did not expose its generated sample")
    workspace = output_dir / adapter.workspace_name
    metadata = json.loads((output_dir / adapter.metadata_filename).read_text(encoding="utf-8"))
    return workspace, adapter.captured_sample, sample_bundle.generated_sample_path, metadata


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_comparison(native_workspace: Path, adapter_workspace: Path) -> dict[str, Any]:
    import torch

    relative = Path("ModelStore") / "model-data" / "model-weights.pt"
    native_path = native_workspace / relative
    adapter_path = adapter_workspace / relative
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


def _json_artifact_exact(native_workspace: Path, adapter_workspace: Path, relative: Path) -> bool:
    native_path = native_workspace / relative
    adapter_path = adapter_workspace / relative
    return json.loads(native_path.read_text(encoding="utf-8")) == json.loads(adapter_path.read_text(encoding="utf-8"))


def run_validation(
    repo_root: Path,
    output_dir: Path,
    evidence_path: Path,
    wheel_path: Path,
    source_archive_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"TabularARGN validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_before = _verify_package(repo_root, wheel_path, source_archive_path)
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
            native_workspace, native_raw = _run_native(native_root, dataset_spec, seed)
            adapter_workspace, adapter_raw, adapter_csv, metadata = _run_adapter(
                repo_root, adapter_root, dataset_spec, seed
            )
            native_csv = native_root / "samples.csv"
            TabularARGNAdapter(repo_root)._write_dataframe_csv(native_raw, native_csv)
            checkpoints = _checkpoint_comparison(native_workspace, adapter_workspace)
            configs_exact = _json_artifact_exact(
                native_workspace,
                adapter_workspace,
                Path("ModelStore") / "model-data" / "model-configs.json",
            )
            stats_exact = _json_artifact_exact(
                native_workspace,
                adapter_workspace,
                Path("ModelStore") / "tgt-stats" / "stats.json",
            )
            raw_exact = native_raw.equals(adapter_raw)
            csv_exact = native_csv.read_bytes() == adapter_csv.read_bytes()
            frame = pd.read_csv(adapter_csv)
            metadata_valid = (
                metadata["package"]["upstream_commit"] == UPSTREAM_COMMIT
                and metadata["package"]["wheel_sha256"] == WHEEL_SHA256
                and metadata["source_rows"] == TRAIN_ROWS
                and metadata["training_rows"] == TRAIN_ROWS
                and metadata["training"]["constructor"]["random_state"] == seed
                and metadata["raw_or_encoded_training_rows_retained"] is False
            )
            categorical_valid = set(frame["group"].astype(str)).issubset({"g0", "g1", "g2"})
            if variant == "binary":
                categorical_valid = categorical_valid and set(frame["target"].astype(str)).issubset({"0", "1"})
            elif variant == "multiclass":
                categorical_valid = categorical_valid and set(frame["target"].astype(str)).issubset({"0", "1", "2"})
            numerical = ["first", "second", *(["target"] if variant == "regression" else [])]
            finite = bool(np.isfinite(frame[numerical].to_numpy(dtype=float)).all())
            original_data_pruned = not (adapter_workspace / "OriginalData").exists()
            passed = (
                checkpoints["tensors_exact"]
                and configs_exact
                and stats_exact
                and raw_exact
                and csv_exact
                and len(frame) == SAMPLE_ROWS
                and frame.columns.tolist() == dataset_spec.column_names
                and not bool(frame.isna().any().any())
                and finite
                and categorical_valid
                and metadata_valid
                and original_data_pruned
                and adapter_workspace.resolve().is_relative_to(adapter_root.resolve())
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
                        "model_config_semantics_exact": configs_exact,
                        "target_stats_semantics_exact": stats_exact,
                        "raw_samples_exact": raw_exact,
                        "sample_bytes_exact": csv_exact,
                        "sample_sha256": _sha256_file(adapter_csv),
                        "sample_rows": len(frame),
                        "sample_columns": frame.columns.tolist(),
                        "missing_values": int(frame.isna().sum().sum()),
                        "finite_numerical_output": finite,
                        "categorical_domains_valid": categorical_valid,
                        "adapter_metadata_valid": metadata_valid,
                        "raw_and_encoded_training_data_pruned": original_data_pruned,
                        "checkpoint_output_local": adapter_workspace.resolve().is_relative_to(adapter_root.resolve()),
                    },
                }
            )

    package_after = _verify_package(repo_root, wheel_path, source_archive_path)
    package_unchanged = package_before == package_after
    passed = package_unchanged and all(case["status"] == "pass" for case in cases)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabularargn",
        "reproduction_target": "method-author-official-flat-tabular-package",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "package": package_before,
        "package_unchanged_after_validation": package_unchanged,
        "environment_lock": {
            "path": "requirements-tabularargn-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-tabularargn-validation.txt"),
        },
        "environment": environment,
        "compatibility_boundary": {
            "source_patches": [],
            "output_workspace": "official workspace_dir is placed beneath output_dir",
            "declared_type_coercion": "DatasetSpec categorical roles are strings in adapter and native control",
            "sampling_rng_reset": "adapter and native control call the official set_random_state function",
            "persistent_artifact": (
                "OriginalData is removed; official ModelStore is integrity-manifested and official weights_only load is used"
            ),
            "categorical_domain": (
                "adapter and native control select official rare_category_replacement_method=SAMPLE"
            ),
        },
        "validated_scope": {
            "model": "MOSTLY_AI/Small",
            "mode": "flat-single-table-unconditional-generation",
            "tasks": ["binary-classification", "multiclass-classification", "regression"],
            "value_protection": True,
            "differential_privacy": "outside-this-parity-protocol",
            "sequential_and_relational_modes": "outside-current-single-table-contract",
            "conditional_generation_prediction_imputation": "outside-current-generation-adapter",
        },
        "seed_cases": list(SEED_CASES),
        "variants": list(VARIANTS),
        "cases": cases,
    }
    atomic_write_bytes(evidence_path, (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("TabularARGN official-package parity failed; inspect the evidence record")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TabularARGN official-package parity protocol")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    parser.add_argument("--source-archive-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(
            args.repo_root,
            args.output_dir,
            args.evidence_path,
            args.wheel_path,
            args.source_archive_path,
        )
    except Exception as exc:
        if not args.evidence_path.exists():
            atomic_write_bytes(
                args.evidence_path,
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "protocol_id": PROTOCOL_ID,
                            "model_id": "tabularargn",
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
