from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from standardized_tabular_diffusion.compat.goggle_launcher import _official_import_boundary
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.goggle import GoggleAdapter
from standardized_tabular_diffusion.upstream_sources import (
    default_source_path,
    load_source_manifest,
    validate_upstream_source,
)

PROTOCOL_ID = "goggle-method-author-native-parity-v1"
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
TRAIN_ROWS = 12
EXPECTED_SAMPLE_ROWS = 7


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _copy_verified_source(source_root: Path, destination: Path) -> dict[str, Any]:
    source = validate_upstream_source("goggle", source_root)
    manifest = load_source_manifest("goggle")
    for record in manifest["runtime_files"]:
        source_path = source_root / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    validate_upstream_source("goggle", destination)
    return source


def _write_fixture(case_root: Path, variant: str) -> tuple[DatasetSpec, dict[str, Any]]:
    import pandas as pd

    dataset_name = f"goggle_parity_{variant}"

    def frame(count: int, offset: int) -> pd.DataFrame:
        row = np.arange(offset, offset + count)
        first = row * 0.17 + (row % 4) * 0.09
        second = ((row * 7) % 19) / 4.0 + 0.125
        group = np.asarray([f"g{value % 3}" for value in row])
        if variant == "binary":
            target: Any = ((row + row // 3) % 2).astype(np.int64)
        elif variant == "multiclass":
            target = ((row * 2 + row // 2) % 3).astype(np.int64)
        elif variant == "regression":
            target = first * 0.35 - second * 0.18 + (row % 3) * 0.07
        else:
            raise ValueError(f"Unknown Goggle validation variant: {variant}")
        return pd.DataFrame({"first": first, "second": second, "group": group, "target": target})

    data_root = case_root / "fixture"
    data_root.mkdir(parents=True)
    train = frame(TRAIN_ROWS, 0)
    test = frame(6, 100)
    train_path = data_root / "train.csv"
    test_path = data_root / "test.csv"
    metadata_path = data_root / "info.json"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "name": dataset_name,
                "task_type": "regression" if variant == "regression" else "classification",
                "column_names": ["first", "second", "group", "target"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_spec = DatasetSpec(
        name=dataset_name,
        task_type="regression" if variant == "regression" else "classification",
        column_names=["first", "second", "group", "target"],
        numerical_columns=["first", "second"],
        categorical_columns=["group"],
        target_columns=["target"],
        metadata_path=metadata_path,
        train_data_path=train_path,
        test_data_path=test_path,
    )
    return dataset_spec, {
        "dataset_name": dataset_name,
        "variant": variant,
        "task_type": dataset_spec.task_type,
        "training_rows": TRAIN_ROWS,
        "test_rows": len(test),
        "requested_sample_rows": EXPECTED_SAMPLE_ROWS,
        "numerical_features": 2,
        "categorical_features": 1,
        "target_features": 1,
        "missing_values": 0,
    }


def _adapter_config() -> dict[str, Any]:
    return {
        "encoder_dim": 8,
        "encoder_l": 2,
        "het_encoding": True,
        "decoder_dim": 8,
        "decoder_l": 1,
        "threshold": 0.1,
        "decoder_arch": "gcn",
        "graph_prior": None,
        "prior_mask": None,
        "alpha": 0.1,
        "beta": 0.1,
        "iter_opt": True,
        "learning_rate": 0.005,
        "weight_decay": 0.001,
        "epochs": 1,
        "batch_size": 4,
        "patience": 2,
        "logging": 1,
        "num_threads": 1,
    }


def _seed_native(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _construct_native(GoggleModel: type[Any], execution: dict[str, Any]) -> Any:
    return GoggleModel(
        ds_name=execution["dataset"],
        input_dim=execution["input_dim"],
        encoder_dim=execution["encoder_dim"],
        encoder_l=execution["encoder_l"],
        het_encoding=execution["het_encoding"],
        decoder_dim=execution["decoder_dim"],
        decoder_l=execution["decoder_l"],
        threshold=execution["threshold"],
        decoder_arch=execution["decoder_arch"],
        graph_prior=None,
        prior_mask=None,
        device="cpu",
        alpha=execution["alpha"],
        beta=execution["beta"],
        seed=execution["seed"],
        iter_opt=execution["iter_opt"],
        learning_rate=execution["learning_rate"],
        weight_decay=execution["weight_decay"],
        epochs=execution["epochs"],
        batch_size=execution["batch_size"],
        patience=execution["patience"],
        logging=execution["logging"],
    )


def _run_native(
    source_root: Path,
    output_dir: Path,
    transformed: Any,
    execution: dict[str, Any],
) -> tuple[Path, Any]:
    import pandas as pd
    import torch

    output_dir.mkdir(parents=True)
    input_path = output_dir / "transformed-training.csv"
    transformed.to_csv(input_path, index=False)
    native_input = pd.read_csv(input_path)
    _seed_native(execution["seed"])
    with _official_import_boundary(source_root) as GoggleModel:
        model = _construct_native(GoggleModel, execution)
        previous_cwd = Path.cwd()
        try:
            os.chdir(output_dir)
            model.fit(native_input)
        finally:
            os.chdir(previous_cwd)
    checkpoint = output_dir / "tmp" / f"{execution['dataset']}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Native Goggle checkpoint is missing: {checkpoint}")
    _seed_native(execution["seed"])
    with _official_import_boundary(source_root) as GoggleModel:
        sample_model = _construct_native(GoggleModel, execution)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sample_model.model.load_state_dict(state)
        raw = sample_model.model.sample(EXPECTED_SAMPLE_ROWS).detach().cpu().numpy()
    return checkpoint, raw


class _CapturingGoggleAdapter(GoggleAdapter):
    def __init__(self, repo_root: Path, raw_path: Path) -> None:
        super().__init__(repo_root)
        self._raw_path = raw_path

    def _inverse_transform(self, raw: np.ndarray, transform: dict[str, Any]) -> Any:
        np.save(self._raw_path, raw, allow_pickle=False)
        return super()._inverse_transform(raw, transform)


def _run_adapter(
    repo_root: Path,
    source_root: Path,
    output_dir: Path,
    dataset_spec: DatasetSpec,
    seed: int,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    raw_path = output_dir.parent / "adapter-raw.npy"
    adapter = _CapturingGoggleAdapter(repo_root, raw_path)
    common = {
        "dataset_spec": dataset_spec.to_dict(),
        "source_dir": str(source_root),
    }
    train_spec = RunSpec(
        model="goggle",
        dataset=dataset_spec.name,
        output_dir=output_dir,
        device="cpu",
        seed=seed,
        extra={**common, **_adapter_config()},
    )
    adapter.train(train_spec)
    sample_spec = RunSpec(
        model="goggle",
        dataset=dataset_spec.name,
        output_dir=output_dir,
        device="cpu",
        seed=seed,
        num_samples=EXPECTED_SAMPLE_ROWS,
        extra={**common, "num_threads": 1},
    )
    sample_bundle = adapter.sample(sample_spec)
    if sample_bundle.generated_sample_path is None:
        raise RuntimeError("Goggle adapter did not report its sample artifact.")
    metadata = json.loads((output_dir / "goggle-model-metadata.json").read_text(encoding="utf-8"))
    return output_dir / "model.pt", raw_path, sample_bundle.generated_sample_path, metadata


def _checkpoint_comparison(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import torch

    native = torch.load(native_path, map_location="cpu", weights_only=True)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    keys_exact = list(native) == list(adapter)
    tensors_exact = keys_exact and all(torch.equal(native[key], adapter[key]) for key in native)

    def state_digest(state: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        for key in sorted(state):
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()

    return {
        "keys_exact": keys_exact,
        "tensors_exact": tensors_exact,
        "tensor_count": len(native),
        "native_state_sha256": state_digest(native),
        "adapter_state_sha256": state_digest(adapter),
        "native_file_sha256": _sha256_file(native_path),
        "adapter_file_sha256": _sha256_file(adapter_path),
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment_versions() -> dict[str, str]:
    actual = {
        "dgl": _version("dgl"),
        "networkx": _version("networkx"),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "psutil": _version("psutil"),
        "pytest": _version("pytest"),
        "requests": _version("requests"),
        "scikit_learn": _version("scikit-learn"),
        "scipy": _version("scipy"),
        "torch": _version("torch"),
        "torch_geometric": _version("torch-geometric"),
        "tqdm": _version("tqdm"),
    }
    expected = {
        "dgl": "1.1.3",
        "networkx": "3.3",
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "psutil": "6.0.0",
        "pytest": "8.3.5",
        "requests": "2.32.3",
        "scikit_learn": "1.5.2",
        "scipy": "1.13.1",
        "torch": "2.3.0",
        "torch_geometric": "2.5.3",
        "tqdm": "4.66.5",
    }
    normalized = {**actual, "torch": actual["torch"].split("+")[0]}
    if normalized != expected:
        raise RuntimeError(f"Goggle validation environment mismatch: expected={expected}, actual={actual}")
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("Goggle formal validation requires Linux and Python 3.11.")
    return actual


def run_validation(
    repo_root: Path,
    output_dir: Path,
    evidence_path: Path,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Goggle validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_source = source_dir or default_source_path(repo_root, "goggle")
    selected_source = selected_source.resolve(strict=True)
    source = validate_upstream_source("goggle", selected_source)
    environment = _environment_versions()
    cases: list[dict[str, Any]] = []
    case_number = 0
    for variant in VARIANTS:
        for seed in SEED_CASES:
            case_number += 1
            case_root = output_dir / f"case-{case_number:02d}-{variant}-seed-{seed}"
            native_source = case_root / "native-source"
            adapter_source = case_root / "adapter-source"
            _copy_verified_source(selected_source, native_source)
            _copy_verified_source(selected_source, adapter_source)
            dataset_spec, fixture = _write_fixture(case_root, variant)
            preparation_adapter = GoggleAdapter(repo_root)
            frame = preparation_adapter._load_training_frame(dataset_spec)
            transformed, transform = preparation_adapter._transform_training_frame(frame, dataset_spec)
            execution = {
                **_adapter_config(),
                "dataset": dataset_spec.name,
                "seed": seed,
                "input_dim": transform["input_dim"],
                "training_rows": transform["training_rows"],
            }
            native_checkpoint, native_raw = _run_native(
                native_source, case_root / "native-output", transformed, execution
            )
            adapter_checkpoint, adapter_raw_path, adapter_sample_path, metadata = _run_adapter(
                repo_root,
                adapter_source,
                case_root / "adapter-output",
                dataset_spec,
                seed,
            )
            adapter_raw = np.load(adapter_raw_path, allow_pickle=False)
            native_frame = preparation_adapter._inverse_transform(native_raw, transform)
            native_sample_path = case_root / "native-samples.csv"
            preparation_adapter._write_dataframe_csv(native_frame, native_sample_path)
            adapter_frame = pd.read_csv(adapter_sample_path)
            native_roundtrip = pd.read_csv(native_sample_path)
            checkpoints = _checkpoint_comparison(native_checkpoint, adapter_checkpoint)
            raw_exact = np.array_equal(native_raw, adapter_raw)
            samples_exact = native_sample_path.read_bytes() == adapter_sample_path.read_bytes()
            source_after = validate_upstream_source("goggle", adapter_source)
            source_pure = not (adapter_source / "tmp").exists()
            metadata_valid = (
                metadata["source"]["upstream_commit"] == source["upstream_commit"]
                and metadata["source"]["runtime_files_verified"] == 18
                and metadata["execution_config"] == execution
                and metadata["transform"] == transform
            )
            case_passed = (
                checkpoints["tensors_exact"]
                and raw_exact
                and samples_exact
                and native_roundtrip.equals(adapter_frame)
                and len(adapter_frame) == EXPECTED_SAMPLE_ROWS
                and adapter_frame.columns.tolist() == dataset_spec.column_names
                and not bool(adapter_frame.isna().any().any())
                and bool(np.isfinite(adapter_frame[["first", "second"]].to_numpy()).all())
                and metadata_valid
                and source_after["manifest_sha256"] == source["manifest_sha256"]
                and source_pure
                and adapter_checkpoint.resolve().is_relative_to((case_root / "adapter-output").resolve())
            )
            cases.append(
                {
                    "case": case_number,
                    "variant": variant,
                    "seed": seed,
                    "status": "pass" if case_passed else "fail",
                    "fixture": fixture,
                    "execution_config": execution,
                    "comparisons": {
                        "checkpoints": checkpoints,
                        "raw_samples_exact": raw_exact,
                        "raw_sample_sha256": _sha256_array(native_raw),
                        "sample_bytes_exact": samples_exact,
                        "sample_frames_exact": native_roundtrip.equals(adapter_frame),
                        "sample_sha256": _sha256_file(adapter_sample_path),
                        "sample_rows": len(adapter_frame),
                        "sample_columns": adapter_frame.columns.tolist(),
                        "missing_values": int(adapter_frame.isna().sum().sum()),
                        "finite_numerical_output": bool(
                            np.isfinite(adapter_frame[["first", "second"]].to_numpy()).all()
                        ),
                        "adapter_metadata_valid": metadata_valid,
                        "adapter_source_remained_exact": source_after["manifest_sha256"]
                        == source["manifest_sha256"],
                        "adapter_checkpoint_outside_source": source_pure,
                    },
                }
            )
    passed = all(case["status"] == "pass" for case in cases)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "goggle",
        "reproduction_target": "method-author-original-core",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "environment_lock": {
            "path": "requirements-goggle-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-goggle-validation.txt"),
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), **environment},
        "compatibility_boundary": {
            "source_patches": [],
            "synthcity_import_bridge": "evaluation-only imports are stubbed; fit and core sample do not execute them",
            "rgcn_import_bridge": "used only when torch-sparse is absent; gcn parity does not instantiate RGCNConv",
            "sampling_target": "official Goggle.model.sample before centralized inverse transformation",
        },
        "seed_cases": list(SEED_CASES),
        "variants": list(VARIANTS),
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("Goggle method-author native-parity protocol failed; inspect the evidence record.")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Goggle method-author native-parity protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path, args.source_dir)
    except Exception as exc:
        if not args.evidence_path.exists():
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": PROTOCOL_ID,
                        "model_id": "goggle",
                        "status": "fail",
                        "repository_commit": _repository_commit(args.repo_root.resolve()),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
