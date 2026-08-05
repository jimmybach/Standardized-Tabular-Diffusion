"""Executable parity protocol for the official be-great package."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import platform
import subprocess
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.great import (
    GREAT_RUNTIME_SHA256,
    GREAT_UPSTREAM_COMMIT,
    GREAT_WHEEL_SHA256,
    GReaTAdapter,
    verify_great_distribution,
)
from standardized_tabular_diffusion.validation._tiny_lm import build_tiny_gpt2

PROTOCOL_ID = "be-great-official-package-parity-v1"
WHEEL_FILENAME = "be_great-0.0.14-py3-none-any.whl"
WHEEL_BYTES = 54_019
EXPECTED_WHEEL_FILES = 19
EXPECTED_PACKAGE_FILES = 14
EXPECTED_PACKAGE_AGGREGATE = "3e98e1e0e68ce614b62d425c8fcf559ab0387cbf85dd0e27166c22b360fcfaca"
SEEDS = (0, 19, 73)
TRAIN_ROWS = 24
SAMPLE_ROWS = 7


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _aggregate(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _verify_wheel(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.name != WHEEL_FILENAME:
        raise ValueError("be-great wheel path is missing or unsafe")
    if path.stat().st_size != WHEEL_BYTES or sha256_file(path) != GREAT_WHEEL_SHA256:
        raise ValueError("be-great wheel differs from the PyPI lock")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("be-great wheel contains duplicate paths")
        files: dict[str, bytes] = {}
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe be-great wheel member: {name!r}")
            if not name.endswith("/"):
                files[name] = archive.read(name)
    if len(files) != EXPECTED_WHEEL_FILES:
        raise ValueError("be-great wheel file count differs from the lock")
    package_files = {name: payload for name, payload in files.items() if name.startswith("be_great/")}
    if len(package_files) != EXPECTED_PACKAGE_FILES or _aggregate(package_files) != EXPECTED_PACKAGE_AGGREGATE:
        raise ValueError("be-great package bytes differ from the 0.0.14 source tag")
    for name, expected in GREAT_RUNTIME_SHA256.items():
        if name not in package_files or hashlib.sha256(package_files[name]).hexdigest() != expected:
            raise ValueError(f"be-great runtime file differs from the lock: {name}")
    metadata = files["be_great-0.0.14.dist-info/METADATA"].decode("utf-8")
    if "Name: be_great\n" not in metadata or "Version: 0.0.14\n" not in metadata:
        raise ValueError("be-great wheel metadata differs from the lock")
    license_payload = files["be_great-0.0.14.dist-info/licenses/LICENSE"]
    if hashlib.sha256(license_payload).hexdigest() != "8cca1b8d2a9b78541e2ed5a92b5133a87ba535a64565d2b8aee2f520dcf49f23":
        raise ValueError("be-great wheel MIT license differs from the lock")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "wheel_files_verified": len(files),
        "package_files_verified": len(package_files),
        "package_aggregate_sha256": _aggregate(package_files),
        "license": "MIT",
        "source_commit": GREAT_UPSTREAM_COMMIT,
    }


def _frame() -> tuple[pd.DataFrame, DatasetSpec]:
    index = np.arange(TRAIN_ROWS)
    frame = pd.DataFrame(
        {
            "color": np.array(["red", "blue"])[index % 2],
            "shape": np.array(["round", "square", "triangle"])[index % 3],
            "target": np.array(["no", "yes"])[(index // 2) % 2],
        }
    )
    spec = DatasetSpec(
        name="great-tiny-mixed",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=[],
        categorical_columns=["color", "shape"],
        target_columns=["target"],
        metadata_path=Path("unused.json"),
    )
    return frame, spec


def _texts(frame: pd.DataFrame) -> list[str]:
    values: list[str] = []
    columns = list(frame.columns)
    for _, row in frame.iterrows():
        values.append(", ".join(f"{column} is {row[column]}" for column in columns))
        values.append(", ".join(f"{column} {row[column]}" for column in columns))
    for column in columns:
        for value in sorted(frame[column].unique()):
            values.extend([f"{column} is {value}", f"{column} {value},"])
    return values


def _state_dict(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _assert_state_exact(left: dict[str, Any], right: dict[str, Any]) -> None:
    import torch

    if set(left) != set(right):
        raise AssertionError("GReaT model tensor names differ")
    for name in left:
        if left[name].dtype != right[name].dtype or left[name].shape != right[name].shape:
            raise AssertionError(f"GReaT tensor contract differs: {name}")
        if not torch.equal(left[name], right[name]):
            raise AssertionError(f"GReaT tensor bytes differ: {name}")


def _run_case(
    repo_root: Path,
    output_dir: Path,
    tiny_model: Path,
    dataset_spec: DatasetSpec,
    frame: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    case_root = output_dir / f"seed-{seed}"
    case_root.mkdir(parents=True, exist_ok=True)
    dataset_spec = DatasetSpec(**{**dataset_spec.__dict__})
    dataset_spec.metadata_path = case_root / "dataset.json"
    dataset_spec.train_data_path = case_root / "train.csv"
    atomic_write_json(dataset_spec.metadata_path, dataset_spec.to_dict())
    frame.to_csv(dataset_spec.train_data_path, index=False)
    adapter = GReaTAdapter(repo_root)
    model_class = adapter._import_model_class()
    train_kwargs = {
        "save_strategy": "no",
        "disable_tqdm": True,
        "seed": seed,
        "data_seed": seed,
        "learning_rate": 0.01,
    }
    with adapter._scoped_randomness(seed, 1):
        direct = model_class(
            str(tiny_model),
            experiment_dir=str(case_root / "direct-trainer"),
            epochs=8,
            batch_size=8,
            report_to=[],
            **train_kwargs,
        )
        direct.fit(frame.copy(), conditional_col="target", random_conditional_col=False)
    direct_state = _state_dict(direct.model)
    train_spec = RunSpec(
        model="great",
        dataset=dataset_spec.name,
        output_dir=case_root / "adapter",
        seed=seed,
        device="cpu",
        extra={
            "dataset_spec": dataset_spec.to_dict(),
            "llm": str(tiny_model),
            "epochs": 8,
            "batch_size": 8,
            "conditional_col": "target",
            "random_conditional_col": False,
            "num_threads": 1,
            "train_kwargs": {"learning_rate": 0.01},
        },
    )
    python_before = __import__("random").getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    adapter.train(train_spec)
    if __import__("random").getstate() != python_before:
        raise AssertionError("GReaT adapter leaked Python RNG state")
    numpy_after = np.random.get_state()
    if any(
        not np.array_equal(before, after) if isinstance(before, np.ndarray) else before != after
        for before, after in zip(numpy_before, numpy_after)
    ):
        raise AssertionError("GReaT adapter leaked NumPy RNG state")
    if not torch.equal(torch_before, torch.random.get_rng_state()) or torch.get_num_threads() != threads_before:
        raise AssertionError("GReaT adapter leaked PyTorch RNG/thread state")
    reloaded = AutoModelForCausalLM.from_pretrained(train_spec.output_dir / "great_model" / "transformer")
    _assert_state_exact(direct_state, _state_dict(reloaded))
    with adapter._scoped_randomness(seed, 1):
        direct_sample = direct.sample(
            n_samples=SAMPLE_ROWS,
            temperature=0.7,
            k=SAMPLE_ROWS,
            max_length=96,
            drop_nan=True,
            device="cpu",
            guided_sampling=True,
            random_feature_order=False,
        )
    sample_spec = RunSpec(
        model="great",
        dataset=dataset_spec.name,
        output_dir=train_spec.output_dir,
        seed=seed,
        num_samples=SAMPLE_ROWS,
        device="cpu",
        extra={
            "dataset_spec": dataset_spec.to_dict(),
            "temperature": 0.7,
            "k": SAMPLE_ROWS,
            "max_length": 96,
            "guided_sampling": True,
            "random_feature_order": False,
            "num_threads": 1,
        },
    )
    bundle = adapter.sample(sample_spec)
    observed = pd.read_csv(bundle.generated_sample_path, dtype=str)
    expected = direct_sample[dataset_spec.column_names].astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=True)
    if bundle.generated_sample_path.read_bytes() != expected.to_csv(index=False).encode("utf-8"):
        raise AssertionError("GReaT adapter CSV bytes differ from direct official package")
    return {
        "seed": seed,
        "train_rows": len(frame),
        "sample_rows": len(observed),
        "state_tensors_exact": len(direct_state),
        "csv_sha256": sha256_file(bundle.generated_sample_path),
        "integrity_sha256": sha256_file(train_spec.output_dir / "great_model" / "great-integrity.json"),
        "exact": True,
    }


def run_protocol(repo_root: Path, output_dir: Path, evidence_path: Path, wheel_path: Path) -> dict[str, Any]:
    wheel = _verify_wheel(wheel_path)
    package_before = verify_great_distribution()
    frame, dataset_spec = _frame()
    tiny_model = output_dir / "tiny-gpt2"
    tiny = build_tiny_gpt2(tiny_model, _texts(frame))
    cases = [_run_case(repo_root, output_dir, tiny_model, dataset_spec, frame, seed) for seed in SEEDS]
    package_after = verify_great_distribution()
    if package_after != package_before:
        raise AssertionError("Installed be-great package changed during validation")
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "great",
        "status": "pass",
        "repository_commit": _repository_commit(repo_root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in (
                "accelerate",
                "be-great",
                "chugchug",
                "datasets",
                "numpy",
                "pandas",
                "safetensors",
                "scikit-learn",
                "torch",
                "transformers",
            )
        },
        "wheel": wheel,
        "installed_package": package_before,
        "tiny_offline_checkpoint": tiny,
        "cases": cases,
        "result_summary": {
            "cases_passed": len(cases),
            "cases_total": len(SEEDS),
            "package_unchanged": True,
            "official_fit_executed": True,
            "official_guided_sample_executed": True,
            "state_tensors_exact": True,
            "safe_safetensors_checkpoint": True,
            "sample_frames_exact": True,
            "sample_csv_bytes_exact": True,
            "rng_and_thread_state_restored": True,
        },
        "claim_limit": (
            "Parity covers the unchanged be-great 0.0.14 package with a deterministic offline tiny GPT-2 "
            "checkpoint and guided categorical sampling. Full-scale pretrained-model quality, central evaluation, "
            "privacy, runtime budgets, Official Results, and release support remain separate gates."
        ),
    }
    atomic_write_json(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_protocol(args.repo_root.resolve(), args.output_dir, args.evidence_path, args.wheel_path.resolve())
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            args.evidence_path,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "model_id": "great",
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
