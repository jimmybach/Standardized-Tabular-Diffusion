"""Executable parity protocol for the locked method-author TabuLa source."""

from __future__ import annotations

import argparse
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
from standardized_tabular_diffusion.models.tabula import TabulaAdapter
from standardized_tabular_diffusion.upstream_sources import source_manifest_path, validate_upstream_source
from standardized_tabular_diffusion.validation._tiny_lm import build_tiny_gpt2

PROTOCOL_ID = "tabula-method-author-source-parity-v1"
UPSTREAM_COMMIT = TabulaAdapter.upstream_commit
ARCHIVE_BYTES = 59_173
ARCHIVE_SHA256 = "dfb69d55cf4e669f979325bf10b118a084dde49d680680dcebf7dcdaed024a26"
SEEDS = (0, 19, 73)
TRAIN_ROWS = 24
SAMPLE_ROWS = 5


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _verify_archive(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("TabuLa source archive is missing or unsafe")
    if path.stat().st_size != ARCHIVE_BYTES or sha256_file(path) != ARCHIVE_SHA256:
        raise ValueError("TabuLa source archive differs from the commit lock")
    files = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("TabuLa source archive contains duplicate paths")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe TabuLa archive member: {name!r}")
            if not name.endswith("/"):
                files += 1
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": len(names),
        "regular_files": files,
        "source_commit": UPSTREAM_COMMIT,
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
        name="tabula-tiny-categorical",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=[],
        categorical_columns=["color", "shape"],
        target_columns=["target"],
        metadata_path=Path("unused.json"),
    )
    return frame, spec


def _texts(frame: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    encoded = frame.copy()
    for column in encoded.columns:
        mapping = {value: index for index, value in enumerate(sorted(encoded[column].unique()))}
        encoded[column] = encoded[column].map(mapping)
    for _, row in encoded.iterrows():
        texts.append(", ".join(f"{column} {row[column]}" for column in encoded.columns))
    for column in encoded.columns:
        for value in sorted(encoded[column].unique()):
            texts.append(f"{column} {value},")
    return texts


def _state_dict(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _assert_state_exact(left: dict[str, Any], right: dict[str, Any]) -> None:
    import torch

    if set(left) != set(right):
        raise AssertionError("TabuLa model tensor names differ")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise AssertionError(f"TabuLa tensor bytes differ: {name}")


def _run_case(
    repo_root: Path,
    source_root: Path,
    output_dir: Path,
    tiny_model: Path,
    base_spec: DatasetSpec,
    frame: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    case_root = output_dir / f"seed-{seed}"
    case_root.mkdir(parents=True, exist_ok=True)
    dataset_spec = DatasetSpec(**{**base_spec.__dict__})
    dataset_spec.metadata_path = case_root / "dataset.json"
    dataset_spec.train_data_path = case_root / "train.csv"
    atomic_write_json(dataset_spec.metadata_path, dataset_spec.to_dict())
    frame.to_csv(dataset_spec.train_data_path, index=False)
    adapter = TabulaAdapter(repo_root)
    train_kwargs = {
        "learning_rate": 0.01,
        "disable_tqdm": True,
        "seed": seed,
        "data_seed": seed,
        "report_to": [],
    }
    with adapter._official_class(source_root) as model_class, adapter._scoped_randomness(seed, 1):
        direct = model_class(
            str(tiny_model),
            experiment_dir=str(case_root / "direct-trainer"),
            epochs=12,
            batch_size=8,
            categorical_columns=list(frame.columns),
            **train_kwargs,
        )
        direct.fit(frame.copy(), conditional_col="target")
    direct_state = _state_dict(direct.model)
    train_spec = RunSpec(
        model="tabula",
        dataset=dataset_spec.name,
        output_dir=case_root / "adapter",
        seed=seed,
        device="cpu",
        extra={
            "dataset_spec": dataset_spec.to_dict(),
            "source_dir": str(source_root),
            "llm": str(tiny_model),
            "epochs": 12,
            "batch_size": 8,
            "conditional_col": "target",
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
        raise AssertionError("TabuLa adapter leaked Python RNG state")
    numpy_after = np.random.get_state()
    if any(
        not np.array_equal(before, after) if isinstance(before, np.ndarray) else before != after
        for before, after in zip(numpy_before, numpy_after)
    ):
        raise AssertionError("TabuLa adapter leaked NumPy RNG state")
    if not torch.equal(torch_before, torch.random.get_rng_state()) or torch.get_num_threads() != threads_before:
        raise AssertionError("TabuLa adapter leaked PyTorch RNG/thread state")
    reloaded = AutoModelForCausalLM.from_pretrained(train_spec.output_dir / "tabula_model" / "transformer")
    _assert_state_exact(direct_state, _state_dict(reloaded))
    start_dist = {str(index): 1.0 / 2.0 for index in range(2)}
    with adapter._scoped_randomness(seed, 1), adapter._sampling_timeout(300, allow_unbounded=False):
        direct_batches: list[pd.DataFrame] = []
        direct_remaining = SAMPLE_ROWS
        direct_empty_batches = 0
        while direct_remaining:
            direct_batch = direct.sample(
                n_samples=direct_remaining,
                start_col="target",
                start_col_dist=start_dist,
                temperature=0.5,
                k=SAMPLE_ROWS,
                max_length=64,
                device="cpu",
            )
            if not isinstance(direct_batch, pd.DataFrame) or len(direct_batch) > direct_remaining:
                raise AssertionError("Direct TabuLa source violated its per-call row boundary")
            if direct_batch.empty:
                direct_empty_batches += 1
                if direct_empty_batches >= 8:
                    raise AssertionError("Direct TabuLa source repeatedly returned zero usable rows")
                continue
            direct_empty_batches = 0
            direct_batches.append(direct_batch)
            direct_remaining -= len(direct_batch)
        direct_sample = pd.concat(direct_batches, ignore_index=True).head(SAMPLE_ROWS)
    sample_spec = RunSpec(
        model="tabula",
        dataset=dataset_spec.name,
        output_dir=train_spec.output_dir,
        seed=seed,
        num_samples=SAMPLE_ROWS,
        device="cpu",
        extra={
            "dataset_spec": dataset_spec.to_dict(),
            "source_dir": str(source_root),
            "start_col": "target",
            "start_col_dist": start_dist,
            "temperature": 0.5,
            "k": SAMPLE_ROWS,
            "max_length": 64,
            "timeout_seconds": 300,
            "num_threads": 1,
            "max_empty_batches": 8,
        },
    )
    bundle = adapter.sample(sample_spec)
    observed = pd.read_csv(bundle.generated_sample_path, dtype=str, keep_default_na=False)
    expected = direct_sample[dataset_spec.column_names].astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=True)
    if bundle.generated_sample_path.read_bytes() != expected.to_csv(index=False).encode("utf-8"):
        raise AssertionError("TabuLa adapter CSV bytes differ from direct method-author source")
    return {
        "seed": seed,
        "train_rows": len(frame),
        "sample_rows": len(observed),
        "state_tensors_exact": len(direct_state),
        "csv_sha256": sha256_file(bundle.generated_sample_path),
        "integrity_sha256": sha256_file(train_spec.output_dir / "tabula_model" / "tabula-integrity.json"),
        "exact": True,
    }


def run_protocol(
    repo_root: Path,
    source_root: Path,
    archive_path: Path,
    output_dir: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    archive = _verify_archive(archive_path)
    source_before = validate_upstream_source("tabula", source_root)
    if source_before["manifest_sha256"] != sha256_file(source_manifest_path("tabula")):
        raise ValueError("TabuLa source manifest identity differs from the lock")
    frame, dataset_spec = _frame()
    tiny_model = output_dir / "tiny-gpt2"
    tiny = build_tiny_gpt2(tiny_model, _texts(frame))
    cases = [
        _run_case(repo_root, source_root, output_dir, tiny_model, dataset_spec, frame, seed) for seed in SEEDS
    ]
    source_after = validate_upstream_source("tabula", source_root)
    if source_after != source_before:
        raise AssertionError("TabuLa source changed during validation")
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabula",
        "status": "pass",
        "repository_commit": _repository_commit(repo_root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in (
                "accelerate",
                "datasets",
                "numpy",
                "pandas",
                "safetensors",
                "scikit-learn",
                "torch",
                "transformers",
            )
        },
        "archive": archive,
        "source": source_before,
        "tiny_offline_checkpoint": tiny,
        "cases": cases,
        "result_summary": {
            "cases_passed": len(cases),
            "cases_total": len(SEEDS),
            "source_unchanged": True,
            "official_fit_executed": True,
            "official_sample_executed": True,
            "state_tensors_exact": True,
            "safe_safetensors_checkpoint": True,
            "sample_frames_exact": True,
            "sample_csv_bytes_exact": True,
            "rng_and_thread_state_restored": True,
        },
        "claim_limit": (
            "Parity covers the locked method-author TabuLa source with a deterministic offline tiny GPT-2 config. "
            "The upstream repository has no declared license, so redistribution, Official Results, and release "
            "support remain blocked regardless of technical parity."
        ),
    }
    atomic_write_json(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_protocol(
            args.repo_root.resolve(),
            args.source_root.resolve(),
            args.archive_path.resolve(),
            args.output_dir,
            args.evidence_path,
        )
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            args.evidence_path,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "model_id": "tabula",
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
