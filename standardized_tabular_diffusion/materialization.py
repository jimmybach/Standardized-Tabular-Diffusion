from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.datasets import get_dataset_spec, validate_dataset_name
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json


def materialization_root(repo_root: Path | None = None) -> Path:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    return repo_root / "materialized_datasets"


def manifest_path(dataset_name: str, repo_root: Path | None = None) -> Path:
    validate_dataset_name(dataset_name)
    return materialization_root(repo_root=repo_root) / dataset_name / "manifest.json"


def load_manifest(dataset_name: str, repo_root: Path | None = None) -> dict[str, Any] | None:
    path = manifest_path(dataset_name, repo_root=repo_root)
    if not path.exists():
        return None
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Materialization manifest must be a JSON object: {path}")
    return manifest


def _run_python(args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def _build_manifest(dataset_name: str, repo_root: Path) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    upstream_root = repo_root / "TabDiff-main"
    data_dir = upstream_root / "data" / dataset_name
    synthetic_dir = upstream_root / "synthetic" / dataset_name
    info_path = data_dir / "info.json"

    def repository_path(path: Path) -> str:
        return path.relative_to(repo_root).as_posix()

    manifest = {
        "manifest_schema_version": "1.0.0",
        "path_base": "repository-root",
        "dataset": dataset_name,
        "materialized_by": "TabDiff-main",
        "metadata_path": repository_path(info_path),
        "train_data_path": repository_path(data_dir / "train.csv"),
        "val_data_path": repository_path(data_dir / "val.csv") if (data_dir / "val.csv").exists() else None,
        "test_data_path": repository_path(data_dir / "test.csv"),
        "synthetic_real_path": (
            repository_path(synthetic_dir / "real.csv") if (synthetic_dir / "real.csv").exists() else None
        ),
        "synthetic_val_path": (
            repository_path(synthetic_dir / "val.csv") if (synthetic_dir / "val.csv").exists() else None
        ),
        "synthetic_test_path": (
            repository_path(synthetic_dir / "test.csv") if (synthetic_dir / "test.csv").exists() else None
        ),
        "exists": {
            "metadata_path": info_path.exists(),
            "train_data_path": (data_dir / "train.csv").exists(),
            "val_data_path": (data_dir / "val.csv").exists(),
            "test_data_path": (data_dir / "test.csv").exists(),
            "synthetic_real_path": (synthetic_dir / "real.csv").exists(),
            "synthetic_val_path": (synthetic_dir / "val.csv").exists(),
            "synthetic_test_path": (synthetic_dir / "test.csv").exists(),
        },
        "synced_roots": {},
    }
    return manifest


def _sync_processed_dataset(dataset_name: str, repo_root: Path) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    sync_report: dict[str, Any] = {}
    source_data_dir = repo_root / "TabDiff-main" / "data" / dataset_name
    source_synth_dir = repo_root / "TabDiff-main" / "synthetic" / dataset_name

    targets = [
        repo_root / "TabSyn-main",
    ]
    for target_root in targets:
        target_data_dir = target_root / "data" / dataset_name
        target_synth_dir = target_root / "synthetic" / dataset_name
        target_data_dir.parent.mkdir(parents=True, exist_ok=True)
        target_synth_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_data_dir, target_data_dir, dirs_exist_ok=True)
        shutil.copytree(source_synth_dir, target_synth_dir, dirs_exist_ok=True)
        sync_report[target_root.relative_to(repo_root).as_posix()] = {
            "data_dir": target_data_dir.relative_to(repo_root).as_posix(),
            "synthetic_dir": target_synth_dir.relative_to(repo_root).as_posix(),
        }
    return sync_report


def materialize_dataset(
    dataset_name: str,
    repo_root: Path | None = None,
    *,
    cache_root: str | Path | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    if dataset_name in {"adult", "sick"}:
        from standardized_tabular_diffusion.official_datasets import (
            materialize_official_adult,
            materialize_official_sick,
        )

        builder = materialize_official_adult if dataset_name == "adult" else materialize_official_sick
        return builder(
            repo_root=repo_root,
            cache_root=cache_root,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
        )
    upstream_root = repo_root / "TabDiff-main"

    _run_python(
        [
            "-c",
            "import sys; import download_dataset; download_dataset.download_from_uci(sys.argv[1])",
            dataset_name,
        ],
        upstream_root,
    )
    _run_python(["process_dataset.py", "--dataname", dataset_name], upstream_root)

    manifest = _build_manifest(dataset_name, repo_root)
    manifest["synced_roots"] = _sync_processed_dataset(dataset_name, repo_root)
    out_path = manifest_path(dataset_name, repo_root=repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, manifest)
    return manifest


def materialization_status(dataset_name: str, repo_root: Path | None = None) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    base_spec = get_dataset_spec(dataset_name, repo_root=repo_root)
    manifest = load_manifest(dataset_name, repo_root=repo_root)

    return {
        "dataset": dataset_name,
        "base_spec": base_spec.to_dict(),
        "manifest": manifest,
        "materialized": manifest is not None,
    }
