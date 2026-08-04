from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.tabsyn import TabSynAdapter

PROTOCOL_ID = "tabsyn-native-parity-v1"
MANIFEST_RELATIVE_PATH = Path("standardized_tabular_diffusion/resources/upstream/tabsyn-source-manifest.json")
DATASET_NAME = "tabsyn_parity"
EXPECTED_SAMPLE_ROWS = 12
SEED = 19


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_lf(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"
    return _sha256_bytes(normalized)


def verify_sources(repo_root: Path) -> dict[str, Any]:
    """Fail closed unless every scoped file matches the pinned official TabSyn tree."""

    manifest_path = repo_root / MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream_root = repo_root / "TabSyn-main"
    failures: list[dict[str, str]] = []
    for record in manifest["files"]:
        path = upstream_root / record["path"]
        actual = "missing" if not path.is_file() else _sha256_lf(path)
        if actual != record["sha256_lf"]:
            failures.append({"path": str(path), "expected": record["sha256_lf"], "actual": actual})
    local_zero = sorted(str(path.relative_to(repo_root)) for path in (upstream_root / "zero").glob("*.py"))
    if local_zero:
        failures.append(
            {
                "path": "TabSyn-main/zero",
                "expected": "absent; use the frozen libzero distribution",
                "actual": ", ".join(local_zero),
            }
        )
    if failures:
        raise RuntimeError(f"TabSyn source-integrity validation failed: {json.dumps(failures, indent=2)}")
    return {
        "manifest_path": str(MANIFEST_RELATIVE_PATH),
        "manifest_sha256": _sha256_file(manifest_path),
        "upstream_files_verified": len(manifest["files"]),
        "upstream_commit": manifest["upstream_commit"],
        "upstream_tree": manifest["upstream_tree"],
        "dependency_resolution": manifest["dependencies"],
    }


def _copy_verified_source(repo_root: Path, destination: Path) -> None:
    manifest = json.loads((repo_root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    source_root = repo_root / "TabSyn-main"
    for record in manifest["files"]:
        source = source_root / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_fixture(upstream_root: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    data_dir = upstream_root / "data" / DATASET_NAME
    data_dir.mkdir(parents=True)

    def split(count: int, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        row = np.arange(offset, offset + count)
        first = (row * 0.31 + (row % 4) * 0.07).astype(np.float32)
        second = ((row * 7) % 19).astype(np.float32)
        group = np.array([f"g{value % 3}" for value in row])
        target = ((row + row // 2) % 2).astype(np.int64)
        frame = pd.DataFrame({"0": first, "1": second, "2": group, "3": target})
        return np.column_stack([first, second]), group.reshape(-1, 1), target, frame

    train = split(24, 0)
    test = split(12, 100)
    for name, payload in (("train", train), ("test", test)):
        numerical, categorical, target, frame = payload
        np.save(data_dir / f"X_num_{name}.npy", numerical)
        np.save(data_dir / f"X_cat_{name}.npy", categorical)
        np.save(data_dir / f"y_{name}.npy", target)
        frame.to_csv(data_dir / f"{name}.csv", index=False)
    info = {
        "name": DATASET_NAME,
        "task_type": "binclass",
        "column_names": ["0", "1", "2", "3"],
        "num_col_idx": [0, 1],
        "cat_col_idx": [2],
        "target_col_idx": [3],
        "idx_mapping": {str(index): index for index in range(4)},
        "inverse_idx_mapping": {str(index): index for index in range(4)},
        "idx_name_mapping": {str(index): str(index) for index in range(4)},
        "train_num": 24,
        "test_num": 12,
        "n_classes": 2,
    }
    (data_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {
        "kind": "deterministic-mixed-type-binary-classification",
        "dataset_name": DATASET_NAME,
        "train_rows": 24,
        "test_rows": 12,
        "numerical_features": 2,
        "categorical_features": 1,
        "target_features": 1,
        "missing_values": 0,
    }


def _replace_once(path: Path, old: str, new: str) -> dict[str, str]:
    before = path.read_text(encoding="utf-8")
    if before.count(old) != 1:
        raise RuntimeError(f"Expected one validation override target in {path}: {old!r}")
    base_sha256 = _sha256_file(path)
    path.write_text(before.replace(old, new), encoding="utf-8")
    return {"base_sha256": base_sha256, "execution_sha256": _sha256_file(path)}


def _configure_runtime(upstream_root: Path) -> dict[str, Any]:
    overrides: list[dict[str, Any]] = []

    def replace(relative: str, old: str, new: str, purpose: str) -> None:
        record = _replace_once(upstream_root / relative, old, new)
        overrides.append({"path": relative, "from": old, "to": new, "purpose": purpose, **record})

    replace("tabsyn/vae/main.py", "num_workers = 4", "num_workers = 0", "bounded CI worker count")
    replace("tabsyn/vae/main.py", "num_epochs = 4000", "num_epochs = 2", "bounded VAE smoke training")
    replace("tabsyn/main.py", "num_workers = 4", "num_workers = 0", "bounded CI worker count")
    replace("tabsyn/main.py", "num_epochs = 10000 + 1", "num_epochs = 2", "bounded diffusion smoke training")
    replace("tabsyn/main.py", "MLPDiffusion(in_dim, 1024)", "MLPDiffusion(in_dim, 64)", "bounded MLP width")
    replace("tabsyn/sample.py", "MLPDiffusion(in_dim, 1024)", "MLPDiffusion(in_dim, 64)", "match training width")
    replace(
        "tabsyn/sample.py",
        "num_samples = train_z.shape[0]",
        f"num_samples = {EXPECTED_SAMPLE_ROWS}",
        "bounded requested row count",
    )
    replace(
        "tabsyn/sample.py",
        "sample(model.denoise_fn_D, num_samples, sample_dim)",
        "sample(model.denoise_fn_D, num_samples, sample_dim, num_steps=steps, device=device)",
        "explicit official sampler controls for CPU execution",
    )
    sitecustomize = upstream_root / "sitecustomize.py"
    sitecustomize.write_text(
        "import os\n"
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "seed = int(os.environ['TABSYN_VALIDATION_SEED'])\n"
        "random.seed(seed)\n"
        "np.random.seed(seed)\n"
        "torch.manual_seed(seed)\n"
        "torch.set_num_threads(1)\n",
        encoding="utf-8",
    )
    return {
        "treatment": "validation-only execution overrides applied equally to two isolated verified copies",
        "tracked_upstream_source_modified": False,
        "overrides": overrides,
        "sitecustomize_sha256": _sha256_file(sitecustomize),
    }


def _environment(upstream_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(SEED),
            "PYTHONPATH": str(upstream_root),
            "STANDARDIZED_TABSYN_NUM_THREADS": "1",
            "TABSYN_VALIDATION_SEED": str(SEED),
        }
    )
    return environment


def _run(command: list[str], upstream_root: Path) -> None:
    subprocess.run(command, cwd=upstream_root, check=True, env=_environment(upstream_root))


def _native_commands(upstream_root: Path, sample_path: Path) -> list[list[str]]:
    commands = [
        [sys.executable, "main.py", "--dataname", DATASET_NAME, "--method", "vae", "--mode", "train", "--gpu", "-1"],
        [sys.executable, "main.py", "--dataname", DATASET_NAME, "--method", "tabsyn", "--mode", "train", "--gpu", "-1"],
        [
            sys.executable,
            "main.py",
            "--dataname",
            DATASET_NAME,
            "--method",
            "tabsyn",
            "--mode",
            "sample",
            "--gpu",
            "-1",
            "--steps",
            "4",
            "--save_path",
            str(sample_path),
        ],
    ]
    for command in commands:
        _run(command, upstream_root)
    return commands


def _adapter_run(repo_root: Path, upstream_root: Path, output_root: Path) -> tuple[TabSynAdapter, list[Path]]:
    os.environ["STANDARDIZED_TABSYN_NUM_THREADS"] = "1"
    adapter = TabSynAdapter(repo_root)
    adapter.upstream_root = upstream_root
    train_bundle = adapter.train(
        RunSpec(model="tabsyn", dataset=DATASET_NAME, output_dir=output_root / "train", device="cpu", seed=SEED)
    )
    sample_bundle = adapter.sample(
        RunSpec(
            model="tabsyn",
            dataset=DATASET_NAME,
            output_dir=output_root / "sample",
            device="cpu",
            seed=SEED,
            num_samples=EXPECTED_SAMPLE_ROWS,
            extra={"steps": 4},
        )
    )
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("TabSyn adapter did not record its generated sample path.")
    manifests = [train_bundle.output_dir / "artifacts.json", sample_bundle.output_dir / "artifacts.json"]
    return adapter, manifests


def _compare_state_dicts(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import torch

    native = torch.load(native_path, map_location="cpu")
    adapter = torch.load(adapter_path, map_location="cpu")
    keys_exact = list(native) == list(adapter)
    mismatches = [key for key in native if key not in adapter or not torch.equal(native[key], adapter[key])]
    return {
        "keys_exact": keys_exact,
        "tensor_values_exact": not mismatches,
        "mismatches": mismatches,
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _compare_array(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import numpy as np

    native = np.load(native_path, allow_pickle=False)
    adapter = np.load(adapter_path, allow_pickle=False)
    return {
        "exact": bool(np.array_equal(native, adapter)),
        "finite": bool(np.isfinite(native).all()),
        "shape": list(native.shape),
        "dtype": str(native.dtype),
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _compare_samples(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    native = pd.read_csv(native_path)
    adapter = pd.read_csv(adapter_path)
    numerical = native[["0", "1"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    return {
        "exact_bytes": native_path.read_bytes() == adapter_path.read_bytes(),
        "exact_frame": native.equals(adapter),
        "rows": len(native),
        "columns": list(native.columns),
        "finite_numerical_values": bool(np.isfinite(numerical).all()),
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def run_validation(repo_root: Path, output_dir: Path, evidence_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = verify_sources(repo_root)

    native_root = output_dir / "native" / "TabSyn-main"
    adapter_root = output_dir / "adapter" / "TabSyn-main"
    _copy_verified_source(repo_root, native_root)
    _copy_verified_source(repo_root, adapter_root)
    fixture = _write_fixture(native_root)
    _write_fixture(adapter_root)
    native_runtime = _configure_runtime(native_root)
    adapter_runtime = _configure_runtime(adapter_root)
    if native_runtime != adapter_runtime:
        raise AssertionError("Native and adapter runtime controls differ.")

    native_sample = output_dir / "native-samples.csv"
    native_commands = _native_commands(native_root, native_sample)
    adapter, manifests = _adapter_run(repo_root, adapter_root, output_dir / "adapter-manifests")
    adapter_sample = output_dir / "adapter-manifests" / "sample" / "samples.csv"

    relative_checkpoints = [
        "tabsyn/vae/ckpt/tabsyn_parity/model.pt",
        "tabsyn/vae/ckpt/tabsyn_parity/encoder.pt",
        "tabsyn/vae/ckpt/tabsyn_parity/decoder.pt",
        "tabsyn/ckpt/tabsyn_parity/model.pt",
        "tabsyn/ckpt/tabsyn_parity/model_0.pt",
    ]
    checkpoints = {
        relative: _compare_state_dicts(native_root / relative, adapter_root / relative)
        for relative in relative_checkpoints
    }
    latent = _compare_array(
        native_root / "tabsyn/vae/ckpt/tabsyn_parity/train_z.npy",
        adapter_root / "tabsyn/vae/ckpt/tabsyn_parity/train_z.npy",
    )
    samples = _compare_samples(native_sample, adapter_sample)
    manifests_valid = all(
        json.loads(path.read_text(encoding="utf-8"))["model"] == "tabsyn" for path in manifests
    )
    passed = (
        all(record["keys_exact"] and record["tensor_values_exact"] for record in checkpoints.values())
        and latent["exact"]
        and latent["finite"]
        and samples["exact_bytes"]
        and samples["finite_numerical_values"]
        and samples["rows"] == EXPECTED_SAMPLE_ROWS
        and samples["columns"] == ["0", "1", "2", "3"]
        and manifests_valid
    )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabsyn",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "environment_lock": {
            "path": "requirements-tabsyn-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-tabsyn-validation.txt"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _version("torch"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit_learn": _version("scikit-learn"),
            "scipy": _version("scipy"),
            "category_encoders": _version("category-encoders"),
            "libzero": _version("libzero"),
        },
        "fixture": fixture,
        "runtime_config": native_runtime,
        "seed": SEED,
        "native_commands": native_commands,
        "adapter_commands": {
            "vae_train": ["compat/tabsyn.py", "--action", "vae-train"],
            "diffusion_train": ["compat/tabsyn.py", "--action", "diffusion-train"],
            "sample": ["compat/tabsyn.py", "--action", "sample", "--num-samples", "12", "--steps", "4"],
        },
        "comparisons": {
            "checkpoints": checkpoints,
            "latent_embeddings": latent,
            "samples": samples,
            "adapter_manifests_valid": manifests_valid,
        },
        "adapter_upstream_root": str(adapter.upstream_root),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("TabSyn native-parity protocol failed; inspect the evidence record.")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TabSyn native-parity validation protocol.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_validation(args.repo_root, args.output_dir, args.evidence_path)
    except Exception as exc:
        if not args.evidence_path.exists():
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": PROTOCOL_ID,
                        "model_id": "tabsyn",
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
