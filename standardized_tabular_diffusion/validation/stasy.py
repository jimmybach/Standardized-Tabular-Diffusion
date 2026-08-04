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
from standardized_tabular_diffusion.models.vendored_baselines import STaSyAdapter
from standardized_tabular_diffusion.upstream_sources import load_source_manifest, validate_upstream_source

PROTOCOL_ID = "stasy-tabsyn-snapshot-parity-v1"
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
EXPECTED_SAMPLE_ROWS = 12


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _copy_verified_source(repo_root: Path, destination: Path) -> dict[str, Any]:
    source = validate_upstream_source("stasy", repo_root / "TabSyn-main")
    manifest = load_source_manifest("stasy")
    for record in manifest["runtime_files"]:
        source_path = repo_root / "TabSyn-main" / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    return source


def _write_fixture(upstream_root: Path, variant: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    dataset_name = f"stasy_parity_{variant}"
    data_dir = upstream_root / "data" / dataset_name
    data_dir.mkdir(parents=True)

    def split(count: int, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        row = np.arange(offset, offset + count)
        first = (row * 0.19 + (row % 3) * 0.11).astype(np.float32)
        second = (((row * 5) % 17) / 3.0).astype(np.float32)
        group = np.array([f"g{value % 3}" for value in row])
        if variant == "binary":
            target = ((row + row // 3) % 2).astype(np.int64)
        elif variant == "multiclass":
            target = ((row * 2 + row // 2) % 3).astype(np.int64)
        elif variant == "regression":
            target = (first * 0.4 - second * 0.2 + (row % 4) * 0.05).astype(np.float32)
        else:
            raise ValueError(f"Unknown STaSy validation variant: {variant}")
        frame = pd.DataFrame({"0": first, "1": second, "2": group, "3": target})
        return np.column_stack([first, second]), group.reshape(-1, 1), target, frame

    train = split(EXPECTED_SAMPLE_ROWS, 0)
    test = split(6, 100)
    for split_name, payload in (("train", train), ("test", test)):
        numerical, categorical, target, frame = payload
        np.save(data_dir / f"X_num_{split_name}.npy", numerical)
        np.save(data_dir / f"X_cat_{split_name}.npy", categorical)
        np.save(data_dir / f"y_{split_name}.npy", target)
        frame.to_csv(data_dir / f"{split_name}.csv", index=False)
    task_type = {"binary": "binclass", "multiclass": "multiclass", "regression": "regression"}[variant]
    info = {
        "name": dataset_name,
        "task_type": task_type,
        "column_names": ["0", "1", "2", "3"],
        "num_col_idx": [0, 1],
        "cat_col_idx": [2],
        "target_col_idx": [3],
        "idx_mapping": {str(index): index for index in range(4)},
        "inverse_idx_mapping": {str(index): index for index in range(4)},
        "idx_name_mapping": {str(index): str(index) for index in range(4)},
        "train_num": EXPECTED_SAMPLE_ROWS,
        "test_num": 6,
    }
    if variant != "regression":
        info["n_classes"] = 2 if variant == "binary" else 3
    (data_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset_name": dataset_name,
        "variant": variant,
        "task_type": task_type,
        "train_rows": EXPECTED_SAMPLE_ROWS,
        "test_rows": 6,
        "numerical_features": 2,
        "categorical_features": 1,
        "target_features": 1,
        "missing_values": 0,
    }


def _replace_once(path: Path, old: str, new: str, purpose: str) -> dict[str, str]:
    before = path.read_text(encoding="utf-8")
    if before.count(old) != 1:
        raise RuntimeError(f"Expected one STaSy validation override target in {path}: {old!r}")
    base_sha256 = _sha256_file(path)
    path.write_text(before.replace(old, new), encoding="utf-8")
    return {
        "path": path.as_posix(),
        "from": old,
        "to": new,
        "purpose": purpose,
        "base_sha256": base_sha256,
        "execution_sha256": _sha256_file(path),
    }


def _configure_native_runtime(upstream_root: Path) -> dict[str, Any]:
    records = [
        _replace_once(
            upstream_root / "baselines/stasy/configs/default_tabular_configs.py",
            "training.epoch = 10000",
            "training.epoch = 0",
            "one inclusive-loop smoke epoch",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/configs/default_tabular_configs.py",
            "model.num_scales = 50",
            "model.num_scales = 2",
            "bounded predictor-corrector sampling",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/configs/config.py",
            "config.training.batch_size = 1000",
            "config.training.batch_size = 8",
            "bounded smoke batch size",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/configs/config.py",
            "config.eval.batch_size = 1000",
            "config.eval.batch_size = 8",
            "bounded smoke evaluation batch size",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/configs/config.py",
            "sampling.method = 'ode'",
            "sampling.method = 'pc'",
            "bounded official predictor-corrector sampler",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/configs/config.py",
            "model.nf = 64",
            "model.nf = 8",
            "bounded smoke Fourier width",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/configs/config.py",
            "model.hidden_dims = (1024, 2048, 1024, 1024)",
            "model.hidden_dims = (16, 16)",
            "bounded smoke score-network width",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/main.py",
            "config.device = torch.device(f'cuda:{args.gpu}')",
            "config.device = torch.device('cpu')",
            "CPU validation device",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/main.py",
            "num_workers=4",
            "num_workers=0",
            "deterministic validation loader",
        ),
        _replace_once(
            upstream_root / "baselines/stasy/sample.py",
            "config.device = torch.device(f'cuda:{args.gpu}')",
            "config.device = torch.device('cpu')",
            "CPU validation device",
        ),
        _replace_once(
            upstream_root / "src/data.py",
            "handle_unknown='ignore', sparse=False, dtype=np.float32",
            "handle_unknown='ignore', sparse_output=False, dtype=np.float32",
            "forward the unchanged dense-output boolean to scikit-learn's renamed keyword",
        ),
    ]
    return {
        "treatment": "native-reference-only temporary overrides; tracked repository source remains exact",
        "adapter_equivalent_config": {
            "epochs": 1,
            "batch_size": 8,
            "nf": 8,
            "hidden_dims": [16, 16],
            "num_scales": 2,
            "num_workers": 0,
            "num_threads": 1,
            "sampler": "pc",
            "spl": True,
        },
        "compatibility_bridge": {
            "id": "stasy-sklearn-onehot-keyword-v1",
            "from": "sparse=False",
            "to": "sparse_output=False",
            "semantic_effect": "none; the same OneHotEncoder dense-output boolean is forwarded",
        },
        "overrides": records,
    }


def _write_sitecustomize(upstream_root: Path) -> str:
    path = upstream_root / "sitecustomize.py"
    path.write_text(
        "import os\n"
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "seed = int(os.environ['STASY_VALIDATION_SEED'])\n"
        "random.seed(seed)\n"
        "np.random.seed(seed)\n"
        "torch.manual_seed(seed)\n"
        "torch.set_num_threads(1)\n",
        encoding="utf-8",
    )
    return _sha256_file(path)


def _environment(upstream_root: Path, seed: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(seed),
            "PYTHONPATH": str(upstream_root),
            "STASY_VALIDATION_SEED": str(seed),
        }
    )
    return environment


def _run_native(upstream_root: Path, dataset: str, sample_path: Path, seed: int) -> list[list[str]]:
    commands = [
        [sys.executable, "main.py", "--method", "stasy", "--mode", "train", "--dataname", dataset, "--gpu", "0"],
        [
            sys.executable,
            "main.py",
            "--method",
            "stasy",
            "--mode",
            "sample",
            "--dataname",
            dataset,
            "--gpu",
            "0",
            "--save_path",
            str(sample_path),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=upstream_root, check=True, env=_environment(upstream_root, seed))
    return commands


def _run_adapter(
    repo_root: Path,
    upstream_root: Path,
    output_dir: Path,
    fixture: dict[str, Any],
    seed: int,
) -> tuple[list[Path], Path]:
    adapter = STaSyAdapter(repo_root)
    adapter.upstream_root = upstream_root
    config = {
        "epochs": 1,
        "batch_size": 8,
        "nf": 8,
        "hidden_dims": [16, 16],
        "num_scales": 2,
        "num_workers": 0,
        "num_threads": 1,
        "sampler": "pc",
        "spl": True,
    }
    train_bundle = adapter.train(
        RunSpec(
            model="stasy",
            dataset=fixture["dataset_name"],
            output_dir=output_dir,
            device="cpu",
            seed=seed,
            extra=config,
        )
    )
    sample_bundle = adapter.sample(
        RunSpec(
            model="stasy",
            dataset=fixture["dataset_name"],
            output_dir=output_dir,
            device="cpu",
            seed=seed,
            num_samples=EXPECTED_SAMPLE_ROWS,
            extra={"num_threads": 1, "sampler": "pc"},
        )
    )
    if sample_bundle.generated_sample_path is None:
        raise AssertionError("STaSy adapter did not record its sample path.")
    manifests = [train_bundle.output_dir / "artifacts.json", sample_bundle.output_dir / "artifacts.json"]
    return manifests, sample_bundle.generated_sample_path


def _equal_state(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(_equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _compare_checkpoints(native_path: Path, adapter_path: Path) -> dict[str, Any]:
    import torch

    native = torch.load(native_path, map_location="cpu")
    adapter = torch.load(adapter_path, map_location="cpu")
    sections = {key: _equal_state(native.get(key), adapter.get(key)) for key in ("model", "optimizer", "ema")}
    scalars = {key: native.get(key) == adapter.get(key) for key in ("step", "epoch")}
    return {
        "sections_exact": sections,
        "scalars_exact": scalars,
        "state_exact": all(sections.values()) and all(scalars.values()),
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
        "missing_values": int(native.isna().sum().sum()),
        "finite_numerical": bool(np.isfinite(numerical).all()),
        "native_sha256": _sha256_file(native_path),
        "adapter_sha256": _sha256_file(adapter_path),
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment_versions() -> dict[str, str]:
    actual = {
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("scikit-learn"),
        "scipy": _version("scipy"),
        "category_encoders": _version("category-encoders"),
        "ml_collections": _version("ml-collections"),
        "libzero": _version("libzero"),
    }
    expected = {
        "torch": "2.3.0",
        "numpy": "1.26.4",
        "pandas": "2.2.3",
        "scikit_learn": "1.5.2",
        "scipy": "1.13.1",
        "category_encoders": "2.6.4",
        "ml_collections": "0.1.1",
        "libzero": "0.0.8",
    }
    normalized = {**actual, "torch": actual["torch"].split("+")[0]}
    if normalized != expected:
        raise RuntimeError(f"STaSy validation environment mismatch: expected={expected}, actual={actual}")
    return actual


def run_validation(repo_root: Path, output_dir: Path, evidence_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"STaSy validation output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = validate_upstream_source("stasy", repo_root / "TabSyn-main")
    environment = _environment_versions()
    cases: list[dict[str, Any]] = []
    runtime_record: dict[str, Any] | None = None
    case_number = 0
    for variant in VARIANTS:
        for seed in SEED_CASES:
            case_number += 1
            case_root = output_dir / f"case-{case_number:02d}-{variant}-seed-{seed}"
            native_root = case_root / "native" / "TabSyn-main"
            adapter_root = case_root / "adapter" / "TabSyn-main"
            _copy_verified_source(repo_root, native_root)
            _copy_verified_source(repo_root, adapter_root)
            native_fixture = _write_fixture(native_root, variant)
            adapter_fixture = _write_fixture(adapter_root, variant)
            if native_fixture != adapter_fixture:
                raise AssertionError("STaSy native and adapter fixtures differ.")
            native_runtime = _configure_native_runtime(native_root)
            sitecustomize_sha256 = _write_sitecustomize(native_root)
            if runtime_record is None:
                runtime_record = native_runtime
            elif runtime_record["adapter_equivalent_config"] != native_runtime["adapter_equivalent_config"]:
                raise AssertionError("STaSy runtime controls changed between cases.")
            native_sample = case_root / "native-samples.csv"
            native_commands = _run_native(native_root, native_fixture["dataset_name"], native_sample, seed)
            adapter_output = case_root / "adapter-output"
            manifests, adapter_sample = _run_adapter(repo_root, adapter_root, adapter_output, adapter_fixture, seed)
            native_checkpoint = (
                native_root / "baselines" / "stasy" / "ckpt" / native_fixture["dataset_name"] / "model.pth"
            )
            adapter_checkpoint = adapter_output / "ckpt" / adapter_fixture["dataset_name"] / "model.pth"
            checkpoint = _compare_checkpoints(native_checkpoint, adapter_checkpoint)
            samples = _compare_samples(native_sample, adapter_sample)
            metadata = json.loads((adapter_output / "stasy-model-metadata.json").read_text(encoding="utf-8"))
            sample_metadata = json.loads(
                (adapter_output / "stasy-sample-metadata.json").read_text(encoding="utf-8")
            )
            manifests_valid = all(
                json.loads(path.read_text(encoding="utf-8"))["model"] == "stasy" for path in manifests
            )
            adapter_source_after = validate_upstream_source("stasy", adapter_root)
            source_pure = not (adapter_root / "baselines" / "stasy" / "ckpt").exists()
            case_passed = (
                checkpoint["state_exact"]
                and samples["exact_bytes"]
                and samples["exact_frame"]
                and samples["rows"] == EXPECTED_SAMPLE_ROWS
                and samples["columns"] == ["0", "1", "2", "3"]
                and samples["missing_values"] == 0
                and samples["finite_numerical"]
                and manifests_valid
                and metadata["source"]["runtime_files_verified"] == 30
                and metadata["training_config"] == native_runtime["adapter_equivalent_config"]
                and sample_metadata["rows"] == EXPECTED_SAMPLE_ROWS
                and adapter_source_after["manifest_sha256"] == source["manifest_sha256"]
                and source_pure
            )
            cases.append(
                {
                    "case": case_number,
                    "variant": variant,
                    "seed": seed,
                    "status": "pass" if case_passed else "fail",
                    "fixture": native_fixture,
                    "native_commands": native_commands,
                    "adapter_runtime": metadata["training_config"],
                    "native_sitecustomize_sha256": sitecustomize_sha256,
                    "comparisons": {
                        "checkpoint": checkpoint,
                        "samples": samples,
                        "adapter_manifests_valid": manifests_valid,
                        "adapter_metadata_valid": metadata["source"]["runtime_files_verified"] == 30,
                        "adapter_source_remained_exact": adapter_source_after["manifest_sha256"]
                        == source["manifest_sha256"],
                        "adapter_checkpoint_outside_source": source_pure,
                    },
                }
            )
    if runtime_record is None:
        raise AssertionError("STaSy validation executed no cases.")
    passed = all(case["status"] == "pass" for case in cases)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "stasy",
        "reproduction_target": "tabsyn-benchmark-snapshot",
        "status": "pass" if passed else "fail",
        "repository_commit": _repository_commit(repo_root),
        "source": source,
        "method_author_boundary": load_source_manifest("stasy")["method_author_repository"],
        "environment_lock": {
            "path": "requirements-stasy-validation.txt",
            "sha256": _sha256_file(repo_root / "requirements-stasy-validation.txt"),
        },
        "environment": {"platform": platform.platform(), "python": platform.python_version(), **environment},
        "runtime_config": runtime_record,
        "seed_cases": list(SEED_CASES),
        "variants": list(VARIANTS),
        "cases": cases,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("STaSy TabSyn-snapshot parity protocol failed; inspect the evidence record.")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the STaSy TabSyn-snapshot parity protocol.")
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
                        "model_id": "stasy",
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
