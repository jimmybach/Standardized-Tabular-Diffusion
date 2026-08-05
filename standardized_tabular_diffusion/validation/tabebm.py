"""Gated-aware validation protocol for the official TabEBM package boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import tarfile
import traceback
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import scipy.special

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.tabebm import (
    TABEBM_PACKAGE_VERSION,
    TABEBM_RUNTIME_SHA256,
    TABEBM_SDIST_SHA256,
    TABEBM_UPSTREAM_COMMIT,
    TabEBMAdapter,
    verify_tabebm_distribution,
)

PROTOCOL_ID = "tabebm-official-package-core-validation-v1"
SDIST_FILENAME = "tabebm-2025.8.19.tar.gz"
SDIST_BYTES = 19_178
EXPECTED_ARCHIVE_FILES = 12
EXPECTED_SOURCE_FILES = {
    "src/tabebm/TabEBM.py": TABEBM_RUNTIME_SHA256["tabebm/TabEBM.py"],
    "src/tabebm/__init__.py": TABEBM_RUNTIME_SHA256["tabebm/__init__.py"],
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "setup.py": "845c989600f9a40006383a1a733bd1ee0809028d297f507b4154be9a11e48707",
}


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_sdist(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.name != SDIST_FILENAME:
        raise ValueError("TabEBM source distribution path is missing or unsafe")
    if path.stat().st_size != SDIST_BYTES or sha256_file(path) != TABEBM_SDIST_SHA256:
        raise ValueError("TabEBM source distribution differs from the PyPI lock")
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe TabEBM archive member: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported TabEBM archive member: {member.name!r}")
            if member.isfile():
                relative = member.name.split("/", 1)[1]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read TabEBM archive member: {member.name!r}")
                files[relative] = extracted.read()
    if len(files) != EXPECTED_ARCHIVE_FILES:
        raise ValueError("TabEBM source archive file count differs from the lock")
    for name, expected in EXPECTED_SOURCE_FILES.items():
        if name not in files or _sha256(files[name]) != expected:
            raise ValueError(f"TabEBM locked source file differs: {name}")
    metadata = Parser().parsestr(files["PKG-INFO"].decode("utf-8"))
    if (
        metadata.get("Name") != "tabebm"
        or metadata.get("Version") != TABEBM_PACKAGE_VERSION
        or metadata.get("License") != "Apache Software License"
        or metadata.get("Home-page") != "https://github.com/andreimargeloiu/TabEBM"
    ):
        raise ValueError("TabEBM package metadata differs from the lock")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "regular_files_verified": len(files),
        "critical_files_verified": len(EXPECTED_SOURCE_FILES),
        "license": "Apache-2.0",
        "source_commit": TABEBM_UPSTREAM_COMMIT,
    }


def _verify_record() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("tabebm")
    record = Path(distribution.locate_file(f"tabebm-{TABEBM_PACKAGE_VERSION}.dist-info/RECORD"))
    if record.is_symlink() or not record.is_file():
        raise ValueError("Installed TabEBM RECORD is missing or unsafe")
    verified = 0
    rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        if len(row) != 3:
            raise ValueError("Installed TabEBM RECORD contains a malformed row")
        if row[1].startswith("sha256="):
            path = Path(distribution.locate_file(row[0]))
            if not path.is_file():
                raise ValueError(f"Installed TabEBM RECORD file is missing: {row[0]}")
            verified += 1
    return {"record_rows": len(rows), "record_hashed_files_present": verified}


def _exercise_official_core() -> dict[str, Any]:
    module = importlib.import_module("tabebm.TabEBM")
    official = module.TabEBM
    logits = np.asarray([[1.0, -0.5], [0.2, 0.7], [-1.0, 2.0]], dtype=np.float64)
    observed_energy = official.compute_energy(logits)
    expected_energy = -scipy.special.logsumexp(logits, axis=1)
    np.testing.assert_array_equal(observed_energy, expected_energy)
    source = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], dtype=np.float64)
    state = np.random.get_state()
    try:
        np.random.seed(19)
        first = official.add_surrogate_negative_samples(source, distance_negative_class=5.0)
        np.random.seed(19)
        second = official.add_surrogate_negative_samples(source, distance_negative_class=5.0)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    if first[0].shape[0] <= source.shape[0] or not set(np.unique(first[1])).issuperset({0, 1}):
        raise AssertionError("Official TabEBM surrogate-negative helper returned an invalid result")
    split = official.train_test_split_allow_full_train(source, np.asarray([0, 1]), test_size=0, shuffle=False)
    np.testing.assert_array_equal(split[0], source)
    return {
        "energy_exact": True,
        "surrogate_negatives_seeded": True,
        "full_train_split_exercised": True,
    }


class _FakeOfficialTabEBM:
    calls: list[dict[str, Any]] = []

    def __init__(self, max_data_size: int = 10_000) -> None:
        self.max_data_size = max_data_size

    def generate(self, X: np.ndarray, y: np.ndarray, num_samples: int, **kwargs: Any) -> dict[str, np.ndarray]:
        type(self).calls.append(
            {
                "X": X.copy(),
                "y": y.copy(),
                "num_samples": num_samples,
                "max_data_size": self.max_data_size,
                "kwargs": dict(kwargs),
            }
        )
        result: dict[str, np.ndarray] = {}
        for class_id in sorted(np.unique(y)):
            class_rows = X[y == class_id]
            indices = np.arange(num_samples) % len(class_rows)
            result[f"class_{int(class_id)}"] = class_rows[indices] + float(class_id) * 0.01
        return result


def _frame(variant: str) -> tuple[pd.DataFrame, DatasetSpec]:
    index = np.arange(36)
    classes = np.array(["no", "yes"] if variant == "binary" else ["a", "b", "c"])
    frame = pd.DataFrame(
        {
            "value": index * 0.25 + index % 4,
            "group": np.array(["g0", "g1", "g2"])[index % 3],
            "target": classes[index % len(classes)],
        }
    )
    spec = DatasetSpec(
        name=f"tabebm-{variant}",
        task_type="classification",
        column_names=list(frame.columns),
        numerical_columns=["value"],
        categorical_columns=["group"],
        target_columns=["target"],
        metadata_path=Path("unused.json"),
    )
    return frame, spec


def _exercise_adapter_boundary(repo_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for variant in ("binary", "multiclass"):
        frame, dataset_spec = _frame(variant)
        case_root = output_dir / variant
        case_root.mkdir(parents=True, exist_ok=True)
        dataset_spec.metadata_path = case_root / "dataset.json"
        dataset_spec.train_data_path = case_root / "train.csv"
        atomic_write_json(dataset_spec.metadata_path, dataset_spec.to_dict())
        frame.to_csv(dataset_spec.train_data_path, index=False)
        adapter = TabEBMAdapter(repo_root)
        train_spec = RunSpec(
            model="tabebm",
            dataset=dataset_spec.name,
            output_dir=case_root / "adapter",
            seed=19,
            extra={"dataset_spec": dataset_spec.to_dict(), "sgld_steps": 3},
        )
        adapter.train(train_spec)
        state = read_json(train_spec.output_dir / "model.tabebm.json")
        if "train_data" in json.dumps(state).lower() and "train_data_sha256" not in state:
            raise AssertionError("TabEBM safe checkpoint unexpectedly retained training rows")
        _FakeOfficialTabEBM.calls.clear()
        adapter._import_official_class = lambda: _FakeOfficialTabEBM  # type: ignore[method-assign]
        sample_spec = RunSpec(
            model="tabebm",
            dataset=dataset_spec.name,
            output_dir=train_spec.output_dir,
            seed=19,
            num_samples=13,
            device="cpu",
            extra={
                "dataset_spec": dataset_spec.to_dict(),
                "allow_gated_model": True,
                "sgld_steps": 3,
            },
        )
        bundle = adapter.sample(sample_spec)
        observed = pd.read_csv(bundle.generated_sample_path)
        if len(observed) != 13 or list(observed.columns) != dataset_spec.column_names:
            raise AssertionError("TabEBM exact-row adapter boundary failed")
        call = _FakeOfficialTabEBM.calls[-1]
        if call["num_samples"] != int(np.ceil(13 / len(np.unique(call["y"])))):
            raise AssertionError("TabEBM per-class official call count is incorrect")
        if call["kwargs"]["seed"] != 19 or call["kwargs"]["sgld_steps"] != 3:
            raise AssertionError("TabEBM official generate arguments differ from the declared recipe")
        results.append(
            {
                "variant": variant,
                "rows": len(observed),
                "official_generate_called": True,
                "checkpoint_sha256": sha256_file(train_spec.output_dir / "model.tabebm.json"),
                "csv_sha256": sha256_file(bundle.generated_sample_path),
            }
        )
    return results


def run_protocol(repo_root: Path, output_dir: Path, evidence_path: Path, sdist_path: Path) -> dict[str, Any]:
    sdist = _verify_sdist(sdist_path)
    package_before = verify_tabebm_distribution()
    record = _verify_record()
    core = _exercise_official_core()
    adapter_cases = _exercise_adapter_boundary(repo_root, output_dir)
    package_after = verify_tabebm_distribution()
    if package_after != package_before:
        raise AssertionError("Installed TabEBM package changed during validation")
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabebm",
        "status": "pass",
        "validation_level": "smoke-validated",
        "repository_commit": _repository_commit(repo_root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "tabebm", "tabpfn", "torch")
        },
        "source_distribution": sdist,
        "installed_package": {**package_before, **record},
        "official_core": core,
        "adapter_cases": adapter_cases,
        "result_summary": {
            "package_unchanged": True,
            "safe_json_checkpoint": True,
            "official_pure_core_exercised": True,
            "official_generate_delegation_exercised_with_test_double": True,
            "full_tabpfn_generation_executed": False,
            "adapter_cases_passed": len(adapter_cases),
        },
        "claim_limit": (
            "This is smoke validation, not native parity. The installed official package, pure deterministic core, "
            "safe preprocessing state, and generate delegation boundary are validated. Full generation is not run "
            "because TabPFN-v2 model access requires external accepted terms and credentials."
        ),
    }
    atomic_write_json(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--sdist-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_protocol(args.repo_root.resolve(), args.output_dir, args.evidence_path, args.sdist_path.resolve())
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            args.evidence_path,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "model_id": "tabebm",
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
