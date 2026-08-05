"""Exact adapter parity protocol for the locked method-author TabSDS source."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import io
import platform
import subprocess
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.paper_gap_baselines import TabSDSAdapter
from standardized_tabular_diffusion.upstream_sources import source_manifest_path, validate_upstream_source

PROTOCOL_ID = "tabsds-official-source-parity-v1"
UPSTREAM_COMMIT = TabSDSAdapter.upstream_commit
EXPECTED_ROWS = 37
REQUESTED_ROWS = 53
SEEDS = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _load(path: Path, name: str, initial_globals: dict[str, Any] | None = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official TabSDS file: {path}")
    module = importlib.util.module_from_spec(spec)
    if initial_globals:
        module.__dict__.update(initial_globals)
    spec.loader.exec_module(module)
    return module


def _official_wrapper(source_root: Path) -> ModuleType:
    core = _load(source_root / "utility_functions_syn_tab_sjppds_for_icml_2025.py", "_tabsds_direct_core")
    wrapper = _load(
        source_root / "utility_functions_additional_for_icml_2025.py",
        "_tabsds_direct_wrapper",
        {"np": np, "pd": pd},
    )
    wrapper.np = np
    wrapper.pd = pd
    for name in ("sequential_jppds", "cat_sjppds", "categorical_to_numeric", "numeric_to_categorical"):
        setattr(wrapper, name, getattr(core, name))
    return wrapper


def _frame(variant: str) -> tuple[pd.DataFrame, DatasetSpec]:
    index = np.arange(EXPECTED_ROWS)
    if variant == "binary":
        frame = pd.DataFrame(
            {
                "amount": (index * 1.75 + (index % 5) * 0.2).astype(float),
                "region": np.array(["north", "south", "west"])[index % 3],
                "target": np.array(["no", "yes"])[index % 2],
            }
        )
        task = "classification"
        numerical = ["amount"]
        categorical = ["region"]
    elif variant == "multiclass":
        frame = pd.DataFrame(
            {
                "count": (index * 3 + index % 4).astype(float),
                "segment": np.array(["a", "b", "c", "d"])[index % 4],
                "target": np.array(["low", "mid", "high"])[index % 3],
            }
        )
        task = "classification"
        numerical = ["count"]
        categorical = ["segment"]
    elif variant == "regression":
        frame = pd.DataFrame(
            {
                "feature": (index / 7.0 + (index % 3) * 0.11).astype(float),
                "group": np.array(["g0", "g1", "g2"])[index % 3],
                "target": (index * 0.45 - (index % 4) * 0.2).astype(float),
            }
        )
        task = "regression"
        numerical = ["feature"]
        categorical = ["group"]
    else:
        raise ValueError(variant)
    spec = DatasetSpec(
        name=f"tabsds-{variant}",
        task_type=task,
        column_names=list(frame.columns),
        numerical_columns=numerical,
        categorical_columns=categorical,
        target_columns=["target"],
        metadata_path=Path("unused.json"),
    )
    return frame, spec


def _direct(
    frame: pd.DataFrame,
    dataset_spec: DatasetSpec,
    wrapper: ModuleType,
    *,
    seed: int,
    n_levels: int,
) -> pd.DataFrame:
    numerical = list(dataset_spec.numerical_columns)
    categorical = list(dataset_spec.categorical_columns)
    target = dataset_spec.target_columns[0]
    (categorical if dataset_spec.task_type == "classification" else numerical).append(target)
    num_indices = [dataset_spec.column_names.index(column) for column in numerical] or None
    cat_indices = [dataset_spec.column_names.index(column) for column in categorical] or None
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        blocks: list[pd.DataFrame] = []
        remaining = REQUESTED_ROWS
        while remaining:
            generated = wrapper.tab_sjppds(
                dat=frame.copy(),
                num_variables=num_indices,
                cat_variables=cat_indices,
                n_levels=n_levels,
                shuffle_type="simple",
                verbose=False,
            )
            generated.columns = dataset_spec.column_names
            take = min(remaining, len(generated))
            blocks.append(generated.iloc[:take].copy())
            remaining -= take
    finally:
        np.random.set_state(state)
    return pd.concat(blocks, ignore_index=True)[dataset_spec.column_names]


def run_protocol(repo_root: Path, source_root: Path, output_dir: Path, evidence_path: Path) -> dict[str, Any]:
    source_before = validate_upstream_source("tabsds", source_root)
    manifest_digest = sha256_file(source_manifest_path("tabsds"))
    if source_before["upstream_commit"] != UPSTREAM_COMMIT or source_before["manifest_sha256"] != manifest_digest:
        raise ValueError("TabSDS source identity differs from the protocol lock")
    wrapper = _official_wrapper(source_root)
    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            case_root = output_dir / f"{variant}-seed-{seed}"
            case_root.mkdir(parents=True, exist_ok=True)
            frame, dataset_spec = _frame(variant)
            dataset_spec.metadata_path = case_root / "dataset.json"
            dataset_spec.train_data_path = case_root / "train.csv"
            atomic_write_json(dataset_spec.metadata_path, dataset_spec.to_dict())
            frame.to_csv(dataset_spec.train_data_path, index=False)
            direct_input = pd.read_csv(dataset_spec.train_data_path)[dataset_spec.column_names]
            direct = _direct(direct_input, dataset_spec, wrapper, seed=seed, n_levels=7)
            adapter = TabSDSAdapter(repo_root)
            train_spec = RunSpec(
                model="tabsds",
                dataset=dataset_spec.name,
                output_dir=case_root / "adapter",
                seed=seed,
                extra={"dataset_spec": dataset_spec.to_dict(), "source_dir": str(source_root), "n_levels": 7},
            )
            adapter.train(train_spec)
            numpy_before = np.random.get_state()
            sample_spec = RunSpec(
                model="tabsds",
                dataset=dataset_spec.name,
                output_dir=train_spec.output_dir,
                seed=seed,
                num_samples=REQUESTED_ROWS,
                extra={"dataset_spec": dataset_spec.to_dict(), "source_dir": str(source_root)},
            )
            bundle = adapter.sample(sample_spec)
            numpy_after = np.random.get_state()
            if any(
                not np.array_equal(before, after) if isinstance(before, np.ndarray) else before != after
                for before, after in zip(numpy_before, numpy_after)
            ):
                raise AssertionError("TabSDS adapter leaked NumPy RNG state")
            direct_csv = direct.to_csv(index=False).encode("utf-8")
            observed_csv = bundle.generated_sample_path.read_bytes()
            if observed_csv != direct_csv:
                raise AssertionError("TabSDS adapter CSV bytes differ from direct official source")
            observed = pd.read_csv(bundle.generated_sample_path)
            direct_round_trip = pd.read_csv(io.BytesIO(direct_csv))
            pd.testing.assert_frame_equal(observed, direct_round_trip, check_dtype=False, check_exact=True)
            results.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "rows": len(observed),
                    "csv_sha256": sha256_file(bundle.generated_sample_path),
                    "checkpoint_sha256": sha256_file(train_spec.output_dir / "model.tabsds.json"),
                    "exact": True,
                }
            )
    source_after = validate_upstream_source("tabsds", source_root)
    if source_after != source_before:
        raise AssertionError("TabSDS source changed during validation")
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "tabsds",
        "status": "pass",
        "repository_commit": _repository_commit(repo_root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "tqdm")
        },
        "source": source_before,
        "result_summary": {
            "cases_passed": len(results),
            "cases_total": len(VARIANTS) * len(SEEDS),
            "source_unchanged": True,
            "safe_json_checkpoint": True,
            "rng_restored": True,
            "sample_frames_exact": True,
            "sample_csv_bytes_exact": True,
            "requested_rows_per_case": REQUESTED_ROWS,
        },
        "cases": results,
        "claim_limit": (
            "Exact parity covers the locked method-author Python simple-shuffle path and the declared exact-row "
            "repeat/truncate boundary. The upstream repository has no license and remains release-blocked."
        ),
    }
    atomic_write_json(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_protocol(args.repo_root.resolve(), args.source_root.resolve(), args.output_dir, args.evidence_path)
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            args.evidence_path,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "model_id": "tabsds",
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
