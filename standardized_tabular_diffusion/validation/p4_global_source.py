"""Execute the locked TabEval Global Utility source and the P4 adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
import types
from contextlib import contextmanager
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, sha256_file

PROTOCOL_ID = "p4-global-source-runtime-pilot-v1"
SOURCE_MODULE_NAME = "_standardized_tabular_diffusion_locked_tabeval_p4"
REPO_ROOT = Path(__file__).resolve().parents[2]
NUMERICAL_ABSOLUTE_TOLERANCE = 1e-8


class P4GlobalSourceValidationError(RuntimeError):
    """Raised when source identity or executable parity cannot be established."""


class _SourceDataLoader:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.copy(deep=True)


class _SourceMetricEvaluator:
    def __init__(
        self,
        *,
        reduction: str = "mean",
        workspace: Path = Path("logs/tabeval_workspace"),
        use_cache: bool = False,
        default_metric: str | None = None,
        **_: Any,
    ) -> None:
        self._reduction = reduction
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._use_cache = use_cache
        self._default_metric = default_metric or reduction

    def use_cache(self, path: Path) -> bool:
        return path.exists() and self._use_cache


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _repository_commit() -> str:
    if github_sha := os.environ.get("GITHUB_SHA"):
        return github_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _manifest() -> dict[str, Any]:
    resource = resources.files("standardized_tabular_diffusion").joinpath(
        "resources/evaluation/upstream/tabeval-p4-source.json"
    )
    with resource.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise P4GlobalSourceValidationError("The packaged TabEval P4 source manifest is invalid")
    return payload


def _lf_normalized_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_tabeval_source(source_path: Path, license_path: Path | None = None) -> dict[str, Any]:
    """Fail closed unless source and, when supplied, license match the approved revision."""

    manifest = _manifest()
    if _lf_normalized_sha256(source_path) != manifest["source_sha256"]:
        raise P4GlobalSourceValidationError(
            "TabEval eval_structure.py does not match the LF-normalized approved source hash"
        )
    text = source_path.read_text(encoding="utf-8")
    required_fragments = (
        "class UtilityPerFeature(StructureEvaluator):",
        'custom_hyperparameters["XGB"] = {}',
        'custom_hyperparameters["KNN"] = {}',
        "custom_hyperparameters[CustomTabPFNModel] = {}",
        "class CustomTabPFNModel(AbstractModel):",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise P4GlobalSourceValidationError("Approved TabEval implementation symbols or predictor panel are absent")
    license_digest = None
    if license_path is not None:
        license_digest = _lf_normalized_sha256(license_path)
        if license_digest != manifest["license_sha256"]:
            raise P4GlobalSourceValidationError("TabEval license does not match the approved Apache-2.0 file")
    return {
        "source_id": manifest["source_id"],
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "release_version": manifest["release_version"],
        "source_path": manifest["source_path"],
        "source_sha256": manifest["source_sha256"],
        "source_hash_algorithm": manifest["source_hash_algorithm"],
        "license_spdx": manifest["license_spdx"],
        "license_sha256": license_digest,
        "implementation_symbols": manifest["implementation_symbols"],
        "upstream_dependency_disclosure": manifest["upstream_dependency_disclosure"],
    }


def _package_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


@contextmanager
def _tabeval_import_stubs() -> Iterator[None]:
    """Provide only the upstream infrastructure types needed to import the exact file."""

    modules: dict[str, types.ModuleType] = {
        "tabeval": _package_module("tabeval"),
        "tabeval.metrics": _package_module("tabeval.metrics"),
        "tabeval.metrics.core": types.ModuleType("tabeval.metrics.core"),
        "tabeval.plugins": _package_module("tabeval.plugins"),
        "tabeval.plugins.core": _package_module("tabeval.plugins.core"),
        "tabeval.plugins.core.dataloader": types.ModuleType("tabeval.plugins.core.dataloader"),
        "tabeval.utils": _package_module("tabeval.utils"),
        "tabeval.utils.reproducibility": types.ModuleType("tabeval.utils.reproducibility"),
        "tabeval.utils.serialization": types.ModuleType("tabeval.utils.serialization"),
    }
    modules["tabeval.metrics.core"].MetricEvaluator = _SourceMetricEvaluator  # type: ignore[attr-defined]
    modules["tabeval.plugins.core.dataloader"].DataLoader = _SourceDataLoader  # type: ignore[attr-defined]
    modules["tabeval.utils.reproducibility"].clear_cache = lambda: None  # type: ignore[attr-defined]
    modules["tabeval.utils.serialization"].load_from_file = lambda path: json.loads(  # type: ignore[attr-defined]
        Path(path).read_text(encoding="utf-8")
    )
    modules["tabeval.utils.serialization"].save_to_file = lambda path, payload: Path(path).write_text(  # type: ignore[attr-defined]
        json.dumps(payload), encoding="utf-8"
    )
    previous = {name: sys.modules.get(name) for name in [*modules, SOURCE_MODULE_NAME]}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def load_tabeval_source_module(source_path: Path) -> types.ModuleType:
    """Import the checksum-verified upstream file without installing all of TabEval."""

    verify_tabeval_source(source_path)
    with _tabeval_import_stubs():
        spec = importlib.util.spec_from_file_location(SOURCE_MODULE_NAME, source_path)
        if spec is None or spec.loader is None:
            raise P4GlobalSourceValidationError("Cannot construct an import specification for TabEval source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[SOURCE_MODULE_NAME] = module
        spec.loader.exec_module(module)
    for symbol in ("UtilityPerFeature", "CustomTabPFNModel", "TabularPredictor"):
        if not hasattr(module, symbol):
            raise P4GlobalSourceValidationError(f"Imported TabEval source lacks {symbol}")
    return module


def _verify_checkpoint(path: Path, expected: dict[str, str]) -> dict[str, Any]:
    if path.name != expected["filename"]:
        raise P4GlobalSourceValidationError(
            f"Unexpected checkpoint filename {path.name!r}; expected {expected['filename']!r}"
        )
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        raise P4GlobalSourceValidationError(f"Checkpoint {path.name} does not match the approved hash")
    return {
        "repository": expected["repository"],
        "revision": expected["revision"],
        "filename": expected["filename"],
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def verify_pilot_runtime() -> dict[str, Any]:
    """Attest the benchmark-selected runtime and distinguish it from an upstream lock."""

    manifest = _manifest()
    expected = manifest["approved_pilot_runtime"]
    actual = {
        name: _distribution_version(name)
        for name in (
            "autogluon.common",
            "autogluon.core",
            "autogluon.features",
            "autogluon.tabular",
            "tabpfn",
        )
    }
    actual["xgboost_distribution"] = "xgboost-cpu"
    actual["xgboost"] = _distribution_version("xgboost-cpu")
    try:
        import torch
        import xgboost
    except (ImportError, OSError) as exc:
        raise P4GlobalSourceValidationError(f"The approved P4 source runtime is unavailable: {exc}") from exc
    actual["torch"] = torch.__version__
    actual["xgboost_import"] = xgboost.__version__
    for name in ("autogluon.common", "autogluon.core", "autogluon.features", "autogluon.tabular", "tabpfn"):
        if actual[name] != expected[name]:
            raise P4GlobalSourceValidationError(
                f"Runtime drift for {name}: observed {actual[name]!r}, expected {expected[name]!r}"
            )
    if actual["xgboost"] != expected["xgboost"] or actual["xgboost_import"] != expected["xgboost"]:
        raise P4GlobalSourceValidationError("The approved CPU XGBoost distribution/runtime is not installed")
    if not str(actual["torch"]).startswith("2.3.0+"):
        raise P4GlobalSourceValidationError(f"Runtime drift for torch: observed {actual['torch']!r}")
    if torch.cuda.is_available():
        raise P4GlobalSourceValidationError("The approved pilot is CPU-only but CUDA is available")
    return {
        "status": expected["status"],
        "selection_basis": expected["selection_basis"],
        "upstream_official_environment_claimed": False,
        "versions": actual,
        "torch_cuda_available": False,
    }


@contextmanager
def _controlled_runtime(seed: int) -> Iterator[None]:
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    columns = ["binary_target", "numeric_target"]
    train_rows = 64
    test_rows = 32
    train = pd.DataFrame(
        {
            "binary_target": [index % 2 for index in range(train_rows)],
            "numeric_target": [float((index % 29) + (index % 2) * 0.25) for index in range(train_rows)],
        }
    )
    test = pd.DataFrame(
        {
            "binary_target": [index % 2 for index in range(test_rows)],
            "numeric_target": [float(((index * 3) % 29) + (index % 2) * 0.25) for index in range(test_rows)],
        }
    )
    return train, test, columns


class _TracingPredictor:
    """Transparent source probe that retains actual AutoGluon calls and leaderboards."""

    implementation: Any
    traces: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._trace: dict[str, Any] = {
            "label": kwargs.get("label"),
            "constructor_problem_type": kwargs.get("problem_type"),
            "fit": None,
            "leaderboard": None,
        }
        self._inner = self.implementation(*args, **kwargs)
        self.traces.append(self._trace)

    @property
    def problem_type(self) -> str:
        return self._inner.problem_type

    def fit(self, *args: Any, **kwargs: Any) -> _TracingPredictor:
        hyperparameters = kwargs.get("hyperparameters", {})
        self._trace["fit"] = {
            "hyperparameters": sorted(
                key if isinstance(key, str) else key.__name__ for key in hyperparameters
            ),
            "fit_weighted_ensemble": kwargs.get("fit_weighted_ensemble"),
            "presets": kwargs.get("presets"),
            "time_limit": kwargs.get("time_limit"),
        }
        self._inner.fit(*args, **kwargs)
        return self

    def leaderboard(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        leaderboard = self._inner.leaderboard(*args, **kwargs)
        extra_metrics = list(kwargs.get("extra_metrics", []))
        self._trace["leaderboard"] = {
            "problem_type": self.problem_type,
            "extra_metrics": extra_metrics,
            "models": sorted(str(value) for value in leaderboard["model"].tolist()),
            "records": leaderboard.to_dict(orient="records"),
        }
        return leaderboard


def _instrument_source(module: types.ModuleType, traces: list[dict[str, Any]]) -> None:
    probe = type(
        "TracingTabularPredictor",
        (_TracingPredictor,),
        {"implementation": module.TabularPredictor, "traces": traces},
    )
    module.TabularPredictor = probe


def _source_high_cardinality_guard(module: types.ModuleType, workspace: Path) -> dict[str, Any]:
    model = module.CustomTabPFNModel(
        path=str(workspace),
        name="source-high-cardinality-guard",
        problem_type="multiclass",
        eval_metric="balanced_accuracy",
    )
    features = pd.DataFrame({"feature": np.arange(22, dtype=np.float32)})
    target = pd.Series([index % 11 for index in range(22)])
    try:
        model._fit(features, target)
    except ValueError as exc:
        if "up to 10 classes" not in str(exc):
            raise
        return {"executed": True, "class_count": 11, "rejected_before_model_fit": True, "error": str(exc)}
    raise P4GlobalSourceValidationError("The exact source TabPFN wrapper accepted an eleven-class target")


def _source_score(result: dict[str, Any], target: str, task_type: str) -> float:
    if task_type == "classification":
        value = result["balanced_accuracy"][target][0]
    else:
        value = abs(result["negative_RMSE"][target][0])
    score = float(value)
    if not math.isfinite(score):
        raise P4GlobalSourceValidationError(f"Non-finite source score for {target}")
    return score


def _serialize_backend_result(result: Any) -> dict[str, Any]:
    return {
        "score": result.score,
        "predictors": list(result.predictors),
        "predictor_scores": result.predictor_scores,
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _pip_freeze() -> list[str]:
    try:
        output = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise P4GlobalSourceValidationError(f"Cannot record the installed runtime: {exc}") from exc
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _assert_primary_environment() -> None:
    if platform.system() != "Linux" or platform.python_version_tuple()[:2] != ("3", "11"):
        raise P4GlobalSourceValidationError("Authoritative P4 source evidence requires Linux and Python 3.11")


def _locked_files() -> dict[str, str]:
    paths = (
        ".github/workflows/p4-global-source-validation.yml",
        "THIRD_PARTY_NOTICES.md",
        "requirements-p4-global-source-validation.txt",
        "standardized_tabular_diffusion/evaluation/tabstruct.py",
        "standardized_tabular_diffusion/evaluation/utility.py",
        "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-utility-pilot-v1.json",
        "standardized_tabular_diffusion/resources/evaluation/upstream/tabeval-p4-source.json",
        "standardized_tabular_diffusion/validation/p4_global_source.py",
        "tests/evaluation/test_p4_global_source_validation.py",
    )
    return {path: sha256_file(REPO_ROOT / path) for path in paths}


def run_validation(
    output: Path,
    *,
    source_path: Path,
    license_path: Path,
    classifier_checkpoint: Path,
    regressor_checkpoint: Path,
    time_limit_seconds: int,
    require_primary_environment: bool = False,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "phase": "P4 Global source-runtime pilot",
        "status": "fail",
        "repository_commit": _repository_commit(),
        "claim_boundary": (
            "Executes the exact locked TabEval UtilityPerFeature and CustomTabPFNModel source with real "
            "AutoGluon, CPU XGBoost, KNN, and checksum-locked TabPFN-v2 checkpoints, then compares the P4 "
            "adapter on classification and regression fixtures. This reconstructs a benchmark-approved pilot "
            "runtime because upstream published no dependency lock; it does not freeze Official Results or "
            "complete Adult/Sick multi-seed admission."
        ),
        "environment": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
            "primary_environment_required": require_primary_environment,
        },
    }
    started = time.perf_counter()
    try:
        if time_limit_seconds < 1:
            raise P4GlobalSourceValidationError("time_limit_seconds must be positive")
        if require_primary_environment:
            _assert_primary_environment()
        source = verify_tabeval_source(source_path, license_path)
        manifest = _manifest()
        checkpoints = {
            "classifier": _verify_checkpoint(
                classifier_checkpoint, manifest["tabpfn_checkpoints"]["classifier"]
            ),
            "regressor": _verify_checkpoint(
                regressor_checkpoint, manifest["tabpfn_checkpoints"]["regressor"]
            ),
        }
        runtime = verify_pilot_runtime()
        module = load_tabeval_source_module(source_path)
        traces: list[dict[str, Any]] = []
        _instrument_source(module, traces)
        train, test, columns = _fixture()
        seed = 42
        with tempfile.TemporaryDirectory(prefix="p4-global-source-") as temporary:
            workspace = Path(temporary)
            evaluator = module.UtilityPerFeature(workspace=workspace / "source", use_cache=False)
            source_started = time.perf_counter()
            with _controlled_runtime(seed):
                source_result = evaluator._evaluate(
                    _SourceDataLoader(test),
                    _SourceDataLoader(train),
                    columns,
                    time_limit_seconds,
                )
            source_seconds = time.perf_counter() - source_started
            guard = _source_high_cardinality_guard(module, workspace / "guard")

            from standardized_tabular_diffusion.evaluation.utility import _default_global_scorer

            adapter_started = time.perf_counter()
            with _controlled_runtime(seed):
                classification = _default_global_scorer(
                    train,
                    test,
                    "binary_target",
                    "classification",
                    seed,
                    time_limit_seconds,
                    "source-pilot",
                )
                regression = _default_global_scorer(
                    train,
                    test,
                    "numeric_target",
                    "regression",
                    seed,
                    time_limit_seconds,
                    "source-pilot",
                )
            adapter_seconds = time.perf_counter() - adapter_started

        source_scores = {
            "binary_target": _source_score(source_result, "binary_target", "classification"),
            "numeric_target": _source_score(source_result, "numeric_target", "regression"),
        }
        adapter_results = {
            "binary_target": _serialize_backend_result(classification),
            "numeric_target": _serialize_backend_result(regression),
        }
        numerical_differences = {
            target: abs(source_scores[target] - adapter_results[target]["score"])
            for target in source_scores
        }
        if any(value > NUMERICAL_ABSOLUTE_TOLERANCE for value in numerical_differences.values()):
            raise P4GlobalSourceValidationError(
                f"Source/adapter aggregate scores differ beyond {NUMERICAL_ABSOLUTE_TOLERANCE}: "
                f"{numerical_differences}"
            )
        expected_panel = {"CustomTabPFNModel", "KNN", "XGB"}
        for trace in traces:
            if set(trace["fit"]["hyperparameters"]) != expected_panel:
                raise P4GlobalSourceValidationError("Exact source did not receive the XGB/KNN/TabPFN panel")
            models = trace["leaderboard"]["models"]
            if not all(
                any(token in model.lower() for model in models)
                for token in ("xgboost", "neighbor", "tabpfn")
            ):
                raise P4GlobalSourceValidationError(f"Exact source omitted a required trained family: {models}")
        evidence.update(
            {
                "source": source,
                "runtime": runtime,
                "checkpoints": checkpoints,
                "execution": {
                    "exact_source_evaluate_completed": True,
                    "source_results": _json_compatible(source_result),
                    "source_scores_normalized_for_comparison": source_scores,
                    "source_predictor_traces": _json_compatible(traces),
                    "adapter_results": _json_compatible(adapter_results),
                    "absolute_differences": numerical_differences,
                    "absolute_tolerance": NUMERICAL_ABSOLUTE_TOLERANCE,
                    "high_cardinality_source_guard": guard,
                    "seed": seed,
                    "rows": {"train": len(train), "test": len(test)},
                    "time_limit_per_target_seconds": time_limit_seconds,
                    "source_elapsed_seconds": source_seconds,
                    "adapter_elapsed_seconds": adapter_seconds,
                },
                "exit_gates": {
                    "exact_lf_normalized_source_attested": "pass",
                    "apache_license_attested": "pass",
                    "benchmark_selected_runtime_attested": "pass",
                    "classifier_and_regressor_checkpoints_attested": "pass",
                    "exact_source_classification_and_regression_executed": "pass",
                    "real_xgb_knn_tabpfn_families_trained": "pass",
                    "source_adapter_aggregate_numerical_parity": "pass",
                    "source_high_cardinality_guard_executed": "pass",
                    "official_results_admission": "not-assessed",
                },
                "installed_distributions": _pip_freeze(),
                "locked_files": _locked_files(),
                "elapsed_seconds": time.perf_counter() - started,
                "status": "pass",
            }
        )
    except Exception as exc:  # retain diagnostic evidence for every scientific failure
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
        evidence["elapsed_seconds"] = time.perf_counter() - started
    atomic_write_json(output, evidence)
    if evidence["status"] != "pass":
        raise SystemExit(1)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--license-path", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--regressor-checkpoint", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=int, default=120)
    parser.add_argument("--require-primary-environment", action="store_true")
    args = parser.parse_args()
    run_validation(
        args.output,
        source_path=args.source_path.resolve(),
        license_path=args.license_path.resolve(),
        classifier_checkpoint=args.classifier_checkpoint.resolve(),
        regressor_checkpoint=args.regressor_checkpoint.resolve(),
        time_limit_seconds=args.time_limit_seconds,
        require_primary_environment=args.require_primary_environment,
    )


if __name__ == "__main__":
    main()
