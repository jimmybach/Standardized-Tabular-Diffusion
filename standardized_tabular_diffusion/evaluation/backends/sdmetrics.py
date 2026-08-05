"""Isolated adapter for the exact SDMetrics source selected by P2."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

_SOURCE_EXECUTION_LOCK = threading.Lock()


class SDMetricsBackendError(RuntimeError):
    """Raised when the selected official SDMetrics source cannot be trusted or run."""


class SDMetricsSourceError(SDMetricsBackendError):
    """Raised when the imported source fails the P2 provenance lock."""


class SDMetricsExecutionError(SDMetricsBackendError):
    """Raised when attested upstream code fails during metric execution."""


@dataclass(frozen=True)
class SDMetricsQualityResult:
    """Unmodified upstream property outputs, before benchmark contract mapping."""

    column_shapes_score: float
    column_pair_trends_score: float
    column_shapes_details: pd.DataFrame
    column_pair_trends_details: pd.DataFrame
    source: dict[str, Any]


@contextmanager
def _controlled_numpy_random_state(seed: int) -> Iterator[None]:
    """Control and restore the legacy RNG used by upstream pandas sampling.

    The locked SDMetrics properties call ``DataFrame.sample`` without an
    explicit random state when a table exceeds the 50,000-row subsample limit.
    Serializing these calls prevents two P2 evaluations from interleaving their
    source RNG state inside one process; the caller's NumPy state is restored.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise SDMetricsExecutionError("evaluator_seed must be an integer in [0, 2**32)")
    with _SOURCE_EXECUTION_LOCK:
        previous_state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(previous_state)


def _manifest() -> dict[str, Any]:
    resource = resources.files("standardized_tabular_diffusion").joinpath(
        "resources/evaluation/upstream/sdmetrics-p2-source.json"
    )
    with resource.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise SDMetricsSourceError("The packaged SDMetrics source manifest is invalid")
    return payload


def _source_tree_digest(root: Path) -> tuple[int, str]:
    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return len(files), digest.hexdigest()


def _installed_license_digest() -> str:
    try:
        installed = distribution("sdmetrics")
    except PackageNotFoundError as exc:
        raise SDMetricsSourceError("Cannot locate installed SDMetrics distribution metadata") from exc
    candidates = [
        item
        for item in (installed.files or ())
        if item.as_posix().endswith(".dist-info/licenses/LICENSE")
    ]
    if len(candidates) != 1:
        raise SDMetricsSourceError("Installed SDMetrics distribution must retain exactly one MIT LICENSE")
    try:
        return hashlib.sha256(Path(str(installed.locate_file(candidates[0]))).read_bytes()).hexdigest()
    except OSError as exc:
        raise SDMetricsSourceError(f"Cannot attest the installed SDMetrics license: {exc}") from exc


def verify_sdmetrics_source() -> dict[str, Any]:
    """Fail closed unless the imported package is the exact P2 source tree."""

    try:
        import sdmetrics
    except ImportError as exc:
        raise SDMetricsSourceError(
            "P2 requires the checksum-locked official SDMetrics source; install the evaluation extra"
        ) from exc
    manifest = _manifest()
    version = getattr(sdmetrics, "__version__", None)
    if version != manifest["distribution_version"]:
        raise SDMetricsSourceError(
            f"Unsupported SDMetrics version {version!r}; expected {manifest['distribution_version']!r}"
        )
    package_root = Path(sdmetrics.__file__).resolve().parent
    try:
        count, digest = _source_tree_digest(package_root)
    except OSError as exc:
        raise SDMetricsSourceError(f"Cannot attest the imported SDMetrics source: {exc}") from exc
    if count != manifest["python_source_file_count"] or digest != manifest["python_source_tree_sha256"]:
        raise SDMetricsSourceError(
            "Imported SDMetrics Python sources do not match the approved commit; metric execution refused"
        )
    license_digest = _installed_license_digest()
    if license_digest != manifest["license_sha256"]:
        raise SDMetricsSourceError("Installed SDMetrics license does not match the approved MIT license")
    return {
        "source_id": manifest["source_id"],
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "distribution_version": version,
        "python_source_file_count": count,
        "python_source_tree_sha256": digest,
        "license_spdx": manifest["license_spdx"],
        "license_sha256": license_digest,
        "implementation_symbols": manifest["implementation_symbols"],
        "report_defaults": manifest["report_defaults"],
    }


def evaluate_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    evaluator_seed: int = 0,
) -> SDMetricsQualityResult:
    """Run official Column Shapes and Column Pair Trends without local formulas.

    The properties are instantiated directly so only the selected source
    implementation runs. Pinned Quality Report defaults are applied explicitly;
    this also makes threshold and subsampling semantics visible in provenance.
    """

    source = verify_sdmetrics_source()
    try:
        from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes

        with _controlled_numpy_random_state(evaluator_seed):
            shapes = ColumnShapes()
            trends = ColumnPairTrends()
            defaults = source["report_defaults"]
            shapes.num_rows_subsample = defaults["num_rows_subsample"]
            trends.num_rows_subsample = defaults["num_rows_subsample"]
            trends.real_correlation_threshold = defaults["real_correlation_threshold"]
            trends.real_association_threshold = defaults["real_association_threshold"]
            shapes_score = shapes.get_score(
                real_data.copy(deep=True), synthetic_data.copy(deep=True), metadata
            )
            trends_score = trends.get_score(
                real_data.copy(deep=True), synthetic_data.copy(deep=True), metadata
            )
    except Exception as exc:
        if isinstance(exc, SDMetricsExecutionError):
            raise
        raise SDMetricsExecutionError(
            f"Official SDMetrics execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    source = {
        **source,
        "execution": {
            "evaluator_seed": evaluator_seed,
            "randomness_control": "serialized legacy NumPy state with caller state restoration",
        },
    }
    return SDMetricsQualityResult(
        column_shapes_score=float(shapes_score),
        column_pair_trends_score=float(trends_score),
        column_shapes_details=shapes.details.copy(deep=True),
        column_pair_trends_details=trends.details.copy(deep=True),
        source=source,
    )
