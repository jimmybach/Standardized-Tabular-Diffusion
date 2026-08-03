"""Public API for the standardized tabular diffusion benchmark.

The public symbols remain import-compatible with the original package, while
optional data, evaluation, and model dependencies are loaded only on demand.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from standardized_tabular_diffusion.config import ExperimentConfig
    from standardized_tabular_diffusion.dataset_onboarding import process_registered_dataset, register_dataset
    from standardized_tabular_diffusion.dataset_sources import fetch_dataset_source, get_dataset_source
    from standardized_tabular_diffusion.datasets import get_dataset_spec
    from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
    from standardized_tabular_diffusion.official_datasets import materialize_official_adult, materialize_official_sick
    from standardized_tabular_diffusion.preprocessing import MissingValuePolicy, preprocess_splits
    from standardized_tabular_diffusion.registry import get_adapter, list_datasets, list_models
    from standardized_tabular_diffusion.runner import build_run_context, run_action, run_pipeline


try:
    __version__ = version("standardized-tabular-diffusion")
except PackageNotFoundError:
    __version__ = "0+unknown"


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactBundle": ("standardized_tabular_diffusion.interfaces", "ArtifactBundle"),
    "DatasetSpec": ("standardized_tabular_diffusion.interfaces", "DatasetSpec"),
    "ExperimentConfig": ("standardized_tabular_diffusion.config", "ExperimentConfig"),
    "RunSpec": ("standardized_tabular_diffusion.interfaces", "RunSpec"),
    "build_run_context": ("standardized_tabular_diffusion.runner", "build_run_context"),
    "get_adapter": ("standardized_tabular_diffusion.registry", "get_adapter"),
    "get_dataset_spec": ("standardized_tabular_diffusion.datasets", "get_dataset_spec"),
    "get_dataset_source": ("standardized_tabular_diffusion.dataset_sources", "get_dataset_source"),
    "fetch_dataset_source": ("standardized_tabular_diffusion.dataset_sources", "fetch_dataset_source"),
    "list_datasets": ("standardized_tabular_diffusion.registry", "list_datasets"),
    "list_models": ("standardized_tabular_diffusion.registry", "list_models"),
    "materialize_official_adult": (
        "standardized_tabular_diffusion.official_datasets",
        "materialize_official_adult",
    ),
    "materialize_official_sick": (
        "standardized_tabular_diffusion.official_datasets",
        "materialize_official_sick",
    ),
    "process_registered_dataset": (
        "standardized_tabular_diffusion.dataset_onboarding",
        "process_registered_dataset",
    ),
    "register_dataset": ("standardized_tabular_diffusion.dataset_onboarding", "register_dataset"),
    "MissingValuePolicy": ("standardized_tabular_diffusion.preprocessing", "MissingValuePolicy"),
    "preprocess_splits": ("standardized_tabular_diffusion.preprocessing", "preprocess_splits"),
    "run_action": ("standardized_tabular_diffusion.runner", "run_action"),
    "run_pipeline": ("standardized_tabular_diffusion.runner", "run_pipeline"),
}

__all__ = [*list(_LAZY_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
