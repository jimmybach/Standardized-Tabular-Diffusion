"""Evaluation APIs with optional backends loaded on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AtomicResult",
    "EvaluationRequest",
    "IncompleteRunBundleWriter",
    "METRIC_DEFINITIONS",
    "MetricState",
    "evaluate_table_to_bundle",
    "evaluate_validity",
    "validate_validity_profile",
    "validate_tables",
    "validate_result_bundle",
]

_LAZY_EXPORTS = {
    "AtomicResult": ("standardized_tabular_diffusion.evaluation.contracts", "AtomicResult"),
    "EvaluationRequest": ("standardized_tabular_diffusion.evaluation.contracts", "EvaluationRequest"),
    "IncompleteRunBundleWriter": (
        "standardized_tabular_diffusion.evaluation.bundle",
        "IncompleteRunBundleWriter",
    ),
    "MetricState": ("standardized_tabular_diffusion.evaluation.contracts", "MetricState"),
    "evaluate_table_to_bundle": (
        "standardized_tabular_diffusion.evaluation.evaluate_table",
        "evaluate_table_to_bundle",
    ),
    "validate_tables": ("standardized_tabular_diffusion.evaluation.table", "validate_tables"),
    "validate_result_bundle": ("standardized_tabular_diffusion.evaluation.bundle", "validate_result_bundle"),
    "evaluate_validity": ("standardized_tabular_diffusion.evaluation.validity", "evaluate_validity"),
    "validate_validity_profile": (
        "standardized_tabular_diffusion.evaluation.validity",
        "validate_validity_profile",
    ),
}


def __getattr__(name: str) -> Any:
    if name == "METRIC_DEFINITIONS":
        value = import_module("standardized_tabular_diffusion.evaluation.tabstruct").METRIC_DEFINITIONS
        globals()[name] = value
        return value
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
