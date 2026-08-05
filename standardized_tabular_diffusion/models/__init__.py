"""Model adapters exposed through lazy imports.

Importing this package is metadata-only. Optional model runtimes are loaded only
when their adapter class is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from standardized_tabular_diffusion.models.base import BaseModelAdapter
    from standardized_tabular_diffusion.models.ctabgan import CTABGANAdapter
    from standardized_tabular_diffusion.models.final_wave_baselines import ARFAdapter
    from standardized_tabular_diffusion.models.goggle import GoggleAdapter
    from standardized_tabular_diffusion.models.great import GReaTAdapter
    from standardized_tabular_diffusion.models.next_wave_baselines import (
        CTABGANPlusAdapter,
        NRGBoostAdapter,
    )
    from standardized_tabular_diffusion.models.paper_gap_baselines import TabSDSAdapter
    from standardized_tabular_diffusion.models.realtabformer import REaLTabFormerAdapter
    from standardized_tabular_diffusion.models.sample_baselines import CTGANAdapter, SMOTEAdapter, TVAEAdapter
    from standardized_tabular_diffusion.models.structured_baselines import BNAdapter, NFlowAdapter
    from standardized_tabular_diffusion.models.tabddpm import TabDDPMAdapter
    from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter
    from standardized_tabular_diffusion.models.tabebm import TabEBMAdapter
    from standardized_tabular_diffusion.models.tabsyn import TabSynAdapter
    from standardized_tabular_diffusion.models.tabula import TabulaAdapter
    from standardized_tabular_diffusion.models.tabularargn import TabularARGNAdapter
    from standardized_tabular_diffusion.models.vendored_baselines import CoDiAdapter, STaSyAdapter


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BaseModelAdapter": ("standardized_tabular_diffusion.models.base", "BaseModelAdapter"),
    "ARFAdapter": ("standardized_tabular_diffusion.models.final_wave_baselines", "ARFAdapter"),
    "GReaTAdapter": ("standardized_tabular_diffusion.models.great", "GReaTAdapter"),
    "TabEBMAdapter": ("standardized_tabular_diffusion.models.tabebm", "TabEBMAdapter"),
    "CTABGANPlusAdapter": ("standardized_tabular_diffusion.models.next_wave_baselines", "CTABGANPlusAdapter"),
    "NRGBoostAdapter": ("standardized_tabular_diffusion.models.next_wave_baselines", "NRGBoostAdapter"),
    "REaLTabFormerAdapter": ("standardized_tabular_diffusion.models.realtabformer", "REaLTabFormerAdapter"),
    "TabSDSAdapter": ("standardized_tabular_diffusion.models.paper_gap_baselines", "TabSDSAdapter"),
    "TabularARGNAdapter": ("standardized_tabular_diffusion.models.tabularargn", "TabularARGNAdapter"),
    "CTGANAdapter": ("standardized_tabular_diffusion.models.sample_baselines", "CTGANAdapter"),
    "SMOTEAdapter": ("standardized_tabular_diffusion.models.sample_baselines", "SMOTEAdapter"),
    "TVAEAdapter": ("standardized_tabular_diffusion.models.sample_baselines", "TVAEAdapter"),
    "BNAdapter": ("standardized_tabular_diffusion.models.structured_baselines", "BNAdapter"),
    "GoggleAdapter": ("standardized_tabular_diffusion.models.goggle", "GoggleAdapter"),
    "NFlowAdapter": ("standardized_tabular_diffusion.models.structured_baselines", "NFlowAdapter"),
    "TabDDPMAdapter": ("standardized_tabular_diffusion.models.tabddpm", "TabDDPMAdapter"),
    "TabDiffAdapter": ("standardized_tabular_diffusion.models.tabdiff", "TabDiffAdapter"),
    "TabSynAdapter": ("standardized_tabular_diffusion.models.tabsyn", "TabSynAdapter"),
    "TabulaAdapter": ("standardized_tabular_diffusion.models.tabula", "TabulaAdapter"),
    "CTABGANAdapter": ("standardized_tabular_diffusion.models.ctabgan", "CTABGANAdapter"),
    "CoDiAdapter": ("standardized_tabular_diffusion.models.vendored_baselines", "CoDiAdapter"),
    "STaSyAdapter": ("standardized_tabular_diffusion.models.vendored_baselines", "STaSyAdapter"),
}

__all__ = list(_LAZY_EXPORTS)


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
