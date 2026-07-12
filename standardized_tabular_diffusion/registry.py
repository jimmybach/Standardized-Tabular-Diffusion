from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.datasets import discover_dataset_specs
from standardized_tabular_diffusion.models import (
    ARFAdapter,
    BNAdapter,
    CTABGANPlusAdapter,
    CTGANAdapter,
    GoggleAdapter,
    GReaTAdapter,
    NRGBoostAdapter,
    NFlowAdapter,
    REaLTabFormerAdapter,
    SMOTEAdapter,
    TabEBMAdapter,
    TVAEAdapter,
    TabDDPMAdapter,
    TabDiffAdapter,
    TabSynAdapter,
)


def _registry(repo_root: Path):
    return {
        "arf": ARFAdapter(repo_root),
        "bn": BNAdapter(repo_root),
        "ctab-gan-plus": CTABGANPlusAdapter(repo_root),
        "ctgan": CTGANAdapter(repo_root),
        "goggle": GoggleAdapter(repo_root),
        "great": GReaTAdapter(repo_root),
        "nrgboost": NRGBoostAdapter(repo_root),
        "nflow": NFlowAdapter(repo_root),
        "realtabformer": REaLTabFormerAdapter(repo_root),
        "smote": SMOTEAdapter(repo_root),
        "tabebm": TabEBMAdapter(repo_root),
        "tabdiff": TabDiffAdapter(repo_root),
        "tabsyn": TabSynAdapter(repo_root),
        "tabddpm": TabDDPMAdapter(repo_root),
        "tvae": TVAEAdapter(repo_root),
    }


def get_adapter(model_name: str, repo_root: Path | None = None):
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    return _registry(repo_root)[model_name]


def list_models() -> list[str]:
    return [
        "arf",
        "bn",
        "ctab-gan-plus",
        "ctgan",
        "goggle",
        "great",
        "nflow",
        "nrgboost",
        "realtabformer",
        "smote",
        "tabebm",
        "tabdiff",
        "tabddpm",
        "tabsyn",
        "tvae",
    ]


def list_datasets(repo_root: Path | None = None) -> list[str]:
    return list(discover_dataset_specs(repo_root=repo_root).keys())
