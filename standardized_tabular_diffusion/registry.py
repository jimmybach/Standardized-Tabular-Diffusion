from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.datasets import discover_dataset_specs
from standardized_tabular_diffusion.models import TabDDPMAdapter, TabDiffAdapter, TabSynAdapter


def _registry(repo_root: Path):
    return {
        "tabdiff": TabDiffAdapter(repo_root),
        "tabsyn": TabSynAdapter(repo_root),
        "tabddpm": TabDDPMAdapter(repo_root),
    }


def get_adapter(model_name: str, repo_root: Path | None = None):
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    return _registry(repo_root)[model_name]


def list_models() -> list[str]:
    return ["tabdiff", "tabsyn", "tabddpm"]


def list_datasets(repo_root: Path | None = None) -> list[str]:
    return list(discover_dataset_specs(repo_root=repo_root).keys())
