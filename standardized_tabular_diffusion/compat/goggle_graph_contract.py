"""Dependency-free identity for Goggle's validated graph compatibility backend."""

from __future__ import annotations

BACKEND_ID = "standardized-goggle-torch-graph"
BACKEND_VERSION = "1.0.0"
DGL_SEMANTIC_TARGET = "1.1.3"


def graph_backend_record() -> dict[str, str]:
    """Return a fresh JSON-safe backend identity record."""

    return {
        "id": BACKEND_ID,
        "version": BACKEND_VERSION,
        "semantic_target": f"DGL GraphConv {DGL_SEMANTIC_TARGET}",
        "scope": "Goggle homogeneous GCN path only",
    }
