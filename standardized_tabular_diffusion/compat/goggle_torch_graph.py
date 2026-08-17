"""Minimal, auditable PyTorch graph backend for Goggle's validated GCN path.

This module is intentionally not a general DGL replacement.  It implements
only the three DGL 1.1.3 interfaces executed by the checksum-locked Goggle
source: ``dgl.graph``, ``dgl.batch``, and ``dgl.nn.GraphConv`` with the
arguments used by Goggle.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from standardized_tabular_diffusion.compat.goggle_graph_contract import BACKEND_VERSION


class GoggleGraphCompatibilityError(RuntimeError):
    """Raised when a graph falls outside the deliberately narrow contract."""


@dataclass(frozen=True)
class GoggleGraph:
    """Directed homogeneous graph in DGL edge order."""

    source: torch.Tensor
    destination: torch.Tensor
    node_count: int

    def __post_init__(self) -> None:
        if self.source.ndim != 1 or self.destination.ndim != 1:
            raise GoggleGraphCompatibilityError("Goggle graph endpoints must be one-dimensional tensors.")
        if self.source.shape != self.destination.shape:
            raise GoggleGraphCompatibilityError("Goggle graph endpoint tensors must have identical shape.")
        if self.source.device != self.destination.device:
            raise GoggleGraphCompatibilityError("Goggle graph endpoints must use the same device.")
        if self.source.dtype != self.destination.dtype:
            raise GoggleGraphCompatibilityError("Goggle graph endpoints must use the same index dtype.")
        if self.source.dtype not in (torch.int32, torch.int64) or self.destination.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise GoggleGraphCompatibilityError("Goggle graph endpoints must use int32 or int64 indices.")
        if self.node_count < 0:
            raise GoggleGraphCompatibilityError("Goggle graph node_count must be non-negative.")
        if self.source.numel() and (
            bool((self.source < 0).any())
            or bool((self.destination < 0).any())
            or int(torch.max(torch.stack((self.source.max(), self.destination.max()))).item()) >= self.node_count
        ):
            raise GoggleGraphCompatibilityError("Goggle graph endpoints fall outside node_count.")

    @property
    def device(self) -> torch.device:
        return self.source.device

    def num_edges(self) -> int:
        return int(self.source.numel())

    def in_degrees(self) -> torch.Tensor:
        return torch.bincount(self.destination.to(torch.int64), minlength=self.node_count)

    def out_degrees(self) -> torch.Tensor:
        return torch.bincount(self.source.to(torch.int64), minlength=self.node_count)


def graph(edges: tuple[torch.Tensor, torch.Tensor], num_nodes: int | None = None, **kwargs: Any) -> GoggleGraph:
    """Construct the homogeneous graph shape used by official Goggle."""

    if kwargs:
        raise GoggleGraphCompatibilityError(f"Unsupported dgl.graph arguments for Goggle: {', '.join(sorted(kwargs))}")
    if not isinstance(edges, tuple) or len(edges) != 2:
        raise GoggleGraphCompatibilityError("Goggle requires dgl.graph((source, destination)).")
    source, destination = edges
    if not isinstance(source, torch.Tensor) or not isinstance(destination, torch.Tensor):
        raise GoggleGraphCompatibilityError("Goggle graph endpoints must already be PyTorch tensors.")
    if num_nodes is None:
        if source.numel() == 0:
            inferred = 0
        else:
            inferred = int(torch.max(torch.stack((source.max(), destination.max()))).item()) + 1
        num_nodes = inferred
    if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
        raise GoggleGraphCompatibilityError("Goggle graph num_nodes must be an integer.")
    return GoggleGraph(source=source, destination=destination, node_count=num_nodes)


def batch(graphs: Iterable[GoggleGraph]) -> GoggleGraph:
    """Create DGL-compatible disjoint-union edge order for a graph batch."""

    graph_list = list(graphs)
    if not graph_list:
        raise GoggleGraphCompatibilityError("Goggle cannot batch an empty graph sequence.")
    if not all(isinstance(item, GoggleGraph) for item in graph_list):
        raise GoggleGraphCompatibilityError("Goggle graph batches may contain only GoggleGraph values.")
    device = graph_list[0].device
    dtype = graph_list[0].source.dtype
    if any(item.device != device or item.source.dtype != dtype for item in graph_list):
        raise GoggleGraphCompatibilityError("All Goggle graphs in a batch must share device and index dtype.")
    sources: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    offset = 0
    for item in graph_list:
        sources.append(item.source + offset)
        destinations.append(item.destination + offset)
        offset += item.node_count
    return GoggleGraph(
        source=torch.cat(sources),
        destination=torch.cat(destinations),
        node_count=offset,
    )


class GraphConv(nn.Module):
    """DGL 1.1.3 ``GraphConv`` semantics for Goggle's homogeneous tensors."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        norm: str = "both",
        weight: bool = True,
        bias: bool = True,
        activation: Any = None,
        allow_zero_in_degree: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(in_feats, int) or isinstance(in_feats, bool) or in_feats <= 0:
            raise GoggleGraphCompatibilityError("GraphConv in_feats must be a positive integer.")
        if not isinstance(out_feats, int) or isinstance(out_feats, bool) or out_feats <= 0:
            raise GoggleGraphCompatibilityError("GraphConv out_feats must be a positive integer.")
        if norm not in {"none", "both", "right", "left"}:
            raise GoggleGraphCompatibilityError('GraphConv norm must be one of "none", "both", "right", or "left".')
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        if weight:
            self.weight = nn.Parameter(torch.empty(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_feats))
        else:
            self.register_parameter("bias", None)
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise GoggleGraphCompatibilityError("GraphConv allow_zero_in_degree must be a boolean.")
        self._allow_zero_in_degree = value

    @staticmethod
    def _reshape_node_norm(norm: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return norm.reshape(norm.shape + (1,) * (feature.ndim - 1))

    @staticmethod
    def _aggregate(
        graph_value: GoggleGraph,
        feature: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        messages = feature.index_select(0, graph_value.source.to(torch.int64))
        if edge_weight is not None:
            shape = (edge_weight.shape[0],) + (1,) * (messages.ndim - 1)
            messages = messages * edge_weight.reshape(shape)
        output_shape = (graph_value.node_count, *messages.shape[1:])
        output = torch.zeros(output_shape, dtype=messages.dtype, device=messages.device)
        output.index_add_(0, graph_value.destination.to(torch.int64), messages)
        return output

    def forward(
        self,
        graph_value: GoggleGraph,
        feat: torch.Tensor,
        weight: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(graph_value, GoggleGraph):
            raise GoggleGraphCompatibilityError("GraphConv accepts only the validated GoggleGraph type.")
        if not isinstance(feat, torch.Tensor) or feat.ndim < 2 or feat.shape[0] != graph_value.node_count:
            raise GoggleGraphCompatibilityError("GraphConv feature rows must equal the graph node count.")
        if feat.shape[-1] != self._in_feats:
            raise GoggleGraphCompatibilityError(
                f"GraphConv expected {self._in_feats} input features, found {feat.shape[-1]}."
            )
        if feat.device != graph_value.device:
            raise GoggleGraphCompatibilityError("GraphConv graph and features must use the same device.")
        if edge_weight is not None:
            if edge_weight.ndim != 1 or edge_weight.shape[0] != graph_value.num_edges():
                raise GoggleGraphCompatibilityError("GraphConv edge_weight must contain one scalar per edge.")
            if edge_weight.device != feat.device or edge_weight.dtype != feat.dtype:
                raise GoggleGraphCompatibilityError("GraphConv edge weights must match feature device and dtype.")
        in_degrees = graph_value.in_degrees()
        if not self._allow_zero_in_degree and bool((in_degrees == 0).any()):
            raise GoggleGraphCompatibilityError(
                "GraphConv found a zero-in-degree node; official Goggle normally prevents this with diagonal edges."
            )
        feature_source = feat
        if self._norm in {"left", "both"}:
            degrees = graph_value.out_degrees().to(feat).clamp(min=1)
            source_norm = torch.pow(degrees, -0.5) if self._norm == "both" else 1.0 / degrees
            feature_source = feature_source * self._reshape_node_norm(source_norm, feature_source)
        if weight is not None and self.weight is not None:
            raise GoggleGraphCompatibilityError(
                "GraphConv cannot combine an external weight with its own weight parameter."
            )
        effective_weight = self.weight if weight is None else weight
        if effective_weight is not None and effective_weight.shape != (self._in_feats, self._out_feats):
            raise GoggleGraphCompatibilityError("GraphConv weight shape is inconsistent with the layer contract.")
        if effective_weight is not None and (
            effective_weight.device != feat.device or effective_weight.dtype != feat.dtype
        ):
            raise GoggleGraphCompatibilityError("GraphConv weight must match feature device and dtype.")
        if self._in_feats > self._out_feats:
            if effective_weight is not None:
                feature_source = torch.matmul(feature_source, effective_weight)
            result = self._aggregate(graph_value, feature_source, edge_weight)
        else:
            result = self._aggregate(graph_value, feature_source, edge_weight)
            if effective_weight is not None:
                result = torch.matmul(result, effective_weight)
        if self._norm in {"right", "both"}:
            degrees = in_degrees.to(feat).clamp(min=1)
            destination_norm = torch.pow(degrees, -0.5) if self._norm == "both" else 1.0 / degrees
            result = result * self._reshape_node_norm(destination_norm, result)
        if self.bias is not None:
            result = result + self.bias
        if self._activation is not None:
            result = self._activation(result)
        return result

    def extra_repr(self) -> str:
        return f"in={self._in_feats}, out={self._out_feats}, normalization={self._norm}, activation={self._activation}"


def dgl_compatibility_modules() -> dict[str, types.ModuleType]:
    """Return private module objects consumed by unmodified Goggle imports."""

    dgl_module = types.ModuleType("dgl")
    dgl_module.__path__ = []  # type: ignore[attr-defined]
    dgl_module.__version__ = f"compat-{BACKEND_VERSION}"  # type: ignore[attr-defined]
    dgl_module.graph = graph  # type: ignore[attr-defined]
    dgl_module.batch = batch  # type: ignore[attr-defined]
    nn_module = types.ModuleType("dgl.nn")
    nn_module.GraphConv = GraphConv  # type: ignore[attr-defined]

    class _UnavailableSAGEConv:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise GoggleGraphCompatibilityError(
                "Goggle decoder_arch='sage' is not supported by the validated pure-PyTorch GCN backend."
            )

    nn_module.SAGEConv = _UnavailableSAGEConv  # type: ignore[attr-defined]
    dgl_module.nn = nn_module  # type: ignore[attr-defined]
    return {"dgl": dgl_module, "dgl.nn": nn_module}
