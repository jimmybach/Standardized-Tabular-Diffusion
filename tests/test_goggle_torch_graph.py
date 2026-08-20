from __future__ import annotations

import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from standardized_tabular_diffusion.compat.goggle_graph_contract import BACKEND_VERSION
from standardized_tabular_diffusion.compat.goggle_launcher import _official_import_boundary, _run_sample
from standardized_tabular_diffusion.compat.goggle_torch_graph import (
    GoggleGraphCompatibilityError,
    GraphConv,
    batch,
    graph,
)
from standardized_tabular_diffusion.validation import goggle as goggle_validation


def _fixture_graph():
    source = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    destination = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.int64)
    return source, destination, graph((source, destination), num_nodes=4)


def test_goggle_graph_and_batch_preserve_edge_order_and_degrees() -> None:
    source, destination, graph_value = _fixture_graph()
    batched = batch([graph_value, graph_value])

    assert torch.equal(graph_value.in_degrees(), torch.tensor([2, 2, 2, 2]))
    assert torch.equal(graph_value.out_degrees(), torch.tensor([2, 2, 2, 2]))
    assert torch.equal(batched.source, torch.cat((source, source + 4)))
    assert torch.equal(batched.destination, torch.cat((destination, destination + 4)))
    assert batched.node_count == 8


def test_goggle_graphconv_matches_the_manual_weighted_both_normalization() -> None:
    source, destination, graph_value = _fixture_graph()
    features = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    edge_weights = torch.linspace(0.25, 2.0, 8, dtype=torch.float64)
    layer = GraphConv(3, 2, norm="both", bias=True).to(torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.5, -0.25], [1.0, 0.75], [-0.5, 0.125]], dtype=torch.float64))
        layer.bias.copy_(torch.tensor([0.2, -0.3], dtype=torch.float64))

    normalized = features / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
    projected = normalized @ layer.weight
    messages = projected.index_select(0, source) * edge_weights[:, None]
    aggregated = torch.zeros(4, 2, dtype=torch.float64)
    aggregated.index_add_(0, destination, messages)
    expected = aggregated / torch.sqrt(torch.tensor(2.0, dtype=torch.float64)) + layer.bias

    assert torch.allclose(layer(graph_value, features, edge_weight=edge_weights), expected, rtol=1e-15, atol=1e-15)


def test_goggle_graphconv_rejects_inputs_outside_the_narrow_contract() -> None:
    source, destination, graph_value = _fixture_graph()
    with pytest.raises(GoggleGraphCompatibilityError, match="same index dtype"):
        graph((source.to(torch.int32), destination), num_nodes=4)
    with pytest.raises(GoggleGraphCompatibilityError, match="zero-in-degree"):
        GraphConv(2, 2)(
            graph((torch.tensor([0]), torch.tensor([0])), num_nodes=2),
            torch.ones(2, 2),
        )
    with pytest.raises(GoggleGraphCompatibilityError, match="one scalar per edge"):
        GraphConv(2, 2)(graph_value, torch.ones(4, 2), edge_weight=torch.ones(2))


def test_goggle_official_import_boundary_needs_neither_dgl_nor_pyg_at_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / ".cache" / "upstream-sources" / "goggle" / "1a3d87ad8a5dffe0f67f844e7b10f1f0dcef73e0"
    if not source_root.is_dir():
        pytest.skip("checksum-locked Goggle source has not been materialized")

    prior_dgl = sys.modules.get("dgl")
    with _official_import_boundary(source_root, graph_backend="torch") as model_type:
        assert model_type.__name__ == "GoggleModel"
        assert sys.modules["dgl"].__version__ == f"compat-{BACKEND_VERSION}"
        assert sys.modules["torch_geometric.utils"].dense_to_sparse is not None
    assert sys.modules.get("dgl") is prior_dgl


def test_goggle_launcher_reapplies_generation_seed_after_model_construction(tmp_path: Path) -> None:
    class Core:
        @staticmethod
        def load_state_dict(_state_dict):
            return None

        @staticmethod
        def sample(rows: int):
            return torch.randn(rows, 3)

    class Model:
        def __init__(self, **kwargs):
            torch.manual_seed(kwargs["seed"])
            self.model = Core()

    config = {
        "dataset": "fixture",
        "input_dim": 3,
        "encoder_dim": 8,
        "encoder_l": 1,
        "het_encoding": True,
        "decoder_dim": 8,
        "decoder_l": 1,
        "threshold": 0.1,
        "decoder_arch": "gcn",
        "graph_prior": None,
        "prior_mask": None,
        "alpha": 0.1,
        "beta": 0.1,
        "seed": 13,
        "iter_opt": True,
        "learning_rate": 0.005,
        "weight_decay": 0.001,
        "epochs": 1,
        "batch_size": 4,
        "patience": 1,
        "logging": 1,
    }
    checkpoint = tmp_path / "model.pt"
    torch.save({}, checkpoint)

    def sample(seed: int, filename: str):
        output = tmp_path / filename
        args = Namespace(
            checkpoint=checkpoint,
            seed=seed,
            num_threads=1,
            num_samples=4,
            raw_output=output,
            output_dir=tmp_path,
        )
        _run_sample(args, Model, config, "cpu")
        return torch.from_numpy(np.load(output, allow_pickle=False))

    seed_17_first = sample(17, "seed-17-first.npy")
    seed_29 = sample(29, "seed-29.npy")
    seed_17_second = sample(17, "seed-17-second.npy")
    assert not torch.equal(seed_17_first, seed_29)
    assert torch.equal(seed_17_first, seed_17_second)


def test_goggle_dgl_oracle_reapplies_the_same_generation_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    class Core:
        @staticmethod
        def load_state_dict(_state_dict):
            return None

        @staticmethod
        def sample(rows: int):
            return torch.randn(rows, 3)

    class Model:
        def __init__(self, **kwargs):
            torch.manual_seed(kwargs["seed"])
            torch.randn(11)
            self.dataset = kwargs["ds_name"]
            self.model = Core()

        def fit(self, _frame):
            checkpoint_dir = Path("tmp")
            checkpoint_dir.mkdir()
            torch.save({}, checkpoint_dir / f"{self.dataset}.pt")

    @contextmanager
    def fake_import_boundary(_source_root: Path, *, graph_backend: str):
        assert graph_backend == "dgl-reference"
        yield Model

    monkeypatch.setattr(goggle_validation, "_official_import_boundary", fake_import_boundary)
    execution = {
        **goggle_validation._adapter_config(),
        "dataset": "fixture",
        "input_dim": 3,
        "seed": 29,
    }
    _, observed = goggle_validation._run_native(
        tmp_path / "source",
        tmp_path / "native-output",
        pd.DataFrame(np.zeros((4, 3))),
        execution,
    )
    torch.manual_seed(29)
    expected = torch.randn(goggle_validation.EXPECTED_SAMPLE_ROWS, 3).numpy()

    assert np.array_equal(observed, expected)


def test_goggle_graphconv_matches_dgl_forward_gradients_and_state_contract() -> None:
    dgl = pytest.importorskip("dgl")
    from dgl.nn import GraphConv as DGLGraphConv

    source, destination, candidate_graph = _fixture_graph()
    reference_graph = dgl.graph((source, destination), num_nodes=4)
    for case_number, (in_features, out_features) in enumerate(((5, 2), (2, 5)), start=1):
        generator = torch.Generator().manual_seed(9100 + case_number)
        reference_features = torch.randn(4, in_features, generator=generator, dtype=torch.float64).requires_grad_()
        candidate_features = reference_features.detach().clone().requires_grad_()
        reference_edges = torch.randn(8, generator=generator, dtype=torch.float64).requires_grad_()
        candidate_edges = reference_edges.detach().clone().requires_grad_()
        reference = DGLGraphConv(in_features, out_features, norm="both", bias=True).to(torch.float64)
        candidate = GraphConv(in_features, out_features, norm="both", bias=True).to(torch.float64)
        candidate.load_state_dict(reference.state_dict(), strict=True)

        reference_output = reference(reference_graph, reference_features, edge_weight=reference_edges)
        candidate_output = candidate(candidate_graph, candidate_features, edge_weight=candidate_edges)
        assert torch.allclose(reference_output, candidate_output, rtol=1e-12, atol=1e-12)
        reference_output.square().sum().backward()
        candidate_output.square().sum().backward()
        assert list(reference.state_dict()) == list(candidate.state_dict()) == ["weight", "bias"]
        for reference_gradient, candidate_gradient in (
            (reference_features.grad, candidate_features.grad),
            (reference_edges.grad, candidate_edges.grad),
            (reference.weight.grad, candidate.weight.grad),
            (reference.bias.grad, candidate.bias.grad),
        ):
            assert reference_gradient is not None and candidate_gradient is not None
            assert torch.allclose(reference_gradient, candidate_gradient, rtol=1e-11, atol=1e-12)
