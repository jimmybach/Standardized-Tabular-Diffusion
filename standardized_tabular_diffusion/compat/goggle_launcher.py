from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _seed_everything(seed: int, num_threads: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(device: str) -> str:
    import torch

    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized in {"cuda", "gpu"}:
        normalized = "cuda:0"
    if not normalized.startswith("cuda:"):
        raise ValueError("Goggle supports device='cpu', 'cuda', 'gpu', or 'cuda:<non-negative index>'.")
    try:
        index = int(normalized.split(":", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid Goggle CUDA device: {device!r}") from exc
    if index < 0:
        raise ValueError(f"Invalid Goggle CUDA device: {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {normalized} was requested, but CUDA is unavailable.")
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {normalized} was requested, but only {torch.cuda.device_count()} devices exist."
        )
    return normalized


class _UnavailableSchema:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError(
            "The official Synthcity Schema postprocessor is intentionally outside the Goggle adapter runtime. "
            "Standardized categorical and target constraints are applied after official core sampling."
        )


@contextmanager
def _official_import_boundary(source_dir: Path) -> Iterator[type[Any]]:
    """Import the untouched official package without its unused Synthcity evaluator stack."""

    module_names = (
        "synthcity",
        "synthcity.metrics",
        "synthcity.plugins",
        "synthcity.plugins.core",
        "synthcity.plugins.core.schema",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    rgcn_module_name = "goggle.model.RGCNConv"
    previous_rgcn = sys.modules.get(rgcn_module_name)
    synthcity = types.ModuleType("synthcity")
    synthcity.__path__ = []  # type: ignore[attr-defined]
    metrics = types.ModuleType("synthcity.metrics")
    for name in ("eval_detection", "eval_performance", "eval_statistical"):
        setattr(metrics, name, types.ModuleType(f"synthcity.metrics.{name}"))
    plugins = types.ModuleType("synthcity.plugins")
    plugins.__path__ = []  # type: ignore[attr-defined]
    core = types.ModuleType("synthcity.plugins.core")
    core.__path__ = []  # type: ignore[attr-defined]
    schema = types.ModuleType("synthcity.plugins.core.schema")
    schema.Schema = _UnavailableSchema  # type: ignore[attr-defined]
    replacements = {
        "synthcity": synthcity,
        "synthcity.metrics": metrics,
        "synthcity.plugins": plugins,
        "synthcity.plugins.core": core,
        "synthcity.plugins.core.schema": schema,
    }
    source_path = str((source_dir / "src").resolve())
    existing_source = source_path in sys.path
    for name in tuple(sys.modules):
        if name == "goggle" or name.startswith("goggle."):
            sys.modules.pop(name, None)
    try:
        import torch_sparse  # noqa: F401
    except ModuleNotFoundError:
        rgcn = types.ModuleType(rgcn_module_name)

        class _UnavailableRGCNConv:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise ModuleNotFoundError(
                    "Goggle decoder_arch='het' requires the official torch-sparse/torch-scatter extension stack. "
                    "The validated gcn and sage paths do not execute RGCNConv."
                )

        rgcn.RGCNConv = _UnavailableRGCNConv  # type: ignore[attr-defined]
        sys.modules[rgcn_module_name] = rgcn
    sys.modules.update(replacements)
    if not existing_source:
        sys.path.insert(0, source_path)
    try:
        from goggle.GoggleModel import GoggleModel

        yield GoggleModel
    finally:
        for name in tuple(sys.modules):
            if name == "goggle" or name.startswith("goggle."):
                sys.modules.pop(name, None)
        if previous_rgcn is not None:
            sys.modules[rgcn_module_name] = previous_rgcn
        if not existing_source:
            sys.path.remove(source_path)
        for name in module_names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def _model_kwargs(config: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    graph_prior = config.get("graph_prior")
    prior_mask = config.get("prior_mask")
    if graph_prior is not None:
        graph_prior = torch.tensor(graph_prior, dtype=torch.float32)
        prior_mask = torch.tensor(prior_mask, dtype=torch.float32)
    return {
        "ds_name": config["dataset"],
        "input_dim": config["input_dim"],
        "encoder_dim": config["encoder_dim"],
        "encoder_l": config["encoder_l"],
        "het_encoding": config["het_encoding"],
        "decoder_dim": config["decoder_dim"],
        "decoder_l": config["decoder_l"],
        "threshold": config["threshold"],
        "decoder_arch": config["decoder_arch"],
        "graph_prior": graph_prior,
        "prior_mask": prior_mask,
        "device": device,
        "alpha": config["alpha"],
        "beta": config["beta"],
        "seed": config["seed"],
        "iter_opt": config["iter_opt"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "patience": config["patience"],
        "logging": config["logging"],
    }


def _safe_dataset_id(value: str) -> str:
    candidate = Path(value)
    if (
        not value
        or "\x00" in value
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name in {".", ".."}
        or any(separator in value for separator in ("/", "\\", ":"))
    ):
        raise ValueError("Goggle dataset must be a single safe identifier, not a path.")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Goggle {label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _output_file(path: Path, output_dir: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(output_dir):
        raise ValueError(f"Goggle {label} must remain inside output_dir: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(_regular_file(path, "configuration").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Goggle configuration must be a JSON object.")
    return payload


def _run_train(args: argparse.Namespace, GoggleModel: type[Any], config: dict[str, Any], device: str) -> None:
    import pandas as pd

    input_csv = _regular_file(args.input_csv, "transformed training frame")
    frame = pd.read_csv(input_csv)
    if frame.shape != (config["training_rows"], config["input_dim"]):
        raise ValueError(
            "Goggle transformed training frame does not match the locked adapter metadata: "
            f"observed={frame.shape}, expected={(config['training_rows'], config['input_dim'])}."
        )
    model = GoggleModel(**_model_kwargs(config, device))
    previous_cwd = Path.cwd()
    try:
        os.chdir(args.output_dir)
        model.fit(frame)
    finally:
        os.chdir(previous_cwd)
    official_checkpoint = args.output_dir / "tmp" / f"{config['dataset']}.pt"
    official_checkpoint = _regular_file(official_checkpoint, "official checkpoint")
    checkpoint = _output_file(args.checkpoint, args.output_dir, "checkpoint")
    os.replace(official_checkpoint, checkpoint)


def _run_sample(args: argparse.Namespace, GoggleModel: type[Any], config: dict[str, Any], device: str) -> None:
    import numpy as np
    import torch

    checkpoint = _regular_file(args.checkpoint, "checkpoint")
    model = GoggleModel(**_model_kwargs(config, device))
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.model.load_state_dict(state_dict)
    raw = model.model.sample(args.num_samples).detach().cpu().numpy()
    if raw.shape != (args.num_samples, config["input_dim"]):
        raise RuntimeError(
            f"Goggle official core sampler returned shape {raw.shape}; "
            f"expected {(args.num_samples, config['input_dim'])}."
        )
    if not bool(np.isfinite(raw).all()):
        raise RuntimeError("Goggle official core sampler produced non-finite values.")
    raw_output = _output_file(args.raw_output, args.output_dir, "raw sample output")
    np.save(raw_output, raw, allow_pickle=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run checksum-locked, unmodified method-author Goggle source through an adapter-only boundary."
    )
    parser.add_argument("--action", choices=("train", "sample"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--raw-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_threads <= 0:
        raise ValueError("Goggle num_threads must be positive.")
    if args.output_dir.is_symlink():
        raise ValueError(f"Goggle output_dir must not be a symlink: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.output_dir.resolve(strict=True)
    args.source_dir = args.source_dir.resolve(strict=True)
    required = args.source_dir / "src" / "goggle" / "GoggleModel.py"
    _regular_file(required, "official source entry point")
    config = _load_config(args.config)
    _safe_dataset_id(str(config.get("dataset", "")))
    if args.action == "train" and args.input_csv is None:
        raise ValueError("Goggle training requires --input-csv.")
    if args.action == "sample":
        if args.num_samples is None or args.num_samples <= 0 or args.raw_output is None:
            raise ValueError("Goggle sampling requires positive --num-samples and --raw-output.")
    os.environ.setdefault("DGLBACKEND", "pytorch")
    os.environ.setdefault("PYTHONHASHSEED", str(config["seed"]))
    cache_dir = args.output_dir / ".runtime-cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    device = _resolve_device(args.device)
    _seed_everything(config["seed"], args.num_threads)
    with _official_import_boundary(args.source_dir) as GoggleModel:
        if args.action == "train":
            _run_train(args, GoggleModel, config, device)
        else:
            _run_sample(args, GoggleModel, config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
