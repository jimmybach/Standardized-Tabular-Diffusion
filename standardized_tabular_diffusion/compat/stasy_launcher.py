from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


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
        raise ValueError("STaSy supports device='cpu', 'cuda', 'gpu', or 'cuda:<non-negative index>'.")
    try:
        index = int(normalized.split(":", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid STaSy CUDA device: {device!r}") from exc
    if index < 0:
        raise ValueError(f"Invalid STaSy CUDA device: {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {normalized} was requested, but CUDA is unavailable.")
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {normalized} was requested, but only {torch.cuda.device_count()} devices exist."
        )
    return normalized


def _configured_get_config(
    official_get_config: Callable[[str], Any],
    args: argparse.Namespace,
    device: str,
) -> Callable[[str], Any]:
    import torch

    def get_config(name: str) -> Any:
        config = official_get_config(name)
        config.device = torch.device(device)
        config.training.epoch = args.epochs - 1
        config.training.batch_size = args.batch_size
        config.training.spl = args.spl
        config.eval.batch_size = args.batch_size
        config.model.nf = args.nf
        config.model.hidden_dims = tuple(args.hidden_dims)
        config.model.num_scales = args.num_scales
        config.sampling.method = args.sampler
        return config

    return get_config


class _TorchDeviceProxy:
    def __init__(self, torch_module: Any, device: str) -> None:
        self._torch = torch_module
        self._device = device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def device(self, value: Any = None, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, str) and value.startswith("cuda:"):
            value = self._device
        return self._torch.device(value, *args, **kwargs)


def _checkpoint_root(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _install_sklearn_onehot_bridge() -> tuple[Any, Any]:
    import sklearn.preprocessing

    official_encoder = sklearn.preprocessing.OneHotEncoder

    def compatible_encoder(*args: Any, **kwargs: Any) -> Any:
        if "sparse" in kwargs:
            if "sparse_output" in kwargs:
                raise TypeError("STaSy OneHotEncoder cannot receive both sparse and sparse_output.")
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return official_encoder(*args, **kwargs)

    sklearn.preprocessing.OneHotEncoder = compatible_encoder
    return sklearn.preprocessing, official_encoder


def _run_train(args: argparse.Namespace, device: str) -> None:
    from baselines.stasy import main as train_module

    train_module.get_config = _configured_get_config(train_module.get_config, args, device)
    official_loader = train_module.DataLoader

    def configured_loader(*loader_args: Any, **loader_kwargs: Any) -> Any:
        loader_kwargs["num_workers"] = args.num_workers
        return official_loader(*loader_args, **loader_kwargs)

    train_module.DataLoader = configured_loader
    train_module.__file__ = str(_checkpoint_root(args.output_dir) / "upstream-stasy-main.py")
    official_torch = train_module.torch
    train_module.torch = _TorchDeviceProxy(official_torch, device)
    try:
        train_module.main(SimpleNamespace(dataname=args.dataname, gpu=0))
    finally:
        train_module.torch = official_torch


def _run_sample(args: argparse.Namespace, device: str) -> None:
    from baselines.stasy import sample as sample_module

    sample_module.get_config = _configured_get_config(sample_module.get_config, args, device)
    sample_module.__file__ = str(_checkpoint_root(args.output_dir) / "upstream-stasy-sample.py")
    official_json_load = sample_module.json.load
    expected_info = (Path.cwd() / "data" / args.dataname / "info.json").resolve()

    def configured_json_load(handle: Any, *load_args: Any, **load_kwargs: Any) -> Any:
        payload = official_json_load(handle, *load_args, **load_kwargs)
        handle_name = getattr(handle, "name", None)
        is_dataset_info = handle_name is not None and Path(handle_name).resolve() == expected_info
        if args.num_samples is not None and is_dataset_info and isinstance(payload, dict) and "train_num" in payload:
            payload = dict(payload)
            payload["train_num"] = args.num_samples
        return payload

    sample_module.json.load = configured_json_load
    official_torch = sample_module.torch
    sample_module.torch = _TorchDeviceProxy(official_torch, device)
    try:
        sample_module.main(
            SimpleNamespace(
                dataname=args.dataname,
                gpu=0,
                save_path=str(args.save_path.resolve()),
            )
        )
    finally:
        sample_module.json.load = official_json_load
        sample_module.torch = official_torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invoke the checksum-locked TabSyn STaSy snapshot through an adapter-only compatibility boundary."
    )
    parser.add_argument("--action", choices=("train", "sample"), required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10001)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 2048, 1024, 1024])
    parser.add_argument("--num-scales", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--sampler", choices=("ode", "pc"), default="ode")
    parser.add_argument("--spl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--save-path", type=Path)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("STaSy seed must be non-negative.")
    if args.epochs <= 0:
        raise ValueError("STaSy epochs must be positive.")
    if args.batch_size <= 0 or args.nf <= 0:
        raise ValueError("STaSy batch_size and nf must be positive.")
    if not args.hidden_dims or any(value <= 0 for value in args.hidden_dims):
        raise ValueError("STaSy hidden_dims must contain positive integers.")
    if args.num_scales < 2:
        raise ValueError("STaSy num_scales must be at least two.")
    if args.num_workers < 0 or args.num_threads <= 0:
        raise ValueError("STaSy num_workers must be non-negative and num_threads must be positive.")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("STaSy num_samples must be positive.")
    if args.action == "sample" and args.save_path is None:
        raise ValueError("STaSy sampling requires --save-path.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    upstream_root = Path.cwd().resolve()
    required = upstream_root / "baselines" / "stasy" / "main.py"
    if not required.is_file():
        raise FileNotFoundError(f"STaSy launcher must run from the locked TabSyn source root: {upstream_root}")
    data_info = upstream_root / "data" / args.dataname / "info.json"
    if not data_info.is_file():
        raise FileNotFoundError(f"STaSy dataset metadata is missing: {data_info}")
    sys.path.insert(0, str(upstream_root))
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    _seed_everything(args.seed, args.num_threads)
    device = _resolve_device(args.device)
    sklearn_preprocessing, official_encoder = _install_sklearn_onehot_bridge()
    try:
        if args.action == "train":
            _run_train(args, device)
        else:
            _run_sample(args, device)
    finally:
        sklearn_preprocessing.OneHotEncoder = official_encoder
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
