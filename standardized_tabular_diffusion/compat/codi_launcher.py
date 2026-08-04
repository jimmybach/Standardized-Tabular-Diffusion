from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


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


def _resolve_device(device: str) -> tuple[str, int]:
    import torch

    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu", 0
    if normalized in {"cuda", "gpu"}:
        normalized = "cuda:0"
    if not normalized.startswith("cuda:"):
        raise ValueError("CoDi supports device='cpu', 'cuda', 'gpu', or 'cuda:<non-negative index>'.")
    try:
        index = int(normalized.split(":", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid CoDi CUDA device: {device!r}") from exc
    if index < 0:
        raise ValueError(f"Invalid CoDi CUDA device: {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {normalized} was requested, but CUDA is unavailable.")
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {normalized} was requested, but only {torch.cuda.device_count()} devices exist."
        )
    return normalized, index


class _CudaProxy:
    def __init__(self, cuda_module: Any, *, enabled: bool) -> None:
        self._cuda = cuda_module
        self._enabled = enabled

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cuda, name)

    def is_available(self) -> bool:
        return self._enabled

    def device_count(self) -> int:
        return 1


class _TorchProxy:
    def __init__(self, torch_module: Any, *, cuda_enabled: bool) -> None:
        self._torch = torch_module
        self.cuda = _CudaProxy(torch_module.cuda, enabled=cuda_enabled)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)


def _official_args(args: argparse.Namespace, gpu_index: int) -> SimpleNamespace:
    return SimpleNamespace(
        dataname=args.dataname,
        gpu=gpu_index,
        training_batch_size=args.training_batch_size,
        eval_batch_size=args.eval_batch_size,
        T=args.diffusion_steps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        lr_con=args.lr_con,
        lr_dis=args.lr_dis,
        total_epochs_both=args.epochs,
        grad_clip=args.grad_clip,
        sample_step=args.sample_step,
        lambda_con=args.lambda_con,
        lambda_dis=args.lambda_dis,
        nf_con=args.nf_con,
        nf_dis=args.nf_dis,
        encoder_dim_con=",".join(str(value) for value in args.encoder_dim_con),
        encoder_dim_dis=",".join(str(value) for value in args.encoder_dim_dis),
        activation=args.activation,
        mean_type=args.mean_type,
        var_type=args.var_type,
        save_path=None if args.save_path is None else str(args.save_path.resolve()),
    )


def _run_train(args: argparse.Namespace, *, cuda_enabled: bool, gpu_index: int) -> None:
    import torch
    from baselines.codi import main as train_module

    output_anchor = args.output_dir.resolve() / "upstream-codi-main.py"
    official_module_torch = train_module.torch
    official_loader_torch = train_module.tabular_dataload.torch
    train_module.__file__ = str(output_anchor)
    train_module.torch = _TorchProxy(torch, cuda_enabled=cuda_enabled)
    train_module.tabular_dataload.torch = _TorchProxy(torch, cuda_enabled=cuda_enabled)
    try:
        train_module.main(_official_args(args, gpu_index))
    finally:
        train_module.torch = official_module_torch
        train_module.tabular_dataload.torch = official_loader_torch


def _run_sample(args: argparse.Namespace, *, cuda_enabled: bool, gpu_index: int) -> None:
    import numpy as np
    import torch
    from baselines.codi import sample as sample_module

    output_anchor = args.output_dir.resolve() / "upstream-codi-sample.py"
    official_module_torch = sample_module.torch
    official_loader_torch = sample_module.tabular_dataload.torch
    official_get_dataset = sample_module.tabular_dataload.get_dataset

    def configured_get_dataset(flags: Any, evaluation: bool = False) -> tuple[Any, ...]:
        result = official_get_dataset(flags, evaluation=evaluation)
        if args.num_samples is None:
            return result
        train, train_con, train_dis, test, transformers, con_idx, dis_idx = result
        if len(train) == args.num_samples:
            return result
        indices = np.arange(args.num_samples) % len(train)
        return train[indices], train_con[indices], train_dis[indices], test, transformers, con_idx, dis_idx

    sample_module.__file__ = str(output_anchor)
    sample_module.torch = _TorchProxy(torch, cuda_enabled=cuda_enabled)
    sample_module.tabular_dataload.torch = _TorchProxy(torch, cuda_enabled=cuda_enabled)
    sample_module.tabular_dataload.get_dataset = configured_get_dataset
    try:
        sample_module.main(_official_args(args, gpu_index))
    finally:
        sample_module.tabular_dataload.get_dataset = official_get_dataset
        sample_module.tabular_dataload.torch = official_loader_torch
        sample_module.torch = official_module_torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invoke the checksum-locked TabSyn CoDi snapshot through an adapter-only compatibility boundary."
    )
    parser.add_argument("--action", choices=("train", "sample"), required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=2100)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-1", type=float, default=0.00001)
    parser.add_argument("--beta-T", type=float, default=0.02)
    parser.add_argument("--lr-con", type=float, default=0.002)
    parser.add_argument("--lr-dis", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-step", type=int, default=2000)
    parser.add_argument("--lambda-con", type=float, default=0.2)
    parser.add_argument("--lambda-dis", type=float, default=0.2)
    parser.add_argument("--nf-con", type=int, default=16)
    parser.add_argument("--nf-dis", type=int, default=64)
    parser.add_argument("--encoder-dim-con", type=int, nargs="+", default=[512, 1024, 1024, 512])
    parser.add_argument("--encoder-dim-dis", type=int, nargs="+", default=[512, 1024, 1024, 512])
    parser.add_argument(
        "--activation", choices=("elu", "relu", "lrelu", "swish", "tanh", "softplus"), default="relu"
    )
    parser.add_argument("--mean-type", choices=("xprev", "xstart", "epsilon"), default="epsilon")
    parser.add_argument("--var-type", choices=("fixedlarge", "fixedsmall"), default="fixedsmall")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--save-path", type=Path)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("CoDi seed must be non-negative.")
    for name in ("training_batch_size", "eval_batch_size", "epochs", "sample_step", "num_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"CoDi {name} must be positive.")
    if args.diffusion_steps < 2 or args.nf_con < 4 or args.nf_dis < 4:
        raise ValueError("CoDi diffusion_steps must be at least two and nf widths must be at least four.")
    if len(args.encoder_dim_con) < 2 or len(args.encoder_dim_dis) < 2:
        raise ValueError("CoDi encoder dimensions must contain at least two widths.")
    if any(value <= 0 for value in [*args.encoder_dim_con, *args.encoder_dim_dis]):
        raise ValueError("CoDi encoder dimensions must be positive.")
    if not 0 < args.beta_1 < args.beta_T < 1:
        raise ValueError("CoDi requires 0 < beta_1 < beta_T < 1.")
    if args.lr_con <= 0 or args.lr_dis <= 0 or args.grad_clip <= 0:
        raise ValueError("CoDi learning rates and grad_clip must be positive.")
    if args.lambda_con < 0 or args.lambda_dis < 0:
        raise ValueError("CoDi contrastive weights must be non-negative.")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("CoDi num_samples must be positive.")
    if args.action == "sample" and args.save_path is None:
        raise ValueError("CoDi sampling requires --save-path.")
    if args.output_dir.is_symlink():
        raise ValueError(f"CoDi output_dir must not be a symlink: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_dir.resolve()
    if args.save_path is not None:
        if not args.save_path.resolve().is_relative_to(output_root):
            raise ValueError("CoDi save_path must remain inside output_dir.")
        args.save_path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    upstream_root = Path.cwd().resolve()
    required = upstream_root / "baselines" / "codi" / "main.py"
    if not required.is_file():
        raise FileNotFoundError(f"CoDi launcher must run from the locked TabSyn source root: {upstream_root}")
    data_info = upstream_root / "data" / args.dataname / "info.json"
    if not data_info.is_file():
        raise FileNotFoundError(f"CoDi dataset metadata is missing: {data_info}")
    sys.path.insert(0, str(upstream_root))
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    _seed_everything(args.seed, args.num_threads)
    device, gpu_index = _resolve_device(args.device)
    if args.action == "train":
        _run_train(args, cuda_enabled=device.startswith("cuda:"), gpu_index=gpu_index)
    else:
        _run_sample(args, cuda_enabled=device.startswith("cuda:"), gpu_index=gpu_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
