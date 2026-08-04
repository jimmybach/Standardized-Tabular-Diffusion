from __future__ import annotations

import argparse
import functools
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configured_threads = os.environ.get("STANDARDIZED_TABSYN_NUM_THREADS")
    if configured_threads is not None:
        torch.set_num_threads(int(configured_threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(gpu: int) -> str:
    import torch

    if gpu == -1:
        return "cpu"
    if gpu < -1:
        raise ValueError("TabSyn GPU must be -1 for CPU or a non-negative CUDA index.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device cuda:{gpu} was requested, but CUDA is unavailable.")
    if gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device cuda:{gpu} was requested, but only {torch.cuda.device_count()} devices exist.")
    return f"cuda:{gpu}"


def _namespace(args: argparse.Namespace, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataname=args.dataname,
        device=device,
        gpu=args.gpu,
        lambd=args.lambd,
        max_beta=args.max_beta,
        min_beta=args.min_beta,
        num_samples=args.num_samples,
        save_path=args.save_path,
        steps=args.steps,
    )


def _run_sample(args: argparse.Namespace, upstream_args: SimpleNamespace) -> None:
    from tabsyn import sample as sample_module

    official_sample: Callable[..., Any] = sample_module.sample

    @functools.wraps(official_sample)
    def configured_sample(
        net: Any,
        native_num_samples: int,
        dim: int,
        num_steps: int = 50,
        device: str = "cuda:0",
    ) -> Any:
        del num_steps, device
        requested_rows = args.num_samples if args.num_samples is not None else native_num_samples
        return official_sample(
            net,
            requested_rows,
            dim,
            num_steps=args.steps,
            device=upstream_args.device,
        )

    sample_module.sample = configured_sample
    sample_module.main(upstream_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invoke the unmodified official TabSyn implementation through a compatibility boundary."
    )
    parser.add_argument("--action", choices=("vae-train", "diffusion-train", "sample"), required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-beta", type=float, default=1e-2)
    parser.add_argument("--min-beta", type=float, default=1e-5)
    parser.add_argument("--lambd", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("TabSyn seed must be non-negative.")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("TabSyn num_samples must be positive.")
    if args.steps < 2:
        raise ValueError("TabSyn diffusion sampling requires at least two steps.")
    if args.action == "sample" and args.save_path is None:
        raise ValueError("TabSyn sampling requires --save-path.")

    upstream_root = Path.cwd().resolve()
    if not (upstream_root / "tabsyn" / "model.py").is_file():
        raise FileNotFoundError(f"TabSyn launcher must run from an official source root: {upstream_root}")
    sys.path.insert(0, str(upstream_root))

    _seed_everything(args.seed)
    device = _resolve_device(args.gpu)
    upstream_args = _namespace(args, device)

    if args.action == "vae-train":
        from tabsyn.vae.main import main as train_vae

        train_vae(upstream_args)
    elif args.action == "diffusion-train":
        from tabsyn.main import main as train_diffusion

        train_diffusion(upstream_args)
    else:
        _run_sample(args, upstream_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
