from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

PATCH_RECORD_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "upstream" / "tabdiff-configurable-seed-overlay-v1.json"
)
RUNTIME_COMPATIBILITY_RECORD_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "upstream" / "tabdiff-pytorch-runtime-compatibility-v1.json"
)
DIAGNOSTIC_PLOT_BYPASS_RECORD_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "upstream" / "tabdiff-diagnostic-plot-bypass-v1.json"
)
CONFIG_PATH_OVERLAY_RECORD_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "upstream" / "tabdiff-config-path-overlay-v1.json"
)


class TabDiffSeedOverlayError(RuntimeError):
    """Raised when the approved TabDiff seed overlay cannot be applied exactly."""


def _sha256_lf(payload: bytes) -> str:
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"
    return hashlib.sha256(normalized).hexdigest()


def load_patch_record() -> dict[str, Any]:
    payload = json.loads(PATCH_RECORD_PATH.read_text(encoding="utf-8"))
    if payload.get("patch_schema_version") != "1.0.0":
        raise TabDiffSeedOverlayError("Unsupported TabDiff seed-overlay record version.")
    if payload.get("patch_id") != "tabdiff-configurable-seed-overlay-v1":
        raise TabDiffSeedOverlayError("Unexpected TabDiff seed-overlay identity.")
    return payload


def load_diagnostic_plot_bypass_record() -> dict[str, Any]:
    payload = json.loads(DIAGNOSTIC_PLOT_BYPASS_RECORD_PATH.read_text(encoding="utf-8"))
    if payload.get("patch_schema_version") != "1.0.0":
        raise TabDiffSeedOverlayError("Unsupported TabDiff diagnostic-plot bypass record version.")
    if payload.get("patch_id") != "tabdiff-diagnostic-plot-bypass-v1":
        raise TabDiffSeedOverlayError("Unexpected TabDiff diagnostic-plot bypass identity.")
    return payload


def load_config_path_overlay_record() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH_OVERLAY_RECORD_PATH.read_text(encoding="utf-8"))
    if payload.get("patch_schema_version") != "1.0.0":
        raise TabDiffSeedOverlayError("Unsupported TabDiff config-path overlay record version.")
    if payload.get("patch_id") != "tabdiff-config-path-overlay-v1":
        raise TabDiffSeedOverlayError("Unexpected TabDiff config-path overlay identity.")
    return payload


def install_pytorch_compatibility(torch_module: Any) -> list[str]:
    """Install reviewed semantic no-op bridges required by newer PyTorch releases."""

    scheduler_module = torch_module.optim.lr_scheduler
    scheduler_cls = scheduler_module.ReduceLROnPlateau
    if "verbose" in inspect.signature(scheduler_cls).parameters:
        return []

    class ReduceLROnPlateauVerboseBridge(scheduler_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, verbose: bool = False, **kwargs: Any) -> None:
            del verbose
            super().__init__(*args, **kwargs)

    ReduceLROnPlateauVerboseBridge.__name__ = scheduler_cls.__name__
    ReduceLROnPlateauVerboseBridge.__qualname__ = scheduler_cls.__qualname__
    scheduler_module.ReduceLROnPlateau = ReduceLROnPlateauVerboseBridge
    return ["tabdiff-pytorch-reduce-lr-verbose-bridge-v1"]


def apply_seed_overlay(source: str, record: dict[str, Any]) -> str:
    """Apply only the reviewed literal seed substitutions to verified official source."""

    patched = source
    for edit in record["edits"]:
        before = edit["before"]
        after = edit["after"]
        expected = edit["expected_occurrences"]
        observed = patched.count(before)
        if observed != expected:
            raise TabDiffSeedOverlayError(
                f"TabDiff seed-overlay anchor mismatch for {before!r}: expected {expected}, observed {observed}."
            )
        patched = patched.replace(before, after)
    return patched


def apply_config_path_overlay(source: str, record: dict[str, Any]) -> str:
    """Expose an optional config path without changing any configuration value."""

    edit = record["edit"]
    before = edit["before"]
    after = edit["after"]
    expected = edit["expected_occurrences"]
    observed = source.count(before)
    if observed != expected:
        raise TabDiffSeedOverlayError(
            f"TabDiff config-path overlay anchor mismatch: expected {expected}, observed {observed}."
        )
    return source.replace(before, after)


def apply_diagnostic_plot_bypass(source: str, record: dict[str, Any]) -> str:
    """Disable only upstream PNG rendering while retaining all metric calculations."""

    edit = record["edit"]
    before = edit["before"]
    after = edit["after"]
    expected = edit["expected_occurrences"]
    observed = source.count(before)
    if observed != expected:
        raise TabDiffSeedOverlayError(
            f"TabDiff diagnostic-plot bypass anchor mismatch: expected {expected}, observed {observed}."
        )
    return source.replace(before, after)


def load_patched_trainer(upstream_root: Path) -> None:
    """Load the official trainer with only optional PNG rendering disabled in memory."""

    record = load_diagnostic_plot_bypass_record()
    source_path = upstream_root / record["source_path"]
    if source_path.is_symlink() or not source_path.is_file():
        raise TabDiffSeedOverlayError(f"TabDiff official trainer is unavailable: {source_path}")
    source_bytes = source_path.read_bytes()
    observed_sha256 = _sha256_lf(source_bytes)
    if observed_sha256 != record["source_sha256_lf"]:
        raise TabDiffSeedOverlayError(
            "TabDiff official trainer differs from the diagnostic-plot bypass authority: "
            f"expected {record['source_sha256_lf']}, observed {observed_sha256}."
        )

    source = source_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    patched_source = apply_diagnostic_plot_bypass(source, record)
    if _sha256_lf(patched_source.encode("utf-8")) != record["patched_sha256_lf"]:
        raise TabDiffSeedOverlayError("The in-memory TabDiff diagnostic-plot bypass has an unexpected digest.")

    module_name = "tabdiff.trainer"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = "tabdiff"
    module.__spec__ = None
    sys.modules[module_name] = module
    exec(compile(patched_source, str(source_path), "exec"), module.__dict__)


def load_patched_main(upstream_root: Path) -> Callable[[argparse.Namespace], Any]:
    """Load the verified official module with the approved seed-only overlay in memory."""

    record = load_patch_record()
    source_path = upstream_root / record["source_path"]
    if source_path.is_symlink() or not source_path.is_file():
        raise TabDiffSeedOverlayError(f"TabDiff official source file is unavailable: {source_path}")
    source_bytes = source_path.read_bytes()
    observed_sha256 = _sha256_lf(source_bytes)
    if observed_sha256 != record["source_sha256_lf"]:
        raise TabDiffSeedOverlayError(
            "TabDiff official source differs from the seed-overlay authority: "
            f"expected {record['source_sha256_lf']}, observed {observed_sha256}."
        )

    source = source_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    patched_source = apply_seed_overlay(source, record)
    if _sha256_lf(patched_source.encode("utf-8")) != record["patched_sha256_lf"]:
        raise TabDiffSeedOverlayError("The in-memory TabDiff seed overlay has an unexpected digest.")

    config_record = load_config_path_overlay_record()
    if _sha256_lf(patched_source.encode("utf-8")) != config_record["input_sha256_lf"]:
        raise TabDiffSeedOverlayError("The TabDiff config-path overlay received an unexpected input digest.")
    patched_source = apply_config_path_overlay(patched_source, config_record)
    if _sha256_lf(patched_source.encode("utf-8")) != config_record["patched_sha256_lf"]:
        raise TabDiffSeedOverlayError("The in-memory TabDiff config-path overlay has an unexpected digest.")

    upstream_path = str(upstream_root)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    load_patched_trainer(upstream_root)
    module_name = "tabdiff.main"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = "tabdiff"
    module.__spec__ = None
    sys.modules[module_name] = module
    exec(compile(patched_source, str(source_path), "exec"), module.__dict__)
    main = module.__dict__.get("main")
    if not callable(main):
        raise TabDiffSeedOverlayError("Patched TabDiff module does not expose a callable main().")
    return main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invoke checksum-locked official TabDiff with the approved configurable-seed overlay."
    )
    parser.add_argument("--dataname", default="adult")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--method", default="tabdiff")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--exp_name")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--y_only", action="store_true")
    parser.add_argument("--non_learnable_schedule", action="store_true")
    parser.add_argument("--num_samples_to_generate", type=int)
    parser.add_argument("--ckpt_path")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--impute", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_size", type=int, default=50)
    parser.add_argument("--resample_rounds", type=int, default=1)
    parser.add_argument("--impute_condition", default="x_t")
    parser.add_argument("--y_only_model_path")
    parser.add_argument("--w_num", type=float, default=0.6)
    parser.add_argument("--w_cat", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("TabDiff seed must be a non-negative integer.")
    if not args.deterministic:
        raise ValueError("The configurable TabDiff seed contract requires --deterministic.")
    if args.gpu < -1:
        raise ValueError("TabDiff GPU must be -1 for CPU or a non-negative CUDA index.")

    import torch

    if args.gpu == -1:
        args.device = "cpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device cuda:{args.gpu} was requested, but CUDA is unavailable.")
        if args.gpu >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device cuda:{args.gpu} was requested, but only {torch.cuda.device_count()} devices exist."
            )
        args.device = f"cuda:{args.gpu}"

    upstream_root = Path.cwd().resolve()
    install_pytorch_compatibility(torch)
    patched_main = load_patched_main(upstream_root)
    patched_main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
