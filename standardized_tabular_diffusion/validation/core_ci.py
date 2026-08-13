"""Cross-platform core CI checks that avoid shell-specific inline programs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REQUIRED_WHEEL_FILES = {
    "standardized_tabular_diffusion/resources/datasets/sources.json",
    "standardized_tabular_diffusion/resources/datasets/adult-uci-2-v1.json",
    "standardized_tabular_diffusion/resources/datasets/sick-uci-102-v1.json",
    "standardized_tabular_diffusion/resources/upstream/source-lock.json",
    "standardized_tabular_diffusion/resources/upstream/ctabgan-source-manifest.json",
    "standardized_tabular_diffusion/schemas/evaluation/atomic-result.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/metric-registry-entry.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/hardware-profile.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/orchestration-run.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/orchestration-stage-record.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/software-profile.schema.json",
    "standardized_tabular_diffusion/resources/evaluation/metrics/legacy-tabstruct-aligned-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/protocols/development-p1.json",
    "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-dataset-scale-windows-gpu-stable-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-utility-stable-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/evaluators/p5-high-order-privacy-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/metrics/p5-excluded-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/metrics/p5-high-order-privacy-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/protocols/p5-high-order-privacy.json",
    "standardized_tabular_diffusion/resources/evaluation/upstream/p5-sources.json",
    "standardized_tabular_diffusion/resources/evaluation/upstream/tabeval-p4-windows-gpu-runtime.json",
}


def verify_dependency_free_surface() -> None:
    import_check = """
import sys
import standardized_tabular_diffusion as package

assert package.list_models()
assert not {"numpy", "pandas", "sklearn", "torch"} & set(sys.modules)
"""
    with tempfile.TemporaryDirectory(prefix="std-tabular-core-ci-") as directory:
        subprocess.run([sys.executable, "-c", import_check], cwd=directory, check=True)
        subprocess.run(
            [sys.executable, "-m", "standardized_tabular_diffusion.cli", "--help"],
            cwd=directory,
            check=True,
        )


def inspect_wheel_contents(dist_directory: Path) -> None:
    wheels = sorted(dist_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel under {dist_directory}, observed {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    missing = sorted(REQUIRED_WHEEL_FILES - names)
    if missing:
        raise RuntimeError(f"Wheel is missing required files: {missing}")
    forbidden_prefixes = ("research_inputs/", "TabDDPM-main/", "TabDiff-main/", "TabSyn-main/")
    forbidden = sorted(name for name in names if name.startswith(forbidden_prefixes))
    if forbidden:
        raise RuntimeError(f"Wheel contains excluded reference-tree files: {forbidden}")


def install_single_wheel(dist_directory: Path) -> None:
    wheels = sorted(dist_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel under {dist_directory}, observed {len(wheels)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dependency-free-surface")
    wheel = subparsers.add_parser("inspect-wheel")
    wheel.add_argument("--dist-directory", type=Path, default=Path("dist"))
    install_wheel = subparsers.add_parser("install-wheel")
    install_wheel.add_argument("--dist-directory", type=Path, default=Path("dist"))
    args = parser.parse_args()
    if args.command == "dependency-free-surface":
        verify_dependency_free_surface()
    elif args.command == "inspect-wheel":
        inspect_wheel_contents(args.dist_directory)
    else:
        install_single_wheel(args.dist_directory)


if __name__ == "__main__":
    main()
