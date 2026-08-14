"""Dependency-light runtime guards shared by optional model adapters."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path

from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec


@contextlib.contextmanager
def temporary_sys_path(path: Path) -> Iterator[None]:
    """Temporarily prepend one import path and restore the original state."""

    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path_str)


@contextlib.contextmanager
def isolated_module_tree(path: Path, namespace: str) -> Iterator[None]:
    """Import one uninstalled source tree without leaking or reusing its namespace."""

    prefix = f"{namespace}."
    previous = {
        name: module for name, module in tuple(sys.modules.items()) if name == namespace or name.startswith(prefix)
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        with temporary_sys_path(path):
            yield
    finally:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(prefix):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


@contextlib.contextmanager
def disable_torchvision_for_transformers() -> Iterator[None]:
    """Temporarily prevent optional torchvision probing by Transformers."""

    try:
        import transformers.utils.import_utils as import_utils
    except ModuleNotFoundError:
        yield
        return

    previous = import_utils._torchvision_available
    import_utils._torchvision_available = False
    try:
        yield
    finally:
        import_utils._torchvision_available = previous


class SampleFileEvaluatorMixin:
    """Compatibility surface that rejects retired adapter-local evaluation."""

    def _evaluate_from_sample_file(self, spec: RunSpec) -> ArtifactBundle:
        raise RuntimeError(
            "Adapter-local tabstruct-aligned-v1 evaluation was retired in P8. "
            "Use runner.run_action(..., 'evaluate'), `benchmark run`, or `evaluate-table`; "
            "use `import-legacy-summary` only for an existing standardized_summary.json."
        )
