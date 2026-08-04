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
    """Dependency-light mixin that imports the evaluator only when invoked."""

    def _evaluate_from_sample_file(self, spec: RunSpec) -> ArtifactBundle:
        from standardized_tabular_diffusion.evaluation.tabstruct import normalize_tabdiff_or_tabsyn_summary

        self._ensure_output_dir(spec)
        sample_path = spec.extra.get("sample_path")
        if sample_path is None:
            raise ValueError(f"{self.model_name} evaluation requires spec.extra['sample_path'].")
        summary_path = spec.output_dir / "standardized_summary.json"
        normalize_tabdiff_or_tabsyn_summary(
            repo_root=self.repo_root,
            model_name=self.model_name,
            dataset=spec.dataset,
            sample_path=Path(sample_path),
            output_path=summary_path,
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=Path(sample_path),
            standardized_summary_path=summary_path,
        )
        return self._write_bundle(bundle)
