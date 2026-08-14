from __future__ import annotations

import os
from pathlib import Path

from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter


def build_tabddpm_environment(upstream_root: Path) -> dict[str, str]:
    """Expose official imports plus the narrow scikit-learn API bridge."""

    compatibility_root = Path(__file__).resolve().parents[1] / "compat" / "tabddpm_sklearn"
    entries = [str(compatibility_root), str(upstream_root.resolve())]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return {"PYTHONPATH": os.pathsep.join(entries)}


class TabDDPMAdapter(BaseModelAdapter):
    model_name = "tabddpm"
    upstream_dirname = "TabDDPM-main"

    def _require_config(self, spec: RunSpec) -> Path:
        if spec.upstream_config_path is None:
            raise ValueError("TabDDPM requires RunSpec.upstream_config_path.")
        return spec.upstream_config_path

    def train(self, spec: RunSpec) -> ArtifactBundle:
        config_path = self._require_config(spec)
        self._ensure_output_dir(spec)
        self._run_python(
            ["scripts/pipeline.py", "--config", str(config_path), "--train"],
            self.upstream_root,
            env=build_tabddpm_environment(self.upstream_root),
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=["Training/output directories are controlled by the upstream TabDDPM TOML config."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        config_path = self._require_config(spec)
        self._ensure_output_dir(spec)
        self._run_python(
            ["scripts/pipeline.py", "--config", str(config_path), "--sample"],
            self.upstream_root,
            env=build_tabddpm_environment(self.upstream_root),
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        raise RuntimeError(
            "TabDDPM upstream-summary normalization is retired; provide the decoded sample to the central runner."
        )
