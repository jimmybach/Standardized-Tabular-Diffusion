from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.tabstruct import normalize_tabddpm_summary
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter


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
        self._run_python(["scripts/pipeline.py", "--config", str(config_path), "--train"], self.upstream_root)
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
        self._run_python(["scripts/pipeline.py", "--config", str(config_path), "--sample"], self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        summary_path = spec.output_dir / "standardized_summary.json"
        normalize_tabddpm_summary(
            dataset=spec.dataset,
            output_path=summary_path,
            metrics_paths={
                "catboost": Path(spec.extra["results_catboost_path"])
                if spec.extra.get("results_catboost_path")
                else None,
                "mlp": Path(spec.extra["results_mlp_path"]) if spec.extra.get("results_mlp_path") else None,
                "privacy": Path(spec.extra["privacy_path"]) if spec.extra.get("privacy_path") else None,
                "simple": Path(spec.extra["simple_path"]) if spec.extra.get("simple_path") else None,
            },
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            upstream_metrics_path=Path(spec.extra["results_catboost_path"])
            if spec.extra.get("results_catboost_path")
            else None,
            standardized_summary_path=summary_path,
            notes=[
                "TabDDPM normalization is limited to metrics already emitted by the upstream evaluation stack.",
            ],
        )
        return self._write_bundle(bundle)
