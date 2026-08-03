from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.tabstruct import normalize_tabdiff_or_tabsyn_summary
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter


class TabDiffAdapter(BaseModelAdapter):
    model_name = "tabdiff"
    upstream_dirname = "TabDiff-main"

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return spec.checkpoint_path
        exp_name = spec.extra.get("exp_name", spec.output_dir.name)
        ckpt_parent = self.upstream_root / "tabdiff" / "ckpt" / spec.dataset / exp_name
        ckpt_paths = sorted(ckpt_parent.glob("best_ema_model*"))
        if not ckpt_paths:
            raise FileNotFoundError(f"Could not infer TabDiff checkpoint from {ckpt_parent}")
        return ckpt_paths[0]

    def _infer_sample_path(self, checkpoint_path: Path) -> Path:
        epoch = int(checkpoint_path.stem.split("_")[-1])
        parent_parts = list(checkpoint_path.parent.parts)
        try:
            checkpoint_index = len(parent_parts) - 1 - parent_parts[::-1].index("ckpt")
        except ValueError as exc:
            raise ValueError(f"TabDiff checkpoint is not under a ckpt directory: {checkpoint_path}") from exc
        parent_parts[checkpoint_index] = "result"
        result_dir = Path(*parent_parts)
        return result_dir / str(epoch) / "samples.csv"

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        args = [
            "main.py",
            "--dataname",
            spec.dataset,
            "--mode",
            "train",
            "--gpu",
            str(spec.extra.get("gpu", 0)),
            "--exp_name",
            spec.extra.get("exp_name", spec.output_dir.name),
        ]
        if spec.extra.get("debug"):
            args.append("--debug")
        if spec.extra.get("no_wandb", True):
            args.append("--no_wandb")
        self._run_python(args, self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=["Training artifacts are written by the upstream TabDiff code."],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        args = [
            "main.py",
            "--dataname",
            spec.dataset,
            "--mode",
            "test",
            "--gpu",
            str(spec.extra.get("gpu", 0)),
            "--exp_name",
            spec.extra.get("exp_name", spec.output_dir.name),
            "--ckpt_path",
            str(checkpoint_path),
        ]
        if spec.num_samples is not None:
            args.extend(["--num_samples_to_generate", str(spec.num_samples)])
        if spec.extra.get("no_wandb", True):
            args.append("--no_wandb")
        self._run_python(args, self.upstream_root)
        sample_path = self._infer_sample_path(checkpoint_path)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path if sample_path.exists() else None,
            notes=["TabDiff test mode generates samples and evaluation outputs together."],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = spec.extra.get("sample_path")
        if sample_path is None:
            raise ValueError("TabDiff evaluation requires spec.extra['sample_path'].")
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
