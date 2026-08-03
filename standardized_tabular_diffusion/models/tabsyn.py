from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.tabstruct import normalize_tabdiff_or_tabsyn_summary
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter


class TabSynAdapter(BaseModelAdapter):
    model_name = "tabsyn"
    upstream_dirname = "TabSyn-main"

    def _vae_ckpt_dir(self, dataset: str) -> Path:
        return self.upstream_root / "tabsyn" / "vae" / "ckpt" / dataset

    def _diffusion_ckpt_dir(self, dataset: str) -> Path:
        return self.upstream_root / "tabsyn" / "ckpt" / dataset

    def _has_vae_artifacts(self, dataset: str) -> bool:
        ckpt_dir = self._vae_ckpt_dir(dataset)
        return (
            (ckpt_dir / "train_z.npy").exists()
            and (ckpt_dir / "decoder.pt").exists()
            and (ckpt_dir / "encoder.pt").exists()
        )

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        gpu = str(spec.extra.get("gpu", 0))
        skip_vae_if_present = spec.extra.get("skip_vae_if_present", False)

        if not (skip_vae_if_present and self._has_vae_artifacts(spec.dataset)):
            vae_args = [
                "tabsyn.vae.main",
                "--dataname",
                spec.dataset,
                "--gpu",
                gpu,
            ]
            if spec.extra.get("vae_num_epochs") is not None:
                vae_args.extend(["--num_epochs", str(spec.extra["vae_num_epochs"])])
            if spec.extra.get("max_beta") is not None:
                vae_args.extend(["--max_beta", str(spec.extra["max_beta"])])
            if spec.extra.get("min_beta") is not None:
                vae_args.extend(["--min_beta", str(spec.extra["min_beta"])])
            if spec.extra.get("lambd") is not None:
                vae_args.extend(["--lambd", str(spec.extra["lambd"])])
            self._run_python(vae_args, self.upstream_root, module=True)

        args = [
            "main.py",
            "--method",
            spec.extra.get("method", "tabsyn"),
            "--mode",
            "train",
            "--dataname",
            spec.dataset,
            "--gpu",
            gpu,
        ]
        # The shared patched parser defaults to 1,000 epochs for other baselines,
        # while the authoritative TabSyn diffusion default is 10,001.
        args.extend(["--num_epochs", str(spec.extra.get("diffusion_num_epochs", 10001))])
        self._run_python(args, self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                "TabSyn standardized train runs the VAE stage first, then the diffusion stage.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = (spec.output_dir / "samples.csv").resolve()
        args = [
            "main.py",
            "--method",
            spec.extra.get("method", "tabsyn"),
            "--mode",
            "sample",
            "--dataname",
            spec.dataset,
            "--gpu",
            str(spec.extra.get("gpu", 0)),
            "--save_path",
            str(sample_path),
        ]
        if spec.num_samples is not None:
            args.extend(["--num-samples", str(spec.num_samples)])
        if spec.extra.get("steps") is not None:
            args.extend(["--steps", str(spec.extra["steps"])])
        self._run_python(args, self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = spec.extra.get("sample_path")
        if sample_path is None:
            raise ValueError("TabSyn evaluation requires spec.extra['sample_path'].")
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
