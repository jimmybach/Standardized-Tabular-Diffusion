from __future__ import annotations

import os
import subprocess
import sys
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

    def _gpu_argument(self, spec: RunSpec) -> int:
        if "gpu" in spec.extra:
            gpu = int(spec.extra["gpu"])
            if gpu < -1:
                raise ValueError("TabSyn GPU must be -1 for CPU or a non-negative CUDA index.")
            return gpu
        device = spec.device.strip().lower()
        if device == "cpu":
            return -1
        if device in {"cuda", "gpu"}:
            return 0
        if device.startswith("cuda:"):
            try:
                gpu = int(device.split(":", maxsplit=1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid TabSyn CUDA device: {spec.device!r}") from exc
            if gpu < 0:
                raise ValueError(f"Invalid TabSyn CUDA device: {spec.device!r}")
            return gpu
        raise ValueError("TabSyn supports device='cpu', 'cuda', 'gpu', or 'cuda:<non-negative index>'.")

    def _run_tabsyn(self, args: list[str], *, seed: int) -> None:
        launcher = self.repo_root / "standardized_tabular_diffusion" / "compat" / "tabsyn.py"
        if not launcher.is_file():
            raise FileNotFoundError(f"TabSyn compatibility launcher is missing: {launcher}")
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(seed)
        subprocess.run(
            [sys.executable, str(launcher), *args, "--seed", str(seed)],
            cwd=self.upstream_root,
            check=True,
            env=environment,
        )

    def _require_internal_checkpoint(self, path: Path, *, label: str) -> Path:
        if path.is_symlink():
            raise PermissionError(f"Refusing to load a symlinked TabSyn {label}: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.upstream_root.resolve()):
            raise PermissionError(f"TabSyn {label} must remain inside the official source worktree: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"TabSyn {label} must be a regular file: {resolved}")
        return resolved

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        gpu = str(self._gpu_argument(spec))
        skip_vae_if_present = spec.extra.get("skip_vae_if_present", False)

        unsupported_controls = [
            key for key in ("vae_num_epochs", "diffusion_num_epochs") if spec.extra.get(key) is not None
        ]
        if unsupported_controls:
            raise ValueError(
                "The unmodified official TabSyn source does not expose epoch-count controls; remove: "
                + ", ".join(unsupported_controls)
            )

        if not (skip_vae_if_present and self._has_vae_artifacts(spec.dataset)):
            vae_args = [
                "--action",
                "vae-train",
                "--dataname",
                spec.dataset,
                "--gpu",
                gpu,
            ]
            if spec.extra.get("max_beta") is not None:
                vae_args.extend(["--max-beta", str(spec.extra["max_beta"])])
            if spec.extra.get("min_beta") is not None:
                vae_args.extend(["--min-beta", str(spec.extra["min_beta"])])
            if spec.extra.get("lambd") is not None:
                vae_args.extend(["--lambd", str(spec.extra["lambd"])])
            self._run_tabsyn(vae_args, seed=spec.seed)

        args = [
            "--action",
            "diffusion-train",
            "--dataname",
            spec.dataset,
            "--gpu",
            gpu,
        ]
        self._run_tabsyn(args, seed=spec.seed)
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
        if spec.checkpoint_path is not None:
            raise ValueError(
                "Official TabSyn sampling uses its fixed VAE and diffusion checkpoint layout; checkpoint_path is unsupported."
            )
        self._require_internal_checkpoint(self._vae_ckpt_dir(spec.dataset) / "train_z.npy", label="latent array")
        self._require_internal_checkpoint(self._vae_ckpt_dir(spec.dataset) / "decoder.pt", label="decoder checkpoint")
        self._require_internal_checkpoint(
            self._diffusion_ckpt_dir(spec.dataset) / "model.pt", label="diffusion checkpoint"
        )
        sample_path = (spec.output_dir / "samples.csv").resolve()
        args = [
            "--action",
            "sample",
            "--dataname",
            spec.dataset,
            "--gpu",
            str(self._gpu_argument(spec)),
            "--save-path",
            str(sample_path),
            "--steps",
            str(spec.extra.get("steps", 50)),
        ]
        if spec.num_samples is not None:
            args.extend(["--num-samples", str(spec.num_samples)])
        self._run_tabsyn(args, seed=spec.seed)
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
