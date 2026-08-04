from __future__ import annotations

from pathlib import Path

from standardized_tabular_diffusion.evaluation.tabstruct import normalize_tabdiff_or_tabsyn_summary
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter


class TabDiffAdapter(BaseModelAdapter):
    model_name = "tabdiff"
    upstream_dirname = "TabDiff-main"

    @staticmethod
    def _gpu_index(spec: RunSpec) -> int:
        if "gpu" in spec.extra:
            return int(spec.extra["gpu"])
        device = str(spec.device).lower()
        if device == "cpu":
            return -1
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        raise ValueError(f"Unsupported TabDiff device {spec.device!r}; use 'cpu', 'cuda', or 'cuda:<index>'.")

    @staticmethod
    def _validate_seed_contract(spec: RunSpec) -> bool:
        deterministic = bool(spec.extra.get("deterministic", True))
        if spec.seed != 0:
            raise ValueError(
                "The pinned official TabDiff CLI exposes only deterministic seed 0. "
                "Use seed=0; configurable upstream seeds require an approved source change."
            )
        return deterministic

    def _common_args(self, spec: RunSpec) -> list[str]:
        args = ["--gpu", str(self._gpu_index(spec))]
        if spec.extra.get("debug"):
            args.append("--debug")
        if spec.extra.get("no_wandb", True):
            args.append("--no_wandb")
        if self._validate_seed_contract(spec):
            args.append("--deterministic")
        if spec.extra.get("non_learnable_schedule"):
            args.append("--non_learnable_schedule")
        if spec.extra.get("y_only"):
            args.append("--y_only")
        return args

    def _resolve_checkpoint_path(self, spec: RunSpec) -> Path:
        if spec.checkpoint_path is not None:
            return self._validate_trusted_executable_artifact(
                spec,
                spec.checkpoint_path,
                format_name="PyTorch checkpoint",
            )
        exp_name = spec.extra.get("exp_name", spec.output_dir.name)
        ckpt_parent = self.upstream_root / "tabdiff" / "ckpt" / spec.dataset / exp_name
        ckpt_paths = sorted(ckpt_parent.glob("best_ema_model*"))
        if not ckpt_paths:
            raise FileNotFoundError(f"Could not infer TabDiff checkpoint from {ckpt_parent}")
        checkpoint_path = ckpt_paths[0]
        if checkpoint_path.is_symlink():
            raise PermissionError(f"Refusing to load a symlinked PyTorch checkpoint: {checkpoint_path}")
        return checkpoint_path.resolve(strict=True)

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

    def _infer_report_sample_path(self, spec: RunSpec, exp_name: str) -> Path:
        return (
            self.upstream_root
            / "eval"
            / "report_runs"
            / exp_name
            / spec.dataset
            / "all_samples"
            / "samples_0.csv"
        )

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        args = [
            "main.py",
            "--dataname",
            spec.dataset,
            "--mode",
            "train",
            "--exp_name",
            spec.extra.get("exp_name", spec.output_dir.name),
        ]
        args.extend(self._common_args(spec))
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
        exp_name = spec.extra.get("exp_name", spec.output_dir.name)
        args = [
            "main.py",
            "--dataname",
            spec.dataset,
            "--mode",
            "test",
            "--exp_name",
            exp_name,
            "--ckpt_path",
            str(checkpoint_path),
        ]
        if spec.num_samples is not None:
            args.extend(["--num_samples_to_generate", str(spec.num_samples)])
        report = bool(spec.extra.get("report", False))
        if report:
            args.extend(["--report", "--num_runs", str(int(spec.extra.get("num_runs", 1)))])
        args.extend(self._common_args(spec))
        self._run_python(args, self.upstream_root)
        sample_path = (
            self._infer_report_sample_path(spec, exp_name) if report else self._infer_sample_path(checkpoint_path)
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path if sample_path.exists() else None,
            notes=[
                "TabDiff report mode generated samples and evaluation outputs together."
                if report
                else "TabDiff test mode generates samples and evaluation outputs together."
            ],
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
