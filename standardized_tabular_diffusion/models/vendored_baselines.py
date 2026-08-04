from __future__ import annotations

from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models._runtime import SampleFileEvaluatorMixin
from standardized_tabular_diffusion.models.base import BaseModelAdapter


class _TabSynVendoredBaselineAdapter(BaseModelAdapter, SampleFileEvaluatorMixin):
    upstream_dirname = "TabSyn-main"
    tabsyn_method_name: str

    def _build_cli_args(self, spec: RunSpec, *, mode: str) -> list[str]:
        args = [
            "main.py",
            "--method",
            self.tabsyn_method_name,
            "--mode",
            mode,
            "--dataname",
            spec.dataset,
            "--gpu",
            str(spec.extra.get("gpu", 0)),
        ]

        passthrough_map = {
            "num_epochs": "--num_epochs",
            "batch_size": "--batch_size",
            "bs": "--bs",
            "training_batch_size": "--training_batch_size",
            "eval_batch_size": "--eval_batch_size",
            "T": "--T",
            "beta_1": "--beta_1",
            "beta_T": "--beta_T",
            "lr_con": "--lr_con",
            "lr_dis": "--lr_dis",
            "total_epochs_both": "--total_epochs_both",
            "sample_step": "--sample_step",
            "lambda_con": "--lambda_con",
            "lambda_dis": "--lambda_dis",
            "nf_con": "--nf_con",
            "nf_dis": "--nf_dis",
            "encoder_dim_con": "--encoder_dim_con",
            "encoder_dim_dis": "--encoder_dim_dis",
            "steps": "--steps",
        }
        for key, flag in passthrough_map.items():
            if spec.extra.get(key) is not None:
                args.extend([flag, str(spec.extra[key])])

        if mode == "sample":
            args.extend(["--save_path", str((spec.output_dir / "samples.csv").resolve())])
            if spec.num_samples is not None:
                args.extend(["--num-samples", str(spec.num_samples)])
        return args

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        self._run_python(self._build_cli_args(spec, mode="train"), self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                f"Training dispatched through TabSyn vendored baseline method `{self.tabsyn_method_name}`.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        sample_path = (spec.output_dir / "samples.csv").resolve()
        self._run_python(self._build_cli_args(spec, mode="sample"), self.upstream_root)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                f"Sampling dispatched through TabSyn vendored baseline method `{self.tabsyn_method_name}`.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        return self._evaluate_from_sample_file(spec)


class STaSyAdapter(_TabSynVendoredBaselineAdapter):
    model_name = "stasy"
    tabsyn_method_name = "stasy"


class CoDiAdapter(_TabSynVendoredBaselineAdapter):
    model_name = "codi"
    tabsyn_method_name = "codi"
