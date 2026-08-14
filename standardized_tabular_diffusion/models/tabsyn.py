from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.runtime_contracts import bind_native_dataset_view


class TabSynAdapter(BaseModelAdapter):
    model_name = "tabsyn"
    upstream_dirname = "TabSyn-main"

    @staticmethod
    def _runtime_root(spec: RunSpec) -> Path:
        return spec.output_dir / "tabsyn-runtime"

    def _vae_ckpt_dir(self, spec: RunSpec) -> Path:
        return self._runtime_root(spec) / "tabsyn" / "vae" / "ckpt" / spec.dataset

    def _diffusion_ckpt_dir(self, spec: RunSpec) -> Path:
        return self._runtime_root(spec) / "tabsyn" / "ckpt" / spec.dataset

    def _has_vae_artifacts(self, spec: RunSpec) -> bool:
        ckpt_dir = self._vae_ckpt_dir(spec)
        return (
            (ckpt_dir / "train_z.npy").exists()
            and (ckpt_dir / "decoder.pt").exists()
            and (ckpt_dir / "encoder.pt").exists()
        )

    @staticmethod
    def _metadata_path(spec: RunSpec) -> Path:
        return spec.output_dir / "tabsyn-model-metadata.json"

    def _dataset_binding(self, spec: RunSpec) -> dict[str, Any] | None:
        if "dataset_identity" not in spec.extra:
            return None
        dataset_spec = self.resolve_dataset_spec(spec)
        return bind_native_dataset_view(dataset_spec, self.upstream_root / "data" / spec.dataset)

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
        launcher = self.repo_root / "standardized_tabular_diffusion" / "compat" / "tabsyn_launcher.py"
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

    def _require_internal_checkpoint(self, spec: RunSpec, path: Path, *, label: str) -> Path:
        if path.is_symlink():
            raise PermissionError(f"Refusing to load a symlinked TabSyn {label}: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self._runtime_root(spec).resolve()):
            raise PermissionError(f"TabSyn {label} must remain inside the declared run output: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"TabSyn {label} must be a regular file: {resolved}")
        return resolved

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_binding = self._dataset_binding(spec)
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

        runtime_args = ["--runtime-root", str(self._runtime_root(spec).resolve())]
        if not (skip_vae_if_present and self._has_vae_artifacts(spec)):
            vae_args = [
                "--action",
                "vae-train",
                "--dataname",
                spec.dataset,
                "--gpu",
                gpu,
                *runtime_args,
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
            *runtime_args,
        ]
        self._run_tabsyn(args, seed=spec.seed)
        if dataset_binding is not None:
            checkpoints = {
                "train_z.npy": self._require_internal_checkpoint(
                    spec, self._vae_ckpt_dir(spec) / "train_z.npy", label="latent array"
                ),
                "decoder.pt": self._require_internal_checkpoint(
                    spec, self._vae_ckpt_dir(spec) / "decoder.pt", label="decoder checkpoint"
                ),
                "model.pt": self._require_internal_checkpoint(
                    spec, self._diffusion_ckpt_dir(spec) / "model.pt", label="diffusion checkpoint"
                ),
            }
            atomic_write_json(
                self._metadata_path(spec),
                {
                    "schema_version": 1,
                    "model": self.model_name,
                    "dataset": spec.dataset,
                    "training_seed": spec.seed,
                    "device": spec.device,
                    "dataset_binding": dataset_binding,
                    "checkpoints": {
                        name: {"path": str(path), "sha256": sha256_file(path)}
                        for name, path in checkpoints.items()
                    },
                },
            )
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
        dataset_binding = self._dataset_binding(spec)
        if spec.checkpoint_path is not None:
            raise ValueError(
                "Official TabSyn sampling uses its fixed VAE and diffusion checkpoint layout; checkpoint_path is unsupported."
            )
        checkpoints = {
            "train_z.npy": self._require_internal_checkpoint(
                spec, self._vae_ckpt_dir(spec) / "train_z.npy", label="latent array"
            ),
            "decoder.pt": self._require_internal_checkpoint(
                spec, self._vae_ckpt_dir(spec) / "decoder.pt", label="decoder checkpoint"
            ),
            "model.pt": self._require_internal_checkpoint(
                spec, self._diffusion_ckpt_dir(spec) / "model.pt", label="diffusion checkpoint"
            ),
        }
        if dataset_binding is not None:
            metadata_path = self._metadata_path(spec)
            if metadata_path.is_symlink() or not metadata_path.is_file():
                raise FileNotFoundError("TabSyn sampling requires run-owned training metadata.")
            metadata = read_json(metadata_path)
            if (
                metadata.get("model") != self.model_name
                or metadata.get("dataset") != spec.dataset
                or metadata.get("dataset_binding") != dataset_binding
            ):
                raise ValueError("TabSyn training metadata does not match the current model/dataset binding.")
            for name, path in checkpoints.items():
                if metadata.get("checkpoints", {}).get(name, {}).get("sha256") != sha256_file(path):
                    raise ValueError(f"TabSyn checkpoint changed after training: {name}")
        sample_path = (spec.output_dir / "samples.csv").resolve()
        args = [
            "--action",
            "sample",
            "--dataname",
            spec.dataset,
            "--gpu",
            str(self._gpu_argument(spec)),
            "--runtime-root",
            str(self._runtime_root(spec).resolve()),
            "--save-path",
            str(sample_path),
            "--steps",
            str(spec.extra.get("steps", 50)),
        ]
        if spec.num_samples is not None:
            args.extend(["--num-samples", str(spec.num_samples)])
        self._run_tabsyn(args, seed=spec.seed)
        if sample_path.is_symlink() or not sample_path.is_file():
            raise FileNotFoundError(f"TabSyn did not produce the expected sample table: {sample_path}")
        if dataset_binding is not None:
            import numpy as np
            import pandas as pd

            frame = pd.read_csv(sample_path)
            dataset_spec = self.resolve_dataset_spec(spec)
            if list(frame.columns) != dataset_spec.column_names:
                raise RuntimeError("TabSyn sample columns differ from the canonical DatasetSpec order.")
            if spec.num_samples is not None and len(frame) != spec.num_samples:
                raise RuntimeError(f"TabSyn produced {len(frame)} rows; expected {spec.num_samples}.")
            if bool(frame.isna().any().any()):
                raise RuntimeError("TabSyn produced missing values.")
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            if numeric.size and not bool(np.isfinite(numeric).all()):
                raise RuntimeError("TabSyn produced non-finite numerical values.")
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        raise RuntimeError(
            "TabSyn adapter-local legacy evaluation is retired; use the central runner evaluation path."
        )
