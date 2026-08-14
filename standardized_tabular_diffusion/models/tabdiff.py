from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import tomllib
from pathlib import Path

from standardized_tabular_diffusion.compat.tabdiff_seed_launcher import (
    RUNTIME_COMPATIBILITY_RECORD_PATH,
    load_config_path_overlay_record,
    load_diagnostic_plot_bypass_record,
    load_patch_record,
)
from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, atomic_write_json
from standardized_tabular_diffusion.interfaces import ArtifactBundle, RunSpec
from standardized_tabular_diffusion.models.base import BaseModelAdapter
from standardized_tabular_diffusion.runtime_contracts import materialize_bound_dataset_view


class TabDiffAdapter(BaseModelAdapter):
    model_name = "tabdiff"
    upstream_dirname = "TabDiff-main"

    @staticmethod
    def _runtime_root(spec: RunSpec) -> Path:
        return spec.output_dir / "tabdiff-runtime"

    def _prepare_runtime(self, spec: RunSpec) -> dict[str, object]:
        runtime_root = self._runtime_root(spec)
        if runtime_root.is_symlink():
            raise ValueError(f"TabDiff runtime root must not be a symlink: {runtime_root}")
        runtime_root.mkdir(parents=True, exist_ok=True)
        if "dataset_identity" in spec.extra:
            dataset_spec = self.resolve_dataset_spec(spec)
            binding: dict[str, object] = materialize_bound_dataset_view(
                dataset_spec,
                self.upstream_root / "data" / spec.dataset,
                runtime_root / "data" / spec.dataset,
            )
        else:
            binding = {"schema_version": 1, "status": "not-declared-at-direct-runspec-boundary"}
        source_config = self.upstream_root / "tabdiff" / "configs" / "tabdiff_configs.toml"
        runtime_config = runtime_root / "tabdiff" / "configs" / "tabdiff_configs.toml"
        if source_config.is_symlink():
            raise FileNotFoundError(f"TabDiff official default configuration is missing: {source_config}")
        if not source_config.is_file() and "dataset_identity" in spec.extra:
            raise FileNotFoundError(f"TabDiff official default configuration is missing: {source_config}")
        if runtime_config.exists() and source_config.is_file():
            if runtime_config.is_symlink() or not runtime_config.is_file():
                raise ValueError(f"TabDiff runtime configuration path is unsafe: {runtime_config}")
            if self._sha256_file(runtime_config) != self._sha256_file(source_config):
                raise FileExistsError("TabDiff runtime default configuration differs from official source.")
        elif source_config.is_file():
            atomic_write_bytes(runtime_config, source_config.read_bytes())
        return binding

    @staticmethod
    def _pytorch_runtime_metadata() -> dict[str, object]:
        if importlib.util.find_spec("torch") is None:
            return {
                "inspection_state": "unavailable-in-parent-environment",
                "bridge_active": None,
                "torch_version": None,
            }

        import torch

        scheduler_accepts_verbose = (
            "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters
        )
        return {
            "inspection_state": "observed",
            "bridge_active": not scheduler_accepts_verbose,
            "torch_version": torch.__version__,
        }

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
    def _validate_seed_contract(spec: RunSpec) -> None:
        deterministic = bool(spec.extra.get("deterministic", True))
        if isinstance(spec.seed, bool) or not isinstance(spec.seed, int) or spec.seed < 0:
            raise ValueError("TabDiff seed must be a non-negative integer.")
        if not deterministic:
            raise ValueError("TabDiff configurable seed execution requires deterministic=true.")

    def _execution_environment(self, spec: RunSpec) -> dict[str, str]:
        package_import_root = Path(__file__).resolve().parents[2]
        python_paths = list(dict.fromkeys([str(package_import_root), str(self.repo_root.resolve())]))
        if existing_python_path := os.environ.get("PYTHONPATH"):
            python_paths.append(existing_python_path)
        return {
            "PYTHONHASHSEED": str(spec.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(python_paths),
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _resolve_config_path(spec: RunSpec) -> Path | None:
        if spec.upstream_config_path is None:
            return None
        path = spec.upstream_config_path
        if path.is_symlink():
            raise PermissionError(f"Refusing to load a symlinked TabDiff TOML configuration: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.suffix.lower() != ".toml":
            raise ValueError(f"TabDiff upstream_config_path must be a regular TOML file: {resolved}")
        with resolved.open("rb") as stream:
            payload = tomllib.load(stream)
        required_sections = {"data", "unimodmlp_params", "diffusion_params", "train", "sample"}
        missing = sorted(required_sections - payload.keys())
        if missing:
            raise ValueError(f"TabDiff TOML is missing required sections: {missing}")
        return resolved

    def _validate_integer_restoration_config(self, spec: RunSpec, config_path: Path | None) -> None:
        info_path = self.upstream_root / "data" / spec.dataset / "info.json"
        if not info_path.is_file():
            return
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not info.get("int_col_idx"):
            return
        selected_config = config_path or self.upstream_root / "tabdiff" / "configs" / "tabdiff_configs.toml"
        with selected_config.open("rb") as stream:
            config = tomllib.load(stream)
        restoration = config.get("data", {}).get("dequant_dist")
        if restoration == "none" and not bool(spec.extra.get("allow_unstandardized_integer_output", False)):
            raise ValueError(
                "TabDiff dequant_dist='none' does not restore integer-valued columns during inverse transformation. "
                "Use the official 'round' mode for standardized execution, or explicitly set "
                "allow_unstandardized_integer_output=true only for native-parity diagnostics."
            )

    def _validate_generated_sample_contract(self, spec: RunSpec, sample_path: Path) -> dict[str, object]:
        info_path = self.upstream_root / "data" / spec.dataset / "info.json"
        if not info_path.is_file():
            return {"status": "not-evaluated", "reason": "upstream dataset info.json is unavailable"}

        import numpy as np
        import pandas as pd

        info = json.loads(info_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(sample_path)
        expected_columns = info["column_names"]
        if list(frame.columns) != expected_columns:
            raise ValueError("TabDiff generated columns differ from the upstream dataset schema.")
        if frame.isna().any().any():
            raise ValueError("TabDiff generated sample contains missing values.")
        non_integral: dict[str, int] = {}
        for index in info.get("int_col_idx", []):
            column = expected_columns[index]
            numeric = pd.to_numeric(frame[column], errors="coerce")
            values = numeric.to_numpy(dtype=float)
            if numeric.isna().any() or not np.isfinite(values).all():
                raise ValueError(f"TabDiff generated integer column {column!r} contains invalid values.")
            count = int((values != np.rint(values)).sum())
            if count:
                non_integral[column] = count
        if non_integral:
            if bool(spec.extra.get("allow_unstandardized_integer_output", False)):
                return {
                    "status": "diagnostic-bypass",
                    "info_path": str(info_path),
                    "info_sha256": self._sha256_file(info_path),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "missing_values": 0,
                    "integer_columns_integral": False,
                    "non_integral_cells": non_integral,
                }
            raise ValueError(
                "TabDiff generated sample violates the standardized integer contract: "
                f"{non_integral}. Use an upstream config with data.dequant_dist='round'."
            )
        return {
            "status": "passed",
            "info_path": str(info_path),
            "info_sha256": self._sha256_file(info_path),
            "rows": len(frame),
            "columns": len(frame.columns),
            "missing_values": 0,
            "integer_columns_integral": True,
        }

    def _write_run_metadata(
        self,
        spec: RunSpec,
        *,
        action: str,
        generated_sample_path: Path | None = None,
        sample_contract: dict[str, object] | None = None,
        dataset_binding: dict[str, object] | None = None,
    ) -> Path:
        patch = load_patch_record()
        config_path_overlay = load_config_path_overlay_record()
        diagnostic_plot_bypass = load_diagnostic_plot_bypass_record()
        runtime_compatibility = json.loads(RUNTIME_COMPATIBILITY_RECORD_PATH.read_text(encoding="utf-8"))
        pytorch_runtime = self._pytorch_runtime_metadata()
        payload: dict[str, object] = {
            "schema_version": 1,
            "model": self.model_name,
            "dataset": spec.dataset,
            "action": action,
            "seed": spec.seed,
            "deterministic": True,
            "device": spec.device,
            "seed_interface": {
                "kind": "approved-runtime-overlay",
                "patch_id": patch["patch_id"],
                "scope": patch["scope"],
                "upstream_commit": patch["upstream_commit"],
                "source_path": patch["source_path"],
                "source_sha256_lf": patch["source_sha256_lf"],
                "patched_sha256_lf": patch["patched_sha256_lf"],
            },
            "config_interface": {
                "patch_id": config_path_overlay["patch_id"],
                "scope": config_path_overlay["scope"],
                "custom_config_active": spec.upstream_config_path is not None,
                "path": None,
                "sha256": None,
            },
            "runtime_compatibility": {
                "bridge_id": runtime_compatibility["bridge_id"],
                **pytorch_runtime,
                "scientific_effect": runtime_compatibility["scientific_effect"],
                "diagnostic_plot_bypass": {
                    "patch_id": diagnostic_plot_bypass["patch_id"],
                    "active": True,
                    "scope": diagnostic_plot_bypass["scope"],
                    "source_sha256_lf": diagnostic_plot_bypass["source_sha256_lf"],
                    "patched_sha256_lf": diagnostic_plot_bypass["patched_sha256_lf"],
                    "scientific_effect": diagnostic_plot_bypass["scientific_effect"],
                },
            },
            "generated_sample": None,
            "sample_contract": sample_contract,
            "dataset_binding": dataset_binding,
        }
        if spec.upstream_config_path is not None:
            config_path = self._resolve_config_path(spec)
            if config_path is None:  # pragma: no cover - guarded by the condition above
                raise AssertionError("TabDiff config resolution unexpectedly returned None.")
            config_interface = payload["config_interface"]
            if not isinstance(config_interface, dict):  # pragma: no cover - construction invariant
                raise AssertionError("TabDiff config metadata has an unexpected representation.")
            config_interface["path"] = str(config_path)
            config_interface["sha256"] = self._sha256_file(config_path)
        if generated_sample_path is not None:
            payload["generated_sample"] = {
                "path": str(generated_sample_path),
                "sha256": self._sha256_file(generated_sample_path),
            }
        path = spec.output_dir / "tabdiff_run.json"
        atomic_write_json(path, payload)
        return path

    def _common_args(self, spec: RunSpec) -> list[str]:
        self._validate_seed_contract(spec)
        args = ["--gpu", str(self._gpu_index(spec)), "--seed", str(spec.seed)]
        if spec.extra.get("debug"):
            args.append("--debug")
        if spec.extra.get("no_wandb", True):
            args.append("--no_wandb")
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
        ckpt_parent = self._runtime_root(spec) / "tabdiff" / "ckpt" / spec.dataset / exp_name
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
        return self._runtime_root(spec) / "eval" / "report_runs" / exp_name / spec.dataset / "all_samples" / "samples_0.csv"

    def train(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_binding = self._prepare_runtime(spec)
        config_path = self._resolve_config_path(spec)
        self._validate_integer_restoration_config(spec, config_path)
        args = [
            "--runtime-root",
            str(self._runtime_root(spec).resolve()),
            "--dataname",
            spec.dataset,
            "--mode",
            "train",
            "--exp_name",
            spec.extra.get("exp_name", spec.output_dir.name),
        ]
        if config_path:
            args.extend(["--config_path", str(config_path)])
        args.extend(self._common_args(spec))
        self._run_python(
            ["standardized_tabular_diffusion.compat.tabdiff_seed_launcher", *args],
            self.upstream_root,
            module=True,
            env=self._execution_environment(spec),
        )
        self._write_run_metadata(spec, action="train", dataset_binding=dataset_binding)
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            notes=[
                "Training artifacts are written under the declared run-owned TabDiff runtime directory.",
                f"Configurable seed supplied by approved overlay {load_patch_record()['patch_id']}.",
                "Optional upstream density-plot PNG rendering is disabled; serialized metrics remain enabled.",
            ],
        )
        return self._write_bundle(bundle)

    def sample(self, spec: RunSpec) -> ArtifactBundle:
        self._ensure_output_dir(spec)
        dataset_binding = self._prepare_runtime(spec)
        config_path = self._resolve_config_path(spec)
        self._validate_integer_restoration_config(spec, config_path)
        checkpoint_path = self._resolve_checkpoint_path(spec)
        exp_name = spec.extra.get("exp_name", spec.output_dir.name)
        args = [
            "--runtime-root",
            str(self._runtime_root(spec).resolve()),
            "--dataname",
            spec.dataset,
            "--mode",
            "test",
            "--exp_name",
            exp_name,
            "--ckpt_path",
            str(checkpoint_path),
        ]
        if config_path:
            args.extend(["--config_path", str(config_path)])
        if spec.num_samples is not None:
            args.extend(["--num_samples_to_generate", str(spec.num_samples)])
        report = bool(spec.extra.get("report", False))
        if report:
            args.extend(["--report", "--num_runs", str(int(spec.extra.get("num_runs", 1)))])
        args.extend(self._common_args(spec))
        self._run_python(
            ["standardized_tabular_diffusion.compat.tabdiff_seed_launcher", *args],
            self.upstream_root,
            module=True,
            env=self._execution_environment(spec),
        )
        upstream_sample_path = (
            self._infer_report_sample_path(spec, exp_name) if report else self._infer_sample_path(checkpoint_path)
        )
        if not upstream_sample_path.is_file():
            raise FileNotFoundError(f"TabDiff did not produce the expected sample table: {upstream_sample_path}")
        sample_contract = self._validate_generated_sample_contract(spec, upstream_sample_path)
        sample_path = spec.output_dir / "samples.csv"
        atomic_write_bytes(sample_path, upstream_sample_path.read_bytes())
        self._write_run_metadata(
            spec,
            action="sample",
            generated_sample_path=sample_path,
            sample_contract=sample_contract,
            dataset_binding=dataset_binding,
        )
        bundle = ArtifactBundle(
            model=self.model_name,
            dataset=spec.dataset,
            output_dir=spec.output_dir,
            upstream_workdir=self.upstream_root,
            generated_sample_path=sample_path,
            notes=[
                "TabDiff report mode generated samples and evaluation outputs together."
                if report
                else "TabDiff test mode generates samples and evaluation outputs together.",
                f"Configurable seed supplied by approved overlay {load_patch_record()['patch_id']}.",
                "Optional upstream density-plot PNG rendering is disabled; serialized metrics remain enabled.",
                f"The seed-specific sample was copied from {upstream_sample_path} into the run output directory.",
            ],
        )
        return self._write_bundle(bundle)

    def evaluate(self, spec: RunSpec) -> ArtifactBundle:
        raise RuntimeError(
            "TabDiff adapter-local legacy evaluation is retired; use the central runner evaluation path."
        )
