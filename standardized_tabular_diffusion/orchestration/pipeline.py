"""Build the seven-stage P6 plan around existing model adapters."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.config import ExperimentConfig, load_experiment_config
from standardized_tabular_diffusion.datasets import get_dataset_spec
from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, read_json, sha256_file
from standardized_tabular_diffusion.orchestration.engine import StageSpec, execute_plan
from standardized_tabular_diffusion.orchestration.hardware import capture_hardware_profile

_SECRET_KEY = re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret|credential)")


def _contains_secret(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_SECRET_KEY.search(str(key)) or _contains_secret(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item) for item in value)
    return False


def _redact_secret_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted-secret-input>" if _SECRET_KEY.search(str(key)) else _redact_secret_values(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secret_values(item) for item in value]
    return value


def _file_fingerprint(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return sha256_file(candidate)


def _configuration_identity(config: ExperimentConfig) -> tuple[str, bool]:
    payload = copy.deepcopy(config.to_dict())
    payload.pop("output_dir", None)
    payload.pop("tags", None)
    payload.pop("notes", None)
    upstream = payload.pop("upstream_config_path", None)
    payload["upstream_config"] = {
        "present": upstream is not None,
        "sha256": _file_fingerprint(upstream),
    }
    sample = payload.get("sample", {})
    if isinstance(sample, dict):
        checkpoint = sample.pop("checkpoint_path", None)
        sample["checkpoint"] = {
            "present": checkpoint is not None,
            "sha256": _file_fingerprint(checkpoint),
        }
    for section_name in ("sample", "evaluation"):
        section = payload.get(section_name, {})
        extra = section.get("extra", {}) if isinstance(section, dict) else {}
        if isinstance(extra, dict):
            external_sample = extra.pop("sample_path", None)
            extra["external_sample"] = {
                "present": external_sample is not None,
                "sha256": _file_fingerprint(external_sample),
            }
    contains_secret = _contains_secret(payload)
    if contains_secret:
        raise ValueError(
            "Secrets are prohibited in experiment configuration; use an environment-backed credential reference"
        )
    return content_fingerprint(_redact_secret_values(payload)), contains_secret


def _input_fingerprints(
    config: ExperimentConfig,
    repo_root: Path,
    config_fingerprint: str,
) -> dict[str, str]:
    fingerprints: dict[str, str] = {"experiment-config-material": config_fingerprint}
    dataset = get_dataset_spec(config.dataset, repo_root=repo_root)
    external_sample = config.evaluation.extra.get("sample_path") or config.sample.extra.get("sample_path")
    for name, path in (
        ("dataset-metadata", dataset.metadata_path),
        ("dataset-train", dataset.train_data_path),
        ("dataset-validation", dataset.val_data_path),
        ("dataset-test", dataset.test_data_path),
        ("upstream-config", None if config.upstream_config_path is None else Path(config.upstream_config_path)),
        (
            "input-checkpoint",
            None if config.sample.checkpoint_path is None else Path(config.sample.checkpoint_path),
        ),
        ("external-sample", None if not isinstance(external_sample, str) else Path(external_sample)),
        ("source-lock", repo_root / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"),
    ):
        digest = _file_fingerprint(path)
        if digest is not None:
            fingerprints[name] = digest
    return fingerprints


def _worker_command(operation: str, config_path: Path, run_root: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "standardized_tabular_diffusion.orchestration.worker",
        "--operation",
        operation,
        "--config",
        str(config_path),
        "--run-root",
        str(run_root),
    )


def build_benchmark_plan(
    config: ExperimentConfig,
    *,
    config_path: Path,
    repo_root: Path,
    timeout_seconds: float | None,
    memory_limit_bytes: int | None,
    max_retries: int,
) -> tuple[list[StageSpec], str]:
    config_fingerprint, contains_secret = _configuration_identity(config)
    fingerprints = _input_fingerprints(config, repo_root, config_fingerprint)
    root = Path(config.output_dir).resolve()
    external_sample = config.evaluation.extra.get("sample_path") or config.sample.extra.get("sample_path")
    validate_enabled = config.sample.enabled or isinstance(external_sample, str)
    cacheable = not contains_secret
    common: dict[str, Any] = {
        "input_fingerprints": fingerprints,
        "cacheable": cacheable,
        "memory_limit_bytes": memory_limit_bytes,
    }

    def stage(
        name: str,
        *,
        dependencies: tuple[str, ...],
        enabled: bool = True,
        required: bool = True,
        allow_failed_dependencies: bool = False,
        output: str,
        retries: int = 0,
        warmup_policy: str = "not-applicable",
        uses_accelerator: bool = False,
        requested_rows: int | None = None,
        stage_timeout: float | None = timeout_seconds,
    ) -> StageSpec:
        return StageSpec(
            name=name,
            command=_worker_command(name, config_path, root),
            action={
                "prepare": "resolve configuration, adapter, dataset, and immutable inputs",
                "train": "execute the selected official adapter training action",
                "sample": "generate the requested decoded synthetic table",
                "validate": "verify the generated artifact without scientific repair",
                "evaluate": "execute the configured adapter evaluation action",
                "aggregate": "index completed operational stage artifacts without leaderboard aggregation",
                "report": "emit an operational report without recomputing scientific values",
            }[name],
            dependencies=dependencies,
            required=required,
            enabled=enabled,
            allow_failed_dependencies=allow_failed_dependencies,
            identity_inputs={
                "operation": name,
                "configuration_fingerprint": config_fingerprint,
                "model": config.model,
                "dataset": config.dataset,
                "embedded_sensitive_fields_present": contains_secret,
            },
            output_paths=(output,),
            timeout_seconds=stage_timeout,
            requested_rows=requested_rows,
            warmup_policy=warmup_policy,
            uses_accelerator=uses_accelerator,
            max_retries=retries,
            **common,
        )

    uses_accelerator = "cuda" in config.train.device.casefold()
    evaluate_timeout = timeout_seconds or float(config.evaluation.total_time_limit_seconds)
    plan = [
        stage(
            "prepare",
            dependencies=(),
            output="artifacts/execution/run_context.json",
            stage_timeout=min(timeout_seconds, 300.0) if timeout_seconds is not None else 300.0,
        ),
        stage(
            "train",
            dependencies=("prepare",),
            enabled=config.train.enabled,
            output="artifacts/execution/train.json",
            retries=max_retries,
            warmup_policy="none",
            uses_accelerator=uses_accelerator,
            stage_timeout=timeout_seconds,
        ),
        stage(
            "sample",
            dependencies=("train",),
            enabled=config.sample.enabled,
            output="artifacts/execution/sample.json",
            retries=max_retries,
            warmup_policy="cold-single-run-diagnostic",
            uses_accelerator=uses_accelerator,
            requested_rows=config.sample.num_samples,
            stage_timeout=timeout_seconds,
        ),
        stage(
            "validate",
            dependencies=("sample",),
            enabled=validate_enabled,
            output="artifacts/execution/validation.json",
            requested_rows=config.sample.num_samples,
            stage_timeout=min(timeout_seconds, 300.0) if timeout_seconds is not None else 300.0,
        ),
        stage(
            "evaluate",
            dependencies=("validate",),
            enabled=config.evaluation.enabled,
            output="artifacts/execution/evaluate.json",
            retries=max_retries,
            uses_accelerator=uses_accelerator,
            stage_timeout=evaluate_timeout,
        ),
        stage(
            "aggregate",
            dependencies=("prepare", "train", "sample", "validate", "evaluate"),
            allow_failed_dependencies=True,
            output="pipeline_result.json",
            stage_timeout=min(timeout_seconds, 300.0) if timeout_seconds is not None else 300.0,
        ),
        stage(
            "report",
            dependencies=("aggregate",),
            allow_failed_dependencies=True,
            output="benchmark_report.json",
            stage_timeout=min(timeout_seconds, 300.0) if timeout_seconds is not None else 300.0,
        ),
    ]
    return plan, config_fingerprint


def run_benchmark_pipeline(
    config_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    resume: bool = True,
    timeout_seconds: float | None = None,
    memory_limit_bytes: int | None = None,
    max_retries: int = 0,
    hardware_profile_id: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(config_path).resolve()
    config = load_experiment_config(source)
    repository = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    plan, config_fingerprint = build_benchmark_plan(
        config,
        config_path=source,
        repo_root=repository,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        max_retries=max_retries,
    )
    profile = capture_hardware_profile(profile_id=hardware_profile_id)
    return execute_plan(
        plan,
        run_root=config.output_dir,
        config_fingerprint=config_fingerprint,
        repo_root=repository,
        cache_dir=cache_dir,
        use_cache=use_cache,
        resume=resume,
        hardware_profile=profile,
    )


def read_config_output_dir(config_path: str | Path) -> Path:
    payload = read_json(config_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("output_dir"), str):
        raise ValueError("Experiment configuration must declare output_dir")
    return Path(payload["output_dir"])
