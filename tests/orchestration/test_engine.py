from __future__ import annotations

import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json, read_json
from standardized_tabular_diffusion.orchestration.engine import (
    OrchestrationError,
    StageSpec,
    execute_plan,
    validate_orchestration_run,
)
from standardized_tabular_diffusion.orchestration.process import ProcessOutcome

pytestmark = [pytest.mark.core, pytest.mark.evaluation]

CONFIG_FINGERPRINT = "0" * 64


def _write_stage(name: str, output: str, content: str, *, dependencies: tuple[str, ...] = (), **changes) -> StageSpec:
    code = "from pathlib import Path; import os; Path(os.environ['P6_OUTPUT']).write_text(os.environ['P6_CONTENT'])"
    values = {
        "name": name,
        "command": (sys.executable, "-c", code),
        "action": f"write deterministic {name} fixture",
        "dependencies": dependencies,
        "identity_inputs": {"content": content},
        "output_paths": (output,),
        "timeout_seconds": 10.0,
        "environment_updates": {"P6_OUTPUT": output, "P6_CONTENT": content},
    }
    values.update(changes)
    return StageSpec(**values)


def _execute(stages, root, hardware_profile, software_profile, **changes):
    for stage in stages:
        if "P6_OUTPUT" in stage.environment_updates:
            relative = stage.environment_updates["P6_OUTPUT"]
            stage.environment_updates["P6_OUTPUT"] = str(root / relative)
    values = {
        "run_root": root,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "repo_root": Path(__file__).resolve().parents[2],
        "hardware_profile": hardware_profile,
        "software_profile": software_profile,
    }
    values.update(changes)
    return execute_plan(stages, **values)


def _attempts(root: Path) -> list[dict]:
    return [read_json(path) for path in sorted((root / ".orchestration" / "attempts").glob("*.json"))]


def test_cache_hit_requires_exact_identity_and_is_not_efficiency_evidence(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    first = _execute([_write_stage("prepare", "artifact.txt", "first")], root, hardware_profile, software_profile)
    assert first["status"] == "success"
    second = _execute([_write_stage("prepare", "artifact.txt", "first")], root, hardware_profile, software_profile)
    assert second["status"] == "success"
    current = {record["stage_id"]: record for record in _attempts(root)}[second["current_stage_ids"][0]]
    assert current["cache"]["decision"] == "hit"
    assert current["cache"]["identity_verified"] is True
    assert current["cache"]["outputs_verified"] is True
    assert current["resource_usage"]["measurement_mode"] == "cache-reuse"
    assert current["resource_usage"]["efficiency_eligible"] is False
    assert validate_orchestration_run(root)["valid"] is True


def test_changed_identity_is_a_cache_miss(tmp_path: Path, hardware_profile: dict, software_profile: dict) -> None:
    root = tmp_path / "run"
    _execute([_write_stage("prepare", "artifact.txt", "first")], root, hardware_profile, software_profile)
    second = _execute([_write_stage("prepare", "artifact.txt", "second")], root, hardware_profile, software_profile)
    current = {record["stage_id"]: record for record in _attempts(root)}[second["current_stage_ids"][0]]
    assert current["cache"]["decision"] == "miss"
    assert (root / "artifact.txt").read_text() == "second"
    assert validate_orchestration_run(root)["valid"] is True


def test_stage_environment_change_is_part_of_cache_identity(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    first = _write_stage("prepare", "artifact.txt", "first", identity_inputs={"fixture": "same"})
    _execute([first], root, hardware_profile, software_profile)
    second = _write_stage("prepare", "artifact.txt", "second", identity_inputs={"fixture": "same"})
    manifest = _execute([second], root, hardware_profile, software_profile)
    current = {record["stage_id"]: record for record in _attempts(root)}[manifest["current_stage_ids"][0]]
    assert current["cache"]["decision"] == "miss"
    assert (root / "artifact.txt").read_text() == "second"


def test_corrupt_cache_is_rejected_and_fresh_output_is_preserved(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    first = _execute([_write_stage("prepare", "artifact.txt", "safe")], root, hardware_profile, software_profile)
    record = {item["stage_id"]: item for item in _attempts(root)}[first["current_stage_ids"][0]]
    digest = record["outputs"][0]["sha256"]
    blob = root / ".orchestration" / "cache" / "artifacts" / digest[:2] / digest
    blob.write_text("corrupt", encoding="utf-8")
    (root / "artifact.txt").unlink()
    second = _execute([_write_stage("prepare", "artifact.txt", "safe")], root, hardware_profile, software_profile)
    current = {item["stage_id"]: item for item in _attempts(root)}[second["current_stage_ids"][0]]
    assert current["cache"]["decision"] == "invalid"
    assert "stale_cache_rejected" in current["warning_codes"]
    assert (root / "artifact.txt").read_text() == "safe"
    assert validate_orchestration_run(root)["valid"] is True


def test_retry_preserves_failed_attempt_and_links_successful_descendant(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    marker = tmp_path / "first-attempt.marker"
    code = (
        "from pathlib import Path; import os, sys; marker=Path(os.environ['P6_MARKER']); "
        "output=Path(os.environ['P6_OUTPUT']); "
        "(marker.write_text('failed'), sys.exit(2)) if not marker.exists() else output.write_text('recovered')"
    )
    stage = StageSpec(
        name="train",
        command=(sys.executable, "-c", code),
        action="fail once then recover",
        output_paths=("model.txt",),
        identity_inputs={"fixture": "retry"},
        max_retries=1,
        environment_updates={"P6_MARKER": str(marker), "P6_OUTPUT": str(root / "model.txt")},
    )
    manifest = _execute([stage], root, hardware_profile, software_profile)
    records = [item for item in _attempts(root) if item["stage_name"] == "train"]
    assert manifest["status"] == "success"
    assert {item["status"] for item in records} == {"failed", "succeeded"}
    success = next(item for item in records if item["status"] == "succeeded")
    failure = next(item for item in records if item["status"] == "failed")
    assert failure["stage_id"] in success["resume_ancestry"]
    assert (root / "model.txt").read_text() == "recovered"


def test_optional_failure_produces_partial_run_without_erasing_success(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    good = _write_stage("evaluate.good", "good.json", "computed")
    bad = StageSpec(
        name="evaluate.optional",
        command=(sys.executable, "-c", "raise RuntimeError('optional metric failed')"),
        action="simulate optional metric backend failure",
        required=False,
        identity_inputs={"metric": "optional"},
    )
    aggregate = _write_stage(
        "aggregate",
        "aggregate.json",
        "partial-preserved",
        dependencies=("evaluate.good", "evaluate.optional"),
        allow_failed_dependencies=True,
    )
    manifest = _execute([good, bad, aggregate], root, hardware_profile, software_profile)
    assert manifest["status"] == "partial"
    assert manifest["failed_optional_stages"] == ["evaluate.optional"]
    assert (root / "good.json").read_text() == "computed"
    assert (root / "aggregate.json").read_text() == "partial-preserved"
    assert validate_orchestration_run(root)["valid"] is True


def test_interruption_cancels_current_stage_and_skips_dependents(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"

    def interrupted_runner(*args, **kwargs):
        return ProcessOutcome(
            exit_code=None,
            wall_seconds=0.01,
            cpu_seconds=0.0,
            peak_rss_bytes=1,
            peak_accelerator_memory_bytes=None,
            timed_out=False,
            memory_limit_exceeded=False,
            interrupted=True,
            launch_error=None,
            reliability=(),
            diagnostic_tail=(),
        )

    first = _write_stage("train", "model.txt", "never-written")
    second = _write_stage("sample", "sample.csv", "never-written", dependencies=("train",))
    manifest = _execute(
        [first, second], root, hardware_profile, software_profile, process_runner=interrupted_runner
    )
    current = {item["stage_id"]: item for item in _attempts(root)}
    statuses = [current[stage_id]["status"] for stage_id in manifest["current_stage_ids"]]
    assert manifest["status"] == "cancelled"
    assert statuses == ["cancelled", "skipped"]


def test_different_configuration_cannot_resume_same_output_directory(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    _execute([_write_stage("prepare", "artifact.txt", "safe")], root, hardware_profile, software_profile)
    with pytest.raises(OrchestrationError, match="different configuration identity"):
        execute_plan(
            [_write_stage("prepare", "artifact.txt", "safe")],
            run_root=root,
            config_fingerprint="f" * 64,
            repo_root=Path(__file__).resolve().parents[2],
            hardware_profile=hardware_profile,
            software_profile=software_profile,
        )


def test_secret_bearing_identity_is_redacted_and_not_cached(
    tmp_path: Path, hardware_profile: dict, software_profile: dict
) -> None:
    root = tmp_path / "run"
    stage = _write_stage("prepare", "artifact.txt", "safe", identity_inputs={"api_token": "do-not-store"})
    manifest = _execute([stage], root, hardware_profile, software_profile)
    record = {item["stage_id"]: item for item in _attempts(root)}[manifest["current_stage_ids"][0]]
    assert "do-not-store" not in str(record)
    assert record["identity_inputs"]["stage"]["api_token"] == "<redacted-secret-input>"
    assert record["cache"]["decision"] == "bypassed"
    assert "cache_bypassed_secret_input" in record["warning_codes"]


@pytest.mark.parametrize("profile_kind", ["hardware", "software"])
def test_profile_material_tampering_is_rejected(
    tmp_path: Path,
    hardware_profile: dict,
    software_profile: dict,
    profile_kind: str,
) -> None:
    root = tmp_path / "run"
    manifest = _execute(
        [_write_stage("prepare", "artifact.txt", "safe")],
        root,
        hardware_profile,
        software_profile,
    )
    relative = manifest[f"{profile_kind}_profile_ref"]
    path = root.joinpath(*relative.split("/"))
    profile = read_json(path)
    assert isinstance(profile, dict)
    if profile_kind == "hardware":
        profile["cpu"]["model"] = "Tampered CPU"
        expected = "Hardware profile comparison key"
    else:
        profile["packages"].append({"name": "tampered-package", "version": "1.0"})
        expected = "Software profile fingerprint"
    atomic_write_json(path, profile)

    with pytest.raises(OrchestrationError, match=expected):
        validate_orchestration_run(root)
