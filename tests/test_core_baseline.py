from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from standardized_tabular_diffusion import registry
from standardized_tabular_diffusion.config import (
    EvaluationConfig,
    ExperimentConfig,
    SampleConfig,
    TrainConfig,
    load_experiment_config,
)
from standardized_tabular_diffusion.model_inventory import MODEL_INVENTORY

pytestmark = pytest.mark.core

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_top_level_import_and_model_listing_are_dependency_light() -> None:
    script = """
import json
import sys
import standardized_tabular_diffusion as package

payload = {
    "models": package.list_models(),
    "unexpected": sorted({"numpy", "pandas", "sklearn", "torch"} & set(sys.modules)),
}
print(json.dumps(payload))
"""
    completed = _run_isolated(script)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "tabdiff" in payload["models"]
    assert payload["unexpected"] == []


def test_models_package_does_not_eagerly_import_adapter_modules() -> None:
    script = """
import json
import sys
import standardized_tabular_diffusion.models

loaded = sorted(
    name for name in sys.modules
    if name.startswith("standardized_tabular_diffusion.models.")
)
print(json.dumps(loaded))
"""
    completed = _run_isolated(script)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_cli_help_is_available_without_data_or_model_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "standardized_tabular_diffusion.cli", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "list-models" in completed.stdout
    assert "run-action" in completed.stdout


def test_missing_adapter_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing_dependency(module_name: str):
        raise ModuleNotFoundError(
            f"No module named 'optional_runtime' while importing {module_name}", name="optional_runtime"
        )

    monkeypatch.setattr(registry, "import_module", raise_missing_dependency)

    with pytest.raises(registry.AdapterDependencyError) as exc_info:
        registry.get_adapter("arf", repo_root=REPO_ROOT)

    message = str(exc_info.value)
    assert "arf" in message
    assert "optional_runtime" in message
    assert "standardized-tabular-diffusion[models]" in message


def test_source_backed_adapter_requires_complete_repository_checkout(tmp_path: Path) -> None:
    with pytest.raises(registry.AdapterSourceUnavailableError, match="lightweight wheel"):
        registry.get_adapter("tabdiff", repo_root=tmp_path)


def test_pytest_discovery_excludes_upstream_and_research_inputs() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = payload["tool"]["pytest"]["ini_options"]

    assert pytest_options["testpaths"] == ["tests"]
    excluded = set(pytest_options["norecursedirs"])
    assert {"research_inputs", "TabDDPM-main", "TabDiff-main", "TabSyn-main"} <= excluded


def test_adapter_registry_reports_conservative_independent_status_dimensions() -> None:
    records = registry.list_adapter_specs()

    assert set(records) == set(registry.list_models())
    assert records["tabdiff"]["validation_level"] == "native-parity-validated"
    assert all(
        record["validation_level"] == "adapter-complete"
        for name, record in records.items()
        if name != "tabdiff"
    )
    assert all(record["benchmark_track"] == "experimental" for record in records.values())
    assert all(record["support_level"] == "unsupported" for record in records.values())
    assert all(MODEL_INVENTORY[name].validation_level == record["validation_level"] for name, record in records.items())


def test_run_spec_keeps_action_extra_namespaces_isolated() -> None:
    config = ExperimentConfig(
        model="tabsyn",
        dataset="adult",
        output_dir="artifacts/test",
        train=TrainConfig(extra={"epochs": 10}),
        sample=SampleConfig(extra={"epochs": 20}),
        evaluation=EvaluationConfig(enabled=False),
    )

    generic_spec = config.to_run_spec()
    train_spec = config.to_run_spec(action="train")
    sample_spec = config.to_run_spec(action="sample")

    assert "epochs" not in generic_spec.extra
    assert train_spec.extra["epochs"] == 10
    assert sample_spec.extra["epochs"] == 20
    assert generic_spec.extra["action_extras"]["train"]["epochs"] == 10


def test_config_loader_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "tabsyn",
                "dataset": "adult",
                "output_dir": "artifacts/test",
                "typo_output": "ignored-before-fix",
            }
        )
    )

    with pytest.raises(ValueError, match="typo_output"):
        load_experiment_config(config_path)
