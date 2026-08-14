from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from standardized_tabular_diffusion.compat.tabdiff_seed_launcher import (
    TabDiffSeedOverlayError,
    apply_config_path_overlay,
    apply_diagnostic_plot_bypass,
    apply_seed_overlay,
    install_pytorch_compatibility,
    load_config_path_overlay_record,
    load_diagnostic_plot_bypass_record,
    load_patch_record,
)
from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.tabdiff import TabDiffAdapter
from standardized_tabular_diffusion.validation.tabdiff import MANIFEST_RELATIVE_PATH, verify_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabdiff" / "native-parity-run-30866879879.json"


def test_tabdiff_source_manifest_matches_pinned_sources() -> None:
    result = verify_sources(REPO_ROOT)

    assert result["upstream_commit"] == "5ecdb3356261aea72716cc9a779f31d7ad083bf4"
    assert result["upstream_tree"] == "052a505cb1fbee5cbc705eeb0717d90d706ffb91"
    assert result["upstream_files_verified"] == 27


def test_tabdiff_source_manifest_has_unique_complete_paths() -> None:
    payload = json.loads((REPO_ROOT / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    paths = [record["path"] for record in payload["files"]]

    assert len(paths) == len(set(paths)) == 27
    assert all(len(record["sha256_lf"]) == 64 for record in payload["files"])
    assert "eval/mle/mle.py" in paths


def test_tabdiff_native_parity_evidence_is_complete_and_immutable() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert (
        hashlib.sha256(evidence_bytes).hexdigest() == "d879512416994a60a86d3718c611aa1e1fc13d87d3b1cd71e7afdfec8ed5f234"
    )
    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "230adafe96dc7ec224bada220e1ee184972b61ad"
    comparisons = evidence["comparisons"]
    assert comparisons["config_exact"] is True
    assert comparisons["checkpoint"]["tensor_values_exact"] is True
    assert comparisons["training_samples"]["exact_bytes"] is True
    assert comparisons["generated_samples"]["exact_bytes"] is True
    assert comparisons["generated_samples"]["rows"] == 12
    assert comparisons["training_metrics_exact"] is True
    assert comparisons["generated_metrics_exact"] is True
    assert comparisons["adapter_manifests_valid"] is True


def test_tabdiff_adapter_maps_cpu_and_official_deterministic_seed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    upstream_root = tmp_path / "TabDiff-main"
    upstream_root.mkdir()
    adapter = TabDiffAdapter(tmp_path)
    commands: list[tuple[list[str], bool, dict[str, str] | None]] = []

    def fake_run(args, _cwd, *, module=False, env=None):
        commands.append((args, module, env))

    monkeypatch.setattr(adapter, "_run_python", fake_run)

    adapter.train(
        RunSpec(
            model="tabdiff",
            dataset="toy",
            output_dir=tmp_path / "artifacts",
            device="cpu",
            seed=0,
            extra={"debug": True, "exp_name": "smoke"},
        )
    )

    assert commands == [
        (
            [
                "standardized_tabular_diffusion.compat.tabdiff_seed_launcher",
                "--dataname",
                "toy",
                "--mode",
                "train",
                "--exp_name",
                "smoke",
                "--gpu",
                "-1",
                "--seed",
                "0",
                "--debug",
                "--no_wandb",
                "--deterministic",
            ],
            True,
            {
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(tmp_path)]),
            },
        )
    ]


def test_tabdiff_adapter_forwards_nonzero_seed(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "TabDiff-main").mkdir()
    adapter = TabDiffAdapter(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        adapter,
        "_run_python",
        lambda args, _cwd, *, module=False, env=None: commands.append(args),
    )
    spec = RunSpec(
        model="tabdiff",
        dataset="toy",
        output_dir=tmp_path / "artifacts",
        seed=17,
    )

    adapter.train(spec)

    assert "--seed" in commands[0]
    assert commands[0][commands[0].index("--seed") + 1] == "17"
    assert json.loads((spec.output_dir / "tabdiff_run.json").read_text())["seed"] == 17


@pytest.mark.parametrize("seed", [-1, True])
def test_tabdiff_adapter_rejects_invalid_seed(tmp_path: Path, seed: object) -> None:
    (tmp_path / "TabDiff-main").mkdir()
    adapter = TabDiffAdapter(tmp_path)
    with pytest.raises(ValueError, match="non-negative integer"):
        adapter.train(
            RunSpec(model="tabdiff", dataset="toy", output_dir=tmp_path / "artifacts", seed=seed)  # type: ignore[arg-type]
        )


def test_tabdiff_seed_overlay_is_exact_and_fails_closed() -> None:
    record = load_patch_record()
    source = (REPO_ROOT / "TabDiff-main" / record["source_path"]).read_text(encoding="utf-8")
    patched = apply_seed_overlay(source, record)

    assert patched != source
    assert "torch.manual_seed(args.seed)" in patched
    assert "np.random.seed(args.seed)" in patched
    with pytest.raises(TabDiffSeedOverlayError, match="anchor mismatch"):
        apply_seed_overlay(source.replace("torch.manual_seed(0)", "torch.manual_seed(1)"), record)


def test_tabdiff_diagnostic_plot_bypass_is_exact_and_fails_closed() -> None:
    record = load_diagnostic_plot_bypass_record()
    source = (REPO_ROOT / "TabDiff-main" / record["source_path"]).read_text(encoding="utf-8")
    patched = apply_diagnostic_plot_bypass(source, record)

    assert patched.count("plot_density=True") == 0
    assert patched.count("plot_density=False") == source.count("plot_density=False") + 3
    with pytest.raises(TabDiffSeedOverlayError, match="anchor mismatch"):
        apply_diagnostic_plot_bypass(source.replace("plot_density=True", "plot_density=False", 1), record)


def test_tabdiff_config_path_overlay_is_exact_and_fails_closed() -> None:
    seed_record = load_patch_record()
    record = load_config_path_overlay_record()
    source = (REPO_ROOT / "TabDiff-main" / seed_record["source_path"]).read_text(encoding="utf-8")
    seed_patched = apply_seed_overlay(source, seed_record)
    patched = apply_config_path_overlay(seed_patched, record)

    assert "args.config_path or f'{curr_dir}/configs/tabdiff_configs.toml'" in patched
    with pytest.raises(TabDiffSeedOverlayError, match="anchor mismatch"):
        apply_config_path_overlay(patched, record)


def test_tabdiff_adapter_forwards_validated_toml_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    (tmp_path / "TabDiff-main").mkdir()
    config_path = tmp_path / "tabdiff.toml"
    config_path.write_text(
        "[data]\n[unimodmlp_params]\n[diffusion_params]\n[train]\n[sample]\n",
        encoding="utf-8",
    )
    adapter = TabDiffAdapter(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        adapter,
        "_run_python",
        lambda args, _cwd, *, module=False, env=None: commands.append(args),
    )

    adapter.train(
        RunSpec(
            model="tabdiff",
            dataset="toy",
            output_dir=tmp_path / "artifacts",
            upstream_config_path=config_path,
        )
    )

    assert commands[0][commands[0].index("--config_path") + 1] == str(config_path.resolve())
    metadata = json.loads((tmp_path / "artifacts" / "tabdiff_run.json").read_text(encoding="utf-8"))
    assert metadata["config_interface"]["custom_config_active"] is True
    assert metadata["config_interface"]["path"] == str(config_path.resolve())


def test_tabdiff_new_pytorch_scheduler_bridge_discards_only_verbose() -> None:
    class NewScheduler:
        def __init__(self, optimizer: object, *, factor: float = 0.1) -> None:
            self.optimizer = optimizer
            self.factor = factor

    scheduler_module = SimpleNamespace(ReduceLROnPlateau=NewScheduler)
    torch_module = SimpleNamespace(optim=SimpleNamespace(lr_scheduler=scheduler_module))
    optimizer = object()

    active = install_pytorch_compatibility(torch_module)
    scheduler = scheduler_module.ReduceLROnPlateau(optimizer, factor=0.9, verbose=True)

    assert active == ["tabdiff-pytorch-reduce-lr-verbose-bridge-v1"]
    assert scheduler.optimizer is optimizer
    assert scheduler.factor == 0.9
    assert install_pytorch_compatibility(torch_module) == []


def test_tabdiff_adapter_rejects_untrusted_explicit_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "TabDiff-main").mkdir()
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"not loaded")
    adapter = TabDiffAdapter(tmp_path)
    spec = RunSpec(
        model="tabdiff",
        dataset="toy",
        output_dir=tmp_path / "artifacts",
        checkpoint_path=checkpoint,
    )

    with pytest.raises(PermissionError, match="can execute code"):
        adapter.sample(spec)


def test_tabdiff_adapter_maps_official_report_output(tmp_path: Path, monkeypatch) -> None:
    upstream_root = tmp_path / "TabDiff-main"
    output_dir = tmp_path / "artifacts"
    checkpoint = output_dir / "model_4.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not loaded by mocked command")
    sample_path = upstream_root / "eval" / "report_runs" / "parity" / "toy_dcr" / "all_samples" / "samples_0.csv"
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text("0,1\n0.1,a\n")
    adapter = TabDiffAdapter(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        adapter,
        "_run_python",
        lambda args, _cwd, *, module=False, env=None: commands.append(args),
    )

    bundle = adapter.sample(
        RunSpec(
            model="tabdiff",
            dataset="toy_dcr",
            output_dir=output_dir,
            device="cpu",
            checkpoint_path=checkpoint,
            num_samples=12,
            extra={"exp_name": "parity", "report": True, "num_runs": 1},
        )
    )

    assert bundle.generated_sample_path == output_dir / "samples.csv"
    assert bundle.generated_sample_path.read_bytes() == sample_path.read_bytes()
    assert commands == [
        [
            "standardized_tabular_diffusion.compat.tabdiff_seed_launcher",
            "--dataname",
            "toy_dcr",
            "--mode",
            "test",
            "--exp_name",
            "parity",
            "--ckpt_path",
            str(checkpoint.resolve()),
            "--num_samples_to_generate",
            "12",
            "--report",
            "--num_runs",
            "1",
            "--gpu",
            "-1",
            "--seed",
            "0",
            "--no_wandb",
            "--deterministic",
        ]
    ]
