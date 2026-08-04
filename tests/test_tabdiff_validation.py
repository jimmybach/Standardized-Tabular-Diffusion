from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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

    assert hashlib.sha256(evidence_bytes).hexdigest() == "d879512416994a60a86d3718c611aa1e1fc13d87d3b1cd71e7afdfec8ed5f234"
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
    upstream_root = tmp_path / "TabDiff-main"
    upstream_root.mkdir()
    adapter = TabDiffAdapter(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(adapter, "_run_python", lambda args, _cwd: commands.append(args))

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
        [
            "main.py",
            "--dataname",
            "toy",
            "--mode",
            "train",
            "--exp_name",
            "smoke",
            "--gpu",
            "-1",
            "--debug",
            "--no_wandb",
            "--deterministic",
        ]
    ]


def test_tabdiff_adapter_rejects_unrepresentable_seed(tmp_path: Path) -> None:
    (tmp_path / "TabDiff-main").mkdir()
    adapter = TabDiffAdapter(tmp_path)
    spec = RunSpec(
        model="tabdiff",
        dataset="toy",
        output_dir=tmp_path / "artifacts",
        seed=17,
    )

    with pytest.raises(ValueError, match="only deterministic seed 0"):
        adapter.train(spec)


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
    sample_path = (
        upstream_root
        / "eval"
        / "report_runs"
        / "parity"
        / "toy_dcr"
        / "all_samples"
        / "samples_0.csv"
    )
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text("0,1\n0.1,a\n")
    adapter = TabDiffAdapter(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(adapter, "_run_python", lambda args, _cwd: commands.append(args))

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

    assert bundle.generated_sample_path == sample_path
    assert commands == [
        [
            "main.py",
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
            "--no_wandb",
            "--deterministic",
        ]
    ]
