from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import standardized_tabular_diffusion.validation.tabsyn as tabsyn_validation
from standardized_tabular_diffusion.interfaces import RunSpec
from standardized_tabular_diffusion.models.tabsyn import TabSynAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabsyn" / "native-parity-run-30871758645.json"


def test_tabsyn_scoped_sources_match_frozen_official_manifest() -> None:
    evidence = tabsyn_validation.verify_sources(REPO_ROOT)
    manifest = json.loads(
        (REPO_ROOT / tabsyn_validation.MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert evidence["upstream_commit"] == "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"
    assert evidence["upstream_files_verified"] == len(manifest["files"]) == 20
    assert not list((REPO_ROOT / "TabSyn-main" / "zero").glob("*.py"))


def test_tabsyn_retained_native_parity_evidence_is_exact_and_complete() -> None:
    assert hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest() == (
        "3b74600a9c6d5e4e841cf56bd128ac7d17b70a6d186b48a3de78d8ca476d8089"
    )
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "54d419642842d7146d6afa4aa1b3d5167301c51c"
    assert evidence["seed_cases"] == [0, 19, 73]
    assert [case["status"] for case in evidence["cases"]] == ["pass", "pass", "pass"]
    for case in evidence["cases"]:
        comparisons = case["comparisons"]
        assert comparisons["adapter_manifests_valid"] is True
        assert comparisons["latent_embeddings"]["exact"] is True
        assert comparisons["samples"]["exact_bytes"] is True
        assert all(
            checkpoint["keys_exact"] and checkpoint["tensor_values_exact"]
            for checkpoint in comparisons["checkpoints"].values()
        )


def test_tabsyn_rejects_epoch_controls_missing_from_official_source(tmp_path: Path) -> None:
    (tmp_path / "TabSyn-main").mkdir()
    adapter = TabSynAdapter(tmp_path)
    with pytest.raises(ValueError, match="does not expose epoch-count controls"):
        adapter.train(
            RunSpec(
                model="tabsyn",
                dataset="adult",
                output_dir=tmp_path / "artifacts",
                extra={"vae_num_epochs": 2},
            )
        )


def test_tabsyn_sample_maps_controls_at_compatibility_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "TabSyn-main"
    vae = upstream / "tabsyn" / "vae" / "ckpt" / "adult"
    diffusion = upstream / "tabsyn" / "ckpt" / "adult"
    vae.mkdir(parents=True)
    diffusion.mkdir(parents=True)
    for path in (vae / "train_z.npy", vae / "decoder.pt", diffusion / "model.pt"):
        path.write_bytes(b"trusted-test-stub")

    adapter = TabSynAdapter(tmp_path)
    calls: list[tuple[list[str], int]] = []
    monkeypatch.setattr(adapter, "_run_tabsyn", lambda args, *, seed: calls.append((args, seed)))
    bundle = adapter.sample(
        RunSpec(
            model="tabsyn",
            dataset="adult",
            output_dir=tmp_path / "artifacts",
            device="cuda:2",
            seed=11,
            num_samples=7,
            extra={"steps": 9},
        )
    )
    assert calls[0][1] == 11
    assert calls[0][0][:8] == [
        "--action",
        "sample",
        "--dataname",
        "adult",
        "--gpu",
        "2",
        "--save-path",
        str((tmp_path / "artifacts" / "samples.csv").resolve()),
    ]
    assert calls[0][0][-4:] == ["--steps", "9", "--num-samples", "7"]
    assert bundle.generated_sample_path == (tmp_path / "artifacts" / "samples.csv").resolve()


def test_tabsyn_rejects_symlinked_internal_checkpoint(tmp_path: Path) -> None:
    upstream = tmp_path / "TabSyn-main"
    vae = upstream / "tabsyn" / "vae" / "ckpt" / "adult"
    vae.mkdir(parents=True)
    target = tmp_path / "external.npy"
    target.write_bytes(b"stub")
    link = vae / "train_z.npy"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation is unavailable in this test environment.")
    adapter = TabSynAdapter(tmp_path)
    with pytest.raises(PermissionError, match="symlinked"):
        adapter.sample(
            RunSpec(model="tabsyn", dataset="adult", output_dir=tmp_path / "artifacts")
        )
