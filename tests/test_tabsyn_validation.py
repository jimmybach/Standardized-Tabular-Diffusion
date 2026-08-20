from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import standardized_tabular_diffusion.validation.tabsyn as tabsyn_validation
from standardized_tabular_diffusion.compat.tabsyn_launcher import (
    _configured_training_runtime,
    _with_configured_num_workers,
    _without_removed_scheduler_verbose,
)
from standardized_tabular_diffusion.interfaces import ArtifactBundle, DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.tabsyn import TabSynAdapter
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabsyn" / "native-parity-run-32055783087.json"
WINDOWS_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabsyn" / "windows-v2-real-function-6b3f2bc.json"
WINDOWS_EVIDENCE_SHA256 = "8f2c21d38c64484b019d43995d79c7a5d9cc837a1ae3c22d061db3b361758ccf"


def test_tabsyn_scheduler_bridge_removes_only_the_logging_keyword() -> None:
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def scheduler(*args: object, **kwargs: object) -> str:
        observed.append((args, kwargs))
        return "scheduler"

    compatible = _without_removed_scheduler_verbose(scheduler)
    assert compatible("optimizer", mode="min", factor=0.95, patience=10, verbose=True) == "scheduler"
    assert observed == [(("optimizer",), {"mode": "min", "factor": 0.95, "patience": 10})]


def test_tabsyn_worker_bridge_only_replaces_num_workers() -> None:
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def loader(*args: object, **kwargs: object) -> str:
        observed.append((args, kwargs))
        return "loader"

    configured = _with_configured_num_workers(loader, 0)
    assert configured("dataset", batch_size=4096, shuffle=True, num_workers=4) == "loader"
    assert observed == [(("dataset",), {"batch_size": 4096, "shuffle": True, "num_workers": 0})]


def test_tabsyn_training_runtime_bridge_scopes_scheduler_and_loader() -> None:
    class Module:
        pass

    module = Module()
    original_calls: list[tuple[str, dict[str, object]]] = []

    def scheduler(*args: object, **kwargs: object) -> str:
        del args
        original_calls.append(("scheduler", kwargs))
        return "scheduler"

    def loader(*args: object, **kwargs: object) -> str:
        del args
        original_calls.append(("loader", kwargs))
        return "loader"

    module.ReduceLROnPlateau = scheduler
    module.DataLoader = loader
    with _configured_training_runtime(module, 0):
        assert module.ReduceLROnPlateau("optimizer", factor=0.9, verbose=True) == "scheduler"
        assert module.DataLoader("data", batch_size=4096, num_workers=4) == "loader"
    assert module.ReduceLROnPlateau is scheduler
    assert module.DataLoader is loader
    assert original_calls == [
        ("scheduler", {"factor": 0.9}),
        ("loader", {"batch_size": 4096, "num_workers": 0}),
    ]


def test_tabsyn_scoped_sources_match_frozen_official_manifest() -> None:
    evidence = tabsyn_validation.verify_sources(REPO_ROOT)
    manifest = json.loads((REPO_ROOT / tabsyn_validation.MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert evidence["upstream_commit"] == "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7"
    assert evidence["upstream_files_verified"] == len(manifest["files"]) == 20
    assert not list((REPO_ROOT / "TabSyn-main" / "zero").glob("*.py"))


def test_tabsyn_retained_native_parity_evidence_is_exact_and_complete() -> None:
    assert hashlib.sha256(EVIDENCE_PATH.read_bytes()).hexdigest() == (
        "8cbfa66a57b99e5f9fdb0381b21b02eb9f5b062a4f8e4f1ef13de24be3862e48"
    )
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert evidence["protocol_id"] == "tabsyn-native-parity-v2"
    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "4668853b5d0acf7bff779453fb1a6e67f5838384"
    assert evidence["seed_cases"] == [0, 19, 73]
    assert [case["status"] for case in evidence["cases"]] == ["pass", "pass", "pass"]
    for case in evidence["cases"]:
        comparisons = case["comparisons"]
        assert comparisons["adapter_manifests_valid"] is True
        assert comparisons["latent_embeddings"]["exact"] is True
        assert comparisons["samples"]["exact_bytes"] is True
        assert Path(case["adapter_run_root"]).name == "run"
        assert all(
            checkpoint["keys_exact"] and checkpoint["tensor_values_exact"]
            for checkpoint in comparisons["checkpoints"].values()
        )


def test_tabsyn_retained_windows_v2_evidence_is_exact_and_complete() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "6b3f2bca50d79d5e59bb22b798eb8cb0a6a9f8f7"
    assert evidence["entry"]["device"] == "cuda"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["packages"]["libzero"] == "0.0.8"
    assert evidence["environment"]["hardware"] == {
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "gpu": "NVIDIA GeForce RTX 5080",
        "gpu_count": 1,
        "torch": "2.8.0+cu128",
    }
    pip_check = evidence["environment"]["pip_check"]
    assert pip_check["status"] == "pass-with-reviewed-waiver"
    assert pip_check["reviewed_waivers"] == [
        {
            "distribution": "libzero",
            "installed_dependency": "torch",
            "installed_dependency_version": "2.8.0+cu128",
            "reason": (
                "The frozen upstream snapshot imports libzero 0.0.8, whose stale distribution metadata caps "
                "torch below 2; the exact compatibility has already passed the retained native-parity workflow "
                "and this V2 run records the observed conflict."
            ),
            "requirement": "torch<2,>=1.7",
            "version": "0.0.8",
        }
    ]
    assert evidence["environment_lock"]["sha256"] == (
        "d81eda4aa77e35ece1c7dee18b29a14bba7b72c4c62279eae690072b1f8a995f"
    )
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [32, 32]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True
    assert evidence["tracked_repository_unchanged"] is True
    assert evidence["central_evaluation"]["status"] == "pass"
    assert evidence["central_evaluation"]["environment_lock"]["sha256"] == (
        "df78903678a6a8de185bfbc1bf7e1cca74ba01f8f5229f35ee2c495ebf6a02ca"
    )
    assert evidence["central_evaluation"]["validation"]["pending_files"] == 0

    windows = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tabsyn"]["windows_real_function"]
    assert windows["level"] == "minimal-real-passed"
    assert windows["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert windows["repository_commit"] == evidence["repository_commit"]
    assert windows["source_code_modified"] is False
    assert "docs/evidence/tabsyn/windows-v2-real-function-6b3f2bc.json" in get_adapter_spec("tabsyn").evidence_records


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
    (tmp_path / "TabSyn-main").mkdir()
    runtime_root = tmp_path / "artifacts" / "tabsyn-runtime"
    vae = runtime_root / "tabsyn" / "vae" / "ckpt" / "adult"
    diffusion = runtime_root / "tabsyn" / "ckpt" / "adult"
    vae.mkdir(parents=True)
    diffusion.mkdir(parents=True)
    for path in (vae / "train_z.npy", vae / "decoder.pt", diffusion / "model.pt"):
        path.write_bytes(b"trusted-test-stub")

    adapter = TabSynAdapter(tmp_path)
    calls: list[tuple[list[str], int]] = []

    def fake_run(args: list[str], *, seed: int) -> None:
        calls.append((args, seed))
        Path(args[args.index("--save-path") + 1]).write_text("x\n1\n", encoding="utf-8")

    monkeypatch.setattr(adapter, "_run_tabsyn", fake_run)
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
    assert calls[0][0][:9] == [
        "--action",
        "sample",
        "--dataname",
        "adult",
        "--gpu",
        "2",
        "--runtime-root",
        str(runtime_root.resolve()),
        "--save-path",
    ]
    assert calls[0][0][9:] == [
        str((tmp_path / "artifacts" / "samples.csv").resolve()),
        "--steps",
        "9",
        "--num-samples",
        "7",
    ]
    assert bundle.generated_sample_path == (tmp_path / "artifacts" / "samples.csv").resolve()


def test_tabsyn_validation_reuses_the_run_owned_training_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = tmp_path / "TabSyn-main"
    upstream.mkdir()
    output_root = tmp_path / "validation"
    calls: list[tuple[str, Path]] = []

    def fake_train(self: TabSynAdapter, spec: RunSpec) -> ArtifactBundle:
        calls.append(("train", spec.output_dir))
        self._ensure_output_dir(spec)
        runtime_root = self._runtime_root(spec)
        vae = runtime_root / "tabsyn" / "vae" / "ckpt" / spec.dataset
        diffusion = runtime_root / "tabsyn" / "ckpt" / spec.dataset
        vae.mkdir(parents=True)
        diffusion.mkdir(parents=True)
        for path in (vae / "train_z.npy", vae / "decoder.pt", diffusion / "model.pt"):
            path.write_bytes(b"checkpoint")
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.upstream_root,
            )
        )

    def fake_sample(self: TabSynAdapter, spec: RunSpec) -> ArtifactBundle:
        calls.append(("sample", spec.output_dir))
        assert (self._vae_ckpt_dir(spec) / "train_z.npy").is_file()
        assert (self._vae_ckpt_dir(spec) / "decoder.pt").is_file()
        assert (self._diffusion_ckpt_dir(spec) / "model.pt").is_file()
        sample_path = spec.output_dir / "samples.csv"
        sample_path.write_text("x\n1\n", encoding="utf-8")
        return self._write_bundle(
            ArtifactBundle(
                model=self.model_name,
                dataset=spec.dataset,
                output_dir=spec.output_dir,
                upstream_workdir=self.upstream_root,
                generated_sample_path=sample_path,
            )
        )

    monkeypatch.setattr(TabSynAdapter, "train", fake_train)
    monkeypatch.setattr(TabSynAdapter, "sample", fake_sample)
    _, run_root, manifests = tabsyn_validation._adapter_run(
        tmp_path,
        upstream,
        output_root,
        17,
    )

    assert calls == [("train", run_root), ("sample", run_root)]
    assert run_root == output_root / "run"
    assert [json.loads(path.read_text(encoding="utf-8"))["model"] for path in manifests] == [
        "tabsyn",
        "tabsyn",
    ]
    assert manifests[0] != manifests[1]


def test_tabsyn_decodes_declared_integers_and_retains_native_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "TabSyn-main").mkdir()
    output_dir = tmp_path / "artifacts"
    runtime_root = output_dir / "tabsyn-runtime"
    vae = runtime_root / "tabsyn" / "vae" / "ckpt" / "adult"
    diffusion = runtime_root / "tabsyn" / "ckpt" / "adult"
    vae.mkdir(parents=True)
    diffusion.mkdir(parents=True)
    checkpoints = {
        "train_z.npy": vae / "train_z.npy",
        "decoder.pt": vae / "decoder.pt",
        "model.pt": diffusion / "model.pt",
    }
    for name, path in checkpoints.items():
        path.write_bytes(f"trusted-{name}".encode())

    metadata_path = tmp_path / "info.json"
    metadata_path.write_text(json.dumps({"int_columns": ["x"]}), encoding="utf-8")
    dataset_spec = DatasetSpec(
        name="adult",
        task_type="classification",
        column_names=["x", "label"],
        numerical_columns=["x"],
        categorical_columns=[],
        target_columns=["label"],
        metadata_path=metadata_path,
    )
    binding = {"manifest_sha256": "b" * 64}
    training_metadata_path = output_dir / "tabsyn-model-metadata.json"
    training_metadata_path.write_text(
        json.dumps(
            {
                "model": "tabsyn",
                "dataset": "adult",
                "dataset_binding": binding,
                "checkpoints": {
                    name: {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in checkpoints.items()
                },
            }
        ),
        encoding="utf-8",
    )
    training_metadata_before = training_metadata_path.read_bytes()
    adapter = TabSynAdapter(tmp_path)
    monkeypatch.setattr(adapter, "_dataset_binding", lambda spec: binding)
    monkeypatch.setattr(adapter, "resolve_dataset_spec", lambda spec: dataset_spec)

    native_bytes = b"x,label\n1.5,no\n2.5,yes\n"

    def fake_run(args: list[str], *, seed: int) -> None:
        assert seed == 17
        Path(args[args.index("--save-path") + 1]).write_bytes(native_bytes)

    monkeypatch.setattr(adapter, "_run_tabsyn", fake_run)
    bundle = adapter.sample(
        RunSpec(
            model="tabsyn",
            dataset="adult",
            output_dir=output_dir,
            device="cpu",
            seed=17,
            num_samples=2,
            extra={"dataset_identity": {"name": "adult"}, "steps": 5},
        )
    )
    assert training_metadata_path.read_bytes() == training_metadata_before
    assert pd.read_csv(bundle.generated_sample_path)["x"].tolist() == [2, 2]
    native_path = output_dir / "tabsyn-native-samples.csv"
    assert native_path.read_bytes() == native_bytes
    sample_metadata = json.loads((output_dir / "tabsyn-sample-metadata.json").read_text(encoding="utf-8"))
    assert sample_metadata["integer_decoding"]["x"] == {
        "policy": "numpy-rint-ties-to-even-at-adapter-decoding-boundary",
        "changed_rows": 2,
        "clipped_rows": 0,
    }
    assert sample_metadata["native_sample_sha256"] == hashlib.sha256(native_bytes).hexdigest()
    assert sample_metadata["sample_sha256"] == hashlib.sha256(bundle.generated_sample_path.read_bytes()).hexdigest()


def test_tabsyn_rejects_symlinked_internal_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "TabSyn-main").mkdir()
    vae = tmp_path / "artifacts" / "tabsyn-runtime" / "tabsyn" / "vae" / "ckpt" / "adult"
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
        adapter.sample(RunSpec(model="tabsyn", dataset="adult", output_dir=tmp_path / "artifacts"))
