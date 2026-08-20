from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import standardized_tabular_diffusion.validation.tvae as tvae_validation
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tvae" / "native-parity-run-32052308431.json"
EVIDENCE_SHA256 = "5c1a050af546b1b4fa0c7a7bd354430f34c130ca4d0f4c1875d42ae0ebd5fd7e"
WINDOWS_EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tvae" / "windows-v2-real-function-5bf59b0.json"
WINDOWS_EVIDENCE_SHA256 = "a6bcb8eefd81141d5e0f485b9aecb50666d3490e97eb18ce11c6c2a170250301"


def test_tvae_package_lock_matches_registry_and_protocol() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tvae"]
    spec = get_adapter_spec("tvae")

    assert spec.upstream_repository == tvae_validation.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == tvae_validation.UPSTREAM_COMMIT
    assert spec.install_extra == "tvae"
    assert spec.validation_level.value == "native-parity-validated"
    assert source_lock["package_lock"]["version"] == tvae_validation.PACKAGE_VERSION
    assert source_lock["package_lock"]["sha256"] == tvae_validation.WHEEL_SHA256
    assert source_lock["upstream_tree"] == tvae_validation.UPSTREAM_TREE
    assert source_lock["license"] == tvae_validation.LICENSE_EXPRESSION
    assert source_lock["validation"]["status"] == "pass"
    assert EVIDENCE_PATH.as_posix().endswith(source_lock["validation"]["artifact"]["permanent_evidence_path"])


def test_retained_tvae_evidence_is_immutable_and_complete() -> None:
    raw_evidence = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(raw_evidence)

    assert hashlib.sha256(raw_evidence).hexdigest() == EVIDENCE_SHA256
    assert evidence["status"] == "pass"
    assert evidence["model_id"] == "tvae"
    assert evidence["protocol_id"] == tvae_validation.PROTOCOL_ID
    assert evidence["repository_commit"] == "b20e9a50ac95602d3348e870de94e3593f935862"
    assert evidence["seed_cases"] == [
        {"train_seed": 0, "sample_seed": 101},
        {"train_seed": 19, "sample_seed": 7},
        {"train_seed": 73, "sample_seed": 29},
    ]
    assert tuple(
        (case["train_seed"], case["sample_seed"]) for case in evidence["cases"]
    ) == tvae_validation.SEED_CASES
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(tvae_validation._case_passed(case["comparisons"]) for case in evidence["cases"])
    assert evidence["source"]["installed_distribution"]["record_files_verified"] == 20
    assert evidence["source"]["synthesizer"] == {
        "class": "TVAE",
        "legacy_snapshot_absent": True,
        "module": "ctgan.synthesizers.tvae",
    }


def test_retained_tvae_windows_v2_evidence_is_exact_and_complete() -> None:
    evidence_bytes = WINDOWS_EVIDENCE_PATH.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == WINDOWS_EVIDENCE_SHA256
    assert evidence_bytes.endswith(b"\n")
    evidence = json.loads(evidence_bytes)

    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == "pipeline-v2-native-windows-v1"
    assert evidence["repository_commit"] == "5bf59b0effa29a0c2694cdbe05b1a8f40443c481"
    assert evidence["entry"]["device"] == "cuda"
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["hardware"]["gpu"] == "NVIDIA GeForce RTX 5080"
    assert evidence["environment"]["hardware"]["torch"] == "2.8.0+cu128"
    assert evidence["environment"]["hardware"]["cuda_runtime"] == "12.8"
    assert evidence["environment"]["pip_check"]["status"] == "pass"
    assert "libzero" not in evidence["environment"]["packages"]
    assert evidence["environment_lock"]["sha256"] == (
        "cd2cca53950203df9d772cffd8ca57e91fc9874d9663f6af4e08948b48733f7b"
    )
    assert [sample["seed"] for sample in evidence["samples"]] == [17, 29]
    assert [sample["rows"] for sample in evidence["samples"]] == [32, 32]
    assert all(sample["schema_valid"] for sample in evidence["samples"])
    assert all(sample["missing_cells"] == 0 for sample in evidence["samples"])
    assert all(sample["training_artifacts_unchanged"] for sample in evidence["samples"])
    assert evidence["seed_outputs_distinct"] is True
    assert evidence["tracked_repository_unchanged"] is True
    assert evidence["central_evaluation"]["status"] == "pass"
    assert evidence["central_evaluation"]["validation"]["pending_files"] == 0

    component = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tvae"]
    windows = component["windows_real_function"]
    assert windows["level"] == "minimal-real-passed"
    assert windows["evidence_file_sha256"] == WINDOWS_EVIDENCE_SHA256
    assert windows["repository_commit"] == evidence["repository_commit"]
    assert windows["source_code_modified"] is False
    assert "docs/evidence/tvae/windows-v2-real-function-5bf59b0.json" in get_adapter_spec(
        "tvae"
    ).evidence_records


def test_legacy_tvae_snapshot_is_recorded_and_removed() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["tvae"]
    disposition = source_lock["legacy_snapshot_disposition"]

    assert disposition["declared_version"] == "0.5.2.dev0"
    assert disposition["closest_reviewed_upstream_commit"] == (
        "ace3dbc4bd3ef7f4ddc027a1b47e8eb916378893"
    )
    assert disposition["shared_files_at_closest_commit"] == 44
    assert disposition["exact_shared_files_at_closest_commit"] == 39
    assert disposition["files_removed"] == 47
    assert disposition["removed_bytes"] == 168098
    assert disposition["tvae_source_sha256"] == (
        "0b0bc0ed424f295084a395a212eb40f1464df0e6474c313e488e8ad43226689f"
    )
    assert not (REPO_ROOT / "TabDDPM-main" / "CTGAN" / "CTGAN" / "ctgan" / "__init__.py").exists()
    assert not (REPO_ROOT / "TabDDPM-main" / "CTGAN" / "train_sample_tvae.py").exists()


def test_tvae_parity_gate_requires_every_comparison() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "sample_bytes_exact": True,
        "model": {
            "constructor_exact": True,
            "device_exact": True,
            "decoder": {
                "keys_exact": True,
                "tensor_values_exact": True,
                "finite": True,
                "sigma_finite": True,
            },
            "transformer_exact": True,
            "random_state": {"numpy_exact": True, "torch_exact": True},
            "loss_values_exact": True,
        },
        "samples": {
            "rows": tvae_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }

    assert tvae_validation._case_passed(comparisons) is True
    comparisons["model"]["decoder"]["sigma_finite"] = False
    assert tvae_validation._case_passed(comparisons) is False


def test_tvae_native_control_reseeds_loaded_model_for_independent_sampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pandas as pd

    class FakeTVAE:
        instances: list[FakeTVAE] = []

        def __init__(self, **_kwargs: object) -> None:
            self.seeds: list[int] = []
            self.__class__.instances.append(self)

        def set_random_state(self, seed: int) -> None:
            self.seeds.append(seed)

        def fit(self, _frame: pd.DataFrame, *, discrete_columns: list[str]) -> None:
            assert discrete_columns == ["category"]

        @staticmethod
        def save(path: Path) -> None:
            path.write_bytes(b"checkpoint")

        @staticmethod
        def sample(rows: int) -> pd.DataFrame:
            return pd.DataFrame({"value": range(rows), "category": ["a"] * rows})

    monkeypatch.setitem(sys.modules, "ctgan", SimpleNamespace(TVAE=FakeTVAE))
    monkeypatch.setattr(
        tvae_validation,
        "_load_official",
        lambda _path: FakeTVAE.instances[-1],
    )

    _model, samples, _artifacts = tvae_validation._run_native(
        pd.DataFrame({"value": [1, 2], "category": ["a", "b"]}),
        ["category"],
        tmp_path / "native",
        train_seed=17,
        sample_seed=83,
    )

    assert FakeTVAE.instances[-1].seeds == [17, 83]
    assert len(samples) == tvae_validation.EXPECTED_SAMPLE_ROWS
