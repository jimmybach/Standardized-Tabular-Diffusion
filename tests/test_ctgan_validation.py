from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import standardized_tabular_diffusion.validation.ctgan as ctgan_validation
from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "ctgan" / "native-parity-run-32047234665.json"
EVIDENCE_SHA256 = "ce9698605f13c641b033d221a56721a957a90135fb2ea639ad2730922e73ae24"


def _write_test_wheel(path: Path, *, unsafe_member: str | None = None) -> str:
    metadata_path = "ctgan-0.12.1.dist-info/METADATA"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            metadata_path,
            "Metadata-Version: 2.4\nName: ctgan\nVersion: 0.12.1\nLicense-Expression: BUSL-1.1\n",
        )
        archive.writestr("ctgan/__init__.py", "__version__ = '0.12.1'\n")
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ctgan_package_lock_matches_registry_and_protocol() -> None:
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["components"]["ctgan"]
    spec = get_adapter_spec("ctgan")

    assert spec.upstream_repository == ctgan_validation.UPSTREAM_REPOSITORY
    assert spec.upstream_revision == ctgan_validation.UPSTREAM_COMMIT
    assert spec.install_extra == "ctgan"
    assert spec.validation_level.value == "native-parity-validated"
    assert source_lock["package_lock"]["version"] == ctgan_validation.PACKAGE_VERSION
    assert source_lock["package_lock"]["sha256"] == ctgan_validation.WHEEL_SHA256
    assert source_lock["upstream_tree"] == ctgan_validation.UPSTREAM_TREE
    assert source_lock["license"] == ctgan_validation.LICENSE_EXPRESSION


def test_retained_ctgan_v2_evidence_is_immutable_and_complete() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert evidence["status"] == "pass"
    assert evidence["protocol_id"] == ctgan_validation.PROTOCOL_ID
    assert evidence["repository_commit"] == "9d5d7e9415f41976bc5524ce5547595ea11e2043"
    assert evidence["environment"]["platform"].startswith("Linux-")
    assert evidence["environment"]["python"] == "3.11.15"
    assert evidence["environment"]["torch"] == "2.3.0+cpu"
    assert evidence["source"]["installed_distribution"]["record_files_verified"] == 20
    assert evidence["source"]["wheel"]["sha256"] == ctgan_validation.WHEEL_SHA256
    assert evidence["seed_cases"] == [
        {"train_seed": 0, "sample_seed": 101},
        {"train_seed": 19, "sample_seed": 7},
        {"train_seed": 73, "sample_seed": 29},
    ]
    assert tuple(
        (case["train_seed"], case["sample_seed"]) for case in evidence["cases"]
    ) == ctgan_validation.SEED_CASES
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(ctgan_validation._case_passed(case["comparisons"]) for case in evidence["cases"])
    assert all(
        case["adapter_artifacts"]["sample_sha256"] == case["native_artifacts"]["sample_sha256"]
        for case in evidence["cases"]
    )


def test_ctgan_wheel_validation_checks_identity_and_license(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / ctgan_validation.WHEEL_FILENAME
    wheel_sha256 = _write_test_wheel(wheel_path)
    monkeypatch.setattr(ctgan_validation, "WHEEL_SHA256", wheel_sha256)

    record = ctgan_validation._verify_wheel(wheel_path)

    assert record["sha256"] == wheel_sha256
    assert record["license_expression"] == "BUSL-1.1"
    assert record["archive_members"] == 2


def test_ctgan_wheel_validation_rejects_archive_traversal(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / ctgan_validation.WHEEL_FILENAME
    wheel_sha256 = _write_test_wheel(wheel_path, unsafe_member="../escape.py")
    monkeypatch.setattr(ctgan_validation, "WHEEL_SHA256", wheel_sha256)

    with pytest.raises(ValueError, match="Unsafe path"):
        ctgan_validation._verify_wheel(wheel_path)


def test_ctgan_parity_gate_requires_every_comparison() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "sample_bytes_exact": True,
        "model": {
            "constructor_exact": True,
            "generator": {"keys_exact": True, "tensor_values_exact": True, "finite": True},
            "transformer_exact": True,
            "data_sampler": {"arrays_exact": True, "row_ids_exact": True, "scalars_exact": True},
            "random_state": {"numpy_exact": True, "torch_exact": True},
            "loss_values_exact": True,
        },
        "samples": {
            "rows": ctgan_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }

    assert ctgan_validation._case_passed(comparisons) is True
    comparisons["samples"]["frame_exact"] = False
    assert ctgan_validation._case_passed(comparisons) is False


def test_ctgan_native_control_reseeds_loaded_model_for_independent_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    class FakeCTGAN:
        instances: list[FakeCTGAN] = []

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

    monkeypatch.setitem(sys.modules, "ctgan", SimpleNamespace(CTGAN=FakeCTGAN))
    monkeypatch.setattr(
        ctgan_validation,
        "_load_official",
        lambda _path: FakeCTGAN.instances[-1],
    )

    _model, samples, _artifacts = ctgan_validation._run_native(
        pd.DataFrame({"value": [1, 2], "category": ["a", "b"]}),
        ["category"],
        tmp_path / "native",
        train_seed=17,
        sample_seed=83,
    )

    assert FakeCTGAN.instances[-1].seeds == [17, 83]
    assert len(samples) == ctgan_validation.EXPECTED_SAMPLE_ROWS
