from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from standardized_tabular_diffusion.interfaces import ArtifactBundle
from standardized_tabular_diffusion.models.tabddpm import build_tabddpm_environment
from standardized_tabular_diffusion.runtime_contracts import dataset_content_identity
from standardized_tabular_diffusion.validation import tabddpm as validation_module
from standardized_tabular_diffusion.validation.tabddpm import (
    MANIFEST_RELATIVE_PATH,
    _run_adapter,
    _sha256_lf,
    _write_fixture,
    verify_sources,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "tabddpm" / "native-parity-run-30863212268.json"


def test_tabddpm_source_manifest_matches_pinned_sources() -> None:
    result = verify_sources(REPO_ROOT)

    assert result["upstream_commit"] == "b476257dd460b778ba09eb97f7a51d6490fa17f8"
    assert result["upstream_files_verified"] == 64
    assert result["libzero_modules_verified"] == 7


def test_tabddpm_source_manifest_has_unique_complete_paths() -> None:
    payload = json.loads((REPO_ROOT / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    paths = [record["path"] for record in payload["files"]]
    zero_paths = [record["path"] for record in payload["dependencies"]["libzero"]["vendored_modules"]]

    assert len(paths) == len(set(paths)) == 64
    assert len(zero_paths) == len(set(zero_paths)) == 7
    assert all(len(record["sha256_lf"]) == 64 for record in payload["files"])
    assert all(len(record["sha256"]) == 64 for record in payload["dependencies"]["libzero"]["vendored_modules"])


def test_tabddpm_libzero_license_hash_is_line_ending_independent(tmp_path: Path) -> None:
    payload = json.loads((REPO_ROOT / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    expected = payload["dependencies"]["libzero"]["license_sha256_lf"]
    source = (REPO_ROOT / "TabDDPM-main" / "zero" / "LICENSE.libzero").read_text(encoding="utf-8")
    lf_path = tmp_path / "license-lf.txt"
    crlf_path = tmp_path / "license-crlf.txt"
    lf_path.write_bytes(source.replace("\r\n", "\n").encode())
    crlf_path.write_bytes(source.replace("\r\n", "\n").replace("\n", "\r\n").encode())

    assert _sha256_lf(lf_path) == _sha256_lf(crlf_path) == expected


def test_tabddpm_modern_sklearn_bridge_preserves_integral_subsample() -> None:
    environment = os.environ.copy()
    environment.update(build_tabddpm_environment(REPO_ROOT / "TabDDPM-main"))
    command = [
        sys.executable,
        "-c",
        (
            "import sklearn.preprocessing as p; "
            "q=p.QuantileTransformer(subsample=1e9); "
            "print(type(q).__name__, q.subsample)"
        ),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)

    assert completed.stdout.strip() == "IntegralSubsampleQuantileTransformer 1000000000"


def test_tabddpm_parity_fixture_embeds_an_identity_checked_dataset_spec(tmp_path: Path) -> None:
    summary, dataset_spec = _write_fixture(tmp_path / "data")

    assert summary["train_rows"] == 24
    assert dataset_spec.name == "tabddpm-parity-fixture"
    assert dataset_spec.column_names == ["feature_0", "feature_1", "feature_2", "target"]
    assert dataset_spec.train_data_path is not None
    assert dataset_spec.train_data_path.read_text(encoding="utf-8").count("\n") == 25
    identity = dataset_content_identity(dataset_spec)
    assert identity["name"] == dataset_spec.name
    assert all(record["exists"] for record in identity["files"].values())


def test_tabddpm_parity_adapter_uses_one_run_owned_workspace_and_embedded_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_spec = _write_fixture(tmp_path / "data")
    captured = []

    class FakeAdapter:
        def __init__(self, repo_root: Path) -> None:
            self.repo_root = repo_root

        @staticmethod
        def _runtime_root(spec: object) -> Path:
            return spec.output_dir / "tabddpm-runtime"  # type: ignore[attr-defined]

        def _bundle(self, spec: object, *, sample: bool) -> ArtifactBundle:
            output_dir = spec.output_dir  # type: ignore[attr-defined]
            output_dir.mkdir(parents=True, exist_ok=True)
            runtime_root = self._runtime_root(spec)
            runtime_root.mkdir(parents=True, exist_ok=True)
            action = "sample" if sample else "train"
            (runtime_root / f"config-{action}-seed-{spec.seed}.toml").write_text(  # type: ignore[attr-defined]
                "seed = 0\n",
                encoding="utf-8",
            )
            sample_path = output_dir / f"samples-seed-{spec.seed}.csv"  # type: ignore[attr-defined]
            if sample:
                sample_path.write_text("feature_0,feature_1,feature_2,target\n", encoding="utf-8")
            bundle = ArtifactBundle(
                model="tabddpm",
                dataset=spec.dataset,  # type: ignore[attr-defined]
                output_dir=output_dir,
                upstream_workdir=self.repo_root / "TabDDPM-main",
                generated_sample_path=sample_path if sample else None,
            )
            (output_dir / "artifacts.json").write_text(
                json.dumps(bundle.to_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return bundle

        def train(self, spec: object) -> ArtifactBundle:
            captured.append(spec)
            return self._bundle(spec, sample=False)

        def sample(self, spec: object) -> ArtifactBundle:
            captured.append(spec)
            return self._bundle(spec, sample=True)

    monkeypatch.setattr(validation_module, "TabDDPMAdapter", FakeAdapter)
    config_path = tmp_path / "config.toml"
    config_path.write_text("seed = 0\n", encoding="utf-8")
    output_dir = tmp_path / "adapter-run"

    result = _run_adapter(
        tmp_path,
        config_path,
        output_dir,
        dataset_spec,
        training_seed=17,
        sampling_seed=47,
    )

    assert len(captured) == 2
    assert captured[0].output_dir == captured[1].output_dir == output_dir
    assert captured[0].extra["dataset_spec"]["name"] == dataset_spec.name
    assert captured[0].extra["dataset_identity"] == captured[1].extra["dataset_identity"]
    assert result["manifests_valid"] is True
    assert Path(result["runtime_root"]).is_relative_to(output_dir)


def test_tabddpm_native_parity_evidence_is_complete_and_immutable() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)

    assert hashlib.sha256(evidence_bytes).hexdigest() == "8fd277aef64a2e7225626a95379ecf67462ac686a4d688a56748f9ef965dd29e"
    assert evidence["status"] == "pass"
    assert evidence["repository_commit"] == "3339af2603bac7a4736e68d7f369194b6b095653"
    assert len(evidence["cases"]) == 3
    for case in evidence["cases"]:
        assert case["status"] == "pass"
        comparisons = case["comparisons"]
        assert comparisons["config_exact"] is True
        assert comparisons["model"]["tensor_values_exact"] is True
        assert comparisons["ema_model"]["tensor_values_exact"] is True
        assert comparisons["loss_csv_exact"] is True
        assert comparisons["sample_rows"] == 12
        assert all(record["exact"] and record["finite"] for record in comparisons["generated_arrays"].values())
