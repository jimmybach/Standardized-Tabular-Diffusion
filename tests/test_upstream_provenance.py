from __future__ import annotations

import hashlib
import json
from pathlib import Path

from standardized_tabular_diffusion.registry import get_adapter_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "standardized_tabular_diffusion" / "resources" / "upstream" / "source-lock.json"


def _load_source_lock() -> dict[str, object]:
    return json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))


def test_source_lock_matches_primary_adapter_registry() -> None:
    payload = _load_source_lock()
    components = payload["components"]

    assert isinstance(components, dict)
    assert set(components) == {
        "codi",
        "ctab-gan",
        "ctab-gan-plus",
        "ctgan",
        "goggle",
        "nrgboost",
        "realtabformer",
        "smote",
        "stasy",
        "tabddpm",
        "tabdiff",
        "tabularargn",
        "tabsyn",
        "tvae",
    }
    for component_id, component in components.items():
        assert isinstance(component, dict)
        spec = get_adapter_spec(component_id)
        assert spec.upstream_repository == component["authoritative_repository"]
        assert spec.upstream_revision == component["upstream_commit"]
        assert list(spec.patch_set_ids) == [patch["patch_set_id"] for patch in component["patch_sets"]]
        if component["distribution_form"] == "source":
            assert (REPO_ROOT / component["license_path"]).is_file()
        elif component["distribution_form"] == "package":
            assert component["package_lock"]["sha256"]
            assert component["license_url"].startswith("https://")
        elif component["license"] == "NONE-DECLARED":
            assert component["distribution_form"] == "source-on-demand"
            assert component["license_review"]["redistribution_status"] == "not-authorized"
            assert (REPO_ROOT / component["source_manifest"]).is_file()
        else:
            assert component["distribution_form"] == "source-on-demand"
            assert component["license"] == "MIT"
            assert component["source_treatment"].endswith("without-patches")
            assert (REPO_ROOT / component["source_manifest"]).is_file()


def test_source_lock_patch_ids_are_unique_and_classified() -> None:
    payload = _load_source_lock()
    components = payload["components"]
    patch_ids: list[str] = []

    assert isinstance(components, dict)
    for component in components.values():
        assert isinstance(component, dict)
        for patch in component["patch_sets"]:
            assert patch["classification"] in {"adapter-only", "compatibility-patched", "semantic-patched"}
            assert patch["status"] != "approved-parity-validated"
            patch_ids.append(patch["patch_set_id"])

    assert len(patch_ids) == len(set(patch_ids))


def test_tabularargn_release_is_locked_and_parity_validated() -> None:
    payload = _load_source_lock()
    component = payload["components"]["tabularargn"]
    spec = get_adapter_spec("tabularargn")

    assert component["authority"] == "method-author"
    assert component["distribution_form"] == "package"
    assert component["license"] == "Apache-2.0"
    assert component["package_lock"]["installed_files_verified"] == 53
    assert component["wheel_source_comparison"]["exact_shared_source_files"] == 50
    assert component["patch_sets"] == []
    validation = component["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30961590047
    assert validation["protocol_id"] == "tabularargn-official-package-parity-v2"
    assert validation["result_summary"]["parity_cases_passed"] == 9
    assert validation["result_summary"]["contract_normalized_samples_exact"] is True
    assert validation["result_summary"]["raw_samples_exact"] is False
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert spec.validation_level.value == "native-parity-validated"
    assert spec.modification_status == "adapter-only"
    assert spec.revision_status == "pinned-official-package-native-parity-validated"
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"


def test_audited_primary_adapters_fail_closed_for_release_claims() -> None:
    evidence_paths = {
        "codi": "docs/evidence/codi/native-parity-run-30941940893.json",
        "ctab-gan": "docs/evidence/ctabgan/native-parity-run-30930939961.json",
        "ctab-gan-plus": "docs/evidence/ctabgan-plus/native-parity-run-30926267432.json",
        "ctgan": "docs/evidence/ctgan/native-parity-run-30910275922.json",
        "goggle": "docs/evidence/goggle/native-parity-run-30945676747.json",
        "nrgboost": "docs/evidence/nrgboost/native-parity-run-30922326384.json",
        "realtabformer": "docs/evidence/realtabformer/native-parity-run-30950369908.json",
        "smote": "docs/evidence/smote/native-parity-run-30918785254.json",
        "stasy": "docs/evidence/stasy/native-parity-run-30936275831.json",
        "tabddpm": "docs/evidence/tabddpm/native-parity-run-30863212268.json",
        "tabdiff": "docs/evidence/tabdiff/native-parity-run-30866879879.json",
        "tabularargn": "docs/evidence/tabularargn/native-parity-run-30961590047.json",
        "tabsyn": "docs/evidence/tabsyn/native-parity-run-30871758645.json",
        "tvae": "docs/evidence/tvae/native-parity-run-30913867621.json",
    }
    for model_id in (
        "codi",
        "ctab-gan",
        "ctab-gan-plus",
        "ctgan",
        "goggle",
        "nrgboost",
        "realtabformer",
        "smote",
        "stasy",
        "tabddpm",
        "tabdiff",
        "tabularargn",
        "tabsyn",
        "tvae",
    ):
        spec = get_adapter_spec(model_id)
        assert spec.upstream_revision is not None
        assert spec.validation_level.value == "native-parity-validated"
        expected_modification = "compatibility-patched" if model_id == "realtabformer" else "adapter-only"
        assert spec.modification_status == expected_modification
        assert spec.patch_set_ids == ()
        assert evidence_paths[model_id] in spec.evidence_records
        assert spec.benchmark_track == "experimental"
        assert spec.support_level == "unsupported"

    tabddpm = get_adapter_spec("tabddpm")
    assert tabddpm.revision_status == "pinned-complete-native-parity-validated"

    smote = get_adapter_spec("smote")
    assert smote.revision_status == "pinned-canonical-package-native-parity-validated"
    assert smote.reproduction_target == "classical-oversampling-reference"

    # STaSy validates the TabSyn benchmark snapshot, not the distinct method-author source.
    stasy = get_adapter_spec("stasy")
    assert stasy.modification_status == "adapter-only"
    assert stasy.reproduction_target == "tabsyn-benchmark-snapshot"
    assert stasy.validation_level.value == "native-parity-validated"
    assert stasy.revision_status == "pinned-exact-tabsyn-snapshot-parity-validated"
    stasy_lock = _load_source_lock()["components"]["stasy"]
    assert stasy_lock["dependency_resolution"]["resolved_distribution"] == "libzero==0.0.8"
    assert stasy_lock["dependency_resolution"]["wheel_sha256"] == (
        "f7bb46c71433ca19b61c5127d010147bccc6b29d250f30ad48a393ce676a5e9d"
    )
    assert stasy_lock["compatibility_shims"] == [
        {
            "classification": "adapter-only",
            "id": "stasy-sklearn-onehot-keyword-v1",
            "reason": (
                "The pinned snapshot uses sparse=False, while scikit-learn 1.5.2 names the same "
                "dense-output control sparse_output."
            ),
            "semantic_effect": (
                "The bridge forwards the unchanged false value to sparse_output; it changes no "
                "encoder, output representation, source file, or preprocessing operation."
            ),
        }
    ]
    codi = get_adapter_spec("codi")
    assert codi.modification_status == "adapter-only"
    assert codi.reproduction_target == "tabsyn-benchmark-snapshot"
    assert codi.validation_level.value == "native-parity-validated"
    assert codi.revision_status == "pinned-exact-tabsyn-snapshot-parity-validated"
    codi_lock = _load_source_lock()["components"]["codi"]
    assert codi_lock["snapshot_comparison"]["exact_local_codi_source_files"] == 11
    assert codi_lock["selected_runtime_files"] == 24
    assert codi_lock["method_author_source"]["license_file_present"] is False
    assert codi_lock["method_author_source"]["exact_shared_paths"] == 5
    assert codi_lock["dependency_resolution"]["resolved_distribution"] == "libzero==0.0.8"
    assert codi_lock["dependency_resolution"]["actual_runtime_dependencies"] == {"tqdm": "4.66.5"}


def test_goggle_retained_method_author_validation_is_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    goggle = payload["components"]["goggle"]
    assert isinstance(goggle, dict)

    assert goggle["authority"] == "method-author"
    assert goggle["distribution_form"] == "source-on-demand"
    assert goggle["license"] == "MIT"
    assert goggle["source_treatment"].endswith("without-patches")
    assert goggle["selected_runtime_files"] == 18
    assert goggle["retired_snapshot"]["shared_paths_differing_after_text_normalization"] == 9

    validation = goggle["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30945676747
    assert validation["pull_request_head_commit"] == "088aed9d2aab4cca38f1e6f8a79e3eea126020e1"
    assert validation["repository_commit"] == "f13f37d653bbdbc5bd4e99035989ff9bfa3a3444"
    assert validation["environment"]["platform"] == "Linux / x86_64"
    assert validation["environment"]["python"] == "3.11.15"
    assert validation["environment_lock_sha256"] == (
        "df84de4126bdf5816f54107c544b0971704b58bafd21d93a499956892884bd7a"
    )

    summary = validation["result_summary"]
    assert summary["parity_cases_passed"] == summary["parity_cases_total"] == 9
    assert summary["checkpoint_state_exact"] is True
    assert summary["checkpoint_file_bytes_exact"] is True
    assert summary["raw_samples_exact"] is True
    assert summary["sample_bytes_exact"] is True
    assert summary["sample_frames_exact"] is True
    assert summary["adapter_source_remained_exact"] is True
    assert summary["missing_values"] == 0

    artifact = validation["artifact"]
    assert artifact["evidence_file_sha256"] == (
        "1dbcf50194505820cac0650ba72d519f4f331008bbcaac635f8eb846bec7da59"
    )
    assert artifact["evidence_file_sha256"] == artifact["downloaded_evidence_sha256"]
    evidence_path = REPO_ROOT / artifact["permanent_evidence_path"]
    evidence_bytes = evidence_path.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == artifact["evidence_file_sha256"]
    assert evidence_bytes.endswith(b"\n")

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "pass"
    assert evidence["reproduction_target"] == "method-author-original-core"
    assert evidence["source"]["runtime_files_verified"] == 18
    assert len(evidence["cases"]) == 9
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(case["comparisons"]["checkpoints"]["tensors_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["raw_samples_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["sample_bytes_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["adapter_source_remained_exact"] for case in evidence["cases"])
    assert "heterogeneous-decoder-runtime" in goggle["official_eligibility"]
    assert "release-gates" in goggle["official_eligibility"]


def test_codi_retained_tabsyn_snapshot_validation_is_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    codi = payload["components"]["codi"]
    assert isinstance(codi, dict)

    validation = codi["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30941940893
    assert validation["pull_request_head_commit"] == "bcfc4dd1d6b219c578bac44c4bd85606158bfb83"
    assert validation["repository_commit"] == "b0a380cd01ee08378742c231ec5811351103b20c"
    assert validation["result_summary"]["parity_cases_passed"] == 9
    assert validation["result_summary"]["continuous_checkpoint_state_exact"] is True
    assert validation["result_summary"]["discrete_checkpoint_state_exact"] is True
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert validation["result_summary"]["sample_frames_exact"] is True
    assert validation["artifact"]["evidence_file_sha256"] == (
        "14d188b856e44dfc7cb7cf5ab16c5cfd7a03aa4b4d7d71e2bcb4226f13f1f156"
    )
    evidence_path = REPO_ROOT / validation["artifact"]["permanent_evidence_path"]
    evidence_bytes = evidence_path.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == validation["artifact"][
        "evidence_file_sha256"
    ]
    assert evidence_bytes.endswith(b"\n")
    assert validation["artifact"]["evidence_file_sha256"] == validation["artifact"][
        "downloaded_evidence_sha256"
    ]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "pass"
    assert evidence["reproduction_target"] == "tabsyn-benchmark-snapshot"
    assert evidence["environment"]["tqdm"] == "4.66.5"
    assert len(evidence["cases"]) == 9
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert all(case["comparisons"]["checkpoints"]["pair_state_exact"] for case in evidence["cases"])
    assert all(case["comparisons"]["samples"]["exact_bytes"] for case in evidence["cases"])
    assert "benchmark-snapshot-not-method-author-original" in codi["official_eligibility"]


def test_stasy_retained_tabsyn_snapshot_validation_is_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    stasy = payload["components"]["stasy"]
    assert isinstance(stasy, dict)

    validation = stasy["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30936275831
    assert validation["pull_request_head_commit"] == "95f339b60916c82e2cd3987a0e30e369744781ee"
    assert validation["repository_commit"] == "ccfb02e29aa1e9e14f25816e86ba74c3ec088a87"
    assert validation["result_summary"]["parity_cases_passed"] == 9
    assert validation["result_summary"]["checkpoint_model_exact"] is True
    assert validation["result_summary"]["checkpoint_optimizer_exact"] is True
    assert validation["result_summary"]["checkpoint_ema_exact"] is True
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert validation["artifact"]["evidence_file_sha256"] == (
        "53c6bdbe66d38ce1a3d91cee4472ffbb8379c5d7a2ac3aa8c4ffcfa86f44cb67"
    )
    evidence_path = REPO_ROOT / validation["artifact"]["permanent_evidence_path"]
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == validation["artifact"][
        "evidence_file_sha256"
    ]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "pass"
    assert evidence["reproduction_target"] == "tabsyn-benchmark-snapshot"
    assert len(evidence["cases"]) == 9
    assert all(case["status"] == "pass" for case in evidence["cases"])
    assert "benchmark-snapshot-not-method-author-original" in stasy["official_eligibility"]


def test_ctabgan_retained_validation_and_source_license_are_exact() -> None:
    payload = _load_source_lock()
    ctabgan = payload["components"]["ctab-gan"]
    assert isinstance(ctabgan, dict)

    assert ctabgan["distribution_form"] == "source"
    assert ctabgan["license"] == "Apache-2.0"
    assert ctabgan["selected_files"] == 7
    assert ctabgan["modification_status"] == "adapter-only"
    assert ctabgan["compatibility_shims"][0]["id"] == "ctabgan-sklearn-keyword-only-v1"
    validation = ctabgan["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30930939961
    assert validation["pull_request_head_commit"] == "4501d5ef8d552c840aea06035cd3902eaeef7a82"
    assert validation["repository_commit"] == "ecc7e0f1931c61d9ac019d44782385ecb4637fbf"
    assert validation["result_summary"]["parity_cases_passed"] == 6
    assert validation["result_summary"]["checkpoint_state_exact"] is True
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert validation["artifact"]["evidence_file_sha256"] == (
        "41788d11578c55530b55fbf392412de361ec2769c63329a3174fa15c6905d0c6"
    )
    evidence_path = REPO_ROOT / validation["artifact"]["permanent_evidence_path"]
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == validation["artifact"][
        "evidence_file_sha256"
    ]


def test_ctabgan_plus_retained_validation_is_exact_and_license_blocked() -> None:
    payload = _load_source_lock()
    ctabgan_plus = payload["components"]["ctab-gan-plus"]
    assert isinstance(ctabgan_plus, dict)

    assert ctabgan_plus["distribution_form"] == "source-on-demand"
    assert ctabgan_plus["license"] == "NONE-DECLARED"
    assert ctabgan_plus["license_review"]["redistribution_status"] == "not-authorized"
    validation = ctabgan_plus["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30926267432
    assert validation["pull_request_head_commit"] == "473af6334d6f367b75b35736370c4dfa6adf85bf"
    assert validation["repository_commit"] == "48837271b693b8af396f4f35cb68707b5c52e5bc"
    assert validation["result_summary"]["parity_cases_passed"] == 6
    assert validation["result_summary"]["checkpoint_state_exact"] is True
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert validation["artifact"]["evidence_file_sha256"] == (
        "df3bbf0dd46d34e8d57551048c7b7abe60340eddb3738e31d400e44344c5e5f2"
    )
    evidence_path = REPO_ROOT / validation["artifact"]["permanent_evidence_path"]
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == validation["artifact"][
        "evidence_file_sha256"
    ]
    assert str(ctabgan_plus["official_eligibility"]).startswith("blocked-no-upstream-license")


def test_tabddpm_source_lock_records_native_parity_without_overclaiming() -> None:
    payload = _load_source_lock()
    tabddpm = payload["components"]["tabddpm"]
    assert isinstance(tabddpm, dict)
    validation = tabddpm["validation"]

    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30863212268
    assert validation["result_summary"]["seed_cases_passed"] == 3
    assert tabddpm["official_eligibility"] == "pending-separate-official-track-review"


def test_ctgan_package_lock_is_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    ctgan = payload["components"]["ctgan"]
    assert isinstance(ctgan, dict)

    assert ctgan["distribution_form"] == "package"
    assert ctgan["license"] == "BUSL-1.1"
    assert ctgan["package_lock"] == {
        "filename": "ctgan-0.12.1-py3-none-any.whl",
        "name": "ctgan",
        "pypi_url": "https://pypi.org/project/ctgan/0.12.1/",
        "sha256": "38a3b83432643caa8381c74c49e6a079166efa40f8f6c3b7204db44d6d2c8f18",
        "trusted_publishing_source_commit": "826da23f8f9385ad15fd206ecad691e04cb0ccdc",
        "version": "0.12.1",
    }
    validation = ctgan["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30910275922
    assert validation["result_summary"]["seed_cases_passed"] == 3
    assert validation["artifact"]["evidence_file_sha256"] == (
        "748501c8671c272a1e5d54c85fdb6550182d0e5578d550a3ca7681cc712f4570"
    )
    assert str(ctgan["official_eligibility"]).startswith("blocked-pending-license")


def test_tvae_package_lock_and_retained_validation_are_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    tvae = payload["components"]["tvae"]
    assert isinstance(tvae, dict)

    assert tvae["distribution_form"] == "package"
    assert tvae["license"] == "BUSL-1.1"
    assert tvae["package_lock"] == payload["components"]["ctgan"]["package_lock"]
    validation = tvae["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30913867621
    assert validation["result_summary"]["seed_cases_passed"] == 3
    assert validation["artifact"]["evidence_file_sha256"] == (
        "ad539ffdb637084a25dc3ab4ec5d54374ff6831525ca63adca2cfa48c3ef95f7"
    )
    assert str(tvae["official_eligibility"]).startswith("blocked-pending-license")


def test_nrgboost_package_lock_and_retained_validation_are_exact_and_conservatively_gated() -> None:
    payload = _load_source_lock()
    nrgboost = payload["components"]["nrgboost"]
    assert isinstance(nrgboost, dict)

    assert nrgboost["distribution_form"] == "package"
    assert nrgboost["license"] == "MIT"
    assert nrgboost["package_lock"] == {
        "filename": "nrgboost-0.0.3-cp311-cp311-manylinux_2_28_x86_64.whl",
        "name": "nrgboost",
        "pypi_url": "https://pypi.org/project/nrgboost/0.0.3/",
        "sha256": "dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb",
        "trusted_publishing_source_commit": "feef73a3edb20b911c2f7214b13f810909ef20ad",
        "version": "0.0.3",
        "wheel_license_sha256": "3693dc7c451fe74ffead14c00964ac00a1123242c9fc3d8cb13c8fef3091b945",
    }
    validation = nrgboost["validation"]
    assert validation["level"] == "native-parity-validated"
    assert validation["status"] == "pass"
    assert validation["workflow_run_id"] == 30922326384
    assert validation["repository_commit"] == "4cd32c8beedd116c6385463d41cf9cba8b1d5438"
    assert validation["result_summary"]["parity_cases_passed"] == 6
    assert validation["result_summary"]["checkpoint_bytes_exact"] is True
    assert validation["result_summary"]["sample_bytes_exact"] is True
    assert validation["artifact"]["evidence_file_sha256"] == (
        "5958c67261e8c25e60d58891efd5d27f8e8bb6439852862064e831f630cbe56c"
    )
    evidence_path = REPO_ROOT / validation["artifact"]["permanent_evidence_path"]
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == validation["artifact"][
        "evidence_file_sha256"
    ]
    spec = get_adapter_spec("nrgboost")
    assert spec.validation_level.value == "native-parity-validated"
    assert spec.revision_status == "pinned-canonical-package-native-parity-validated"
    assert spec.benchmark_track == "experimental"
    assert spec.support_level == "unsupported"


def test_removed_unverified_checkpoints_are_recorded_and_absent() -> None:
    payload = _load_source_lock()
    components = payload["components"]

    assert isinstance(components, dict)
    tabsyn = components["tabsyn"]
    assert isinstance(tabsyn, dict)
    artifacts = tabsyn["removed_unverified_artifacts"]
    assert len(artifacts) == 7
    assert sum(artifact["bytes"] for artifact in artifacts) == 93_479_868
    for artifact in artifacts:
        assert len(artifact["sha256"]) == 64
        assert not (REPO_ROOT / artifact["original_path"]).exists()
