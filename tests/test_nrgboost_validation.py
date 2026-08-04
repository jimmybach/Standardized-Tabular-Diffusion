from __future__ import annotations

import platform

import pytest

from standardized_tabular_diffusion.validation import nrgboost as nrgboost_validation

pytestmark = pytest.mark.adapter


def test_nrgboost_protocol_constants_lock_the_official_release() -> None:
    assert nrgboost_validation.PACKAGE_NAME == "nrgboost"
    assert nrgboost_validation.PACKAGE_VERSION == "0.0.3"
    assert nrgboost_validation.UPSTREAM_TAG == "v0.0.3"
    assert nrgboost_validation.UPSTREAM_COMMIT == "feef73a3edb20b911c2f7214b13f810909ef20ad"
    assert nrgboost_validation.UPSTREAM_TREE == "e3e84bacc7236a36af93c3d214de14bd308d2767"
    assert nrgboost_validation.WHEEL_SHA256 == (
        "dfe30829ceaf2d0d0ec03eab1744838bed857d56919238e7243c9fb7f273e1fb"
    )
    assert nrgboost_validation.LICENSE_EXPRESSION == "MIT"
    assert nrgboost_validation.SEED_CASES == (0, 19, 73)
    assert nrgboost_validation.VARIANTS == ("classification", "regression")


def test_nrgboost_validation_runtime_is_bounded_and_deterministic() -> None:
    assert nrgboost_validation.TRAINING_PARAMS["num_trees"] == 3
    assert nrgboost_validation.TRAINING_PARAMS["num_threads"] == 1
    assert nrgboost_validation.SAMPLING_PARAMS["num_threads"] == 1
    assert nrgboost_validation.SAMPLING_PARAMS["output_full_chain"] is False
    assert nrgboost_validation.EXPECTED_SAMPLE_ROWS == 16
    extras = nrgboost_validation._adapter_extra()
    assert extras["training_temperature"] == nrgboost_validation.TRAINING_PARAMS["temperature"]
    assert extras["num_steps"] == nrgboost_validation.SAMPLING_PARAMS["num_steps"]


def test_nrgboost_case_gate_fails_closed() -> None:
    comparisons = {
        "adapter_manifests_valid": True,
        "adapter_metadata_exact": True,
        "checkpoint_bytes_exact": True,
        "sample_bytes_exact": True,
        "checkpoint_structure_exact": True,
        "native_global_numpy_state_unchanged": True,
        "adapter_global_numpy_state_unchanged": True,
        "samples": {
            "rows": nrgboost_validation.EXPECTED_SAMPLE_ROWS,
            "columns_exact": True,
            "frame_exact": True,
            "finite_numerical": True,
            "categorical_domains_valid": True,
            "missing_values": 0,
        },
    }
    assert nrgboost_validation._case_passed(comparisons) is True
    comparisons["checkpoint_bytes_exact"] = False
    assert nrgboost_validation._case_passed(comparisons) is False


def test_nrgboost_authoritative_environment_rejects_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(
        nrgboost_validation.importlib.metadata,
        "version",
        lambda name: nrgboost_validation.EXPECTED_DISTRIBUTION_VERSIONS[name],
    )
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "platform", lambda: "Windows-test")
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.15")
    with pytest.raises(RuntimeError, match="requires Linux"):
        nrgboost_validation._verify_environment()
