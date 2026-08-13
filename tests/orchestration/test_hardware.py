from __future__ import annotations

import copy

import pytest

from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.orchestration.hardware import (
    IncompatibleHardwareProfileError,
    assert_efficiency_compatible,
)

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_hardware_profile_is_schema_valid_and_self_compatible(hardware_profile: dict) -> None:
    validate_instance("hardware-profile", hardware_profile)
    assert assert_efficiency_compatible([hardware_profile, hardware_profile]) == hardware_profile["comparison_key"]
    assert hardware_profile["official_efficiency_eligible"] is False


def test_cross_profile_efficiency_comparison_fails_closed(hardware_profile: dict) -> None:
    different = copy.deepcopy(hardware_profile)
    different["comparison_key"] = "f" * 64
    with pytest.raises(IncompatibleHardwareProfileError, match="different hardware profiles"):
        assert_efficiency_compatible([hardware_profile, different])


def test_efficiency_records_require_a_comparison_key() -> None:
    with pytest.raises(IncompatibleHardwareProfileError, match="requires a hardware comparison key"):
        assert_efficiency_compatible([{"wall_seconds": 1.0}])
