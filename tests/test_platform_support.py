from __future__ import annotations

from standardized_tabular_diffusion.platform_support import (
    PRIMARY_RELEASE_PYTHON,
    PRIMARY_RELEASE_SYSTEM,
    SECONDARY_COMPATIBILITY_SYSTEMS,
    classify_release_environment,
)


def test_windows_python_311_is_the_primary_release_environment() -> None:
    assert PRIMARY_RELEASE_SYSTEM == "Windows"
    assert PRIMARY_RELEASE_PYTHON == (3, 11)
    assert classify_release_environment(system="Windows", python_version=(3, 11, 15)) == "primary"


def test_linux_python_311_is_secondary_compatibility() -> None:
    assert SECONDARY_COMPATIBILITY_SYSTEMS == ("Linux",)
    assert classify_release_environment(system="Linux", python_version=(3, 11, 15)) == "secondary"


def test_other_python_or_operating_system_is_not_release_supported() -> None:
    assert classify_release_environment(system="Windows", python_version=(3, 13, 5)) == "unsupported"
    assert classify_release_environment(system="Darwin", python_version=(3, 11, 15)) == "unsupported"
