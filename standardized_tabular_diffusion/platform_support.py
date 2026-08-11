"""Repository-wide release-platform policy and environment classification."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

PRIMARY_RELEASE_FAMILY_SYSTEM = "Windows"
PRIMARY_RELEASE_TARGET = "Windows 11 x86-64 with CPython 3.11"
# Backward-compatible name. This denotes the operating-system family, not an
# assertion that the current host is the exact Windows 11 release target.
PRIMARY_RELEASE_SYSTEM = PRIMARY_RELEASE_FAMILY_SYSTEM
PRIMARY_RELEASE_PYTHON = (3, 11)
SECONDARY_COMPATIBILITY_SYSTEMS = ("Linux",)

EnvironmentTier = Literal["primary", "secondary", "unsupported"]


@dataclass(frozen=True)
class ReleaseEnvironment:
    """OS-family classification; exact Windows-edition qualification is external evidence."""

    system: str
    machine: str
    python: str
    tier: EnvironmentTier

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def classify_release_environment(
    *,
    system: str | None = None,
    python_version: Sequence[int] | None = None,
) -> EnvironmentTier:
    """Classify OS-family compatibility without asserting an exact Windows edition."""

    resolved_system = platform.system() if system is None else system
    resolved_python = sys.version_info[:2] if python_version is None else tuple(python_version[:2])
    if resolved_system == PRIMARY_RELEASE_FAMILY_SYSTEM and resolved_python == PRIMARY_RELEASE_PYTHON:
        return "primary"
    if resolved_system in SECONDARY_COMPATIBILITY_SYSTEMS and resolved_python == PRIMARY_RELEASE_PYTHON:
        return "secondary"
    return "unsupported"


def current_release_environment() -> ReleaseEnvironment:
    return ReleaseEnvironment(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        tier=classify_release_environment(),
    )


def is_primary_release_family_environment() -> bool:
    """Return whether the host matches the Windows/Python primary family.

    Python's portable platform APIs do not reliably distinguish a Windows 11
    consumer installation from a Windows Server installation with the same
    kernel build. Exact Windows 11 qualification therefore requires recorded
    local or self-hosted runner evidence in addition to this classifier.
    """

    return classify_release_environment() == "primary"


def is_primary_release_environment() -> bool:
    """Backward-compatible alias for primary-family classification."""

    return is_primary_release_family_environment()
