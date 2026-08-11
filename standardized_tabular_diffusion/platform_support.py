"""Repository-wide release-platform policy and environment classification."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

PRIMARY_RELEASE_SYSTEM = "Windows"
PRIMARY_RELEASE_PYTHON = (3, 11)
SECONDARY_COMPATIBILITY_SYSTEMS = ("Linux",)

EnvironmentTier = Literal["primary", "secondary", "unsupported"]


@dataclass(frozen=True)
class ReleaseEnvironment:
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
    """Classify an environment without conflating release support and source parity."""

    resolved_system = platform.system() if system is None else system
    resolved_python = sys.version_info[:2] if python_version is None else tuple(python_version[:2])
    if resolved_system == PRIMARY_RELEASE_SYSTEM and resolved_python == PRIMARY_RELEASE_PYTHON:
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


def is_primary_release_environment() -> bool:
    return classify_release_environment() == "primary"

