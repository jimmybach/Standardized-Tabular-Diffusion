from __future__ import annotations

from importlib import import_module
from pathlib import Path


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    module = import_module(package_or_requirement)
    return str(Path(module.__file__).resolve().parent / resource_name)
