"""Capture cache-relevant code and software identity without host secrets."""

from __future__ import annotations

import hashlib
import locale
import os
import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from standardized_tabular_diffusion.evaluation.serialization import content_fingerprint, sha256_file

MATERIAL_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _git(repo_root: Path, *args: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def capture_repository_state(repo_root: str | Path) -> dict[str, Any]:
    """Return a content identity for tracked changes and relevant untracked files."""

    root = Path(repo_root).resolve()
    commit_raw = _git(root, "rev-parse", "HEAD")
    commit = "unknown" if commit_raw is None else commit_raw.decode("ascii", errors="replace").strip()
    diff = _git(root, "diff", "--binary", "HEAD", "--") or b""
    untracked_raw = _git(root, "ls-files", "--others", "--exclude-standard", "-z") or b""
    untracked: list[dict[str, str]] = []
    for raw_relative in untracked_raw.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        if candidate.is_file() and not candidate.is_symlink():
            untracked.append({"path": candidate.relative_to(root).as_posix(), "sha256": sha256_file(candidate)})
    untracked.sort(key=lambda item: item["path"])
    digest = hashlib.sha256()
    digest.update(diff)
    digest.update(content_fingerprint(untracked).encode("ascii"))
    dirty = bool(diff or untracked)
    if commit_raw is None:
        package_root = Path(__file__).resolve().parents[1]
        installed_files = [
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".json"}
        ]
        installed_identity = [
            {"path": path.relative_to(package_root).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(installed_files)
        ]
        installed_tree_sha256 = content_fingerprint(installed_identity)
        commit = f"installed-tree-{installed_tree_sha256[:16]}"
        digest.update(installed_tree_sha256.encode("ascii"))
        dirty = False
    material = {"commit": commit, "dirty": dirty, "patch_sha256": digest.hexdigest()}
    return {**material, "fingerprint": content_fingerprint(material), "untracked_file_count": len(untracked)}


def capture_software_profile(repo_root: str | Path) -> dict[str, Any]:
    """Capture a deterministic package inventory and material runtime settings."""

    packages: dict[str, str] = {}
    for distribution in distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            name = None
        if name:
            packages[name.casefold().replace("_", "-")] = distribution.version
    inventory = [{"name": name, "version": packages[name]} for name in sorted(packages)]
    material_environment = {key: os.environ[key] for key in MATERIAL_ENVIRONMENT_KEYS if key in os.environ}
    repository = capture_repository_state(repo_root)
    material = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": inventory,
        "material_environment": material_environment,
        "repository": {
            "commit": repository["commit"],
            "dirty": repository["dirty"],
            "patch_sha256": repository["patch_sha256"],
        },
        "locale": locale.getlocale()[0] or "unknown",
    }
    return {
        "software_profile_schema_version": "1.0.0",
        **material,
        "fingerprint": content_fingerprint(material),
        "code_fingerprint": repository["fingerprint"],
    }
