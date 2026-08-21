"""Cross-platform core CI checks that avoid shell-specific inline programs."""

from __future__ import annotations

import argparse
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath

_LOCAL_PATH_PATTERNS = (
    re.compile(rb"(?i)\b[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+[\\/]"),
    re.compile(rb"/(?:home|Users)/[^/\s]+/"),
)
_TEXT_SUFFIXES = {".cff", ".cfg", ".csv", ".json", ".md", ".toml", ".txt", ".yaml", ".yml"}
_FORBIDDEN_DISTRIBUTION_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "materialized_datasets",
    "outputs",
    "research_inputs",
    "tmp",
}
_FORBIDDEN_CREDENTIAL_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
}
_FORBIDDEN_CREDENTIAL_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
_WINDOWS_RESERVED_MEMBER_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_WINDOWS_INVALID_MEMBER_CHARACTERS = frozenset('<>:"|?*')

REQUIRED_WHEEL_FILES = {
    "standardized_tabular_diffusion/resources/datasets/sources.json",
    "standardized_tabular_diffusion/resources/datasets/adult-uci-2-v1.json",
    "standardized_tabular_diffusion/resources/datasets/sick-uci-102-v1.json",
    "standardized_tabular_diffusion/resources/upstream/source-lock.json",
    "standardized_tabular_diffusion/resources/upstream/ctabgan-source-manifest.json",
    "standardized_tabular_diffusion/schemas/evaluation/atomic-result.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/admission-record.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/correction-record.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/dataset-summary.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/leaderboard-snapshot.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/legacy-import-record.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/legacy-standardized-summary.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/metric-registry-entry.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/hardware-profile.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/orchestration-run.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/orchestration-stage-record.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/software-profile.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/snapshot-manifest.schema.json",
    "standardized_tabular_diffusion/schemas/evaluation/snapshot-request.schema.json",
    "standardized_tabular_diffusion/resources/evaluation/metrics/legacy-tabstruct-aligned-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/protocols/development-p1.json",
    "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-dataset-scale-windows-gpu-stable-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/evaluators/p4-utility-stable-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/evaluators/p5-high-order-privacy-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/metrics/p5-excluded-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/metrics/p5-high-order-privacy-v1.json",
    "standardized_tabular_diffusion/resources/evaluation/protocols/p5-high-order-privacy.json",
    "standardized_tabular_diffusion/resources/evaluation/upstream/p5-sources.json",
    "standardized_tabular_diffusion/resources/evaluation/upstream/tabeval-p4-windows-gpu-runtime.json",
    "standardized_tabular_diffusion/resources/quickstart/train.csv",
    "standardized_tabular_diffusion/resources/quickstart/metadata.json",
}

REQUIRED_SDIST_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ARCHITECTURE.md",
    "docs/ARCHITECTURE.zh-CN.md",
    "docs/PROJECT_ROADMAP.md",
    "docs/PROJECT_ROADMAP.zh-CN.md",
    "docs/QUICKSTART.md",
    "docs/QUICKSTART.zh-CN.md",
    "docs/evaluation/P8_MIGRATION_AND_RELEASE.md",
    "docs/evaluation/P8_MIGRATION_AND_RELEASE.zh-CN.md",
    "configs/datasets/adult-uci-2-v1.json",
    "configs/smoke/arf-adult-smoke.json",
    "tests/test_p8_release_assets.py",
}


def verify_dependency_free_surface() -> None:
    import_check = """
import sys
import standardized_tabular_diffusion as package

assert package.list_models()
assert not {"numpy", "pandas", "sklearn", "torch"} & set(sys.modules)
"""
    with tempfile.TemporaryDirectory(prefix="std-tabular-core-ci-") as directory:
        subprocess.run([sys.executable, "-c", import_check], cwd=directory, check=True)
        subprocess.run(
            [sys.executable, "-m", "standardized_tabular_diffusion.cli", "--help"],
            cwd=directory,
            check=True,
        )


def _portable_member_key(name: str) -> str:
    """Validate one archive path on every host and return its Windows-portable collision key."""

    normalized = name.replace("\\", "/")
    raw_parts = normalized.split("/")
    if normalized.endswith("/"):
        raw_parts = raw_parts[:-1]
    path = PurePosixPath(normalized)
    if (
        name != normalized
        or not raw_parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or path.is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized) is not None
        or "\x00" in normalized
    ):
        raise RuntimeError(f"Distribution archive contains an unsafe path: {name!r}")
    for part in raw_parts:
        if (
            part.endswith((" ", "."))
            or any(character in _WINDOWS_INVALID_MEMBER_CHARACTERS or ord(character) < 32 for character in part)
            or part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_MEMBER_NAMES
        ):
            raise RuntimeError(f"Distribution archive contains a non-portable path: {name!r}")
    lowered_parts = {part.casefold() for part in raw_parts}
    if lowered_parts & _FORBIDDEN_DISTRIBUTION_PARTS:
        raise RuntimeError(f"Distribution archive contains a forbidden runtime/development path: {name!r}")
    basename = raw_parts[-1].casefold()
    if basename in _FORBIDDEN_CREDENTIAL_NAMES or path.suffix.casefold() in _FORBIDDEN_CREDENTIAL_SUFFIXES:
        raise RuntimeError(f"Distribution archive contains a credential-like file: {name!r}")
    return "/".join(part.casefold() for part in raw_parts)


def _reject_portable_member_collisions(names: list[str]) -> None:
    portable_keys = [_portable_member_key(name) for name in names]
    if len(set(portable_keys)) != len(portable_keys):
        raise RuntimeError("Distribution archive contains duplicate or non-portable-colliding member names")


def _validate_portable_text(name: str, payload: bytes) -> None:
    if Path(name).suffix.casefold() not in _TEXT_SUFFIXES:
        return
    for pattern in _LOCAL_PATH_PATTERNS:
        if pattern.search(payload):
            raise RuntimeError(f"Distribution archive contains a developer-local absolute path in {name!r}")


def inspect_wheel_contents(dist_directory: Path) -> None:
    wheels = sorted(dist_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel under {dist_directory}, observed {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        records = archive.infolist()
        names = {record.filename for record in records}
        if len(names) != len(records):
            raise RuntimeError("Wheel contains duplicate archive member names")
        _reject_portable_member_collisions([record.filename for record in records])
        for record in records:
            mode = record.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise RuntimeError(f"Wheel contains a symbolic link: {record.filename!r}")
            if not record.is_dir():
                _validate_portable_text(record.filename, archive.read(record))
    missing = sorted(REQUIRED_WHEEL_FILES - names)
    if missing:
        raise RuntimeError(f"Wheel is missing required files: {missing}")
    forbidden_prefixes = ("research_inputs/", "TabDDPM-main/", "TabDiff-main/", "TabSyn-main/")
    forbidden = sorted(name for name in names if name.startswith(forbidden_prefixes))
    if forbidden:
        raise RuntimeError(f"Wheel contains excluded reference-tree files: {forbidden}")


def _single_sdist(dist_directory: Path) -> Path:
    archives = sorted(dist_directory.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected exactly one sdist under {dist_directory}, observed {len(archives)}")
    return archives[0]


def inspect_sdist_contents(dist_directory: Path) -> None:
    archive_path = _single_sdist(dist_directory)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        records = archive.getmembers()
        names_with_types: set[str] = set()
        members: list[str] = []
        _reject_portable_member_collisions([member.name for member in records])
        for member in records:
            if member.name in names_with_types:
                raise RuntimeError("Sdist contains duplicate archive member names")
            names_with_types.add(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Sdist contains a link or device member: {member.name!r}")
            if member.isfile():
                members.append(member.name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"Sdist regular file cannot be read: {member.name!r}")
                _validate_portable_text(member.name, stream.read())
    roots = {name.split("/", maxsplit=1)[0] for name in members}
    if len(roots) != 1:
        raise RuntimeError(f"Expected one sdist root directory, observed {sorted(roots)}")
    prefix = next(iter(roots)) + "/"
    names = {name.removeprefix(prefix) for name in members}
    missing = sorted(REQUIRED_SDIST_FILES - names)
    if missing:
        raise RuntimeError(f"Sdist is missing required release files: {missing}")
    forbidden_prefixes = (
        "docs/evidence/",
        "research_inputs/",
        "TabDDPM-main/",
        "TabDiff-main/",
        "TabSyn-main/",
        "tmp/",
    )
    forbidden = sorted(name for name in names if name.startswith(forbidden_prefixes))
    if forbidden:
        raise RuntimeError(f"Sdist contains excluded reference-tree files: {forbidden}")


def install_single_wheel(dist_directory: Path) -> None:
    wheels = sorted(dist_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel under {dist_directory}, observed {len(wheels)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])],
        check=True,
    )


def install_single_sdist(dist_directory: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(_single_sdist(dist_directory))],
        check=True,
    )


def _run_installed_quickstart(python: Path, root: Path) -> None:
    output = root / "quickstart"
    subprocess.run(
        [str(python), "-m", "standardized_tabular_diffusion.cli", "quickstart", "--output", str(output)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "standardized_tabular_diffusion.cli",
            "validate-result",
            "--bundle",
            str(output / "evaluation-result"),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "standardized_tabular_diffusion.cli",
            "validate-leaderboard",
            "--snapshot",
            str(output / "diagnostic-snapshot"),
        ],
        cwd=root,
        check=True,
    )


def verify_installed_quickstart() -> None:
    """Exercise the current installation away from the repository checkout."""

    with tempfile.TemporaryDirectory(prefix="std-tabular-installed-p8-") as directory:
        _run_installed_quickstart(Path(sys.executable), Path(directory))


def verify_clean_distribution_install(dist_directory: Path, artifact_kind: str) -> None:
    """Install one artifact with quickstart dependencies into a fresh virtual environment."""

    if artifact_kind == "wheel":
        artifacts = sorted(dist_directory.glob("*.whl"))
    elif artifact_kind == "sdist":
        artifacts = sorted(dist_directory.glob("*.tar.gz"))
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact_kind!r}")
    if len(artifacts) != 1:
        raise RuntimeError(
            f"Expected exactly one {artifact_kind} under {dist_directory}, observed {len(artifacts)}"
        )

    with tempfile.TemporaryDirectory(prefix=f"std-tabular-clean-{artifact_kind}-") as directory:
        root = Path(directory)
        environment = root / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        requirement = f"{artifacts[0].resolve()}[quickstart]"
        subprocess.run([str(python), "-m", "pip", "install", requirement], cwd=root, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True)
        _run_installed_quickstart(python, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dependency-free-surface")
    subparsers.add_parser("installed-quickstart")
    wheel = subparsers.add_parser("inspect-wheel")
    wheel.add_argument("--dist-directory", type=Path, default=Path("dist"))
    sdist = subparsers.add_parser("inspect-sdist")
    sdist.add_argument("--dist-directory", type=Path, default=Path("dist"))
    install_wheel = subparsers.add_parser("install-wheel")
    install_wheel.add_argument("--dist-directory", type=Path, default=Path("dist"))
    install_sdist = subparsers.add_parser("install-sdist")
    install_sdist.add_argument("--dist-directory", type=Path, default=Path("dist"))
    clean_install = subparsers.add_parser(
        "clean-install",
        help="Install one distribution kind into a fresh environment and run the installed quickstart",
    )
    clean_install.add_argument("--dist-directory", type=Path, default=Path("dist"))
    clean_install.add_argument("--artifact", choices=["wheel", "sdist"], required=True)
    args = parser.parse_args()
    if args.command == "dependency-free-surface":
        verify_dependency_free_surface()
    elif args.command == "installed-quickstart":
        verify_installed_quickstart()
    elif args.command == "inspect-wheel":
        inspect_wheel_contents(args.dist_directory)
    elif args.command == "inspect-sdist":
        inspect_sdist_contents(args.dist_directory)
    elif args.command == "install-wheel":
        install_single_wheel(args.dist_directory)
    elif args.command == "install-sdist":
        install_single_sdist(args.dist_directory)
    else:
        verify_clean_distribution_install(args.dist_directory, args.artifact)


if __name__ == "__main__":
    main()
