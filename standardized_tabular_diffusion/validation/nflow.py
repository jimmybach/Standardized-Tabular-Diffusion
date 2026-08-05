"""Executable recipe-parity protocol for the official nflows package."""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import subprocess
import tarfile
import traceback
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_bytes, read_json, sha256_file
from standardized_tabular_diffusion.interfaces import DatasetSpec, RunSpec
from standardized_tabular_diffusion.models.structured_baselines import NFlowAdapter

PROTOCOL_ID = "nflows-maf-tabular-recipe-parity-v1"
PACKAGE_NAME = "nflows"
PACKAGE_VERSION = "0.14"
SDIST_FILENAME = "nflows-0.14.tar.gz"
SDIST_BYTES = 45_784
SDIST_SHA256 = "6299844a62f9999fcdf2d95cb2d01c091a50136bd17826e303aba646b2d11b55"
SDIST_URL = (
    "https://files.pythonhosted.org/packages/bd/16/"
    "a484db41aab28332f42080435c9342fa87cfc9a4fce5495521ea1e80ca27/"
    "nflows-0.14.tar.gz"
)
UPSTREAM_REPOSITORY = "https://github.com/bayesiains/nflows"
UPSTREAM_TAG = "v0.14"
UPSTREAM_COMMIT = NFlowAdapter.upstream_commit
UPSTREAM_TREE = NFlowAdapter.upstream_tree
LICENSE_EXPRESSION = "MIT"
SOURCE_LICENSE_SHA256 = "74a24abd8e13ac55286f5a8396a88c20da9f67a64cbc5daa8999f31843a8b948"
SOURCE_LICENSE_GIT_BLOB = "785b65b6b446af620425caf7b27b3e1585a74720"
EXPECTED_ARCHIVE_MEMBERS = 96
EXPECTED_ARCHIVE_FILES = 80
EXPECTED_PACKAGE_FILES = 42
EXPECTED_PACKAGE_AGGREGATE_SHA256 = "e87ed4bf20415a470c531592945e662a8026f4632d892cfa9d9de58a03766721"
EXPECTED_CRITICAL_FILES = {f"nflows/{path}": digest for path, digest in NFlowAdapter.runtime_file_sha256.items()}
EXPECTED_REQUIREMENTS = ["matplotlib", "numpy", "tensorboard", "torch", "tqdm"]
EXPECTED_DISTRIBUTION_VERSIONS = {
    "matplotlib": "3.9.2",
    "nflows": PACKAGE_VERSION,
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "setuptools": "80.10.2",
    "tensorboard": "2.18.0",
    "torch": "2.3.0+cpu",
    "tqdm": "4.66.5",
    "wheel": "0.45.1",
}
EXPECTED_SOURCE_ROWS = 48
EXPECTED_SAMPLE_ROWS = 13
SEED_CASES = (0, 19, 73)
VARIANTS = ("binary", "multiclass", "regression")
RECIPE = {
    "num_layers": 2,
    "hidden_features": 16,
    "num_blocks": 1,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 3,
    "num_threads": 1,
    "transform_order": "random-permutation-then-masked-affine-autoregressive",
    "base_distribution": "standard-normal",
    "context_features": None,
    "use_residual_blocks": True,
    "random_mask": False,
    "activation": "relu",
    "dropout_probability": 0.0,
    "use_batch_norm": False,
    "optimizer": "adam",
    "shuffle": True,
    "dtype": "float32",
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_record_hash(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _package_aggregate(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _archive_member_path(name: str) -> str:
    root_name = f"nflows-{PACKAGE_VERSION}"
    if name == root_name:
        return ""
    root = f"{root_name}/"
    if not name.startswith(root):
        raise ValueError(f"Unexpected root in nflows source archive: {name!r}")
    return name[len(root) :]


def _verify_sdist(sdist_path: Path) -> dict[str, Any]:
    if sdist_path.is_symlink() or not sdist_path.is_file():
        raise ValueError(f"nflows source distribution must be a regular non-symlinked file: {sdist_path}")
    if sdist_path.name != SDIST_FILENAME:
        raise ValueError(f"Expected source archive {SDIST_FILENAME!r}, observed {sdist_path.name!r}")
    observed_sha256 = sha256_file(sdist_path)
    if observed_sha256 != SDIST_SHA256 or sdist_path.stat().st_size != SDIST_BYTES:
        raise ValueError(
            "nflows source archive identity differs from the PyPI lock: "
            f"sha256={observed_sha256}, bytes={sdist_path.stat().st_size}"
        )
    files: dict[str, bytes] = {}
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != EXPECTED_ARCHIVE_MEMBERS:
            raise ValueError(
                f"nflows archive member count mismatch: expected={EXPECTED_ARCHIVE_MEMBERS}, observed={len(members)}"
            )
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe path in nflows source archive: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported link/device member in nflows source archive: {member.name!r}")
            relative = _archive_member_path(member.name)
            if member.isfile():
                if relative in files:
                    raise ValueError(f"Duplicate file in nflows source archive: {relative!r}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read nflows archive member: {member.name!r}")
                files[relative] = extracted.read()
    if len(files) != EXPECTED_ARCHIVE_FILES:
        raise ValueError(f"nflows source archive file count differs from the lock: observed={len(files)}")
    package_files = {name: payload for name, payload in files.items() if name.startswith("nflows/")}
    if len(package_files) != EXPECTED_PACKAGE_FILES:
        raise ValueError("nflows package file count differs from the locked source/tag comparison")
    aggregate = _package_aggregate(package_files)
    if aggregate != EXPECTED_PACKAGE_AGGREGATE_SHA256:
        raise ValueError("nflows package files differ from the locked v0.14 tag content")
    for name, expected in EXPECTED_CRITICAL_FILES.items():
        if name not in files or _sha256_bytes(files[name]) != expected:
            raise ValueError(f"nflows critical runtime file differs from the lock: {name}")
    metadata = Parser().parsestr(files["PKG-INFO"].decode("utf-8"))
    if (
        metadata.get("Name") != PACKAGE_NAME
        or metadata.get("Version") != PACKAGE_VERSION
        or metadata.get("License") != LICENSE_EXPRESSION
        or metadata.get("Home-page") != UPSTREAM_REPOSITORY
        or metadata.get("Download-URL") != f"{UPSTREAM_REPOSITORY}/archive/{UPSTREAM_TAG}.tar.gz"
        or files.get("nflows.egg-info/requires.txt", b"").decode("utf-8").splitlines() != EXPECTED_REQUIREMENTS
    ):
        raise ValueError("nflows source metadata differs from the locked release")
    license_members = sorted(name for name in files if PurePosixPath(name).name.lower().startswith("license"))
    if license_members:
        raise ValueError("The audited nflows 0.14 sdist unexpectedly gained a license file; review the new artifact")
    return {
        "filename": sdist_path.name,
        "bytes": sdist_path.stat().st_size,
        "sha256": observed_sha256,
        "archive_members_verified": len(members),
        "regular_files_verified": len(files),
        "package_files_verified": len(package_files),
        "package_aggregate_sha256": aggregate,
        "critical_runtime_files_verified": len(EXPECTED_CRITICAL_FILES),
        "metadata_license": LICENSE_EXPRESSION,
        "license_file_in_sdist": False,
        "source_tag_license_sha256": SOURCE_LICENSE_SHA256,
        "source_tag_license_git_blob": SOURCE_LICENSE_GIT_BLOB,
        "source_commit": UPSTREAM_COMMIT,
        "source_tree": UPSTREAM_TREE,
        "source_tag": UPSTREAM_TAG,
        "tag_type": "lightweight",
        "commit_signature": "unsigned",
        "pypi_trusted_publishing": False,
        "audited_source_relation": (
            "All 42 nflows package files are byte-exact between the PyPI sdist and locked v0.14 Git tree. "
            "The PyPI sdist omits the tag's MIT LICENSE.md file; its metadata still declares MIT."
        ),
    }


def _verify_installed_distribution() -> dict[str, Any]:
    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    if distribution.version != PACKAGE_VERSION:
        raise ValueError(f"Installed nflows version differs from the lock: {distribution.version}")
    metadata = distribution.metadata
    if metadata.get("Name") != PACKAGE_NAME or metadata.get_all("Requires-Dist", []) != EXPECTED_REQUIREMENTS:
        raise ValueError("Installed nflows metadata differs from the audited source release")
    record_path = Path(distribution.locate_file("nflows-0.14.dist-info/RECORD"))
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError("Installed nflows RECORD is missing or unsafe")
    record_rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    verified_hashes = 0
    for row in record_rows:
        if len(row) != 3:
            raise ValueError("Installed nflows RECORD contains an invalid row")
        name, digest_field, size_field = row
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Installed nflows RECORD contains an unsafe path: {name!r}")
        installed = Path(distribution.locate_file(name))
        if installed.is_symlink() or not installed.is_file():
            raise ValueError(f"Installed nflows file is missing or unsafe: {name}")
        if digest_field:
            algorithm, encoded = digest_field.split("=", 1)
            if algorithm != "sha256" or hashlib.sha256(installed.read_bytes()).digest() != _decode_record_hash(encoded):
                raise ValueError(f"Installed nflows RECORD hash mismatch: {name}")
            if int(size_field) != installed.stat().st_size:
                raise ValueError(f"Installed nflows RECORD size mismatch: {name}")
            verified_hashes += 1
    package_root = Path(importlib.import_module("nflows").__file__).resolve().parent
    package_files: dict[str, bytes] = {}
    for path in package_root.rglob("*.py"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Installed nflows package contains an unsafe source path: {path}")
        relative = f"nflows/{path.relative_to(package_root).as_posix()}"
        package_files[relative] = path.read_bytes()
    if len(package_files) != EXPECTED_PACKAGE_FILES or _package_aggregate(package_files) != (
        EXPECTED_PACKAGE_AGGREGATE_SHA256
    ):
        raise ValueError("Installed nflows package files differ from the audited source distribution")
    return {
        "version": distribution.version,
        "record_rows": len(record_rows),
        "record_hashes_verified": verified_hashes,
        "package_files_verified": len(package_files),
        "package_aggregate_sha256": _package_aggregate(package_files),
        "requirements": metadata.get_all("Requires-Dist", []),
    }


def _verify_environment() -> dict[str, Any]:
    if platform.system() != "Linux":
        raise RuntimeError("The authoritative NFlow parity protocol requires Linux.")
    if tuple(map(int, platform.python_version_tuple()[:2])) != (3, 11):
        raise RuntimeError("The authoritative NFlow parity protocol requires Python 3.11.")
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTION_VERSIONS}
    if observed != EXPECTED_DISTRIBUTION_VERSIONS:
        raise RuntimeError(f"Frozen NFlow environment differs from the lock: {observed}")
    torch = importlib.import_module("torch")
    if torch.version.cuda is not None:
        raise RuntimeError("The authoritative NFlow recipe requires the CPU-only PyTorch build.")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "distributions": observed,
        "torch_cuda": torch.version.cuda,
        "torch_num_threads_before_protocol": torch.get_num_threads(),
    }


def _fixture(variant: str) -> tuple[pd.DataFrame, str, list[str], list[str], list[str]]:
    index = np.arange(EXPECTED_SOURCE_ROWS)
    value = 0.18 * index + np.sin(index / 3.0) + (index % 4) * 0.07
    constant = np.full(EXPECTED_SOURCE_ROWS, 3.25)
    segment = np.asarray(["alpha", "beta", "gamma", "alpha"] * 12)
    if variant == "binary":
        target = np.where((index + (segment == "gamma")) % 3 == 0, "yes", "no")
        frame = pd.DataFrame({"value": value, "constant": constant, "segment": segment, "target": target})
        return frame, "classification", ["value", "constant"], ["segment"], ["target"]
    if variant == "multiclass":
        levels = np.asarray(["low", "middle", "high"])
        target = levels[(index + (segment == "beta").astype(int)) % 3]
        frame = pd.DataFrame({"value": value, "constant": constant, "segment": segment, "target": target})
        return frame, "classification", ["value", "constant"], ["segment"], ["target"]
    if variant == "regression":
        target = 1.7 * value - 0.4 * (segment == "beta") + np.cos(index / 5.0)
        frame = pd.DataFrame({"value": value, "constant": constant, "segment": segment, "target": target})
        return frame, "regression", ["value", "constant"], ["segment"], ["target"]
    raise ValueError(f"Unknown NFlow validation variant: {variant}")


def _official_api() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    return {
        "torch": torch,
        "StandardNormal": getattr(importlib.import_module("nflows.distributions"), "StandardNormal"),
        "Flow": getattr(importlib.import_module("nflows.flows"), "Flow"),
        "CompositeTransform": getattr(importlib.import_module("nflows.transforms"), "CompositeTransform"),
        "RandomPermutation": getattr(importlib.import_module("nflows.transforms"), "RandomPermutation"),
        "MaskedAffineAutoregressiveTransform": getattr(
            importlib.import_module("nflows.transforms.autoregressive"), "MaskedAffineAutoregressiveTransform"
        ),
    }


@contextlib.contextmanager
def _torch_scope(torch: Any, seed: int):
    previous_state = torch.random.get_rng_state()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(RECIPE["num_threads"])
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(previous_state)
        torch.set_num_threads(previous_threads)


def _build_direct_flow(num_features: int, api: dict[str, Any]) -> Any:
    transforms = []
    for _ in range(RECIPE["num_layers"]):
        transforms.append(api["RandomPermutation"](features=num_features, dim=1))
        transforms.append(
            api["MaskedAffineAutoregressiveTransform"](
                features=num_features,
                hidden_features=RECIPE["hidden_features"],
                context_features=None,
                num_blocks=RECIPE["num_blocks"],
                use_residual_blocks=True,
                random_mask=False,
                activation=api["torch"].nn.functional.relu,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
    return api["Flow"](
        transform=api["CompositeTransform"](transforms),
        distribution=api["StandardNormal"](shape=[num_features]),
    )


def _direct_path(frame: pd.DataFrame, dataset_spec: DatasetSpec, seed: int) -> dict[str, Any]:
    numeric_columns = list(dataset_spec.numerical_columns)
    categorical_columns = list(dataset_spec.categorical_columns)
    if dataset_spec.task_type == "regression":
        numeric_columns.extend(dataset_spec.target_columns)
    else:
        categorical_columns.extend(dataset_spec.target_columns)
    scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    blocks: list[np.ndarray] = []
    if numeric_columns:
        blocks.append(scaler.fit_transform(frame[numeric_columns].astype(float)))
    if categorical_columns:
        blocks.append(encoder.fit_transform(frame[categorical_columns].astype(str)).astype(np.float32))
    train_array = np.concatenate(blocks, axis=1).astype(np.float32)
    preprocessing = {
        "column_names": dataset_spec.column_names,
        "numerical_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_columns": dataset_spec.target_columns,
        "numerical_mean": [float(value) for value in scaler.mean_] if numeric_columns else [],
        "numerical_scale": [float(value) for value in scaler.scale_] if numeric_columns else [],
        "categorical_levels": {
            column: [str(value) for value in encoder.categories_[idx]]
            for idx, column in enumerate(categorical_columns)
        },
    }
    api = _official_api()
    torch = api["torch"]
    state_before = torch.random.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    losses: list[float] = []
    with _torch_scope(torch, seed):
        flow = _build_direct_flow(train_array.shape[1], api)
        flow.train()
        optimizer = torch.optim.Adam(
            flow.parameters(),
            lr=RECIPE["learning_rate"],
            betas=tuple(RECIPE["adam_betas"]),
            eps=RECIPE["adam_eps"],
            weight_decay=RECIPE["weight_decay"],
            amsgrad=RECIPE["amsgrad"],
        )
        tensor = torch.tensor(train_array, dtype=torch.float32)
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tensor),
            batch_size=RECIPE["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        for _ in range(RECIPE["epochs"]):
            epoch_losses: list[float] = []
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = -flow.log_prob(batch).mean()
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("Direct official nflows training produced a non-finite loss.")
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            losses.append(float(np.mean(epoch_losses)))
    state_after_training = torch.random.get_rng_state().clone()
    flow.eval()
    with _torch_scope(torch, seed):
        with torch.no_grad():
            raw_samples = flow.sample(EXPECTED_SAMPLE_ROWS).detach().cpu().numpy()
    state_after_sampling = torch.random.get_rng_state().clone()
    output = pd.DataFrame(index=range(EXPECTED_SAMPLE_ROWS))
    start = 0
    if numeric_columns:
        stop = len(numeric_columns)
        restored_numeric = scaler.inverse_transform(raw_samples[:, :stop])
        for idx, column in enumerate(numeric_columns):
            output[column] = restored_numeric[:, idx]
        start = stop
    if categorical_columns:
        decoded = np.zeros((EXPECTED_SAMPLE_ROWS, len(categorical_columns)), dtype=np.float32)
        for idx, categories in enumerate(encoder.categories_):
            decoded[:, idx] = np.clip(np.round(raw_samples[:, start + idx]), 0, len(categories) - 1)
        recovered = encoder.inverse_transform(decoded)
        for idx, column in enumerate(categorical_columns):
            output[column] = recovered[:, idx]
    sample_frame = output[dataset_spec.column_names]
    sample_bytes = sample_frame.to_csv(index=False).encode("utf-8")
    return {
        "flow": flow,
        "state_arrays": NFlowAdapter._state_arrays(flow),
        "losses": losses,
        "preprocessing": preprocessing,
        "raw_samples": raw_samples,
        "sample_frame": pd.read_csv(io.BytesIO(sample_bytes)),
        "sample_bytes": sample_bytes,
        "global_state_unchanged": bool(
            torch.equal(state_before, state_after_training) and torch.equal(state_before, state_after_sampling)
        ),
        "thread_count_unchanged": torch.get_num_threads() == threads_before,
    }


def _case_passed(comparisons: dict[str, Any]) -> bool:
    return bool(
        comparisons.get("preprocessing_exact")
        and comparisons.get("losses_exact")
        and comparisons.get("state_tensor_names_exact")
        and comparisons.get("state_tensors_exact")
        and comparisons.get("raw_samples_exact")
        and comparisons.get("sample_frame_exact")
        and comparisons.get("sample_bytes_exact")
        and comparisons.get("adapter_manifests_valid")
        and comparisons.get("checkpoint_metadata_valid")
        and comparisons.get("safe_json_numpy_checkpoint")
        and comparisons.get("row_level_training_data_absent")
        and comparisons.get("privacy_not_overclaimed")
        and comparisons.get("artifact_access_control_required")
        and comparisons.get("direct_global_torch_state_unchanged")
        and comparisons.get("adapter_global_torch_state_unchanged")
        and comparisons.get("thread_count_unchanged")
        and comparisons.get("rows") == EXPECTED_SAMPLE_ROWS
        and comparisons.get("columns_exact")
        and comparisons.get("missing_values") == 0
        and comparisons.get("finite_numerical")
        and comparisons.get("categorical_domains_valid")
    )


def _run_case(root: Path, output_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    frame, task_type, numerical, categorical, targets = _fixture(variant)
    case_dir = output_dir / f"{variant}-seed-{seed}"
    native_dir = case_dir / "direct"
    adapter_dir = case_dir / "adapter"
    native_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    train_path = case_dir / "train.csv"
    metadata_path = case_dir / "dataset.json"
    atomic_write_bytes(train_path, frame.to_csv(index=False).encode("utf-8"))
    atomic_write_bytes(metadata_path, b"{}\n")
    dataset_spec = DatasetSpec(
        name=f"nflow-{variant}",
        task_type=task_type,
        column_names=list(frame.columns),
        numerical_columns=numerical,
        categorical_columns=categorical,
        target_columns=targets,
        metadata_path=metadata_path,
        train_data_path=train_path,
    )
    direct = _direct_path(pd.read_csv(train_path), dataset_spec, seed)
    adapter = NFlowAdapter(root)
    train_spec = RunSpec(
        model="nflow",
        dataset=dataset_spec.name,
        output_dir=adapter_dir,
        device="cpu",
        seed=seed,
        extra={"dataset_spec": dataset_spec.to_dict(), **RECIPE},
    )
    torch = importlib.import_module("torch")
    adapter_state_before = torch.random.get_rng_state().clone()
    adapter_threads_before = torch.get_num_threads()
    adapter.train(train_spec)
    adapter_state_after_training = torch.random.get_rng_state().clone()
    checkpoint_path = adapter_dir / NFlowAdapter.checkpoint_filename
    payload = read_json(checkpoint_path)
    with np.load(adapter._weights_path(checkpoint_path), allow_pickle=False) as archive:
        adapter_arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    sample_spec = RunSpec(
        model="nflow",
        dataset=dataset_spec.name,
        output_dir=adapter_dir,
        device="cpu",
        seed=seed,
        num_samples=EXPECTED_SAMPLE_ROWS,
        checkpoint_path=checkpoint_path,
        extra={"dataset_spec": dataset_spec.to_dict()},
    )
    adapter.sample(sample_spec)
    adapter_state_after_sampling = torch.random.get_rng_state().clone()
    sample_path = adapter_dir / "samples.csv"
    adapter_frame = pd.read_csv(sample_path)
    sample_metadata = read_json(adapter_dir / "nflow_sample_metadata.json")
    checkpoint_metadata = read_json(adapter._metadata_path(checkpoint_path))
    categorical_domains = {
        column: set(frame[column].astype(str).unique()) for column in [*categorical, *(targets if task_type == "classification" else [])]
    }
    numerical_output = [*numerical, *(targets if task_type == "regression" else [])]
    direct_names = set(direct["state_arrays"])
    adapter_names = set(adapter_arrays)
    comparisons = {
        "preprocessing_exact": payload["preprocessing"] == direct["preprocessing"],
        "losses_exact": payload["training"]["epoch_mean_negative_log_likelihood"] == direct["losses"],
        "state_tensor_names_exact": direct_names == adapter_names,
        "state_tensors_exact": direct_names == adapter_names
        and all(np.array_equal(direct["state_arrays"][name], adapter_arrays[name]) for name in direct_names),
        "raw_samples_exact": sample_metadata["raw_sample_sha256"] == _sha256_bytes(direct["raw_samples"].tobytes()),
        "sample_frame_exact": adapter_frame.equals(direct["sample_frame"]),
        "sample_bytes_exact": sample_path.read_bytes() == direct["sample_bytes"],
        "adapter_manifests_valid": read_json(adapter_dir / "artifacts.json")["model"] == "nflow"
        and sample_metadata["sample_sha256"] == sha256_file(sample_path),
        "checkpoint_metadata_valid": checkpoint_metadata["checkpoint_sha256"] == sha256_file(checkpoint_path)
        and checkpoint_metadata["weights_sha256"] == sha256_file(adapter._weights_path(checkpoint_path)),
        "safe_json_numpy_checkpoint": payload["privacy"]["code_executing_checkpoint"] is False
        and payload["weights"]["format"] == "numpy-npz-no-pickle"
        and checkpoint_path.suffix == ".json"
        and adapter._weights_path(checkpoint_path).suffix == ".npz",
        "row_level_training_data_absent": payload["privacy"]["contains_row_level_training_data"] is False
        and "rows" not in payload,
        "privacy_not_overclaimed": payload["privacy"]["privacy_guarantee"] == "none",
        "artifact_access_control_required": payload["privacy"]["trained_artifact_access_control_required"] is True,
        "direct_global_torch_state_unchanged": direct["global_state_unchanged"],
        "adapter_global_torch_state_unchanged": bool(
            torch.equal(adapter_state_before, adapter_state_after_training)
            and torch.equal(adapter_state_before, adapter_state_after_sampling)
        ),
        "thread_count_unchanged": direct["thread_count_unchanged"]
        and torch.get_num_threads() == adapter_threads_before,
        "rows": len(adapter_frame),
        "columns_exact": list(adapter_frame.columns) == dataset_spec.column_names,
        "missing_values": int(adapter_frame.isna().sum().sum()),
        "finite_numerical": not numerical_output
        or bool(np.isfinite(adapter_frame[numerical_output].to_numpy(dtype=float)).all()),
        "categorical_domains_valid": all(
            set(adapter_frame[column].astype(str).unique()) <= domain
            for column, domain in categorical_domains.items()
        ),
    }
    return {
        "variant": variant,
        "seed": seed,
        "status": "pass" if _case_passed(comparisons) else "fail",
        "source_rows": len(frame),
        "sample_rows": len(adapter_frame),
        "comparisons": comparisons,
        "artifacts": {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "weights_sha256": sha256_file(adapter._weights_path(checkpoint_path)),
            "sample_sha256": sha256_file(sample_path),
            "tensor_count": len(adapter_arrays),
        },
    }


def run_protocol(repo_root: Path, output_dir: Path, sdist_path: Path) -> dict[str, Any]:
    environment = _verify_environment()
    source = _verify_sdist(sdist_path)
    installed = _verify_installed_distribution()
    cases = [_run_case(repo_root, output_dir, variant, seed) for variant in VARIANTS for seed in SEED_CASES]
    source_after = _verify_installed_distribution()
    passed = len(cases) == len(VARIANTS) * len(SEED_CASES) and all(case["status"] == "pass" for case in cases)
    repository_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": "nflow",
        "status": "pass" if passed else "fail",
        "reproduction_target": "official-nflows-package-plus-declared-tabular-maf-recipe",
        "claim_boundary": (
            "Validates exact official-library recipe parity only; this is not a paper-native tabular synthesizer "
            "reproduction and does not establish benchmark eligibility or release support."
        ),
        "repository_commit": repository_commit,
        "environment": environment,
        "environment_lock": {
            "path": "requirements-nflow-validation.txt",
            "sha256": sha256_file(repo_root / "requirements-nflow-validation.txt"),
        },
        "source": {
            "authority": "canonical-library",
            "repository": UPSTREAM_REPOSITORY,
            "source_distribution": source,
            "installed_distribution": installed,
            "license": LICENSE_EXPRESSION,
            "upstream_source_modified": False,
        },
        "recipe": RECIPE,
        "cases": cases,
        "summary": {
            "expected_cases": len(VARIANTS) * len(SEED_CASES),
            "passed_cases": sum(case["status"] == "pass" for case in cases),
            "all_exact_comparisons_passed": passed,
            "safe_non_executable_checkpoint": all(
                case["comparisons"]["safe_json_numpy_checkpoint"] for case in cases
            ),
        },
        "source_unchanged_after_validation": installed == source_after,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--sdist-path", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence: dict[str, Any]
    try:
        evidence = run_protocol(args.repo_root.resolve(), args.output_dir.resolve(), args.sdist_path.resolve())
    except Exception as exc:
        evidence = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "model_id": "nflow",
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    atomic_write_bytes(args.evidence_path, (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return 0 if evidence.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
