"""Dataset and protocol profile loading, identity, and legacy migration."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from standardized_tabular_diffusion.evaluation.schema import validate_instance
from standardized_tabular_diffusion.evaluation.serialization import (
    atomic_write_json,
    content_fingerprint,
    read_json,
    read_yaml_safe,
    sha256_file,
)
from standardized_tabular_diffusion.interfaces import DatasetSpec

PROTOCOL_RESOURCE_PACKAGE = "standardized_tabular_diffusion.resources.evaluation.protocols"


class ProfileError(ValueError):
    """Raised when a profile is invalid or cannot be resolved unambiguously."""


@dataclass(frozen=True)
class DatasetProfile:
    payload: dict[str, Any]
    source: str

    @property
    def dataset_id(self) -> str:
        return str(self.payload["dataset_id"])

    @property
    def dataset_profile_version(self) -> str:
        return str(self.payload["dataset_profile_version"])

    @property
    def identity(self) -> tuple[str, str]:
        return self.dataset_id, self.dataset_profile_version

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


@dataclass(frozen=True)
class ProtocolProfile:
    payload: dict[str, Any]
    source: str

    @property
    def protocol_id(self) -> str:
        return str(self.payload["protocol_id"])

    @property
    def protocol_version(self) -> str:
        return str(self.payload["protocol_version"])

    @property
    def identity(self) -> tuple[str, str]:
        return self.protocol_id, self.protocol_version

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def _read_structured(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return read_json(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return read_yaml_safe(path)
    raise ProfileError(f"Profile must use .json, .yaml, or .yml: {path}")


def load_dataset_profile(path: str | Path) -> DatasetProfile:
    source = Path(path)
    payload = _read_structured(source)
    if not isinstance(payload, dict):
        raise ProfileError(f"Dataset Profile must be an object: {source}")
    validate_instance("dataset-profile", payload)
    _validate_dataset_semantics(payload)
    if payload["official_eligible"] and payload["status"] != "frozen":
        raise ProfileError("Only a frozen Dataset Profile can claim official eligibility")
    if payload["official_eligible"] and payload["source_rights"]["redistribution_status"] == "unknown":
        raise ProfileError("A Dataset Profile with unknown rights cannot claim official eligibility")
    return DatasetProfile(copy.deepcopy(payload), str(source))


def load_protocol_profile(path: str | Path) -> ProtocolProfile:
    source = Path(path)
    payload = _read_structured(source)
    if not isinstance(payload, dict):
        raise ProfileError(f"Protocol Profile must be an object: {source}")
    validate_instance("protocol-profile", payload)
    _validate_protocol_semantics(payload)
    if payload["official_results_allowed"] and payload["status"] != "frozen":
        raise ProfileError("Only a frozen protocol can allow Official Results")
    return ProtocolProfile(copy.deepcopy(payload), str(source))


def _validate_dataset_semantics(payload: dict[str, Any]) -> None:
    columns = payload["columns"]
    column_names = [str(column["name"]) for column in columns]
    column_ids = [str(column["column_id"]) for column in columns]
    if len(set(column_names)) != len(column_names):
        raise ProfileError("Dataset Profile column names must be unique")
    if len(set(column_ids)) != len(column_ids):
        raise ProfileError("Dataset Profile column_id values must be unique")
    if any(len(set(column["roles"])) != len(column["roles"]) for column in columns):
        raise ProfileError("Dataset Profile column roles must not contain duplicates")

    table_contract = payload["table_contract"]
    declared_order = table_contract.get("canonical_column_order")
    canonical_names = [
        str(column["name"])
        for column in columns
        if "audit_only" not in column["roles"] and "ignored" not in column["roles"]
    ]
    if declared_order is not None and declared_order != canonical_names:
        raise ProfileError(
            "table_contract.canonical_column_order must exactly match non-audit, non-ignored columns"
        )
    if table_contract.get("unique_column_names") is False:
        raise ProfileError("A valid Dataset Profile cannot declare non-unique column names")
    declared_targets = table_contract.get("target_presence", [])
    if any(target not in column_names for target in declared_targets):
        raise ProfileError("table_contract.target_presence references an unknown column")

    views = payload["views"]
    view_ids = [view.get("view_id") for view in views]
    if any(not isinstance(view_id, str) for view_id in view_ids) or len(set(view_ids)) != len(view_ids):
        raise ProfileError("Dataset Profile views require unique string view_id values")
    if payload["dataset_view"] not in view_ids:
        raise ProfileError("dataset_view must resolve to exactly one declared view")

    columns_by_id = {str(column["column_id"]): column for column in columns}
    task_ids: set[str] = set()
    for task in payload["predictive_tasks"]:
        task_id = task.get("task_id")
        target_id = task.get("target_column_id")
        if not isinstance(task_id, str) or task_id in task_ids:
            raise ProfileError("predictive_tasks require unique string task_id values")
        task_ids.add(task_id)
        if target_id not in columns_by_id:
            raise ProfileError(f"Predictive task {task_id!r} references an unknown target column")
        if not {"primary_target", "secondary_target"} & set(columns_by_id[str(target_id)]["roles"]):
            raise ProfileError(f"Predictive task {task_id!r} must reference a declared target role")

    if payload["official_eligible"]:
        if payload["split"].get("frozen") is not True:
            raise ProfileError("An official-eligible Dataset Profile requires a frozen split")
        if payload["source_rights"]["rights_review"].get("decision") != "approved":
            raise ProfileError("An official-eligible Dataset Profile requires approved rights review")


def _validate_protocol_semantics(payload: dict[str, Any]) -> None:
    selections = payload["metric_selections"]
    identities = [(item["metric_id"], item["metric_version"]) for item in selections]
    if len(set(identities)) != len(identities):
        raise ProfileError("Protocol metric selections must have unique metric identities")
    if len(set(payload["dataset_suites"])) != len(payload["dataset_suites"]):
        raise ProfileError("Protocol dataset_suites must not contain duplicates")
    supported_versions = {
        "atomic_result": "1.0.0",
        "manifest": "1.0.0",
        "metadata": "1.0.0",
        "summary": "1.0.0",
        "stage_record": "1.0.0",
        "artifact_index": "1.0.0",
    }
    if payload["result_schema_versions"] != supported_versions:
        raise ProfileError("Protocol references unsupported result schema versions")


def _packaged_protocols() -> Iterable[ProtocolProfile]:
    for item in sorted(resources.files(PROTOCOL_RESOURCE_PACKAGE).iterdir(), key=lambda candidate: candidate.name):
        if item.name.endswith(".json"):
            with resources.as_file(item) as path:
                yield load_protocol_profile(path)


def list_protocol_profiles(directory: str | Path | None = None) -> tuple[ProtocolProfile, ...]:
    if directory is None:
        profiles = list(_packaged_protocols())
    else:
        root = Path(directory)
        if not root.is_dir():
            raise ProfileError(f"Protocol Profile directory does not exist: {root}")
        paths = sorted((*root.glob("*.json"), *root.glob("*.yaml"), *root.glob("*.yml")))
        profiles = [load_protocol_profile(path) for path in paths]
    indexed: dict[tuple[str, str], ProtocolProfile] = {}
    for profile in profiles:
        if profile.identity in indexed:
            raise ProfileError(f"Duplicate protocol identity: {profile.protocol_id}@{profile.protocol_version}")
        indexed[profile.identity] = profile
    return tuple(indexed[key] for key in sorted(indexed))


def list_dataset_profiles(directory: str | Path) -> tuple[DatasetProfile, ...]:
    root = Path(directory)
    if not root.is_dir():
        raise ProfileError(f"Dataset Profile directory does not exist: {root}")
    paths = sorted((*root.glob("*.json"), *root.glob("*.yaml"), *root.glob("*.yml")))
    indexed: dict[tuple[str, str], DatasetProfile] = {}
    for path in paths:
        profile = load_dataset_profile(path)
        if profile.identity in indexed:
            previous = indexed[profile.identity]
            raise ProfileError(
                f"Duplicate dataset identity {profile.dataset_id}@{profile.dataset_profile_version} "
                f"in {previous.source} and {path}"
            )
        indexed[profile.identity] = profile
    return tuple(indexed[key] for key in sorted(indexed))


def resolve_protocol(protocol_id: str, protocol_version: str, directory: str | Path | None = None) -> ProtocolProfile:
    for profile in list_protocol_profiles(directory):
        if profile.identity == (protocol_id, protocol_version):
            return profile
    raise KeyError(f"Unknown protocol identity: {protocol_id}@{protocol_version}")


def _portable_id(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or fallback


def _source_artifact(path: Path | None, role: str) -> dict[str, Any]:
    if path is None:
        return {"role": role, "name": None, "available": False, "sha256": None}
    available = path.is_file()
    return {
        "role": role,
        "name": path.name,
        "available": available,
        "sha256": sha256_file(path) if available else None,
    }


def import_legacy_dataset_spec(spec: DatasetSpec) -> DatasetProfile:
    """Convert upstream info.json metadata into an explicitly non-official profile."""

    source_artifacts = [
        _source_artifact(spec.metadata_path, "upstream-metadata"),
        _source_artifact(spec.train_data_path, "legacy-train"),
        _source_artifact(spec.val_data_path, "legacy-validation"),
        _source_artifact(spec.test_data_path, "legacy-test"),
    ]
    identity_material = {
        "name": spec.name,
        "task_type": spec.task_type,
        "column_names": spec.column_names,
        "numerical_columns": spec.numerical_columns,
        "categorical_columns": spec.categorical_columns,
        "target_columns": spec.target_columns,
        "metadata_sha256": source_artifacts[0]["sha256"],
    }
    dataset_id = _portable_id(spec.name, fallback="legacy-dataset")
    dataset_version = f"legacy-{content_fingerprint(identity_material)[:16]}"
    used_ids: set[str] = set()
    columns: list[dict[str, Any]] = []
    for index, name in enumerate(spec.column_names):
        candidate = _portable_id(name, fallback=f"column-{index}")
        column_id = candidate
        suffix = 2
        while column_id in used_ids:
            column_id = f"{candidate}-{suffix}"
            suffix += 1
        used_ids.add(column_id)
        is_target = name in spec.target_columns
        if name in spec.categorical_columns:
            semantic_type = "categorical"
        elif name in spec.numerical_columns:
            semantic_type = "continuous"
        else:
            semantic_type = "string"
        columns.append(
            {
                "name": name,
                "column_id": column_id,
                "semantic_type": semantic_type,
                "storage_type": "legacy-metadata-unspecified",
                "roles": ["primary_target" if is_target else "feature"],
                "nullable_raw": True,
                "nullable_model_input": False,
                "valid_domain": None,
                "category_vocabulary_source": None,
                "transformation_policy": "pending centralized preprocessing review",
                "inverse_transformation_policy": "pending review",
                "description": "Imported from upstream info.json; semantics require review.",
                "unit": None,
                "sensitivity": {"status": "not-reviewed", "sensitive": None, "quasi_identifier": None},
                "constraint_refs": [],
            }
        )

    target_ids = {column["name"]: column["column_id"] for column in columns}
    payload = {
        "profile_schema_version": "1.0.0",
        "dataset_profile_version": "0.1.0-legacy",
        "dataset_id": dataset_id,
        "display_name": spec.name,
        "dataset_version": dataset_version,
        "dataset_view": "legacy-canonical-v1",
        "dataset_family": "legacy-import",
        "status": "legacy-imported",
        "owners": ["repository-maintainers"],
        "review_record": {"status": "not-reviewed", "importer": "p1-info-json-importer"},
        "change_log": [{"version": "0.1.0-legacy", "change": "Imported without official eligibility."}],
        "suite_membership": ["diagnostic"],
        "official_eligible": False,
        "source_rights": {
            "canonical_source": "unknown; upstream info.json is not rights evidence",
            "publisher": "unknown",
            "source_version": dataset_version,
            "retrieved_date": None,
            "raw_files": source_artifacts,
            "citation": "pending review",
            "license": "unknown",
            "access_restrictions": ["unknown"],
            "permitted_uses": [],
            "redistribution_status": "unknown",
            "modification_requirements": ["unknown"],
            "attribution_requirements": ["pending source review"],
            "rights_review": {"decision": "pending", "reviewer": None, "date": None},
        },
        "table_contract": {
            "canonical_serialization": spec.extra.get("file_type") or "unknown",
            "encoding": "unknown",
            "delimiter": "unknown",
            "row_count_expectations": {"train": spec.extra.get("train_num"), "test": spec.extra.get("test_num")},
            "unique_column_names": len(set(spec.column_names)) == len(spec.column_names),
            "canonical_column_order": list(spec.column_names),
            "duplicate_row_policy": "pending review",
            "identifier_policy": "pending review",
            "target_presence": list(spec.target_columns),
            "missing_value_inventory": "unknown; preprocessing is mandatory if missingness is present",
            "supported_input_forms": ["legacy upstream files"],
            "canonical_table_checksum": None,
        },
        "columns": columns,
        "preprocessing": {
            "status": "not-reviewed",
            "raw_missingness": "unknown",
            "model_input_missing_values": "prohibited",
            "require_centralized_imputation_if_present": True,
            "fitted_on": "train-only",
            "configuration_version": None,
            "implementation_version": None,
        },
        "views": [
            {
                "view_id": "legacy-canonical-v1",
                "status": "unreviewed",
                "parent_dataset_version": dataset_version,
                "comparability": "diagnostic-only",
            }
        ],
        "split": {
            "split_id": "legacy-unreviewed-split",
            "status": "unreviewed",
            "frozen": False,
            "artifacts": source_artifacts[1:],
        },
        "predictive_tasks": [
            {
                "task_id": f"predict-{target_ids[target]}",
                "target_column_id": target_ids[target],
                "task_type": spec.task_type,
                "status": "unreviewed",
            }
            for target in spec.target_columns
        ],
        "validity": {"status": "unreviewed", "constraints": []},
        "privacy": {"status": "unreviewed", "threat_models": [], "sensitive_roles_reviewed": False},
        "metric_applicability": {
            "status": "unreviewed",
            "included": [],
            "excluded": [],
            "default_behavior": "error-until-reviewed",
        },
    }
    validate_instance("dataset-profile", payload)
    return DatasetProfile(payload, f"legacy:{spec.metadata_path.name}")


def write_dataset_profile(profile: DatasetProfile, path: str | Path) -> None:
    validate_instance("dataset-profile", profile.payload)
    atomic_write_json(path, profile.payload)
