from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from standardized_tabular_diffusion.evaluation.profiles import (
    ProfileError,
    import_legacy_dataset_spec,
    list_dataset_profiles,
    list_protocol_profiles,
    load_dataset_profile,
    load_protocol_profile,
    write_dataset_profile,
)
from standardized_tabular_diffusion.evaluation.registry import (
    MetricRegistryError,
    load_metric_registry,
    validate_metric_record,
)
from standardized_tabular_diffusion.evaluation.schema import SchemaValidationError
from standardized_tabular_diffusion.evaluation.serialization import SerializationError
from standardized_tabular_diffusion.interfaces import DatasetSpec

pytestmark = [pytest.mark.core, pytest.mark.evaluation]


def test_packaged_legacy_registry_is_explicitly_nonofficial() -> None:
    records = load_metric_registry()
    assert len(records) == 8
    assert len({record.identity for record in records}) == len(records)
    assert all(record.payload["lifecycle_status"] == "registered" for record in records)
    assert all(record.payload["planned_leaderboard_role"] == "legacy-diagnostic" for record in records)
    assert all(not record.payload["admission"]["official_results_allowed"] for record in records)


def test_lifecycle_cannot_advance_without_cumulative_evidence() -> None:
    payload = load_metric_registry()[0].to_dict()
    payload["lifecycle_status"] = "unit-validated"
    with pytest.raises(MetricRegistryError, match="definition_review"):
        validate_metric_record(payload)

    evidence = {
        "decision": "approved",
        "reviewer": "test-reviewer",
        "reviewed_at": "2026-08-03T12:00:00Z",
        "evidence_refs": ["tests/evaluation/fixture"],
    }
    payload["lifecycle_evidence"]["definition_review"] = evidence
    with pytest.raises(MetricRegistryError, match="implementation"):
        validate_metric_record(payload)


def test_official_admission_requires_protocol_freeze_and_matching_role() -> None:
    payload = load_metric_registry()[0].to_dict()
    payload["admission"]["official_results_allowed"] = True
    with pytest.raises(MetricRegistryError, match="protocol-frozen"):
        validate_metric_record(payload)


def _legacy_spec(tmp_path: Path) -> DatasetSpec:
    metadata = tmp_path / "info.json"
    train = tmp_path / "train.csv"
    metadata.write_text('{"name":"Fixture"}\n', encoding="utf-8")
    train.write_text("age,city,target\n20,A,0\n", encoding="utf-8")
    return DatasetSpec(
        name="Fixture Dataset",
        task_type="classification",
        column_names=["age", "city", "target"],
        numerical_columns=["age"],
        categorical_columns=["city", "target"],
        target_columns=["target"],
        metadata_path=metadata,
        train_data_path=train,
        extra={"file_type": "csv", "train_num": 1},
    )


def test_legacy_dataset_import_is_deterministic_nonofficial_and_round_trips(tmp_path: Path) -> None:
    first = import_legacy_dataset_spec(_legacy_spec(tmp_path))
    second = import_legacy_dataset_spec(_legacy_spec(tmp_path))
    output = tmp_path / "fixture.profile.json"
    write_dataset_profile(first, output)
    restored = load_dataset_profile(output)

    assert first.fingerprint == second.fingerprint == restored.fingerprint
    assert restored.payload["status"] == "legacy-imported"
    assert restored.payload["official_eligible"] is False
    assert restored.payload["source_rights"]["redistribution_status"] == "unknown"
    assert restored.payload["preprocessing"]["model_input_missing_values"] == "prohibited"
    assert all(
        not Path(item["name"]).is_absolute() for item in restored.payload["source_rights"]["raw_files"] if item["name"]
    )
    assert restored.payload["columns"][2]["roles"] == ["primary_target"]


def test_dataset_profile_unknown_top_level_field_fails_closed(tmp_path: Path) -> None:
    profile = import_legacy_dataset_spec(_legacy_spec(tmp_path)).to_dict()
    profile["unreviewed_extension"] = {}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        load_dataset_profile(path)


def test_dataset_profile_semantic_identity_checks_fail_closed(tmp_path: Path) -> None:
    profile = import_legacy_dataset_spec(_legacy_spec(tmp_path)).to_dict()
    profile["columns"][1]["column_id"] = profile["columns"][0]["column_id"]
    path = tmp_path / "duplicate-column-id.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ProfileError, match="column_id"):
        load_dataset_profile(path)

    profile = import_legacy_dataset_spec(_legacy_spec(tmp_path)).to_dict()
    profile["table_contract"]["canonical_column_order"] = list(reversed(profile["table_contract"]["canonical_column_order"]))
    path = tmp_path / "wrong-column-order.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ProfileError, match="canonical_column_order"):
        load_dataset_profile(path)


def test_dataset_profile_directory_rejects_duplicate_exact_identities(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    profile = import_legacy_dataset_spec(_legacy_spec(source_root))
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    write_dataset_profile(profile, profiles / "first.json")
    write_dataset_profile(profile, profiles / "second.json")
    with pytest.raises(ProfileError, match="Duplicate dataset identity"):
        list_dataset_profiles(profiles)


def test_safe_yaml_loader_rejects_executable_constructor(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    with pytest.raises(SerializationError, match="safely load"):
        load_protocol_profile(path)


def test_packaged_protocols_resolve_exact_versions_and_are_nonofficial() -> None:
    profiles = list_protocol_profiles()
    assert {profile.identity for profile in profiles} == {
        ("development-p1", "0.1.0"),
        ("legacy-tabstruct-aligned", "1.0.0-legacy"),
    }
    assert all(not profile.payload["official_results_allowed"] for profile in profiles)


def test_draft_protocol_cannot_claim_official_results(tmp_path: Path) -> None:
    payload = copy.deepcopy(list_protocol_profiles()[0].to_dict())
    payload["official_results_allowed"] = True
    path = tmp_path / "invalid-protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((SchemaValidationError, ProfileError)):
        load_protocol_profile(path)


def test_protocol_rejects_duplicate_metric_identities_and_unknown_schema_versions(tmp_path: Path) -> None:
    payload = copy.deepcopy(list_protocol_profiles()[1].to_dict())
    duplicate = copy.deepcopy(payload["metric_selections"][0])
    duplicate["required"] = not duplicate["required"]
    payload["metric_selections"].append(duplicate)
    path = tmp_path / "duplicate-metric.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProfileError, match="unique metric identities"):
        load_protocol_profile(path)

    payload = copy.deepcopy(list_protocol_profiles()[0].to_dict())
    payload["result_schema_versions"]["atomic_result"] = "2.0.0"
    path = tmp_path / "unsupported-schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProfileError, match="unsupported result schema"):
        load_protocol_profile(path)
