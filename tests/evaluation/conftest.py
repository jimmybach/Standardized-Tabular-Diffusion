from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from standardized_tabular_diffusion.evaluation.contracts import EvaluationRequest
from standardized_tabular_diffusion.evaluation.profiles import load_dataset_profile, resolve_protocol


@pytest.fixture
def adult_profile():
    return load_dataset_profile(Path("configs/datasets/adult-uci-2-v1.json"))


@pytest.fixture
def p2_protocol():
    return resolve_protocol("p2-shape-trend", "0.2.0")


@pytest.fixture
def p3_protocol():
    return resolve_protocol("p3-validity", "0.3.0")


@pytest.fixture
def adult_frames(adult_profile):
    data: dict[str, list[object]] = {}
    for column in adult_profile.payload["columns"]:
        if column["semantic_type"] in {"continuous", "integer"}:
            minimum = column["valid_domain"].get("minimum", 0)
            data[column["name"]] = [minimum + index for index in range(20)]
        else:
            values = column["valid_domain"]["values"]
            data[column["name"]] = [values[index % min(len(values), 3)] for index in range(20)]
    reference = pd.DataFrame(data)
    synthetic = reference.copy(deep=True)
    return reference, synthetic


@pytest.fixture
def p2_request(adult_profile, p2_protocol):
    metrics = tuple(
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in p2_protocol.payload["metric_selections"]
    )
    return EvaluationRequest(
        subject_type="external-synthetic-table",
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": "0" * 64,
            "row_count": 20,
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": "1" * 64,
            "row_count": 20,
        },
        dataset_profile={
            "dataset_id": adult_profile.dataset_id,
            "dataset_profile_version": adult_profile.dataset_profile_version,
            "sha256": adult_profile.fingerprint,
        },
        protocol={
            "protocol_id": p2_protocol.protocol_id,
            "protocol_version": p2_protocol.protocol_version,
            "sha256": p2_protocol.fingerprint,
        },
        metrics=metrics,
        comparison_track="native",
        generation_seed=17,
        evaluator_seeds=(23,),
        model={"model_id": "fixture-model"},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )


@pytest.fixture
def p3_request(adult_profile, p3_protocol):
    metrics = tuple(
        {"metric_id": item["metric_id"], "metric_version": item["metric_version"]}
        for item in p3_protocol.payload["metric_selections"]
    )
    return EvaluationRequest(
        subject_type="external-synthetic-table",
        reference_artifact={
            "artifact_id": "reference-table",
            "media_type": "text/csv",
            "sha256": "0" * 64,
            "row_count": 20,
        },
        sample_artifact={
            "artifact_id": "synthetic-table",
            "media_type": "text/csv",
            "sha256": "1" * 64,
            "row_count": 20,
        },
        dataset_profile={
            "dataset_id": adult_profile.dataset_id,
            "dataset_profile_version": adult_profile.dataset_profile_version,
            "sha256": adult_profile.fingerprint,
        },
        protocol={
            "protocol_id": p3_protocol.protocol_id,
            "protocol_version": p3_protocol.protocol_version,
            "sha256": p3_protocol.fingerprint,
        },
        metrics=metrics,
        comparison_track="native",
        generation_seed=17,
        evaluator_seeds=(23,),
        model={"model_id": "fixture-model"},
        failure_policy={"structural_gate": "fail-fast", "metric_failure": "partial-bundle"},
    )
