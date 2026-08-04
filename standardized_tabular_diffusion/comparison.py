from __future__ import annotations

from pathlib import Path

import pandas as pd

from standardized_tabular_diffusion.evaluation.serialization import read_json


def load_summary(path: Path) -> dict:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Summary must be a JSON object: {path}")
    return payload


def summary_to_row(summary: dict) -> dict:
    metrics = summary["metrics"]
    return {
        "model": summary["model"],
        "dataset": summary["dataset"],
        "protocol_name": summary["protocol_name"],
        "density_shape": metrics["density"].get("shape_score"),
        "density_trend": metrics["density"].get("trend_score"),
        "density_overall": metrics["density"].get("overall_score"),
        "ml_primary_name": metrics["ml_efficacy"].get("primary_metric_name"),
        "ml_primary_value": metrics["ml_efficacy"].get("primary_metric_value"),
        "detection_logistic": metrics["detection"].get("logistic_detection"),
        "privacy_dcr": metrics["privacy"].get("dcr_score"),
        "global_utility": metrics["structural_fidelity"].get("global_utility"),
    }


def compare_summaries(summary_paths: list[Path]) -> pd.DataFrame:
    rows = [summary_to_row(load_summary(path)) for path in summary_paths]
    frame = pd.DataFrame(rows)
    sort_cols = [col for col in ("dataset", "model") if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    return frame
