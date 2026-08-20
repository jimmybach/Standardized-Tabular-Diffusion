"""Windows-safe subprocess boundary for the unchanged official TabuLa sampling loop."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from standardized_tabular_diffusion.evaluation.serialization import atomic_write_json
from standardized_tabular_diffusion.models.tabula import execute_tabula_sampling_request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    try:
        response = execute_tabula_sampling_request(args.request, args.sample)
    except Exception as exc:
        atomic_write_json(
            args.response,
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    atomic_write_json(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
