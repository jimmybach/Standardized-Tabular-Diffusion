"""Run the preregistered TabDDPM/Adult P5 v1 confirmatory experiment on Windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from standardized_tabular_diffusion.validation.p5_tabddpm_adult_pilot import run_declared_experiment

PROTOCOL_ID = "p5-tabddpm-adult-confirmatory-windows-v1"
GENERATION_SEEDS = (3, 4, 5)
PREREGISTRATION = Path("docs/evidence/evaluation/p5-confirmatory-preregistration-2026-08-13.json")


def run_confirmatory(repo_root: Path, output_root: Path, evidence_path: Path) -> dict[str, object]:
    return run_declared_experiment(
        repo_root,
        output_root,
        evidence_path,
        experiment_protocol_id=PROTOCOL_ID,
        evaluation_protocol_version="1.0.0",
        generation_seeds=GENERATION_SEEDS,
        expected_generation_seeds=GENERATION_SEEDS,
        experiment_kind="preregistered-confirmatory",
        preregistration_path=PREREGISTRATION,
        additional_retained_sources=(
            Path("standardized_tabular_diffusion/validation/p5_tabddpm_adult_confirmatory.py"),
            PREREGISTRATION,
        ),
        claim_boundary=(
            "Preregistered TabDDPM/Adult confirmation for the unchanged P5 v1 scientific identity. "
            "Passing the declared non-numerical gates may support a separate protocol-freeze decision only; "
            "this run does not itself admit Official Results, establish a formal privacy guarantee, assess "
            "regulatory compliance, or generalize to another model, dataset, configuration, seed, or platform."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence = run_confirmatory(args.repo_root, args.output_root, args.evidence_path)
    print(json.dumps({"status": evidence["status"], "protocol_id": evidence["protocol_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
