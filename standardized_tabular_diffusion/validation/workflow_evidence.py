"""Create deterministic fallback evidence when a CI gate fails before its protocol runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def preserve_pre_protocol_failure(
    *,
    output: Path,
    protocol_id: str,
    phase: str,
    message: str,
) -> bool:
    """Write fallback failure evidence only when protocol evidence is absent."""

    if output.is_file():
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence_schema_version": "1.0.0",
        "protocol_id": protocol_id,
        "phase": phase,
        "status": "fail",
        "repository_commit": os.environ.get("GITHUB_SHA", "unknown"),
        "error_type": "PreProtocolFailure",
        "error": message,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    preserve_pre_protocol_failure(
        output=args.output,
        protocol_id=args.protocol_id,
        phase=args.phase,
        message=args.message,
    )


if __name__ == "__main__":
    main()
