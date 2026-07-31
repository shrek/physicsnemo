#!/usr/bin/env python3
"""Record explicit user confirmation of a phase-1 test configuration."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--confirmation-source",
        default="explicit user confirmation in conversation",
        help="Non-secret description of where explicit confirmation was received.",
    )
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    test_config_path = bundle / "test-config.json"
    confirmation_path = bundle / "config-confirmation.json"

    try:
        test_config = load_json(test_config_path)
        confirmation = load_json(confirmation_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if test_config.get("schema_version") != SCHEMA_VERSION:
        print("error: unsupported test-config schema_version", file=sys.stderr)
        return 2
    if confirmation.get("run_id") != test_config.get("run_id"):
        print("error: confirmation and test config run_id differ", file=sys.stderr)
        return 2
    if not args.confirmation_source.strip():
        print("error: --confirmation-source cannot be empty", file=sys.stderr)
        return 2

    confirmation.update(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": 1,
            "run_id": test_config["run_id"],
            "status": "confirmed",
            "confirmed_config_sha256": sha256_file(test_config_path),
            "confirmed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "confirmation_source": args.confirmation_source.strip(),
        }
    )
    confirmation_path.write_text(
        json.dumps(confirmation, indent=2, sort_keys=True) + "\n"
    )
    print(confirmation["confirmed_config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
