#!/usr/bin/env python3
"""Normalize bounded torch.compile graph-break and recompilation logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
GRAPH_BREAK = re.compile(
    r"Graph break(?: in user code)?(?: from user code)? at (.+?):(\d+)",
    re.IGNORECASE,
)
RECOMPILE = re.compile(
    r"Recompiling function (.+?) in (.+?):(\d+)", re.IGNORECASE
)
REASON = re.compile(r"^\s*(?:Reason:|- \d+/\d+:)\s*(.+?)\s*$")
CACHE_LIMIT_MARKERS = ("recompile_limit", "cache_size_limit")
BACKEND_FAILURE_MARKERS = (
    "backendcompilerfailed",
    "compiler error",
    "falling back to eager",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def aggregate(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter(
        tuple(item.get(key) for key in keys) for item in items
    )
    return [
        {**dict(zip(keys, values)), "count": count}
        for values, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def parse_logs(paths: list[Path], run_id: str) -> dict[str, Any]:
    breaks: list[dict[str, Any]] = []
    recompiles: list[dict[str, Any]] = []
    cache_limit_warnings: list[str] = []
    backend_failures: list[str] = []

    for path in paths:
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            graph_match = GRAPH_BREAK.search(line)
            if graph_match:
                reason = None
                for following in lines[index + 1 : index + 12]:
                    reason_match = REASON.match(following)
                    if reason_match:
                        reason = reason_match.group(1)
                        break
                    if GRAPH_BREAK.search(following) or RECOMPILE.search(following):
                        break
                breaks.append(
                    {
                        "path": graph_match.group(1),
                        "line": int(graph_match.group(2)),
                        "reason": reason or "unreported",
                    }
                )
            recompile_match = RECOMPILE.search(line)
            if recompile_match:
                reason = None
                for following in lines[index + 1 : index + 12]:
                    reason_match = REASON.match(following)
                    if reason_match:
                        reason = reason_match.group(1)
                        break
                    if GRAPH_BREAK.search(following) or RECOMPILE.search(following):
                        break
                recompiles.append(
                    {
                        "function": recompile_match.group(1),
                        "path": recompile_match.group(2),
                        "line": int(recompile_match.group(3)),
                        "guard_failure": reason or "unreported",
                    }
                )
            lower = line.lower()
            if any(marker in lower for marker in CACHE_LIMIT_MARKERS):
                cache_limit_warnings.append(line.strip())
            if any(marker in lower for marker in BACKEND_FAILURE_MARKERS):
                backend_failures.append(line.strip())

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "complete",
        "created_at_utc": utc_now(),
        "timing_scope": "diagnostic_not_benchmark",
        "logs": [str(path) for path in paths],
        "graph_breaks": aggregate(breaks, ("path", "line", "reason")),
        "recompilations": aggregate(
            recompiles, ("function", "path", "line", "guard_failure")
        ),
        "cache_limit_warnings": sorted(set(cache_limit_warnings)),
        "backend_failures": sorted(set(backend_failures)),
        "summary": {
            "unique_graph_breaks": len(
                {tuple(item.values()) for item in breaks}
            ),
            "graph_break_occurrences": len(breaks),
            "unique_recompilations": len(
                {tuple(item.values()) for item in recompiles}
            ),
            "recompilation_occurrences": len(recompiles),
            "eager_fallback_or_backend_failure": bool(backend_failures),
        },
        "notes": [
            "Collected in a bounded diagnostic run; do not use its wall time as a benchmark.",
            "No graph-break entry means none were logged, not proof that fullgraph capture succeeded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = parse_logs(
            [path.expanduser().resolve() for path in args.log], args.run_id
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
