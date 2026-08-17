"""Build machine-readable v0 optimizer results from recipe artifacts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


_CANONICAL_RANGES = (
    "train_step",
    "dataloader_wait",
    "host_to_device",
    "forward",
    "loss",
    "backward",
    "optimizer_step",
)
_WARMUP_STEPS = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"metrics artifact was not created: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
        records.append(record)
    return records


def benchmark_result(metrics_path: Path) -> dict[str, float | str]:
    """Return the v0 benchmark result from a completed training run's JSONL log."""
    records = _read_jsonl(metrics_path)
    step_times = [
        float(record["step_time_s"])
        for record in records
        if record.get("phase") == "train_step"
        and isinstance(record.get("step_time_s"), (int, float))
    ]
    if not step_times:
        raise ValueError("metrics log contains no train_step timing records")
    measured_steps = step_times[_WARMUP_STEPS:] or step_times

    validation = next(
        (record for record in reversed(records) if record.get("phase") == "val_summary"),
        None,
    )
    if validation is None or not isinstance(validation.get("loss"), (int, float)):
        raise ValueError("metrics log contains no numeric val_summary loss")

    return {
        "step_time_ms": statistics.median(measured_steps) * 1_000.0,
        "correctness_value": float(validation["loss"]),
        "correctness_metric": "validation_loss",
    }


def trace_result(trace_path: Path) -> dict[str, object]:
    """Return a v0 trace result and report only ranges present in the trace."""
    if not trace_path.is_file():
        raise FileNotFoundError(f"Chrome trace was not created: {trace_path}")
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Chrome trace is invalid JSON: {trace_path}") from error
    events = trace.get("traceEvents") if isinstance(trace, dict) else None
    if not isinstance(events, list):
        raise ValueError(f"Chrome trace has no traceEvents array: {trace_path}")
    names = {
        event.get("name")
        for event in events
        if isinstance(event, dict) and isinstance(event.get("name"), str)
    }
    ranges = [name for name in _CANONICAL_RANGES if name in names]
    return {
        "completed": True,
        "path": str(trace_path),
        "ranges": ranges,
        "summary": (
            f"Chrome trace contains {len(events)} events and "
            f"{len(ranges)} canonical optimizer ranges."
        ),
    }
