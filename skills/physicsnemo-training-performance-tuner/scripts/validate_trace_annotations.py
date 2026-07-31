#!/usr/bin/env python3
"""Validate trace annotation health before HTA analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "0.1"
DEFAULT_REQUIRED_PHASES = (
    "dataloader_wait",
    "host_to_device",
    "forward",
    "loss",
    "backward",
    "optimizer_step",
    "distributed_sync",
)
PROJECTED_CATEGORIES = {"gpu_user_annotation"}
COMPILE_WARNING_PARTS = ("profiler record function", "will be ignored")


class UserError(ValueError):
    """An actionable input or trace-format error."""


def stream_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield Chrome trace events without loading a large trace into memory."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        position = 0
        found = False
        eof = False
        while not found:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                raise UserError("trace does not contain a traceEvents array")
            buffer += chunk
            marker = buffer.find('"traceEvents"')
            if marker < 0:
                buffer = buffer[-64:]
                continue
            array_start = buffer.find("[", marker)
            while array_start < 0:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    raise UserError("traceEvents is not an array")
                buffer += chunk
                array_start = buffer.find("[", marker)
            position = array_start + 1
            found = True

        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer):
                    break
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    eof = True
                    break
                buffer = buffer[position:] + chunk
                position = 0
            if eof:
                raise UserError("traceEvents array is incomplete")
            if buffer[position] == "]":
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    if isinstance(value, dict):
                        yield value
                    break
                except json.JSONDecodeError:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        raise UserError("cannot decode an event in traceEvents")
                    buffer = buffer[position:] + chunk
                    position = 0
            if position > 4 * 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def inspect_hta_rows(path: Path | None) -> dict[str, int | bool]:
    result: dict[str, int | bool] = {
        "duration_invalid_count": 0,
        "end_before_start_count": 0,
        "end_field_inconsistent_count": 0,
        "derived_end_required": False,
    }
    if path is None:
        return result
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames or not {"ts", "dur"}.issubset(rows.fieldnames):
                raise UserError("HTA rows CSV must contain ts and dur columns")
            for row in rows:
                try:
                    ts = float(row["ts"])
                    duration = float(row["dur"])
                except (TypeError, ValueError):
                    result["duration_invalid_count"] += 1
                    continue
                if not math.isfinite(ts) or not math.isfinite(duration) or duration < 0:
                    result["duration_invalid_count"] += 1
                    continue
                raw_end = row.get("end")
                if raw_end in (None, ""):
                    result["derived_end_required"] = True
                    continue
                try:
                    end = float(raw_end)
                except ValueError:
                    result["end_field_inconsistent_count"] += 1
                    result["derived_end_required"] = True
                    continue
                if not math.isfinite(end) or end < ts:
                    result["end_before_start_count"] += 1
                    result["derived_end_required"] = True
                elif not math.isclose(end, ts + duration, rel_tol=1e-9, abs_tol=1e-3):
                    result["end_field_inconsistent_count"] += 1
                    result["derived_end_required"] = True
    except OSError as exc:
        raise UserError(f"cannot read HTA rows CSV: {exc}") from exc
    return result


def analyze(
    trace: Path,
    *,
    run_id: str,
    required_phases: tuple[str, ...],
    expected_steps: int | None,
    allow_reconstruction: bool,
    compile_log: Path | None,
    hta_rows: Path | None,
    variant: str = "unknown",
) -> dict[str, Any]:
    phase_counts: Counter[str] = Counter()
    native_steps = 0
    explicit_steps = 0
    projected_count = 0
    invalid_trace_durations = 0
    original_annotation_count = 0

    for event in stream_trace_events(trace):
        name = event.get("name")
        category = str(event.get("cat", ""))
        if category in PROJECTED_CATEGORIES:
            projected_count += 1
        is_original = category == "user_annotation" or (
            category == "" and event.get("ph") in {"X", "B", "E"}
        )
        if not is_original:
            continue
        original_annotation_count += 1
        if isinstance(name, str):
            if name.startswith("ProfilerStep#"):
                native_steps += 1
            elif name == "train_step":
                explicit_steps += 1
            if name in required_phases:
                phase_counts[name] += 1
        duration = event.get("dur")
        if event.get("ph") == "X" and (
            not finite_number(duration) or float(duration) < 0
        ):
            invalid_trace_durations += 1

    compile_warning = False
    if compile_log is not None:
        try:
            log_text = compile_log.read_text(errors="replace").lower()
        except OSError as exc:
            raise UserError(f"cannot read compile log: {exc}") from exc
        compile_warning = all(part in log_text for part in COMPILE_WARNING_PARTS)

    timestamps = inspect_hta_rows(hta_rows)
    timestamps["duration_invalid_count"] = (
        int(timestamps["duration_invalid_count"]) + invalid_trace_durations
    )
    missing = [phase for phase in required_phases if phase_counts[phase] == 0]
    enough_native = native_steps > 0 and (
        expected_steps is None or native_steps >= expected_steps
    )
    enough_explicit = explicit_steps > 0 and (
        expected_steps is None or explicit_steps >= expected_steps
    )
    reconstruction_anchors = (
        not missing
        and phase_counts["dataloader_wait"] > 0
        and phase_counts["distributed_sync"] > 0
    )

    provenance = "pending"
    marker: str | None = None
    selection_rule: str | None = None
    status = "failed"
    fallback_used = False
    attribution_method = "hta_user_annotation"
    recapture_reason: str | None = None
    if enough_native:
        provenance = "native_profiler_step"
        marker = "ProfilerStep#N"
        status = "pass"
    elif enough_explicit:
        provenance = "explicit_train_step"
        marker = "train_step"
        status = "pass"
    elif allow_reconstruction and reconstruction_anchors:
        provenance = "reconstructed"
        selection_rule = (
            "Use only original user_annotation rows; derive end as ts + dur; "
            "group canonical phases in trace order from dataloader_wait through "
            "distributed_sync; correlate GPU kernels through index_correlation."
        )
        status = "reconstructed"
        fallback_used = True
        attribution_method = "index_correlation"
    else:
        recapture_reason = (
            "No complete native ProfilerStep or explicit train_step boundary was "
            "found, and safe reconstruction was not enabled or lacked phase anchors."
        )

    if missing:
        status = "failed"
        recapture_reason = "Missing required phase annotations: " + ", ".join(missing)
    if int(timestamps["duration_invalid_count"]) > 0:
        status = "failed"
        recapture_reason = "Trace contains invalid or negative annotation durations."

    notes: list[str] = []
    if projected_count:
        notes.append(
            "Projected gpu_user_annotation rows were detected; exclude them from "
            "CPU phase interval construction to avoid duplicates."
        )
    if bool(timestamps["derived_end_required"]):
        notes.append(
            "HTA end fields are missing or inconsistent; derive interval end as ts + dur."
        )
    if compile_warning:
        notes.append(
            "torch.compile ignored nested record_function annotations; verify required "
            "outer ranges directly in the raw trace."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "variant": variant,
        "status": status,
        "trace": str(trace),
        "step_boundary": {
            "provenance": provenance,
            "marker": marker,
            "count": native_steps if enough_native else explicit_steps if enough_explicit else 0,
            "selection_rule": selection_rule,
        },
        "annotations": {
            "required_phases": list(required_phases),
            "phase_counts": dict(phase_counts),
            "missing_required_phases": missing,
            "original_annotation_count": original_annotation_count,
            "duplicate_projected_annotations": projected_count > 0,
            "projected_annotation_count": projected_count,
            "compile_record_function_warning": compile_warning,
        },
        "timestamps": {
            **timestamps,
            "end_rule": "Use valid HTA end only when consistent; otherwise ts + dur.",
        },
        "kernel_attribution": {
            "method": attribution_method,
            "fallback_used": fallback_used,
        },
        "recapture": {
            "required": status == "failed",
            "reason": recapture_reason,
        },
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--variant", choices=("eager", "compiled"), required=True)
    parser.add_argument(
        "--required-phase",
        action="append",
        dest="required_phases",
        help="Required phase name; repeat to override the default phase set.",
    )
    parser.add_argument("--allow-reconstruction", action="store_true")
    parser.add_argument("--compile-log", type=Path)
    parser.add_argument(
        "--hta-rows",
        type=Path,
        help="Optional HTA annotation CSV with ts, dur, and optional end columns.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_steps is not None and args.expected_steps < 1:
        print("error: --expected-steps must be at least 1", file=sys.stderr)
        return 2
    required = tuple(args.required_phases or DEFAULT_REQUIRED_PHASES)
    if not required or any(not phase for phase in required):
        print("error: required phases must be non-empty", file=sys.stderr)
        return 2
    try:
        result = analyze(
            args.trace.expanduser().resolve(),
            run_id=args.run_id,
            required_phases=required,
            expected_steps=args.expected_steps,
            allow_reconstruction=args.allow_reconstruction,
            compile_log=args.compile_log,
            hta_rows=args.hta_rows,
            variant=args.variant,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, UserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0 if result["status"] in {"pass", "reconstructed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
