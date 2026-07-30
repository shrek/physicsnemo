#!/usr/bin/env python3
"""Render report-ready CPU/GPU and kernel timelines from normalized HTA JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
WIDTH = 1400
LEFT = 180
RIGHT = 180
PLOT_WIDTH = WIDTH - LEFT - RIGHT
COLORS = {
    "data": "#4e79a7",
    "transfer": "#76b7b2",
    "forward": "#59a14f",
    "backward": "#af7aa1",
    "optimizer": "#f28e2b",
    "communication": "#edc948",
    "idle": "#d9d9d9",
    "bubble": "#e15759",
    "radius_search": "#c94c16",
    "bvh_sdf": "#0e8a8a",
    "gemm": "#2f62e8",
    "attention": "#7b35de",
    "triton": "#0b996e",
    "reduction": "#cc8500",
    "memory": "#66788a",
    "other": "#94a3b8",
}


class InputError(ValueError):
    """Invalid normalized timeline input."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read valid JSON from {path}: {exc}") from exc


def number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InputError(f"{label} must be numeric")
    return float(value)


def validate_span(span: Any, label: str, duration: float | None = None) -> None:
    if not isinstance(span, dict):
        raise InputError(f"{label} must be an object")
    if not isinstance(span.get("label"), str) or not span["label"]:
        raise InputError(f"{label}.label must be a non-empty string")
    start = number(span.get("start_ms"), f"{label}.start_ms")
    end = number(span.get("end_ms"), f"{label}.end_ms")
    if start < 0 or end <= start:
        raise InputError(f"{label} must satisfy 0 <= start_ms < end_ms")
    if duration is not None and end > duration + 1e-9:
        raise InputError(f"{label}.end_ms exceeds its enclosing duration")
    if not isinstance(span.get("category", "other"), str):
        raise InputError(f"{label}.category must be a string")


def validate_document(document: Any) -> None:
    if not isinstance(document, dict):
        raise InputError("diagram input must be a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise InputError(f"schema_version must equal {SCHEMA_VERSION!r}")
    pipeline = document.get("pipeline")
    forward = document.get("forward")
    if not isinstance(pipeline, dict):
        raise InputError("pipeline is required and must be an object")
    duration = number(pipeline.get("duration_ms"), "pipeline.duration_ms")
    if duration <= 0:
        raise InputError("pipeline.duration_ms must be positive")
    lanes = pipeline.get("lanes")
    if not isinstance(lanes, list) or len(lanes) < 2:
        raise InputError("pipeline.lanes must contain at least CPU and GPU lanes")
    lane_names: set[str] = set()
    for lane_index, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            raise InputError(f"pipeline.lanes[{lane_index}] must be an object")
        name = lane.get("name")
        if not isinstance(name, str) or not name:
            raise InputError(f"pipeline.lanes[{lane_index}].name is required")
        lane_names.add(name.upper())
        spans = lane.get("spans")
        if not isinstance(spans, list):
            raise InputError(f"pipeline.lanes[{lane_index}].spans must be an array")
        for span_index, span in enumerate(spans):
            validate_span(
                span,
                f"pipeline.lanes[{lane_index}].spans[{span_index}]",
                duration,
            )
    if not {"CPU", "GPU"}.issubset(lane_names):
        raise InputError("pipeline.lanes must include named CPU and GPU lanes")

    if not isinstance(forward, dict):
        raise InputError("forward is required and must be an object")
    steps = forward.get("steps")
    if not isinstance(steps, list) or not steps:
        raise InputError("forward.steps must be a non-empty array")
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise InputError(f"forward.steps[{step_index}] must be an object")
        step_duration = number(
            step.get("duration_ms"), f"forward.steps[{step_index}].duration_ms"
        )
        if step_duration <= 0:
            raise InputError(
                f"forward.steps[{step_index}].duration_ms must be positive"
            )
        kernels = step.get("kernels")
        if not isinstance(kernels, list) or not kernels:
            raise InputError(f"forward.steps[{step_index}].kernels must not be empty")
        for kernel_index, kernel in enumerate(kernels):
            validate_span(
                kernel,
                f"forward.steps[{step_index}].kernels[{kernel_index}]",
                step_duration,
            )

    multi = document.get("multi_step")
    if multi is not None:
        if not isinstance(multi, dict):
            raise InputError("multi_step must be an object")
        multi_steps = multi.get("steps")
        if not isinstance(multi_steps, list) or len(multi_steps) < 2:
            raise InputError("multi_step.steps must contain at least two steps")
        last_start = -math.inf
        for index, step in enumerate(multi_steps):
            if not isinstance(step, dict):
                raise InputError(f"multi_step.steps[{index}] must be an object")
            start = number(step.get("start_ms"), f"multi_step.steps[{index}].start_ms")
            end = number(step.get("end_ms"), f"multi_step.steps[{index}].end_ms")
            busy_start = number(
                step.get("gpu_busy_start_ms"),
                f"multi_step.steps[{index}].gpu_busy_start_ms",
            )
            busy_end = number(
                step.get("gpu_busy_end_ms"),
                f"multi_step.steps[{index}].gpu_busy_end_ms",
            )
            if not (start <= busy_start < busy_end <= end):
                raise InputError(
                    f"multi_step.steps[{index}] must contain its GPU busy interval"
                )
            if start < last_start:
                raise InputError("multi_step.steps must be ordered by start_ms")
            last_start = start


def svg_begin(height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" viewBox="0 0 {WIDTH} {height}">',
        "<style>",
        "text{font-family:Arial,sans-serif;fill:#172033}"
        ".title{font-size:25px;font-weight:700}"
        ".subtitle{font-size:14px;fill:#42526b}"
        ".label{font-size:14px}"
        ".small{font-size:12px}"
        ".lane{font-size:17px;font-weight:700}"
        ".grid{stroke:#dce3ec;stroke-width:1}"
        ".axis{stroke:#53657a;stroke-width:1.2}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="28" y="38" class="title">{escape(title)}</text>',
    ]


def finish(lines: list[str], output: Path) -> None:
    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def xscale(value: float, start: float, end: float) -> float:
    return LEFT + (value - start) / (end - start) * PLOT_WIDTH


def ticks(lines: list[str], start: float, end: float, y0: int, y1: int) -> None:
    for index in range(6):
        value = start + (end - start) * index / 5
        x = xscale(value, start, end)
        lines.append(
            f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y1 + 22}" text-anchor="middle" '
            f'class="small">{value:.1f}</text>'
        )


def span_color(category: str) -> str:
    return COLORS.get(category, COLORS["other"])


def render_pipeline(document: dict[str, Any], output: Path) -> None:
    pipeline = document["pipeline"]
    duration = float(pipeline["duration_ms"])
    lanes = pipeline["lanes"]
    height = 150 + len(lanes) * 115
    title = pipeline.get("title") or "CPU/GPU pipeline"
    lines = svg_begin(height, title)
    lines.append(
        f'<text x="28" y="65" class="subtitle">Step {escape(str(pipeline.get("step_id", "")))}'
        f" · {duration:.1f} ms · grey/red regions identify GPU idle bubbles</text>"
    )
    y_start = 105
    ticks(lines, 0, duration, y_start - 20, height - 55)
    for lane_index, lane in enumerate(lanes):
        y = y_start + lane_index * 115
        lines.append(
            f'<text x="{LEFT - 18}" y="{y + 31}" text-anchor="end" '
            f'class="lane">{escape(lane["name"])}</text>'
        )
        lines.append(
            f'<rect x="{LEFT}" y="{y}" width="{PLOT_WIDTH}" height="52" '
            'fill="#eef2f7" stroke="#8fa0b5"/>'
        )
        for span in lane["spans"]:
            x = xscale(float(span["start_ms"]), 0, duration)
            width = max(1.5, xscale(float(span["end_ms"]), 0, duration) - x)
            category = span.get("category", "other")
            color = span_color(category)
            opacity = "0.95" if category not in ("idle", "bubble") else "0.75"
            lines.append(
                f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="52" '
                f'fill="{color}" fill-opacity="{opacity}" stroke="#ffffff"/>'
            )
            duration_ms = float(span["end_ms"]) - float(span["start_ms"])
            if width >= 68:
                text_color = "#172033" if category in ("idle", "bubble") else "#ffffff"
                lines.append(
                    f'<text x="{x + width / 2:.1f}" y="{y + 23}" '
                    f'text-anchor="middle" class="small" fill="{text_color}" '
                    f'style="fill:{text_color}">{escape(span["label"])}</text>'
                )
                lines.append(
                    f'<text x="{x + width / 2:.1f}" y="{y + 40}" '
                    f'text-anchor="middle" class="small" fill="{text_color}" '
                    f'style="fill:{text_color}">{duration_ms:.1f} ms</text>'
                )
            if span.get("annotation"):
                lines.append(
                    f'<text x="{x + width / 2:.1f}" y="{y + 75}" '
                    f'text-anchor="middle" class="small">'
                    f'{escape(str(span["annotation"]))}</text>'
                )
    summary = pipeline.get("summary")
    if summary:
        lines.append(
            f'<text x="{LEFT}" y="{height - 14}" class="subtitle">'
            f'{escape(str(summary))}</text>'
        )
    finish(lines, output)


def family_order(kernels: list[dict[str, Any]]) -> list[str]:
    totals: dict[str, float] = {}
    for kernel in kernels:
        family = str(kernel.get("family") or kernel.get("category") or "other")
        totals[family] = totals.get(family, 0.0) + (
            float(kernel["end_ms"]) - float(kernel["start_ms"])
        )
    return sorted(totals, key=lambda item: (-totals[item], item))


def render_forward(document: dict[str, Any], output: Path) -> None:
    steps = document["forward"]["steps"]
    families: list[str] = []
    for step in steps:
        for family in family_order(step["kernels"]):
            if family not in families:
                families.append(family)
    row_height = 30
    panel_height = 105 + (len(families) + 1) * row_height
    height = 95 + len(steps) * panel_height + 65
    title = document["forward"].get("title") or "Forward-pass dominant kernels"
    lines = svg_begin(height, title)
    lines.append(
        '<text x="28" y="64" class="subtitle">Kernel families are ordered by '
        "total GPU time; the CPU forward span provides the wall-time boundary.</text>"
    )
    for step_index, step in enumerate(steps):
        top = 92 + step_index * panel_height
        duration = float(step["duration_ms"])
        lines.append(
            f'<rect x="18" y="{top - 18}" width="{WIDTH - 36}" '
            f'height="{panel_height - 8}" rx="8" fill="#ffffff" stroke="#d8e0ea"/>'
        )
        lines.append(
            f'<text x="35" y="{top + 8}" class="lane">'
            f'{escape(str(step.get("step_id", f"Step {step_index + 1}")))}</text>'
        )
        axis_top = top + 30
        ticks(
            lines,
            0,
            duration,
            axis_top,
            axis_top + (len(families) + 1) * row_height,
        )
        lines.append(
            f'<text x="{LEFT - 12}" y="{axis_top + 20}" text-anchor="end" '
            'class="label">CPU forward</text>'
        )
        lines.append(
            f'<rect x="{LEFT}" y="{axis_top + 4}" width="{PLOT_WIDTH}" height="20" '
            'rx="3" fill="#cfd8e6" stroke="#9aabc0"/>'
        )
        for family_index, family in enumerate(families):
            y = axis_top + (family_index + 1) * row_height + 4
            family_kernels = [
                item
                for item in step["kernels"]
                if str(item.get("family") or item.get("category") or "other") == family
            ]
            total = sum(
                float(item["end_ms"]) - float(item["start_ms"])
                for item in family_kernels
            )
            lines.append(
                f'<text x="{LEFT - 12}" y="{y + 15}" text-anchor="end" '
                f'class="label">{escape(family)}</text>'
            )
            for kernel in family_kernels:
                x = xscale(float(kernel["start_ms"]), 0, duration)
                width = max(
                    1.2, xscale(float(kernel["end_ms"]), 0, duration) - x
                )
                lines.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="20" '
                    f'rx="2" fill="{span_color(family)}"><title>'
                    f'{escape(kernel["label"])}: '
                    f'{float(kernel["end_ms"]) - float(kernel["start_ms"]):.3f} ms'
                    "</title></rect>"
                )
                if width >= 42:
                    lines.append(
                        f'<text x="{x + width / 2:.1f}" y="{y + 14}" '
                        'text-anchor="middle" class="small" '
                        'style="fill:#ffffff">'
                        f'{float(kernel["end_ms"]) - float(kernel["start_ms"]):.1f}'
                        "</text>"
                    )
            lines.append(
                f'<text x="{LEFT + PLOT_WIDTH + 12}" y="{y + 15}" '
                f'class="small">{total:.2f} ms ({len(family_kernels)})</text>'
            )
    finish(lines, output)


def render_idle_bubbles(document: dict[str, Any], output: Path) -> bool:
    pipeline = document["pipeline"]
    gpu = next(lane for lane in pipeline["lanes"] if lane["name"].upper() == "GPU")
    bubbles = [
        span for span in gpu["spans"] if span.get("category") in ("idle", "bubble")
    ]
    if not bubbles:
        return False
    duration = float(pipeline["duration_ms"])
    lines = svg_begin(285, "In-step GPU idle bubbles")
    lines.append(
        '<text x="28" y="64" class="subtitle">Red spans are explicitly classified '
        "bubbles; grey spans are unclassified idle time.</text>"
    )
    ticks(lines, 0, duration, 90, 215)
    lines.append(
        f'<text x="{LEFT - 18}" y="145" text-anchor="end" class="lane">GPU</text>'
    )
    lines.append(
        f'<rect x="{LEFT}" y="112" width="{PLOT_WIDTH}" height="52" '
        'fill="#59a14f" fill-opacity="0.2" stroke="#8fa0b5"/>'
    )
    total = 0.0
    for span in bubbles:
        start = float(span["start_ms"])
        end = float(span["end_ms"])
        total += end - start
        x = xscale(start, 0, duration)
        width = max(2, xscale(end, 0, duration) - x)
        lines.append(
            f'<rect x="{x:.1f}" y="112" width="{width:.1f}" height="52" '
            f'fill="{span_color(span.get("category", "idle"))}" stroke="#ffffff"/>'
        )
        lines.append(
            f'<text x="{x + width / 2:.1f}" y="185" text-anchor="middle" '
            f'class="small">{escape(span["label"])} · {end - start:.1f} ms</text>'
        )
    lines.append(
        f'<text x="{LEFT}" y="248" class="subtitle">Total classified idle: '
        f"{total:.1f} ms ({total / duration * 100:.1f}% of step).</text>"
    )
    finish(lines, output)
    return True


def render_multi_step(document: dict[str, Any], output: Path) -> bool:
    multi = document.get("multi_step")
    if not multi:
        return False
    steps = multi["steps"]
    start = float(steps[0]["start_ms"])
    end = float(steps[-1]["end_ms"])
    lines = svg_begin(330, multi.get("title") or "GPU compute and dataload bubbles")
    lines.append(
        '<text x="28" y="64" class="subtitle">GPU busy envelopes across consecutive '
        "steps; red gaps expose inter-step pipeline starvation.</text>"
    )
    ticks(lines, start, end, 90, 250)
    lines.append(
        f'<text x="{LEFT - 18}" y="145" text-anchor="end" class="lane">GPU</text>'
    )
    lines.append(
        f'<rect x="{LEFT}" y="112" width="{PLOT_WIDTH}" height="52" '
        'fill="#e6ebf2" stroke="#8fa0b5"/>'
    )
    previous_busy_end: float | None = None
    total_busy = 0.0
    total_bubble = 0.0
    for step in steps:
        busy_start = float(step["gpu_busy_start_ms"])
        busy_end = float(step["gpu_busy_end_ms"])
        if previous_busy_end is not None and busy_start > previous_busy_end:
            x = xscale(previous_busy_end, start, end)
            width = xscale(busy_start, start, end) - x
            total_bubble += busy_start - previous_busy_end
            lines.append(
                f'<rect x="{x:.1f}" y="112" width="{width:.1f}" height="52" '
                f'fill="{COLORS["bubble"]}" fill-opacity="0.9"/>'
            )
            if width >= 55:
                lines.append(
                    f'<text x="{x + width / 2:.1f}" y="143" text-anchor="middle" '
                    'class="small" style="fill:#ffffff">'
                    f"bubble {busy_start - previous_busy_end:.1f} ms</text>"
                )
        x = xscale(busy_start, start, end)
        width = xscale(busy_end, start, end) - x
        total_busy += busy_end - busy_start
        lines.append(
            f'<rect x="{x:.1f}" y="112" width="{width:.1f}" height="52" '
            f'fill="{COLORS["forward"]}" stroke="#ffffff"/>'
        )
        lines.append(
            f'<text x="{x + width / 2:.1f}" y="143" text-anchor="middle" '
            f'class="small" style="fill:#ffffff">'
            f'{escape(str(step.get("step_id", "step")))}</text>'
        )
        previous_busy_end = busy_end
    observed = end - start
    lines.append(
        f'<text x="{LEFT}" y="222" class="subtitle">Observed window: {observed:.1f} ms'
        f" · GPU busy: {total_busy:.1f} ms · inter-step bubbles: "
        f"{total_bubble:.1f} ms.</text>"
    )
    finish(lines, output)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate input without rendering."
    )
    args = parser.parse_args()
    try:
        document = load_json(args.input)
        validate_document(document)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(f"OK: valid diagram input: {args.input}")
        return 0
    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "cpu_gpu_pipeline": output_dir / "cpu-gpu-pipeline.svg",
        "forward_dominant_kernels": output_dir / "forward-dominant-kernels.svg",
    }
    render_pipeline(document, outputs["cpu_gpu_pipeline"])
    render_forward(document, outputs["forward_dominant_kernels"])
    idle = output_dir / "in-step-gpu-bubbles.svg"
    if render_idle_bubbles(document, idle):
        outputs["in_step_gpu_bubbles"] = idle
    multi = output_dir / "multi-step-gpu-bubbles.svg"
    if render_multi_step(document, multi):
        outputs["multi_step_gpu_bubbles"] = multi
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": str(args.input.expanduser().resolve()),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = output_dir / "diagram-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
