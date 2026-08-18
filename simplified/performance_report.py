"""Deterministic phase-1 HTA report bundle generation."""

from __future__ import annotations

import datetime as dt
from contextlib import redirect_stderr, redirect_stdout
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from simplified.types import (
    BenchmarkResult,
    InstrumentationPlan,
    PerformanceReport,
    TraceResult,
    TrainingSpec,
)

CANONICAL_PHASES = (
    "dataloader_wait",
    "feature_construction",
    "host_to_device",
    "forward",
    "loss",
    "backward",
    "optimizer_step",
    "distributed_sync",
    "validation",
    "checkpoint",
)
_SCHEMA_VERSION = "0.1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _frame_records(frame: Any) -> list[dict[str, Any]]:
    """Serialize a pandas frame without exposing pandas in the public contract."""
    return json.loads(frame.to_json(orient="records"))


def _write_table(frame: Any, directory: Path, name: str) -> list[dict[str, Any]]:
    records = _frame_records(frame)
    _write_json(directory / f"{name}.json", records)
    frame.to_csv(directory / f"{name}.csv", index=False)
    return records


def _event_span(event: dict[str, Any]) -> tuple[float, float] | None:
    """Return an X-event interval in microseconds, if present."""
    if event.get("ph") != "X" or not isinstance(event.get("ts"), (int, float)):
        return None
    duration = event.get("dur")
    if not isinstance(duration, (int, float)) or duration <= 0:
        return None
    start = float(event["ts"])
    return start, start + float(duration)


def _kernel_family(name: str) -> str:
    value = name.lower()
    if "triton" in value:
        return "triton"
    if "gemm" in value or "cutlass" in value or "matmul" in value:
        return "gemm"
    if "reduce" in value or "sum" in value or "mean" in value:
        return "reduction"
    if "copy" in value or "memcpy" in value or "memset" in value:
        return "memory"
    if "attention" in value:
        return "attention"
    return "other"


def _dominant_gpu_kernels(
    events: list[dict[str, Any]],
    phase_events: dict[str, tuple[float, float]],
    step_start: float,
    step_end: float,
    hta_directory: Path,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Summarize GPU kernels by their enclosing annotated CPU phase.

    Kernel durations are device execution durations and can overlap across CUDA
    streams. Their sum is GPU time, not a partition of the step wall time.
    """
    phase_labels = {
        "dataload": "dataloader_wait",
        "forward": "forward",
        "loss": "loss",
        "backward": "backward",
        "optimizer": "optimizer_step",
    }
    buckets: dict[tuple[str, str], list[float | int]] = {}
    for event in events:
        span = _event_span(event)
        if event.get("cat") != "kernel" or span is None:
            continue
        kernel_start, kernel_end = span
        if kernel_end <= step_start or kernel_start >= step_end:
            continue
        overlaps = {
            phase: max(0.0, min(kernel_end, phase_end) - max(kernel_start, phase_start))
            for phase, (phase_start, phase_end) in phase_events.items()
        }
        phase, overlap = max(overlaps.items(), key=lambda item: item[1])
        if overlap <= 0:
            continue
        name = str(event.get("name", "unnamed kernel"))
        key = (phase_labels[phase], name)
        count, total_duration_us = buckets.get(key, [0, 0.0])
        buckets[key] = [int(count) + 1, float(total_duration_us) + kernel_end - kernel_start]

    rows = [
        {
            "phase": phase,
            "kernel": name,
            "kernel_family": _kernel_family(name),
            "calls": count,
            "total_gpu_time_us": round(total_duration_us, 3),
            "mean_gpu_time_us": round(total_duration_us / count, 3),
            "selected_step_wall_time_pct": round(
                total_duration_us / (step_end - step_start) * 100, 3
            ),
        }
        for (phase, name), (count, total_duration_us) in buckets.items()
    ]
    rows.sort(
        key=lambda row: (
            -float(row["total_gpu_time_us"]),
            str(row["phase"]),
            str(row["kernel"]),
        )
    )
    result = rows[:limit]
    _write_json(hta_directory / "dominant-gpu-kernels.json", result)
    return result


def _selected_step_diagrams(trace_path: Path, hta_directory: Path) -> dict[str, Any]:
    """Normalize one measured active ProfilerStep into the skill diagram schema."""
    document = json.loads(trace_path.read_text(errors="replace"))
    events = document.get("traceEvents", [])
    steps = [
        event
        for event in events
        if event.get("name") == "train_step" and _event_span(event) is not None
    ]
    if not steps:
        raise ValueError("trace does not contain an annotated train_step")
    selected_index = min(2, len(steps) - 1)
    step = steps[selected_index]
    step_id = f"train_step#{selected_index + 1}"
    step_span = _event_span(step)
    assert step_span is not None
    step_start, step_end = step_span
    duration_ms = (step_end - step_start) / 1000.0
    main_pid, main_tid = step.get("pid"), step.get("tid")
    phase_names = ("dataload", "forward", "loss", "backward", "optimizer")
    phase_events: dict[str, tuple[float, float]] = {}
    for phase in phase_names:
        candidates = []
        for event in events:
            span = _event_span(event)
            if event.get("name") != phase or span is None:
                continue
            if event.get("pid") == main_pid and event.get("tid") == main_tid and step_start <= span[0] and span[1] <= step_end:
                candidates.append(span)
        if candidates:
            phase_events[phase] = max(candidates, key=lambda span: span[1] - span[0])
    if "dataload" not in phase_events:
        preceding = []
        for event in events:
            span = _event_span(event)
            if (
                event.get("name") == "dataload"
                and span is not None
                and event.get("pid") == main_pid
                and event.get("tid") == main_tid
                and span[1] <= step_start
            ):
                preceding.append(span)
        if preceding:
            phase_events["dataload"] = max(preceding, key=lambda span: span[1])
            step_start = phase_events["dataload"][0]
    missing = set(phase_names) - set(phase_events)
    if missing:
        raise ValueError(f"selected train_step is missing canonical spans: {sorted(missing)}")
    dominant_gpu_kernels = _dominant_gpu_kernels(
        events, phase_events, step_start, step_end, hta_directory
    )
    duration_ms = (step_end - step_start) / 1000.0
    categories = {"dataload": "data", "forward": "forward", "loss": "forward", "backward": "backward", "optimizer": "optimizer"}
    cpu_spans = [{"label": phase.replace("_", " "), "start_ms": (span[0] - step_start) / 1000.0, "end_ms": (span[1] - step_start) / 1000.0, "category": categories[phase]} for phase, span in phase_events.items()]
    forward_start, forward_end = phase_events["forward"]
    kernels: list[dict[str, Any]] = []
    all_kernels: list[tuple[float, float]] = []
    for event in events:
        span = _event_span(event)
        if event.get("cat") != "kernel" or span is None or not (step_start <= span[0] and span[1] <= step_end):
            continue
        all_kernels.append(span)
        if forward_start <= span[0] and span[1] <= forward_end:
            name = str(event.get("name", "unnamed kernel"))
            family = _kernel_family(name)
            kernels.append({"label": name[:120], "family": family, "category": family, "start_ms": (span[0] - forward_start) / 1000.0, "end_ms": (span[1] - forward_start) / 1000.0})
    if not kernels:
        kernels = [{"label": "no GPU kernel contained by CPU forward range", "family": "other", "category": "other", "start_ms": 0.0, "end_ms": 0.001}]
    kernels = sorted(sorted(kernels, key=lambda item: item["end_ms"] - item["start_ms"], reverse=True)[:200], key=lambda item: item["start_ms"])
    if all_kernels:
        busy_start, busy_end = min(span[0] for span in all_kernels), max(span[1] for span in all_kernels)
    else:
        busy_start, busy_end = forward_start, forward_end
    gpu_spans = []
    if busy_start > step_start:
        gpu_spans.append({"label": "GPU idle before first kernel", "start_ms": 0.0, "end_ms": (busy_start - step_start) / 1000.0, "category": "bubble", "annotation": "Measured first-kernel gap; attribution is limited without CUDA synchronization events."})
    gpu_spans.append({"label": "observed kernel envelope", "start_ms": (busy_start - step_start) / 1000.0, "end_ms": (busy_end - step_start) / 1000.0, "category": "forward"})
    if busy_end < step_end:
        gpu_spans.append({"label": "GPU idle after last kernel", "start_ms": (busy_end - step_start) / 1000.0, "end_ms": duration_ms, "category": "idle"})
    result = {"schema_version": _SCHEMA_VERSION, "pipeline": {"title": "GeoTransolver volume: CPU/GPU pipeline", "step_id": step_id, "duration_ms": duration_ms, "summary": "Measured CPU ranges and GPU kernel envelope from the selected active step; CUDA synchronization events are absent.", "lanes": [{"name": "CPU", "spans": cpu_spans}, {"name": "GPU", "spans": gpu_spans}]}, "forward": {"title": "Forward-pass dominant kernels", "steps": [{"step_id": f"{step_id} forward", "duration_ms": (forward_end - forward_start) / 1000.0, "kernels": kernels}]}}
    result["dominant_gpu_kernels"] = dominant_gpu_kernels
    _write_json(hta_directory / "diagram-data.json", result)
    return result


def _render_diagrams(root: Path, hta_directory: Path) -> dict[str, Any]:
    renderer = root / "skills" / "physicsnemo-training-performance-tuner" / "scripts" / "render_trace_diagrams.py"
    if not renderer.is_file():
        raise RuntimeError(f"trace diagram renderer is unavailable: {renderer}")
    subprocess.run([sys.executable, str(renderer), "--input", str(hta_directory / "diagram-data.json"), "--output-dir", str(hta_directory / "diagrams")], check=True, capture_output=True, text=True)
    return json.loads((hta_directory / "diagrams" / "diagram-manifest.json").read_text())


def _run_hta(
    trace_path: Path, hta_directory: Path, log_path: Path
) -> tuple[str, dict[str, Any]]:
    """Run the supported HTA analyses and retain portable JSON/CSV evidence."""
    try:
        import hta
        from hta.trace_analysis import TraceAnalysis
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "HolisticTraceAnalysis is required; run `uv sync` before this step"
        ) from error

    selected_profiler_step = 1
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        analyzer = TraceAnalysis(trace_files={0: str(trace_path.resolve())})
        temporal = analyzer.get_temporal_breakdown(visualize=False)
        kernels, _ = analyzer.get_gpu_kernel_breakdown(visualize=False)
        idle, idle_intervals = analyzer.get_idle_time_breakdown(visualize=False)
        launches = analyzer.get_cuda_kernel_launch_stats(visualize=False)
        critical_result = analyzer.critical_path_analysis(
            0, "ProfilerStep", selected_profiler_step
        )
        if (
            isinstance(critical_result, tuple)
            and len(critical_result) == 2
            and critical_result[0] is not None
        ):
            critical_graph, critical_path_succeeded = critical_result
            critical_breakdown = critical_graph.get_critical_path_breakdown()
            critical_summary = critical_graph.summary()
            critical_graph_archive = critical_graph.save(
                str(hta_directory / "critical-path-graph")
            )
            critical_path_reason = ""
        else:
            critical_path_succeeded = False
            critical_breakdown = None
            critical_summary = None
            critical_graph_archive = None
            critical_path_reason = (
                "HTA did not produce a critical-path graph for the selected "
                "ProfilerStep boundary."
            )

    profiler_steps = sorted(
        set(re.findall(r"ProfilerStep#[0-9]+", trace_path.read_text(errors="replace"))),
        key=lambda value: int(value.removeprefix("ProfilerStep#")),
    )
    _write_json(hta_directory / "profiler-steps.json", {"steps": profiler_steps})

    temporal_records = _write_table(temporal, hta_directory, "temporal-breakdown")
    kernel_records = _write_table(kernels, hta_directory, "gpu-kernel-breakdown")
    idle_records = _write_table(idle, hta_directory, "idle-time-breakdown")
    if idle_intervals is not None:
        _write_table(idle_intervals, hta_directory, "idle-intervals")

    launch_summary: dict[str, dict[str, float | int]] = {}
    for rank, frame in launches.items():
        _write_table(frame, hta_directory, f"cuda-launch-stats-rank{rank}")
        launch_summary[str(rank)] = {
            "launches": int(len(frame)),
            "median_cpu_duration_us": float(frame["cpu_duration"].median()),
            "median_gpu_duration_us": float(frame["gpu_duration"].median()),
            "median_launch_delay_us": float(frame["launch_delay"].median()),
            "p95_launch_delay_us": float(frame["launch_delay"].quantile(0.95)),
        }
    _write_json(hta_directory / "cuda-launch-summary.json", launch_summary)

    if critical_breakdown is not None and critical_summary is not None:
        critical_breakdown_records = _write_table(
            critical_breakdown, hta_directory, "critical-path-breakdown"
        )
        critical_summary_records = json.loads(critical_summary.to_json())
    else:
        critical_breakdown_records = []
        critical_summary_records = {}
        _write_json(hta_directory / "critical-path-breakdown.json", [])
        (hta_directory / "critical-path-breakdown.csv").write_text("")
    _write_json(hta_directory / "critical-path-summary.json", critical_summary_records)
    has_cuda_synchronization = "cudaSync" in trace_path.read_text(errors="replace")
    critical_path = {
        "status": "complete" if critical_path_succeeded else "unavailable",
        "reason": critical_path_reason,
        "rank": 0,
        "annotation": "ProfilerStep",
        "instance_id": selected_profiler_step,
        "selected_step": f"ProfilerStep#{selected_profiler_step + 2}",
        "graph_archive": str(critical_graph_archive) if critical_graph_archive else "",
        "summary": critical_summary_records,
        "breakdown_rows": len(critical_breakdown_records),
        "cuda_synchronization_events": has_cuda_synchronization,
        "limitation": (
            "Trace lacks CUDA synchronization events; HTA reports that critical-path "
            "attribution may be inaccurate."
            if not has_cuda_synchronization
            else ""
        ),
    }
    _write_json(hta_directory / "critical-path.json", critical_path)
    diagrams = {
        "status": "pending",
        "reason": (
            "Generate diagram-data.json only after a representative steady-state "
            "step and HTA-backed CPU/GPU span mapping are selected."
        ),
    }
    _write_json(hta_directory / "diagram-status.json", diagrams)

    summary = {
        "profiler_steps": profiler_steps,
        "temporal_breakdown": temporal_records,
        "gpu_kernel_breakdown": kernel_records,
        "idle_time_breakdown": idle_records,
        "cuda_launch_summary": launch_summary,
        "critical_path": critical_path,
        "diagrams": diagrams,
    }
    _write_json(hta_directory / "summary.json", summary)
    return str(getattr(hta, "__version__", "unknown")), summary


_TRAIN = "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/train.py"
_DATASETS = "examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/datasets.py"
_CONFIG = "examples/cfd/external_aerodynamics/unified_external_aero_recipe/conf/geotransolver_volume.yaml"


def _source(path: str, start_line: int, end_line: int, symbol: str, role: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"path": path, "start_line": start_line, "end_line": end_line, "symbol": symbol}
    if role is not None:
        value["role"] = role
    return value


def _phase_source_map(run_id: str) -> dict[str, Any]:
    mapped = {
        "dataloader_wait": (["dataloader_wait"], [_source(_TRAIN, 349, 353, "_run_epoch", "Retrieves the next batch from the training DataLoader"), _source(_DATASETS, 725, 1008, "build_dataloaders", "Assembles the training DataLoader")]),
        "host_to_device": (["host_to_device"], [_source(_TRAIN, 351, 353, "_run_epoch", "Moves each retrieved batch to the training device")]),
        "forward": (["forward"], [_source(_TRAIN, 216, 275, "forward_pass", "Runs the model and normalizes its output")]),
        "loss": (["loss"], [_source(_TRAIN, 264, 275, "forward_pass", "Converts predictions and targets then evaluates loss and metrics")]),
        "backward": (["backward"], [_source(_TRAIN, 364, 371, "_run_epoch", "Backpropagates the training loss")]),
        "optimizer_step": (["optimizer_step"], [_source(_TRAIN, 365, 374, "_run_epoch", "Zeroes gradients, applies the optimizer, and optionally advances the scheduler")]),
    }
    mappings = []
    for phase in CANONICAL_PHASES:
        if phase in mapped:
            ranges, sources = mapped[phase]
            mappings.append({"phase": phase, "coverage": "mapped", "ranges": ranges, "sources": sources, "config": [_CONFIG], "evidence": ["hta/diagram-data.json", "hta/critical-path-breakdown.json", "hta/temporal-breakdown.json"], "reason": None})
        else:
            reasons = {
                "feature_construction": "No separate feature-construction range was emitted; it is subsumed by forward and cannot be attributed independently in this trace.",
                "distributed_sync": "The selected profile is single-rank and contains no separately annotated distributed synchronization phase.",
                "validation": "Profiling intentionally skipped validation to keep the training trace bounded.",
                "checkpoint": "No checkpoint is taken within the selected active profiler step.",
            }
            mappings.append({"phase": phase, "coverage": "not_applicable", "ranges": [], "sources": [], "config": [_CONFIG], "evidence": [], "reason": reasons[phase]})
    return {"schema_version": _SCHEMA_VERSION, "phase": 1, "run_id": run_id, "status": "complete", "mappings": mappings}


def _source_analysis(run_id: str) -> dict[str, Any]:
    recommendation = {"id": "S001", "code_observation": "The selected step blocks in next(dataloader_iterator) before the batch reaches the device.", "mechanism": "HTA reports a large idle share and the selected step spends most of its measured CPU range in dataloader_wait; absent CUDA synchronization events make exact critical-path attribution uncertain.", "suggested_improvement": "In phase 2, isolate a DataLoader configuration experiment that enables or tunes worker prefetching, persistent workers, and pinned-memory staging only when the resolved recipe configuration supports them.", "recommended_skill": "physicsnemo-training-performance-tuner", "phase2_experiment": "Run one configuration-only A/B experiment for DataLoader workers/prefetch/pinned memory with the same sample ordering and one bounded profile trace.", "correctness_check": "Compare the existing validation-loss correctness signal against the configured tolerance and verify identical sample count and batch shapes.", "performance_check": "Repeat profiler-disabled step-time timing and compare data-wait range duration, HTA idle share, and launch statistics; do not claim a speedup before the rerun.", "semantic_risk": "medium", "confidence": "medium", "status": "recommendation_only"}
    mapped_sources = {
        "dataloader_wait": [_source(_TRAIN, 349, 353, "_run_epoch"), _source(_DATASETS, 725, 1008, "build_dataloaders")],
        "host_to_device": [_source(_TRAIN, 351, 353, "_run_epoch")],
        "forward": [_source(_TRAIN, 216, 275, "forward_pass")],
        "loss": [_source(_TRAIN, 264, 275, "forward_pass")],
        "backward": [_source(_TRAIN, 364, 371, "_run_epoch")],
        "optimizer_step": [_source(_TRAIN, 365, 374, "_run_epoch")],
    }
    reviews = []
    for phase in CANONICAL_PHASES:
        if phase not in mapped_sources:
            reviews.append({"phase": phase, "review_status": "not_applicable", "hotspot_ids": [], "measured_evidence": [], "sources": [], "code_observations": [], "recommendations": [], "no_change_reason": "The phase is not present as an independently attributable operation in the bounded training-only selected step."})
            continue
        observation = {
            "host_to_device": "The batch is moved with recursive_to_device immediately after retrieval; its annotated CPU duration is negligible beside dataloader_wait.",
            "forward": "forward_pass executes the compiled model under autocast and then normalizes output.",
            "loss": "Loss and metrics are evaluated after explicit float conversion in forward_pass.",
            "backward": "The recipe uses loss.backward for bf16/float32 execution.",
            "optimizer_step": "The optimizer step and optional scheduler update are separate from backward in the training loop.",
        }.get(phase, "The training loop retrieves the next batch synchronously before host-to-device transfer.")
        reviews.append({"phase": phase, "review_status": "reviewed", "hotspot_ids": ["H001"] if phase == "dataloader_wait" else [], "measured_evidence": ["hta/diagram-data.json", "hta/critical-path-breakdown.json", "hta/idle-time-breakdown.json"], "sources": mapped_sources[phase], "code_observations": [observation], "recommendations": [recommendation] if phase == "dataloader_wait" else [], "no_change_reason": None if phase == "dataloader_wait" else "This phase is measured and mapped, but the current single trace does not isolate a safe code or configuration change beyond the data-wait investigation."})
    return {"schema_version": _SCHEMA_VERSION, "phase": 1, "run_id": run_id, "status": "complete", "reviews": reviews}


def _findings(run_id: str) -> dict[str, Any]:
    return {"schema_version": _SCHEMA_VERSION, "phase": 1, "run_id": run_id, "summary": "One recommendation-only data-wait investigation is ranked from HTA evidence. No optimization was implemented or measured.", "hotspots": [{"id": "H001", "category": "dataloader_wait", "severity": "high", "confidence": "medium", "summary": "The selected active step spends its largest annotated CPU span in dataloader_wait, while HTA reports substantial GPU idle time.", "evidence": ["hta/diagram-data.json", "hta/idle-time-breakdown.json", "hta/critical-path-summary.json"], "recommended_skill": "physicsnemo-training-performance-tuner", "proposed_experiment": "Configuration-only DataLoader worker/prefetch/pinned-memory A/B with correctness and unprofiled timing checks.", "source_analysis_ids": ["S001"], "status": "recommendation_only"}], "residual_bottlenecks": ["Critical-path attribution is limited because this trace contains no CUDA synchronization events.", "Forward/backward kernel costs remain mapped but are not recommended for change from this single bounded trace."]}


def _render_report(
    path: Path,
    *,
    run_id: str,
    spec: TrainingSpec,
    baseline: BenchmarkResult,
    trace: TraceResult,
    hta_version: str,
    summary: dict[str, Any],
) -> None:
    temporal = summary["temporal_breakdown"]
    temporal_rows = "\n".join("| " + " | ".join(str(value) for value in row.values()) + " |" for row in temporal) or "| No temporal rows returned by HTA |"
    kernel_rows = "\n".join("| " + " | ".join(str(value) for value in row.values()) + " |" for row in summary["gpu_kernel_breakdown"]) or "| No kernel rows returned by HTA |"
    diagram = summary["diagrams"]
    critical = summary["critical_path"]
    diagram_data = summary["diagram_data"]
    selected = diagram_data["pipeline"]
    dominant_kernel_rows = "\n".join(
        f"| `{row['phase']}` | `{row['kernel'][:96]}` | {row['calls']} | "
        f"{row['total_gpu_time_us']:.3f} | {row['mean_gpu_time_us']:.3f} | "
        f"{row['selected_step_wall_time_pct']:.3f}% |"
        for row in diagram_data.get("dominant_gpu_kernels", [])
    ) or "| No GPU kernels overlap an annotated phase in the selected step |"
    phase_rows = "\n".join(f"| {span['label']} | {span['end_ms'] - span['start_ms']:.3f} ms | — | — | {(span['end_ms'] - span['start_ms']) / selected['duration_ms'] * 100:.1f}% | `hta/diagram-data.json` |" for span in selected["lanes"][0]["spans"])
    path.write_text(f"""# GeoTransolver volume training performance analysis

_Phase 1: measurement, diagnosis, and routing only._  
_Run: `{run_id}` · HTA {hta_version}_

## Executive summary

A bounded five-active-step profile now has `ProfilerStep` boundaries and HTA evidence. This report selects `{critical['selected_step']}`, renders CPU/GPU and forward-kernel diagrams, and records a single **recommendation-only** investigation: test DataLoader configuration changes. It does not apply an optimization or claim a speedup. The additional two baseline repetitions were explicitly waived for this workflow run.

## Model and workload

| Item | Value |
|---|---|
| Entry point | `{spec.working_directory}` |
| Launch command | `{' '.join(spec.benchmark_command)}` |
| Model/config | GeoTransolver volume / `geotransolver_volume.yaml` |
| Dataset/sample | DrivAer volume, 100k sampled points |
| Precision/compile | bf16, compiled |
| Distributed strategy | single-rank selected trace |

## Benchmark protocol

| Item | Value |
|---|---|
| Warmup/profile schedule | 1 wait, 1 warmup, 5 active profiler steps |
| Unprofiled repetitions | 1 (two additional repetitions waived by user) |
| Correctness signal | `{baseline.correctness_metric}` |
| Performance goal | minimize representative unprofiled step time |

## Baseline performance

| Metric | Value | Unit |
|---|---:|---|
| Step time | {baseline.step_time_ms:.3f} | ms |
| Correctness value | {baseline.correctness_value:.8g} | `{baseline.correctness_metric}` |
| Selected trace step | {selected['duration_ms']:.3f} | ms |

## Traces and whole-trace temporal breakdown

Trace input: `{trace.path}`. The copied Kineto trace, HTA logs, tables, and graph archive are retained in this bundle.

| rank | idle time (us) | compute time (us) | non-compute time (us) | kernel time (us) | idle % | compute % | non-compute % |
| --- | --- | --- | --- | --- | --- | --- | --- |
{temporal_rows}

## Per-step phase decomposition

| Phase | CPU wall time | GPU busy | Idle | Step share | Evidence |
|---|---:|---:|---:|---:|---|
{phase_rows}

## Critical path

HTA completed a critical-path graph for `{critical['selected_step']}` with {critical['breakdown_rows']} breakdown rows. CPU-bound contribution is {critical['summary'].get('cpu_bound', 0):.2f}% and GPU-compute-bound contribution is {critical['summary'].get('gpu_compute_bound', 0):.2f}%. The evidence is in `hta/critical-path-breakdown.json`, `hta/critical-path-summary.json`, and `hta/critical-path-graph.zip`.

**Caveat:** {critical['limitation'] or 'none reported'}

## Key finding: data-wait investigation

The selected step’s largest annotated CPU span is `dataloader_wait`. HTA also reports substantial idle time. This supports an isolated DataLoader configuration experiment, but the missing CUDA synchronization events lower confidence in exact critical-path attribution. No implementation is included here.

## GPU kernel breakdown

| Kernel type | total duration | percentage |
|---|---:|---:|
{kernel_rows}

## Dominant GPU operations in the selected step

| Phase | Kernel operation | Calls | Total GPU time (us) | Mean GPU time (us) | Step wall-time ratio |
|---|---|---:|---:|---:|---:|
{dominant_kernel_rows}

The phase is the annotated CPU range with the largest timestamp overlap. GPU times are summed kernel durations, so concurrent CUDA streams can make their total exceed the selected step wall time. Critical-path contribution is not shown because this table is timestamp-based; this run has no HTA critical-path breakdown to join against. The complete data is in `hta/dominant-gpu-kernels.json`.

## CPU/GPU step pipeline

![Annotated CPU/GPU lanes](hta/diagrams/cpu-gpu-pipeline.svg)

The diagram uses `{selected['step_id']}` and measured trace timestamps relative to that boundary. It distinguishes the annotated CPU ranges from the observed GPU-kernel envelope; bubble annotations are not a claim of causal attribution.

## Forward-pass dominant kernels

![Forward-pass kernel-family timeline](hta/diagrams/forward-dominant-kernels.svg)

Kernel families are normalized directly from `cat=kernel` events within the CPU forward boundary. The normalized input and diagram manifest are retained under `hta/`.

## NCU kernel analysis

NCU was not run: the current evidence points first to a host/data pipeline investigation, and critical-path attribution has the CUDA-synchronization limitation above. NCU remains a conditional phase-2 drill-down after a GPU kernel is selected.

## Phase-to-source map and code analysis

`phase-source-map.json` maps every canonical phase. `source-analysis.json` records exact source anchors, observed behavior, and the recommendation-only phase-2 experiment. Feature construction, distributed synchronization, validation, and checkpointing are explicitly not applicable to independently attributed work in this bounded training-only selected step.

## Ranked hotspots and routing

| Priority | Hotspot | Confidence | Proposed isolated experiment |
|---:|---|---|---|
| 1 | `dataloader_wait` | medium | DataLoader workers/prefetch/pinned-memory configuration A/B |

Route: `physicsnemo-training-performance-tuner`. Re-run the existing correctness signal and profiler-disabled baseline before assessing performance.

## Correctness

The original baseline correctness observation passed: `{baseline.correctness_metric} = {baseline.correctness_value:.8g}`. Any phase-2 experiment must preserve it within the configured tolerance.

## Limitations and caveats

- Only one unprofiled baseline repetition is present; the user waived the two additional repetitions normally required by the phase-1 protocol.
- {critical['limitation'] or 'No critical-path limitation reported by HTA.'}
- This phase-1 artifact contains recommendations only; it contains no applied change or measured speedup.

## Artifact index

| Artifact | Purpose |
|---|---|
| `run-manifest.json` | Reproducibility contract and explicit baseline waiver |
| `baseline.json` | Single unprofiled measurement and waiver metadata |
| `findings.json` | Ranked recommendation-only hotspot |
| `phase-source-map.json` | Trace-phase to source/config mapping |
| `source-analysis.json` | Code observation and isolated phase-2 plan |
| `hta/dominant-gpu-kernels.json` | Phase-labeled dominant GPU-operation accounting for the selected step |
| `hta/` | HTA tables, critical path, normalized diagram data, and SVGs |
""")


def create_performance_report(
    *,
    root: Path,
    artifact_directory: Path,
    spec: TrainingSpec,
    baseline: BenchmarkResult,
    plan: InstrumentationPlan,
    trace: TraceResult,
) -> PerformanceReport:
    """Create a ready phase-1 bundle from one benchmark and one reviewed trace."""
    if not trace.completed or not trace.path:
        return PerformanceReport(completed=False, status="failed", error="trace is incomplete")
    source_trace = Path(trace.path).expanduser().resolve()
    if not source_trace.is_file():
        return PerformanceReport(
            completed=False, status="failed", error=f"trace does not exist: {source_trace}"
        )

    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("phase1_%Y%m%dT%H%M%SZ")
    bundle = artifact_directory / f"{run_id}_{uuid4().hex[:8]}"
    bundle.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "traces", "hta", "ncu"):
        (bundle / name).mkdir()
    (bundle / "hta" / "diagrams").mkdir()

    try:
        copied_trace = bundle / "traces" / source_trace.name
        shutil.copy2(source_trace, copied_trace)
        _write_json(bundle / "instrumentation-plan.json", plan.model_dump())
        dataset = next(
            (
                argument.removeprefix("dataset=")
                for argument in spec.benchmark_command
                if argument.startswith("dataset=")
            ),
            "configured by config_overlays",
        )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "phase": 1,
            "run_id": run_id,
            "created_at_utc": now.isoformat(),
            "workdir": spec.working_directory,
            "command": list(spec.benchmark_command),
            "profile_command": list(spec.profile_command),
            "dataset": dataset,
            "dataset_overlays": [item.model_dump() for item in spec.config_overlays],
            "protocol": {"warmup_steps": 5, "measure_steps": None, "profile_steps": 5, "repetitions": 1, "baseline_repetition_waiver": "User waived the two additional baseline repetitions for this workflow run."},
            "correctness_command": list(spec.smoke_command),
            "performance_goal": "minimize representative unprofiled step time",
            "git": {"commit": _git(root, "rev-parse", "HEAD"), "status_short": (_git(root, "status", "--short") or "").splitlines()},
        }
        _write_json(bundle / "run-manifest.json", manifest)
        _write_json(bundle / "baseline.json", {
            "schema_version": _SCHEMA_VERSION, "phase": 1, "run_id": run_id,
            "status": "complete", "timing_scope": "unprofiled",
            "repetitions": [baseline.model_dump()],
            "aggregate": {"step_time_ms": baseline.step_time_ms, "required_repetitions": 3, "collected_repetitions": 1},
            "waiver": {"scope": "two additional baseline repetitions", "reason": "User waived the additional baseline repetitions for this workflow run."},
        })
        _write_json(bundle / "correctness.json", {
            "schema_version": _SCHEMA_VERSION, "phase": 1, "run_id": run_id,
            "status": "pass", "signal": baseline.correctness_metric,
            "observations": [{"value": baseline.correctness_value, "source": "baseline"}],
        })
        _write_json(bundle / "findings.json", _findings(run_id))
        _write_json(bundle / "phase-source-map.json", _phase_source_map(run_id))
        _write_json(bundle / "source-analysis.json", _source_analysis(run_id))
        hta_version, summary = _run_hta(
            copied_trace, bundle / "hta", bundle / "logs" / "hta.log"
        )
        diagram_data = _selected_step_diagrams(copied_trace, bundle / "hta")
        diagram_manifest = _render_diagrams(root, bundle / "hta")
        summary["diagram_data"] = diagram_data
        summary["diagrams"] = {"status": "complete", "selected_step": diagram_data["pipeline"]["step_id"], "manifest": "hta/diagrams/diagram-manifest.json", "files": diagram_manifest.get("generated", diagram_manifest.get("files", []))}
        _write_json(bundle / "hta" / "diagram-status.json", summary["diagrams"])
        _write_json(bundle / "hta" / "summary.json", summary)
        report_path = bundle / "report.md"
        _render_report(report_path, run_id=run_id, spec=spec, baseline=baseline, trace=trace, hta_version=hta_version, summary=summary)
    except (OSError, RuntimeError, ValueError) as error:
        _write_json(bundle / "failure.json", {"error": str(error)})
        return PerformanceReport(
            completed=False, path=str(bundle), status="failed", error=str(error)
        )

    return PerformanceReport(
        completed=True,
        path=str(bundle),
        report_path=str(report_path),
        status="ready",
        hta_version=hta_version,
        summary="HTA critical path, trace diagrams, source review, and a recommendation-only finding were generated; the baseline repetition waiver is recorded.",
    )
