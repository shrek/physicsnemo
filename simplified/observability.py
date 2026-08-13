"""NOOA viewer wiring for one trace session per optimizer run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from nooa.tracing import enable_tracing, exporters, flush_traces, set_session

from simplified.types import OptimizationResult, TrainingRequest
from simplified.workflow import TrainingOptimizer


@dataclass(frozen=True)
class ObservedRun:
    session_id: str
    result: OptimizationResult


def trace_to_viewer(endpoint: str = "http://localhost:5001/v1/traces") -> None:
    """Send complete traces to the built-in NOOA viewer."""
    enable_tracing(
        exporters=[exporters.journal(endpoint)],
        experiment="training-optimizer",
    )


def trace_to_files(directory: str | Path = "traces") -> None:
    """Write portable NOOA traces when a live viewer is not running."""
    enable_tracing(
        exporters=[exporters.journal_file(directory)],
        experiment="training-optimizer",
    )


async def run_observed(
    optimizer: TrainingOptimizer,
    request: TrainingRequest,
    *,
    session_id: str | None = None,
) -> ObservedRun:
    """Run the optimizer in a fresh trace session and flush it before returning."""
    session_id = session_id or f"training-optimizer-{uuid4().hex[:12]}"
    set_session(session_id)
    try:
        result = await optimizer.run(request)
        return ObservedRun(session_id=session_id, result=result)
    finally:
        flush_traces()
