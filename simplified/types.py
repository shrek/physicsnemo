"""Values passed between training-optimizer agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingRequest(Value):
    """Complete, user-authored contract for one optimization run.

    Version zero requires every execution detail up front. The workflow never
    asks an LLM to infer commands, dataset locations, or hardware choices.
    """

    objective: str = Field(
        min_length=1,
        description="Concrete performance objective supplied to the patch proposer.",
    )
    spec: "TrainingSpec" = Field(
        description="Fully specified, reproducible training commands and correctness rule."
    )
    allowed_change_paths: tuple[str, ...] = Field(
        min_length=1,
        description="Repository-relative paths an LLM-proposed optimization may modify.",
    )


class HelloResponse(Value):
    message: str = Field(min_length=1)


class ConfigOverlay(Value):
    """A YAML mapping merged into an existing config only in execution worktrees."""

    path: str = Field(
        min_length=1,
        description="Repository-relative YAML file copied and modified only in execution worktrees.",
    )
    format: Literal["yaml"] = "yaml"
    merge: dict[str, Any] = Field(
        min_length=1,
        description="YAML mapping merged into the disposable copy of path.",
    )


class TrainingSpec(Value):
    """Validated command contract extracted from a detailed TrainingRequest."""

    working_directory: str = Field(default=".", min_length=1, description="Repository-relative command directory.")
    smoke_command: tuple[str, ...] = Field(min_length=1, description="Direct argv for a short real-data training update.")
    benchmark_command: tuple[str, ...] = Field(min_length=1, description="Direct argv whose final stdout line is BenchmarkResult JSON.")
    profile_command: tuple[str, ...] = Field(min_length=1, description="Direct argv whose final stdout line is TraceResult JSON.")
    correctness_tolerance: float = Field(ge=0, description="Maximum baseline-to-candidate correctness difference.")
    config_overlays: tuple[ConfigOverlay, ...] = Field(default=(), description="YAML updates applied only in disposable worktrees.")
    unresolved: tuple[str, ...] = Field(default=(), description="Legacy field; v0 requires it to remain empty.")


class Critique(Value):
    accepted: bool
    feedback: str = ""
    requires_human: bool = False


class RunResult(Value):
    completed: bool
    error: str = ""


class BenchmarkLog(Value):
    completed: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""


class BenchmarkResult(Value):
    """Machine-readable benchmark output emitted by the training command."""

    step_time_ms: float = Field(gt=0, description="Representative post-warmup step latency in milliseconds.")
    correctness_value: float = Field(allow_inf_nan=False, description="Comparable scalar correctness invariant.")
    correctness_metric: str = Field(default="correctness_value", min_length=1, description="Name of correctness_value.")


class BenchmarkInterpretation(Value):
    benchmark: BenchmarkResult | None = None
    error: str = ""


class InstrumentationPlan(Value):
    """LLM-proposed profiling patch checked by deterministic runtime validation."""

    patch: str = Field(description="Unified Git diff containing opt-in profiling instrumentation.")
    ranges: tuple[str, ...] = Field(description="Profiler range names emitted by the patch.")


class TraceResult(Value):
    """Machine-readable final line emitted by the profile command."""

    completed: bool = Field(description="Whether profiling completed successfully.")
    path: str = Field(default="", description="Trace path inside the temporary execution worktree.")
    ranges: tuple[str, ...] = Field(default=(), description="Profiler range names recorded in the trace.")
    summary: str = Field(default="", description="Factual bounded summary used for hotspot analysis.")
    error: str = Field(default="", description="Failure explanation when completed is false.")


class Phase1Report(Value):
    """Typed location and status of a deterministic HTA phase-1 report bundle."""

    completed: bool
    path: str = ""
    report_path: str = ""
    status: Literal["draft", "ready", "failed"] = "draft"
    hta_version: str = ""
    summary: str = ""
    error: str = ""


HotspotKind = Literal[
    "dataloader_wait",
    "python_preprocessing",
    "geometry_math",
    "model_forward_backward",
    "host_to_device",
    "distributed_scaling",
]


class Hotspot(Value):
    kind: HotspotKind
    evidence: str = Field(min_length=1)


class HotspotAnalysis(Value):
    hotspots: tuple[Hotspot, ...]


class Route(Value):
    skill: str


class ChangeProposal(Value):
    """One LLM-proposed source change evaluated by a deterministic benchmark."""

    patch: str = Field(min_length=1, description="Unified Git diff for one targeted optimization.")
    rationale: str = Field(min_length=1, description="Explanation linking the patch to measured hotspot evidence.")


class CandidateResult(Value):
    completed: bool
    benchmark: BenchmarkResult | None = None
    error: str = ""


class OptimizationResult(Value):
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    hotspot: Hotspot
    proposal: ChangeProposal
    speedup: float
    report: str
    phase1_report: Phase1Report | None = None
