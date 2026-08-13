"""Values passed between training-optimizer agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingRequest(Value):
    description: str = Field(min_length=1)


class TrainingSpec(Value):
    smoke_command: tuple[str, ...]
    benchmark_command: tuple[str, ...]
    profile_command: tuple[str, ...]
    correctness_tolerance: float = Field(ge=0)
    unresolved: tuple[str, ...] = ()


class Critique(Value):
    accepted: bool
    feedback: str = ""


class RunResult(Value):
    completed: bool
    error: str = ""


class BenchmarkResult(Value):
    step_time_ms: float = Field(gt=0)
    correctness_value: float = Field(allow_inf_nan=False)


class InstrumentationPlan(Value):
    patch: str
    ranges: tuple[str, ...]


class TraceResult(Value):
    completed: bool
    path: str = ""
    ranges: tuple[str, ...] = ()
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
    patch: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


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
