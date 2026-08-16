"""Interfaces to source code and training execution."""

from __future__ import annotations

from typing import Protocol

from simplified.types import (
    BenchmarkLog,
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    InstrumentationPlan,
    RunResult,
    TraceResult,
    TrainingSpec,
)


class SourceEnvironment(Protocol):
    def list_files(self, pattern: str = "**/*", limit: int = 200) -> list[str]: ...

    def search(self, text: str, limit: int = 100) -> list[str]: ...

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str: ...


class InputValidationEnvironment(Protocol):
    def preflight(self, spec: TrainingSpec) -> RunResult: ...

    def smoke(self, spec: TrainingSpec) -> RunResult: ...


class TrainingEnvironment(InputValidationEnvironment, Protocol):
    def validate_instrumentation(
        self, spec: TrainingSpec, plan: InstrumentationPlan
    ) -> RunResult: ...
    def validate_instrumentation_runtime(
        self, spec: TrainingSpec, plan: InstrumentationPlan
    ) -> RunResult: ...


    def benchmark_log(self, spec: TrainingSpec) -> BenchmarkLog: ...

    def benchmark(self, spec: TrainingSpec) -> BenchmarkResult: ...

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult: ...

    def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult: ...

    def benchmark_candidate_log(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> BenchmarkLog:
        ...
