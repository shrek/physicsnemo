"""NOOA agents: ellipsis methods propose; Python methods execute and verify."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from nooa import Agent, CodeActStrategy, PredictStrategy, hidden, strategy
from nooa.unifiedllm import UnifiedLLM

from simplified.environment import SourceEnvironment, TrainingEnvironment
from simplified.types import (
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    Hotspot,
    HotspotAnalysis,
    InstrumentationPlan,
    OptimizationResult,
    Route,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)


REQUIRED_RANGES = (
    "data_loading",
    "host_to_device",
    "forward",
    "loss",
    "backward",
    "optimizer_step",
)


class SourceAgent(Agent):
    source: Annotated[SourceEnvironment | None, hidden]

    def __init__(self, *, source: SourceEnvironment | None = None, **kwargs):
        super().__init__(**kwargs)
        self.source = source

    def list_files(self, pattern: str = "**/*", limit: int = 200) -> list[str]:
        """List source files matching a glob."""
        return self._source().list_files(pattern, limit)

    def search(self, text: str, limit: int = 100) -> list[str]:
        """Find text in the source repository."""
        return self._source().search(text, limit)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        """Read a bounded source excerpt."""
        return self._source().read_file(path, start_line, end_line)

    @hidden
    def _source(self) -> SourceEnvironment:
        if self.source is None:
            raise RuntimeError("source environment is required")
        return self.source


class InputProposer(SourceAgent):
    @strategy(CodeActStrategy())
    async def propose(
        self, request: TrainingRequest, previous: Critique | None
    ) -> TrainingSpec:
        """Inspect the repository and propose reproducible smoke, benchmark, and profile
        commands plus an absolute tolerance for the numeric correctness signal. Resolve the
        critic's feedback. Record uncertain or missing facts in unresolved.
        """
        ...


class InputCritic(Agent):
    def review(self, spec: TrainingSpec) -> Critique:
        missing = []
        for name in ("smoke_command", "benchmark_command", "profile_command"):
            if not getattr(spec, name):
                missing.append(name)
        missing.extend(spec.unresolved)
        return Critique(
            accepted=not missing,
            feedback="Missing or unresolved: " + ", ".join(missing) if missing else "",
        )


class Runner(Agent):
    environment: Annotated[TrainingEnvironment, hidden]

    def __init__(self, *, environment: TrainingEnvironment, **kwargs):
        super().__init__(**kwargs)
        self.environment = environment

    def smoke(self, spec: TrainingSpec) -> RunResult:
        return self.environment.smoke(spec)

    def benchmark(self, spec: TrainingSpec) -> BenchmarkResult:
        return self.environment.benchmark(spec)

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult:
        return self.environment.profile(spec, plan)

    def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult:
        return self.environment.benchmark_candidate(spec, proposal)


class InstrumentationProposer(SourceAgent):
    @strategy(CodeActStrategy())
    async def propose(
        self, spec: TrainingSpec, required_ranges: tuple[str, ...], previous: Critique | None
    ) -> InstrumentationPlan:
        """Inspect the training loop and propose the smallest instrumentation patch covering
        every required range. Resolve the verifier's feedback without changing training behavior.
        """
        ...


class TraceCritic(Agent):
    def review(self, trace: TraceResult, required_ranges: tuple[str, ...]) -> Critique:
        missing = sorted(set(required_ranges) - set(trace.ranges))
        problems = []
        if not trace.completed:
            problems.append(trace.error or "profile run failed")
        if not trace.path:
            problems.append("trace path is empty")
        if not trace.summary:
            problems.append("trace summary is empty")
        if missing:
            problems.append("missing ranges: " + ", ".join(missing))
        return Critique(accepted=not problems, feedback="; ".join(problems))


class HotspotAnalyzer(Agent):
    @strategy(PredictStrategy())
    async def analyze(
        self, trace: TraceResult, previous: Critique | None
    ) -> HotspotAnalysis:
        """Rank the trace's supported bottlenecks using only the allowed hotspot kinds.
        Each hotspot must cite concrete evidence from the trace summary. Resolve the critic's
        feedback and do not recommend changes.
        """
        ...


class HotspotCritic(Agent):
    @strategy(PredictStrategy())
    async def review(self, trace: TraceResult, analysis: HotspotAnalysis) -> Critique:
        """Accept only a non-empty, ranked hotspot analysis whose evidence is supported by
        the trace. Give concise, actionable revision feedback when rejecting it.
        """
        ...


class Router(Agent):
    def route(self, hotspot: Hotspot) -> Route:
        routes = {
            "dataloader_wait": "physicsnemo-datapipe-adapter",
            "python_preprocessing": "physicsnemo-datapipe-adapter",
            "geometry_math": "physicsnemo-functionals-integrator",
            "model_forward_backward": "physicsnemo-functionals-integrator",
            "host_to_device": "physicsnemo-shard-tensor-scaling",
            "distributed_scaling": "physicsnemo-shard-tensor-scaling",
        }
        return Route(skill=routes[hotspot.kind])


class ChangeProposer(SourceAgent):
    @strategy(CodeActStrategy())
    async def propose(
        self,
        spec: TrainingSpec,
        hotspot: Hotspot,
        route: Route,
        previous: Critique | None,
    ) -> ChangeProposal:
        """Inspect the source and propose one minimal patch for the routed hotspot. Preserve
        training semantics and resolve the candidate verifier's feedback. Do not apply the patch
        or claim a speedup.
        """
        ...


class CandidateCritic(Agent):
    def review(
        self, spec: TrainingSpec, baseline: BenchmarkResult, candidate: CandidateResult
    ) -> Critique:
        problems = []
        result = candidate.benchmark
        if not candidate.completed or result is None:
            problems.append(candidate.error or "candidate run failed")
        else:
            if abs(result.correctness_value - baseline.correctness_value) > spec.correctness_tolerance:
                problems.append("candidate failed correctness tolerance")
            if result.step_time_ms >= baseline.step_time_ms:
                problems.append("candidate did not improve step time")
        return Critique(accepted=not problems, feedback="; ".join(problems))


class ReportBuilder(Agent):
    def build(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
        hotspot: Hotspot,
        proposal: ChangeProposal,
    ) -> OptimizationResult:
        speedup = baseline.step_time_ms / candidate.step_time_ms
        report = (
            f"Hotspot: {hotspot.kind}\n"
            f"Baseline: {baseline.step_time_ms:.3f} ms\n"
            f"Candidate: {candidate.step_time_ms:.3f} ms\n"
            f"Verified speedup: {speedup:.3f}x\n"
            f"Change: {proposal.rationale}"
        )
        return OptimizationResult(
            baseline=baseline,
            candidate=candidate,
            hotspot=hotspot,
            proposal=proposal,
            speedup=speedup,
            report=report,
        )


@dataclass(frozen=True)
class Agents:
    inputs: InputProposer
    input_critic: InputCritic
    runner: Runner
    instrumentation: InstrumentationProposer
    trace_critic: TraceCritic
    hotspots: HotspotAnalyzer
    hotspot_critic: HotspotCritic
    router: Router
    changes: ChangeProposer
    candidate_critic: CandidateCritic
    report: ReportBuilder


def create_agents(
    llm: UnifiedLLM,
    source: SourceEnvironment,
    environment: TrainingEnvironment,
) -> Agents:
    return Agents(
        inputs=InputProposer(llm=llm, source=source),
        input_critic=InputCritic(llm=llm),
        runner=Runner(llm=llm, environment=environment),
        instrumentation=InstrumentationProposer(llm=llm, source=source),
        trace_critic=TraceCritic(llm=llm),
        hotspots=HotspotAnalyzer(llm=llm),
        hotspot_critic=HotspotCritic(llm=llm),
        router=Router(llm=llm),
        changes=ChangeProposer(llm=llm, source=source),
        candidate_critic=CandidateCritic(llm=llm),
        report=ReportBuilder(llm=llm),
    )
