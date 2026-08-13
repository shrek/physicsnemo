from __future__ import annotations

import asyncio

from nooa import Agent
from nooa.tracing import enable_tracing
from nooa.unifiedllm import FakeLLMClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from simplified.agents import (
    Agents,
    CandidateCritic,
    HotspotAnalyzer,
    InputCritic,
    ReportBuilder,
    Router,
    TraceCritic,
)
from simplified.observability import run_observed
from simplified.types import (
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    InstrumentationPlan,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)
from simplified.workflow import TrainingOptimizer


SPEC = TrainingSpec(
    smoke_command=("smoke",),
    benchmark_command=("benchmark",),
    profile_command=("profile",),
    correctness_tolerance=0.01,
)


class InputStub(Agent):
    async def propose(self, request, previous):
        return SPEC


class InstrumentationStub(Agent):
    attempts = 0

    async def propose(self, spec, required_ranges, previous):
        self.attempts += 1
        return InstrumentationPlan(patch=f"attempt-{self.attempts}", ranges=required_ranges)


class HotspotCriticStub(Agent):
    async def review(self, trace, analysis):
        return Critique(accepted=True)


class ChangeStub(Agent):
    async def propose(self, spec, hotspot, route, previous):
        return ChangeProposal(patch="prefetch", rationale="prefetch batches")


class RunnerStub(Agent):
    profile_attempts = 0

    def smoke(self, spec):
        return RunResult(completed=True)

    def benchmark(self, spec):
        return BenchmarkResult(step_time_ms=100, correctness_value=1)

    def profile(self, spec, plan):
        self.profile_attempts += 1
        if self.profile_attempts == 1:
            return TraceResult(completed=False, error="profiler failed")
        return TraceResult(
            completed=True,
            path="trace.json",
            ranges=plan.ranges,
            summary="Dataloader wait is 40% of the step.",
        )

    def benchmark_candidate(self, spec, proposal):
        return CandidateResult(
            completed=True,
            benchmark=BenchmarkResult(step_time_ms=80, correctness_value=1),
        )


def test_optimizer_run_exposes_workflow_nodes_retries_and_generation():
    exporter = InMemorySpanExporter()
    enable_tracing(exporters=[exporter], experiment="training-optimizer-test")
    llm = FakeLLMClient.simple_message(
        '{"hotspots":[{"kind":"dataloader_wait","evidence":"40% of the step"}]}'
    )
    agents = Agents(
        inputs=InputStub(llm=llm),
        input_critic=InputCritic(llm=llm),
        runner=RunnerStub(llm=llm),
        instrumentation=InstrumentationStub(llm=llm),
        trace_critic=TraceCritic(llm=llm),
        hotspots=HotspotAnalyzer(llm=llm),
        hotspot_critic=HotspotCriticStub(llm=llm),
        router=Router(llm=llm),
        changes=ChangeStub(llm=llm),
        candidate_critic=CandidateCritic(llm=llm),
        report=ReportBuilder(llm=llm),
    )
    optimizer = TrainingOptimizer(agents, llm=llm)

    observed = asyncio.run(
        run_observed(
            optimizer,
            TrainingRequest(description="tune the training loop"),
            session_id="optimizer-test-run",
        )
    )
    spans = exporter.get_finished_spans()

    assert observed.session_id == "optimizer-test-run"
    assert observed.result.speedup == 1.25
    root = next(span for span in spans if span.name == "method.run")
    trace_critic = next(
        span for span in spans if span.attributes.get("agent.name") == "TraceCritic"
    )
    assert root.attributes["session.id"] == "optimizer-test-run"
    assert trace_critic.parent.span_id == root.context.span_id
    assert sum(
        span.name == "method.propose"
        and span.attributes.get("agent.name") == "InstrumentationStub"
        for span in spans
    ) == 2
    assert any(span.name == "generation" for span in spans)
    assert "input.value" in trace_critic.attributes
    assert "output.value" in trace_critic.attributes
