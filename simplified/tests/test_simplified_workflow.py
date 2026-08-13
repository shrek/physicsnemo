from __future__ import annotations

import asyncio

import pytest
from nooa.unifiedllm import FakeLLMClient

from simplified.agents import (
    Agents,
    CandidateCritic,
    InputCritic,
    ReportBuilder,
    Router,
    TraceCritic,
)
from simplified.types import (
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    Hotspot,
    HotspotAnalysis,
    InstrumentationPlan,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)
from simplified.workflow import TrainingOptimizer, WorkflowError


SPEC = TrainingSpec(
    smoke_command=("python", "train.py", "--steps", "1"),
    benchmark_command=("python", "train.py", "--steps", "20"),
    profile_command=("python", "train.py", "--profile"),
    correctness_tolerance=0.01,
)
BASELINE = BenchmarkResult(step_time_ms=100, correctness_value=1.0)


class Proposer:
    def __init__(self, values):
        self.values = iter(values)
        self.feedback = []

    async def propose(self, *args):
        self.feedback.append(args[-1])
        return next(self.values)


class Analyzer:
    def __init__(self, analysis):
        self.analysis = analysis
        self.feedback = []

    async def analyze(self, trace, previous):
        self.feedback.append(previous)
        return self.analysis


class HotspotReviews:
    def __init__(self, reviews):
        self.reviews = iter(reviews)

    async def review(self, trace, analysis):
        return next(self.reviews)


class Runner:
    def __init__(self, traces, candidates):
        self.traces = iter(traces)
        self.candidates = iter(candidates)

    def smoke(self, spec):
        return RunResult(completed=True)

    def benchmark(self, spec):
        return BASELINE

    def profile(self, spec, plan):
        return next(self.traces)

    def benchmark_candidate(self, spec, proposal):
        return CandidateResult(completed=True, benchmark=next(self.candidates))


def test_feedback_drives_revision_until_every_stage_is_accepted():
    llm = FakeLLMClient()
    bad_trace = TraceResult(completed=False, error="profiler failed")
    good_trace = TraceResult(
        completed=True,
        path="trace.json",
        ranges=(
            "data_loading",
            "host_to_device",
            "forward",
            "loss",
            "backward",
            "optimizer_step",
        ),
        summary="Dataloader consumes 40% of each step.",
    )
    analysis = HotspotAnalysis(
        hotspots=(Hotspot(kind="dataloader_wait", evidence="40% of step"),)
    )
    inputs = Proposer([SPEC])
    instrumentation = Proposer(
        [
            InstrumentationPlan(patch="first", ranges=()),
            InstrumentationPlan(patch="second", ranges=good_trace.ranges),
        ]
    )
    hotspots = Analyzer(analysis)
    changes = Proposer(
        [
            ChangeProposal(patch="first", rationale="prefetch"),
            ChangeProposal(patch="second", rationale="prefetch safely"),
        ]
    )
    runner = Runner(
        [bad_trace, good_trace],
        [
            BenchmarkResult(step_time_ms=100, correctness_value=1.0),
            BenchmarkResult(step_time_ms=80, correctness_value=1.005),
        ],
    )
    agents = Agents(
        inputs=inputs,
        input_critic=InputCritic(llm=llm),
        runner=runner,
        instrumentation=instrumentation,
        trace_critic=TraceCritic(llm=llm),
        hotspots=hotspots,
        hotspot_critic=HotspotReviews(
            [Critique(accepted=False, feedback="cite the percentage"), Critique(accepted=True)]
        ),
        router=Router(llm=llm),
        changes=changes,
        candidate_critic=CandidateCritic(llm=llm),
        report=ReportBuilder(llm=llm),
    )

    result = asyncio.run(
        TrainingOptimizer(agents, llm=llm).run(TrainingRequest(description="tune"))
    )

    assert result.speedup == 1.25
    assert instrumentation.feedback[1].feedback == (
        "profiler failed; trace path is empty; trace summary is empty; missing ranges: "
        "backward, data_loading, forward, host_to_device, loss, optimizer_step"
    )
    assert hotspots.feedback[1].feedback == "cite the percentage"
    assert changes.feedback[1].feedback == "candidate did not improve step time"


def test_trace_retry_exhaustion_stops_before_analysis():
    llm = FakeLLMClient()
    analysis = Analyzer(HotspotAnalysis(hotspots=()))
    failed_trace = TraceResult(completed=False, error="no trace")
    agents = Agents(
        inputs=Proposer([SPEC]),
        input_critic=InputCritic(llm=llm),
        runner=Runner([failed_trace], []),
        instrumentation=Proposer([InstrumentationPlan(patch="x", ranges=())]),
        trace_critic=TraceCritic(llm=llm),
        hotspots=analysis,
        hotspot_critic=HotspotReviews([]),
        router=Router(llm=llm),
        changes=Proposer([]),
        candidate_critic=CandidateCritic(llm=llm),
        report=ReportBuilder(llm=llm),
    )

    with pytest.raises(WorkflowError, match="instrumentation failed"):
        asyncio.run(
            TrainingOptimizer(agents, llm=llm, max_attempts=1).run(
                TrainingRequest(description="tune")
            )
        )

    assert analysis.feedback == []
