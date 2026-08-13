from nooa import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient

from simplified.agents import (
    CandidateCritic,
    ChangeProposer,
    HotspotAnalyzer,
    HotspotCritic,
    InputProposer,
    InstrumentationProposer,
    Router,
)
from simplified.types import BenchmarkResult, CandidateResult, Hotspot, TrainingSpec


def test_agentic_methods_use_the_smallest_suitable_strategy():
    for cls in (InputProposer, InstrumentationProposer, ChangeProposer):
        assert isinstance(cls.propose._strategy_override, CodeActStrategy)
    assert isinstance(HotspotAnalyzer.analyze._strategy_override, PredictStrategy)
    assert isinstance(HotspotCritic.review._strategy_override, PredictStrategy)


def test_router_uses_the_improofbench_hotspot_mapping():
    assert Router(llm=FakeLLMClient()).route(
        Hotspot(kind="host_to_device", evidence="copy stalls")
    ).skill == (
        "physicsnemo-shard-tensor-scaling"
    )


def test_candidate_critic_checks_correctness_and_measured_speedup():
    spec = TrainingSpec(
        smoke_command=("smoke",),
        benchmark_command=("benchmark",),
        profile_command=("profile",),
        correctness_tolerance=0.1,
    )
    baseline = BenchmarkResult(step_time_ms=10, correctness_value=1)
    critic = CandidateCritic(llm=FakeLLMClient())

    accepted = critic.review(
        spec,
        baseline,
        CandidateResult(
            completed=True,
            benchmark=BenchmarkResult(step_time_ms=8, correctness_value=1.05),
        ),
    )
    rejected = critic.review(
        spec,
        baseline,
        CandidateResult(
            completed=True,
            benchmark=BenchmarkResult(step_time_ms=11, correctness_value=1.2),
        ),
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert "correctness" in rejected.feedback
    assert "step time" in rejected.feedback
