import asyncio
import json

import pytest

from nooa import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient
from nooa.unifiedllm.unifiedllm import LLMResponse, ToolCall

from simplified.agents import (
    BenchmarkLogParser,
    CandidateCritic,
    ChangeProposer,
    CompatiblePredictStrategy,
    HotspotAnalyzer,
    HotspotCritic,
    InputCritic,
    InputProposer,
    InstrumentationProposer,
    Router,
)
from simplified.types import (
    BenchmarkLog,
    BenchmarkResult,
    CandidateResult,
    Hotspot,
    TrainingSpec,
)


def test_agentic_methods_use_the_smallest_suitable_strategy():
    for method in (
        InputProposer.propose,
        InstrumentationProposer.propose,
        ChangeProposer.propose,
        BenchmarkLogParser.parse,
    ):
        assert isinstance(method._strategy_override, CodeActStrategy)
    assert InputProposer.propose._strategy_override.config.max_iterations == 30
    assert isinstance(
        InputCritic._review._strategy_override, CompatiblePredictStrategy
    )
    assert InputCritic._review._strategy_override.config.max_retries == 1
    assert isinstance(HotspotAnalyzer.analyze._strategy_override, PredictStrategy)
    assert isinstance(HotspotCritic.review._strategy_override, PredictStrategy)


def test_benchmark_log_parser_exposes_only_bounded_numbered_log_tools():
    parser = BenchmarkLogParser(
        llm=FakeLLMClient(),
        benchmark_log=BenchmarkLog(
            completed=True,
            stdout="setup complete\nStep 7: 12.5 ms\nvalidation loss: 0.25",
            stderr="warning only",
        ),
    )

    assert "stdout: 3 lines" in parser.log_info()
    assert parser.search_log("STEP") == ["3: Step 7: 12.5 ms"]
    assert parser.read_log(2, 4) == (
        "2: setup complete\n"
        "3: Step 7: 12.5 ms\n"
        "4: validation loss: 0.25"
    )

    with pytest.raises(ValueError, match="must not be empty"):
        parser.search_log(" ")
    with pytest.raises(ValueError, match="limited to 200"):
        parser.read_log(1, 201)


def _tool_response(name: str, arguments: dict, call_id: str) -> LLMResponse:
    payload = json.dumps(arguments)
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=payload)],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": payload},
                }
            ],
        },
        reasoning=None,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def test_benchmark_log_parser_runs_as_a_scoped_codeact_agent():
    llm = FakeLLMClient(
        [
            _tool_response(
                "execute_python",
                {"code": "print(await self.search_log('step'))"},
                "read-log",
            ),
            _tool_response(
                "return_result",
                {
                    "result": {
                        "benchmark": {
                            "step_time_ms": 12.5,
                            "correctness_value": 0.25,
                            "correctness_metric": "validation loss",
                        },
                        "error": "",
                    }
                },
                "return-benchmark",
            ),
        ]
    )
    parser = BenchmarkLogParser(
        llm=llm,
        benchmark_log=BenchmarkLog(
            completed=True,
            stdout="unselected private line\nStep 7: 12.5 ms\nvalidation loss: 0.25",
        ),
    )

    interpretation = asyncio.run(parser.parse())

    assert interpretation.benchmark == BenchmarkResult(
        step_time_ms=12.5,
        correctness_value=0.25,
        correctness_metric="validation loss",
    )
    assert "unselected private line" not in json.dumps(llm.last_messages)


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


def test_candidate_critic_requires_the_same_correctness_metric():
    spec = TrainingSpec(
        smoke_command=("smoke",),
        benchmark_command=("benchmark",),
        profile_command=("profile",),
        correctness_tolerance=1,
    )
    baseline = BenchmarkResult(
        step_time_ms=10,
        correctness_value=1,
        correctness_metric="validation loss",
    )
    candidate = CandidateResult(
        completed=True,
        benchmark=BenchmarkResult(
            step_time_ms=8,
            correctness_value=1,
            correctness_metric="training loss",
        ),
    )

    critique = CandidateCritic(llm=FakeLLMClient()).review(
        spec, baseline, candidate
    )

    assert not critique.accepted
