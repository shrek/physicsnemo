from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from nooa.unifiedllm import FakeLLMClient

from simplified.agents import (
    Agents,
    ClosedHumanGate,
    HumanGateError,
    InputAcceptanceAgent,
    InputAcceptanceError,
    InputContractCritic,
    TerminalHumanGate,
)
from simplified.cli import _main, _parser
from simplified.types import Critique, RunResult, TrainingRequest, TrainingSpec
from simplified.workflow import TrainingOptimizer, WorkflowError


REQUEST = TrainingRequest(description="Optimize GeoTransolver volume training.")
SPEC = TrainingSpec(
    smoke_command=("python", "train.py", "--steps", "1"),
    benchmark_command=("python", "train.py", "--steps", "20"),
    profile_command=("python", "train.py", "--profile"),
    correctness_tolerance=0.01,
)
UNRESOLVED_SPEC = SPEC.model_copy(
    update={"unresolved": ("The DrivAerML dataset path is unknown.",)}
)


class Proposer:
    def __init__(self, values):
        self.values = iter(values)
        self.feedback = []

    async def propose(self, request, previous):
        self.feedback.append(previous)
        return next(self.values)


class SlowProposer:
    async def propose(self, request, previous):
        await asyncio.sleep(1)
        return SPEC


class AcceptingSemanticCritic:
    def __init__(self):
        self.specs = []

    async def review(self, spec):
        self.specs.append(spec)
        return Critique(accepted=True)


class SemanticCritic:
    def __init__(self, reviews):
        self.reviews = iter(reviews)
        self.specs = []

    async def review(self, spec):
        self.specs.append(spec)
        return next(self.reviews)


class ContractReviewGate:
    enabled = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.specs = []

    def review_contract(self, spec):
        self.specs.append(spec)
        return next(self.responses)

    def request_input(self, feedback):
        raise AssertionError("clarification should not be requested")


class RecordingGate:
    enabled = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.feedback = []

    def request_input(self, feedback):
        self.feedback.append(feedback)
        return next(self.responses)


class FailIfCalledRunner:
    called = False

    def smoke(self, spec):
        self.called = True
        raise AssertionError("smoke must not run before input acceptance")


def _acceptance(
    proposer,
    gate,
    *,
    semantic_critic=None,
    validator=None,
    max_attempts=3,
    agent_timeout=300,
):
    llm = FakeLLMClient()
    return InputAcceptanceAgent(
        llm=llm,
        proposer=proposer,
        contract_critic=InputContractCritic(llm=llm),
        critic=semantic_critic or AcceptingSemanticCritic(),
        human_gate=gate,
        validator=validator,
        max_attempts=max_attempts,
        agent_timeout=agent_timeout,
    )


def test_input_contract_critic_distinguishes_retry_from_human_input():
    critic = InputContractCritic(llm=FakeLLMClient())
    missing_command = SPEC.model_copy(update={"smoke_command": ()})

    retryable = critic.review(missing_command)
    human_required = critic.review(UNRESOLVED_SPEC)

    assert not retryable.accepted
    assert not retryable.requires_human
    assert not human_required.accepted
    assert human_required.requires_human


def test_input_contract_critic_rejects_shell_wrapper_without_human():
    critic = InputContractCritic(llm=FakeLLMClient())
    shell_spec = SPEC.model_copy(
        update={
            "smoke_command": (
                "/bin/bash",
                "-c",
                "sed -i s/old/new/ config.yaml && python train.py",
            )
        }
    )

    review = critic.review(shell_spec)

    assert not review.accepted
    assert not review.requires_human
    assert "must not invoke a shell" in review.feedback


def test_llm_semantic_criticism_is_sent_to_proposer_for_revision():
    proposer = Proposer([SPEC, SPEC])
    semantic_critic = SemanticCritic(
        [
            Critique(accepted=False, feedback="Profile does not warm up first."),
            Critique(accepted=True),
        ]
    )

    result = asyncio.run(
        _acceptance(
            proposer,
            ClosedHumanGate(),
            semantic_critic=semantic_critic,
            max_attempts=2,
        ).accept(REQUEST)
    )

    assert result == SPEC
    assert proposer.feedback[1].feedback == "Profile does not warm up first."
    assert semantic_critic.specs == [SPEC, SPEC]


def test_input_acceptance_enforces_per_stage_agent_timeout():
    with pytest.raises(InputAcceptanceError, match="input proposal exceeded"):
        asyncio.run(
            _acceptance(
                SlowProposer(),
                ClosedHumanGate(),
                max_attempts=1,
                agent_timeout=0.01,
            ).accept(REQUEST)
        )


def test_input_acceptance_retries_agent_correctable_feedback_without_human():
    missing_command = SPEC.model_copy(update={"smoke_command": ()})
    proposer = Proposer([missing_command, SPEC])

    result = asyncio.run(
        _acceptance(proposer, ClosedHumanGate(), max_attempts=2).accept(REQUEST)
    )

    assert result == SPEC
    assert proposer.feedback[0] is None
    assert proposer.feedback[1] is not None
    assert "smoke_command" in proposer.feedback[1].feedback


def test_human_correction_is_reproposed_then_approval_skips_llm_review():
    corrected = SPEC.model_copy(update={"working_directory": "examples/recipe"})
    proposer = Proposer([SPEC, corrected])
    semantic_critic = AcceptingSemanticCritic()
    gate = ContractReviewGate(
        ["Use the examples/recipe working directory.", None]
    )

    result = asyncio.run(
        _acceptance(
            proposer,
            gate,
            semantic_critic=semantic_critic,
            max_attempts=1,
        ).accept(REQUEST)
    )

    assert result == corrected
    assert "Human command-contract correction" in proposer.feedback[1].feedback
    assert "examples/recipe" in proposer.feedback[1].feedback
    assert gate.specs == [SPEC, corrected]
    assert semantic_critic.specs == []


def test_input_acceptance_uses_human_text_for_llm_revision():
    proposer = Proposer([UNRESOLVED_SPEC, SPEC])
    gate = RecordingGate(
        ["/data/drivaer_ml contains manifest.json with train and val splits."]
    )

    result = asyncio.run(
        _acceptance(proposer, gate, max_attempts=1).accept(REQUEST)
    )

    assert result == SPEC
    assert gate.feedback == [
        "Missing or unresolved: The DrivAerML dataset path is unknown."
    ]
    assert "Human clarification: /data/drivaer_ml" in proposer.feedback[1].feedback


def test_input_acceptance_fails_closed_immediately_when_human_is_required():
    proposer = Proposer([UNRESOLVED_SPEC, SPEC])

    with pytest.raises(InputAcceptanceError, match="requires human clarification"):
        asyncio.run(_acceptance(proposer, ClosedHumanGate()).accept(REQUEST))

    assert len(proposer.feedback) == 1


def test_terminal_gate_contract_review_accepts_empty_response():
    output_stream = StringIO()
    gate = TerminalHumanGate(
        input_stream=StringIO("\n"),
        output_stream=output_stream,
        require_tty=False,
        color="never",
    )

    correction = gate.review_contract(SPEC)

    assert correction is None
    output = output_stream.getvalue()
    assert "command contract review" in output
    assert "working directory: ." in output
    assert "smoke" in output and "python train.py --steps 1" in output
    assert "benchmark" in output
    assert "profile" in output
    assert "approved" in output


def test_terminal_gate_contract_review_returns_correction_without_echoing_it():
    correction = "Use examples/recipe and two measured steps."
    output_stream = StringIO()
    gate = TerminalHumanGate(
        input_stream=StringIO(correction + "\n"),
        output_stream=output_stream,
        require_tty=False,
        color="never",
    )

    result = gate.review_contract(SPEC)

    assert result == correction
    output = output_stream.getvalue()
    assert "correction received" in output
    assert correction not in output


def test_terminal_gate_accepts_nonempty_text_and_does_not_echo_it():
    input_stream = StringIO("\n/data/drivaer_ml has manifest.json\n")
    output_stream = StringIO()
    gate = TerminalHumanGate(
        input_stream=input_stream,
        output_stream=output_stream,
        require_tty=False,
        color="never",
    )

    response = gate.request_input("Dataset path is unresolved.")

    assert response == "/data/drivaer_ml has manifest.json"
    output = output_stream.getvalue()
    assert "human checkpoint" in output
    assert "Please provide a non-empty clarification." in output
    assert "human response" in output
    assert response not in output


def test_terminal_gate_refuses_noninteractive_stdin():
    with pytest.raises(HumanGateError, match="interactive terminal"):
        TerminalHumanGate(
            input_stream=StringIO("answer\n"),
            output_stream=StringIO(),
        ).request_input("Missing input")


def test_workflow_stops_before_smoke_when_hitl_is_disabled():
    llm = FakeLLMClient()
    runner = FailIfCalledRunner()
    agents = Agents(
        inputs=Proposer([UNRESOLVED_SPEC]),
        input_contract_critic=InputContractCritic(llm=llm),
        input_critic=AcceptingSemanticCritic(),
        runner=runner,
        instrumentation=None,
        trace_critic=None,
        hotspots=None,
        hotspot_critic=None,
        router=None,
        changes=None,
        candidate_critic=None,
        report=None,
    )

    with pytest.raises(WorkflowError, match="requires human clarification"):
        asyncio.run(
            TrainingOptimizer(agents, llm=llm).run(REQUEST)
        )

    assert not runner.called


def test_cli_exposes_accept_inputs_and_hitl_mode():
    args = _parser().parse_args(
        [
            "accept-inputs",
            "request.json",
            "--human-in-the-loop",
            "--client-type",
            "responses",
            "--output",
            "spec.json",
        ]
    )

    assert args.step == "accept-inputs"
    assert args.human_in_the_loop
    assert args.client_type == "responses"
    assert args.turn_detail == "compact"
    assert args.trace_dir == "traces"
    assert args.agent_timeout == 300

    disabled = _parser().parse_args(
        ["hello", "--no-trace", "--output", "hello.json"]
    )
    assert disabled.trace_dir is None


def test_cli_traces_by_default(monkeypatch, tmp_path):
    calls = {}

    async def execute(_args):
        return RunResult(completed=True)

    def trace_for_cli(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("simplified.cli._execute", execute)
    monkeypatch.setattr("simplified.cli.trace_for_cli", trace_for_cli)
    monkeypatch.setattr("simplified.cli.set_session", lambda _session: None)
    monkeypatch.setattr("simplified.cli.flush_traces", lambda: None)
    args = _parser().parse_args(
        ["hello", "--output", str(tmp_path / "hello.json")]
    )

    asyncio.run(_main(args))

    assert calls["directory"] == "traces"
    assert calls["show_turns"] is False


class ValidationRunner:
    def __init__(self, preflights, smokes):
        self.preflights = iter(preflights)
        self.smokes = iter(smokes)
        self.preflight_calls = 0
        self.smoke_calls = 0

    def preflight(self, spec):
        self.preflight_calls += 1
        return next(self.preflights)

    def smoke(self, spec):
        self.smoke_calls += 1
        return next(self.smokes)


def test_preflight_failure_is_sent_to_llm_for_automatic_revision():
    bad = SPEC.model_copy(update={"working_directory": "."})
    fixed = SPEC.model_copy(update={"working_directory": "examples/recipe"})
    proposer = Proposer([bad, fixed])
    semantic_critic = AcceptingSemanticCritic()
    validator = ValidationRunner(
        [
            RunResult(
                completed=False,
                error=(
                    "smoke_command script does not exist relative to "
                    "working_directory .: train.py"
                ),
            ),
            RunResult(completed=True),
        ],
        [RunResult(completed=True)],
    )

    result = asyncio.run(
        _acceptance(
            proposer,
            ClosedHumanGate(),
            semantic_critic=semantic_critic,
            validator=validator,
            max_attempts=2,
        ).accept(REQUEST)
    )

    assert result == fixed
    assert "Command preflight failed" in proposer.feedback[1].feedback
    assert "train.py" in proposer.feedback[1].feedback
    assert semantic_critic.specs == [fixed]
    assert validator.preflight_calls == 2
    assert validator.smoke_calls == 1


def test_smoke_dataset_failure_uses_human_text_then_revalidates():
    proposer = Proposer([SPEC, SPEC])
    gate = RecordingGate(["The dataset root is /data/drivaer_ml."])
    validator = ValidationRunner(
        [RunResult(completed=True), RunResult(completed=True)],
        [
            RunResult(
                completed=False,
                error="manifest.json: No such file or directory",
            ),
            RunResult(completed=True),
        ],
    )

    result = asyncio.run(
        _acceptance(
            proposer,
            gate,
            validator=validator,
            max_attempts=2,
        ).accept(REQUEST)
    )

    assert result == SPEC
    assert len(gate.feedback) == 1
    assert "Smoke validation failed" in gate.feedback[0]
    assert "Human clarification: The dataset root" in proposer.feedback[1].feedback
    assert validator.smoke_calls == 2
