"""Ordinary Python orchestration for the bounded proposal/critique loops."""

from __future__ import annotations

from typing import Annotated

from nooa import Agent, hidden
from nooa.unifiedllm import UnifiedLLM

from simplified.agents import (
    REQUIRED_RANGES,
    Agents,
    ClosedHumanGate,
    HumanGate,
    InputAcceptanceAgent,
    InputAcceptanceError,
)
from simplified.types import Critique, OptimizationResult, TrainingRequest


class WorkflowError(RuntimeError):
    pass


class TrainingOptimizer(Agent):
    agents: Annotated[Agents, hidden]
    input_acceptance: Annotated[InputAcceptanceAgent, hidden]

    def __init__(
        self,
        agents: Agents,
        *,
        llm: UnifiedLLM,
        max_attempts: int = 3,
        human_gate: HumanGate | None = None,
        agent_timeout: float = 300,
    ):
        super().__init__(llm=llm)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.agents = agents
        self.max_attempts = max_attempts
        self.input_acceptance = InputAcceptanceAgent(
            llm=llm,
            proposer=agents.inputs,
            contract_critic=agents.input_contract_critic,
            critic=agents.input_critic,
            human_gate=human_gate or ClosedHumanGate(),
            validator=agents.runner,
            max_attempts=max_attempts,
            agent_timeout=agent_timeout,
        )

    async def run(self, request: TrainingRequest) -> OptimizationResult:
        try:
            spec = await self.input_acceptance.accept(request)
        except InputAcceptanceError as error:
            raise WorkflowError(str(error)) from error

        baseline = await self.agents.runner.benchmark(spec)

        feedback = None
        for _ in range(self.max_attempts):
            plan = await self.agents.instrumentation.propose(spec, REQUIRED_RANGES, feedback)
            trace = self.agents.runner.profile(spec, plan)
            feedback = self.agents.trace_critic.review(trace, REQUIRED_RANGES)
            if feedback.accepted:
                break
        else:
            self._failed("instrumentation", feedback)

        feedback = None
        for _ in range(self.max_attempts):
            analysis = await self.agents.hotspots.analyze(trace, feedback)
            feedback = await self.agents.hotspot_critic.review(trace, analysis)
            if feedback.accepted:
                break
        else:
            self._failed("hotspot analysis", feedback)
        if not analysis.hotspots:
            raise WorkflowError("hotspot analysis returned no hotspots")

        hotspot = analysis.hotspots[0]
        route = self.agents.router.route(hotspot)
        feedback = None
        for _ in range(self.max_attempts):
            proposal = await self.agents.changes.propose(spec, hotspot, route, feedback)
            candidate = await self.agents.runner.benchmark_candidate(spec, proposal)
            feedback = self.agents.candidate_critic.review(spec, baseline, candidate)
            if feedback.accepted:
                break
        else:
            self._failed("candidate", feedback)

        if candidate.benchmark is None:
            raise WorkflowError("accepted candidate has no benchmark")
        return self.agents.report.build(baseline, candidate.benchmark, hotspot, proposal)

    @staticmethod
    def _failed(stage: str, critique: Critique | None) -> None:
        feedback = critique.feedback if critique else "no critique"
        raise WorkflowError(f"{stage} failed after all attempts: {feedback}")
