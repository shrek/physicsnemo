"""Deterministic orchestration with bounded LLM proposal steps."""

from __future__ import annotations

from typing import Annotated

from nooa import Agent, hidden
from nooa.unifiedllm import UnifiedLLM

from simplified.agents import REQUIRED_RANGES, Agents
from simplified.types import Critique, OptimizationResult, TrainingRequest, TrainingSpec


class WorkflowError(RuntimeError):
    """Raised when a user contract or deterministic verifier fails."""


class TrainingOptimizer(Agent):
    agents: Annotated[Agents, hidden]

    def __init__(
        self,
        agents: Agents,
        *,
        llm: UnifiedLLM,
        max_attempts: int = 3,
    ):
        super().__init__(llm=llm)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.agents = agents
        self.max_attempts = max_attempts

    async def run(self, request: TrainingRequest) -> OptimizationResult:
        """Validate the detailed user contract, then execute one bounded run."""
        self._validate_request(request)
        spec = request.spec
        preflight = self.agents.runner.preflight(spec)
        if not preflight.completed:
            raise WorkflowError("request preflight failed: " + preflight.error)
        smoke = self.agents.runner.smoke(spec)
        if not smoke.completed:
            raise WorkflowError("request smoke failed: " + smoke.error)

        baseline = await self.agents.runner.benchmark(spec)

        feedback = None
        for _ in range(self.max_attempts):
            plan = await self.agents.instrumentation.propose(spec, REQUIRED_RANGES, feedback)
            feedback = self._validate_patch_scope(plan.patch, request.allowed_change_paths)
            if feedback.accepted:
                feedback = self.agents.instrumentation_critic.review(
                    spec, plan, REQUIRED_RANGES
                )
            if not feedback.accepted:
                continue
            trace = self.agents.runner.profile(spec, plan)
            feedback = self.agents.trace_critic.review(trace, REQUIRED_RANGES)
            if feedback.accepted:
                break
        else:
            self._failed("instrumentation", feedback)

        phase1_report = self.agents.runner.create_performance_report(
            spec, baseline, plan, trace
        )
        if not phase1_report.completed:
            raise WorkflowError(
                "phase-1 performance report failed: " + phase1_report.error
            )

        analysis = await self.agents.hotspots.analyze(trace, None)
        self._validate_analysis(trace, analysis)

        hotspot = analysis.hotspots[0]
        route = self.agents.router.route(hotspot)
        feedback = None
        for _ in range(self.max_attempts):
            proposal = await self.agents.changes.propose(
                spec,
                hotspot,
                route,
                feedback,
                objective=request.objective,
                allowed_change_paths=request.allowed_change_paths,
            )
            feedback = self._validate_patch_scope(proposal.patch, request.allowed_change_paths)
            if not feedback.accepted:
                continue
            candidate = await self.agents.runner.benchmark_candidate(spec, proposal)
            feedback = self.agents.candidate_critic.review(spec, baseline, candidate)
            if feedback.accepted:
                break
        else:
            self._failed("candidate", feedback)

        if candidate.benchmark is None:
            raise WorkflowError("accepted candidate has no benchmark")
        result = self.agents.report.build(baseline, candidate.benchmark, hotspot, proposal)
        return result.model_copy(update={"phase1_report": phase1_report})

    @staticmethod
    def _validate_request(request: TrainingRequest) -> None:
        """Reject incomplete v0 request files before any LLM or command runs."""
        if request.spec.unresolved:
            raise WorkflowError("request must not contain unresolved inputs")
        for path in request.allowed_change_paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise WorkflowError("allowed_change_paths must be repository-relative")

    @staticmethod
    def _validate_patch_scope(patch: str, allowed_paths: tuple[str, ...]) -> Critique:
        """Reject patches that touch files outside the user's explicit scope."""
        paths: list[str] = []
        for line in patch.splitlines():
            if not line.startswith("+++ b/"):
                continue
            path = line.removeprefix("+++ b/")
            if path != "/dev/null":
                paths.append(path)
        if not paths:
            return Critique(accepted=False, feedback="patch contains no changed files")
        disallowed = [path for path in paths if not any(path == root.rstrip("/") or path.startswith(root.rstrip("/") + "/") for root in allowed_paths)]
        if disallowed:
            return Critique(accepted=False, feedback="patch changes paths outside allowed_change_paths: " + ", ".join(disallowed))
        return Critique(accepted=True)

    @staticmethod
    def _validate_analysis(trace, analysis) -> None:
        """Apply the small deterministic hotspot-output contract."""
        if not analysis.hotspots:
            raise WorkflowError("hotspot analysis returned no hotspots")
        if not all(hotspot.evidence.strip() for hotspot in analysis.hotspots):
            raise WorkflowError("hotspot analysis contains empty evidence")

    @staticmethod
    def _failed(stage: str, critique: Critique | None) -> None:
        feedback = critique.feedback if critique else "no critique"
        raise WorkflowError(f"{stage} failed after all attempts: {feedback}")
