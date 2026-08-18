"""Composable workflows and command entrypoint for training optimization."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass
from inspect import getdoc
from pathlib import Path
from typing import Annotated, Awaitable, Callable, get_type_hints
from uuid import uuid4

from nooa import Agent, hidden
from nooa.tracing import flush_traces, set_session
from nooa.unifiedllm import FakeLLMClient, UnifiedLLM, get_llm_client
from pydantic import BaseModel

from simplified.agents import (
    REQUIRED_RANGES,
    Agents,
    HelloAgent,
    create_agents,
)
from simplified.environment import (
    LocalSourceEnvironment,
    LocalTrainingEnvironment,
    SourceEnvironment,
    TrainingEnvironment,
)
from simplified.types import (
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    HelloResponse,
    Hotspot,
    HotspotAnalysis,
    InstrumentationPlan,
    OptimizationResult,
    PerformanceAnalysis,
    PerformanceReport,
    Route,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class RunContext:
    """Filesystem boundary for one named workflow invocation."""

    id: str
    directory: Path

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None,
        output_root: str | Path,
        step: str,
    ) -> "RunContext":
        identifier = run_id or f"{step}-{uuid4().hex[:12]}"
        if _RUN_ID.fullmatch(identifier) is None:
            raise ValueError(
                "run_id must contain only letters, digits, dots, underscores, or hyphens"
            )
        directory = Path(output_root).resolve() / identifier
        directory.mkdir(parents=True, exist_ok=True)
        return cls(id=identifier, directory=directory)

    def path(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("run output paths must be relative to the run directory")
        return self.directory / candidate

    def output_path(self, output: str | None, step: str) -> Path | None:
        if output == "-":
            return None
        path = self.path(output or f"{step}.json")
        if path == self.directory / "run.json":
            raise ValueError("run.json is reserved for run metadata")
        return path

    def write_manifest(
        self,
        *,
        step: str,
        status: str,
        output: Path | None = None,
        error: str = "",
    ) -> None:
        payload = {
            "id": self.id,
            "step": step,
            "status": status,
            "output": (
                output.relative_to(self.directory).as_posix() if output else None
            ),
            "error": error,
        }
        self.path("run.json").write_text(json.dumps(payload, indent=2) + "\n")


class WorkflowCheckpointStore:
    """Durable, input-bound results for restartable composite workflows."""

    def __init__(self, run: RunContext, workflow: str, inputs: BaseModel, *, resume: bool):
        self.directory = run.path(Path("checkpoints") / workflow)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.workflow = workflow
        self.input_digest = hashlib.sha256(inputs.model_dump_json().encode()).hexdigest()
        self.manifest = self.directory / "manifest.json"
        if self.manifest.exists():
            saved = json.loads(self.manifest.read_text())
            if saved.get("input_digest") != self.input_digest:
                raise WorkflowError(
                    f"run ID already contains checkpoints for different {workflow} inputs; "
                    "choose a new --run-id"
                )
        self._write_manifest()

    def _write_manifest(self) -> None:
        completed = sorted(path.stem for path in self.directory.glob("*.json") if path != self.manifest)
        self._atomic_write(
            self.manifest,
            json.dumps({"version": 1, "workflow": self.workflow, "input_digest": self.input_digest, "completed_steps": completed}, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(content)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, step: str, model: type[BaseModel]) -> BaseModel | None:
        if not self.resume:
            return None
        path = self.directory / f"{step}.json"
        if not path.is_file():
            return None
        try:
            return model.model_validate_json(path.read_text())
        except (OSError, ValueError) as error:
            raise WorkflowError(f"invalid checkpoint {path}: {error}") from error

    def save(self, step: str, value: BaseModel) -> None:
        self._atomic_write(self.directory / f"{step}.json", value.model_dump_json(indent=2) + "\n")
        self._write_manifest()
class WorkflowError(RuntimeError):
    """Raised when a user contract or deterministic verifier fails."""


class RequestValidationWorkflow(Agent):
    """Validate a complete request and prove its commands can run."""

    agents: Annotated[Agents, hidden]

    def __init__(self, agents: Agents, *, llm: UnifiedLLM):
        super().__init__(llm=llm)
        self.agents = agents

    def run(self, request: TrainingRequest) -> TrainingSpec:
        self.validate_request(request)
        for name, result in (
            ("preflight", self.agents.runner.preflight(request.spec)),
            ("smoke", self.agents.runner.smoke(request.spec)),
        ):
            if not result.completed:
                raise WorkflowError(f"request {name} failed: {result.error}")
        return request.spec

    def validate_request(self, request: TrainingRequest) -> None:
        """Validate the request contract before executing it."""
        if request.spec.unresolved:
            raise WorkflowError("request must not contain unresolved inputs")
        for path in request.allowed_change_paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise WorkflowError("allowed_change_paths must be repository-relative")


class InstrumentationWorkflow(Agent):
    """Accept profiling instrumentation and capture a validated trace."""

    agents: Annotated[Agents, hidden]

    def __init__(self, agents: Agents, *, llm: UnifiedLLM, max_attempts: int):
        super().__init__(llm=llm)
        self.agents = agents
        self.max_attempts = max_attempts

    async def accept(
        self,
        spec: TrainingSpec,
        previous: Critique | None = None,
    ) -> InstrumentationPlan:
        feedback = previous
        for _ in range(self.max_attempts):
            plan = await self.agents.instrumentation.propose(
                spec, REQUIRED_RANGES, feedback
            )
            feedback = self.agents.instrumentation_critic.review(
                spec, plan, REQUIRED_RANGES
            )
            if feedback.accepted:
                return plan
        self.failed("instrumentation", feedback)
        raise AssertionError("unreachable")

    async def run(
        self,
        request: TrainingRequest,
        plan: InstrumentationPlan | None = None,
        on_plan_accepted: Callable[[InstrumentationPlan], None] | None = None,
    ) -> tuple[InstrumentationPlan, TraceResult]:
        feedback = None
        if plan is None:
            for _ in range(self.max_attempts):
                candidate = await self.agents.instrumentation.propose(
                    request.spec, REQUIRED_RANGES, feedback
                )
                feedback = self.validate_patch_scope(
                    candidate.patch, request.allowed_change_paths
                )
                if feedback.accepted:
                    feedback = self.agents.instrumentation_critic.review(
                        request.spec, candidate, REQUIRED_RANGES
                    )
                if feedback.accepted:
                    plan = candidate
                    if on_plan_accepted:
                        on_plan_accepted(plan)
                    break
            if plan is None:
                self.failed("instrumentation", feedback)

        for _ in range(self.max_attempts):
            trace = self.agents.runner.profile(request.spec, plan)
            feedback = self.agents.trace_critic.review(trace, REQUIRED_RANGES)
            if feedback.accepted:
                return plan, trace
        self.failed("trace", feedback)
        raise AssertionError("unreachable")

    def validate_patch_scope(
        self, patch: str, allowed_paths: tuple[str, ...]
    ) -> Critique:
        return TrainingOptimizer.validate_patch_scope(patch, allowed_paths)

    def failed(self, stage: str, critique: Critique | None) -> None:
        TrainingOptimizer.failed(stage, critique)


class PerformanceReportWorkflow(Agent):
    """Create the deterministic HTA-backed performance evidence bundle."""

    agents: Annotated[Agents, hidden]

    def __init__(self, agents: Agents, *, llm: UnifiedLLM):
        super().__init__(llm=llm)
        self.agents = agents

    def run(
        self,
        spec: TrainingSpec,
        baseline: BenchmarkResult,
        plan: InstrumentationPlan,
        trace: TraceResult,
    ) -> PerformanceReport:
        report = self.agents.runner.create_performance_report(
            spec, baseline, plan, trace
        )
        if not report.completed:
            raise WorkflowError("performance report failed: " + report.error)
        return report


class HotspotWorkflow(Agent):
    """Analyze an accepted trace and deterministically choose its first route."""

    agents: Annotated[Agents, hidden]

    def __init__(self, agents: Agents, *, llm: UnifiedLLM):
        super().__init__(llm=llm)
        self.agents = agents

    async def analyze(
        self,
        trace: TraceResult,
        previous: Critique | None = None,
    ) -> HotspotAnalysis:
        analysis = await self.agents.hotspots.analyze(trace, previous)
        self.validate_analysis(trace, analysis)
        return analysis

    async def run(self, trace: TraceResult) -> tuple[Hotspot, Route]:
        analysis = await self.analyze(trace)
        hotspot = analysis.hotspots[0]
        return hotspot, self.agents.router.route(hotspot)

    def validate_analysis(
        self, trace: TraceResult, analysis: HotspotAnalysis
    ) -> None:
        TrainingOptimizer.validate_analysis(trace, analysis)


class CandidateWorkflow(Agent):
    """Propose and benchmark one scoped source optimization."""

    agents: Annotated[Agents, hidden]

    def __init__(self, agents: Agents, *, llm: UnifiedLLM, max_attempts: int):
        super().__init__(llm=llm)
        self.agents = agents
        self.max_attempts = max_attempts

    async def run(
        self,
        request: TrainingRequest,
        baseline: BenchmarkResult,
        hotspot: Hotspot,
        route: Route,
    ) -> tuple[ChangeProposal, CandidateResult]:
        feedback = None
        for _ in range(self.max_attempts):
            proposal = await self.agents.changes.propose(
                request.spec,
                hotspot,
                route,
                feedback,
                objective=request.objective,
                allowed_change_paths=request.allowed_change_paths,
            )
            feedback = self.validate_patch_scope(
                proposal.patch, request.allowed_change_paths
            )
            if not feedback.accepted:
                continue
            candidate = await self.agents.runner.benchmark_candidate(
                request.spec, proposal
            )
            feedback = self.agents.candidate_critic.review(
                request.spec, baseline, candidate
            )
            if feedback.accepted:
                return proposal, candidate
        self.failed("candidate", feedback)
        raise AssertionError("unreachable")

    def validate_patch_scope(
        self, patch: str, allowed_paths: tuple[str, ...]
    ) -> Critique:
        return TrainingOptimizer.validate_patch_scope(patch, allowed_paths)

    def failed(self, stage: str, critique: Critique | None) -> None:
        TrainingOptimizer.failed(stage, critique)


class PerformanceAnalysisWorkflow(Agent):
    """Build validated baseline, trace, and report evidence for optimization."""

    agents: Annotated[Agents, hidden]

    def __init__(
        self,
        agents: Agents,
        *,
        llm: UnifiedLLM,
        max_attempts: int,
        checkpoints: WorkflowCheckpointStore | None = None,
    ):
        super().__init__(llm=llm)
        self.agents = agents
        self.max_attempts = max_attempts
        self.checkpoints = checkpoints

    async def run(self, request: TrainingRequest) -> PerformanceAnalysis:
        spec = self.checkpoints.load("validate-request", TrainingSpec) if self.checkpoints else None
        if spec is None:
            spec = RequestValidationWorkflow(self.agents, llm=self._llm).run(request)
            if self.checkpoints:
                self.checkpoints.save("validate-request", spec)

        baseline = self.checkpoints.load("benchmark", BenchmarkResult) if self.checkpoints else None
        if baseline is None:
            baseline = await self.agents.runner.benchmark(spec)
            if self.checkpoints:
                self.checkpoints.save("benchmark", baseline)

        plan = self.checkpoints.load("instrumentation", InstrumentationPlan) if self.checkpoints else None
        trace = self.checkpoints.load("profile", TraceResult) if self.checkpoints else None
        if plan is None or trace is None:
            plan, trace = await InstrumentationWorkflow(
                self.agents, llm=self._llm, max_attempts=self.max_attempts
            ).run(
                request,
                plan=plan,
                on_plan_accepted=(
                    lambda accepted: self.checkpoints.save("instrumentation", accepted)
                    if self.checkpoints
                    else None
                ),
            )
            if self.checkpoints:
                self.checkpoints.save("instrumentation", plan)
                self.checkpoints.save("profile", trace)

        performance_report = (
            self.checkpoints.load("create-performance-report", PerformanceReport)
            if self.checkpoints
            else None
        )
        if performance_report is None:
            performance_report = PerformanceReportWorkflow(
                self.agents, llm=self._llm
            ).run(spec, baseline, plan, trace)
            if self.checkpoints:
                self.checkpoints.save("create-performance-report", performance_report)

        return PerformanceAnalysis(
            request=request,
            baseline=baseline,
            instrumentation=plan,
            trace=trace,
            performance_report=performance_report,
        )


class PerformanceOptimizationWorkflow(Agent):
    """Evaluate one scoped source optimization from validated performance evidence."""

    agents: Annotated[Agents, hidden]

    def __init__(
        self,
        agents: Agents,
        *,
        llm: UnifiedLLM,
        max_attempts: int,
        checkpoints: WorkflowCheckpointStore | None = None,
    ):
        super().__init__(llm=llm)
        self.agents = agents
        self.max_attempts = max_attempts
        self.checkpoints = checkpoints

    async def run(self, analysis: PerformanceAnalysis) -> OptimizationResult:
        if not analysis.performance_report.completed:
            raise WorkflowError("performance analysis report is incomplete")

        hotspot = self.checkpoints.load("analyze-hotspots", Hotspot) if self.checkpoints else None
        route = self.checkpoints.load("route-hotspot", Route) if self.checkpoints else None
        if hotspot is None or route is None:
            hotspot, route = await HotspotWorkflow(self.agents, llm=self._llm).run(
                analysis.trace
            )
            if self.checkpoints:
                self.checkpoints.save("analyze-hotspots", hotspot)
                self.checkpoints.save("route-hotspot", route)

        proposal = self.checkpoints.load("propose-change", ChangeProposal) if self.checkpoints else None
        candidate = self.checkpoints.load("benchmark-candidate", CandidateResult) if self.checkpoints else None
        if proposal is None or candidate is None:
            proposal, candidate = await CandidateWorkflow(
                self.agents, llm=self._llm, max_attempts=self.max_attempts
            ).run(analysis.request, analysis.baseline, hotspot, route)
            if self.checkpoints:
                self.checkpoints.save("propose-change", proposal)
                self.checkpoints.save("benchmark-candidate", candidate)

        if candidate.benchmark is None:
            raise WorkflowError("accepted candidate has no benchmark")
        result = self.agents.report.build(
            analysis.baseline, candidate.benchmark, hotspot, proposal
        )
        return result.model_copy(
            update={"performance_report": analysis.performance_report}
        )


class TrainingOptimizer(Agent):
    """Compose the independently invocable optimization workflows."""

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
        """Run performance analysis, then optimize from its evidence artifact."""
        analysis = await PerformanceAnalysisWorkflow(
            self.agents, llm=self._llm, max_attempts=self.max_attempts
        ).run(request)
        return await PerformanceOptimizationWorkflow(
            self.agents, llm=self._llm, max_attempts=self.max_attempts
        ).run(analysis)

    @staticmethod
    def _validate_request(request: TrainingRequest) -> None:
        """Backward-compatible pure request-contract validator."""
        if request.spec.unresolved:
            raise WorkflowError("request must not contain unresolved inputs")
        for path in request.allowed_change_paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise WorkflowError("allowed_change_paths must be repository-relative")

    @staticmethod
    def validate_patch_scope(patch: str, allowed_paths: tuple[str, ...]) -> Critique:
        paths: list[str] = []
        for line in patch.splitlines():
            if line.startswith("+++ b/"):
                path = line.removeprefix("+++ b/")
                if path != "/dev/null":
                    paths.append(path)
        if not paths:
            return Critique(accepted=False, feedback="patch contains no changed files")
        disallowed = [
            path
            for path in paths
            if not any(
                path == root.rstrip("/") or path.startswith(root.rstrip("/") + "/")
                for root in allowed_paths
            )
        ]
        if disallowed:
            return Critique(
                accepted=False,
                feedback=(
                    "patch changes paths outside allowed_change_paths: "
                    + ", ".join(disallowed)
                ),
            )
        return Critique(accepted=True)

    _validate_patch_scope = staticmethod(validate_patch_scope)

    @staticmethod
    def validate_analysis(trace: TraceResult, analysis: HotspotAnalysis) -> None:
        if not analysis.hotspots:
            raise WorkflowError("hotspot analysis returned no hotspots")
        if not all(hotspot.evidence.strip() for hotspot in analysis.hotspots):
            raise WorkflowError("hotspot analysis contains empty evidence")

    _validate_analysis = staticmethod(validate_analysis)

    @staticmethod
    def failed(stage: str, critique: Critique | None) -> None:
        feedback = critique.feedback if critique else "no critique"
        raise WorkflowError(f"{stage} failed after all attempts: {feedback}")

    _failed = staticmethod(failed)


class WorkflowSteps(Agent):
    """Individually invocable workflow stages over shared dependencies."""

    source: Annotated[SourceEnvironment, hidden]
    environment: Annotated[TrainingEnvironment, hidden]

    def __init__(
        self,
        source: SourceEnvironment,
        environment: TrainingEnvironment,
        llm: UnifiedLLM,
        max_attempts: int = 3,
        run_context: RunContext | None = None,
        resume: bool = True,
    ):
        super().__init__(llm=llm)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.source = source
        self.environment = environment
        self.max_attempts = max_attempts
        self.run_context = run_context
        self.resume = resume

    @property
    def agents(self) -> Agents:
        return create_agents(self._llm, self.source, self.environment)

    def _analysis_checkpoints(
        self, request: TrainingRequest
    ) -> WorkflowCheckpointStore | None:
        if self.run_context is None:
            return None
        return WorkflowCheckpointStore(
            self.run_context, "performance-analysis", request, resume=self.resume
        )

    def _optimization_checkpoints(
        self, analysis: PerformanceAnalysis
    ) -> WorkflowCheckpointStore | None:
        if self.run_context is None:
            return None
        return WorkflowCheckpointStore(
            self.run_context, "performance-optimization", analysis, resume=self.resume
        )

    async def hello(self, name: str) -> HelloResponse:
        """Generate a greeting from the configured LLM."""
        return HelloResponse.model_validate(await HelloAgent(llm=self._llm).hello(name))

    async def run_all(self, request: TrainingRequest) -> OptimizationResult:
        """Execute the complete optimization pipeline."""
        analysis = await self.analyze_performance(request)
        return await self.optimize_performance(analysis)

    async def analyze_performance(
        self, request: TrainingRequest
    ) -> PerformanceAnalysis:
        """Build validated performance evidence through the report boundary."""
        return await PerformanceAnalysisWorkflow(
            self.agents,
            llm=self._llm,
            max_attempts=self.max_attempts,
            checkpoints=self._analysis_checkpoints(request),
        ).run(request)

    async def optimize_performance(
        self, analysis: PerformanceAnalysis
    ) -> OptimizationResult:
        """Evaluate one optimization from a completed performance analysis."""
        return await PerformanceOptimizationWorkflow(
            self.agents,
            llm=self._llm,
            max_attempts=self.max_attempts,
            checkpoints=self._optimization_checkpoints(analysis),
        ).run(analysis)

    def validate_request(self, request: TrainingRequest) -> TrainingSpec:
        """Validate the request and run its preflight and smoke checks."""
        return RequestValidationWorkflow(self.agents, llm=self._llm).run(request)

    def smoke(self, spec: TrainingSpec) -> RunResult:
        """Run the configured bounded training smoke command."""
        return self.agents.runner.smoke(spec)

    async def benchmark(self, spec: TrainingSpec) -> BenchmarkResult:
        """Run the unmodified benchmark command and parse its metrics."""
        return await self.agents.runner.benchmark(spec)

    async def propose_instrumentation(
        self, spec: TrainingSpec, previous: Critique | None
    ) -> InstrumentationPlan:
        """Propose and validate profiling instrumentation."""
        return await InstrumentationWorkflow(
            self.agents, llm=self._llm, max_attempts=self.max_attempts
        ).accept(spec, previous)

    def review_instrumentation(
        self, spec: TrainingSpec, plan: InstrumentationPlan
    ) -> Critique:
        """Deterministically review an instrumentation plan."""
        return self.agents.instrumentation_critic.review(spec, plan, REQUIRED_RANGES)

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult:
        """Apply accepted instrumentation and capture a trace."""
        return self.agents.runner.profile(spec, plan)

    def review_trace(self, trace: TraceResult) -> Critique:
        """Deterministically validate a captured trace."""
        return self.agents.trace_critic.review(trace, REQUIRED_RANGES)

    def create_performance_report(
        self,
        spec: TrainingSpec,
        baseline: BenchmarkResult,
        plan: InstrumentationPlan,
        trace: TraceResult,
    ) -> PerformanceReport:
        """Create the deterministic phase-one HTA evidence bundle."""
        return self.agents.runner.create_performance_report(
            spec, baseline, plan, trace
        )

    async def analyze_hotspots(
        self, trace: TraceResult, previous: Critique | None
    ) -> HotspotAnalysis:
        """Rank supported hotspots from the trace summary."""
        return await self.agents.hotspots.analyze(trace, previous)

    def route_hotspot(self, hotspot: Hotspot) -> Route:
        """Map one hotspot to its implementation route."""
        return self.agents.router.route(hotspot)

    async def propose_change(
        self,
        spec: TrainingSpec,
        hotspot: Hotspot,
        route: Route,
        previous: Critique | None,
    ) -> ChangeProposal:
        """Propose one source optimization for the routed hotspot."""
        return await self.agents.changes.propose(spec, hotspot, route, previous)

    async def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult:
        """Benchmark a proposed source optimization in an isolated worktree."""
        return await self.agents.runner.benchmark_candidate(spec, proposal)

    def review_candidate(
        self,
        spec: TrainingSpec,
        baseline: BenchmarkResult,
        candidate: CandidateResult,
    ) -> Critique:
        """Deterministically accept or reject a benchmarked candidate."""
        return self.agents.candidate_critic.review(spec, baseline, candidate)

    def build_report(
        self,
        baseline: BenchmarkResult,
        candidate: CandidateResult,
        hotspot: Hotspot,
        proposal: ChangeProposal,
    ) -> OptimizationResult:
        """Build the final optimization result from accepted artifacts."""
        if candidate.benchmark is None:
            raise ValueError("build-report requires a candidate containing a benchmark")
        return self.agents.report.build(baseline, candidate.benchmark, hotspot, proposal)


InputContract = tuple[tuple[str, type[BaseModel]], ...]

INPUTS: dict[str, InputContract] = {
    "hello": (),
    "run-all": (("request", TrainingRequest),),
    "analyze-performance": (("request", TrainingRequest),),
    "optimize-performance": (("analysis", PerformanceAnalysis),),
    "validate-request": (("request", TrainingRequest),),
    "smoke": (("spec", TrainingSpec),),
    "benchmark": (("spec", TrainingSpec),),
    "propose-instrumentation": (("spec", TrainingSpec),),
    "review-instrumentation": (("spec", TrainingSpec), ("plan", InstrumentationPlan)),
    "profile": (("spec", TrainingSpec), ("plan", InstrumentationPlan)),
    "review-trace": (("trace", TraceResult),),
    "create-performance-report": (
        ("spec", TrainingSpec),
        ("baseline", BenchmarkResult),
        ("plan", InstrumentationPlan),
        ("trace", TraceResult),
    ),
    "analyze-hotspots": (("trace", TraceResult),),
    "route-hotspot": (("hotspot", Hotspot),),
    "propose-change": (
        ("spec", TrainingSpec),
        ("hotspot", Hotspot),
        ("route", Route),
    ),
    "benchmark-candidate": (("spec", TrainingSpec), ("proposal", ChangeProposal)),
    "review-candidate": (
        ("spec", TrainingSpec),
        ("baseline", BenchmarkResult),
        ("candidate", CandidateResult),
    ),
    "build-report": (
        ("baseline", BenchmarkResult),
        ("candidate", CandidateResult),
        ("hotspot", Hotspot),
        ("proposal", ChangeProposal),
    ),
}

LLM_STEPS = {
    "hello",
    "run-all",
    "analyze-performance",
    "optimize-performance",
    "propose-instrumentation",
    "analyze-hotspots",
    "propose-change",
}


def workflow_catalog() -> str:
    """Render command help from workflow method documentation and type contracts."""
    lines = ["Workflow catalog:"]
    for name, contract in INPUTS.items():
        method = getattr(WorkflowSteps, name.replace("-", "_"))
        description = (getdoc(method) or "No description.").splitlines()[0]
        output = get_type_hints(method).get("return", BaseModel)
        output_name = getattr(output, "__name__", str(output))
        inputs = (
            ", ".join(f"{field}: {model.__name__}" for field, model in contract)
            or "none"
        )
        lines.extend(
            (
                f"  {name}",
                f"    {description}",
                f"    Inputs: {inputs}",
                f"    Output: {output_name}",
            )
        )
    return "\n".join(lines)


def load_artifact(path: str, model: type[BaseModel]) -> BaseModel:
    text = Path(path).read_text()
    if model is TrainingRequest:
        if text.lstrip().startswith("{"):
            return TrainingRequest.model_validate_json(text)
        try:
            return TrainingRequest.model_validate(tomllib.loads(text))
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"request file must contain valid TOML: {error}") from error
    return model.model_validate_json(text)


def write_artifact(value: BaseModel, output: Path | None) -> None:
    rendered = value.model_dump_json(indent=2) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


def create_llm(
    model: str | None = None,
    *,
    client_type: str | None = None,
) -> UnifiedLLM:
    model = model or os.getenv("SIMPLIFIED_MODEL")
    if not model:
        raise ValueError("an LLM step requires --model or SIMPLIFIED_MODEL")
    api_key = (
        os.getenv("SIMPLIFIED_API_TOKEN")
        or os.getenv("SIMPLIFIED_API_KEY")
        or os.getenv("INFERENCE_HUB_KEY")
    )
    client_type = client_type or os.getenv("SIMPLIFIED_CLIENT_TYPE")
    if client_type not in {None, "completion", "responses"}:
        raise ValueError("SIMPLIFIED_CLIENT_TYPE must be 'completion' or 'responses'")
    options = {}
    if client_type:
        options["client_type"] = client_type
    if api_key:
        options["api_key"] = api_key
    if api_base := os.getenv("SIMPLIFIED_API_BASE"):
        options["api_base"] = api_base
    return get_llm_client(model, **options)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run one named optimizer workflow. Training requests are complete TOML "
            "contracts; all other inputs and outputs are typed JSON artifacts."
        )
    )
    argument_parser.add_argument("step", choices=INPUTS)
    argument_parser.add_argument(
        "inputs", nargs="*", help="input artifact files in the order shown below"
    )
    argument_parser.add_argument(
        "-o",
        "--output",
        help="output filename within the run directory, or - for stdout",
    )
    argument_parser.add_argument(
        "--run-id",
        help="caller-selected run ID; a step-prefixed ID is generated by default",
    )
    argument_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="ignore successful checkpoints and execute the workflow again",
    )
    argument_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="parent directory for per-run outputs (default: outputs/runs)",
    )
    argument_parser.add_argument("--repo", default=".", help="source/training repository")
    argument_parser.add_argument("--model", default=os.getenv("SIMPLIFIED_MODEL"))
    argument_parser.add_argument(
        "--client-type",
        choices=("completion", "responses"),
        help="NOOA LLM transport; defaults to SIMPLIFIED_CLIENT_TYPE",
    )
    argument_parser.add_argument("--name", default="PhysicsNeMo")
    argument_parser.add_argument("--previous", help="optional Critique JSON for an LLM retry")
    argument_parser.add_argument("--timeout", type=float, default=600)
    argument_parser.add_argument("--max-attempts", type=int, default=3)
    argument_parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="artifact subdirectory within the run directory",
    )
    argument_parser.add_argument(
        "--trace-dir",
        default="traces",
        help="trace subdirectory within the run directory",
    )
    argument_parser.add_argument("--no-trace", dest="trace_dir", action="store_const", const=None)
    argument_parser.add_argument("--show-turns", action="store_true")
    argument_parser.add_argument(
        "--turn-detail", choices=("compact", "full"), default="compact"
    )
    argument_parser.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto"
    )
    argument_parser.epilog = workflow_catalog()
    return argument_parser


async def execute(args: argparse.Namespace, run: RunContext) -> BaseModel:
    contract = INPUTS[args.step]
    if len(args.inputs) != len(contract):
        raise ValueError(
            f"{args.step} expects {len(contract)} inputs: "
            + ", ".join(name for name, _ in contract)
        )
    values = {
        name: load_artifact(path, model)
        for path, (name, model) in zip(args.inputs, contract, strict=True)
    }
    previous = load_artifact(args.previous, Critique) if args.previous else None
    source = LocalSourceEnvironment(args.repo)
    environment = LocalTrainingEnvironment(
        args.repo, timeout=args.timeout, artifact_directory=run.path(args.artifacts)
    )
    llm: UnifiedLLM = (
        create_llm(args.model, client_type=args.client_type)
        if args.step in LLM_STEPS
        else FakeLLMClient()
    )
    steps = WorkflowSteps(
        source,
        environment,
        llm,
        args.max_attempts,
        run_context=run,
        resume=args.resume,
    )
    try:
        handlers: dict[str, Callable[[], Awaitable[BaseModel] | BaseModel]] = {
            "hello": lambda: steps.hello(args.name),
            "run-all": lambda: steps.run_all(values["request"]),
            "analyze-performance": lambda: steps.analyze_performance(
                values["request"]
            ),
            "optimize-performance": lambda: steps.optimize_performance(
                values["analysis"]
            ),
            "validate-request": lambda: steps.validate_request(values["request"]),
            "smoke": lambda: steps.smoke(values["spec"]),
            "benchmark": lambda: steps.benchmark(values["spec"]),
            "propose-instrumentation": lambda: steps.propose_instrumentation(values["spec"], previous),
            "review-instrumentation": lambda: steps.review_instrumentation(values["spec"], values["plan"]),
            "profile": lambda: steps.profile(values["spec"], values["plan"]),
            "review-trace": lambda: steps.review_trace(values["trace"]),
            "create-performance-report": lambda: steps.create_performance_report(values["spec"], values["baseline"], values["plan"], values["trace"]),
            "analyze-hotspots": lambda: steps.analyze_hotspots(values["trace"], previous),
            "route-hotspot": lambda: steps.route_hotspot(values["hotspot"]),
            "propose-change": lambda: steps.propose_change(values["spec"], values["hotspot"], values["route"], previous),
            "benchmark-candidate": lambda: steps.benchmark_candidate(values["spec"], values["proposal"]),
            "review-candidate": lambda: steps.review_candidate(values["spec"], values["baseline"], values["candidate"]),
            "build-report": lambda: steps.build_report(values["baseline"], values["candidate"], values["hotspot"], values["proposal"]),
        }
        result = handlers[args.step]()
        if isinstance(result, Awaitable):
            return await result
        return result
    finally:
        await llm.aclose()


async def _main(args: argparse.Namespace) -> None:
    from simplified.observability import trace_for_cli

    run = RunContext.create(
        run_id=args.run_id,
        output_root=args.output_root,
        step=args.step,
    )
    output = run.output_path(args.output, args.step)
    trace_directory = run.path(args.trace_dir) if args.trace_dir else None
    run.write_manifest(step=args.step, status="running", output=output)
    tracing = bool(trace_directory or args.show_turns)
    if tracing:
        trace_for_cli(
            directory=trace_directory,
            show_turns=args.show_turns,
            color=args.color,
            turn_detail=args.turn_detail,
        )
        set_session(f"run-{run.id}-{args.step}")
    try:
        result = await execute(args, run)
        write_artifact(result, output)
    except Exception as error:
        run.write_manifest(
            step=args.step, status="failed", output=output, error=str(error)
        )
        raise
    else:
        run.write_manifest(step=args.step, status="completed", output=output)
    finally:
        if tracing:
            flush_traces()


def main(argv: list[str] | None = None) -> None:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        asyncio.run(_main(args))
    except (OSError, RuntimeError, ValueError) as error:
        argument_parser.error(str(error))


if __name__ == "__main__":
    main()
