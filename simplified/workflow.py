"""Composable workflows and command entrypoint for training optimization."""

from __future__ import annotations

import argparse
import asyncio
import os
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
    Phase1Report,
    Route,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)


class WorkflowError(RuntimeError):
    """Raised when a user contract or deterministic verifier fails."""


class RequestValidationWorkflow:
    """Validate a complete request and prove its commands can run."""

    def __init__(self, environment: TrainingEnvironment):
        self.environment = environment

    def run(self, request: TrainingRequest) -> TrainingSpec:
        self.validate_request(request)
        for name, result in (
            ("preflight", self.environment.preflight(request.spec)),
            ("smoke", self.environment.smoke(request.spec)),
        ):
            if not result.completed:
                raise WorkflowError(f"request {name} failed: {result.error}")
        return request.spec

    @staticmethod
    def validate_request(request: TrainingRequest) -> None:
        """Validate the request and run its preflight and smoke checks."""
        if request.spec.unresolved:
            raise WorkflowError("request must not contain unresolved inputs")
        for path in request.allowed_change_paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise WorkflowError("allowed_change_paths must be repository-relative")


class InstrumentationWorkflow:
    """Accept profiling instrumentation and capture a validated trace."""

    def __init__(self, agents: Agents, *, max_attempts: int):
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
        TrainingOptimizer.failed("instrumentation", feedback)
        raise AssertionError("unreachable")

    async def run(
        self,
        request: TrainingRequest,
    ) -> tuple[InstrumentationPlan, TraceResult]:
        feedback = None
        for _ in range(self.max_attempts):
            plan = await self.agents.instrumentation.propose(
                request.spec, REQUIRED_RANGES, feedback
            )
            feedback = TrainingOptimizer.validate_patch_scope(
                plan.patch, request.allowed_change_paths
            )
            if feedback.accepted:
                feedback = self.agents.instrumentation_critic.review(
                    request.spec, plan, REQUIRED_RANGES
                )
            if not feedback.accepted:
                continue
            trace = self.agents.runner.profile(request.spec, plan)
            feedback = self.agents.trace_critic.review(trace, REQUIRED_RANGES)
            if feedback.accepted:
                return plan, trace
        TrainingOptimizer.failed("instrumentation", feedback)
        raise AssertionError("unreachable")


class PerformanceReportWorkflow:
    """Create the deterministic HTA-backed performance evidence bundle."""

    def __init__(self, environment: TrainingEnvironment):
        self.environment = environment

    def run(
        self,
        spec: TrainingSpec,
        baseline: BenchmarkResult,
        plan: InstrumentationPlan,
        trace: TraceResult,
    ) -> Phase1Report:
        report = self.environment.create_performance_report(
            spec, baseline, plan, trace
        )
        if not report.completed:
            raise WorkflowError("performance report failed: " + report.error)
        return report


class HotspotWorkflow:
    """Analyze an accepted trace and deterministically choose its first route."""

    def __init__(self, agents: Agents):
        self.agents = agents

    async def analyze(
        self,
        trace: TraceResult,
        previous: Critique | None = None,
    ) -> HotspotAnalysis:
        analysis = await self.agents.hotspots.analyze(trace, previous)
        TrainingOptimizer.validate_analysis(trace, analysis)
        return analysis

    async def run(self, trace: TraceResult) -> tuple[Hotspot, Route]:
        analysis = await self.analyze(trace)
        hotspot = analysis.hotspots[0]
        return hotspot, self.agents.router.route(hotspot)


class CandidateWorkflow:
    """Propose and benchmark one scoped source optimization."""

    def __init__(self, agents: Agents, *, max_attempts: int):
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
            feedback = TrainingOptimizer.validate_patch_scope(
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
        TrainingOptimizer.failed("candidate", feedback)
        raise AssertionError("unreachable")


class PerformanceAnalysisWorkflow:
    """Build validated baseline, trace, and report evidence for optimization."""

    def __init__(self, agents: Agents, *, max_attempts: int):
        self.agents = agents
        self.max_attempts = max_attempts

    async def run(self, request: TrainingRequest) -> PerformanceAnalysis:
        spec = RequestValidationWorkflow(self.agents.runner.environment).run(request)
        baseline = await self.agents.runner.benchmark(spec)
        plan, trace = await InstrumentationWorkflow(
            self.agents, max_attempts=self.max_attempts
        ).run(request)
        performance_report = PerformanceReportWorkflow(
            self.agents.runner.environment
        ).run(spec, baseline, plan, trace)
        return PerformanceAnalysis(
            request=request,
            baseline=baseline,
            instrumentation=plan,
            trace=trace,
            performance_report=performance_report,
        )


class PerformanceOptimizationWorkflow:
    """Evaluate one scoped source optimization from validated performance evidence."""

    def __init__(self, agents: Agents, *, max_attempts: int):
        self.agents = agents
        self.max_attempts = max_attempts

    async def run(self, analysis: PerformanceAnalysis) -> OptimizationResult:
        if not analysis.performance_report.completed:
            raise WorkflowError("performance analysis report is incomplete")
        hotspot, route = await HotspotWorkflow(self.agents).run(analysis.trace)
        proposal, candidate = await CandidateWorkflow(
            self.agents, max_attempts=self.max_attempts
        ).run(analysis.request, analysis.baseline, hotspot, route)
        if candidate.benchmark is None:
            raise WorkflowError("accepted candidate has no benchmark")
        result = self.agents.report.build(
            analysis.baseline, candidate.benchmark, hotspot, proposal
        )
        return result.model_copy(
            update={"phase1_report": analysis.performance_report}
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
            self.agents, max_attempts=self.max_attempts
        ).run(request)
        return await PerformanceOptimizationWorkflow(
            self.agents, max_attempts=self.max_attempts
        ).run(analysis)

    _validate_request = staticmethod(RequestValidationWorkflow.validate_request)

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


@dataclass(frozen=True)
class WorkflowSteps:
    """Individually invocable workflow stages over shared dependencies."""

    source: SourceEnvironment
    environment: TrainingEnvironment
    llm: UnifiedLLM
    max_attempts: int = 3

    @property
    def agents(self) -> Agents:
        return create_agents(self.llm, self.source, self.environment)

    async def hello(self, name: str) -> HelloResponse:
        """Generate a greeting from the configured LLM."""
        return HelloResponse.model_validate(await HelloAgent(llm=self.llm).hello(name))

    async def run_all(self, request: TrainingRequest) -> OptimizationResult:
        """Execute the complete optimization pipeline."""
        return await TrainingOptimizer(
            self.agents, llm=self.llm, max_attempts=self.max_attempts
        ).run(request)

    async def analyze_performance(
        self, request: TrainingRequest
    ) -> PerformanceAnalysis:
        """Build validated performance evidence through the report boundary."""
        return await PerformanceAnalysisWorkflow(
            self.agents, max_attempts=self.max_attempts
        ).run(request)

    async def optimize_performance(
        self, analysis: PerformanceAnalysis
    ) -> OptimizationResult:
        """Evaluate one optimization from a completed performance analysis."""
        return await PerformanceOptimizationWorkflow(
            self.agents, max_attempts=self.max_attempts
        ).run(analysis)

    def validate_request(self, request: TrainingRequest) -> TrainingSpec:
        """Validate the request and run its preflight and smoke checks."""
        return RequestValidationWorkflow(self.environment).run(request)

    def smoke(self, spec: TrainingSpec) -> RunResult:
        """Run the configured bounded training smoke command."""
        return self.environment.smoke(spec)

    async def benchmark(self, spec: TrainingSpec) -> BenchmarkResult:
        """Run the unmodified benchmark command and parse its metrics."""
        return await self.agents.runner.benchmark(spec)

    async def propose_instrumentation(
        self, spec: TrainingSpec, previous: Critique | None
    ) -> InstrumentationPlan:
        """Propose and validate profiling instrumentation."""
        return await InstrumentationWorkflow(
            self.agents, max_attempts=self.max_attempts
        ).accept(spec, previous)

    def review_instrumentation(
        self, spec: TrainingSpec, plan: InstrumentationPlan
    ) -> Critique:
        """Deterministically review an instrumentation plan."""
        return self.agents.instrumentation_critic.review(spec, plan, REQUIRED_RANGES)

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult:
        """Apply accepted instrumentation and capture a trace."""
        return self.environment.profile(spec, plan)

    def review_trace(self, trace: TraceResult) -> Critique:
        """Deterministically validate a captured trace."""
        return self.agents.trace_critic.review(trace, REQUIRED_RANGES)

    def create_performance_report(
        self,
        spec: TrainingSpec,
        baseline: BenchmarkResult,
        plan: InstrumentationPlan,
        trace: TraceResult,
    ) -> Phase1Report:
        """Create the deterministic phase-one HTA evidence bundle."""
        return self.environment.create_performance_report(
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


def write_artifact(value: BaseModel, output: str) -> None:
    rendered = value.model_dump_json(indent=2) + "\n"
    if output == "-":
        print(rendered, end="")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


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
        "-o", "--output", required=True, help="output JSON file, or - for stdout"
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
    argument_parser.add_argument("--artifacts", default="runs")
    argument_parser.add_argument("--trace-dir", default="traces")
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


async def execute(args: argparse.Namespace) -> BaseModel:
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
        args.repo, timeout=args.timeout, artifact_directory=args.artifacts
    )
    llm: UnifiedLLM = (
        create_llm(args.model, client_type=args.client_type)
        if args.step in LLM_STEPS
        else FakeLLMClient()
    )
    steps = WorkflowSteps(source, environment, llm, args.max_attempts)
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

    tracing = bool(args.trace_dir or args.show_turns)
    if tracing:
        trace_for_cli(
            directory=args.trace_dir,
            show_turns=args.show_turns,
            color=args.color,
            turn_detail=args.turn_detail,
        )
        set_session(f"step-{args.step}-{uuid4().hex[:12]}")
    try:
        write_artifact(await execute(args), args.output)
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
