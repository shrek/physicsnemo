"""NOOA agents: ellipsis methods propose; Python methods execute and verify."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from nooa import Agent, CodeActStrategy, PredictStrategy, hidden, strategy
from nooa.config.strategy_config import CodeActConfig, PredictConfig
from nooa.errors import GenerationError
from nooa.events import Feedback
from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel

from simplified.environment import (
    SourceEnvironment,
    TrainingEnvironment,
)
from simplified.knowledge import (
    ContextBuilder,
    FilesystemKnowledgeStore,
    JsonlRunMemory,
    KnowledgeStore,
    RunMemory,
)
from simplified.types import (
    AgentContext,
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    HelloResponse,
    Hotspot,
    HotspotAnalysis,
    InstrumentationPlan,
    KnowledgeItem,
    KnowledgeQuery,
    MemoryItem,
    OptimizationResult,
    PerformanceReport,
    Route,
    RunResult,
    TraceResult,
    TrainingSpec,
)



REQUIRED_RANGES = (
    "train_step",
    "dataload",
    "forward",
    "loss",
    "backward",
    "optimizer",
)


_TEXT_ONLY_STRUCTURED_OUTPUT_MODELS = (
    "moonshotai/kimi-k3-max-preview",
)


def _rejects_json_schema(error: Exception) -> bool:
    message = str(error).lower()
    return "json_schema" in message and (
        "only \"text\" is supported" in message
        or "unsupported 'response_format.type'" in message
    )


class CompatiblePredictStrategy(PredictStrategy):
    """Use typed Predict validation with a text fallback for limited endpoints."""

    def __init__(self, config: PredictConfig | None = None):
        super().__init__(config)
        self._schema_rejected = False

    def _use_text_output(self, runtime) -> bool:
        preference = os.getenv("SIMPLIFIED_STRUCTURED_OUTPUT", "auto").lower()
        if preference == "text":
            return True
        if preference == "schema":
            return False
        model = str(getattr(runtime.agent.llm, "model", "")).lower()
        return self._schema_rejected or any(
            fragment in model for fragment in _TEXT_ONLY_STRUCTURED_OUTPUT_MODELS
        )

    async def _call_llm_raw(self, runtime, response_model):
        if not self._use_text_output(runtime):
            try:
                return await super()._call_llm_raw(runtime, response_model)
            except Exception as error:
                if not _rejects_json_schema(error):
                    raise
                self._schema_rejected = True
        return await runtime.generate(
            tools=None,
            output_model=None,
            **self._build_sampling_kwargs(),
        )


class _PlainTextPredictStrategy(CompatiblePredictStrategy):
    """Accept text-only model output without NOOA's JSON decoding step."""

    def _parse_llm_response(self, llm_response, method_name: str) -> dict[str, str]:
        del method_name
        value = getattr(llm_response, "content", None)
        if not value:
            value = getattr(llm_response, "reasoning", None)
        if isinstance(value, BaseModel):
            value = value.model_dump_json()
        elif not isinstance(value, str):
            value = json.dumps(value, default=str) if value is not None else ""
        if not value.strip():
            raise ValueError("the progress judge returned no text")
        return {"value": value.strip()}


class ProgressVerdict(BaseModel):
    """Generic semantic decision made at an adaptive CodeAct boundary."""

    decision: Literal["extend", "finalize", "stop"]
    reason: str = ""


class SemanticProgressJudge(Agent):
    @strategy(
        _PlainTextPredictStrategy(
            PredictConfig(max_retries=2, max_tokens=4096, max_param_chars=24_000)
        )
    )
    async def assess(
        self,
        objective: str,
        previous_checkpoint: str,
        recent_activity: str,
    ) -> str:
        """Judge whether a generic CodeAct session is making semantic progress.

        Progress means becoming materially closer to returning a valid result for
        objective. Activity alone is not progress. New reads count only when they
        resolve a relevant unknown. Repeated investigation, unchanged drafts,
        repeated errors, and cosmetic state changes are stagnation.

        Use extend when recent activity materially advances the result. Use finalize
        when enough evidence exists and the working agent should stop investigating
        and return its best valid result. Use stop when the agent is cycling or
        cannot make meaningful progress.

        Do not reason at length. Respond in at most three short plain-text lines:
        decision=extend|finalize|stop
        checkpoint=<completed work and remaining obstacle>
        reason=<brief reason for the decision>
        """
        ...


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Task-independent policy for extending bounded CodeAct sessions."""

    extension_iterations: int = 10
    finalization_grace_iterations: int = 3
    hard_max_iterations: int | None = 100
    recent_event_limit: int = 30
    max_event_chars: int = 800

    def __post_init__(self) -> None:
        if self.extension_iterations < 1:
            raise ValueError("extension_iterations must be positive")
        if self.finalization_grace_iterations < 1:
            raise ValueError("finalization_grace_iterations must be positive")
        if self.hard_max_iterations is not None and self.hard_max_iterations < 1:
            raise ValueError("hard_max_iterations must be positive or None")
        if self.recent_event_limit < 1 or self.max_event_chars < 1:
            raise ValueError("event summary limits must be positive")


ProgressEvaluator = Callable[[str, str, str], Awaitable[ProgressVerdict]]


class AdaptiveCodeActStrategy(CodeActStrategy):
    """Extend CodeAct in semantic-progress chunks using only public NOOA APIs.

    Each chunk is a normal ``CodeActStrategy.execute`` call. NOOA writes the
    exhausted chunk's locals back to ``CurrentCall.session_locals`` and retains
    its conversation events, so the next chunk resumes without copying or
    overriding NOOA's generation loop.
    """

    def __init__(
        self,
        config: CodeActConfig,
        *,
        adaptive: AdaptiveBudgetConfig | None = None,
        progress_evaluator: ProgressEvaluator | None = None,
        error_formatter=None,
    ) -> None:
        if config.max_iterations is None or config.max_iterations < 1:
            raise ValueError("adaptive CodeAct requires a positive initial max_iterations")
        super().__init__(config, error_formatter=error_formatter)
        self.adaptive = adaptive or AdaptiveBudgetConfig()
        self.progress_evaluator = progress_evaluator

    @staticmethod
    def _iteration_exhausted(error: GenerationError) -> bool:
        message = str(error)
        return "max_iterations=" in message and "Unable to complete" in message

    def _recent_activity(self, runtime, call) -> str:
        events = runtime.event_manager.filter(
            call_id=call.id,
            limit=self.adaptive.recent_event_limit,
        )
        rendered = []
        for event in events:
            try:
                payload = event.model_dump(exclude_none=True, mode="json")
                event_text = json.dumps(payload, sort_keys=True, default=str)
            except Exception:
                event_text = repr(event)
            rendered.append(
                f"{getattr(event, 'event_type', type(event).__name__)}: "
                f"{event_text[: self.adaptive.max_event_chars]}"
            )
        return "\n".join(rendered) or "No recorded activity."

    @staticmethod
    def _objective(call) -> str:
        return "\n".join(
            part
            for part in (
                f"method: {call.method_name}",
                f"signature: {call.signature}" if call.signature else "",
                f"return type: {call.return_type}" if call.return_type else "",
                call.docstring or "",
            )
            if part
        )

    async def _assess_progress(
        self,
        runtime,
        call,
        checkpoint: str,
    ) -> ProgressVerdict:
        objective = self._objective(call)
        activity = self._recent_activity(runtime, call)
        try:
            if self.progress_evaluator is not None:
                return await self.progress_evaluator(objective, checkpoint, activity)
            response = await SemanticProgressJudge(llm=runtime.agent.llm).assess(
                objective,
                checkpoint,
                activity,
            )
            return self._parse_progress_verdict(response, checkpoint)
        except Exception as error:
            runtime.event_manager.add(
                Feedback(
                    content=(
                        "Semantic progress review was unavailable; continue for "
                        "one bounded extension and return a valid result if ready. "
                        f"Reviewer error: {type(error).__name__}: {error}"
                    )
                )
            )
            return ProgressVerdict(
                decision="extend",
                checkpoint=checkpoint,
                reason=f"semantic progress reviewer failed: {type(error).__name__}",
            )

    @staticmethod
    def _parse_progress_verdict(
        response: str,
        previous_checkpoint: str = "",
    ) -> ProgressVerdict:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            if payload.get("decision") in {"extend", "finalize", "stop"}:
                return ProgressVerdict.model_validate(payload)
            if isinstance(payload.get("value"), str):
                text = payload["value"].strip()

        decision_match = re.search(
            r"(?im)^\s*decision\s*[:=]\s*[\"']?(extend|finalize|stop)\b",
            text,
        )

        def field(name: str) -> str:
            match = re.search(
                rf"(?im)^\s*{name}\s*[:=]\s*(.+?)\s*$",
                text,
            )
            return match.group(1).strip().strip("\"'") if match else ""

        if decision_match:
            return ProgressVerdict(
                decision=decision_match.group(1).lower(),
                checkpoint=field("checkpoint") or previous_checkpoint,
                reason=field("reason"),
            )

        return ProgressVerdict(
            decision="finalize",
            checkpoint=previous_checkpoint,
            reason="semantic progress verdict was not parseable",
        )

    def _chunk_strategy(self, iterations: int) -> CodeActStrategy:
        config = self.config.model_copy(update={"max_iterations": iterations})
        return CodeActStrategy(config, error_formatter=self.error_formatter)

    async def execute(self, runtime, call):
        if call.session_locals is None:
            call = replace(call, session_locals={})

        initial = self.config.max_iterations
        assert initial is not None
        total_iterations = 0
        chunk_iterations = initial
        checkpoint = ""
        finalizing = False

        while True:
            try:
                return await self._chunk_strategy(chunk_iterations).execute(runtime, call)
            except GenerationError as error:
                if not self._iteration_exhausted(error):
                    raise
                total_iterations += chunk_iterations
                if finalizing:
                    raise GenerationError(
                        "Adaptive CodeAct could not return a valid result during "
                        "the finalization grace window."
                    ) from error

                verdict = await self._assess_progress(runtime, call, checkpoint)
                checkpoint = verdict.checkpoint
                if verdict.decision == "stop":
                    raise GenerationError(
                        "Adaptive CodeAct stopped after semantic stagnation: "
                        + (verdict.reason or "no meaningful progress")
                    ) from error

                grace = self.adaptive.finalization_grace_iterations
                hard_limit = self.adaptive.hard_max_iterations
                extension_room = None
                if hard_limit is not None:
                    extension_room = max(hard_limit - total_iterations - grace, 0)

                if verdict.decision == "extend" and (
                    extension_room is None or extension_room > 0
                ):
                    chunk_iterations = self.adaptive.extension_iterations
                    if extension_room is not None:
                        chunk_iterations = min(chunk_iterations, extension_room)
                    runtime.event_manager.add(
                        Feedback(
                            content=(
                                "Semantic progress review: continue from this "
                                f"checkpoint: {checkpoint or verdict.reason}. "
                                "Return the valid result as soon as it is complete."
                            )
                        )
                    )
                    continue

                finalizing = True
                chunk_iterations = grace
                runtime.event_manager.add(
                    Feedback(
                        content=(
                            "Semantic progress review: stop further investigation. "
                            "Return the best valid result now using the evidence and "
                            f"state already collected. Reason: {verdict.reason or checkpoint}."
                        )
                    )
                )


_INSTRUMENTATION_PROPOSER_STRATEGY = AdaptiveCodeActStrategy(
    CodeActConfig(
        max_iterations=50,
        max_retries=1,
        max_consecutive_text_only=2,
        cell_timeout=30,
        max_tokens=32_768,
        max_tool_calls=50,
    ),
    adaptive=AdaptiveBudgetConfig(
        extension_iterations=10,
        finalization_grace_iterations=3,
        hard_max_iterations=100,
    ),
)



class HelloAgent(Agent):
    @strategy(CompatiblePredictStrategy())
    async def hello(self, name: str) -> HelloResponse:
        """Return one short, friendly greeting addressed to name."""
        ...


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


class Runner(Agent):
    environment: Annotated[TrainingEnvironment, hidden]

    def __init__(self, *, environment: TrainingEnvironment, **kwargs):
        super().__init__(**kwargs)
        self.environment = environment

    def preflight(self, spec: TrainingSpec) -> RunResult:
        return self.environment.preflight(spec)

    def smoke(self, spec: TrainingSpec) -> RunResult:
        return self.environment.smoke(spec)

    async def benchmark(self, spec: TrainingSpec) -> BenchmarkResult:
        """Read the typed benchmark JSON emitted by the training command."""
        return self.environment.benchmark(spec)

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult:
        return self.environment.profile(spec, plan)

    def create_performance_report(
        self,
        spec: TrainingSpec,
        baseline: BenchmarkResult,
        plan: InstrumentationPlan,
        trace: TraceResult,
    ) -> PerformanceReport:
        return self.environment.create_performance_report(spec, baseline, plan, trace)

    async def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult:
        return self.environment.benchmark_candidate(spec, proposal)


class InstrumentationProposer(SourceAgent):
    knowledge: Annotated[KnowledgeStore, hidden]
    memory: Annotated[RunMemory, hidden]

    def __init__(
        self,
        *,
        knowledge: KnowledgeStore | None = None,
        memory: RunMemory | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knowledge = knowledge or FilesystemKnowledgeStore()
        self.memory = memory or JsonlRunMemory()

    def search_knowledge(self, query: str, limit: int = 6) -> tuple[KnowledgeItem, ...]:
        """Search verified, published instrumentation guidance."""
        return self.knowledge.search(KnowledgeQuery(text=query, limit=limit))

    def record_discovery(self, observation: str) -> MemoryItem:
        """Append an untrusted observation for later review; never edits the wiki."""
        return self.memory.record("agent_discovery", observation, trust="agent_claim")

    def create_knowledge_draft(self, draft: KnowledgeItem) -> str:
        """Save a draft for review; this can never publish or overwrite guidance."""
        return str(self.knowledge.create_draft(draft))

    @strategy(_INSTRUMENTATION_PROPOSER_STRATEGY)
    async def propose(
        self,
        spec: TrainingSpec,
        required_ranges: tuple[str, ...],
        previous: Critique | None,
        context: AgentContext | None = None,
    ) -> InstrumentationPlan:
        """Return the smallest applicable opt-in instrumentation diff.

        First consult the instrumentation wiki through `search_knowledge` using
        terms grounded in the requested recipe (for example, its profiler,
        framework, loop shape, or distributed mode). Use the returned verified
        guidance to inform the plan, and use the supplied bounded context as a
        starting index rather than a replacement for that lookup. Then inspect
        the requested source as authoritative. Resolve previous critic feedback,
        inspect the requested entry point, and reuse its existing profiler and
        result-output path; do not create a
        competing profiler or output protocol. Keep normal execution and
        training semantics unchanged when profiling is disabled.

        The patch must be a valid unified Git diff that applies to the supplied
        repository. Its declared ranges must exactly match the named ranges it
        emits around real work. Cover {required_ranges} where those phases run;
        never add empty or misleading ranges. Keep data loading separate from
        device transfer and train-step computation, and keep the capture
        bounded by the repository's existing profiling controls.

        Do not apply or run the patch yourself. Return the best complete diff
        promptly; the critic performs static and runtime validation.
        Published knowledge pages may suggest patterns, but must never replace
        inspection of the supplied source. Any discovery can be recorded only as
        an untrusted run-memory observation; do not modify canonical knowledge.
        Examples under `simplified/examples/instrumentation/` remain illustrative.
        """
        ...


class InstrumentationCritic(Agent):
    environment: Annotated[TrainingEnvironment, hidden]

    def __init__(self, *, environment: TrainingEnvironment, **kwargs):
        super().__init__(**kwargs)
        self.environment = environment

    def review(
        self,
        spec: TrainingSpec,
        plan: InstrumentationPlan,
        required_ranges: tuple[str, ...],
    ) -> Critique:
        """Apply, statically check, then execute a bounded profile validation."""
        missing = sorted(set(required_ranges) - set(plan.ranges))
        problems = []
        if missing:
            problems.append("missing ranges: " + ", ".join(missing))
        validation = self.environment.validate_instrumentation(spec, plan)
        if not validation.completed:
            problems.append(validation.error or "instrumentation preflight failed")
        if not problems:
            runtime = self.environment.validate_instrumentation_runtime(spec, plan)
            if not runtime.completed:
                problems.append(
                    runtime.error or "instrumentation runtime validation failed"
                )
        return Critique(accepted=not problems, feedback="; ".join(problems))



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
    @strategy(CompatiblePredictStrategy())
    async def analyze(
        self, trace: TraceResult, previous: Critique | None
    ) -> HotspotAnalysis:
        """Rank the trace's supported bottlenecks using only the allowed hotspot kinds.
        Each hotspot must cite concrete evidence from the trace summary. Resolve the critic's
        feedback and do not recommend changes.
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
        objective: str = "",
        allowed_change_paths: tuple[str, ...] = (),
    ) -> ChangeProposal:
        """Inspect the source and propose one minimal patch for the routed hotspot.

        Objective: {objective}. Modify only: {allowed_change_paths}. Preserve
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
            if result.correctness_metric != baseline.correctness_metric:
                problems.append(
                    "candidate correctness metric does not match baseline: "
                    f"{result.correctness_metric!r} != "
                    f"{baseline.correctness_metric!r}"
                )
            if (
                abs(result.correctness_value - baseline.correctness_value)
                > spec.correctness_tolerance
            ):
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
    runner: Runner
    instrumentation: InstrumentationProposer
    context_builder: ContextBuilder
    instrumentation_critic: InstrumentationCritic
    trace_critic: TraceCritic
    hotspots: HotspotAnalyzer
    router: Router
    changes: ChangeProposer
    candidate_critic: CandidateCritic
    report: ReportBuilder


def create_agents(
    llm: UnifiedLLM,
    source: SourceEnvironment,
    environment: TrainingEnvironment,
    *,
    knowledge: KnowledgeStore | None = None,
    memory: RunMemory | None = None,
) -> Agents:
    knowledge = knowledge or FilesystemKnowledgeStore()
    memory = memory or JsonlRunMemory()
    return Agents(
        runner=Runner(llm=llm, environment=environment),
        instrumentation=InstrumentationProposer(
            llm=llm, source=source, knowledge=knowledge, memory=memory
        ),
        context_builder=ContextBuilder(knowledge, memory),
        instrumentation_critic=InstrumentationCritic(
            llm=llm, environment=environment
        ),
        trace_critic=TraceCritic(llm=llm),
        hotspots=HotspotAnalyzer(llm=llm),
        router=Router(llm=llm),
        changes=ChangeProposer(llm=llm, source=source),
        candidate_critic=CandidateCritic(llm=llm),
        report=ReportBuilder(llm=llm),
    )
