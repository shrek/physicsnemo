"""NOOA agents: ellipsis methods propose; Python methods execute and verify."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, Protocol, TextIO

from nooa import Agent, CodeActStrategy, PredictStrategy, hidden, strategy
from nooa.config.strategy_config import CodeActConfig, PredictConfig
from nooa.errors import GenerationError
from nooa.events import Feedback
from nooa.unifiedllm import UnifiedLLM
from pydantic import BaseModel

from simplified.environment import (
    InputValidationEnvironment,
    SourceEnvironment,
    TrainingEnvironment,
)
from simplified.types import (
    BenchmarkInterpretation,
    BenchmarkLog,
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    HelloResponse,
    Hotspot,
    HotspotAnalysis,
    InstrumentationPlan,
    OptimizationResult,
    Phase1Report,
    Route,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHELL_EXECUTABLES = {"bash", "csh", "dash", "fish", "sh", "tcsh", "zsh"}
_SHELL_OPERATORS = {"&&", "||", ";", "|", "<", ">", ">>"}
ColorMode = Literal["auto", "always", "never"]


class HumanGateError(RuntimeError):
    """Raised when required human clarification cannot be obtained."""


class HumanGate(Protocol):
    """Review proposed commands and clarify rejected proposals."""

    enabled: bool

    def review_contract(self, spec: TrainingSpec) -> str | None:
        """Return correction text, or None to approve the command contract."""
        ...

    def request_input(self, feedback: str) -> str:
        """Return non-empty human clarification or raise :class:`HumanGateError`."""
        ...


class ClosedHumanGate:
    """Skip optional reviews and fail on required human-only information."""

    enabled = False

    def review_contract(self, spec: TrainingSpec) -> str | None:
        return None

    def request_input(self, feedback: str) -> str:
        raise HumanGateError(f"input acceptance requires human clarification: {feedback}")


class TerminalHumanGate:
    """Review command contracts and read clarifications from a terminal."""

    enabled = True

    def __init__(
        self,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        require_tty: bool = True,
        color: ColorMode = "auto",
    ) -> None:
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stderr
        self.require_tty = require_tty
        self._use_color = color == "always" or (
            color == "auto"
            and os.getenv("NO_COLOR") is None
            and bool(getattr(self.output_stream, "isatty", lambda: False)())
        )

    def review_contract(self, spec: TrainingSpec) -> str | None:
        if self.require_tty and not bool(
            getattr(self.input_stream, "isatty", lambda: False)()
        ):
            raise HumanGateError(
                "--human-in-the-loop requires an interactive terminal on stdin"
            )

        self._line("human checkpoint", "command contract review", "\033[96m")
        print(
            f"working directory: {spec.working_directory}",
            file=self.output_stream,
            flush=True,
        )
        for label, command in (
            ("smoke", spec.smoke_command),
            ("benchmark", spec.benchmark_command),
            ("profile", spec.profile_command),
        ):
            print(
                f"{label:>17}: {shlex.join(command)}",
                file=self.output_stream,
                flush=True,
            )
        for overlay in spec.config_overlays:
            print(
                f"{'config overlay':>17}: {overlay.path} <- "
                f"{json.dumps(overlay.merge, sort_keys=True)}",
                file=self.output_stream,
                flush=True,
            )
        print(
            "Press Enter to approve, or type corrections for the LLM:",
            file=self.output_stream,
            flush=True,
        )
        print("> ", end="", file=self.output_stream, flush=True)
        try:
            line = self.input_stream.readline()
        except KeyboardInterrupt as error:
            print(file=self.output_stream, flush=True)
            raise HumanGateError("human input aborted") from error
        if line == "":
            raise HumanGateError("human input aborted: end of input")
        correction = line.strip()
        if correction:
            self._line("human response", "correction received", "\033[93m")
            return correction
        self._line("human response", "approved", "\033[92m")
        return None

    def request_input(self, feedback: str) -> str:
        if self.require_tty and not bool(
            getattr(self.input_stream, "isatty", lambda: False)()
        ):
            raise HumanGateError(
                "--human-in-the-loop requires an interactive terminal on stdin"
            )

        self._line("human checkpoint", "input clarification required", "\033[96m")
        print(feedback, file=self.output_stream, flush=True)
        print(
            "Provide the missing information, or press Ctrl+C to abort:",
            file=self.output_stream,
            flush=True,
        )
        while True:
            print("> ", end="", file=self.output_stream, flush=True)
            try:
                line = self.input_stream.readline()
            except KeyboardInterrupt as error:
                print(file=self.output_stream, flush=True)
                raise HumanGateError("human input aborted") from error
            if line == "":
                raise HumanGateError("human input aborted: end of input")
            clarification = line.strip()
            if clarification:
                self._line("human response", "received", "\033[92m")
                return clarification
            print(
                "Please provide a non-empty clarification.",
                file=self.output_stream,
                flush=True,
            )

    def _line(self, label: str, message: str, color: str) -> None:
        if self._use_color:
            label = f"{color}{label}\033[0m"
        print(f"{label:>16}  {message}", file=self.output_stream, flush=True)


REQUIRED_RANGES = (
    "train_step",
    "dataloader_wait",
    "host_to_device",
    "forward",
    "loss",
    "backward",
    "optimizer_step",
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
    checkpoint: str = ""
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

_INPUT_PROPOSER_STRATEGY = CodeActStrategy(
    CodeActConfig(
        max_iterations=30,
        max_retries=1,
        max_consecutive_text_only=2,
        cell_timeout=30,
        max_tokens=4096,
        max_tool_calls=30,
    )
)

_INPUT_CRITIC_STRATEGY = CompatiblePredictStrategy(
    PredictConfig(
        max_retries=1,
        max_tokens=2048,
        max_param_chars=30_000,
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


class InputProposer(SourceAgent):
    @strategy(_INPUT_PROPOSER_STRATEGY)
    async def propose(
        self, request: TrainingRequest, previous: Critique | None
    ) -> TrainingSpec:
        """Inspect the repository and propose a reproducible training specification.

        Resolve the critic's feedback and set working_directory relative to the
        repository. Every command must be one direct argv invocation. Put leading
        environment assignments in separate argv entries. Never use a shell, shell
        operators, sed, repository-writing setup, or embedded repetition loops.

        When a requested external path must update a repository YAML config rather
        than a supported command-line option, add a config_overlays entry. Its path
        must be an existing repository-relative .yaml or .yml file and merge must
        contain only the required mapping values. The runner applies these overlays
        only in disposable execution worktrees; never encode the update in a shell
        command or leave a provided dataset path unresolved.


        The smoke command must use representative real data and perform only one or
        two complete training updates: load data, forward, finite loss, backward, and
        optimizer step. Disable profiling. It may disable compilation only to keep
        this viability check short.

        The benchmark command must be an unprofiled, production-representative
        invocation with fixed model, data, sample or batch size, seed, precision,
        compile mode, GPU count, warmup, and measured-step budget where the repository
        supports them. Prefer fixed steady-state steps over epochs. This command is
        one repetition; never implement repetitions with a shell loop.

        The profile command must preserve the benchmark model, data, sample or batch
        size, seed, precision, compile mode, and GPU count. Enable the existing
        profiler, warm up the dataloader, allocator, and compiled graph, then capture
        a bounded steady-state window. Prefer five active training iterations and use
        an epoch count only as a safety ceiling.

        Choose a correctness tolerance for a numeric invariant shared by baseline and
        candidate. Record only genuinely external missing facts such as a dataset
        location, credential, hardware choice, or correctness requirement in
        unresolved. A dirty working tree is context, not unresolved human input.

        Work economically. Source methods are synchronous; never await them. Begin
        with targeted searches for distinctive model, dataset, or recipe terms from
        the request; do not list the entire repository. Use list_files, search, and
        read_file instead of direct filesystem traversal. Inspect only the entry
        point and directly relevant configuration, and reserve the final iteration
        for returning the specification. You have at most thirty CodeAct iterations,
        including the final result.
        """
        ...


class InputContractCritic(Agent):
    """Deterministically enforce the non-negotiable input command contract."""

    def review(self, spec: TrainingSpec) -> Critique:
        missing: list[str] = []
        violations: list[str] = []
        for name in ("smoke_command", "benchmark_command", "profile_command"):
            command = getattr(spec, name)
            if not command:
                missing.append(name)
                continue
            problem = self._command_problem(command)
            if problem:
                violations.append(f"{name} {problem}")
        seen_overlay_paths: set[str] = set()
        for overlay in spec.config_overlays:
            problem = self._overlay_problem(overlay.path, overlay.merge)
            if overlay.path in seen_overlay_paths:
                problem = "is declared more than once"
            seen_overlay_paths.add(overlay.path)
            if problem:
                violations.append(f"config overlay {overlay.path!r} {problem}")

        contract_problems = [*missing, *violations]
        if contract_problems:
            feedback = "Input contract violations: " + ", ".join(contract_problems)
        elif spec.unresolved:
            feedback = "Missing or unresolved: " + ", ".join(spec.unresolved)
        else:
            feedback = ""
        return Critique(
            accepted=not contract_problems and not spec.unresolved,
            feedback=feedback,
            requires_human=bool(spec.unresolved) and not contract_problems,
        )

    @staticmethod
    def _command_problem(command: tuple[str, ...]) -> str:
        try:
            argv = shlex.split(command[0]) if len(command) == 1 else list(command)
        except ValueError as error:
            return f"is not valid argv: {error}"
        if not argv:
            return "is empty"
        if any(argument in _SHELL_OPERATORS for argument in argv):
            return "contains a shell operator"
        while argv:
            variable, separator, _value = argv[0].partition("=")
            if not separator or _ENVIRONMENT_NAME.fullmatch(variable) is None:
                break
            argv.pop(0)
        if not argv:
            return "contains environment assignments but no executable"
        if Path(argv[0]).name.lower() in _SHELL_EXECUTABLES:
            return "must not invoke a shell"
        return ""

    @staticmethod
    def _overlay_problem(path: str, merge: dict[str, object]) -> str:
        target = Path(path)
        if target.is_absolute() or ".." in target.parts:
            return "must use a repository-relative path without '..'"
        if target.suffix.lower() not in {".yaml", ".yml"}:
            return "must target a .yaml or .yml file"
        if not merge:
            return "must contain at least one value to merge"
        return ""



class InputCritic(SourceAgent):
    """Perform one LLM review over bounded, deterministically collected evidence."""

    async def review(self, spec: TrainingSpec) -> Critique:
        """Review spec without starting an autonomous repository investigation."""
        return await self._review(spec, self._repository_evidence(spec))

    @strategy(_INPUT_CRITIC_STRATEGY)
    async def _review(self, spec: TrainingSpec, repository_evidence: str) -> Critique:
        """Semantically review the spec using the supplied repository evidence.

        Accept only when the commands match real entry points and supported options.
        Smoke must use representative data, disable profiling, and run one or two
        complete training updates including backward and optimizer step. Benchmark
        must be unprofiled and use a fixed steady-state warmup and measured-step
        budget when supported. Profile must match benchmark model, data, size, seed,
        precision, compile mode, and GPU count while capturing a bounded window after
        warmup; prefer five active iterations. Epoch counts are safety ceilings, not
        substitutes for explicit measurement or capture budgets.

        Reject shell wrappers, repository mutations, embedded loops, profiled
        benchmark timings, cold-versus-warm comparisons, synthetic data presented as
        representative, or commands that invent unsupported arguments. Give concise,
        actionable feedback that the proposer can use in its next attempt.

        When direct configuration files need external values, require a
        config_overlays entry that targets the relevant YAML file and merges the
        supplied value. Reject a spec that relies on a dirty primary checkout for
        such configuration; every execution uses a clean disposable worktree.


        Set requires_human only for genuinely external missing facts such as dataset
        location, credentials, unavailable hardware choices, or an unspecified
        correctness requirement. Repository dirtiness is not missing human input.
        Never request more repository exploration, execute commands, or modify files.
        Treat the evidence packet as deliberately bounded: reject concrete conflicts,
        but do not reject merely because unrelated repository content is omitted.
        Return one concise Critique.
        """
        ...

    def _repository_evidence(self, spec: TrainingSpec) -> str:
        source = self._source()
        working_directory = spec.working_directory or "."
        sections = [f"working_directory: {working_directory}"]

        if spec.config_overlays:
            sections.append(
                "config overlays:\n"
                + "\n".join(
                    f"{overlay.path}: {json.dumps(overlay.merge, sort_keys=True)}"
                    for overlay in spec.config_overlays
                )
            )


        pattern = (
            "**/*"
            if working_directory == "."
            else f"{working_directory.rstrip('/')}/**/*"
        )
        try:
            files = source.list_files(pattern, limit=80)
            sections.append("recipe files:\n" + "\n".join(files))
        except (OSError, RuntimeError, ValueError) as error:
            sections.append(f"recipe file listing unavailable: {error}")


        referenced_paths: set[str] = set()
        selectors: set[str] = set()
        referenced_paths.update(
            overlay.path for overlay in spec.config_overlays
        )

        for command in (
            spec.smoke_command,
            spec.benchmark_command,
            spec.profile_command,
        ):
            argv = self._command_argv(command)
            for argument in argv[1:]:
                if argument.endswith((".py", ".yaml", ".yml", ".toml")):
                    path = Path(argument)
                    if not path.is_absolute():
                        path = Path(working_directory) / path
                    referenced_paths.add(path.as_posix())
                key, separator, value = argument.partition("=")
                if separator and key in {"model", "dataset"} and value:
                    selectors.add(value)

        for selector in sorted(selectors):
            try:
                matches = source.list_files(f"*{selector}*", limit=12)
            except (OSError, RuntimeError, ValueError) as error:
                sections.append(f"files matching {selector!r} unavailable: {error}")
            else:
                sections.append(
                    f"files matching {selector!r}:\n" + "\n".join(matches)
                )
                referenced_paths.update(
                    path
                    for path in matches
                    if path.endswith((".py", ".yaml", ".yml", ".toml"))
                )

        excerpt_budget = 10_000
        missing_paths: list[str] = []
        for path in sorted(referenced_paths):
            if excerpt_budget <= 0:
                break
            try:
                excerpt = source.read_file(path, 1, 220)
            except (OSError, RuntimeError, ValueError) as error:
                missing_paths.append(f"{path}: {error}")
                continue
            excerpt = excerpt[: min(4_000, excerpt_budget)]
            excerpt_budget -= len(excerpt)
            sections.append(f"excerpt {path}:\n{excerpt}")
        if missing_paths:
            sections.append(
                "missing referenced paths:\n" + "\n".join(missing_paths)
            )

        evidence = "\n\n".join(sections)
        if len(evidence) > 20_000:
            evidence = evidence[:19_940] + "\n...[evidence truncated]"
        return evidence

    @staticmethod
    def _command_argv(command: tuple[str, ...]) -> list[str]:
        try:
            argv = shlex.split(command[0]) if len(command) == 1 else list(command)
        except ValueError:
            return []
        while argv:
            variable, separator, _value = argv[0].partition("=")
            if not separator or _ENVIRONMENT_NAME.fullmatch(variable) is None:
                break
            argv.pop(0)
        return argv


class InputAcceptanceError(RuntimeError):
    """Raised when no reviewed training specification can be obtained."""


class InputAcceptanceAgent(Agent):
    """Own proposal, review, runtime validation, and human clarification."""

    proposer: Annotated[InputProposer, hidden]
    contract_critic: Annotated[InputContractCritic, hidden]
    critic: Annotated[InputCritic, hidden]
    human_gate: Annotated[HumanGate, hidden]
    validator: Annotated[InputValidationEnvironment | None, hidden]

    def __init__(
        self,
        *,
        proposer: InputProposer,
        contract_critic: InputContractCritic,
        critic: InputCritic,
        human_gate: HumanGate,
        validator: InputValidationEnvironment | None = None,
        max_attempts: int = 3,
        agent_timeout: float = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if agent_timeout <= 0:
            raise ValueError("agent_timeout must be positive")
        self.proposer = proposer
        self.contract_critic = contract_critic
        self.critic = critic
        self.human_gate = human_gate
        self.validator = validator
        self.max_attempts = max_attempts
        self.agent_timeout = agent_timeout

    async def accept(self, request: TrainingRequest) -> TrainingSpec:
        """Return a human- or LLM-reviewed, smoke-tested training specification."""
        feedback: Critique | None = None
        attempts_since_human = 0
        while attempts_since_human < self.max_attempts:
            attempts_since_human += 1
            try:
                spec = await asyncio.wait_for(
                    self.proposer.propose(request, feedback),
                    timeout=self.agent_timeout,
                )
            except TimeoutError as error:
                raise InputAcceptanceError(
                    f"input proposal exceeded the {self.agent_timeout:g}s agent timeout"
                ) from error
            feedback = self.contract_critic.review(spec)
            if feedback.accepted:
                feedback = self._validate_preflight(spec)
            human_approved = False
            if feedback.accepted and getattr(self.human_gate, "enabled", False):
                correction = self._review_human_contract(spec)
                if correction:
                    feedback = Critique(
                        accepted=False,
                        feedback=(
                            "Human command-contract correction: "
                            f"{correction[:8000]}"
                        ),
                    )
                    attempts_since_human = 0
                    continue
                human_approved = True
            if feedback.accepted and not human_approved:
                try:
                    feedback = await asyncio.wait_for(
                        self.critic.review(spec),
                        timeout=self.agent_timeout,
                    )
                except TimeoutError as error:
                    raise InputAcceptanceError(
                        f"input review exceeded the {self.agent_timeout:g}s agent timeout"
                    ) from error
            if feedback.accepted:
                feedback = self._validate_smoke(spec)
                if feedback.accepted:
                    return spec
            if feedback.requires_human:
                feedback = self._with_human_clarification(feedback)
                attempts_since_human = 0

        final_feedback = feedback.feedback if feedback else "no critique"
        raise InputAcceptanceError(
            "input acceptance failed after all attempts: " + final_feedback
        )

    def _review_human_contract(self, spec: TrainingSpec) -> str | None:
        review = getattr(self.human_gate, "review_contract", None)
        if review is None:
            return None
        try:
            return review(spec)
        except HumanGateError as error:
            raise InputAcceptanceError(str(error)) from error

    def _validate_preflight(self, spec: TrainingSpec) -> Critique:
        if self.validator is None:
            return Critique(accepted=True)

        preflight = getattr(self.validator, "preflight", None)
        if preflight is None:
            return Critique(accepted=True)

        result = preflight(spec)
        if result.completed:
            return Critique(accepted=True)
        return Critique(
            accepted=False,
            feedback=(
                "Command preflight failed. "
                f"working_directory={spec.working_directory!r}; "
                f"error={result.error}"
            ),
        )

    def _validate_smoke(self, spec: TrainingSpec) -> Critique:
        if self.validator is None:
            return Critique(accepted=True)

        result = self.validator.smoke(spec)
        if result.completed:
            return Critique(accepted=True)
        return self._runtime_critique(spec, result.error)

    def _validate_runtime(self, spec: TrainingSpec) -> Critique:
        """Compatibility helper for callers that validate both stages directly."""
        feedback = self._validate_preflight(spec)
        return feedback if not feedback.accepted else self._validate_smoke(spec)

    def _with_human_clarification(self, feedback: Critique) -> Critique:
        try:
            clarification = self.human_gate.request_input(feedback.feedback)
        except HumanGateError as error:
            raise InputAcceptanceError(str(error)) from error
        return Critique(
            accepted=False,
            feedback=(
                f"{feedback.feedback}\n"
                f"Human clarification: {clarification[:8000]}"
            ),
        )

    @staticmethod
    def _runtime_critique(spec: TrainingSpec, error: str) -> Critique:
        lowered = error.lower()
        agent_correctable = (
            "can't open file",
            "cannot find primary config",
            "could not override",
            "modulenotfounderror",
            "unrecognized arguments",
        )
        human_required = (
            "dataset_paths",
            "manifest.json",
            "missingmandatoryvalue",
            "no such file or directory",
            "permission denied",
            "no cuda",
            "no gpu",
            "nvidia driver",
            "cuda-capable device",
        )
        requires_human = not any(
            marker in lowered for marker in agent_correctable
        ) and any(marker in lowered for marker in human_required)
        return Critique(
            accepted=False,
            feedback=(
                "Smoke validation failed. "
                f"working_directory={spec.working_directory!r}; "
                f"smoke_command={list(spec.smoke_command)!r}; "
                f"error={error}"
            ),
            requires_human=requires_human,
        )


_BENCHMARK_LOG_STRATEGY = CodeActStrategy(
    CodeActConfig(
        max_iterations=20,
        max_retries=1,
        max_consecutive_text_only=2,
        cell_timeout=30,
        max_tokens=3072,
        max_tool_calls=30,
    )
)


class BenchmarkLogParser(Agent):
    """Interpret one captured benchmark log through bounded, log-only tools."""

    benchmark_log: Annotated[BenchmarkLog, hidden]

    def __init__(self, *, benchmark_log: BenchmarkLog, **kwargs):
        super().__init__(**kwargs)
        if not benchmark_log.completed:
            raise ValueError("cannot parse an unsuccessful benchmark log")
        self.benchmark_log = benchmark_log

    def log_info(self) -> str:
        """Return stream, line, and character counts without returning log content."""
        stdout_lines = len(self.benchmark_log.stdout.splitlines())
        stderr_lines = len(self.benchmark_log.stderr.splitlines())
        return (
            f"stdout: {stdout_lines} lines, {len(self.benchmark_log.stdout)} chars; "
            f"stderr: {stderr_lines} lines, {len(self.benchmark_log.stderr)} chars; "
            f"combined: {len(self._log_lines())} numbered lines"
        )

    def search_log(self, text: str, limit: int = 50) -> list[str]:
        """Case-insensitively find literal text in this benchmark log only."""
        if not text.strip():
            raise ValueError("search text must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        needle = text.casefold()
        matches = []
        for number, line in enumerate(self._log_lines(), start=1):
            if needle in line.casefold():
                matches.append(f"{number}: {line[:1000]}")
                if len(matches) == limit:
                    break
        return matches

    def read_log(self, start_line: int = 1, end_line: int = 200) -> str:
        """Read at most 200 numbered lines from this benchmark log only."""
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid line range")
        if end_line - start_line + 1 > 200:
            raise ValueError("read_log is limited to 200 lines per call")
        lines = self._log_lines()
        excerpt = "\n".join(
            f"{number}: {lines[number - 1][:2000]}"
            for number in range(start_line, min(end_line, len(lines)) + 1)
        )
        if len(excerpt) > 20_000:
            return excerpt[:20_000] + "\n...[excerpt truncated]"
        return excerpt

    @hidden
    def _log_lines(self) -> list[str]:
        lines = ["=== STDOUT ===", *self.benchmark_log.stdout.splitlines()]
        if self.benchmark_log.stderr:
            lines.extend(["=== STDERR ===", *self.benchmark_log.stderr.splitlines()])
        return lines

    @strategy(_BENCHMARK_LOG_STRATEGY)
    async def parse(self) -> BenchmarkInterpretation:
        """Interpret the captured successful training log.

        Begin with log_info. Use search_log to locate likely timing/performance/step/
        latency and loss/accuracy/validation/correctness measurements, then use
        read_log for their surrounding context. These tools expose only bounded
        excerpts from this run; do not ask for repository files.

        Extract an explicitly reported representative steady-state training step
        time and convert it to milliseconds. Prefer a measured post-warmup window;
        ignore setup, compilation, and warmup timings when the log identifies them.
        Do not use total process wall time as step time.

        Extract one explicitly reported scalar correctness invariant suitable for
        comparing an unchanged baseline and candidate. Prefer a final validation
        loss or validation metric over a training metric, and record its exact
        semantic name in correctness_metric. Never invent or infer missing values.

        Return benchmark=None and a concise error if either value is absent or
        ambiguous. Logs may use arbitrary frameworks, formatting, units, and metric
        names; interpret meaning rather than relying on a recipe-specific pattern.
        """
        ...


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
    ) -> Phase1Report:
        return self.environment.create_performance_report(spec, baseline, plan, trace)

    async def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult:
        return self.environment.benchmark_candidate(spec, proposal)


class InstrumentationProposer(SourceAgent):
    @strategy(_INSTRUMENTATION_PROPOSER_STRATEGY)
    async def propose(
        self,
        spec: TrainingSpec,
        required_ranges: tuple[str, ...],
        previous: Critique | None,
    ) -> InstrumentationPlan:
        """Inspect the training loop and return a valid, minimal instrumentation patch.

        For this GeoTransolver task, begin finalizing by turn 18. You may use the
        existing 50-turn CodeAct budget only while each additional turn makes material
        progress toward a complete, applicable unified diff; otherwise return the best
        complete InstrumentationPlan immediately. Finish repository discovery by turn
        12 and reserve turns 13 through 18 for composing and checking the unified diff.
        Do not execute, apply, or validate the patch
        yourself and do not invoke subprocess or shell commands; a separate critic
        applies the patch, performs static checks, and runs a bounded profile validation
        in a disposable Git worktree. Any failure is returned as feedback for your next
        proposal, so resolve exact runtime API errors rather than merely compiling.

        Resolve all previous critic feedback. Inspect existing profiler support first
        and integrate with it instead of creating a competing profiler. The patch must
        be a syntactically valid unified Git diff against repository HEAD with accurate
        hunk counts and context; it must apply using git apply.

        For the GeoTransolver volume recipe, before inspecting implementation details,
        use read_file to read these repository-local instructions:
        - skills/physicsnemo-training-performance-tuner/SKILL.md
        - skills/physicsnemo-training-performance-tuner/references/geotransolver-volume.md
        - skills/physicsnemo-training-performance-tuner/references/phase1-protocol.md
        Then inspect only these implementation files: the recipe's src/train.py,
        src/v0_results.py, conf/train.yaml, conf/model/geotransolver_volume.yaml,
        datasets/drivaer_ml_volume.yaml, and the PhysicsNeMo profiler interface and
        torch wrapper. Do not call list_files after reading the three guidance files.
        Apply the guidance as an instrumentation contract: retain the representative compiled
        workload, capture a bounded post-warmup window, and separate dataloader,
        geometry/feature construction, forward, loss, backward, and optimizer work
        when those phases exist. Do not copy historical findings or make an
        optimization recommendation before trace evidence exists.

        This recipe already has an opt-in +v0_result=true bridge that emits the final
        TraceResult after profiler finalization. Preserve that bridge and do not add a
        second stdout-result mechanism. Ensure its trace contains the required range
        names rather than merely returning declared names.

        Preserve numerics, sample order, optimizer behavior, and normal execution when
        profiling is disabled. Instrumentation must be opt-in. Emit every required
        range name, including one outer train_step range and a dataloader_wait range
        that isolates iterator advancement. Place profiler.step() exactly once per
        logical training step. Also instrument feature construction, distributed
        synchronization, validation, or checkpoint work when those phases actually
        exist; do not add misleading empty ranges.

        Capture a bounded Kineto trace after model/compiler, dataloader, and allocator
        warmup, with five steady-state active iterations by default. Use unique trace
        names per rank for distributed runs. Keep stack and shape collection disabled
        in the primary capture unless the existing profiler requires them.

        After profiler finalization, the final non-empty stdout line must be a JSON
        object accepted as TraceResult with completed=true, the actual trace path
        inside the working tree, the emitted range names, and a non-empty factual
        summary. Print nothing after that JSON object. Do not import the simplified
        package from the training repository merely to construct this JSON.

        Make no optimization or unrelated source change and never claim profiled wall
        time is baseline performance.
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



class InstrumentationAcceptanceError(RuntimeError):
    """Raised when no instrumentation patch passes runtime validation."""


class InstrumentationAcceptanceAgent(Agent):
    """Propose instrumentation patches until a bounded profile validation passes."""

    proposer: Annotated[InstrumentationProposer, hidden]
    critic: Annotated[InstrumentationCritic, hidden]

    def __init__(
        self,
        *,
        proposer: InstrumentationProposer,
        critic: InstrumentationCritic,
        max_attempts: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.proposer = proposer
        self.critic = critic
        self.max_attempts = max_attempts

    async def accept(
        self,
        spec: TrainingSpec,
        required_ranges: tuple[str, ...],
        previous: Critique | None = None,
    ) -> InstrumentationPlan:
        """Return the first patch that passes static and runtime validation."""
        feedback = previous
        for _attempt in range(self.max_attempts):
            plan = await self.proposer.propose(spec, required_ranges, feedback)
            feedback = self.critic.review(spec, plan, required_ranges)
            if feedback.accepted:
                return plan
        raise InstrumentationAcceptanceError(
            "instrumentation proposal failed after all attempts: "
            + (feedback.feedback if feedback else "no critique")
        )
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
) -> Agents:
    return Agents(
        runner=Runner(llm=llm, environment=environment),
        instrumentation=InstrumentationProposer(llm=llm, source=source),
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
