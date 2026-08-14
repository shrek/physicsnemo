"""NOOA agents: ellipsis methods propose; Python methods execute and verify."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TextIO

from nooa import Agent, CodeActStrategy, PredictStrategy, hidden, strategy
from nooa.config.strategy_config import CodeActConfig, PredictConfig
from nooa.unifiedllm import UnifiedLLM

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
    "data_loading",
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
        log = self.environment.benchmark_log(spec)
        if not log.completed:
            raise RuntimeError(log.error or "benchmark command failed")
        interpretation = await BenchmarkLogParser(
            llm=self.llm, benchmark_log=log
        ).parse()
        if interpretation.benchmark is None:
            raise RuntimeError(
                interpretation.error or "benchmark log could not be interpreted"
            )
        return interpretation.benchmark

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult:
        return self.environment.profile(spec, plan)

    async def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult:
        log = self.environment.benchmark_candidate_log(spec, proposal)
        if not log.completed:
            return CandidateResult(
                completed=False, error=log.error or "candidate benchmark failed"
            )
        try:
            interpretation = await BenchmarkLogParser(
                llm=self.llm, benchmark_log=log
            ).parse()
        except Exception as error:
            return CandidateResult(completed=False, error=str(error))
        if interpretation.benchmark is None:
            return CandidateResult(
                completed=False,
                error=interpretation.error
                or "candidate benchmark log could not be interpreted",
            )
        return CandidateResult(completed=True, benchmark=interpretation.benchmark)


class InstrumentationProposer(SourceAgent):
    @strategy(CodeActStrategy())
    async def propose(
        self,
        spec: TrainingSpec,
        required_ranges: tuple[str, ...],
        previous: Critique | None,
    ) -> InstrumentationPlan:
        """Inspect the training loop and propose the smallest instrumentation patch covering
        every required range. Resolve the verifier's feedback without changing training behavior.
        """
        ...


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


class HotspotCritic(Agent):
    @strategy(CompatiblePredictStrategy())
    async def review(self, trace: TraceResult, analysis: HotspotAnalysis) -> Critique:
        """Accept only a non-empty, ranked hotspot analysis whose evidence is supported by
        the trace. Give concise, actionable revision feedback when rejecting it.
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
    ) -> ChangeProposal:
        """Inspect the source and propose one minimal patch for the routed hotspot. Preserve
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
    inputs: InputProposer
    input_contract_critic: InputContractCritic
    input_critic: InputCritic
    runner: Runner
    instrumentation: InstrumentationProposer
    trace_critic: TraceCritic
    hotspots: HotspotAnalyzer
    hotspot_critic: HotspotCritic
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
        inputs=InputProposer(llm=llm, source=source),
        input_contract_critic=InputContractCritic(llm=llm),
        input_critic=InputCritic(llm=llm, source=source),
        runner=Runner(llm=llm, environment=environment),
        instrumentation=InstrumentationProposer(llm=llm, source=source),
        trace_critic=TraceCritic(llm=llm),
        hotspots=HotspotAnalyzer(llm=llm),
        hotspot_critic=HotspotCritic(llm=llm),
        router=Router(llm=llm),
        changes=ChangeProposer(llm=llm, source=source),
        candidate_critic=CandidateCritic(llm=llm),
        report=ReportBuilder(llm=llm),
    )
