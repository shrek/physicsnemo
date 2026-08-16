"""Run one training-optimizer step using typed JSON artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import yaml
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from nooa.tracing import flush_traces, set_session
from nooa.unifiedllm import FakeLLMClient, UnifiedLLM, get_llm_client
from pydantic import BaseModel

from simplified.agents import (
    REQUIRED_RANGES,
    CandidateCritic,
    ChangeProposer,
    ClosedHumanGate,
    HumanGate,
    TerminalHumanGate,
    HotspotAnalyzer,
    HotspotCritic,
    HelloAgent,
    InputAcceptanceAgent,
    InputContractCritic,
    InputCritic,
    InputProposer,
    InstrumentationAcceptanceAgent,
    InstrumentationCritic,
    InstrumentationProposer,
    ReportBuilder,
    Runner,
    Router,
    TraceCritic,
    create_agents,
)
from simplified.observability import trace_for_cli
from simplified.workflow import TrainingOptimizer
from simplified.types import (
    BenchmarkLog,
    BenchmarkResult,
    CandidateResult,
    ChangeProposal,
    Critique,
    Hotspot,
    HotspotAnalysis,
    HelloResponse,
    InstrumentationPlan,
    Route,
    RunResult,
    TraceResult,
    TrainingRequest,
    TrainingSpec,
)


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHELL_EXECUTABLES = {"bash", "csh", "dash", "fish", "sh", "tcsh", "zsh"}


class LocalSourceEnvironment:
    """Bounded, read-only access to a local source tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)

    def list_files(self, pattern: str = "**/*", limit: int = 200) -> list[str]:
        if shutil.which("rg") is not None:
            result = subprocess.run(
                ["rg", "--files", "-g", pattern],
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr.strip() or "source listing failed")
            return result.stdout.splitlines()[:limit]

        patterns = {pattern}
        candidate = pattern
        while "**/" in candidate:
            candidate = candidate.replace("**/", "", 1)
            patterns.add(candidate)
        matches = (
            path
            for path in self._iter_source_files()
            if any(fnmatch(path, candidate) for candidate in patterns)
        )
        result = []
        for path in matches:
            result.append(path)
            if len(result) >= limit:
                break
        return result

    def search(self, text: str, limit: int = 100) -> list[str]:
        if shutil.which("rg") is not None:
            result = subprocess.run(
                ["rg", "-n", "--fixed-strings", "--", text, "."],
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr.strip() or "source search failed")
            return result.stdout.splitlines()[:limit]

        matches: list[str] = []
        for relative_path in self._iter_source_files():
            path = self.root / relative_path
            try:
                with path.open("rb") as stream:
                    payload = stream.read(2_000_001)
            except OSError:
                continue
            if len(payload) > 2_000_000 or b"\0" in payload:
                continue
            for line_number, line in enumerate(
                payload.decode(errors="replace").splitlines(), start=1
            ):
                if text in line:
                    matches.append(f"{relative_path}:{line_number}:{line[:500]}")
                    if len(matches) >= limit:
                        return matches
        return matches

    def _iter_source_files(self) -> Iterator[str]:
        ignored = {".git", ".venv", "__pycache__", "runs", "traces"}
        for directory, directories, files in os.walk(self.root):
            directories[:] = sorted(
                name for name in directories if name not in ignored
            )
            base = Path(directory)
            for name in sorted(files):
                yield (base / name).relative_to(self.root).as_posix()

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid line range")
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            raise ValueError(f"path is outside the source tree: {path}")
        lines = target.read_text(errors="replace").splitlines()
        return "\n".join(lines[start_line - 1 : end_line])


class LocalTrainingEnvironment:
    """Execute real commands, isolating proposed patches in Git worktrees.

    Portable training specs use ``python``. When the repository has its usual
    local virtual environment, this runner resolves that portable executable to
    the repository interpreter and exposes the repository on ``PYTHONPATH``.
    The same preparation is used for validation and every execution path.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        timeout: float = 600,
        artifact_directory: str | Path = "runs",
    ):
        self.root = Path(root).resolve(strict=True)
        self.timeout = timeout
        self.artifact_directory = Path(artifact_directory).resolve()

    @staticmethod
    def _merge_yaml_mappings(
        existing: dict[str, object],
        update: dict[str, object],
    ) -> dict[str, object]:
        result = dict(existing)
        for key, value in update.items():
            previous = result.get(key)
            if isinstance(previous, dict) and isinstance(value, dict):
                result[key] = LocalTrainingEnvironment._merge_yaml_mappings(
                    previous, value
                )
            else:
                result[key] = value
        return result

    def _apply_config_overlays(self, spec: TrainingSpec, worktree: Path) -> None:
        updated_paths: set[Path] = set()
        for overlay in spec.config_overlays:
            relative_path = Path(overlay.path)
            if relative_path.is_absolute():
                raise ValueError(
                    f"config overlay path must be relative to the repository: {overlay.path}"
                )
            if relative_path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(
                    f"config overlay must target a YAML file: {overlay.path}"
                )
            target = (worktree / relative_path).resolve()
            if not target.is_relative_to(worktree):
                raise ValueError(
                    f"config overlay path escapes the repository: {overlay.path}"
                )
            if target in updated_paths:
                raise ValueError(
                    f"config overlay targets {overlay.path!r} more than once"
                )
            if not target.is_file():
                raise ValueError(f"config overlay file does not exist: {overlay.path}")
            try:
                loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                raise ValueError(
                    f"could not parse config overlay file {overlay.path}: {error}"
                ) from error
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"config overlay file must contain a YAML mapping: {overlay.path}"
                )
            merged = self._merge_yaml_mappings(loaded, overlay.merge)
            target.write_text(
                yaml.safe_dump(merged, sort_keys=False),
                encoding="utf-8",
            )
            updated_paths.add(target)

    @contextmanager
    def _execution_worktree(
        self, spec: TrainingSpec, patch: str = ""
    ) -> Iterator[Path]:
        with self._patched_worktree(patch) as worktree:
            self._apply_config_overlays(spec, worktree)
            yield worktree


    def validate_instrumentation(
        self, spec: TrainingSpec, plan: InstrumentationPlan
    ) -> RunResult:
        """Apply a plan in a disposable worktree and run cheap static checks."""
        try:
            with self._execution_worktree(spec, plan.patch) as worktree:
                command_cwd = self._working_directory(spec, worktree)
                self._validate_command(
                    "profile_command",
                    spec.profile_command,
                    command_cwd,
                    repository_root=worktree,
                )
                diff_check = subprocess.run(
                    ["git", "-C", str(worktree), "diff", "--check"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if diff_check.returncode != 0:
                    raise RuntimeError(
                        diff_check.stderr.strip()
                        or diff_check.stdout.strip()
                        or "instrumentation patch failed git diff --check"
                    )
                changed = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(worktree),
                        "diff",
                        "--name-only",
                        "--diff-filter=ACMR",
                        "HEAD",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if changed.returncode != 0:
                    raise RuntimeError(
                        changed.stderr.strip() or "could not inspect patched files"
                    )
                for relative in changed.stdout.splitlines():
                    target = (worktree / relative).resolve()
                    if target.suffix != ".py" or not target.is_relative_to(worktree):
                        continue
                    compile(
                        target.read_text(errors="replace"),
                        relative,
                        "exec",
                    )
        except (OSError, RuntimeError, SyntaxError, ValueError) as error:
            return RunResult(
                completed=False,
                error=f"instrumentation preflight failed: {error}",
            )
        return RunResult(completed=True)

    def validate_instrumentation_runtime(
        self, spec: TrainingSpec, plan: InstrumentationPlan
    ) -> RunResult:
        """Run a bounded profiled execution of an applied instrumentation plan.

        This disposable validation capture proves that the patch can initialize
        the recipe's profiler API and emit a usable trace. It does not preserve
        the trace or replace the full profile step.
        """
        try:
            with self._execution_worktree(spec, plan.patch) as worktree:
                command_cwd = self._working_directory(spec, worktree)
                self._validate_command(
                    "profile_command",
                    spec.profile_command,
                    command_cwd,
                    repository_root=worktree,
                )
                result = self._run(
                    spec.profile_command,
                    command_cwd,
                    timeout=min(self.timeout, 180),
                )
                if result.returncode != 0:
                    return RunResult(
                        completed=False,
                        error=(
                            "instrumentation runtime validation failed: "
                            + self._error(result)
                        ),
                    )
                trace = TraceResult.model_validate(self._last_json(result.stdout))
                if not trace.completed:
                    return RunResult(
                        completed=False,
                        error=(
                            "instrumentation runtime validation reported failure: "
                            + (trace.error or "unknown error")
                        ),
                    )
                if not trace.summary:
                    return RunResult(
                        completed=False,
                        error="instrumentation runtime validation emitted no trace summary",
                    )
                missing_ranges = sorted(set(plan.ranges) - set(trace.ranges))
                if missing_ranges:
                    return RunResult(
                        completed=False,
                        error=(
                            "instrumentation runtime validation did not report ranges: "
                            + ", ".join(missing_ranges)
                        ),
                    )
                if not trace.path:
                    return RunResult(
                        completed=False,
                        error="instrumentation runtime validation emitted no trace path",
                    )
                trace_path = Path(trace.path)
                trace_path = (
                    trace_path
                    if trace_path.is_absolute()
                    else command_cwd / trace_path
                ).resolve()
                if not trace_path.is_file() or not trace_path.is_relative_to(worktree):
                    return RunResult(
                        completed=False,
                        error=(
                            "instrumentation runtime validation emitted a missing "
                            "or out-of-worktree trace path: "
                            f"{trace.path}"
                        ),
                    )
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            return RunResult(
                completed=False,
                error=f"instrumentation runtime validation failed: {error}",
            )
        return RunResult(completed=True)

    def preflight(self, spec: TrainingSpec) -> RunResult:
        """Validate commands and overlays in the same clean checkout used for runs."""
        try:
            with self._execution_worktree(spec) as worktree:
                cwd = self._working_directory(spec, worktree)
                for name, command in (
                    ("smoke_command", spec.smoke_command),
                    ("benchmark_command", spec.benchmark_command),
                    ("profile_command", spec.profile_command),
                ):
                    self._validate_command(
                        name,
                        command,
                        cwd,
                        repository_root=worktree,
                    )
        except (OSError, RuntimeError, ValueError) as error:
            return RunResult(completed=False, error=str(error))
        return RunResult(completed=True)

    def smoke(self, spec: TrainingSpec) -> RunResult:
        try:
            with self._execution_worktree(spec) as worktree:
                cwd = self._working_directory(spec, worktree)
                self._validate_command(
                    "smoke_command",
                    spec.smoke_command,
                    cwd,
                    repository_root=worktree,
                )
                result = self._run(spec.smoke_command, cwd)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            return RunResult(completed=False, error=str(error))
        return RunResult(
            completed=result.returncode == 0,
            error="" if result.returncode == 0 else self._error(result),
        )

    def benchmark_log(self, spec: TrainingSpec) -> BenchmarkLog:
        try:
            with self._execution_worktree(spec) as worktree:
                cwd = self._working_directory(spec, worktree)
                self._validate_command(
                    "benchmark_command",
                    spec.benchmark_command,
                    cwd,
                    repository_root=worktree,
                )
                result = self._run(spec.benchmark_command, cwd)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            return BenchmarkLog(completed=False, error=str(error))
        if result.returncode != 0:
            return BenchmarkLog(
                completed=False,
                stdout=result.stdout,
                stderr=result.stderr,
                error=self._error(result),
            )
        return BenchmarkLog(
            completed=True, stdout=result.stdout, stderr=result.stderr
        )

    def benchmark(self, spec: TrainingSpec) -> BenchmarkResult:
        log = self.benchmark_log(spec)
        if not log.completed:
            raise RuntimeError(log.error)
        return BenchmarkResult.model_validate(self._last_json(log.stdout))

    def profile(self, spec: TrainingSpec, plan: InstrumentationPlan) -> TraceResult:
        try:
            with self._execution_worktree(spec, plan.patch) as worktree:
                command_cwd = self._working_directory(spec, worktree)
                result = self._run(spec.profile_command, command_cwd)
                if result.returncode != 0:
                    return TraceResult(completed=False, error=self._error(result))
                trace = TraceResult.model_validate(self._last_json(result.stdout))
                return self._preserve_trace(trace, worktree, command_cwd)
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            return TraceResult(completed=False, error=str(error))

    def benchmark_candidate_log(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> BenchmarkLog:
        try:
            with self._execution_worktree(spec, proposal.patch) as worktree:
                result = self._run(
                    spec.benchmark_command,
                    self._working_directory(spec, worktree),
                )
                if result.returncode != 0:
                    return BenchmarkLog(
                        completed=False,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        error=self._error(result),
                    )
                return BenchmarkLog(
                    completed=True, stdout=result.stdout, stderr=result.stderr
                )
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
            return BenchmarkLog(completed=False, error=str(error))

    def benchmark_candidate(
        self, spec: TrainingSpec, proposal: ChangeProposal
    ) -> CandidateResult:
        log = self.benchmark_candidate_log(spec, proposal)
        if not log.completed:
            return CandidateResult(completed=False, error=log.error)
        try:
            benchmark = BenchmarkResult.model_validate(self._last_json(log.stdout))
        except ValueError as error:
            return CandidateResult(completed=False, error=str(error))
        return CandidateResult(completed=True, benchmark=benchmark)

    def _working_directory(self, spec: TrainingSpec, root: Path) -> Path:
        root = root.resolve()
        configured = Path(spec.working_directory)
        if configured.is_absolute():
            raise ValueError("working_directory must be relative to the repository")
        cwd = (root / configured).resolve()
        if not cwd.is_relative_to(root):
            raise ValueError("working_directory escapes the repository")
        if not cwd.is_dir():
            raise ValueError(
                f"working_directory does not exist: {spec.working_directory}"
            )
        return cwd

    def _validate_command(
        self,
        name: str,
        command: tuple[str, ...],
        cwd: Path,
        *,
        repository_root: Path | None = None,
    ) -> None:
        repository_root = (repository_root or self.root).resolve()
        argv, environment = self._prepare_command(command)

        executable = Path(argv[0])
        if executable.parent != Path("."):
            target = (cwd / executable).resolve()
            if not target.is_file():
                raise ValueError(
                    f"{name} executable does not exist relative to "
                    f"working_directory: {argv[0]}"
                )
        elif shutil.which(argv[0], path=environment.get("PATH")) is None:
            raise ValueError(f"{name} executable was not found: {argv[0]}")

        script = next(
            (argument for argument in argv[1:] if argument.endswith(".py")),
            None,
        )
        if script is None:
            return
        script_path = Path(script)
        target = (
            script_path.resolve()
            if script_path.is_absolute()
            else (cwd / script_path).resolve()
        )
        if not target.is_relative_to(repository_root):
            raise ValueError(f"{name} script is outside the repository: {script}")
        if not target.is_file():
            raise ValueError(
                f"{name} script does not exist relative to working_directory "
                f"{cwd.relative_to(repository_root)}: {script}"
            )

    def _prepare_command(
        self,
        command: tuple[str, ...],
    ) -> tuple[list[str], dict[str, str]]:
        if not command:
            raise ValueError("command is empty")
        argv = shlex.split(command[0]) if len(command) == 1 else list(command)
        if not argv:
            raise ValueError("command is empty")
        shell_operators = {"&&", "||", ";", "|", "<", ">", ">>"}
        if any(argument in shell_operators for argument in argv):
            raise ValueError(
                "shell operators are not supported; provide a single command"
            )

        environment = os.environ.copy()
        while argv:
            variable, separator, value = argv[0].partition("=")
            if not separator or _ENVIRONMENT_NAME.fullmatch(variable) is None:
                break
            environment[variable] = value
            argv.pop(0)
        if not argv:
            raise ValueError(
                "command contains environment assignments but no executable"
            )
        if Path(argv[0]).name.lower() in _SHELL_EXECUTABLES:
            raise ValueError(
                "shell executables are not supported; provide direct argv"
            )

        # A generated spec should not need to know where a repository's
        # environment lives. Resolve only conventional portable Python names;
        # an explicit interpreter remains the user's choice.
        repository_python = self.root / ".venv" / "bin" / "python"
        if argv[0] in {"python", "python3"} and repository_python.is_file():
            argv[0] = str(repository_python)

        # Training source imports should resolve from the repository regardless
        # of the disposable worktree used for execution. Preserve a user-supplied
        # PYTHONPATH, including one already equal to the repository root.
        current_pythonpath = environment.get("PYTHONPATH", "")
        pythonpath_entries = [
            entry for entry in current_pythonpath.split(os.pathsep) if entry
        ]
        repository_path = str(self.root)
        if repository_path not in pythonpath_entries:
            environment["PYTHONPATH"] = os.pathsep.join(
                [repository_path, *pythonpath_entries]
            )
        elif current_pythonpath:
            environment["PYTHONPATH"] = current_pythonpath
        else:
            environment["PYTHONPATH"] = repository_path
        return argv, environment

    def _run(
        self, command: tuple[str, ...], cwd: Path, *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        argv, environment = self._prepare_command(command)
        return subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=self.timeout if timeout is None else timeout,
            check=False,
        )

    @staticmethod
    def _last_json(stdout: str) -> object:
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise ValueError("command emitted no JSON result")
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise ValueError("the final stdout line is not valid JSON") from error

    @staticmethod
    def _error(result: subprocess.CompletedProcess[str]) -> str:
        return (
            result.stderr or result.stdout or f"command exited {result.returncode}"
        )[-4000:]

    @contextmanager
    def _patched_worktree(self, patch: str) -> Iterator[Path]:
        parent = Path(tempfile.mkdtemp(prefix="simplified-step-"))
        worktree = parent / "repo"
        add = subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                "HEAD",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if add.returncode != 0:
            shutil.rmtree(parent, ignore_errors=True)
            raise RuntimeError(add.stderr.strip() or "could not create Git worktree")
        try:
            if patch.strip():
                applied = subprocess.run(
                    ["git", "-C", str(worktree), "apply", "--whitespace=nowarn", "-"],
                    input=patch,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if applied.returncode != 0:
                    raise RuntimeError(
                        applied.stderr.strip() or "could not apply patch"
                    )
            yield worktree
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.root),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ],
                capture_output=True,
                check=False,
            )
            shutil.rmtree(parent, ignore_errors=True)

    def _preserve_trace(
        self,
        trace: TraceResult,
        worktree: Path,
        command_cwd: Path,
    ) -> TraceResult:
        if not trace.path:
            return trace
        source = Path(trace.path)
        source = (source if source.is_absolute() else command_cwd / source).resolve()
        if not source.exists() or not source.is_relative_to(worktree):
            return trace
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        destination = (
            self.artifact_directory / f"trace-{uuid4().hex[:12]}{source.suffix}"
        )
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return trace.model_copy(update={"path": str(destination)})


INPUTS: dict[str, tuple[tuple[str, type[BaseModel]], ...]] = {
    "hello": (),
    "run-all": (("request", TrainingRequest),),
    "accept-inputs": (("request", TrainingRequest),),
    "propose-inputs": (("request", TrainingRequest),),
    "review-inputs": (("spec", TrainingSpec),),
    "smoke": (("spec", TrainingSpec),),
    "benchmark": (("spec", TrainingSpec),),
    "propose-instrumentation": (("spec", TrainingSpec),),
    "review-instrumentation": (
        ("spec", TrainingSpec),
        ("plan", InstrumentationPlan),
    ),
    "profile": (("spec", TrainingSpec), ("plan", InstrumentationPlan)),
    "review-trace": (("trace", TraceResult),),
    "analyze-hotspots": (("trace", TraceResult),),
    "review-hotspots": (("trace", TraceResult), ("analysis", HotspotAnalysis)),
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
    "accept-inputs",
    "benchmark",
    "benchmark-candidate",
    "review-inputs",
    "propose-inputs",
    "propose-instrumentation",
    "analyze-hotspots",
    "review-hotspots",
    "propose-change",
}


def _load(path: str, model: type[BaseModel]) -> BaseModel:
    text = Path(path).read_text()
    if model is TrainingRequest and not text.lstrip().startswith("{"):
        return TrainingRequest(description=text.strip())
    return model.model_validate_json(text)


def _write(value: BaseModel, output: str) -> None:
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
    """Create the configured completion or Responses API client."""
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
        raise ValueError(
            "SIMPLIFIED_CLIENT_TYPE must be 'completion' or 'responses'"
        )

    options = {}
    if client_type:
        options["client_type"] = client_type
    if api_key:
        options["api_key"] = api_key
    if api_base := os.getenv("SIMPLIFIED_API_BASE"):
        options["api_base"] = api_base
    return get_llm_client(model, **options)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one optimizer step. Training requests may be plain text; "
            "all other inputs and outputs are typed JSON artifacts."
        )
    )
    parser.add_argument("step", choices=INPUTS)
    parser.add_argument(
        "inputs", nargs="*", help="input artifact files in the order shown below"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="output JSON file, or - for stdout"
    )
    parser.add_argument("--repo", default=".", help="source/training repository")
    parser.add_argument("--model", default=os.getenv("SIMPLIFIED_MODEL"))
    parser.add_argument(
        "--client-type",
        choices=("completion", "responses"),
        help=(
            "NOOA LLM transport; defaults to SIMPLIFIED_CLIENT_TYPE, then "
            "the model registry, then completion"
        ),
    )
    parser.add_argument(
        "--name", default="PhysicsNeMo", help="name used by the hello step"
    )
    parser.add_argument("--previous", help="optional Critique JSON for an LLM retry")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=300,
        help="maximum seconds for each input proposal or review (default: 300)",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--human-in-the-loop",
        action="store_true",
        help="ask for free-form terminal clarification when required",
    )
    parser.add_argument(
        "--artifacts", default="runs", help="directory for preserved trace files"
    )
    parser.add_argument(
        "--trace-dir",
        default="traces",
        help="write complete NOOA traces to this directory (default: traces)",
    )
    parser.add_argument(
        "--no-trace",
        dest="trace_dir",
        action="store_const",
        const=None,
        help="disable persistent trace files",
    )
    parser.add_argument(
        "--show-turns",
        action="store_true",
        help="show compact LLM, CodeAct, tool, and agent activity on stderr",
    )
    parser.add_argument(
        "--turn-detail",
        choices=("compact", "full"),
        default="compact",
        help="detail level for --show-turns (default: compact)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="color mode for --show-turns (default: auto)",
    )
    parser.epilog = "Input order:\n" + "\n".join(
        f"  {name}: " + ", ".join(field for field, _ in fields)
        for name, fields in INPUTS.items()
    )
    return parser


def _human_gate(args: argparse.Namespace) -> HumanGate:
    if args.human_in_the_loop:
        return TerminalHumanGate(color=args.color)
    return ClosedHumanGate()


async def _execute(args: argparse.Namespace) -> BaseModel:
    contract = INPUTS[args.step]
    if len(args.inputs) != len(contract):
        raise ValueError(
            f"{args.step} expects {len(contract)} inputs: "
            + ", ".join(name for name, _ in contract)
        )
    values = {
        name: _load(path, model)
        for path, (name, model) in zip(args.inputs, contract, strict=True)
    }
    previous = _load(args.previous, Critique) if args.previous else None
    source = LocalSourceEnvironment(args.repo)
    environment = LocalTrainingEnvironment(
        args.repo, timeout=args.timeout, artifact_directory=args.artifacts
    )
    llm: UnifiedLLM = (
        create_llm(args.model, client_type=args.client_type)
        if args.step in LLM_STEPS
        else FakeLLMClient()
    )
    try:
        if args.step == "hello":
            return HelloResponse.model_validate(
                await HelloAgent(llm=llm).hello(args.name)
            )
        if args.step == "run-all":
            agents = create_agents(llm, source, environment)
            return await TrainingOptimizer(
                agents,
                llm=llm,
                max_attempts=args.max_attempts,
                human_gate=_human_gate(args),
                agent_timeout=args.agent_timeout,
            ).run(values["request"])
        if args.step == "accept-inputs":
            proposer = InputProposer(llm=llm, source=source)
            contract_critic = InputContractCritic(llm=llm)
            critic = InputCritic(llm=llm, source=source)
            validator = Runner(llm=llm, environment=environment)
            acceptance = InputAcceptanceAgent(
                llm=llm,
                proposer=proposer,
                contract_critic=contract_critic,
                critic=critic,
                human_gate=_human_gate(args),
                validator=validator,
                max_attempts=args.max_attempts,
                agent_timeout=args.agent_timeout,
            )
            return await acceptance.accept(values["request"])
        if args.step == "propose-inputs":
            try:
                return await asyncio.wait_for(
                    InputProposer(llm=llm, source=source).propose(
                        values["request"], previous
                    ),
                    timeout=args.agent_timeout,
                )
            except TimeoutError as error:
                raise RuntimeError(
                    f"input proposal exceeded the {args.agent_timeout:g}s agent timeout"
                ) from error
        if args.step == "review-inputs":
            contract_review = InputContractCritic(llm=llm).review(values["spec"])
            if not contract_review.accepted:
                return contract_review
            try:
                return await asyncio.wait_for(
                    InputCritic(llm=llm, source=source).review(values["spec"]),
                    timeout=args.agent_timeout,
                )
            except TimeoutError as error:
                raise RuntimeError(
                    f"input review exceeded the {args.agent_timeout:g}s agent timeout"
                ) from error
        if args.step == "smoke":
            return RunResult.model_validate(environment.smoke(values["spec"]))
        if args.step == "benchmark":
            return await Runner(llm=llm, environment=environment).benchmark(
                values["spec"]
            )
        if args.step == "propose-instrumentation":
            return await InstrumentationAcceptanceAgent(
                llm=llm,
                proposer=InstrumentationProposer(llm=llm, source=source),
                critic=InstrumentationCritic(llm=llm, environment=environment),
                max_attempts=args.max_attempts,
            ).accept(values["spec"], REQUIRED_RANGES, previous)
        if args.step == "review-instrumentation":
            return InstrumentationCritic(
                llm=llm, environment=environment
            ).review(values["spec"], values["plan"], REQUIRED_RANGES)
        if args.step == "profile":
            return environment.profile(values["spec"], values["plan"])
        if args.step == "review-trace":
            return TraceCritic(llm=llm).review(values["trace"], REQUIRED_RANGES)
        if args.step == "analyze-hotspots":
            return await HotspotAnalyzer(llm=llm).analyze(values["trace"], previous)
        if args.step == "review-hotspots":
            return await HotspotCritic(llm=llm).review(
                values["trace"], values["analysis"]
            )
        if args.step == "route-hotspot":
            return Router(llm=llm).route(values["hotspot"])
        if args.step == "propose-change":
            return await ChangeProposer(llm=llm, source=source).propose(
                values["spec"], values["hotspot"], values["route"], previous
            )
        if args.step == "benchmark-candidate":
            return await Runner(llm=llm, environment=environment).benchmark_candidate(
                values["spec"], values["proposal"]
            )
        if args.step == "review-candidate":
            return CandidateCritic(llm=llm).review(
                values["spec"], values["baseline"], values["candidate"]
            )
        candidate = values["candidate"].benchmark
        if candidate is None:
            raise ValueError("build-report requires a candidate containing a benchmark")
        return ReportBuilder(llm=llm).build(
            values["baseline"], candidate, values["hotspot"], values["proposal"]
        )
    finally:
        await llm.aclose()


async def _main(args: argparse.Namespace) -> None:
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
        _write(await _execute(args), args.output)
    finally:
        if tracing:
            flush_traces()


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_main(args))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
