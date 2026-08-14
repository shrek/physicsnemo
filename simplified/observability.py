"""NOOA trace wiring for optimizer runs and the step CLI."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO
from uuid import uuid4

from nooa.tracing import enable_tracing, exporters, flush_traces, set_session
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from simplified.types import OptimizationResult, TrainingRequest
from simplified.workflow import TrainingOptimizer


ColorMode = Literal["auto", "always", "never"]
TurnDetail = Literal["compact", "full"]


@dataclass(frozen=True)
class ObservedRun:
    session_id: str
    result: OptimizationResult


def trace_to_viewer(endpoint: str = "http://localhost:5001/v1/traces") -> None:
    """Send complete traces to the built-in NOOA viewer."""
    enable_tracing(
        exporters=[exporters.journal(endpoint)],
        experiment="training-optimizer",
    )


def trace_to_files(directory: str | Path = "traces") -> None:
    """Write portable NOOA traces when a live viewer is not running."""
    enable_tracing(
        exporters=[exporters.jsonl(directory)],
        experiment="training-optimizer",
    )


def trace_for_cli(
    *,
    directory: str | Path | None = None,
    show_turns: bool = False,
    color: ColorMode = "auto",
    turn_detail: TurnDetail = "compact",
) -> None:
    """Configure file and/or concise live tracing for ``simplified-step``."""
    configured: list[SpanExporter] = []
    if directory is not None:
        configured.append(exporters.jsonl(directory))
    if show_turns:
        configured.append(LiveTurnExporter(color=color, detail=turn_detail))
    if configured:
        with redirect_stdout(sys.stderr):
            enable_tracing(exporters=configured, experiment="training-optimizer")


async def run_observed(
    optimizer: TrainingOptimizer,
    request: TrainingRequest,
    *,
    session_id: str | None = None,
) -> ObservedRun:
    """Run the optimizer in a fresh trace session and flush it before returning."""
    session_id = session_id or f"training-optimizer-{uuid4().hex[:12]}"
    set_session(session_id)
    try:
        result = await optimizer.run(request)
        return ObservedRun(session_id=session_id, result=result)
    finally:
        flush_traces()


class LiveTurnExporter(SpanExporter):
    """Render LLM turns, generated code, and method calls to stderr."""

    synchronous = True

    _COLORS = {
        "turn": "\033[96m",
        "call": "\033[95m",
        "code": "\033[93m",
        "agent": "\033[94m",
        "ok": "\033[92m",
        "error": "\033[91m",
        "detail": "\033[90m",
        "reset": "\033[0m",
    }

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        color: ColorMode = "auto",
        detail: TurnDetail = "compact",
        max_value_chars: int | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.detail = detail
        self.max_value_chars = max_value_chars or (240 if detail == "compact" else 800)
        self.turn = 0
        self._use_color = color == "always" or (
            color == "auto"
            and os.getenv("NO_COLOR") is None
            and bool(getattr(self.stream, "isatty", lambda: False)())
        )

    def export(self, spans) -> SpanExportResult:
        try:
            for span in spans:
                self._render(span)
        except Exception:
            # Observability must never make the agent operation fail.
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def describe(self) -> str:
        return "live turns on stderr"

    def _paint(self, style: str, value: str) -> str:
        if not self._use_color:
            return value
        return f"{self._COLORS[style]}{value}{self._COLORS['reset']}"

    def _line(self, style: str, label: str, body: str) -> None:
        print(f"{self._paint(style, label):>16}  {body}", file=self.stream, flush=True)

    def _render(self, span: ReadableSpan) -> None:
        attrs = dict(span.attributes or {})
        kind = str(attrs.get("openinference.span.kind", ""))
        plugin = str(attrs.get("nooa.viewer.plugin", ""))
        suffix, is_error = self._suffix(span, attrs)

        if kind == "LLM":
            self.turn += 1
            model = str(attrs.get("llm.model_name", span.name))
            self._line("turn", f"turn {self.turn}", f"LLM {model}  {suffix}")
            if self.detail == "full":
                self._detail("response", attrs.get("output.value"))
            return

        if plugin == "code_execution" or span.name == "code_execution":
            code = self._code_from(attrs.get("input.value"))
            if self.detail == "compact":
                action = self._quantity("code", code)
                result = self._summary(attrs.get("output.value"))
                self._line(
                    "code", "code action", f"{suffix}  {action} -> {result}"
                )
                return
            self._line("code", "generated code", suffix)
            self._detail("code", code)
            self._detail("result", attrs.get("output.value"))
            return

        if span.name.startswith("method_call."):
            method = attrs.get("method.name") or span.name.removeprefix("method_call.")
            agent = attrs.get("agent.name")
            owner = f"{agent}." if agent else ""
            if self.detail == "compact":
                arguments = self._summary(attrs.get("input.value"))
                result = self._summary(attrs.get("output.value"))
                self._line(
                    "call",
                    "method call",
                    f"{owner}{method}  {suffix}  {arguments} -> {result}",
                )
                return
            self._line("call", "method call", f"{owner}{method}  {suffix}")
            self._detail("arguments", attrs.get("input.value"))
            self._detail("result", attrs.get("output.value"))
            return

        if plugin == "tool_execution" or (kind == "TOOL" and attrs.get("tool.name")):
            tool = str(attrs.get("tool.name", span.name))
            if self.detail == "compact":
                if tool == "return_result":
                    return
                arguments = self._summary(attrs.get("input.value"))
                result = self._summary(attrs.get("output.value"))
                self._line(
                    "call", "tool call", f"{tool}  {suffix}  {arguments} -> {result}"
                )
                return
            self._line("call", "tool call", f"{tool}  {suffix}")
            self._detail("arguments", attrs.get("input.value"))
            self._detail("result", attrs.get("output.value"))
            return

        if span.name.startswith("method."):
            method = attrs.get("agent.method") or span.name.removeprefix("method.")
            agent = attrs.get("agent.name", "Agent")
            self._line("agent", "agent method", f"{agent}.{method}  {suffix}")
            result = attrs.get("output.value")
            if self.detail == "compact":
                result = self._summary(result)
            self._detail("result", result)
            if is_error:
                self._detail("error", attrs.get("error.message"))
            return

        if span.name == "generation":
            if self.detail == "compact":
                return
            strategy = attrs.get("generation.strategy", "generation")
            agent = attrs.get("agent.name", "Agent")
            method = attrs.get("agent.method", "")
            self._line(
                "agent", "generation", f"{agent}.{method} via {strategy}  {suffix}"
            )

    def _suffix(self, span: ReadableSpan, attrs: dict[str, Any]) -> tuple[str, bool]:
        duration = self._duration(span)
        code = getattr(getattr(span, "status", None), "status_code", None)
        is_error = getattr(code, "name", str(code or "OK")) == "ERROR"
        status = self._paint(
            "error" if is_error else "ok", "error" if is_error else "ok"
        )
        suffix = f"{duration}  {status}"
        tokens = attrs.get("llm.token_count.total")
        if tokens is not None:
            try:
                tokens = f"{int(tokens):,}"
            except (TypeError, ValueError):
                pass
            suffix = f"{suffix}  {tokens} tokens"
        return suffix, is_error

    @staticmethod
    def _duration(span: ReadableSpan) -> str:
        if span.start_time is None or span.end_time is None:
            return ""
        seconds = max(0.0, (span.end_time - span.start_time) / 1_000_000_000)
        return f"{seconds:.2f}s"

    def _detail(self, label: str, value: Any) -> None:
        if value is None or value == "" or value == "none":
            return
        rendered = self._compact(value)
        prefix = self._paint("detail", f"  {label}:")
        print(f"{prefix} {rendered}", file=self.stream, flush=True)

    def _compact(self, value: Any) -> str:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str)
        value = " ".join(value.strip().split())
        if len(value) > self.max_value_chars:
            value = value[: self.max_value_chars - 1] + "…"
        return value

    def _summary(self, value: Any) -> str:
        if value is None or value == "":
            return "none"
        decoded = value
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
        if isinstance(decoded, dict):
            streams = [
                self._quantity(name, decoded.get(name))
                for name in ("stdout", "stderr")
                if decoded.get(name)
            ]
            if streams:
                return ", ".join(streams)
            if "code" in decoded:
                return self._quantity("code", decoded["code"])
            return self._compact(decoded)
        if isinstance(decoded, (list, tuple)):
            return f"{type(decoded).__name__}[{len(decoded)}]"
        text = str(decoded).strip()
        normalized = " ".join(text.split())
        if len(normalized) <= self.max_value_chars:
            return normalized
        kind = normalized.split("(", 1)[0]
        if kind == "Critique" and "(" in normalized:
            return self._compact(normalized)
        if kind.isidentifier() and "(" in normalized:
            return f"{kind} ({len(text):,} chars)"
        return self._quantity("value", text)

    @staticmethod
    def _quantity(label: str, value: Any) -> str:
        if value is None:
            return f"{label} 0 chars"
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str)
        return f"{label} {len(value):,} chars"

    @staticmethod
    def _code_from(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("code", value)
        if not isinstance(value, str):
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return decoded.get("code", value) if isinstance(decoded, dict) else value
