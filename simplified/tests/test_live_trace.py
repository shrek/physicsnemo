from io import StringIO
from types import SimpleNamespace

from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Status, StatusCode

from simplified.observability import LiveTurnExporter


def _span(name, attributes, *, seconds=1.0, status=StatusCode.OK):
    return SimpleNamespace(
        name=name,
        attributes=attributes,
        start_time=0,
        end_time=int(seconds * 1_000_000_000),
        status=Status(status),
    )


def test_live_turn_exporter_renders_llm_and_method_call_without_color():
    stream = StringIO()
    exporter = LiveTurnExporter(stream=stream, color="never", detail="full")

    result = exporter.export(
        [
            _span(
                "litellm.acompletion",
                {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "deepseek-v4-pro",
                    "llm.token_count.total": 321,
                    "output.value": "I will inspect the training recipe.",
                },
                seconds=1.2,
            ),
            _span(
                "method_call.list_files",
                {
                    "openinference.span.kind": "TOOL",
                    "agent.name": "InputProposer",
                    "method.name": "list_files",
                    "input.value": '{"args":["**/*geotransolver*"],"kwargs":{}}',
                    "output.value": '["geotransolver_volume.yaml"]',
                },
                seconds=0.01,
            ),
        ]
    )

    output = stream.getvalue()
    assert result is SpanExportResult.SUCCESS
    assert "turn 1" in output
    assert "LLM deepseek-v4-pro" in output
    assert "321 tokens" in output
    assert "InputProposer.list_files" in output
    assert "geotransolver_volume.yaml" in output
    assert "\033[" not in output


def test_live_turn_exporter_colors_errors_and_bounds_details():
    stream = StringIO()
    exporter = LiveTurnExporter(
        stream=stream,
        color="always",
        max_value_chars=20,
    )

    exporter.export(
        [
            _span(
                "method.propose",
                {
                    "openinference.span.kind": "AGENT",
                    "agent.name": "InputProposer",
                    "agent.method": "propose",
                    "error.message": "x" * 100,
                },
                status=StatusCode.ERROR,
            )
        ]
    )

    output = stream.getvalue()
    assert "\033[91merror\033[0m" in output
    assert "InputProposer.propose" in output
    assert "xxxxxxxxxxxxxxxxxxx…" in output


def test_compact_turns_hide_payloads_and_summarize_codeact_activity():
    stream = StringIO()
    exporter = LiveTurnExporter(stream=stream, color="never")
    response = "private response " * 100
    code = "print('inspect')\n" * 100
    stdout = "repository contents\n" * 200
    training_spec = "TrainingSpec(" + ("command=" * 200) + ")"

    exporter.export(
        [
            _span(
                "litellm.acompletion",
                {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "deepseek-v4-pro",
                    "output.value": response,
                },
            ),
            _span(
                "code_execution",
                {
                    "nooa.viewer.plugin": "code_execution",
                    "input.value": {"code": code},
                    "output.value": {"stdout": stdout},
                },
                seconds=0.01,
            ),
            _span(
                "tool.return_result",
                {
                    "openinference.span.kind": "TOOL",
                    "nooa.viewer.plugin": "tool_execution",
                    "tool.name": "return_result",
                    "input.value": training_spec,
                    "output.value": training_spec,
                },
                seconds=0.01,
            ),
            _span(
                "generation",
                {
                    "agent.name": "InputProposer",
                    "agent.method": "propose",
                    "generation.strategy": "CODEACT",
                },
            ),
            _span(
                "method.propose",
                {
                    "agent.name": "InputProposer",
                    "agent.method": "propose",
                    "output.value": training_spec,
                },
            ),
        ]
    )

    output = stream.getvalue()
    assert "turn 1" in output
    assert "code action" in output
    assert f"code {len(code):,} chars" in output
    assert f"stdout {len(stdout):,} chars" in output
    assert f"TrainingSpec ({len(training_spec):,} chars)" in output
    assert "private response" not in output
    assert "repository contents" not in output
    assert "print('inspect')" not in output
    assert "return_result" not in output
    assert "generation" not in output


def test_compact_turns_include_long_critic_feedback():
    stream = StringIO()
    exporter = LiveTurnExporter(stream=stream, color="never")
    critique = (
        "Critique(accepted=False, feedback='Profile must include a bounded "
        + ("steady-state capture window. " * 30)
        + "', requires_human=False)"
    )

    exporter.export(
        [
            _span(
                "method.review",
                {
                    "agent.name": "InputCritic",
                    "agent.method": "review",
                    "output.value": critique,
                },
            )
        ]
    )

    output = stream.getvalue()
    assert "accepted=False" in output
    assert "Profile must include a bounded" in output
    assert "Critique (" not in output
