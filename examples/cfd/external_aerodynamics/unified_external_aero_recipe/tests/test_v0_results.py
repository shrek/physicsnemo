import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from v0_results import benchmark_result, trace_result


def test_benchmark_result_uses_post_warmup_median_and_validation_loss(tmp_path):
    records = [
        {"phase": "train_step", "step_time_s": value}
        for value in (0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080)
    ]
    records.append({"phase": "val_summary", "loss": 0.125})
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    result = benchmark_result(metrics_path)

    assert result == {
        "step_time_ms": 70.0,
        "correctness_value": 0.125,
        "correctness_metric": "validation_loss",
    }


def test_trace_result_reports_only_canonical_ranges_present_in_trace(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "forward"},
                    {"name": "unrelated_kernel"},
                    {"name": "optimizer_step"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = trace_result(trace_path)

    assert result["completed"] is True
    assert result["path"] == str(trace_path)
    assert result["ranges"] == ["forward", "optimizer_step"]
    assert "3 events" in result["summary"]
