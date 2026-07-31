"""Tests for trace annotation health and reconstruction provenance."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_trace_annotations.py"
)
SPEC = importlib.util.spec_from_file_location("validate_trace_annotations", SCRIPT)
assert SPEC and SPEC.loader
annotations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotations)


def event(name: str, *, category: str = "user_annotation", ts: int = 1) -> dict:
    return {"name": name, "cat": category, "ph": "X", "ts": ts, "dur": 1}


def write_trace(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps({"traceEvents": events}))


class TraceAnnotationTest(unittest.TestCase):
    def analyze(
        self,
        root: Path,
        events: list[dict],
        *,
        allow_reconstruction: bool = False,
        compile_log: Path | None = None,
        hta_rows: Path | None = None,
    ) -> dict:
        trace = root / "trace.json"
        write_trace(trace, events)
        return annotations.analyze(
            trace,
            run_id="test-run",
            required_phases=annotations.DEFAULT_REQUIRED_PHASES,
            expected_steps=1,
            allow_reconstruction=allow_reconstruction,
            compile_log=compile_log,
            hta_rows=hta_rows,
        )

    def phase_events(self) -> list[dict]:
        return [
            event(name, ts=index + 10)
            for index, name in enumerate(annotations.DEFAULT_REQUIRED_PHASES)
        ]

    def test_native_profiler_step_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(
                Path(directory),
                [event("ProfilerStep#0"), *self.phase_events()],
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["step_boundary"]["provenance"], "native_profiler_step"
            )

    def test_missing_marker_can_be_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(
                Path(directory),
                self.phase_events(),
                allow_reconstruction=True,
            )
            self.assertEqual(result["status"], "reconstructed")
            self.assertEqual(
                result["kernel_attribution"]["method"], "index_correlation"
            )

    def test_projected_annotations_are_detected_as_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events = self.phase_events()
            events.extend(
                event(item["name"], category="gpu_user_annotation", ts=item["ts"])
                for item in self.phase_events()
            )
            result = self.analyze(
                Path(directory),
                [event("ProfilerStep#0"), *events],
            )
            self.assertTrue(
                result["annotations"]["duplicate_projected_annotations"]
            )
            self.assertEqual(
                result["annotations"]["projected_annotation_count"],
                len(annotations.DEFAULT_REQUIRED_PHASES),
            )

    def test_invalid_hta_end_requires_derived_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = root / "hta.csv"
            rows.write_text("ts,dur,end\n100,10,5\n")
            result = self.analyze(
                root,
                [event("ProfilerStep#0"), *self.phase_events()],
                hta_rows=rows,
            )
            self.assertTrue(result["timestamps"]["derived_end_required"])
            self.assertEqual(result["timestamps"]["end_before_start_count"], 1)

    def test_compile_warning_is_reported_without_hiding_outer_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "profile.log"
            log.write_text(
                "Profiler record function foo will be ignored under torch.compile\n"
            )
            result = self.analyze(
                root,
                [event("ProfilerStep#0"), *self.phase_events()],
                compile_log=log,
            )
            self.assertEqual(result["status"], "pass")
            self.assertTrue(
                result["annotations"]["compile_record_function_warning"]
            )


if __name__ == "__main__":
    unittest.main()
