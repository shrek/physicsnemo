"""Tests for dependency-free HTA report diagram rendering."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "render_trace_diagrams.py"
)
SPEC = importlib.util.spec_from_file_location("render_trace_diagrams", SCRIPT)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def diagram_data() -> dict:
    return {
        "schema_version": "0.1",
        "pipeline": {
            "title": "Golden CPU/GPU pipeline",
            "step_id": "ProfilerStep#3",
            "duration_ms": 100.0,
            "summary": "GPU idle is dominated by dataload.",
            "lanes": [
                {
                    "name": "CPU",
                    "spans": [
                        {
                            "label": "dataload",
                            "start_ms": 0.0,
                            "end_ms": 35.0,
                            "category": "data",
                        },
                        {
                            "label": "forward",
                            "start_ms": 35.0,
                            "end_ms": 75.0,
                            "category": "forward",
                        },
                    ],
                },
                {
                    "name": "GPU",
                    "spans": [
                        {
                            "label": "idle before batch",
                            "start_ms": 0.0,
                            "end_ms": 30.0,
                            "category": "bubble",
                            "annotation": "CPU dataload",
                        },
                        {
                            "label": "forward kernels",
                            "start_ms": 35.0,
                            "end_ms": 70.0,
                            "category": "forward",
                        },
                    ],
                },
            ],
        },
        "forward": {
            "title": "Forward dominant kernels",
            "steps": [
                {
                    "step_id": "ProfilerStep#3",
                    "duration_ms": 40.0,
                    "kernels": [
                        {
                            "label": "radius_search",
                            "family": "radius_search",
                            "category": "radius_search",
                            "start_ms": 2.0,
                            "end_ms": 22.0,
                        },
                        {
                            "label": "gemm",
                            "family": "gemm",
                            "category": "gemm",
                            "start_ms": 24.0,
                            "end_ms": 30.0,
                        },
                    ],
                }
            ],
        },
        "multi_step": {
            "title": "Consecutive steps",
            "steps": [
                {
                    "step_id": "step 3",
                    "start_ms": 0.0,
                    "end_ms": 100.0,
                    "gpu_busy_start_ms": 30.0,
                    "gpu_busy_end_ms": 70.0,
                },
                {
                    "step_id": "step 4",
                    "start_ms": 100.0,
                    "end_ms": 200.0,
                    "gpu_busy_start_ms": 125.0,
                    "gpu_busy_end_ms": 170.0,
                },
            ],
        },
    }


class RenderTraceDiagramsTest(unittest.TestCase):
    def test_validates_and_renders_all_supported_views(self) -> None:
        document = diagram_data()
        renderer.validate_document(document)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            renderer.render_pipeline(document, output / "cpu-gpu-pipeline.svg")
            renderer.render_forward(
                document, output / "forward-dominant-kernels.svg"
            )
            self.assertTrue(
                renderer.render_idle_bubbles(
                    document, output / "in-step-gpu-bubbles.svg"
                )
            )
            self.assertTrue(
                renderer.render_multi_step(
                    document, output / "multi-step-gpu-bubbles.svg"
                )
            )
            expected = (
                "cpu-gpu-pipeline.svg",
                "forward-dominant-kernels.svg",
                "in-step-gpu-bubbles.svg",
                "multi-step-gpu-bubbles.svg",
            )
            for name in expected:
                text = (output / name).read_text()
                self.assertIn("<svg", text)
                self.assertIn("</svg>", text)
                ET.parse(output / name)
            self.assertIn(
                "radius_search",
                (output / "forward-dominant-kernels.svg").read_text(),
            )

    def test_rejects_span_outside_step(self) -> None:
        document = diagram_data()
        document["pipeline"]["lanes"][0]["spans"][0]["end_ms"] = 101.0
        with self.assertRaisesRegex(renderer.InputError, "exceeds"):
            renderer.validate_document(document)

    def test_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "diagram-data.json"
            output = root / "diagrams"
            input_path.write_text(json.dumps(diagram_data()))
            original_argv = renderer.sys.argv
            try:
                renderer.sys.argv = [
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output),
                ]
                self.assertEqual(renderer.main(), 0)
            finally:
                renderer.sys.argv = original_argv
            manifest = json.loads((output / "diagram-manifest.json").read_text())
            self.assertEqual(len(manifest["outputs"]), 4)


if __name__ == "__main__":
    unittest.main()
