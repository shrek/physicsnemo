"""Tests for bounded torch.compile diagnostic normalization."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_compile_logs.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_compile_logs", SCRIPT)
assert SPEC and SPEC.loader
compile_logs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compile_logs)


class CompileLogTest(unittest.TestCase):
    def test_deduplicates_breaks_and_recompiles_with_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            graph_break = (
                "Graph break in user code at /repo/train.py:41\n"
                "Reason: Unsupported: Tensor.item\n"
            )
            recompile = (
                "Recompiling function forward in /repo/model.py:12\n"
                "    - 0/0: tensor size mismatch at index 0\n"
            )
            log.write_text(graph_break + graph_break + recompile + recompile)
            result = compile_logs.parse_logs([log], "run")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["graph_breaks"][0]["count"], 2)
            self.assertEqual(result["recompilations"][0]["count"], 2)
            self.assertEqual(result["summary"]["unique_graph_breaks"], 1)
            self.assertEqual(result["summary"]["unique_recompilations"], 1)

    def test_records_cache_limit_and_eager_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text(
                "torch._dynamo hit config.recompile_limit (8)\n"
                "BackendCompilerFailed: falling back to eager\n"
            )
            result = compile_logs.parse_logs([log], "run")
            self.assertEqual(len(result["cache_limit_warnings"]), 1)
            self.assertTrue(
                result["summary"]["eager_fallback_or_backend_failure"]
            )

    def test_empty_diagnostic_log_is_complete_but_not_fullgraph_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "compile.log"
            log.write_text("")
            result = compile_logs.parse_logs([log], "run")
            self.assertEqual(result["graph_breaks"], [])
            self.assertIn("not proof", result["notes"][1])


if __name__ == "__main__":
    unittest.main()
