"""Tests for phase-1 bundle source-analysis contracts."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]


def load_script(name: str):
    path = SKILL / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


creator = load_script("create_run_bundle.py")
confirmer = load_script("confirm_run_config.py")
validator = load_script("validate_phase1_bundle.py")


def run_main(module, argv: list[str]) -> tuple[int, str]:
    original = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.argv = [module.__file__, *argv]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = module.main()
    finally:
        sys.argv = original
    return result, stdout.getvalue() + stderr.getvalue()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


class Phase1BundleTest(unittest.TestCase):
    def create_bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        result, output = run_main(
            creator,
            [
                "--output",
                str(bundle),
                "--workdir",
                str(REPOSITORY),
                "--eager-command",
                "python train.py compile=false",
                "--compiled-command",
                "python train.py compile=true",
                "--dataset",
                "sample",
                "--correctness-command",
                "finite loss",
                "--performance-goal",
                "step time",
                "--entry-point",
                "train.py",
                "--model-config",
                "model=test",
                "--batch-size",
                "1",
                "--hardware",
                "test GPU",
                "--framework-stack",
                "PyTorch test stack",
                "--precision",
                "float32",
            ],
        )
        self.assertEqual(result, 0, output)
        return bundle

    def test_new_bundle_contains_source_analysis_skeletons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.create_bundle(Path(directory))
            source_map = json.loads((bundle / "phase-source-map.json").read_text())
            analysis = json.loads((bundle / "source-analysis.json").read_text())
            self.assertEqual(
                {item["phase"] for item in source_map["mappings"]},
                set(creator.CANONICAL_PHASES),
            )
            self.assertEqual(
                {item["phase"] for item in analysis["reviews"]},
                set(creator.CANONICAL_PHASES),
            )
            self.assertTrue((bundle / "test-config.json").is_file())
            self.assertTrue((bundle / "test-config.md").is_file())
            confirmation = json.loads(
                (bundle / "config-confirmation.json").read_text()
            )
            self.assertEqual(confirmation["status"], "pending")
            for variant in ("eager", "compiled"):
                annotation_health = json.loads(
                    (bundle / f"annotation-health-{variant}.json").read_text()
                )
                self.assertEqual(annotation_health["status"], "pending")
                self.assertEqual(annotation_health["variant"], variant)
            config = json.loads((bundle / "test-config.json").read_text())
            self.assertFalse(config["comparison"]["variants"]["eager"]["compile"])
            self.assertTrue(config["comparison"]["variants"]["compiled"]["compile"])
            result, output = run_main(validator, [str(bundle)])
            self.assertEqual(result, 0, output)
            result, output = run_main(validator, [str(bundle), "--confirmed"])
            self.assertEqual(result, 1, output)

    def test_bundle_rejects_invalid_compile_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.create_bundle(Path(directory))
            config_path = bundle / "test-config.json"
            config = json.loads(config_path.read_text())
            config["comparison"]["variants"]["eager"]["compile"] = True
            write_json(config_path, config)
            result, output = run_main(validator, [str(bundle)])
            self.assertEqual(result, 1, output)
            self.assertIn("eager variant must set compile=false", output)

    def test_confirmation_matches_current_config_and_rejects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.create_bundle(Path(directory))
            result, output = run_main(confirmer, [str(bundle)])
            self.assertEqual(result, 0, output)
            result, output = run_main(validator, [str(bundle), "--confirmed"])
            self.assertEqual(result, 0, output)

            config_path = bundle / "test-config.json"
            config = json.loads(config_path.read_text())
            config["protocol"]["measure_steps"] += 1
            write_json(config_path, config)
            result, output = run_main(validator, [str(bundle), "--confirmed"])
            self.assertEqual(result, 1, output)
            self.assertIn("reconfirm", output)

    def test_ready_bundle_requires_complete_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.create_bundle(Path(directory))
            run_id = json.loads((bundle / "run-manifest.json").read_text())["run_id"]

            baseline = json.loads((bundle / "baseline.json").read_text())
            baseline["status"] = "complete"
            for variant in ("eager", "compiled"):
                baseline["variants"][variant].update(
                    {"status": "complete", "repetitions": [{"step_ms": 10}]}
                )
            baseline["variants"]["compiled"]["cold_start"].update(
                {
                    "status": "complete",
                    "compilation_time_ms": 100.0,
                    "first_step_time_ms": 110.0,
                    "cache_state": "cold",
                }
            )
            baseline["comparison"].update(
                {
                    "status": "complete",
                    "steady_state_speedup": 1.1,
                    "classification": "beneficial",
                }
            )
            write_json(bundle / "baseline.json", baseline)
            correctness = json.loads((bundle / "correctness.json").read_text())
            correctness["status"] = "pass"
            for variant in ("eager", "compiled"):
                correctness["variants"][variant]["status"] = "pass"
            correctness["comparison"]["status"] = "pass"
            write_json(bundle / "correctness.json", correctness)

            for variant in ("eager", "compiled"):
                path = bundle / f"annotation-health-{variant}.json"
                annotation_health = json.loads(path.read_text())
                annotation_health.update(
                    {
                        "status": "pass",
                        "trace": f"traces/{variant}/rank0.json",
                        "step_boundary": {
                            "provenance": "native_profiler_step",
                            "marker": "ProfilerStep#N",
                            "count": 1,
                            "selection_rule": None,
                        },
                    }
                )
                annotation_health["annotations"]["required_phases"] = ["forward"]
                annotation_health["annotations"]["phase_counts"] = {"forward": 1}
                annotation_health["kernel_attribution"] = {
                    "method": "hta_user_annotation",
                    "fallback_used": False,
                }
                write_json(path, annotation_health)

            compile_analysis = json.loads((bundle / "compile-analysis.json").read_text())
            compile_analysis.update(
                {
                    "status": "complete",
                    "logs": ["logs/compiled/torch-compile.log"],
                    "summary": {"unique_graph_breaks": 0},
                }
            )
            write_json(bundle / "compile-analysis.json", compile_analysis)
            compile_comparison = json.loads(
                (bundle / "compile-comparison.json").read_text()
            )
            compile_comparison.update(
                {
                    "status": "complete",
                    "classification": "beneficial",
                    "correctness_status": "pass",
                    "steady_state_speedup": 1.1,
                    "compile_time_ms": 100.0,
                    "amortization_steps": 100,
                    "conclusion": "Compilation improves steady-state time.",
                    "evidence": ["baseline.json"],
                }
            )
            write_json(bundle / "compile-comparison.json", compile_comparison)

            source = {
                "path": "train.py",
                "start_line": 1,
                "end_line": 10,
                "symbol": "train",
            }
            mappings = []
            reviews = []
            for phase in creator.CANONICAL_PHASES:
                if phase in ("distributed_sync", "validation", "checkpoint"):
                    mappings.append(
                        {
                            "phase": phase,
                            "coverage": "not_applicable",
                            "ranges": [],
                            "sources": [],
                            "config": [],
                            "evidence": [],
                            "reason": "Not used by this smoke workload.",
                        }
                    )
                    reviews.append(
                        {
                            "phase": phase,
                            "review_status": "not_applicable",
                            "hotspot_ids": [],
                            "measured_evidence": [],
                            "sources": [],
                            "code_observations": [],
                            "recommendations": [],
                            "no_change_reason": "Not used by this smoke workload.",
                        }
                    )
                    continue
                mapping_source = {**source, "role": f"Implements {phase}"}
                mappings.append(
                    {
                        "phase": phase,
                        "coverage": "mapped",
                        "ranges": [phase],
                        "sources": [mapping_source],
                        "config": [],
                        "evidence": [f"hta/{phase}.json"],
                        "reason": None,
                    }
                )
                review = {
                    "phase": phase,
                    "review_status": "reviewed",
                    "hotspot_ids": ["H001"] if phase == "forward" else [],
                    "measured_evidence": [f"hta/{phase}.json"],
                    "sources": [source],
                    "code_observations": [f"Reviewed {phase} implementation."],
                    "recommendations": [],
                    "no_change_reason": "No material change justified.",
                }
                if phase == "forward":
                    review["no_change_reason"] = None
                    review["recommendations"] = [
                        {
                            "id": "S001",
                            "code_observation": "A custom operation repeats work.",
                            "mechanism": "Repeated launches add critical-path work.",
                            "suggested_improvement": "Reuse the computed result.",
                            "recommended_skill": "physicsnemo-functionals-integrator",
                            "phase2_experiment": "Reuse one result for one step.",
                            "correctness_check": "Compare outputs and loss.",
                            "performance_check": "Compare launches and step time.",
                            "semantic_risk": "medium",
                            "confidence": "high",
                            "status": "recommendation_only",
                        }
                    ]
                reviews.append(review)

            write_json(
                bundle / "phase-source-map.json",
                {
                    "schema_version": "0.1",
                    "phase": 1,
                    "run_id": run_id,
                    "status": "complete",
                    "mappings": mappings,
                },
            )
            write_json(
                bundle / "source-analysis.json",
                {
                    "schema_version": "0.1",
                    "phase": 1,
                    "run_id": run_id,
                    "status": "complete",
                    "reviews": reviews,
                },
            )
            write_json(
                bundle / "findings.json",
                {
                    "schema_version": "0.1",
                    "phase": 1,
                    "run_id": run_id,
                    "summary": "Forward repeats custom work.",
                    "residual_bottlenecks": [],
                    "hotspots": [
                        {
                            "id": "H001",
                            "category": "custom_math",
                            "severity": "high",
                            "confidence": "high",
                            "summary": "Repeated custom work is on the critical path.",
                            "evidence": [{"artifact": "hta/forward.json"}],
                            "recommended_skill": "physicsnemo-functionals-integrator",
                            "proposed_experiment": "Reuse one result.",
                            "source_analysis_ids": ["S001"],
                            "status": "recommendation_only",
                        }
                    ],
                },
            )
            for variant in ("eager", "compiled"):
                (bundle / "traces" / variant / "rank0.json").write_text("{}\n")
                (bundle / "hta" / variant / "diagram-data.json").write_text("{}\n")
                diagrams = bundle / "hta" / variant / "diagrams"
                (diagrams / "diagram-manifest.json").write_text("{}\n")
                (diagrams / "cpu-gpu-pipeline.svg").write_text("<svg/>\n")
                (diagrams / "forward-dominant-kernels.svg").write_text("<svg/>\n")
            report = (bundle / "report.md").read_text()
            (bundle / "report.md").write_text(
                re.sub(r"\{\{[^{}]+\}\}", "completed", report)
            )

            result, output = run_main(confirmer, [str(bundle)])
            self.assertEqual(result, 0, output)
            result, output = run_main(validator, [str(bundle), "--ready"])
            self.assertEqual(result, 0, output)

            baseline = json.loads((bundle / "baseline.json").read_text())
            baseline["comparison"]["timing_basis"] = "profiled"
            write_json(bundle / "baseline.json", baseline)
            result, output = run_main(validator, [str(bundle), "--ready"])
            self.assertEqual(result, 1, output)
            self.assertIn("unprofiled_steady_state", output)


if __name__ == "__main__":
    unittest.main()
