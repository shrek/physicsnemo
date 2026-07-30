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
                "--command",
                "python train.py",
                "--dataset",
                "sample",
                "--correctness-command",
                "finite loss",
                "--performance-goal",
                "step time",
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
            result, output = run_main(validator, [str(bundle)])
            self.assertEqual(result, 0, output)

    def test_ready_bundle_requires_complete_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.create_bundle(Path(directory))
            run_id = json.loads((bundle / "run-manifest.json").read_text())["run_id"]

            baseline = json.loads((bundle / "baseline.json").read_text())
            baseline.update({"status": "complete", "repetitions": [{"step_ms": 10}]})
            write_json(bundle / "baseline.json", baseline)
            correctness = json.loads((bundle / "correctness.json").read_text())
            correctness["status"] = "pass"
            write_json(bundle / "correctness.json", correctness)

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
            (bundle / "traces" / "rank0.json").write_text("{}\n")
            (bundle / "hta" / "diagram-data.json").write_text("{}\n")
            diagrams = bundle / "hta" / "diagrams"
            (diagrams / "diagram-manifest.json").write_text("{}\n")
            (diagrams / "cpu-gpu-pipeline.svg").write_text("<svg/>\n")
            (diagrams / "forward-dominant-kernels.svg").write_text("<svg/>\n")
            report = (bundle / "report.md").read_text()
            (bundle / "report.md").write_text(
                re.sub(r"\{\{[^{}]+\}\}", "completed", report)
            )

            result, output = run_main(validator, [str(bundle), "--ready"])
            self.assertEqual(result, 0, output)


if __name__ == "__main__":
    unittest.main()
