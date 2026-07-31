"""Tests for the standalone NCU workflow."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ncu_profile.py"
SPEC = importlib.util.spec_from_file_location("ncu_profile", SCRIPT)
assert SPEC and SPEC.loader
ncu_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ncu_profile)


def capture_spec(root: Path) -> dict:
    return {
        "schema_version": "0.1",
        "candidate": {
            "id": "H003",
            "reason": "Kernel is on the critical path.",
            "question": "What limits this kernel?",
            "source": "findings.json",
        },
        "selection": {
            "kernel_regex": ".*radius_search.*",
            "nvtx_include": "neighbor_search/",
            "profile_from_start": True,
            "launch_skip": 0,
            "launch_count": 10,
        },
        "collection": {
            "set": "default",
            "sections": [],
            "target_processes": "all",
            "replay_mode": None,
        },
        "workload": {
            "argv": ["python", "train.py"],
            "cwd": str(root),
            "env": {},
        },
        "output": {
            "prefix": str(root / "H003-default"),
            "force_overwrite": False,
        },
    }


class NcuProfileTest(unittest.TestCase):
    def test_validate_accepts_bounded_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                ncu_profile.validate_spec(capture_spec(Path(directory))), []
            )

    def test_validate_rejects_full_and_unbounded_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = capture_spec(Path(directory))
            spec["selection"].update(
                {
                    "kernel_regex": None,
                    "nvtx_include": None,
                    "profile_from_start": True,
                    "launch_count": 11,
                }
            )
            spec["collection"]["set"] = "full"
            errors = ncu_profile.validate_spec(spec)
            self.assertTrue(any("not bounded" in error for error in errors))
            self.assertTrue(any("prohibited" in error for error in errors))

    def test_command_preserves_argv_and_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = capture_spec(Path(directory))
            command = ncu_profile.build_ncu_command("/opt/ncu", spec)
            self.assertEqual(command[0], "/opt/ncu")
            self.assertIn("--kernel-name", command)
            self.assertIn("--nvtx-include", command)
            self.assertEqual(command[-2:], ["python", "train.py"])

    def test_fake_ncu_capture_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_ncu = root / "ncu"
            fake_ncu.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if '--version' in args:\n"
                "    print('NVIDIA Nsight Compute CLI 2099.1')\n"
                "elif '--list-sections' in args:\n"
                "    print('SpeedOfLight')\n"
                "elif '--import' in args:\n"
                "    print('\"Kernel Name\",\"Metric Name\",\"Metric Value\"')\n"
                "    print('\"radius_search\",\"dram__throughput\",\"72.5\"')\n"
                "else:\n"
                "    prefix = pathlib.Path(args[args.index('-o') + 1])\n"
                "    pathlib.Path(str(prefix) + '.ncu-rep').write_bytes(b'report')\n"
            )
            fake_ncu.chmod(0o755)
            spec_path = root / "capture-spec.json"
            spec_path.write_text(json.dumps(capture_spec(root)))

            parser = ncu_profile.build_parser()
            approval_path = root / "capture-approval.json"
            approve_args = parser.parse_args(
                [
                    "approve",
                    str(spec_path),
                    "--output",
                    str(approval_path),
                    "--confirmation-source",
                    "user confirmed exact print-only plan",
                ]
            )
            self.assertEqual(ncu_profile.cmd_approve(approve_args), 0)
            capture_args = parser.parse_args(
                [
                    "capture",
                    str(spec_path),
                    "--ncu",
                    str(fake_ncu),
                    "--manifest",
                    str(root / "capture-manifest.json"),
                    "--approval",
                    str(approval_path),
                ]
            )
            self.assertEqual(ncu_profile.cmd_capture(capture_args), 0)
            self.assertTrue((root / "H003-default.ncu-rep").is_file())

            summary_args = parser.parse_args(
                [
                    "summarize",
                    str(root / "H003-default.ncu-rep"),
                    "--output",
                    str(root / "H003-summary.json"),
                    "--ncu",
                    str(fake_ncu),
                ]
            )
            self.assertEqual(ncu_profile.cmd_summarize(summary_args), 0)
            summary = json.loads((root / "H003-summary.json").read_text())
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["rows"][0]["Kernel Name"], "radius_search")

    def test_capture_requires_approval_bound_to_current_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "capture-spec.json"
            spec_path.write_text(json.dumps(capture_spec(root)))
            approval_path = root / "capture-approval.json"
            with self.assertRaisesRegex(ncu_profile.UserError, "approval is missing"):
                ncu_profile.require_capture_approval(spec_path, approval_path)
            approval_path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.1",
                        "status": "approved",
                        "spec_sha256": ncu_profile.sha256_file(spec_path),
                        "confirmation_source": "user confirmation",
                    }
                )
            )
            ncu_profile.require_capture_approval(spec_path, approval_path)
            spec = capture_spec(root)
            spec["selection"]["launch_count"] = 2
            spec_path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ncu_profile.UserError, "changed after approval"):
                ncu_profile.require_capture_approval(spec_path, approval_path)

    def test_permission_failure_is_classified(self) -> None:
        self.assertEqual(
            ncu_profile.classify_failure("ERR_NVGPUCTRPERM", ""),
            "gpu_performance_counter_permission",
        )


if __name__ == "__main__":
    unittest.main()
