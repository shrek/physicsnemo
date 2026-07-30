#!/usr/bin/env python3
"""Create a deterministic phase-1 performance-analysis artifact bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
CANONICAL_PHASES = (
    "dataloader_wait",
    "feature_construction",
    "host_to_device",
    "forward",
    "loss",
    "backward",
    "optimizer_step",
    "distributed_sync",
    "validation",
    "checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--command", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--correctness-command", required=True)
    parser.add_argument("--performance-goal", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--workload", default="PyTorch training loop")
    parser.add_argument("--entry-point", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--batch-size", default="")
    parser.add_argument("--hardware", default="")
    parser.add_argument("--framework-stack", default="")
    parser.add_argument("--precision", default="")
    parser.add_argument("--distributed", default="single-process")
    parser.add_argument("--warmup-steps", type=positive_int, default=5)
    parser.add_argument("--measure-steps", type=positive_int, default=20)
    parser.add_argument("--profile-steps", type=positive_int, default=5)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    return parser.parse_args()


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def run_git(workdir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ensure_new_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not workdir.is_dir():
        print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
        return 2
    for name in ("command", "dataset", "correctness_command", "performance_goal"):
        if not getattr(args, name).strip():
            print(f"error: --{name.replace('_', '-')} cannot be empty", file=sys.stderr)
            return 2

    try:
        ensure_new_output(output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    now = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or now.strftime("perf_%Y%m%dT%H%M%SZ")
    git_commit = run_git(workdir, "rev-parse", "HEAD")
    git_status = run_git(workdir, "status", "--short")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "workload": args.workload,
        "workdir": str(workdir),
        "entry_point": args.entry_point,
        "command": args.command,
        "dataset": args.dataset,
        "model_config": args.model_config,
        "batch_size": args.batch_size,
        "hardware": args.hardware,
        "framework_stack": args.framework_stack,
        "precision": args.precision,
        "distributed": args.distributed,
        "protocol": {
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "profile_steps": args.profile_steps,
            "repetitions": args.repetitions,
        },
        "correctness_command": args.correctness_command,
        "performance_goal": args.performance_goal,
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status),
            "status_short": git_status.splitlines() if git_status else [],
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "notes": [
            "Resolve blank hardware/framework fields before benchmarking.",
            "Do not add secrets or dump the complete process environment.",
        ],
    }

    baseline = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "timing_scope": "unprofiled",
        "repetitions": [],
        "aggregate": {
            "step_time_ms": {},
            "throughput": {},
            "peak_gpu_allocated_mib": None,
            "peak_gpu_reserved_mib": None,
            "data_wait_ms": {},
        },
    }
    correctness = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "signal": args.correctness_command,
        "observations": [],
    }
    findings = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "summary": "",
        "hotspots": [],
        "residual_bottlenecks": [],
    }
    phase_source_map = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "mappings": [
            {
                "phase": phase,
                "coverage": "unmapped",
                "ranges": [],
                "sources": [],
                "config": [],
                "evidence": [],
                "reason": None,
            }
            for phase in CANONICAL_PHASES
        ],
    }
    source_analysis = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "reviews": [
            {
                "phase": phase,
                "review_status": "pending",
                "hotspot_ids": [],
                "measured_evidence": [],
                "sources": [],
                "code_observations": [],
                "recommendations": [],
                "no_change_reason": None,
            }
            for phase in CANONICAL_PHASES
        ],
    }

    for dirname in ("logs", "traces", "hta", "ncu"):
        (output / dirname).mkdir()
    (output / "hta" / "diagrams").mkdir()
    write_json(output / "run-manifest.json", manifest)
    write_json(output / "baseline.json", baseline)
    write_json(output / "correctness.json", correctness)
    write_json(output / "findings.json", findings)
    write_json(output / "phase-source-map.json", phase_source_map)
    write_json(output / "source-analysis.json", source_analysis)

    template = Path(__file__).resolve().parent.parent / "assets" / "report-template.md"
    report = output / "report.md"
    shutil.copyfile(template, report)
    replacements = {
        "{{WORKLOAD}}": args.workload,
        "{{RUN_ID}}": run_id,
        "{{DATE}}": now.date().isoformat(),
        "{{GIT_COMMIT}}": git_commit or "not-a-git-worktree",
        "{{ENTRY_POINT}}": args.entry_point or "TBD",
        "{{COMMAND}}": args.command.replace("`", "'"),
        "{{MODEL_CONFIG}}": args.model_config or "TBD",
        "{{DATASET}}": args.dataset,
        "{{BATCH_SIZE}}": args.batch_size or "TBD",
        "{{PRECISION_COMPILE}}": args.precision or "TBD",
        "{{HARDWARE}}": args.hardware or "TBD",
        "{{DISTRIBUTED}}": args.distributed,
        "{{WARMUP_STEPS}}": str(args.warmup_steps),
        "{{MEASURE_STEPS}}": str(args.measure_steps),
        "{{REPETITIONS}}": str(args.repetitions),
        "{{PROFILE_STEPS}}": str(args.profile_steps),
        "{{CORRECTNESS_SIGNAL}}": args.correctness_command,
        "{{PERFORMANCE_GOAL}}": args.performance_goal,
    }
    report_text = report.read_text()
    for marker, value in replacements.items():
        report_text = report_text.replace(marker, value)
    report.write_text(report_text)

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
