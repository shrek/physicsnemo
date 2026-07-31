#!/usr/bin/env python3
"""Create a deterministic phase-1 performance-analysis artifact bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
    parser.add_argument(
        "--command",
        help="Legacy alias for --eager-command; paired runs still require --compiled-command.",
    )
    parser.add_argument("--eager-command")
    parser.add_argument("--compiled-command", required=True)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", default="false")
    parser.add_argument("--compile-dynamic", default="default")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--correctness-command", required=True)
    parser.add_argument("--performance-goal", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--workload", default="PyTorch training loop")
    parser.add_argument("--entry-point", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--batch-size", default="")
    parser.add_argument("--resolved-config", type=Path)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display(value: Any) -> str:
    if value in (None, "", []):
        return "UNKNOWN — resolve before confirmation"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_test_config(config: dict[str, Any]) -> str:
    eager_command = str(config["comparison"]["variants"]["eager"]["command"]).replace("`", "'")
    compiled_command = str(config["comparison"]["variants"]["compiled"]["command"]).replace("`", "'")
    rows = (
        ("Working directory", config["workload"]["workdir"]),
        ("Eager command (`torch.compile=false`)", f"`{eager_command}`"),
        ("Compiled command (`torch.compile=true`)", f"`{compiled_command}`"),
        ("Compile backend", config["comparison"]["variants"]["compiled"]["backend"]),
        ("Compile mode", config["comparison"]["variants"]["compiled"]["mode"]),
        ("Compile fullgraph", config["comparison"]["variants"]["compiled"]["fullgraph"]),
        ("Compile dynamic", config["comparison"]["variants"]["compiled"]["dynamic"]),
        ("Source revision", config["source"]["git_commit"]),
        ("Local patch state", "dirty" if config["source"]["dirty"] else "clean"),
        ("Entry point", config["workload"]["entry_point"]),
        ("Model/config", config["workload"]["model_config"]),
        ("Dataset/sample", config["data"]["dataset"]),
        ("Batch/sample size", config["data"]["batch_size"]),
        ("Hardware", config["runtime"]["hardware"]),
        ("Framework stack", config["runtime"]["framework_stack"]),
        ("Precision", config["runtime"]["precision"]),
        ("Distributed", config["runtime"]["distributed"]),
        ("Warmup steps", config["protocol"]["warmup_steps"]),
        ("Measured steps", config["protocol"]["measure_steps"]),
        ("Profiled steps", config["protocol"]["profile_steps"]),
        ("Repetitions", config["protocol"]["repetitions"]),
        ("Correctness signal", config["correctness"]["signal"]),
        ("Performance goal", config["performance"]["goal"]),
        ("Artifact output", config["artifacts"]["output"]),
        ("Resolved config", config["workload"]["resolved_config_artifact"]),
    )
    body = "\n".join(f"| {label} | {display(value)} |" for label, value in rows)
    return (
        "# Test configuration — confirmation required\n\n"
        "| Item | Value |\n|---|---|\n"
        f"{body}\n"
    )


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
    resolved_config = (
        args.resolved_config.expanduser().resolve() if args.resolved_config else None
    )
    eager_command = args.eager_command or args.command

    if not workdir.is_dir():
        print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
        return 2
    if resolved_config is not None and not resolved_config.is_file():
        print(
            f"error: resolved config does not exist: {resolved_config}",
            file=sys.stderr,
        )
        return 2
    required_values = {
        "eager-command": eager_command,
        "compiled-command": args.compiled_command,
        "dataset": args.dataset,
        "correctness-command": args.correctness_command,
        "performance-goal": args.performance_goal,
    }
    for name, value in required_values.items():
        if not value or not value.strip():
            print(f"error: --{name} cannot be empty", file=sys.stderr)
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

    resolved_config_artifact = None
    resolved_config_sha256 = None
    if resolved_config is not None:
        suffix = resolved_config.suffix or ".txt"
        resolved_config_artifact = f"resolved-config{suffix}"
        destination = output / resolved_config_artifact
        shutil.copyfile(resolved_config, destination)
        resolved_config_sha256 = sha256_file(destination)

    test_config = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "source": {
            "git_commit": git_commit,
            "dirty": bool(git_status),
            "status_short": git_status.splitlines() if git_status else [],
        },
        "workload": {
            "name": args.workload,
            "workdir": str(workdir),
            "entry_point": args.entry_point,
            "command": eager_command,
            "model_config": args.model_config,
            "resolved_config_artifact": resolved_config_artifact,
            "resolved_config_sha256": resolved_config_sha256,
        },
        "data": {"dataset": args.dataset, "batch_size": args.batch_size},
        "runtime": {
            "hardware": args.hardware,
            "framework_stack": args.framework_stack,
            "precision": args.precision,
            "distributed": args.distributed,
        },
        "comparison": {
            "design": "paired_eager_compiled",
            "only_intended_difference": "torch.compile state and declared compile options",
            "variants": {
                "eager": {
                    "compile": False,
                    "command": eager_command,
                    "artifact_prefix": "eager",
                },
                "compiled": {
                    "compile": True,
                    "command": args.compiled_command,
                    "backend": args.compile_backend,
                    "mode": args.compile_mode,
                    "fullgraph": args.compile_fullgraph,
                    "dynamic": args.compile_dynamic,
                    "artifact_prefix": "compiled",
                },
            },
            "cold_start_separate": True,
            "steady_state_speedup_source": "unprofiled_baseline_only",
        },
        "protocol": {
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "profile_steps": args.profile_steps,
            "repetitions": args.repetitions,
        },
        "correctness": {"signal": args.correctness_command},
        "performance": {"goal": args.performance_goal},
        "artifacts": {"output": str(output)},
    }
    test_config_path = output / "test-config.json"
    write_json(test_config_path, test_config)
    test_config_sha256 = sha256_file(test_config_path)
    confirmation = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "confirmed_config_sha256": None,
        "confirmed_at_utc": None,
        "confirmation_source": None,
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "workload": args.workload,
        "workdir": str(workdir),
        "entry_point": args.entry_point,
        "command": eager_command,
        "commands": {"eager": eager_command, "compiled": args.compiled_command},
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
        "comparison": {
            "design": "paired_eager_compiled",
            "compile_settings": {
                "backend": args.compile_backend,
                "mode": args.compile_mode,
                "fullgraph": args.compile_fullgraph,
                "dynamic": args.compile_dynamic,
            },
            "cold_start_separate": True,
            "steady_state_speedup_source": "unprofiled_baseline_only",
        },
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
        "test_config": {
            "artifact": "test-config.json",
            "sha256": test_config_sha256,
        },
        "confirmation": "config-confirmation.json",
        "notes": [
            "Resolve blank hardware/framework fields before confirmation.",
            "Show test-config.md to the user and obtain explicit confirmation.",
            "Do not execute the workload while config-confirmation.json is pending.",
            "Do not add secrets or dump the complete process environment.",
        ],
    }

    empty_aggregate = {
        "step_time_ms": {},
        "throughput": {},
        "peak_gpu_allocated_mib": None,
        "peak_gpu_reserved_mib": None,
        "data_wait_ms": {},
    }
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "timing_scope": "unprofiled_steady_state",
        "variants": {
            "eager": {
                "status": "pending",
                "repetitions": [],
                "aggregate": dict(empty_aggregate),
            },
            "compiled": {
                "status": "pending",
                "cold_start": {
                    "status": "pending",
                    "compilation_time_ms": None,
                    "first_step_time_ms": None,
                    "cache_state": None,
                },
                "repetitions": [],
                "aggregate": dict(empty_aggregate),
            },
        },
        "comparison": {
            "status": "pending",
            "steady_state_speedup": None,
            "classification": "pending",
            "amortization_steps": None,
            "timing_basis": "unprofiled_steady_state",
        },
        "repetitions": [],
        "aggregate": dict(empty_aggregate),
    }
    correctness = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "signal": args.correctness_command,
        "observations": [],
        "variants": {
            "eager": {"status": "pending", "observations": []},
            "compiled": {"status": "pending", "observations": []},
        },
        "comparison": {"status": "pending", "tolerances": {}, "observations": []},
    }
    def annotation_health(variant: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": 1,
            "run_id": run_id,
            "variant": variant,
            "status": "pending",
            "trace": None,
            "step_boundary": {
                "provenance": "pending",
                "marker": None,
                "count": 0,
                "selection_rule": None,
            },
            "annotations": {
                "required_phases": [],
                "phase_counts": {},
                "missing_required_phases": [],
                "original_annotation_count": 0,
                "duplicate_projected_annotations": False,
                "projected_annotation_count": 0,
                "compile_record_function_warning": False,
            },
            "timestamps": {
                "duration_invalid_count": 0,
                "end_before_start_count": 0,
                "end_field_inconsistent_count": 0,
                "derived_end_required": False,
                "end_rule": "Use valid HTA end only when consistent; otherwise ts + dur.",
            },
            "kernel_attribution": {"method": "pending", "fallback_used": False},
            "recapture": {"required": False, "reason": None},
            "notes": [],
        }

    annotation_health_eager = annotation_health("eager")
    annotation_health_compiled = annotation_health("compiled")
    compile_analysis = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "timing_scope": "diagnostic_not_benchmark",
        "logs": [],
        "graph_breaks": [],
        "recompilations": [],
        "cache_limit_warnings": [],
        "backend_failures": [],
        "summary": {},
        "notes": [],
    }
    compile_comparison = {
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "run_id": run_id,
        "status": "pending",
        "classification": "pending",
        "correctness_status": "pending",
        "steady_state_speedup": None,
        "compile_time_ms": None,
        "amortization_steps": None,
        "timing_basis": "unprofiled_steady_state",
        "trace_deltas": {},
        "conclusion": None,
        "evidence": [],
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
    for variant in ("eager", "compiled"):
        (output / "logs" / variant).mkdir()
        (output / "traces" / variant).mkdir()
        (output / "hta" / variant).mkdir()
        (output / "hta" / variant / "diagrams").mkdir()
    write_json(output / "run-manifest.json", manifest)
    write_json(output / "config-confirmation.json", confirmation)
    (output / "test-config.md").write_text(render_test_config(test_config))
    write_json(output / "baseline.json", baseline)
    write_json(output / "correctness.json", correctness)
    write_json(output / "annotation-health-eager.json", annotation_health_eager)
    write_json(output / "annotation-health-compiled.json", annotation_health_compiled)
    write_json(output / "compile-analysis.json", compile_analysis)
    write_json(output / "compile-comparison.json", compile_comparison)
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
        "{{COMMAND}}": eager_command.replace("`", "'"),
        "{{EAGER_COMMAND}}": eager_command.replace("`", "'"),
        "{{COMPILED_COMMAND}}": args.compiled_command.replace("`", "'"),
        "{{COMPILE_BACKEND}}": args.compile_backend,
        "{{COMPILE_MODE}}": args.compile_mode,
        "{{COMPILE_FULLGRAPH}}": args.compile_fullgraph,
        "{{COMPILE_DYNAMIC}}": args.compile_dynamic,
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
