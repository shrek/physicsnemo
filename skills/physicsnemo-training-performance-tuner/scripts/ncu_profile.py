#!/usr/bin/env python3
"""Plan, run, summarize, and validate bounded Nsight Compute captures."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
DEFAULT_LAUNCH_COUNT = 10
MAX_LAUNCH_COUNT = 100
PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "permission to access gpu performance counters")


class UserError(ValueError):
    """An actionable error in a user-provided capture plan."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UserError(f"cannot read valid JSON from {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_ncu(explicit: str | None) -> str | None:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        found = shutil.which(explicit)
        return found
    found = shutil.which("ncu")
    if found:
        return found
    fallback = Path("/usr/local/cuda/bin/ncu")
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    return None


def run_command(
    argv: list[str], *, cwd: str | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def command_result(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        result = run_command(argv, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def select_finding(document: Any, candidate_id: str | None) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise UserError("HTA finding must be a JSON object")
    candidates = document.get("hotspots")
    if isinstance(candidates, list):
        if candidate_id:
            matches = [
                item
                for item in candidates
                if isinstance(item, dict) and item.get("id") == candidate_id
            ]
            if len(matches) != 1:
                raise UserError(f"expected one HTA hotspot with id {candidate_id!r}")
            return matches[0]
        ncu_candidates = [
            item
            for item in candidates
            if isinstance(item, dict) and item.get("ncu_needed") is True
        ]
        if len(ncu_candidates) != 1:
            raise UserError(
                "select --candidate-id when the HTA document does not contain "
                "exactly one hotspot with ncu_needed=true"
            )
        return ncu_candidates[0]
    return document


def load_workload(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if isinstance(value, list):
        value = {"argv": value}
    if not isinstance(value, dict):
        raise UserError("workload JSON must be an argv array or an object")
    argv = value.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise UserError("workload argv must be a non-empty array of strings")
    cwd = value.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise UserError("workload cwd must be a string or null")
    env = value.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in env.items()
    ):
        raise UserError("workload env must map strings to strings")
    return {"argv": argv, "cwd": cwd, "env": env}


def validate_spec(spec: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["capture spec must be a JSON object"]
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION!r}")
    selection = spec.get("selection")
    collection = spec.get("collection")
    workload = spec.get("workload")
    output = spec.get("output")
    if not isinstance(selection, dict):
        errors.append("selection must be an object")
        selection = {}
    if not isinstance(collection, dict):
        errors.append("collection must be an object")
        collection = {}
    if not isinstance(workload, dict):
        errors.append("workload must be an object")
        workload = {}
    if not isinstance(output, dict):
        errors.append("output must be an object")
        output = {}

    count = selection.get("launch_count")
    skip = selection.get("launch_skip")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        errors.append("selection.launch_count must be a positive integer")
    elif count > MAX_LAUNCH_COUNT:
        errors.append(
            f"selection.launch_count must not exceed {MAX_LAUNCH_COUNT}; "
            "split the capture instead"
        )
    if not isinstance(skip, int) or isinstance(skip, bool) or skip < 0:
        errors.append("selection.launch_skip must be a non-negative integer")
    profile_from_start = selection.get("profile_from_start", True)
    if not isinstance(profile_from_start, bool):
        errors.append("selection.profile_from_start must be a boolean")
    selectors = (
        bool(selection.get("kernel_regex")),
        bool(selection.get("nvtx_include")),
        profile_from_start is False,
        isinstance(count, int)
        and not isinstance(count, bool)
        and 1 <= count <= DEFAULT_LAUNCH_COUNT,
    )
    if not any(selectors):
        errors.append(
            "capture is not bounded: use a kernel regex, NVTX include, profiler "
            f"gate, or launch_count <= {DEFAULT_LAUNCH_COUNT}"
        )

    set_name = collection.get("set", "default")
    if not isinstance(set_name, str) or not set_name:
        errors.append("collection.set must be a non-empty string")
    elif set_name.lower() == "full":
        errors.append("collection.set='full' is prohibited for training captures")
    sections = collection.get("sections", [])
    if not isinstance(sections, list) or not all(
        isinstance(item, str) and item for item in sections
    ):
        errors.append("collection.sections must be an array of non-empty strings")
    elif len(sections) > 8:
        errors.append("collection.sections must contain at most 8 focused sections")

    argv = workload.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(item, str) and item for item in argv
    ):
        errors.append("workload.argv must be a non-empty array of strings")
    env = workload.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in env.items()
    ):
        errors.append("workload.env must map strings to strings")
    prefix = output.get("prefix")
    if not isinstance(prefix, str) or not prefix:
        errors.append("output.prefix must be a non-empty string")
    elif prefix.endswith(".ncu-rep"):
        errors.append("output.prefix must omit the .ncu-rep suffix")
    return errors


def build_ncu_command(ncu: str, spec: dict[str, Any]) -> list[str]:
    errors = validate_spec(spec)
    if errors:
        raise UserError("; ".join(errors))
    selection = spec["selection"]
    collection = spec["collection"]
    output = spec["output"]
    command = [ncu, "--target-processes", collection.get("target_processes", "all")]
    sections = collection.get("sections", [])
    if sections:
        for section in sections:
            command.extend(["--section", section])
    else:
        command.extend(["--set", collection.get("set", "default")])
    command.extend(["--launch-skip", str(selection["launch_skip"])])
    command.extend(["--launch-count", str(selection["launch_count"])])
    if selection.get("kernel_regex"):
        command.extend(["--kernel-name", selection["kernel_regex"]])
    if selection.get("nvtx_include"):
        command.extend(["--nvtx", "--nvtx-include", selection["nvtx_include"]])
    if selection.get("profile_from_start") is False:
        command.extend(["--profile-from-start", "off"])
    if collection.get("replay_mode"):
        command.extend(["--replay-mode", collection["replay_mode"]])
    if output.get("force_overwrite") is True:
        command.append("--force-overwrite")
    command.extend(["-o", output["prefix"]])
    command.extend(spec["workload"]["argv"])
    return command


def classify_failure(stderr: str, stdout: str) -> str | None:
    message = f"{stderr}\n{stdout}".lower()
    if any(marker.lower() in message for marker in PERMISSION_MARKERS):
        return "gpu_performance_counter_permission"
    if "no kernels were profiled" in message:
        return "no_matching_kernel"
    if "failed to connect" in message or "target application returned" in message:
        return "target_application_failure"
    return None


def cmd_doctor(args: argparse.Namespace) -> int:
    ncu = resolve_ncu(args.ncu)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": utc_now(),
        "ncu": {"path": ncu, "available": ncu is not None},
        "gpu": {"nvidia_smi": shutil.which("nvidia-smi")},
        "status": "unavailable",
    }
    if ncu:
        version = command_result([ncu, "--version"])
        sections = command_result([ncu, "--list-sections"])
        document["ncu"].update({"version": version, "sections": sections})
        document["status"] = "ready" if version["returncode"] == 0 else "error"
    smi = document["gpu"]["nvidia_smi"]
    if smi:
        document["gpu"]["inventory"] = command_result(
            [
                smi,
                "--query-gpu=index,name,driver_version",
                "--format=csv,noheader",
            ]
        )
    if args.output:
        write_json(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["status"] == "ready" else 1


def cmd_plan(args: argparse.Namespace) -> int:
    finding: dict[str, Any] = {}
    if args.hta_finding:
        finding = select_finding(read_json(args.hta_finding), args.candidate_id)
    kernel_regex = args.kernel_regex or finding.get("kernel_regex")
    nvtx_include = args.nvtx_include or finding.get("nvtx_range")
    candidate_id = args.candidate_id or finding.get("id")
    reason = args.reason or finding.get("summary") or ""
    question = args.question or finding.get("ncu_question") or ""
    workload = load_workload(args.workload_json)
    spec = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "candidate": {
            "id": candidate_id,
            "reason": reason,
            "question": question,
            "source": str(args.hta_finding) if args.hta_finding else None,
        },
        "selection": {
            "kernel_regex": kernel_regex,
            "nvtx_include": nvtx_include,
            "profile_from_start": args.profile_from_start == "on",
            "launch_skip": args.launch_skip,
            "launch_count": args.launch_count,
        },
        "collection": {
            "set": args.set_name,
            "sections": args.section,
            "target_processes": args.target_processes,
            "replay_mode": args.replay_mode,
        },
        "workload": workload,
        "output": {
            "prefix": str(args.output_prefix.expanduser().resolve()),
            "force_overwrite": args.force_overwrite,
        },
    }
    errors = validate_spec(spec)
    if errors:
        raise UserError("; ".join(errors))
    write_json(args.spec, spec)
    print(args.spec)
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    spec = read_json(args.spec)
    errors = validate_spec(spec)
    if errors:
        raise UserError("; ".join(errors))
    ncu = resolve_ncu(args.ncu)
    if not ncu:
        raise UserError(
            "ncu was not found; install Nsight Compute or pass --ncu /path/to/ncu"
        )
    command = build_ncu_command(ncu, spec)
    manifest_path = args.manifest or args.spec.with_name("capture-manifest.json")
    report_path = Path(spec["output"]["prefix"] + ".ncu-rep")
    if report_path.exists() and not spec["output"].get("force_overwrite"):
        raise UserError(
            f"report already exists: {report_path}; choose a new prefix or explicitly "
            "plan with --force-overwrite"
        )
    if args.print_only:
        print(json.dumps({"argv": command, "cwd": spec["workload"].get("cwd")}, indent=2))
        return 0

    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    environment = os.environ.copy()
    environment.update(spec["workload"].get("env", {}))
    try:
        result = subprocess.run(
            command,
            cwd=spec["workload"].get("cwd"),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        failure = classify_failure(result.stderr, result.stdout)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "spec": str(args.spec),
            "argv": command,
            "cwd": spec["workload"].get("cwd"),
            "returncode": result.returncode,
            "failure_class": failure,
            "report": str(report_path),
            "report_exists": report_path.is_file(),
        }
        log_path = manifest_path.with_name("capture.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "STDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr
        )
        manifest["log"] = str(log_path)
        write_json(manifest_path, manifest)
        print(manifest_path)
        return 0 if result.returncode == 0 and report_path.is_file() else 1
    except subprocess.TimeoutExpired as exc:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "spec": str(args.spec),
            "argv": command,
            "cwd": spec["workload"].get("cwd"),
            "returncode": None,
            "failure_class": "timeout",
            "timeout_seconds": args.timeout,
            "report": str(report_path),
            "report_exists": report_path.is_file(),
            "stderr": str(exc),
        }
        write_json(manifest_path, manifest)
        print(manifest_path)
        return 1


def parse_ncu_csv(raw: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = [
        line
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("==")
    ]
    if not lines:
        return [], []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    if reader.fieldnames is None:
        return [], []
    return reader.fieldnames, [dict(row) for row in reader]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_summarize(args: argparse.Namespace) -> int:
    report = args.report.expanduser().resolve()
    if not report.is_file():
        raise UserError(f"NCU report does not exist: {report}")
    ncu = resolve_ncu(args.ncu)
    if not ncu:
        raise UserError(
            "ncu was not found; the same or a compatible Nsight Compute CLI is "
            "required to import the report"
        )
    command = [ncu, "--import", str(report), "--csv", "--page", args.page]
    result = run_command(command, timeout=args.timeout)
    if result.returncode != 0:
        raise UserError(f"ncu import failed: {result.stderr.strip()}")
    columns, rows = parse_ncu_csv(result.stdout)
    raw_csv = args.output.with_suffix(".csv")
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_csv.write_text(result.stdout)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "report": str(report),
        "report_sha256": sha256_file(report),
        "import_argv": command,
        "page": args.page,
        "columns": columns,
        "row_count": len(rows),
        "rows": rows,
        "raw_csv": str(raw_csv),
        "interpretation": {
            "limiting_mechanism": None,
            "evidence": [],
            "note": "Populate from focused NCU metrics; do not use replay time as speedup.",
        },
    }
    write_json(args.output, summary)
    print(args.output)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    if args.spec:
        spec = read_json(args.spec)
        errors.extend(f"spec: {item}" for item in validate_spec(spec))
    if args.report and not args.report.is_file():
        errors.append(f"report does not exist: {args.report}")
    if args.summary:
        summary = read_json(args.summary)
        if not isinstance(summary, dict):
            errors.append("summary must be a JSON object")
        else:
            for key in ("schema_version", "report", "report_sha256", "rows"):
                if key not in summary:
                    errors.append(f"summary is missing {key!r}")
            if summary.get("schema_version") != SCHEMA_VERSION:
                errors.append(
                    f"summary schema_version must equal {SCHEMA_VERSION!r}"
                )
    if not any((args.spec, args.report, args.summary)):
        errors.append("provide at least one of --spec, --report, or --summary")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: NCU artifacts passed validation")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check NCU and GPU visibility.")
    doctor.add_argument("--ncu")
    doctor.add_argument("--output", type=Path)
    doctor.set_defaults(func=cmd_doctor)

    plan = subparsers.add_parser("plan", help="Create a bounded capture spec.")
    plan.add_argument("--workload-json", required=True, type=Path)
    plan.add_argument("--spec", required=True, type=Path)
    plan.add_argument("--output-prefix", required=True, type=Path)
    plan.add_argument("--hta-finding", type=Path)
    plan.add_argument("--candidate-id")
    plan.add_argument("--reason")
    plan.add_argument("--question")
    plan.add_argument("--kernel-regex")
    plan.add_argument("--nvtx-include")
    plan.add_argument("--profile-from-start", choices=("on", "off"), default="on")
    plan.add_argument("--launch-skip", type=int, default=0)
    plan.add_argument("--launch-count", type=int, default=DEFAULT_LAUNCH_COUNT)
    plan.add_argument("--set", dest="set_name", default="default")
    plan.add_argument("--section", action="append", default=[])
    plan.add_argument("--target-processes", default="all")
    plan.add_argument("--replay-mode")
    plan.add_argument("--force-overwrite", action="store_true")
    plan.set_defaults(func=cmd_plan)

    capture = subparsers.add_parser("capture", help="Run a capture from a spec.")
    capture.add_argument("spec", type=Path)
    capture.add_argument("--ncu")
    capture.add_argument("--manifest", type=Path)
    capture.add_argument("--timeout", type=int, default=1800)
    capture.add_argument("--print-only", action="store_true")
    capture.set_defaults(func=cmd_capture)

    summarize = subparsers.add_parser(
        "summarize", help="Import an .ncu-rep file into JSON and CSV."
    )
    summarize.add_argument("report", type=Path)
    summarize.add_argument("--output", required=True, type=Path)
    summarize.add_argument("--ncu")
    summarize.add_argument("--page", default="raw")
    summarize.add_argument("--timeout", type=int, default=300)
    summarize.set_defaults(func=cmd_summarize)

    validate = subparsers.add_parser("validate", help="Validate NCU artifacts.")
    validate.add_argument("--spec", type=Path)
    validate.add_argument("--report", type=Path)
    validate.add_argument("--summary", type=Path)
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
