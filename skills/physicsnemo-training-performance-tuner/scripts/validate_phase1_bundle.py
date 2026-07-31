#!/usr/bin/env python3
"""Validate a phase-1 training-performance artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_JSON = (
    "run-manifest.json",
    "test-config.json",
    "config-confirmation.json",
    "baseline.json",
    "correctness.json",
    "annotation-health-eager.json",
    "annotation-health-compiled.json",
    "compile-analysis.json",
    "compile-comparison.json",
    "findings.json",
    "phase-source-map.json",
    "source-analysis.json",
)
REQUIRED_DIRS = ("logs", "traces", "hta", "ncu")
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
RECOMMENDATION_KEYS = (
    "id",
    "code_observation",
    "mechanism",
    "suggested_improvement",
    "recommended_skill",
    "phase2_experiment",
    "correctness_check",
    "performance_check",
    "semantic_risk",
    "confidence",
    "status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: cannot read valid JSON: {exc}")
        return {}


def require_keys(
    value: Any, keys: tuple[str, ...], label: str, errors: list[str]
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label}: expected a JSON object")
        return
    for key in keys:
        if key not in value:
            errors.append(f"{label}: missing key {key!r}")


def validate_source_location(
    source: Any, label: str, errors: list[str], require_role: bool = False
) -> None:
    keys = ("path", "start_line", "end_line", "symbol")
    if require_role:
        keys += ("role",)
    require_keys(source, keys, label, errors)
    if not isinstance(source, dict):
        return
    if not isinstance(source.get("path"), str) or not source.get("path"):
        errors.append(f"{label}: path must be a non-empty string")
    elif Path(source["path"]).is_absolute() or ".." in Path(source["path"]).parts:
        errors.append(f"{label}: path must be repository-relative")
    start = source.get("start_line")
    end = source.get("end_line")
    if not isinstance(start, int) or isinstance(start, bool) or start < 1:
        errors.append(f"{label}: start_line must be a positive integer")
    if not isinstance(end, int) or isinstance(end, bool) or end < 1:
        errors.append(f"{label}: end_line must be a positive integer")
    if isinstance(start, int) and isinstance(end, int) and end < start:
        errors.append(f"{label}: end_line must be >= start_line")


def require_phase_coverage(
    entries: Any, label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        errors.append(f"{label}: expected an array")
        return {}
    by_phase: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        item_label = f"{label}[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{item_label}: expected an object")
            continue
        phase = entry.get("phase")
        if phase not in CANONICAL_PHASES:
            errors.append(f"{item_label}: unknown phase {phase!r}")
            continue
        if phase in by_phase:
            errors.append(f"{item_label}: duplicate phase {phase!r}")
            continue
        by_phase[phase] = entry
    missing = sorted(set(CANONICAL_PHASES) - set(by_phase))
    if missing:
        errors.append(f"{label}: missing canonical phases: {', '.join(missing)}")
    return by_phase


def validate_annotation_health(
    value: Any, label: str, errors: list[str], *, ready: bool
) -> None:
    require_keys(
        value,
        (
            "schema_version",
            "phase",
            "run_id",
            "variant",
            "status",
            "trace",
            "step_boundary",
            "annotations",
            "timestamps",
            "kernel_attribution",
            "recapture",
            "notes",
        ),
        label,
        errors,
    )
    if not ready or not isinstance(value, dict):
        return
    if value.get("status") not in ("pass", "reconstructed"):
        errors.append(f"{label}: status must be 'pass' or 'reconstructed' with --ready")
    provenance = nested(value, "step_boundary", "provenance")
    if provenance not in (
        "native_profiler_step",
        "explicit_train_step",
        "reconstructed",
    ):
        errors.append(f"{label}: step boundary provenance is not usable")
    if nested(value, "recapture", "required") is not False:
        errors.append(f"{label}: resolve the required recapture before --ready")
    missing = nested(value, "annotations", "missing_required_phases")
    if not isinstance(missing, list) or missing:
        errors.append(f"{label}: missing_required_phases must be empty with --ready")
    if provenance == "reconstructed":
        if not nested(value, "step_boundary", "selection_rule"):
            errors.append(f"{label}: reconstructed boundaries need a selection_rule")
        if nested(value, "kernel_attribution", "method") != "index_correlation":
            errors.append(
                f"{label}: reconstructed boundaries must use index_correlation"
            )
        if nested(value, "kernel_attribution", "fallback_used") is not True:
            errors.append(f"{label}: reconstructed boundaries need fallback_used=true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--confirmed",
        action="store_true",
        help="Require explicit confirmation of the current test-config fingerprint.",
    )
    parser.add_argument(
        "--ready",
        action="store_true",
        help="Require completed baseline, correctness, findings, traces, and report.",
    )
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    errors: list[str] = []

    if not bundle.is_dir():
        print(f"error: bundle directory does not exist: {bundle}", file=sys.stderr)
        return 2

    for name in REQUIRED_JSON:
        if not (bundle / name).is_file():
            errors.append(f"missing required file: {name}")
    if not (bundle / "report.md").is_file():
        errors.append("missing required file: report.md")
    if not (bundle / "test-config.md").is_file():
        errors.append("missing required file: test-config.md")
    for name in REQUIRED_DIRS:
        if not (bundle / name).is_dir():
            errors.append(f"missing required directory: {name}/")
    for variant in ("eager", "compiled"):
        for dirname in ("logs", "traces", "hta"):
            if not (bundle / dirname / variant).is_dir():
                errors.append(f"missing required directory: {dirname}/{variant}/")
        if not (bundle / "hta" / variant / "diagrams").is_dir():
            errors.append(f"missing required directory: hta/{variant}/diagrams/")

    documents = {
        name: load_json(bundle / name, errors)
        for name in REQUIRED_JSON
        if (bundle / name).is_file()
    }
    manifest = documents.get("run-manifest.json", {})
    test_config = documents.get("test-config.json", {})
    confirmation = documents.get("config-confirmation.json", {})
    baseline = documents.get("baseline.json", {})
    correctness = documents.get("correctness.json", {})
    annotation_health = {
        "eager": documents.get("annotation-health-eager.json", {}),
        "compiled": documents.get("annotation-health-compiled.json", {}),
    }
    compile_analysis = documents.get("compile-analysis.json", {})
    compile_comparison = documents.get("compile-comparison.json", {})
    findings = documents.get("findings.json", {})
    phase_source_map = documents.get("phase-source-map.json", {})
    source_analysis = documents.get("source-analysis.json", {})

    require_keys(
        manifest,
        (
            "schema_version",
            "phase",
            "run_id",
            "workdir",
            "command",
            "commands",
            "comparison",
            "dataset",
            "protocol",
            "correctness_command",
            "performance_goal",
        ),
        "run-manifest.json",
        errors,
    )
    require_keys(
        test_config,
        (
            "schema_version",
            "phase",
            "run_id",
            "source",
            "workload",
            "data",
            "runtime",
            "comparison",
            "protocol",
            "correctness",
            "performance",
            "artifacts",
        ),
        "test-config.json",
        errors,
    )
    require_keys(
        confirmation,
        (
            "schema_version",
            "phase",
            "run_id",
            "status",
            "confirmed_config_sha256",
            "confirmed_at_utc",
            "confirmation_source",
        ),
        "config-confirmation.json",
        errors,
    )
    require_keys(
        baseline,
        (
            "schema_version",
            "phase",
            "run_id",
            "status",
            "timing_scope",
            "variants",
            "comparison",
        ),
        "baseline.json",
        errors,
    )
    require_keys(
        correctness,
        (
            "schema_version",
            "phase",
            "run_id",
            "status",
            "signal",
            "observations",
            "variants",
            "comparison",
        ),
        "correctness.json",
        errors,
    )
    for variant, document in annotation_health.items():
        validate_annotation_health(
            document,
            f"annotation-health-{variant}.json",
            errors,
            ready=False,
        )
        if isinstance(document, dict) and document.get("variant") != variant:
            errors.append(
                f"annotation-health-{variant}.json: variant must equal {variant!r}"
            )
    require_keys(
        compile_analysis,
        (
            "schema_version",
            "phase",
            "run_id",
            "status",
            "timing_scope",
            "logs",
            "graph_breaks",
            "recompilations",
            "cache_limit_warnings",
            "backend_failures",
            "summary",
            "notes",
        ),
        "compile-analysis.json",
        errors,
    )
    require_keys(
        compile_comparison,
        (
            "schema_version",
            "phase",
            "run_id",
            "status",
            "classification",
            "correctness_status",
            "steady_state_speedup",
            "compile_time_ms",
            "amortization_steps",
            "timing_basis",
            "trace_deltas",
            "conclusion",
            "evidence",
        ),
        "compile-comparison.json",
        errors,
    )
    require_keys(
        findings,
        ("schema_version", "phase", "run_id", "summary", "hotspots"),
        "findings.json",
        errors,
    )
    require_keys(
        phase_source_map,
        ("schema_version", "phase", "run_id", "status", "mappings"),
        "phase-source-map.json",
        errors,
    )
    require_keys(
        source_analysis,
        ("schema_version", "phase", "run_id", "status", "reviews"),
        "source-analysis.json",
        errors,
    )

    for name, document in documents.items():
        if document.get("phase") != 1:
            errors.append(f"{name}: phase must equal 1")
        if manifest and document.get("run_id") != manifest.get("run_id"):
            errors.append(f"{name}: run_id does not match run-manifest.json")

    test_config_path = bundle / "test-config.json"
    current_config_sha256 = (
        sha256_file(test_config_path) if test_config_path.is_file() else None
    )
    manifest_config_sha256 = nested(manifest, "test_config", "sha256")
    if current_config_sha256 and manifest_config_sha256 != current_config_sha256:
        errors.append(
            "run-manifest.json: test_config.sha256 does not match test-config.json"
        )

    resolved_artifact = nested(
        test_config, "workload", "resolved_config_artifact"
    )
    resolved_sha256 = nested(test_config, "workload", "resolved_config_sha256")
    if resolved_artifact:
        resolved_path = bundle / resolved_artifact
        if not resolved_path.is_file():
            errors.append(
                f"test-config.json: missing resolved config artifact {resolved_artifact!r}"
            )
        elif sha256_file(resolved_path) != resolved_sha256:
            errors.append(
                "test-config.json: resolved config fingerprint does not match artifact"
            )

    if args.confirmed or args.ready:
        if confirmation.get("status") != "confirmed":
            errors.append(
                "config-confirmation.json: status must be 'confirmed' before execution"
            )
        if confirmation.get("confirmed_config_sha256") != current_config_sha256:
            errors.append(
                "config-confirmation.json: confirmed fingerprint does not match "
                "test-config.json; show the changed config and reconfirm"
            )
        for key in ("confirmed_at_utc", "confirmation_source"):
            if not isinstance(confirmation.get(key), str) or not confirmation.get(key):
                errors.append(f"config-confirmation.json: {key} must be non-empty")
        required_config_values = (
            ("workload", "workdir"),
            ("workload", "entry_point"),
            ("workload", "command"),
            ("workload", "model_config"),
            ("comparison", "variants", "eager", "command"),
            ("comparison", "variants", "compiled", "command"),
            ("comparison", "variants", "compiled", "backend"),
            ("comparison", "variants", "compiled", "mode"),
            ("comparison", "variants", "compiled", "fullgraph"),
            ("comparison", "variants", "compiled", "dynamic"),
            ("data", "dataset"),
            ("data", "batch_size"),
            ("runtime", "hardware"),
            ("runtime", "framework_stack"),
            ("runtime", "precision"),
            ("runtime", "distributed"),
            ("correctness", "signal"),
            ("performance", "goal"),
            ("artifacts", "output"),
        )
        for path in required_config_values:
            value = nested(test_config, *path)
            if value in (None, "", []):
                errors.append(
                    f"test-config.json: {'.'.join(path)} must be resolved before confirmation"
                )

    eager_variant = nested(test_config, "comparison", "variants", "eager")
    compiled_variant = nested(test_config, "comparison", "variants", "compiled")
    if not isinstance(eager_variant, dict) or eager_variant.get("compile") is not False:
        errors.append("test-config.json: eager variant must set compile=false")
    if not isinstance(compiled_variant, dict) or compiled_variant.get("compile") is not True:
        errors.append("test-config.json: compiled variant must set compile=true")
    if nested(test_config, "comparison", "design") != "paired_eager_compiled":
        errors.append("test-config.json: comparison.design must be paired_eager_compiled")

    hotspots = findings.get("hotspots", [])
    if not isinstance(hotspots, list):
        errors.append("findings.json: hotspots must be a list")
        hotspots = []
    for index, hotspot in enumerate(hotspots):
        label = f"findings.json hotspot[{index}]"
        require_keys(
            hotspot,
            (
                "id",
                "category",
                "severity",
                "confidence",
                "summary",
                "evidence",
                "recommended_skill",
                "proposed_experiment",
                "source_analysis_ids",
                "status",
            ),
            label,
            errors,
        )
        if hotspot.get("status") != "recommendation_only":
            errors.append(f"{label}: status must be 'recommendation_only'")
        if not isinstance(hotspot.get("evidence"), list):
            errors.append(f"{label}: evidence must be a list")
        if not isinstance(hotspot.get("source_analysis_ids"), list):
            errors.append(f"{label}: source_analysis_ids must be a list")
        elif not all(
            isinstance(item, str) and item for item in hotspot["source_analysis_ids"]
        ):
            errors.append(
                f"{label}: source_analysis_ids must contain non-empty strings"
            )

    mappings = require_phase_coverage(
        phase_source_map.get("mappings"), "phase-source-map.json mappings", errors
    )
    for phase, mapping in mappings.items():
        label = f"phase-source-map.json {phase}"
        require_keys(
            mapping,
            (
                "phase",
                "coverage",
                "ranges",
                "sources",
                "config",
                "evidence",
                "reason",
            ),
            label,
            errors,
        )
        if mapping.get("coverage") not in ("unmapped", "mapped", "not_applicable"):
            errors.append(f"{label}: invalid coverage {mapping.get('coverage')!r}")
        for key in ("ranges", "sources", "config", "evidence"):
            if not isinstance(mapping.get(key), list):
                errors.append(f"{label}: {key} must be a list")
        for key in ("ranges", "config", "evidence"):
            if isinstance(mapping.get(key), list) and not all(
                isinstance(item, str) and item for item in mapping[key]
            ):
                errors.append(f"{label}: {key} must contain non-empty strings")
        if isinstance(mapping.get("sources"), list):
            for index, source in enumerate(mapping["sources"]):
                validate_source_location(
                    source, f"{label} source[{index}]", errors, require_role=True
                )

    reviews = require_phase_coverage(
        source_analysis.get("reviews"), "source-analysis.json reviews", errors
    )
    recommendation_ids: set[str] = set()
    reviewed_hotspot_ids: set[str] = set()
    for phase, review in reviews.items():
        label = f"source-analysis.json {phase}"
        require_keys(
            review,
            (
                "phase",
                "review_status",
                "hotspot_ids",
                "measured_evidence",
                "sources",
                "code_observations",
                "recommendations",
                "no_change_reason",
            ),
            label,
            errors,
        )
        if review.get("review_status") not in (
            "pending",
            "reviewed",
            "not_applicable",
        ):
            errors.append(
                f"{label}: invalid review_status {review.get('review_status')!r}"
            )
        for key in (
            "hotspot_ids",
            "measured_evidence",
            "sources",
            "code_observations",
            "recommendations",
        ):
            if not isinstance(review.get(key), list):
                errors.append(f"{label}: {key} must be a list")
        for key in ("hotspot_ids", "measured_evidence", "code_observations"):
            if isinstance(review.get(key), list) and not all(
                isinstance(item, str) and item for item in review[key]
            ):
                errors.append(f"{label}: {key} must contain non-empty strings")
        if isinstance(review.get("hotspot_ids"), list):
            reviewed_hotspot_ids.update(
                item for item in review["hotspot_ids"] if isinstance(item, str)
            )
        if isinstance(review.get("sources"), list):
            for index, source in enumerate(review["sources"]):
                validate_source_location(
                    source, f"{label} source[{index}]", errors
                )
        if isinstance(review.get("recommendations"), list):
            for index, recommendation in enumerate(review["recommendations"]):
                rec_label = f"{label} recommendation[{index}]"
                require_keys(
                    recommendation, RECOMMENDATION_KEYS, rec_label, errors
                )
                if not isinstance(recommendation, dict):
                    continue
                rec_id = recommendation.get("id")
                if not isinstance(rec_id, str) or not rec_id:
                    errors.append(f"{rec_label}: id must be a non-empty string")
                elif rec_id in recommendation_ids:
                    errors.append(f"{rec_label}: duplicate id {rec_id!r}")
                else:
                    recommendation_ids.add(rec_id)
                if recommendation.get("status") != "recommendation_only":
                    errors.append(
                        f"{rec_label}: status must be 'recommendation_only'"
                    )
                for key in RECOMMENDATION_KEYS:
                    if key == "status":
                        continue
                    if not isinstance(recommendation.get(key), str) or not (
                        recommendation.get(key)
                    ):
                        errors.append(
                            f"{rec_label}: {key} must be a non-empty string"
                        )

    for index, hotspot in enumerate(hotspots):
        if not isinstance(hotspot, dict):
            continue
        label = f"findings.json hotspot[{index}]"
        source_ids = hotspot.get("source_analysis_ids", [])
        if isinstance(source_ids, list):
            for source_id in source_ids:
                if source_id not in recommendation_ids:
                    errors.append(
                        f"{label}: unknown source_analysis_id {source_id!r}"
                    )

    report_path = bundle / "report.md"
    report_text = report_path.read_text() if report_path.is_file() else ""
    serialized_findings = json.dumps(findings).lower()
    if "measured speedup" in serialized_findings or "post_change" in findings:
        errors.append("findings.json: phase 1 must not contain optimization results")

    if args.ready:
        if baseline.get("status") != "complete":
            errors.append("baseline.json: status must be 'complete' with --ready")
        if baseline.get("timing_scope") != "unprofiled_steady_state":
            errors.append(
                "baseline.json: timing_scope must be unprofiled_steady_state"
            )
        baseline_variants = baseline.get("variants", {})
        for variant in ("eager", "compiled"):
            result = baseline_variants.get(variant, {}) if isinstance(baseline_variants, dict) else {}
            if result.get("status") != "complete":
                errors.append(f"baseline.json: {variant} status must be complete")
            if not result.get("repetitions"):
                errors.append(f"baseline.json: {variant} repetitions cannot be empty")
        cold_start = nested(baseline, "variants", "compiled", "cold_start")
        if not isinstance(cold_start, dict) or cold_start.get("status") != "complete":
            errors.append("baseline.json: compiled cold_start must be complete")
        elif not isinstance(cold_start.get("compilation_time_ms"), (int, float)):
            errors.append(
                "baseline.json: compiled cold_start.compilation_time_ms must be numeric"
            )
        if nested(baseline, "comparison", "timing_basis") != "unprofiled_steady_state":
            errors.append(
                "baseline.json: comparison must use unprofiled_steady_state timing"
            )
        if correctness.get("status") != "pass":
            errors.append("correctness.json: status must be 'pass' with --ready")
        for variant in ("eager", "compiled"):
            if nested(correctness, "variants", variant, "status") != "pass":
                errors.append(f"correctness.json: {variant} status must be pass")
        if nested(correctness, "comparison", "status") != "pass":
            errors.append("correctness.json: paired comparison status must be pass")
        for variant, document in annotation_health.items():
            validate_annotation_health(
                document,
                f"annotation-health-{variant}.json",
                errors,
                ready=True,
            )
        if compile_analysis.get("status") != "complete":
            errors.append("compile-analysis.json: status must be complete with --ready")
        if compile_analysis.get("timing_scope") != "diagnostic_not_benchmark":
            errors.append(
                "compile-analysis.json: diagnostic timing must not be benchmark timing"
            )
        if not compile_analysis.get("logs"):
            errors.append("compile-analysis.json: at least one diagnostic log is required")
        if compile_comparison.get("status") != "complete":
            errors.append("compile-comparison.json: status must be complete with --ready")
        if compile_comparison.get("classification") not in (
            "beneficial",
            "regression",
            "neutral",
            "inconclusive",
        ):
            errors.append("compile-comparison.json: invalid classification")
        if compile_comparison.get("correctness_status") != "pass":
            errors.append("compile-comparison.json: correctness_status must be pass")
        if compile_comparison.get("timing_basis") != "unprofiled_steady_state":
            errors.append(
                "compile-comparison.json: speedup must use unprofiled steady-state timing"
            )
        if not isinstance(
            compile_comparison.get("steady_state_speedup"), (int, float)
        ):
            errors.append(
                "compile-comparison.json: steady_state_speedup must be numeric"
            )
        if not isinstance(compile_comparison.get("compile_time_ms"), (int, float)):
            errors.append("compile-comparison.json: compile_time_ms must be numeric")
        if not compile_comparison.get("conclusion"):
            errors.append("compile-comparison.json: conclusion is required")
        if not compile_comparison.get("evidence"):
            errors.append("compile-comparison.json: evidence is required")
        if not hotspots:
            errors.append("findings.json: at least one hotspot is required with --ready")
        if "{{" in report_text or "}}" in report_text:
            errors.append("report.md: unresolved template markers remain with --ready")
        for variant in ("eager", "compiled"):
            traces_dir = bundle / "traces" / variant
            hta_dir = bundle / "hta" / variant
            if traces_dir.is_dir() and not any(traces_dir.iterdir()):
                errors.append(f"traces/{variant}/: at least one trace is required")
            diagram_data = hta_dir / "diagram-data.json"
            diagram_manifest = hta_dir / "diagrams" / "diagram-manifest.json"
            required_diagrams = (
                hta_dir / "diagrams" / "cpu-gpu-pipeline.svg",
                hta_dir / "diagrams" / "forward-dominant-kernels.svg",
            )
            if not diagram_data.is_file():
                errors.append(f"hta/{variant}/diagram-data.json is required")
            if not diagram_manifest.is_file():
                errors.append(
                    f"hta/{variant}/diagrams/diagram-manifest.json is required"
                )
            for diagram in required_diagrams:
                if not diagram.is_file():
                    errors.append(
                        f"hta/{variant}/diagrams/{diagram.name} is required"
                    )
        if phase_source_map.get("status") != "complete":
            errors.append(
                "phase-source-map.json: status must be 'complete' with --ready"
            )
        if source_analysis.get("status") != "complete":
            errors.append(
                "source-analysis.json: status must be 'complete' with --ready"
            )
        for phase in CANONICAL_PHASES:
            mapping = mappings.get(phase, {})
            review = reviews.get(phase, {})
            coverage = mapping.get("coverage")
            review_status = review.get("review_status")
            if coverage not in ("mapped", "not_applicable"):
                errors.append(
                    f"phase-source-map.json {phase}: coverage must be mapped or "
                    "not_applicable with --ready"
                )
            if coverage == "mapped":
                if not mapping.get("sources"):
                    errors.append(
                        f"phase-source-map.json {phase}: mapped phase needs sources"
                    )
                if not mapping.get("evidence"):
                    errors.append(
                        f"phase-source-map.json {phase}: mapped phase needs evidence"
                    )
                if review_status != "reviewed":
                    errors.append(
                        f"source-analysis.json {phase}: mapped phase must be reviewed"
                    )
                if not review.get("sources"):
                    errors.append(
                        f"source-analysis.json {phase}: reviewed phase needs sources"
                    )
                if not review.get("measured_evidence"):
                    errors.append(
                        f"source-analysis.json {phase}: reviewed phase needs "
                        "measured_evidence"
                    )
                if not review.get("code_observations"):
                    errors.append(
                        f"source-analysis.json {phase}: reviewed phase needs "
                        "code_observations"
                    )
                if not review.get("recommendations") and not review.get(
                    "no_change_reason"
                ):
                    errors.append(
                        f"source-analysis.json {phase}: provide recommendations "
                        "or no_change_reason"
                    )
            if coverage == "not_applicable":
                if not mapping.get("reason"):
                    errors.append(
                        f"phase-source-map.json {phase}: not_applicable needs reason"
                    )
                if review_status != "not_applicable":
                    errors.append(
                        f"source-analysis.json {phase}: expected not_applicable review"
                    )
        for index, hotspot in enumerate(hotspots):
            if not isinstance(hotspot, dict):
                continue
            label = f"findings.json hotspot[{index}]"
            source_ids = hotspot.get("source_analysis_ids", [])
            if not source_ids:
                errors.append(
                    f"{label}: source_analysis_ids cannot be empty with --ready"
                )
            hotspot_id = hotspot.get("id")
            if hotspot_id not in reviewed_hotspot_ids:
                errors.append(
                    f"{label}: hotspot id is not linked from source-analysis.json"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    mode = "ready" if args.ready else "confirmed" if args.confirmed else "structural"
    print(f"OK: phase-1 bundle passed {mode} validation: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
