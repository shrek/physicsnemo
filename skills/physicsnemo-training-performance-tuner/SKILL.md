---
name: physicsnemo-training-performance-tuner
description: Profile custom PyTorch and PhysicsNeMo training loops, establish reproducible baselines, analyze Kineto traces with HolisticTraceAnalysis, correlate bottlenecks with phase-specific source code, drill into selected CUDA kernels with a bounded Nsight Compute workflow, and produce evidence-backed PhysicsNeMo optimization recommendations. Use for slow training, low GPU utilization, dataloader stalls, expensive feature or geometry construction, excessive memory movement, custom-kernel bottlenecks, or poor distributed scaling. This initial version diagnoses and routes work but does not apply optimizations.
---

# PhysicsNeMo Training Performance Tuner

## Overview

Measure a training loop before recommending changes. Produce a phase-1 report
covering the baseline, correctness, trace evidence, kernel evidence when needed,
ranked hotspots, and downstream PhysicsNeMo skill routing.

Do not apply optimizations in this version. End with testable hypotheses and
`recommendation_only` findings.

## Required inputs

Collect or infer:

- Training entry point, complete launch command, and working directory.
- Representative config, dataset or sample, and batch/sample size.
- CPU, GPU type/count, and interconnect.
- PyTorch, PhysicsNeMo, precision, compile, and distributed settings.
- Warmup, measurement, profiling, and repetition budgets.
- An executable correctness check or explicit invariant.
- A primary goal such as step time, throughput, memory, utilization, or scaling.

Ask one targeted question when the command, representative data, or correctness
check is missing. Do not start profiling until all three are known.

## Phase-1 workflow

### 1. Prepare the run bundle

Read `references/phase1-protocol.md`. For the repository golden case, also read
`references/geotransolver-volume.md`.

Create a fresh artifact directory with:

```bash
python scripts/create_run_bundle.py \
  --output <artifact-dir> \
  --workdir <training-workdir> \
  --command '<complete launch command>' \
  --dataset '<dataset path, manifest, or resolved dataset identifier>' \
  --correctness-command '<command or explicit invariant>' \
  --performance-goal '<goal>'
```

Fill the generated manifest with resolved hardware and framework details before
running the benchmark. Never record secrets, authentication tokens, or the
complete process environment.

### 2. Establish the baseline

Run a short smoke test first. Then run an unprofiled benchmark with fixed seed,
config, batch size, sampling, precision, compile mode, GPU count, warmup, and
measured steps. Use at least three repetitions unless the user explicitly limits
the budget.

Record distributions, not only a single timing:

- Step time and throughput.
- Peak allocated and reserved GPU memory.
- CPU time/utilization and data-wait time when observable.
- Per-rank results for distributed jobs.
- The correctness result.

Do not use profiled timings as the baseline. Do not compare a cold baseline with
a warm candidate.

### 3. Add minimal instrumentation

Prefer existing training-loop profiler support. Otherwise add opt-in
`torch.profiler.record_function` or NVTX ranges around:

`dataloader_wait`, `feature_construction`, `host_to_device`, `forward`, `loss`,
`backward`, `optimizer_step`, `distributed_sync`, `validation`, and
`checkpoint`.

Keep instrumentation reversible and disabled by default. Capture a bounded
Kineto trace after warmup. Start with five steady-state active iterations.
Collect expensive stack or shape data only in a secondary capture when required.

### 4. Analyze with HolisticTraceAnalysis

Use HolisticTraceAnalysis directly as the only Python analysis dependency. Do
not require Scope or an MCP server. Inspect the installed HTA version before
using APIs because supported analyses and output schemas can vary.

For each representative steady-state trace:

1. Inspect trace/rank coverage and profiler-step boundaries.
2. Compute temporal and idle-time breakdowns.
3. Compute GPU kernel breakdown and CUDA launch statistics.
4. For distributed traces, inspect communication/compute overlap and stragglers.
5. Derive the critical path for a representative steady-state step.
6. Normalize the representative step and forward-kernel spans into the schema in
   `references/diagram-schema.md`.
7. Generate the required CPU/GPU pipeline and forward dominant-kernel diagrams
   with `scripts/render_trace_diagrams.py`.
8. Generate in-step and multi-step bubble diagrams when the trace supports them.
9. Correlate ranges and kernels with the instrumented source.

Save normalized JSON/CSV tables and visualization paths under `hta/`. Record
the HTA version, trace inputs, analysis command, and assumptions.

The CPU/GPU pipeline and forward dominant-kernel diagrams are required report
artifacts. Diagram timings must come from HTA evidence, not visual estimation.

### 5. Inspect the code behind every phase

Read `references/source-review-protocol.md`. Map every canonical training phase
to its implementing ranges, symbols, files, configuration, and measured
evidence in `phase-source-map.json`. Explicitly mark genuinely absent phases
`not_applicable`; do not silently omit them.

Review each mapped phase with depth proportional to its critical-path impact.
Write `source-analysis.json` containing:

- Exact repository-relative paths, symbols, and line bounds.
- Measured trace or NCU evidence.
- Code observations separated from performance-mechanism explanations.
- Code-specific improvements only when the evidence supports them.
- One isolated phase-2 experiment per recommendation.
- Correctness, performance, semantic-risk, and confidence fields.
- A `no_change_reason` when no change is justified.

Link every hotspot in `findings.json` to recommendation IDs through
`source_analysis_ids`. Keep every recommendation `recommendation_only`. Do not
edit the reviewed code during phase 1.

### 6. Use NCU conditionally

Read `references/ncu-profiling.md` before running NCU. Run Nsight Compute only
after HTA identifies a CUDA kernel on the critical
path whose limiting mechanism affects the recommendation.

- Record `ncu_needed`, a precise `ncu_question`, and the kernel/NVTX selector in
  the HTA hotspot.
- Run `scripts/ncu_profile.py doctor`.
- Create a bounded plan with `scripts/ncu_profile.py plan` and inspect it with
  `capture --print-only` before execution.
- Start with at most ten launches and NCU's `default` set.
- Collect deeper focused sections only for a surviving kernel candidate.
- Preserve the spec, manifest, log, `.ncu-rep`, CSV, and normalized JSON summary.
- Treat NCU replay timings as diagnostic evidence, never benchmark speed.

Skip NCU when the bottleneck is already explained by dataloading, host wait,
communication, or launch overhead. State why it was skipped.

Do not install NCU, enable GPU performance counters, use `sudo`, or change
container capabilities without explicit user approval.

### 7. Classify and route

Classify each hotspot using the routing table in
`references/phase1-protocol.md`. Include:

- Severity and confidence.
- Metric, range, kernel, trace, and source evidence.
- Estimated critical-path contribution when defensible.
- Recommended downstream skill.
- One isolated experiment for phase 2.
- Status exactly `recommendation_only`.

Do not invoke downstream optimization skills during phase 1.

### 8. Write and validate the report

Use `assets/report-template.md` as the report skeleton. Follow the narrative
style of the GeoTransolver volume reference: measurements first, diagrams second,
interpretation third, and hypotheses last.

Distinguish measured values, estimates, and theoretical upper bounds. Do not
claim a speedup because phase 1 has no optimized rerun.

Validate the bundle:

```bash
python scripts/validate_phase1_bundle.py <artifact-dir>
```

Use `--ready` only for a completed report. Resolve every reported error before
delivery.

## Completion criteria

Finish phase 1 only when:

- The smoke test and unprofiled baseline are reproducible.
- Baseline variability and correctness are reported.
- HTA analyzed at least one representative trace.
- NCU produced evidence or the report explains why it was unnecessary.
- Every conclusion points to an artifact, metric, range, kernel, or source line.
- Every canonical phase is source-mapped or explicitly not applicable.
- Every hotspot has a code observation, supported mechanism, source location,
  isolated experiment, correctness check, and performance check.
- The report contains a CPU/GPU lane diagram and a forward dominant-kernel
  diagram generated from normalized HTA evidence.
- Findings are ranked and remain `recommendation_only`.
- Residual bottlenecks and next experiments are explicit.

## Guardrails

- Do not optimize before measuring.
- Do not alter physics fields, units, targets, loss, or metrics.
- Do not hide correctness failures behind performance results.
- Do not generalize from tiny synthetic inputs unless the user requested only a
  smoke benchmark.
- Do not modify unrelated user changes.
- Do not report projected savings as measured speedups.
- Do not use NCU or profiler-instrumented wall time for performance comparison.
- Do not recommend a code change from source inspection alone; require measured
  evidence linking it to the bottleneck.
- Do not present a candidate improvement from the phase checklist as a finding
  unless the inspected implementation and trace support it.

## Resources

- `references/phase1-protocol.md` — benchmark, evidence, routing, and reporting
  contract.
- `references/geotransolver-volume.md` — golden workflow and evaluation
  expectations.
- `references/diagram-schema.md` — normalized HTA diagram input and output
  contract.
- `references/source-review-protocol.md` — phase mapping, code inspection,
  recommendation, and source-analysis schema.
- `references/ncu-profiling.md` — bounded NCU planning, capture, and failure
  handling.
- `references/ncu-output-schema.md` — NCU artifact contract.
- `assets/report-template.md` — phase-1 Markdown report skeleton.
- `scripts/create_run_bundle.py` — initialize deterministic phase-1 artifacts.
- `scripts/render_trace_diagrams.py` — render report-ready SVG timelines using
  only the Python standard library.
- `scripts/ncu_profile.py` — check, plan, capture, summarize, and validate NCU.
- `scripts/validate_phase1_bundle.py` — validate structure and phase boundary.
