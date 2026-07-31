---
name: physicsnemo-training-performance-tuner
description: Profile custom PyTorch and PhysicsNeMo training loops, establish reproducible baselines, analyze Kineto traces with HolisticTraceAnalysis, compare eager and torch.compile steady-state performance, inventory graph breaks and recompilations, correlate bottlenecks with phase-specific source code, drill into selected CUDA kernels with a bounded Nsight Compute workflow, and produce evidence-backed PhysicsNeMo optimization recommendations. Use for slow training, compile regressions, low GPU utilization, dataloader stalls, expensive feature or geometry construction, excessive memory movement, custom-kernel bottlenecks, or poor distributed scaling. This initial version diagnoses and routes work but does not apply optimizations.
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
- PyTorch, PhysicsNeMo, precision, distributed settings, and the compiled
  backend/mode/fullgraph/dynamic-shape settings.
- Warmup, measurement, profiling, and repetition budgets.
- An executable correctness check or explicit invariant.
- A primary goal such as step time, throughput, memory, utilization, or scaling.

Ask one targeted question when the command, representative data, or correctness
check is missing. Do not start workload execution until all three are known and
the user has confirmed the resolved test configuration.

## Phase-1 workflow

### 1. Prepare the run bundle

Read `references/phase1-protocol.md`. For the repository golden case, also read
`references/geotransolver-volume.md`.

Create a fresh artifact directory with:

```bash
python scripts/create_run_bundle.py \
  --output <artifact-dir> \
  --workdir <training-workdir> \
  --eager-command '<complete command with torch.compile=false>' \
  --compiled-command '<complete command with torch.compile=true>' \
  --dataset '<dataset path, manifest, or resolved dataset identifier>' \
  --correctness-command '<command or explicit invariant>' \
  --performance-goal '<goal>'
```

Fill the generated manifest with resolved hardware and framework details before
running the benchmark. Pass `--resolved-config <path>` when the workload has a
resolved application configuration. Never record secrets, authentication
tokens, or the complete process environment.

### 2. Show and confirm the test configuration

Before any smoke test, baseline, profiler, or NCU execution:

1. Resolve defaults and overrides as the workload will see them.
2. Show the user the generated `test-config.md` confirmation block. Include the
   complete eager and compiled launch commands, compile backend/mode/fullgraph/
   dynamic settings, working directory, source revision/local patch
   state, model/config, dataset and sample selection, batch/sample size, seed,
   precision, compile mode, GPU/distributed settings, dataloader settings,
   warmup/measurement/profile/repetition budgets, correctness check,
   performance goal, and artifact output.
3. Link the full resolved configuration snapshot when one exists. Do not dump
   secrets or an unfiltered process environment.
4. Ask the user to explicitly confirm this configuration.

Treat confirmation as a hard execution gate. Read-only discovery and config
resolution are allowed before confirmation; training, smoke tests, profiling,
and NCU are not. After explicit confirmation, record it with:

```bash
python scripts/confirm_run_config.py <artifact-dir>
python scripts/validate_phase1_bundle.py <artifact-dir> --confirmed
```

If any confirmed field or resolved-config content changes, regenerate the
confirmation block, show it again, and obtain a new confirmation. Never infer
confirmation from an earlier plan or from the user's request to profile.

### 3. Establish paired eager and compiled baselines

Read `references/compile-comparison.md`. Run correctness and a short smoke test
for both confirmed variants. Then run unprofiled eager and compiled benchmarks
with source, seed, config, data/sample order, batch size, sampling, precision,
GPU topology, warmup, and measured steps held invariant. Use at least three
steady-state repetitions per variant unless the user explicitly limits the
budget.

Measure compiled cold-start/compilation cost separately. Warm compilation and
intended recompilations before compiled steady-state timing. Do not mix cold
compiled steps into steady-state aggregates. Record distributions, not only a
single timing:

- Step time and throughput.
- Peak allocated and reserved GPU memory.
- CPU time/utilization and data-wait time when observable.
- Per-rank results for distributed jobs.
- The correctness result.

Do not use profiled timings as the baseline. Derive compile speedup only from
unprofiled steady-state results. Report compile time, first-step time, cache
state, and amortization separately.

### 4. Add minimal instrumentation

Prefer existing training-loop profiler support. Otherwise add opt-in
`torch.profiler.record_function` or NVTX ranges around:

`dataloader_wait`, `feature_construction`, `host_to_device`, `forward`, `loss`,
`backward`, `optimizer_step`, `distributed_sync`, `validation`, and
`checkpoint`.

Keep instrumentation reversible and disabled by default. Capture the same bounded
logical steps once with `torch.compile=false` and once with `torch.compile=true`.
Configure an explicit profiler schedule that emits native `ProfilerStep#N` ranges; if the wrapper cannot
guarantee that, add one outer `train_step` range per logical iteration. Capture a
bounded Kineto trace after warmup. Start with five steady-state active iterations.
Collect expensive stack or shape data only in a secondary capture when required.

Read `references/annotation-health.md` and run
`scripts/validate_trace_annotations.py` separately for eager and compiled traces,
writing `annotation-health-eager.json` and `annotation-health-compiled.json`,
before HTA. Treat missing required phases,
invalid durations, or an untrustworthy step boundary as a recapture condition.
When `torch.compile` reports that a nested `record_function` was ignored, verify
the required outer ranges in the raw trace. Use the documented reconstruction
fallback only when recapture is impractical, and label its provenance explicitly.

### 5. Diagnose torch.compile behavior

Run a separate bounded compiled diagnostic with graph-break and recompilation
logging as defined in `references/compile-comparison.md`. Normalize it with
`scripts/analyze_compile_logs.py` into `compile-analysis.json`. This diagnostic
run is not benchmark evidence.

List unique graph breaks with source/reason/count, recompilations with failed
guards, cache-limit warnings, backend failures, and eager fallbacks. Use
`torch._dynamo.explain` or `TORCH_TRACE`/`tlparse` only when appropriate and
available. Diagnose and route; do not rewrite code to remove graph breaks in
phase 1.

### 6. Analyze with HolisticTraceAnalysis

Use HolisticTraceAnalysis directly as the only Python analysis dependency. Do
not require Scope or an MCP server. Inspect the installed HTA version before
using APIs because supported analyses and output schemas can vary.

For each eager and compiled representative steady-state trace:

1. Inspect trace/rank coverage and the validated logical-step boundary, including
   whether its provenance is native, explicit, or reconstructed.
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

### 7. Inspect the code behind every phase

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

### 8. Use NCU conditionally

Read `references/ncu-profiling.md` before running NCU. Run Nsight Compute only
after HTA identifies a CUDA kernel on the critical
path whose limiting mechanism affects the recommendation.

- Record `ncu_needed`, a precise `ncu_question`, and the kernel/NVTX selector in
  the HTA hotspot.
- Run `scripts/ncu_profile.py doctor`; treat it as discovery, not proof that an
  actual capture is operationally permitted.
- Create a bounded plan with `scripts/ncu_profile.py plan`, show the exact
  `capture --print-only` output to the user, and obtain separate explicit
  confirmation for that capture.
- Bind confirmation to the capture-spec fingerprint with
  `scripts/ncu_profile.py approve`. Actual capture must reject a missing or stale
  approval. If `ncu` is unavailable, record that fact and skip the capture.
- Start with at most ten launches and NCU's `default` set.
- Collect deeper focused sections only for a surviving kernel candidate.
- Preserve the spec, manifest, log, `.ncu-rep`, CSV, and normalized JSON summary.
- Treat NCU replay timings as diagnostic evidence, never benchmark speed.

Skip NCU when the bottleneck is already explained by dataloading, host wait,
communication, or launch overhead. State why it was skipped.

Do not install NCU, enable GPU performance counters, use `sudo`, or change
container capabilities without explicit user approval.

### 9. Classify and route

Classify each hotspot using the routing table in
`references/phase1-protocol.md`. Include:

- Severity and confidence.
- Metric, range, kernel, trace, and source evidence.
- Estimated critical-path contribution when defensible.
- Recommended downstream skill.
- One isolated experiment for phase 2.
- Status exactly `recommendation_only`.

Do not invoke downstream optimization skills during phase 1.

### 10. Write and validate the report

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

- The executed configuration has explicit user confirmation and its fingerprint
  still matches `test-config.json`.

- Eager and compiled smoke tests and unprofiled steady-state baselines are
  reproducible with only declared compile settings different.
- Correctness passes for each variant and for their declared comparison.
- Baseline variability, cold compilation cost, cache state, steady-state delta,
  and amortization are reported.
- `annotation-health-eager.json` and `annotation-health-compiled.json` record
  usable logical-step boundaries before HTA analysis.
- `compile-analysis.json` lists graph breaks, recompilations, cache-limit warnings,
  and backend fallback evidence from a non-benchmark diagnostic run.
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
- Do not execute the workload before the user confirms the displayed test
  configuration.
- Do not reuse confirmation after a test-config fingerprint changes.
- Do not alter physics fields, units, targets, loss, or metrics.
- Do not hide correctness failures behind performance results.
- Do not generalize from tiny synthetic inputs unless the user requested only a
  smoke benchmark.
- Do not modify unrelated user changes.
- Do not report projected savings as measured speedups.
- Do not use NCU, compiler-log, or profiler-instrumented wall time for performance
  comparison.
- Do not compare eager steady state with compiled cold-start or compilation time.
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
- `references/annotation-health.md` — trace-boundary validation and bounded
  reconstruction contract.
- `references/compile-comparison.md` — paired eager/compiled baseline, diagnostic,
  trace, and routing contract.
- `references/source-review-protocol.md` — phase mapping, code inspection,
  recommendation, and source-analysis schema.
- `references/ncu-profiling.md` — bounded NCU planning, capture, and failure
  handling.
- `references/ncu-output-schema.md` — NCU artifact contract.
- `assets/report-template.md` — phase-1 Markdown report skeleton.
- `scripts/create_run_bundle.py` — initialize deterministic phase-1 artifacts.
- `scripts/confirm_run_config.py` — bind explicit user confirmation to the
  current test-config fingerprint.
- `scripts/validate_trace_annotations.py` — validate phase annotations and record
  logical-step provenance before HTA.
- `scripts/analyze_compile_logs.py` — normalize graph breaks, recompilations,
  cache-limit warnings, and backend fallbacks.
- `scripts/render_trace_diagrams.py` — render report-ready SVG timelines using
  only the Python standard library.
- `scripts/ncu_profile.py` — check, plan, capture, summarize, and validate NCU.
- `scripts/validate_phase1_bundle.py` — validate structure and phase boundary.
