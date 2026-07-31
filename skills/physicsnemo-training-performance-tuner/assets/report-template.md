# {{WORKLOAD}} training performance analysis

_Phase 1: measurement, diagnosis, and routing only._  
_Run: `{{RUN_ID}}` · Date: `{{DATE}}` · Commit: `{{GIT_COMMIT}}`_

## Executive summary

{{EXECUTIVE_SUMMARY}}

## Model and workload

| Item | Value |
|---|---|
| Entry point | {{ENTRY_POINT}} |
| Eager command | `{{EAGER_COMMAND}}` |
| Compiled command | `{{COMPILED_COMMAND}}` |
| Compile backend/mode | {{COMPILE_BACKEND}} / {{COMPILE_MODE}} |
| Compile fullgraph/dynamic | {{COMPILE_FULLGRAPH}} / {{COMPILE_DYNAMIC}} |
| Model/config | {{MODEL_CONFIG}} |
| Dataset/sample | {{DATASET}} |
| Batch/sample size | {{BATCH_SIZE}} |
| Precision | {{PRECISION_COMPILE}} |
| Hardware | {{HARDWARE}} |
| Distributed strategy | {{DISTRIBUTED}} |

The tested configuration was shown to the user before workload execution and
confirmed against the fingerprint recorded in `config-confirmation.json`.

## Benchmark protocol

| Item | Value |
|---|---|
| Warmup steps | {{WARMUP_STEPS}} |
| Measured steps | {{MEASURE_STEPS}} |
| Repetitions | {{REPETITIONS}} |
| Profiled iterations | {{PROFILE_STEPS}} |
| Correctness signal | {{CORRECTNESS_SIGNAL}} |
| Performance goal | {{PERFORMANCE_GOAL}} |

Describe synchronization, sample ordering, profiler-disabled baseline timing,
and any deviations from the manifest.

## Paired eager and compiled baseline

| Metric | Eager | Compiled steady state | Delta/ratio | Unit |
|---|---:|---:|---:|---|
| Step time median | {{EAGER_STEP_TIME}} | {{COMPILED_STEP_TIME}} | {{STEP_TIME_DELTA}} | ms |
| Step-time dispersion | {{EAGER_STEP_DISPERSION}} | {{COMPILED_STEP_DISPERSION}} | — | ms |
| Throughput | {{EAGER_THROUGHPUT}} | {{COMPILED_THROUGHPUT}} | {{THROUGHPUT_DELTA}} | {{THROUGHPUT_UNIT}} |
| Peak GPU allocated | {{EAGER_GPU_ALLOCATED}} | {{COMPILED_GPU_ALLOCATED}} | {{GPU_ALLOCATED_DELTA}} | MiB |
| Data wait | {{EAGER_DATA_WAIT}} | {{COMPILED_DATA_WAIT}} | {{DATA_WAIT_DELTA}} | ms |

| Compilation item | Result |
|---|---:|
| Cold compilation time | {{COMPILE_TIME}} ms |
| First compiled step | {{FIRST_COMPILED_STEP}} ms |
| Compile cache state | {{COMPILE_CACHE_STATE}} |
| Steady-state speedup | {{COMPILE_SPEEDUP}} |
| Estimated amortization | {{AMORTIZATION_STEPS}} steps |
| Classification | {{COMPILE_CLASSIFICATION}} |

Speedup must come from unprofiled steady-state measurements. Do not include
cold compilation, verbose compiler logging, or profiler overhead.

## Paired traces

List eager and compiled traces, rank, active iteration, annotation-health result,
and HTA analysis input. State whether the same samples and logical steps were
used.

## Annotation health and step provenance

| Item | Eager | Compiled |
|---|---|---|
| Logical-step provenance | {{EAGER_STEP_PROVENANCE}} | {{COMPILED_STEP_PROVENANCE}} |
| Boundary/selection rule | {{EAGER_STEP_BOUNDARY}} | {{COMPILED_STEP_BOUNDARY}} |
| Required phase coverage | {{EAGER_ANNOTATION_COVERAGE}} | {{COMPILED_ANNOTATION_COVERAGE}} |
| Duplicate projected annotations | {{EAGER_PROJECTED_ANNOTATIONS}} | {{COMPILED_PROJECTED_ANNOTATIONS}} |
| Timestamp/end handling | {{EAGER_TIMESTAMP_HANDLING}} | {{COMPILED_TIMESTAMP_HANDLING}} |
| Compile annotation warning | {{EAGER_COMPILE_WARNING}} | {{COMPILED_COMPILE_WARNING}} |

State whether each boundary is a native `ProfilerStep`, an explicit `train_step`,
or a reconstructed logical step. Link both annotation-health artifacts and
describe fallbacks without presenting reconstructed boundaries as native output.

## torch.compile diagnostics

| Diagnostic | Result | Evidence |
|---|---:|---|
| Unique graph breaks | {{GRAPH_BREAK_COUNT}} | {{GRAPH_BREAK_EVIDENCE}} |
| Graph-break occurrences | {{GRAPH_BREAK_OCCURRENCES}} | {{GRAPH_BREAK_EVIDENCE}} |
| Unique recompilations | {{RECOMPILE_COUNT}} | {{RECOMPILE_EVIDENCE}} |
| Recompilation occurrences | {{RECOMPILE_OCCURRENCES}} | {{RECOMPILE_EVIDENCE}} |
| Cache-limit warnings | {{CACHE_LIMIT_WARNINGS}} | {{COMPILE_LOG}} |
| Backend failure/eager fallback | {{COMPILE_FALLBACK}} | {{COMPILE_LOG}} |

List each graph-break source location, reason, and count. List recompilation
failed guards. Treat this as diagnostic evidence, not benchmark timing.

## Whole-trace temporal breakdown

| Rank/trace | Compute | Communication | Memory | Idle | Host-wait share |
|---|---:|---:|---:|---:|---:|
| {{TRACE_ROW}} | | | | | |

## Per-step phase decomposition

| Phase | Wall time | GPU busy | Idle | Step share | Evidence |
|---|---:|---:|---:|---:|---|
| {{PHASE_ROW}} | | | | | |

## Critical path

{{CRITICAL_PATH}}

## Key finding: {{KEY_FINDING_TITLE}}

{{KEY_FINDING}}

## GPU kernel breakdown

| Kernel family | Calls | Total time | Mean time | Phase/range |
|---|---:|---:|---:|---|
| {{KERNEL_ROW}} | | | | |

## CPU/GPU step pipeline

### Eager

![Eager annotated CPU/GPU lanes](hta/eager/diagrams/cpu-gpu-pipeline.svg)

### Compiled

![Compiled annotated CPU/GPU lanes](hta/compiled/diagrams/cpu-gpu-pipeline.svg)

Compare validated logical-step provenance, stage taxonomy, GPU busy time, idle
time, launches, and the cause of each annotated bubble.

## Forward-pass dominant kernels

### Eager

![Eager forward kernel timeline](hta/eager/diagrams/forward-dominant-kernels.svg)

### Compiled

![Compiled forward kernel timeline](hta/compiled/diagrams/forward-dominant-kernels.svg)

Explain changes in dominant kernel families, aggregate time, launch count,
compiled/fused regions, and whether the critical path changes.

## NCU kernel analysis

State whether NCU was run and why.

| Kernel | Limiting mechanism | Evidence | Report |
|---|---|---|---|
| {{NCU_ROW}} | | | |

## Host/GPU pipeline and idle bubbles

{{OPTIONAL_BUBBLE_DIAGRAMS}}

Add the generated in-step or multi-step bubble diagrams only when they
materially explain execution order or idle. Define the boundary used by every
diagram.

## Phase-to-source map

| Phase | Trace range/evidence | Source and symbol | Coverage |
|---|---|---|---|
| {{PHASE_SOURCE_ROW}} | | | |

Every canonical phase must be mapped or explicitly marked not applicable.

## Phase-by-phase code analysis

| Phase | Measured bottleneck | Code observation | Performance mechanism |
|---|---|---|---|
| {{SOURCE_ANALYSIS_ROW}} | | | |

Separate observed implementation behavior from the inferred performance
mechanism. Cite exact source locations and profiler evidence.

## Code-specific improvement candidates

| ID | Phase/source | Suggested improvement | Correctness risk | Isolated phase-2 experiment | Performance check |
|---|---|---|---|---|---|
| {{SOURCE_RECOMMENDATION_ROW}} | | | | | |

These are evidence-backed recommendations, not implemented or measured
optimizations.

## Ranked hotspots

| Priority | Hotspot | Evidence | Confidence | Critical-path impact |
|---:|---|---|---|---:|
| {{HOTSPOT_ROW}} | | | | |

## PhysicsNeMo skill routing

| Hotspot | Recommended skill | Proposed isolated experiment |
|---|---|---|
| {{ROUTING_ROW}} | | |

## Hypotheses to verify

1. {{HYPOTHESIS_1}}
2. {{HYPOTHESIS_2}}
3. {{HYPOTHESIS_3}}

These are recommendations, not measured improvements.

## Correctness

{{CORRECTNESS_RESULT}}

## Limitations and caveats

- {{LIMITATION_1}}
- {{LIMITATION_2}}

## Residual bottlenecks

{{RESIDUAL_BOTTLENECKS}}

## Artifact index

| Artifact | Purpose |
|---|---|
| `run-manifest.json` | Resolved reproducibility contract |
| `baseline.json` | Raw and aggregate unprofiled measurements |
| `test-config.json` | Machine-readable configuration shown for confirmation |
| `test-config.md` | Human-readable configuration confirmation block |
| `config-confirmation.json` | User-confirmed test-config fingerprint |
| `correctness.json` | Correctness result and evidence |
| `annotation-health-eager.json` | Eager annotation integrity and step provenance |
| `annotation-health-compiled.json` | Compiled annotation integrity and step provenance |
| `compile-analysis.json` | Graph breaks, recompilations, and fallback diagnostics |
| `compile-comparison.json` | Paired steady-state and trace conclusion |
| `findings.json` | Machine-readable ranked recommendations |
| `phase-source-map.json` | Trace-phase to source/configuration mapping |
| `source-analysis.json` | Code observations and phase-2 recommendations |
| `traces/eager/`, `traces/compiled/` | Paired Kineto traces |
| `hta/eager/`, `hta/compiled/` | Paired HTA tables and diagrams |
| `hta/<variant>/diagram-data.json` | Normalized paired diagram spans |
| `hta/<variant>/diagrams/diagram-manifest.json` | Per-variant diagram inventory |
| `ncu/` | Conditional Nsight Compute reports |
| `logs/` | Smoke, baseline, and profiling logs |
