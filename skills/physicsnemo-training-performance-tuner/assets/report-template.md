# {{WORKLOAD}} training performance analysis

_Phase 1: measurement, diagnosis, and routing only._  
_Run: `{{RUN_ID}}` · Date: `{{DATE}}` · Commit: `{{GIT_COMMIT}}`_

## Executive summary

{{EXECUTIVE_SUMMARY}}

## Model and workload

| Item | Value |
|---|---|
| Entry point | {{ENTRY_POINT}} |
| Launch command | `{{COMMAND}}` |
| Model/config | {{MODEL_CONFIG}} |
| Dataset/sample | {{DATASET}} |
| Batch/sample size | {{BATCH_SIZE}} |
| Precision/compile | {{PRECISION_COMPILE}} |
| Hardware | {{HARDWARE}} |
| Distributed strategy | {{DISTRIBUTED}} |

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

## Baseline performance

| Metric | Median | Dispersion | Unit |
|---|---:|---:|---|
| Step time | {{STEP_TIME}} | {{STEP_TIME_DISPERSION}} | ms |
| Throughput | {{THROUGHPUT}} | {{THROUGHPUT_DISPERSION}} | {{THROUGHPUT_UNIT}} |
| Peak GPU allocated | {{GPU_ALLOCATED}} | — | MiB |
| Peak GPU reserved | {{GPU_RESERVED}} | — | MiB |
| Data wait | {{DATA_WAIT}} | {{DATA_WAIT_DISPERSION}} | ms |

## Traces

List each trace, rank, active iteration, compile mode, and HTA analysis input.

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

![Annotated CPU/GPU lanes](hta/diagrams/cpu-gpu-pipeline.svg)

Define the ProfilerStep boundary, stage taxonomy, GPU busy time, idle time, and
the cause of each annotated bubble.

## Forward-pass dominant kernels

![Forward-pass kernel-family timeline](hta/diagrams/forward-dominant-kernels.svg)

Explain the dominant kernel family, its aggregate time and launch count, and
whether its cost changes the next recommendation.

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
| `correctness.json` | Correctness result and evidence |
| `findings.json` | Machine-readable ranked recommendations |
| `phase-source-map.json` | Trace-phase to source/configuration mapping |
| `source-analysis.json` | Code observations and phase-2 recommendations |
| `traces/` | Kineto traces |
| `hta/` | HolisticTraceAnalysis tables and diagrams |
| `hta/diagram-data.json` | Normalized spans used for report diagrams |
| `hta/diagrams/diagram-manifest.json` | Generated diagram inventory |
| `ncu/` | Conditional Nsight Compute reports |
| `logs/` | Smoke, baseline, and profiling logs |
