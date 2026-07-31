# Phase-1 profiling protocol

## Contents

1. Configuration confirmation contract
2. Benchmark contract
3. Instrumentation contract
4. Evidence contract
5. Hotspot routing
6. NCU decision gate
7. Report contract

## Configuration confirmation contract

Resolve and display the exact performance-relevant test configuration before
executing the workload. The confirmation block must include:

- Working directory, complete launch command, source revision, and local patch
  state.
- Entry point, model/config, and full resolved-config artifact when available.
- Dataset path or identifier, split, sample selection and ordering, batch/sample
  size, and sampling resolution.
- Seed/determinism, precision, GPU count/type, distributed mode, and dataloader
  settings.
- Complete eager (`torch.compile=false`) and compiled (`torch.compile=true`)
  commands plus backend, mode, fullgraph, dynamic-shape, and cache settings.
- Warmup, measured, profiled, and repetition budgets.
- Correctness signal, performance goal, synchronization method, and artifact
  output.

Mark unknown values explicitly; do not silently substitute defaults. Ask one
targeted question for a missing command, representative dataset, or correctness
signal.

Require explicit user confirmation after displaying the configuration. A
request to run the tuner or approval of a plan is not configuration
confirmation. Do not run a smoke test, baseline, Kineto capture, or NCU before
confirmation.

Store the displayed configuration in `test-config.json` and `test-config.md`.
Store confirmation separately in `config-confirmation.json`, bound to the
SHA-256 fingerprint of `test-config.json`. Re-display and reconfirm whenever the
fingerprint changes. Validate the gate with:

```bash
python scripts/validate_phase1_bundle.py <artifact-dir> --confirmed
```

## Benchmark contract

Keep the following invariant across comparable runs:

- Source commit and local patch.
- Dataset, split, sample selection, and sample ordering.
- Batch size and sampling resolution.
- Seed and deterministic settings.
- Precision and gradient accumulation.
- GPU model/count and distributed strategy.
- Launch command apart from the explicit compile state/options under comparison.
- Warmup count, measured-step count, and synchronization points.

Run correctness and a smoke test for eager and compiled variants before spending
the profiling budget. Run both baselines without the PyTorch profiler, compile
logging, or NCU. Measure compiled cold start separately, warm the compiled path,
and use at least three steady-state repetitions per variant by default. Report
the median plus dispersion and store raw per-step measurements so aggregates can
be recomputed. Derive speedup only from unprofiled steady-state measurements.
Read `references/compile-comparison.md` for the paired protocol.

Record:

| Dimension | Required measurement |
|---|---|
| Latency | Per-step wall time after warmup |
| Throughput | Samples, points, tokens, or batches per second |
| GPU memory | Peak allocated and reserved bytes |
| CPU | Wall time and utilization when available |
| Pipeline | Dataloader wait and host-to-device span |
| Distributed | Per-rank step time and exposed communication |
| Correctness | Loss, metric, shape, or smoke-test result |

Record synchronization used for timing. CUDA work is asynchronous, so an
unexplained CPU timer is insufficient evidence for GPU phase durations.

## Instrumentation contract

Prefer a small, opt-in instrumentation layer. Preserve the training code's
normal behavior when profiling is disabled.

Use consistent range names:

```text
train_step
  dataloader_wait
  feature_construction
  host_to_device
  forward
  loss
  backward
  optimizer_step
  distributed_sync
  validation
  checkpoint
```

Place `profiler.step()` exactly once per logical training step and configure an
explicit schedule that emits active `ProfilerStep#N` markers. If the profiler
wrapper cannot guarantee those markers, use the outer `train_step` range as the
explicit boundary. Warm the model, compiler, dataloader, and allocator before
the active trace window. In distributed runs, produce uniquely named traces per
rank.

Capture the same logical steps for eager and compiled variants. Validate both raw
traces with `scripts/validate_trace_annotations.py` before HTA. Read
`references/annotation-health.md` for duplicate projected annotations,
inconsistent interval ends, compile-warning handling, recapture rules, and the
bounded reconstruction fallback. Record `annotation-health-eager.json` and
`annotation-health-compiled.json` separately.

Run `TORCH_LOGS=graph_breaks,recompiles` only in a separate bounded compiled
diagnostic, normalize it with `scripts/analyze_compile_logs.py`, and never use
its wall time as benchmark evidence.

## Evidence contract

Every hotspot record must contain:

```json
{
  "id": "H001",
  "category": "data_loading",
  "severity": "high",
  "confidence": "high",
  "summary": "GPU is starved between steps while the host loads the next batch.",
  "critical_path_fraction_percent": 35.0,
  "evidence": [
    {
      "kind": "hta_metric",
      "name": "host_wait_share_of_idle",
      "value": 96.0,
      "unit": "percent",
      "artifact": "hta/idle-time.json",
      "source": null
    }
  ],
  "recommended_skill": "physicsnemo-datapipe-adapter",
  "proposed_experiment": "Prefetch batch N+1 while batch N executes.",
  "source_analysis_ids": ["S001"],
  "status": "recommendation_only"
}
```

Use `critical_path_fraction_percent: null` when it cannot be defended. Do not
turn an upper bound into a prediction. Label evidence as one of:

- `measured`
- `estimated`
- `theoretical_upper_bound`

The JSON bundle may include this label in each evidence item or explain it in
the report.

## Hotspot routing

| Evidence | Category | Route | Candidate action |
|---|---|---|---|
| Long dataloader range; host-wait idle between steps | `data_loading` | `physicsnemo-datapipe-adapter` | Prefetch, batch, cache, or replace the loader |
| Python feature loops or expensive collation | `feature_construction` | `physicsnemo-datapipe-adapter` | Vectorize or move/cached transforms |
| Geometry, interpolation, neighborhood, or loss kernels dominate | `custom_math` | `physicsnemo-functionals-integrator` or renamed model-enhancement skill | Replace or fuse custom math |
| Peak memory or dense large-input intermediates | `memory_pressure` | `physicsnemo-shard-tensor-scaling` | Shard tensors or domain |
| Exposed collectives, imbalance, or stragglers | `distributed_scaling` | `physicsnemo-shard-tensor-scaling` | Change sharding or distributed strategy |
| Shapes, features, precision, or launch settings implicated | `configuration` | `physicsnemo-recipe-configurator` | Change validated configuration |
| Proposed change affects fields, units, targets, timestep, or metrics | `semantic_risk` | `physicsnemo-solver-recipe-consistency-auditor` | Audit before experimentation |

When two routes apply, name a primary route and explain the fallback. Do not
invoke either route during phase 1.

Before routing, complete the phase-to-source review in
`references/source-review-protocol.md`. Every hotspot must link to at least one
source-analysis recommendation. Recommendations must distinguish observed code,
the evidence-backed performance mechanism, the candidate change, correctness
risk, and the metric to remeasure.

## NCU decision gate

Run NCU only when all are true:

1. HTA identifies a candidate kernel on the critical path.
2. Kernel-level mechanism changes the next recommendation.
3. A bounded warm iteration or NVTX range can be selected.
4. The profiling budget permits replay.
5. The exact print-only command has separate, fingerprint-bound user approval.
6. `ncu` is available on the target system; `doctor` alone is not proof that a
   real capture is permitted.

Useful NCU classifications include memory-bandwidth limitation, low occupancy,
latency/dependency limitation, poor tensor-core utilization, excessive global
memory traffic, synchronization/atomic contention, and unfavorable layout or
shape.

Avoid broad `--set full` captures over complete training runs. Preserve the
report version, command filters, replay mode, and `.ncu-rep` file.

Use `scripts/ncu_profile.py` for the target check, capture plan, explicit
approval binding, execution, summary, and artifact validation. Read `references/ncu-profiling.md` before
running the capture.

## Report contract

Write the report from `assets/report-template.md`. Preserve these distinctions:

- Baseline wall time and compile speedup come from paired unprofiled steady-state
  runs.
- Cold compilation, compiler-log runs, and profiler runs are reported separately.
- Phase/kernel timing comes from profiler evidence.
- NCU explains selected kernels but does not establish end-to-end speed.
- Phase-1 opportunities are hypotheses, not measured improvements.
- Correctness is a first-class result.
- Code suggestions are supported by both measured profiler evidence and exact
  source inspection.

Generate diagrams from normalized HTA evidence using
`scripts/render_trace_diagrams.py` and `references/diagram-schema.md`. Always
include one representative CPU/GPU lane diagram and one forward-pass
dominant-kernel timeline. Add the focused in-step bubble and consecutive-step
dataload-bubble views when those relationships materially explain the hotspot.

End the report with ranked hypotheses, routing, limitations, and an artifact
index. Do not append optimization results until phase 2 exists and reruns the
same benchmark protocol.
