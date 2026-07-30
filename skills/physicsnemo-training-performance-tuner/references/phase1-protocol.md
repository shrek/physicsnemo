# Phase-1 profiling protocol

## Contents

1. Benchmark contract
2. Instrumentation contract
3. Evidence contract
4. Hotspot routing
5. NCU decision gate
6. Report contract

## Benchmark contract

Keep the following invariant across comparable runs:

- Source commit and local patch.
- Dataset, split, sample selection, and sample ordering.
- Batch size and sampling resolution.
- Seed and deterministic settings.
- Precision, compile mode, and gradient accumulation.
- GPU model/count, distributed strategy, and launch command.
- Warmup count, measured-step count, and synchronization points.

Run a smoke test before spending the profiling budget. Run the baseline without
the PyTorch profiler or NCU. Use at least three repetitions by default and report
the median plus dispersion. Store raw per-step measurements so an aggregate can
be recomputed.

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

Place `profiler.step()` exactly once per logical training step. Warm the model,
compiler, dataloader, and allocator before the active trace window. In
distributed runs, produce uniquely named traces per rank.

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

Useful NCU classifications include memory-bandwidth limitation, low occupancy,
latency/dependency limitation, poor tensor-core utilization, excessive global
memory traffic, synchronization/atomic contention, and unfavorable layout or
shape.

Avoid broad `--set full` captures over complete training runs. Preserve the
report version, command filters, replay mode, and `.ncu-rep` file.

Use `scripts/ncu_profile.py` for the target check, capture plan, execution,
summary, and artifact validation. Read `references/ncu-profiling.md` before
running the capture.

## Report contract

Write the report from `assets/report-template.md`. Preserve these distinctions:

- Baseline wall time comes from unprofiled runs.
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
