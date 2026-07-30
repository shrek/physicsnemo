# Phase-to-source review protocol

Use this protocol after HTA establishes phase timings and before final hotspot
routing. Inspect code to explain measured behavior, not to generate an
unprioritized optimization checklist.

## Required artifacts

Create:

```text
phase-source-map.json
source-analysis.json
```

Cover these canonical phases:

```text
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

Mark a phase `not_applicable` with a reason when the workload genuinely omits
it. Do not silently omit phases.

## Review sequence

For each phase:

1. Start from an HTA range, kernel, CPU operator, idle interval, or launch
   statistic.
2. Locate the entry point, called symbols, configuration, and distributed
   wrapper that implement the phase.
3. Record exact repository-relative paths, symbols, and line bounds.
4. Describe the code behavior without proposing a change.
5. Explain the performance mechanism supported by trace or NCU evidence.
6. Propose a change only when that mechanism and code path support it.
7. Define one isolated phase-2 experiment, correctness check, and unprofiled
   performance check.
8. Record `no_change_reason` when the phase is already efficient, immaterial, or
   insufficiently evidenced.

Inspect callees deeply enough to explain the measured hotspot. Do not review
every helper indiscriminately. Give more depth to phases with higher
critical-path contribution.

## Phase checklist

| Phase | Inspect | Candidate mechanisms and improvements |
|---|---|---|
| `dataloader_wait` | Dataset access, `__getitem__`, transforms, collation, workers, sampling | Synchronous reads, repeated transforms, poor batching; prefetch, persistent workers, caching, PhysicsNeMo datapipe |
| `feature_construction` | Python loops, tensor creation, geometry/SDF/BVH/neighborhood setup | Repeated search, scalar loops, allocations; vectorize, cache, reuse, replace with PhysicsNeMo functionals |
| `host_to_device` | `.to()`, casts, pinning, staging buffers | Pageable copies, serialized casts, allocations; pinned memory, nonblocking copies, fused conversion, buffer reuse |
| `forward` | Module call graph, custom ops, graph breaks, shapes, compile boundaries | Repeated work, launch overhead, unfused math, dynamic recompilation; reuse, fuse, compile continuity, functional replacement |
| `loss` | Indexing, reductions, masks, host reads, temporaries | Redundant reductions, synchronization, allocation; vectorize, fuse, keep values device-side |
| `backward` | Saved activations, recompute, gradient hooks, synchronization | Activation pressure, graph breaks, exposed collectives; checkpoint selectively, adjust precision or sharding |
| `optimizer_step` | Optimizer, scaler, parameter groups, zeroing | Many small launches, host synchronization, allocation; fused optimizer, `set_to_none`, group consolidation |
| `distributed_sync` | DDP/FSDP/domain-parallel setup, collectives, waits, rank conditionals | Poor overlap, bucket behavior, imbalance; bucket/strategy changes, ShardTensor/domain parallelism |
| `validation` | Frequency, device transfers, metric aggregation | Full validation in hot path, host metric reads; reduce cadence, batch aggregation, asynchronous staging |
| `checkpoint` | Serialization, barriers, device copies, frequency | Synchronous I/O or global barriers; asynchronous staging, sharded checkpoints, lower cadence |

Candidate improvements are prompts for inspection, not automatic
recommendations.

## Phase-source map schema

```json
{
  "schema_version": "0.1",
  "phase": 1,
  "run_id": "perf_...",
  "status": "complete",
  "mappings": [
    {
      "phase": "feature_construction",
      "coverage": "mapped",
      "ranges": ["neighbor_search"],
      "sources": [
        {
          "path": "physicsnemo/.../context_projector.py",
          "start_line": 604,
          "end_line": 690,
          "symbol": "extract_shared_context_and_local_features",
          "role": "Constructs multi-scale neighborhood features"
        }
      ],
      "config": ["conf/model/geotransolver_volume.yaml"],
      "evidence": ["hta/kernel-breakdown.json#radius_search"],
      "reason": null
    }
  ]
}
```

Use `coverage: unmapped` only while work is in progress. Ready reports permit
only `mapped` or `not_applicable`.

## Source-analysis schema

```json
{
  "schema_version": "0.1",
  "phase": 1,
  "run_id": "perf_...",
  "status": "complete",
  "reviews": [
    {
      "phase": "feature_construction",
      "review_status": "reviewed",
      "hotspot_ids": ["H003"],
      "measured_evidence": [
        "hta/kernel-breakdown.json#radius_search"
      ],
      "sources": [
        {
          "path": "physicsnemo/.../context_projector.py",
          "start_line": 604,
          "end_line": 690,
          "symbol": "extract_shared_context_and_local_features"
        }
      ],
      "code_observations": [
        "Context and local features each invoke the same six-radius search."
      ],
      "recommendations": [
        {
          "id": "S001",
          "code_observation": "The same neighborhood is constructed twice.",
          "mechanism": "Duplicate radius-search launches add critical-path GPU work.",
          "suggested_improvement": "Construct the shared neighborhood once.",
          "recommended_skill": "physicsnemo-functionals-integrator",
          "phase2_experiment": "Reuse one result for both feature views.",
          "correctness_check": "Compare indices, tensors, shapes, and loss.",
          "performance_check": "Compare launches, forward GPU time, and unprofiled step time.",
          "semantic_risk": "medium",
          "confidence": "high",
          "status": "recommendation_only"
        }
      ],
      "no_change_reason": null
    }
  ]
}
```

Every recommendation must:

- Cite measured evidence and exact source through its enclosing review.
- Separate observation, mechanism, and proposed change.
- State correctness risk and the metric to remeasure.
- Remain `recommendation_only`.
- Avoid a speedup claim until phase 2 implements and reruns the baseline.

Every hotspot in `findings.json` must list its related recommendation IDs in
`source_analysis_ids`. A reviewed phase may contain no recommendation only when
`no_change_reason` explains why.
