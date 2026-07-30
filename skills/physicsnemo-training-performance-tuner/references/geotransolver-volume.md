# GeoTransolver volume golden workflow

## Purpose

Use the unified external-aerodynamics GeoTransolver volume case as the first
end-to-end evaluation of this skill. Do not copy historical conclusions into a
new report; rediscover or refute them from the current run.

## Repository inputs

Resolve paths from the PhysicsNeMo repository root:

- Workdir:
  `examples/cfd/external_aerodynamics/unified_external_aero_recipe`
- Entry point: `src/train.py`
- Training config: `conf/train.yaml`
- Model config: `conf/model/geotransolver_volume.yaml`
- Dataset config: `datasets/drivaer_ml_volume.yaml`
- Model selection: `model=geotransolver_volume`
- Dataset selection: `dataset=drivaer_ml_volume`
- Compile mode: `compile=true`

The minimal command is:

```bash
python src/train.py \
  model=geotransolver_volume \
  dataset=drivaer_ml_volume \
  compile=true
```

Add explicit seed, sampling resolution, step budget, run ID, and output path for
the test environment. Record every override in the run manifest.

## Initial test shape

- Start with one GPU.
- Use bf16 when supported and record the resolved precision.
- Warm the dataloader, allocator, and compiled graph before measurement.
- Collect at least three unprofiled repetitions.
- Capture five steady-state profiler iterations.
- Use a representative DrivAerML volume sample, not a tiny synthetic mesh.
- Use finite loss, output/target shape agreement, and loss parity against the
  unprofiled smoke run as the initial correctness contract.

Do not assume a dataset location. Ask for it when the configured manifest or
dataset root cannot be resolved.

## Phase taxonomy

Decompose a steady-state step into:

1. Dataloader wait.
2. SDF/BVH work.
3. Multi-scale neighbor search.
4. Transformer forward.
5. Loss and backward.
6. Optimizer/end-of-step.
7. Inter-step idle.

Map actual ranges and kernels to these phases from evidence. Do not force an
event into a phase when the trace cannot support the mapping.

## HTA analysis

Use the standard HolisticTraceAnalysis sequence from `SKILL.md`. In addition:

- Compare total GPU busy time with full CPU `ProfilerStep` wall time.
- Separate in-step idle from inter-step dataload bubbles.
- Group forward kernels into radius search, BVH/SDF, attention, GEMM, Triton
  pointwise/reduction, memory, and other families when symbols permit.
- Inspect consecutive steady-state iterations because volume sample sizes vary.
- State whether a diagram includes dataloading in the step boundary.
- Generate the CPU/GPU step pipeline and forward dominant-kernel views.
- Generate the in-step idle-bubble view when custom operations cause host gaps.
- Generate the consecutive-step bubble view when dataloading separates GPU busy
  envelopes.

## Golden evaluation expectations

Historical traces have shown these behaviors. They are evaluation targets, not
facts for a new run:

- Compilation can reduce transformer math enough to move the bottleneck.
- Warp `radius_search` and `bvh_query` can remain opaque to compilation.
- Repeated radius searches can dominate geometry preprocessing.
- Host waits around custom operations can introduce GPU idle gaps.
- Inter-step synchronous loading can dominate the remaining wall time.
- Removing GPU work can increase idle percentage without being a regression.

The skill passes this golden evaluation when it:

1. Produces a reproducible unprofiled baseline.
2. Separates CPU-step wall time, GPU busy time, and idle.
3. Identifies the actual critical path from the current trace.
4. Selects NCU only for kernels requiring mechanism-level evidence.
5. Maps every canonical training phase to the responsible GeoTransolver source
   and configuration or marks it not applicable.
6. Produces code-specific `recommendation_only` findings with source,
   mechanism, correctness, and phase-2 experiment fields.
7. Ends at hypotheses and routing without changing training behavior.

The diagram set should mirror the useful analytical views in the reference
report without copying its measured values:

1. Full CPU/GPU lanes for one representative step.
2. Forward-pass kernel timeline grouped by dominant family.
3. Focused in-step GPU idle bubbles when present.
4. GPU busy envelopes and dataload bubbles across consecutive steps when
   present.

## Historical optimization boundary

A historical branch deduplicated the six-radius neighbor search, reducing twelve
calls per step to six. The current run must verify whether that implementation
is present before using this as a hypothesis. Never report its historical
speedup as a result for the current branch.

Potential phase-2 hypotheses include:

- Deduplicate or reuse neighbor-search results.
- Preserve compiled-graph continuity around custom operations.
- Cache SDF/BVH results when geometry semantics permit.
- Prefetch or double-buffer dataloading.
- Reduce optimizer/end-of-step host waits.

Phase 1 may recommend these experiments but must not apply them.

For each hypothesis, inspect the current branch before recommending it. For
example, verify whether the shared neighborhood-search path is already enabled,
whether the dataloader already overlaps samples, and whether SDF/BVH results are
stable enough to cache. Historical behavior is not source evidence for the
current run.

## Golden NCU drill-down

When the current HTA evidence places `radius_search`, `bvh_query`, or another
custom CUDA kernel on the critical path, create exactly one first-pass NCU
candidate. Do not select it merely because it appeared in historical notes.

For a radius-search candidate:

- Prefer a stable kernel regex such as `.*radius_search.*`.
- Add an NVTX selector when the training loop exposes a neighbor-search range.
- Capture no more than ten launches with the `default` set first.
- Ask whether memory traffic, occupancy, dependency latency, or candidate
  scanning explains the observed kernel cost.
- Run a focused second pass only when the first-pass metrics justify it.

Store all evidence under `ncu/` and keep HTA's critical-path evidence as the
reason for the drill-down.
