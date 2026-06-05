# GeoTransolver Radius Search Optimization Notes

These notes summarize profiling observations for the DrivAerML volume
GeoTransolver recipe and outline likely optimization paths for the Warp
radius-search hotspot.

## Context

The profiles compared two runs with the same model and sampling resolution:

- Uncompiled: `runs/geotransolver_volume__drivaer_ml_volume__20260605_204103`
- Compiled: `runs/geotransolver_volume__drivaer_ml_volume__20260605_204524`

Both used `sampling_resolution=100000`. The compiled run reduced overall model
step time and memory use, but did not reduce the Warp radius-search cost.

## Key Finding

The radius-search kernel is effectively unchanged by `torch.compile`.

Approximate profiled cost:

- Uncompiled radius search: `775 ms / 5 profiled steps`
- Compiled radius search: `775 ms / 5 profiled steps`
- Per-step radius-search cost: about `155 ms`
- Launches per step: `12`

The `12` launches come from `6` radii being used twice per forward pass: once
for context features and once for local features.

## Relevant Code Paths

Radius-search configuration:

- `conf/model/geotransolver_volume.yaml`
  - `radii: [0.01, 0.05, 0.25, 1.0, 2.5, 5.0]`
  - `neighbors_in_radius: [8, 16, 32, 32, 64, 128]`

GeoTransolver local feature construction:

- `physicsnemo/experimental/models/geotransolver/context_projector.py`
  - `GeometricFeatureProcessor` creates `BQWarp`
  - `MultiScaleFeatureExtractor` creates one processor per radius
  - context and local extraction each traverse all radii

Ball query wrapper:

- `physicsnemo/nn/module/ball_query.py`
  - `BQWarp.forward()` calls `radius_search(..., return_points=True)`

Warp implementation:

- `physicsnemo/nn/functional/neighbors/radius_search/_warp_impl.py`
  - builds grid IDs
  - converts tensors to Warp arrays
  - allocates output tensors
  - launches the limited radius-search kernel

Actual Warp kernel:

- `physicsnemo/nn/functional/neighbors/radius_search/kernels.py`
  - `radius_search_limited_select_batched`
  - one thread per query point
  - queries the Warp hash grid
  - checks candidate distances
  - writes neighbor mapping, optional distances, and optional neighbor points
  - stops when `neighbors_found == max_points`

## Why Compile Does Not Help This Kernel

`torch.compile` can fuse PyTorch elementwise work and reduce Python/framework
overhead, but this radius search is an external custom Warp kernel. The compiled
profile confirms that most PyTorch elementwise overhead was fused away while the
Warp kernel time stayed flat.

This makes the Warp path the dominant remaining forward-pass cost after compile.
It is likely memory and irregular-access bound rather than dense-compute bound.

## Correctness Definition

For these notes, a correctness-preserving optimization means:

- same sampled input tensors
- same model architecture and trained parameters
- same configured radii and neighbor caps
- same neighbor values and neighbor ordering consumed by the downstream MLPs
- same padding behavior for missing neighbors

Bitwise equality is a stricter standard. Some options, such as
`torch.compile`, should preserve model semantics but may not guarantee bitwise
identical floating-point results because fused kernels can change evaluation
order. For strict validation, compare outputs and gradients on a fixed batch.

## Per-Radius Cost

The larger radii dominate the runtime. Approximate per-call timings at
`sampling_resolution=100000`:

| Radius | Neighbors | Approx. cost per call |
| --- | ---: | ---: |
| `0.01` | `8` | `1.2 ms` |
| `0.05` | `16` | `2.6 ms` |
| `0.25` | `32` | `12.3 ms` |
| `1.0` | `32` | `19.1 ms` |
| `2.5` | `64` | `20.9 ms` |
| `5.0` | `128` | `21.5 ms` |

Because each radius is currently called twice, the largest three radii account
for roughly `120 ms/step` of the `155 ms/step` radius-search cost.

## Correctness-Preserving Options

### 1. Reuse Duplicate Processor Outputs Within a Forward

For the current DrivAerML volume GeoTransolver config, both `geometry` and
`local_positions` are `interior.points`. That means the context feature path and
local feature path call the same per-radius processors with effectively the same
inputs.

Current structure:

- context path computes `processor(spatial_coords, geometry)` for each radius
- local path computes `processor(geometry, spatial_coords)` for each radius

For this config, `spatial_coords == geometry`, so the two calls should produce
the same processed features. A correctness-preserving refactor is to compute
each per-radius processor output once, then:

- pass the processed feature to the tokenizer for context features
- concatenate the same processed features for the local skip path

This should preserve outputs and gradients because the same tensor is consumed by
both branches instead of recomputing the same function twice. It would remove six
of the twelve radius-search launches per step and also remove duplicate MLP work.

This optimization should be guarded by an explicit check or limited to configs
where the two inputs are known to be the same. If `geometry` and
`local_positions` differ, this is not automatically correctness-preserving.

### 2. Add a Point-Output-Only Radius Search Path

`GeometricFeatureProcessor.forward()` ignores the neighbor mapping returned by
`BQWarp` and only consumes the neighbor point/features tensor:

```python
_, neighbors = self.bq_warp(query_points, key_features)
```

The current limited Warp path still allocates and writes:

- `indices`
- `num_neighbors`
- optional `pts_out`

For this model path, a specialized `BQWarp` mode could preserve correctness while
returning only the neighbor points/features actually used by the MLP. To remain
correct, the specialized kernel must preserve:

- the same candidate traversal behavior
- the same selected neighbor ordering
- the same zero-padding for missing neighbors
- the same output dtype conversion behavior

This would reduce memory allocation and write bandwidth, but it would not reduce
the candidate-search loop itself.

### 3. Remove Redundant Copies Around Warp Inputs

The Warp implementation casts non-fp32 inputs to fp32 and calls `.contiguous()`
before converting tensors to Warp arrays. If upstream code can provide
contiguous fp32 coordinate tensors once, repeated conversion/copy work can be
reduced without changing neighbor results.

This is only useful if the profiler shows meaningful `aten::to`,
`aten::contiguous`, or copy time around radius search. If the tensors are already
contiguous and fp32, this option has little effect.

### 4. Combine Allocation Initialization With the Kernel

The current limited path initializes output tensors before launching the Warp
kernel:

- `indices = torch.full(..., 0, ...)`
- `num_neighbors = torch.zeros(...)`
- `pts_out = torch.zeros(...)` when `return_points=True`

Those zero values are semantically important because missing neighbors are padded
with zeros. A specialized kernel could allocate with `torch.empty()` and write
the same zero padding itself. That can be correctness-preserving if every output
element receives exactly the same value as before.

This is a lower-confidence optimization than reusing duplicate processor outputs:
it saves separate initialization work, but the kernel still has to write padding
values somehow.

### 5. Improve Host/GPU Overlap

This does not reduce the Warp kernel time, but it can reduce visible step time if
the GPU is waiting for data or host-side work. These changes preserve model math:

```bash
python src/train.py profile=true compile=true sampling_resolution=100000 training.num_epochs=3 \
  dataloader.use_streams=true dataloader.prefetch_factor=1 dataloader.num_streams=1
```

If memory allows:

```bash
python src/train.py profile=true compile=true sampling_resolution=100000 training.num_epochs=3 \
  dataloader.use_streams=true dataloader.prefetch_factor=2 dataloader.num_streams=2
```

Keep `num_workers=1` initially. Increasing worker parallelism may change random
subsampling order and can introduce TensorDict construction issues noted in the
datapipe.

### 6. Keep `torch.compile=true`

`torch.compile` did not improve the Warp kernel, but it did reduce surrounding
PyTorch work and memory use. It is a semantics-preserving optimization for the
model configuration, subject to normal floating-point non-bitwise caveats.

Use it together with the Warp-focused changes rather than expecting it to solve
radius search directly.

## Quality-Tradeoff Options

### 1. Reduce Large-Radius Work

The fastest experiment is to reduce the number of large radii or lower their
neighbor caps.

Example: remove the two largest radii.

```bash
python src/train.py compile=true sampling_resolution=100000 \
  model.radii='[0.01,0.05,0.25,1.0]' \
  model.neighbors_in_radius='[8,16,32,32]'
```

Example: keep all radii but reduce large-radius caps.

```bash
python src/train.py compile=true sampling_resolution=100000 \
  model.neighbors_in_radius='[8,16,32,16,32,64]'
```

This changes the model's geometric receptive field, so it may affect model
quality. It is the easiest performance/quality tradeoff to test.

### 2. Reuse Hash Grids Across Different Radii

The current Warp path builds/query-prepares hash grid state inside each radius
search call. With six radii and two feature paths, the same point set may be
prepared repeatedly within a single forward pass.

However, the implementation builds each grid with `radius=0.5 * search_radius`.
Reusing one grid across different radii can change candidate traversal and
neighbor ordering. Since the MLP consumes flattened neighbors, ordering changes
can change model outputs.

Grid reuse is correctness-preserving only for identical input/radius calls, or
if the implementation proves it preserves the exact selected neighbor order.

### 3. Fuse Multi-Radius Search

The current implementation launches one kernel per radius. Since the configured
radii are nested, a custom multi-radius kernel could query the largest radius
once and fill buckets for smaller radii during the same candidate traversal.

This is likely the most invasive option. The output shapes differ by radius and
neighbor cap, and preserving the current neighbor ordering may be difficult. It
should be treated as a potential behavior change unless validated carefully.

## Recommended Order

1. Add profiler labels around each radius call so the profile directly names
   `radius` and `neighbors_in_radius`.
2. Prototype reusing duplicate per-radius processor outputs for the current
   DrivAerML volume config.
3. Validate output and gradient equality on a fixed batch with `compile=false`.
4. Re-run the compiled profile and confirm radius-search launches drop from
   `12` to `6` per step.
5. Then consider a point-output-only Warp path to avoid unused mapping writes.
6. Treat large-radius ablations and multi-radius rewrites as quality/performance
   experiments, not strict correctness-preserving changes.

## Test Results: Optimizations 1 and 2

A local harness was added at `scripts/test_radius_search_optimizations.py`. It
prototypes the two correctness-preserving options without modifying the shared
PhysicsNeMo package.

The most relevant run used one actual DrivAerML train dataloader batch with
`sampling_resolution=100000`. The config resolved both `geometry` and
`local_positions` to `interior.points`, and the runtime tensor equality check
confirmed they were identical for the tested batch.

Result file:

- `runs/radius_search_optimization_test_real_100k.json`

Forward equivalence on the real batch:

| Comparison | Max abs diff |
| --- | ---: |
| Optimization 1 reuse vs baseline | `0.0` |
| Optimization 1 + 2 vs Optimization 1 | `0.0` |

Extractor benchmark on the real 100K batch, H100 80GB, inference-mode timing:

| Case | Searches | Time / iter | Speedup vs baseline |
| --- | ---: | ---: | ---: |
| Baseline original | `12` | `183.81 ms` | `1.00x` |
| Optimization 1: reuse processor outputs | `6` | `105.00 ms` | `1.75x` |
| Optimization 2: point-output-only kernel | `12` | `185.24 ms` | `0.99x` |
| Optimization 1 + 2 | `6` | `104.40 ms` | `1.76x` |

A smaller gradient check also passed in the practical sense:

- Optimization 1 forward max abs diff: `0.0`
- Optimization 1 parameter-gradient max abs diff: `5.9e-6`
- Optimization 2 forward max abs diff: `0.0`
- Optimization 2 parameter-gradient max abs diff: `0.0`

The small nonzero Optimization 1 gradient difference is from floating-point
accumulation order: the baseline backpropagates through two identical processor
calls, while the optimized path backpropagates through one shared processor
output consumed by two branches. This is mathematically equivalent, but not
bitwise-identical in GPU floating point.

Conclusion:

- Optimization 1 is the clear implementation candidate. It removes duplicate
  radius-search and MLP work and saved about `79 ms` in the isolated extractor
  path for one real 100K DrivAerML batch.
- Optimization 2 is not worth implementing first. On the real batch, removing
  mapping/neighbor-count writes did not improve runtime, likely because the
  candidate traversal and neighbor-point output dominate.

## Full Training/Profile Validation After Optimization 1

After implementing `reuse_context_local_features` and enabling it for
`geotransolver_volume.yaml`, matched training/profile runs were executed against
the earlier 100K setup:

```bash
python src/train.py profile=true compile=false sampling_resolution=100000 training.num_epochs=3 \
  run_id=opt1_uncompiled_100k_20260605_1

python src/train.py profile=true compile=true sampling_resolution=100000 training.num_epochs=3 \
  run_id=opt1_compiled_100k_20260605_1
```

Both runs completed successfully. The eager runtime guard confirmed the optimized
path was valid for this config because `geometry` and `local_positions` shared
the same tensor storage.

Profiler comparison against the earlier unoptimized profile runs:

| Run | Compile | Radius launches / profiled step | Radius kernel time / step | Train forward-pass range / step | Profiler step wall / step |
| --- | --- | ---: | ---: | ---: | ---: |
| Earlier baseline `20260605_204103` | false | `12` | `155.1 ms` | `280.2 ms` | `1100.6 ms` |
| Optimized `opt1_uncompiled_100k_20260605_1` | false | `6` | `77.6 ms` | `199.2 ms` | `1032.1 ms` |
| Earlier baseline `20260605_204524` | true | `12` | `155.0 ms` | `221.1 ms` | `753.3 ms` |
| Optimized `opt1_compiled_100k_20260605_1` | true | `6` | `77.7 ms` | `141.8 ms` | `673.5 ms` |

Full-epoch training log comparison for epoch 2, outside the truncated profiler
window:

| Run | Compile | Epoch 2 train avg step | Memory reported |
| --- | --- | ---: | ---: |
| Earlier baseline `20260605_204103` | false | about `1.079 s/step` | about `61.8 GB` |
| Optimized `opt1_uncompiled_100k_20260605_1` | false | `0.996 s/step` | `61.4 GB` |
| Earlier baseline `20260605_204524` | true | about `0.722 s/step` | about `48.0 GB` |
| Optimized `opt1_compiled_100k_20260605_1` | true | `0.636 s/step` | `48.0 GB` |

Final epoch metrics remained in the same range as the earlier runs. The training
path is not expected to be bitwise-identical because the optimized path shares
one processor output across two loss branches, changing floating-point gradient
accumulation order.

| Run | Compile | Epoch 2 train loss | Epoch 2 val loss |
| --- | --- | ---: | ---: |
| Earlier baseline `20260605_204103` | false | `0.072981` | `0.070428` |
| Optimized `opt1_uncompiled_100k_20260605_1` | false | `0.073187` | `0.071515` |
| Earlier baseline `20260605_204524` | true | `0.072274` | `0.070246` |
| Optimized `opt1_compiled_100k_20260605_1` | true | `0.072368` | `0.070141` |

Conclusion: Optimization 1 validated in both eager and compiled training. It
halves the radius-search launches and radius-kernel time in the profiled window.
The visible full-step improvement is smaller because data loading, optimizer,
backward, and other model work remain in the step.

## Profiling Command Examples

Baseline compiled profiling run:

```bash
python src/train.py profile=true compile=true sampling_resolution=100000 training.num_epochs=3
```

Compiled run with streams enabled to check overlap with data loading:

```bash
python src/train.py profile=true compile=true sampling_resolution=100000 training.num_epochs=3 \
  dataloader.use_streams=true dataloader.prefetch_factor=1 dataloader.num_streams=1
```

Large-radius ablation:

```bash
python src/train.py profile=true compile=true sampling_resolution=100000 training.num_epochs=3 \
  model.radii='[0.01,0.05,0.25,1.0]' \
  model.neighbors_in_radius='[8,16,32,32]'
```

## Summary

The Warp radius-search kernel is now the main forward-pass bottleneck. Compile
helps the surrounding PyTorch work, but not this custom kernel. The highest
leverage correctness-preserving optimization is to avoid duplicate per-radius
processor work when the context and local feature paths use the same inputs.
After that, consider a point-output-only Warp path to remove unused mapping
writes. Reducing radii, changing grid strategy, or fusing multi-radius search
should be treated as quality/performance experiments unless exact output
equivalence is proven.
