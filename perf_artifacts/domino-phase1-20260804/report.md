# DoMINO eager training performance analysis

Phase 1 report. Measurements only; recommendations are not implemented results.

## Executive summary

On one H100 80 GB GPU, the 38-sample eager training epoch was 79.34–79.93 s
(median 79.58 s). A validated Kineto capture and HTA analysis identify two
evidence-backed bottleneck classes: repeated geometry neighborhood search on
GPU and host-wait idle time between GPU work.

## Configuration and correctness

- Entry point: `examples/cfd/external_aerodynamics/domino/src/train.py`.
- Data: deterministic 38/10 Zarr train/validation split under
  `artifacts/domino-data-split/`.
- Model: combined DoMINO, batch size 1, AMP fp16, single H100, no sharding.
- Scaling factors: `scaling_factors/scaling_factors.pkl`.
- Correctness: smoke and all three baselines completed with finite train and
  validation losses. Peak observed NVML memory was about 17.6 GB.

## Unprofiled eager baseline

| Epoch | Training time (s) |
|---:|---:|
| 1 | 79.93 |
| 2 | 79.58 |
| 3 | 79.34 |

The profiler run is excluded from these timing results.

## Trace integrity

`annotation-health-eager.json` passes after adding explicit `train_step` and
CPU-side `forward`, `loss`, `backward`, and `optimizer_step` ranges. HTA does
not recognize the explicit boundary as native `ProfilerStep`; therefore HTA
whole-trace aggregates are used, but no per-step HTA timing or generated SVG
timeline is claimed.

## HTA evidence

HTA temporal breakdown: 30.22 s compute (34.33%), 57.22 s idle (65.01%), and
0.58 s non-compute. Idle classification is predominantly host wait.

| Kernel | Total GPU time (s) | Interpretation |
|---|---:|---|
| `radius_search_limited_select_batched` | 19.70 | Dominant geometry neighborhood-search kernel |
| `_bvh_query_distance` | 2.22 | Geometry/BVH distance query |

GPU kernel classification was 98.0% computation and 1.9% memory.

## Source correlation and recommendations

1. **Geometry neighborhood search** — `geometry_rep.py` ball-query path near
   line 662 and multi-scale neighbor settings in `config.yaml`. Test reusing
   equivalent radius-search outputs where geometry semantics allow. Verify
   output/loss parity and remeasure unprofiled step time. Status:
   `recommendation_only`.
2. **BVH/SDF preparation** — `domino_datapipe.py:process_data` near line 663.
   Test DoMINO cached preprocessing for invariant geometry-derived features.
   Verify feature equality and loss parity. Status: `recommendation_only`.
3. **Host wait** — synchronous iteration at `train.py:222`, with preload depth
   one. Test a larger validated prefetch depth or cached dataset. Verify sample
   order/loss behavior and remeasure idle share plus unprofiled step time.
   Status: `recommendation_only`.

## Limitations

- Eager-only: no torch.compile comparison was requested.
- No Nsight Compute capture: HTA already identifies the kernel candidates; NCU
  needs separate capture-plan approval.
- HTA did not emit native ProfilerStep groups, so required report SVG timelines
  are deferred pending raw-trace step reconstruction.

## Artifact index

- `annotation-health-eager.json` — validated annotation health.
- `hta/temporal-breakdown.csv`, `hta/gpu-kernel-summary.csv`,
  `hta/idle-time-breakdown.csv` — HTA evidence.
- `runs/baseline-*/train.log` — unprofiled baseline logs.
- `scaling_factors/` — generated normalization factors.
