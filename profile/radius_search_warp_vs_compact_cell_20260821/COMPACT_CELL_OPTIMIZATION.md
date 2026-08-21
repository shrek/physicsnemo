# Compact Cell `radius_search` — Optimization Deep-Dive

**Kernel:** `radius_search` (`physicsnemo/nn/functional/neighbors/radius_search/kernels/compact_cell_points.cu`)
**Target GPU:** NVIDIA B200 (148 SM, CC 10.0)
**Baseline:** 2.046 ms, `--set full` capture, 200,000-point DrivAerML volume sample, radius=5.0, max_points=128 (see `REPORT.md` for the full warp-vs-compact-cell comparison this builds on)
**Run directory:** `profile/radius_search_warp_vs_compact_cell_20260821/`

This note goes one level deeper than `REPORT.md` §2.3/§2.6, focused only on `compact_cell`, with normalized stall percentages and concrete, line-cited code changes.

---

## 1. Where the time actually goes

`smsp__pcsamp_sample_count = 34,982` total samples for this kernel. Stall-reason breakdown (normalized, not raw counts):

| Stall reason | Samples | % of kernel |
|---|---:|---:|
| `long_scoreboard` | 23,083 | **66.0%** |
| `wait` | 5,148 | 14.7% |
| `not_selected` | 1,787 | 5.1% |
| `selected` (productive) | 1,414 | 4.0% |
| everything else | ~3,550 | ~10.2% |

And within `long_scoreboard`, one line dominates everything else combined:

| File:Line | Samples | % of kernel | % of all `long_scoreboard` |
|---|---:|---:|---:|
| `compact_cell_points.cu:306` | 19,756 | **56.5%** | 85.6% |
| `compact_cell_points.cu:247` | 844 | 2.4% | 3.7% |
| `compact_cell_points.cu:327` | 556 | 1.6% | 2.4% |
| `compact_cell_points.cu:328` | 453 | 1.3% | 2.0% |
| everything else | ~1,474 | ~4.2% | ~6.3% |

**The kernel spends 56.5% of all sampled cycles stalled on one dependency: waiting for the load at line 300 (`compact_points[candidate_offset]`) to return, consumed at line 306.** Only 4.0% of samples are "selected" (actually issuing). This is the single biggest lever in the kernel — bigger than every other finding in `REPORT.md` §2.6 combined.

```cuda
297  bool hit = false;
298
299  if (candidate_offset < cell_end) {
300      const CompactPoint compact_point = compact_points[candidate_offset];   // <- the load
301      point_id = compact_point.point_id;
302      const float px = compact_point.x;
303      const float py = compact_point.y;
304      const float pz = compact_point.z;
305      const float xdiff = px - qx;
306      const float ydiff = py - qy;                                          // <- 56.5% of kernel stalls here
307      const float zdiff = pz - qz;
308      dist_sq = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;
309      hit = dist_sq <= radius_sq;
310  }
```

Why this load is expensive: `l1tex__t_sector_hit_rate.pct = 85.29%` but `lts__t_sector_hit_rate.pct = 35.06%` (from `REPORT.md` §1) — most `compact_points` reads miss L2 and go to DRAM. This is expected: 200,000 points scattered across the point cloud, each visited by relatively few of the 200,000 queries, so there's little cross-warp reuse to exploit in L2. **This is a genuine DRAM-latency problem, not a coalescing problem** — the load itself (`candidate_offset = candidate_base + lane`, 32 lanes reading 32 consecutive 16-byte `CompactPoint` structs = one contiguous 512-byte region) is already about as coalesced as it can be.

Critically: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed = 0.068%` (§1 of `REPORT.md`). There is essentially unlimited spare DRAM bandwidth here — the problem is 100% latency, 0% bandwidth. That reframes the fix: don't reduce bytes moved, **hide the latency by keeping more independent loads in flight**.

---

## 2. Recommendations, ranked by impact

### Priority 1 — Software-pipeline the candidate load (double-buffer the per-chunk `compact_points` fetch)

The inner loop (`compact_cell_points.cu:292-337`) is fully serial per 32-candidate chunk: issue the load (line 300) → wait for it → compute distance → `__ballot_sync`/compact-write → loop back and issue the *next* load. Nothing overlaps the ~line 300 load latency with useful work from the next iteration. With 66% of the kernel's cycles already attributed to `long_scoreboard` and only 4% "selected", there is essentially no other work available to hide behind — the fix has to create more independent in-flight memory operations.

Concretely: issue the load for chunk `candidate_base + warp_size` *before* consuming/processing chunk `candidate_base`, so the next chunk's `compact_points[...]` load is in flight while this chunk's `__ballot_sync`/`__popc`/compaction-write executes:

```cuda
CompactPoint cur = (cell_start < cell_end) ? compact_points[cell_start + lane] : CompactPoint{};
for (int candidate_base = cell_start;
     candidate_base < cell_end && found < max_points;
     candidate_base += warp_size) {
    const int next_base = candidate_base + warp_size;
    CompactPoint next;
    bool has_next = next_base < cell_end;
    if (has_next) {
        next = compact_points[next_base + lane];   // prefetch issued before we consume `cur`
    }
    // ... process `cur` (distance test, ballot, compaction write) using the already-in-flight load ...
    cur = next;
}
```

This is the standard "issue-ahead" prefetch pattern — it does not reduce bytes moved (DRAM bandwidth is not the constraint, per §1 above), it increases memory-level parallelism so the L2-miss latency of chunk N+1 overlaps chunk N's compute/compaction instead of blocking issue.

**Evidence:** `smsp__pcsamp_warps_issue_stalled_long_scoreboard` = 66.0% of all samples, 56.5% on line 306 alone; `smsp__pcsamp_warps_issue_stalled_selected` = only 4.0%; `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` = 0.068% (bandwidth headroom is not the constraint — latency is).

**Expected impact:** This is the largest single-line stall in the kernel by a wide margin (>15x the next-largest line). Even partial latency-hiding here (e.g. cutting effective wait time in half) should be visible in `gpu__time_duration.sum`.

**Effort:** Medium. Requires restructuring the innermost loop to carry one candidate chunk of look-ahead state; register cost is small (one extra `CompactPoint` per lane, 16 bytes) and current occupancy has headroom (`launch__occupancy_limit_registers = 8` blocks/SM, matches `launch__occupancy_limit_warps = 8` — not tightly register-bound today, so a modest register increase is unlikely to drop occupancy).

### Priority 2 — Eliminate the separate `cell_counts` array; derive it from `cell_offsets`

`cell_offsets` is already built as an **exclusive prefix sum over the full `cell_counts` array** in the host code:

```python
# physicsnemo/nn/functional/neighbors/radius_search/_compact_cell_points_impl.py:280-284
cell_prefix = torch.cumsum(cell_counts, dim=0, dtype=torch.int32)
cell_offsets = torch.empty_like(cell_counts)
cell_offsets[0] = 0
if table_capacity > 1:
    cell_offsets[1:] = cell_prefix[:-1]
```

This means `cell_offsets[slot + 1] - cell_offsets[slot] == cell_counts[slot]` for every `slot < table_capacity - 1`. The kernel currently reads both arrays separately at `compact_cell_points.cu:289-291`:

```cuda
289  const int cell_start = cell_offsets[table_slot];
290  const int cell_count = cell_counts[table_slot];      // <- redundant: derivable from cell_offsets[table_slot + 1]
291  const int cell_end = cell_start + cell_count;
```

replacing this with `cell_end = cell_offsets[table_slot + 1]` turns a scattered gather into a **separate global array** (`cell_counts`, different base address, different cache line) into an access to an *adjacent element of the array already being read* (`cell_offsets[table_slot]` and `cell_offsets[table_slot + 1]` are 4 bytes apart — same cache line, essentially free after the first load; can even be issued as a single aligned `int2`/vectorized load).

This requires bounds-safety for `table_slot == table_capacity - 1`: either allocate `cell_offsets` with one extra sentinel slot (`cell_offsets[table_capacity] = total_points`, one extra `torch.cumsum` output element) or keep a 1-element fallback. Cheap to add on the host side.

**Evidence:** `compact_cell_points.cu:289` and `:291` together account for 813 + 344 = 1,157 stall samples (3.3% of the kernel) — smaller than Priority 1, but this is a **free elimination of a whole tensor and a whole scattered global load**, not a latency-hiding trick, so it's low-risk, low-effort, and stacks with Priority 1.

**Expected impact:** ~3% direct stall-time reduction, plus one fewer kernel argument, one fewer `total_points`-sized `torch.zeros` allocation, and one less array for `search_kernel`/`count_kernel` to keep coherent.

**Effort:** Low. Delete `cell_counts` tensor and `count_point_cells`'s `atomicAdd(&cell_counts[slot], 1)` (replace with an equivalent counter, or keep `cell_counts` only as the pre-`cumsum` build-time scratch buffer — it's `cell_offsets` that's redundant to carry into `radius_search`, not `cell_counts` itself, which is still needed to compute the prefix sum). Concretely: keep `cell_counts` as the count-kernel's write target and cumsum input (as today), but **stop passing `cell_counts` into `radius_search`** and read `cell_offsets[table_slot + 1]` instead.

### Priority 3 — Batch the per-cell offset lookup up front, like `table_slot` already is

Right now the kernel already batches one gather across all 27 cells before the serial loop — `lane_table_slot` is resolved once per lane (`compact_cell_points.cu:264-278`, `if (lane < 27) { ... lane_table_slot = table_lookup(...); }`) and then broadcast per-cell via `__shfl_sync` inside the loop (line 283-284). But the *next* gather — `cell_offsets`/`cell_counts` (or, after Priority 2, just `cell_offsets[slot]` and `cell_offsets[slot+1]`) — still happens **serially inside the loop**, once per `cell_index`, blocking that iteration:

```cuda
280  for (int cell_index = 0; cell_index < 27 && found < max_points; ++cell_index) {
281      const int table_slot = __shfl_sync(full_mask, lane_table_slot, cell_index);
...
289      const int cell_start = cell_offsets[table_slot];   // <- serial gather, once per loop iteration
290      const int cell_count = cell_counts[table_slot];
291      const int cell_end = cell_start + cell_count;
```

Apply the same trick already used for `table_slot`: have each of the (up to) 27 active lanes fetch its own `cell_offsets[lane_table_slot]` (and, pre-Priority-2, `cell_counts[lane_table_slot]`) **once, up front, in the same `if (lane < 27)` block**, alongside the `table_lookup` call — then broadcast both `cell_start`/`cell_end` per cell via `__shfl_sync` inside the loop, exactly like `table_slot` is broadcast today. This turns 27 serial single-lane-relevant gathers into one batch of up to 27 independent, concurrent lane loads, which the SM's multiple outstanding-load slots can overlap far better than one-at-a-time.

**Evidence:** builds on the same stall lines as Priority 2 (`compact_cell_points.cu:289/291`); combine the two changes rather than doing them separately.

**Expected impact:** Smaller than Priority 1, complementary to Priority 2 — mostly removes latency exposure at cell-loop start-up, which matters most for queries with few resolved cells (sparse regions) where loop overhead is proportionally larger.

**Effort:** Low — mechanical extension of the pattern already present in the kernel for `table_slot`.

### Priority 4 (lower priority) — Fuse the hash-table probe's scattered reads

`table_lookup` (`compact_cell_points.cu:63-92`, called from the `if (lane < 27)` block) reads `table_states[slot]`, then on a candidate match reads `table_batches[slot]` and `table_cell_coords[slot*3 + 0..2]` via `cell_equal` (`compact_cell_points.cu:47-61`) — four separate global arrays at the same (hashed, effectively random) `slot`, i.e. up to 4 separate cache-line fetches per probe. Stall lines `compact_cell_points.cu:57,58,78` total ~500 samples (1.4% of kernel) — real, but an order of magnitude smaller than Priority 1.

**Fix direction:** pack `table_cell_coords` (3×int64) and `table_batches` (1×int64) into a single 32-byte-aligned struct-of-4 array, so `cell_equal` becomes one coalesced 32-byte load instead of 4 scattered 8-byte loads. `table_states` must stay separate (it's polled with `atomicCAS`/busy-wait semantics in `table_lookup_or_insert`, `compact_cell_points.cu:108-122`, which doesn't fit into a packed read-only struct).

**Effort:** Medium — touches `count_point_cells`'s build-time insert path (`table_lookup_or_insert`) as well as the read-only `table_lookup` path used by `radius_search`, plus the host-side tensor allocation (`_compact_cell_points_impl.py:233-238`). Given the small evidence-based payoff (1.4% of kernel time), only worth doing after Priorities 1-3.

---

## 3. What *not* to chase here

- **DRAM bandwidth / coalescing on `compact_points` (line 300 itself):** already well-coalesced (32 consecutive lanes → one contiguous 512-byte region) and DRAM is at 0.068% of peak — there is no bandwidth problem to fix, only a latency one (Priority 1 addresses this directly).
- **Occupancy tuning:** `launch__occupancy_limit_registers = launch__occupancy_limit_warps = 8` (tied), theoretical occupancy is already 100%. The 30.64% achieved occupancy shortfall is a workload-imbalance/tail effect (`WorkloadDistribution/WorkloadImbalance` rule: SMs range from 77.68% above to 69.05% below average active cycles), not a launch-config problem — more registers/shared-mem headroom won't move this number.
- **Tensor cores / compute pipes:** not applicable; this is a gather/compare kernel, 0% tensor-core-shaped work, and compute pipes are already confirmed under-utilized (§2.4/§1 of `REPORT.md`) as a symptom of the same latency-bound story, not an independent finding.

---

## 4. Suggested validation loop

1. Apply Priority 2 (cheapest, mechanical, easiest to verify for correctness — pure derivation, no behavior change) and re-profile; confirm `compact_cell_points.cu:289/291` stall samples drop to ~0 and `gpu__time_duration.sum` drops by roughly the expected ~3%.
2. Apply Priority 3 on top; re-profile; expect a modest further reduction concentrated in low-candidate-count queries.
3. Apply Priority 1 (the real lever); re-profile; expect `long_scoreboard` on line 306 to drop substantially and `smsp__pcsamp_warps_issue_stalled_selected` (currently 4.0%) to rise. This is the one to validate most carefully — re-check `launch__occupancy_limit_registers` didn't regress from the added look-ahead state.
4. Re-run `helpers/extract_stall_hotspots.py` and `helpers/analyze_reports.py` against the new capture, tagged distinctly (e.g. `compact_cell_v2`) per the skill's directory convention — do not overwrite this run's `reports/`.
