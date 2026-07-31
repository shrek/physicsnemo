# Trace annotation health

Validate annotation integrity before HTA metrics or diagrams. A trace that opens
successfully can still have unusable step boundaries, duplicated projected
annotations, or inconsistent interval ends.

## Capture contract

Use an explicit profiler schedule that produces active `ProfilerStep#N` ranges.
When the workload's profiler wrapper does not guarantee those ranges, wrap the
entire logical iteration in an explicit `train_step` range. Keep the canonical
outer phase ranges outside compiled functions when possible:

`dataloader_wait`, `feature_construction`, `host_to_device`, `forward`, `loss`,
`backward`, `optimizer_step`, `distributed_sync`, `validation`, and
`checkpoint`.

Select the phases required by the actual workload. Record optional or combined
phases as not applicable in the phase-to-source map rather than requiring fake
ranges.

## Health gate

Run before HTA analysis:

```bash
python scripts/validate_trace_annotations.py \
  --trace <bundle>/traces/<variant>/rank0.json \
  --output <bundle>/annotation-health-<variant>.json \
  --run-id <run-id> \
  --variant <eager-or-compiled> \
  --expected-steps <active-step-count> \
  --compile-log <bundle>/logs/profile.log
```

Repeat `--required-phase <name>` to override the default core phase set. If HTA
annotation rows have been exported, pass a CSV containing `ts`, `dur`, and
optional `end` columns with `--hta-rows`.

Recapture when required phases or a trustworthy step boundary are missing.
Warnings that `torch.compile` ignored a `record_function` do not by themselves
invalidate the trace: verify the required outer ranges directly in the raw
trace and record the warning.

## Bounded reconstruction fallback

Use `--allow-reconstruction` only when recapture is impractical and all required
canonical phase anchors are present. The fallback contract is:

1. Use only original `user_annotation` rows for CPU phase intervals. Exclude
   projected `gpu_user_annotation` rows to prevent duplicate intervals.
2. Derive each interval end as `ts + dur` when HTA's `end` is missing,
   non-finite, before `ts`, or inconsistent with the duration.
3. Reconstruct logical steps by canonical phase order, bounded from
   `dataloader_wait` through `distributed_sync`.
4. Attribute GPU kernels through HTA `index_correlation` launch links.
5. Mark the boundary provenance `reconstructed`; never label it a native
   `ProfilerStep`.

The fallback must fail closed when a required phase is absent or a duration is
invalid. Validate eager and compiled traces independently; do not reuse one
variant's health result for the other. Each annotation-health artifact records the selection rule, duplicate
annotations, compile warning, timestamp checks, kernel attribution method, and
whether recapture remains required.
