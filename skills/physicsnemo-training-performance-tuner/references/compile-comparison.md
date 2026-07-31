# Eager versus torch.compile comparison

Treat eager and compiled execution as a mandatory paired experiment. The goal is
to determine whether compilation improves the representative workload and, when
it does not, whether graph breaks, recompilations, backend fallback, compilation
cost, or an unchanged critical path explains the result.

## Paired configuration

Show and confirm two complete commands:

- `eager`: `torch.compile=false`
- `compiled`: `torch.compile=true`

Hold source, local patch, dataset and sample order, sampling resolution, seed,
batch size, precision, optimizer, GPU topology, distributed settings, warmup,
measurement window, and correctness signal invariant. Record the compiled
backend, mode, `fullgraph`, dynamic-shape setting, and compile-cache state. If
another setting must differ, disclose it and classify the comparison
`inconclusive` unless its effect is isolated.

## Measurement sequence

1. Run correctness for eager and compiled variants.
2. Run unprofiled eager steady-state repetitions.
3. Run one compiled cold-start measurement and record compilation/first-step
   cost separately.
4. Warm the compiled path until compilation and intended recompilations finish.
5. Run unprofiled compiled steady-state repetitions.
6. Capture bounded Kineto traces for both variants over the same logical steps.
7. Validate each trace independently with the annotation-health gate.
8. Run compile diagnostics separately from timing and Kineto capture.

Do not include compilation, verbose compiler logging, or profiler overhead in
steady-state speedup. Report the median and dispersion for both variants.
Calculate amortization steps only when compiled steady-state is faster:

```text
amortization_steps = compile_time / (eager_step_time - compiled_step_time)
```

## Compile diagnostic run

Start with a bounded representative run:

```bash
TORCH_LOGS=graph_breaks,recompiles <compiled-command> \
  > <bundle>/logs/compiled/torch-compile.log 2>&1

python scripts/analyze_compile_logs.py \
  --log <bundle>/logs/compiled/torch-compile.log \
  --output <bundle>/compile-analysis.json \
  --run-id <run-id>
```

Use `torch._dynamo.explain` only when a representative callable and inputs can be
exercised without changing workload semantics. Use `TORCH_TRACE` and `tlparse`
when available and needed; do not install extra tools without approval. Record:

- Unique graph breaks by source location and reason, plus occurrence counts.
- Recompilations by function and failed guard.
- Recompilation/cache-limit warnings.
- Backend failures or eager fallback.
- Compile latency and cache state.
- Diagnostic commands, logs, and tool versions.

No logged graph breaks is not proof of a single full graph. Use `fullgraph=True`
or an appropriate graph-break error policy only in a separate diagnostic
experiment when failure is acceptable.

## Trace comparison

Compare:

- Step time and throughput from unprofiled baselines.
- CPU overhead, GPU busy/idle time, and exposed host gaps.
- Kernel and CUDA launch counts.
- Dominant kernel families and critical-path phase.
- Fusion or compiled-region boundaries.
- Memory allocation and transfer behavior.

Compiled traces may omit nested `record_function` ranges. Keep outer logical
step and phase ranges outside compiled functions when possible, validate both
traces independently, and disclose reconstructed boundaries.

## Classification and routing

Classify the paired result:

- `beneficial`: correctness passes and compiled steady state is materially faster.
- `regression`: correctness passes and compiled steady state is materially slower.
- `neutral`: correctness passes and the difference is within the declared noise
  threshold.
- `inconclusive`: correctness fails, invariants differ, evidence is incomplete,
  or compilation never reaches a stable steady state.

The performance tuner lists and explains graph breaks, recompilations, and
fallbacks but does not rewrite code to fix them. Recommend a dedicated
torch.compile diagnostic or implementation skill only for deeper Dynamo,
AOTAutograd, Inductor, guard/dynamic-shape, custom-op, minifier, or compiler
correctness work.
