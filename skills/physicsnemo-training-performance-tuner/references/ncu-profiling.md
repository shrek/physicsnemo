# Bounded Nsight Compute workflow

Use NCU only after HTA identifies a CUDA kernel on the critical path and the
kernel's limiting mechanism will change the recommendation. NCU is an optional
system capability: the skill requires an NVIDIA GPU, driver access to hardware
performance counters, and an installed `ncu` executable. It does not depend on
Colossus, Scope, or an NCU Python package.

## 1. Check the target

Run this on the GPU host or inside the workload container:

```bash
python scripts/ncu_profile.py doctor --output <bundle>/ncu/doctor.json
```

The command checks `--ncu`, then `PATH`, then `/usr/local/cuda/bin/ncu`. A
non-ready result is evidence to record, not permission to install software,
change driver policy, use `sudo`, or add container capabilities. A ready result
proves executable and GPU discovery only; it does not prove counter permission
for the planned workload and is not user approval to capture.

## 2. Record one HTA candidate

Add NCU selection fields to the relevant hotspot in `findings.json`:

```json
{
  "id": "H003",
  "category": "custom_math",
  "summary": "Radius search is on the steady-state critical path.",
  "kernel_regex": ".*radius_search.*",
  "nvtx_range": "neighbor_search/",
  "ncu_needed": true,
  "ncu_question": "Is the kernel limited by memory traffic, occupancy, or scanning?"
}
```

The trailing `/` in an NVTX push/pop filter is significant to NCU. Prefer a
kernel regex or NVTX range. If neither is stable, gate a short region in the
workload with `cudaProfilerStart`/`cudaProfilerStop` and plan with
`--profile-from-start off`.

## 3. Create and inspect a capture plan

Represent the workload as an argv array so it is executed without a shell:

```json
{
  "argv": ["python", "src/train.py", "model=geotransolver_volume"],
  "cwd": "/absolute/path/to/unified_external_aero_recipe",
  "env": {}
}
```

Do not put secrets in the environment object. Create the first-pass plan:

```bash
python scripts/ncu_profile.py plan \
  --hta-finding <bundle>/findings.json \
  --candidate-id H003 \
  --workload-json <bundle>/ncu/workload.json \
  --output-prefix <bundle>/ncu/H003-default \
  --spec <bundle>/ncu/capture-spec.json

python scripts/ncu_profile.py capture \
  <bundle>/ncu/capture-spec.json \
  --print-only
```

The default captures at most ten launches with NCU's `default` set. The
validator rejects `--set full`, more than 100 launches, and captures over ten
launches that have no kernel filter, NVTX filter, or profiler gate. Existing
reports are not overwritten unless the plan explicitly records
`force_overwrite`. `--print-only` works even when `ncu` is absent so the proposed
command can still be reviewed.

Show the exact print-only output to the user and ask for separate explicit
confirmation. After confirmation, bind it to the current spec fingerprint:

```bash
python scripts/ncu_profile.py approve \
  <bundle>/ncu/capture-spec.json \
  --output <bundle>/ncu/capture-approval.json \
  --confirmation-source 'user confirmed exact print-only plan'
```

Any spec change invalidates this approval and requires the plan to be shown and
confirmed again. If `ncu` is unavailable, record it and skip capture; do not
install it or change the platform unless the user separately authorizes that
operation.

## 4. Capture and summarize

Run:

```bash
python scripts/ncu_profile.py capture \
  <bundle>/ncu/capture-spec.json \
  --approval <bundle>/ncu/capture-approval.json

python scripts/ncu_profile.py summarize \
  <bundle>/ncu/H003-default.ncu-rep \
  --output <bundle>/ncu/H003-summary.json
```

The capture produces `capture-manifest.json`, `capture.log`, and the NCU report.
The summary preserves all CSV rows plus the report hash. Fill its
`interpretation` from the metrics relevant to the question.

Only after the first pass isolates a stable candidate, make a second plan with
one or more focused sections, such as:

- `SpeedOfLight`
- `SpeedOfLight_RooflineChart`
- `MemoryWorkloadAnalysis`
- `LaunchStats`
- `Occupancy`

Confirm section availability in `doctor.json`; section names vary by NCU
version. Keep the second pass narrow.

## 5. Failure and safety handling

- `gpu_performance_counter_permission` means the platform owner must decide
  whether to enable counter access. Do not change it automatically.
- `no_matching_kernel` means the regex, NVTX range, warmup, or launch window
  needs correction.
- Profile one rank/GPU first for distributed workloads. Use unique output
  prefixes per rank before expanding.
- Container capability changes such as `SYS_ADMIN` are privileged operational
  changes and require explicit user approval.
- NCU replay perturbs execution. Use it to explain a kernel, never to claim
  end-to-end speedup.
- Preserve the capture spec, fingerprint-bound approval, manifest, log,
  `.ncu-rep`, normalized summary, NCU version, and the HTA candidate that
  justified the run.
