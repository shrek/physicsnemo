# NCU artifact contract

The NCU subdirectory contains:

| Artifact | Required when NCU runs | Purpose |
|---|---:|---|
| `doctor.json` | yes | Tool, version, section, and GPU visibility |
| `workload.json` | yes | Shell-free workload argv, cwd, and scoped env |
| `capture-spec.json` | yes | Reviewable filters, launch bound, and question |
| `capture-manifest.json` | yes | Exact argv, result, failure class, and paths |
| `capture.log` | yes | NCU stdout and stderr |
| `<candidate>.ncu-rep` | successful capture | Native report |
| `<candidate>-summary.csv` | successful import | Raw NCU CSV export |
| `<candidate>-summary.json` | successful import | Normalized evidence |

`capture-spec.json` has this shape:

```json
{
  "schema_version": "0.1",
  "candidate": {
    "id": "H003",
    "reason": "Candidate kernel is on the critical path.",
    "question": "Is it memory, occupancy, or latency limited?",
    "source": "../findings.json"
  },
  "selection": {
    "kernel_regex": ".*radius_search.*",
    "nvtx_include": "neighbor_search/",
    "profile_from_start": true,
    "launch_skip": 0,
    "launch_count": 10
  },
  "collection": {
    "set": "default",
    "sections": [],
    "target_processes": "all",
    "replay_mode": null
  },
  "workload": {
    "argv": ["python", "src/train.py"],
    "cwd": "/absolute/workdir",
    "env": {}
  },
  "output": {
    "prefix": "/absolute/bundle/ncu/H003-default",
    "force_overwrite": false
  }
}
```

The JSON summary deliberately keeps raw metric rows because metric names and
section contents vary across NCU versions. Its `interpretation` is a small,
reviewable conclusion written after inspecting the relevant metrics. It must
not contain a measured speedup derived from NCU replay timing.
