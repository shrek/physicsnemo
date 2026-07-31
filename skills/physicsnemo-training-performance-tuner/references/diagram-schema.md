# HTA report diagram contract

Generate diagrams from normalized HTA evidence with:

```bash
python scripts/render_trace_diagrams.py \
  --input <bundle>/hta/<variant>/diagram-data.json \
  --output-dir <bundle>/hta/<variant>/diagrams
```

Run the renderer once for `eager` and once for `compiled`. The renderer uses only
Python's standard library and emits self-contained SVG.
Do not hand-enter timings inferred from screenshots. Every span must come from
the selected trace and use milliseconds relative to its declared boundary.

## Required views

1. `cpu-gpu-pipeline.svg`: CPU and GPU lanes for one representative step,
   including stage labels, duration labels, idle regions, and annotations.
2. `forward-dominant-kernels.svg`: one or more forward-pass timelines grouped by
   kernel family and ordered by aggregate GPU time.

## Conditional views

- `in-step-gpu-bubbles.svg`: generated when GPU spans contain `idle` or `bubble`
  categories.
- `multi-step-gpu-bubbles.svg`: generated when `multi_step` contains at least
  two consecutive steps. Use this to expose inter-step dataload starvation and
  sample-size variability.

Use stable categories where possible:

```text
CPU/GPU phases: data, transfer, forward, backward, optimizer, communication
Idle: idle, bubble
Kernel families: radius_search, bvh_sdf, gemm, attention, triton,
                 reduction, memory, other
```

## Input shape

```json
{
  "schema_version": "0.1",
  "pipeline": {
    "title": "GeoTransolver volume: CPU/GPU pipeline",
    "step_id": "ProfilerStep#3",
    "duration_ms": 825.8,
    "summary": "GPU busy 42.2%; dataload is the largest bubble.",
    "lanes": [
      {
        "name": "CPU",
        "spans": [
          {
            "label": "dataload",
            "start_ms": 0.0,
            "end_ms": 412.2,
            "category": "data"
          }
        ]
      },
      {
        "name": "GPU",
        "spans": [
          {
            "label": "idle before batch",
            "start_ms": 0.0,
            "end_ms": 344.0,
            "category": "bubble",
            "annotation": "CPU waits for dataload"
          }
        ]
      }
    ]
  },
  "forward": {
    "title": "Forward-pass dominant kernels",
    "steps": [
      {
        "step_id": "ProfilerStep#3",
        "duration_ms": 225.8,
        "kernels": [
          {
            "label": "radius_search",
            "family": "radius_search",
            "category": "radius_search",
            "start_ms": 8.0,
            "end_ms": 18.7
          }
        ]
      }
    ]
  },
  "multi_step": {
    "title": "GPU compute and dataload bubbles",
    "steps": [
      {
        "step_id": "step 3",
        "start_ms": 0.0,
        "end_ms": 657.0,
        "gpu_busy_start_ms": 0.0,
        "gpu_busy_end_ms": 352.9
      },
      {
        "step_id": "step 4",
        "start_ms": 601.7,
        "end_ms": 1258.7,
        "gpu_busy_start_ms": 601.7,
        "gpu_busy_end_ms": 1019.5
      }
    ]
  }
}
```

The renderer validates required CPU/GPU lanes, span bounds, forward kernels,
and ordered multi-step windows before writing files. Preserve
`diagram-data.json`, generated SVGs, and `diagram-manifest.json`.
