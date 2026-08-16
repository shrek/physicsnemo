# Minimal NOOA training optimizer

An isolated reimplementation of the ImproofBench training optimizer. It tries one
ranked hotspot and reports a speedup only after a deterministic benchmark and
correctness check.

```text
inputs proposal <-> critic
  -> smoke -> baseline
  -> instrumentation proposal <-> profile/trace critic
  -> hotspot analysis <-> LLM critic
  -> route -> change proposal <-> candidate benchmark/critic
  -> report
```

## Design principles

These follow the [NOOA object-oriented agent model](https://arxiv.org/pdf/2607.20709).

- **Python is the workflow.** Classes are agents, methods are nodes, `for` loops
  are retries, exceptions are failures, and `async` is orchestration.
- **Types are boundaries.** Agent methods exchange validated Python values by
  reference; prompts and dictionaries are not workflow state.
- **LLMs propose; Python verifies.** Semantic proposals use NOOA `Predict` or
  `CodeAct`; execution, routing, arithmetic, and acceptance use normal methods.
- **Feedback is explicit.** A rejected `Critique` is passed to the next proposal,
  and every loop has a hard attempt limit.
- **Authority follows capability.** Proposal agents receive only read-only source
  methods. The injected training environment owns execution and candidate isolation.
- **DRY and YAGNI.** There is one workflow, one environment boundary, no YAML/DAG
  engine, persistence layer, generic retry framework, or speculative extension API.

The implementation lives in `simplified`: `types.py` defines values,
`environment.py` defines external capabilities, `agents.py` defines nodes, and
`workflow.py` orders them.

```python
agents = create_agents(llm, source_environment, training_environment)
optimizer = TrainingOptimizer(agents, llm=llm)
result = await optimizer.run(TrainingRequest(description="..."))
```

## Trace viewer

NOOA's bundled viewer shows optimizer runs, nested agent methods, typed inputs and
outputs, retries, LLM turns, CodeAct execution, errors, tokens, and cost.

```bash
uv run python -m nooa.viewer
```

Enable tracing before constructing the agents, then give each run its own session:

```python
trace_to_viewer()
agents = create_agents(llm, source_environment, training_environment)
optimizer = TrainingOptimizer(agents, llm=llm)
run = await run_observed(optimizer, TrainingRequest(description="..."))
print(run.session_id)
```

Use `trace_to_files("traces")` instead when the viewer is offline. Source-reading
and environment implementations should avoid returning credentials or unbounded
logs; NOOA additionally scrubs recognized secrets before export.

## Debug one step at a time

Credentials and the model can stay in environment variables. For the NVIDIA
OpenAI-compatible endpoint:

```bash
export SIMPLIFIED_MODEL="openai/nvidia/moonshotai/kimi-k3-max-preview"
export SIMPLIFIED_API_BASE="https://inference-api.nvidia.com/v1"
export SIMPLIFIED_API_TOKEN="$INFERENCE_HUB_KEY"

uv run simplified-step hello --name PhysicsNeMo -o hello.json
cat hello.json
```

`SIMPLIFIED_API_KEY` is accepted as an alias. If neither variable is set, the
runner uses `INFERENCE_HUB_KEY` directly. Keep tokens out of `pyproject.toml` and
committed files; `uv run` inherits exported shell variables.

For a model exposed only through the OpenAI Responses API, explicitly select
NOOA's Responses client:

```bash
export SIMPLIFIED_MODEL="openai/switchyard/openai/gpt-4o-mini"
export SIMPLIFIED_API_BASE="https://your-gateway.example/v1"
export SIMPLIFIED_API_KEY="$SWITCHYARD_API_KEY"
export SIMPLIFIED_CLIENT_TYPE="responses"

uv run simplified-step hello --name PhysicsNeMo -o hello.json
```

The equivalent one-command override is `--client-type responses`. When neither
the flag nor `SIMPLIFIED_CLIENT_TYPE` is set, a NOOA registry alias may select
`client_type: responses`; otherwise NOOA defaults to the completion client.

Predict-based steps request JSON Schema when the endpoint supports it. The Kimi K3
Max preview endpoint is automatically run in text-output mode because it rejects
`response_format.type=json_schema`; its JSON text is still parsed and validated
against the declared Pydantic return type. Set
`SIMPLIFIED_STRUCTURED_OUTPUT=text` to force the same compatibility mode for
another text-only endpoint.

A training request may be JSON or plain text. For example, `request.json`:

    {"description": "Train GeoTransolver volume using examples/cfd/external_aerodynamics/unified_external_aero_recipe and the DrivAerML dataset at /mymount/ramu-data/datasets/drivaerml with model=geotransolver_volume, dataset=drivaer_ml_volume, and sampling_resolution=100000."}

The equivalent, easier-to-edit `request.txt` is:

    Train the GeoTransolver volume model.
    Training recipe: examples/cfd/external_aerodynamics/unified_external_aero_recipe
    Dataset path: /mymount/ramu-data/datasets/drivaerml
    Model: geotransolver_volume
    Dataset config: drivaer_ml_volume
    Sampling resolution: 100000

All later-stage inputs and every output remain typed JSON artifacts. Install the
standalone environment, then run a step with either request format:

```bash
uv sync
uv run simplified-step accept-inputs request.txt \
  --repo /path/to/training/repo --human-in-the-loop -o spec.json

# Lower-level proposal and review steps remain available for debugging.
uv run simplified-step propose-inputs request.txt \
  --repo /path/to/training/repo -o spec.json
uv run simplified-step review-inputs spec.json -o input-critique.json

uv run simplified-step smoke spec.json --repo /path/to/training/repo -o smoke.json
uv run simplified-step benchmark spec.json --repo /path/to/training/repo -o baseline.json
uv run simplified-step propose-instrumentation spec.json \
  --repo /path/to/training/repo -o instrumentation-plan.json
uv run simplified-step review-instrumentation spec.json instrumentation-plan.json \
  --repo /path/to/training/repo -o instrumentation-critique.json
uv run simplified-step profile spec.json instrumentation-plan.json \
  --repo /path/to/training/repo -o trace.json
uv run simplified-step run-all request.txt \
  --repo /path/to/training/repo --human-in-the-loop -o result.json
```

`accept-inputs` uses a dedicated acceptance agent to propose, review, and revise
the training specification. A deterministic contract critic first rejects malformed,
shell-based, or unresolved inputs, then command preflight verifies the working
directory, executable, and referenced script paths. With
`--human-in-the-loop`, every preflight-valid contract is displayed for final
review; Enter approves it and bypasses the LLM critic, while free-form correction
text returns to the proposer
for another revision. Human feedback starts a fresh bounded proposal cycle even
when the preceding proposal used the final automated attempt. Without the flag,
the separate single-pass LLM critic receives
a bounded evidence packet collected from the entry point and relevant configuration.
It checks
that smoke performs a bounded complete training update, benchmark is an unprofiled
representative steady-state run, and profile matches the benchmark workload while
capturing a bounded post-warmup window. Its feedback returns to the proposer. Input
proposal is capped at thirty CodeAct iterations; each proposal and review has a
300-second default deadline, configurable with `--agent-timeout`.

Instrumentation proposal uses the local generic `AdaptiveCodeActStrategy`, which
delegates unchanged execution chunks to NOOA's public `CodeActStrategy`. It starts
with 50 iterations. At a boundary, a separate tool-free semantic judge compares
the method objective, its previous checkpoint, and a bounded summary of recent
events. Material progress grants 10 more iterations; sufficient evidence triggers
a three-iteration finalization window; semantic stagnation stops the run. The
configured hard ceiling is 100 iterations. Conversation events and CodeAct session
locals are retained across chunks, and no installed NOOA source is modified.

After human or semantic acceptance, the agent runs the smoke command before returning
`spec.json`. Commands must be direct argv; shell executables, shell
operators, repository-mutating setup, and embedded repetition loops are not accepted.
Command construction failures are sent back to the LLM for automatic revision.
External failures such as an unresolved dataset, missing
hardware, or permissions trigger one free-form terminal clarification when
`--human-in-the-loop` is enabled; the LLM uses that text to revise and revalidate the
specification. Without that flag the workflow fails closed. HITL mode also fails
immediately when stdin is not an interactive terminal, so automated jobs never hang.

`working_directory` in `spec.json` is always relative to `--repo`. Commands execute
from that directory. For the GeoTransolver unified external-aero recipe, use:

```json
{
  "working_directory": "examples/cfd/external_aerodynamics/unified_external_aero_recipe"
}
```

After `accept-inputs` succeeds, its smoke test has passed. The next required step is
`benchmark`; the standalone `smoke` command remains useful for diagnostics.

Run `uv run simplified-step --help` for every step and its input order. LLM steps
require `--model` or `SIMPLIFIED_MODEL`. Complete NOOA traces are written to
`./traces` by default; use `--trace-dir PATH` to select another directory or
`--no-trace` to disable persistent tracing. `--previous critique.json` supplies
retry feedback.

A spec can carry YAML values that are needed at runtime but must not be committed,
such as a local dataset root. Each overlay targets an existing repository-relative
YAML file and is deep-merged only in the temporary worktree used by smoke,
benchmark, instrumentation validation, profile, and candidate benchmarks:

```json
{
  "config_overlays": [
    {
      "path": "examples/cfd/external_aerodynamics/unified_external_aero_recipe/datasets/dataset_paths.yaml",
      "format": "yaml",
      "merge": {
        "drivaer_ml": "/mymount/ramu-data/datasets/drivaerml"
      }
    }
  ]
}
```

The source checkout and its tracked configuration are never modified. This is the
right mechanism when a recipe reads a YAML file directly and does not support the
same value as a command-line override.


Add `--show-turns` to follow LLM turns, generated code, method/tool calls including
`InputCritic.review`, `Runner.preflight`, and `Runner.smoke`, bounded arguments,
and results live in the terminal. Its default compact view shows one-line activity
and payload sizes without dumping generated code or captured stdout. Add
`--turn-detail full` when the terminal needs complete payloads; the default trace
directory always preserves the complete trace. The live log is written to stderr, so
`-o -` remains machine-readable JSON on stdout. Colors are automatic for terminals;
use `--color always` or `--color never` to override detection. Live and file tracing
can be enabled together:

```bash
uv run simplified-step accept-inputs request.txt --human-in-the-loop \
  --repo /path/to/training/repo --show-turns \
  -o spec.json
```

Smoke commands succeed by exit status. Benchmark commands also succeed by exit
status; no final JSON line is required. A dedicated LLM node interprets their
captured stdout and stderr using scoped, bounded `log_info`, `search_log`, and
`read_log` tools. Those declared tools accept no filesystem paths, and unselected
log lines are not injected into the LLM prompt. The node returns a typed
`BenchmarkResult` containing steady-state step time plus a named correctness
metric. Selected log excerpts are sent to the configured LLM, so benchmark output
should still avoid secrets. Profile commands must emit a
`TraceResult`. Instrumentation and candidate patches are applied
in temporary Git worktrees, so their repository must be committed; trace files

`propose-instrumentation` asks for opt-in, behavior-preserving ranges around the
complete training step, dataloader wait, host-to-device transfer, forward, loss,
backward, and optimizer step. It also asks for a bounded five-iteration
steady-state capture after warmup and a final `TraceResult` JSON line.
`review-instrumentation` applies the proposed patch in a disposable Git worktree,
checks the profile command, runs `git diff --check`, and compiles every changed
Python file without running the expensive training workload. The subsequent
`profile` step is the runtime test: it executes the patched command, and the trace
critic verifies successful output, the trace path, summary, and required ranges.
created in the worktree are copied to `--artifacts` before cleanup.

The normal functional tests use real local processes but no external LLM:

```bash
uv run pytest
```

Set a model only when intentionally running the independently selectable real-LLM
tests:

```bash
uv run pytest -m llm -s
uv run pytest tests/test_steps.py::test_llm_hello -s

SIMPLIFIED_TEST_MODEL=another-model \
  uv run pytest tests/test_steps.py::test_llm_analyze_hotspots -s
```
