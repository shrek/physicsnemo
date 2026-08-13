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

The implementation lives in `src/simplified`: `types.py` defines values,
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

```bash
uv run pytest tests/test_simplified_agents.py tests/test_simplified_workflow.py \
  tests/test_simplified_observability.py
```
