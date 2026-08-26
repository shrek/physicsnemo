---
id: contract-instrumentation-ranges
title: Canonical instrumentation ranges
kind: contract
status: published
trust: verified
version: 1
tags: [instrumentation, profiler, validation]
evidence_ids: [manual-protocol-v1]
---

# Canonical instrumentation ranges

Emit a range only around work that actually occurs. `dataload` covers iterator
retrieval only; `train_step` excludes it and covers device transfer through the
optimizer. `forward`, `loss`, `backward`, and `optimizer` have non-overlapping,
source-supported boundaries. Keep aliases at retrieval time only; traces use the
canonical names configured by the workflow.
