---
id: failure-duplicate-profiler
title: Duplicate profiler lifecycle
kind: failure_pattern
status: published
trust: verified
version: 1
tags: [instrumentation, profiler, failure]
evidence_ids: [manual-protocol-v1]
---

# Duplicate profiler lifecycle

Creating a new profiler around a recipe that already owns a profiler can change
trace scheduling, output paths, overhead, and training behavior. Inspect the
existing lifecycle first and extend it only when profiling is enabled.
