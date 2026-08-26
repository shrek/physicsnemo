---
id: playbook-reuse-existing-profiler
title: Reuse an existing recipe-owned profiler
kind: playbook
status: published
trust: verified
version: 1
tags: [instrumentation, profiler, kineto, profile-output]
evidence_ids: [manual-protocol-v1]
---

# Reuse the recipe-owned profiler

Extend an existing profiler lifecycle with opt-in annotation ranges and preserve
its trace schedule and result-output protocol. Do not add a second profiler,
change the training command's normal output, or use profiled wall time as a
benchmark result.
