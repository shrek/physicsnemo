---
id: playbook-explicit-pytorch-loop
title: Explicit PyTorch training loop
kind: playbook
status: published
trust: verified
version: 1
tags: [instrumentation, pytorch, kineto, explicit-loop]
applies_when:
  framework: pytorch
  loop_shape: explicit
evidence_ids: [manual-protocol-v1]
---

# Explicit PyTorch training loop

Locate batch retrieval, device transfer, model invocation, loss computation,
backward, and optimizer update in the existing loop. Use the recipe-owned
profiler when one exists. Keep profile-disabled execution a no-op, place exactly
one profiler step at each logical training step, and avoid moving work merely to
make ranges easier to emit.
