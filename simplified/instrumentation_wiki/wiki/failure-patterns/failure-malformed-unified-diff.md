---
id: failure-malformed-unified-diff
title: Malformed unified Git diff
kind: failure_pattern
status: published
trust: verified
version: 1
tags:
- instrumentation
- validation
- failure
applies_when: {}
evidence_ids:
- run-geotransolver-malformed-diff-20260826
---

# Malformed unified Git diff

## Symptom

Instrumentation preflight rejects the proposed patch with `patch fragment
without header`, often because the response begins with an `@@` hunk rather
than a complete unified diff.

## Guard

Return one complete patch beginning with a `diff --git a/<path> b/<path>`
header, followed by `---` and `+++` file headers and only then hunk markers.
Before finalizing, verify that every hunk belongs to a declared changed file.

## Applicability

This is a format failure independent of the target recipe. It does not imply
that the proposed instrumentation boundaries are wrong.
