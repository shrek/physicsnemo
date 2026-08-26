# Instrumentation knowledge base

This is a small, generic knowledge base for profiling training recipes. It has
three trust boundaries: immutable source observations under `sources/`,
retrievable reviewed guidance under `wiki/`, and manual/agent-authored drafts
under `candidates/`. Proposers read only published verified pages. They may
append untrusted observations to run-local memory but cannot modify this wiki.

Use `instrumentation-knowledge validate`, `search`, `show`, and `promote` to
manage the store. New pages start in `candidates/` with `status: draft` and are
promoted only after validation and review.
