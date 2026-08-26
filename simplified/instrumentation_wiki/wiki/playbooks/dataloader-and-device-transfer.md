---
id: playbook-dataloader-and-device-transfer
title: Separate batch retrieval from device transfer
kind: playbook
status: published
trust: verified
version: 1
tags: [instrumentation, dataloader, device-transfer, validation]
evidence_ids: [manual-protocol-v1]
---

# Separate batch retrieval from device transfer

Annotate the call that retrieves the next batch independently from recursive
host-to-device movement. Do not label collation, preprocessing, or asynchronous
work as data wait unless the target loop makes that boundary observable.
