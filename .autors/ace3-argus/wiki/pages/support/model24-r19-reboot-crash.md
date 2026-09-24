---
title: Model24 r19 reboot-crash terminal boundary
description: Immutable r18 crash ancestry and exit-code-free infrastructure-crash sealing for Model24 r19.
---

# Model24 r19 reboot-crash terminal boundary

The r19 package preserves the complete r18 package, accepted review, original
and consumed authority, durable-runner terminal tree, materialized controller,
and partial output tree as recursively read-only ancestry. The ancestry binds
the consumed authority and crashed receipt, treats layers 0 through 8 as
authenticated partial evidence, and classifies layer 9 as incomplete because
it has no completion record. No r18 nonce, task, review, authority, output, or
terminal namespace is reusable by r19.

The lifecycle finalizer retains the strict sidecar and launch-terminal contract
for normal `done`, `error`, and `timeout` receipts. A `crashed` receipt may
instead seal `INFRASTRUCTURE_CRASH` only when its task, run, process, command,
working directory, logs, and expected sidecar path are consistent, the sidecar
is absent, and no launch terminal exists. The terminal records the exit code as
unobserved and `null`; it never fabricates an exit status.

Preparation validates all inherited fixtures plus reboot-crash positives and
negatives, compiles layers 0 and 23 from recursively read-only source into a
fresh external post-seal build root without executing either binary, and
requires canonical review, authority, output, receipt, logs, and terminal
namespaces to remain absent.
