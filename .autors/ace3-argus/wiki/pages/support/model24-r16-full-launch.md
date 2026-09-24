---
title: Model24 r16 full durable-launch boundary
description: Exact-source overlay, production preflight, ancestry, review, and authority constraints for the full Model24 r16 launch package.
---

# Model24 r16 full durable-launch boundary

The r16 package materializes accepted commit
`42c895ce1e5fea00525e9f2f7fef66f0fbb8e118` and changes exactly four
authenticated compile-closure paths: `Makefile`, the decoder layer-0 RTL, the
SiLU gate RTL, and the decoder layer-0 C++ testbench. The sealed r15 successor
v2 package is compile-only provenance for those overlays; it is neither an
executable predecessor nor a package that may be wrapped for launch. The r16
seal binds v2's package manifest, seal, review request, validator, source-tree
manifest, internal source-record tree hash, and layer-0 compile result.

Before review or Model24 execution, production preflight derives the actual
per-layer Make command from the sealed cascade controller, resolves the Make
target at layers 0 and 23, cleans prior build state, and compiles accurate-SiLU
layer 0 from the exact materialized source. The resulting executable and
preflight logs are hash-bound inside the package. The source tree sealed after
preflight contains no generated build state.

The package preserves r12 and r13 terminal provenance and byte-preserving r14
and r15 terminal ancestry. r14 and r15 executions are classified as
unauthorized because no Manager directive existed; their reviews and
authority documents cannot be reused. A future independent L2 review must bind
the full package and report all six positive fixtures and 48 negative cases.
Only a later exact Manager authority matching the sealed authority schema can
authorize one launch.

Preparation leaves the nonce-qualified r16 review, authority, consumed
authority, terminal root, durable receipt and logs, output root, controller
simulation directory, and RTL cascade directory absent.
