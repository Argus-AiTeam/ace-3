---
title: Position-3 traversal evidence boundary
description: Canonical vector paths, build identity, and validation-time replay required for complete position-3 traversal evidence.
---

# Position-3 traversal evidence boundary

Complete position-3 traversal evidence binds all 24 ordered decoder layers to
the accepted position-3 input and position-2 K/V parentage. Each layer requires
the exact 26 checkpoint tensors, checkpoint metadata and values, canonical
serialized bytes, and the position-3 RoPE coefficients.

The Verilator harness consumes fixed names beneath each canonical
`layerNN/vectors` directory: `manifest.json`, `inputs.hex`,
`rope_coefficients.hex`, and the layer-derived files under `tensors/`.
Authenticated records must point to those exact paths. A content-valid record
for an alternate path cannot authenticate a substituted file at the path the
harness actually opens.

Validation performs fresh current-worktree builds and replays each canonical
binary into temporary outputs. Stored compile and simulation logs, terminal,
trace, state, output, terminal counts, activation lineage, and exact integer
oracle comparison must match the fresh replay before evidence can retain
`COMPLETE` status.

The accepted-v10 continuation package at
`build/model24_selected_token_position3_continuations/v10-accepted-review-gated-r3`
prepares 26 checkpointed transactions without executing them: the position-2
final RMSNorm/tied `lm_head`, 24 ordered position-3 decoder layers, and the
position-3 final RMSNorm/tied `lm_head`. Its atomic resume manifest starts at
transaction 0 and preserves each authenticated completed unit instead of
recomputing it. The package binds the accepted v10 receipt, all 72 raw
terminals, layer-23 hidden output, all 24 position-2 K/V states, official
checkpoint, fixture manifests, tied weight, and continuation source closure.

Every completion receipt must satisfy its transaction's required result
semantics. Transaction 0 propagates its authenticated selected token, FP16
logit, checkpoint-bound tied embedding, and embedding semantic hash into
layer 0. Each decoder transaction propagates authenticated hidden and
same-layer position-4 K/V state records into the next ledger binding before
the resume cursor is atomically replaced. Validation supports inert,
authorized-ready, in-progress, and complete prefixes while rejecting gaps,
stale bindings, missing receipts, and semantic mismatches. If interruption
occurs after the first pending transaction's valid receipt is durably installed
but before the resume manifest replacement, restart authenticates and adopts
that checkpoint, then advances to the next incomplete transaction without
recomputing the completed unit.

The package is not execution authority and contains zero completion receipts
or runtime-shaped outputs. Only an independent Reviewer PASS binding the exact
package manifest and v10 acceptance receipt can authorize a later bounded
continuation execution.
