---
title: Position-2 terminal status boundary
description: Current canonical evidence, consumed-failure, and successor-advancement limits for position 2.
---

# Position-2 terminal status boundary

ACE-3 has independently accepted canonical position-2 runtime PASS evidence.
Fresh v10 identity
`ace3-position2-fresh-v10-20260831t110456z` reached a natural zero-exit
terminal after invoking the accepted extracted canonical validator once for
`generate` and once for `validate`. Its candidate evidence records 24 ordered
causal layers, 72 position-0/1/2 raw natural-terminal records, exact
integer-oracle agreement, FP16 K/V parentage, and the post-layer-23 hidden
hash. The independent scoped v10 runtime review accepted those records. The
hash-complete acceptance receipt at
`build/model24_selected_token_position3_continuations/v10-accepted-review-gated-r2/v10_acceptance_receipt.json`
binds the review, terminal, runtime submission, canonical evidence, durable
runner receipt, all 72 raw terminals, and layer-23 hidden parentage. Acceptance
did not rerun or mutate v10.

V10 is distinct from v9: v9 exercised a rehearsal-only driver for one RTL
transaction with zero canonical-validator invocations, while v10 ran the
source-bound validator through the full 24-layer position-2 traversal.

The earlier reviewed
fresh v6 authority was consumed once: one durable submission reached one
payload, driver, and generate invocation, then ended naturally with a
non-retriable failure before validation. The read-only terminal seal records
one failure verdict and zero retry, replay, resume, relaunch, or validator
invocations.

Layer 0 compiled, but its first position-0 simulation stopped while loading
replay vectors because `trace.hex` was absent. The same simulator input
directory also lacked its required `final.hex` and `boundary_manifest.json`.
The authenticated source archive used the published
`a4809cf9517b6436ad8f22fde02b8677ea4f2612` harness, whose legacy constructor
still opened those oracle files before accepting a transaction. The fresh
position-2 validator generated only `inputs.hex` and `rope_coefficients.hex`
for each replay because raw trace and final output are captured from the live
transaction and compared afterward by its independent integer oracle.

The corrected worktree keeps those oracle vectors as explicit transaction
inputs, writes them from the independent integer oracle before invoking
`--vector-dir`, and checks their records in the live harness while separately
capturing raw RTL outputs. The focused regression builds a fresh
extracted-source Verilator executable with an external Mdir, runs it through
the first accepted trace, and reproduces the v6 `cannot open .../trace.hex`
failure after deleting that file. All 19 focused checks pass.

A fresh v7 package now seals the corrected 30-file consumed-source closure and
an exact source archive. Its independent package review passed without
creating runtime output or granting execution. The first generated v7
authority was rejected before execution because its preflight required the
bound `/usr/bin/g++` path itself not to be a symlink. It remains unconsumed and
non-authoritative for runtime.

A distinct v8 execution authority bound the accepted v7 package, reviewed
source archive, exact runtime driver, resolved regular tool executables, and
project-root durable-runner invocation. Its L2 authority review passed at zero
authority consumption, durable submission, payload, runtime, validator, and
terminal counts.

The sole v8 durable submission then ended naturally after 0.7 seconds with
exit code 2. The durable runner had created its exact task receipt before
starting the child, while the child launch-time freshness gate required that
receipt to remain absent. The driver therefore sealed a non-retryable terminal
with one submission and one driver invocation, but zero authority consumption,
payload, generate, validate, vector, or runtime-output counts. The v8 terminal
is awaiting independent review and cannot support a position-2 runtime claim
or a relaunch of the same identity.

The consumed v6 identity cannot be reused or validated retroactively.
Position-2 `lm_head` and position-3 preparation are unblocked by v10
acceptance, but execution remains withheld until the separate continuation
package receives an independent review PASS. Dialogue has not advanced.

The exact project-relative evidence paths, hashes, consumed lifecycle counts,
source identities, and zero-advancement seal counts are bound by
`ace3/contracts/position2_canonical_evidence_status.json`.
