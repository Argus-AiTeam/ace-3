# Stage11 exact-integer o_proj: software checkpoint, science unmeasured

Observation: **2026-09-28**, with exact time and live owner status in
[the progress snapshot](../continuation/progress-2026-09-28.json).

| Evidence | Observed outcome | What it does not establish |
|---|---|---|
| Latest S13-S17 composition `92e4223ef310` | One actual nine-control run; independently reviewed REJECTED; 0/9 S18 checks and 0/18 paired directions pass | A successful repair, full-model admission or permission to repeat the same composition |
| Exact-integer Stage11 implementation `d12e372181f0` | Four implementation/test files; normal independent review and native DONE | A compatible production entry or a numerical result |
| First science owner `7a2022cef21c` | Reviewer requested replanning; native failed before handoff/root/claim; zero science invocations | Numerical rejection of the hypothesis or a consumed numerical attempt |
| Software repair `aba2b6c0f3d0` | Running at snapshot | Accepted repair or fresh science execution |

## Mechanism and fixed scope

Replace the L23/P0 `self_attn.o_proj` binary64 dequantization/`torch.mv`
accumulation with exact-integer G128 reduction and one Q47.48-to-FP16
conversion. Keep the same retained Stage10 FP16 inputs and G128 INT4 tensors,
then recompute S12-S18, final RMSNorm and only the tied rows `[34319,13]`.
The prediction is a source-derived Stage11 difference that may improve the
unchanged downstream criteria. Zero Stage11 change or failure of those criteria
would reject sufficiency; the present entry failure has not tested that prediction.

Scope remains CPU-only width896/L23/P0, native S16 RTZ, FP16 operator/KV
boundaries and **wider Q24 residual state**, not strict-FP16-state W4A16.
References are read-only comparators, never inputs for choosing output bits.
Retained value-substitution ancestry is not validation of a native V repair.
There is no hardware, new-token, full-model, admission or chat-quality claim.

## Known defects and historical validation

The software checkpoint recorded 17 focused tests passing, but the subsequent
native execution context exposed two gaps: the wrapper expects a different
node key and no dependencies, and wrapper-first imports count 63 evaluator
members where 62 are required. Its Manager handoff also was not provisioned.
No irreversible claim or numerical execution occurred.

The source snapshot deliberately preserves the four reviewed bytes and records
these defects. It is not a release advertised as runnable. Fresh-process
integration and canonical mission compatibility must be repaired and
independently reviewed before a coherent fresh science owner is issued.
Do not mutate or revive the failed owner to make the history look successful.

Publication validation ran **16 synthetic/no-claim tests successfully** in an
isolated clone, excluding the one original-host retained-input test. The first
run on public `main` had 13 passes, two failures and one error at the hard-coded
branch guard. The successful run temporarily used the required branch name only
inside the publication clone; it neither changed the live repository nor removed
the guard. These tests use the original suite import order, not native entry order.

A separate fresh-process, import-only probe reproduced the disclosed defect:
64 wrapper source members, but **63 evaluator members instead of 62**.
Source hashes, compilation and these synthetic checks do not establish production
entry acceptance. No retained scientific computation or production claim was run
as part of publication.

## Publication boundary

[The manifest](../continuation/source-manifest-2026-09-28.json) pins 69
source/test files, including unchanged dependencies and companion tests.
The small [Stage16 summary](STAGE16_REJECTED_20260927.json) cites the original
result/review hashes; it is not an authoritative receipt.

Model weights, raw arrays, original handoffs/reviews, active authorities,
claims, provider transcripts and credentials remain host-only. Running
wrappers from this public clone is neither authorized nor a way to reproduce
closed experiments. Later repair edits are intentionally excluded.
The original September 26 source basis remains at its unchanged dated tag.
