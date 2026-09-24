# Decoder/model-root Q24 adoption specification

Attempt `q24_decoder_adoption_c39ca7b7722a_attempt001`, mission `c39ca7b7722a`.
**Specification selected for normal independent Host review; not runtime adoption.**
The adjacent JSON freezes authority, policy/source/reference identities and the
absence of an admitted Q24 model parent. This is the requested substantive
specification, not a launch receipt or a new request to resolve the old wait.

## Controlling authority and retained evidence

The operator's Qwen/Qwen2.5-0.5B-Instruct-AWQ objective covers computer-local
RTL/model implementation, simulation, host/runtime integration and persistent
KV, with W4A16 first. The supervising Manager's wait resolution at
`1788937474.7925315` explicitly resolves the stale general implementation wait
and retains HOLD in the RTL stage. Its objective fingerprint matches the older
`authorize_bounded_semantics_specification` directive at `1788933417.9750104`.
The JSON binds both actual records, not an Engineer-authored substitute verdict.

The existing `ce914530c478` binding and its genuine `round-0001.json` review
apply that resolution to an **isolated primitive/state route only**.
The current plan-7b926b45d9ac v3 mission selects the **decoder/model-root
adoption specification** as its next, separate bounded artifact. That selection
extends the specification's subject, not attempt003's authority or results.
The later wait resolution is not itself a decoder arithmetic/reference adoption
or permission to ignore this mission's explicit no-execution restriction.
No new human answer, Manager decision, Reviewer verdict or runtime grant is
issued here. No same-level unresolved objective conflict is asserted: the
current mission can be completed without executing the prospective cone.

The genuine primitive review is `a8b9021c084b/round-0002.json`, created at
`1788939741.4158454`, producer `reviewer`, status `done`. It postdates
attempt003; round 1 requested further evidence and is not the acceptance.
Retain attempt003's isolated 64,818-row/one-simulator result and its older
pending-review field unchanged. This mission does not rerun it or promote its
fixtures/codec payloads into model state. Existing primitive policy/state/ABI
IDs keep their isolated scope. New decoder policy/state IDs in the JSON are
prospective opt-in specifications, not replacements for those files.

The independently reviewed `a79ca09169d1/round-0001.json` supports the retained
single-round P0/L0-L8 accepted prefix and L9/P0/S18/index62 mandatory global
FAIL. This is attributed retained evidence, not a fresh numerical measurement.
The old four-suffix result and every failed checker/attempt remain immutable.
Neither correct local RNE nor primitive acceptance repairs that trajectory.

## General recurrence and precision

For each coordinate, let `B(x)=2^24*val_FP16(x)` for finite binary16 x.
Authoritative state `(I,Z)` is a signed 64-bit Q24 integer and a negative-zero
tag. Define `A((I,Z),x)=(J,Znew)` with exact `J=I+B(x)`. `Znew=1` precisely
when `J=0`, `I=0`, `Z=1` and x is negative zero; otherwise it is zero.
`Q(I,Z)` is binary16 round-to-nearest ties-to-even of `I*2^-24`, using Z
for exact zero. No coordinate-dependent correction or separately editable
carry exists.

```text
(I0,Z0)          = (B(authenticated embedding E), is_negative_zero(E))
H_l             = Q(I_l,Z_l)
(O_l,K_l,V_l)    = ATTENTION16_l(H_l, own_layer_prior_KV, canonical_controls)
(T_l,ZT_l)       = A((I_l,Z_l), actual_S11_O_l)
R_l             = Q(T_l,ZT_l)                              [S12]
D_l             = MLP16_l(actual_R_l)                      [S13-S17]
(I_next,Z_next)  = A((T_l,ZT_l), actual_S17_D_l)
H_next          = Q(I_next,Z_next)                         [S18]
```

The root loader first produces I0/Z0 from the actual embedding, not an oracle.
The current layer owns T/ZT, rounded R and D; successful completion produces
paired I_next/Z_next/H_next. The next residual operand is I_next, never
B(H_next). Similarly T is not replaced by B(R). Unlike the old
`RNE16(H+O+D)`, information discarded by an inter-layer FP16 view survives.
The derived quantity `(I-B(H))*2^-24` is diagnostic only, never an input.

FP16 decoding is exact on the Q24 grid, including subnormals; no flush to zero.
B(x) fits signed 41 bits. Sign-extend for a checked signed 65-bit addition,
then narrow only after checking signed 64-bit range. Reject nonfinite operands,
noncanonical tags, participating X/Z, integer overflow, and RNE16 infinity.
Require `abs(I)<65520*2^24` at every observable/restorable FP16 view; the
65520 tie rejects. Values between 65504 and 65520 may round to finite 65504.
This does not change the separate global-reference limit of 65504. Nonzero
cancellation produces +0; only negative-zero plus negative-zero preserves -0.
Fault placeholders, saturation and partial outputs are never admissible data.

Weights remain native G128 asymmetric packed INT4, GEMM nibble order
`[0,4,1,5,2,6,3,7]`, no qzero plus-one. Scales, canonical biases/norm/RoPE
inputs, all operator activations, materialized H/R and KV remain FP16.
Non-residual operator temporaries/rounding remain the source-bound accepted
single-round implementation, not arbitrary host float arithmetic.
Persistent residual and scratch T are **64+1 bits**, additions 65 bits:
this is wider hidden state, **not unchanged strict W4A16**, model completion,
or promotion to another precision. Width, grid, activation/KV encoding, weight
format and policy IDs are separate fields; only the declared mode is specified.

Prospective rationale: root plus the 20 O/D terms through ten layers is an
exact grid sum; `21*65504<2^21`, comfortably within the stated integer width.
Preserving residual rounding remainders can reduce discarded information but
does not ensure better nonlinear trajectories or any numerical PASS. No global
reference, known failing coordinate or desired output selects the correction.

## State, public interface and fail-closed ancestry

Use new decoder state/policy IDs, retaining the primitive's byte representation:
896 increasing-coordinate records, each signed int64 little-endian plus u8 Z,
exactly 8064 bytes; Z is 0/1 and must be 0 for nonzero I. No padding, trailing
bytes, numeric conversions or implicit defaults. The model envelope must bind
schema/arithmetic/policy/public ABI, width/grid/encoding/geometry, model and
canonical tensor identities, token history, slot, position, next layer, root,
immediate actual producer, source closure, paired H bytes and admission.
It separately binds each own-layer FP16 KV artifact, valid context and producer.
Equality of caller-supplied strings alone authenticates none of these.

At P0 each layer has its own empty prior KV, not another layer's opaque save.
Each new position starts at its own embedding, not the preceding position's
terminal residual. The present scope is only P0/history [9707]; no actual root
artifact or admitted Q24 parent is claimed. `null` in the JSON means absent
and therefore unusable for execution, never an instruction to synthesize state.
Root-only FP16 lift is allowed prospectively; non-root FP16-only seeding,
invented zero carry, reference-minus-H carry, failed parents, mixed A/B state,
old save-file reinterpretation and primitive fixtures as parents are rejected.

Freeze the prospective top `ace3_decoder_token_engine_q24_v1` by the exact
hash-bound accepted decoder header plus **all** added ports and transfer
semantics in section 5 of the bound source-change plan. Preserve original
ports/directions/widths and LAYER_INDEX/ACCURATE_SILU defaults; add only its
RESIDUAL_WIDTH=64/RESIDUAL_FRAC_BITS=24 and the listed state/owner/fault ports.
No compatibility alias, old module replacement or new default FP16 fallback.
That referenced table is normative here, not permission to guess a header.
There is no candidate literal header or compiled decoder ABI in this attempt.
A later implementation must freeze and compile the complete exact signature
before its first semantic attempt, preserving primitive ABI fault definitions
separately from the decoder's additional ownership/pairing faults.

Transfers occur on rising clock edges, asynchronous active-low reset and
synchronous active-high clear taking priority. State/H share load/trace/final
handshakes; update once per accepted transfer. Hold payload/index/last/fault
stable under backpressure. Do not start a partially loaded vector. Root mode
is legal only at layer 0; non-root paired H must equal Q(I,Z) bitwise, including
zero sign. Successful done requires all 896 final pairs, drained traces and
complete own-layer KV work. No fixed whole-decoder cycle count is invented.
Only successful independent numerical/state checks admit an atomic whole-layer
checkpoint. Reset/clear invalidate live and pending state/cache validity;
faulted or partial vectors/KV cannot be forwarded. Preserve old checkpoints.
Save/restore is completed-idle only, with exact lossless payload and trusted
metadata. Reject duplicate/unknown/missing fields, nonfinite JSON, malformed
length/tags, wrong layer/slot/history/source and H/state disagreement.
Portable payload and source/tool/binary-bound simulator saves are distinct
ABIs; neither is a migration route from old FP16-only ancestry.

## Reference applicability and affected cone

New local-reference identity changes S12 operands to authenticated I/Z/O.
S0-S11 and S13-S17 retain canonical v3 same-input references and S13 consumes
actual rounded S12. Independent implementation must not import candidate
arithmetic or use candidate outputs as expectations. Exact state/zero-tag
transitions and producer-to-consumer identity are additional mandatory checks.
Authenticate canonical weights, controls, shapes, layout, indices and causal
KV independently; locally correct arithmetic on corrupt inputs is a failure.
The local finite/absolute/relative/ULP predicate and denominator in JSON are
unchanged, including inclusive 1/8 and strict 1/1000.

S18 retains the ORIGINAL-input independently propagated binary64-v1 reference
and exact rational excess predicate. Never quantize/re-anchor r, clamp negative
excess, seed global history with actual hidden/KV/local outputs, or replace it
with exact Q24 recurrence agreement. Preserve separate `local_operator_fp16`,
`binary64_v1`, residual lineage, KV lineage and legacy FP16 trajectory diagnostic
reports. Missing/unsupported mandatory evidence blocks admission; no empty-pass
result. Original independent reference identity is inherited explicitly from
the unchanged bound v3 policy, not a newly generated reference or tensor set.

The **specified prospective affected cone** is model root through L0-L9/P0;
the **authorized execution cone in this mission is empty**. Representation
changes at root, the first inter-layer retained remainder is L0/S18, and
the first potentially changed FP16 result under identical inputs/operators
is L1/S12. Its MLP/H2 and later hidden/KV dependencies are then affected.
Old L0 and L1/S0-S11 may be arithmetically compatible, but no new state producer
is inferred from that observation. Reuse authenticated compatible tensors,
original references and reviewed primitive evidence without replay. A future
root/O/D-derived state producer would need explicit scope, full provenance and
independent checks; it is neither created here nor called actual RTL state.
Old L1/S12 onward is not assumed compatible merely because FP16 words match.
No L9-only attachment to accepted H8 is legitimate.

Full causal consequences extend to L10-L23, affected later positions/KV, final
RMSNorm/head/top-k, token feedback and new roots. Naming those dependencies
does not include them in this bounded cone. All remain excluded and
NOT_EVALUATED; accepted old FP16 prefix/tail/token evidence keeps its original
lineage and claim limits. No unchanged ancestor replay or failed-state forwarding.

## Ownership, falsification and this attempt's checks

The bound plan maps later candidate work to N02 decoder integration, N05/N07
policy and local-reference adapter, N08 root/state/capture/host/controller
adapter and N09 ordered runner. These are future responsibilities, not edits
or launches here. Preserve primitive N01/N03/N04/N06 and attempt003 evidence;
reuse compatible implementation only under explicit decoder IDs. The accepted
single-round source snapshot, not mutable production two-round wiring, is the
baseline; a complete future changed-source/binary closure remains required.

Future independently authorized evidence must cover root/state losslessness,
signed zero, subnormals, ties/overflow, nonzero retained remainder, H/state
pairing, reset/stall/fault atomicity, wrong owners/versions, independent KV and
all mandatory local/global coordinates in layer order. First failure stops
forwarding. Preserve failure taxonomy, hypothesis, regression and fresh attempt
IDs. Primitive correctness or local improvement without global/lineage admission
falsifies any claim of a repaired decoder; no witness tuning or waived gate.
Independent document review alone never supplies these results or a launch.

This attempt permits only the JSON schema/binding checks, safe import/Python
compilation of the new static checker, and declarative rejection controls.
The exact one-shot command and Python version are frozen in JSON; its actual
output is recorded in the designated mission CHECKPOINT. No simulator, oracle,
arithmetic prototype, numerical admission, decoder/model runtime, tokenizer,
synthesis, timing/PPA, FPGA or hardware execution occurs. Runtime is
**NO_EXECUTION**; L0-L9 numerical correctness and downstream model scopes are
**NOT_EVALUATED**, not FAIL, PASS or evaluator unavailability.
