# Exact-grid residual recurrence: prospective mathematical/state specification

**Disposition: one concrete hypothesis for independent document review; NOT ADOPTED.**
Mission: `51efa8e34614`, `residual-recurrence-semantics-spec`.
Arithmetic proposal: `ace3-residual-exact-grid-q24-v1`.
State proposal: `ace3-residual-exact-grid-q24-state-v1`.
This document authorizes no implementation, compilation, simulator, capture,
admission, software trajectory, arithmetic prototype, oracle, regression, replay,
restore, or forwarding. Document acceptance is not runtime-launch permission.
Only this new document and its directly related review/checkpoint records change.

## 1. Source basis and existing boundary

The normal Host review
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/a79ca09169d1/round-0001.json`
is a `round_reviewed_handoff`, producer `reviewer`, status `done`. Its conclusion
accepts a contract-decision stop, not a repaired model. Its referenced
`build/l9_w4a16_hypothesis_adjudication_a79ca09169d1_attempt001/adjudication.md`
retains accepted actual P0/L0-L8 and actual mandatory-global L9 FAIL at S18/index62.
It records four canonical suffixes still producing `662d`, and separates inherited
hidden discrepancy from current increment discrepancy. These are reused reviewed
observations, not measurements repeated here or unique upstream fault localization.
Its old pending-review/checker-failure prose predates the genuine review; neither
the failed checker log nor any upstream control is rerun or rewritten.

Read source anchors:

| Anchor | What it establishes |
|---|---|
| `build/single_round_residual_rtl_execution_fb06353c5f53_attempt001/source/ace3_fp16_single_round_residual_core.sv` | Exact signed 41-bit Q24 operand decoding, signed 43-bit three-term addition, one RNE16 conversion, overflow rejection and ready/valid behavior |
| `build/single_round_residual_rtl_execution_fb06353c5f53_attempt001/source/ace3_decoder_layer0_token_engine.sv` | S12 uses activation/O; S18 separately uses activation/O/down; public FP16 data ports have no residual-state channel |
| `ace3/contracts/candidates/decoder_gate_policy_v3.json` | Frozen same-input S12 rule, rounded S12-to-S13 edge, canonical AWQ metadata and unchanged accuracy thresholds |
| `ace3/model/candidates/local_operator_reference_v3.py` | Actual operand mapping and `validate_lineage` single-round residual ownership check |
| `ace3/contracts/candidates/w4a16_residual_state_recurrence_23f4d06c67a4_attempt001.json` | Prior rejection of unspecified persistent compensation under the old contract; not adoption of this concrete representation |
| `ace3/rtl/ace3_model24_layer_controller.sv` | Ordered layer completion/checkpoint control; no hidden/carry payload interface |

Let `val` decode binary16 exactly and `Q` denote RNE16. Current per-coordinate
semantics are `R = Q(val(H)+val(O))`, `D = MLP16(R)`, and
`H_next = Q(val(H)+val(O)+val(D))`. Nonlinear/projection operators receive FP16,
not an unrounded residual. Exact intra-layer recovery of
`val(H)+val(O)-val(R)` merely refactors this existing single-round S18; it is
not the proposal's benefit. Only H16 currently crosses the layer boundary.

The live delegated decision permits specifying, but not adopting, a different
recurrence. This proposal makes the previously unspecified full-residual variant
concrete. It does not reopen primitive canonicalization or the parent's already
resolved choice to commission a specification.

## 2. Exact recurrence and operand ownership

Use layer `l=0..23`, position `p`, cache slot `s`, coordinate `i=0..895`.
The equations apply uniformly to every coordinate/layer/position, without witness
indices, reference-derived adjustments, tuned biases, or layer-specific exceptions.
Let `B(x)=2^24*val(x)`, an exact signed integer for every finite binary16 word.
The authoritative residual state is `(I_l,Z_l)` per coordinate:
`I_l` is signed 64-bit two's-complement with value `I_l*2^-24`;
`Z_l` is a negative-zero tag, meaningful only when `I_l=0`.
There is no separately stored floating carry and no accumulator rounding.

Define exact signed-zero addition `A((I,Z),x)` for finite FP16 x:
its integer is `J=I+B(x)`. If J is nonzero its zero tag is 0; if J is zero its
tag is 1 only when `(I,Z)=(0,1)` and x is negative zero, otherwise 0.
Thus cancellation of nonzero operands yields positive zero, and a sum of only
negative zeros remains negative zero. Integer addition must be overflow-checked.
`Q24(I,Z)` is correctly rounded binary16 of `I*2^-24`, using Z for exact zero.

For each layer, the *only* selected recurrence is:

```text
H_l             = Q24(I_l, Z_l)
(O_l, K_l, V_l)  = ATTENTION16_l(H_l, own_layer_prior_KV, canonical_controls)
(T_l, ZT_l)     = A((I_l, Z_l), O_l)
R_l             = Q24(T_l, ZT_l)                         [S12]
D_l             = MLP16_l(R_l)                          [S13-S17]
(I_(l+1),Z_(l+1)) = A((T_l, ZT_l), D_l)
H_(l+1)         = Q24(I_(l+1), Z_(l+1))                 [S18]
```

These are elementwise residual equations; attention and MLP retain their full
vector/head dependencies. O is the actual same-layer S11 output, D the actual
same-layer S17 output produced from this R, and H the materialized view of this
exact state's own producer. ATTENTION16 includes the existing norm1/QKV/RoPE/
causal attention/output-projection path. MLP16 includes norm2/gate/up/SiLU/down.
Their candidate arithmetic stays bound to the accepted single-round source
closure; canonical local references remain independently implemented.

The residual edge uses I, not `B(H)`, and the next layer receives I as well as
the derived FP16 view. A diagnostic carry may be *derived* as
`C_l = (I_l-B(H_l))*2^-24`; it is not an additional input or independently
editable field. `C_(l+1)` is first derivable from this layer's actual O and D.
No global reference, candidate expected output, old-layer reference difference,
software hidden, or opaque preceding-layer cache may produce any operand.

## 3. Rounding, exceptional values, and precision accounting

Q24 uses round-to-nearest ties-to-even, gradual underflow, and no saturation
admission. FP16 subnormals are exact integer grid points; no nonzero exact-grid
sum falls below the least positive subnormal `2^-24`. Signed-zero handling is
defined above, including root negative zero. Infinity/NaN inputs, invalid tags,
unknown participating data/control (X/Z), integer overflow, or a conversion
whose IEEE RNE result is infinite reject the transaction. At magnitude 65520
the round-to-nearest overflow tie rejects; values just above 65504 but below
that threshold may round to finite 65504. This does not relax the separate
global-reference requirement `abs(r)<=65504`. Error/status must remain explicit;
an implementation's placeholder zero/saturated word is never admissible data.

| Boundary | Proposed precision and rounding |
|---|---|
| Model root | Authenticated existing FP16 embedding/root vector, lifted exactly by B; no claim to recover root quantization loss |
| Weights | Native asymmetric G128 packed INT4 qweight/qzeros, GEMM nibble order `[0,4,1,5,2,6,3,7]`, no qzero +1; FP16 scales unchanged |
| Biases, norm parameters, RoPE inputs | Existing source-bound FP16 payloads and canonical selection; unchanged |
| All S0-S17 operator inputs/outputs, including R at S12/S13 | FP16; existing operator rounding/accumulation rules unchanged except S12's newly declared I operand |
| Authoritative inter-layer residual | 64 signed integer bits with 24 fractional bits (39 integer magnitude bits plus sign), and 1 zero-sign bit per coordinate; effective persistent state is wider than FP16 |
| Residual scratch T and pending next state | Same 64+1 representation, not an FP16-rounded R substituted for T |
| Residual decode/add temporaries | B(x) fits signed 41 bits; sign-extend into a signed 65-bit checked addition of 64-bit I and B(x), then exact narrowing after range check; same for T+D |
| Other operator temporaries | Unchanged bit widths, fixed-point scales, reduction order, rounding and bias placement in the bound accepted source closure; no replacement with wider host floating arithmetic |
| Materialized H/R, external load/trace/final activation data | FP16; H is a checked view, not sufficient persistent state |
| Per-layer K/V storage/read/write | FP16, unchanged position/slot/head layout and causal ownership; no carry in cache padding |
| Tail | Final RMSNorm consumes H24 FP16, not I24; tied head/logits/Top-K retain their separately bound precision and selection contracts |
| New portable state payload | 64-bit integer plus zero-sign tag, lossless; serialization performs no numeric conversion |

For N=24 layers a coordinate's state is the exact sum of at most 49 finite
FP16 terms (root plus O and D for each layer). Its magnitude is at most
`49*65504 < 2^22`, hence the scaled integer magnitude is below `2^46`.
64 signed bits therefore suffice for this declared model without approximation;
65-bit checked sums also expose illegal restored state or unsupported growth.
Each intermediate FP16 view must still pass its finite conversion check.
This bound proves capacity, not model accuracy or measured hardware cost.

**This is not unchanged strict W4A16 hidden state.** It preserves INT4 weights,
FP16 operator activations/ports/KV, but proposes a wider persistent residual
stream. Calling it unchanged W4A16 because the external port is 16 bits would
be incorrect. Its adoption is a separate delegated-parent decision after review;
it cannot count as W4A16 completion or precision-milestone advancement.
The grid exponent, residual width, activation encoding and policy identity must
remain explicit, separable contract fields in any later parameterized design.
This FP16-specific grid is not an implicit BF16/W8A16 implementation or policy.

## 4. Initialization, updates, reset and restore

The root producer for every position is that position's authenticated existing
FP16 token embedding E under its actual token history/model. Initialize
`I_0=B(E)` and `Z_0=(E is negative zero)`. H0 is exactly E. This root lift is the
only zero-carry initialization: never initialize from an arbitrary retained H8
or H9. Each position starts a new residual chain; I24 is not next-position I0.
Only own-layer KV and the causal token history persist between positions.

The root-loader owns I0/Z0; layer l owns T/ZT, R and D; successful layer-l
completion produces I(l+1)/Z(l+1)/H(l+1) as a single logical record. Consumers
cannot accept a mix of old H and new I. A materialized H must bitwise equal
Q24 of its state, including signed zero. Ownership binds arithmetic/state
version, canonical model, history, slot, position, next-layer index and coordinate.
The state layer index denotes the *next* layer to execute, 0 through 24.

Future RTL must preserve rising-edge valid/ready transfers, asynchronous
active-low reset and synchronous active-high clear priority. Each accepted
operand updates its pending coordinate exactly once, not once per cycle while
valid stays high. Pending result, index, last, state and fault must remain stable
under backpressure. Do not expose a vector to the next layer until all 896
paired H/state coordinates and the layer completion are acknowledged.
An incomplete/faulted vector is not a continuation parent, even if some K/V
payloads were already written. No exact cycle latency is promised by this
mathematical specification; cycle/stall behavior must be fixed in a later ABI.

Reset/clear cancel pending transfers and invalidate residual validity, partial
vector progress and pending state, alongside existing activation/context/cache
validity semantics. Stale RAM bits need not be zeroed, but must not become a
valid implicit root. Fresh root initialization reestablishes validity. Clearing
and resuming is not restore; later-position cache context must not be silently
lost. A fault stops admission/forwarding rather than rolling back invisibly.

Portable state version `ace3-residual-exact-grid-q24-state-v1` is specified at
completed idle layer boundaries only. No mid-layer save is permitted. The
residual payload contains 896 coordinate records in increasing i order:
8 bytes little-endian signed two's-complement I, then one byte Z (only 0 or 1).
Require Z=0 when I is nonzero, exactly 8064 payload bytes, and no trailing data.
The enclosing existing authenticated manifest must explicitly bind schema,
arithmetic/policy IDs, grid exponent -24, width 64, hidden size 896,
slot 0..3, position 0..32767, next layer 0..24, completed/idle validity,
model/token-history/root identity and immediate actual producer.
It must bind the separately owned per-layer K/V artifacts and their context
length/validity plus source/binary/generated-save ABI where opaque saves are used.
These are logical fields for later implementation, not an executable schema
or a modification of existing manifests.

Restore requires all fields and source/history bindings, canonical dimensions,
finite H projection, exact H/state agreement where H is stored, and an admissible
actual producer chain back to the root. Wrong/missing/truncated data, unsupported
versions, noncanonical Z, or wrong layer/slot/history fail closed. It must produce
the same subsequent semantic transactions as uninterrupted execution, including
KV and stalls' lack of numerical effect. Old generated Verilator snapshots have
no such residual payload: no reinterpretation of unused ports/memory, inferred
carry, default zeros, or dimension-only migration is permitted. A future changed
generated ABI needs its own matching restore path; it cannot import an old
snapshot merely because the public FP16 signature matches.

## 5. Dependency graph and earliest affected cone

```text
actual token E_p -> root (I0,Z0) -> H0
(Il,Zl) -> Hl -> norm1/QKV/RoPE -> own-layer Kp,Vp
Hl + own-layer causal KV[0..p] -> attention/output projection -> Ol
(Il,Zl) + Ol -> (Tl,ZTl) -> Rl -> norm2/MLP -> Dl
(Tl,ZTl) + Dl -> (I(l+1),Z(l+1)) -> H(l+1) -> next layer
H24 -> final RMSNorm -> tied head -> logits/Top-K -> selected token
selected token -> later token root; K/V writes -> same-layer later attention
```

There is no edge from original-global reference to actual state, from a later
layer to an earlier layer, or from residual state of position p to root p+1.
The new *representation* starts at each model root. With zero root carry and
identical actual inputs/controls, layer0 S12 and S18 equal the existing exact
single-round recurrence, including signed zeros. T does not bypass S12 into
norm2. The first newly retained rounding remainder is produced at L0/S18;
the first potentially different FP16 result is L1/S12. L1/S13-S18 follow it.
L1/S0-S11 and its K/V are earlier than that change. H2 may then differ,
affecting L2/S0 onward, K/V, all later layers and the tail.

For fixed authenticated token roots and unchanged attention arithmetic, L0
outputs/KV and L1 input/attention/KV can remain compatible across positions:
their producing residual views precede the change. This is a conditional
dependency proof, not a new acceptance of existing artifacts. Starting at L2,
changed cached K/V can affect every reachable later position in the same layer;
changed hidden feeds later-layer caches too. If the candidate head selects a
different token, that token's new embedding root affects even L0 and L1 at
subsequent positions. Fixed old token IDs are experimental inputs, not evidence
of candidate-native greedy feedback.

The complete prospective execution cone is the changed residual-state chain
from root/L0-S18, L1-S12 through its MLP/final state, L2-L23 descendants,
their affected K/V over causal positions, final RMSNorm/head/logits/Top-K and
subsequent host/tokenizer feedback. Actual differences may be smaller, but
matching FP16 H alone does not prove equality of the exact residual state.
Any reuse argument must cover both arithmetic lineage and separate cache lineage.
**The presently authorized execution cone is empty.**

| Retained artifact | Compatibility boundary |
|---|---|
| Official tensors/tokenizer/control constants | Reuse unchanged with their existing authentic bindings; no new model conversion |
| Accepted old L0 execution and its H/O/D | Potential reuse for identical FP16 outputs; complete authenticated operands can support a future independently checked first state producer without replaying unchanged operators |
| Old L1 S0-S11 and its own K/V | Potential reuse only with identical input H, source/control/causal KV and canonical bindings |
| Old accepted L1-S12 onward through L8 | Remains accepted old-profile evidence, not a ready-made new residual chain or new-state certificate |
| H8/H9 alone or any FP16-only saved state | Insufficient to seed the new chain; missing information cannot be reconstructed from H alone |
| Failed L9 output/state and canonical suffix controls | Immutable negative evidence/regression motivation, never a continuation parent or a new-state initializer |
| Historical other-lineage full-model tail/tokens/KV | Remain scoped to that lineage; cannot be spliced into this one |
| Original global binary64 references | Reuse only as independent original-input accuracy references; never as execution state |

Any future derivation of a first state from retained actual operands is a newly
scoped state-producer/evidence operation, not permitted here and not implied by
review of this document. If operands/provenance are insufficient, report that
specific gap; do not launch an unchanged ancestor replay merely for metadata.
A new residual-root lineage must not erase the old accepted prefix.

## 6. Mathematical rationale and limits

For fixed branch outputs b_l=val(O_l)+val(D_l), the invariant is
`I_L*2^-24 = val(E) + sum_(l<L) b_l`: all residual additions are exact.
The displayed H_L differs from that sum only by its final RNE16 error.
In contrast, the old recurrence with the *same fixed increments* has
`h_L = val(E) + sum b_l + sum epsilon_l`, where each epsilon_l is its
layer-final rounding error. Error-feedback preserves those discarded dyadic
remainders instead of repeatedly forgetting them. It does not change the
correct ties-to-even rule to select a preferred final encoding.

This is a conditional algebraic rationale, not a measured model improvement.
In the real network the increments are not fixed: R changes nonlinear MLP
outputs, later H changes attention, and changed K/V affects future positions.
Removing repeated residual rounding need not reduce error against the original
global binary64 trajectory, improve logits, or choose readable tokens. The
proposal cannot undo embedding/weight quantization, operator approximation,
FP16 branch-output rounding, or other lost information. Carry-free error could
also have fortuitously cancelled those errors. The retained L9 coordinate
motivates examining lost history, but determines no constants or correction.

## 7. Policy, public ABI, and existing-code ownership

The current opt-in
`ace3-w4a16-local-operator-global-binary64-authority-v3` is unchanged and does
not admit this proposal. In particular, its S12 reference is Q(H+O), whereas
this proposal is Q(I*2^-24+O). `local_operator_reference_v3.validate_lineage`
also enforces the old exact H+O+D S18 ownership. Both differences must be
explicitly adopted under a separate prospective policy identity, for example
`ace3-residual-exact-grid-q24-local-global-policy-v1`, before implementation.
That name is a proposal, not a registered policy or default selection.

Required S0-S17 local accuracy remains exactly:
`finite AND (abs_error <= 1/8 OR (relative_error < 1/1000 AND ordered_FP16_ULP <= 1))`,
denominator `max(abs(independent local FP16 reference),2^-14)`.
Any later new S12 reference must independently evaluate exact authenticated I/O,
and verify the new state producer and H view; conditioning on I is not permission
to bless a corrupted carry. S13 still consumes rounded actual R. S18 state
transition correctness is additionally exact, not relaxed by numerical tolerance.

Global S18 remains the ORIGINAL-input independently propagated binary64-v1 r:
finite valid operands, `abs(r)<=65504`, `q(r)=min_finite_FP16_h |h-r|`,
exact `|actual_H-r|-q(r) <= 1/8`, no negative-excess clamp. Candidate hidden,
KV, local outputs and residual state never seed that reference. Missing required
evidence blocks. Legacy/v2/v3 definitions and FAILs remain intact; original
whole-FP16 trajectory diagnostics keep truthful results. Tail/logit/token,
whole-model, state/lineage and independent-review requirements are not waived.

The exact existing public contract remains normative at
`build/l9_l23_continuation_10d99fb0a67e_attempt002/freeze.json#/public_contract`:
`ace3_decoder_layer0_token_engine`, integer `LAYER_INDEX=0`,
integer `ACCURATE_SILU=(LAYER_INDEX>=3)`; retained L9 uses 9 and 1.
Its full header, not an alias list, is the signature authority. Existing
`load_f16_i`, `trace_f16_o`, `final_f16_o` remain 16-bit. This document adds no
module, port or parameter. A later isolated versioned wrapper/top must explicitly
carry the 64+1 state with H and transaction identity; no hidden use of old fields.
Its exact header, layer parameters, handshake, fault/commit channel and generated
save ABI must be frozen and compiled before its first arithmetic attempt.
No such compile occurs now; existing compile receipts are not new conformance.

| Existing owner/surface | Prospective responsibility; no edits now |
|---|---|
| Frozen decoder and single-round residual core cited in section 1 | Isolated new residual state datapath/checked rounding; preserve unchanged operator source and original top |
| `ace3/tb/ace3_single_round_residual_tb.sv` | Existing primitive handshake/corner-case patterns, not already a persistent-state test |
| `ace3/model/candidates/local_operator_reference_v3.py` | Independent new-policy residual reference/ownership checks in a separately scoped implementation, not changing v3 in place |
| `ace3/model/candidates/decoder_gate_policy_v3.py` | Keep `local_operator_fp16`, original global and trajectory diagnostics distinguishable; no policy substitution |
| `ace3/model/candidates/runtime_admission_v3.py` | Future distinct state/source/actual-operand authentication; existing old-ABI validator cannot self-admit the extension |
| `ace3/model/candidates/run_single_round_residual_rtl.py` | Future ordered actual-state/output threading and affected-cone reuse, never software-as-RTL substitution |
| `ace3/rtl/ace3_model24_layer_controller.sv` and owning host | Pair state completion with hidden completion; bind separate own-layer KV and checkpoint ownership |
| Existing model/tail/tokenizer host owners | Consume H24 at the unchanged tail boundary and candidate-bound selected tokens, not unrounded I24 |

## 8. Prospective evidence plan and falsification

All items here are future work requiring separate scope/adoption after independent
document review, not commands authorized by this specification.

1. Independent Reviewer accepts/rejects this document against the six live parent
   requirements, including root causality, wider-state disclosure, compatibility,
   unchanged thresholds and no-execution scope. The Host supplies that review;
   Engineer does not create a Reviewer verdict or spawn a Reviewer.
2. Before any arithmetic implementation/attempt, obtain the separate technical
   disposition; freeze exact semantics/new public ABI, canonical inputs, reference
   independence, tool versions, evidence ordering and scoring. Preserve first
   official attempt immutably and assign fresh repair attempts. Inspect available
   local tools/declared containers then, not as an execution pretext now.
3. Define an independent exact-integer/rational residual oracle that does not
   import/call candidate arithmetic. It independently decodes authenticated actual
   FP16 words and applies these equations and bit-level rounding/zero rules.
   Preserve separate canonical operator metadata authentication and original-input
   binary64 propagation. The present P0-only local reference support is not
   silently extended to later positions: general causal/RoPE/KV references need
   separately implemented, reviewed coverage before such evidence is admissible.
4. Later general primitive/state cases cover signs/zeros, subnormals, cancellation,
   binade ties, rounding midpoints, finite limits/overflow, illegal X/Z/count/state,
   root lift, multi-layer exact-sum invariant, no cross-position carry, stall and
   reset priority, duplicate inputs, partial vectors, wrong ownership, missing
   state, incompatible saves and uninterrupted-versus-restored equivalence.
   They also demonstrate that discarding the carried remainder returns the old
   single-round recurrence, rather than disguising intra-layer canonicalization.
5. After independent prerequisite review, any authorized model evidence proceeds
   causally from the valid root/state producer and earliest affected boundary,
   covering every applicable coordinate and local gate before each global S18.
   Reuse compatible executions; stop forwarding at the first mandatory failure.
   Compare unchanged-contract controls by retained evidence where compatible,
   not repeated control runs. Retain the L9 FAIL and canonical suffixes as
   prospective regressions, never as expected words hard-coded in the candidate.
6. Only subsequently scoped actual RTL may establish RTL behavior, with actual
   outputs/state feeding descendants, independently authenticated simulator input,
   independent oracle and genuine reviewer validation. New K/V histories and all
   affected tail/host dependencies are prerequisites for any token claim.
   Preserve command sidecars, logs, failing seeds/waveforms and actual tool status;
   evaluator no-execution is distinct from numerical failure. No hardware,
   synthesis, PPA, bitstream, board or deployment evidence is claimed.

Reject this specification if its invariant/root/width/zero definitions are
inconsistent, if state requires an unavailable invented producer, or if it
silently represents a policy/ABI widening as unchanged strict W4A16.
In a later authorized implementation, any state-invariant/bit-rounding mismatch,
lost update, reset/restore discrepancy, forbidden source edge, or incompatible
state reuse falsifies conformance. A mandatory local or original-global miss
rejects that candidate trajectory even if mean error improves or local gates
pass. Retained L9 remaining failed after coherent propagation would falsify
the proposed repair for that target, not the algebraic invariant or all possible
future architectures. No guaranteed PASS, readable dialogue, or timing benefit
is inferred from the mathematics.

For each future failed attempt record exactly one primary taxonomy, one supported
root-cause hypothesis, and one discriminating regression; preserve the original
failure. Appropriate classes include arithmetic-conformance, state-lineage,
mandatory-global-numerical, and evaluator-no-execution. The existing L9 failure
remains mandatory-global-numerical; its inherited-history hypothesis is not a
proven unique bug. This document has no numerical attempt and no numerical PASS.
