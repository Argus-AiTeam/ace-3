# Exact-grid Q24 residual adoption boundary

**DOCUMENT_ONLY_ADOPTION_BOUNDARY_NOT_ADOPTED. Authorized execution cone: empty.**
Mission `1952d3601928`; node `exact-grid-q24-adoption-boundary`; attempt001.
This records the boundary for a reviewed mathematical proposal. It adopts no
arithmetic, reference rule, public ABI, precision profile, or runtime candidate.
Independent review of this record is pending through the normal Host Reviewer.

## 1. Reviewed basis and honest frontier

The normative proposal is
`ace3/contracts/candidates/residual_exact_grid_q24_semantics_51efa8e34614_attempt001.md`.
Its later Host receipt,
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/51efa8e34614/round-0001.json`,
is `round_reviewed_handoff`, `producer_role=reviewer`, `review.status=done`,
created_at `1788934442.939863`. Its reason explicitly accepts the bounded
specification against the parent-authorized requirements. The older specification
check record's pending wording is historical, not a missing-review blocker.
This is document acceptance, not adoption, a measured repair, or launch authority.

The reviewed proposal cites the prerequisite a79ca09169d1 adjudication.
The inspected
`build/l9_w4a16_hypothesis_adjudication_a79ca09169d1_attempt001/adjudication.md`
retains accepted actual single-round P0/L0-L8 and an actual mandatory-global
L9/P0/S18/index62 FAIL. All four retained canonical suffix controls still fail.
Those are attributed retained observations, not rerun or re-authenticated
numerical measurements in this task. No newer passing trajectory is established
by the Q24 document review. The exact accounting is neither a unique upstream
bug localization nor a guarantee that wider residual state repairs the model.

The Wiki's numerical-boundary page was consulted for the v3/continuation surface.
Frozen source and the accepted specification, rather than mutable Wiki prose
or older task summaries, govern this prospective compatibility map.
There is no `ambiguous_objective`: the live task is the narrower non-executing
adoption record following the accepted specification.

## 2. Prospective identities and policy delta

| Identity | Meaning and adoption status |
|---|---|
| `ace3-residual-exact-grid-q24-v1` | Reviewed recurrence proposal; not implemented/adopted by this record |
| `ace3-residual-exact-grid-q24-state-v1` | Reviewed portable residual-state proposal; not an existing simulator save ABI |
| `ace3-residual-exact-grid-q24-local-global-policy-v1` | Selected prospective policy name from the specification; no registration, default, or production opt-in exists here |
| `ace3-w4a16-operators-fp16-kv-q24-residual-proposal-v1` | Prospective claim label: INT4 weights, FP16 operator activations/KV, wider persistent residual; NOT unchanged strict W4A16 |
| `ace3-w4a16-local-operator-global-binary64-authority-v3` | Existing policy stays unchanged; cannot admit the new residual operands by relabeling an old result |
| `ace3-w4a16-layer-final-binary64-fp16-excess-v1` | Existing mandatory global numerical profile remains unchanged |

The proposal is exactly the accepted recurrence, not a second hypothesis.
Let `B(x)=2^24*val_FP16(x)` and let `(I,Z)` be a signed 64-bit integer
with scale `2^-24` and its canonical negative-zero tag. Q is binary16 RNE
of that exact value; A is the specification's checked exact addition of a
finite FP16 operand, including its signed-zero rule:

```text
E_p -> (I0,Z0) = (B(E_p), is_negative_zero(E_p)); H0 = E_p
Hl = Q(Il,Zl)
(Ol,Kl,Vl) = ATTENTION16_l(Hl, own_layer_causal_KV, canonical_controls)
(Tl,ZTl) = A((Il,Zl),Ol); Rl = Q(Tl,ZTl)                    [S12]
Dl = MLP16_l(Rl)                                         [S13-S17]
(I(l+1),Z(l+1)) = A((Tl,ZTl),Dl); H(l+1) = Q(I(l+1),Z(l+1)) [S18]
```

S12 changes from `RNE16(H+O)` to `Q(I+B(O))`; S18 exact ownership changes
from `RNE16(H+O+D)` to `Q(I+B(O)+B(D))`. S13 still receives rounded R,
never T. Carry is only a derived diagnostic `I*2^-24-val(H)`, not an input.
Retaining T within one layer alone merely recovers existing single-round S18;
the prospective difference is preserving I across layers.

Finite FP16 subnormals are exact Q24 grid points. RNE, gradual underflow,
negative-zero handling and canonical tags remain as specified: cancellation
gives positive zero; all-negative-zero addition preserves negative zero.
Reject nonfinite operands, participating X/Z, invalid tags, signed-integer
overflow and conversion to infinity, including the magnitude-65520 overflow
tie. Do not saturate or admit a placeholder. Values between 65504 and 65520
may round to finite 65504; the global reference must still satisfy
`abs(r)<=65504`. No new exception or tolerance is introduced here.

Native G128 asymmetric packed INT4, GEMM lane order `[0,4,1,5,2,6,3,7]`,
no qzero plus-one, FP16 scales, existing FP16 operator boundaries and FP16 KV
remain fixed. Residual storage and scratch become 64+1 bits per coordinate;
checked additions use 65 bits, not host floating-point accumulation. Other
operator temporaries retain their source-bound widths and rounding. The
49-term/24-layer capacity argument is mathematical, not an area/timing result.
INT4/FP16 ports do not conceal or legitimize the wider persistent hidden stream.
This proposal cannot close unchanged strict-W4A16 dialogue or advance the
W8A16/BF16/FP16 roadmap. Later interfaces must expose mode, activation encoding,
residual width/scale and numerical policy separately; no later mode is implemented.

## 3. Required source, contract and evidence touchpoints

Paths below are existing ownership anchors, not files authorized for mutation.
Any later implementation must be isolated/versioned and preserve accepted sources,
legacy/v2/v3 behavior, immutable controls and all historical FAIL records.

| Existing surface | Required later delta or explicitly preserved responsibility |
|---|---|
| `build/single_round_residual_rtl_execution_fb06353c5f53_attempt001/source/ace3_decoder_layer0_token_engine.sv` and its `ace3_fp16_single_round_residual_core.sv` | Accepted-source baseline: S12 reads activation/O; S18 reads activation/O/down. New isolated datapath must use I/T and produce paired H/state; never edit this frozen source |
| `ace3/rtl/ace3_decoder_layer0_token_engine.sv`, `ace3/rtl/candidates/single_round_residual_v1/ace3_fp16_single_round_residual_core.sv` | Reusable RTL ownership locations, not proof mutable production equals the accepted closure; preserve unchanged operators and old public top |
| `ace3/contracts/decoder_layer0_token_engine.json`, `ace3/contracts/candidates/decoder_gate_policy_v3.json` | Separate versioned residual operands/precision/ABI contract; do not widen either existing definition in place |
| `ace3/model/candidates/local_operator_reference_v3.py` | `OPERANDS[12]`, `local_reference`, and `validate_lineage` currently use H/O and exact H+O+D. New independent policy branch must authenticate I/Z, check Q(I)=H and exact state transitions; do not weaken v3 checks |
| `ace3/model/candidates/decoder_gate_policy_v3.py`, `ace3/model/candidates/remaining_decoder_gate_policy_v3.py` | Distinct opt-in evaluator identity and adapters; separate local, global, exact state/lineage, and trajectory-diagnostic results. The remaining-layer adapter is not a new root/state certificate |
| `ace3/model/candidates/binary64_fp16_excess_v1.py`, `ace3/contracts/candidates/binary64_fp16_excess_v1.json` | Preserve exact-rational global metric, reference policy and rejection rules; no Q24-specific tolerance or reference recurrence |
| `ace3/model/candidates/reevaluate_local_operator_v3.py` | Existing original-binary64 reference binding/loading and independent FP16 trajectory provenance are reference-integration anchors. Any new-policy software screen is separate; never regenerate old expectations from candidate state |
| `ace3/model/candidates/runtime_admission_v3.py`, `ace3/model/candidates/host_capture_v3.py`, `ace3/model/candidates/capture_harness_v3.py` | Later source/binary/operand authentication and raw state capture must cover new payload/commit semantics. Old validation/capture does not automatically cover I/Z |
| `ace3/model/candidates/run_single_round_residual_rtl.py`, `ace3/model/candidates/remaining_layers_v3.py` | Later ordered runner must thread actual paired H/state, classify compatible ancestors and halt before forwarding a mandatory failure; no old-L8 launch-root shortcut |
| `ace3/rtl/ace3_model24_layer_controller.sv`, `ace3/contracts/model24_layer_controller.json`, `ace3/contracts/model24_execution.json` | Controller currently sequences completion, not carry data. Define atomic vector/state completion, owning host payload, next-layer index and fault semantics |
| `ace3/model/model24_persistent_kv_runtime.py`, `ace3/model/controller_model24_rtl_cascade.py`, `ace3/model/model24_host_runtime.py` | Explicit root/state threading, independent per-layer KV lineage, checkpoint/restore, selected-token ancestry and host receipts; no opaque-save import by dimensions |
| `ace3/model/run_position2_tail.py`, `ace3/model/run_tied_lm_head_topk_from_final_rmsnorm.py`, `ace3/contracts/position2_tail_binding.json` | Preserve final RMSNorm input H24 FP16, tied-head/logit/Top-K contracts; later outputs must bind the new candidate H24, not historical tail output |
| `ace3/model/model24_oracle.py`, `ace3/model/official_model24_next_token.py` | Preserve canonical config/tensor/control/tokenizer authentication; distinguish software next-token evidence from actual RTL-generated feedback |
| `ace3/tb/ace3_single_round_residual_tb.sv`, `ace3/tb/ace3_decoder_layer0_token_engine_main.cpp`, `ace3/tb/ace3_model24_layer_controller_tb.sv` | Later isolated residual-state, cycle/backpressure/reset, payload and restore tests; existing tests are not Q24 state coverage |

The new local oracle must independently decode exact integers/rationals and
round to FP16 without importing candidate arithmetic or reading candidate output
as expectation. Canonical tensor/control authentication and producer-consumer
identity are mandatory even for same-input arithmetic. Current
`local_operator_reference_v3.validate_lineage` explicitly supports only
P0/history `[9707]`, empty prior KV. Later causal positions/RoPE/masks/KV need
separately implemented and reviewed reference support, not a P0 fallback.

S0-S17 mandatory numerical accuracy remains:
`finite AND (abs_error <= 1/8 OR (relative_error < 1/1000 AND ordered_FP16_ULP <= 1))`,
with denominator `max(abs(independent local FP16 reference),2^-14)`.
The necessary change is S12's authenticated operand/state semantics, not the
threshold. State transitions and H/state identity are additionally exact.

S18 retains ORIGINAL-input independently propagated binary64-v1 r, valid finite
operands and in-range r, `q(r)=min_finite_FP16_h |h-r|`, and exact
`|actual_H-r|-q(r) <= 1/8`, without clamping negative excess. Missing/unverifiable
references block. Neither candidate hidden/KV/I nor new local outputs seed this
global reference. Original independent whole-FP16 comparisons remain truthful
trajectory diagnostics; they are not silently redefined as the Q24 trajectory.
Tail, logits, tokens, full-model accuracy and independent-review gates persist.

## 4. Public ABI and state lifecycle boundary

The exact old module/port/parameter authority remains
`build/l9_l23_continuation_10d99fb0a67e_attempt002/freeze.json#/public_contract`:
`ace3_decoder_layer0_token_engine`, integer `LAYER_INDEX=0`, integer
`ACCURATE_SILU=(LAYER_INDEX>=3)`; retained L9 binds 9 and 1.
The complete frozen header, not this summary, defines the contract.
`load_f16_i`, `trace_f16_o`, `final_f16_o` remain 16-bit.
This record adds no aliases, ports, parameters, or presumed compatibility.

A future separately versioned top/wrapper must explicitly transport 64-bit I
and Z with FP16 H, coordinate/transaction identity, validity and fault/commit
semantics. Exact new names, widths, parameter values and cycle latency remain
unfrozen: implementation is not authorized to guess them. Freeze and compile the
exact public signature before the first arithmetic attempt in that later scope.
No compilation occurs here; historical compile success proves no new behavior.

Follow the reviewed active-low asynchronous reset, synchronous clear priority,
rising-edge valid/ready transfer, once-per-accepted-operand update and stable
pending output under backpressure. Commit only a complete acknowledged 896-entry
paired vector. Fault/reset/clear invalidate partial state and continuation,
including any uncommitted KV writes; stale RAM contents do not become a root.
The root loader alone produces I0/Z0. Layer l owns T/R/D and produces
I(l+1)/Z(l+1)/H(l+1); a cache belongs to its own layer, position and history.

The portable residual payload is exactly 896 ordered coordinate records,
each 8 little-endian two's-complement I bytes plus one Z byte: 8064 bytes.
Z is 0 or 1 and must be zero for nonzero I. The enclosing state manifest binds
policy/arithmetic/state versions, width 64, exponent -24, hidden size 896,
slot 0..3, position 0..32767, next-layer 0..24, completed idle validity,
canonical model/root/history and immediate producer, plus separate own-layer KV
context and source/binary/save ABI. No numeric conversion occurs on serialization.

Save/restore is only at completed idle layer boundaries, never mid-layer.
Reject missing/truncated/trailing payload, wrong versions/ownership, invalid
tags, nonfinite H projection or Q(I,Z) != H. Restore must preserve uninterrupted
semantic transactions and KV. Old FP16-only or generated simulator snapshots
have no residual payload; padding, default zero carry and opaque ABI reuse
cannot migrate them. A later restore path requires its own matching ABI evidence.

## 5. Earliest lineage and complete future affected cone

**Root lineage starts at each authenticated token embedding, not at L9.**
At position p, I0/Z0 is the exact lift of that position's FP16 E.
Position p+1 has a fresh residual root; I24 is never its initializer.
Only causal token history and own-layer KV carry across positions.

```text
E_p -> new root/state -> L0/S18 first retained rounding remainder
    -> L1/S12 first potentially changed FP16 R -> L1/S13-S18 and new state
    -> L2-L23 hidden/operator/state descendants -> H24 FP16
    -> final RMSNorm -> tied lm_head/logits/Top-K -> selected token -> new E
changed per-layer K/V -> every reachable later attention in that layer
changed later hidden -> downstream layers' K/V and tail at those positions
```

With identical authenticated roots, controls and operator arithmetic, L0 FP16
outputs and KV remain compatible with single-round semantics. L1/S0-S11 and
its KV precede the first potential FP16 difference. This does not eliminate the
new root/L0 state-producer obligation. From L1/S12, include all MLP/final-state
descendants; from H2, include L2/S0 onward, their caches, all later layers/tail
and reachable later-position dependencies. A changed selected token also changes
later roots, invalidating even L0/L1 reuse for that new token history.

Thus the future cone is root-state establishment plus every changed residual
edge and downstream hidden/KV/tail/host edge, not merely L9-L23. Execute only
causal layer/position order if separately authorized. Matching FP16 H alone
does not establish matching I or state lineage. Arithmetic reuse and cache
reuse require distinct proofs; no A/B/single-round/two-round state splice.

## 6. Retained artifact classification

| Retained class | Permitted reuse versus incompatibility |
|---|---|
| Official weights, scales, quantization metadata, tokenizer and canonical controls | Reuse compatible authenticated assets unchanged; no conversion or oracle regeneration in this task |
| Original-global binary64 references and original FP16 trajectories | Reuse only as their independently bound references/diagnostics for matching original inputs; never execution state |
| Accepted old L0 H/O/D and KV for the same root/source/history | Conditionally reusable operator evidence. Complete actual operands may support a separately authorized new first-state producer; no unchanged operator replay merely for metadata |
| Old L1/S0-S11 and own KV for identical H1/control/causal state | Conditionally compatible before first changed FP16 boundary; not a complete new-state/layer certificate |
| Accepted old L1/S12-L8 outputs, hidden and saves | Retain old-profile acceptance; not presumed Q24-compatible ancestors. No bulk old-prefix import into the new recurrence |
| H8/H9 alone, FP16-only state, or old generated saves | Incompatible as a root/restore seed: missing residual cannot be inferred from H, reference-minus-H, spare bits or invented carry |
| Failed L9 state/output and canonical suffix controls | Immutable failure/regression evidence only; never forward, admit, replay here, or patch toward a desired FP16 word |
| Historical other-lineage 24-layer tail, selected tokens and KV | Retain their scoped historical evidence; cannot seed this residual lineage or prove its greedy token choices |
| Existing source/tests and old compile receipts | Reuse patterns and compatible sources, not claims that the new state/ABI was compiled, simulated or independently validated |

Any future retained-operand reconstruction is a new, explicitly scoped
state-production operation with root-to-producer authentication. A software
reconstruction is not RTL state-production evidence or a generated-save image.
If required actual O/D or provenance is missing, record the precise gap and
justify the smallest new affected evidence scope; do not replay accepted
ancestors automatically. No reconstruction, capture, import or admission occurs now.

## 7. Later implementation sequence and pre-authorization criteria

The sequence is prospective. Independent acceptance of either document grants
none of these execution stages.

1. Complete normal Host review of this adoption-boundary record. The delegated
   parent can then consider a specifically scoped semantics/ABI adoption and
   implementation decision, retaining the explicit wider-state claim limitation.
2. Only under that separate decision, freeze an isolated complete public ABI and
   source/contract/reference plan; implement the residual datapath, independent
   oracle, exact ownership/state checks, policy adapters and checkpoint wiring.
   Preserve all old entry points/defaults and source closures.
3. In a separately permitted evidence phase, compile the frozen exact contract
   before arithmetic attempts, then check general primitive/state semantics,
   unchanged-policy regressions and independent-reference integrity. Record
   actual tool versions and inspect host PATH/declared containers then, not now.
4. After the required independent prerequisite review and scoped runtime
   authority, establish new root/state causally, reuse proven compatible actual
   operands, and evaluate the complete affected cone in order. Stop at the first
   mandatory failure; preserve original logs/seeds/waveforms and distinct repair
   attempts. No software result may substitute for an actual RTL output.
5. Only after decoder prerequisites pass with review, bind candidate H24 to the
   existing tail contracts and candidate-selected token to tokenizer/host and
   persistent KV feedback. Extend later-position reference support before using
   it. Fixed historical tokens remain fixed-history experiments, not generation.

Before requesting **initial adoption/implementation authorization**, all of these
must be explicit: accepted normal reviews of both documents; the exact policy and
wider-state claim delta; an owner-reviewed full ABI design with no aliases or
hidden payload; unchanged mandatory thresholds/reference provenance; a bounded
root/affected-cone and per-artifact reuse/gap map; independent oracle ownership,
general falsification cases and no reference injection; a precise requested
phase/writable scope that cannot be read as model-launch or precision promotion.
Implementation results are not a circular prerequisite for that first request.
Unresolved required semantics/ABI fields block that request rather than being
filled opportunistically by execution.

Before requesting **numerical or RTL launch authorization**, additionally require
the actual adopted/versioned implementation scope and independent review of
its source, oracle independence, canonical input authentication, policy wiring,
full public signature and state/restore design. Freeze the prompt, evaluator,
official public inputs, tool versions, score policy, ordering and stop criteria
before the first official attempt; keep that attempt immutable and record repairs
separately. Where a prerequisite check itself executes arithmetic, it needs its
own explicitly scoped evidence permission, not presumed permission from this list.
Never inspect hidden harness sources/golden outputs.

General future falsification must cover signs/negative zero, subnormals, ties,
binade and overflow boundaries, exact multi-layer recurrence, no cross-position
carry, handshake stalls, reset/clear, partial/duplicate vectors, invalid X/Z,
missing/wrong state/metadata/history, and save/restore equivalence. A real local
arithmetic error must still fail; global drift despite locally correct operators
must still fail binary64-v1; missing/nonfinite references must block/fail.
Legacy/v2/v3 compatibility and truthful trajectory diagnostics are regressions,
not new tests weakened to admit Q24.

The retained L9 FAIL is one prospective regression, not the definition or desired
word. A coherent Q24 trajectory still missing any mandatory gate rejects that
candidate path; failure to improve the proposed residual-drift mechanism rejects
the repair claim, not a universal theorem about all wider arithmetic. Each future
failed attempt must state one failure-taxonomy class, one root-cause hypothesis
and one regression, preserving its evidence. No performance cause, threshold
benefit, readable dialogue, full-model admission or hardware result is claimed.

## 8. This record's closure boundary

Only this new adoption-boundary artifact, its directly related check record and
the explicitly designated external CHECKPOINT.md are written. No source,
production contract/evaluator/reference, existing artifact, Wiki or Skill is
changed. The proposal is not adopted durable project knowledge or a validated
new reusable procedure, so no Wiki/Skill update is warranted or in scope.

The one structural check verifies document shape, required boundary statements
and named source-path presence only. It is not independent mathematical review,
source equivalence, numerical accuracy or runtime evidence. No arithmetic
implementation, oracle regeneration, software trajectory, RTL/simulator,
capture/admission, L9 replay, failed-state forwarding, commit or push occurs.
Normal independent Reviewer disposition is the next boundary; no new operator
question or runtime authorization request is issued by this record.
