# Exact-grid Q24 residual: bounded source-change plan

**DOCUMENT_ONLY_SOURCE_CHANGE_PLAN_NOT_ADOPTED. Authorized execution cone: empty.**
Mission `9885c390465c`; node `exact-grid-q24-source-change-plan`; attempt001.
This document proposes later implementation work, not arithmetic code, an adopted
contract, a compiled ABI, a runtime candidate or a numerical result. Normal
independent Host Reviewer disposition of this plan is still required.

## 1. Authority, inspected evidence and source-basis guard

The normative recurrence is the accepted
`ace3/contracts/candidates/residual_exact_grid_q24_semantics_51efa8e34614_attempt001.md`.
The bounded adoption record is
`ace3/contracts/candidates/residual_exact_grid_q24_adoption_boundary_1952d3601928_attempt001.md`.
Its separate assessment is
`ace3/contracts/candidates/residual_exact_grid_q24_adoption_boundary_1952d3601928_attempt001.independent_review_5ae91c13caf2_attempt001.md`.
Read later genuine Host receipts:
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/5df93323345b/round-0001.json`
accepts the prospective specification assessment, and
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/5ae91c13caf2/round-0001.json`
has producer `reviewer`, status `done`, reason "requested outcome is materially
complete". These receipts do not adopt Q24 or erase either earlier failed
delivery check. Frozen pending-review prose predates later review.

The reviewed documents retain the actual single-round P0/L0-L8 prefix and
mandatory-global L9/P0/S18/index62 FAIL. That frontier is an attributed retained
observation, not re-adjudicated here; no newer passing trajectory is established.
The Wiki INDEX and numerical-boundary page were read as context, not authority.
No same-level conflict was found requiring `ambiguous_objective`.

**Observed migration hazard:** mutable
`ace3/rtl/ace3_decoder_layer0_token_engine.sv` lines 458-465 currently instantiate
two `ace3_fp16_residual_add_core` instances; S18 consumes `res1_mem` and
`down_mem`. That is not the accepted frozen single-round activation/O/down
wiring. The standalone candidate single-round core has a signed 43-bit sum and
one conversion; its mere presence does not make the mutable decoder equivalent.
The first later source task must compare the complete accepted source closure,
not blindly copy the mutable top, change only a width, or overwrite frozen files.
Existing dirty worktree changes are other work and must remain untouched.

Source inspection here covered the current decoder header/residual wiring,
fixed-point helpers, single-round core, local reference, evaluator and
capture/runner/controller symbol boundaries. Other paths below are concrete
ownership anchors from the reviewed adoption map, not claims that their full
implementations or historical binaries have been authenticated in this mission.

## 2. Selected recurrence and non-negotiable precision

For every position independently, lift its authenticated FP16 embedding E:
`I0 = B(E)`, `Z0 = is_negative_zero(E)`, where `B(x)=2^24*val_FP16(x)`.
For each layer: `H=Q(I,Z)`; attention consumes H and its own causal FP16 KV
and produces actual O; `(T,ZT)=A((I,Z),O)`; `R=Q(T,ZT)` at S12;
MLP consumes rounded R and produces actual D; `(I_next,Z_next)=A((T,ZT),D)`;
`H_next=Q(I_next,Z_next)` at S18. Only actual same-transaction O/D are operands.
Never replace T with B(R), I with B(H), or state with global-reference-minus-H.

A is exact checked integer addition of B(x). Its zero tag is negative only when
the prior state is tagged negative zero and x is negative zero; nonzero results
have Z=0 and nonzero cancellation gives positive zero. Q is binary16 RNE,
ties-to-even, gradual underflow. Reject NaN/Inf, invalid tags, participating
X/Z, integer overflow and conversion to infinity. Magnitude 65520 is the
rejecting overflow tie; a value above 65504 but below 65520 can round to 65504.
No saturated or placeholder word is admissible.

Persistent I, scratch T and pending I_next are signed 64-bit Q24 plus one
zero-sign bit. Decode FP16 into signed 41 bits, explicitly sign-extend, add in
65 bits, range-check before narrowing, and round only H/R views. Every finite
FP16 subnormal is already on this grid. For 24 layers the 49-term magnitude
bound is below 2^46 in Q24 units; this is a capacity proof, not measured benefit.
Under fixed O/D increments the exact-sum invariant removes repeated residual
rounding, but changed nonlinear branches/KV need not improve global accuracy.
No constants, special cases or RNE choices may target the retained L9 word.

Keep native G128 asymmetric packed INT4, GEMM order `[0,4,1,5,2,6,3,7]`,
no qzero plus-one, FP16 scales/operator activations/KV and unchanged other
operator temporaries. Final RMSNorm receives H24 FP16, never I24. Wider
persistent residual state is **not unchanged strict W4A16** and does not close
that milestone or implement W8A16/BF16/FP16. Keep weight mode, activation/KV
encoding, residual width/grid and numerical policy separate in metadata.

## 3. Concrete existing-source ownership map

Each E row names an existing surface. "Adapter" means a later isolated,
explicitly selected candidate path, not an in-place change to frozen v3.

| ID | Existing source/contract surface | Required later change or preservation |
|---|---|---|
| E01 | `build/single_round_residual_rtl_execution_fb06353c5f53_attempt001/source/ace3_decoder_layer0_token_engine.sv` | Read-only accepted arithmetic baseline; derive the candidate from its authenticated closure and prove all non-residual operators unchanged. |
| E02 | `build/single_round_residual_rtl_execution_fb06353c5f53_attempt001/source/ace3_fp16_single_round_residual_core.sv` | Preserve immutable single-round implementation as baseline, not a 64-bit state carrier. |
| E03 | `build/l9_l23_continuation_10d99fb0a67e_attempt002/freeze.json` | Preserve `public_contract` as the complete old header authority; new top has a distinct header and ABI. |
| E04 | `ace3/rtl/ace3_decoder_layer0_token_engine.sv` | Ownership anchor for S12/S18 memories/FSM, loads, traces and final/done. Do not use its current two-round wiring as accepted single-round source. |
| E05 | `ace3/rtl/candidates/single_round_residual_v1/ace3_fp16_single_round_residual_core.sv` | Reuse elastic handshake pattern, not its 43-bit accumulator or three-FP16 public input contract. |
| E06 | `ace3/rtl/ace3_fp16_fixed.sv` | Reuse source-compatible `ace3_fp16_to_q24` and parameterized `ace3_q24_to_fp16_rne` only after WIDTH=64 boundary coverage; consume saturation status as rejection. No production helper rewrite presumed. |
| E07 | `ace3/contracts/decoder_layer0_token_engine.json` | Preserve old ports/precision/defaults; separate candidate contract specifies I/T/state and paired outputs. |
| E08 | `ace3/contracts/candidates/decoder_gate_policy_v3.json` | Preserve v3 identity/operand meanings; new policy changes S12 state operands and exact S18 lineage only as specified below. |
| E09 | `ace3/model/candidates/local_operator_reference_v3.py` | `OPERANDS[12]`, `local_reference`, `validate_lineage` require a separate Q24 adapter. Reuse unchanged tensor authentication and non-residual local operators; do not call the old H+O+D lineage check on a Q24 transaction. |
| E10 | `ace3/model/candidates/decoder_gate_policy_v3.py` | Preserve `evaluate_decoder_stage`, `evaluate_p0_transaction`, `evaluate_actual_rtl_result`; new equivalents require Q24 provenance and exact state checks before numerical admission. Existing transaction scope is L5-L8/P0. |
| E11 | `ace3/model/candidates/remaining_decoder_gate_policy_v3.py` | Later-layer adapter pattern only; not a Q24 root certificate or implied extension to new positions. |
| E12 | `ace3/model/candidates/binary64_fp16_excess_v1.py` | Reuse exact global metric without modifying reference, q(r), tolerance or negative-excess rejection. |
| E13 | `ace3/contracts/candidates/binary64_fp16_excess_v1.json` | Keep numerical profile immutable and bind its identity separately from the new operand policy. |
| E14 | `ace3/model/candidates/reevaluate_local_operator_v3.py` | Reuse original-reference binding conventions; separate new software screen must not recompute global/legacy expectations from candidate state. |
| E15 | `ace3/model/candidates/runtime_admission_v3.py` | `validate_runtime`, `saved_state`, trusted context/source checks are ownership anchors. New schema authenticates residual input/output bytes, commit and generated save ABI; no old-schema reinterpretation. |
| E16 | `ace3/model/candidates/host_capture_v3.py` | New plan/actual-array/capture/admission adapter records state sidebands and root producers; old `prepare`/`capture` paths must not acquire Q24 defaults. |
| E17 | `ace3/model/candidates/capture_harness_v3.py` | Extend only in a separate candidate harness; capture raw I/Z plus FP16 views and fault/completion, not state inferred from final.hex. |
| E18 | `ace3/model/candidates/run_single_round_residual_rtl.py` | Preserve old CLI and retained verification; new ordered runner must not use its accepted-L4 continuation shortcut. |
| E19 | `ace3/model/candidates/remaining_layers_v3.py` | Preserve old L8-root suffix plan/public-header equality. A Q24 runner needs root-state provenance rather than loosening `validate_plan`. |
| E20 | `ace3/rtl/ace3_model24_layer_controller.sv` | Existing `layer_done_*`/`checkpoint_*` control interface need not widen. Candidate bridge must withhold successful completion until paired vector/KV and host admission are complete; faults map to `layer_done_fault_i`. |
| E21 | `ace3/contracts/model24_layer_controller.json` | Retain legacy control ABI; new companion binding defines what qualifies as Q24 completed/admitted layer state. |
| E22 | `ace3/contracts/model24_execution.json` | Separate versioned execution binding names root, arithmetic/state policy, layer order and independent KV lineage, not a relaxed legacy execution contract. |
| E23 | `ace3/model/model24_persistent_kv_runtime.py` | Candidate adapter owns per-layer causal KV and completed idle restore; no residual carry between token roots or layer-to-layer opaque-cache import. |
| E24 | `ace3/model/controller_model24_rtl_cascade.py` | Thread actual paired H/I/Z and separate KV dependencies; numerical failure prevents next-layer dispatch/checkpoint admission. |
| E25 | `ace3/model/model24_host_runtime.py` | Candidate root/load/serialization adapter binds token/model/history and immediate producer; default host behavior unchanged. |
| E26 | `ace3/model/run_position2_tail.py` | Bind later candidate H24 to unchanged final RMSNorm/tail checks; do not replay or relabel the accepted historical tail. |
| E27 | `ace3/model/run_tied_lm_head_topk_from_final_rmsnorm.py` | Preserve tied head/logit/top-k arithmetic; changed H24 requires affected tail evidence, not old selected-token import. |
| E28 | `ace3/contracts/position2_tail_binding.json` | Preserve old lineage binding; future Q24-tail binding is distinct and includes candidate H24/selected-token ancestry. |
| E29 | `ace3/model/model24_oracle.py` | Reuse canonical model/config/control authentication; original independently propagated reference remains isolated from candidate data. |
| E30 | `ace3/model/official_model24_next_token.py` | Reuse official tokenizer/model authentication; actual candidate-selected token, not frozen old token IDs, must drive later greedy roots. |
| E31 | `ace3/tb/ace3_single_round_residual_tb.sv` | Existing primitive harness pattern; separate state-core tests must cover persistence and protocol, not just compile or one sum. |
| E32 | `ace3/tb/ace3_decoder_layer0_token_engine_main.cpp` | Candidate host harness loads/captures paired state and uses its own generated model/save symbols; old binary/save layout remains untouched. |
| E33 | `ace3/tb/ace3_model24_layer_controller_tb.sv` | Add candidate-bridge regressions for completion, fault and checkpoint ordering without changing the legacy control contract. |
| E34 | `ace3/model/tests/test_decoder_gate_policy.py` | Preserve legacy/v2 compatibility regressions. |
| E35 | `ace3/model/tests/test_decoder_gate_policy_v3.py` | Preserve local/global/diagnostic separation and genuine local-error/global-drift rejection. |
| E36 | `ace3/model/tests/test_runtime_admission_v3.py` | Preserve old provenance rejection; add separate Q24 tamper/restore tests, not exceptions in old tests. |

Proposed new surfaces below do not exist as delivered capabilities. All reusable
implementation would stay in the four permitted source trees; generated
vectors/logs/objects/results belong under fresh ignored build attempts.

| ID | Proposed later surface | Responsibility |
|---|---|---|
| N01 | `ace3/rtl/candidates/residual_exact_grid_q24_v1/ace3_residual_exact_grid_q24_core.sv` | Exact state-plus-FP16 primitive with paired state/view output and explicit faults. |
| N02 | `ace3/rtl/candidates/residual_exact_grid_q24_v1/ace3_decoder_token_engine_q24_v1.sv` | Isolated accepted-baseline decoder integration, root lift, I/T/pending memories and commit gating. |
| N03 | `ace3/contracts/candidates/residual_exact_grid_q24_v1.json` | Executable arithmetic/precision and exact candidate public-interface contract. |
| N04 | `ace3/contracts/candidates/residual_exact_grid_q24_state_v1.json` | Portable payload, ownership and idle save/restore contract. |
| N05 | `ace3/contracts/candidates/residual_exact_grid_q24_policy_v1.json` | Explicit opt-in local/global/state policy plus execution/tail binding requirements. |
| N06 | `ace3/model/candidates/residual_exact_grid_q24_reference_v1.py` | Independent integer/rational residual oracle and exact root/state-transition checks; no candidate arithmetic import. |
| N07 | `ace3/model/candidates/residual_exact_grid_q24_policy_v1.py` | Numerical adapter; reuse unchanged comparison primitives and non-residual references, not old transaction assumptions. |
| N08 | `ace3/model/candidates/residual_exact_grid_q24_runtime_v1.py` | Root/state codec, capture/admission/host/controller/tail adapter with separate evidence roles and explicit policy selection. |
| N09 | `ace3/model/candidates/run_residual_exact_grid_q24_v1.py` | Ordered, stop-on-failure candidate runner; no default launch or implicit restore. |
| N10 | `ace3/tb/ace3_residual_exact_grid_q24_tb.sv` | General primitive/state/handshake tests; separate candidate C++ decoder harness under ace3/tb is also required. |
| N11 | `ace3/model/tests/test_residual_exact_grid_q24_v1.py` | Independent reference, codec, provenance, policy and integration-unit regressions using existing unittest tooling. |

## 4. Versioned policy and report additions

Retain the selected prospective identities: arithmetic
`ace3-residual-exact-grid-q24-v1`; portable state
`ace3-residual-exact-grid-q24-state-v1`; policy
`ace3-residual-exact-grid-q24-local-global-policy-v1`; claim label
`ace3-w4a16-operators-fp16-kv-q24-residual-proposal-v1`.
Propose distinct public ABI `ace3-decoder-q24-public-abi-v1` and runtime schema
`ace3-q24-actual-runtime-v1`. These are plan selections, not registered versions.
Generated simulator-save ABI is additionally source/tool/binary-bound, never
equated with portable-state or public-port version.

Require exact policy/arithmetic/state/ABI tuple, activation and KV encoding,
weight profile, residual width 64/grid exponent -24, canonical geometry,
source/binary provenance, input/output records and actual producer identities.
Unsupported/missing versions, implicit policy defaults and cross-profile
restore fail explicitly. A future parameter interface may expose width/grid,
but v1 accepts only its declared 64/24 and FP16 encoding; it implements no other
precision by coercion. Existing legacy/v2/v3 entry points and defaults stay intact.

S0-S17 retain finite AND
`(abs_error <= 1/8 OR (relative_error < 1/1000 AND ordered_FP16_ULP <= 1))`,
denominator `max(abs(independent local FP16 reference),2^-14)`. S12 alone
replaces H/O operands with authenticated I/Z/O; S13 still consumes actual rounded
S12. Exact state lineage checks additionally prove T, next I/Z and Q(I)=H,
including signed zero. Authenticate canonical weights, dimensions, RoPE,
positions, mask and cache/control selection independently of candidate choices.

S18 retains ORIGINAL-input independently propagated binary64 r, finite valid
actual H and reference, `abs(r)<=65504`, exact
`q(r)=min_finite_FP16_h |h-r|`, and `|actual_H-r|-q(r) <= 1/8`.
Never clamp negative excess, round/re-anchor r, relax tolerance, or seed global
or legacy trajectory references from candidate hidden/I/KV/local outputs.
S18 exact state-transition validation is additional, not a substitute local
accuracy reference. Missing/unverifiable mandatory evidence blocks/fails.

Report separately: `local_operator_fp16` mandatory S0-S17;
`binary64_v1` mandatory S18; exact `residual_state_lineage`;
independent `kv_lineage`; source/runtime admission; and original whole-FP16
`fp16` trajectory diagnostics with truthful errors and failures. Aggregate
acceptance requires every mandatory component. Unsupported scope/provenance
must be BLOCKED with numerical_status NOT_EVALUATED, not successful zero work.
Software, primitive RTL, actual decoder output and host generation remain
different evidence kinds. Current local reference is P0/history [9707] only:
later positions require independent causal/RoPE/KV support before admission.

## 5. Prospective public ABI and state lifecycle design

This is a reviewable design selection, not an already frozen/compiled interface.
The new decoder top is `ace3_decoder_token_engine_q24_v1`. Preserve every old
header port and width from E03, and integer `LAYER_INDEX=0`,
`ACCURATE_SILU=(LAYER_INDEX>=3)` defaults, under that distinct module name.
Add integer `RESIDUAL_WIDTH=64`, `RESIDUAL_FRAC_BITS=24`, rejecting unsupported
values and layer indices outside 0..23. No alias using the old module name.
The following additions exhaust the proposed decoder sideband ports; they use
the existing load/trace/final/done handshakes rather than independently drifting
state channels. Width expressions refer to the new residual-width parameter.

| Added port | Direction/type/width | Transfer meaning |
|---|---|---|
| load_state_root_i | input wire, 1 | With accepted load_kind_i=0 only: 1 selects authenticated embedding lift and is legal only for L0; 0 selects an actual paired predecessor state for later layers. |
| load_state_q24_i | input wire signed [RESIDUAL_WIDTH-1:0] | Non-root I; root mode requires zero sideband and produces I internally from load_f16_i, never from an oracle. |
| load_state_negzero_i | input wire, 1 | Non-root canonical Z; root mode requires zero sideband and derives Z from the embedding word. |
| load_state_layer_i | input wire [4:0] | Next layer to execute, equal to LAYER_INDEX for all loaded coordinates. |
| load_state_slot_i | input wire [1:0] | Paired-state owner, later matched to start_cache_slot_i. |
| load_state_position_i | input wire [14:0] | Paired-state owner, later matched to start_position_i. |
| trace_state_valid_o | output wire, 1 | Qualifies I/Z sideband on the existing accepted trace transfer, only at S12 or S18. |
| trace_state_q24_o | output wire signed [RESIDUAL_WIDTH-1:0] | T at S12, I_next at S18. |
| trace_state_negzero_o | output wire, 1 | Corresponding canonical zero tag; non-state trace transfers have valid=0 and zero state sidebands. |
| final_state_q24_o | output wire signed [RESIDUAL_WIDTH-1:0] | I_next paired with final_f16_o/index/last under final_valid_o/final_ready_i. |
| final_state_negzero_o | output wire, 1 | Z_next on the same final transfer. |
| done_fault_o | output wire, 1 | Successful complete-vector result or explicit rejected transaction, stable with done_valid_o. |
| done_fault_code_o | output wire [3:0] | 0 success; 1 invalid/nonfinite/XZ data/control; 2 bad tag; 3 integer overflow; 4 FP16 overflow; 5 index/owner/partial-state error; 6 H/state disagreement. Lowest applicable nonzero code wins if simultaneous. Other encodings reserved/rejected. |
| done_next_layer_o | output wire [4:0] | LAYER_INDEX+1; after L23 this is 24 and only a tail consumer is legal. |

Non-root loads require Q(I,Z) bitwise equal to load_f16_i; all 896 coordinates
arrive exactly once in increasing load_index_i order. No start before a complete,
consistent owner/mode vector and required existing parameters are loaded.
Sidebands are nonparticipating for other load kinds. Model/history/root and
producer authentication belong to the host manifest; RTL range checks do not
authenticate a fabricated but internally consistent state. No old FP16-only
caller can silently provide zeros as residual state.

The separate core N01 uses parameters VECTOR_SIZE=896, STATE_WIDTH=64,
FRAC_BITS=24; its proposed ports are clk_i/rst_ni/clear_i; start_valid_i,
start_ready_o, element_count_i[12:0]; in_valid_i/in_ready_o;
in_state_q24_i signed [STATE_WIDTH-1:0], in_state_negzero_i,
in_addend_f16_i[15:0]; out_valid_o/out_ready_i, out_state_q24_o signed
[STATE_WIDTH-1:0], out_state_negzero_o, out_f16_o[15:0],
out_index_o[12:0], out_last_o, out_fault_code_o[3:0], busy_o.
All unspecified widths are one bit. Input/output directions follow the suffixes.
One accepted element produces one registered paired result on the following
cycle; one transfer per cycle when unstalled, with a one-entry elastic output.
Start is accepted only idle, and elements start the following cycle. Fault
codes share the decoder definitions. Invalid counts 0 or >VECTOR_SIZE cannot
start and are diagnosed by the owning harness/decoder, not silently successful.

Both blocks use rising-edge transfers, asynchronous active-low reset, and
synchronous clear with priority over all transfers. Payload/index/last/fault
remain stable under backpressure; high valid without ready never repeats an
update. No new fixed whole-decoder cycle count is promised: existing operator
iterations and stalls determine completion. Successful done becomes eligible
only after the last paired final transfer, drained traces and complete KV work.
On fault, cancel pending numeric transfers, discard partial staged state/KV
validity, and present faulted done; after acknowledgement remain faulted until
reset/clear. Already observed partial outputs are unusable, not a parent.

The host stages all 896 final pairs and current-layer KV before commit. RTL done
is not numerical admission: only successful mandatory checks allow the controller
bridge to acknowledge a usable layer/checkpoint. Clear/reset invalidate live
residual/context/cache validity; stale RAM is not initialization. Retained prior
checkpoints remain immutable; recovery needs explicitly authenticated restore,
not invisible rollback or continuation from a faulted in-memory cache.

Portable state is exactly 896 ordered records of 8 little-endian two's-complement
I bytes plus one Z byte: **8064 bytes**, no padding/trailing data or conversion.
Z must be 0/1 and zero for nonzero I. The enclosing manifest binds arithmetic,
policy, state and ABI versions; width/grid/hidden size; slot 0..3; position
0..32767; next layer 0..24; completed idle validity; model/root/token history;
immediate actual producer; paired H; and separately owned per-layer KV
context/validity/artifacts. Position encoding does not expand the current
CONTEXT_MAX=128 cache capacity; overflow of supported causal storage rejects.

Save/restore only at completed idle layer boundaries. Reject missing, truncated,
trailing, malformed, wrong-version/layer/slot/history/source state, nonfinite
views and Q(I,Z)!=H. Restore must reproduce uninterrupted semantic transactions
and KV under differing stalls. Portable data and generated simulator saves
require different explicit loaders; generated save requires matching compiled
layout/tool/source/binary identity. Old snapshots, padding and FP16-only H are
never migration inputs. Before a later first attempt, the owner must turn this
design into a complete literal header/parameter/latency contract, review it,
freeze it, then compile that exact signature; no compilation occurs here.

## 6. Migration guards and complete later dependency cone

Every position has a new root I0/Z0 from its own embedding; I24 is not the next
position's residual root. Arithmetic/state lineage and own-layer KV lineage
stay separate. New representation begins at root, the first retained remainder
is L0/S18, and the first potentially changed FP16 result is L1/S12 for identical
roots/operators/controls. Include L1/S13-S18, H2 and L2-L23, affected per-layer
KV and every reachable later position, H24/tail/head/top-k, selected-token
feedback and resulting roots. Changed token history invalidates even early-layer
reuse at that position. An L9-only suffix is not this cone.

| Retained class | Later reuse condition / explicit prohibition |
|---|---|
| Official tensors/tokenizer/controls; original references | Reuse compatible authenticated assets and original-input references unchanged, never as execution state. |
| Accepted L0 outputs, O/D and own KV | FP16 arithmetic may be compatible; a new I1 producer still needs complete root/O/D authentication and independent checking. Derivation is separately authorized state production, not an operation done here or software-as-RTL evidence. |
| L1/S0-S11 and own KV | Conditional reuse only for identical H1, arithmetic, control and causal cache ancestry; not proof of new T/I2. |
| Accepted old L1/S12 through L8 | Preserve old-profile acceptance; not presumed new-state compatibility. Require a genuine affected-edge proof, not equal dimensions or selected matching H words. |
| H8/H9, FP16-only or old generated saves | Insufficient to seed/restore Q24; no invented zero carry, global-reference-derived carry or L9 attachment. |
| Failed L9 and suffix controls | Immutable negative evidence/prospective regression only. No forwarding, admission, replay or target-word correction here. |
| Historical other-lineage tail/tokens/KV | Preserve their accepted scope; never splice into the new chain or claim candidate-native generation from fixed token IDs. |

Do not reopen compatible accepted operators for metadata. Missing complete
operands/provenance is a concrete evidence gap, not automatic permission to
replay ancestors. A future state producer and complete affected execution need
their own scoped authorization and independent review. Stop on the first
mandatory numerical, state or provenance failure; do not continue to L9 merely
because that is the retained witness. Later-position oracle support is a gate,
not a reason to apply P0 semantics to a new cache history.

## 7. Unit-level obligations before any model-runtime candidate

These are unexecuted acceptance requirements for a separately authorized
implementation/evidence mission, not results. Freeze independent general case
generation and public inputs before measurement; no candidate-output expected
values, hidden harness inspection or candidate arithmetic import in the oracle.
Use independent exact integers/rationals and independent RNE selection for
residual/state expectations, avoiding host float accumulation/cast as a Q24
oracle. Reuse the existing independent non-residual references only within their
declared scope. Test all scalar/vector entries, not only the retained witness.

| ID | Required invariant / discriminating regression | Surfaces |
|---|---|---|
| U01 | Every finite FP16 encoding lifts/round-trips exactly including signed zero; NaN/Inf reject. No FP16 subnormal flush-to-zero. | N01/N02/N06 |
| U02 | Exact root-plus-actual-O/D sum across multiple layers; a constructed nonzero remainder survives to a later S12. Same H with different valid I is not identical state. No T=B(R) or I=B(H) shortcut. | N01/N02/N06 |
| U03 | Both signs, all signed-zero combinations and nonzero cancellation follow A; Z=1 with nonzero I rejects. | N01/N06/N11 |
| U04 | Adjacent-binade ties, even/odd retained mantissas, subnormal/normal edge, 65504, both sides of 65520 and its exact tie. No saturated placeholder admission. | E06/N01/N06 |
| U05 | Explicit signed 41-to-65 extension, 64-bit extremes and checked narrowing; integer and FP16 overflow diagnosed independently. 43-bit legacy sum is never persistent storage. | N01/N06 |
| U06 | H=Q(I), R=Q(T), next H=Q(I_next) exact including zero sign; S13 consumes rounded R and actual S17 D has that producer. | N02/N06/N07 |
| U07 | Start/load completeness; exactly 896 ordered pairs; count/index/last, duplicate/missing coordinates, layer/slot/position and root-mode errors reject before commit. | N02/N08/N10 |
| U08 | Single-step versus randomly stalled runs have identical transaction sequence and state; held valid updates once, full output/fault tuple stable while stalled, elastic replacement correct. | N01/N02/N10 |
| U09 | Reset/clear during load, arithmetic, trace, final and done cancel pending work; fault after partial KV/final output cannot commit or forward. Clear priority over handshake. | N02/N08/N10 |
| U10 | Participating X/Z data/control rejected with a real four-state simulation test; two-state Verilator alone is not evidence of X/Z handling. Invalid-count rejection is explicitly observed, not counted as zero-work PASS. | N01/N02/N10 |
| U11 | Codec exactly 8064 bytes, endianness/tags and byte round-trip; missing/trailing data and wrong ABI/source/history/root reject. Idle restore equals uninterrupted state/KV under stalls; mid-layer save rejects. | N04/N08/N11 |
| U12 | Each new position lifts its own E; own-layer KV causal order and context lengths independent of residual chain. Cross-layer cache, prior-position I24, forged owner or FP16-only state cannot seed. | N06/N08/N11 |
| U13 | Legacy/v2/v3 unchanged; explicit new policy only. Uniform local errors still fail; locally correct but globally drifting outputs still fail original binary64-v1. Strict relative and inclusive absolute/excess edges unchanged. | E34/E35/N07/N11 |
| U14 | Missing/nonfinite/wrong-source references and wrong canonical tensors/metadata reject; changing candidate hidden/I/KV cannot modify original-global/legacy expectations. Tampered actual sideband/producer fails even when local numbers agree. | E36/N06/N07/N08/N11 |
| U15 | Captured raw I/Z/H, state traces, source/header/binary/save layout and host-provided trust binding agree; software screen, actual RTL and NOT_EVALUATED reports cannot be interchanged. | N08/candidate C++ harness |
| U16 | L0 single-round compatibility and L1 attention reuse require full dependency proof; new root/state is still produced. Tail consumes H24 only and a candidate-selected token alone drives subsequent greedy root. Failed gates prevent dispatch. | N02/N08/N09/N11 |

Later use existing unittest tooling, grouping E34-E36 and N11 in one invocation;
primitive/cycle tests use a project-local candidate harness and fresh ignored
output paths, not new stale success stamps. Probe current host tools, then
declared local containers only when execution is actually authorized; serialize
shared container runtime access. First freeze and compile exact public contracts,
then execute semantic tests with real tools. Compile success is not arithmetic
PASS. No current tool availability, simulator/formal result, performance root
cause, timing, synthesis, area, FPGA or hardware evidence is asserted here.

## 8. Failed delivery-check evidence hygiene

The adoption-boundary assessment named in section 1 retains its exact command
and **FAILED: one invocation, exit code 1; no rerun**, with no stdout and an
AssertionError at its final assertion. Its raw Markdown substring count includes
the three-backtick literal inside its own fenced Python command. The historical
record attributes the failure to that check, after the earlier assertions had
been reached successfully; this plan does not rerun them or invent success stdout.
Primary taxonomy: **document-validation**. Root-cause hypothesis: counting
inline fence substrings instead of delimiter lines causes a false rejection.
Regression: a valid fenced command containing that literal must parse, and an
unclosed fenced command must fail. The earlier specification-assessment
terminal-newline failure is a different record and must not be conflated.

Preserve the original assessment, command, traceback, failed exit and later Host
receipt unchanged. No old FAIL-to-PASS rewrite, failed-check replacement or
extra review receipt is allowed. The correction is additive: this mission's
directly related check record freezes a different, line-aware document check
before its single invocation, records actual status/stdout/stderr, and labels
its parser regression and delivery scope only. It may read the historical
Markdown as data, but never executes the embedded historical command or any
arithmetic/evaluator. Genuine review acceptance and successful formatting
validation remain distinct; neither repairs RTL or grants a launch.

The only mission writes are this new plan, its directly related record at
`build/exact_grid_q24_source_change_plan_9885c390465c_attempt001/check.md`,
and the explicitly designated external CHECKPOINT.md. No production semantic
change, contract/evaluator/reference mutation, arithmetic prototype, oracle
regeneration, RTL/simulator invocation, output capture/admission, L9 replay or
failed-state forwarding occurs. No Wiki/Skill update, commit or push is in scope.

## 9. Reviewer-ready checklist for the later implementation mission

- [ ] Accept/reject this plan through the normal independent Host Reviewer path; distinguish document acceptance from adoption and any launch permission.
- [ ] Obtain the separate delegated-parent scoped adoption/implementation disposition, including the wider-state claim label; do not ask again for the already accepted decision to specify this hypothesis.
- [ ] Owner-review the complete candidate decoder/core headers, parameters, fault/commit/reset/stall protocol and portable/generated-save boundaries; unresolved design fields block adoption rather than being guessed by execution.
- [ ] Re-probe source closure and dirty worktree; isolate N01-N11 from the accepted frozen single-round basis, preserve non-residual operators and all legacy/v2/v3 defaults/FAILs.
- [ ] Implement explicit arithmetic/state/policy/ABI schemas and migration rejection; wire root loader, paired state capture, independent local/state oracle, unchanged global references, host/controller and separate KV ownership end to end.
- [ ] Under separately permitted prerequisite evidence scope, freeze prompt/evaluator/public inputs/tools/score policy and exact headers before the first official attempt; compile the exact contract, then execute U01-U16 as applicable with independent expectations and authenticated inputs.
- [ ] Preserve every failing seed/log/waveform and immutable first attempt; record each repair separately with one taxonomy, root-cause hypothesis and regression. Never weaken tests or change a reference to obtain PASS.
- [ ] Obtain genuine independent review of source, oracle independence, unit evidence and provenance before requesting any model-runtime scope; document-only checks cannot satisfy that gate.
- [ ] Resolve per-artifact root/L0/L1 reuse and missing-state-producer gaps, then authorize only the complete genuinely affected causal cone. No H8/H9 seed, failed-state forwarding, old-tail splice or unchanged replay for metadata.
- [ ] Add independently reviewed later-position reference/cache support before using those positions; admit every layer's actual paired output before its dependent dispatch.
- [ ] Only after separately authorized decoder/tail gates pass, bind actual H24, final RMSNorm, tied head/top-k, tokenizer and candidate-selected feedback with persistent own-layer KV. Report the exact achieved scope, not dialogue, precision promotion or hardware.

