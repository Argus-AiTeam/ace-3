# Q24 decoder/model-root runtime directive reconciliation

Attempt `q24_runtime_directive_reconciliation_3b2ad3e1ac7b_attempt001`;
mission `3b2ad3e1ac7b`, node `q24-runtime-directive-reconciliation`;
plan `plan-57521c722cb7` v5, bounded `rtl` stage.

**AUTHORIZED_FOR_BOUNDED_FOLLOW_ON_IMPLEMENTATION_AND_RUNTIME: P0/L0-L8.**
The current Manager wait resolution is controlling for this cone. The older
specification-only prohibition is **RETIRED_FOR_P0_L0_L8**, not erased from
history. This reconciliation executes nothing and establishes neither launch
readiness nor an accepted Q24 model parent. Normal Host Reviewer assessment of
this fresh disposition is still required.

## Primary authority and exact conflict

The direct Manager decision is retained in
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/agent_io.jsonl`,
line 26103, event `type=agent.io.stream`, `stream=stdout`,
`ts=1788937473.8096473`. Decode the event's `line` as JSON, then read
`data.content`'s first-line `ARGUS_ROLE_DECISION`. Its actual payload has
`role=manager`, `action=hold`, `target_stage=rtl`, **`resolves_wait=true`**.
The reason is verbatim:

> HOLD in rtl: the checklist is not evidenced enough to advance, but the stale implementation block is resolved because the pre-existing operator objective explicitly authorizes computer-local W4A16 RTL/model implementation, simulation, host/runtime integration, regression, and persistent K/V work. Planner should replan current-stage implementation/verification work instead of waiting on the specification-only boundary.

The corresponding persisted resolution is
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/planner-waiting-contract-a575af38a55f9f0c.json`:
`wait_id=16806f94e8c87b9de43ab422`,
`manager_resolution.resolved_at=1788937474.7925315`,
`manager_resolution.target_stage=rtl`, `active=false`,
`operator_action_required=false`, `stage_reconciliation_required=false`.
Its `manager_resolution.reason` equals the direct decision's reason.
The persisted wait object does not itself serialize `resolves_wait`; the
Boolean above is sourced from the actual Manager message, not invented as a
field in that object. Its durable-task recheck condition is not a Q24 launch
receipt or a direction to rerun that task.

The conflicting text remains physically present in
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/active_manager_directive.json`,
revision `b1be43f06593421ca76130d0210dc43a`,
`source=operator.delegated-parent-residual-specification`,
`set_at=1788933417.9750104`:

> Decision: authorize_bounded_semantics_specification.
>
> One bounded, non-executing mathematical/state specification for a general
> residual recurrence that may preserve information lost at FP16 boundaries.
>
> No RTL/simulator/capture/admission, no candidate software trajectory, no
> prototype arithmetic implementation, no oracle regeneration or sweep, no
> accepted-ancestor replay, no L9 replay, no L10-L23 execution, no failed-state
> forwarding, no production source/contract/evaluator/reference changes.
>
> Passing that document review alone never grants a runtime launch.

Both authority records bind objective
`a575af38a55f9f0cdac7db22bfeeba28cbb33d6a38a55ea2d07a635843bad80d`.
The explicit wait resolution postdates that directive. The current mission
objective specifically requires applying this resolution to the decoder/
model-root cone and retiring the stale restriction there. Repeated injection
of the same older revision is not a materially different authority.

Accordingly, the earlier non-execution-only limit no longer controls
prospective P0/L0-L8 candidate implementation, software evidence, RTL
simulation, capture/admission, or necessary independent-reference work.
The Manager decision and this live scope instruction supply that replacement;
neither Engineer authorship nor document acceptance creates Manager authority.
The stage HOLD remains: no project-stage advancement is granted. All unrelated
restrictions survive, including no failed-state forwarding, production/legacy
mutation, unchanged replay, numerical waiver, or automatic launch on document
review. No unresolved `ambiguous_objective` remains for this bounded
reconciliation on the inspected authorities. A different active revision or
an explicit later cone-specific prohibition would require a fresh disposition,
not reuse of this finding.

## Historical inputs remain historical

Repository-relative paths below refer to `/home/argustest/ace3-argus`.

| Preserved identity | Unchanged result boundary |
|---|---|
| `ace3/contracts/candidates/residual_exact_grid_q24_model_root_runtime_parent_eae362d506e6_attempt001.json` and adjacent `.md`; attempt `q24_model_root_runtime_parent_eae362d506e6_attempt001` | Completed runtime-parent binding, inherited here without replacement. Normal Host receipt `/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/eae362d506e6/round-0001.json` has `producer_role=reviewer`, `review.status=done`, `created_at=1788941489.4470797`. Its frozen pending-review strings predate this receipt; its `NO_EXECUTION` state remains true. |
| `ace3/contracts/candidates/residual_exact_grid_q24_decoder_adoption_c39ca7b7722a_attempt001.json` and adjacent `.md` | Earlier no-execution adoption specification, not retroactively a runtime grant or a root/L8 producer. It remains the eae binding's baseline. |
| `ace3/contracts/candidates/residual_exact_grid_q24_authority_reconciliation_dcec8877084c_attempt002.md` | Retained `CLEARANCE_NOT_ESTABLISHED. IMPLEMENTATION_BLOCKED.` document. Its narrowed request was a request, not the later Manager decision. |
| `/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/e16f727570de/round-0001.json`, `created_at=1788941640.6165469` | Retained genuine `replan_requested` review: `ambiguous_objective` because a project-local binding alone cannot replace the controlling specification directive. No L0-L8 execution or authenticated L8 parent was established. This reconciliation binds the actual Manager replacement and live corrective objective; it does not rewrite that review as PASS. |
| `/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/6e9c03218ac8/latest.json` and its referenced sealed handoff | Earlier no-execution diagnostic identity retained, not re-adjudicated or promoted to runtime evidence. |
| Isolated binding `ce914530c478`, primitive `q24_primitive_a8b9021c084b_attempt003` and its existing review | Remain isolated primitive/state evidence, not decoder/model execution, candidate root state or a whole-model result. |

This new attempt does not overwrite, renew, consume, or relabel any of these
records. The preservation check records their observed file identities; it
does not manufacture model-call attestations or recertify old numerical data.

## Exact follow-on scope and remaining prerequisites

The eae JSON's `selected_contract`, `scope`, `state_production`,
`prerequisites`, and inherited baseline sections remain the technical binding.
Primary surfaces are N02 separate decoder RTL, N08 root/state codec,
capture/admission and controller/host checkpoint plumbing, and N09 the ordered
software/RTL runner, with N03-N07/N10-N11 support only as needed for this cone.
Permission is to implement and establish evidence, not to overwrite existing
candidate files or replace a legacy module.

Start with authenticated model-root embedding for fixed P0 history `[9707]`;
produce L0 through L8 in order, terminating at paired `I9/Z9/H9`,
`next_layer=9`. Each layer owns its separate empty P0 prior FP16 K/V.
Only actual independently admitted paired `I/Z/H` outputs may feed a non-root
consumer. No FP16-only non-root seed, invented or reference-derived carry,
primitive fixture as model parent, incompatible restore, A/B splice, or
failed-state forwarding is permitted.

Before a first semantic attempt, freeze and compile the complete exact public
module/port/parameter contract without aliases; authenticate source, binary,
tools, canonical inputs and original references. Independent software/state
admission and its required review precede decoder RTL. Check every applicable
mandatory gate before forwarding; retain failed attempts and obtain independent
oracle plus normal Reviewer validation before advancing the capability.
Compatible accepted work is reusable; only genuinely affected or missing
state/hidden/K/V dependencies justify fresh execution. This is not another
per-layer permission requirement inside the admitted bounded route.

Native G128 asymmetric packed INT4 GEMM nibble ordering, no qzero plus-one,
FP16 scales/operator activations/KV, exact Q24 residual state and zero tags
remain as bound by eae. The wider-state candidate is explicitly
`ace3-w4a16-operators-fp16-kv-q24-residual-proposal-v1`, **not unchanged strict
W4A16** or another precision milestone. Keep the explicit changed S12 operands,
unchanged mandatory local FP16 thresholds, ORIGINAL-input global binary64-v1
excess rule, independent residual/KV lineages and legacy trajectory diagnostics.
No policy, reference, rounding, threshold or arithmetic repair is adopted here.

L9-L23, later positions, tail, token generation and dialogue are outside this
authorization. No full-model, hardware, synthesis, PPA, bitstream, deployment,
publication, commit, push, new worker or runtime reconfiguration is authorized.

## Actual check and handoff boundary

One document/metadata check is provided at
`build/q24_runtime_directive_reconciliation_3b2ad3e1ac7b_attempt001/check.py`;
its exclusive-create `validation.log` records the actual outcome. It checks
the direct Boolean decision, persisted reason/chronology/objective identity,
the exact conflicting directive revision, historical review/status boundaries,
and this document's required scope. No new runtime schema or public RTL ABI is
implemented; candidate-module imports, RTL compilation and arithmetic checks
are inapplicable here and are not run.

This mission's RTL/model/state result is **NO_EXECUTION** and numerical
correctness is **NOT_EVALUATED**, deliberately, not evaluator unavailability.
Static success is not RTL correctness, launch readiness or a Reviewer verdict.
Only this fresh reconciliation, its local check/log and the designated
CHECKPOINT are authored. Pipeline state, active directive, immutable evidence,
production sources, Wiki and Skills are not edited. The Host owns the required
independent Reviewer step.
