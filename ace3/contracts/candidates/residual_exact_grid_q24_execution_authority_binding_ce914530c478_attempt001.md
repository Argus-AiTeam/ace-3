# Q24 isolated primitive/state execution authority binding

Mission `ce914530c478`; node `q24-authority-binding-for-execution`;
plan `plan-1426105e5608` v3; attempt001.

**DOCUMENT-ONLY BINDING OF EXISTING MANAGER AUTHORITY.**
Computer-local W4A16 RTL/model implementation, simulation, host/runtime
integration, regression, and persistent K/V work are authorized for the
follow-on **Q24 isolated primitive/state route** defined below. This mission
implements and executes none of those surfaces. Normal independent Host
Reviewer assessment of this binding is pending.

## 1. Primary authority and temporal reconciliation

Authoritative source:
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/planner-waiting-contract-a575af38a55f9f0c.json`,
field `manager_resolution.reason`, verbatim:

> HOLD in rtl: the checklist is not evidenced enough to advance, but the stale implementation block is resolved because the pre-existing operator objective explicitly authorizes computer-local W4A16 RTL/model implementation, simulation, host/runtime integration, regression, and persistent K/V work. Planner should replan current-stage implementation/verification work instead of waiting on the specification-only boundary.

The same record has `manager_resolution.resolved_at=1788937474.7925315`,
`manager_resolution.target_stage="rtl"`, `active=false`, and
`operator_action_required=false`. **HOLD concerns stage advancement, not an
unresolved specification-only implementation wait.** The record's
`recheck_condition` names a separate durable task; it supplies no Q24 execution,
numerical result, or accepted state and is not a task to rerun here.

The older specification directive is still retained at
`/home/argustest/.argus-skill-ace3/projects/s-62150b05/active_manager_directive.json`,
with `set_at=1788933417.9750104`,
`source="operator.delegated-parent-residual-specification"`, and decision
`authorize_bounded_semantics_specification`. Its file name or repeated
injection does not make its earlier blanket prohibition newer than the
explicit Manager wait resolution. The two records bind the same objective
fingerprint. The current mission contract explicitly selects this later
resolution for the isolated route; it does not authorize execution in this
document-only mission.

The following retained no-execution conclusions are **superseded for this
route**, not overwritten or declared erroneous when originally issued:

| Retained record | Earlier restriction and later genuine review |
|---|---|
| `ace3/contracts/candidates/residual_exact_grid_q24_source_change_plan_9885c390465c_attempt001.md` | `DOCUMENT_ONLY_SOURCE_CHANGE_PLAN_NOT_ADOPTED. Authorized execution cone: empty.` Normal Host receipt `/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/9885c390465c/round-0001.json` has `producer_role=reviewer`, `review.status=done`, `created_at=1788936235.2373126`. |
| `ace3/contracts/candidates/residual_exact_grid_q24_authority_reconciliation_dcec8877084c_attempt002.md` | `CLEARANCE_NOT_ESTABLISHED. IMPLEMENTATION_BLOCKED.` Normal Host receipt `/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/dcec8877084c/round-0002.json` has `producer_role=reviewer`, `review.status=done`, `created_at=1788937370.5214102`. |

Both reviews predate the Manager resolution. Their acceptance of documents
never granted implementation authority; the later Manager resolution supplies
the authority bound here. In particular, the reconciliation's requested
implementation-only exclusion of compilation, oracle execution, regressions,
and simulation is not the scope of that later resolution. Its request is not
an issued Manager decision. Earlier specification/adoption documents and
frozen pending-review wording remain historical, not additional launch waits.
No old review, FAIL, freeze, manifest, seal, source, or consumed authority is
modified, renewed, or relabeled by this binding.

There is no unresolved `ambiguous_objective` for this bounded binding: the
dated replacement and current mission resolve the route-specific restriction.
Restrictions outside this isolated route remain intact. This is neither a
new human answer nor an Engineer-issued Manager/Reviewer verdict.

## 2. Exact permitted follow-on surfaces

Paths below are repository-relative ownership targets from the reviewed
source-change plan, not claims that their implementation already exists.
Permission is restricted by responsibility, not merely by a filename.
Existing files are not disposable; any follow-on must integrate existing work
or use a fresh candidate path without replacing historical evidence.

| Surface | Permitted bounded implementation and execution |
|---|---|
| Q24 ABI/header/contract freeze: `ace3/contracts/candidates/residual_exact_grid_q24_v1.json`, `ace3/contracts/candidates/residual_exact_grid_q24_state_v1.json`, `ace3/contracts/candidates/residual_exact_grid_q24_policy_v1.json` (N03-N05) | Freeze the exact primitive module/top, ports, directions, widths, signedness, parameters, clock/reset/handshake/fault behavior, and portable state/codec boundary. Limit policy fields to the isolated primitive/state route; this does not adopt a decoder/model policy. |
| Primitive RTL: `ace3/rtl/candidates/residual_exact_grid_q24_v1/ace3_residual_exact_grid_q24_core.sv` (N01) | Implement the general exact-grid residual primitive and its state/FP16-view interface under the separately frozen contract, not a witness-specific repair or a decoder integration. |
| Independent oracle: `ace3/model/candidates/residual_exact_grid_q24_reference_v1.py` (N06); primitive policy checks: `ace3/model/candidates/residual_exact_grid_q24_policy_v1.py` (N07) | Implement and execute independent integer/rational arithmetic, FP16 conversion, root/state-transition and rejection checks. The oracle must not import candidate arithmetic or read candidate outputs as expectations. No global/legacy model-reference regeneration or decoder re-adjudication. |
| Independent codec and host/runtime state boundary: the isolated codec portion of `ace3/model/candidates/residual_exact_grid_q24_runtime_v1.py` (N08), bound to N04 | Implement/test root initialization, serialization, reset, save/restore, payload identity and invalid-state rejection. Persistent K/V work is limited to separate FP16 cache ownership, identity and persistence plumbing exercised with isolated fixtures; it does not create or admit candidate model K/V history. Keep residual arithmetic lineage separate from K/V/state lineage. |
| Primitive RTL harness: `ace3/tb/ace3_residual_exact_grid_q24_tb.sv` (primitive portion of N10); focused runner portion of `ace3/model/candidates/run_residual_exact_grid_q24_v1.py` (N09) | Build and run only the isolated primitive/state harness against independently generated, authenticated inputs. N09 is not permission for its proposed ordered decoder/model runner. Any host harness must remain primitive-only under `ace3/tb`. |
| Compile/import/elaboration and focused primitive regressions: the above candidate files and `ace3/model/tests/test_residual_exact_grid_q24_v1.py` (primitive/codec portion of N11) | Compile the exact frozen public contract before the first semantic attempt; import the isolated oracle/codec; elaborate the primitive top; run focused arithmetic, state, protocol and rejection regressions with the real available tools. Compile success alone is not numerical or RTL PASS. |

The ABI must be explicit before execution; no guessed compatibility aliases.
Focused regressions must cover signed zero, finite/subnormal/rounding and
overflow behavior, invalid encodings/tags, participating X/Z, reset,
backpressure/output stability, and portable-state round trips/rejection as
applicable to the frozen contract. Do not weaken expected values or assertions.
Generated vectors, simulator objects, logs, waveforms and failed attempts
belong in fresh ignored build attempts authorized by the follow-on task, not
in reusable source or any old attempt. Preserve the first attempt and record
repairs separately.

This binding grants no result admission. Actual independent oracle evidence
and normal independent Reviewer validation remain required before advancing
an implemented capability or any later affected cone. No Reviewer subagent
or substitute approval record is authorized by this document.

## 3. Explicit exclusions and unchanged precision/evidence boundary

- **No decoder/model/token execution or claims:** N02
  `ace3/rtl/candidates/residual_exact_grid_q24_v1/ace3_decoder_token_engine_q24_v1.sv`,
  decoder C++ integration, N08 capture/admission/controller/tail integration,
  the N09 layer/model runner, full-model reference sweeps, layer replay,
  final RMSNorm/lm_head/top-k/tokenizer generation, third-token or dialogue
  claims are outside this isolated route. Primitive success is not model
  fidelity, a repaired L9, a completed W4A16 path, or stage advancement.
- **No accepted-state forwarding:** no root reconstruction from a rounded
  accepted hidden vector, invented carry at L9, reference-minus-actual carry,
  old accepted/failed state used to seed a Q24 continuation, A/B splicing,
  incompatible opaque-cache import, or forwarding primitive test state as an
  admitted decoder/model checkpoint. Historical outputs may remain read-only
  context; accepted or consumed evidence must not be replayed merely to restamp
  authority.
- **No hardware work or claims:** FPGA, synthesis, timing/PPA, bitstream,
  board/accelerator deployment and deployed-hardware evidence remain forbidden.
  Only computer-local numerical reference, RTL simulation and isolated
  host/runtime/state integration are within the follow-on permission.
- **No historical evidence mutation:** preserve every failure, reference,
  attempt, review, source/binary snapshot and arithmetic/state lineage.
  No production/legacy source or evaluator mutation, commit, push, runtime
  reconfiguration, duplicate worker or interference with ACE-2 is authorized.

Native official AWQ remains G128 asymmetric packed INT4 in GEMM nibble order,
without qzero plus-one, with FP16 scales, operator activations and K/V. Keep
all active local FP16 and ORIGINAL-input global binary64-v1 references,
thresholds, diagnostics and failures unchanged. Do not inject reference
hidden/Q/state into execution or treat local agreement as global admission.

The prospective label
`ace3-w4a16-operators-fp16-kv-q24-residual-proposal-v1` explicitly denotes
wider persistent residual state: signed 64-bit Q24 plus a zero-sign tag.
It is **not unchanged strict W4A16**, an adopted whole-model arithmetic/
reference boundary, or advancement to W8A16/BF16/FP16. The primitive experiment
does not promise that wider state repairs the model. Keep weight, activation,
K/V, residual and policy interfaces distinct for the later precision roadmap.

## 4. This mission's actual result boundary

This fresh Markdown binding and the designated mission CHECKPOINT are the
only files authored here. Repository/Manager metadata and existing documents
were inspected read-only. No production or candidate implementation, public
ABI freeze, project-module import, compilation, elaboration, oracle arithmetic,
test/regression, simulator, model, tokenizer, host generation or K/V production
was performed.

RTL/numerical status is **NOT_EVALUATED (deliberate document-only scope)**.
It is not a numerical PASS/FAIL or an evaluator/tool-unavailability finding.
The document check is limited to the quoted authority, chronology, scoped
surface/exclusion coverage and mission binding. Normal independent Host
Reviewer acceptance remains a separate, pending outcome.
