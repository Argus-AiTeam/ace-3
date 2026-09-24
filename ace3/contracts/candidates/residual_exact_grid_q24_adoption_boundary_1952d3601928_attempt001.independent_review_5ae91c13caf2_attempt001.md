# Independent no-execution assessment of the Q24 adoption boundary

Mission `5ae91c13caf2`; node `exact-grid-q24-adoption-boundary-review`;
review attempt001. Assessment author role: Engineer, separate from the
adoption-record author. Normal independent Host Reviewer closure of this
mission remains **PENDING_NORMAL_HOST_REVIEWER**; this file is not a
Host-issued `producer_role=reviewer` receipt.

**Assessment verdict: ACCEPT for the bounded adoption-boundary document.**
No blocking discrepancy was found against the accepted specification and its
later review. This verdict accepts the accuracy and limits of the document,
not Q24 adoption, a complete new ABI, arithmetic correctness in execution,
numerical improvement, an implementation, or a launch. Authorized execution
cone remains empty. No `ambiguous_objective` conflict was identified: the live
no-execution review is narrower than the standing model-continuation mission.

## Evidence actually inspected

- Subject: `ace3/contracts/candidates/residual_exact_grid_q24_adoption_boundary_1952d3601928_attempt001.md`, sections 1-8.
- Normative basis: `ace3/contracts/candidates/residual_exact_grid_q24_semantics_51efa8e34614_attempt001.md`, sections 1-8.
- Later specification review: `/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/51efa8e34614/round-0001.json`. Observed `kind=round_reviewed_handoff`, `producer_role=reviewer`, mission `51efa8e34614`, round 1, `review.status=done`, created_at `1788934442.939863`; reason: "independent review accepts the bounded specification against the parent-authorized requirements".
- The older specification `.review.md` was read. It is an Engineer structural-check record with pending-review wording, not a competing later rejection or the independent acceptance receipt.
- Frozen residual core and decoder wiring under `build/single_round_residual_rtl_execution_fb06353c5f53_attempt001/source/`; the old exact header in `build/l9_l23_continuation_10d99fb0a67e_attempt002/freeze.json#/public_contract`.
- Current `ace3/contracts/candidates/decoder_gate_policy_v3.json`, relevant operand/lineage definitions in `ace3/model/candidates/local_operator_reference_v3.py`, and controller completion/checkpoint declarations in `ace3/rtl/ace3_model24_layer_controller.sv`.
- Retained `build/l9_w4a16_hypothesis_adjudication_a79ca09169d1_attempt001/adjudication.md`; Wiki INDEX and the numerical-boundary page's v3 discussion as secondary context.

This is direct source/document inspection, not a repeat authentication of all
historical execution closures. The retained adjudication states accepted actual
P0/L0-L8, mandatory-global L9/S18/index62 FAIL, and four failing canonical
suffixes. Those remain attributed retained observations, not measurements
performed by this review. The specification review does not establish a later
passing trajectory. Its receipt metadata was inspected as stored; this task
does not manufacture or independently re-attest its underlying model call.

## Requirement-level findings

| ID | Requirement and subject location | Finding and rationale | Disposition |
|---|---|---|---|
| R01 | Policy delta; sections 2-3 | Correctly distinguishes `ace3-residual-exact-grid-q24-v1`, `ace3-residual-exact-grid-q24-state-v1`, and prospective `ace3-residual-exact-grid-q24-local-global-policy-v1`. Existing v3 is not silently extended. Its S12 H/O reference and exact S18 H+O+D lineage check need separate new-state semantics. Original local thresholds and original-input global binary64-v1 remain mandatory. | ACCEPT |
| R02 | Claim/precision label; section 2 | `ace3-w4a16-operators-fp16-kv-q24-residual-proposal-v1` discloses wider persistent residual state, not unchanged strict W4A16. Native G128 packed INT4, nibble ordering, no qzero +1, FP16 scales/operator activations/KV, and FP16 tail input are preserved. Width/scale/encoding/policy remain separately exposed for later modes, with no milestone promotion. | ACCEPT |
| R03 | Recurrence and ownership; sections 2 and 4 | Uses exact I+O then T+D with separately rounded R for MLP input, not rounded R as residual state. Signed 64-bit Q24 plus canonical zero tag, checked 65-bit sums, gradual underflow and overflow rejection agree with specification sections 2-4. Derived carry is diagnostic only; neither reference-minus-H nor a witness-selected correction produces state. | ACCEPT |
| R04 | Source and contract touchpoints; sections 3-4 | Identifies frozen baseline separately from mutable RTL, local oracle/lineage, policy adapters, unchanged global metric, capture/admission, ordered runners, controller/host, persistent KV, tail/tokenizer and test owners. Inspected frozen decoder lines 458-465 feed activation/O to S12 and activation/O/down to S18. Local reference lines 12-18, 77-101 and 140-141 confirm the old operand and P0 restrictions. Controller declarations distinguish completion metadata from a residual payload. Other named surfaces are ownership anchors, not an assertion of source equivalence or implemented Q24 support. | ACCEPT |
| R05 | Public ABI/lifecycle; section 4 | Names the exact frozen public header rather than inventing aliases. Existing LAYER_INDEX/ACCURATE_SILU defaults and FP16 load/trace/final ports match that header. Explicit new state transport, atomic 896-coordinate completion, reset/clear/fault invalidation, stable stalled transfers and idle-only restore remain future ABI obligations. The 896 ordered 9-byte records, 8064-byte payload, canonical Z and root/history/layer/KV bindings match specification section 4. | ACCEPT |
| R06 | Earliest root lineage; section 5 | Every position starts from its authenticated FP16 embedding. L0/S18 is the first newly retained remainder; L1/S12 the first potentially changed FP16 result under identical controls/arithmetic. L0/root state establishment is still necessary despite compatible old L0 FP16 results. Neither H8/H9 nor I24 from another position is a root. | ACCEPT |
| R07 | Full future affected cone; section 5 | Includes L1/S12 MLP/final-state descendants, H2 and L2-L23, changed own-layer K/V and reachable later positions, H24/final RMSNorm/head/logits/Top-K, selected-token feedback and subsequent embeddings. Changed token selection invalidates even early-layer reuse for the changed history. Does not mislabel an L9-only suffix or fixed-history experiment as candidate-native generation. | ACCEPT |
| R08 | Reuse versus incompatible evidence; section 6 | Conditionally reuses authenticated tensors, original references and compatible L0/L1-prefix operator evidence; requires separate arithmetic and cache compatibility. Old L1/S12-L8 acceptance remains old-profile evidence. FP16-only saves, H8/H9, failed L9 and other-lineage tail/KV are not new-state seeds. Retained-operand state reconstruction would itself be separately authorized state production, never software represented as RTL. | ACCEPT |
| R09 | Later sequence/pre-authorization; section 7 | Separates document review, delegated-parent adoption/implementation decision, ABI/source/oracle work, explicitly permitted prerequisite evidence, independently reviewed runtime scope, ordered affected-cone work and tail/feedback. Requires a complete owner-reviewed ABI design before an initial adoption request, without requiring implementation results circularly. Unresolved names/timing/design are declared blockers to that request, not hidden authorization. | ACCEPT |
| R10 | Falsification, scope and preserved failures; sections 7-8 | Requires general rounding/state/handshake/restore and invalid-input cases, continued local-error/global-drift rejection, legacy/v2/v3 preservation, unchanged original-global provenance, later-position oracle support, immutable attempts and stop-on-failure. The known L9 word is a regression, not the target definition. No arithmetic, oracle, simulation, output admission, failed-state forwarding, performance cause, dialogue or hardware result is asserted. | ACCEPT |

The recurrence's root argument is algebraic: I0 is B(E), so its first layer
has the same exact residual sums as the old single-round layer. Carry first
survives that layer's final rounding; it can then change the next S12 and its
MLP, hence H2 and downstream cache producers. No candidate arithmetic was
evaluated to make this dependency comparison.

## Concrete limits retained by acceptance

1. **No adoption or launch readiness.** The new full signature, latency,
   fault/commit protocol, save ABI and implementation are intentionally not
   supplied. Section 7 correctly blocks an initial adoption request until its
   owner-reviewed design prerequisites exist. Accepting this boundary record
   must not be interpreted as satisfying those prerequisites.
2. **No later-position oracle support established.** Current v3 validation
   explicitly restricts position/history to P0 and `[9707]`; future causal
   positions need independent, reviewed reference implementation before use.
   Naming the whole future cone does not execute or admit it.
3. **No reusable Q24 state exists by inference.** Compatible old FP16 operator
   outputs can reduce future replay, but cannot alone prove I/Z or a new
   saved-state producer. The exact root-to-state construction and its evidence
   remain separate obligations; matching H is insufficient.
4. **No measured repair claim.** The fixed-increment exact-sum rationale in the
   accepted specification is conditional. Changed nonlinear outputs/KV may
   defeat global improvement. L9 remains retained failure evidence and cannot
   be forwarded. No independent oracle or numerical result was generated here.

These are explicit deferred requirements, not defects requiring modification
of the reviewed adoption record. No production or normative-source amendment
is requested by this assessment.

## Single delivery-check command sidecar

The following literal command is recorded before its one invocation. It checks
delivery structure, named path existence, the already inspected review metadata
and old public signature only. It imports no project arithmetic or evaluator,
does not run tests/RTL/oracles, and does not establish source equivalence.

```sh
python3 - <<'PY'
from pathlib import Path
import json
import re
import subprocess

root = Path("/home/argustest/ace3-argus")
subject = root / "ace3/contracts/candidates/residual_exact_grid_q24_adoption_boundary_1952d3601928_attempt001.md"
spec = root / "ace3/contracts/candidates/residual_exact_grid_q24_semantics_51efa8e34614_attempt001.md"
record = root / "ace3/contracts/candidates/residual_exact_grid_q24_adoption_boundary_1952d3601928_attempt001.independent_review_5ae91c13caf2_attempt001.md"
receipt = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/51efa8e34614/round-0001.json")
branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
assert branch == "argus/full-projection", branch
subject_text = subject.read_text()
assert spec.is_file()
anchors = set(re.findall(r"`((?:ace3|build)/[^`\n]+)`", subject_text))
missing = sorted(path for path in anchors if not (root / path.split("#", 1)[0]).is_file())
assert not missing, "Missing subject anchors: " + repr(missing)
review = json.loads(receipt.read_text())
assert review["kind"] == "round_reviewed_handoff"
assert review["producer_role"] == "reviewer" and review["mission_id"] == "51efa8e34614"
assert review["round"] == 1 and review["review"]["status"] == "done"
assert review["review"]["reason"] == "independent review accepts the bounded specification against the parent-authorized requirements"
freeze = json.loads((root / "build/l9_l23_continuation_10d99fb0a67e_attempt002/freeze.json").read_text())
header = freeze["public_contract"]
for token in ("module ace3_decoder_layer0_token_engine", "parameter integer LAYER_INDEX = 0", "parameter integer ACCURATE_SILU = (LAYER_INDEX >= 3)", "input wire [15:0] load_f16_i", "output wire [15:0] trace_f16_o", "output wire [15:0] final_f16_o"):
    assert token in header, token
text = record.read_text()
assert all(text.count("| R%02d |" % i) == 1 for i in range(1, 11))
assert "**Assessment verdict: ACCEPT for the bounded adoption-boundary document.**" in text
assert "PENDING_NORMAL_HOST_REVIEWER" in text
assert text.count("```") % 2 == 0
print("DOCUMENT_DELIVERY_CHECK_COMPLETE")
print("branch=" + branch)
print("requirement_rows=10 missing_subject_anchors=0")
print("prior_review=done old_public_signature=matched")
print("scope=document/source inspection only; no arithmetic, oracle, evaluator or RTL invocation")
PY
```

## Delivery-check result

**FAILED: one invocation, exit code 1; no rerun.** No stdout was produced.
The reported error was:

```text
Traceback (most recent call last):
  File "<stdin>", line 31, in <module>
AssertionError
```

The failed assertion is the raw Markdown-fence substring count. The command
itself embeds that same three-backtick string literal inside the recorded code
block, so it counts an inline Python string as a third fence. The actual command
block has its opening and closing fence lines. This is a delivery-checker defect,
not a missing document, source anchor, prior review, signature or requirement
row: execution reached the final assertion after those assertions succeeded.
The original command and failure are retained unchanged; no success stdout is
inferred or manufactured.

Primary failure taxonomy: **document-validation**. Root-cause hypothesis:
counting raw fence substrings instead of Markdown fence lines falsely rejects
a valid embedded command. Prospective regression: a fenced command containing
an inline three-backtick string must remain structurally valid, while an
unclosed fence must fail; use fence-line-aware parsing in a separately scoped
future check. That regression was not executed. The one-validation constraint
is preserved.

The direct requirement-level assessment remains **ACCEPT**; it does not depend
on the defective formatting assertion or claim a successful automated check.
Normal independent Host Reviewer closure remains pending. No numerical
evaluator was invoked, so this checker failure establishes no RTL correctness
or numerical outcome.

Only this review record and the explicitly designated external CHECKPOINT may
be written by this mission. No existing subject/specification, production source,
contract/evaluator/reference, attempt, Wiki or Skill is modified. No commit,
push, simulator, compilation, arithmetic prototype, oracle regeneration, model
trajectory, output capture/admission, L9 replay or failed-state forwarding is
performed. There is no new durable procedure or adopted project fact warranting
an out-of-scope Wiki/Skill edit.
