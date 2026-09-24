"""Stdout-only exact accounting of the reviewed nearest-passable final suffix."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_l23_p0_nearest_passable_final_margin_bridge_v1"
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
WIDTH = 896
ROWS = (13, 319, 34319)
PAIRS = ((319, 34319), (34319, 319), (34319, 13))
ENVELOPE = {
    "path": str(ROOT / "build/q24_s16_l23_p0_s18_nearest_passable_envelope_v1_attempt001/result.json"),
    "bytes": 30734523,
    "sha256": "30a041d42ba461fb1368ce6042cc83218e5f7efe8b1094ff67eb84d67816c4d1",
}
BOUNDARY_PARENT = {
    "path": str(ROOT / "build/q24_s16_l23_p0_s18_binary64_reference_boundary_v1_attempt002/result.json"),
    "bytes": 19109472,
    "sha256": "366d9595ced4c7d9038bb00ce32854f472aec68b9d12947f5d38bb227380afc6",
}
COMBINED = {
    "path": str(ROOT / "build/q24_s16_l23_p0_s18_immediate_operand_combined_sufficiency_v1_attempt001/result.json"),
    "bytes": 15877095,
    "sha256": "30746da7c96c1647352c0f550e8a023451a3808e9bd6ae11e44cf30d59877740",
}
REVIEWS = (
    ("8e448d376f47", 1, "4f4c5262a7fecf29f037836a32e4563faf7e974f3de468d6089ea83145cadbf2"),
    ("209cf7a65bdc", 2, "ef2fd442c019d2062e34c7da27c7cd64f1fb3e5dfdf70ebecc36fea9e1aae8c8"),
)
TERMS = ("direct_hidden", "scale", "interaction", "boundary_remainder_delta")
CLAIM_BOUNDARY = (
    "Read-only exact-rational decomposition of retained combined-operand versus "
    "nearest-passable intervened final outputs, not a new intervention. Accounting "
    "anchors are 2^24 / retained root_q24, not newly generated binary64 inverse "
    "norms or recovered unrounded operands. Boundary remainders include all "
    "differences from this explicitly declared rational accounting basis; they "
    "are not rounding-only or causal attributions. Independent original-input "
    "FP16 and binary64 references, exact thresholds, historical FAIL/UNKNOWN, "
    "source/operand/state/KV/lineage gates and closed middle/outer outcomes remain "
    "unchanged. Q24 residual state is wider than FP16; native S16 RTZ, G128 "
    "asymmetric packed INT4 native GEMM nibble ordering, no qzero plus-one, FP16 "
    "scales/operator boundaries/KV remain unchanged. No model operator, prefix, "
    "S18/reference producer, admission, candidate/R17/1780/framework maintenance, "
    "coordinate retry, ten-anchor capture, hardware/GPU/RTL/FPGA or ACE2 change. "
    "No new-token, strict-FP16-state W4A16 or full-model admission. Final rescue "
    "is NOT_DEFINED. Normal independent Host Reviewer closure REQUIRED."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def record(path):
    path = Path(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def bound_bytes(pin):
    path = Path(pin["path"])
    path = path if path.is_absolute() else ROOT / path
    data = path.read_bytes()
    same(hashlib.sha256(data).hexdigest(), pin["sha256"], "bound content hash changed: " + str(path))
    if "bytes" in pin:
        same(len(data), pin["bytes"], "bound byte count changed: " + str(path))
    return data


def authenticate_pins(value, files):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = Path(value["path"])
            path = path if path.is_absolute() else ROOT / path
            key = str(path)
            if key not in files:
                files[key] = record(path)
            same(files[key]["sha256"], value["sha256"], "source/evidence pin changed: " + key)
            if "bytes" in value:
                same(files[key]["bytes"], value["bytes"], "source/evidence size changed: " + key)
        for child in value.values():
            authenticate_pins(child, files)
    elif isinstance(value, list):
        for child in value:
            authenticate_pins(child, files)


def review_bindings(files):
    reviews = []
    for task, number, digest in REVIEWS:
        directory = HANDOFFS / task
        pin = {"path": str(directory / f"round-{number:04d}.json"),
               "bytes": 690, "sha256": digest}
        review = json.loads(bound_bytes(pin))
        index_path = directory / "latest.json"
        index = json.loads(index_path.read_bytes())
        same(index["kind"], "handoff_ref", "review index is not sealed")
        same(index["handoff"]["path"], pin["path"], "review index changed")
        same(index["mission"]["path"], str(directory / "mission.json"), "review mission path changed")
        same((review["kind"], review["mission_id"], review["producer_role"],
              review["round"], review["review"]["status"]),
             ("round_reviewed_handoff", task, "reviewer", number, "done"),
             "independent parent review is not done")
        authenticate_pins(pin, files)
        files[str(index_path)] = record(index_path)
        reviews.append(pin)
    return reviews


def archive(pin):
    with np.load(io.BytesIO(bound_bytes(pin)), allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def words(array, length=WIDTH):
    require(isinstance(array, np.ndarray) and array.shape == (length,)
            and array.dtype.str == "<u2"
            and not np.any((array & 0x7c00) == 0x7c00), "invalid retained FP16 words")
    return tuple(Fraction(float(value)) for value in array.view("<f2"))


def preserve(original, retained):
    require(set(original).issubset(retained), "retained field missing")
    for name, value in original.items():
        other = retained[name]
        require(value.shape == other.shape and value.dtype == other.dtype
                and value.tobytes() == other.tobytes(), "protected field changed: " + name)


def load_weights(assets):
    same(assets["tensors"]["lm_head.weight"], assets["tensors"]["model.embed_tokens.weight"],
         "selected head is no longer tied")
    with safe_open(assets["checkpoint"]["path"], framework="numpy") as model:
        norm = model.get_tensor("model.norm.weight")
        head = model.get_tensor("lm_head.weight")
    for name, array, shape in (
        ("model.norm.weight", norm, (WIDTH,)),
        ("lm_head.weight", head, (151936, WIDTH)),
    ):
        pin = assets["tensors"][name]
        require(array.shape == shape and array.dtype.str == "<f2", "operand shape/dtype changed")
        same(pin["shape"], list(shape), "operand shape binding changed")
        same(pin["dtype"], "float16", "operand dtype binding changed")
        same(hashlib.sha256(array.tobytes()).hexdigest(), pin["sha256"], "operand hash changed")
    return words(norm.view("<u2")), {row: words(head[row].view("<u2")) for row in ROWS}


def authenticate():
    files = {}
    reviews = review_bindings(files)
    envelope, boundary, combined = [
        json.loads(bound_bytes(pin)) for pin in (ENVELOPE, BOUNDARY_PARENT, COMBINED)
    ]
    for pin, result in zip((ENVELOPE, BOUNDARY_PARENT, COMBINED),
                           (envelope, boundary, combined), strict=True):
        authenticate_pins(pin, files)
        authenticate_pins(result, files)
    same((envelope["task_id"], envelope["revision"], envelope["execution_status"]),
         ("8e448d376f47", 1, "COMPLETE"), "envelope identity changed")
    same((boundary["task_id"], boundary["revision"], boundary["status"]),
         ("209cf7a65bdc", 2, "COMPLETE"), "boundary parent identity changed")
    same(envelope["authentication"]["diagnostic_result"], BOUNDARY_PARENT, "boundary parent splice")
    same(boundary["authentication"]["combined_result"], COMBINED, "combined baseline splice")
    same(envelope["authentication"]["diagnostic_review"]["pin"], reviews[1], "review splice")
    same(envelope["thresholds"], combined["thresholds"], "threshold splice")
    same(boundary["thresholds"], combined["thresholds"], "boundary threshold splice")
    same(envelope["preflight"], combined["preflight"], "source/reference lineage splice")
    controls = [row["control"] for row in envelope["controls"]]
    require(len(controls) == len(set(controls)) == 9, "nine-control census changed")
    for result in (boundary, combined):
        same([row["control"] for row in result["controls"]], controls, "parent control order changed")
    for result in (envelope, boundary, combined):
        for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
            same(result["flags"][flag], False, "non-admission flag changed")
        same(result["final_rescued"], "NOT_DEFINED", "final rescue predicate invented")
    summary = combined["preflight"]
    refs = summary["final_reference"]["reference"]
    for branch in ("fp16", "binary64"):
        same(refs["input_" + branch],
             summary["L23_original_reference"]["reference"][branch], "original reference reanchored")
    require(refs["reference_only"] and refs["position"] == 0 and refs["history"] == [9707],
            "independent original-input reference scope changed")
    same(refs["reference_policy"], "legacy-binary64-AWQ-fully-independent-propagation",
         "reference policy changed")
    reference_logits = {}
    for branch, dtype in (("fp16", "<u2"), ("binary64", "<f8")):
        array = np.load(io.BytesIO(bound_bytes(refs["logits_" + branch])), allow_pickle=False)
        require(array.shape == (151936,) and array.dtype.str == dtype, "reference logits changed")
        if branch == "fp16":
            values = words(array, 151936)
            reference_logits[branch] = {row: values[row] for row in ROWS}
        else:
            require(np.all(np.isfinite(array)), "nonfinite original binary64 reference")
            reference_logits[branch] = {row: Fraction(float(array[row])) for row in ROWS}
    weights, head = load_weights(summary["assets"])
    return envelope, boundary, combined, weights, head, reference_logits, files, reviews


def retained_anchor(details):
    root = details["root_q24"]
    require(type(root) is int and root > 0
            and type(details["mean_q48"]) is int and details["mean_q48"] >= 0,
            "invalid retained integer RMSNorm details")
    return Fraction(1 << 24, root)


def coordinate_terms(h0, h1, y0, y1, weight, head_weight, s0, s1):
    dh, ds = h1 - h0, s1 - s0
    direct = head_weight * weight * dh * s0
    scale = head_weight * weight * h0 * ds
    interaction = head_weight * weight * dh * ds
    boundary0 = head_weight * (y0 - weight * h0 * s0)
    boundary1 = head_weight * (y1 - weight * h1 * s1)
    terms = (direct, scale, interaction, boundary1 - boundary0)
    observed = head_weight * (y1 - y0)
    same(sum(terms, Fraction()), observed, "coordinate identity failed")
    return terms, observed


def row_account(weights, y0, y1, logit0, logit1):
    dot0 = sum((w * y for w, y in zip(weights, y0, strict=True)), Fraction())
    dot1 = sum((w * y for w, y in zip(weights, y1, strict=True)), Fraction())
    return {
        "baseline_exact_dot": str(dot0), "intervened_exact_dot": str(dot1),
        "exact_dot_delta": str(dot1 - dot0),
        "baseline_retained_logit": str(logit0), "intervened_retained_logit": str(logit1),
        "baseline_head_boundary": str(logit0 - dot0),
        "intervened_head_boundary": str(logit1 - dot1),
        "head_boundary_delta": str((logit1 - dot1) - (logit0 - dot0)),
        "retained_logit_delta": str(logit1 - logit0),
        "closure": (dot1 - dot0) + ((logit1 - dot1) - (logit0 - dot0)) == logit1 - logit0,
    }


def pair_account(pair, h0, h1, y0, y1, weights, head, s0, s1, changed, rows):
    left, right = pair
    coordinates = []
    totals = [Fraction() for _ in TERMS]
    changed_direct, unchanged_observed, changed_remainder = Fraction(), Fraction(), Fraction()
    dot_delta = Fraction()
    for i in range(WIDTH):
        terms, observed = coordinate_terms(
            h0[i], h1[i], y0[i], y1[i], weights[i], head[left][i] - head[right][i], s0, s1)
        totals = [a + b for a, b in zip(totals, terms, strict=True)]
        dot_delta += observed
        if i in changed:
            changed_direct += terms[0]
            changed_remainder += sum(terms[1:], Fraction())
        else:
            same(terms[0], Fraction(), "unchanged coordinate has nonzero hidden delta")
            unchanged_observed += observed
        coordinates.append({
            "index": i, "changed": i in changed, "hidden_delta": str(h1[i] - h0[i]),
            **{name: str(value) for name, value in zip(TERMS, terms, strict=True)},
            "retained_row_difference_contribution": str(observed), "closure": True,
        })
    row_delta = Fraction(rows[left]["exact_dot_delta"]) - Fraction(rows[right]["exact_dot_delta"])
    same(dot_delta, row_delta, "selected tied-head row contribution closure failed")
    head_delta = Fraction(rows[left]["head_boundary_delta"]) - Fraction(rows[right]["head_boundary_delta"])
    margin_delta = Fraction(rows[left]["retained_logit_delta"]) - Fraction(rows[right]["retained_logit_delta"])
    decomposition = changed_direct + unchanged_observed + changed_remainder + head_delta
    same(decomposition, margin_delta, "final margin decomposition failed")
    return {
        "pair": list(pair), "coordinates": coordinates,
        "term_totals": {name: str(value) for name, value in zip(TERMS, totals, strict=True)},
        "changed_coordinate_direct_hidden": str(changed_direct),
        "unchanged_coordinate_direct_hidden": "0",
        "unchanged_coordinate_contribution": str(unchanged_observed),
        "changed_coordinate_scale_interaction_boundary_remainder": str(changed_remainder),
        "selected_tied_head_row_difference_delta": str(row_delta),
        "head_boundary_delta": str(head_delta), "retained_margin_delta": str(margin_delta),
        "decomposition_sum": str(decomposition), "closure_residual": "0",
        "selected_row_closure": True, "exact_rational_closure": True,
    }


def check_margin_bindings(retained, pairs, rows, references):
    same([(entry["reference_branch"], entry["pair"]) for entry in retained],
         [(branch, list(pair)) for branch in ("fp16", "binary64") for pair in PAIRS],
         "ordered pair/reference census changed")
    for entry in retained:
        left, right = entry["pair"]
        branch = entry["reference_branch"]
        baseline = Fraction(rows[left]["baseline_retained_logit"]) - Fraction(rows[right]["baseline_retained_logit"])
        intervened = Fraction(rows[left]["intervened_retained_logit"]) - Fraction(rows[right]["intervened_retained_logit"])
        reference = references[branch][left] - references[branch][right]
        for key, value in (
            ("retained_actual_margin", baseline), ("substituted_actual_margin", intervened),
            ("retained_reference_margin", reference), ("paired_intervention_delta", intervened - baseline),
            ("retained_margin_change", baseline - reference),
            ("substituted_margin_change", intervened - reference),
        ):
            same(Fraction(entry[key]), value, "retained reference/margin closure changed: " + key)
        pair_report = pairs[PAIRS.index((left, right))]
        same(Fraction(pair_report["retained_margin_delta"]), intervened - baseline,
             "pair margin report splice")


def report(evidence):
    envelope, boundary, combined, weights, head, references, files, reviews = evidence
    controls = []
    for row, boundary_row, baseline_row in zip(
            envelope["controls"], boundary["controls"], combined["controls"], strict=True):
        baseline = archive(baseline_row["raw_payload"])
        parent_arrays = archive(boundary_row["raw_payload"])
        intervened = archive(row["raw_payload"])
        same(row["intervention_input"], boundary_row["raw_payload"], "intervention input splice")
        same(row["baseline_final_margins_source"], baseline_row["raw_payload"], "baseline source splice")
        same(row["parent"], baseline_row["parent"], "source/state/KV lineage splice")
        same(row["source_operand_state_KV_lineage"], "AUTHENTICATED", "lineage gate changed")
        same(baseline_row["source_operand_state_KV_lineage_checks"], "PASS", "baseline lineage gate changed")
        require(row["preservation_checks"] and all(row["preservation_checks"].values()),
                "retained preservation gate changed")
        same(set(baseline), set(parent_arrays), "boundary retained field census changed")
        preserve(baseline, parent_arrays)
        preserve(parent_arrays, intervened)
        same(set(intervened) - set(baseline),
             {"intervened_stage18", "intervened_final_rmsnorm", "intervened_final_logits"},
             "intervened field census changed")
        h0, h1 = words(baseline["stage18"]), words(intervened["intervened_stage18"])
        y0, y1 = words(baseline["final_rmsnorm"]), words(intervened["intervened_final_rmsnorm"])
        logits0 = words(baseline["final_logits"], 151936)
        logits1 = words(intervened["intervened_final_logits"], 151936)
        changed = {i for i in range(WIDTH)
                   if baseline["stage18"][i] != intervened["intervened_stage18"][i]}
        same([item["index"] for item in row["coordinate_reports"]], list(range(WIDTH)),
             "retained coordinate order changed")
        same(changed, {item["index"] for item in row["coordinate_reports"] if item["changed"]},
             "retained changed coordinate census changed")
        same((len(changed), row["changed_coordinates"], row["retained_S18_failures"],
              row["passing_coordinates_preserved"]), (110, 110, 110, 786),
             "retained scalar envelope census changed")
        for i, item in enumerate(row["coordinate_reports"]):
            same(item["retained_word"], f"{int(baseline['stage18'][i]):04x}", "baseline word splice")
            same(item["selected_word"], f"{int(intervened['intervened_stage18'][i]):04x}",
                 "intervened word splice")
            same(Fraction(item["signed_delta"]), h1[i] - h0[i], "retained hidden delta splice")
            require(item["intervened_gate"]["accepted"] and item["passing_word_preserved"],
                    "closed intervention scalar gate changed")
            same(Fraction(item["intervened_gate"]["excess_budget"]), Fraction(1, 8), "budget changed")
        s0 = retained_anchor(baseline_row["rmsnorm_integer_details"])
        s1 = retained_anchor(row["rmsnorm_integer_details"])
        rows = {token: row_account(head[token], y0, y1, logits0[token], logits1[token])
                for token in ROWS}
        pairs = [pair_account(pair, h0, h1, y0, y1, weights, head, s0, s1, changed, rows)
                 for pair in PAIRS]
        check_margin_bindings(row["paired_final_margins"], pairs, rows, references)
        controls.append({
            "control": row["control"], "changed_indices": sorted(changed),
            "changed_coordinates": len(changed), "unchanged_coordinates": WIDTH - len(changed),
            "anchors": {
                "basis": "exact rational 2^24 / retained root_q24; no sqrt or fresh anchor capture",
                "baseline_retained_details": baseline_row["rmsnorm_integer_details"],
                "intervened_retained_details": row["rmsnorm_integer_details"],
                "baseline": str(s0), "intervened": str(s1),
            },
            "selected_rows": {str(token): value for token, value in rows.items()},
            "pairs": pairs, "retained_reference_margin_rows": row["paired_final_margins"],
            "historical_parent": row["parent"], "combined_S18_status": baseline_row["status"],
            "historical_S18_status": row["historical_S18_status"],
            "source_operand_state_KV_lineage": "AUTHENTICATED",
            "retained_fields_byte_preserved": True,
        })
    return {
        "diagnostic_id": NAME, "revision": 1, "task_id": "f0e950f143c6", "status": "COMPLETE",
        "delivery_status": "AWAITING_INDEPENDENT_REVIEW", "independent_review_required": True,
        "final_rescued": "NOT_DEFINED", "flags": envelope["flags"],
        "thresholds": envelope["thresholds"], "controls": controls,
        "parent_result_pins": [ENVELOPE, BOUNDARY_PARENT, COMBINED],
        "parent_review_pins": reviews, "authenticated_current_files": list(files.values()),
        "current_source_and_test": [record(SOURCE), record(TEST)],
        "original_reference_bindings": combined["preflight"]["final_reference"],
        "closed_middle_outer_intervention_outcomes": "UNCHANGED; no dispatch or reinterpretation",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def new_audit():
    return {"evidence_writes": 0, "forbidden_dispatches": 0, "model_operator_calls": 0}


@contextmanager
def read_only(audit):
    active = [True]
    own_names = {__name__, "ace3.model.candidates." + NAME}

    def audit_event(event, args):
        if not active[0]:
            return
        if event == "open":
            _, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+"))
            writing = writing or (isinstance(flags, int) and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing:
                audit["evidence_writes"] += 1
                raise RuntimeError("diagnostic forbids file writes")
        if event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link", "os.symlink",
                     "os.chmod", "os.chown", "os.truncate", "os.utime"}:
            audit["evidence_writes"] += 1
            raise RuntimeError("diagnostic forbids filesystem mutation")
        if (event in {"subprocess.Popen", "os.system", "os.fork", "os.forkpty",
                      "os.posix_spawn", "os.exec", "os.spawn", "socket.connect", "socket.bind"}
                or (event == "import" and args[0].startswith("ace3."))):
            audit["forbidden_dispatches"] += 1
            raise RuntimeError("diagnostic forbids external/model dispatch")

    def profile(frame, event, arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            if module.startswith("ace3.") and module not in own_names:
                audit["model_operator_calls"] += 1
                raise RuntimeError("diagnostic forbids model operator calls")

    previous = sys.getprofile()
    sys.addaudithook(audit_event)
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        active[0] = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    audit = new_audit()
    try:
        with read_only(audit):
            result = report(authenticate())
            same(audit, new_audit(), "nonzero write/dispatch audit")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result = {"diagnostic_id": NAME, "revision": 1, "status": "UNKNOWN",
                  "error": f"{type(error).__name__}: {error}",
                  "candidate_admitted": False, "policy_adopted": False,
                  "successor_published": False, "independent_review_required": True,
                  "claim_boundary": CLAIM_BOUNDARY}
    result["audit"] = audit
    result["audit_scope"] = "in-process Python filesystem/external audit and model call guard"
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
