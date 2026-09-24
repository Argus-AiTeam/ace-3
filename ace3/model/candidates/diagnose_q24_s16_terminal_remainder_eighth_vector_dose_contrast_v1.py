"""Exact eighth-terminal-vector polarities through a non-admission CPU suffix."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_quarter_vector_dose_contrast_v1 as quarter


half, reverse, vector, complement, closed, bridge, base, parent, hotspot = (
    quarter.half, quarter.reverse, quarter.vector, quarter.complement, quarter.closed,
    quarter.bridge, quarter.base, quarter.parent, quarter.hotspot)
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_eighth_vector_dose_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES, POLARITIES, DOSE = tuple(range(896)), ("forward", "reverse"), Fraction(1, 8)
EXPECTED_TESTS = 14
QUARTER_CHAIN = (
    quarter, "14ecb1be9d4b",
    "79d7bcd53c1c2e30f12b66d0ff728bd01839f4b172e110a29508485421295527",
    "e745ccf4925b562571a41f39dcf89ba02e64737db9d785b7ef307e9e73b36f4d",
    "c45dcab071e047ccebbb4c9379eb9f4fc6c9d44e1aa7a3503fb249c31da51f3c",
)
QUARTER_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "508b7af6-18cc-49a5-8d29-13ccd21f0caa/events.jsonl",
    "sha256": "776961dc5015d535311176adce3d03192ff081ec34c0abe529222ac89e96f3b3",
}
QUARTER_CALL = "call_ahJxJq84JYf1PhHeFPmTlZCp"
RECEIPTS = (reverse.PREDICTION_RECEIPT, half.REVERSE_RECEIPT,
            quarter.HALF_RECEIPT, QUARTER_RECEIPT)
PREREGISTRATION = {
    "selection": "All 896 coordinates including zeros, in input order, both outer pairs "
                 "and polarities; no coordinate ranking, union reconstruction or dose search.",
    "operand": "Freeze delta = original_FP16_stage18 - original_binary64_terminal. "
               "Working[i] = native_RNE_FP16(actual_FP16_stage18[i] +/- delta[i]/8), "
               "minus forward, plus reverse. Exact eighthing precedes one conversion; "
               "never average rounded quarter/half/full-dose operands.",
    "prediction": "All 72 control/polarity/ordered-pair/original-reference contrasts move "
                  "in the corresponding retained terminal prediction's strict nonzero "
                  "direction, with 0 < abs(eighth movement) < abs(same-polarity quarter "
                  "movement), and below same-polarity half/full movement when comparable.",
    "comparison": "Authenticate closed full/reverse/half/quarter receipts and reviewed "
                  "source/test/review pins with identical source/token/operand/Q24/state/"
                  "KV/lineage/reference/weight identities. Missing or incompatible "
                  "comparators are UNKNOWN, never omitted passing rows. The quarter "
                  "receipt retains both ordered pairs in binary64; bind all 36 rows "
                  "directly. Only the half receipt needs opposite-pair antisymmetry. "
                  "Fixed-reference subtraction cancels in margin movement for either "
                  "original branch. Algebraic comparator derivations are not "
                  "reconstructed detailed coordinate gate censuses.",
    "supported": "Conversion-resilient sub-quarter local terminal-boundary dose sensitivity; "
                 "a smaller fixed-dose successor is justified.",
    "rejected": "Below-quarter FP16 conversion-threshold/asymmetry/cancellation follow-up; "
                "any zero, opposite, equal-quarter/half/full or larger response rejects.",
    "UNKNOWN": "Authentication or integrity defect; no scientific interpretation.",
    "boundary": "CPU-only reference-guided final-suffix non-admission diagnostic. Original "
                "prefix/admission/reference producers and closed coordinate-363/full/non-363/"
                "reverse/half/quarter/middle-pair interventions are not replayed. Historical "
                "FAIL/UNKNOWN, exact thresholds and independent original global references "
                "remain fixed. Native S16 RTZ, wider-than-FP16 Q24 state, native G128 "
                "asymmetric packed INT4 weights with native GEMM nibble order and no qzero "
                "plus-one, FP16 scales/operator/KV boundaries unchanged. No hardware/GPU/"
                "RTL/ACE2, precision/scale expansion, strict-FP16-state W4A16, new-token/"
                "full-model admission or reference-independent repair. No continuous-dose "
                "monotonicity or causal mechanism is established.",
}
FLAGS = {**quarter.FLAGS, "closed_quarter_vector_intervention_replay": False,
         "quarter_dose_recomputation": False}


def authenticate_chain():
    pins = quarter.authenticate_chain()
    module, mission, source, test, review = QUARTER_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0001.json"
    for path, digest in ((module.SOURCE, source), (module.TEST, test), (review_path, review)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    record = json.loads(base.read_bound(pins[-1]))
    same((record["kind"], record["mission_id"], record["producer_role"],
          record["round"], record["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
         "closed quarter independent review gate failed")
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed quarter review lineage changed")
    pins.append(parent.record(latest_path))
    return pins


def retained_doses():
    previous = quarter.retained_doses()
    events = [json.loads(line) for line in base.read_bound(QUARTER_RECEIPT).splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == QUARTER_CALL]
    same(len(matches), 1, "quarter dose receipt missing or duplicated")
    require(matches[0]["success"] is True, "quarter dose receipt execution failed")
    lines = [line for line in matches[0]["result"]["content"].splitlines()
             if line.startswith('{"changed_coordinate_counts":')]
    same(len(lines), 1, "quarter dose summary missing or duplicated")
    return (*previous, json.loads(lines[0]))


@contextmanager
def no_closed_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed quarter-dose intervention replay forbidden")

    with quarter.no_closed_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare"):
            stack.enter_context(patch.object(quarter, name, refuse))
        yield


def selection_gate():
    same((IDS, PAIRS, BRANCHES, COORDINATES, POLARITIES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          tuple(range(896)), ("forward", "reverse")), "eighth-dose selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(1, 8), "eighth dose changed")


def prepare(words, delta, polarity):
    selection_gate()
    require(polarity in POLARITIES, "invalid eighth-dose polarity")
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid actual FP16 operand")
    require(isinstance(delta, tuple) and len(delta) == 896
            and all(isinstance(v, Fraction) for v in delta), "invalid frozen terminal vector")
    signed_dose = -DOSE if polarity == "forward" else DOSE
    working, targets, rounding = words.copy(), [], []
    for i in COORDINATES:
        target = Fraction(float(words.view("<f2")[i])) + signed_dose * delta[i]
        require(target.denominator & (target.denominator - 1) == 0, "nondyadic target")
        bits, saturated = parent.head.fixed_to_f16(
            target.numerator, target.denominator.bit_length() - 1)
        require(not saturated, "eighth-dose operand saturated")
        working[i] = bits
        targets.append(str(target))
        rounding.append(str(Fraction(float(working.view("<f2")[i])) - target))
    working.flags.writeable = False
    return working, {
        "coordinates": list(COORDINATES), "coordinate_count": 896,
        "coordinate_order": "all_input_order", "polarity": polarity, "dose": str(DOSE),
        "signed_dose": str(signed_dose), "exact_targets": targets,
        "rounding_remainders": rounding, "working_fp16_words": working.tolist(),
        "changed_coordinates": np.flatnonzero(words != working).tolist(),
        "frozen_vector_identity": closed.digest(delta),
    }


def bind_doses(evidence, forward, backward, retained_half, retained_quarter):
    selection_gate()
    delta, predictions = quarter.bind_doses(evidence, forward, backward, retained_half)
    same((retained_quarter["status"], retained_quarter["native_exit"],
          retained_quarter["stdout_json_documents"], retained_quarter["contrast_count"],
          retained_quarter["directional_agreements"], retained_quarter["strict_dose_agreements"],
          retained_quarter["retained_common_component"]),
         ("supported", 0, 1, 72, 72, 72, "UNKNOWN"), "closed quarter outcome/census changed")
    same([retained_quarter["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [14, 0, 0, 0], "closed quarter test gate failed")
    same(retained_quarter["tests"]["compiled"],
         [parent.record(quarter.SOURCE), parent.record(quarter.TEST)],
         "closed quarter source/test binding changed")
    same(retained_quarter["dispatch_and_write_audit"],
         {"forbidden_calls": 0, "final_rmsnorm_invocations": 27,
          "selected_row_head_invocations": 27}, "closed quarter dispatch changed")
    same(retained_quarter["protected_input_identity"], closed.digest(evidence),
         "quarter comparison source/token/operand/state/KV/lineage/reference drift")
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"])
          for r in retained_quarter["contrasts"]],
         [(c, p, l, r, "binary64") for c in parent.CONTROLS
          for p in POLARITIES for l, r in PAIRS],
         "quarter receipt comparator census changed")
    comparisons = {}
    for row in retained_quarter["contrasts"]:
        control, polarity = row["control"], row["polarity"]
        left, right = row["left_id"], row["right_id"]
        prediction = predictions[control, left, right, polarity]
        actual = evidence["arrays"][control]["logits"].view("<f2")
        ref = evidence["references"]["logits_binary64"]
        retained = (Fraction(float(actual[left])) - Fraction(float(actual[right]))
                    - Fraction(float(ref[left])) + Fraction(float(ref[right])))
        for key in ("predicted_delta", "same_polarity_half_dose_delta",
                    "same_polarity_full_dose_delta"):
            require(Fraction(row[key]) == Fraction(prediction[key]),
                    "quarter prediction or half/full comparator lineage changed")
        require(Fraction(row["retained_margin_change"]) == retained,
                "quarter retained original-reference obstruction changed")
        movement = Fraction(row["observed_delta"])
        require(Fraction(row["intervened_margin_change"]) - retained == movement,
                "quarter closed margin movement closure changed")
        require(Fraction(row["predicted_delta"]) * movement > 0
                and 0 < abs(movement) < abs(Fraction(row["same_polarity_half_dose_delta"]))
                < abs(Fraction(row["same_polarity_full_dose_delta"])),
                "closed quarter strict dose outcome changed")
        comparisons[control, left, right, polarity] = {
            **prediction, "predicted_delta": str(Fraction(prediction["predicted_delta"]) / 2),
            "same_polarity_quarter_dose_delta": str(movement),
            "quarter_receipt_pair": [left, right],
            "quarter_comparator_derivation": "retained_ordered_pair",
        }
    for control in parent.CONTROLS:
        for polarity in POLARITIES:
            require(Fraction(comparisons[control, 319, 34319, polarity]
                             ["same_polarity_quarter_dose_delta"])
                    == -Fraction(comparisons[control, 34319, 319, polarity]
                                 ["same_polarity_quarter_dose_delta"]),
                    "retained quarter ordered-pair antisymmetry changed")
    return delta, comparisons


def classify(rows):
    selection_gate()
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"])
          for r in rows],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS for b in BRANCHES], "incomplete or reordered eighth-dose census")
    supported = True
    for row in rows:
        prediction, observed, quarter_move, half_move, full = (Fraction(row[k]) for k in
            ("predicted_delta", "observed_delta", "same_polarity_quarter_dose_delta",
             "same_polarity_half_dose_delta", "same_polarity_full_dose_delta"))
        require(row["dose_comparable"] is True and prediction * quarter_move > 0
                and prediction * half_move > 0 and prediction * full > 0
                and 0 < abs(quarter_move) < abs(half_move) < abs(full),
                "missing or incompatible closed dose comparator")
        supported &= (prediction * observed > 0 and 0 < abs(observed) < abs(quarter_move)
                      and abs(observed) < abs(half_move) and abs(observed) < abs(full))
    return "supported" if supported else "rejected"


def run_tests(evidence, report, outputs, identity, delta, summaries):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_eighth_dose_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.OUTPUTS = evidence, report, outputs
    module.IDENTITY, module.DELTA, module.SUMMARIES = identity, delta, summaries
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "errors": len(result.errors),
            "failures": len(result.failures), "skipped": len(result.skipped),
            "compiled": [parent.record(SOURCE), parent.record(TEST)]}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter gate failed")
    pins, summaries = authenticate_chain(), retained_doses()
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "terminal_eighth_dose_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
             "selected_row_head_invocations": 0}
    with hotspot.read_only(audit), no_closed_replay(audit):
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        delta, comparisons = bind_doses(evidence, *summaries)
        prepared = {(c, p): prepare(evidence["archives"][c]["stage18"], delta, p)
                    for c in parent.CONTROLS for p in POLARITIES}
    identity = closed.digest(evidence)
    frozen = closed.digest((delta, comparisons, prepared, summaries, PREREGISTRATION, FLAGS))
    jobs = [words for c in parent.CONTROLS
            for words in (evidence["archives"][c]["stage18"],
                          *(prepared[c, p][0] for p in POLARITIES))]
    head_rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs, obstructions = [], {}, []
    with no_closed_replay(audit), closed.suffix_only(audit, jobs, evidence["weight_array"], head_rows):
        for control in parent.CONTROLS:
            normalized, _ = parent.rmsnorm(evidence["archives"][control]["stage18"],
                                           evidence["weight_array"])
            baseline = parent.logits(normalized, head_rows)
            closed.same_array(normalized, evidence["arrays"][control]["rmsnorm"],
                              "baseline final RMSNorm closure failed")
            closed.same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)],
                              "baseline selected-row closure failed")
            for polarity in POLARITIES:
                words, operand = prepared[control, polarity]
                normalized, scalars = parent.rmsnorm(words, evidence["weight_array"])
                changed = parent.logits(normalized, head_rows)
                require(np.isfinite(normalized.view("<f2")).all()
                        and np.isfinite(changed.view("<f2")).all(), "nonfinite suffix output")
                outputs[control, polarity] = {
                    "working_stage18": words, "rmsnorm": normalized, "logits": changed,
                    "scalars": scalars, "operand": operand}
                for left, right in PAIRS:
                    li, ri = IDS.index(left), IDS.index(right)
                    old = Fraction(float(baseline.view("<f2")[li])) - Fraction(float(baseline.view("<f2")[ri]))
                    new = Fraction(float(changed.view("<f2")[li])) - Fraction(float(changed.view("<f2")[ri]))
                    comparison = comparisons[control, left, right, polarity]
                    for branch in BRANCHES:
                        ref = evidence["references"]["logits_" + branch]
                        ref = ref.view("<f2") if branch == "fp16" else ref
                        reference_margin = Fraction(float(ref[left])) - Fraction(float(ref[right]))
                        obstruction = {
                            "control": control, "polarity": polarity,
                            "left_id": left, "right_id": right, "branch": branch,
                            "retained_actual_margin": str(old),
                            "fixed_reference_margin": str(reference_margin),
                            "retained_margin_change": str(old - reference_margin),
                            "retained_common_component": "UNKNOWN",
                            "retained_branch_terminal_component_signed":
                                comparison["retained_terminal_component_signed"]
                                if branch == "binary64" else "0",
                        }
                        obstructions.append(obstruction)
                        contrasts.append({
                            **obstruction, **comparison, "observed_delta": str(new - old),
                            "intervened_actual_margin": str(new),
                            "intervened_margin_change": str(new - reference_margin),
                        })
    closed.protect(evidence, identity)
    report = {
        "preregistration": PREREGISTRATION, "classification": classify(contrasts),
        "contrasts": contrasts,
        "operands": {p: {c: outputs[c, p]["operand"] for c in parent.CONTROLS}
                     for p in POLARITIES},
        "frozen_original_terminal_vector": list(map(str, delta)),
        "frozen_original_terminal_vector_identity": closed.digest(delta),
        "retained_dose_summaries": dict(zip(
            ("full_forward", "full_reverse", "half", "quarter"), summaries, strict=True)),
        "retained_dose_receipts": list(RECEIPTS),
        "retained_dose_calls": [reverse.PREDICTION_CALL, half.REVERSE_CALL,
                               quarter.HALF_CALL, QUARTER_CALL],
        "retained_common_selection_obstructions": obstructions,
        "retained_common_selection_rule": bridge.SELECTION_RULE,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "obstruction_scope": "Retained signed terminal component and fixed margin change; "
                             "missing detailed coordinate gate censuses are not reconstructed.",
        "closed_results": [
            {"mission": chain[1], "review_sha256": chain[4], "classification": "supported",
             "directional_contrasts":
                 "72/72" if chain in (quarter.HALF_CHAIN, QUARTER_CHAIN) else "36/36",
             "replayed": False}
            for chain in (vector.CLOSED_CHAIN, complement.VECTOR_CHAIN,
                          reverse.COMPLEMENT_CHAIN, half.REVERSE_CHAIN,
                          quarter.HALF_CHAIN, QUARTER_CHAIN)],
    }
    output_identity = closed.digest((report, outputs))
    with hotspot.read_only(audit), no_closed_replay(audit):
        tests = run_tests(evidence, report, outputs, identity, delta, summaries)
        closed.protect(evidence, identity)
        same(closed.digest((delta, comparisons, prepared, summaries, PREREGISTRATION, FLAGS)),
             frozen, "frozen dose/operand/contract drift")
        same(closed.digest((report, outputs)), output_identity, "suffix result drift")
        for pin in (*pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"], *RECEIPTS):
            base.read_bound(pin)
    same(audit, {"forbidden_calls": 0, "final_rmsnorm_invocations": 27,
                 "selected_row_head_invocations": 27}, "suffix dispatch census changed")
    return {
        "diagnostic_id": NAME, "version": 1, "status": report["classification"],
        "command": COMMAND, "report": report, "tests": tests,
        "diagnostic_sources": origins, "reviewed_parent_pins": pins,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"], "protected_input_identity": identity,
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "flags": FLAGS, "dispatch_and_write_audit": audit,
        "normal_host_review": "REQUIRED", "claim_boundary": PREREGISTRATION["boundary"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, ArithmeticError, LookupError,
            TypeError, ImportError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN",
                          "integrity_error": f"{type(error).__name__}: {error}",
                          "flags": FLAGS, "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
