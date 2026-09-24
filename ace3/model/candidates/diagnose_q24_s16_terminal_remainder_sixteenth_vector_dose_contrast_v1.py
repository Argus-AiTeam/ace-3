"""Exact sixteenth-terminal-vector polarities through a non-admission CPU suffix."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_eighth_vector_dose_contrast_v1 as eighth


quarter, half, reverse, vector, complement, closed, bridge, base, parent, hotspot = (
    eighth.quarter, eighth.half, eighth.reverse, eighth.vector, eighth.complement,
    eighth.closed, eighth.bridge, eighth.base, eighth.parent, eighth.hotspot)
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_sixteenth_vector_dose_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES, POLARITIES, DOSE = tuple(range(896)), ("forward", "reverse"), Fraction(1, 16)
EXPECTED_TESTS = 14
EIGHTH_CHAIN = (
    eighth, "a97c298b38bc",
    "a32843bd2b86ca099374ac44267dc982410b8b83bb85446a6b2b6c21eb4bd71a",
    "6216e5307881d6d8e16dd4559c5df17939b6dceb20b1591f4fee82dfccc01443",
    "ef3c75820ec6788a91eee07188f516cb05d59c1ccaef2fbf567d912fd2faaf9e",
)
EIGHTH_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "074a11ef-3aa4-4f6e-9a9e-4de7849d1520/events.jsonl",
    "sha256": "393c8843edeeae52165250e93f3f4f29fa9504740293fb51bd934fa69a03ede8",
}
EIGHTH_CALL = "call_j8D066JhmL2JJfXm5eCOSnM9"
RECEIPTS = (*eighth.RECEIPTS, EIGHTH_RECEIPT)
DOSE_KEYS = tuple(f"same_polarity_{dose}_dose_delta"
                  for dose in ("eighth", "quarter", "half", "full"))
PREREGISTRATION = {
    "selection": "All 896 coordinates including zeros, in input order, both outer pairs "
                 "and polarities; no ranking, selected-union reconstruction or dose search.",
    "operand": "Freeze delta = original_FP16_stage18 - original_binary64_terminal. "
               "Working[i] = native_RNE_FP16(actual_FP16_stage18[i] +/- delta[i]/16), "
               "minus forward, plus reverse. Exact sixteenthing precedes one conversion; "
               "never average rounded eighth/quarter/half/full operands.",
    "prediction": "All 72 control/polarity/ordered-pair/original-reference contrasts move "
                  "in the corresponding retained terminal prediction's strict nonzero "
                  "direction, with 0 < abs(sixteenth movement) < abs(same-polarity eighth "
                  "movement), and below same-polarity quarter/half/full when comparable.",
    "comparison": "Authenticate reviewed source/test/review pins and closed full/reverse/"
                  "half/quarter/eighth receipts with identical source/token/operand/Q24/"
                  "state/KV/lineage/reference/weight identities. The compact eighth receipt "
                  "retains (polarity, movement, quarter, half, full) signatures for pair "
                  "(319,34319). Bind each control by its exact same-polarity quarter/half/"
                  "full tuple: require unique keys and use every signature. Derive the "
                  "opposite pair by exact margin antisymmetry; fixed-reference subtraction "
                  "cancels for either original branch. Missing, ambiguous or incompatible "
                  "comparators are UNKNOWN, never omitted passing rows. These identities "
                  "do not reconstruct detailed coordinate gate censuses or a selected union.",
    "supported": "Conversion-resilient sub-eighth local terminal-boundary dose sensitivity; "
                 "a smaller fixed-dose successor is justified.",
    "rejected": "Below-eighth FP16 conversion-threshold/asymmetry/cancellation follow-up; "
                "any zero, opposite, equal-bound or larger response rejects.",
    "UNKNOWN": "Authentication or integrity defect; no scientific interpretation.",
    "boundary": eighth.PREREGISTRATION["boundary"].replace(
        "reverse/half/quarter/middle-pair", "reverse/half/quarter/eighth/middle-pair"),
}
FLAGS = {**eighth.FLAGS, "closed_eighth_vector_intervention_replay": False,
         "eighth_dose_recomputation": False}


def authenticate_chain():
    pins = eighth.authenticate_chain()
    module, mission, source, test, review = EIGHTH_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0001.json"
    for path, digest in ((module.SOURCE, source), (module.TEST, test), (review_path, review)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    record = json.loads(base.read_bound(pins[-1]))
    same((record["kind"], record["mission_id"], record["producer_role"],
          record["round"], record["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
         "closed eighth independent review gate failed")
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed eighth review lineage changed")
    pins.append(parent.record(latest_path))
    return pins


def retained_doses():
    previous = eighth.retained_doses()
    events = [json.loads(line) for line in base.read_bound(EIGHTH_RECEIPT).splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == EIGHTH_CALL]
    same(len(matches), 1, "eighth dose receipt missing or duplicated")
    require(matches[0]["success"] is True, "eighth dose receipt execution failed")
    text = matches[0]["result"]["content"]
    marker = '{\n  "changed_coordinate_counts":'
    same(text.count(marker), 1, "eighth dose summary missing or duplicated")
    summary, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    return (*previous, summary)


@contextmanager
def no_closed_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed eighth-dose intervention replay forbidden")

    with eighth.no_closed_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare"):
            stack.enter_context(patch.object(eighth, name, refuse))
        yield


def selection_gate():
    same((IDS, PAIRS, BRANCHES, COORDINATES, POLARITIES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          tuple(range(896)), ("forward", "reverse")), "sixteenth-dose selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(1, 16), "sixteenth dose changed")


def prepare(words, delta, polarity):
    selection_gate()
    require(polarity in POLARITIES, "invalid sixteenth-dose polarity")
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
        require(not saturated, "sixteenth-dose operand saturated")
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


def bind_doses(evidence, forward, backward, retained_half, retained_quarter, retained_eighth):
    selection_gate()
    delta, predictions = eighth.bind_doses(
        evidence, forward, backward, retained_half, retained_quarter)
    same([retained_eighth[k] for k in (
        "status", "native_exit", "stdout_json_documents", "contrast_count",
        "directional_agreements", "strict_dose_agreements", "retained_common_component")],
        ["supported", 0, 1, 72, 72, 72, "UNKNOWN"], "closed eighth outcome/census changed")
    same([retained_eighth["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [14, 0, 0, 0], "closed eighth test gate failed")
    same(retained_eighth["tests"]["compiled"],
         [parent.record(eighth.SOURCE), parent.record(eighth.TEST)],
         "closed eighth source/test binding changed")
    same(retained_eighth["dispatch_and_write_audit"],
         {"forbidden_calls": 0, "final_rmsnorm_invocations": 27,
          "selected_row_head_invocations": 27}, "closed eighth dispatch changed")
    same(retained_eighth["protected_input_identity"], closed.digest(evidence),
         "eighth comparison source/token/operand/state/KV/lineage/reference drift")
    same(retained_eighth["changed_coordinate_counts"],
         {p: {c: 858 for c in parent.CONTROLS} for p in POLARITIES},
         "closed eighth operand census changed")
    signatures = retained_eighth["distinct_forward_pair_movements"]
    require(isinstance(signatures, list) and len(signatures) == 4,
            "closed eighth signature census changed")
    same(signatures, sorted(signatures), "closed eighth signature order changed")
    lookup = {}
    for signature in signatures:
        require(isinstance(signature, list) and len(signature) == 5
                and signature[0] in POLARITIES, "invalid eighth comparator signature")
        polarity, movement, quarter_move, half_move, full = signature
        movement, quarter_move, half_move, full = map(
            Fraction, (movement, quarter_move, half_move, full))
        key = (polarity, quarter_move, half_move, full)
        require(key not in lookup, "ambiguous eighth comparator signature")
        require(movement * quarter_move > 0 and movement * half_move > 0
                and movement * full > 0
                and 0 < abs(movement) < abs(quarter_move) < abs(half_move) < abs(full),
                "closed eighth strict dose outcome changed")
        lookup[key] = movement
    comparisons, used = {}, set()
    for control in parent.CONTROLS:
        for polarity in POLARITIES:
            forward_prediction = predictions[control, 319, 34319, polarity]
            key = (polarity, *(Fraction(forward_prediction[k]) for k in DOSE_KEYS[1:]))
            require(key in lookup, "missing same-control eighth comparator signature")
            used.add(key)
            for left, right in PAIRS:
                sign = 1 if left == 319 else -1
                prediction = predictions[control, left, right, polarity]
                for field in ("predicted_delta", *DOSE_KEYS[1:]):
                    require(Fraction(prediction[field]) == sign * Fraction(forward_prediction[field]),
                            "eighth ordered-pair comparator antisymmetry changed")
                comparisons[control, left, right, polarity] = {
                    **prediction,
                    "predicted_delta": str(Fraction(prediction["predicted_delta"]) / 2),
                    DOSE_KEYS[0]: str(sign * lookup[key]),
                    "eighth_receipt_pair": [319, 34319],
                    "eighth_comparator_signature": [polarity, *map(str, key[1:])],
                    "eighth_comparator_derivation": "unique_closed_dose_signature"
                        + ("" if sign == 1 else "_and_exact_margin_antisymmetry"),
                }
    require(used == set(lookup), "unused eighth comparator signature")
    return delta, comparisons


def classify(rows):
    selection_gate()
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"])
          for r in rows],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS for b in BRANCHES], "incomplete or reordered sixteenth-dose census")
    supported = True
    for row in rows:
        prediction, observed = Fraction(row["predicted_delta"]), Fraction(row["observed_delta"])
        movements = [Fraction(row[k]) for k in DOSE_KEYS]
        require(row["dose_comparable"] is True
                and all(prediction * v > 0 for v in movements)
                and all(abs(a) < abs(b) for a, b in zip(movements, movements[1:])),
                "missing or incompatible closed dose comparator")
        supported &= prediction * observed > 0 and all(
            0 < abs(observed) < abs(v) for v in movements)
    return "supported" if supported else "rejected"


def run_tests(evidence, report, outputs, identity, delta, summaries):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_sixteenth_dose_tests", TEST)
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
               "terminal_sixteenth_dose_tests": parent.record(TEST)}
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
            ("full_forward", "full_reverse", "half", "quarter", "eighth"), summaries, strict=True)),
        "retained_dose_receipts": list(RECEIPTS),
        "retained_dose_calls": [reverse.PREDICTION_CALL, half.REVERSE_CALL,
                               quarter.HALF_CALL, eighth.QUARTER_CALL, EIGHTH_CALL],
        "retained_common_selection_obstructions": obstructions,
        "retained_common_selection_rule": bridge.SELECTION_RULE,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "obstruction_scope": "Retained signed terminal component and fixed margin change; "
                             "missing detailed coordinate gate censuses are not reconstructed.",
        "closed_results": [
            {"mission": chain[1], "review_sha256": chain[4], "classification": "supported",
             "directional_contrasts": "72/72" if index >= 4 else "36/36", "replayed": False}
            for index, chain in enumerate((
                vector.CLOSED_CHAIN, complement.VECTOR_CHAIN, reverse.COMPLEMENT_CHAIN,
                half.REVERSE_CHAIN, quarter.HALF_CHAIN, eighth.QUARTER_CHAIN, EIGHTH_CHAIN))],
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
