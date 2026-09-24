"""Exact three-sixty-fourths terminal-vector bracket; CPU suffix, never admission."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_thirtysecond_vector_dose_contrast_v1 as thirtysecond


sixteenth, eighth, quarter, half, reverse, vector, complement, closed, bridge, base, parent, hotspot = (
    thirtysecond.sixteenth, thirtysecond.eighth, thirtysecond.quarter, thirtysecond.half,
    thirtysecond.reverse, thirtysecond.vector, thirtysecond.complement, thirtysecond.closed,
    thirtysecond.bridge, thirtysecond.base, thirtysecond.parent, thirtysecond.hotspot)
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_three_sixtyfourths_vector_dose_bracket_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES, POLARITIES, DOSE = tuple(range(896)), ("forward", "reverse"), Fraction(3, 64)
EXPECTED_TESTS = 14
THIRTYSECOND_CHAIN = (
    thirtysecond, "b5d8f57125c8",
    "16264e95f1afb2cf7dcc327db293ab0406916b71e73d6c258d6b4878a52cb5ad",
    "dc4b83a350d2e25fd4ac41ca063fe442ad8c06f96d3fdcef2a8946d2aa3833b9",
    "927a04873d38749ce7e929e183df17e1e4c2523c7ec3801e1eba2851d3decead",
)
THIRTYSECOND_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "94dd5886-094c-4234-937d-3864ece95a91/events.jsonl",
    "sha256": "ad375f7730b93efc4a5b91cf88ebfeb03511b72a488dd22bb8ce4b5fdcc4c7d9",
}
THIRTYSECOND_CALL = "call_H1TDNvYkLzJ0aIJ4YO8D9LsE"
RECEIPTS = (*thirtysecond.RECEIPTS, THIRTYSECOND_RECEIPT)
DOSE_KEYS = ("same_polarity_thirtysecond_dose_delta", *thirtysecond.DOSE_KEYS)
CONTRAST_FIELDS = (
    "control", "polarity", "left_id", "right_id", "branch", "predicted_delta",
    "observed_delta", *DOSE_KEYS[1:], "retained_margin_change", "intervened_margin_change")
PREREGISTRATION = {
    "selection": "All 896 coordinates including zeros, in input order, both outer pairs "
                 "and polarities; no ranking, selected-union reconstruction or dose search.",
    "operand": "Freeze delta = original_FP16_stage18 - original_binary64_terminal. "
               "Working[i] = native_RNE_FP16(actual_FP16_stage18[i] +/- 3*delta[i]/64), "
               "minus forward, plus reverse. Exact rational multiplication precedes one "
               "conversion; never substitute or average rounded closed-dose operands.",
    "prediction": "All 72 contrasts retain strict nonzero predicted direction, including "
                  "recovery of the 32 thirty-second reverse-zero contrasts. Require "
                  "abs(three-sixty-fourths) <= abs(same-polarity sixteenth), strictly "
                  "above abs(same-polarity thirty-second) wherever nonzero, and below "
                  "same-polarity eighth/quarter/half/full movement.",
    "comparison": "Authenticate reviewed source/test/review pins and closed full/reverse/"
                  "half/quarter/eighth/sixteenth/thirty-second receipts. Bind identical "
                  "source/token/operand/Q24/state/KV/lineage/reference/weight identities, "
                  "ordered census, predictions, older comparators and original-reference "
                  "margin closure. Fixed-reference subtraction cancels for both branches. "
                  "Preserve closed supported outcomes and the rejected thirty-second "
                  "40/72 outcome with 32 reverse zeros. No closed operand or intervention "
                  "is recomputed. A different changed-word census from each neighboring "
                  "closed dose proves that neither rounded operand was substituted.",
    "supported": "Reverse-obstruction conversion threshold bracket (1/32, 3/64]; "
                 "select a tighter threshold-localization successor, not continuous "
                 "monotonicity or a causal boundary attribution.",
    "rejected": "Select suffix-boundary asymmetry/cancellation localization rather than "
                "smaller fixed-dose continuation; any zero, opposite, excessive upper "
                "movement or non-strict nonzero lower comparison falsifies the prediction.",
    "UNKNOWN": "Authentication or integrity defect; no scientific interpretation.",
    "boundary": thirtysecond.PREREGISTRATION["boundary"]
                + " Closed thirty-second rejection and all earlier outcomes remain fixed.",
}
FLAGS = {**thirtysecond.FLAGS, "closed_thirtysecond_vector_intervention_replay": False,
         "thirtysecond_dose_recomputation": False}


def authenticate_chain():
    pins = thirtysecond.authenticate_chain()
    module, mission, source, test, review = THIRTYSECOND_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0001.json"
    for path, digest in ((module.SOURCE, source), (module.TEST, test), (review_path, review)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    record = json.loads(base.read_bound(pins[-1]))
    same((record["kind"], record["mission_id"], record["producer_role"],
          record["round"], record["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
         "closed thirty-second independent review gate failed")
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed thirty-second review lineage changed")
    pins.append(parent.record(latest_path))
    return pins


def retained_doses():
    previous = thirtysecond.retained_doses()
    events = [json.loads(line) for line in base.read_bound(THIRTYSECOND_RECEIPT).splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == THIRTYSECOND_CALL]
    same(len(matches), 1, "thirty-second dose receipt missing or duplicated")
    require(matches[0]["success"] is True, "thirty-second dose receipt execution failed")
    text = matches[0]["result"]["content"]
    marker = '{"changed_coordinate_counts":'
    same(text.count(marker), 1, "thirty-second dose summary missing or duplicated")
    summary, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    return (*previous, summary)


@contextmanager
def no_closed_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed thirty-second-dose intervention replay forbidden")

    with thirtysecond.no_closed_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare"):
            stack.enter_context(patch.object(thirtysecond, name, refuse))
        yield


def selection_gate():
    same((IDS, PAIRS, BRANCHES, COORDINATES, POLARITIES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          tuple(range(896)), ("forward", "reverse")), "three-sixty-fourths selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(3, 64), "three-sixty-fourths dose changed")


def prepare(words, delta, polarity):
    selection_gate()
    require(polarity in POLARITIES, "invalid three-sixty-fourths polarity")
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid actual FP16 operand")
    require(isinstance(delta, tuple) and len(delta) == 896
            and all(isinstance(v, Fraction) and v.denominator & (v.denominator - 1) == 0
                    for v in delta), "invalid frozen terminal vector")
    signed_dose = -DOSE if polarity == "forward" else DOSE
    working, targets, rounding = words.copy(), [], []
    for i in COORDINATES:
        target = Fraction(float(words.view("<f2")[i])) + signed_dose * delta[i]
        require(target.denominator & (target.denominator - 1) == 0, "nondyadic target")
        bits, saturated = parent.head.fixed_to_f16(
            target.numerator, target.denominator.bit_length() - 1)
        require(not saturated, "three-sixty-fourths operand saturated")
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


def bind_doses(evidence, *summaries):
    selection_gate()
    same(len(summaries), 7, "closed dose census changed")
    delta, predictions = thirtysecond.bind_doses(evidence, *summaries[:-1])
    retained = summaries[-1]
    same([retained[k] for k in (
        "status", "native_exit", "stdout_json_documents", "contrast_count",
        "directional_agreements", "strict_dose_agreements", "retained_common_component")],
        ["rejected", 0, 1, 72, 40, 40, "UNKNOWN"], "closed thirty-second outcome/census changed")
    same([retained["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [14, 0, 0, 0], "closed thirty-second test gate failed")
    same(retained["tests"]["compiled"],
         [parent.record(thirtysecond.SOURCE), parent.record(thirtysecond.TEST)],
         "closed thirty-second source/test binding changed")
    same(retained["dispatch_and_write_audit"],
         {"forbidden_calls": 0, "final_rmsnorm_invocations": 27,
          "selected_row_head_invocations": 27}, "closed thirty-second dispatch changed")
    same(retained["protected_input_identity"], closed.digest(evidence),
         "thirty-second source/token/operand/state/KV/lineage/reference drift")
    same(retained["changed_coordinate_counts"],
         {p: {c: 742 for c in parent.CONTROLS} for p in POLARITIES},
         "closed thirty-second operand census changed")
    same(retained["contrast_fields"], list(CONTRAST_FIELDS), "closed contrast schema changed")
    values = retained["contrast_values"]
    require(isinstance(values, list) and len(values) == 36
            and all(isinstance(v, list) and len(v) == len(CONTRAST_FIELDS) for v in values),
            "closed thirty-second contrast shape changed")
    rows = [dict(zip(CONTRAST_FIELDS, v, strict=True)) for v in values]
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in rows],
         [(c, p, l, r, "binary64") for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS], "closed thirty-second ordered contrast census changed")
    comparisons = {}
    zero_count = 0
    for row in rows:
        control, polarity, left, right = (row[k] for k in CONTRAST_FIELDS[:4])
        prediction = predictions[control, left, right, polarity]
        for field in ("predicted_delta", *DOSE_KEYS[1:]):
            same(row[field], prediction[field], "closed thirty-second prediction/dose drift")
        movement = Fraction(row["observed_delta"])
        expected = (Fraction(-1, 128) if polarity == "forward" else
                    Fraction(1, 128) if control == "mapped_all" else Fraction())
        same(str(movement), str(expected if left == 319 else -expected),
             "closed thirty-second directional/zero outcome changed")
        zero_count += movement == 0
        actual = evidence["arrays"][control]["logits"].view("<f2")
        reference = evidence["references"]["logits_binary64"]
        margin = (Fraction(float(actual[left])) - Fraction(float(actual[right]))
                  - Fraction(float(reference[left])) + Fraction(float(reference[right])))
        same(row["retained_margin_change"], str(margin), "closed original-reference margin drift")
        same(row["intervened_margin_change"], str(margin + movement), "closed margin closure failed")
        comparisons[control, left, right, polarity] = {
            **prediction,
            "predicted_delta": str(Fraction(prediction["predicted_delta"]) * Fraction(3, 2)),
            DOSE_KEYS[0]: str(movement), "thirtysecond_zero_obstruction": movement == 0,
            "thirtysecond_receipt_branch": "binary64",
            "thirtysecond_comparator_derivation": "closed_ordered_pair_margin_movement",
        }
    same(zero_count, 16, "closed binary64 reverse-zero census changed")
    return delta, comparisons


def dose_agreement(row):
    prediction, observed = Fraction(row["predicted_delta"]), Fraction(row["observed_delta"])
    lower, upper = (Fraction(row[k]) for k in DOSE_KEYS[:2])
    return (prediction * observed > 0 and abs(observed) <= abs(upper)
            and (lower == 0 or abs(observed) > abs(lower))
            and all(abs(observed) < abs(Fraction(row[k])) for k in DOSE_KEYS[2:]))


def classify(rows):
    selection_gate()
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in rows],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS for b in BRANCHES], "incomplete or reordered bracket census")
    for row in rows:
        prediction = Fraction(row["predicted_delta"])
        Fraction(row["observed_delta"])
        lower, *older = [Fraction(row[k]) for k in DOSE_KEYS]
        require(row["dose_comparable"] is True and all(prediction * v > 0 for v in older)
                and all(abs(a) < abs(b) for a, b in zip(older, older[1:]))
                and abs(lower) < abs(older[0])
                and (lower == 0 and row["polarity"] == "reverse" or prediction * lower > 0)
                and row["thirtysecond_zero_obstruction"] is (lower == 0),
                "missing or incompatible closed bracket comparator")
    return "supported" if all(dose_agreement(row) for row in rows) else "rejected"


def run_tests(evidence, report, outputs, identity, delta, summaries):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_three_sixtyfourths_tests", TEST)
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
               "terminal_three_sixtyfourths_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0, "selected_row_head_invocations": 0}
    with hotspot.read_only(audit), no_closed_replay(audit):
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        delta, comparisons = bind_doses(evidence, *summaries)
        prepared = {(c, p): prepare(evidence["archives"][c]["stage18"], delta, p)
                    for c in parent.CONTROLS for p in POLARITIES}
        for (control, polarity), (_, operand) in prepared.items():
            count = len(operand["changed_coordinates"])
            require(all(count != receipt["changed_coordinate_counts"][polarity][control]
                        for receipt in summaries[-2:]),
                    "new operand not proven distinct from closed neighboring doses")
    identity = closed.digest(evidence)
    frozen = closed.digest((delta, comparisons, prepared, summaries, PREREGISTRATION, FLAGS))
    jobs = [words for c in parent.CONTROLS
            for words in (evidence["archives"][c]["stage18"], *(prepared[c, p][0] for p in POLARITIES))]
    head_rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs, obstructions = [], {}, []
    with no_closed_replay(audit), closed.suffix_only(audit, jobs, evidence["weight_array"], head_rows):
        for control in parent.CONTROLS:
            normalized, _ = parent.rmsnorm(evidence["archives"][control]["stage18"], evidence["weight_array"])
            baseline = parent.logits(normalized, head_rows)
            closed.same_array(normalized, evidence["arrays"][control]["rmsnorm"], "baseline RMSNorm closure failed")
            closed.same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)], "baseline head closure failed")
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
                                comparison["retained_terminal_component_signed"] if branch == "binary64" else "0",
                        }
                        obstructions.append(obstruction)
                        contrasts.append({
                            **obstruction, **comparison, "observed_delta": str(new - old),
                            "intervened_actual_margin": str(new),
                            "intervened_margin_change": str(new - reference_margin)})
    closed.protect(evidence, identity)
    report = {
        "preregistration": PREREGISTRATION, "classification": classify(contrasts), "contrasts": contrasts,
        "operands": {p: {c: outputs[c, p]["operand"] for c in parent.CONTROLS} for p in POLARITIES},
        "frozen_original_terminal_vector": list(map(str, delta)),
        "frozen_original_terminal_vector_identity": closed.digest(delta),
        "retained_dose_summaries": dict(zip(
            ("full_forward", "full_reverse", "half", "quarter", "eighth", "sixteenth", "thirtysecond"),
            summaries, strict=True)),
        "retained_dose_receipts": list(RECEIPTS),
        "retained_dose_calls": [reverse.PREDICTION_CALL, half.REVERSE_CALL, quarter.HALF_CALL,
                               eighth.QUARTER_CALL, sixteenth.EIGHTH_CALL,
                               thirtysecond.SIXTEENTH_CALL, THIRTYSECOND_CALL],
        "retained_common_selection_obstructions": obstructions,
        "retained_common_selection_rule": bridge.SELECTION_RULE,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "obstruction_scope": "Retained signed terminal component and fixed margin change; "
                             "missing detailed coordinate gate censuses are not reconstructed.",
        "closed_results": [
            {"mission": chain[1], "review_sha256": chain[4],
             "classification": "rejected" if index == 8 else "supported",
             "directional_contrasts": "40/72" if index == 8 else "72/72" if index >= 4 else "36/36",
             "replayed": False}
            for index, chain in enumerate((
                vector.CLOSED_CHAIN, complement.VECTOR_CHAIN, reverse.COMPLEMENT_CHAIN,
                half.REVERSE_CHAIN, quarter.HALF_CHAIN, eighth.QUARTER_CHAIN,
                sixteenth.EIGHTH_CHAIN, thirtysecond.SIXTEENTH_CHAIN, THIRTYSECOND_CHAIN))],
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
    except (ValueError, RuntimeError, OSError, ArithmeticError, LookupError, TypeError, ImportError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN",
                          "integrity_error": f"{type(error).__name__}: {error}",
                          "flags": FLAGS, "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
