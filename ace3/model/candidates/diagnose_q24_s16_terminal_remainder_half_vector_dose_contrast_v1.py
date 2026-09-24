"""Both half-terminal-vector polarities through a CPU-only non-admission suffix."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_reverse_vector_suffix_contrast_v1 as reverse


vector, complement, closed, bridge, base, parent, hotspot = (
    reverse.vector, reverse.complement, reverse.closed, reverse.bridge,
    reverse.base, reverse.parent, reverse.hotspot)
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_half_vector_dose_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES, POLARITIES, DOSE = tuple(range(896)), ("forward", "reverse"), Fraction(1, 2)
EXPECTED_TESTS = 14
REVERSE_CHAIN = (
    reverse, "850f03bc18c4",
    "06b0f63b18a4dfebbea6eb5ecf603c7d7c55c9acaa17e9517152da3626eff6bd",
    "243c41280a056f842ecf1dd3c505e1c7832539e5333621f66b99b2c8eadb27ab",
    "3d5e5b1589f0331bfd0a2993fa1fc7d768b12eb33797bf70d5967bb8701cdd8a",
)
REVERSE_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "d35d5372-7654-4be0-bbd9-a90c20604b42/events.jsonl",
    "sha256": "ea75ae4c5fd778f4a6c2aa6b5e64159a96fbde68af2af075090ec328202cd452",
}
REVERSE_CALL = "call_te6Sf0n0xmL6q0olRivAak0U"
PREREGISTRATION = {
    "selection": "All 896 coordinates in input order, including zeros; no ranking, "
                 "selected-coordinate union, fitting, coordinate search or dose search.",
    "operand": "Freeze delta = original_FP16_stage18 - original_binary64_terminal. "
               "Working[i] = native_RNE_FP16(actual_FP16_stage18[i] +/- delta[i]/2), "
               "minus for forward and plus for reverse. Divide exactly before one FP16 "
               "conversion, not by averaging rounded full-dose operands.",
    "prediction": "For every retained control, ordered outer pair and original reference "
                  "branch, both half-dose margin changes have the corresponding retained "
                  "terminal prediction's strict nonzero sign and absolute movement "
                  "strictly between zero and the same-polarity full-dose movement.",
    "comparison": "Use only authenticated closed execution receipts, never full-dose "
                  "replay. All 72 comparisons must bind the same controls, original inputs, "
                  "references, weights, linear bridge and retained obstruction. A missing "
                  "or mismatched comparator is UNKNOWN, never an omitted passing row. "
                  "Actual-margin movement also equals fixed-reference margin-change movement.",
    "supported": "Local monotone terminal-boundary dose sensitivity, not proportionality.",
    "rejected": "FP16 conversion threshold, asymmetry or cancellation follow-up; any zero, "
                "opposite, equal-full-dose or larger-than-full-dose response rejects.",
    "UNKNOWN": "Authentication or integrity defect; no scientific interpretation.",
    "boundary": reverse.PREREGISTRATION["boundary"] +
                " The reverse full-vector outcome also remains closed. This tests only "
                "the preregistered half dose, not a threshold interval or an algorithm repair.",
}
FLAGS = {**reverse.FLAGS, "closed_reverse_vector_intervention_replay": False,
         "full_dose_recomputation": False, "dose_search": False}


def authenticate_chain():
    pins = reverse.authenticate_chain()
    module, mission, source, test, review = REVERSE_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0002.json"
    for path, digest in ((module.SOURCE, source), (module.TEST, test), (review_path, review)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    record = json.loads(base.read_bound(pins[-1]))
    same((record["kind"], record["mission_id"], record["producer_role"],
          record["round"], record["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 2, "done"),
         "closed reverse independent review gate failed")
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed reverse review lineage changed")
    pins.append(parent.record(latest_path))
    return pins


def retained_doses():
    forward = reverse.retained_predictions()
    events = [json.loads(line) for line in base.read_bound(REVERSE_RECEIPT).splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == REVERSE_CALL]
    same(len(matches), 1, "reverse dose receipt missing or duplicated")
    require(matches[0]["success"] is True, "reverse dose receipt execution failed")
    lines = [line for line in matches[0]["result"]["content"].splitlines()
             if line.startswith('{"changed_coordinate_counts":')]
    same(len(lines), 1, "reverse dose summary missing or duplicated")
    return forward, json.loads(lines[0])


@contextmanager
def no_closed_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed reverse-vector intervention replay forbidden")

    with reverse.no_reselection_or_closed_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare", "preregister"):
            stack.enter_context(patch.object(reverse, name, refuse))
        yield


def selection_gate():
    same((IDS, PAIRS, BRANCHES, COORDINATES, POLARITIES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          tuple(range(896)), ("forward", "reverse")), "half-dose selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(1, 2), "half dose changed")


def prepare(words, delta, polarity):
    selection_gate()
    require(polarity in POLARITIES, "invalid half-dose polarity")
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
        require(not saturated, "half-dose operand saturated")
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


def bind_doses(evidence, forward, backward):
    selection_gate()
    for summary, module in ((forward, vector), (backward, reverse)):
        same(summary["status"], "supported", "closed full-dose outcome changed")
        same([summary["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
             [14, 0, 0, 0], "closed full-dose test gate failed")
        same(summary["tests"]["compiled"], [parent.record(module.SOURCE), parent.record(module.TEST)],
             "closed full-dose source/test binding changed")
        same(summary["dispatch_and_write_audit"],
             {"forbidden_calls": 0, "final_rmsnorm_invocations": 18,
              "selected_row_head_invocations": 18}, "closed full-dose dispatch changed")
    same((forward["single_stdout_json"], forward["directional_rows"]),
         (True, 36), "closed forward stdout/census changed")
    same((backward["native_exit"], backward["stdout_json_documents"],
          backward["contrast_count"], backward["directional_agreements"],
          backward["retained_common_component"]), (0, 1, 36, 36, "UNKNOWN"),
         "closed reverse outcome/census changed")
    same(backward["prediction_receipt"], reverse.PREDICTION_RECEIPT,
         "closed reverse prediction lineage changed")
    same(closed.digest(evidence), backward["protected_input_identity"],
         "full-dose comparison source/token/operand/state/KV/lineage/reference drift")
    same([(r["control"], tuple(r["pair"])) for r in forward["contrasts"]],
         [(c, p) for c in parent.CONTROLS for p in PAIRS], "forward comparator census changed")
    same([(r["control"], r["left_id"], r["right_id"], r["branch"])
          for r in backward["contrasts"]],
         [(c, l, r, "binary64") for c in parent.CONTROLS for l, r in PAIRS],
         "reverse comparator census changed")
    reference, rows = hotspot.bound_vectors(evidence)
    delta = tuple(a - b for a, b in zip(
        reference["stage18"], evidence["hidden"]["binary64"], strict=True))
    require(any(delta), "zero original terminal vector")
    anchor = Fraction(bridge.hidden.norm_summary(evidence["hidden"]["binary64"])
                      ["inverse_norm_anchor"])
    comparisons = {}
    for fwd, rev in zip(forward["contrasts"], backward["contrasts"], strict=True):
        control, (left, right) = fwd["control"], fwd["pair"]
        linear = sum((v * w * anchor * (a - b) for v, w, a, b in
                      zip(delta, evidence["weights"], rows[left], rows[right], strict=True)),
                     Fraction())
        require(linear == Fraction(rev["full_vector_linear_bridge_delta"])
                == -Fraction(fwd["full_vector_linear_bridge_delta"]),
                "original-input/weight/reference linear bridge changed")
        terminal = Fraction(rev["predicted_delta"])
        require(terminal != 0 and terminal == -Fraction(fwd["predicted_delta"]),
                "retained terminal prediction changed")
        actual = evidence["arrays"][control]["logits"].view("<f2")
        ref = evidence["references"]["logits_binary64"]
        old = Fraction(float(actual[left])) - Fraction(float(actual[right]))
        reference_margin = Fraction(float(ref[left])) - Fraction(float(ref[right]))
        require(old - reference_margin == Fraction(rev["retained_margin_change"]),
                "retained full-dose margin obstruction changed")
        require(Fraction(rev["intervened_margin_change"]) - Fraction(rev["retained_margin_change"])
                == Fraction(rev["observed_delta"]), "closed reverse margin closure changed")
        for polarity, row, sign in (("forward", fwd, -1), ("reverse", rev, 1)):
            observed = Fraction(row["observed_delta"])
            require(sign * terminal * observed > 0, "closed full-dose direction changed")
            comparisons[control, left, right, polarity] = {
                "predicted_delta": str(sign * terminal * DOSE),
                "retained_terminal_component_signed": str(terminal),
                "full_vector_linear_bridge_delta": str(sign * linear),
                "same_polarity_full_dose_delta": str(observed),
                "dose_comparable": True,
            }
    return delta, comparisons


def classify(rows):
    selection_gate()
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"])
          for r in rows],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS for b in BRANCHES], "incomplete or reordered half-dose census")
    supported = True
    for row in rows:
        prediction, observed, full = (Fraction(row[k]) for k in
                                     ("predicted_delta", "observed_delta",
                                      "same_polarity_full_dose_delta"))
        require(row["dose_comparable"] is True and prediction * full > 0,
                "missing, zero or incompatible full-dose comparator")
        supported &= prediction * observed > 0 and 0 < abs(observed) < abs(full)
    return "supported" if supported else "rejected"


def run_tests(evidence, report, outputs, identity, delta, summaries):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_half_dose_tests", TEST)
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
    pins = authenticate_chain()
    summaries = retained_doses()
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "terminal_half_dose_tests": parent.record(TEST)}
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
    rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs, obstructions = [], {}, []
    with no_closed_replay(audit), closed.suffix_only(audit, jobs, evidence["weight_array"], rows):
        for control in parent.CONTROLS:
            normalized, _ = parent.rmsnorm(evidence["archives"][control]["stage18"],
                                           evidence["weight_array"])
            baseline = parent.logits(normalized, rows)
            closed.same_array(normalized, evidence["arrays"][control]["rmsnorm"],
                              "baseline final RMSNorm closure failed")
            closed.same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)],
                              "baseline selected-row closure failed")
            for polarity in POLARITIES:
                words, operand = prepared[control, polarity]
                normalized, scalars = parent.rmsnorm(words, evidence["weight_array"])
                changed = parent.logits(normalized, rows)
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
        "retained_full_dose_summaries": dict(zip(POLARITIES, summaries, strict=True)),
        "retained_dose_receipts": [reverse.PREDICTION_RECEIPT, REVERSE_RECEIPT],
        "retained_dose_calls": [reverse.PREDICTION_CALL, REVERSE_CALL],
        "retained_common_selection_obstructions": obstructions,
        "retained_common_selection_rule": bridge.SELECTION_RULE,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "obstruction_scope": "Retained signed terminal component and fixed margin change; "
                             "missing detailed coordinate gate censuses are not reconstructed.",
        "closed_results": [
            {"mission": chain[1], "review_sha256": chain[4], "classification": "supported",
             "directional_contrasts": "36/36", "replayed": False}
            for chain in (vector.CLOSED_CHAIN, complement.VECTOR_CHAIN,
                          reverse.COMPLEMENT_CHAIN, REVERSE_CHAIN)],
    }
    output_identity = closed.digest((report, outputs))
    with hotspot.read_only(audit), no_closed_replay(audit):
        tests = run_tests(evidence, report, outputs, identity, delta, summaries)
        closed.protect(evidence, identity)
        same(closed.digest((delta, comparisons, prepared, summaries, PREREGISTRATION, FLAGS)),
             frozen, "frozen dose/operand/contract drift")
        same(closed.digest((report, outputs)), output_identity, "suffix result drift")
        for pin in (*pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"],
                    reverse.PREDICTION_RECEIPT, REVERSE_RECEIPT):
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
