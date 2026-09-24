"""Frozen non-363 terminal correction through the isolated final suffix only."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_vector_suffix_contrast_v1 as vector


closed, bridge, base, parent, hotspot = (
    vector.closed, vector.bridge, vector.base, vector.parent, vector.hotspot)
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_non363_complement_suffix_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
EXCLUDED_COORDINATE = 363
COORDINATES = tuple(i for i in range(896) if i != 363)
EXPECTED_TESTS = 16
VECTOR_CHAIN = (
    vector, "6bcf67a50804",
    "a245b7c63250b8b681fd46bb9ae07a9847549cc9fe76b9cc28fb4c1854505abf",
    "d3aba4a0577dc9070f38ed7242406cd6168bbeaddd018754f0775cd0fee4fa5c",
    "a05de88c96c5d640234fc5ff5c7f8b35a22ea0c2a7db82b1242ea2a501daee3a",
)
PREREGISTRATION = {
    "question": "Is the terminal complement directionally sufficient beyond the "
                "closed coordinate-363 hotspot and closed all-896 intervention?",
    "selection": "Exactly the 895 coordinates other than 363, in input order, "
                 "including zeros and retained selected coordinates. No ranking, "
                 "coordinate re-selection, fitting or alternative mask.",
    "operand": "Freeze delta[i] = original_FP16_stage18[i] - "
               "original_binary64_terminal[i]. For i != 363 use native_RNE_FP16("
               "actual_stage18[i] - delta[i]); copy actual FP16 word 363 bit-for-bit. "
               "Disclose every applied exact target and conversion remainder.",
    "prediction": "For all nine controls, both outer pairs and both unchanged "
                  "original-input reference branches, the actual final margin delta "
                  "has the strict nonzero sign opposite the retained binary64 "
                  "non-363 terminal complement contribution. That contribution is "
                  "the retained unselected terminal component minus its coordinate-363 "
                  "term, not the separate all-895 working-operand linear bridge.",
    "supported": "Authentication, integrity and independent suffix-oracle agreement "
                 "plus all 36 strict directional contrasts support distributed "
                 "terminal-boundary complement sufficiency for this bounded question.",
    "rejected": "Authenticated, oracle-verified zero, cancelled or opposite movement "
                "in any contrast selects hotspot-dominated or cancellation follow-up.",
    "UNKNOWN": "Authentication, source/operand/state/KV/lineage/reference drift, "
               "nonfinite/saturated values, invalid prediction census, dispatch, "
               "compilation, test or execution-integrity failure.",
    "successors": {
        "supported": "Investigate distributed terminal-boundary complement sufficiency; "
                     "do not infer a reference-independent native repair.",
        "rejected": "Investigate hotspot domination or cancellation, without choosing "
                    "or executing another coordinate here.",
        "UNKNOWN": "Resolve the specific integrity defect before interpreting direction.",
    },
    "boundary": vector.PREREGISTRATION["boundary"] +
                " Both closed interventions stay fixed and are never replayed. "
                "Coordinate 363 of the working stage18 operand is unchanged.",
}
FLAGS = {
    **vector.FLAGS,
    "closed_full_vector_intervention_replay": False,
    "coordinate363_operand_mutated": False,
}


def authenticate_chain():
    pins = vector.authenticate_chain()
    module, mission, source_hash, test_hash, review_hash = VECTOR_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0001.json"
    for path, digest in ((module.SOURCE, source_hash), (module.TEST, test_hash),
                         (review_path, review_hash)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed full-vector review lineage changed")
    review = json.loads(base.read_bound(pins[-1]))
    same((review["kind"], review["mission_id"], review["producer_role"],
          review["round"], review["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
         "closed full-vector independent review gate failed")
    pins.append(parent.record(latest_path))
    return pins


@contextmanager
def no_closed_interventions(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed full-vector intervention/check/selection replay forbidden")

    with vector.no_closed_intervention(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare", "preregister"):
            stack.enter_context(patch.object(vector, name, refuse))
        yield


def selection_gate():
    same((IDS, PAIRS, BRANCHES, EXCLUDED_COORDINATE, COORDINATES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          363, tuple(i for i in range(896) if i != 363)),
         "frozen non-363 complement/pair selection changed")


def preregister(evidence):
    selection_gate()
    reference, rows = hotspot.bound_vectors(evidence)
    census = hotspot.census.census(evidence["report"])
    same((census["common_component"], census["stop_nested_bridge_expansion"]),
         ("UNKNOWN", True), "historical common obstruction changed")
    same([c["control"] for c in evidence["report"]["controls"]],
         list(parent.CONTROLS), "retained control order changed")
    delta = tuple(a-b for a, b in zip(reference["stage18"],
                                     evidence["hidden"]["binary64"], strict=True))
    require(len(delta) == 896 and any(delta[i] for i in COORDINATES),
            "invalid original terminal complement vector")
    predictions, obstructions = {}, []
    for control in evidence["report"]["controls"]:
        for left, right in PAIRS:
            matches = [p for p in control["pairs"]
                       if (p["left_id"], p["right_id"]) == (left, right)]
            same(len(matches), 1, "retained outer pair missing or duplicated")
            pair = matches[0]
            binary = pair["branches"]["binary64"]
            gate = binary["gate_values"]
            excluded = gate["excluded_selected_coordinates"]
            same(excluded, [r["coordinate"] for r in
                            binary["unchanged_unselected_bridge"]["selected_coordinates"]],
                 "retained selected-coordinate binding changed")
            require(excluded == sorted(set(excluded))
                    and all(type(i) is int and 0 <= i < 896 for i in excluded)
                    and 62 in excluded and 8 <= len(excluded) <= 25
                    and 363 not in excluded, "invalid retained coordinate complement")
            anchor = Fraction(binary["unchanged_unselected_bridge"]
                              ["reference_norm"]["inverse_norm_anchor"])
            require(anchor > 0, "invalid retained inverse norm anchor")
            terms = tuple(v*w*anchor*(a-b) for v, w, a, b in
                          zip(delta, evidence["weights"], rows[left], rows[right], strict=True))
            retained_terms = [v for i, v in enumerate(terms) if i not in excluded]
            terminal = gate["component_totals"][hotspot.TERMINAL]
            same(bridge.mass(retained_terms), terminal,
                 "retained terminal signed/absolute account changed")
            complement = sum((terms[i] for i in COORDINATES if i not in excluded), Fraction())
            require(complement == Fraction(terminal["signed"])-terms[363],
                    "retained non-363 complement closure failed")
            require(complement != 0, "zero retained complement prediction")
            predictions[control["control"], left, right] = {
                "predicted_delta": str(-complement),
                "retained_non363_complement_contribution": str(complement),
                "retained_coordinate363_contribution": str(terms[363]),
                "retained_terminal_component": terminal,
                "all895_linear_bridge_delta": str(-sum((terms[i] for i in COORDINATES),
                                                       Fraction())),
            }
            for branch in BRANCHES:
                obstructions.append({
                    "control": control["control"], "left_id": left, "right_id": right,
                    "branch": branch,
                    "retained_gate_values": pair["branches"][branch]["gate_values"],
                })
    same(len(predictions), 18, "outer pair/control census changed")
    return delta, predictions, {"selection": evidence["report"]["selection"], "rows": obstructions}


def prepare(words, delta):
    selection_gate()
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid actual FP16 complement operand")
    require(isinstance(delta, tuple) and len(delta) == 896
            and all(isinstance(v, Fraction) for v in delta),
            "full frozen original-input terminal vector required")
    working = words.copy()
    targets, rounding = [], []
    for i in COORDINATES:
        target = Fraction(float(words.view("<f2")[i])) - delta[i]
        require(target.denominator & (target.denominator-1) == 0, "nondyadic complement target")
        bits, saturated = parent.head.fixed_to_f16(
            target.numerator, target.denominator.bit_length()-1)
        require(not saturated, "complement intervention operand saturated")
        working[i] = bits
        targets.append(str(target))
        rounding.append(str(Fraction(float(working.view("<f2")[i]))-target))
    same(int(working[363]), int(words[363]), "protected stage18 coordinate 363 changed")
    working.flags.writeable = False
    return working, {
        "coordinate_count": 895, "coordinates": list(COORDINATES),
        "coordinate_order": "input_order_excluding_363",
        "exact_targets": targets, "rounding_remainders": rounding,
        "working_fp16_words": working.tolist(),
        "changed_coordinates": np.flatnonzero(words != working).tolist(),
        "coordinate363_original_word": int(words[363]),
        "coordinate363_working_word": int(working[363]),
        "frozen_vector_identity": closed.digest(delta),
    }


classify = vector.classify


def run_tests(evidence, report, outputs, identity, delta, retained):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_non363_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.OUTPUTS = evidence, report, outputs
    module.IDENTITY, module.DELTA, module.RETAINED = identity, delta, retained
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "compiled": [parent.record(SOURCE), parent.record(TEST)]}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter gate failed")
    pins = authenticate_chain()
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "terminal_non363_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
             "selected_row_head_invocations": 0}
    with bridge.read_only(audit), no_closed_interventions(audit):
        evidence = bridge.measure()
    identity = closed.digest(evidence)
    with hotspot.read_only(audit), no_closed_interventions(audit):
        delta, predictions, retained = preregister(evidence)
        prepared = {c: prepare(evidence["archives"][c]["stage18"], delta)
                    for c in parent.CONTROLS}
    frozen_identity = closed.digest((delta, predictions, retained, prepared))
    jobs = [words for c in parent.CONTROLS
            for words in (evidence["archives"][c]["stage18"], prepared[c][0])]
    rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs = [], {}
    with closed.suffix_only(audit, jobs, evidence["weight_array"], rows), no_closed_interventions(audit):
        for control in parent.CONTROLS:
            normalized, _ = parent.rmsnorm(evidence["archives"][control]["stage18"],
                                           evidence["weight_array"])
            baseline = parent.logits(normalized, rows)
            closed.same_array(normalized, evidence["arrays"][control]["rmsnorm"],
                              "baseline final RMSNorm closure failed")
            closed.same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)],
                              "baseline selected-row head closure failed")
            words, operand = prepared[control]
            normalized, scalars = parent.rmsnorm(words, evidence["weight_array"])
            changed = parent.logits(normalized, rows)
            require(np.isfinite(normalized.view("<f2")).all()
                    and np.isfinite(changed.view("<f2")).all(), "nonfinite final suffix")
            outputs[control] = {"working_stage18": words, "rmsnorm": normalized,
                                "logits": changed, "scalars": scalars, "operand": operand}
            for left, right in PAIRS:
                li, ri = IDS.index(left), IDS.index(right)
                old = Fraction(float(baseline.view("<f2")[li]))-Fraction(float(baseline.view("<f2")[ri]))
                new = Fraction(float(changed.view("<f2")[li]))-Fraction(float(changed.view("<f2")[ri]))
                for branch in BRANCHES:
                    ref = evidence["references"]["logits_"+branch]
                    ref = ref.view("<f2") if branch == "fp16" else ref
                    reference_margin = Fraction(float(ref[left]))-Fraction(float(ref[right]))
                    contrasts.append({
                        "control": control, "left_id": left, "right_id": right, "branch": branch,
                        **predictions[control, left, right],
                        "observed_delta": str(new-old), "retained_actual_margin": str(old),
                        "intervened_actual_margin": str(new),
                        "fixed_reference_margin": str(reference_margin),
                        "retained_margin_change": str(old-reference_margin),
                        "intervened_margin_change": str(new-reference_margin),
                    })
    closed.protect(evidence, identity)
    report = {
        "preregistration": PREREGISTRATION, "contrasts": contrasts,
        "classification": classify(contrasts),
        "frozen_original_terminal_vector": list(map(str, delta)),
        "frozen_original_terminal_vector_identity": closed.digest(delta),
        "operands": {c: v["operand"] for c, v in outputs.items()},
        "retained_outer_pair_common_selection_obstructions": retained,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "common_selection_note": "Directional response does not relabel retained component "
                                 "accounts, historical failures or common UNKNOWN.",
        "closed_coordinate363_result": {
            "classification": "supported", "directional_contrasts": "36/36",
            "mission": vector.CLOSED_CHAIN[1], "review_sha256": vector.CLOSED_CHAIN[4],
            "replayed": False,
        },
        "closed_full_vector_result": {
            "classification": "supported", "directional_contrasts": "36/36",
            "mission": VECTOR_CHAIN[1], "review_sha256": VECTOR_CHAIN[4], "replayed": False,
        },
    }
    output_identity = closed.digest((report, outputs))
    with hotspot.read_only(audit), no_closed_interventions(audit):
        tests = run_tests(evidence, report, outputs, identity, delta, retained)
        closed.protect(evidence, identity)
        same(closed.digest((delta, predictions, retained, prepared)), frozen_identity,
             "frozen complement/selection/working operand drift")
        same(closed.digest((report, outputs)), output_identity, "suffix result drift")
        for pin in (*pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
    same(audit, {"forbidden_calls": 0, "final_rmsnorm_invocations": 18,
                 "selected_row_head_invocations": 18}, "suffix dispatch census changed")
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
