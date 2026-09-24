"""Frozen full-vector terminal correction through an isolated final suffix only."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_coordinate363_suffix_intervention_v1 as closed


bridge, base, parent, hotspot = closed.bridge, closed.base, closed.parent, closed.hotspot
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_vector_suffix_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES = tuple(range(896))
EXPECTED_TESTS = 14
CLOSED_CHAIN = (
    closed, "f400ca52ef46",
    "10bb5b7cdf9a3302a679f41e669918b8257d020b50230b09ab1380ddfebff1a0",
    "e98acb61768d2b1a2d64f1177881ce07b900640ba56460a2a8f3d89e172b0b23",
    "5ad54e5be716cdf7fc4afba72cfa95f340ca70d208519ee92f0c728f7355191a",
)
PREREGISTRATION = {
    "question": "Is the aggregate terminal boundary directionally sufficient for "
                "both closed divergent outer pairs, rather than coordinate 363 alone?",
    "selection": "All 896 coordinates in input order, including selected coordinates "
                 "and zeros, with no ranking, mask, coordinate re-selection or fitting.",
    "operand": "Freeze delta[i] = original_FP16_stage18[i] - "
               "original_binary64_terminal[i] once, independently of actual controls. "
               "Working_actual_stage18[i] = native_RNE_FP16(actual_stage18[i] - delta[i]). "
               "Preserve the exact target and conversion remainder for every coordinate.",
    "prediction": "For each of nine retained controls, both outer pairs and both "
                  "fixed original-input reference branches, the final margin delta "
                  "has the strict nonzero sign of minus the retained binary64 "
                  "terminal-remainder bridge contribution. Bind each contrast to "
                  "its unchanged common-selection obstruction, not a new component census.",
    "supported": "Authenticated, finite, independently oracle-verified execution "
                 "with all 36 directional contrasts supports aggregate terminal-boundary "
                 "sufficiency for this bounded directional question only.",
    "rejected": "Any zero/cancelled or opposite directional response after successful "
                "authentication and oracle verification selects coordinate-specific/"
                "cancellation follow-up, without choosing or running another coordinate.",
    "UNKNOWN": "Authentication, source/operand/state/KV/lineage/reference drift, "
               "nonfinite/saturated values, dispatch or execution/test integrity failure.",
    "successors": {
        "supported": "Planner may investigate aggregate terminal-boundary sufficiency "
                     "without treating this reference-guided intervention as a native repair.",
        "rejected": "Planner may investigate coordinate-specific versus aggregate cancellation.",
        "UNKNOWN": "Resolve the specific integrity defect before scientific interpretation.",
    },
    "boundary": "CPU-only non-admission counterfactual: final RMSNorm and two tied-head "
                "rows only, with one baseline closure per control. No prefix, admission, "
                "reference-producer, decoder or closed-intervention replay. Original "
                "references, source/token, Q24 state, native-S16 RTZ, INT4 G128 asymmetric "
                "GEMM ordering with no qzero plus-one, FP16 scales/operators/KV, controls, "
                "lineage, exact thresholds and historical FAIL/UNKNOWN facts remain fixed. "
                "Q24 residual state is wider than FP16. No strict-FP16-state W4A16, "
                "new-token, full-model, admission or reference-independent repair claim.",
}
FLAGS = {
    **closed.FLAGS,
    "coordinate_reselection": False,
    "closed_coordinate363_intervention_replay": False,
    "middle_pair_intervention_replay": False,
    "reference_independent_native_repair_claim": False,
}


def authenticate_chain():
    pins = closed.authenticate_chain()
    module, mission, source_hash, test_hash, review_hash = CLOSED_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0001.json"
    for path, digest in ((module.SOURCE, source_hash), (module.TEST, test_hash),
                         (review_path, review_hash)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed scalar review lineage changed")
    review = json.loads(base.read_bound(pins[-1]))
    same((review["kind"], review["mission_id"], review["producer_role"],
          review["round"], review["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
         "closed scalar independent review gate failed")
    pins.append(parent.record(latest_path))
    return pins


@contextmanager
def no_closed_intervention(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed scalar intervention/check/selection replay forbidden")

    with ExitStack() as stack:
        for name in ("check", "run_tests", "prepare", "preregister"):
            stack.enter_context(patch.object(closed, name, refuse))
        yield


def preregister(evidence):
    same((IDS, PAIRS, BRANCHES, COORDINATES),
         ((319, 34319), ((319, 34319), (34319, 319)),
          ("fp16", "binary64"), tuple(range(896))),
         "frozen full-vector/pair selection changed")
    reference, rows = hotspot.bound_vectors(evidence)
    census = hotspot.census.census(evidence["report"])
    dominance = hotspot.dominance.audit_census(census, evidence["report"])
    same((census["common_component"], census["stop_nested_bridge_expansion"]),
         ("UNKNOWN", True), "historical common obstruction changed")
    delta = tuple(a-b for a, b in zip(reference["stage18"],
                                     evidence["hidden"]["binary64"], strict=True))
    require(len(delta) == 896 and any(delta), "invalid full terminal vector")
    predictions, obstructions = {}, []
    for control in evidence["report"]["controls"]:
        for left, right in PAIRS:
            pair = next(p for p in control["pairs"]
                        if (p["left_id"], p["right_id"]) == (left, right))
            binary = pair["branches"]["binary64"]
            terminal = binary["gate_values"]["component_totals"][hotspot.TERMINAL]
            predicted = -Fraction(terminal["signed"])
            require(predicted != 0, "zero retained terminal bridge prediction")
            anchor = Fraction(binary["unchanged_unselected_bridge"]
                              ["reference_norm"]["inverse_norm_anchor"])
            full_bridge = -sum((v*w*anchor*(a-b) for v, w, a, b in
                                zip(delta, evidence["weights"], rows[left],
                                    rows[right], strict=True)), Fraction())
            predictions[control["control"], left, right] = {
                "predicted_delta": str(predicted),
                "full_vector_linear_bridge_delta": str(full_bridge),
                "retained_terminal_component": terminal,
            }
            for branch in BRANCHES:
                obstructions.append({
                    "control": control["control"], "left_id": left, "right_id": right,
                    "branch": branch,
                    "retained_gate_values": pair["branches"][branch]["gate_values"],
                })
    same(len(predictions), 18, "outer pair/control census changed")
    return delta, predictions, {
        "selection": evidence["report"]["selection"],
        "outer_pair_dominance": [p for p in dominance["pairs"]
                                 if (p["left_id"], p["right_id"]) in PAIRS],
        "rows": obstructions,
    }


def prepare(words, delta):
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid full-vector actual FP16 operand")
    require(isinstance(delta, tuple) and len(delta) == 896
            and all(isinstance(v, Fraction) for v in delta),
            "full original-input remainder vector required")
    working = words.copy()
    targets, rounding = [], []
    for i in COORDINATES:
        target = Fraction(float(words.view("<f2")[i])) - delta[i]
        require(target.denominator & (target.denominator-1) == 0, "nondyadic vector target")
        bits, saturated = parent.head.fixed_to_f16(
            target.numerator, target.denominator.bit_length()-1)
        require(not saturated, "full-vector intervention operand saturated")
        working[i] = bits
        targets.append(str(target))
        rounding.append(str(Fraction(float(working.view("<f2")[i])) - target))
    working.flags.writeable = False
    return working, {
        "coordinate_count": len(COORDINATES), "coordinate_order": "all_input_order",
        "exact_targets": targets, "rounding_remainders": rounding,
        "working_fp16_words": working.tolist(),
        "changed_coordinates": np.flatnonzero(words != working).tolist(),
        "frozen_vector_identity": closed.digest(delta),
    }


def classify(rows):
    same([(r["control"], r["left_id"], r["right_id"], r["branch"]) for r in rows],
         [(c, l, r, b) for c in parent.CONTROLS for l, r in PAIRS for b in BRANCHES],
         "incomplete, duplicate or reordered directional contrast census")
    require(all(Fraction(r["predicted_delta"]) != 0 for r in rows),
            "zero bridge prediction prevents classification")
    return "supported" if all(Fraction(r["predicted_delta"])*Fraction(r["observed_delta"]) > 0
                              for r in rows) else "rejected"


def run_tests(evidence, report, outputs, identity, delta, retained):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_vector_tests", TEST)
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
               "terminal_vector_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
             "selected_row_head_invocations": 0}
    with bridge.read_only(audit):
        evidence = bridge.measure()
    identity = closed.digest(evidence)
    with hotspot.read_only(audit), no_closed_intervention(audit):
        delta, predictions, retained = preregister(evidence)
        prepared = {c: prepare(evidence["archives"][c]["stage18"], delta)
                    for c in parent.CONTROLS}
    frozen_identity = closed.digest((delta, predictions, retained, prepared))
    jobs = [words for c in parent.CONTROLS
            for words in (evidence["archives"][c]["stage18"], prepared[c][0])]
    rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs = [], {}
    with closed.suffix_only(audit, jobs, evidence["weight_array"], rows), no_closed_intervention(audit):
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
            outputs[control] = {
                "working_stage18": words, "rmsnorm": normalized, "logits": changed,
                "scalars": scalars, "operand": operand,
            }
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
        "frozen_original_terminal_vector": [str(v) for v in delta],
        "frozen_original_terminal_vector_identity": closed.digest(delta),
        "operands": {c: v["operand"] for c, v in outputs.items()},
        "retained_outer_pair_common_selection_obstructions": retained,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "common_selection_note": "Margins respond against fixed references; old component "
                                 "accounts and common UNKNOWN are not counterfactually relabelled.",
        "closed_coordinate363_result": {
            "classification": "supported", "directional_contrasts": "36/36",
            "mission": CLOSED_CHAIN[1], "review_sha256": CLOSED_CHAIN[4], "replayed": False,
        },
    }
    output_identity = closed.digest((report, outputs))
    with hotspot.read_only(audit), no_closed_intervention(audit):
        tests = run_tests(evidence, report, outputs, identity, delta, retained)
        closed.protect(evidence, identity)
        same(closed.digest((delta, predictions, retained, prepared)), frozen_identity,
             "frozen vector/selection/working operand drift")
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
