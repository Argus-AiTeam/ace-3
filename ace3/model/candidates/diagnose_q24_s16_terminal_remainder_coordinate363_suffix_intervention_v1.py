"""One actual-FP16 coordinate-363 suffix correction; references remain immutable."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_terminal_remainder_coordinate_hotspot_audit_v1 as hotspot


bridge, base, parent = hotspot.bridge, hotspot.base, hotspot.parent
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_coordinate363_suffix_intervention_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
COORDINATE, IDS, EXPECTED_TESTS = 363, (319, 34319), 11
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
CHAIN = (
    (hotspot, "26f4868db92e",
     "03168fd515211892cb77962b3f4154721f5f24ae69ecdb0704c2311a09c1a43b",
     "682ecd307d8b3f36de370cca15332fdbb3c2842ed82c7a45ec9ecb8193602988",
     "bcaf9f2d4e9382e155a56ebac88f23bb64aed76a3ff5b0c5d3c962956518c589"),
    (hotspot.dominance, "d423f3395067",
     "276ac257b0ed320354b84480379f5bdaec74797b754ba20017dbef65907f1e9a",
     "3992eb1371c9cd1141b822cc7ac25c8b89e5d124c23d48ca7ae714570988ba47",
     "5d84c60739edb3a8a4b01ddfc0a1a432524fe55c1cb10a521097b1bb279ef6b0"),
    (hotspot.census, "ea58d8308658",
     "291be54c421df28d38d2ecdcf9ee66432376a0e005e0b02e369919ea3aed4c93",
     "92e6bab2302bdd4dc7081d2c98d30887a8119554fd466b0b5cdcb3bcfcdfeaec",
     "ed2d40ab77db1011e581de8d7f8c0146ab57981af9c48c7b0625fd479ab84033"),
)
PREREGISTRATION = {
    "question": "Does correcting only working actual FP16 stage18 coordinate 363 "
                "by the immutable original-FP16 minus original-binary64 discrepancy "
                "move both outer-pair margins opposite the retained contribution?",
    "selection": "Coordinate 363 must be the sole maximum-absolute terminal hotspot "
                 "in every binary64 outer-pair/control row before execution.",
    "operand": "Working final stage18[363] becomes native RNE FP16 of "
               "actual_stage18[363] - (original_FP16_stage18[363] - "
               "original_binary64_terminal[363]); all other coordinates unchanged.",
    "prediction": "For both outer pairs, all nine controls and each original-input "
                  "reference branch, the actual pair-margin delta must have the "
                  "strict nonzero sign of minus the retained binary64 coordinate "
                  "contribution. References never move. Rounding is disclosed.",
    "supported": "Authenticated, finite, oracle-verified final-suffix execution "
                 "and the strict directional contrast in all 36 rows.",
    "rejected": "Authenticated, oracle-verified execution with any zero/opposite contrast.",
    "UNKNOWN": "Authentication, source/operand/state/KV/lineage, dispatch, test, "
               "nonfinite-output or execution-integrity failure.",
    "stop": "One fresh coordinate-363 classification closes this branch regardless "
            "of supported/rejected outcome, pending independent Reviewer verdict. "
            "No retry, alternate coordinate or middle-pair continuation.",
    "boundary": "Final RMSNorm and two tied-head rows only; no prefix, decoder, "
                "S16, reference or admission replay. Q24 state remains wider than "
                "FP16. Native INT4 G128 asymmetric GEMM ordering, no qzero plus-one, "
                "FP16 scales/operators/KV and S16 RTZ are unchanged. Historical "
                "S18 FAIL and common UNKNOWN remain. No final-head admission "
                "threshold exists; no strict-FP16-state W4A16, new-token or full-model claim.",
}
FLAGS = {key: value for key, value in hotspot.FLAGS.items() if key not in {
    "final_rmsnorm_invocations", "lm_head_invocations", "rmsnorm_recomputation",
    "selected_row_head_replay", "rmsnorm_operator_replay", "head_operator_replay",
    "counterfactual_operator_replay", "local_operator_replay", "row_dot_operator_replay",
    "local_exact_row_dot_computations",
}}
FLAGS.update({"historical_failures_preserved": True,
              "original_global_reference_unchanged": True})


def authenticate_chain():
    pins = []
    for module, mission, source_hash, test_hash, review_hash in CHAIN:
        review_path = HANDOFFS / mission / "round-0001.json"
        for path, digest in ((module.SOURCE, source_hash), (module.TEST, test_hash),
                             (review_path, review_hash)):
            pin = {"path": str(path), "sha256": digest}
            base.read_bound(pin)
            pins.append(pin)
        latest_path = review_path.with_name("latest.json")
        latest = json.loads(latest_path.read_bytes())
        same(latest["kind"], "handoff_ref", "parent not independently reviewed")
        same(latest["handoff"]["path"], str(review_path), "review lineage changed")
        review = json.loads(base.read_bound(pins[-1]))
        same((review["kind"], review["mission_id"], review["producer_role"],
              review["round"], review["review"]["status"]),
             ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
             "independent parent review gate failed")
        pins.append(parent.record(latest_path))
    return pins


def digest(value):
    result = hashlib.sha256()

    def consume(item):
        result.update(type(item).__name__.encode() + b":")
        if isinstance(item, np.ndarray):
            result.update(str((item.dtype.str, item.shape)).encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                consume(key)
                consume(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                consume(child)
        else:
            result.update(repr(item).encode())
        result.update(b";")

    consume(value)
    return result.hexdigest()


def protect(evidence, identity):
    same(digest(evidence), identity, "source/operand/state/KV/lineage/reference drift")


def preregister(report):
    same((COORDINATE, IDS), (363, (319, 34319)), "frozen coordinate/row selection changed")
    same([(p["left_id"], p["right_id"]) for p in report["pairs"]],
         list(hotspot.PAIRS), "outer pair selection changed")
    same(report["common_component"], "UNKNOWN", "historical common gate changed")
    same(report["stop_nested_bridge_expansion"], True, "historical stop gate changed")
    predictions = {}
    for pair in report["pairs"]:
        same([(r["control"], r["branch"]) for r in pair["rows"]],
             [(c, b) for c in parent.CONTROLS for b in ("fp16", "binary64")],
             "hotspot census changed")
        for row in pair["rows"]:
            require(COORDINATE not in row["excluded_selected_coordinates"],
                    "hotspot entered retained selected-coordinate union")
            if row["branch"] == "fp16":
                require(row["fp16_terminal_zero_boundary"], "FP16 zero boundary changed")
                continue
            same(row["top_terminal_magnitude_ties"], [COORDINATE],
                 "coordinate 363 is not the unique retained hotspot")
            coordinate = next(v for v in row["coordinates"] if v["coordinate"] == COORDINATE)
            predicted = -Fraction(coordinate["terminal_signed_contribution"])
            require(predicted != 0, "zero prediction prevents classification")
            predictions[(pair["left_id"], pair["right_id"], row["control"])] = predicted
    return predictions


def prepare(actual, reference, binary64):
    delta = Fraction(float(reference["stage18"].view("<f2")[COORDINATE])) - Fraction(
        float(binary64[COORDINATE]))
    target = Fraction(float(actual["stage18"].view("<f2")[COORDINATE])) - delta
    require(target.denominator & (target.denominator - 1) == 0, "nondyadic operand")
    bits, saturated = parent.head.fixed_to_f16(
        target.numerator, target.denominator.bit_length() - 1)
    require(not saturated, "intervention operand saturated")
    words = actual["stage18"].copy()
    words[COORDINATE] = bits
    words.flags.writeable = False
    return words, {"coordinate": COORDINATE, "terminal_hidden_delta": str(delta),
                   "exact_target": str(target), "working_fp16_bits": f"{bits:04x}",
                   "rounding_remainder": str(Fraction(float(words.view("<f2")[COORDINATE]))-target)}


def same_array(actual, expected, message):
    require(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
            and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
            message)


@contextmanager
def suffix_only(audit, jobs, weights, rows):
    raw_norm, raw_logits = parent.rmsnorm, parent.logits
    raw_core, raw_decode = parent.norm.rmsnorm, parent.head.decode_array_q24
    active = {"job": None, "phase": None, "normalized": None}
    remaining = iter(jobs)

    def refuse(message):
        audit["forbidden_calls"] += 1
        raise RuntimeError(message)

    def norm(words, supplied_weights):
        if active["phase"] is not None:
            return refuse("unexpected RMSNorm phase")
        expected = next(remaining, None)
        if expected is None:
            return refuse("final suffix budget exhausted")
        same_array(words, expected, "unregistered final stage18 operand")
        same_array(supplied_weights, weights, "norm tensor drift")
        active.update(job=expected, phase="norm")
        output, scalars = raw_norm(words, supplied_weights)
        active.update(phase="head", normalized=output)
        audit["final_rmsnorm_invocations"] += 1
        return output, scalars

    def core(activations, supplied_weights):
        if active["phase"] != "norm":
            return refuse("out-of-cone RMSNorm core")
        same(activations, active["job"].tolist(), "RMSNorm activation splice")
        same(supplied_weights, weights.view("<u2").tolist(), "RMSNorm weight splice")
        return raw_core(activations, supplied_weights)

    def decode(words):
        if active["phase"] != "head":
            return refuse("out-of-cone head decode")
        expected = active["normalized"] if words.ndim == 1 else rows.view("<u2")
        same_array(words, expected, "head decode operand splice")
        return raw_decode(words)

    def logits(words, supplied_rows):
        if active["phase"] != "head":
            return refuse("out-of-order selected-row head")
        same_array(words, active["normalized"], "head hidden splice")
        same_array(supplied_rows, rows, "head row identity/scale expansion")
        output = raw_logits(words, supplied_rows)
        active.update(job=None, phase=None, normalized=None)
        audit["selected_row_head_invocations"] += 1
        return output

    with hotspot.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests"):
            stack.enter_context(patch.object(hotspot, name,
                                            lambda *a, **k: refuse("predecessor check forbidden")))
        for module, name, function in (
            (parent, "rmsnorm", norm), (parent, "logits", logits),
            (parent.norm, "rmsnorm", core), (parent.head, "decode_array_q24", decode),
        ):
            stack.enter_context(patch.object(module, name, function))
        yield
        require(active["phase"] is None and next(remaining, None) is None,
                "incomplete registered final suffix")


def classify(rows):
    require(len(rows) == 36 and all(Fraction(r["predicted_delta"]) != 0 for r in rows),
            "incomplete or zero directional contrast")
    return "supported" if all(Fraction(r["predicted_delta"])*Fraction(r["observed_delta"]) > 0
                              for r in rows) else "rejected"


def run_tests(evidence, report, outputs, identity, retained):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_suffix_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT = evidence, report
    module.OUTPUTS, module.IDENTITY = outputs, identity
    module.RETAINED = retained
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
               "terminal_suffix_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
             "selected_row_head_invocations": 0}
    with bridge.read_only(audit):
        evidence = bridge.measure()
    with hotspot.read_only(audit):
        census = hotspot.census.census(evidence["report"])
        dominance = hotspot.dominance.audit_census(census, evidence["report"])
        retained = hotspot.audit_coordinates(evidence, census, dominance)
        predictions = preregister(retained)
    identity = digest(evidence)
    jobs, prepared = [], {}
    for control in parent.CONTROLS:
        words, operand = prepare(evidence["archives"][control],
                                 evidence["reference_archive"], evidence["binary64"])
        prepared[control] = (words, operand)
        jobs.extend((evidence["archives"][control]["stage18"], words))
    rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs = [], {}
    with suffix_only(audit, jobs, evidence["weight_array"], rows):
        for control in parent.CONTROLS:
            normalized, _ = parent.rmsnorm(evidence["archives"][control]["stage18"],
                                           evidence["weight_array"])
            baseline = parent.logits(normalized, rows)
            same_array(normalized, evidence["arrays"][control]["rmsnorm"],
                       "baseline final RMSNorm closure failed")
            same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)],
                       "baseline selected-row head closure failed")
            words, operand = prepared[control]
            normalized, scalars = parent.rmsnorm(words, evidence["weight_array"])
            changed = parent.logits(normalized, rows)
            require(np.isfinite(normalized.view("<f2")).all()
                    and np.isfinite(changed.view("<f2")).all(), "nonfinite suffix")
            outputs[control] = {"working_stage18": words, "rmsnorm": normalized,
                                "logits": changed, "scalars": scalars, "operand": operand}
            for left, right in hotspot.PAIRS:
                li, ri = IDS.index(left), IDS.index(right)
                old = Fraction(float(baseline.view("<f2")[li]))-Fraction(float(baseline.view("<f2")[ri]))
                new = Fraction(float(changed.view("<f2")[li]))-Fraction(float(changed.view("<f2")[ri]))
                for branch in ("fp16", "binary64"):
                    ref = evidence["references"]["logits_"+branch]
                    ref = ref.view("<f2") if branch == "fp16" else ref
                    reference_margin = Fraction(float(ref[left]))-Fraction(float(ref[right]))
                    contrasts.append({
                        "left_id": left, "right_id": right, "control": control, "branch": branch,
                        "predicted_delta": str(predictions[left, right, control]),
                        "observed_delta": str(new-old), "retained_actual_margin": str(old),
                        "intervened_actual_margin": str(new), "fixed_reference_margin": str(reference_margin),
                        "retained_margin_change": str(old-reference_margin),
                        "intervened_margin_change": str(new-reference_margin),
                    })
    protect(evidence, identity)
    report = {"preregistration": PREREGISTRATION, "contrasts": contrasts,
              "classification": classify(contrasts), "retained_common_component": "UNKNOWN",
              "counterfactual_common_component": "UNKNOWN",
              "common_selection_note": "Directional response is not a new component census "
                                       "or authorization to replace the historical common gate.",
              "operands": {c: value["operand"] for c, value in outputs.items()},
              "retained_hotspot_identity": digest(retained)}
    with hotspot.read_only(audit):
        tests = run_tests(evidence, report, outputs, identity, retained)
        protect(evidence, identity)
        for pin in (*pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
    same(audit, {"forbidden_calls": 0, "final_rmsnorm_invocations": 18,
                 "selected_row_head_invocations": 18}, "suffix dispatch census changed")
    return {"diagnostic_id": NAME, "version": 1, "status": report["classification"],
            "command": COMMAND, "report": report, "tests": tests,
            "diagnostic_sources": origins, "reviewed_parent_pins": pins,
            "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
            "assets": evidence["assets"], "protected_input_identity": identity,
            "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
            "retained_controls_and_failure_gates": evidence["result"]["controls"],
            "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
            "flags": FLAGS, "dispatch_and_write_audit": audit,
            "normal_host_review": "REQUIRED", "claim_boundary": PREREGISTRATION["boundary"]}


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
