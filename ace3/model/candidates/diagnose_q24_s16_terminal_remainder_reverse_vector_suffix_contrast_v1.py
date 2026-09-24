"""Input-ordered reverse terminal vector through a non-admission final suffix."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_non363_complement_suffix_contrast_v1 as complement


vector, closed, bridge, base, parent, hotspot = (
    complement.vector, complement.closed, complement.bridge,
    complement.base, complement.parent, complement.hotspot)
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_reverse_vector_suffix_contrast_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES = tuple(range(896))
EXPECTED_TESTS = 14
COMPLEMENT_CHAIN = (
    complement, "a16e613b99f9",
    "a5013d7277ecf312d90dbf288ff4886aa983e4a27cd0215eda426dffbb69fa6f",
    "d713547230a1f87a56e4ac3ff0b9ca1954cae0dc5b38c46bb9a1aa424af22fbe",
    "af0ba36cef62404a6af2b895cdfd392a02fffb57d95525a64f8f56fdb783c19a",
)
PREDICTION_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "b00973d1-0134-4a8d-9d39-52182b241dc2/events.jsonl",
    "sha256": "ba6762bab7c23edec05dd34051b028babc3c714e980fe2a837161143d7ee776e",
}
PREDICTION_CALL = "call_NivGaAgukMWbSutzqlmWkKdl"
PREREGISTRATION = {
    "selection": "Exactly all 896 coordinates in input order, including zeros. "
                 "No ranking, selected-coordinate union, fitting or coordinate search.",
    "operand": "Freeze delta = original_FP16_stage18 - original_binary64_terminal. "
               "For every control, working_stage18[i] = native_RNE_FP16("
               "actual_FP16_stage18[i] + delta[i]). Disclose exact targets and rounding.",
    "prediction": "Every outer-pair/control/reference final margin delta has the strict "
                  "nonzero sign of the retained binary64 terminal-remainder bridge "
                  "contribution, i.e. minus the recorded forward prediction. The "
                  "full-vector linear bridge is a binding check, not that prediction.",
    "supported": "Local sign-symmetric aggregate terminal-boundary sensitivity.",
    "rejected": "Conversion/asymmetry/cancellation follow-up; any zero or opposite response.",
    "UNKNOWN": "Authentication or integrity defect; no scientific interpretation.",
    "boundary": vector.PREREGISTRATION["boundary"] +
                " The coordinate-363, all-896 forward, non-363 and middle-pair "
                "interventions remain closed. Historical common UNKNOWN is not "
                "reclassified; detailed coordinate gate censuses are not reconstructed.",
}
FLAGS = {
    **vector.FLAGS,
    "closed_full_vector_intervention_replay": False,
    "closed_non363_intervention_replay": False,
    "coordinate_ranking": False,
    "selected_union_dependency": False,
    "selected_union_reconstruction": False,
}


def authenticate_chain():
    pins = complement.authenticate_chain()
    module, mission, source, test, review = COMPLEMENT_CHAIN
    review_path = closed.HANDOFFS / mission / "round-0001.json"
    for path, digest in ((module.SOURCE, source), (module.TEST, test), (review_path, review)):
        pin = {"path": str(path), "sha256": digest}
        base.read_bound(pin)
        pins.append(pin)
    latest_path = review_path.with_name("latest.json")
    latest = json.loads(latest_path.read_bytes())
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", str(review_path)), "closed complement review lineage changed")
    record = json.loads(base.read_bound(pins[-1]))
    same((record["kind"], record["mission_id"], record["producer_role"],
          record["round"], record["review"]["status"]),
         ("round_reviewed_handoff", mission, "reviewer", 1, "done"),
         "closed complement independent review gate failed")
    pins.append(parent.record(latest_path))
    return pins


def retained_predictions():
    events = [json.loads(line) for line in base.read_bound(PREDICTION_RECEIPT).splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == PREDICTION_CALL]
    same(len(matches), 1, "retained prediction receipt missing or duplicated")
    require(matches[0]["success"] is True, "retained prediction execution failed")
    lines = [line for line in matches[0]["result"]["content"].splitlines()
             if line.startswith('{"changed_coordinate_counts":')]
    same(len(lines), 1, "retained prediction summary missing or duplicated")
    summary = json.loads(lines[0])
    same((summary["status"], summary["single_stdout_json"], summary["directional_rows"]),
         ("supported", True, 36), "closed vector outcome changed")
    tests = summary["tests"]
    same([tests[k] for k in ("executed", "errors", "failures", "skipped")],
         [14, 0, 0, 0], "retained prediction tests failed")
    same(tests["compiled"], [parent.record(vector.SOURCE), parent.record(vector.TEST)],
         "retained prediction source/test identity changed")
    rows = summary["contrasts"]
    same([(r["control"], tuple(r["pair"])) for r in rows],
         [(c, p) for c in parent.CONTROLS for p in PAIRS],
         "retained prediction control/pair census changed")
    require(all(Fraction(r["predicted_delta"]) * Fraction(r["observed_delta"]) > 0
                for r in rows), "closed vector directional outcome changed")
    return summary


@contextmanager
def no_reselection_or_closed_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("coordinate ranking/union or closed intervention replay forbidden")

    with ExitStack() as stack:
        for module in (closed, vector, complement):
            for name in ("check", "run_tests", "prepare", "preregister"):
                stack.enter_context(patch.object(module, name, refuse))
        for module, names in (
            (bridge, ("measure", "report")),
            (bridge.margin, ("report", "selected_coordinates")),
            (bridge.hidden, ("report",)),
            (hotspot, ("coordinate_row", "audit_coordinates")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, refuse))
        yield


def selection_gate():
    same((IDS, PAIRS, BRANCHES, COORDINATES),
         ((319, 34319), ((319, 34319), (34319, 319)),
          ("fp16", "binary64"), tuple(range(896))), "reverse vector selection changed")


def prepare(words, delta):
    selection_gate()
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,) and np.isfinite(words.view("<f2")).all(),
            "invalid actual FP16 operand")
    require(isinstance(delta, tuple) and len(delta) == 896
            and all(isinstance(v, Fraction) for v in delta), "invalid frozen terminal vector")
    working, targets, rounding = words.copy(), [], []
    for i in COORDINATES:
        target = Fraction(float(words.view("<f2")[i])) + delta[i]
        require(target.denominator & (target.denominator - 1) == 0, "nondyadic target")
        bits, saturated = parent.head.fixed_to_f16(
            target.numerator, target.denominator.bit_length() - 1)
        require(not saturated, "reverse vector operand saturated")
        working[i] = bits
        targets.append(str(target))
        rounding.append(str(Fraction(float(working.view("<f2")[i])) - target))
    working.flags.writeable = False
    return working, {
        "coordinates": list(COORDINATES), "coordinate_count": 896,
        "coordinate_order": "all_input_order", "polarity": "plus_original_terminal_remainder",
        "exact_targets": targets, "rounding_remainders": rounding,
        "working_fp16_words": working.tolist(),
        "changed_coordinates": np.flatnonzero(words != working).tolist(),
        "frozen_vector_identity": closed.digest(delta),
    }


def preregister(evidence, summary):
    selection_gate()
    reference, rows = hotspot.bound_vectors(evidence)
    delta = tuple(a - b for a, b in zip(
        reference["stage18"], evidence["hidden"]["binary64"], strict=True))
    require(any(delta), "zero terminal vector")
    anchor = Fraction(bridge.hidden.norm_summary(evidence["hidden"]["binary64"])
                      ["inverse_norm_anchor"])
    predictions = {}
    for row in summary["contrasts"]:
        left, right = row["pair"]
        linear = sum((v * w * anchor * (a - b) for v, w, a, b in
                      zip(delta, evidence["weights"], rows[left], rows[right], strict=True)),
                     Fraction())
        require(linear == -Fraction(row["full_vector_linear_bridge_delta"]),
                "retained prediction original-input/weight/reference binding changed")
        predictions[row["control"], left, right] = {
            "predicted_delta": str(-Fraction(row["predicted_delta"])),
            "retained_terminal_component_signed": str(-Fraction(row["predicted_delta"])),
            "full_vector_linear_bridge_delta": str(linear),
        }
    same(list(predictions), [(c, l, r) for c in parent.CONTROLS for l, r in PAIRS],
         "retained prediction census changed")
    return delta, predictions


classify = vector.classify


def run_tests(evidence, report, outputs, identity, delta):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("terminal_reverse_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.OUTPUTS = evidence, report, outputs
    module.IDENTITY, module.DELTA = identity, delta
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
    summary = retained_predictions()
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "terminal_reverse_tests": parent.record(TEST)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("ace3.") and path and Path(path).suffix == ".py":
            require(Path(path).resolve().is_relative_to(ROOT), "foreign source origin")
            origins[name] = parent.record(Path(path).resolve())
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
             "selected_row_head_invocations": 0}
    with hotspot.read_only(audit), no_reselection_or_closed_replay(audit):
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        delta, predictions = preregister(evidence, summary)
        prepared = {c: prepare(evidence["archives"][c]["stage18"], delta)
                    for c in parent.CONTROLS}
    identity = closed.digest(evidence)
    frozen = closed.digest((delta, predictions, prepared, summary))
    jobs = [words for c in parent.CONTROLS
            for words in (evidence["archives"][c]["stage18"], prepared[c][0])]
    rows = np.stack([evidence["rows"][i] for i in IDS])
    contrasts, outputs, obstructions = [], {}, []
    with no_reselection_or_closed_replay(audit), closed.suffix_only(
            audit, jobs, evidence["weight_array"], rows):
        for control in parent.CONTROLS:
            normalized, _ = parent.rmsnorm(evidence["archives"][control]["stage18"],
                                           evidence["weight_array"])
            baseline = parent.logits(normalized, rows)
            closed.same_array(normalized, evidence["arrays"][control]["rmsnorm"],
                              "baseline final RMSNorm closure failed")
            closed.same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)],
                              "baseline selected-row closure failed")
            words, operand = prepared[control]
            normalized, scalars = parent.rmsnorm(words, evidence["weight_array"])
            changed = parent.logits(normalized, rows)
            require(np.isfinite(normalized.view("<f2")).all()
                    and np.isfinite(changed.view("<f2")).all(), "nonfinite suffix output")
            outputs[control] = {"working_stage18": words, "rmsnorm": normalized,
                                "logits": changed, "scalars": scalars, "operand": operand}
            for left, right in PAIRS:
                li, ri = IDS.index(left), IDS.index(right)
                old = Fraction(float(baseline.view("<f2")[li])) - Fraction(float(baseline.view("<f2")[ri]))
                new = Fraction(float(changed.view("<f2")[li])) - Fraction(float(changed.view("<f2")[ri]))
                for branch in BRANCHES:
                    ref = evidence["references"]["logits_" + branch]
                    ref = ref.view("<f2") if branch == "fp16" else ref
                    reference_margin = Fraction(float(ref[left])) - Fraction(float(ref[right]))
                    obstruction = {
                        "control": control, "left_id": left, "right_id": right, "branch": branch,
                        "retained_actual_margin": str(old),
                        "fixed_reference_margin": str(reference_margin),
                        "retained_margin_change": str(old - reference_margin),
                        "retained_common_component": "UNKNOWN",
                        "retained_branch_terminal_component_signed":
                            predictions[control, left, right]["predicted_delta"]
                            if branch == "binary64" else "0",
                    }
                    obstructions.append(obstruction)
                    contrasts.append({
                        **obstruction, **predictions[control, left, right],
                        "observed_delta": str(new - old), "intervened_actual_margin": str(new),
                        "intervened_margin_change": str(new - reference_margin),
                    })
    closed.protect(evidence, identity)
    report = {
        "preregistration": PREREGISTRATION, "classification": classify(contrasts),
        "contrasts": contrasts, "operands": {c: v["operand"] for c, v in outputs.items()},
        "frozen_original_terminal_vector": list(map(str, delta)),
        "retained_prediction_receipt": PREDICTION_RECEIPT,
        "retained_prediction_call": PREDICTION_CALL,
        "retained_forward_summary": summary,
        "retained_common_selection_obstructions": obstructions,
        "retained_common_selection_rule": bridge.SELECTION_RULE,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "obstruction_scope": "Retained signed terminal prediction and fixed margin change, "
                             "not a reconstruction of the missing coordinate gate census.",
        "closed_results": [
            {"mission": chain[1], "review_sha256": chain[4],
             "classification": "supported", "directional_contrasts": "36/36", "replayed": False}
            for chain in (vector.CLOSED_CHAIN, complement.VECTOR_CHAIN, COMPLEMENT_CHAIN)
        ],
    }
    output_identity = closed.digest((report, outputs))
    with hotspot.read_only(audit), no_reselection_or_closed_replay(audit):
        tests = run_tests(evidence, report, outputs, identity, delta)
        closed.protect(evidence, identity)
        same(closed.digest((delta, predictions, prepared, summary)), frozen, "frozen operand drift")
        same(closed.digest((report, outputs)), output_identity, "suffix result drift")
        for pin in (*pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"], PREDICTION_RECEIPT):
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
