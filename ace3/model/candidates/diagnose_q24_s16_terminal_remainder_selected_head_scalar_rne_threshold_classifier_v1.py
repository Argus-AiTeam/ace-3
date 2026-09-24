"""Selected-row scalar RNE split at retained final hidden; CPU, never admission."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_final_rmsnorm_headonly_three_sixtyfourths_vector_dose_classifier_v1 as retained


base, parent, closed, bridge = retained.base, retained.parent, retained.closed, retained.bridge
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_scalar_rne_threshold_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS = (319, 34319), ((319, 34319), (34319, 319))
BRANCHES, POLARITIES = ("fp16", "binary64"), ("forward", "reverse")
COORDINATES, DOSE, EXPECTED_TESTS = tuple(range(896)), Fraction(3, 64), 12
PARENT_MISSION = "d398f0a6bcda"
PARENT_PINS = (
    {"path": str(retained.SOURCE), "sha256": "16e535380ccdc50a1de18667b91f8894ab255515a3bef33fa7a66fdf60943053"},
    {"path": str(retained.TEST), "sha256": "01e938d50efb0ad928dcb6be5e8d627438d18d648f7b672849fbd91f3be4b689"},
    {"path": str(closed.HANDOFFS / PARENT_MISSION / "round-0001.json"),
     "sha256": "97f633469596f4764ab7b753fabd217b17d13090c980f21d5d213ac59a4652fa"},
)
PARENT_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "c4cbd787-a7a8-4019-9440-26e66f3cf44d/events.jsonl",
    "sha256": "1082345222a2e2701aadd475a79b2eabc16ea2b574c571d80a394b0a9f9cf9ec",
}
PARENT_CALL = "call_RDPOzhUT05f87GMy7lSLdGRZ"
EXACT_SUCCESSOR = "exact-dose accumulator threshold localization"
HIDDEN_SUCCESSOR = "hidden-RNE-induced accumulator threshold crossing"
PREREGISTRATION = {
    "operand": "Use retained final-RMSNorm hidden and selected FP16 tied-head rows "
               "319 and 34319 only. H is original-input FP16 final hidden minus "
               "independently propagated original-input binary64 final hidden. "
               "Recover exact targets actual +/- 3*H/64 and their native FP16 RNE "
               "working operands in all-896-coordinate input order. Do not replay "
               "any RMSNorm, closed diagnostic, stage18, prefix or reference producer.",
    "prediction": "Round individual exact target-dot scalars and individual "
                  "post-hidden-RNE working-dot scalars, never their differences. "
                  "Compare both with the authenticated 72/72 closed rounded-margin "
                  "pattern using the same individually rounded baseline scalars.",
    "exact_target_72_of_72": EXACT_SUCCESSOR,
    "only_working_72_of_72": HIDDEN_SUCCESSOR,
    "neither_or_integrity_failure": "UNKNOWN/integrity; no scientific successor",
    "boundary": retained.PREREGISTRATION["boundary"] + " Scalar threshold classifier "
                "only; no causal dominance, strict-FP16-state W4A16 or repair claim.",
}
FLAGS = {**retained.FLAGS, "closed_headonly_classifier_replay": False,
         "strict_fp16_state_w4a16_claim": False, "rounding_only_causal_claim": False}
PARENT_FIELDS = (
    "control", "polarity", "left_id", "right_id", "branch", "target_dot_delta",
    "pre_head_rne_delta", "observed_delta", "closed_bracket_delta",
    "retained_margin_change", "intervened_margin_change",
)


def selection_gate():
    same((IDS, PAIRS, BRANCHES, POLARITIES, COORDINATES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          ("forward", "reverse"), tuple(range(896))), "scalar selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(3, 64), "scalar dose changed")


@contextmanager
def read_only(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("scalar classifier forbids closed replay and head dispatch")

    with retained.hotspot.read_only(audit), retained.no_closed_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare", "head_only"):
            stack.enter_context(patch.object(retained, name, refuse))
        stack.enter_context(patch.object(parent.head, "decode_array_q24", refuse))
        yield


def authenticate_parent():
    pins = retained.authenticate_chain()
    for pin in (*PARENT_PINS, PARENT_RECEIPT):
        base.read_bound(pin)
    review = json.loads(base.read_bound(PARENT_PINS[-1]))
    same((review["kind"], review["mission_id"], review["producer_role"],
          review["round"], review["review"]["status"]),
         ("round_reviewed_handoff", PARENT_MISSION, "reviewer", 1, "done"),
         "selected-head independent review gate failed")
    latest_pin = parent.record(Path(PARENT_PINS[-1]["path"]).with_name("latest.json"))
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", PARENT_PINS[-1]["path"]), "selected-head review lineage changed")
    events = (json.loads(line) for line in base.read_bound(PARENT_RECEIPT).splitlines())
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == PARENT_CALL]
    same(len(matches), 1, "selected-head receipt missing or duplicated")
    require(matches[0]["success"] is True, "selected-head receipt execution failed")
    text, marker = matches[0]["result"]["content"], '{"changed_coordinate_counts":'
    same(text.count(marker), 1, "selected-head receipt document missing or duplicated")
    summary, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    return [*pins, *PARENT_PINS, latest_pin], summary


def bind_parent(evidence, summary, comparisons):
    same([summary[k] for k in ("status", "native_exit", "stdout_json_documents",
                              "pattern_agreements", "exact_dose_separated_contrasts")],
         ["supported", 0, 1, 72, 72], "closed head-only outcome changed")
    same(summary["protected_input_identity"], closed.digest(evidence),
         "closed head-only source/token/Q24/state/KV/lineage/reference/row drift")
    same(summary["flags"], retained.FLAGS, "closed head-only claim boundary changed")
    same(summary["dispatch_and_write_audit"],
         {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
          "selected_row_head_invocations": 27}, "closed head-only dispatch drift")
    same([summary["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [12, 0, 0, 0], "closed head-only test result changed")
    same(summary["tests"]["compiled"], [parent.record(retained.SOURCE), parent.record(retained.TEST)],
         "closed head-only source/test drift")
    same(summary["changed_coordinate_counts"],
         {p: {c: 801 for c in parent.CONTROLS} for p in POLARITIES},
         "closed head-only hidden operand census changed")
    same(summary["contrast_fields"], list(PARENT_FIELDS), "closed head-only schema changed")
    values = summary["contrast_values"]
    require(isinstance(values, list) and len(values) == 36
            and all(isinstance(v, list) and len(v) == len(PARENT_FIELDS) for v in values),
            "closed head-only contrast shape changed")
    rows = [dict(zip(PARENT_FIELDS, v, strict=True)) for v in values]
    same([tuple(r[k] for k in PARENT_FIELDS[:5]) for r in rows],
         [(c, p, l, r, "binary64") for c in parent.CONTROLS for p in POLARITIES for l, r in PAIRS],
         "closed head-only ordered contrast census changed")
    for row in rows:
        key = tuple(row[k] for k in PARENT_FIELDS[:4])
        same((row["observed_delta"], row["closed_bracket_delta"]),
             (comparisons[key], comparisons[key]), "closed head-only pattern drift")
    return rows


def rne(value):
    require(type(value) is Fraction and value.denominator & (value.denominator - 1) == 0,
            "native RNE requires an exact dyadic scalar")
    bits, saturated = parent.head.fixed_to_f16(value.numerator, value.denominator.bit_length() - 1)
    require(not saturated, "selected scalar saturated")
    rounded = Fraction(float(np.asarray(bits, dtype="<u2").view("<f2")))
    return {"exact": str(value), "fp16_word": bits, "rounded": str(rounded),
            "rne_remainder": str(rounded - value)}


def accumulate(values, rows, audit):
    selection_gate()
    require(isinstance(values, tuple) and len(values) == 896
            and all(type(v) is Fraction for v in values), "invalid selected-dot operand")
    require(isinstance(rows, np.ndarray) and rows.shape == (2, 896)
            and rows.dtype == np.dtype("<f2") and np.isfinite(rows).all(),
            "invalid selected head rows")
    result = retained.dots(values, rows)
    audit["exact_selected_row_accumulations"] += len(result)
    return result


def decide(target, working, total):
    require(total == 72 and type(total) is int
            and all(type(n) is int and 0 <= n <= total for n in (target, working)),
            "invalid scalar agreement census")
    if target == total:
        return "supported", EXACT_SUCCESSOR
    if working == total:
        return "supported", HIDDEN_SUCCESSOR
    return "UNKNOWN", None


def classify(contrasts):
    selection_gate()
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in contrasts],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS for b in BRANCHES], "scalar contrast census changed")
    for row in contrasts:
        expected = Fraction(-1, 128) if row["polarity"] == "forward" else Fraction(
            1, 64 if row["control"] == "mapped_all" else 128)
        require(Fraction(row["closed_bracket_delta"]) == (
            expected if row["left_id"] == 319 else -expected), "scalar closed pattern drift")
        old, ref = (Fraction(row[k]) for k in ("baseline_rounded_margin", "fixed_reference_margin"))
        require(Fraction(row["retained_margin_change"]) == old - ref, "scalar reference reanchored")
        for name in ("target", "working"):
            new = Fraction(row[name + "_rounded_margin"])
            require(Fraction(row[name + "_delta"]) == new - old, "scalar rounded-margin closure failed")
            require(Fraction(row[name + "_margin_change"]) == new - ref, "scalar global-reference closure failed")
    target = sum(r["target_delta"] == r["closed_bracket_delta"] for r in contrasts)
    working = sum(r["working_delta"] == r["closed_bracket_delta"] for r in contrasts)
    status, successor = decide(target, working, len(contrasts))
    return {"classification": status, "successor": successor,
            "target_pattern_agreements": target, "working_pattern_agreements": working,
            "contrast_count": len(contrasts),
            "integrity_error": None if successor else "Neither scalar branch reproduces the closed pattern"}


def measure(evidence, comparisons, audit):
    selection_gate()
    rows = np.stack([evidence["rows"][i] for i in IDS])
    delta = tuple(a - b for a, b in zip(
        retained.exact(evidence["references"]["rmsnorm_fp16"].view("<f2")),
        retained.exact(evidence["references"]["rmsnorm_binary64"]), strict=True))
    require(any(delta), "zero frozen final-hidden residual")
    residual_dots = accumulate(delta, rows, audit)
    scalars, contrasts, operands = [], [], []
    for control in parent.CONTROLS:
        words = evidence["arrays"][control]["rmsnorm"]
        require(words.shape == (896,) and words.dtype == np.dtype("<u2"),
                "invalid retained final-hidden operand")
        original = retained.exact(words.view("<f2"))
        baseline_dots = accumulate(original, rows, audit)
        baseline = [rne(v) for v in baseline_dots]
        audit["accumulator_rne_scalars"] += 2
        same([v["fp16_word"] for v in baseline],
             evidence["arrays"][control]["logits"][list(IDS)].tolist(),
             "baseline individual selected-row closure failed")
        for polarity in POLARITIES:
            signed = -DOSE if polarity == "forward" else DOSE
            targets = tuple(a + signed * h for a, h in zip(original, delta, strict=True))
            working_words = [rne(v)["fp16_word"] for v in targets]
            audit["hidden_rne_scalars"] += len(targets)
            working_values = retained.exact(np.asarray(working_words, dtype="<u2").view("<f2"))
            target_dots = accumulate(targets, rows, audit)
            working_dots = accumulate(working_values, rows, audit)
            target, working = ([rne(v) for v in dots] for dots in (target_dots, working_dots))
            audit["accumulator_rne_scalars"] += 4
            require(tuple(t - b for t, b in zip(target_dots, baseline_dots, strict=True))
                    == tuple(signed * h for h in residual_dots), "individual exact-dose closure failed")
            operands.append({
                "control": control, "polarity": polarity, "signed_dose": str(signed),
                "baseline_hidden_words": words.tolist(), "exact_targets": list(map(str, targets)),
                "working_hidden_words": working_words, "coordinate_order": list(COORDINATES),
            })
            for i, row_id in enumerate(IDS):
                scalars.append({"control": control, "polarity": polarity, "row_id": row_id,
                                "baseline": baseline[i], "exact_target": target[i], "working": working[i],
                                "frozen_hidden_dot_residual": str(residual_dots[i]),
                                "hidden_rne_dot_remainder": str(working_dots[i] - target_dots[i])})
            for left, right in PAIRS:
                i, j = IDS.index(left), IDS.index(right)
                old, new_target, new_working = (
                    Fraction(v[i]["rounded"]) - Fraction(v[j]["rounded"])
                    for v in (baseline, target, working))
                for branch in BRANCHES:
                    reference = evidence["references"]["logits_" + branch][list(IDS)]
                    reference = retained.exact(reference.view("<f2") if branch == "fp16" else reference)
                    ref = reference[i] - reference[j]
                    contrasts.append({
                        "control": control, "polarity": polarity, "left_id": left, "right_id": right,
                        "branch": branch, "baseline_rounded_margin": str(old),
                        "target_rounded_margin": str(new_target), "working_rounded_margin": str(new_working),
                        "target_delta": str(new_target - old), "working_delta": str(new_working - old),
                        "target_dot_delta": str(target_dots[i] - target_dots[j] - baseline_dots[i] + baseline_dots[j]),
                        "pre_head_rne_delta": str(working_dots[i] - working_dots[j] - baseline_dots[i] + baseline_dots[j]),
                        "fixed_reference_margin": str(ref), "retained_margin_change": str(old - ref),
                        "target_margin_change": str(new_target - ref),
                        "working_margin_change": str(new_working - ref),
                        "closed_bracket_delta": comparisons[control, polarity, left, right],
                    })
    return {"scalars": scalars, "contrasts": contrasts, "operands": operands,
            "frozen_final_hidden_residual": list(map(str, delta)),
            "selected_head_row_identity": closed.digest(rows),
            "frozen_final_hidden_residual_identity": closed.digest(delta)}


def run_tests(evidence, report, summaries, parent_summary, comparisons, pins):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("selected_scalar_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.SUMMARIES = evidence, report, summaries
    module.PARENT_SUMMARY, module.COMPARISONS, module.PINS = parent_summary, comparisons, pins
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
            "skipped": len(result.skipped), "compiled": [parent.record(SOURCE), parent.record(TEST)]}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "selected_scalar_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0, "selected_row_head_invocations": 0,
             "prefix_replays": 0, "admission_replays": 0, "reference_producer_replays": 0,
             "full_vocabulary_replays": 0, "closed_stage18_replays": 0,
             "closed_headonly_replays": 0, "writes": 0, "artifact_overwrites": 0,
             "exact_selected_row_accumulations": 0, "hidden_rne_scalars": 0, "accumulator_rne_scalars": 0}
    with read_only(audit):
        pins, parent_summary = authenticate_parent()
        summaries = retained.retained_doses()
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        comparisons = retained.bind_doses(evidence, summaries)
        closed_rows = bind_parent(evidence, parent_summary, comparisons)
        identity = closed.digest(evidence)
        frozen = closed.digest((summaries, parent_summary, comparisons, PREREGISTRATION, FLAGS))
        report = measure(evidence, comparisons, audit)
        binary64_rows = [r for r in report["contrasts"] if r["branch"] == "binary64"]
        for actual, previous in zip(binary64_rows, closed_rows, strict=True):
            for key in ("target_dot_delta", "pre_head_rne_delta", "retained_margin_change"):
                same(actual[key], previous[key], "recovered scalar/closed receipt disagreement")
            same(actual["working_delta"], previous["observed_delta"], "working scalar/closed pattern disagreement")
            same(actual["working_margin_change"], previous["intervened_margin_change"],
                 "working scalar/closed reference disagreement")
        report.update(classify(report["contrasts"]))
        report.update({"preregistration": PREREGISTRATION, "retained_common_component": "UNKNOWN",
                       "counterfactual_common_component": "UNKNOWN",
                       "retained_common_selection_rule": bridge.SELECTION_RULE})
        result_identity = closed.digest(report)
        tests = run_tests(evidence, report, summaries, parent_summary, comparisons, pins)
        closed.protect(evidence, identity)
        same(closed.digest(report), result_identity, "scalar result mutation")
        same(closed.digest((summaries, parent_summary, comparisons, PREREGISTRATION, FLAGS)),
             frozen, "closed evidence/contract mutation")
        for pin in (*pins, *origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"], *retained.RECEIPTS, PARENT_RECEIPT):
            base.read_bound(pin)
    same({k: v for k, v in audit.items() if k not in (
        "exact_selected_row_accumulations", "hidden_rne_scalars", "accumulator_rne_scalars")},
         {k: 0 for k in audit if k not in (
             "exact_selected_row_accumulations", "hidden_rne_scalars", "accumulator_rne_scalars")},
         "forbidden dispatch/write census changed")
    same([audit[k] for k in ("exact_selected_row_accumulations", "hidden_rne_scalars", "accumulator_rne_scalars")],
         [92, 16128, 90], "bounded scalar budget changed")
    return {
        "diagnostic_id": NAME, "version": 1, "status": report["classification"], "command": COMMAND,
        "report": report, "tests": tests, "diagnostic_sources": origins, "reviewed_parent_pins": pins,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"], "protected_input_identity": identity,
        "retained_dose_receipts": [*retained.RECEIPTS, PARENT_RECEIPT],
        "retained_dose_summaries": summaries, "retained_headonly_summary": parent_summary,
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "flags": FLAGS, "dispatch_and_write_audit": audit, "normal_host_review": "REQUIRED",
        "claim_boundary": PREREGISTRATION["boundary"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", required=True, action="store_true")
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, ArithmeticError, LookupError, TypeError, ImportError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "classification": "UNKNOWN/integrity",
                          "successor": None, "integrity_error": f"{type(error).__name__}: {error}",
                          "flags": FLAGS, "normal_host_review": "REQUIRED"}, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "supported" else 1


if __name__ == "__main__":
    sys.exit(main())
