"""Frozen final-hidden 3/64 contrast through two head rows; CPU, never admission."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_three_sixtyfourths_vector_dose_bracket_v1 as bracket


base, parent, closed, bridge, hotspot = (
    bracket.base, bracket.parent, bracket.closed, bracket.bridge, bracket.hotspot)
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_final_rmsnorm_headonly_three_sixtyfourths_vector_dose_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, BRANCHES = (319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64")
COORDINATES, POLARITIES, DOSE = tuple(range(896)), ("forward", "reverse"), Fraction(3, 64)
EXPECTED_TESTS = 12
BRACKET_MISSION = "f3d86d60d275"
BRACKET_PINS = (
    {"path": str(bracket.SOURCE), "sha256": "73e0f4cb94dd93ffea0ff610a14ddc90cd16da4a7c0b728fff030362adf69f75"},
    {"path": str(bracket.TEST), "sha256": "aaa8a9b702cb2da78089a0e1ea78499759259dd81db7e31969bedf3e73788de0"},
    {"path": str(closed.HANDOFFS / BRACKET_MISSION / "round-0001.json"),
     "sha256": "6669f0c0aa68a85267235cfeedf2c2c7939355cfe68f9a72663e9a490e7c63cd"},
)
BRACKET_RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "32dae28e-eb59-43e5-a3f1-cf7380230f35/events.jsonl",
    "sha256": "f4984358e7c3c387ed93dfd8679afd4a1263edabb4065b6ccae340ac7352ef2e",
}
BRACKET_CALL = "call_c2VB09Msd2dcjtLxlHaSsu3E"
RECEIPTS = (*bracket.RECEIPTS, BRACKET_RECEIPT)
CONTRAST_FIELDS = (*bracket.CONTRAST_FIELDS[:7], bracket.DOSE_KEYS[0],
                   *bracket.CONTRAST_FIELDS[7:])
HEAD_SUCCESSOR = "selected-head accumulation/RNE threshold localization"
NORM_SUCCESSOR = "final-RMSNorm input/scale/cancellation localization"
PREREGISTRATION = {
    "operand": "Freeze H = original-input final_RMSNorm_FP16 hidden minus independently "
               "propagated original-input final_RMSNorm_binary64 hidden. For every "
               "retained actual final hidden, traverse all 896 coordinates in input order: "
               "native_RNE_FP16(actual[i] +/- 3*H[i]/64), minus forward, plus reverse. "
               "Exact rational targets precede one conversion. No stage18 operand, "
               "RMSNorm, closed intervention or original-reference producer is replayed.",
    "prediction": "The fixed head-only 3/64 intervention reproduces the closed 3/64 "
                  "rounded ordered-pair margin movement in all 72 contrasts: for "
                  "(319,34319), forward -1/128; reverse +1/128 except mapped_all +1/64. "
                  "Opposite pairs negate; original FP16/binary64 references remain fixed.",
    "separation": "Exact target row-dot movement must equal +/-3/64 times the frozen "
                  "hidden row-dot residual, nonzero and strictly between its analytical "
                  "1/32 and 1/16 movements in magnitude. These are scalar identities, "
                  "not neighboring-dose operands or executions. Retain separately the "
                  "exact post-hidden-RNE/pre-head-RNE dots and conversion remainders.",
    "supported": HEAD_SUCCESSOR,
    "rejected": NORM_SUCCESSOR,
    "UNKNOWN": "Authentication, integrity or exact-separation failure; no scientific successor.",
    "boundary": bracket.PREREGISTRATION["boundary"] + " Closed 3/64 rejection stays fixed. "
                "Reference-guided head-only counterfactual; no root-cause or rounding-only claim.",
}
FLAGS = {
    **bracket.FLAGS, "closed_three_sixtyfourths_intervention_replay": False,
    "stage18_operand_construction": False, "rmsnorm_recomputation": False,
    "rmsnorm_operator_replay": False, "neighboring_hidden_dose_execution": False,
}


def authenticate_chain():
    pins = bracket.authenticate_chain()
    for pin in BRACKET_PINS:
        base.read_bound(pin)
    record = json.loads(base.read_bound(BRACKET_PINS[-1]))
    same((record["kind"], record["mission_id"], record["producer_role"],
          record["round"], record["review"]["status"]),
         ("round_reviewed_handoff", BRACKET_MISSION, "reviewer", 1, "done"),
         "closed bracket independent review gate failed")
    latest_path = Path(BRACKET_PINS[-1]["path"]).with_name("latest.json")
    latest_pin = parent.record(latest_path)
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", BRACKET_PINS[-1]["path"]), "closed bracket review lineage changed")
    return [*pins, *BRACKET_PINS, latest_pin]


def retained_doses():
    earlier = bracket.retained_doses()
    events = [json.loads(line) for line in base.read_bound(BRACKET_RECEIPT).splitlines()]
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == BRACKET_CALL]
    same(len(matches), 1, "closed bracket receipt missing or duplicated")
    require(matches[0]["success"] is True, "closed bracket execution failed")
    text = matches[0]["result"]["content"]
    marker = '{"changed_coordinate_counts":'
    same(text.count(marker), 1, "closed bracket summary missing or duplicated")
    summary, _ = json.JSONDecoder().raw_decode(text[text.index(marker):])
    return (*earlier, summary)


def bind_doses(evidence, summaries):
    same(len(summaries), 8, "closed dose census changed")
    _, predictions = bracket.bind_doses(evidence, *summaries[:-1])
    retained = summaries[-1]
    same([retained[k] for k in (
        "status", "native_exit", "stdout_json_documents", "contrast_count",
        "directional_agreements", "strict_dose_agreements", "retained_common_component",
        "reverse_zero_obstructions", "recovered_reverse_zero_contrasts")],
        ["rejected", 0, 1, 72, 72, 36, "UNKNOWN", 32, 32],
        "closed bracket outcome changed")
    same([retained["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [14, 0, 0, 0], "closed bracket tests changed")
    same(retained["tests"]["compiled"], [parent.record(bracket.SOURCE), parent.record(bracket.TEST)],
         "closed bracket source/test changed")
    same(retained["flags"], bracket.FLAGS, "closed bracket claim boundary changed")
    same(retained["dispatch_and_write_audit"],
         {"forbidden_calls": 0, "final_rmsnorm_invocations": 27, "selected_row_head_invocations": 27},
         "closed bracket dispatch changed")
    same(retained["protected_input_identity"], closed.digest(evidence),
         "closed bracket source/token/state/KV/lineage/reference/head identity drift")
    same(retained["changed_coordinate_counts"],
         {p: {c: 795 for c in parent.CONTROLS} for p in POLARITIES},
         "closed bracket operand census changed")
    same(retained["contrast_fields"], list(CONTRAST_FIELDS), "closed bracket schema changed")
    values = retained["contrast_values"]
    require(isinstance(values, list) and len(values) == 36
            and all(isinstance(v, list) and len(v) == len(CONTRAST_FIELDS) for v in values),
            "closed bracket contrast shape changed")
    rows = [dict(zip(CONTRAST_FIELDS, v, strict=True)) for v in values]
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in rows],
         [(c, p, l, r, "binary64") for c in parent.CONTROLS for p in POLARITIES for l, r in PAIRS],
         "closed bracket ordered contrast census changed")
    comparisons = {}
    for row in rows:
        c, p, l, r = (row[k] for k in CONTRAST_FIELDS[:4])
        prediction = predictions[c, l, r, p]
        for field in ("predicted_delta", *bracket.DOSE_KEYS):
            same(row[field], prediction[field], "closed bracket prediction/comparator drift")
        expected = Fraction(-1, 128) if p == "forward" else Fraction(1, 64 if c == "mapped_all" else 128)
        same(row["observed_delta"], str(expected if l == 319 else -expected),
             "closed bracket asymmetry changed")
        actual = exact(evidence["arrays"][c]["logits"][list(IDS)].view("<f2"))
        reference = exact(evidence["references"]["logits_binary64"][list(IDS)])
        i, j = IDS.index(l), IDS.index(r)
        margin = actual[i] - actual[j] - reference[i] + reference[j]
        same(row["retained_margin_change"], str(margin), "closed original-reference margin drift")
        same(row["intervened_margin_change"], str(margin + Fraction(row["observed_delta"])),
             "closed bracket margin closure failed")
        comparisons[c, p, l, r] = row["observed_delta"]
    return comparisons


def selection_gate():
    same((IDS, PAIRS, BRANCHES, COORDINATES, POLARITIES),
         ((319, 34319), ((319, 34319), (34319, 319)), ("fp16", "binary64"),
          tuple(range(896)), ("forward", "reverse")), "head-only selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(3, 64), "head-only fixed dose changed")


def exact(values):
    require(np.isfinite(values).all(), "nonfinite retained operand")
    return tuple(Fraction(float(v)) for v in values)


def dots(values, rows):
    return tuple(sum((a * b for a, b in zip(values, exact(row), strict=True)), Fraction())
                 for row in rows)


def prepare(words, delta, polarity):
    selection_gate()
    require(polarity in POLARITIES, "invalid head-only polarity")
    require(isinstance(words, np.ndarray) and words.dtype == np.dtype("<u2")
            and words.shape == (896,), "invalid retained final hidden")
    require(isinstance(delta, tuple) and len(delta) == 896
            and all(type(v) is Fraction and v.denominator & (v.denominator - 1) == 0 for v in delta),
            "invalid frozen final-hidden residual")
    signed = -DOSE if polarity == "forward" else DOSE
    targets = tuple(a + signed * b for a, b in zip(exact(words.view("<f2")), delta, strict=True))
    working = words.copy()
    for i in COORDINATES:
        value = targets[i]
        bits, saturated = parent.head.fixed_to_f16(value.numerator, value.denominator.bit_length() - 1)
        require(not saturated, "head-only hidden saturated")
        working[i] = bits
    rounded = exact(working.view("<f2"))
    working.flags.writeable = False
    return working, targets, {
        "boundary": "retained_final_rmsnorm_hidden_to_selected_head",
        "coordinate_order": "all_input_order", "coordinates": list(COORDINATES),
        "polarity": polarity, "dose": str(DOSE), "signed_dose": str(signed),
        "exact_targets": list(map(str, targets)), "working_fp16_words": working.tolist(),
        "rounding_remainders": [str(a - b) for a, b in zip(rounded, targets, strict=True)],
        "changed_coordinates": np.flatnonzero(words != working).tolist(),
        "frozen_hidden_vector_identity": closed.digest(delta),
    }


@contextmanager
def no_closed_replay(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed stage18/bracket replay forbidden")

    with bracket.no_closed_replay(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "prepare"):
            stack.enter_context(patch.object(bracket, name, refuse))
        yield


@contextmanager
def head_only(audit, jobs, rows):
    raw_logits, raw_decode = parent.logits, parent.head.decode_array_q24
    position, active = 0, None
    decode_step = 0

    def refuse(message):
        audit["forbidden_calls"] += 1
        raise RuntimeError(message)

    def decode(words):
        nonlocal decode_step
        if active is None or decode_step >= 2:
            return refuse("out-of-cone head decode")
        expected = active if decode_step == 0 else rows.view("<u2")
        closed.same_array(words, expected, "selected head decode splice")
        decode_step += 1
        return raw_decode(words)

    def logits(words, supplied_rows):
        nonlocal position, active, decode_step
        if active is not None or position >= len(jobs):
            return refuse("head-only budget exhausted")
        closed.same_array(words, jobs[position], "unregistered final-hidden operand")
        closed.same_array(supplied_rows, rows, "selected head row/scale splice")
        active, decode_step = words, 0
        output = raw_logits(words, supplied_rows)
        same(decode_step, 2, "selected head decode census changed")
        active = None
        position += 1
        audit["selected_row_head_invocations"] += 1
        return output

    with hotspot.read_only(audit), no_closed_replay(audit), ExitStack() as stack:
        stack.enter_context(patch.object(parent, "logits", logits))
        stack.enter_context(patch.object(parent.head, "decode_array_q24", decode))
        yield
        same(position, len(jobs), "head-only job census incomplete")


def classify(rows):
    selection_gate()
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in rows],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES for l, r in PAIRS for b in BRANCHES],
         "head-only ordered contrast census changed")
    for row in rows:
        target, linear = Fraction(row["target_dot_delta"]), Fraction(row["frozen_hidden_dot_residual"])
        sign = -1 if row["polarity"] == "forward" else 1
        require(target == sign * DOSE * linear and target != 0
                and abs(linear / 32) < abs(target) < abs(linear / 16),
                "exact pre-conversion dose separation failed")
        old, new = Fraction(row["baseline_dot_margin"]), Fraction(row["working_dot_margin"])
        require(Fraction(row["target_dot_margin"]) - old == target
                and new - old == Fraction(row["pre_head_rne_delta"]),
                "exact selected-row dot closure failed")
        old, new, ref = (Fraction(row[k]) for k in (
            "retained_actual_margin", "intervened_actual_margin", "fixed_reference_margin"))
        require(new - old == Fraction(row["observed_delta"])
                and old - ref == Fraction(row["retained_margin_change"])
                and new - ref == Fraction(row["intervened_margin_change"]),
                "fixed original-reference margin closure failed")
        expected = Fraction(-1, 128) if row["polarity"] == "forward" else Fraction(
            1, 64 if row["control"] == "mapped_all" else 128)
        require(Fraction(row["closed_bracket_delta"]) == (
            expected if row["left_id"] == 319 else -expected), "closed comparison pattern drift")
    status = "supported" if all(r["observed_delta"] == r["closed_bracket_delta"] for r in rows) else "rejected"
    return status, HEAD_SUCCESSOR if status == "supported" else NORM_SUCCESSOR


def run_tests(evidence, report, outputs, identity, delta, summaries, pins):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("final_hidden_headonly_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.OUTPUTS = evidence, report, outputs
    module.IDENTITY, module.DELTA, module.SUMMARIES, module.PINS = identity, delta, summaries, pins
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
               "headonly_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0, "final_rmsnorm_invocations": 0, "selected_row_head_invocations": 0}
    with hotspot.read_only(audit), no_closed_replay(audit):
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        comparisons = bind_doses(evidence, summaries)
        delta = tuple(a - b for a, b in zip(
            exact(evidence["references"]["rmsnorm_fp16"].view("<f2")),
            exact(evidence["references"]["rmsnorm_binary64"]), strict=True))
        require(any(delta), "zero frozen final-hidden residual")
        prepared = {(c, p): prepare(evidence["arrays"][c]["rmsnorm"], delta, p)
                    for c in parent.CONTROLS for p in POLARITIES}
    identity = closed.digest(evidence)
    frozen = closed.digest((delta, comparisons, prepared, summaries, PREREGISTRATION, FLAGS))
    rows = np.stack([evidence["rows"][i] for i in IDS])
    jobs = [words for c in parent.CONTROLS for words in (
        evidence["arrays"][c]["rmsnorm"], *(prepared[c, p][0] for p in POLARITIES))]
    contrasts, outputs = [], {}
    residual_dots = dots(delta, rows)
    with head_only(audit, jobs, rows):
        for control in parent.CONTROLS:
            original = evidence["arrays"][control]["rmsnorm"]
            baseline = parent.logits(original, rows)
            closed.same_array(baseline, evidence["arrays"][control]["logits"][list(IDS)],
                              "retained selected-head closure failed")
            baseline_dots = dots(exact(original.view("<f2")), rows)
            for polarity in POLARITIES:
                working, targets, operand = prepared[control, polarity]
                require(not np.array_equal(working, evidence["archives"][control]["stage18"]),
                        "stage18 substituted for final hidden")
                logits = parent.logits(working, rows)
                target_dots, working_dots = dots(targets, rows), dots(exact(working.view("<f2")), rows)
                old_values, new_values = exact(baseline.view("<f2")), exact(logits.view("<f2"))
                outputs[control, polarity] = {
                    "operand": operand, "working_hidden": working, "logits": logits,
                    "baseline_logits": baseline, "baseline_dots": baseline_dots,
                    "target_dots": target_dots, "working_dots": working_dots,
                }
                for left, right in PAIRS:
                    i, j = IDS.index(left), IDS.index(right)
                    old, new = old_values[i] - old_values[j], new_values[i] - new_values[j]
                    before, target, after = (v[i] - v[j] for v in (baseline_dots, target_dots, working_dots))
                    for branch in BRANCHES:
                        reference = evidence["references"]["logits_" + branch][list(IDS)]
                        reference = exact(reference.view("<f2") if branch == "fp16" else reference)
                        ref = reference[i] - reference[j]
                        contrasts.append({
                            "control": control, "polarity": polarity, "left_id": left,
                            "right_id": right, "branch": branch,
                            "frozen_hidden_dot_residual": str(residual_dots[i] - residual_dots[j]),
                            "baseline_row_dots": list(map(str, baseline_dots)),
                            "target_row_dots": list(map(str, target_dots)),
                            "working_row_dots": list(map(str, working_dots)),
                            "baseline_dot_margin": str(before), "target_dot_margin": str(target),
                            "working_dot_margin": str(after), "target_dot_delta": str(target - before),
                            "pre_head_rne_delta": str(after - before),
                            "hidden_rne_margin_remainder": str(after - target),
                            "baseline_head_rne_remainder": str(old - before),
                            "working_head_rne_remainder": str(new - after),
                            "baseline_logit_words": baseline.tolist(), "working_logit_words": logits.tolist(),
                            "baseline_selected_logits": list(map(str, old_values)),
                            "working_selected_logits": list(map(str, new_values)),
                            "retained_actual_margin": str(old), "intervened_actual_margin": str(new),
                            "fixed_reference_margin": str(ref), "retained_margin_change": str(old - ref),
                            "intervened_margin_change": str(new - ref), "observed_delta": str(new - old),
                            "closed_bracket_delta": comparisons[control, polarity, left, right],
                        })
    closed.protect(evidence, identity)
    status, successor = classify(contrasts)
    report = {
        "preregistration": PREREGISTRATION, "classification": status, "successor": successor,
        "contrasts": contrasts, "pattern_agreements": sum(
            r["observed_delta"] == r["closed_bracket_delta"] for r in contrasts),
        "exact_dose_separated_contrasts": len(contrasts),
        "operands": {p: {c: outputs[c, p]["operand"] for c in parent.CONTROLS} for p in POLARITIES},
        "frozen_final_hidden_residual": list(map(str, delta)),
        "frozen_final_hidden_residual_identity": closed.digest(delta),
        "selected_head_row_identity": closed.digest(rows),
        "retained_dose_summaries": dict(zip(
            ("full_forward", "full_reverse", "half", "quarter", "eighth", "sixteenth",
             "thirtysecond", "three_sixtyfourths"), summaries, strict=True)),
        "retained_dose_receipts": list(RECEIPTS), "retained_bracket_call": BRACKET_CALL,
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
        "retained_common_selection_rule": bridge.SELECTION_RULE,
    }
    output_identity = closed.digest((report, outputs))
    with hotspot.read_only(audit), no_closed_replay(audit):
        tests = run_tests(evidence, report, outputs, identity, delta, summaries, pins)
        closed.protect(evidence, identity)
        same(closed.digest((delta, comparisons, prepared, summaries, PREREGISTRATION, FLAGS)),
             frozen, "frozen hidden/operand/receipt/contract drift")
        same(closed.digest((report, outputs)), output_identity, "head-only result drift")
        for pin in (*pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"], *RECEIPTS):
            base.read_bound(pin)
    same(audit, {"forbidden_calls": 0, "final_rmsnorm_invocations": 0,
                 "selected_row_head_invocations": 27}, "head-only dispatch census changed")
    return {
        "diagnostic_id": NAME, "version": 1, "status": status, "command": COMMAND,
        "report": report, "tests": tests, "diagnostic_sources": origins,
        "reviewed_parent_pins": pins, "authenticated_files": evidence["files"],
        "hidden_pins": evidence["hidden_pins"], "assets": evidence["assets"],
        "protected_input_identity": identity,
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
