"""Retained selected-row midpoint attribution; stdout-only CPU non-admission."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_scalar_rne_threshold_classifier_v1 as scalar


base, parent, closed, bridge = scalar.base, scalar.parent, scalar.closed, scalar.bridge
require, same, ROOT = base.require, base.same, base.ROOT
NAME = "diagnose_q24_s16_terminal_remainder_selected_head_rne_crossing_attribution_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
IDS, PAIRS, POLARITIES, BRANCHES = scalar.IDS, scalar.PAIRS, scalar.POLARITIES, scalar.BRANCHES
DOSE, EXPECTED_TESTS = Fraction(3, 64), 12
MISSION = "8e5ec50c8f6b"
PINS = (
    {"path": str(scalar.SOURCE), "sha256": "7b1db7fbd94b840a716fe00abc91696f8cb0d15a9f8683ad4cc6d88951b09bab"},
    {"path": str(scalar.TEST), "sha256": "15c9402cc195b673b95bea5702ac66c76f5ca07b07508383b9471ffb2e69d123"},
    {"path": str(closed.HANDOFFS / MISSION / "round-0001.json"),
     "sha256": "6e66f7537281b80620be8ac30504164233c6b5f56a903141edf5af891a1e1b8f"},
)
RECEIPT = {
    "path": "/home/argustest/.argus-skill-ace3/copilot-home/session-state/"
            "f8194436-d8f1-4de0-abf4-1366a14f59ce/events.jsonl",
    "sha256": "f93fc729632d06a46ebc2442eaea860585dba6b13bcd5325ad3100e43bff40eb",
}
CALL = "call_ogKxy7s4Y9ZfEV2F7j4rxYBr"
FIELDS = ("control", "polarity", "row_id", "baseline_exact", "baseline_word",
          "target_exact", "target_word", "working_exact", "working_word")
SINGLE = "single-row RNE threshold localization"
COUPLED = "two-row coupled threshold localization"
PREREGISTRATION = {
    "operand": "Only authenticated retained individual baseline/exact-target/working "
               "accumulator scalars and FP16 words for rows 319 and 34319. "
               "No scalar recovery, hidden construction, row dots or operator replay.",
    "prediction": "Every reproduced nonzero rounded-margin contrast is attributable "
                  "to one selected row crossing while its paired row stays non-crossing.",
    "single_row_in_every_contrast": SINGLE,
    "any_two_row_crossing_or_opposite_compensation": COUPLED,
    "authentication_midpoint_or_contrast_failure": "UNKNOWN/integrity; no scientific successor",
    "precedence": "Exact-dose branch first; retain working accounts separately, never "
                  "attribute the margin by rounding the row difference.",
    "boundary": scalar.PREREGISTRATION["boundary"] + " Retained scalar midpoint "
                "accounting only; no new intervention, repair or production action.",
}
FLAGS = {**scalar.FLAGS, "closed_scalar_diagnostic_replay": False,
         "selected_row_accumulator_recomputation": False, "production_claim": False}


def selection_gate():
    same((IDS, PAIRS, POLARITIES, BRANCHES),
         ((319, 34319), ((319, 34319), (34319, 319)),
          ("forward", "reverse"), ("fp16", "binary64")), "crossing selection changed")
    require(type(DOSE) is Fraction and DOSE == Fraction(3, 64), "crossing dose changed")


@contextmanager
def read_only(audit):
    def refuse(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("crossing attribution forbids scalar/hidden/dot replay")

    with scalar.read_only(audit), ExitStack() as stack:
        for module, names in (
            (scalar, ("check", "run_tests", "measure", "accumulate", "rne", "classify")),
            (scalar.retained, ("dots", "prepare")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, refuse))
        yield


def authenticate_receipt():
    for pin in (*PINS, RECEIPT):
        base.read_bound(pin)
    review = json.loads(base.read_bound(PINS[-1]))
    same([review[k] for k in ("kind", "mission_id", "producer_role", "round")],
         ["round_reviewed_handoff", MISSION, "reviewer", 1], "scalar review identity changed")
    same(review["review"]["status"], "done", "scalar independent review incomplete")
    latest_pin = parent.record(Path(PINS[-1]["path"]).with_name("latest.json"))
    latest = json.loads(base.read_bound(latest_pin))
    same((latest["kind"], latest["handoff"]["path"]),
         ("handoff_ref", PINS[-1]["path"]), "scalar review lineage changed")
    events = (json.loads(line) for line in base.read_bound(RECEIPT).splitlines())
    matches = [e["data"] for e in events if e["type"] == "tool.execution_complete"
               and e["data"]["toolCallId"] == CALL]
    same(len(matches), 1, "scalar receipt missing or duplicated")
    require(matches[0]["success"] is True, "scalar receipt tool failed")
    content, marker = matches[0]["result"]["content"], '{"classification":'
    same(content.count(marker), 1, "scalar receipt JSON missing or duplicated")
    summary, _ = json.JSONDecoder().raw_decode(content[content.index(marker):])
    return summary, [*PINS, RECEIPT, latest_pin]


def bind_receipt(summary, evidence, chain_pins):
    same([summary[k] for k in ("status", "classification", "native_exit", "stdout_json_documents",
                              "contrast_count", "target_pattern_agreements", "working_pattern_agreements",
                              "successor", "native_command", "reviewed_parent_pin_count")],
         ["supported", "supported", 0, 1, 72, 72, 72, scalar.EXACT_SUCCESSOR,
          scalar.COMMAND, len(chain_pins)], "reviewed scalar outcome changed")
    same(summary["protected_input_identity"], closed.digest(evidence),
         "scalar source/token/Q24/state/KV/lineage/reference/head-row identity drift")
    same([summary["tests"][k] for k in ("executed", "errors", "failures", "skipped")],
         [12, 0, 0, 0], "reviewed scalar tests changed")
    same(summary["tests"]["compiled"], [parent.record(scalar.SOURCE), parent.record(scalar.TEST)],
         "reviewed scalar source/test drift")
    expected_audit = dict.fromkeys((
        "forbidden_calls", "final_rmsnorm_invocations", "selected_row_head_invocations",
        "prefix_replays", "admission_replays", "reference_producer_replays",
        "full_vocabulary_replays", "closed_stage18_replays", "closed_headonly_replays",
        "writes", "artifact_overwrites"), 0)
    expected_audit.update(exact_selected_row_accumulations=92, hidden_rne_scalars=16128,
                          accumulator_rne_scalars=90)
    same(summary["dispatch_and_write_audit"], expected_audit, "reviewed scalar budget drift")
    same(summary["scalar_fields"], list(FIELDS), "retained scalar schema changed")
    values = summary["scalar_values"]
    require(type(values) is list and len(values) == 36
            and all(type(v) is list and len(v) == len(FIELDS) for v in values),
            "retained scalar shape changed")
    rows = [dict(zip(FIELDS, v, strict=True)) for v in values]
    same([(r["control"], r["polarity"], r["row_id"]) for r in rows],
         [(c, p, i) for c in parent.CONTROLS for p in POLARITIES for i in IDS],
         "retained scalar census changed")
    return rows


def rational(text):
    require(type(text) is str, "non-string retained rational")
    value = Fraction(text)
    require(str(value) == text and value.denominator & (value.denominator - 1) == 0,
            "noncanonical or nondyadic retained scalar")
    return value


def decode(word):
    require(type(word) is int and 0 <= word <= 0xffff
            and (word >> 10) & 31 != 31, "invalid finite FP16 word")
    magnitude = word & 0x7fff
    exponent, mantissa = magnitude >> 10, magnitude & 1023
    shift = exponent - 25 if exponent else -24
    integer = mantissa + (1024 if exponent else 0)
    value = Fraction(integer << shift) if shift >= 0 else Fraction(integer, 1 << -shift)
    return -value if word & 0x8000 else value


def cell(value, word):
    center = decode(word)
    magnitude = word & 0x7fff
    below = decode(magnitude - 1) if magnitude else -decode(1)
    above = decode(magnitude + 1) if magnitude < 0x7bff else Fraction(65536)
    if word & 0x8000:
        below, above = -above, -below
    lower, upper = (below + center) / 2, (center + above) / 2
    even = word & 1 == 0
    require(lower <= value <= upper and (even or lower < value < upper),
            "retained scalar midpoint/RNE closure failed")
    require((word & 0x8000 != 0) == (value < 0), "retained scalar sign/zero closure failed")
    return {"exact": str(value), "fp16_word": word, "rounded": str(center),
            "lower_midpoint": str(lower), "upper_midpoint": str(upper),
            "lower_distance": str(value - lower), "upper_distance": str(upper - value),
            "even_significand": even, "tie": value in (lower, upper),
            "tie_decision": "retained_even_word" if value in (lower, upper) else "not_at_midpoint",
            "rne_remainder": str(center - value)}


def movement(baseline, endpoint):
    start, end = rational(baseline["exact"]), rational(endpoint["exact"])
    delta = end - start
    direction = (delta > 0) - (delta < 0)
    word_delta = decode(endpoint["fp16_word"]) - decode(baseline["fp16_word"])
    threshold = rational(baseline["upper_midpoint" if direction >= 0 else "lower_midpoint"])
    initial_slack = direction * (threshold - start)
    remaining_slack = direction * (threshold - end)
    crossed = endpoint["fp16_word"] != baseline["fp16_word"]
    require(not crossed or direction * word_delta > 0, "nonmonotonic RNE row movement")
    expected_crossing = bool(direction and (
        remaining_slack < 0 or (remaining_slack == 0 and not baseline["even_significand"])))
    require(crossed == expected_crossing, "row threshold/crossing closure failed")
    require(initial_slack - abs(delta) == remaining_slack, "row slack closure failed")
    return {"signed_movement": str(delta), "direction": direction,
            "individual_rounded_delta": str(word_delta), "word_changed": crossed,
            "threshold": str(threshold) if direction else None,
            "initial_slack": str(initial_slack), "remaining_slack": str(remaining_slack),
            "signed_overshoot": str(-remaining_slack), "slack_closure_residual": "0",
            "endpoint_at_baseline_midpoint": bool(direction and remaining_slack == 0),
            "baseline_midpoint_tie_winner": "baseline" if baseline["even_significand"] else "neighbor",
            "word_change": endpoint["fp16_word"] - baseline["fp16_word"]}


def attribute(rows, evidence, comparisons, head_rows):
    selection_gate()
    accounts, lookup, slopes, baselines = [], {}, {}, {}
    for row in rows:
        c, p, i = (row[k] for k in FIELDS[:3])
        cells = {name: cell(rational(row[name + "_exact"]), row[name + "_word"])
                 for name in ("baseline", "target", "working")}
        same(row["baseline_word"], int(evidence["arrays"][c]["logits"][i]),
             "individual baseline retained-logit closure failed")
        baseline = (row["baseline_exact"], row["baseline_word"])
        if (c, i) in baselines:
            same(baseline, baselines[c, i], "polarity baseline changed")
        baselines[c, i] = baseline
        target = movement(cells["baseline"], cells["target"])
        working = movement(cells["baseline"], cells["working"])
        signed_dose = -DOSE if p == "forward" else DOSE
        slope = rational(target["signed_movement"]) / signed_dose
        require(slope != 0, "zero retained exact-dose movement")
        if i in slopes:
            require(slope == slopes[i], "frozen exact-dose row slope changed")
        slopes[i] = slope
        account = {**row, "cells": cells, "target": target, "working": working,
                   "signed_dose": str(signed_dose), "frozen_row_dose_slope": str(slope),
                   "hidden_rne_dot_remainder": str(rational(row["working_exact"]) -
                                                 rational(row["target_exact"]))}
        accounts.append(account)
        lookup[c, p, i] = account
    old_head = {(r["control"], r["polarity"], r["left_id"], r["right_id"]): r for r in head_rows}
    contrasts = []
    for c in parent.CONTROLS:
        for p in POLARITIES:
            for left, right in PAIRS:
                lrow, rrow = lookup[c, p, left], lookup[c, p, right]
                margins = {name: decode(lrow[name + "_word"]) - decode(rrow[name + "_word"])
                           for name in ("baseline", "target", "working")}
                previous = old_head[c, p, left, right]
                for name, field in (("target", "target_dot_delta"), ("working", "pre_head_rne_delta")):
                    exact_delta = (rational(lrow[name + "_exact"]) - rational(rrow[name + "_exact"])
                                   - rational(lrow["baseline_exact"]) + rational(rrow["baseline_exact"]))
                    same(str(exact_delta), previous[field], "retained exact dot-margin closure failed")
                for branch in BRANCHES:
                    refs = evidence["references"]["logits_" + branch]
                    ref = (decode(int(refs[left])) - decode(int(refs[right])) if branch == "fp16"
                           else Fraction(float(refs[left])) - Fraction(float(refs[right])))
                    record = {"control": c, "polarity": p, "left_id": left, "right_id": right,
                              "branch": branch, "fixed_reference_margin": str(ref),
                              "retained_margin_change": str(margins["baseline"] - ref),
                              "baseline_rounded_margin": str(margins["baseline"]),
                              "closed_bracket_delta": comparisons[c, p, left, right]}
                    for name in ("target", "working"):
                        ld, rd = (rational(r[name]["individual_rounded_delta"]) for r in (lrow, rrow))
                        delta = margins[name] - margins["baseline"]
                        require(ld - rd == delta, "individual row-to-margin closure failed")
                        same(str(delta), record["closed_bracket_delta"], "rounded contrast closure failed")
                        record[name] = {
                            "rounded_margin": str(margins[name]), "delta": str(delta),
                            "margin_change": str(margins[name] - ref),
                            "left_contribution": str(ld), "right_contribution": str(-rd),
                            "crossing_rows": [i for i, r in ((left, lrow), (right, rrow))
                                              if r[name]["word_changed"]],
                            "opposite_row_compensation": ld * -rd < 0,
                            "row_closure_residual": "0", "reference_closure_residual": "0",
                        }
                    if branch == "binary64":
                        same(record["retained_margin_change"], previous["retained_margin_change"],
                             "original reference reanchored")
                        same(record["working"]["margin_change"], previous["intervened_margin_change"],
                             "working original-reference closure failed")
                    contrasts.append(record)
    return {"rows": accounts, "contrasts": contrasts,
            "individual_midpoint_closures": len(accounts) * 3,
            "individual_slack_closures": len(accounts) * 2, **classify(contrasts)}


def classify(contrasts):
    same([(r["control"], r["polarity"], r["left_id"], r["right_id"], r["branch"]) for r in contrasts],
         [(c, p, l, r, b) for c in parent.CONTROLS for p in POLARITIES
          for l, r in PAIRS for b in BRANCHES], "crossing contrast census changed")
    single, coupled, compensation = 0, 0, 0
    for row in contrasts:
        for name in ("target", "working"):
            branch = row[name]
            left, right = (rational(branch[k]) for k in ("left_contribution", "right_contribution"))
            expected_rows = [i for i, v in ((row["left_id"], left), (row["right_id"], right)) if v]
            same(branch["crossing_rows"], expected_rows, "crossing row attribution changed")
            require(left + right == rational(branch["delta"]) != 0, "row contribution closure failed")
            same(branch["delta"], row["closed_bracket_delta"], "contrast pattern changed")
            require(rational(branch["rounded_margin"]) - rational(row["baseline_rounded_margin"])
                    == rational(branch["delta"]), "rounded margin changed")
            require(rational(branch["margin_change"]) == rational(branch["rounded_margin"])
                    - rational(row["fixed_reference_margin"]), "global-reference closure changed")
            require(rational(row["retained_margin_change"]) == rational(row["baseline_rounded_margin"])
                    - rational(row["fixed_reference_margin"]), "retained global reference changed")
            same(branch["opposite_row_compensation"], left * right < 0, "compensation flag changed")
        chosen = row["target"]
        count = len(chosen["crossing_rows"])
        single += count == 1
        coupled += count == 2
        compensation += chosen["opposite_row_compensation"]
    require(single + coupled == 72, "incomplete crossing attribution")
    return {"classification": "supported", "successor": COUPLED if coupled or compensation else SINGLE,
            "single_row_prediction": "rejected" if coupled or compensation else "supported",
            "selected_branch": "exact_target", "contrast_count": 72,
            "target_contrast_closures": 72, "working_contrast_closures": 72,
            "single_row_contrasts": single, "two_row_contrasts": coupled,
            "opposite_compensation_contrasts": compensation, "integrity_error": None}


def run_tests(evidence, report, summary, rows, comparisons, head_rows, pins):
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    spec = importlib.util.spec_from_file_location("crossing_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.REPORT, module.SUMMARY = evidence, report, summary
    module.ROWS, module.COMPARISONS, module.HEAD_ROWS, module.PINS = rows, comparisons, head_rows, pins
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
    selection_gate()
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE), "crossing_tests": parent.record(TEST)}
    audit = dict.fromkeys((
        "forbidden_calls", "prefix_replays", "admission_replays", "reference_producer_replays",
        "final_rmsnorm_invocations", "selected_row_head_invocations", "full_vocabulary_replays",
        "closed_stage18_replays", "closed_headonly_replays", "closed_scalar_diagnostic_replays",
        "exact_selected_row_accumulations", "hidden_rne_scalars", "writes", "artifact_overwrites"), 0)
    with read_only(audit):
        summary, pins = authenticate_receipt()
        chain_pins, head_summary = scalar.authenticate_parent()
        summaries = scalar.retained.retained_doses()
        for pin in bridge.margin.rows.PINS.values():
            base.read_bound(pin)
        evidence = bridge.hidden.authenticate()
        evidence["rows"] = bridge.margin.contributions.load_rows(evidence["assets"], IDS)
        base.check_history(evidence["result"])
        comparisons = scalar.retained.bind_doses(evidence, summaries)
        head_rows = scalar.bind_parent(evidence, head_summary, comparisons)
        rows = bind_receipt(summary, evidence, chain_pins)
        identity = closed.digest(evidence)
        frozen = closed.digest((summary, summaries, head_summary, rows, comparisons, head_rows,
                                PREREGISTRATION, FLAGS))
        report = attribute(rows, evidence, comparisons, head_rows)
        report_identity = closed.digest(report)
        tests = run_tests(evidence, report, summary, rows, comparisons, head_rows, chain_pins)
        closed.protect(evidence, identity)
        same(closed.digest(report), report_identity, "crossing result mutation")
        same(closed.digest((summary, summaries, head_summary, rows, comparisons, head_rows,
                            PREREGISTRATION, FLAGS)), frozen, "retained evidence/contract mutation")
        for pin in (*pins, *chain_pins, *origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *evidence["hidden_pins"],
                    *scalar.retained.RECEIPTS, scalar.PARENT_RECEIPT):
            base.read_bound(pin)
    same(audit, dict.fromkeys(audit, 0), "forbidden dispatch/write census changed")
    return {
        "diagnostic_id": NAME, "version": 1, "status": report["classification"], "command": COMMAND,
        "report": report, "preregistration": PREREGISTRATION, "tests": tests,
        "retained_scalar_receipt": summary, "reviewed_parent_pins": [*pins, *chain_pins],
        "diagnostic_sources": origins, "protected_input_identity": identity,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"], "selected_head_row_identity": closed.digest(evidence["rows"]),
        "retained_dose_receipts": [*scalar.retained.RECEIPTS, scalar.PARENT_RECEIPT],
        "retained_dose_summaries": summaries, "retained_headonly_summary": head_summary,
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "retained_common_component": "UNKNOWN", "counterfactual_common_component": "UNKNOWN",
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
