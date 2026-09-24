"""Read-only cutoff sensitivity of authenticated final-head attempt001 arrays.

--check compiles this source and its focused tests in memory, runs those tests,
and emits JSON only to stdout. No model operator or prior check is rerun.
"""

import argparse
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_impact_census_v1 as base


ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_cutoff_margin_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {base.parent.PYTHON}"
    f" -B -m {MODULE} --check"
)
AUTHENTICATOR_PIN = {
    "path": str(base.SOURCE), "bytes": 19580,
    "sha256": "035da1e497d0bbec6a56d0c1c461928e86cd8d2888ac4d93aaff011dd26d6adb",
}
EXPECTED_TESTS = 26
require, same = base.require, base.same
FLAGS = {
    **base.FLAGS, "token_selected_for_feedback": False,
    "direct_binary64_to_fp16_rounding_claimed": False,
    "upstream_cause_identified": False,
}
BOUNDARY = (
    base.BOUNDARY.replace("cross-control statistics", "top-k cutoff-margin statistics")
    .replace("this new census", "this new cutoff diagnostic")
)


def exact(value):
    return Fraction.from_float(float(value))


def profile(values, k):
    require(isinstance(values, np.ndarray) and values.dtype.str == "<f8"
            and values.ndim == 1 and np.all(np.isfinite(values)),
            "cutoff values must be a finite binary64 vector")
    require(type(k) is int and 0 < k < len(values), "cutoff requires 0 < k < vocabulary")
    order = np.lexsort((np.arange(len(values)), -values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    included, excluded = map(int, order[k - 1:k + 1])

    def entry(index):
        return {"token_id": index, "rank": int(ranks[index]),
                "value_hex": float(values[index]).hex()}

    gap = exact(values[included]) - exact(values[excluded])
    return {
        "values": values, "order": order, "ranks": ranks,
        "summary": {
            "elements": len(values), "all_finite": True,
            "top_k_ids_diagnostic_only": list(map(int, order[:k])),
            "lowest_included": entry(included), "highest_excluded": entry(excluded),
            "cutoff_gap": str(gap), "cutoff_tie": gap == 0,
        },
    }


def token_position(index, ranked, k):
    values, ranks = ranked["values"], ranked["ranks"]
    included = ranked["summary"]["lowest_included"]["token_id"]
    excluded = ranked["summary"]["highest_excluded"]["token_id"]
    inside = bool(ranks[index] <= k)
    neighbor = excluded if inside else included
    return {
        "rank": int(ranks[index]), "included": inside,
        "value_hex": float(values[index]).hex(),
        "margin_to_lowest_included": str(exact(values[index]) - exact(values[included])),
        "margin_to_highest_excluded": str(exact(values[index]) - exact(values[excluded])),
        "nearest_opposite_membership": {
            "token_id": neighbor, "rank": int(ranks[neighbor]),
            "signed_margin": str(exact(values[index]) - exact(values[neighbor])),
        },
    }


def compare_profiles(actual, reference, k):
    a_ids = actual["summary"]["top_k_ids_diagnostic_only"]
    r_ids = reference["summary"]["top_k_ids_diagnostic_only"]
    actual_only, reference_only = sorted(set(a_ids) - set(r_ids)), sorted(set(r_ids) - set(a_ids))
    a, r = actual["values"], reference["values"]
    pairs = []
    for gained in actual_only:
        for lost in reference_only:
            reference_margin = exact(r[lost]) - exact(r[gained])
            actual_margin = exact(a[gained]) - exact(a[lost])
            gained_change = exact(a[gained]) - exact(r[gained])
            lost_change = exact(a[lost]) - exact(r[lost])
            relative_change = gained_change - lost_change
            require(reference_margin >= 0 and actual_margin >= 0
                    and relative_change == reference_margin + actual_margin,
                    "cutoff reversal identity failed")
            pairs.append({
                "actual_only_id": gained, "reference_only_id": lost,
                "reference_preference_margin": str(reference_margin),
                "actual_preference_margin": str(actual_margin),
                "actual_only_value_change": str(gained_change),
                "reference_only_value_change": str(lost_change),
                "relative_value_change": str(relative_change),
                "relative_change_minus_reference_margin": str(relative_change - reference_margin),
                "exact_reversal_identity": True,
                "tie_at_either_boundary": reference_margin == 0 or actual_margin == 0,
                "adjacent_cutoff_in_both": all(
                    {int(p["ranks"][gained]), int(p["ranks"][lost])} == {k, k + 1}
                    for p in (actual, reference)),
                "max_distance_from_cutoff_rank": max(
                    abs(int(p["ranks"][i]) - k)
                    for p in (actual, reference) for i in (gained, lost)),
            })
    return {
        **base.overlap(a_ids, r_ids), "actual_only_ids": actual_only,
        "reference_only_ids": reference_only, "cutoff_reversals": pairs,
    }


def cutoff_control(actual, binary64, fp16, k=10):
    require(actual.shape == binary64.shape == fp16.shape, "cutoff vector shapes differ")
    profiles = {name: profile(values, k) for name, values in (
        ("actual_fp16", actual), ("binary64", binary64), ("fp16", fp16))}
    comparisons = {
        key: compare_profiles(profiles["actual_fp16"], profiles[key], k)
        for key in ("binary64", "fp16")}
    reference_comparison = compare_profiles(profiles["binary64"], profiles["fp16"], k)
    relevant = sorted({
        int(index) for p in profiles.values() for index in p["order"][:k + 1]})
    differing = sorted({
        index for comparison in (*comparisons.values(), reference_comparison)
        for key in ("actual_only_ids", "reference_only_ids") for index in comparison[key]})
    positions = [
        {"token_id": index, **{key: token_position(index, p, k)
                              for key, p in profiles.items()}} for index in relevant]
    pairs = comparisons["fp16"]["cutoff_reversals"]
    observed_pattern = (comparisons["binary64"]["overlap_count"] == k
                        and comparisons["fp16"]["overlap_count"] == k - 1)
    adjacent = bool(pairs) and all(pair["adjacent_cutoff_in_both"] for pair in pairs)
    tied = any(pair["tie_at_either_boundary"] for pair in pairs)
    supported = observed_pattern and adjacent and not tied
    if supported:
        classification = "FINITE_ADJACENT_CUTOFF_REVERSAL"
    elif not observed_pattern:
        classification = "OBSERVED_OVERLAP_PATTERN_NOT_PRESENT"
    elif tied:
        classification = "FINITE_REVERSAL_WITH_TIE_BREAK"
    else:
        classification = "FINITE_REVERSAL_NOT_ADJACENT_TO_CUTOFF"
    return {
        "summaries": {key: p["summary"] for key, p in profiles.items()},
        "comparisons": comparisons, "binary64_vs_fp16_reference": reference_comparison,
        "rank_positions": positions,
        "differing_token_diagnostics": [row for row in positions if row["token_id"] in differing],
        "explanation": {
            "classification": classification,
            "observed_k_vs_k_minus_one_overlap": observed_pattern,
            "finite_near_boundary_explained": supported,
            "near_boundary_definition": "the exchanged IDs occupy ranks k and k+1 in both compared vectors",
            "all_values_finite": True,
            "scope": (
                "Exact retained-logit order geometry only, not an upstream causal diagnosis. "
                "The reviewed FP16 trajectory is independent, not a cast of the binary64 logits."
            ),
        },
    }


def report(result, arrays, references):
    base.check_history(result)
    require(tuple(arrays) == tuple(base.parent.CONTROLS), "diagnostic control census changed")
    rows = []
    for retained in result["controls"]:
        label = retained["control"]
        row = cutoff_control(
            arrays[label]["logits"].view("<f2").astype("<f8"),
            references["logits_binary64"], references["logits_fp16"].view("<f2").astype("<f8"))
        top = retained["comparisons"]["top_k"]
        for key in ("binary64", "fp16"):
            same({field: row["comparisons"][key][field] for field in ("overlap_count", "same_order")},
                 {field: top[key][field] for field in ("overlap_count", "same_order")},
                 "reviewed overlap/order changed")
        actual_top = row["summaries"]["actual_fp16"]["top_k_ids_diagnostic_only"]
        same(actual_top, [entry["token_id"] for entry in top["actual_diagnostic_only"]],
             "reviewed actual diagnostic IDs changed")
        l23 = retained["parent"]["retained_L23"]
        rows.append({
            "control": label, **row,
            "L21_L22_L23_status": ["FAIL"] * 3,
            "retained_failures": {
                "L21": l23["retained_L21"]["failures"],
                "L22": [l23["L22_failure"]], "L23": retained["parent"]["L23_failures"],
            },
            "retained_L23_and_ancestral_lineage": l23,
        })
    return {
        "control_count": len(rows), "controls": rows, "k": 10,
        "order": "descending finite value, ascending numeric diagnostic ID on ties",
        "margin_units": "exact rational logit differences, not decoder thresholds or ULP budgets",
        "rank_positions_scope": "union of each vector's top k and first excluded ID; ranks are 1-based",
        "margin_semantics": (
            "Signed token value minus the named cutoff neighbor. Nearest opposite membership "
            "is the highest excluded for an included ID, or lowest included for an excluded ID."
        ),
        "finite_near_boundary_explained_control_count": sum(
            row["explanation"]["finite_near_boundary_explained"] for row in rows),
        "retained_L23_failures": 9,
        "retained_L23_stage_reports": result["preflight"]["retained_L23_stage_reports"],
        "thresholds": result["preflight"]["thresholds"],
        "original_global_reference": result["preflight"]["final_reference"],
        "retained_flags": result["flags"],
    }


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(base.parent.record(path))
    spec = importlib.util.spec_from_file_location("final_head_cutoff_margin_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "focused cutoff tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors),
            "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == base.parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    base.read_bound(AUTHENTICATOR_PIN)
    origins = {**base.parent.source_context(), MODULE: base.parent.record(SOURCE),
               "cutoff_focused_test": base.parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with base.read_only(audit):
        result, arrays, references, files = base.authenticate()
        measured = report(result, arrays, references)
        tests = focused_tests((result, arrays, references, measured))
        for pin in origins.values():
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "cutoff diagnostic attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1, "status": "READ_ONLY_CUTOFF_MARGIN_DIAGNOSTIC",
        "command": COMMAND, "pins": {**base.PINS, "authenticator": AUTHENTICATOR_PIN},
        "original_execution_sources": base.parent.RETAINED_SOURCES,
        "authenticated_files": files, "diagnostic_sources": origins,
        "input_terminal_review": {
            "mission_id": base.MISSION, "round": 2, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done",
        },
        "authentication_scope": (
            "Pinned reviewed result, retained original execution source/test identities, "
            "live reviewed validator/test and other origins; all 20 attempt001 files, "
            "18 retained arrays, four original-input final reference arrays and their "
            "manifest/input pins, retained L21/L22/L23 state/gate archives. Operand and "
            "ancestral gates are authenticated retained facts, never replayed."
        ),
        "controls_verified": 9, "attempt001_files_verified": 20,
        "arrays_verified": 18, "reference_arrays_verified": 4,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        **measured,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
