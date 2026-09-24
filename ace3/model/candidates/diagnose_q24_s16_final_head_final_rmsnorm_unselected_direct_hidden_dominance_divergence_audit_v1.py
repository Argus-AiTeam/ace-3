"""Stdout-only exact dominance divergence over the authenticated gate census."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_gate_obstruction_census_v1 as census


bridge, base, parent = census.bridge, census.base, census.parent
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_dominance_divergence_audit_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
CONTROLS, BRANCHES, COMPONENTS = census.CONTROLS, census.BRANCHES, census.COMPONENTS
MLP, TERMINAL = "mlp_stage17", "fp16_to_branch_terminal_remainder"
EXPECTED_TESTS = 15
FLAGS = {**census.FLAGS, "dominance_divergence_causal_allocation": False}
BOUNDARY = (
    "Exact row-level dominance means ordering absolute NET SIGNED corrections, "
    "not absolute coordinate mass, an upstream cause, a dominant operator or "
    "a performance bottleneck. Signed magnitude gaps are |component|-|anchor|; "
    "positive means larger, zero means equal, and negative means smaller. "
    "Partitions preserve overlapping pair/control/branch identities and do not "
    "count independent samples. This audit consumes the existing authenticated "
    "in-memory bridge/census interface, preserves the full census including "
    "nonshared pairs, and makes no new split or counterfactual execution. "
    + census.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("dominance audit forbids predecessor checks and test replay")

    with census.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests"):
            stack.enter_context(patch.object(census, name, forbidden))
        yield


def row_dominance(row):
    values = {k: Fraction(row["component_totals"][k]["signed"]) for k in COMPONENTS}
    magnitudes = {k: abs(v) for k, v in values.items()}
    maximum = max(magnitudes.values())
    winners = [k for k in COMPONENTS if magnitudes[k] == maximum]
    return {
        "control": row["control"], "branch": row["branch"],
        "largest_components": winners,
        "unique_largest_component": winners[0] if len(winners) == 1 else "UNKNOWN",
        "largest_net_signed_magnitude": str(maximum),
        "mlp_minus_terminal_net_magnitude": str(magnitudes[MLP]-magnitudes[TERMINAL]),
        "components": [{
            "component": k, "signed_correction": str(values[k]),
            "net_signed_magnitude": str(magnitudes[k]),
            "magnitude_gap_against_mlp_stage17": str(magnitudes[k]-magnitudes[MLP]),
            "magnitude_gap_against_terminal_remainder": str(magnitudes[k]-magnitudes[TERMINAL]),
            "larger_components": [other for other in COMPONENTS if magnitudes[other] > magnitudes[k]],
            "equal_net_magnitude_components": [
                other for other in COMPONENTS if other != k and magnitudes[other] == magnitudes[k]],
            "unique_largest_net_signed_magnitude": len(winners) == 1 and winners[0] == k,
        } for k in COMPONENTS],
    }


def partitions(rows):
    return {
        "by_branch": [{
            "branch": branch,
            "controls_by_unique_largest_component": {
                k: [r["control"] for r in rows if r["branch"] == branch
                    and r["unique_largest_component"] == k] for k in COMPONENTS},
            "tied_controls": [r["control"] for r in rows if r["branch"] == branch
                              and r["unique_largest_component"] == "UNKNOWN"],
        } for branch in BRANCHES],
        "by_control": [{
            "control": control,
            "largest_components_by_branch": {
                r["branch"]: r["largest_components"] for r in rows if r["control"] == control},
            "branch_winners_differ": len({
                tuple(r["largest_components"]) for r in rows if r["control"] == control}) > 1,
        } for control in CONTROLS],
    }


def audit_census(observed, bridge_report):
    # Bind every supplied census row and stored gate to the authenticated report.
    same(observed, census.census(bridge_report), "dominance census/report binding changed")
    pairs, nonshared = [], []
    for pair in observed["pairs"]:
        identity = {"left_id": pair["left_id"], "right_id": pair["right_id"]}
        if not pair["complete_nine_control_two_branch_census"]:
            nonshared.append(identity)
            continue
        rows = []
        for index, source in enumerate(pair["exact_accounting_rows"]):
            row = row_dominance(source)
            for component, evaluation in zip(row["components"], pair["component_evaluations"], strict=True):
                gate = evaluation["rows"][index]
                same(component["component"], evaluation["component"], "component order changed")
                for key in ("signed_correction", "net_signed_magnitude", "larger_components",
                            "equal_net_magnitude_components", "unique_largest_net_signed_magnitude"):
                    same(component[key], gate[key], "dominance/gate divergence: " + key)
                component["retained_gate_row"] = gate
            rows.append(row)
        failures, zeros = [], []
        for row in rows:
            components = {c["component"]: c for c in row["components"]}
            if not components[MLP]["unique_largest_net_signed_magnitude"]:
                failures.append({
                    "control": row["control"], "branch": row["branch"],
                    "largest_components": row["largest_components"],
                    "mlp_minus_terminal_net_magnitude": row["mlp_minus_terminal_net_magnitude"],
                    **components[MLP],
                })
            if row["branch"] == "fp16" and components[TERMINAL]["signed_correction"] == "0":
                zeros.append({"control": row["control"], "branch": row["branch"],
                              **components[TERMINAL]})
        pairs.append({
            **identity, "component": pair["component"],
            "stop_nested_bridge_expansion": pair["stop_nested_bridge_expansion"],
            "row_count": len(rows), "component_row_count": len(rows)*len(COMPONENTS),
            "census_proof": {k: pair[k] for k in (
                "complete_nine_control_two_branch_census", "expected_row_count", "observed_row_count",
                "missing_rows", "unexpected_rows", "duplicate_rows", "canonical_row_order")},
            "largest_tie_rows": [{"control": r["control"], "branch": r["branch"],
                                 "largest_components": r["largest_components"]}
                                for r in rows if len(r["largest_components"]) != 1],
            "equal_magnitude_component_row_count": sum(
                bool(c["equal_net_magnitude_components"]) for r in rows for c in r["components"]),
            "rows": rows, "dominance_partitions": partitions(rows),
            "mlp_stage17_unique_largest_failure_count": len(failures),
            "mlp_stage17_unique_largest_failure_rows": failures,
            "mlp_stage17_failure_partitions": {
                "by_branch": [{"branch": b, "controls": [
                    r["control"] for r in failures if r["branch"] == b]} for b in BRANCHES],
                "by_control": [{"control": c, "branches": [
                    r["branch"] for r in failures if r["control"] == c]} for c in CONTROLS],
            },
            "terminal_fp16_zero_boundary_rows": zeros,
        })
    same(len(pairs), observed["shared_pair_count"], "shared pair census changed")
    same(len(nonshared), observed["nonshared_pair_count"], "nonshared pair census changed")
    return {
        "gap_convention": "|component|-|anchor|, using net signed corrections, not absolute accounts",
        "common_component": observed["common_component"],
        "stop_nested_bridge_expansion": observed["stop_nested_bridge_expansion"],
        "shared_pair_count": len(pairs), "nonshared_pair_count": len(nonshared),
        "shared_row_count": sum(p["row_count"] for p in pairs),
        "shared_component_row_count": sum(p["component_row_count"] for p in pairs),
        "all_shared_censuses_complete": all(
            p["census_proof"]["complete_nine_control_two_branch_census"] for p in pairs),
        "shared_largest_tie_row_count": sum(len(p["largest_tie_rows"]) for p in pairs),
        "shared_equal_magnitude_component_row_count": sum(
            p["equal_magnitude_component_row_count"] for p in pairs),
        "pairs": pairs, "nonshared_pair_identities": nonshared,
        "retained_obstruction_census": observed,
    }


def run_tests(evidence, observed, report):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("dominance_divergence_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.CENSUS, module.OBSERVED = evidence, observed, report
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "dominance audit tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "dominance_divergence_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with bridge.read_only(audit):
        evidence = bridge.measure()
    with read_only(audit):
        observed = census.census(evidence["report"])
        report = audit_census(observed, evidence["report"])
        tests = run_tests(evidence, observed, report)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "dominance audit attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_UNSELECTED_DIRECT_HIDDEN_DOMINANCE_DIVERGENCE_AUDIT",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "existing in-memory authenticated bridge/census reports; rational accounting only",
        "report": report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
