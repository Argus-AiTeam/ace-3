"""Stdout-only retained-coordinate MLP-stage17 audit; no operator replay."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_terminal_remainder_coordinate_hotspot_audit_v1 as terminal


bridge, census, dominance = terminal.bridge, terminal.census, terminal.dominance
base, parent = terminal.base, terminal.parent
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_coordinate_hotspot_audit_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
PAIR = (34319, 13)
MLP, TERMINAL = dominance.MLP, dominance.TERMINAL
EXPECTED_TESTS = 14
FLAGS = {**dominance.FLAGS, "mlp_stage17_coordinate_causal_allocation": False}
BOUNDARY = (
    "Descending absolute MLP-stage17 coordinate contributions are retained "
    "accounting only, not a new component split, causality, a dominant operator "
    "or performance attribution. Every unchanged complement coordinate is "
    "included; ties use ascending coordinate ID. Coordinate magnitude gaps "
    "compare MLP and terminal contributions at the SAME coordinate, not net "
    "component magnitudes. FP16 terminal zeros are preserved, not hotspots. "
    "No binary64 internal stage is reconstructed. Middle-pair localization "
    "does not override the common UNKNOWN or authorize nested expansion. "
    + dominance.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("MLP coordinate audit forbids predecessor checks and replay")

    with terminal.read_only(audit), \
            patch.object(terminal, "check", forbidden), \
            patch.object(terminal, "run_tests", forbidden), \
            patch.object(terminal, "audit_coordinates", forbidden):
        yield


def coordinate_order(coordinates):
    ordered = sorted(coordinates, key=lambda r: (
        -Fraction(r["mlp_stage17_absolute_contribution"]), r["coordinate"]))
    groups = {}
    for row in ordered:
        groups.setdefault(row["mlp_stage17_absolute_contribution"], []).append(row["coordinate"])
    nonzero = [r for r in ordered if Fraction(r["mlp_stage17_absolute_contribution"])]
    return {
        "mlp_stage17_absolute_coordinate_order": [r["coordinate"] for r in ordered],
        "top_mlp_stage17_coordinates": nonzero[:8],
        "top_mlp_stage17_magnitude_ties": (
            groups[nonzero[0]["mlp_stage17_absolute_contribution"]] if nonzero else []),
        "mlp_stage17_magnitude_tie_groups": [
            {"absolute_contribution": value, "coordinates": ids}
            for value, ids in groups.items() if len(ids) > 1],
        "nonzero_mlp_stage17_coordinate_count": len(nonzero),
        "mlp_stage17_hotspot_order_meaningful": bool(nonzero),
    }


def coordinate_row(evidence, reference, rows, pair, branch):
    same((pair["left_id"], pair["right_id"]), PAIR, "middle pair identity changed")
    require(branch in census.BRANCHES, "unsupported reference branch")
    gate = pair["branches"][branch]["gate_values"]
    require(gate["control"] in census.CONTROLS and gate["branch"] == branch,
            "coordinate control/branch identity changed")
    row = terminal.coordinate_row(evidence, reference, rows, pair, branch)
    for coordinate in row["coordinates"]:
        coordinate["mlp_minus_terminal_coordinate_magnitude"] = str(
            Fraction(coordinate["mlp_stage17_absolute_contribution"])
            - Fraction(coordinate["terminal_absolute_contribution"]))
    row.update(coordinate_order(row["coordinates"]))
    row["terminal_remainder_comparison_meaningful"] = row["terminal_hotspot_order_meaningful"]
    row["terminal_remainder_comparison_rows"] = (
        row["top_mlp_stage17_coordinates"] if row["terminal_hotspot_order_meaningful"] else [])
    row["coordinate_closure_residuals"] = {
        component: {
            field: str(Fraction(row["component_totals"][component][field])
                       - Fraction(gate["component_totals"][component][field]))
            for field in ("signed", "absolute", "cancellation_absolute_mass")}
        for component in (MLP, TERMINAL)}
    return row


def audit_coordinates(evidence, observed_census, observed_dominance):
    base.check_history(evidence["result"])
    same(observed_dominance, dominance.audit_census(observed_census, evidence["report"]),
         "MLP audit dominance input binding changed")
    reference, rows = terminal.bound_vectors(evidence)
    matches = [p for p in observed_dominance["pairs"]
               if (p["left_id"], p["right_id"]) == PAIR]
    same(len(matches), 1, "middle pair census missing or duplicated")
    retained = matches[0]
    require(retained["census_proof"]["complete_nine_control_two_branch_census"],
            "incomplete middle pair census")
    same(retained["component"], MLP, "middle pair component gate changed")
    same(retained["stop_nested_bridge_expansion"], False, "middle pair stop gate changed")
    for row in retained["rows"]:
        same(row["unique_largest_component"], MLP, "middle pair unique-largest component changed")
        require(Fraction(row["mlp_minus_terminal_net_magnitude"]) > 0,
                "MLP no longer strictly exceeds terminal net magnitude")
    accounts = []
    for control in evidence["report"]["controls"]:
        pairs = [p for p in control["pairs"] if (p["left_id"], p["right_id"]) == PAIR]
        same(len(pairs), 1, "middle pair control census missing or duplicated")
        accounts.extend(coordinate_row(evidence, reference, rows, pairs[0], b)
                        for b in census.BRANCHES)
    same([(r["control"], r["branch"]) for r in accounts],
         [(c, b) for c in census.CONTROLS for b in census.BRANCHES],
         "MLP coordinate row census changed")
    same(observed_dominance["common_component"], "UNKNOWN", "common gate changed")
    same(observed_dominance["stop_nested_bridge_expansion"], True, "common stop gate changed")
    return {
        "pairs": [{"left_id": PAIR[0], "right_id": PAIR[1], "rows": accounts,
                   "component": retained["component"],
                   "stop_nested_bridge_expansion": retained["stop_nested_bridge_expansion"]}],
        "pair_count": 1, "row_count": len(accounts),
        "coordinate_order_rule": "descending absolute MLP-stage17 contribution, then ascending coordinate ID",
        "coordinate_gap_rule": "|mlp_stage17 contribution at i|-|terminal contribution at i|",
        "common_component": observed_dominance["common_component"],
        "stop_nested_bridge_expansion": observed_dominance["stop_nested_bridge_expansion"],
        "retained_bridge_selection": evidence["report"]["selection"],
        "retained_dominance_audit": observed_dominance,
        "prior_pair_and_common_gates_unchanged": True,
    }


def run_tests(evidence, observed_census, observed_dominance, report):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("mlp_stage17_coordinate_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.CENSUS = evidence, observed_census
    module.DOMINANCE, module.OBSERVED = observed_dominance, report
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "MLP coordinate tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               terminal.MODULE: parent.record(terminal.SOURCE),
               "mlp_stage17_coordinate_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with bridge.read_only(audit):
        evidence = bridge.measure()
    with read_only(audit):
        observed_census = census.census(evidence["report"])
        observed_dominance = dominance.audit_census(observed_census, evidence["report"])
        report = audit_coordinates(evidence, observed_census, observed_dominance)
        tests = run_tests(evidence, observed_census, observed_dominance, report)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "MLP coordinate audit attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_UNSELECTED_DIRECT_HIDDEN_MLP_STAGE17_COORDINATE_HOTSPOT_AUDIT",
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
        "input_mode": "existing authenticated in-memory bridge/census/dominance data; exact accounting only",
        "report": report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
