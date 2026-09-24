"""Stdout-only retained-coordinate terminal-remainder audit; no operator replay."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_dominance_divergence_audit_v1 as dominance


bridge, census, base, parent = dominance.bridge, dominance.census, dominance.base, dominance.parent
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_terminal_remainder_coordinate_hotspot_audit_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
PAIRS = ((319, 34319), (34319, 319))
MLP, TERMINAL = dominance.MLP, dominance.TERMINAL
EXPECTED_TESTS = 12
FLAGS = {**dominance.FLAGS, "terminal_coordinate_causal_allocation": False}
BOUNDARY = (
    "Ordering coordinates by absolute terminal-remainder contribution is only "
    "retained accounting, not component selection, a new split, causality or "
    "performance attribution. Coordinate gaps compare contributions at the SAME "
    "coordinate, not total component magnitudes. All complement coordinates are "
    "retained, including zero contributions; ties use ascending coordinate ID. "
    "FP16 zeros have no meaningful terminal hotspot ordering. No binary64 internal "
    "stage is reconstructed. The unchanged common UNKNOWN still stops nested "
    "bridge expansion; this ordering does not authorize it. " + dominance.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("terminal coordinate audit forbids predecessor checks and replay")

    with dominance.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests"):
            stack.enter_context(patch.object(dominance, name, forbidden))
        yield


def bound_vectors(evidence):
    reference = bridge.residual.operands(evidence["reference_archive"], actual=False)
    same(list(evidence["actual"]), list(census.CONTROLS), "actual control census changed")
    for control in census.CONTROLS:
        require(evidence["actual"][control]
                == bridge.residual.operands(evidence["archives"][control], actual=True),
                "retained actual vector splice")
    same(list(evidence["hidden"]), list(census.BRANCHES), "terminal branch census changed")
    require(evidence["hidden"]["fp16"] == reference["stage18"], "FP16 terminal vector splice")
    require(evidence["hidden"]["binary64"]
            == bridge.margin.contributions.exact_vector(evidence["binary64"]),
            "binary64 terminal vector splice")
    require(evidence["weights"] == bridge.hidden.validate_weight(
        evidence["weight_array"], evidence["assets"]["tensors"]["model.norm.weight"]),
        "norm weight vector splice")
    rows = {i: bridge.margin.contributions.exact_vector(v) for i, v in evidence["rows"].items()}
    for vector in (*evidence["hidden"].values(), evidence["weights"], *rows.values()):
        require(len(vector) == bridge.WIDTH and all(isinstance(v, Fraction) for v in vector),
                "invalid exact retained vector")
    return reference, rows


def coordinate_row(evidence, reference, rows, pair, branch):
    old = pair["branches"][branch]
    gate = old["gate_values"]
    selected = gate["excluded_selected_coordinates"]
    require(all(type(i) is int and 0 <= i < bridge.WIDTH for i in selected)
            and selected == sorted(set(selected)) and 62 in selected
            and 8 <= len(selected) <= 25, "invalid unchanged selected coordinate union")
    same(selected, [r["coordinate"] for r in
                    old["unchanged_unselected_bridge"]["selected_coordinates"]],
         "selected coordinate binding changed")
    anchor = Fraction(old["unchanged_unselected_bridge"]["reference_norm"]["inverse_norm_anchor"])
    require(anchor > 0, "invalid retained inverse norm anchor")
    left, right = rows[pair["left_id"]], rows[pair["right_id"]]
    actual = evidence["actual"][gate["control"]]
    excluded = set(selected)
    coordinates = []
    terminal_values, mlp_values = [], []
    for i in range(bridge.WIDTH):
        if i in excluded:
            continue
        factor = evidence["weights"][i] * anchor * (left[i]-right[i])
        terminal_delta = reference["stage18"][i]-evidence["hidden"][branch][i]
        mlp_delta = actual["stage17"][i]-reference["stage17"][i]
        terminal, mlp = factor*terminal_delta, factor*mlp_delta
        terminal_values.append(terminal)
        mlp_values.append(mlp)
        coordinates.append({
            "coordinate": i, "retained_weighted_row_factor": str(factor),
            "terminal_hidden_delta": str(terminal_delta), "mlp_stage17_hidden_delta": str(mlp_delta),
            "terminal_signed_contribution": str(terminal),
            "terminal_absolute_contribution": str(abs(terminal)),
            "mlp_stage17_signed_contribution": str(mlp),
            "mlp_stage17_absolute_contribution": str(abs(mlp)),
            "terminal_minus_mlp_coordinate_magnitude": str(abs(terminal)-abs(mlp)),
        })
    totals = {TERMINAL: bridge.mass(terminal_values), MLP: bridge.mass(mlp_values)}
    for component, total in totals.items():
        same(total, gate["component_totals"][component],
             "coordinate signed/absolute closure changed: " + component)
    same(len(coordinates), gate["coordinate_count"], "coordinate complement count changed")
    if branch == "fp16":
        require(all(v == 0 for v in terminal_values), "FP16 terminal coordinate boundary changed")
    ordered = sorted(coordinates, key=lambda r: (-Fraction(r["terminal_absolute_contribution"]),
                                                 r["coordinate"]))
    nonzero = [r for r in ordered if Fraction(r["terminal_absolute_contribution"]) != 0]
    top = nonzero[:8]
    return {
        "control": gate["control"], "branch": branch,
        "coordinate_count": len(coordinates), "excluded_selected_coordinates": selected,
        "hidden_reference": "original_input_L23_"+branch,
        "rmsnorm_reference": "original_input_final_attempt003_"+branch,
        "binary64_internal_stages": old["binary64_internal_stages"],
        "coordinates": coordinates,
        "terminal_absolute_coordinate_order": [r["coordinate"] for r in ordered],
        "top_terminal_coordinates": top,
        "top_terminal_magnitude_ties": [r["coordinate"] for r in nonzero
                                      if r["terminal_absolute_contribution"]
                                      == nonzero[0]["terminal_absolute_contribution"]],
        "nonzero_terminal_coordinate_count": len(nonzero),
        "terminal_hotspot_order_meaningful": bool(nonzero),
        "fp16_terminal_zero_boundary": branch == "fp16" and not nonzero,
        "component_totals": totals,
        "exact_signed_and_absolute_closure": True,
        "retained_gate_values": gate,
    }


def audit_coordinates(evidence, observed_census, observed_dominance):
    base.check_history(evidence["result"])
    same(observed_dominance, dominance.audit_census(observed_census, evidence["report"]),
         "terminal audit dominance input binding changed")
    reference, rows = bound_vectors(evidence)
    pairs = []
    for identity in PAIRS:
        retained = next(p for p in observed_dominance["pairs"]
                        if (p["left_id"], p["right_id"]) == identity)
        require(retained["census_proof"]["complete_nine_control_two_branch_census"],
                "incomplete divergent pair census")
        for row in retained["rows"]:
            same(row["unique_largest_component"], TERMINAL if row["branch"] == "binary64" else MLP,
                 "retained divergent branch dominance changed")
            if row["branch"] == "binary64":
                require(Fraction(row["mlp_minus_terminal_net_magnitude"]) < 0,
                        "terminal remainder no longer strictly exceeds MLP")
        accounts = []
        for control in evidence["report"]["controls"]:
            pair = next(p for p in control["pairs"] if (p["left_id"], p["right_id"]) == identity)
            accounts.extend(coordinate_row(evidence, reference, rows, pair, b) for b in census.BRANCHES)
        same([(r["control"], r["branch"]) for r in accounts],
             [(c, b) for c in census.CONTROLS for b in census.BRANCHES],
             "terminal coordinate row census changed")
        pairs.append({
            "left_id": identity[0], "right_id": identity[1], "rows": accounts,
            "component": retained["component"],
            "stop_nested_bridge_expansion": retained["stop_nested_bridge_expansion"],
        })
    same(observed_dominance["common_component"], "UNKNOWN", "common gate changed")
    require(observed_dominance["stop_nested_bridge_expansion"] is True, "common stop gate changed")
    return {
        "pairs": pairs, "pair_count": len(pairs), "row_count": sum(len(p["rows"]) for p in pairs),
        "coordinate_order_rule": "descending absolute terminal contribution, then ascending coordinate ID",
        "coordinate_gap_rule": "|terminal contribution at i|-|mlp_stage17 contribution at i|",
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
    spec = importlib.util.spec_from_file_location("terminal_coordinate_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.CENSUS = evidence, observed_census
    module.DOMINANCE, module.OBSERVED = observed_dominance, report
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "terminal coordinate tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "terminal_coordinate_tests": parent.record(TEST)}
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
        same(audit, {"forbidden_calls": 0}, "terminal coordinate audit attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_UNSELECTED_DIRECT_HIDDEN_TERMINAL_REMAINDER_COORDINATE_HOTSPOT_AUDIT",
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
