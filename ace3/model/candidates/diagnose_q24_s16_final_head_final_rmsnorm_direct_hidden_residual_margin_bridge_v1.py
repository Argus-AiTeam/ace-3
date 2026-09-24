"""Read-only retained direct-hidden residual margin accounting; stdout JSON only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_logit_margin_bridge_v1 as margin


hidden, residual = margin.hidden, margin.hidden.residual
base, parent = margin.base, margin.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_direct_hidden_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 22
BRANCHES = margin.BRANCHES
require, same = base.require, base.same
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
FLAGS = {**margin.FLAGS, "binary64_internal_stages_substituted": False}
REFERENCE_SCOPE = (
    "L23 branch operands are original-input FP16 retained stages only. Binary64 "
    "internal stages are NOT_RETAINED_NO_RECONSTRUCTION. The explicit FP16 terminal "
    "minus independent branch terminal remainder is zero for FP16 and retains the "
    "original binary64 terminal difference for binary64; it is not a binary64 "
    "input/attention/MLP decomposition or a replacement reference."
)
BOUNDARY = (
    "Only selected direct_hidden weighted terms are split into L23 residual "
    "components. Global scale, interaction, RMSNorm boundary, unselected-coordinate "
    "and head-boundary terms are preserved, not reassigned. " + REFERENCE_SCOPE
    + " No RMSNorm, head, row-dot or local-operator replay; no earlier-check dispatch. "
    + margin.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("direct-hidden bridge forbids operators, prior checks and writes")

    with margin.hotspots.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.") and name != MODULE:
                for attribute in (
                    "rmsnorm", "_torch_rmsnorm", "logits", "decode_array_q24",
                    "load_operands", "reference_suffix", "compute", "projection",
                    "local_reference", "attention_value", "rne", "toward_zero",
                    "check", "focused_tests", "measure",
                ):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        yield


def mass(values):
    values = list(values)
    signed = sum(values, Fraction())
    absolute = sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute - abs(signed))}


def split_coordinate(actual, reference, branch_terminal, reference_anchor, row):
    i = row["coordinate"]
    require(type(i) is int and 0 <= i < WIDTH, "coordinate outside retained width")
    require(isinstance(branch_terminal, Fraction)
            and isinstance(reference_anchor, Fraction) and reference_anchor > 0,
            "invalid independent terminal or scalar anchor")
    h = row["hidden_bridge"]
    same(h["coordinate"], i, "hidden coordinate splice")
    a, r = actual["stage18"][i], reference["stage18"][i]
    require(Fraction(h["actual_hidden"]) == a, "actual hidden splice")
    require(Fraction(h["reference_hidden"]) == branch_terminal, "reference hidden splice")
    require(Fraction(h["actual_raw_q24_hidden"]) == actual["output_q24"][i],
            "Q24 output splice")
    difference = Fraction(row["left_weight"]) - Fraction(row["right_weight"])
    require(Fraction(row["row_difference"]) == difference, "tied-row difference splice")
    weight = Fraction(h["weight"])
    factor = weight * reference_anchor * difference
    local = residual.account(actual, reference, i)
    parts = local["actual_q24_boundary_parts"]
    values = {
        "input_hidden": actual["input_hidden"][i] - reference["input_hidden"][i],
        "attention_stage11": actual["stage11"][i] - reference["stage11"][i],
        "mlp_stage17": actual["stage17"][i] - reference["stage17"][i],
        "actual_residual_boundary": sum(
            (Fraction(parts[key]) for key in residual.ACTUAL_PARTS[:-1]), Fraction()),
        "negative_reference_residual_boundary": -Fraction(local["reference_boundary_remainder"]),
        "q24_to_fp16_conversion": a - actual["output_q24"][i],
        "fp16_to_branch_terminal_remainder": r - branch_terminal,
    }
    direct = Fraction(row["weighted_terms"]["direct_hidden"])
    require(sum(values.values(), Fraction()) == a - branch_terminal
            and Fraction(h["hidden_delta"]) == a - branch_terminal
            and Fraction(h["direct_hidden"]) == weight * reference_anchor * (a - branch_terminal)
            and direct == factor * (a - branch_terminal)
            and Fraction(h["actual_q24_to_stage18_conversion"]) == values["q24_to_fp16_conversion"],
            "direct-hidden/residual/conversion identity changed")
    weighted = {key: value * factor for key, value in values.items()}
    require(sum(weighted.values(), Fraction()) == direct, "weighted residual closure failed")
    return {
        "coordinate": i, "reference_inverse_norm_anchor": str(reference_anchor),
        "weight_times_reference_anchor_times_row_difference": str(factor),
        "retained_fp16_residual_account": local,
        "hidden_components": {key: str(value) for key, value in values.items()},
        "weighted_components": {key: str(value) for key, value in weighted.items()},
        "weighted_actual_residual_boundary_parts": {
            key: str(Fraction(parts[key]) * factor) for key in residual.ACTUAL_PARTS[:-1]},
        "weighted_negative_reference_residual_boundary_parts": {
            key: str(-Fraction(value) * factor)
            for key, value in local["reference_fp16_boundary_parts"].items()},
        "direct_hidden_weighted_term": str(direct),
        "hidden_component_mass": mass(values.values()),
        "weighted_component_mass": mass(weighted.values()),
        "exact_direct_hidden_identity": True,
    }


def selected_account(rows, upstream):
    totals = {key: mass(Fraction(row["weighted_components"][key]) for row in rows)
              for key in COMPONENTS}
    direct = mass(Fraction(row["direct_hidden_weighted_term"]) for row in rows)
    same({key: direct[key] for key in ("signed", "absolute")},
         upstream["selected_term_totals"]["direct_hidden"],
         "selected direct-hidden totals changed")
    require(sum((Fraction(row["signed"]) for row in totals.values()), Fraction())
            == Fraction(direct["signed"]), "selected residual component closure failed")
    absolute = sum((Fraction(row["absolute"]) for row in totals.values()), Fraction())
    return {
        "component_totals": totals, "selected_direct_hidden": direct,
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute - Fraction(direct["absolute"])),
        "across_coordinate_cancellation_mass": direct["cancellation_absolute_mass"],
        "total_component_cancellation_mass": str(absolute - abs(Fraction(direct["signed"]))),
        "exact_selected_identity": True,
    }


def report(evidence):
    base.check_history(evidence["result"])
    upstream = evidence["margin_report"]
    same([c["control"] for c in upstream["controls"]], list(parent.CONTROLS),
         "margin control order changed")
    reference = residual.operands(evidence["reference_archive"], actual=False)
    controls, accounts = [], 0
    for control, hidden_control in zip(upstream["controls"],
                                      evidence["hidden_report"]["controls"], strict=True):
        label = control["control"]
        same(hidden_control["control"], label, "hidden/margin control splice")
        pairs = []
        for pair in control["pairs"]:
            same(list(pair["branches"]), list(BRANCHES), "reference branch order changed")
            branches = {}
            for branch, upstream_branch in pair["branches"].items():
                same(upstream_branch["hidden_reference"], "original_input_L23_" + branch,
                     "reference branch identity changed")
                selected = upstream_branch["selected_coordinates"]
                indices = [row["coordinate"] for row in selected]
                require(indices == sorted(set(indices)) and 62 in indices
                        and 8 <= len(indices) <= 25, "selected coordinate census changed")
                sr = Fraction(hidden_control["branches"][branch]["reference_norm"]["inverse_norm_anchor"])
                split = [split_coordinate(evidence["actual"][label], reference,
                                          evidence["hidden"][branch][row["coordinate"]], sr, row)
                         for row in selected]
                branches[branch] = {
                    "hidden_reference": upstream_branch["hidden_reference"],
                    "residual_internal_reference": "original_input_L23_fp16",
                    "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
                    "selected_coordinates": split,
                    "direct_hidden_accounting": selected_account(split, upstream_branch["accounting"]),
                    "unchanged_margin_accounting": upstream_branch["accounting"],
                }
                accounts += len(split)
            pairs.append({"left_id": pair["left_id"], "right_id": pair["right_id"],
                          "roles": pair["roles"], "branches": branches})
        controls.append({"control": label, "pairs": pairs})
    return {
        "control_count": len(controls),
        "diagnostic_pair_count": upstream["diagnostic_pair_count"],
        "pair_branch_count": upstream["pair_branch_count"],
        "coordinate62_accounts": upstream["coordinate62_accounts"],
        "selected_coordinate_accounts": accounts,
        "weighted_component_count": accounts * len(COMPONENTS),
        "reference_scope": REFERENCE_SCOPE,
        "identity": (
            "direct_hidden=dW*w*s_ref*(delta_input_hidden+delta_stage11+delta_stage17"
            "+actual_residual_boundary-reference_residual_boundary"
            "+Q24_to_FP16+FP16_terminal_minus_independent_branch_terminal)"),
        "lineage_separation": upstream["lineage_separation"],
        "retained_final_rmsnorm_logit_margin_bridge": upstream,
        "controls": controls,
    }


def measure():
    for pin in margin.rows.PINS.values():
        base.read_bound(pin)
    evidence = hidden.authenticate()
    evidence["hidden_report"] = hidden.report(evidence)
    evidence["geometry"] = margin.cutoff.report(
        evidence["result"], evidence["arrays"], evidence["references"])
    ids = {i for control in evidence["geometry"]["controls"]
           for pair in margin.contributions.pairs_for(control) for i in pair}
    evidence["rows"] = margin.contributions.load_rows(evidence["assets"], ids)
    evidence["margin_report"] = margin.report(evidence)
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("direct_hidden_residual_margin_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "direct-hidden bridge focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors),
            "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "direct_hidden_residual_margin_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "direct-hidden bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_DIRECT_HIDDEN_RESIDUAL_MARGIN_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "report": evidence["report"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
