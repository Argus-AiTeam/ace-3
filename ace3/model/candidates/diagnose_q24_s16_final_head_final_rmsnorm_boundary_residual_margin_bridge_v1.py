"""Read-only retained final RMSNorm boundary accounting; stdout JSON only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_interaction_residual_margin_bridge_v1 as interaction


scale, direct, margin = interaction.scale, interaction.direct, interaction.margin
hidden, residual, base, parent = (
    interaction.hidden, interaction.residual, interaction.base, interaction.parent)
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_boundary_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 20
BRANCHES = hidden.BRANCHES
COMPONENTS = ("actual_boundary_conversion", "negative_reference_boundary_conversion")
require, same, mass = base.require, base.same, direct.mass
FLAGS = {**interaction.FLAGS, "boundary_split_causal_allocation": False}
BOUNDARY = (
    "Only the selected weighted boundary_remainder_delta term is split into "
    "actual and negative independent-reference boundary/conversion remainders. "
    "b=y-w*h*s uses the retained outputs and unchanged binary64 scalar anchors. "
    "These combined implementation, epsilon, normalization, multiply/conversion "
    "and scalar-anchor effects are not separately identifiable and not rounding-only. "
    "Q24 input conversion is disclosed, not reassigned to final RMSNorm boundaries. "
    "This is accounting closure only, not causal attribution. Direct-hidden, "
    "scalar-scale, interaction, unselected-coordinate and head-boundary accounts "
    "remain unchanged. " + direct.REFERENCE_SCOPE + " " + margin.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("boundary bridge forbids operators, prior checks and writes")

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


def split_coordinate(row, actual_norm, reference_norm):
    i, h = row["coordinate"], row["hidden_bridge"]
    require(type(i) is int and 0 <= i < WIDTH, "coordinate outside retained width")
    same(h["coordinate"], i, "boundary coordinate splice")
    dw = Fraction(row["left_weight"]) - Fraction(row["right_weight"])
    require(Fraction(row["row_difference"]) == dw, "tied-row difference splice")
    a, r, ya, yr, w = (Fraction(h[k]) for k in (
        "actual_hidden", "reference_hidden", "actual_rmsnorm", "reference_rmsnorm", "weight"))
    sa, sr = (Fraction(n["inverse_norm_anchor"]) for n in (actual_norm, reference_norm))
    require(sa > 0 and sr > 0, "nonpositive scalar anchor")
    require(Fraction(h["hidden_delta"]) == a-r
            and Fraction(h["inverse_norm_anchor_delta"]) == sa-sr,
            "hidden/scalar delta splice")
    ba, br = ya-w*a*sa, yr-w*r*sr
    expected = dict(zip(hidden.TERMS, (
        w*(a-r)*sr, w*r*(sa-sr), w*(a-r)*(sa-sr), ba-br), strict=True))
    same({k: h[k] for k in expected}, {k: str(v) for k, v in expected.items()},
         "retained hidden term splice")
    same(row["weighted_terms"], {k: str(dw*v) for k, v in expected.items()},
         "retained weighted term splice")
    require(Fraction(h["actual_boundary_conversion_remainder"]) == ba
            and Fraction(h["reference_boundary_conversion_remainder"]) == br
            and Fraction(row["weighted_actual_boundary_conversion_remainder"]) == dw*ba
            and Fraction(row["weighted_reference_boundary_conversion_remainder"]) == dw*br,
            "actual/reference boundary conversion splice")
    require(Fraction(h["retained_rmsnorm_delta"]) == ya-yr
            and Fraction(h["signed_term_sum"]) == sum(expected.values(), Fraction()) == ya-yr
            and Fraction(row["coordinate_margin_change"]) == dw*(ya-yr),
            "retained output delta splice")
    require(Fraction(h["actual_q24_to_stage18_conversion"])
            == a-Fraction(h["actual_raw_q24_hidden"]), "Q24 input conversion splice")
    components = dict(zip(COMPONENTS, (dw*ba, -dw*br), strict=True))
    target = Fraction(row["weighted_terms"]["boundary_remainder_delta"])
    totals = mass(components.values())
    require(Fraction(totals["signed"]) == target, "boundary split closure failed")
    return {
        "coordinate": i, "selection_reasons": row["selection_reasons"],
        "row_difference": str(dw), "norm_weight": str(w),
        "actual": {
            "retained_hidden": str(a), "retained_rmsnorm": str(ya),
            "inverse_norm_anchor": str(sa), "anchored_product": str(w*a*sa),
            "boundary_conversion_remainder": str(ba),
            "weighted_boundary_conversion_remainder": str(dw*ba),
        },
        "reference": {
            "retained_hidden": str(r), "retained_rmsnorm": str(yr),
            "inverse_norm_anchor": str(sr), "anchored_product": str(w*r*sr),
            "boundary_conversion_remainder": str(br),
            "weighted_boundary_conversion_remainder": str(dw*br),
        },
        "actual_raw_q24_hidden": h["actual_raw_q24_hidden"],
        "actual_q24_to_stage18_conversion": h["actual_q24_to_stage18_conversion"],
        "weighted_components": {k: str(v) for k, v in components.items()},
        "boundary_remainder_delta_weighted_term": str(target),
        "weighted_boundary_mass": totals, "exact_boundary_identity": True,
    }


def selected_account(rows, upstream):
    for row in rows:
        same(list(row["weighted_components"]), list(COMPONENTS), "boundary component census changed")
        values = [Fraction(row["weighted_components"][k]) for k in COMPONENTS]
        same(row["weighted_boundary_mass"], mass(values), "boundary coordinate mass splice")
        require(sum(values, Fraction()) == Fraction(row["boundary_remainder_delta_weighted_term"]),
                "boundary coordinate target splice")
    totals = {k: mass(Fraction(row["weighted_components"][k]) for row in rows) for k in COMPONENTS}
    target = mass(Fraction(row["boundary_remainder_delta_weighted_term"]) for row in rows)
    same({k: target[k] for k in ("signed", "absolute")},
         upstream["selected_term_totals"]["boundary_remainder_delta"], "selected boundary totals changed")
    signed = sum((Fraction(v["signed"]) for v in totals.values()), Fraction())
    absolute = sum((Fraction(v["absolute"]) for v in totals.values()), Fraction())
    require(signed == Fraction(target["signed"]) and absolute >= Fraction(target["absolute"]),
            "selected boundary split closure failed")
    return {
        "component_totals": totals, "selected_boundary_remainder_delta": target,
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute-Fraction(target["absolute"])),
        "across_coordinate_cancellation_mass": target["cancellation_absolute_mass"],
        "total_component_cancellation_mass": str(absolute-abs(signed)),
        "exact_selected_identity": True,
    }


def report(evidence):
    upstream, retained = evidence["margin_report"], evidence["interaction_report"]
    for outer, key, inner in (
        (retained, "retained_final_rmsnorm_scalar_scale_residual_margin_bridge", evidence["scalar_report"]),
        (evidence["scalar_report"], "retained_final_rmsnorm_direct_hidden_residual_margin_bridge",
         evidence["direct_report"]),
        (evidence["direct_report"], "retained_final_rmsnorm_logit_margin_bridge", upstream),
        (upstream, "hidden_delta_bridge", evidence["hidden_report"]),
    ):
        same(outer[key], inner, "retained bridge linkage splice")
    for old in (upstream, retained, evidence["hidden_report"], evidence["scalar_report"], evidence["direct_report"]):
        same([c["control"] for c in old["controls"]], list(parent.CONTROLS), "boundary control census changed")
    base.check_history(evidence["result"])
    controls, count, pair_count = [], 0, 0
    for control, ic, hc in zip(
            upstream["controls"], retained["controls"], evidence["hidden_report"]["controls"], strict=True):
        pairs = []
        for pair, ip in zip(control["pairs"], ic["pairs"], strict=True):
            same(tuple(ip[k] for k in ("left_id", "right_id", "roles")),
                 tuple(pair[k] for k in ("left_id", "right_id", "roles")), "boundary pair splice")
            for p in (pair, ip):
                same(list(p["branches"]), list(BRANCHES), "reference branch census changed")
            branches = {}
            for branch in BRANCHES:
                original, old = pair["branches"][branch], ip["branches"][branch]
                for other in (original, old):
                    same(other["hidden_reference"], "original_input_L23_"+branch, "reference identity changed")
                same(original["rmsnorm_reference"], "original_input_final_attempt003_"+branch,
                     "final reference identity changed")
                same(old["unchanged_margin_accounting"], original["accounting"], "margin remainder splice")
                same([(r["coordinate"], r["selection_reasons"]) for r in old["selected_coordinates"]],
                     [(r["coordinate"], r["selection_reasons"]) for r in original["selected_coordinates"]],
                     "selected union splice")
                selected = [split_coordinate(r, hc["actual_norm"], hc["branches"][branch]["reference_norm"])
                            for r in original["selected_coordinates"]]
                indices = [r["coordinate"] for r in selected]
                require(indices == sorted(set(indices)) and 62 in indices and 8 <= len(indices) <= 25,
                        "selected coordinate census changed")
                branches[branch] = {
                    "hidden_reference": original["hidden_reference"],
                    "rmsnorm_reference": original["rmsnorm_reference"],
                    "residual_internal_reference": old["residual_internal_reference"],
                    "binary64_internal_stages": old["binary64_internal_stages"],
                    "reference_norm": hc["branches"][branch]["reference_norm"],
                    "selected_coordinates": selected,
                    "boundary_accounting": selected_account(selected, original["accounting"]),
                    "unchanged_margin_accounting": original["accounting"],
                }
                same(branches[branch]["residual_internal_reference"], "original_input_L23_fp16",
                     "residual internal reference changed")
                same(branches[branch]["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
                     "binary64 internal stages substituted")
                count += len(selected)
            pairs.append({"left_id": pair["left_id"], "right_id": pair["right_id"],
                          "roles": pair["roles"], "branches": branches})
        controls.append({"control": control["control"], "actual_norm": hc["actual_norm"], "pairs": pairs})
        pair_count += len(pairs)
    census = {"control_count": len(controls), "diagnostic_pair_count": pair_count,
              "pair_branch_count": pair_count*len(BRANCHES),
              "coordinate62_accounts": pair_count*len(BRANCHES),
              "selected_coordinate_accounts": count}
    for old in (upstream, retained, evidence["scalar_report"], evidence["direct_report"]):
        for key, value in census.items():
            same(old[key], value, "boundary account census changed: "+key)
    return {
        **census, "weighted_boundary_component_count": count*len(COMPONENTS),
        "lineage_separation": upstream["lineage_separation"],
        "reference_scope": direct.REFERENCE_SCOPE,
        "identity": "dW*delta_b=dW*(y_actual-w*h_actual*s_actual)-dW*(y_ref-w*h_ref*s_ref)",
        "remainder_scope": evidence["hidden_report"]["remainder_scope"],
        "scalar_anchor": evidence["hidden_report"]["scalar_anchor"],
        "retained_final_rmsnorm_interaction_residual_margin_bridge": retained,
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
    evidence["direct_report"] = direct.report(evidence)
    evidence["scalar_report"] = scale.report(evidence)
    evidence["interaction_report"] = interaction.report(evidence)
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("boundary_residual_margin_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "boundary bridge focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "boundary_residual_margin_test": parent.record(TEST)}
    for module in (interaction, scale, direct, margin, hidden, residual):
        origins[module.MODULE] = parent.record(module.SOURCE)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "boundary bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_BOUNDARY_RESIDUAL_MARGIN_BRIDGE",
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
