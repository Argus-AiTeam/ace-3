"""Read-only retained RMSNorm interaction cross-product accounting; stdout JSON only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_scalar_scale_residual_margin_bridge_v1 as scale


direct, margin, hidden, residual = scale.direct, scale.margin, scale.hidden, scale.residual
base, parent = scale.base, scale.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_interaction_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 22
BRANCHES = scale.BRANCHES
HIDDEN_COMPONENTS, SCALE_COMPONENTS = direct.COMPONENTS, scale.COMPONENTS
require, same, mass = base.require, base.same, direct.mass
FLAGS = {**scale.FLAGS, "interaction_split_causal_allocation": False}
BOUNDARY = (
    "Only the selected weighted interaction term is split into the 7 by 9 "
    "cross-products of retained direct-hidden components and global scalar-scale "
    "delta components. This grid is accounting closure only, not causal attribution, "
    "causal percentages, or independent branch-error effects. Actual and negative "
    "reference scalar-anchor defects remain separate columns. Direct hidden, "
    "global scale, RMSNorm boundary, unselected-coordinate and head-boundary "
    "accounts remain unchanged. " + direct.REFERENCE_SCOPE + " " + margin.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("interaction bridge forbids operators, prior checks and writes")

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


def split_coordinate(row, direct_row, scale_row, scalar):
    i = row["coordinate"]
    require(type(i) is int and 0 <= i < WIDTH, "coordinate outside retained width")
    h = row["hidden_bridge"]
    for other in (h, direct_row, scale_row, scale_row["selected_residual_energy"]):
        same(other["coordinate"], i, "interaction coordinate splice")
    same(scale_row["selection_reasons"], row["selection_reasons"], "selection reason splice")
    same(direct_row["hidden_components"],
         scale_row["selected_residual_energy"]["hidden_components"],
         "direct-hidden/scale residual component splice")
    same(list(direct_row["hidden_components"]), list(HIDDEN_COMPONENTS),
         "hidden component census changed")
    same(list(scalar["scale_components"]), list(SCALE_COMPONENTS),
         "scalar component census changed")
    components = {k: Fraction(v) for k, v in direct_row["hidden_components"].items()}
    scalars = {k: Fraction(v) for k, v in scalar["scale_components"].items()}
    a, r, w = (Fraction(h[k]) for k in ("actual_hidden", "reference_hidden", "weight"))
    sa, sr = (Fraction(scalar[k]["inverse_norm_anchor"])
              for k in ("actual_norm", "reference_norm"))
    require(sa > 0 and sr > 0, "nonpositive scalar anchor")
    dh, ds = a-r, sa-sr
    require(sum(components.values(), Fraction()) == dh
            and Fraction(h["hidden_delta"]) == dh, "hidden component closure changed")
    require(sum(scalars.values(), Fraction()) == ds
            and Fraction(scalar["inverse_norm_anchor_delta"]) == ds
            and Fraction(h["inverse_norm_anchor_delta"]) == ds,
            "scalar component closure changed")
    dw = Fraction(row["left_weight"]) - Fraction(row["right_weight"])
    require(Fraction(row["row_difference"]) == dw, "tied-row difference splice")
    factor = dw*w
    require(Fraction(direct_row["reference_inverse_norm_anchor"]) == sr
            and Fraction(direct_row["weight_times_reference_anchor_times_row_difference"]) == factor*sr,
            "direct-hidden anchor/factor splice")
    same(direct_row["weighted_components"],
         {k: str(factor*sr*v) for k, v in components.items()},
         "direct-hidden weighted component splice")
    require(Fraction(direct_row["direct_hidden_weighted_term"])
            == Fraction(row["weighted_terms"]["direct_hidden"]) == factor*sr*dh
            and Fraction(h["direct_hidden"]) == w*sr*dh,
            "retained direct-hidden term splice")
    require(Fraction(scale_row["row_difference_times_weight_times_reference_hidden"]) == factor*r,
            "scalar weighting factor splice")
    same(scale_row["weighted_components"],
         {k: str(factor*r*v) for k, v in scalars.items()},
         "retained weighted scalar component splice")
    require(Fraction(scale_row["global_scale_weighted_term"])
            == Fraction(row["weighted_terms"]["global_scale"]) == factor*r*ds
            and Fraction(h["global_scale"]) == w*r*ds, "retained global-scale term splice")
    target = Fraction(row["weighted_terms"]["interaction"])
    require(Fraction(h["interaction"]) == w*dh*ds and target == factor*dh*ds,
            "retained interaction term splice")
    grid = {k: {j: factor*c*s for j, s in scalars.items()} for k, c in components.items()}
    total = mass(v for cells in grid.values() for v in cells.values())
    require(Fraction(total["signed"]) == target, "interaction cross-product closure failed")
    return {
        "coordinate": i, "selection_reasons": row["selection_reasons"],
        "row_difference_times_norm_weight": str(factor),
        "weighted_cross_products": {k: {j: str(v) for j, v in cells.items()}
                                    for k, cells in grid.items()},
        "hidden_component_totals": {k: mass(cells.values()) for k, cells in grid.items()},
        "scalar_component_totals": {j: mass(cells[j] for cells in grid.values())
                                    for j in SCALE_COMPONENTS},
        "interaction_weighted_term": str(target),
        "weighted_cross_product_mass": total,
        "exact_interaction_identity": True,
    }


def selected_account(rows, upstream):
    totals = {
        k: {j: mass(Fraction(row["weighted_cross_products"][k][j]) for row in rows)
            for j in SCALE_COMPONENTS}
        for k in HIDDEN_COMPONENTS
    }
    interaction = mass(Fraction(row["interaction_weighted_term"]) for row in rows)
    same({k: interaction[k] for k in ("signed", "absolute")},
         upstream["selected_term_totals"]["interaction"], "selected interaction totals changed")
    signed = sum((Fraction(v["signed"]) for cells in totals.values() for v in cells.values()), Fraction())
    absolute = sum((Fraction(v["absolute"]) for cells in totals.values() for v in cells.values()), Fraction())
    require(signed == Fraction(interaction["signed"]), "selected interaction grid closure failed")
    within = absolute-Fraction(interaction["absolute"])
    require(within >= 0, "negative within-coordinate cancellation")
    return {
        "cross_product_totals": totals, "selected_interaction": interaction,
        "cross_product_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(within),
        "across_coordinate_cancellation_mass": interaction["cancellation_absolute_mass"],
        "total_cross_product_cancellation_mass": str(absolute-abs(signed)),
        "exact_selected_identity": True,
    }


def report(evidence):
    upstream, retained, scalar_report = (
        evidence[k] for k in ("margin_report", "direct_report", "scalar_report"))
    same(scalar_report["retained_final_rmsnorm_direct_hidden_residual_margin_bridge"],
         retained, "scalar/direct-hidden report splice")
    same(retained["retained_final_rmsnorm_logit_margin_bridge"], upstream,
         "direct-hidden/margin report splice")
    same(upstream["hidden_delta_bridge"], evidence["hidden_report"], "hidden/margin report splice")
    for old in (upstream, retained, scalar_report, evidence["hidden_report"]):
        same([c["control"] for c in old["controls"]], list(parent.CONTROLS),
             "interaction control order changed")
    base.check_history(evidence["result"])
    controls, count, pairs_count = [], 0, 0
    for control, dc, sc, hc in zip(
            upstream["controls"], retained["controls"], scalar_report["controls"],
            evidence["hidden_report"]["controls"], strict=True):
        same(list(sc["global_scalar_accounts"]), list(BRANCHES), "scalar branch census changed")
        for branch in BRANCHES:
            scalar = sc["global_scalar_accounts"][branch]
            same(scalar["actual_norm"], hc["actual_norm"], "actual norm disclosure splice")
            same(scalar["reference_norm"], hc["branches"][branch]["reference_norm"],
                 "independent norm disclosure splice")
            expected = scale.scalar_account(
                {"global_mean_square_delta": scalar["global_mean_square_delta"],
                 "totals": scalar["residual_energy_totals"]},
                scalar["actual_norm"], scalar["reference_norm"])
            same({k: scalar[k] for k in expected}, expected, "retained scalar account splice")
        pairs = []
        for pair, dp, sp in zip(control["pairs"], dc["pairs"], sc["pairs"], strict=True):
            for other in (dp, sp):
                same(tuple(other[k] for k in ("left_id", "right_id", "roles")),
                     tuple(pair[k] for k in ("left_id", "right_id", "roles")),
                     "diagnostic pair splice")
            for other in (pair, dp, sp):
                same(list(other["branches"]), list(BRANCHES), "reference branch census changed")
            branches = {}
            for branch in BRANCHES:
                original, old_direct, old_scale = (p["branches"][branch] for p in (pair, dp, sp))
                for other in (original, old_direct, old_scale):
                    same(other["hidden_reference"], "original_input_L23_"+branch,
                         "independent branch identity changed")
                for other in (old_direct, old_scale):
                    same(other["unchanged_margin_accounting"], original["accounting"],
                         "retained margin remainder splice")
                    same(other["residual_internal_reference"], "original_input_L23_fp16",
                         "residual internal reference changed")
                    same(other["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
                         "binary64 internal stages substituted")
                selected = [
                    split_coordinate(row, dr, sr, sc["global_scalar_accounts"][branch])
                    for row, dr, sr in zip(original["selected_coordinates"],
                                           old_direct["selected_coordinates"],
                                           old_scale["selected_coordinates"], strict=True)
                ]
                indices = [r["coordinate"] for r in selected]
                require(indices == sorted(set(indices)) and 62 in indices
                        and 8 <= len(indices) <= 25, "selected coordinate census changed")
                branches[branch] = {
                    "hidden_reference": original["hidden_reference"],
                    "residual_internal_reference": "original_input_L23_fp16",
                    "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
                    "selected_coordinates": selected,
                    "interaction_accounting": selected_account(selected, original["accounting"]),
                    "unchanged_margin_accounting": original["accounting"],
                }
                count += len(selected)
            pairs.append({"left_id": pair["left_id"], "right_id": pair["right_id"],
                          "roles": pair["roles"], "branches": branches})
        controls.append({"control": control["control"], "pairs": pairs})
        pairs_count += len(pairs)
    for field, value in (
        ("control_count", len(controls)), ("diagnostic_pair_count", pairs_count),
        ("pair_branch_count", pairs_count*len(BRANCHES)),
        ("coordinate62_accounts", pairs_count*len(BRANCHES)),
        ("selected_coordinate_accounts", count),
    ):
        for old in (upstream, retained, scalar_report):
            same(old[field], value, "interaction account census changed: "+field)
    return {
        **{k: upstream[k] for k in ("control_count", "diagnostic_pair_count", "pair_branch_count",
                                   "coordinate62_accounts", "lineage_separation")},
        "selected_coordinate_accounts": count,
        "weighted_cross_product_count": count*len(HIDDEN_COMPONENTS)*len(SCALE_COMPONENTS),
        "cross_product_shape": {"hidden_components": len(HIDDEN_COMPONENTS),
                                "scalar_components": len(SCALE_COMPONENTS)},
        "reference_scope": direct.REFERENCE_SCOPE,
        "identity": "interaction=dW*w*sum(c_k)*sum(delta_s_j)=sum_k_j(dW*w*c_k*delta_s_j)",
        "allocation_scope": "accounting closure only, not causal attribution or precision/scale expansion",
        "retained_final_rmsnorm_scalar_scale_residual_margin_bridge": scalar_report,
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
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("interaction_residual_margin_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "interaction bridge focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "interaction_residual_margin_test": parent.record(TEST)}
    for module in (scale, direct, margin, hidden, residual):
        origins[module.MODULE] = parent.record(module.SOURCE)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "interaction bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_INTERACTION_RESIDUAL_MARGIN_BRIDGE",
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
