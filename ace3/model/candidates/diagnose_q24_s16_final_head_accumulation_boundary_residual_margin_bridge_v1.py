"""Read-only selected-row final-head boundary accounting; one stdout JSON."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_direct_hidden_residual_margin_bridge_v1 as direct
from ace3.model.candidates import diagnose_q24_s16_final_head_mixed_rmsnorm_counterfactual_v1 as mixed


margin = direct.margin
hidden, contributions, base, parent = margin.hidden, margin.contributions, margin.base, margin.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_accumulation_boundary_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 18
BRANCHES = hidden.BRANCHES
COMPONENTS = ("actual_head_boundary", "negative_reference_head_boundary")
require, same, mass = base.require, base.same, direct.mass
FLAGS = {key: value for key, value in margin.FLAGS.items()
         if key != "local_exact_row_dot_computations"}
FLAGS.update({
    "bounded_selected_row_arithmetic_permitted": True,
    "artifact_overwrites": 0, "causal_root_cause_claimed": False,
    "binary64_internal_stages_substituted": False,
})
BOUNDARY = (
    "Only exact selected-row arithmetic on existing numeric diagnostic pairs is "
    "permitted and counted, with cached row dots kept distinct by actual control "
    "and independent reference branch. The actual retained-logit margin minus "
    "exact actual row-dot margin and the negative reference retained-logit margin "
    "minus exact reference row-dot margin close the existing head boundary. "
    "These combined accumulation/order/conversion effects are not rounding-only "
    "or causal root-cause allocations. No RMSNorm or full-vocabulary head replay, "
    "mixed-head counterfactual replay, earlier checks, artifact overwrites or "
    "precision/scale expansion. No category selection or change-sign gate is "
    "performed; this is not nested unselected-coordinate bridge expansion. "
    + contributions.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("head boundary bridge forbids replay and writes")

    with margin.hotspots.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.") and name != MODULE:
                for attribute in (
                    "rmsnorm", "_torch_rmsnorm", "logits", "decode_array_q24",
                    "load_operands", "reference_suffix", "compute", "projection",
                    "local_reference", "attention_value", "rne", "toward_zero",
                    "exact_head", "float_head", "check", "focused_tests", "measure",
                ):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        yield


def selected_row_dot(vector, row, audit):
    require(vector.shape == row.shape == (WIDTH,) and row.dtype.str == "<f2",
            "selected-row width or FP16 weight boundary changed")
    x, w = contributions.exact_vector(vector), contributions.exact_vector(row)
    audit["local_exact_row_dot_computations"] += 1
    audit["selected_row_scalar_products"] += WIDTH
    return sum((a*b for a, b in zip(x, w, strict=True)), Fraction())


def split_boundary(actual_margin, reference_margin, actual_dots, reference_dots,
                   effect, accounting):
    operands = (actual_margin, reference_margin, *actual_dots, *reference_dots)
    require(len(actual_dots) == len(reference_dots) == 2
            and all(isinstance(v, Fraction) for v in operands),
            "head margin operands must be exact selected-row pairs")
    ea, er = actual_dots[0]-actual_dots[1], reference_dots[0]-reference_dots[1]
    vector = Fraction(effect["exact_sum"])
    boundary = Fraction(accounting["head_boundary_remainder_change"])
    change = actual_margin-reference_margin
    require(Fraction(effect["retained_actual_margin"]) == actual_margin
            and Fraction(effect["retained_reference_margin"]) == reference_margin
            and Fraction(effect["retained_margin_change"]) == change
            and Fraction(accounting["retained_margin_change"]) == change,
            "retained actual/reference margin splice")
    require(ea-er == vector, "selected-row versus vector-coordinate closure failed")
    ba, br = actual_margin-ea, reference_margin-er
    require(ba-br == boundary == Fraction(effect["boundary_remainder_change"]),
            "actual/reference head-boundary split failed")
    selected = Fraction(accounting["selected_coordinate_signed_sum"])
    unselected = Fraction(accounting["unselected_coordinate_signed_remainder"])
    require(selected+unselected == vector and vector+boundary == change,
            "selected/unselected/head margin closure failed")
    require(Fraction(effect["exact_sum_of_absolute_contributions"]) >= abs(vector)
            and Fraction(accounting["unselected_coordinate_absolute_remainder"]) >= abs(unselected),
            "coordinate absolute account invalid")
    components = dict(zip(COMPONENTS, (ba, -br), strict=True))
    return {
        "retained_actual_margin": str(actual_margin),
        "retained_reference_margin": str(reference_margin),
        "retained_margin_change": str(change),
        "actual_exact_left_row_dot": str(actual_dots[0]),
        "actual_exact_right_row_dot": str(actual_dots[1]),
        "reference_exact_left_row_dot": str(reference_dots[0]),
        "reference_exact_right_row_dot": str(reference_dots[1]),
        "actual_exact_row_dot_margin": str(ea),
        "reference_exact_row_dot_margin": str(er),
        "vector_coordinate_sum": str(vector),
        "vector_coordinate_absolute_sum": effect["exact_sum_of_absolute_contributions"],
        "vector_coordinate_cancellation_mass": str(
            Fraction(effect["exact_sum_of_absolute_contributions"])-abs(vector)),
        "actual_head_boundary": str(ba),
        "reference_head_boundary": str(br),
        "negative_reference_head_boundary": str(-br),
        "head_boundary_remainder_change": str(boundary),
        "boundary_components": {k: str(v) for k, v in components.items()},
        "boundary_accounting": mass(components.values()),
        "margin_accounting": mass((vector, ba, -br)),
        "unchanged_margin_accounting": accounting,
        "closure_residuals": {
            "actual_retained_minus_exact_and_boundary": str(actual_margin-ea-ba),
            "reference_retained_minus_exact_and_boundary": str(reference_margin-er-br),
            "row_dot_change_minus_vector_sum": str(ea-er-vector),
            "boundary_split_minus_retained_remainder": str(ba-br-boundary),
            "selected_unselected_boundary_minus_retained_change": str(
                selected+unselected+ba-br-change),
        },
        "exact_boundary_identity": True,
        "exact_margin_identity": True,
    }


def report(evidence, audit):
    base.check_history(evidence["result"])
    upstream, geometry = evidence["margin_report"], evidence["geometry"]
    same(upstream["hidden_delta_bridge"], evidence["hidden_report"], "hidden linkage splice")
    same(upstream["cutoff_margin_report"], geometry, "cutoff linkage splice")
    for controls in (upstream["controls"], geometry["controls"]):
        same([c["control"] for c in controls], list(parent.CONTROLS), "control census changed")
    same(list(evidence["arrays"]), list(parent.CONTROLS), "array control census changed")
    expected_ids = {i for c in geometry["controls"]
                    for pair in contributions.pairs_for(c) for i in pair}
    same(sorted(evidence["rows"]), sorted(expected_ids), "selected tied-head row census changed")
    cache, controls = {}, []
    branch_accounts = {b: [] for b in BRANCHES}

    def dots(trajectory, vector, ids):
        values = []
        for index in ids:
            key = trajectory, index
            if key not in cache:
                cache[key] = selected_row_dot(vector, evidence["rows"][index], audit)
            values.append(cache[key])
        return tuple(values)

    for control, g in zip(upstream["controls"], geometry["controls"], strict=True):
        label = control["control"]
        expected_pairs = contributions.pairs_for(g)
        same([(p["left_id"], p["right_id"]) for p in control["pairs"]],
             list(expected_pairs), "diagnostic pair census changed")
        av = evidence["arrays"][label]["rmsnorm"].view("<f2")
        al = evidence["arrays"][label]["logits"].view("<f2")
        pairs = []
        for pair in control["pairs"]:
            ids = pair["left_id"], pair["right_id"]
            same(pair["roles"], expected_pairs[ids], "diagnostic pair roles changed")
            same(list(pair["branches"]), list(BRANCHES), "reference branch census changed")
            for branch in BRANCHES:
                old = pair["branches"][branch]
                same(old["hidden_reference"], "original_input_L23_"+branch, "hidden reference splice")
                same(old["rmsnorm_reference"], "original_input_final_attempt003_"+branch,
                     "final reference splice")
            ad = dots(("actual", label), av, ids)
            am = margin.cutoff.exact(al[ids[0]])-margin.cutoff.exact(al[ids[1]])
            branches = {}
            for branch in BRANCHES:
                old = pair["branches"][branch]
                rv = evidence["references"]["rmsnorm_"+branch]
                rl = evidence["references"]["logits_"+branch]
                if branch == "fp16":
                    rv, rl = rv.view("<f2"), rl.view("<f2")
                rd = dots(("reference", branch), rv, ids)
                rm = margin.cutoff.exact(rl[ids[0]])-margin.cutoff.exact(rl[ids[1]])
                account = split_boundary(am, rm, ad, rd, old["retained_pair_effect"], old["accounting"])
                branches[branch] = {
                    "hidden_reference": old["hidden_reference"],
                    "rmsnorm_reference": old["rmsnorm_reference"],
                    "accounting": account,
                }
                branch_accounts[branch].append(account)
            pairs.append({"left_id": ids[0], "right_id": ids[1],
                          "roles": pair["roles"], "branches": branches})
        controls.append({"control": label, "pairs": pairs})
    pair_count = sum(len(c["pairs"]) for c in controls)
    same(audit["local_exact_row_dot_computations"], len(cache), "selected-row arithmetic count changed")
    same(audit["selected_row_scalar_products"], WIDTH*len(cache), "scalar product count changed")
    return {
        "control_count": len(controls), "diagnostic_pair_count": pair_count,
        "pair_branch_count": pair_count*len(BRANCHES),
        "selected_row_ids": sorted(expected_ids),
        "local_exact_row_dot_computations": len(cache),
        "selected_row_scalar_products": WIDTH*len(cache),
        "cache_scope": "one exact dot per control/selected row or independent reference branch/selected row",
        "identity": "retained_margin_change = vector_coordinate_sum + actual_head_boundary - reference_head_boundary",
        "boundary_identity": "head_boundary_remainder_change = actual_head_boundary + negative_reference_head_boundary",
        "branch_totals": {
            b: {
                "pair_accounts": len(accounts),
                "component_totals": {
                    k: mass(Fraction(a[k]) for a in accounts)
                    for k in (*COMPONENTS, "vector_coordinate_sum", "retained_margin_change",
                              "head_boundary_remainder_change")
                },
                "boundary_absolute_account": str(sum(
                    (Fraction(a["boundary_accounting"]["absolute"]) for a in accounts), Fraction())),
                "within_pair_boundary_cancellation": str(sum(
                    (Fraction(a["boundary_accounting"]["cancellation_absolute_mass"])
                     for a in accounts), Fraction())),
                "scope": "repeated ordered diagnostic pairs, not independent samples or causal shares",
            } for b, accounts in branch_accounts.items()
        },
        "lineage_separation": contributions.LINEAGE,
        "mixed_rmsnorm_input_binding": evidence["mixed_input_binding"],
        "retained_final_rmsnorm_logit_margin_bridge": upstream,
        "controls": controls,
    }


def measure(audit):
    for pin in (*margin.rows.PINS.values(), mixed.CUTOFF_PIN):
        base.read_bound(pin)
    evidence = hidden.authenticate()
    final = evidence["result"]["preflight"]["final_reference"]
    manifest = json.loads(base.read_bound(final["manifest"]))
    same(manifest["arithmetic"]["runtime"],
         {"torch": mixed.torch.__version__, "numpy": mixed.np.__version__},
         "reviewed mixed-RMSNorm CPU numerical runtime changed")
    evidence["mixed_input_binding"] = {
        "final_reference_manifest": final["manifest"],
        "runtime": manifest["arithmetic"]["runtime"],
        "source": parent.record(mixed.SOURCE), "test": parent.record(mixed.TEST),
        "scope": "same authenticated retained actual/reference RMSNorm and logits; no mixed-head recomputation",
    }
    evidence["hidden_report"] = hidden.report(evidence)
    evidence["geometry"] = margin.cutoff.report(
        evidence["result"], evidence["arrays"], evidence["references"])
    ids = {i for c in evidence["geometry"]["controls"]
           for pair in contributions.pairs_for(c) for i in pair}
    evidence["rows"] = contributions.load_rows(evidence["assets"], ids)
    evidence["margin_report"] = margin.report(evidence)
    evidence["report"] = report(evidence, audit)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("head_accumulation_boundary_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "head boundary focused tests failed, errored or skipped")
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
               "head_accumulation_boundary_test": parent.record(TEST),
               "mixed_rmsnorm_test": parent.record(mixed.TEST)}
    audit = {"forbidden_calls": 0, "local_exact_row_dot_computations": 0,
             "selected_row_scalar_products": 0}
    with read_only(audit):
        evidence = measure(audit)
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit["forbidden_calls"], 0, "head boundary bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_HEAD_ACCUMULATION_BOUNDARY_RESIDUAL_MARGIN_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "dispatch_and_write_audit": {**FLAGS, **audit}, "flags": FLAGS,
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
