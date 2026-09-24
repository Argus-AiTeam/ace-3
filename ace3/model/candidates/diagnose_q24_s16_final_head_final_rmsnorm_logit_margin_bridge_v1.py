"""Read-only retained final RMSNorm term-to-logit-margin accounting; stdout only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_hidden_delta_bridge_v1 as hidden
from ace3.model.candidates import diagnose_q24_s16_final_head_row_difference_hotspots_v1 as rows


hotspots, contributions = hidden.hotspots, hidden.contributions
base, parent, cutoff = hidden.base, hidden.parent, hotspots.cutoff
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_logit_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, EXPECTED_TESTS = 896, 8, 20
BRANCHES, TERMS = hidden.BRANCHES, hidden.TERMS
require, same = base.require, base.same
FLAGS = {**hidden.FLAGS, "row_dot_operator_replay": 0,
         "token_decode_or_publication": False}
BOUNDARY = (
    "Read-only CPU-software local final RMSNorm-to-logit-margin accounting for "
    "existing cutoff/exchanged numeric diagnostic pairs, not new token selection. "
    "Selected hidden bridge terms are multiplied by authenticated tied-head row "
    "differences. Unselected coordinate products and head-boundary remainders stay "
    "separate; signed effects and cancellation are not causal percentages. "
    "No RMSNorm, head, row-dot or other operator is replayed. " + hidden.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("margin bridge forbids operator and prior-check replay")

    with hotspots.read_only(audit), ExitStack() as stack:
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


def weighted_account(account, left, right):
    require(isinstance(left, Fraction) and isinstance(right, Fraction),
            "row difference operands must be exact")
    terms = {key: Fraction(account[key]) for key in TERMS}
    delta = Fraction(account["retained_rmsnorm_delta"])
    require(sum(terms.values(), Fraction()) == delta, "hidden term identity changed")
    difference = left - right
    products = {key: value * difference for key, value in terms.items()}
    observed = delta * difference
    require(observed == delta * left - delta * right
            and sum(products.values(), Fraction()) == observed,
            "weighted hidden bridge identity failed")
    absolute = sum(map(abs, products.values()), Fraction())
    return {
        "coordinate": account["coordinate"],
        "left_weight": str(left), "right_weight": str(right),
        "row_difference": str(difference), "hidden_bridge": account,
        "weighted_terms": {key: str(value) for key, value in products.items()},
        "weighted_actual_boundary_conversion_remainder": str(
            Fraction(account["actual_boundary_conversion_remainder"]) * difference),
        "weighted_reference_boundary_conversion_remainder": str(
            Fraction(account["reference_boundary_conversion_remainder"]) * difference),
        "coordinate_margin_change": str(observed),
        "absolute_term_sum": str(absolute),
        "cancellation_absolute_mass": str(absolute - abs(observed)),
        "exact_weighted_identity": True,
    }


def selected_coordinates(hidden_branch, effect, row_order):
    reasons = {62: ["required_coordinate_62"]}
    for reason, indices in (
        ("retained_rmsnorm_delta_hotspot",
         [row["coordinate"] for row in hidden_branch["hotspots"]]),
        ("pair_margin_delta_hotspot",
         [row["coordinate"] for row in effect["largest_absolute_signed"]]),
        ("tied_head_row_difference_hotspot", row_order[:LIMIT]),
    ):
        for i in indices:
            reasons.setdefault(i, []).append(reason)
    return {i: sorted(reasons[i]) for i in sorted(reasons)}


def margin_account(selected, effect):
    products = [Fraction(row["coordinate_margin_change"]) for row in selected]
    signed = sum(products, Fraction())
    absolute = sum(map(abs, products), Fraction())
    term_totals = {
        key: {
            "signed": str(sum((Fraction(row["weighted_terms"][key])
                               for row in selected), Fraction())),
            "absolute": str(sum((abs(Fraction(row["weighted_terms"][key]))
                                 for row in selected), Fraction())),
        } for key in TERMS
    }
    total, full_absolute = (Fraction(effect[key]) for key in
                            ("exact_sum", "exact_sum_of_absolute_contributions"))
    unselected, unselected_absolute = total - signed, full_absolute - absolute
    boundary = Fraction(effect["boundary_remainder_change"])
    retained = Fraction(effect["retained_margin_change"])
    require(unselected_absolute >= abs(unselected)
            and sum(Fraction(row["signed"]) for row in term_totals.values()) == signed
            and signed + unselected + boundary == retained,
            "selected/unselected/head-boundary margin closure failed")
    term_absolute = sum(Fraction(row["absolute"]) for row in term_totals.values())
    return {
        "selected_term_totals": term_totals,
        "selected_coordinate_signed_sum": str(signed),
        "selected_coordinate_absolute_sum": str(absolute),
        "selected_coordinate_cancellation_mass": str(absolute - abs(signed)),
        "selected_term_absolute_sum": str(term_absolute),
        "selected_term_cancellation_mass": str(term_absolute - abs(signed)),
        "unselected_coordinate_signed_remainder": str(unselected),
        "unselected_coordinate_absolute_remainder": str(unselected_absolute),
        "full_coordinate_cancellation_mass": str(full_absolute - abs(total)),
        "head_boundary_remainder_change": str(boundary),
        "retained_margin_change": str(retained),
        "exact_margin_identity": True,
    }


def report(evidence):
    base.check_history(evidence["result"])
    same(list(evidence["arrays"]), list(parent.CONTROLS), "margin control census changed")
    hidden_report = evidence["hidden_report"]
    same([row["control"] for row in hidden_report["controls"]], list(parent.CONTROLS),
         "hidden bridge control order changed")
    geometry = evidence["geometry"]
    same([row["control"] for row in geometry["controls"]], list(parent.CONTROLS),
         "cutoff control order changed")
    controls, pair_count, account_count = [], 0, 0
    for control, bridge in zip(geometry["controls"], hidden_report["controls"], strict=True):
        label = control["control"]
        a = evidence["actual"][label]["stage18"]
        ya = hidden.residual.words(evidence["arrays"][label]["rmsnorm"])
        vectors = {
            "actual_fp16": ya,
            "fp16": hidden.residual.words(evidence["references"]["rmsnorm_fp16"]),
            "binary64": contributions.exact_vector(evidence["references"]["rmsnorm_binary64"]),
        }
        logits = {
            "actual_fp16": evidence["arrays"][label]["logits"].view("<f2"),
            "fp16": evidence["references"]["logits_fp16"].view("<f2"),
            "binary64": evidence["references"]["logits_binary64"],
        }
        sa = Fraction(bridge["actual_norm"]["inverse_norm_anchor"])
        energies = {b: hidden.energy_delta(a, evidence["hidden"][b]) for b in BRANCHES}
        pairs = []
        for (left, right), roles in contributions.pairs_for(control).items():
            pair = hotspots.pair_products(left, right, vectors, evidence["rows"], logits, roles)
            _, difference, wl, wr = rows.row_summary(
                evidence["rows"][left], evidence["rows"][right],
                {62: ["required_coordinate_62"]})
            row_order = sorted(range(WIDTH), key=lambda i: (-abs(difference[i]), i))
            branches = {}
            for branch in BRANCHES:
                hb = bridge["branches"][branch]
                r, yr = evidence["hidden"][branch], vectors[branch]
                sr = Fraction(hb["reference_norm"]["inverse_norm_anchor"])
                effect = pair["actual_minus_independent_reference"][branch]
                reasons = selected_coordinates(hb, effect, row_order)
                selected = []
                for i, why in reasons.items():
                    account = hidden.account(
                        a[i], r[i], ya[i], yr[i], evidence["weights"][i], sa, sr,
                        i, *energies[branch], evidence["actual"][label]["output_q24"][i])
                    selected.append({
                        **weighted_account(account, wl[i], wr[i]),
                        "selection_reasons": why,
                        "row_difference_absolute_rank": row_order.index(i) + 1,
                    })
                branches[branch] = {
                    "hidden_reference": hb["hidden_reference"],
                    "rmsnorm_reference": hb["rmsnorm_reference"],
                    "retained_pair_effect": effect,
                    "selected_coordinates": selected,
                    "accounting": margin_account(selected, effect),
                }
                account_count += len(selected)
            pairs.append({"left_id": left, "right_id": right, "roles": roles,
                          "row_difference_hotspots": row_order[:LIMIT], "branches": branches})
        controls.append({"control": label, "pairs": pairs})
        pair_count += len(pairs)
    return {
        "control_count": len(controls), "diagnostic_pair_count": pair_count,
        "pair_branch_count": pair_count * len(BRANCHES),
        "selected_coordinate_accounts": account_count,
        "weighted_term_count": account_count * len(TERMS),
        "coordinate62_accounts": pair_count * len(BRANCHES),
        "selection": (
            "union of existing top-eight absolute RMSNorm delta, pair-margin delta "
            "and row-difference coordinates plus 62; magnitude ties use ascending "
            "coordinate; union emitted in coordinate order, repeats across pairs retained"),
        "identity": (
            "delta_margin=sum_selected(dW*(direct_hidden+global_scale+interaction+"
            "boundary_remainder_delta))+unselected_coordinate_sum+head_boundary_remainder"),
        "remainder_scope": (
            "RMSNorm boundary/conversion terms include scalar-anchor and implementation "
            "differences, not rounding alone; head boundary is retained margin change "
            "minus all exact delta_y*dW products, not an operator replay or rounding-only cause"),
        "hidden_delta_bridge": hidden_report,
        "cutoff_margin_report": geometry,
        "lineage_separation": contributions.LINEAGE,
        "controls": controls,
    }


def measure():
    for pin in rows.PINS.values():
        base.read_bound(pin)
    evidence = hidden.authenticate()
    evidence["hidden_report"] = hidden.report(evidence)
    evidence["geometry"] = cutoff.report(
        evidence["result"], evidence["arrays"], evidence["references"])
    ids = {i for control in evidence["geometry"]["controls"]
           for pair in contributions.pairs_for(control) for i in pair}
    evidence["rows"] = contributions.load_rows(evidence["assets"], ids)
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("final_rmsnorm_margin_bridge_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "margin bridge focused tests failed, errored or skipped")
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
               "margin_bridge_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "margin bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_LOGIT_MARGIN_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **rows.PINS},
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
