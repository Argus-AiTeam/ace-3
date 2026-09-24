"""Read-only retained unselected-coordinate RMSNorm margin accounting; stdout only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_boundary_residual_margin_bridge_v1 as boundary


interaction, scale, direct, margin = (
    boundary.interaction, boundary.scale, boundary.direct, boundary.margin)
hidden, residual, base, parent = (
    boundary.hidden, boundary.residual, boundary.base, boundary.parent)
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_coordinate_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 20
BRANCHES, TERMS = hidden.BRANCHES, hidden.TERMS
require, same, mass = base.require, base.same, direct.mass
FLAGS = {**boundary.FLAGS, "unselected_split_causal_allocation": False,
         "counterfactual_operator_replay": 0}
SELECTION_RULE = (
    "For each existing numeric pair present in all nine controls, require one "
    "unique largest absolute NET SIGNED category in both FP16 and binary64, "
    "with the same nonzero direction across every control (including mapped_all "
    "and inherited variants). In every row it must either explain strictly more "
    "than half of the unselected signed remainder in that remainder's direction, "
    "or strictly reverse the retained-margin-change sign when its signed aggregate "
    "is subtracted, holding selected terms, other unselected terms and head boundary "
    "fixed. Zero is not a sign reversal. This neutralization is an accounting "
    "counterfactual, not a recomputed operator/token margin. Ties, missing controls "
    "or failed conditions mean UNKNOWN; stop nested bridge expansion. A common "
    "mechanism is selected only if all shared pairs select the same category. "
    "Nonshared pairs remain UNKNOWN and are not certified by that selection."
)
BOUNDARY = (
    "Only unselected_coordinate_signed_remainder and "
    "unselected_coordinate_absolute_remainder are resolved over the complement "
    "of each already-selected coordinate union. Aggregate direct_hidden, "
    "global_scale, interaction and boundary_remainder_delta keep signed and "
    "absolute mass separate, with explicit within/across-coordinate cancellation. "
    "Selected accounts including coordinate 62 and head-boundary remainders stay "
    "unchanged. Mechanism selection is bounded accounting localization, not "
    "upstream causality, a dominant stage, or a performance diagnosis. The retained "
    "P0-only probability conclusion is unchanged. " + direct.REFERENCE_SCOPE + " "
    + hidden.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("unselected bridge forbids operators, prior checks and writes")

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


def split_unselected(actual, reference, output, reference_output, weights,
                     left, right, sa, sr, selected, upstream):
    vectors = (actual, reference, output, reference_output, weights, left, right)
    require(all(len(v) == WIDTH and all(isinstance(x, Fraction) for x in v)
                for v in vectors), "invalid exact retained vector")
    require(all(isinstance(s, Fraction) and s > 0 for s in (sa, sr)),
            "invalid retained scalar anchor")
    require(all(type(i) is int and 0 <= i < WIDTH for i in selected)
            and selected == sorted(set(selected)) and 62 in selected
            and 8 <= len(selected) <= 25, "selected coordinate union changed")
    excluded = set(selected)
    indices = [i for i in range(WIDTH) if i not in excluded]
    components = {term: [] for term in TERMS}
    deltas = []
    for i in indices:
        a, r, ya, yr, w, wl, wr = (v[i] for v in vectors)
        dw, dh, ds = wl-wr, a-r, sa-sr
        # Boundary terms use retained outputs, never an RMSNorm implementation.
        terms = (dw*w*dh*sr, dw*w*r*ds, dw*w*dh*ds,
                 dw*((ya-w*a*sa)-(yr-w*r*sr)))
        delta = dw*(ya-yr)
        require(sum(terms, Fraction()) == delta, "unselected coordinate identity failed")
        for term, value in zip(TERMS, terms, strict=True):
            components[term].append(value)
        deltas.append(delta)
    totals = {term: mass(values) for term, values in components.items()}
    target = mass(deltas)
    same(target["signed"], upstream["unselected_coordinate_signed_remainder"],
         "unselected signed remainder changed")
    same(target["absolute"], upstream["unselected_coordinate_absolute_remainder"],
         "unselected absolute remainder changed")
    signed = sum((Fraction(t["signed"]) for t in totals.values()), Fraction())
    absolute = sum((Fraction(t["absolute"]) for t in totals.values()), Fraction())
    net_absolute = sum((abs(Fraction(t["signed"])) for t in totals.values()), Fraction())
    require(signed == Fraction(target["signed"])
            and absolute >= Fraction(target["absolute"]) >= abs(signed),
            "unselected aggregate closure failed")
    retained = Fraction(upstream["retained_margin_change"])
    require(Fraction(upstream["selected_coordinate_signed_sum"])+signed
            +Fraction(upstream["head_boundary_remainder_change"]) == retained,
            "selected/unselected/head closure changed")
    return {
        "coordinate_count": len(indices), "excluded_selected_coordinates": selected,
        "component_totals": totals, "unselected_coordinate_remainder": target,
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute-Fraction(target["absolute"])),
        "across_coordinate_cancellation_mass": target["cancellation_absolute_mass"],
        "total_component_cancellation_mass": str(absolute-abs(signed)),
        "within_category_across_coordinate_cancellation_mass": str(absolute-net_absolute),
        "between_category_net_cancellation_mass": str(net_absolute-abs(signed)),
        "neutralized_retained_margin_change": {
            term: str(retained-Fraction(totals[term]["signed"])) for term in TERMS},
        "exact_unselected_identity": True, "exact_margin_identity": True,
    }


def select_mechanism(rows):
    expected = [(c, b) for c in parent.CONTROLS for b in BRANCHES]
    complete = [(r["control"], r["branch"]) for r in rows] == expected
    evaluations = []
    for term in TERMS:
        directions, checks = [], []
        for row in rows:
            account, original = row["unselected_accounting"], row["unchanged_margin_accounting"]
            values = {k: Fraction(v["signed"]) for k, v in account["component_totals"].items()}
            correction = values[term]
            remainder = Fraction(original["unselected_coordinate_signed_remainder"])
            retained = Fraction(original["retained_margin_change"])
            neutralized = retained-correction
            same(str(neutralized), account["neutralized_retained_margin_change"][term],
                 "neutralized margin splice")
            largest = abs(correction) > max(abs(v) for k, v in values.items() if k != term)
            majority = correction*remainder > 0 and abs(correction)*2 > abs(remainder)
            reversal = retained*neutralized < 0
            directions.append((correction > 0)-(correction < 0))
            checks.append({
                "control": row["control"], "branch": row["branch"],
                "signed_correction": str(correction),
                "unique_largest_net_signed_magnitude": largest,
                "signed_remainder_share": str(correction/remainder) if remainder else None,
                "explains_more_than_half": majority,
                "neutralized_retained_margin_change": str(neutralized),
                "retained_margin_sign_reversed": reversal,
                "row_qualifies": largest and (majority or reversal),
            })
        stable = bool(directions) and len(set(directions)) == 1 and directions[0] != 0
        evaluations.append({
            "category": term, "directionally_stable": stable,
            "qualifies": complete and stable and all(r["row_qualifies"] for r in checks),
            "rows": checks,
        })
    selected = [e["category"] for e in evaluations if e["qualifies"]]
    require(len(selected) <= 1, "nonunique mechanism selection")
    return {
        "complete_nine_control_two_branch_census": complete,
        "mechanism": selected[0] if selected else "UNKNOWN",
        "stop_nested_bridge_expansion": not selected,
        "category_evaluations": evaluations,
    }


def report(evidence):
    upstream, retained = evidence["margin_report"], evidence["boundary_report"]
    for outer, key, inner in (
        (retained, "retained_final_rmsnorm_interaction_residual_margin_bridge", evidence["interaction_report"]),
        (evidence["interaction_report"], "retained_final_rmsnorm_scalar_scale_residual_margin_bridge",
         evidence["scalar_report"]),
        (evidence["scalar_report"], "retained_final_rmsnorm_direct_hidden_residual_margin_bridge",
         evidence["direct_report"]),
        (evidence["direct_report"], "retained_final_rmsnorm_logit_margin_bridge", upstream),
        (upstream, "hidden_delta_bridge", evidence["hidden_report"]),
    ):
        same(outer[key], inner, "retained bridge linkage splice")
    for old in (upstream, retained, evidence["hidden_report"], evidence["interaction_report"],
                evidence["scalar_report"], evidence["direct_report"]):
        same([c["control"] for c in old["controls"]], list(parent.CONTROLS),
             "unselected control census changed")
    for key in ("control_count", "diagnostic_pair_count", "pair_branch_count",
                "coordinate62_accounts", "selected_coordinate_accounts"):
        same(retained[key], upstream[key], "retained account census changed")
    base.check_history(evidence["result"])
    refs = {
        "fp16": residual.words(evidence["references"]["rmsnorm_fp16"]),
        "binary64": margin.contributions.exact_vector(evidence["references"]["rmsnorm_binary64"]),
    }
    row_vectors = {i: margin.contributions.exact_vector(v) for i, v in evidence["rows"].items()}
    controls, groups, count = [], {}, 0
    for control, bc, hc in zip(upstream["controls"], retained["controls"],
                               evidence["hidden_report"]["controls"], strict=True):
        label = control["control"]
        actual = evidence["actual"][label]["stage18"]
        output = residual.words(evidence["arrays"][label]["rmsnorm"])
        sa = Fraction(hc["actual_norm"]["inverse_norm_anchor"])
        pairs = []
        for pair, bp in zip(control["pairs"], bc["pairs"], strict=True):
            same(tuple(bp[k] for k in ("left_id", "right_id", "roles")),
                 tuple(pair[k] for k in ("left_id", "right_id", "roles")), "pair identity splice")
            same(list(pair["branches"]), list(BRANCHES), "margin reference census changed")
            same(list(bp["branches"]), list(BRANCHES), "boundary reference census changed")
            key = (pair["left_id"], pair["right_id"])
            branches = {}
            for branch in BRANCHES:
                original, old = pair["branches"][branch], bp["branches"][branch]
                same(old["unchanged_margin_accounting"], original["accounting"],
                     "selected/unselected/head account splice")
                same([(r["coordinate"], r["selection_reasons"]) for r in old["selected_coordinates"]],
                     [(r["coordinate"], r["selection_reasons"]) for r in original["selected_coordinates"]],
                     "selected union splice")
                same(original["hidden_reference"], "original_input_L23_"+branch,
                     "hidden reference identity changed")
                same(original["rmsnorm_reference"], "original_input_final_attempt003_"+branch,
                     "final reference identity changed")
                same(old["reference_norm"], hc["branches"][branch]["reference_norm"],
                     "scalar anchor splice")
                indices = [r["coordinate"] for r in original["selected_coordinates"]]
                account = split_unselected(
                    actual, evidence["hidden"][branch], output, refs[branch], evidence["weights"],
                    row_vectors[key[0]], row_vectors[key[1]], sa,
                    Fraction(old["reference_norm"]["inverse_norm_anchor"]), indices, original["accounting"])
                branches[branch] = {
                    "hidden_reference": original["hidden_reference"],
                    "rmsnorm_reference": original["rmsnorm_reference"],
                    "reference_norm": old["reference_norm"],
                    "selected_coordinates": original["selected_coordinates"],
                    "unchanged_margin_accounting": original["accounting"],
                    "unselected_accounting": account,
                }
                groups.setdefault(key, []).append({
                    "control": label, "branch": branch,
                    "unselected_accounting": account,
                    "unchanged_margin_accounting": original["accounting"],
                })
                count += account["coordinate_count"]
            pairs.append({"left_id": key[0], "right_id": key[1], "roles": pair["roles"],
                          "branches": branches})
        controls.append({"control": label, "actual_norm": hc["actual_norm"], "pairs": pairs})
    decisions = [{"left_id": key[0], "right_id": key[1], **select_mechanism(rows)}
                 for key, rows in sorted(groups.items())]
    shared = [d["mechanism"] for d in decisions if d["complete_nine_control_two_branch_census"]]
    mechanism = shared[0] if shared and len(set(shared)) == 1 else "UNKNOWN"
    same(count+upstream["selected_coordinate_accounts"], WIDTH*upstream["pair_branch_count"],
         "selected/complement census changed")
    return {
        "control_count": len(controls), "diagnostic_pair_count": upstream["diagnostic_pair_count"],
        "pair_branch_count": upstream["pair_branch_count"],
        "coordinate62_accounts": upstream["coordinate62_accounts"],
        "selected_coordinate_accounts": upstream["selected_coordinate_accounts"],
        "unselected_coordinate_accounts": count, "weighted_unselected_term_count": count*len(TERMS),
        "identity": "sum_unselected(dW*(direct_hidden+global_scale+interaction+boundary_remainder_delta))",
        "absolute_identity": "sum_unselected(abs(sum_four_terms)), not sum_unselected(sum(abs(terms)))",
        "lineage_separation": upstream["lineage_separation"],
        "reference_scope": direct.REFERENCE_SCOPE,
        "retained_final_rmsnorm_boundary_residual_margin_bridge": retained,
        "selection": {
            "rule": SELECTION_RULE, "scope": "shared numeric pairs, current P0 and nine controls only",
            "mechanism": mechanism, "stop_nested_bridge_expansion": mechanism == "UNKNOWN",
            "pairs": decisions,
        },
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
    for key, module in (
        ("margin_report", margin), ("direct_report", direct), ("scalar_report", scale),
        ("interaction_report", interaction), ("boundary_report", boundary),
    ):
        evidence[key] = module.report(evidence)
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("unselected_residual_margin_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "unselected bridge focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "unselected_residual_margin_test": parent.record(TEST)}
    for module in (boundary, interaction, scale, direct, margin, hidden, residual):
        origins[module.MODULE] = parent.record(module.SOURCE)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "unselected bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_UNSELECTED_COORDINATE_RESIDUAL_MARGIN_BRIDGE",
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
