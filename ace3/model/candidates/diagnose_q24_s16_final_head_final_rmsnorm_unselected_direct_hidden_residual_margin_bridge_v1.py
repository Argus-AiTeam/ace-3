"""Read-only retained unselected direct-hidden residual margin bridge; stdout only."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_coordinate_residual_margin_bridge_v1 as unselected


boundary, interaction, scale, direct, margin = (
    unselected.boundary, unselected.interaction, unselected.scale,
    unselected.direct, unselected.margin)
hidden, residual, base, parent = (
    unselected.hidden, unselected.residual, unselected.base, unselected.parent)
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_residual_margin_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, EXPECTED_TESTS = 896, 20
BRANCHES, COMPONENTS = direct.BRANCHES, direct.COMPONENTS
require, same, mass = base.require, base.same, direct.mass
FLAGS = {**unselected.FLAGS, "unselected_direct_hidden_causal_allocation": False}
SELECTION_RULE = (
    "Only after the shared-pair parent gate selects direct_hidden, split that "
    "category over its unchanged complement. Select a residual component only "
    "if uniquely largest in absolute NET SIGNED correction in all nine controls "
    "and both separate branches, with the same nonzero direction in every row. "
    "Each row must either have that direction in the unselected direct_hidden "
    "aggregate and strictly exceed half its magnitude, or strictly reverse the "
    "full retained_actual_margin minus retained_reference_margin CHANGE after "
    "subtracting only this component. Selected coordinates, other unselected "
    "categories and the head boundary remain fixed. Zero is not reversal. "
    "Ties, incomplete census, unstable direction or a failed row give UNKNOWN. "
    "Continue only if every shared numeric pair selects the same component. "
    "Nonshared pairs remain UNKNOWN. This is accounting-only neutralization, "
    "not an actual pair-order/ranking flip, token choice or executed counterfactual."
)
BOUNDARY = (
    "Only the prior unselected direct_hidden category is split into retained L23 "
    "input_hidden, attention_stage11, mlp_stage17, actual_residual_boundary, "
    "negative_reference_residual_boundary, q24_to_fp16_conversion and "
    "fp16_to_branch_terminal_remainder. All selected accounts including 62, "
    "global_scale/interaction/boundary unselected accounts and head remainders "
    "are preserved. Signed corrections are not absolute accounts; cancellation "
    "within coordinates, across coordinates and between net components is explicit. "
    + direct.REFERENCE_SCOPE + " " + hidden.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("unselected direct-hidden bridge forbids replay, prior checks and writes")

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


def component_vectors(actual, reference, terminal):
    require(len(terminal) == WIDTH and all(isinstance(v, Fraction) for v in terminal),
            "invalid independent branch terminal")
    vectors = {key: [] for key in COMPONENTS}
    for i in range(WIDTH):
        local = residual.account(actual, reference, i)
        parts = local["actual_q24_boundary_parts"]
        values = (
            actual["input_hidden"][i] - reference["input_hidden"][i],
            actual["stage11"][i] - reference["stage11"][i],
            actual["stage17"][i] - reference["stage17"][i],
            sum((Fraction(parts[k]) for k in residual.ACTUAL_PARTS[:-1]), Fraction()),
            -Fraction(local["reference_boundary_remainder"]),
            actual["stage18"][i] - actual["output_q24"][i],
            reference["stage18"][i] - terminal[i],
        )
        require(sum(values, Fraction()) == actual["stage18"][i] - terminal[i],
                "retained residual component identity changed")
        for key, value in zip(COMPONENTS, values, strict=True):
            vectors[key].append(value)
    return vectors


def split_unselected(components, weights, left, right, anchor, selected, upstream, retained):
    same(list(components), list(COMPONENTS), "residual component census changed")
    require(all(len(v) == WIDTH and all(isinstance(x, Fraction) for x in v)
                for v in (*components.values(), weights, left, right)),
            "invalid exact residual or tied-row vector")
    require(isinstance(anchor, Fraction) and anchor > 0, "invalid reference scalar anchor")
    require(all(type(i) is int and 0 <= i < WIDTH for i in selected)
            and selected == sorted(set(selected)) and 62 in selected
            and 8 <= len(selected) <= 25, "selected coordinate union changed")
    excluded = set(selected)
    values = {key: [] for key in COMPONENTS}
    corrections = []
    for i in range(WIDTH):
        if i in excluded:
            continue
        factor = weights[i] * anchor * (left[i] - right[i])
        terms = [components[key][i] * factor for key in COMPONENTS]
        for key, value in zip(COMPONENTS, terms, strict=True):
            values[key].append(value)
        corrections.append(sum(terms, Fraction()))
    totals = {key: mass(v) for key, v in values.items()}
    target = mass(corrections)
    same(target, upstream["component_totals"]["direct_hidden"],
         "unselected direct_hidden signed/absolute account changed")
    absolute = sum((Fraction(v["absolute"]) for v in totals.values()), Fraction())
    net_absolute = sum((abs(Fraction(v["signed"])) for v in totals.values()), Fraction())
    signed, direct_absolute = Fraction(target["signed"]), Fraction(target["absolute"])
    require(sum((Fraction(v["signed"]) for v in totals.values()), Fraction()) == signed
            and absolute >= direct_absolute >= abs(signed), "residual aggregate closure failed")
    return {
        "coordinate_count": WIDTH-len(selected),
        "excluded_selected_coordinates": selected,
        "component_totals": totals, "unselected_direct_hidden": target,
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute-direct_absolute),
        "across_coordinate_cancellation_mass": target["cancellation_absolute_mass"],
        "total_component_cancellation_mass": str(absolute-abs(signed)),
        "within_component_across_coordinate_cancellation_mass": str(absolute-net_absolute),
        "between_component_net_cancellation_mass": str(net_absolute-abs(signed)),
        "neutralized_retained_margin_change": {
            key: str(Fraction(retained)-Fraction(v["signed"])) for key, v in totals.items()},
        "exact_unselected_direct_hidden_identity": True,
    }


def select_component(rows):
    expected = [(c, b) for c in parent.CONTROLS for b in BRANCHES]
    complete = [(r["control"], r["branch"]) for r in rows] == expected
    evaluations = []
    for key in COMPONENTS:
        directions, checks = [], []
        for row in rows:
            values = {k: Fraction(v["signed"]) for k, v in row["component_totals"].items()}
            same(list(values), list(COMPONENTS), "selection component census changed")
            retained = Fraction(row["retained_margin_change"])
            require(Fraction(row["retained_actual_margin"])
                    - Fraction(row["retained_reference_margin"]) == retained,
                    "retained actual-minus-reference margin splice")
            correction = values[key]
            neutralized = retained-correction
            same(str(neutralized), row["neutralized_retained_margin_change"][key],
                 "neutralized retained margin change splice")
            remainder = Fraction(row["unselected_direct_hidden"]["signed"])
            largest = all(abs(correction) > abs(v) for k, v in values.items() if k != key)
            half = correction*remainder > 0 and 2*abs(correction) > abs(remainder)
            reversal = retained*neutralized < 0
            directions.append((correction > 0)-(correction < 0))
            checks.append({
                "control": row["control"], "branch": row["branch"],
                "signed_correction": str(correction),
                "absolute_account": row["component_totals"][key]["absolute"],
                "unique_largest_net_signed_magnitude": largest,
                "explains_more_than_half": half,
                "retained_margin_change_sign_reversed": reversal,
                "neutralized_retained_margin_change": str(neutralized),
                "row_qualifies": largest and (half or reversal),
            })
        stable = bool(directions) and len(set(directions)) == 1 and directions[0] != 0
        for row in checks:
            row["directionally_stable"] = stable
            row["row_qualifies"] = row["row_qualifies"] and stable and complete
        evaluations.append({
            "component": key, "directionally_stable": stable,
            "qualifies": complete and stable and all(r["row_qualifies"] for r in checks),
            "rows": checks,
        })
    chosen = [e["component"] for e in evaluations if e["qualifies"]]
    require(len(chosen) <= 1, "nonunique residual component selection")
    return {
        "complete_nine_control_two_branch_census": complete,
        "component": chosen[0] if chosen else "UNKNOWN",
        "stop_nested_bridge_expansion": not chosen,
        "component_evaluations": evaluations,
    }


def common_selection(decisions):
    shared = [d["component"] for d in decisions if d["complete_nine_control_two_branch_census"]]
    return shared[0] if shared and len(set(shared)) == 1 else "UNKNOWN"


def report(evidence):
    upstream = evidence["unselected_report"]
    same(upstream["retained_final_rmsnorm_boundary_residual_margin_bridge"],
         evidence["boundary_report"], "unselected/boundary bridge linkage splice")
    base.check_history(evidence["result"])
    same([c["control"] for c in upstream["controls"]], list(parent.CONTROLS),
         "unselected control census changed")
    groups = {}
    for control in upstream["controls"]:
        for pair in control["pairs"]:
            same(list(pair["branches"]), list(BRANCHES), "unselected branch census changed")
            for branch, old in pair["branches"].items():
                groups.setdefault((pair["left_id"], pair["right_id"]), []).append(
                    {"control": control["control"], "branch": branch, **old})
    parent_decisions = [{"left_id": key[0], "right_id": key[1], **unselected.select_mechanism(rows)}
                        for key, rows in sorted(groups.items())]
    same(parent_decisions, upstream["selection"]["pairs"], "parent selection splice")
    shared = [d["mechanism"] for d in parent_decisions if d["complete_nine_control_two_branch_census"]]
    require(shared and set(shared) == {"direct_hidden"}
            and upstream["selection"]["mechanism"] == "direct_hidden"
            and upstream["selection"]["stop_nested_bridge_expansion"] is False,
            "parent gate does not authorize nested direct_hidden localization")
    reference = residual.operands(evidence["reference_archive"], actual=False)
    row_vectors = {i: margin.contributions.exact_vector(v) for i, v in evidence["rows"].items()}
    controls, groups, count = [], {}, 0
    for control, mc, dc in zip(upstream["controls"], evidence["margin_report"]["controls"],
                               evidence["direct_report"]["controls"], strict=True):
        label = control["control"]
        same((mc["control"], dc["control"]), (label, label), "retained control splice")
        vectors = {b: component_vectors(evidence["actual"][label], reference, evidence["hidden"][b])
                   for b in BRANCHES}
        pairs = []
        for pair, mp, dp in zip(control["pairs"], mc["pairs"], dc["pairs"], strict=True):
            identity = tuple(pair[k] for k in ("left_id", "right_id", "roles"))
            for old_pair in (mp, dp):
                same(tuple(old_pair[k] for k in ("left_id", "right_id", "roles")), identity,
                     "retained ordered pair splice")
            left, right = pair["left_id"], pair["right_id"]
            branches = {}
            for branch, old in pair["branches"].items():
                original = mp["branches"][branch]
                same(old["selected_coordinates"], original["selected_coordinates"], "selected union splice")
                same(old["unchanged_margin_accounting"], original["accounting"], "retained margin splice")
                same(old["hidden_reference"], "original_input_L23_"+branch, "hidden branch splice")
                same(old["rmsnorm_reference"], "original_input_final_attempt003_"+branch,
                     "final reference splice")
                effect, accounting = original["retained_pair_effect"], original["accounting"]
                actual_margin, reference_margin = (
                    Fraction(effect[k]) for k in ("retained_actual_margin", "retained_reference_margin"))
                retained = actual_margin-reference_margin
                same(str(retained), accounting["retained_margin_change"], "retained margin change splice")
                indices = [r["coordinate"] for r in old["selected_coordinates"]]
                split = split_unselected(
                    vectors[branch], evidence["weights"], row_vectors[left], row_vectors[right],
                    Fraction(old["reference_norm"]["inverse_norm_anchor"]), indices,
                    old["unselected_accounting"], retained)
                unselected_account = old["unselected_accounting"]
                total = Fraction(split["unselected_direct_hidden"]["signed"]) + sum(
                    (Fraction(unselected_account["component_totals"][k]["signed"])
                     for k in hidden.TERMS if k != "direct_hidden"), Fraction())
                require(total == Fraction(accounting["unselected_coordinate_signed_remainder"])
                        and total+Fraction(accounting["selected_coordinate_signed_sum"])
                        +Fraction(accounting["head_boundary_remainder_change"]) == retained,
                        "selected/unselected/head margin closure changed")
                row = {
                    "control": label, "branch": branch,
                    "retained_actual_margin": str(actual_margin),
                    "retained_reference_margin": str(reference_margin),
                    "retained_margin_change": str(retained),
                    "selected_coordinate_signed_sum": accounting["selected_coordinate_signed_sum"],
                    "head_boundary_remainder_change": accounting["head_boundary_remainder_change"],
                    "unselected_coordinate_remainder": unselected_account["unselected_coordinate_remainder"],
                    "unchanged_unselected_categories": {
                        k: v for k, v in unselected_account["component_totals"].items()
                        if k != "direct_hidden"},
                    **split,
                }
                branches[branch] = {
                    "gate_values": row,
                    "unchanged_selected_direct_hidden": dp["branches"][branch],
                    "unchanged_unselected_bridge": old,
                    "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
                }
                groups.setdefault((left, right), []).append(row)
                count += split["coordinate_count"]
            pairs.append({"left_id": left, "right_id": right, "roles": pair["roles"], "branches": branches})
        controls.append({"control": label, "pairs": pairs})
    same(count, upstream["unselected_coordinate_accounts"], "unselected coordinate census changed")
    decisions = [{"left_id": k[0], "right_id": k[1], **select_component(v)}
                 for k, v in sorted(groups.items())]
    common = common_selection(decisions)
    return {
        "control_count": len(controls),
        "diagnostic_pair_count": upstream["diagnostic_pair_count"],
        "pair_branch_count": upstream["pair_branch_count"],
        "coordinate62_accounts": upstream["coordinate62_accounts"],
        "selected_coordinate_accounts": upstream["selected_coordinate_accounts"],
        "unselected_coordinate_accounts": count, "weighted_residual_component_count": count*len(COMPONENTS),
        "reference_scope": direct.REFERENCE_SCOPE, "lineage_separation": upstream["lineage_separation"],
        "retained_final_rmsnorm_unselected_coordinate_residual_margin_bridge": upstream,
        "selection": {
            "rule": SELECTION_RULE, "parent_category": "direct_hidden",
            "scope": "existing shared numeric pairs, nine controls and P0 only",
            "component": common, "stop_nested_bridge_expansion": common == "UNKNOWN",
            "pairs": decisions,
        },
        "controls": controls,
    }


def measure():
    for pin in margin.rows.PINS.values():
        base.read_bound(pin)
    evidence = hidden.authenticate()
    evidence["hidden_report"] = hidden.report(evidence)
    evidence["geometry"] = margin.cutoff.report(evidence["result"], evidence["arrays"], evidence["references"])
    ids = {i for control in evidence["geometry"]["controls"]
           for pair in margin.contributions.pairs_for(control) for i in pair}
    evidence["rows"] = margin.contributions.load_rows(evidence["assets"], ids)
    for key, module in (
        ("margin_report", margin), ("direct_report", direct), ("scalar_report", scale),
        ("interaction_report", interaction), ("boundary_report", boundary),
        ("unselected_report", unselected),
    ):
        evidence[key] = module.report(evidence)
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("unselected_direct_hidden_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "unselected direct-hidden tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "unselected_direct_hidden_tests": parent.record(TEST)}
    for module in (unselected, boundary, interaction, scale, direct, margin, hidden, residual):
        origins[module.MODULE] = parent.record(module.SOURCE)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *margin.rows.PINS.values(), *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "unselected direct-hidden attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_UNSELECTED_DIRECT_HIDDEN_RESIDUAL_MARGIN_BRIDGE",
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
