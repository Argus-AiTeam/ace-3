"""Stdout-only exact gate census over the authenticated direct-hidden bridge."""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_residual_margin_bridge_v1 as bridge


base, parent = bridge.base, bridge.parent
require, same = base.require, base.same
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_gate_obstruction_census_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
CONTROLS, BRANCHES, COMPONENTS = parent.CONTROLS, bridge.BRANCHES, bridge.COMPONENTS
EXPECTED_TESTS = 16
FLAGS = {**bridge.FLAGS, "obstruction_census_causal_allocation": False}
GATES = (
    "unique_largest_net_signed_magnitude", "directionally_stable",
    "explains_more_than_half", "retained_margin_change_sign_reversed",
    "complete_nine_control_two_branch_census",
)
BOUNDARY = (
    "Declarative obstruction accounting over the existing authenticated bridge "
    "report interface, not a new residual split or an executed counterfactual. "
    "Half and reversal are alternatives, not two mandatory gates. Counts are "
    "over overlapping diagnostic rows, not independent samples or causal shares. "
    "An absent obstruction has count zero; UNKNOWN is not evidence that every "
    "gate fails. No prior check or native/decoder/prefix/admission/reference/"
    "RMSNorm/head/row-dot/local operator is replayed; no evidence is written. "
    + bridge.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("obstruction census forbids bridge regeneration, checks and operators")

    with bridge.read_only(audit), ExitStack() as stack:
        for name in ("check", "focused_tests", "measure", "report",
                     "component_vectors", "split_unselected"):
            stack.enter_context(patch.object(bridge, name, forbidden))
        yield


def exact(value):
    require(isinstance(value, str), "gate value is not an exact rational string")
    return Fraction(value)


def magnitude(account):
    signed, absolute = exact(account["signed"]), exact(account["absolute"])
    require(absolute >= abs(signed)
            and exact(account["cancellation_absolute_mass"]) == absolute-abs(signed),
            "signed/absolute/cancellation account splice")
    return signed, absolute


def row_values(row):
    same(list(row["component_totals"]), list(COMPONENTS), "component census changed")
    values, absolutes = {}, {}
    for key, account in row["component_totals"].items():
        values[key], absolutes[key] = magnitude(account)
    aggregate, aggregate_absolute = magnitude(row["unselected_direct_hidden"])
    absolute_sum = sum(absolutes.values(), Fraction())
    net_absolute_sum = sum(map(abs, values.values()), Fraction())
    require(sum(values.values(), Fraction()) == aggregate
            and absolute_sum >= aggregate_absolute, "component/aggregate closure splice")
    for key, expected in (
        ("component_absolute_sum", absolute_sum),
        ("within_coordinate_cancellation_mass", absolute_sum-aggregate_absolute),
        ("across_coordinate_cancellation_mass", aggregate_absolute-abs(aggregate)),
        ("total_component_cancellation_mass", absolute_sum-abs(aggregate)),
        ("within_component_across_coordinate_cancellation_mass", absolute_sum-net_absolute_sum),
        ("between_component_net_cancellation_mass", net_absolute_sum-abs(aggregate)),
    ):
        require(exact(row[key]) == expected, "cancellation total splice: " + key)
    retained = exact(row["retained_actual_margin"])-exact(row["retained_reference_margin"])
    require(exact(row["retained_margin_change"]) == retained, "retained margin change splice")
    same(list(row["neutralized_retained_margin_change"]), list(COMPONENTS),
         "neutralized component census changed")
    for key, correction in values.items():
        require(exact(row["neutralized_retained_margin_change"][key]) == retained-correction,
                "neutralized margin change splice")
    return values, aggregate, retained, net_absolute_sum


def pair_census(rows):
    expected = [(c, b) for c in CONTROLS for b in BRANCHES]
    identities = [(r["control"], r["branch"]) for r in rows]
    complete = identities == expected
    counts = Counter(identities)
    parsed = [row_values(row) for row in rows]
    evaluations = []
    for key in COMPONENTS:
        directions = [(v[key] > 0)-(v[key] < 0) for v, _, _, _ in parsed]
        stable = bool(directions) and len(set(directions)) == 1 and directions[0] != 0
        checks = []
        for row, (values, aggregate, retained, net_absolute) in zip(rows, parsed, strict=True):
            correction = values[key]
            neutralized = retained-correction
            larger = [k for k in COMPONENTS if abs(values[k]) > abs(correction)]
            ties = [k for k in COMPONENTS if k != key and abs(values[k]) == abs(correction)]
            largest = not larger and not ties
            half = correction*aggregate > 0 and 2*abs(correction) > abs(aggregate)
            reversal = retained*neutralized < 0
            gates = dict(zip(GATES, (largest, stable, half, reversal, complete), strict=True))
            blockers = [g for g in (GATES[0], GATES[1], GATES[4]) if not gates[g]]
            if not (half or reversal):
                blockers.append("neither_half_nor_reversal")
            zeros = {
                "correction": correction == 0, "direct_hidden_aggregate": aggregate == 0,
                "retained_margin_change": retained == 0, "neutralized_margin_change": neutralized == 0,
            }
            checks.append({
                "control": row["control"], "branch": row["branch"],
                "signed_correction": str(correction),
                "absolute_account": row["component_totals"][key]["absolute"],
                "net_signed_magnitude": str(abs(correction)),
                "largest_competitor_net_magnitude": str(max(abs(v) for k, v in values.items() if k != key)),
                "larger_components": larger, "equal_net_magnitude_components": ties,
                "tied_for_largest": bool(ties) and not larger,
                "direct_hidden_signed": str(aggregate),
                "twice_magnitude_minus_aggregate_magnitude": str(2*abs(correction)-abs(aggregate)),
                "aggregate_direction_product": str(correction*aggregate),
                "strict_half_equality": 2*abs(correction) == abs(aggregate),
                "retained_actual_margin": row["retained_actual_margin"],
                "retained_reference_margin": row["retained_reference_margin"],
                "retained_margin_change": str(retained),
                "neutralized_retained_margin_change": str(neutralized),
                "reversal_product": str(retained*neutralized),
                "component_net_magnitude_sum": str(net_absolute),
                **gates, "failed_gates": [g for g in GATES if not gates[g]],
                "blocking_conditions": blockers, "zero_boundaries": zeros,
                "row_qualifies": not blockers,
            })
        evaluations.append({
            "component": key, "direction_counts": {
                str(sign): directions.count(sign) for sign in (-1, 0, 1)},
            "directionally_stable": stable,
            "qualifies": complete and stable and all(r["row_qualifies"] for r in checks),
            "gate_failure_counts": {g: sum(not r[g] for r in checks) for g in GATES},
            "neither_half_nor_reversal_count": sum(
                "neither_half_nor_reversal" in r["blocking_conditions"] for r in checks),
            "tied_for_largest_count": sum(r["tied_for_largest"] for r in checks),
            "equal_net_magnitude_count": sum(bool(r["equal_net_magnitude_components"]) for r in checks),
            "strict_half_equality_count": sum(r["strict_half_equality"] for r in checks),
            "zero_boundary_counts": {
                k: sum(r["zero_boundaries"][k] for r in checks)
                for k in ("correction", "direct_hidden_aggregate", "retained_margin_change",
                          "neutralized_margin_change")},
            "rows": checks,
        })
    chosen = [e["component"] for e in evaluations if e["qualifies"]]
    require(len(chosen) <= 1, "nonunique component selection")
    return {
        "complete_nine_control_two_branch_census": complete,
        "expected_row_count": len(expected), "observed_row_count": len(rows),
        "missing_rows": [list(i) for i in expected if i not in counts],
        "unexpected_rows": [list(i) for i in counts if i not in expected],
        "duplicate_rows": [{"control": c, "branch": b, "count": n}
                           for (c, b), n in counts.items() if n > 1],
        "canonical_row_order": identities == [i for i in expected if i in counts],
        "component": chosen[0] if chosen else "UNKNOWN",
        "stop_nested_bridge_expansion": not chosen,
        "component_evaluations": evaluations,
        "exact_accounting_rows": rows,
    }


def census(report):
    upstream = report["retained_final_rmsnorm_unselected_coordinate_residual_margin_bridge"]
    same([c["control"] for c in report["controls"]], list(CONTROLS), "control census changed")
    same(report["selection"]["rule"], bridge.SELECTION_RULE, "selection rule changed")
    same(report["selection"]["parent_category"], "direct_hidden", "parent category changed")
    same(report["lineage_separation"], upstream["lineage_separation"], "lineage splice")
    same(report["reference_scope"], bridge.direct.REFERENCE_SCOPE, "reference scope changed")
    parent_groups, groups = {}, {}
    for control, old_control in zip(report["controls"], upstream["controls"], strict=True):
        same(control["control"], old_control["control"], "parent control splice")
        seen = set()
        for pair, old_pair in zip(control["pairs"], old_control["pairs"], strict=True):
            identity = pair["left_id"], pair["right_id"]
            require(identity not in seen, "duplicate diagnostic pair")
            seen.add(identity)
            same([pair[k] for k in ("left_id", "right_id", "roles")],
                 [old_pair[k] for k in ("left_id", "right_id", "roles")], "pair identity splice")
            same(list(pair["branches"]), list(BRANCHES), "branch census changed")
            same(list(old_pair["branches"]), list(BRANCHES), "parent branch census changed")
            for branch, account in pair["branches"].items():
                row, old = account["gate_values"], old_pair["branches"][branch]
                same(account["unchanged_unselected_bridge"], old, "unselected bridge splice")
                same(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
                     "binary64 internal stage substitution")
                same((row["control"], row["branch"]), (control["control"], branch), "row identity splice")
                same(old["hidden_reference"], "original_input_L23_"+branch, "hidden reference splice")
                same(old["rmsnorm_reference"], "original_input_final_attempt003_"+branch,
                     "RMSNorm reference splice")
                same(row["unselected_direct_hidden"],
                     old["unselected_accounting"]["component_totals"]["direct_hidden"],
                     "direct-hidden aggregate splice")
                same(row["unchanged_unselected_categories"], {
                    k: v for k, v in old["unselected_accounting"]["component_totals"].items()
                    if k != "direct_hidden"}, "other category splice")
                accounting = old["unchanged_margin_accounting"]
                for field in ("retained_margin_change", "selected_coordinate_signed_sum",
                              "head_boundary_remainder_change"):
                    same(row[field], accounting[field], "retained fixed account splice")
                same(row["unselected_coordinate_remainder"],
                     old["unselected_accounting"]["unselected_coordinate_remainder"],
                     "unselected remainder splice")
                selected = [r["coordinate"] for r in old["selected_coordinates"]]
                same(row["excluded_selected_coordinates"], selected, "selected complement splice")
                same([r["coordinate"] for r in account["unchanged_selected_direct_hidden"]["selected_coordinates"]],
                     selected, "selected direct-hidden coordinate splice")
                require(62 in selected and selected == sorted(set(selected))
                        and row["coordinate_count"] == bridge.WIDTH-len(selected)
                        and row["exact_unselected_direct_hidden_identity"] is True,
                        "complement/coordinate62 identity changed")
                values, aggregate, retained, _ = row_values(row)
                other = sum((magnitude(v)[0] for v in row["unchanged_unselected_categories"].values()), Fraction())
                require(aggregate+other == exact(accounting["unselected_coordinate_signed_remainder"])
                        and aggregate+other+exact(row["selected_coordinate_signed_sum"])
                        +exact(row["head_boundary_remainder_change"]) == retained,
                        "fixed selected/unselected/head closure splice")
                if branch == "fp16":
                    same(row["component_totals"]["fp16_to_branch_terminal_remainder"],
                         {"signed": "0", "absolute": "0", "cancellation_absolute_mass": "0"},
                         "FP16 terminal remainder changed")
                groups.setdefault(identity, []).append(row)
                parent_groups.setdefault(identity, []).append(
                    {"control": control["control"], "branch": branch, **old})
    parent_decisions = [{"left_id": i[0], "right_id": i[1], **bridge.unselected.select_mechanism(rows)}
                        for i, rows in sorted(parent_groups.items())]
    same(parent_decisions, upstream["selection"]["pairs"], "parent gate splice")
    shared = [d["mechanism"] for d in parent_decisions if d["complete_nine_control_two_branch_census"]]
    require(shared and set(shared) == {"direct_hidden"}
            and upstream["selection"]["mechanism"] == "direct_hidden"
            and upstream["selection"]["stop_nested_bridge_expansion"] is False,
            "parent gate does not authorize component localization")
    pairs = []
    original_decisions = []
    for identity, rows in sorted(groups.items()):
        decision = pair_census(rows)
        original = bridge.select_component(rows)
        for key in ("complete_nine_control_two_branch_census", "component", "stop_nested_bridge_expansion"):
            same(decision[key], original[key], "bridge gate semantics changed")
        for evaluation, old in zip(decision["component_evaluations"], original["component_evaluations"], strict=True):
            for key in ("component", "directionally_stable", "qualifies"):
                same(evaluation[key], old[key], "component gate semantics changed")
            for row, old_row in zip(evaluation["rows"], old["rows"], strict=True):
                same({k: row[k] for k in old_row}, old_row, "row gate semantics changed")
        original_decisions.append({"left_id": identity[0], "right_id": identity[1], **original})
        pairs.append({"left_id": identity[0], "right_id": identity[1], **decision})
    same(original_decisions, report["selection"]["pairs"], "retained component gate splice")
    common = bridge.common_selection(original_decisions)
    same(report["selection"]["component"], common, "common component splice")
    same(report["selection"]["stop_nested_bridge_expansion"], common == "UNKNOWN", "stop decision splice")
    for key in ("control_count", "diagnostic_pair_count", "pair_branch_count", "coordinate62_accounts",
                "selected_coordinate_accounts", "unselected_coordinate_accounts"):
        same(report[key], upstream[key], "retained census splice: " + key)
    require(report["control_count"] == 9
            and report["pair_branch_count"] == sum(len(r) for r in groups.values())
            and report["unselected_coordinate_accounts"] == sum(
                r["coordinate_count"] for rows in groups.values() for r in rows)
            and report["weighted_residual_component_count"] == 7*report["unselected_coordinate_accounts"],
            "coordinate/component census splice")
    return {
        "common_component": common, "stop_nested_bridge_expansion": common == "UNKNOWN",
        "shared_pair_count": sum(p["complete_nine_control_two_branch_census"] for p in pairs),
        "nonshared_pair_count": sum(not p["complete_nine_control_two_branch_census"] for p in pairs),
        "rule": bridge.SELECTION_RULE, "pairs": pairs,
        "retained_bridge_selection": report["selection"],
        "reference_scope": report["reference_scope"], "lineage_separation": report["lineage_separation"],
        "count_scope": "Per numeric pair and component; branch/control rows overlap. No causal shares.",
    }


def run_tests(evidence, observed):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("gate_obstruction_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE, module.OBSERVED = evidence, observed
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "obstruction census tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "gate_obstruction_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    # The upstream interface authenticates retained operands and builds rational
    # accounts; its guard forbids operators. Do not run its --check or its tests.
    with bridge.read_only(audit):
        evidence = bridge.measure()
    with read_only(audit):
        observed = census(evidence["report"])
        tests = run_tests(evidence, observed)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "obstruction census attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_UNSELECTED_DIRECT_HIDDEN_GATE_OBSTRUCTION_CENSUS",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "existing in-memory bridge report from guarded retained-input authentication",
        "report": observed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
