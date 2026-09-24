"""Stdout-only retained L23 SiLU/product accounting on unselected down inputs."""

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

import jsonschema

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_silu_product_bridge_v1 as bridge


down, residual, channels, base, parent = (
    bridge.down, bridge.residual, bridge.channels, bridge.base, bridge.parent)
ROOT = bridge.ROOT
NAME = "diagnose_q24_s16_final_head_l23_mlp_unselected_down_input_silu_product_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, EXPECTED_TESTS = bridge.WIDTH, bridge.LIMIT, 14
TERMS, FLAGS = bridge.TERM_NAMES, dict(bridge.FLAGS)
require, same = base.require, base.same
obj, ordered, array_schema, RATIONAL = (
    bridge.obj, bridge.ordered, bridge.array_schema, bridge.RATIONAL)
BOUNDARY = bridge.BOUNDARY + (
    " This extension splits only each hotspot's unchanged complement of eight "
    "selected stage16 coordinates: 4856 inputs per hotspot, 699264 logical "
    "input accounts and 2797056 weighted terms across 144 hotspots. The four "
    "weighted category net signed sums close the original unselected down-input "
    "signed remainder. Category absolute accounts retain cancellation and need "
    "not equal the absolute unselected input remainder. Actual and reference "
    "weighted b=y-g*u remainders remain separate. Selected-coordinate accounts "
    "and the down-projection boundary remainder are unchanged. Both final-head "
    "selection branches still use original-input L23 FP16 internal operands; "
    "binary64 internals are not reconstructed. This local stage17 account does "
    "not inform or extend the final RMSNorm pair-selection gate, select a next "
    "bridge, neutralize a token margin, or execute a counterfactual."
)

SPLIT_SCHEMA = obj({
    "reference": {"const": "original_input_L23_fp16"},
    "baseline": {"const": "raw_gate_times_up_NOT_SILU"},
    "separate_silu_product_rounding": {"const": "NOT_IDENTIFIABLE_WITHOUT_REPLAY"},
    "coordinate_count": {"const": WIDTH - LIMIT},
    "excluded_selected_coordinates": {
        **array_schema({"type": "integer", "minimum": 0, "maximum": WIDTH - 1}, LIMIT),
        "uniqueItems": True,
    },
    "weighted_terms_in_operand_order": ordered([
        obj({"term": {"const": name}, "totals": bridge.TOTAL_SCHEMA}) for name in TERMS
    ]),
    "unselected_down_input_totals": bridge.TOTAL_SCHEMA,
    **{key: RATIONAL for key in (
        "weighted_actual_silu_product_remainder",
        "weighted_reference_silu_product_remainder",
        "component_absolute_sum", "within_coordinate_cancellation_mass",
        "across_coordinate_cancellation_mass",
        "within_category_across_coordinate_cancellation_mass",
        "between_category_net_cancellation_mass", "total_component_cancellation_mass",
        "selected_down_input_signed_sum", "down_projection_boundary_remainder",
        "retained_stage17_delta",
    )},
    "exact_unselected_identity": {"const": True},
    "exact_stage17_identity": {"const": True},
})


def branch_schema(branch):
    previous = bridge.branch_schema(branch)
    hotspots = []
    for rank in range(LIMIT):
        old = previous["properties"]["hotspots"]["prefixItems"][rank]
        hotspots.append(obj({**old["properties"], "unselected_bridge": SPLIT_SCHEMA}))
    return obj({**previous["properties"], "hotspots": ordered(hotspots)})


REPORT_SCHEMA = obj({
    **{key: {"const": value} for key, value in {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "selected_coordinate_account_count": 1152,
        "unselected_coordinate_account_count": 144 * (WIDTH - LIMIT),
        "weighted_term_count": 4 * 144 * (WIDTH - LIMIT), "layer": 23, "position": 0,
    }.items()},
    "controls": ordered([
        obj({"control": {"const": control},
             "branches": ordered([branch_schema(branch) for branch in channels.BRANCHES])})
        for control in parent.CONTROLS
    ]),
    "silu_product_bridge_report": bridge.REPORT_SCHEMA,
    "L23_reference": bridge.REPORT_SCHEMA["properties"]["L23_reference"],
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in bridge.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary",
                      "dispatch_and_write_audit", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_L23_MLP_UNSELECTED_DOWN_INPUT_SILU_PRODUCT_BRIDGE"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "dispatch_and_write_audit": {"const": {**FLAGS, "forbidden_calls": 0}},
    "tests": obj({
        "compiled": array_schema({"type": "object"}, 2),
        "executed": {"const": EXPECTED_TESTS},
        **{key: {"const": 0} for key in ("failures", "errors", "skipped")},
    }),
    "report": REPORT_SCHEMA,
})


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("unselected down-input bridge forbids replay and writes")

    with channels.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("projection", "local_reference", "attention_value",
                                  "check", "focused_tests", "rne", "toward_zero", "_stages"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
            if name in ("torch.nn.functional", "numpy", "math", "torch"):
                for attribute in ("silu", "sigmoid", "exp", "expm1"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        for attribute in ("measure", "report"):
            stack.enter_context(patch.object(residual, attribute, forbidden))
        yield


def split_unselected(actual, reference, weights, hotspot):
    require(all(set(source) == set(bridge.OPERAND_NAMES)
                and all(len(vector) == WIDTH and all(isinstance(x, Fraction) for x in vector)
                        for vector in source.values()) for source in (actual, reference))
            and len(weights) == WIDTH and all(isinstance(x, Fraction) for x in weights),
            "invalid exact unselected bridge operands")
    selected = [item["selected_down_input"]["coordinate"]
                for item in hotspot["stage16_coordinates"]]
    require(len(selected) == LIMIT and len(set(selected)) == LIMIT
            and all(type(i) is int and 0 <= i < WIDTH for i in selected),
            "selected stage16 complement changed")
    selected_terms = []
    for item in hotspot["stage16_coordinates"]:
        pin = item["selected_down_input"]
        i = pin["coordinate"]
        same(pin["weight"], str(weights[i]), "selected down weight changed")
        delta = actual["stage16"][i] - reference["stage16"][i]
        same(pin["input_delta"], str(delta), "selected stage16 delta changed")
        same(pin["signed_contribution"], str(weights[i] * delta),
             "selected down contribution changed")
        selected_terms.append(weights[i] * delta)
    selected_totals = bridge.totals(selected_terms)
    summary = hotspot["summary"]
    same(selected_totals, summary["selected_weighted_down_totals"],
         "selected down-input totals changed")
    excluded = set(selected)
    components, deltas, actual_boundary, reference_boundary = [[], [], [], []], [], [], []
    for i in range(WIDTH):
        if i in excluded:
            continue
        ag, au, ay = (actual[key][i] for key in bridge.OPERAND_NAMES)
        rg, ru, ry = (reference[key][i] for key in bridge.OPERAND_NAMES)
        dg, du, w = ag - rg, au - ru, weights[i]
        ab, rb = w * (ay - ag * au), w * (ry - rg * ru)
        values = (w * dg * ru, w * rg * du, w * dg * du, ab - rb)
        delta = w * (ay - ry)
        require(sum(values, Fraction()) == delta, "unselected coordinate closure failed")
        for component, value in zip(components, values, strict=True):
            component.append(value)
        deltas.append(delta)
        actual_boundary.append(ab)
        reference_boundary.append(rb)
    totals = [bridge.totals(values) for values in components]
    target = bridge.totals(deltas)
    same(target["signed_sum"], summary["unselected_down_input_signed_remainder"],
         "unselected signed remainder changed")
    same(target["absolute_sum"], summary["unselected_down_input_absolute_remainder"],
         "unselected absolute remainder changed")
    signed = sum((Fraction(t["signed_sum"]) for t in totals), Fraction())
    absolute = sum((Fraction(t["absolute_sum"]) for t in totals), Fraction())
    net_absolute = sum((abs(Fraction(t["signed_sum"])) for t in totals), Fraction())
    target_absolute = Fraction(target["absolute_sum"])
    boundary = Fraction(summary["down_projection_boundary_remainder"])
    require(signed == Fraction(target["signed_sum"])
            and absolute >= target_absolute >= abs(signed)
            and absolute >= net_absolute >= abs(signed),
            "unselected category closure failed")
    require(Fraction(selected_totals["signed_sum"]) + signed + boundary
            == Fraction(summary["retained_stage17_delta"]), "stage17 closure changed")
    return {
        "reference": "original_input_L23_fp16",
        "baseline": "raw_gate_times_up_NOT_SILU",
        "separate_silu_product_rounding": "NOT_IDENTIFIABLE_WITHOUT_REPLAY",
        "coordinate_count": len(deltas), "excluded_selected_coordinates": sorted(selected),
        "weighted_terms_in_operand_order": [
            {"term": name, "totals": total} for name, total in zip(TERMS, totals, strict=True)],
        "unselected_down_input_totals": target,
        "weighted_actual_silu_product_remainder": str(sum(actual_boundary, Fraction())),
        "weighted_reference_silu_product_remainder": str(sum(reference_boundary, Fraction())),
        "component_absolute_sum": str(absolute),
        "within_coordinate_cancellation_mass": str(absolute - target_absolute),
        "across_coordinate_cancellation_mass": target["cancellation_absolute_mass"],
        "within_category_across_coordinate_cancellation_mass": str(absolute - net_absolute),
        "between_category_net_cancellation_mass": str(net_absolute - abs(signed)),
        "total_component_cancellation_mass": str(absolute - abs(signed)),
        "selected_down_input_signed_sum": selected_totals["signed_sum"],
        "down_projection_boundary_remainder": summary["down_projection_boundary_remainder"],
        "retained_stage17_delta": summary["retained_stage17_delta"],
        "exact_unselected_identity": True, "exact_stage17_identity": True,
    }


def report(evidence):
    previous, actual, reference, selected_report = evidence
    same(list(actual), list(parent.CONTROLS), "unselected control order changed")
    same([row["control"] for row in selected_report["controls"]], list(parent.CONTROLS),
         "selected report control order changed")
    controls, columns = [], {}
    for row in selected_report["controls"]:
        same([b["final_head_reference_branch"] for b in row["branches"]],
             list(channels.BRANCHES), "selection branch order changed")
        branches = []
        for branch in row["branches"]:
            hotspots = []
            for hotspot in branch["hotspots"]:
                coordinate = hotspot["final_head_channel"]["coordinate"]
                if coordinate not in columns:
                    columns[coordinate] = down.weight_column(previous[0][5], coordinate)
                split = split_unselected(actual[row["control"]], reference,
                                         columns[coordinate], hotspot)
                hotspots.append({**hotspot, "unselected_bridge": split})
            branches.append({**branch, "hotspots": hotspots})
        controls.append({**row, "branches": branches})
    return {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "selected_coordinate_account_count": 1152,
        "unselected_coordinate_account_count": 144 * (WIDTH - LIMIT),
        "weighted_term_count": 4 * 144 * (WIDTH - LIMIT), "layer": 23, "position": 0,
        "controls": controls, "silu_product_bridge_report": selected_report,
        "L23_reference": selected_report["L23_reference"],
    }


def measure():
    evidence, files, assets = bridge.measure()
    return (evidence, report(evidence)), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("l23_unselected_down_input_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "unselected down-input focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "unselected_down_input_focused_test": parent.record(TEST),
               **{module.MODULE: parent.record(module.SOURCE)
                  for module in (bridge, bridge.gate_up, down, residual, channels)}}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        compiled = []
        for path in (SOURCE, TEST):
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(parent.record(path))
        evidence, files, assets = measure()
        tests = {"compiled": compiled, **focused_tests(evidence)}
        for pin in (*origins.values(), *channels.PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_L23_MLP_UNSELECTED_DOWN_INPUT_SILU_PRODUCT_BRIDGE",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "L23_archives_verified": 10, "projection_tensors_verified": 9,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "tests": tests, "report": evidence[1],
        }
        jsonschema.Draft202012Validator.check_schema(OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
