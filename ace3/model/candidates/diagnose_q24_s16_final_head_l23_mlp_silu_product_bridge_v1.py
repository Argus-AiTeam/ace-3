"""Read-only exact operand bridge across the retained L23 SiLU/product boundary."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_gate_up_projection_input_coordinates_v1 as gate_up


down, residual = gate_up.down, gate_up.residual
channels, base, parent = gate_up.channels, gate_up.base, gate_up.parent
ROOT = gate_up.ROOT
NAME = "diagnose_q24_s16_final_head_l23_mlp_silu_product_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, EXPECTED_TESTS = 4864, 8, 16
require, same = base.require, base.same
obj, ordered, array_schema, RATIONAL = (
    gate_up.obj, gate_up.ordered, gate_up.array_schema, gate_up.RATIONAL)
FLAGS = {**gate_up.FLAGS, "product_operator_replay": 0, "reference_replay": 0}
TERM_NAMES = ("gate_delta_times_reference_up", "reference_gate_times_up_delta",
              "gate_up_interaction", "silu_product_boundary_delta")
OPERAND_NAMES = ("gate", "up", "stage16")
VALUES = (
    "gate_delta", "up_delta", "retained_stage16_delta",
    "actual_raw_gate_up_product", "reference_raw_gate_up_product",
    "raw_gate_up_product_delta", "actual_silu_product_remainder",
    "reference_silu_product_remainder", "silu_product_boundary_delta",
    "down_weight", "weighted_stage16_delta",
    "weighted_actual_silu_product_remainder", "weighted_reference_silu_product_remainder",
)
BOUNDARY = residual.BOUNDARY + (
    " This version preserves the 1152 retained stage16 selections and completed "
    "gate/up projection accounts. For g=stage14, u=stage15 and y=stage16, "
    "the raw product baseline g*u is NOT a SiLU output. Define b=y-g*u "
    "separately on actual and original-input FP16 reference trajectories. "
    "Exactly delta_y=delta_g*u_ref+g_ref*delta_u+delta_g*delta_u+delta_b. "
    "The interaction is explicit, not allocated to either operand. The b "
    "remainders jointly contain the nonlinear SiLU effect and implementation/"
    "product/conversion differences; separate SiLU and product-rounding effects "
    "are not identifiable from retained stages without forbidden replay. "
    "No sigmoid, exponential, SiLU, product operator or reference is evaluated. "
    "Exact rational arithmetic is accounting, not a new operator precision. "
    "Signed/absolute sums, descending-absolute ranks with index tie breaks, "
    "and cancellation mass (absolute sum minus absolute signed sum) include "
    "all four terms. Multiplication by the authenticated down weight closes each "
    "selected down-input contribution; unselected inputs and the original "
    "down-projection remainder close stage17 without propagation or causality "
    "claims. Repeated logical selections are counted separately."
)

TOTAL_SCHEMA = obj({key: RATIONAL for key in (
    "signed_sum", "absolute_sum", "cancellation_absolute_mass")})
TERM_SCHEMA = obj({
    "term_index": {"type": "integer", "minimum": 0, "maximum": 3},
    "term": {"enum": list(TERM_NAMES)}, "signed_contribution": RATIONAL,
})
TERM_SCHEMA["allOf"] = [
    {"if": {"properties": {"term_index": {"const": i}}},
     "then": {"properties": {"term": {"const": name}}}}
    for i, name in enumerate(TERM_NAMES)
]
ACCOUNT_SCHEMA = obj({
    "coordinate": {"type": "integer", "minimum": 0, "maximum": WIDTH - 1},
    "reference": {"const": "original_input_L23_fp16"},
    "baseline": {"const": "raw_gate_times_up_NOT_SILU"},
    "separate_silu_product_rounding": {"const": "NOT_IDENTIFIABLE_WITHOUT_REPLAY"},
    **{side: obj({key: RATIONAL for key in OPERAND_NAMES}) for side in ("actual", "reference_operands")},
    **{key: RATIONAL for key in VALUES},
    "terms_in_operand_order": ordered([
        {"allOf": [TERM_SCHEMA, {"properties": {"term_index": {"const": i}}}]}
        for i in range(4)
    ]),
    "ranked_absolute_terms": array_schema(TERM_SCHEMA, 4),
    "weighted_terms_in_operand_order": ordered([
        {"allOf": [TERM_SCHEMA, {"properties": {"term_index": {"const": i}}}]}
        for i in range(4)
    ]),
    **{key: TOTAL_SCHEMA for key in ("operand_totals", "closure_totals", "weighted_closure_totals")},
    "exact_local_identity": {"const": True},
})
HOTSPOT_SUMMARY_SCHEMA = obj({
    "selected_coordinate_count": {"const": LIMIT},
    "selected_stage16_totals": TOTAL_SCHEMA,
    "selected_weighted_down_totals": TOTAL_SCHEMA,
    "weighted_operand_terms_in_order": ordered([
        obj({"term": {"const": name}, "totals": TOTAL_SCHEMA}) for name in TERM_NAMES
    ]),
    "stage16_coordinates_ranked_by_absolute_delta": array_schema(
        {"type": "integer", "minimum": 0, "maximum": WIDTH - 1}, LIMIT),
    **{key: RATIONAL for key in (
        "unselected_down_input_signed_remainder", "unselected_down_input_absolute_remainder",
        "down_projection_boundary_remainder", "retained_stage17_delta")},
    "exact_selected_to_stage17_identity": {"const": True},
})


def branch_schema(branch):
    return obj({
        "final_head_reference_branch": {"const": branch},
        "numeric_id": channels.ROW_SCHEMA["properties"]["numeric_id"],
        "hotspots": ordered([
            obj({
                "hotspot_rank": {"const": rank}, "final_head_channel": channels.ENTRY_SCHEMA,
                "stage16_coordinates": ordered([
                    obj({
                        "stage16_coordinate_rank": {"const": coordinate_rank},
                        "selected_down_input": down.TERM_SCHEMA,
                        "bridge": ACCOUNT_SCHEMA,
                    }) for coordinate_rank in range(1, LIMIT + 1)
                ]),
                "summary": HOTSPOT_SUMMARY_SCHEMA,
            }) for rank in range(1, LIMIT + 1)
        ]),
    })


REPORT_SCHEMA = obj({
    **{key: {"const": value} for key, value in {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "bridge_account_count": 1152, "operand_term_count": 3456,
        "interaction_term_count": 1152, "closure_term_count": 4608,
        "layer": 23, "position": 0,
    }.items()},
    "controls": ordered([
        obj({"control": {"const": control},
             "branches": ordered([branch_schema(branch) for branch in channels.BRANCHES])})
        for control in parent.CONTROLS
    ]),
    "gate_up_projection_input_coordinate_report": gate_up.REPORT_SCHEMA,
    "L23_reference": residual.REPORT_SCHEMA["properties"]["L23_reference"],
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in gate_up.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary",
                      "projection_tensors_verified", "dispatch_and_write_audit", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_L23_MLP_SILU_PRODUCT_BRIDGE"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "projection_tensors_verified": {"const": 9},
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
        raise RuntimeError("SiLU/product bridge forbids replay, earlier checks and writes")

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


def totals(values):
    signed = sum(values, Fraction())
    absolute = sum(map(abs, values), Fraction())
    return {"signed_sum": str(signed), "absolute_sum": str(absolute),
            "cancellation_absolute_mass": str(absolute - abs(signed))}


def operands(archive):
    return {key: residual.words(archive[stage], (WIDTH,))
            for key, stage in zip(OPERAND_NAMES, ("stage14", "stage15", "stage16"), strict=True)}


def account(actual, reference, coordinate, weight):
    require(type(coordinate) is int and 0 <= coordinate < WIDTH
            and isinstance(weight, Fraction)
            and all(set(source) == set(OPERAND_NAMES)
                    and all(isinstance(v, Fraction) for v in source.values())
                    for source in (actual, reference)),
            "invalid exact SiLU/product operands, coordinate or down weight")
    ag, au, ay = (actual[key] for key in OPERAND_NAMES)
    rg, ru, ry = (reference[key] for key in OPERAND_NAMES)
    dg, du, dy = ag - rg, au - ru, ay - ry
    ap, rp = ag * au, rg * ru
    ab, rb = ay - ap, ry - rp
    values = [dg * ru, rg * du, dg * du, ab - rb]
    require(sum(values, Fraction()) == dy, "exact stage16 bridge failed")

    def terms(items):
        return [{"term_index": i, "term": name, "signed_contribution": str(value)}
                for i, (name, value) in enumerate(zip(TERM_NAMES, items, strict=True))]

    entries = terms(values)
    return {
        "coordinate": coordinate, "reference": "original_input_L23_fp16",
        "baseline": "raw_gate_times_up_NOT_SILU",
        "separate_silu_product_rounding": "NOT_IDENTIFIABLE_WITHOUT_REPLAY",
        "actual": {key: str(value) for key, value in actual.items()},
        "reference_operands": {key: str(value) for key, value in reference.items()},
        **{key: str(value) for key, value in zip(VALUES, (
            dg, du, dy, ap, rp, ap - rp, ab, rb, ab - rb, weight, weight * dy,
            weight * ab, weight * rb), strict=True)},
        "terms_in_operand_order": entries,
        "ranked_absolute_terms": sorted(entries, key=lambda e: (
            -abs(Fraction(e["signed_contribution"])), e["term_index"])),
        "weighted_terms_in_operand_order": terms([weight * value for value in values]),
        "operand_totals": totals(values[:3]), "closure_totals": totals(values),
        "weighted_closure_totals": totals([weight * value for value in values]),
        "exact_local_identity": True,
    }


def hotspot_summary(coordinates, previous):
    accounts = [item["bridge"] for item in coordinates]
    stage16 = [Fraction(item["retained_stage16_delta"]) for item in accounts]
    weighted = [Fraction(item["weighted_stage16_delta"]) for item in accounts]
    selected = sum(weighted, Fraction())
    same(str(selected), previous["ranked_signed_sum"], "selected down-input sum changed")
    same(str(sum(map(abs, weighted), Fraction())), previous["ranked_absolute_sum"],
         "selected down-input absolute sum changed")
    remainder = Fraction(previous["unranked_signed_remainder"])
    boundary = Fraction(previous["down_projection_boundary_remainder"])
    require(selected + remainder + boundary == Fraction(previous["retained_projection_delta"]),
            "stage17 bridge closure failed")
    return {
        "selected_coordinate_count": LIMIT,
        "selected_stage16_totals": totals(stage16),
        "selected_weighted_down_totals": totals(weighted),
        "weighted_operand_terms_in_order": [
            {"term": name, "totals": totals([
                Fraction(item["weighted_terms_in_operand_order"][i]["signed_contribution"])
                for item in accounts])} for i, name in enumerate(TERM_NAMES)
        ],
        "stage16_coordinates_ranked_by_absolute_delta": [
            item["coordinate"] for item in sorted(accounts, key=lambda a: (
                -abs(Fraction(a["retained_stage16_delta"])), a["coordinate"]))],
        "unselected_down_input_signed_remainder": str(remainder),
        "unselected_down_input_absolute_remainder": previous["unranked_absolute_remainder"],
        "down_projection_boundary_remainder": str(boundary),
        "retained_stage17_delta": previous["retained_projection_delta"],
        "exact_selected_to_stage17_identity": True,
    }


def report(previous, actual, reference):
    jsonschema.Draft202012Validator(gate_up.REPORT_SCHEMA).validate(previous)
    same(list(actual), list(parent.CONTROLS), "SiLU/product control order changed")
    require(all(set(source) == set(OPERAND_NAMES)
                and all(len(values) == WIDTH for values in source.values())
                for source in (*actual.values(), reference)), "SiLU/product archive geometry changed")
    controls = []
    down_report = previous["down_projection_input_coordinate_report"]
    for row, down_row in zip(previous["controls"], down_report["controls"], strict=True):
        branches = []
        for branch, down_branch in zip(row["branches"], down_row["branches"], strict=True):
            hotspots = []
            for hotspot, down_hotspot in zip(branch["hotspots"], down_branch["hotspots"], strict=True):
                same(hotspot["final_head_channel"], down_hotspot["final_head_channel"],
                     "retained channel selection changed")
                selected = hotspot["stage16_coordinates"]
                same([item["selected_down_input"] for item in selected],
                     down_hotspot["down_projection"]["largest_absolute_coordinates"],
                     "retained stage16 selection changed")
                require(len({s["selected_down_input"]["coordinate"] for s in selected}) == LIMIT,
                        "duplicate stage16 selection")
                coordinates = []
                for item in selected:
                    pin = item["selected_down_input"]
                    coordinate = pin["coordinate"]
                    a, r = ({key: values[coordinate] for key, values in source.items()}
                            for source in (actual[row["control"]], reference))
                    for kind, key in (("gate_proj", "gate"), ("up_proj", "up")):
                        output = item[kind]
                        same(output["output_coordinate"], coordinate, "projection coordinate splice")
                        same(output["actual_projection_output"], str(a[key]), "actual projection splice")
                        same(output["reference_projection_output"], str(r[key]), "reference projection splice")
                        delta = a[key] - r[key]
                        same(output["retained_projection_delta"], str(delta), "projection delta splice")
                        require(Fraction(output["exact_input_delta"])
                                + Fraction(output["projection_boundary_remainder"]) == delta,
                                "projection input/boundary closure changed")
                    bridge = account(a, r, coordinate, Fraction(pin["weight"]))
                    same(bridge["retained_stage16_delta"], pin["input_delta"], "stage16 delta splice")
                    same(bridge["weighted_stage16_delta"], pin["signed_contribution"],
                         "down-input contribution splice")
                    coordinates.append({
                        "stage16_coordinate_rank": item["stage16_coordinate_rank"],
                        "selected_down_input": pin, "bridge": bridge,
                    })
                hotspots.append({
                    "hotspot_rank": hotspot["hotspot_rank"],
                    "final_head_channel": hotspot["final_head_channel"],
                    "stage16_coordinates": coordinates,
                    "summary": hotspot_summary(coordinates, down_hotspot["down_projection"]),
                })
            branches.append({
                "final_head_reference_branch": branch["final_head_reference_branch"],
                "numeric_id": branch["numeric_id"], "hotspots": hotspots,
            })
        controls.append({"control": row["control"], "branches": branches})
    return {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "bridge_account_count": 1152, "operand_term_count": 3456,
        "interaction_term_count": 1152, "closure_term_count": 4608,
        "layer": 23, "position": 0, "controls": controls,
        "gate_up_projection_input_coordinate_report": previous,
        "L23_reference": previous["L23_reference"],
    }


def measure():
    previous, files, assets = gate_up.measure()
    actual = {control: operands(archive) for control, archive in previous[0][3].items()}
    reference = operands(previous[0][4])
    return (previous, actual, reference, report(previous[4], actual, reference)), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("l23_silu_product_bridge_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "SiLU/product focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "silu_product_focused_test": parent.record(TEST),
               **{module.MODULE: parent.record(module.SOURCE)
                  for module in (gate_up, down, residual, channels)}}
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
            "status": "READ_ONLY_L23_MLP_SILU_PRODUCT_BRIDGE",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "L23_archives_verified": 10, "projection_tensors_verified": 9,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "tests": tests, "report": evidence[3],
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
