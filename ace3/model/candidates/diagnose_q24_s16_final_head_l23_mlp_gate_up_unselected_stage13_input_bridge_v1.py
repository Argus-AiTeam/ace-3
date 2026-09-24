"""Stdout-only exact G128 accounting for unranked retained L23 stage13 inputs."""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
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


down, residual, channels, base, parent = (
    gate_up.down, gate_up.residual, gate_up.channels, gate_up.base, gate_up.parent)
ROOT = gate_up.ROOT
NAME = "diagnose_q24_s16_final_head_l23_mlp_gate_up_unselected_stage13_input_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, GROUPS, EXPECTED_TESTS = 896, 8, 7, 18
FLAGS = dict(gate_up.FLAGS)
require, same = base.require, base.same
obj, ordered, array_schema, RATIONAL = (
    gate_up.obj, gate_up.ordered, gate_up.array_schema, gate_up.RATIONAL)
BOUNDARY = gate_up.BOUNDARY + (
    " This extension splits only the complement of each account's unchanged "
    "eight ranked stage13 inputs into seven contiguous G128 unranked accounts "
    "and exact signed/absolute totals. There are 888 unranked coordinates per "
    "projection account, 2045952 logical unranked terms and 16128 G128 accounts "
    "across 2304 gate/up accounts, counting repeated selections separately. "
    "The unranked totals close the existing unranked_signed_remainder and "
    "unranked_absolute_remainder exactly. Selected entries, full-input groups, "
    "projection-boundary remainders and retained projection deltas are unchanged. "
    "This local post-RMSNorm account does not inform or extend the final RMSNorm "
    "pair-selection gate, select a next bridge, neutralize a token margin, "
    "establish dominance or execute a counterfactual."
)

SPLIT_SCHEMA = obj({
    "reference": {"const": "original_input_L23_fp16"},
    "coordinate_count": {"const": WIDTH - LIMIT},
    "excluded_ranked_coordinates": {
        **array_schema({"type": "integer", "minimum": 0, "maximum": WIDTH - 1}, LIMIT),
        "uniqueItems": True,
    },
    "groups_in_input_order": ordered([
        obj({
            "input_group": {"const": group},
            "start_coordinate": {"const": group * 128},
            "end_coordinate_exclusive": {"const": (group + 1) * 128},
            "coordinate_count": {"type": "integer", "minimum": 120, "maximum": 128},
            "excluded_ranked_coordinate_count": {"type": "integer", "minimum": 0, "maximum": 8},
            "signed_contribution": RATIONAL,
            "sum_absolute_contributions": RATIONAL,
        }) for group in range(GROUPS)
    ]),
    "unranked_signed_sum": RATIONAL,
    "unranked_absolute_sum": RATIONAL,
    "exact_unranked_identity": {"const": True},
    "exact_projection_identity": {"const": True},
})
ACCOUNT_SCHEMA = obj({
    **gate_up.ACCOUNT_SCHEMA["properties"],
    "unranked_stage13_inputs": SPLIT_SCHEMA,
})
ACCOUNT_SCHEMA["allOf"] = gate_up.ACCOUNT_SCHEMA["allOf"]


def branch_schema(branch):
    schema = deepcopy(gate_up.branch_schema(branch))
    for hotspot in schema["properties"]["hotspots"]["prefixItems"]:
        for selected in hotspot["properties"]["stage16_coordinates"]["prefixItems"]:
            for kind in gate_up.STAGES:
                selected["properties"][kind] = {
                    "allOf": [ACCOUNT_SCHEMA, {"properties": {"projection": {"const": kind}}}]}
    return schema


REPORT_SCHEMA = obj({
    **gate_up.REPORT_SCHEMA["properties"],
    "unranked_input_coordinate_count": {"const": 2304 * (WIDTH - LIMIT)},
    "controls": ordered([
        obj({"control": {"const": control},
             "branches": ordered([branch_schema(branch) for branch in channels.BRANCHES])})
        for control in parent.CONTROLS
    ]),
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in gate_up.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary",
                      "dispatch_and_write_audit", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_L23_MLP_GATE_UP_UNSELECTED_STAGE13_INPUT_BRIDGE"},
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
        raise RuntimeError("unranked stage13 bridge forbids operators, earlier checks and writes")

    with channels.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("projection", "local_reference", "attention_value",
                                  "check", "focused_tests", "rne", "toward_zero", "_stages"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        for attribute in ("measure", "report"):
            stack.enter_context(patch.object(residual, attribute, forbidden))
        yield


def split_unranked(actual, reference, weights, account):
    kind, coordinate = account["projection"], account["output_coordinate"]
    require(kind in gate_up.STAGES and type(coordinate) is int
            and 0 <= coordinate < gate_up.OUTPUT_WIDTH, "invalid gate/up output")
    stage = gate_up.STAGES[kind]
    require(all(len(source["stage13"]) == WIDTH
                and len(source[stage]) == gate_up.OUTPUT_WIDTH
                and all(isinstance(value, Fraction) for name in ("stage13", stage)
                        for value in source[name]) for source in (actual, reference))
            and len(weights) == WIDTH and all(isinstance(w, Fraction) for w in weights),
            "invalid exact unranked stage13 operands")
    selected = account["largest_absolute_coordinates"]
    excluded = [entry["coordinate"] for entry in selected]
    require(len(excluded) == LIMIT and len(set(excluded)) == LIMIT
            and all(type(i) is int and 0 <= i < WIDTH for i in excluded),
            "ranked stage13 complement changed")
    same(account["reference"], "original_input_L23_fp16", "projection reference changed")
    same(account["input_stage"], "stage13_fp16", "projection input stage changed")
    same(account["output_stage"], stage + "_fp16", "projection output stage changed")
    delta = [a - r for a, r in zip(actual["stage13"], reference["stage13"], strict=True)]
    terms = [value * weight for value, weight in zip(delta, weights, strict=True)]
    same(excluded, sorted(range(WIDTH), key=lambda i: (-abs(terms[i]), i))[:LIMIT],
         "retained ranked stage13 order changed")
    for entry in selected:
        i = entry["coordinate"]
        same(entry, {"coordinate": i, "input_group": i // 128,
                     "input_delta": str(delta[i]), "weight": str(weights[i]),
                     "signed_contribution": str(terms[i])}, "ranked stage13 entry changed")
    excluded_set = set(excluded)
    groups = []
    for group in range(GROUPS):
        indices = range(group * 128, (group + 1) * 128)
        unranked = [terms[i] for i in indices if i not in excluded_set]
        same(account["groups_in_input_order"][group], {
            "input_group": group, "start_coordinate": group * 128,
            "end_coordinate_exclusive": (group + 1) * 128, "coordinate_count": 128,
            "signed_contribution": str(sum((terms[i] for i in indices), Fraction())),
            "sum_absolute_contributions": str(sum((abs(terms[i]) for i in indices), Fraction())),
        }, "full-input G128 account changed")
        groups.append({
            "input_group": group, "start_coordinate": group * 128,
            "end_coordinate_exclusive": (group + 1) * 128,
            "coordinate_count": len(unranked),
            "excluded_ranked_coordinate_count": 128 - len(unranked),
            "signed_contribution": str(sum(unranked, Fraction())),
            "sum_absolute_contributions": str(sum(map(abs, unranked), Fraction())),
        })
    signed = sum((Fraction(g["signed_contribution"]) for g in groups), Fraction())
    absolute = sum((Fraction(g["sum_absolute_contributions"]) for g in groups), Fraction())
    ranked = sum((terms[i] for i in excluded), Fraction())
    ranked_absolute = sum((abs(terms[i]) for i in excluded), Fraction())
    a, r = actual[stage][coordinate], reference[stage][coordinate]
    for key, value in {
        "unranked_signed_remainder": signed, "unranked_absolute_remainder": absolute,
        "ranked_signed_sum": ranked, "ranked_absolute_sum": ranked_absolute,
        "exact_input_delta": ranked + signed,
        "sum_absolute_contributions": ranked_absolute + absolute,
        "actual_projection_output": a, "reference_projection_output": r,
        "retained_projection_delta": a - r,
        "projection_boundary_remainder": a - r - ranked - signed,
    }.items():
        same(account[key], str(value), f"unranked projection closure changed: {key}")
    require(sum(g["coordinate_count"] for g in groups) == WIDTH - LIMIT
            and absolute >= abs(signed), "unranked G128 partition failed")
    return {
        "reference": "original_input_L23_fp16", "coordinate_count": WIDTH - LIMIT,
        "excluded_ranked_coordinates": excluded, "groups_in_input_order": groups,
        "unranked_signed_sum": str(signed), "unranked_absolute_sum": str(absolute),
        "exact_unranked_identity": True, "exact_projection_identity": True,
    }


def report(evidence):
    _, actual, reference, tensors, previous = evidence
    jsonschema.Draft202012Validator(gate_up.REPORT_SCHEMA).validate(previous)
    same(list(actual), list(parent.CONTROLS), "unranked control order changed")
    controls, columns = [], {}
    for row in previous["controls"]:
        branches = []
        for branch in row["branches"]:
            hotspots = []
            for hotspot in branch["hotspots"]:
                coordinates = []
                for selected in hotspot["stage16_coordinates"]:
                    entry = dict(selected)
                    for kind in gate_up.STAGES:
                        account = selected[kind]
                        coordinate = account["output_coordinate"]
                        same(coordinate, selected["selected_down_input"]["coordinate"],
                             "gate/up stage16 selection changed")
                        if (kind, coordinate) not in columns:
                            columns[kind, coordinate] = gate_up.weight_column(tensors[kind], coordinate)
                        split = split_unranked(actual[row["control"]], reference,
                                               columns[kind, coordinate], account)
                        entry[kind] = {**account, "unranked_stage13_inputs": split}
                    coordinates.append(entry)
                hotspots.append({**hotspot, "stage16_coordinates": coordinates})
            branches.append({**branch, "hotspots": hotspots})
        controls.append({**row, "branches": branches})
    return {**previous, "controls": controls, "unranked_input_coordinate_count": 2304 * (WIDTH - LIMIT)}


def measure():
    evidence, files, assets = gate_up.measure()
    return (evidence, report(evidence)), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("l23_unranked_stage13_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "unranked stage13 focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "unranked_stage13_focused_test": parent.record(TEST),
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
            "status": "READ_ONLY_L23_MLP_GATE_UP_UNSELECTED_STAGE13_INPUT_BRIDGE",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "L23_archives_verified": 10, "projection_tensors_verified": 6,
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
