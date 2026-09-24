"""Stdout-only exact G128 complement of retained V projection stage00 inputs."""

import argparse
from contextlib import contextmanager
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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_value_projection_input_coordinates_v1 as inputs


base, parent, channels = inputs.base, inputs.parent, inputs.channels
ROOT = inputs.ROOT
NAME = "diagnose_q24_s16_final_head_attention_value_projection_unselected_stage00_complement_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, GROUPS, EXPECTED_TESTS = inputs.WIDTH, inputs.LIMIT, 7, 28
FLAGS = dict(inputs.FLAGS)
require, same = base.require, base.same
obj, array_schema, RATIONAL = inputs.obj, inputs.array_schema, inputs.RATIONAL
BOUNDARY = inputs.BOUNDARY + (
    " This extension splits only the complement outside each retained V account's "
    "unchanged eight ranked stage00 coordinates into exact signed and absolute "
    "accounts for seven contiguous G128 groups in input order and 888-coordinate "
    "totals. Across 1152 retained projection accounts this is 1022976 logical "
    "unselected terms and 8064 G128 complement accounts, counting repeated "
    "selections separately. Signed totals close the existing unranked_signed_remainder; "
    "absolute totals close the full absolute sum minus selected absolute terms. "
    "Selected entries, full-input group rankings, retained V deltas, projection "
    "boundary remainders and value-component weighted identities remain unchanged. "
    "This local accounting neither selects a bridge nor establishes a dominant "
    "cause, a margin-change reversal, token ranking change or executed counterfactual."
)

SPLIT_SCHEMA = obj({
    "attention_reference": {"const": "original_input_fp16"},
    "coordinate_count": {"const": WIDTH - LIMIT},
    "excluded_ranked_coordinates": {
        **array_schema({"type": "integer", "minimum": 0, "maximum": WIDTH - 1}, LIMIT),
        "uniqueItems": True,
    },
    "groups_in_input_order": {
        "type": "array", "minItems": GROUPS, "maxItems": GROUPS, "items": False,
        "prefixItems": [
            obj({
                "input_group": {"const": group},
                "start_coordinate": {"const": group * 128},
                "end_coordinate_exclusive": {"const": (group + 1) * 128},
                "coordinate_count": {"type": "integer", "minimum": 120, "maximum": 128},
                "excluded_ranked_coordinate_count": {
                    "type": "integer", "minimum": 0, "maximum": LIMIT},
                "signed_contribution": RATIONAL,
                "sum_absolute_contributions": RATIONAL,
            }) for group in range(GROUPS)
        ],
    },
    "unselected_signed_sum": RATIONAL,
    "unselected_absolute_sum": RATIONAL,
    "exact_complement_identity": {"const": True},
    "exact_projection_identity": {"const": True},
    "exact_value_component_identity": {"const": True},
})
ACCOUNT_SCHEMA = obj({
    **inputs.ACCOUNT_SCHEMA["properties"], "unselected_stage00_inputs": SPLIT_SCHEMA,
})
REPORT_SCHEMA = deepcopy(inputs.REPORT_SCHEMA)
REPORT_SCHEMA["properties"]["unselected_input_coordinate_count"] = {"const": 1022976}
REPORT_SCHEMA["required"].append("unselected_input_coordinate_count")
for branch_schema in REPORT_SCHEMA["properties"]["controls"]["items"]["properties"]["branches"]["properties"].values():
    component_schema = branch_schema["properties"]["hotspots"]["items"]["properties"]["value_components"]["items"]
    component_schema["properties"]["value_projection"] = ACCOUNT_SCHEMA
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in inputs.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_ATTENTION_VALUE_PROJECTION_UNSELECTED_STAGE00_COMPLEMENT"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
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
        raise RuntimeError("V stage00 complement forbids earlier checks and operator replay")

    with inputs.read_only(audit), patch.object(inputs, "check", forbidden), \
            patch.object(inputs, "focused_tests", forbidden):
        yield


def split_unselected(actual, reference, weights, account):
    require(all(len(values) == WIDTH and all(isinstance(v, Fraction) for v in values)
                for values in (actual, reference, weights)), "invalid exact stage00 operands")
    selected = account["largest_absolute_coordinates"]
    excluded = [entry["coordinate"] for entry in selected]
    require(len(excluded) == LIMIT and len(set(excluded)) == LIMIT
            and all(type(i) is int and 0 <= i < WIDTH for i in excluded),
            "ranked stage00 complement changed")
    same((account["projection"], account["layer"], account["position"],
          account["input_stage"], account["output_stage"], account["coordinate_count"]),
         ("v_proj", 23, 0, "stage00_fp16", "stage03_fp16", WIDTH),
         "V projection scope changed")
    deltas = [a - r for a, r in zip(actual, reference, strict=True)]
    terms = [delta * weight for delta, weight in zip(deltas, weights, strict=True)]
    same(excluded, sorted(range(WIDTH), key=lambda i: (-abs(terms[i]), i))[:LIMIT],
         "ranked stage00 order changed")
    for entry in selected:
        i = entry["coordinate"]
        same(entry, {"coordinate": i, "input_group": i // 128,
                     "input_delta": str(deltas[i]), "weight": str(weights[i]),
                     "signed_contribution": str(terms[i])}, "selected stage00 entry changed")
    excluded_set = set(excluded)
    groups, full_groups = [], []
    for group in range(GROUPS):
        indices = range(group * 128, (group + 1) * 128)
        unselected = [terms[i] for i in indices if i not in excluded_set]
        full_groups.append({
            "input_group": group, "start_coordinate": group * 128,
            "end_coordinate_exclusive": (group + 1) * 128, "coordinate_count": 128,
            "signed_contribution": str(sum((terms[i] for i in indices), Fraction())),
            "sum_absolute_contributions": str(sum((abs(terms[i]) for i in indices), Fraction())),
        })
        groups.append({
            "input_group": group, "start_coordinate": group * 128,
            "end_coordinate_exclusive": (group + 1) * 128,
            "coordinate_count": len(unselected),
            "excluded_ranked_coordinate_count": 128 - len(unselected),
            "signed_contribution": str(sum(unselected, Fraction())),
            "sum_absolute_contributions": str(sum(map(abs, unselected), Fraction())),
        })
    full_groups.sort(key=lambda g: (-abs(Fraction(g["signed_contribution"])), g["input_group"]))
    same(account["groups_by_absolute_signed_sum"], full_groups, "full-input G128 accounts changed")
    signed = sum((Fraction(g["signed_contribution"]) for g in groups), Fraction())
    absolute = sum((Fraction(g["sum_absolute_contributions"]) for g in groups), Fraction())
    ranked = sum((terms[i] for i in excluded), Fraction())
    ranked_absolute = sum((abs(terms[i]) for i in excluded), Fraction())
    retained = Fraction(account["actual_v"]) - Fraction(account["reference_v"])
    boundary = retained - ranked - signed
    factor = Fraction(account["value_component_factor"])
    for key, value in {
        "unranked_signed_remainder": signed, "ranked_signed_sum": ranked,
        "exact_input_delta": ranked + signed,
        "sum_absolute_contributions": ranked_absolute + absolute,
        "retained_projection_delta": retained, "projection_boundary_remainder": boundary,
        "weighted_input_contribution": (ranked + signed) * factor,
        "weighted_boundary_contribution": boundary * factor,
        "retained_value_contribution": retained * factor,
    }.items():
        same(account[key], str(value), f"V complement closure changed: {key}")
    require(sum(g["coordinate_count"] for g in groups) == WIDTH - LIMIT
            and absolute >= abs(signed), "unselected G128 partition failed")
    return {
        "attention_reference": "original_input_fp16", "coordinate_count": WIDTH - LIMIT,
        "excluded_ranked_coordinates": excluded, "groups_in_input_order": groups,
        "unselected_signed_sum": str(signed), "unselected_absolute_sum": str(absolute),
        "exact_complement_identity": True, "exact_projection_identity": True,
        "exact_value_component_identity": True,
    }


def report(evidence):
    previous, actual, reference, tensors, inherited = evidence
    jsonschema.Draft202012Validator(inputs.REPORT_SCHEMA).validate(inherited)
    same(list(actual), list(parent.CONTROLS), "V complement control order changed")
    measured = deepcopy(inherited)
    columns, count = {}, 0
    for row in measured["controls"]:
        for branch in row["branches"].values():
            for hotspot in branch["hotspots"]:
                for entry in hotspot["value_components"]:
                    account = entry["value_projection"]
                    coordinate = account["output_coordinate"]
                    same(coordinate, entry["retained_value_component"]["value_coordinate"],
                         "V complement component splice")
                    same((account["actual_v"], account["reference_v"]),
                         (str(previous[1][row["control"]][1][coordinate]),
                          str(previous[2][1][coordinate])), "retained V output splice")
                    same(account["retained_value_contribution"],
                         entry["retained_value_component"]["value"], "weighted V component splice")
                    if coordinate not in columns:
                        columns[coordinate] = inputs.weight_column(tensors, coordinate)
                    account["unselected_stage00_inputs"] = split_unselected(
                        actual[row["control"]], reference, columns[coordinate], account)
                    count += account["unselected_stage00_inputs"]["coordinate_count"]
    measured["unselected_input_coordinate_count"] = count
    return measured


def measure():
    evidence, files, assets = inputs.measure()
    return (evidence, report(evidence)), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("v_stage00_complement_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "V complement focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "v_stage00_complement_focused_test": parent.record(TEST)}
    for module in (inputs, inputs.attribution, channels):
        origins[module.MODULE] = parent.record(module.SOURCE)
        origins[module.MODULE + ".focused_test"] = parent.record(module.TEST)
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
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or evidence write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_ATTENTION_VALUE_PROJECTION_UNSELECTED_STAGE00_COMPLEMENT",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 7,
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
