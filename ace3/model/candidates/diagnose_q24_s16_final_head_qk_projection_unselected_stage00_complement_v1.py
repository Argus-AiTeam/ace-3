"""Stdout-only exact G128 complement of retained Q/K projection stage00 inputs."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_qk_projection_input_coordinates_v1 as inputs


base, parent, channels = inputs.base, inputs.parent, inputs.channels
ROOT = inputs.ROOT
NAME = "diagnose_q24_s16_final_head_qk_projection_unselected_stage00_complement_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, GROUPS, EXPECTED_TESTS = inputs.WIDTH, inputs.LIMIT, 7, 28
FLAGS = dict(inputs.FLAGS)
require, same = base.require, base.same
obj, array_schema, RATIONAL = inputs.obj, inputs.array_schema, inputs.RATIONAL
SIDES = ("reference", "actual")
BOUNDARY = inputs.BOUNDARY + (
    " This extension accounts only for the 888 stage00 coordinates outside each "
    "Q/K projection account's unchanged eight selected inputs. Seven contiguous "
    "G128 complements retain exact signed sums and sums of absolute coordinate "
    "terms, not absolute net group sums. The 1152 retained accounts contain "
    "1022976 logical unselected terms and 8064 complement groups, including "
    "repeated Q/K selections. Selected plus unselected plus the unchanged "
    "projection boundary equals the retained projection delta at both original "
    "reference-partner and actual-partner QK/8 factors. Both operand-swap "
    "identities remain unchanged. This is accounting only, not bridge selection, "
    "a dominant-cause claim, margin-change reversal, token ranking change or an "
    "executed counterfactual. No additional reference trajectory is reconstructed."
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
                **{f"qk_at_{side}_partner": RATIONAL for side in SIDES},
            }) for group in range(GROUPS)
        ],
    },
    "unselected_signed_sum": RATIONAL,
    "unselected_absolute_sum": RATIONAL,
    **{f"qk_unselected_at_{side}_partner": RATIONAL for side in SIDES},
    "exact_complement_identity": {"const": True},
    "exact_projection_identity": {"const": True},
    "exact_qk_partner_identities": {"const": True},
})
ACCOUNT_SCHEMA = deepcopy(inputs.ACCOUNT_SCHEMA)
ACCOUNT_SCHEMA["properties"]["unselected_stage00_inputs"] = SPLIT_SCHEMA
ACCOUNT_SCHEMA["required"].append("unselected_stage00_inputs")
REPORT_SCHEMA = deepcopy(inputs.REPORT_SCHEMA)
REPORT_SCHEMA["properties"]["unselected_input_coordinate_count"] = {"const": 1022976}
REPORT_SCHEMA["required"].append("unselected_input_coordinate_count")
_dimension_schema = REPORT_SCHEMA["properties"]["controls"]["items"]["properties"][
    "score_hotspots"]["items"]["properties"]["dimensions"]["items"]
for _kind in inputs.OUTPUT_WIDTHS:
    _dimension_schema["properties"][_kind] = ACCOUNT_SCHEMA
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in inputs.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_QK_PROJECTION_UNSELECTED_STAGE00_COMPLEMENT"},
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
        raise RuntimeError("Q/K stage00 complement forbids earlier checks and operator replay")

    with inputs.read_only(audit), patch.object(inputs, "check", forbidden), \
            patch.object(inputs, "focused_tests", forbidden):
        yield


def split_unselected(actual, reference, weights, component, kind, account):
    same(account, inputs.account(actual, reference, weights, component, kind),
         "retained Q/K projection account changed")
    selected = account["largest_absolute_coordinates"]
    excluded = [entry["coordinate"] for entry in selected]
    factors = {side: Fraction(account[side + "_partner_over_8"]) for side in SIDES}
    groups = []
    for full in sorted(account["groups_by_absolute_signed_sum"], key=lambda g: g["input_group"]):
        removed = [Fraction(entry["signed_contribution"]) for entry in selected
                   if entry["input_group"] == full["input_group"]]
        signed = Fraction(full["signed_contribution"]) - sum(removed, Fraction())
        absolute = Fraction(full["sum_absolute_contributions"]) - sum(map(abs, removed), Fraction())
        require(absolute >= abs(signed), "invalid G128 complement absolute sum")
        groups.append({
            "input_group": full["input_group"],
            "start_coordinate": full["start_coordinate"],
            "end_coordinate_exclusive": full["end_coordinate_exclusive"],
            "coordinate_count": 128 - len(removed),
            "excluded_ranked_coordinate_count": len(removed),
            "signed_contribution": str(signed), "sum_absolute_contributions": str(absolute),
            **{f"qk_at_{side}_partner": str(signed * factor) for side, factor in factors.items()},
        })
    signed = sum((Fraction(g["signed_contribution"]) for g in groups), Fraction())
    absolute = sum((Fraction(g["sum_absolute_contributions"]) for g in groups), Fraction())
    ranked = sum((Fraction(entry["signed_contribution"]) for entry in selected), Fraction())
    ranked_absolute = sum((abs(Fraction(entry["signed_contribution"])) for entry in selected), Fraction())
    boundary = Fraction(account["projection_boundary_delta"])
    same(str(signed), account["unranked_signed_remainder"], "signed Q/K complement failed")
    same(str(absolute + ranked_absolute), account["sum_absolute_contributions"],
         "absolute Q/K complement failed")
    same(str(ranked + signed + boundary), account["retained_projection_delta"],
         "Q/K complement projection identity failed")
    for side, factor in factors.items():
        same(str((ranked + signed + boundary) * factor),
             account[f"qk_retained_at_{side}_partner"], "Q/K complement partner identity failed")
    require(sum(g["coordinate_count"] for g in groups) == WIDTH - LIMIT,
            "Q/K complement coordinate census failed")
    return {
        "attention_reference": "original_input_fp16", "coordinate_count": WIDTH - LIMIT,
        "excluded_ranked_coordinates": excluded, "groups_in_input_order": groups,
        "unselected_signed_sum": str(signed), "unselected_absolute_sum": str(absolute),
        **{f"qk_unselected_at_{side}_partner": str(signed * factor)
           for side, factor in factors.items()},
        "exact_complement_identity": True, "exact_projection_identity": True,
        "exact_qk_partner_identities": True,
    }


def report(evidence):
    previous, actual, reference, tensors, inherited = evidence
    jsonschema.Draft202012Validator(inputs.REPORT_SCHEMA).validate(inherited)
    same(list(actual), list(parent.CONTROLS), "Q/K complement input control order changed")
    same([row["control"] for row in inherited["controls"]], list(parent.CONTROLS),
         "Q/K complement control order changed")
    same(inherited["qk_logit_attribution_report"], previous[3], "Q/K attribution splice")
    measured = deepcopy(inherited)
    columns, count = {}, 0
    for row, prior in zip(measured["controls"], previous[3]["controls"], strict=True):
        same(row["control"], prior["control"], "Q/K complement control splice")
        for hotspot, old in zip(row["score_hotspots"], prior["score_hotspots"], strict=True):
            source = old["source_tokens"][0]
            for key in ("score_hotspot_rank", "retained_score"):
                same(hotspot[key], old[key], "Q/K complement score selection changed")
            for key in ("source_position", "source_token_id"):
                same(hotspot[key], source[key], "Q/K complement source changed")
            components = source["qk_account"]["largest_absolute_components"]
            for rank, (entry, component) in enumerate(zip(hotspot["dimensions"], components, strict=True), 1):
                same(entry["dimension_hotspot_rank"], rank, "Q/K complement dimension order changed")
                same(entry["retained_qk_component"], component, "Q/K complement component splice")
                for kind in inputs.OUTPUT_WIDTHS:
                    account = entry[kind]
                    key = kind, component[kind + "_coordinate"]
                    if key not in columns:
                        columns[key] = inputs.weight_column(tensors, *key)
                    account["unselected_stage00_inputs"] = split_unselected(
                        actual[row["control"]], reference, columns[key], component, kind, account)
                    count += account["unselected_stage00_inputs"]["coordinate_count"]
                q, k = entry["query"], entry["key"]
                qr, qa = (Fraction(q[f"qk_retained_at_{side}_partner"]) for side in SIDES)
                kr, ka = (Fraction(k[f"qk_retained_at_{side}_partner"]) for side in SIDES)
                require(tuple(Fraction(component[part]) for part in inputs.qk.PARTS)
                        == (qr, kr, qa - qr, qr + ka), "Q/K complement operand identity changed")
                require(qr + ka == kr + qa, "Q/K complement swap identity failed")
    measured["unselected_input_coordinate_count"] = count
    return measured


def measure():
    evidence, files, assets = inputs.measure()
    return (evidence, report(evidence)), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("qk_stage00_complement_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "Q/K complement focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "qk_stage00_complement_focused_test": parent.record(TEST)}
    for module in (inputs, inputs.qk, inputs.qk.scores, inputs.attribution, channels):
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
            "status": "READ_ONLY_QK_PROJECTION_UNSELECTED_STAGE00_COMPLEMENT",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 8,
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
