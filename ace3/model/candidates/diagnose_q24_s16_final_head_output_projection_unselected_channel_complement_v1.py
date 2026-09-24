"""Stdout-only exact G128 complement of retained tied-head input-channel hotspots."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_output_projection_channel_hotspots_v1 as channels


base, parent = channels.base, channels.parent
ROOT = channels.ROOT
NAME = "diagnose_q24_s16_final_head_output_projection_unselected_channel_complement_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, GROUP_SIZE = channels.WIDTH, channels.RANK_LIMIT, 128
GROUPS, EXPECTED_TESTS = WIDTH // GROUP_SIZE, 24
BRANCHES, PINS = channels.BRANCHES, channels.PINS
FLAGS = dict(channels.FLAGS)
require, same = base.require, base.same
obj, RATIONAL = channels.object_schema, channels.RATIONAL
BOUNDARY = channels.BOUNDARY + (
    " This extension preserves every selected row/reference hotspot account and "
    "splits only the 888 input channels outside its unchanged largest-absolute eight "
    "into seven contiguous G128 groups in input order. Each group reports the exact "
    "signed sum and sum of absolute individual contributions, not the absolute net "
    "group sum. The 18 accounts contain 15984 logical unselected terms and 126 "
    "complement groups. Selected plus unselected plus the unchanged head-boundary "
    "remainder equals the retained signed residual; selected and unselected absolute "
    "sums equal the full absolute channel account. This is individual-row accounting, "
    "not pair-margin bridge selection, a dominance claim, ranking reversal or an "
    "executed counterfactual. No additional reference trajectory is reconstructed."
)

SPLIT_SCHEMA = obj({
    "coordinate_count": {"const": WIDTH - LIMIT},
    "excluded_ranked_coordinates": {
        "type": "array", "minItems": LIMIT, "maxItems": LIMIT, "uniqueItems": True,
        "items": channels.INDEX,
    },
    "groups_in_input_order": {
        "type": "array", "minItems": GROUPS, "maxItems": GROUPS, "items": False,
        "prefixItems": [
            obj({
                "input_group": {"const": group},
                "start_coordinate": {"const": group * GROUP_SIZE},
                "end_coordinate_exclusive": {"const": (group + 1) * GROUP_SIZE},
                "coordinate_count": {
                    "type": "integer", "minimum": GROUP_SIZE - LIMIT, "maximum": GROUP_SIZE},
                "excluded_ranked_coordinate_count": {
                    "type": "integer", "minimum": 0, "maximum": LIMIT},
                "signed_contribution": RATIONAL, "sum_absolute_contributions": RATIONAL,
            }) for group in range(GROUPS)
        ],
    },
    "selected_signed_sum": RATIONAL, "selected_absolute_sum": RATIONAL,
    "unselected_signed_sum": RATIONAL, "unselected_absolute_sum": RATIONAL,
    "exact_complement_identity": {"const": True},
    "exact_absolute_identity": {"const": True},
    "exact_residual_identity": {"const": True},
})
ROW_SCHEMA = obj({**channels.ROW_SCHEMA["properties"], "unselected_input_channels": SPLIT_SCHEMA})
REPORT_SCHEMA = deepcopy(channels.REPORT_SCHEMA)
REPORT_SCHEMA["properties"].update({
    "unselected_input_coordinate_count": {"const": 18 * (WIDTH - LIMIT)},
    "complement_group_count": {"const": 18 * GROUPS},
})
REPORT_SCHEMA["required"].extend(("unselected_input_coordinate_count", "complement_group_count"))
REPORT_SCHEMA["properties"]["controls"]["items"]["properties"]["branches"] = obj({
    branch: ROW_SCHEMA for branch in BRANCHES})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in channels.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_OUTPUT_PROJECTION_UNSELECTED_CHANNEL_COMPLEMENT"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "tests": obj({
        "compiled": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "object"}},
        "executed": {"const": EXPECTED_TESTS},
        **{key: {"const": 0} for key in ("failures", "errors", "skipped")},
    }),
    "report": REPORT_SCHEMA,
})


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("head complement forbids earlier checks and operator replay")

    with channels.read_only(audit), patch.object(channels, "check", forbidden), \
            patch.object(channels, "focused_tests", forbidden):
        yield


def split_unselected(account):
    jsonschema.Draft202012Validator(channels.ROW_SCHEMA).validate(account)
    summary = account["channels"]
    full = summary["full_absolute_signed_ranking"]
    same(sorted(entry["coordinate"] for entry in full), list(range(WIDTH)),
         "head complement coordinate census changed")
    same(full, sorted(full, key=lambda entry: (
        -abs(Fraction(entry["signed_contribution"])), entry["coordinate"])),
        "head complement ranking changed")
    selected = summary["largest_absolute_signed"]
    same(selected, full[:LIMIT], "head complement selected channels changed")
    excluded = [entry["coordinate"] for entry in selected]
    values = {entry["coordinate"]: Fraction(entry["signed_contribution"]) for entry in full}
    groups = []
    for group in range(GROUPS):
        start, end = group * GROUP_SIZE, (group + 1) * GROUP_SIZE
        terms = [values[i] for i in range(start, end) if i not in excluded]
        groups.append({
            "input_group": group, "start_coordinate": start, "end_coordinate_exclusive": end,
            "coordinate_count": len(terms),
            "excluded_ranked_coordinate_count": GROUP_SIZE - len(terms),
            "signed_contribution": str(sum(terms, Fraction())),
            "sum_absolute_contributions": str(sum(map(abs, terms), Fraction())),
        })
    signed = sum((Fraction(group["signed_contribution"]) for group in groups), Fraction())
    absolute = sum((Fraction(group["sum_absolute_contributions"]) for group in groups), Fraction())
    ranked = sum((values[i] for i in excluded), Fraction())
    ranked_absolute = sum((abs(values[i]) for i in excluded), Fraction())
    same(str(ranked), summary["ranked_signed_sum"], "selected signed account changed")
    same(str(signed), summary["unranked_exact_remainder"], "signed head complement failed")
    same(str(ranked + signed), summary["exact_sum"], "full signed head account failed")
    same(str(ranked_absolute + absolute), summary["exact_sum_of_absolute_contributions"],
         "absolute head complement failed")
    same(str(ranked + signed + Fraction(account["boundary_remainder"])),
         account["retained_signed_residual"], "head complement residual identity failed")
    require(sum(group["coordinate_count"] for group in groups) == WIDTH - LIMIT,
            "head complement coordinate count failed")
    return {
        "coordinate_count": WIDTH - LIMIT, "excluded_ranked_coordinates": excluded,
        "groups_in_input_order": groups,
        "selected_signed_sum": str(ranked), "selected_absolute_sum": str(ranked_absolute),
        "unselected_signed_sum": str(signed), "unselected_absolute_sum": str(absolute),
        "exact_complement_identity": True, "exact_absolute_identity": True,
        "exact_residual_identity": True,
    }


def report(evidence):
    result, arrays, references, selected, weights, inherited = evidence
    jsonschema.Draft202012Validator(channels.REPORT_SCHEMA).validate(inherited)
    require(selected == channels.selection(result, arrays, references),
            "head complement retained row/reference selection changed")
    same(inherited, channels.report(result, arrays, references, selected, weights),
         "head complement inherited hotspot account changed")
    measured = deepcopy(inherited)
    count, groups = 0, 0
    for row in measured["controls"]:
        for branch in BRANCHES:
            account = row["branches"][branch]
            split = split_unselected(account)
            account["unselected_input_channels"] = split
            count += split["coordinate_count"]
            groups += len(split["groups_in_input_order"])
    measured["unselected_input_coordinate_count"] = count
    measured["complement_group_count"] = groups
    return measured


def measure():
    evidence, files, assets = channels.measure()
    return (evidence, report(evidence)), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("head_channel_complement_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "head complement focused tests failed, errored or skipped")
    return {"executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "head_channel_complement_focused_test": parent.record(TEST)}
    for module in (channels, channels.hotspots):
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
        for pin in (*origins.values(), *base.PINS.values(), *PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or evidence write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_OUTPUT_PROJECTION_UNSELECTED_CHANNEL_COMPLEMENT",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "pins": {**base.PINS, **PINS}, "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attempt001_files_verified": 20, "arrays_verified": 18, "reference_arrays_verified": 4,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "flags": FLAGS,
            "tests": tests, "report": evidence[1],
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
