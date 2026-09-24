"""Stdout-only exact complement of retained attention source/value components."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_source_value_attribution_v1 as attribution


base, parent, channels = attribution.base, attribution.parent, attribution.channels
ROOT = attribution.ROOT
NAME = "diagnose_q24_s16_final_head_attention_source_value_unselected_component_complement_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
COMPONENTS, LIMIT, EXPECTED_TESTS = 128, attribution.LIMIT, 24
PARTS, FLAGS = attribution.PARTS, dict(attribution.FLAGS)
require, same = base.require, base.same
obj, RATIONAL = attribution.obj, attribution.RATIONAL
BOUNDARY = attribution.BOUNDARY + (
    " This extension accounts only for the 120 value components outside each "
    "unchanged eight-component selection, directly from the full retained exact "
    "ranking. There are 17280 logical unselected components across 144 hotspot "
    "accounts, including repeated selections. Absolute accounts sum absolute "
    "NET component contributions after GQA query-head aggregation, not absolute "
    "individual query-head products; separate absolute probability/value/interaction "
    "accounts are also retained. Selected plus unselected signed totals close "
    "exact_attention_sum; adding unchanged AV and o_projection boundary remainders "
    "closes unchanged retained_o_projection_delta. Full rankings and source-token "
    "totals are preserved. This is no bridge selection, dominance attribution, "
    "margin/ranking flip or executed counterfactual, and does not extend MLP/RMSNorm."
)


def array_schema(items, count):
    return {"type": "array", "minItems": count, "maxItems": count, "items": items}


TOTAL_SCHEMA = obj({
    **{key: RATIONAL for key in (*PARTS, "signed_contribution",
                                "sum_absolute_contributions")},
    "absolute_parts": obj({key: RATIONAL for key in PARTS}),
})
TOTALS_SCHEMA = {key: TOTAL_SCHEMA for key in ("selected", "unselected", "full")}
COORDINATE_SCHEMA = {"type": "integer", "minimum": 0, "maximum": COMPONENTS - 1}
SPLIT_SCHEMA = obj({
    "attention_reference": {"const": "original_input_fp16"},
    "selected_component_count": {"const": LIMIT},
    "unselected_component_count": {"const": COMPONENTS - LIMIT},
    "unselected_query_value_product_count": {"const": (COMPONENTS - LIMIT) * 7},
    "excluded_ranked_value_coordinates": {
        **array_schema(COORDINATE_SCHEMA, LIMIT), "uniqueItems": True},
    "unselected_value_coordinates_in_input_order": {
        **array_schema(COORDINATE_SCHEMA, COMPONENTS - LIMIT), "uniqueItems": True},
    **TOTALS_SCHEMA,
    "source_tokens": array_schema(obj({
        "source_position": {"const": 0}, "source_token_id": {"const": 9707},
        "selected_component_count": {"const": LIMIT},
        "unselected_component_count": {"const": COMPONENTS - LIMIT},
        **TOTALS_SCHEMA,
    }), 1),
    **{key: {"const": True} for key in (
        "exact_complement_identity", "exact_source_token_identity",
        "exact_attention_identity", "exact_local_identity")},
})
ACCOUNT_SCHEMA = obj({
    **attribution.ACCOUNT_SCHEMA["properties"],
    "unselected_value_component_complement": SPLIT_SCHEMA,
})
REPORT_SCHEMA = deepcopy(attribution.REPORT_SCHEMA)
for key, count in (("selected_value_component_count", 1152),
                   ("unselected_value_component_count", 17280),
                   ("unselected_query_value_product_count", 120960)):
    REPORT_SCHEMA["properties"][key] = {"const": count}
    REPORT_SCHEMA["required"].append(key)
for branch in REPORT_SCHEMA["properties"]["controls"]["items"]["properties"]["branches"]["properties"].values():
    branch["properties"]["hotspots"]["items"]["properties"]["attention"] = ACCOUNT_SCHEMA
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in attribution.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_ATTENTION_SOURCE_VALUE_UNSELECTED_COMPONENT_COMPLEMENT"},
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
        raise RuntimeError("value-component complement forbids prior checks and operator replay")

    with attribution.read_only(audit), patch.object(attribution, "check", forbidden), \
            patch.object(attribution, "focused_tests", forbidden):
        yield


def component_totals(entries):
    return {
        **{key: str(sum((Fraction(e[key]) for e in entries), Fraction()))
           for key in (*PARTS, "signed_contribution")},
        "sum_absolute_contributions": str(sum(
            (abs(Fraction(e["signed_contribution"])) for e in entries), Fraction())),
        "absolute_parts": {
            key: str(sum((abs(Fraction(e[key])) for e in entries), Fraction()))
            for key in PARTS},
    }


def split_unselected(account):
    jsonschema.Draft202012Validator(attribution.ACCOUNT_SCHEMA).validate(account)
    ranked = account["full_value_component_ranking"]
    selected = account["largest_absolute_value_components"]
    same(sorted(e["value_coordinate"] for e in ranked), list(range(COMPONENTS)),
         "value-component partition changed")
    same(ranked, sorted(ranked, key=lambda e: (
        -abs(Fraction(e["signed_contribution"])), e["value_coordinate"])),
        "full value-component ranking changed")
    same(selected, ranked[:LIMIT], "selected value-component entries changed")
    for entry in ranked:
        kv, dim = divmod(entry["value_coordinate"], 64)
        same((entry["source_position"], entry["source_token_id"], entry["kv_head"],
              entry["head_dimension"], entry["query_heads"]),
             (0, 9707, kv, dim, list(range(kv * 7, (kv + 1) * 7))),
             "source/value/GQA identity changed")
        require(Fraction(entry["signed_contribution"]) ==
                sum((Fraction(entry[key]) for key in PARTS), Fraction()),
                "probability/value/interaction component identity changed")
        require(Fraction(entry["probability"]) == Fraction(entry["interaction"]) == 0,
                "retained singleton-P0 probability/interaction changed")
    excluded = [e["value_coordinate"] for e in selected]
    complement = sorted(ranked[LIMIT:], key=lambda e: e["value_coordinate"])
    totals = {key: component_totals(entries) for key, entries in
              (("selected", selected), ("unselected", complement), ("full", ranked))}
    for key in (*PARTS, "signed_contribution", "sum_absolute_contributions"):
        require(Fraction(totals["selected"][key]) + Fraction(totals["unselected"][key]) ==
                Fraction(totals["full"][key]), "selected/unselected complement identity changed")
    token = account["source_tokens"][0]
    for key in (*PARTS, "signed_contribution"):
        same(token[key], totals["full"][key], "source-token total changed")
    for key, value in (
        ("ranked_signed_sum", totals["selected"]["signed_contribution"]),
        ("unranked_signed_remainder", totals["unselected"]["signed_contribution"]),
        ("exact_attention_sum", totals["full"]["signed_contribution"]),
    ):
        same(account[key], value, f"attention complement closure changed: {key}")
    exact = Fraction(totals["full"]["signed_contribution"])
    av = Fraction(account["av_boundary_remainder"])
    projection = Fraction(account["o_projection_boundary_remainder"])
    require(av == 0, "retained singleton-P0 AV boundary changed")
    require(exact + av + projection == Fraction(account["retained_o_projection_delta"]),
            "retained o_projection boundary identity changed")
    return {
        "attention_reference": "original_input_fp16",
        "selected_component_count": LIMIT,
        "unselected_component_count": len(complement),
        "unselected_query_value_product_count": len(complement) * 7,
        "excluded_ranked_value_coordinates": excluded,
        "unselected_value_coordinates_in_input_order": [e["value_coordinate"] for e in complement],
        **totals,
        "source_tokens": [{
            "source_position": token["source_position"], "source_token_id": token["source_token_id"],
            "selected_component_count": LIMIT, "unselected_component_count": len(complement),
            **deepcopy(totals),
        }],
        "exact_complement_identity": True, "exact_source_token_identity": True,
        "exact_attention_identity": True, "exact_local_identity": True,
    }


def report(inherited):
    jsonschema.Draft202012Validator(attribution.REPORT_SCHEMA).validate(inherited)
    same([row["control"] for row in inherited["controls"]], list(parent.CONTROLS),
         "attention complement control order changed")
    measured = deepcopy(inherited)
    count = 0
    for row, original in zip(measured["controls"],
                             inherited["final_head_channel_report"]["controls"], strict=True):
        same(row["control"], original["control"], "final-head control splice")
        same(list(row["branches"]), list(channels.BRANCHES), "reference branch order changed")
        for name, branch in row["branches"].items():
            source = original["branches"][name]
            same(branch["numeric_id"], source["numeric_id"], "final-head row splice")
            same(branch["retained_signed_residual"], source["retained_signed_residual"],
                 "final-head residual splice")
            same([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, LIMIT + 1)),
                 "hotspot rank changed")
            same([h["final_head_channel"] for h in branch["hotspots"]],
                 source["channels"]["largest_absolute_signed"], "final-head hotspot splice")
            for hotspot in branch["hotspots"]:
                account = hotspot["attention"]
                # Repeated hotspots can share an account in the inherited report.
                if "unselected_value_component_complement" not in account:
                    account["unselected_value_component_complement"] = split_unselected(account)
                count += account["unselected_value_component_complement"]["unselected_component_count"]
    measured["selected_value_component_count"] = measured["hotspot_count"] * LIMIT
    measured["unselected_value_component_count"] = count
    measured["unselected_query_value_product_count"] = count * 7
    same(count, 17280, "logical unselected component census changed")
    return measured


def measure():
    evidence, files, assets = attribution.measure()
    return (evidence, report(evidence[4])), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("attention_component_complement_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "value-component complement focused tests failed, errored or skipped")
    return {"executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "attention_component_complement_focused_test": parent.record(TEST)}
    for module in (attribution, channels):
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
            "status": "READ_ONLY_ATTENTION_SOURCE_VALUE_UNSELECTED_COMPONENT_COMPLEMENT",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 3,
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
