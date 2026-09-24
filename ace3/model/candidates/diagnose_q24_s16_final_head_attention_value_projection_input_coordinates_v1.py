"""Read-only stage00 input-coordinate accounting for retained P0 V hotspots.

--check compiles this module and its focused tests in memory, runs only those
tests, and emits one schema-checked JSON document without writing evidence.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_source_value_attribution_v1 as attribution


base, parent, channels = attribution.base, attribution.parent, attribution.channels
ROOT = attribution.ROOT
NAME = "diagnose_q24_s16_final_head_attention_value_projection_input_coordinates_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 18
WIDTH, LIMIT = 896, 8
PROJECTION = "model.layers.23.self_attn.v_proj."
require, same = base.require, base.same
FLAGS = {**attribution.FLAGS, "v_projection_replay": 0}
BOUNDARY = attribution.BOUNDARY + (
    " This version follows each selected largest value component to authenticated "
    "L23 v_proj stage00 FP16 actual-minus-original-reference input coordinates. "
    "The input residual is a post-RMSNorm operator-input difference, not propagation "
    "of raw Q24 residual state through nonlinear RMSNorm. All 896 exact delta-input "
    "times native AWQ weight terms are ranked by absolute signed contribution then "
    "coordinate; seven contiguous G128 groups rank by absolute net sum then group. "
    "Shared authenticated FP16 biases cancel. Retained stage03 V delta minus the "
    "exact input sum is a projection boundary remainder, not solely rounding. "
    "Stage03/stage07/output-cache lineage and original empty P0 KV are preserved. "
    "The original-reference probability times o_proj GQA weight sum links this "
    "local projection identity to the selected retained value contribution only. "
    "No v_proj or prior diagnostic check execution; no binary64 attention "
    "substitution, operator replay, precision/scale expansion or final-head causality."
)
obj, RATIONAL = channels.object_schema, channels.RATIONAL


def array_schema(items, count):
    return {"type": "array", "minItems": count, "maxItems": count, "items": items}


TERM_SCHEMA = obj({
    "coordinate": {"type": "integer", "minimum": 0, "maximum": 895},
    "input_group": {"type": "integer", "minimum": 0, "maximum": 6},
    **{key: RATIONAL for key in ("input_delta", "weight", "signed_contribution")},
})
GROUP_SCHEMA = obj({
    "input_group": {"type": "integer", "minimum": 0, "maximum": 6},
    "start_coordinate": {"enum": list(range(0, WIDTH, 128))},
    "end_coordinate_exclusive": {"enum": list(range(128, WIDTH + 1, 128))},
    "coordinate_count": {"const": 128},
    **{key: RATIONAL for key in ("signed_contribution", "sum_absolute_contributions")},
})
ACCOUNT_SCHEMA = obj({
    "projection": {"const": "v_proj"}, "layer": {"const": 23},
    "position": {"const": 0}, "input_stage": {"const": "stage00_fp16"},
    "output_stage": {"const": "stage03_fp16"},
    "output_coordinate": {"type": "integer", "minimum": 0, "maximum": 127},
    "coordinate_count": {"const": WIDTH},
    "largest_absolute_coordinates": array_schema(TERM_SCHEMA, LIMIT),
    "groups_by_absolute_signed_sum": array_schema(GROUP_SCHEMA, 7),
    **{key: RATIONAL for key in (
        "actual_v", "reference_v", "exact_input_delta", "retained_projection_delta",
        "projection_boundary_remainder", "ranked_signed_sum", "unranked_signed_remainder",
        "sum_absolute_contributions", "value_component_factor",
        "weighted_input_contribution", "weighted_boundary_contribution",
        "retained_value_contribution")},
    "exact_local_identity": {"const": True},
    "exact_value_component_identity": {"const": True},
})
COMPONENT_SCHEMA = obj({
    "value_hotspot_rank": {"type": "integer", "minimum": 1, "maximum": LIMIT},
    "retained_value_component": attribution.COMPONENT_SCHEMA,
    "value_projection": ACCOUNT_SCHEMA,
})
REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "selected_row_count": {"const": 18},
    "hotspot_count": {"const": 144}, "selected_value_component_count": {"const": 1152},
    "projection_account_count": {"const": 1152},
    "input_coordinate_term_count": {"const": 1032192},
    "selected_input_coordinate_count": {"const": 9216},
    "input_group_count": {"const": 8064},
    "attention_reference": {"const": "original_input_fp16"},
    "controls": array_schema(obj({
        "control": {"enum": list(parent.CONTROLS)},
        "branches": obj({branch: obj({
            "numeric_id": channels.ROW_SCHEMA["properties"]["numeric_id"],
            "retained_signed_residual": RATIONAL,
            "hotspots": array_schema(obj({
                "hotspot_rank": {"type": "integer", "minimum": 1, "maximum": LIMIT},
                "final_head_channel": channels.ENTRY_SCHEMA,
                "value_components": array_schema(COMPONENT_SCHEMA, LIMIT),
            }), LIMIT),
        }) for branch in channels.BRANCHES}),
    }), 9),
    "source_value_attribution_report": attribution.REPORT_SCHEMA,
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in attribution.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary",
                      "dispatch_and_write_audit", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_ATTENTION_VALUE_PROJECTION_INPUT_COORDINATES"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "projection_tensors_verified": {"const": 7},
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
        raise RuntimeError("V input diagnostic forbids prior checks and operator replay")

    with attribution.read_only(audit), patch.object(attribution, "check", forbidden), \
            patch.object(attribution, "focused_tests", forbidden):
        yield


def retained_input(pin, *, actual, expected_attention):
    with np.load(io.BytesIO(base.read_bound(pin)), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate projection fields")
        require(attribution.attention_operands(archive, actual=actual) == expected_attention,
                "projection/attention archive splice")
        return attribution.words(archive["stage00"], (WIDTH,))


def bind_tensor(value, pin, suffix):
    specs = {"qweight": ((WIDTH, 16), "<i4"), "qzeros": ((7, 16), "<i4"),
             "scales": ((7, 128), "<f2"), "bias": ((128,), "<f2")}
    require(suffix in specs, "unsupported V tensor")
    shape, dtype = specs[suffix]
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.str == dtype
            and pin["name"] == PROJECTION + suffix and pin["shape"] == list(shape)
            and pin["dtype"] == str(value.dtype) and np.all(np.isfinite(value))
            and hashlib.sha256(value.tobytes()).hexdigest() == pin["sha256"],
            "original AWQ V operand binding changed")
    return value


def load_inputs(previous, assets):
    result = previous[0][0]
    binding = result["preflight"]["L23_original_reference"]["reference"]
    reference = retained_input(binding["fp16"], actual=False, expected_attention=previous[2])
    actual = {
        row["control"]: retained_input(row["parent"]["terminal_archive"], actual=True,
                                       expected_attention=previous[1][row["control"]])
        for row in result["controls"]}
    tensors = {}
    with channels.contributions.preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        for suffix in ("qweight", "qzeros", "scales", "bias"):
            name = PROJECTION + suffix
            tensors[suffix] = bind_tensor(model.get_tensor(name), binding["canonical"][name], suffix)
    return actual, reference, tensors


def weight_column(tensors, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < 128,
            "V output coordinate outside bounded width")
    return attribution.weight_column(tensors, coordinate)


def account(actual, reference, weights, coordinate, actual_v, reference_v, factor, component):
    require(type(coordinate) is int and 0 <= coordinate < 128 and all(
        len(values) == WIDTH and all(isinstance(v, Fraction) for v in values)
        for values in (actual, reference, weights))
        and all(isinstance(v, Fraction) for v in (actual_v, reference_v, factor)),
        "invalid exact V projection inputs")
    jsonschema.Draft202012Validator(attribution.COMPONENT_SCHEMA).validate(component)
    same(component["value_coordinate"], coordinate, "V component coordinate splice")
    retained = actual_v - reference_v
    require(Fraction(component["probability"]) == Fraction(component["interaction"]) == 0
            and Fraction(component["value"]) == Fraction(component["signed_contribution"])
            == retained * factor, "retained value component identity changed")
    deltas = [a - r for a, r in zip(actual, reference, strict=True)]
    values = [delta * weight for delta, weight in zip(deltas, weights, strict=True)]
    order = sorted(range(WIDTH), key=lambda i: (-abs(values[i]), i))
    exact = sum(values, Fraction())
    boundary = retained - exact
    ranked = sum((values[i] for i in order[:LIMIT]), Fraction())
    groups = []
    for start in range(0, WIDTH, 128):
        terms = values[start:start + 128]
        groups.append({
            "input_group": start // 128, "start_coordinate": start,
            "end_coordinate_exclusive": start + 128, "coordinate_count": 128,
            "signed_contribution": str(sum(terms, Fraction())),
            "sum_absolute_contributions": str(sum(map(abs, terms), Fraction())),
        })
    groups.sort(key=lambda e: (-abs(Fraction(e["signed_contribution"])), e["input_group"]))
    require(sum(Fraction(g["signed_contribution"]) for g in groups) == exact
            and exact + boundary == retained
            and exact * factor + boundary * factor == Fraction(component["value"]),
            "V projection local identity failed")
    return {
        "projection": "v_proj", "layer": 23, "position": 0,
        "input_stage": "stage00_fp16", "output_stage": "stage03_fp16",
        "output_coordinate": coordinate, "coordinate_count": WIDTH,
        "largest_absolute_coordinates": [
            {"coordinate": i, "input_group": i // 128, "input_delta": str(deltas[i]),
             "weight": str(weights[i]), "signed_contribution": str(values[i])}
            for i in order[:LIMIT]],
        "groups_by_absolute_signed_sum": groups,
        "actual_v": str(actual_v), "reference_v": str(reference_v),
        "exact_input_delta": str(exact), "retained_projection_delta": str(retained),
        "projection_boundary_remainder": str(boundary), "ranked_signed_sum": str(ranked),
        "unranked_signed_remainder": str(exact - ranked),
        "sum_absolute_contributions": str(sum(map(abs, values), Fraction())),
        "value_component_factor": str(factor),
        "weighted_input_contribution": str(exact * factor),
        "weighted_boundary_contribution": str(boundary * factor),
        "retained_value_contribution": str(retained * factor),
        "exact_local_identity": True, "exact_value_component_identity": True,
    }


def report(previous, actual, reference, tensors):
    inherited = previous[4]
    jsonschema.Draft202012Validator(attribution.REPORT_SCHEMA).validate(inherited)
    same(list(actual), list(parent.CONTROLS), "V input control order changed")
    same(list(previous[1]), list(parent.CONTROLS), "V attention control order changed")
    same([row["control"] for row in inherited["controls"]], list(parent.CONTROLS),
         "source/value control order changed")
    controls, columns, output_columns = [], {}, {}
    for row, source in zip(inherited["controls"],
                           inherited["final_head_channel_report"]["controls"], strict=True):
        same(row["control"], source["control"], "source/value control splice")
        same(list(row["branches"]), list(channels.BRANCHES), "reference branch order changed")
        branches = {}
        for name, branch in row["branches"].items():
            original = source["branches"][name]
            same((branch["numeric_id"], branch["retained_signed_residual"]),
                 (original["numeric_id"], original["retained_signed_residual"]),
                 "final-head selected row splice")
            same([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, 9)),
                 "final-head hotspot order changed")
            same([h["final_head_channel"] for h in branch["hotspots"]],
                 original["channels"]["largest_absolute_signed"], "final-head selection changed")
            hotspots = []
            for hotspot in branch["hotspots"]:
                output = hotspot["final_head_channel"]["coordinate"]
                if output not in output_columns:
                    output_columns[output] = attribution.weight_column(previous[3], output)
                ordered = hotspot["attention"]["full_value_component_ranking"]
                same(sorted(e["value_coordinate"] for e in ordered), list(range(128)),
                     "value component census changed")
                same(ordered, sorted(ordered, key=lambda e: (
                    -abs(Fraction(e["signed_contribution"])), e["value_coordinate"])),
                     "value component order changed")
                same(hotspot["attention"]["largest_absolute_value_components"], ordered[:LIMIT],
                     "value hotspot selection changed")
                components = []
                for rank, component in enumerate(ordered[:LIMIT], 1):
                    coordinate = component["value_coordinate"]
                    kv, dim = divmod(coordinate, 64)
                    heads = list(range(kv * 7, (kv + 1) * 7))
                    same((component["kv_head"], component["head_dimension"], component["query_heads"]),
                         (kv, dim, heads), "value GQA mapping changed")
                    if coordinate not in columns:
                        columns[coordinate] = weight_column(tensors, coordinate)
                    factor = sum((previous[2][0][h] * output_columns[output][h * 64 + dim]
                                  for h in heads), Fraction())
                    projection = account(
                        actual[row["control"]], reference, columns[coordinate], coordinate,
                        previous[1][row["control"]][1][coordinate], previous[2][1][coordinate],
                        factor, component)
                    components.append({"value_hotspot_rank": rank,
                                       "retained_value_component": component,
                                       "value_projection": projection})
                hotspots.append({"hotspot_rank": hotspot["hotspot_rank"],
                                 "final_head_channel": hotspot["final_head_channel"],
                                 "value_components": components})
            branches[name] = {"numeric_id": branch["numeric_id"],
                              "retained_signed_residual": branch["retained_signed_residual"],
                              "hotspots": hotspots}
        controls.append({"control": row["control"], "branches": branches})
    rows = sum(len(row["branches"]) for row in controls)
    hotspots = sum(len(b["hotspots"]) for row in controls for b in row["branches"].values())
    accounts = sum(len(h["value_components"]) for row in controls
                   for b in row["branches"].values() for h in b["hotspots"])
    return {
        "control_count": len(controls), "selected_row_count": rows, "hotspot_count": hotspots,
        "selected_value_component_count": accounts, "projection_account_count": accounts,
        "input_coordinate_term_count": accounts * WIDTH,
        "selected_input_coordinate_count": accounts * LIMIT, "input_group_count": accounts * 7,
        "attention_reference": "original_input_fp16", "controls": controls,
        "source_value_attribution_report": inherited,
    }


def measure():
    previous, files, assets = attribution.measure()
    actual, reference, tensors = load_inputs(previous, assets)
    measured = report(previous, actual, reference, tensors)
    return (previous, actual, reference, tensors, measured), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("value_projection_input_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "V projection input focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors),
            "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "value_projection_input_focused_test": parent.record(TEST)}
    for module in (attribution, channels):
        origins[module.MODULE] = parent.record(module.SOURCE)
        origins[module.MODULE + ".focused_test"] = parent.record(module.TEST)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *channels.PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or evidence write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_ATTENTION_VALUE_PROJECTION_INPUT_COORDINATES",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 7,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "tests": tests, "report": evidence[4],
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
