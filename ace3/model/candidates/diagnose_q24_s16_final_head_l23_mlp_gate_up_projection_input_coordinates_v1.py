"""Read-only stage13 input accounting at the retained L23 stage16 selections."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_mlp_down_projection_input_coordinates_v1 as down


residual, channels, base, parent = down.residual, down.channels, down.base, down.parent
ROOT = down.ROOT
NAME = "diagnose_q24_s16_final_head_l23_mlp_gate_up_projection_input_coordinates_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, OUTPUT_WIDTH, GROUPS, LIMIT, EXPECTED_TESTS = 896, 4864, 7, 8, 18
STAGES = {"gate_proj": "stage14", "up_proj": "stage15"}
PREFIXES = {kind: f"model.layers.23.mlp.{kind}." for kind in STAGES}
SPECS = {"qweight": ((896, 608), "<i4"), "qzeros": ((7, 608), "<i4"),
         "scales": ((7, 4864), "<f2")}
require, same = base.require, base.same
obj, ordered, array_schema, RATIONAL = down.obj, down.ordered, down.array_schema, down.RATIONAL
FLAGS = {**down.FLAGS, "mlp_gate_up_projection_replay": 0, "silu_replay": 0,
         "rmsnorm_replay": 0}
BOUNDARY = residual.BOUNDARY + (
    " This version follows the eight already selected stage16 input coordinates "
    "of each retained final-head/down-projection hotspot, without reselection. "
    "For each coordinate, gate_proj stage14 and up_proj stage15 deltas are "
    "accounted for by all 896 exact (stage13_actual-stage13_original_input_FP16)*W "
    "terms. Both selections use the same original-input L23 FP16 trajectory. "
    "There are 1152 logical stage16 selections, 2304 projection accounts, "
    "2064384 input terms, 18432 ranked terms and 16128 contiguous G128 groups, "
    "counting repeated selections separately. Eight terms are ranked by descending "
    "absolute contribution then ascending coordinate; all seven groups remain in "
    "input order. Signed/absolute totals and unranked remainders retain cancellation. "
    "Separate retained gate/up deltas minus exact input sums close the two "
    "projection boundaries; these remainders are not solely rounding. The six "
    "authenticated canonical gate/up tensors have no bias bindings. This is "
    "post-RMSNorm input accounting, not raw Q24 propagation through RMSNorm, "
    "SiLU/product attribution, projection replay or a causal share of stage16, "
    "MLP-down or final-head error. Native S16 RTZ is not recomputed."
)

TERM_SCHEMA = obj({
    "coordinate": {"type": "integer", "minimum": 0, "maximum": WIDTH - 1},
    "input_group": {"type": "integer", "minimum": 0, "maximum": GROUPS - 1},
    **{key: RATIONAL for key in ("input_delta", "weight", "signed_contribution")},
})
ACCOUNT_SCHEMA = obj({
    "projection": {"enum": list(STAGES)},
    "output_coordinate": {"type": "integer", "minimum": 0, "maximum": OUTPUT_WIDTH - 1},
    "input_stage": {"const": "stage13_fp16"},
    "output_stage": {"enum": [stage + "_fp16" for stage in STAGES.values()]},
    "reference": {"const": "original_input_L23_fp16"},
    "coordinate_count": {"const": WIDTH},
    "largest_absolute_coordinates": array_schema(TERM_SCHEMA, LIMIT),
    "groups_in_input_order": ordered([
        obj({
            "input_group": {"const": group},
            "start_coordinate": {"const": group * 128},
            "end_coordinate_exclusive": {"const": (group + 1) * 128},
            "coordinate_count": {"const": 128},
            "signed_contribution": RATIONAL, "sum_absolute_contributions": RATIONAL,
        }) for group in range(GROUPS)
    ]),
    **{key: RATIONAL for key in (
        "actual_projection_output", "reference_projection_output", "retained_projection_delta",
        "exact_input_delta", "projection_boundary_remainder", "sum_absolute_contributions",
        "ranked_signed_sum", "ranked_absolute_sum",
        "unranked_signed_remainder", "unranked_absolute_remainder")},
    "exact_local_identity": {"const": True},
})
ACCOUNT_SCHEMA["allOf"] = [
    {"if": {"properties": {"projection": {"const": kind}}},
     "then": {"properties": {"output_stage": {"const": stage + "_fp16"}}}}
    for kind, stage in STAGES.items()
]


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
                        **{kind: {"allOf": [
                            ACCOUNT_SCHEMA, {"properties": {"projection": {"const": kind}}}
                        ]} for kind in STAGES},
                    }) for coordinate_rank in range(1, LIMIT + 1)
                ]),
            }) for rank in range(1, LIMIT + 1)
        ]),
    })


REPORT_SCHEMA = obj({
    **{key: {"const": value} for key, value in {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "selected_stage16_coordinate_count": 1152, "projection_account_count": 2304,
        "input_coordinate_term_count": 2064384, "selected_input_coordinate_count": 18432,
        "input_group_count": 16128, "layer": 23, "position": 0,
    }.items()},
    "controls": ordered([
        obj({"control": {"const": control},
             "branches": ordered([branch_schema(branch) for branch in channels.BRANCHES])})
        for control in parent.CONTROLS
    ]),
    "down_projection_input_coordinate_report": down.REPORT_SCHEMA,
    "L23_reference": residual.REPORT_SCHEMA["properties"]["L23_reference"],
    "gate_up_projection_tensor_bindings": obj({
        prefix + suffix: {"type": "object"} for prefix in PREFIXES.values() for suffix in SPECS
    }),
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in down.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary",
                      "projection_tensors_verified", "dispatch_and_write_audit", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_L23_MLP_GATE_UP_PROJECTION_INPUT_COORDINATES"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "projection_tensors_verified": {"const": 6},
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
        raise RuntimeError("gate/up diagnostic forbids operators, earlier checks and writes")

    with channels.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("projection", "local_reference", "attention_value",
                                  "check", "focused_tests", "rne", "toward_zero"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        for attribute in ("measure", "report"):
            stack.enter_context(patch.object(residual, attribute, forbidden))
        yield


def tensor_names(canonical, kind):
    require(kind in STAGES, "unsupported gate/up projection")
    expected = [PREFIXES[kind] + suffix for suffix in SPECS]
    same(sorted(key for key in canonical if key.startswith(PREFIXES[kind])),
         sorted(expected), "canonical gate/up tensor set changed (no bias permitted)")
    return expected


def bind_tensor(value, pin, kind, suffix):
    require(kind in STAGES and suffix in SPECS, "unsupported gate/up tensor")
    shape, dtype = SPECS[suffix]
    require(isinstance(value, np.ndarray) and value.shape == shape
            and value.dtype.str == dtype and pin["name"] == PREFIXES[kind] + suffix
            and pin["shape"] == list(shape) and pin["dtype"] == str(value.dtype)
            and np.all(np.isfinite(value))
            and hashlib.sha256(value.tobytes()).hexdigest() == pin["sha256"],
            "original AWQ gate/up operand binding changed")
    return value


def weight_column(tensors, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < OUTPUT_WIDTH,
            "gate/up output coordinate outside bounded width")
    shift, packed = down.SHIFTS[coordinate % 8], coordinate // 8
    return [
        Fraction(float(tensors["scales"][i // 128, coordinate])) * (
            ((int(tensors["qweight"][i, packed]) >> shift) & 15)
            - ((int(tensors["qzeros"][i // 128, packed]) >> shift) & 15))
        for i in range(WIDTH)
    ]


def operands(archive):
    return {stage: residual.words(archive[stage], (width,))
            for stage, width in (("stage13", WIDTH), ("stage14", OUTPUT_WIDTH),
                                 ("stage15", OUTPUT_WIDTH))}


def account(actual, reference, weights, kind, coordinate):
    require(kind in STAGES and type(coordinate) is int and 0 <= coordinate < OUTPUT_WIDTH,
            "invalid gate/up projection or output coordinate")
    stage = STAGES[kind]
    require(all(len(source["stage13"]) == WIDTH and len(source[stage]) == OUTPUT_WIDTH
                for source in (actual, reference))
            and len(weights) == WIDTH
            and all(isinstance(value, Fraction) for source in (actual, reference)
                    for name in ("stage13", stage) for value in source[name])
            and all(isinstance(value, Fraction) for value in weights),
            "invalid exact gate/up account operands")
    delta = [a - r for a, r in zip(actual["stage13"], reference["stage13"], strict=True)]
    terms = [value * weight for value, weight in zip(delta, weights, strict=True)]
    top = sorted(range(WIDTH), key=lambda i: (-abs(terms[i]), i))[:LIMIT]
    exact, absolute = sum(terms, Fraction()), sum(map(abs, terms), Fraction())
    ranked = sum((terms[i] for i in top), Fraction())
    ranked_absolute = sum((abs(terms[i]) for i in top), Fraction())
    a, r = actual[stage][coordinate], reference[stage][coordinate]
    groups = []
    for group in range(GROUPS):
        chunk = terms[group * 128:(group + 1) * 128]
        groups.append({
            "input_group": group, "start_coordinate": group * 128,
            "end_coordinate_exclusive": (group + 1) * 128, "coordinate_count": 128,
            "signed_contribution": str(sum(chunk, Fraction())),
            "sum_absolute_contributions": str(sum(map(abs, chunk), Fraction())),
        })
    require(sum(Fraction(g["signed_contribution"]) for g in groups) == exact
            and sum(Fraction(g["sum_absolute_contributions"]) for g in groups) == absolute,
            "exact gate/up G128 group identity failed")
    return {
        "projection": kind, "output_coordinate": coordinate, "input_stage": "stage13_fp16",
        "output_stage": stage + "_fp16", "reference": "original_input_L23_fp16",
        "coordinate_count": WIDTH,
        "largest_absolute_coordinates": [
            {"coordinate": i, "input_group": i // 128, "input_delta": str(delta[i]),
             "weight": str(weights[i]), "signed_contribution": str(terms[i])} for i in top],
        "groups_in_input_order": groups,
        "actual_projection_output": str(a), "reference_projection_output": str(r),
        "retained_projection_delta": str(a - r), "exact_input_delta": str(exact),
        "projection_boundary_remainder": str(a - r - exact),
        "sum_absolute_contributions": str(absolute),
        "ranked_signed_sum": str(ranked), "ranked_absolute_sum": str(ranked_absolute),
        "unranked_signed_remainder": str(exact - ranked),
        "unranked_absolute_remainder": str(absolute - ranked_absolute),
        "exact_local_identity": True,
    }


def report(previous, actual, reference, tensors, binding):
    jsonschema.Draft202012Validator(down.REPORT_SCHEMA).validate(previous)
    same(list(actual), list(parent.CONTROLS), "gate/up control order changed")
    controls, columns, accounts = [], {}, {}
    for row in previous["controls"]:
        branches = []
        for branch in row["branches"]:
            hotspots = []
            for hotspot in branch["hotspots"]:
                selected = hotspot["down_projection"]["largest_absolute_coordinates"]
                require(len({item["coordinate"] for item in selected}) == LIMIT
                        and selected == sorted(selected, key=lambda item: (
                            -abs(Fraction(item["signed_contribution"])), item["coordinate"])),
                        "retained stage16 selection order or uniqueness changed")
                coordinates = []
                for rank, selected_input in enumerate(selected, 1):
                    coordinate = selected_input["coordinate"]
                    entry = {"stage16_coordinate_rank": rank, "selected_down_input": selected_input}
                    for kind in STAGES:
                        if (kind, coordinate) not in columns:
                            columns[kind, coordinate] = weight_column(tensors[kind], coordinate)
                        key = (row["control"], kind, coordinate)
                        if key not in accounts:
                            accounts[key] = account(actual[row["control"]], reference,
                                                    columns[kind, coordinate], kind, coordinate)
                        entry[kind] = accounts[key]
                    coordinates.append(entry)
                hotspots.append({
                    "hotspot_rank": hotspot["hotspot_rank"],
                    "final_head_channel": hotspot["final_head_channel"],
                    "stage16_coordinates": coordinates,
                })
            branches.append({
                "final_head_reference_branch": branch["final_head_reference_branch"],
                "numeric_id": branch["numeric_id"], "hotspots": hotspots,
            })
        controls.append({"control": row["control"], "branches": branches})
    return {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "selected_stage16_coordinate_count": 1152, "projection_account_count": 2304,
        "input_coordinate_term_count": 2064384, "selected_input_coordinate_count": 18432,
        "input_group_count": 16128, "layer": 23, "position": 0, "controls": controls,
        "down_projection_input_coordinate_report": previous,
        "L23_reference": previous["L23_reference"],
        "gate_up_projection_tensor_bindings": {
            name: binding["reference"]["canonical"][name]
            for kind in STAGES for name in tensor_names(binding["reference"]["canonical"], kind)
        },
    }


def measure():
    previous, files, assets = down.measure()
    actual = {control: operands(archive) for control, archive in previous[3].items()}
    reference = operands(previous[4])
    binding = previous[0][0]["preflight"]["L23_original_reference"]
    canonical = binding["reference"]["canonical"]
    tensors = {}
    with channels.contributions.preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        for kind in STAGES:
            tensor_names(canonical, kind)
            tensors[kind] = {
                suffix: bind_tensor(model.get_tensor(PREFIXES[kind] + suffix),
                                    canonical[PREFIXES[kind] + suffix], kind, suffix)
                for suffix in SPECS
            }
    measured = report(previous[6], actual, reference, tensors, binding)
    return (previous, actual, reference, tensors, measured), files, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("l23_mlp_gate_up_coordinate_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "gate/up focused tests failed, errored or skipped")
    return {"executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "gate_up_focused_test": parent.record(TEST),
               down.MODULE: parent.record(down.SOURCE),
               residual.MODULE: parent.record(residual.SOURCE),
               channels.MODULE: parent.record(channels.SOURCE)}
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
            "status": "READ_ONLY_L23_MLP_GATE_UP_PROJECTION_INPUT_COORDINATES",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "L23_archives_verified": 10, "projection_tensors_verified": 6,
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
