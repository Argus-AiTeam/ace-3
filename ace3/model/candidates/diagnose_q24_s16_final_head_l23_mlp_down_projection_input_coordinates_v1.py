"""Read-only exact stage16-input accounting for retained L23 MLP-down deltas."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_residual_branch_delta_v1 as residual


channels, base, parent = residual.channels, residual.base, residual.parent
ROOT = residual.ROOT
NAME = "diagnose_q24_s16_final_head_l23_mlp_down_projection_input_coordinates_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, OUTPUT_WIDTH, LIMIT, GROUPS, EXPECTED_TESTS = 4864, 896, 8, 38, 16
PREFIX = "model.layers.23.mlp.down_proj."
SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
SPECS = {"qweight": ((WIDTH, 112), "<i4"), "qzeros": ((GROUPS, 112), "<i4"),
         "scales": ((GROUPS, OUTPUT_WIDTH), "<f2")}
require, same = base.require, base.same
obj, ordered, RATIONAL = residual.obj, residual.ordered_schema, residual.RATIONAL
FLAGS = {**residual.FLAGS, "mlp_down_projection_replay": 0}
BOUNDARY = residual.BOUNDARY + (
    " This version accounts for each retained stage17 MLP-down delta with all "
    "4864 exact (stage16_actual-stage16_original_input_FP16)*W_down terms, "
    "eight coordinates ordered by descending absolute contribution then ascending "
    "coordinate, and all 38 contiguous G128 groups in input order. Both final-head "
    "reference-branch hotspot selections use the same original-input L23 FP16 "
    "stage16/stage17 reference, not binary64 internal stages. The separate "
    "down-projection boundary remainder is retained stage17 delta minus the exact "
    "input sum; it is not solely rounding, an MLP replay, or propagation through "
    "SiLU, RMSNorm or the head. Signed/absolute unranked remainders retain every "
    "input term, including cancellation. The 144 logical accounts count repeated "
    "hotspots separately: 700416 terms and 5472 groups. No bias tensor is present "
    "in the authenticated original down-projection bindings."
)


def array_schema(item, count):
    return {"type": "array", "minItems": count, "maxItems": count, "items": item}


TERM_SCHEMA = obj({
    "coordinate": {"type": "integer", "minimum": 0, "maximum": WIDTH - 1},
    "input_group": {"type": "integer", "minimum": 0, "maximum": GROUPS - 1},
    **{key: RATIONAL for key in ("input_delta", "weight", "signed_contribution")},
})
ACCOUNT_SCHEMA = obj({
    "output_coordinate": channels.INDEX,
    "input_stage": {"const": "stage16_fp16"},
    "output_stage": {"const": "stage17_fp16"},
    "reference": {"const": "original_input_L23_fp16"},
    "coordinate_count": {"const": WIDTH},
    "largest_absolute_coordinates": array_schema(TERM_SCHEMA, LIMIT),
    "groups_in_input_order": ordered([
        obj({
            "input_group": {"const": group},
            "start_coordinate": {"const": group * 128},
            "end_coordinate_exclusive": {"const": (group + 1) * 128},
            "coordinate_count": {"const": 128},
            "signed_contribution": RATIONAL,
            "sum_absolute_contributions": RATIONAL,
        }) for group in range(GROUPS)
    ]),
    **{key: RATIONAL for key in (
        "actual_stage17", "reference_stage17", "retained_projection_delta",
        "exact_input_delta", "down_projection_boundary_remainder",
        "sum_absolute_contributions", "ranked_signed_sum", "ranked_absolute_sum",
        "unranked_signed_remainder", "unranked_absolute_remainder")},
    "exact_local_identity": {"const": True},
})


def branch_schema(branch):
    return obj({
        "final_head_reference_branch": {"const": branch},
        "numeric_id": channels.ROW_SCHEMA["properties"]["numeric_id"],
        "hotspots": ordered([
            obj({"hotspot_rank": {"const": rank}, "final_head_channel": channels.ENTRY_SCHEMA,
                 "down_projection": ACCOUNT_SCHEMA})
            for rank in range(1, LIMIT + 1)
        ]),
    })


REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "selected_row_count": {"const": 18},
    "projection_account_count": {"const": 144},
    "input_coordinate_term_count": {"const": 700416},
    "selected_input_coordinate_count": {"const": 1152},
    "input_group_count": {"const": 5472},
    "layer": {"const": 23}, "position": {"const": 0},
    "controls": ordered([
        obj({"control": {"const": control},
             "branches": ordered([branch_schema(branch) for branch in channels.BRANCHES])})
        for control in parent.CONTROLS
    ]),
    "final_head_channel_report": channels.REPORT_SCHEMA,
    "L23_reference": residual.REPORT_SCHEMA["properties"]["L23_reference"],
    "down_projection_tensor_bindings": obj({key: {"type": "object"} for key in SPECS}),
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in residual.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary",
                      "dispatch_and_write_audit", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_L23_MLP_DOWN_PROJECTION_INPUT_COORDINATES"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "projection_tensors_verified": {"const": 3},
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
        raise RuntimeError("MLP-down diagnostic forbids operators, prior checks and writes")

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


def bind_tensor(value, pin, suffix):
    require(suffix in SPECS, "unsupported down-projection tensor")
    shape, dtype = SPECS[suffix]
    require(isinstance(value, np.ndarray) and value.shape == shape
            and value.dtype.str == dtype and pin["name"] == PREFIX + suffix
            and pin["shape"] == list(shape) and pin["dtype"] == str(value.dtype)
            and np.all(np.isfinite(value))
            and hashlib.sha256(value.tobytes()).hexdigest() == pin["sha256"],
            "original AWQ down-projection operand binding changed")
    return value


def weight_column(tensors, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < OUTPUT_WIDTH,
            "down-projection output coordinate outside bounded width")
    shift, packed = SHIFTS[coordinate % 8], coordinate // 8
    return [
        Fraction(float(tensors["scales"][i // 128, coordinate])) * (
            ((int(tensors["qweight"][i, packed]) >> shift) & 15)
            - ((int(tensors["qzeros"][i // 128, packed]) >> shift) & 15))
        for i in range(WIDTH)
    ]


def account(actual, reference, weights, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < OUTPUT_WIDTH
            and all(len(source["stage16"]) == WIDTH
                    and len(source["stage17"]) == OUTPUT_WIDTH
                    for source in (actual, reference))
            and len(weights) == WIDTH
            and all(isinstance(value, Fraction) for source in (actual, reference)
                    for stage in ("stage16", "stage17") for value in source[stage])
            and all(isinstance(value, Fraction) for value in weights),
            "invalid exact down-projection account operands")
    deltas = [a - r for a, r in zip(actual["stage16"], reference["stage16"], strict=True)]
    terms = [delta * weight for delta, weight in zip(deltas, weights, strict=True)]
    top = sorted(range(WIDTH), key=lambda i: (-abs(terms[i]), i))[:LIMIT]
    exact, absolute = sum(terms, Fraction()), sum(map(abs, terms), Fraction())
    ranked = sum((terms[i] for i in top), Fraction())
    ranked_absolute = sum((abs(terms[i]) for i in top), Fraction())
    a17, r17 = actual["stage17"][coordinate], reference["stage17"][coordinate]
    retained = a17 - r17
    groups = []
    for group in range(GROUPS):
        values = terms[group * 128:(group + 1) * 128]
        groups.append({
            "input_group": group, "start_coordinate": group * 128,
            "end_coordinate_exclusive": (group + 1) * 128, "coordinate_count": 128,
            "signed_contribution": str(sum(values, Fraction())),
            "sum_absolute_contributions": str(sum(map(abs, values), Fraction())),
        })
    require(sum(Fraction(g["signed_contribution"]) for g in groups) == exact
            and sum(Fraction(g["sum_absolute_contributions"]) for g in groups) == absolute,
            "exact G128 group identity failed")
    return {
        "output_coordinate": coordinate, "input_stage": "stage16_fp16",
        "output_stage": "stage17_fp16", "reference": "original_input_L23_fp16",
        "coordinate_count": WIDTH,
        "largest_absolute_coordinates": [
            {"coordinate": i, "input_group": i // 128, "input_delta": str(deltas[i]),
             "weight": str(weights[i]), "signed_contribution": str(terms[i])}
            for i in top],
        "groups_in_input_order": groups,
        "actual_stage17": str(a17), "reference_stage17": str(r17),
        "retained_projection_delta": str(retained), "exact_input_delta": str(exact),
        "down_projection_boundary_remainder": str(retained - exact),
        "sum_absolute_contributions": str(absolute),
        "ranked_signed_sum": str(ranked), "ranked_absolute_sum": str(ranked_absolute),
        "unranked_signed_remainder": str(exact - ranked),
        "unranked_absolute_remainder": str(absolute - ranked_absolute),
        "exact_local_identity": True,
    }


def report(previous, actual, reference, tensors, binding):
    jsonschema.Draft202012Validator(channels.REPORT_SCHEMA).validate(previous)
    same(list(actual), list(parent.CONTROLS), "down-projection control order changed")
    same([row["control"] for row in previous["controls"]], list(parent.CONTROLS),
         "hotspot control order changed")
    controls, columns = [], {}
    for row in previous["controls"]:
        same(list(row["branches"]), list(channels.BRANCHES), "hotspot branch order changed")
        branches = []
        for name, branch in row["branches"].items():
            selected = branch["channels"]["largest_absolute_signed"]
            same(selected, branch["channels"]["full_absolute_signed_ranking"][:LIMIT],
                 "retained hotspot ranking changed")
            require(len(selected) == len({e["coordinate"] for e in selected}) == LIMIT,
                    "hotspot count or uniqueness changed")
            hotspots = []
            for rank, channel in enumerate(selected, 1):
                coordinate = channel["coordinate"]
                if coordinate not in columns:
                    columns[coordinate] = weight_column(tensors, coordinate)
                hotspots.append({
                    "hotspot_rank": rank, "final_head_channel": channel,
                    "down_projection": account(actual[row["control"]], reference,
                                               columns[coordinate], coordinate),
                })
            branches.append({"final_head_reference_branch": name,
                             "numeric_id": branch["numeric_id"], "hotspots": hotspots})
        controls.append({"control": row["control"], "branches": branches})
    return {
        "control_count": 9, "selected_row_count": 18, "projection_account_count": 144,
        "input_coordinate_term_count": 700416, "selected_input_coordinate_count": 1152,
        "input_group_count": 5472, "layer": 23, "position": 0, "controls": controls,
        "final_head_channel_report": previous,
        "L23_reference": {
            "fp16": binding["reference"]["fp16"], "manifest": binding["manifest"],
            "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
            "original_global_references_reanchored": False,
        },
        "down_projection_tensor_bindings": {
            suffix: binding["reference"]["canonical"][PREFIX + suffix] for suffix in SPECS},
    }


def measure():
    evidence, files, assets = channels.measure()
    actual, reference, archives, reference_archive, pins = residual.load_residuals(evidence[0])
    reference["stage16"] = residual.words(reference_archive["stage16"], (WIDTH,))
    for control, archive in archives.items():
        actual[control]["stage16"] = residual.words(archive["stage16"], (WIDTH,))
    binding = evidence[0]["preflight"]["L23_original_reference"]
    canonical = binding["reference"]["canonical"]
    same(sorted(key for key in canonical if key.startswith(PREFIX)),
         sorted(PREFIX + suffix for suffix in SPECS), "down-projection tensor set changed")
    tensors = {}
    with channels.contributions.preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        for suffix in SPECS:
            tensors[suffix] = bind_tensor(model.get_tensor(PREFIX + suffix),
                                          canonical[PREFIX + suffix], suffix)
    measured = report(evidence[5], actual, reference, tensors, binding)
    return (evidence, actual, reference, archives, reference_archive, tensors, measured), files + pins, assets


def focused_tests(evidence):
    spec = importlib.util.spec_from_file_location("l23_mlp_down_coordinate_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "MLP-down focused tests failed, errored or skipped")
    return {"executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "mlp_down_focused_test": parent.record(TEST),
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
            "status": "READ_ONLY_L23_MLP_DOWN_PROJECTION_INPUT_COORDINATES",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "L23_archives_verified": 10, "projection_tensors_verified": 3,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "tests": tests,
            "report": evidence[6],
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
