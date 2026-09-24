"""Read-only exact L23 residual-branch deltas at retained final-head hotspots.

--check compiles both new files in memory, runs only focused tests, and emits
one JSON document. Retained boundaries are read, never regenerated.
"""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
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

from ace3.model.candidates import diagnose_q24_s16_final_head_output_projection_channel_hotspots_v1 as channels
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


base, parent = channels.base, channels.parent
ROOT = channels.ROOT
NAME = "diagnose_q24_s16_final_head_l23_residual_branch_delta_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, EXPECTED_TESTS = 896, 8, 20
STAGES = ("input_hidden", "stage11", "stage17", "stage18")
Q24 = 1 << 24
require, same = base.require, base.same
FLAGS = {
    **channels.FLAGS, "residual_operator_replay": 0,
    "local_operator_replay": 0, "earlier_checks": 0,
    "binary64_L23_stages_reconstructed": False,
}
BOUNDARY = (
    "Read-only native-S16-RTZ Q24 CPU-software coordinate accounting on reviewed "
    "final-head attempt001. Nine controls, two retained final-head reference branches "
    "and eight existing channel hotspots per branch give 144 logical L23 accounts. "
    "Both selections use the independently propagated original-input L23 FP16 "
    "trajectory for residual-branch deltas; binary64 L23 internal stages are not "
    "retained and are not substituted or reconstructed. Exact actual-minus-reference "
    "input_hidden, stage11 attention, stage17 MLP down and retained stage18 hidden "
    "differences close with explicit boundary remainders. Actual Q24 input carry, "
    "state updates and output conversion are separated from the FP16 reference "
    "residual-add boundaries; the remainder is not attributed solely to rounding. "
    "This is not propagation through final RMSNorm or a causal decomposition of "
    "final-head errors, an upstream root-cause claim or runtime profiling. All "
    "L21/L22/L23 failures, exact thresholds, original global references and source/"
    "operand/state/KV/lineage gates remain unchanged. Old sparse-cut closure cannot "
    "certify, substitute for or propagate these nine parents. Q24 residual state "
    "is wider than FP16; G128 asymmetric packed INT4 native GEMM ordering without "
    "qzero plus-one, FP16 scales/operator boundaries/KV and native S16 RTZ are "
    "unchanged. No native, prefix, admission, decoder, reference or local operator "
    "replay, earlier checks, accepted-evidence writes, token selection/publication, "
    "RTL, GPU, hardware, simulation or ACE2 changes. No candidate admission, policy "
    "adoption, successor publication, strict-FP16-state W4A16, new-token or full-model "
    "claim. Normal independent Reviewer validation REQUIRED."
)

obj, RATIONAL = channels.object_schema, channels.RATIONAL


def ordered_schema(items):
    return {"type": "array", "minItems": len(items), "maxItems": len(items),
            "prefixItems": items, "items": False}


DELTA_SCHEMA = obj({
    "actual_exact": RATIONAL, "reference_exact": RATIONAL,
    "actual_minus_reference": RATIONAL,
})
ACTUAL_PARTS = ("input_q24_carry", "attention_state_update",
                "mlp_state_update", "output_fp16_boundary")
REFERENCE_PARTS = ("attention_residual_boundary", "mlp_residual_boundary")
ACCOUNT_SCHEMA = obj({
    "coordinate": channels.INDEX,
    "residual_reference": {"const": "original_input_L23_fp16"},
    "boundaries": obj({stage: DELTA_SCHEMA for stage in STAGES}),
    "branch_delta_sum": RATIONAL,
    "actual_boundary_remainder": RATIONAL,
    "reference_boundary_remainder": RATIONAL,
    "boundary_remainder_delta": RATIONAL,
    "actual_q24_boundary_parts": obj({key: RATIONAL for key in ACTUAL_PARTS}),
    "reference_fp16_boundary_parts": obj({key: RATIONAL for key in REFERENCE_PARTS}),
    "exact_residual_branch_identity": {"const": True},
    "exact_actual_boundary_identity": {"const": True},
    "exact_reference_boundary_identity": {"const": True},
})


def branch_schema(branch):
    return obj({
        "final_head_reference_branch": {"const": branch},
        "numeric_id": channels.ROW_SCHEMA["properties"]["numeric_id"],
        "hotspots": ordered_schema([
            obj({"hotspot_rank": {"const": rank},
                 "final_head_channel": channels.ENTRY_SCHEMA,
                 "residual": ACCOUNT_SCHEMA})
            for rank in range(1, LIMIT + 1)
        ]),
    })


REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "selected_row_count": {"const": 18},
    "hotspot_count": {"const": 144}, "boundary_delta_count": {"const": 576},
    "layer": {"const": 23}, "position": {"const": 0},
    "controls": ordered_schema([
        obj({"control": {"const": label},
             "branches": ordered_schema([branch_schema(b) for b in channels.BRANCHES])})
        for label in parent.CONTROLS
    ]),
    "final_head_channel_report": channels.REPORT_SCHEMA,
    "L23_reference": obj({
        "fp16": {"type": "object"}, "manifest": {"type": "object"},
        "binary64_internal_stages": {"const": "NOT_RETAINED_NO_RECONSTRUCTION"},
        "original_global_references_reanchored": {"const": False},
    }),
})
OUTPUT_SCHEMA = obj({
    "diagnostic_id": {"const": NAME}, "version": {"const": 1},
    "status": {"const": "READ_ONLY_L23_RESIDUAL_BRANCH_DELTA"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "normal_host_review": {"const": "REQUIRED"},
    "original_execution_sources": {"const": parent.RETAINED_SOURCES},
    "diagnostic_sources": {"type": "object"},
    "authenticated_files": {"type": "array", "minItems": 1},
    "assets": {"type": "object"},
    "L23_archives_verified": {"const": 10},
    "dispatch_and_write_audit": {"const": {**FLAGS, "forbidden_calls": 0}},
    "tests": obj({
        "compiled": ordered_schema([{"type": "object"}, {"type": "object"}]),
        "executed": {"const": EXPECTED_TESTS},
        **{key: {"const": 0} for key in ("failures", "errors", "skipped")},
    }),
    "report": REPORT_SCHEMA,
})


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("residual diagnostic forbids operators and earlier checks")

    with channels.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("projection", "local_reference", "attention_value",
                                  "check", "focused_tests", "rne", "toward_zero"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        yield


def words(array, shape=(WIDTH,)):
    require(isinstance(array, np.ndarray) and array.dtype.str == "<u2"
            and array.shape == shape and np.all((array & 0x7c00) != 0x7c00),
            "invalid retained FP16 boundary")
    return [Fraction(float(value)) for value in array.view("<f2").reshape(-1)]


def operands(archive, *, actual):
    values = {stage: words(archive[stage]) for stage in (*STAGES, "stage12")}
    if actual:
        for kind, stage in (("k", "stage06"), ("v", "stage07")):
            words(archive["input_cache_" + kind], (0, 128))
            words(archive["output_cache_" + kind], (1, 128))
            words(archive[stage], (128,))
            require(np.array_equal(archive["output_cache_" + kind][0], archive[stage]),
                    "retained P0 output KV lineage changed")
        words(archive["stage03"], (128,))
        require(np.array_equal(archive["stage03"], archive["stage07"]),
                "retained V projection/cache lineage changed")
        for name in ("input", "scratch", "output"):
            integer, zero = archive[name + "_i"], archive[name + "_z"]
            require(integer.shape == zero.shape == (WIDTH,)
                    and integer.dtype.str == "<i8" and zero.dtype.str == "|u1"
                    and np.all(zero <= 1) and np.all(integer[zero != 0] == 0),
                    "invalid retained Q24 state or signed-zero lineage")
            values[name + "_q24"] = [Fraction(int(i), Q24) for i in integer]
    return values


def read_archive(pin):
    with np.load(io.BytesIO(base.read_bound(pin)), allow_pickle=False) as arrays:
        require(len(arrays.files) == len(set(arrays.files)), "duplicate archive fields")
        return {key: arrays[key] for key in arrays.files}


def load_residuals(result):
    binding = result["preflight"]["L23_original_reference"]
    require(binding["status"] == "BOUND_ORIGINAL_INPUT_L23" and not binding["missing"]
            and binding["reference"]["prior_kv"] == "own empty P0",
            "original L23 reference scope changed")
    same(binding["reference"]["fp16"],
         result["preflight"]["final_reference"]["reference"]["input_fp16"],
         "L23/final-head original reference splice")
    same([row["control"] for row in result["controls"]], list(parent.CONTROLS),
         "retained residual control order changed")
    pins = [binding["reference"]["fp16"]]
    reference_archive = read_archive(pins[0])
    reference = operands(reference_archive, actual=False)
    actual, archives = {}, {}
    for row in result["controls"]:
        pins.append(row["parent"]["terminal_archive"])
        archives[row["control"]] = read_archive(pins[-1])
        actual[row["control"]] = operands(archives[row["control"]], actual=True)
    return actual, reference, archives, reference_archive, pins


def account(actual, reference, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < WIDTH,
            "residual coordinate outside bounded width")
    a, r = ({key: values[coordinate] for key, values in source.items()}
            for source in (actual, reference))
    deltas = {stage: a[stage] - r[stage] for stage in STAGES}
    branch_sum = sum(deltas[stage] for stage in STAGES[:3])
    ar = a["stage18"] - sum(a[stage] for stage in STAGES[:3])
    rr = r["stage18"] - sum(r[stage] for stage in STAGES[:3])
    actual_parts = {
        "input_q24_carry": a["input_q24"] - a["input_hidden"],
        "attention_state_update": a["scratch_q24"] - a["input_q24"] - a["stage11"],
        "mlp_state_update": a["output_q24"] - a["scratch_q24"] - a["stage17"],
        "output_fp16_boundary": a["stage18"] - a["output_q24"],
    }
    reference_parts = {
        "attention_residual_boundary": r["stage12"] - r["input_hidden"] - r["stage11"],
        "mlp_residual_boundary": r["stage18"] - r["stage12"] - r["stage17"],
    }
    require(deltas["stage18"] == branch_sum + ar - rr
            and sum(actual_parts.values()) == ar and sum(reference_parts.values()) == rr,
            "exact residual-branch/boundary identity failed")
    return {
        "coordinate": coordinate, "residual_reference": "original_input_L23_fp16",
        "boundaries": {stage: {
            "actual_exact": str(a[stage]), "reference_exact": str(r[stage]),
            "actual_minus_reference": str(deltas[stage]),
        } for stage in STAGES},
        "branch_delta_sum": str(branch_sum),
        "actual_boundary_remainder": str(ar), "reference_boundary_remainder": str(rr),
        "boundary_remainder_delta": str(ar - rr),
        "actual_q24_boundary_parts": {key: str(value) for key, value in actual_parts.items()},
        "reference_fp16_boundary_parts": {key: str(value) for key, value in reference_parts.items()},
        "exact_residual_branch_identity": True, "exact_actual_boundary_identity": True,
        "exact_reference_boundary_identity": True,
    }


def report(channel_report, actual, reference, binding):
    same(list(actual), list(parent.CONTROLS), "residual control order changed")
    same([row["control"] for row in channel_report["controls"]], list(parent.CONTROLS),
         "hotspot control order changed")
    rows = []
    for row in channel_report["controls"]:
        same(list(row["branches"]), list(channels.BRANCHES), "hotspot branch order changed")
        branches = []
        for name, branch in row["branches"].items():
            selected = branch["channels"]["largest_absolute_signed"]
            same(len(selected), LIMIT, "hotspot count changed")
            same(selected, branch["channels"]["full_absolute_signed_ranking"][:LIMIT],
                 "retained hotspot ranking changed")
            require(len({entry["coordinate"] for entry in selected}) == LIMIT,
                    "duplicate hotspot coordinate")
            branches.append({
                "final_head_reference_branch": name, "numeric_id": branch["numeric_id"],
                "hotspots": [
                    {"hotspot_rank": rank, "final_head_channel": channel,
                     "residual": account(actual[row["control"]], reference, channel["coordinate"])}
                    for rank, channel in enumerate(selected, 1)
                ],
            })
        rows.append({"control": row["control"], "branches": branches})
    return {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "boundary_delta_count": 576, "layer": 23, "position": 0, "controls": rows,
        "final_head_channel_report": channel_report,
        "L23_reference": {
            "fp16": binding["reference"]["fp16"], "manifest": binding["manifest"],
            "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
            "original_global_references_reanchored": False,
        },
    }


def measure():
    evidence, files, assets = channels.measure()
    actual, reference, archives, reference_archive, pins = load_residuals(evidence[0])
    measured = report(evidence[5], actual, reference,
                      evidence[0]["preflight"]["L23_original_reference"])
    return (evidence, actual, reference, archives, reference_archive, measured), files + pins, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("l23_residual_branch_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "residual diagnostic focused tests failed, errored or skipped")
    return {
        "compiled": compiled, "executed": outcome.testsRun,
        "failures": len(outcome.failures), "errors": len(outcome.errors),
        "skipped": len(outcome.skipped),
    }


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "residual_focused_test": parent.record(TEST),
               channels.MODULE: parent.record(channels.SOURCE)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *channels.PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_L23_RESIDUAL_BRANCH_DELTA",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "L23_archives_verified": 10, "dispatch_and_write_audit": {**FLAGS, **audit},
            "tests": tests, "report": evidence[5],
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
