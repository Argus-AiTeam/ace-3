"""Read-only input-column accounting for the worst retained tied-head residual rows.

--check compiles the two new Python files in memory, runs only focused tests,
validates the JSON schema, and emits one JSON document without evidence writes.
"""

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
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_rmsnorm_delta_hotspots_v1 as hotspots


contributions = hotspots.contributions
base, parent = hotspots.base, hotspots.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_output_projection_channel_hotspots_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
    f" -B -m {MODULE} --check"
)
PINS = hotspots.PINS
BRANCHES = hotspots.BRANCHES
FLAGS = dict(hotspots.FLAGS)
WIDTH, VOCABULARY, RANK_LIMIT = 896, 151936, 8
EXPECTED_TESTS = 22
require, same = base.require, base.same
BOUNDARY = (
    "Read-only CPU-software input-column accounting for the final tied lm_head output "
    "projection, not an attention o_proj experiment. Select the largest absolute "
    "retained logit residual separately for each of nine controls and each independent "
    "original-input FP16/binary64 reference; numeric IDs are diagnostic only. Exact "
    "coordinate products (actual RMSNorm[i]-reference RMSNorm[i])*W[row,i] are signed "
    "accounting, not nonnegative causal shares. Retained residual minus their sum is "
    "an explicit boundary remainder, not attributed solely to rounding. No RMSNorm, "
    "row-dot, head, native, decoder, prefix, admission or reference replay, full-vocabulary "
    "output recomputation, evidence writes, token decode/publication/selection, RTL, "
    "GPU, hardware or simulation dispatch, or ACE2 changes. All nine L21/L22/L23 "
    "failures, exact thresholds, original global references and source/operand/state/"
    "KV/lineage gates remain unchanged. Old sparse-cut closure cannot certify, "
    "substitute for or propagate these parents. Q24 residual state is wider than FP16; "
    "native S16 RTZ, G128 asymmetric packed INT4 GEMM ordering without qzero plus-one, "
    "FP16 scales/operator boundaries/KV and the FP16 tied head remain unchanged. "
    "No upstream root-cause, runtime-bottleneck, candidate admission, policy adoption, "
    "successor publication, strict-FP16-state W4A16, new-token or full-model claim. "
    "Normal independent Reviewer validation REQUIRED."
)


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


RATIONAL = {"type": "string", "pattern": r"^-?[0-9]+(?:/[1-9][0-9]*)?$"}
INDEX = {"type": "integer", "minimum": 0, "maximum": WIDTH - 1}
ENTRY_SCHEMA = object_schema({
    "coordinate": INDEX, "signed_contribution": RATIONAL,
    "input_delta": RATIONAL, "column_weight": RATIONAL,
})
TOP_SCHEMA = {"type": "array", "maxItems": RANK_LIMIT, "items": ENTRY_SCHEMA}
CHANNEL_SCHEMA = object_schema({
    **{key: TOP_SCHEMA for key in hotspots.RANK_FIELDS},
    "coordinate_count": {"const": WIDTH},
    "exact_sum": RATIONAL, "ranked_signed_sum": RATIONAL,
    "unranked_exact_remainder": RATIONAL,
    "exact_sum_of_absolute_contributions": RATIONAL,
    "zero_coordinate_count": {"type": "integer", "minimum": 0, "maximum": WIDTH},
    "full_absolute_signed_ranking": {
        "type": "array", "minItems": WIDTH, "maxItems": WIDTH, "items": ENTRY_SCHEMA},
    "coordinate62": object_schema({
        **ENTRY_SCHEMA["properties"], "absolute_rank": {
            "type": "integer", "minimum": 1, "maximum": WIDTH},
        "in_largest_absolute": {"type": "boolean"},
    }),
})
ROW_SCHEMA = object_schema({
    "numeric_id": {"type": "integer", "minimum": 0, "maximum": VOCABULARY - 1},
    "absolute_residual": RATIONAL, "retained_signed_residual": RATIONAL,
    "retained_actual_hex": {"type": "string"},
    "retained_reference_hex": {"type": "string"},
    "boundary_remainder": RATIONAL, "exact_residual_identity": {"const": True},
    "channels": CHANNEL_SCHEMA,
})
REPORT_SCHEMA = object_schema({
    "control_count": {"const": 9}, "selected_row_count": {"const": 18},
    "coordinate_product_count": {"const": 18 * WIDTH},
    "unique_numeric_rows": {
        "type": "array", "minItems": 1, "maxItems": 18, "uniqueItems": True,
        "items": ROW_SCHEMA["properties"]["numeric_id"]},
    "controls": {"type": "array", "minItems": 9, "maxItems": 9, "items": object_schema({
        "control": {"enum": list(parent.CONTROLS)},
        "branches": object_schema({branch: ROW_SCHEMA for branch in BRANCHES}),
        "L21_L22_L23_status": {"const": ["FAIL"] * 3},
        "retained_failures": object_schema({
            stage: {"type": "array", "minItems": 1} for stage in ("L21", "L22", "L23")}),
        "retained_L23_and_ancestral_lineage": {"type": "object"},
    })},
    "retained_L23_stage_reports": {"const": {"PASS": 162, "FAIL": 9}},
    "thresholds": {"type": "object"}, "original_global_reference": {"type": "object"},
    "retained_flags": {"type": "object"},
    "lineage_separation": {"const": contributions.LINEAGE},
    "arithmetic": {"type": "object"},
})
OUTPUT_SCHEMA = object_schema({
    "diagnostic_id": {"const": NAME}, "version": {"const": 1},
    "status": {"const": "READ_ONLY_OUTPUT_PROJECTION_CHANNEL_HOTSPOTS"},
    "command": {"const": COMMAND}, "pins": {"type": "object"},
    "original_execution_sources": {"const": parent.RETAINED_SOURCES},
    "diagnostic_sources": {"type": "object"},
    "authenticated_files": {"type": "array", "minItems": 1},
    "assets": {"type": "object"},
    "attempt001_files_verified": {"const": 20},
    "arrays_verified": {"const": 18}, "reference_arrays_verified": {"const": 4},
    "dispatch_and_write_audit": {"const": {**FLAGS, "forbidden_calls": 0}},
    "flags": {"const": FLAGS},
    "tests": object_schema({
        "compiled": {"type": "array", "minItems": 2, "maxItems": 2},
        "executed": {"const": EXPECTED_TESTS},
        "failures": {"const": 0}, "errors": {"const": 0}, "skipped": {"const": 0},
    }),
    "normal_host_review": {"const": "REQUIRED"},
    "claim_boundary": {"const": BOUNDARY}, "report": REPORT_SCHEMA,
})


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("channel diagnostic forbids operator or prior diagnostic replay")

    with hotspots.read_only(audit), ExitStack() as stack:
        for module, names in (
            (hotspots, ("measure", "report", "check", "focused_tests")),
            (hotspots.cutoff, ("report",)),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def worst_row(actual, reference):
    require(isinstance(actual, np.ndarray) and isinstance(reference, np.ndarray)
            and actual.shape == reference.shape == (VOCABULARY,)
            and actual.dtype.str == "<f2" and reference.dtype.str in ("<f2", "<f8")
            and np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)),
            "invalid retained output-projection vectors")
    errors = np.abs(actual.astype("<f8") - reference.astype("<f8"))
    require(np.all(np.isfinite(errors)), "nonfinite residual magnitude")
    # Monotone binary64 rounding retains the exact maximum among rounded maxima.
    candidates = map(int, np.flatnonzero(errors == errors.max()))
    index = max(candidates, key=lambda i: (
        abs(Fraction(float(actual[i])) - Fraction(float(reference[i]))), -i))
    delta = Fraction(float(actual[index])) - Fraction(float(reference[index]))
    return index, delta


def selection(result, arrays, references):
    base.check_history(result)
    same(list(arrays), list(parent.CONTROLS), "channel diagnostic control census changed")
    selected = {}
    for retained in result["controls"]:
        label = retained["control"]
        branches = {}
        for name in BRANCHES:
            reference = references["logits_" + name]
            if name == "fp16":
                reference = reference.view("<f2")
            index, residual = worst_row(arrays[label]["logits"].view("<f2"), reference)
            if name == "binary64":
                comparison = retained["comparisons"]["logits"]
                same(index, comparison["binary64_worst_index"], "retained residual row changed")
                same(str(abs(residual)), comparison["binary64_max_absolute_error"],
                     "retained residual magnitude changed")
            branches[name] = (index, residual)
        selected[label] = branches
    return selected


def channel_summary(actual, reference, weights):
    require(actual.shape == reference.shape == weights.shape == (WIDTH,)
            and actual.dtype.str == weights.dtype.str == "<f2"
            and reference.dtype.str in ("<f2", "<f8"), "invalid column operand shape/dtype")
    a, r, w = map(contributions.exact_vector, (actual, reference, weights))
    deltas = [x - y for x, y in zip(a, r, strict=True)]
    values = [delta * weight for delta, weight in zip(deltas, w, strict=True)]
    require(values == [x * weight - y * weight for x, y, weight in zip(a, r, w, strict=True)],
            "expanded input-column identity failed")
    summary = contributions.ranked(values, RANK_LIMIT)
    order = sorted(range(WIDTH), key=lambda i: (-abs(values[i]), i))

    def entry(i):
        return {"coordinate": i, "signed_contribution": str(values[i]),
                "input_delta": str(deltas[i]), "column_weight": str(w[i])}

    return {
        **summary,
        **{key: [entry(item["coordinate"]) for item in summary[key]]
           for key in hotspots.RANK_FIELDS},
        "full_absolute_signed_ranking": [entry(i) for i in order],
        "coordinate62": {**entry(62), "absolute_rank": order.index(62) + 1,
                         "in_largest_absolute": 62 in order[:RANK_LIMIT]},
    }


def report(result, arrays, references, selected, weights):
    same(list(selected), list(parent.CONTROLS), "selected control order changed")
    ids = sorted({index for branches in selected.values() for index, _ in branches.values()})
    same(list(weights), ids, "missing, extra or reordered selected tied-head rows")
    rows = []
    for retained in result["controls"]:
        label = retained["control"]
        same(list(selected[label]), list(BRANCHES), "independent branch order changed")
        branches = {}
        for name in BRANCHES:
            index, residual = selected[label][name]
            reference = references["rmsnorm_" + name]
            logits = references["logits_" + name]
            if name == "fp16":
                reference, logits = reference.view("<f2"), logits.view("<f2")
            channels = channel_summary(arrays[label]["rmsnorm"].view("<f2"), reference, weights[index])
            actual_logit, reference_logit = float(arrays[label]["logits"].view("<f2")[index]), float(logits[index])
            require(residual == Fraction(actual_logit) - Fraction(reference_logit),
                    "selected retained residual changed")
            remainder = residual - Fraction(channels["exact_sum"])
            branches[name] = {
                "numeric_id": index, "absolute_residual": str(abs(residual)),
                "retained_signed_residual": str(residual),
                "retained_actual_hex": actual_logit.hex(),
                "retained_reference_hex": reference_logit.hex(),
                "boundary_remainder": str(remainder), "exact_residual_identity": True,
                "channels": channels,
            }
        l23 = retained["parent"]["retained_L23"]
        rows.append({
            "control": label, "branches": branches, "L21_L22_L23_status": ["FAIL"] * 3,
            "retained_failures": {
                "L21": l23["retained_L21"]["failures"], "L22": [l23["L22_failure"]],
                "L23": retained["parent"]["L23_failures"]},
            "retained_L23_and_ancestral_lineage": l23,
        })
    return {
        "control_count": len(rows), "selected_row_count": 2 * len(rows),
        "coordinate_product_count": 2 * len(rows) * WIDTH, "unique_numeric_rows": ids,
        "controls": rows,
        "retained_L23_stage_reports": result["preflight"]["retained_L23_stage_reports"],
        "thresholds": result["preflight"]["thresholds"],
        "original_global_reference": result["preflight"]["final_reference"],
        "retained_flags": result["flags"], "lineage_separation": contributions.LINEAGE,
        "arithmetic": {
            "projection": "final tied FP16 lm_head; columns are RMSNorm input channels",
            "row_selection": "largest exact absolute retained logit residual; numeric ID ascending on ties",
            "channel_formula": "(actual RMSNorm[i] - independent reference RMSNorm[i]) * W[row,i]",
            "ranking": "absolute descending; positive descending; negative ascending; channel ascending on ties",
            "rank_limit": RANK_LIMIT, "coordinate_count": WIDTH,
            "scope": "18 row/reference accounts, not pair margins or full-vocabulary head recomputation",
            "remainder": "retained signed residual minus exact coordinate sum; not solely rounding",
        },
    }


def measure():
    result, arrays, references, files, assets = hotspots.authenticate()
    selected = selection(result, arrays, references)
    ids = {index for branches in selected.values() for index, _ in branches.values()}
    require(1 <= len(ids) <= 18, "unbounded tied-head row selection")
    weights = contributions.load_rows(assets, ids)
    measured = report(result, arrays, references, selected, weights)
    return (result, arrays, references, selected, weights, measured), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("output_projection_channel_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "channel diagnostic tests failed, errored or skipped")
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
               "channel_focused_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *PINS.values()):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "channel diagnostic attempted forbidden work")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_OUTPUT_PROJECTION_CHANNEL_HOTSPOTS",
            "command": COMMAND, "pins": {**base.PINS, **PINS},
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attempt001_files_verified": 20, "arrays_verified": 18, "reference_arrays_verified": 4,
            "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
            "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
            "report": evidence[5],
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
