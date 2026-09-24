"""Read-only L20-admissible Q24 interval and L21 input-cell endpoint preparation.

The adjacent contract publishes the repo-bound --check command. It compiles and
runs only this diagnostic's tests, then authenticates frozen evidence. Endpoint
coverage of input RNE cells is not coverage of native L21 residual computations.
"""

import argparse
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from ace3.model.candidates import (
    diagnose_q24_s16_l20_to_l21_coordinate62_input_sensitivity_v1 as upstream,
)


ROOT, PYTHON = upstream.ROOT, upstream.PYTHON
retained, require, gates = upstream.retained, upstream.require, upstream.gates
margin = upstream.margin
rational, scalar_math, Q = margin.rational, margin.scalar_math, margin.Q
no_execution = upstream.no_execution
NAME = "q24_s16_l20_to_l21_coordinate62_admissible_threshold_v1"
ID = "ace3-q24-s16-l20-to-l21-coordinate62-admissible-threshold-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
EVIDENCE = ROOT / ".argus_subagents/e2612f3b804d-input-sensitivity-execute-r2_logs/stdout.log"
EVIDENCE_SHA256 = "d4afa7f861e461cf7cbb26d50c815eebe8df063fa51e6084c88e927da07b41f4"
EXPECTED_TESTS = 20
FLAGS = {**upstream.FLAGS, "native_L9_plus_invocations": 0}
METHOD = {
    "coordinate": 62, "position": 0, "history": [9707],
    "gate": "unchanged full-vector L20 S18 original-input binary64-v1",
    "interval": "all signed int64 Q24 I[62] projecting into the passing FP16 interval",
    "cells": "exact rational midpoint bounds; ties to even; inclusive integer endpoints",
    "candidates": "both Q24 endpoints of every passing input FP16 RNE cell",
    "control": "unchanged frozen parent, delta_Q24_units=0",
    "state": "copy I/Z/H; change only I[62]; preserve all Z; recompute only H[62] by RNE16",
    "coverage": "input cells only; not an exhaustive L21 search or a minimality proof",
}


def equal(actual, expected, message):
    require(type(actual) is type(expected)
            and json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
            message)


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 1, "focused_tests": EXPECTED_TESTS,
        "interface": ["--check"], "native_layers": [], "method": METHOD,
        "input": str(EVIDENCE.relative_to(ROOT)), "input_sha256": EVIDENCE_SHA256,
        "policy_id": gates.POLICY_ID, "local_gate": upstream.producer.LOCAL_GATE,
        "excess_budget": "1/8", "reference_policy": upstream.suffix.REFERENCE_POLICY,
        **FLAGS,
    }
    for key, value in expected.items():
        equal(document[key], value, f"versioned contract mismatch: {key}")


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "candidate/contract origin mismatch")
    records = upstream.origins()
    for name in (MODULE, TEST_MODULE, upstream.LEGACY_TEST):
        require(name in records and records[name]["path"]
                == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"required repository origin missing: {name}")
    return records


def read_observation(inputs):
    require(EVIDENCE.resolve() == EVIDENCE and EVIDENCE.is_relative_to(ROOT),
            "sensitivity evidence outside isolated repository")
    record = retained.record(EVIDENCE)
    require(record["sha256"] == EVIDENCE_SHA256, "pinned sensitivity evidence mismatch")
    last = None
    with inputs.bind(record).open() as stream:
        for line in stream:
            last = line
    require(last is not None, "empty sensitivity evidence")
    return json.loads(last)


def check_observation(document, parent_binding, failure):
    expected = {
        "diagnostic_id": upstream.ID, "status": "INCONCLUSIVE",
        "reason": "L20_GLOBAL_GATE_BOUNDARY", "search_executed": True,
        "search_exhausted": False, "minimality_proven": False,
        "search": upstream.SEARCH, "executor": upstream.EXECUTION,
        "native_layers": [21] * 43, "parent_binding": parent_binding,
        "frozen_first_failure": failure,
        "reference_policy": upstream.suffix.REFERENCE_POLICY,
        "producer_sha256": upstream.PRODUCER_SHA256,
        "suffix_sha256": upstream.SUFFIX_PINS,
        "margin_sha256": upstream.suffix.PINS, "L15_cut_suffix_sha256": margin.PINS,
        **upstream.FLAGS, "native_invocations": 43, "native_layer_invocations": 43,
    }
    for key, value in expected.items():
        equal(document[key], value, f"frozen sensitivity mismatch: {key}")
    upstream.check_validation(document["validation"], upstream.EXPECTED_TESTS)
    samples = document["samples"]
    equal([row["delta_Q24_units"] for row in samples],
          list(upstream.search_points()[:44]), "frozen sensitivity probe order mismatch")
    for index, row in enumerate(samples):
        boundary = "L20_GLOBAL_GATE_BOUNDARY" if index == 43 else None
        equal(row["boundary"], boundary, "frozen sensitivity stop boundary mismatch")
        equal(row["L20_global_gate"]["status"], "FAIL" if index == 43 else "PASS",
              "frozen sensitivity parent gate mismatch")
        reports = row["reports"]
        require(len(reports) == (0 if index == 43 else 19),
                "frozen sensitivity report count mismatch")
        for stage, report in enumerate(reports):
            require(report["node"] == [21, 0, stage] and report["stage"] == stage
                    and report["policy_id"] == gates.POLICY_ID
                    and report["residual_state_lineage"] == report["kv_lineage"] == "PASS",
                    "frozen sensitivity policy/state/KV mismatch")
            mandatory = report["local_operator_fp16"] if stage < 18 else report["binary64_v1"]
            require(mandatory["passed"] is (stage < 18)
                    and report["status"] == ("PASS" if stage < 18 else "FAIL"),
                    "frozen sensitivity mandatory gate mismatch")


def change_coordinate(parent, integer):
    require(type(integer) is int and -(1 << 63) <= integer < (1 << 63),
            "signed int64 Q24 integer required")
    retained.verify_parent(parent, parent)
    changed = {key: value.copy() for key, value in parent.items()}
    changed["i"][62] = integer
    changed["h"][62] = rational.project(integer, int(changed["z"][62]))
    retained.verify_parent(changed, changed)
    require(np.array_equal(changed["z"], parent["z"])
            and set(np.flatnonzero(changed["i"] != parent["i"]).tolist()) <= {62}
            and set(np.flatnonzero(changed["h"] != parent["h"]).tolist()) <= {62},
            "coordinate cut changed another state component")
    return changed


def interval(integer, reference):
    require(type(integer) is int and -(1 << 63) <= integer < (1 << 63)
            and type(reference) is Fraction
            and 1 < reference < 65503
            and Fraction.from_float(float(reference)) == reference,
            "positive interior exact binary64 reference and int64 Q24 required")
    baseline = scalar_math.scalar(rational.project(integer, 0), reference)
    radius = Fraction(baseline["q"]) + Fraction(1, 8)
    lower, upper = reference - radius, reference + radius
    require(rational.fp16_value(1) < lower < upper < rational.fp16_value(0x7BFE),
            "gate outside supported positive interior FP16 cells")
    first = last = int(baseline["nearest_fp16_bits"], 16)
    while rational.fp16_value(first - 1) >= lower:
        first -= 1
    while rational.fp16_value(last + 1) <= upper:
        last += 1
    require(scalar_math.scalar(first, reference)["accepted"] is True
            and scalar_math.scalar(last, reference)["accepted"] is True
            and scalar_math.scalar(first - 1, reference)["accepted"] is False
            and scalar_math.scalar(last + 1, reference)["accepted"] is False,
            "passing FP16 interval is not exhaustive")
    cells, endpoints = [], []
    previous_maximum = None
    for word in range(first, last + 1):
        lo, hi, minimum, maximum = scalar_math.passing_cell(word)
        require(minimum <= maximum
                and (previous_maximum is None or minimum == previous_maximum + 1),
                "RNE cells do not cover one exact Q24 interval")
        previous_maximum = maximum
        cells.append({
            "input_FP16_word": f"{word:04x}", "input_FP16_exact": str(rational.fp16_value(word)),
            "lower_midpoint": str(lo), "upper_midpoint": str(hi),
            "lower_tie_inclusive": not bool(word & 1),
            "upper_tie_inclusive": not bool(word & 1),
            "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
        })
        for side, value in (("minimum", minimum), ("maximum", maximum)):
            require(rational.project(value, 0) == word, "RNE endpoint projection mismatch")
            endpoints.append({
                "cell_endpoint": side, "input_FP16_word": f"{word:04x}",
                "Q24_integer": value, "delta_Q24_units": value - integer,
                "delta_exact": str(Fraction(value - integer, Q)),
            })
    minimum, maximum = cells[0]["minimum_Q24_integer"], cells[-1]["maximum_Q24_integer"]
    return {
        "coordinate": 62, "baseline_Q24_integer": integer,
        "baseline_input_FP16_word": f"{rational.project(integer, 0):04x}",
        "reference_binary64_hex": float(reference).hex(), "reference_exact": str(reference),
        "representation_floor": baseline["q"], "excess_budget": "1/8",
        "gate_lower_exact": str(lower), "gate_upper_exact": str(upper),
        "first_passing_FP16_word": f"{first:04x}", "last_passing_FP16_word": f"{last:04x}",
        "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
        "minimum_delta_Q24_units": minimum - integer,
        "maximum_delta_Q24_units": maximum - integer,
        "minimum_delta_exact": str(Fraction(minimum - integer, Q)),
        "maximum_delta_exact": str(Fraction(maximum - integer, Q)),
        "integer_bounds_inclusive": True, "admissible_Q24_integer_count": maximum - minimum + 1,
        "cells": cells, "candidate_endpoints": endpoints,
        "candidate_endpoint_count": len(endpoints), "baseline_control_delta_Q24_units": 0,
        "input_cell_coverage_complete": True, "L21_search_coverage_complete": False,
        "minimality_proven": False,
    }


def gate_summary(report):
    gate = report["binary64_v1"]
    return {
        "status": report["status"], "passed": gate["passed"],
        "coordinates": gate["coordinates"], "failure_count": gate["failure_count"],
        "failed_indices": [row["index"] for row in gate["failures"]],
    }


def analyze(parent, trajectory, reference):
    baseline_gate = upstream.parent_gate(parent, trajectory, reference)
    require(baseline_gate["status"] == "PASS", "frozen full-vector L20 global gate failed")
    result = interval(int(parent["i"][62]), Fraction.from_float(float(reference[62])))
    require(result["minimum_Q24_integer"] <= int(parent["i"][62])
            <= result["maximum_Q24_integer"], "frozen parent outside admissible interval")
    results = {}
    probes = [("baseline", int(parent["i"][62]), True)]
    probes.extend((f"endpoint_{index}", row["Q24_integer"], True)
                  for index, row in enumerate(result["candidate_endpoints"]))
    probes.extend((
        ("one_below_interval", result["minimum_Q24_integer"] - 1, False),
        ("one_above_interval", result["maximum_Q24_integer"] + 1, False),
    ))
    for label, integer, accepted in probes:
        changed = change_coordinate(parent, integer)
        report = upstream.parent_gate(changed, trajectory, reference)
        summary = gate_summary(report)
        require(summary["passed"] is accepted
                and summary["status"] == ("PASS" if accepted else "FAIL")
                and summary["coordinates"] == 896
                and summary["failed_indices"] == ([] if accepted else [62]),
                "full-vector admissible boundary mismatch")
        results[label] = {"Q24_integer": integer, "delta_Q24_units": integer - int(parent["i"][62]),
                          "input_FP16_word": f"{int(changed['h'][62]):04x}", **summary}
    result["full_vector_gate_checks"] = results
    return result


def authenticate():
    inputs, parent_binding, failure = upstream.authenticate()
    observation = read_observation(inputs)
    check_observation(observation, parent_binding, failure)
    for record in observation["authenticated_inputs"]:
        inputs.bind(record)
    upstream.bind_validation(inputs, observation["validation"], upstream.SOURCE, upstream.CONTRACT)
    frozen = inputs.read(retained.record(margin.INPUT / "result.json"))
    entry20 = next(entry for entry in frozen["layers"] if entry["layer"] == 20)
    entry21 = frozen["layers"][-1]
    require(entry20["status"] == "PASS" and entry21["layer"] == 21
            and entry20["output_state_evidence"] == parent_binding == entry21["input_state_evidence"],
            "frozen L20 output/L21 input linkage mismatch")
    original = margin.upstream.upstream
    freeze = inputs.read(retained.record(original.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    for entry in (entry20, entry21):
        margin.check_reference(entry, extension["layers"][str(entry["layer"])])
    parent = inputs.archive(parent_binding)
    arrays20 = inputs.archive(entry20["actual_stages"])
    retained.verify_parent(parent, retained.state_from(arrays20, "output", "stage18"))
    item = extension["layers"]["20"]
    trajectory = inputs.archive(item["fp16"])
    reference = original.global_reference(inputs, item)
    analysis = analyze(parent, trajectory, reference)
    equal(observation["samples"][0]["reports"], inputs.read(entry21["reports"]),
          "frozen sensitivity baseline reports changed")
    # Only the L20 gate depends solely on H; never deduplicate future native L21 probes by H.
    parent_gates = {}
    for row in observation["samples"]:
        changed = change_coordinate(parent, int(parent["i"][62]) + row["delta_Q24_units"])
        word = int(changed["h"][62])
        if word not in parent_gates:
            parent_gates[word] = upstream.parent_gate(changed, trajectory, reference)
        equal(row["L20_global_gate"], parent_gates[word], "frozen L20 sample gate changed")
    return inputs, parent_binding, item, failure, analysis


def validate():
    context = upstream.producer.context()
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(upstream.TEST_MODULE)
    importlib.import_module(upstream.LEGACY_TEST)
    records = origins()
    contract_record = retained.record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for path in (SOURCE, ROOT.joinpath(*TEST_MODULE.split(".")).with_suffix(".py")):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    require(count == EXPECTED_TESTS, "focused test collection mismatch")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and not result.skipped,
            "focused tests failed")
    require(origins() == records and retained.record(CONTRACT) == contract_record,
            "source/contract changed during validation")
    return {
        **context, "origins": records, "compiled": compiled, "contract": contract_record,
        "contract_parsed": True, "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "legacy_19_tests_executed": False,
        "upstream_tests_executed": False,
    }


def check():
    with no_execution():
        validation = validate()
        inputs, parent_binding, reference, failure, analysis = authenticate()
        for record in (*validation["origins"].values(), validation["contract"]):
            inputs.bind(record)
        require(origins() == validation["origins"], "source origins changed during check")
        for record in inputs.records.values():
            require(retained.record(record["path"]) == record,
                    f"input/source changed during check: {record['path']}")
    return {
        "diagnostic_id": ID, "status": "ADMISSIBLE_INTERVAL_PREPARED", **FLAGS,
        "native_layers": [], "native_search_executed": False,
        "validation": validation, "method": METHOD, "analysis": analysis,
        "authenticated_inputs": list(inputs.records.values()),
        "input": retained.record(EVIDENCE), "parent_binding": parent_binding,
        "original_L20_reference": reference, "frozen_first_failure": failure,
        "frozen_sensitivity": {
            "status": "INCONCLUSIVE", "reason": "L20_GLOBAL_GATE_BOUNDARY",
            "native_layer_invocations": 43, "native_layers": [21] * 43,
            "samples": 44, "search_exhausted": False, "minimality_proven": False,
        },
        "reference_policy": upstream.suffix.REFERENCE_POLICY,
        "lineage": upstream.producer.LINEAGE,
        "claim_boundary": (
            "Exact L20 numerical admissibility and input-cell endpoints only; no native L21 "
            "sufficiency, search exhaustion, minimality, producer attribution or executable "
            "successor. Both endpoints remain distinct despite equal H because I is wider than "
            "FP16. Historical failures and bounded independent acceptances remain unchanged. "
            "Native S16 RTZ, INT4 weights and FP16 operator boundaries/scales/KV are unchanged. "
            "No native L0-L8 or L9+ execution, RTL, GPU, FPGA, hardware, simulation, synthesis, "
            "PPA, strict-FP16-state W4A16, new-token or full-model PASS."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = {"diagnostic_id": ID, "status": "BLOCKED", **FLAGS, "native_layers": []}
    try:
        result = check()
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, SyntaxError, ImportError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"], file=sys.stderr)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ADMISSIBLE_INTERVAL_PREPARED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
