"""Prepare strict-interior L20 Q24 probes without executing any native layer.

The adjacent versioned contract publishes the repository-bound --check command.
This interface authenticates frozen evidence and prepares inputs, not a suffix
executor, numerical acceptance, exhaustive search, or permission to execute.
"""

import argparse
from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_endpoint_l21_l23_execute_v2 as endpoint


v1, prepared, upstream = endpoint.v1, endpoint.prepared, endpoint.upstream
ROOT, PYTHON = endpoint.ROOT, endpoint.PYTHON
retained, require, equal = endpoint.retained, endpoint.require, endpoint.equal
NAME = "q24_s16_l20_interval_l21_l23_interior_scan_v1"
ID = "ace3-q24-s16-l20-interval-l21-l23-interior-scan-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = ROOT / "build/q24_s16_l20_endpoint_l21_l23_execute_v2_attempt001/native_execution_v1"
PINS = {
    "diagnostic.stdout.json": "64558fa6ee596b58ceda0546eadf51391d48cabae5a1bc7426ea1ac5fd9b6c91",
    "terminal.json": "9751122781185bf29fae25bfb37b0705bd51168a25e87cff0e6eb6401ee925b5",
    "command.json": "aafd3a31849bfcb10f54886696c657c5adc9c73358a67ca2f880014825ff2d34",
}
EXPECTED_TESTS = 22
FLAGS = dict(endpoint.FLAGS)
COMMAND = (
    "cd /home/argustest/ace3-argus && "
    "PYTHONPATH=/home/argustest/ace3-argus /home/argustest/miniconda3/bin/python -B "
    "-m ace3.model.candidates.diagnose_q24_s16_l20_interval_l21_l23_interior_scan_v1 --check"
)
METHOD = {
    "coordinate": 62, "position": 0, "history": [9707],
    "boundary": "authenticated frozen L20 output / L21 input",
    "prospective_suffix": [21, 22, 23], "native_layers": [],
    "subdivisions": 16, "probe_count": 15,
    "order": "ascending lo + ((hi - lo) * k) // 16 for k=1..15",
    "exclusions": "both executed endpoints and unchanged frozen baseline",
    "state": "copy I/Z/H; change only I[62]; preserve Z; recompute H[62] by RNE16",
    "precondition": "authenticated single passing RNE cell; unchanged full-vector L20 S18 gate",
    "coverage": "15 strict-interior samples only; no native sufficiency or monotonicity claim",
}
CLAIM = (
    "Preparation-only native-S16-RTZ Q24/P0 CPU software, pending independent Host Reviewer. "
    "Q24 residual state is wider than FP16; G128 asymmetric packed INT4 weights, native GEMM "
    "nibble order, no qzero plus-one, FP16 scales/operator boundaries/KV are unchanged. "
    "Historical endpoint failures and bounded acceptances are preserved. No native execution, "
    "L0-L8 replay, RTL, GPU, FPGA, hardware, simulation, synthesis, PPA, strict-FP16-state "
    "W4A16, new-token or full-model PASS; no admission, publication or execution authorization."
)


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 1, "interface": ["--check"],
        "focused_tests": EXPECTED_TESTS, "method": METHOD,
        "input": str(INPUT.relative_to(ROOT)), "input_sha256": PINS,
        "policy_id": prepared.gates.POLICY_ID,
        "local_gate": upstream.producer.LOCAL_GATE, "excess_budget": "1/8",
        "reference_policy": upstream.suffix.REFERENCE_POLICY,
        "reproduction": COMMAND, "native_layers": [], **FLAGS,
    }
    for key, value in expected.items():
        equal(document[key], value, f"interior contract mismatch: {key}")


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "interior source/contract origin mismatch")
    records = endpoint.origins()
    for name in (MODULE, TEST_MODULE, endpoint.MODULE, endpoint.TEST_MODULE):
        require(name in records and records[name]["path"]
                == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"required repository origin mismatch: {name}")
    return records


def interior_points(lo, hi, baseline):
    require(all(type(value) is int and -(1 << 63) <= value < (1 << 63)
                for value in (lo, hi, baseline)), "signed int64 Q24 bounds required")
    require(lo < baseline < hi and hi - lo >= METHOD["subdivisions"],
            "nonempty strict-interior interval and baseline required")
    points = [lo + (hi - lo) * k // METHOD["subdivisions"]
              for k in range(1, METHOD["subdivisions"])]
    require(len(points) == len(set(points)) == METHOD["probe_count"]
            and all(lo < value < hi and value != baseline for value in points),
            "duplicate, endpoint or baseline in interior schedule")
    return points


def schedule(analysis, parent):
    retained.verify_parent(parent, parent)
    before = v1.state_binding(parent)
    equal(analysis["baseline_Q24_integer"], int(parent["i"][62]), "baseline substitution")
    require(len(analysis["cells"]) == 1, "authenticated single passing RNE cell required")
    cell = analysis["cells"][0]
    lo, hi = analysis["minimum_Q24_integer"], analysis["maximum_Q24_integer"]
    equal([lo, hi], [cell["minimum_Q24_integer"], cell["maximum_Q24_integer"]],
          "interval/cell bounds mismatch")
    points = interior_points(lo, hi, analysis["baseline_Q24_integer"])
    rows = []
    for index, integer in enumerate(points):
        state = prepared.change_coordinate(parent, integer)
        word = f"{int(state['h'][62]):04x}"
        equal(word, cell["input_FP16_word"], "interior RNE cell mismatch")
        require(np.array_equal(state["h"], parent["h"]),
                "interior probe changed the authenticated L20 gate operand")
        rows.append({
            "probe_index": index, "Q24_integer": integer,
            "delta_Q24_units": integer - analysis["baseline_Q24_integer"],
            "delta_exact": str(Fraction(integer - analysis["baseline_Q24_integer"], prepared.Q)),
            "input_FP16_word": word, "state_binding": v1.state_binding(state),
            "status": "NOT_EXECUTED",
        })
    equal(v1.state_binding(parent), before, "frozen parent mutated")
    return rows


def check_history(document, terminal, preparation):
    expected = {
        "diagnostic_id": endpoint.ID, "status": "ENDPOINT_GATE_FAIL",
        "command": endpoint.EXECUTE_COMMAND, "executor": endpoint.EXECUTOR,
        "candidate_endpoint_count": 2, **FLAGS,
        "native_invocations": 2, "native_layer_invocations": 2,
        "native_L9_plus_invocations": 2, "native_layers": [21, 21],
        "native_L21_invocations": 2, "native_L22_invocations": 0, "native_L23_invocations": 0,
        "native_search_executed": True, "minimality_proven": False, "search_exhausted": False,
    }
    for key in ("analysis", "parent_binding", "original_L20_reference", "lineage",
                "reference_policy", "baseline_control", "frozen_first_failure",
                "historical_preparation", "historical_sensitivity"):
        expected[key] = preparation[key]
    for key, value in expected.items():
        equal(document[key], value, f"endpoint execution mismatch: {key}")
    rows = document["candidate_set"]
    equal(len(rows), 2, "executed endpoint count mismatch")
    for row, original in zip(rows, preparation["candidate_set"]):
        for key in ("cell_endpoint", "Q24_integer", "delta_Q24_units", "delta_exact",
                    "input_FP16_word", "state_binding"):
            equal(row[key], original[key], f"executed endpoint binding mismatch: {key}")
        for key, value in {
            "status": "FAIL", "native_layers": [21], "native_L9_plus_invocations": 1,
            "native_L0_L8_invocations": 0, "rtl_invocations": 0,
            "first_failure": {"gate": "binary64_v1", "index": 62, "node": [21, 0, 18]},
        }.items():
            equal(row[key], value, f"executed endpoint outcome mismatch: {key}")
        gate = row["L20_global_gate"]
        require(gate["status"] == "PASS" and gate["policy_id"] == prepared.gates.POLICY_ID
                and gate["binary64_v1"]["passed"] is True
                and gate["binary64_v1"]["coordinates"] == 896,
                "frozen L20 full-vector gate mismatch")
        equal(len(row["layers"]), 1, "historical suffix continued past failure")
        layer = row["layers"][0]
        endpoint.check_layer_result(21, layer, None)
        equal(layer["failure"], row["first_failure"], "historical first failure mismatch")
        equal(row["native_gate_outcomes"], [layer["reports"]], "historical gate outcomes mismatch")
        require(layer["prior_kv"] == "own empty P0"
                and layer["prior_layer_kv_consumed"] is False, "historical P0 KV mismatch")
        for state_key, stage_key in (("i", "input_i"), ("z", "input_z"), ("h", "input_hidden")):
            equal(layer["actual_stage_bindings"][stage_key], row["state_binding"][state_key],
                  "historical native input state mismatch")
    terminal_expected = {
        "status": "EXHAUSTED_NO_PASS", "diagnostic_status": "ENDPOINT_GATE_FAIL",
        "diagnostic_error": None, "returncode": 1, "phase": "complete",
        "command": endpoint.EXECUTE_COMMAND, "executor_invocations": 1,
        "historical_artifacts_unchanged": True, "finite_endpoint_set_exhausted": True,
        "first_passing_candidate_index": None,
    }
    for key in ("native_invocations", "native_layer_invocations", "native_L9_plus_invocations",
                "native_L0_L8_invocations", "native_layers", "native_L21_invocations",
                "native_L22_invocations", "native_L23_invocations", "rtl_invocations",
                "candidate_admitted", "successor_published", "policy_adopted", "normal_host_review"):
        terminal_expected[key] = document[key]
    for key, value in terminal_expected.items():
        equal(terminal[key], value, f"endpoint terminal mismatch: {key}")
    equal(len(terminal["candidates"]), 2, "terminal endpoint count mismatch")
    for summary, row in zip(terminal["candidates"], rows):
        for key, value in summary.items():
            equal(row[key], value, f"terminal endpoint mismatch: {key}")


def authenticate():
    inputs, preparation, _, parent, _ = endpoint.authenticate()
    documents = {name: v1.pinned(inputs, INPUT / name, digest) for name, digest in PINS.items()}
    document, terminal, command = (documents[name] for name in PINS)
    check_history(document, terminal, preparation)
    endpoint.check_contract(json.loads(endpoint.CONTRACT.read_text()))
    validation = document["validation"]
    upstream.check_validation(validation, endpoint.EXPECTED_TESTS)
    equal(validation["command"], endpoint.COMMAND, "historical check command mismatch")
    equal(validation["contract"], retained.record(endpoint.CONTRACT), "endpoint contract changed")
    equal(sorted(validation["compiled"], key=lambda row: row["path"]),
          sorted(validation["origins"].values(), key=lambda row: row["path"]),
          "endpoint compiled closure mismatch")
    current = origins()
    for name, record in validation["origins"].items():
        equal(current[name], record, f"endpoint source origin changed: {name}")
    for record in (*document["authenticated_inputs"], *validation["compiled"],
                   validation["contract"], *terminal["outputs"], *command["source_bindings"],
                   command["capture_source"], command["accepted_capture"],
                   *command["historical_before"]):
        inputs.bind(record)
    for key, value in (("cwd", str(ROOT)), ("executable", str(PYTHON)),
                       ("PYTHONPATH", str(ROOT)), ("command", endpoint.EXECUTE_COMMAND)):
        equal(command[key], value, f"endpoint command context mismatch: {key}")
    return inputs, document, terminal, parent


def check_test_result(outcome, collected):
    equal(collected, EXPECTED_TESTS, "focused test collection mismatch")
    require(outcome.testsRun == EXPECTED_TESTS and outcome.wasSuccessful() and not outcome.skipped,
            "focused tests failed, skipped or incomplete")


def validate(result):
    context = upstream.producer.context()
    tests = importlib.import_module(TEST_MODULE)
    for name in (endpoint.TEST_MODULE, v1.TEST_MODULE, prepared.TEST_MODULE,
                 upstream.TEST_MODULE, upstream.LEGACY_TEST):
        importlib.import_module(name)
    records = origins()
    contract = retained.record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    validation = {
        **context, "command": COMMAND, "origins": records, "compiled": [],
        "contract": contract, "contract_parsed": True,
        "legacy_19_tests_executed": False, "upstream_tests_executed": False,
        "native_L0_L8_invocations": 0, "native_layer_invocations": 0, "rtl_invocations": 0,
    }
    result["validation"] = validation
    for record in records.values():
        path = Path(record["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        validation["compiled"].append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    equal(count, EXPECTED_TESTS, "focused test collection mismatch")
    with v1.no_execution():
        outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    validation.update(collected=count, executed=outcome.testsRun, failures=len(outcome.failures),
                      errors=len(outcome.errors), skipped=len(outcome.skipped))
    check_test_result(outcome, count)
    equal(origins(), records, "source origins changed during validation")
    equal(retained.record(CONTRACT), contract, "contract changed during validation")


def check(result):
    validate(result)
    with v1.no_execution():
        inputs, document, terminal, parent = authenticate()
        probes = schedule(document["analysis"], parent)
        executed = [row["Q24_integer"] for row in document["candidate_set"]]
        require(set(executed).isdisjoint(row["Q24_integer"] for row in probes),
                "already-executed endpoint scheduled")
        for record in (*result["validation"]["origins"].values(), result["validation"]["contract"]):
            inputs.bind(record)
        endpoint.check_inputs(inputs, result["validation"])
    result.update(
        status="INTERIOR_SCAN_PREFLIGHT_VALIDATED", method=METHOD,
        probe_count=len(probes), probes=probes, excluded_executed_endpoints=executed,
        endpoint_exclusion_proven=True, analysis=document["analysis"],
        parent_binding=document["parent_binding"], baseline_control=document["baseline_control"],
        original_L20_reference=document["original_L20_reference"], lineage=document["lineage"],
        reference_policy=document["reference_policy"],
        historical_endpoint_execution={
            "status": document["status"], "terminal_status": terminal["status"],
            "native_layers": document["native_layers"], "native_layer_invocations": 2,
            "first_failures": [row["first_failure"] for row in document["candidate_set"]],
            "evidence": {name: retained.record(INPUT / name) for name in PINS},
        },
        historical_preparation=document["historical_preparation"],
        historical_sensitivity=document["historical_sensitivity"],
        historical_v1=document["historical_v1"],
        authenticated_inputs=list(inputs.records.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = {
        "diagnostic_id": ID, "status": "BLOCKED", "command": COMMAND, **FLAGS,
        "native_layers": [], "native_search_executed": False,
        "minimality_proven": False, "search_exhausted": False, "claim_boundary": CLAIM,
    }
    try:
        check(result)
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, SyntaxError, ImportError) as exc:
        result.update(status="BLOCKED", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "INTERIOR_SCAN_PREFLIGHT_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
