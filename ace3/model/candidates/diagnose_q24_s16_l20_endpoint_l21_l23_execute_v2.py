"""Versioned, non-admitting L20 endpoint / L21-L23 CPU-software executor.

--check compiles and runs only the new focused tests, then authenticates frozen
inputs without native execution. --execute additionally evaluates both distinct
Q24 endpoints, stopping each suffix at its first mandatory numerical failure.
The adjacent contract publishes the exact repository-bound command context.
"""

import argparse
from contextlib import ExitStack, contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_endpoint_l21_l23_executor_v1 as v1


prepared, upstream = v1.prepared, v1.prepared.upstream
engine = upstream.margin.upstream.upstream
native, local = upstream.producer.native, upstream.producer.local
retained, require, equal = v1.retained, v1.require, v1.equal
ROOT, PYTHON = v1.ROOT, v1.PYTHON
NAME = "q24_s16_l20_endpoint_l21_l23_execute_v2"
ID = "ace3-q24-s16-l20-endpoint-l21-l23-execute-v2"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = ROOT / "build/q24_s16_l20_endpoint_l21_l23_executor_v1_attempt001"
PINS = {
    "diagnostic.stdout.json": "37a1c9134fcb6bf50481eaa6b84bfa4fded79e747241e0aa4c11b92e8f8800ca",
    "result.json": "f31b3ac481cf20d4e6051128f53f90b5024fda77bb27ce1d448660578b84f12e",
}
LAYERS = (21, 22, 23)
EXPECTED_TESTS = 25
FLAGS = dict(v1.FLAGS)
COMMAND = (
    "cd /home/argustest/ace3-argus && "
    "PYTHONPATH=/home/argustest/ace3-argus /home/argustest/miniconda3/bin/python -B "
    "-m ace3.model.candidates.diagnose_q24_s16_l20_endpoint_l21_l23_execute_v2 --check"
)
EXECUTE_COMMAND = COMMAND.removesuffix("--check") + "--execute"
EXECUTOR = {
    "version": 2,
    "candidate_order": ["minimum", "maximum"],
    "candidate_endpoint_count": 2,
    "coordinate": 62,
    "position": 0,
    "history": [9707],
    "native_layers": list(LAYERS),
    "maximum_native_layer_invocations": 6,
    "baseline": "unchanged frozen control; not executed",
    "stop": "first mandatory failure per endpoint; BLOCKED or failed L20 precondition aborts batch",
    "state": "copy I/Z/H; change only I[62]; preserve Z; recompute H[62] by RNE16",
    "output": "stdout evidence only; no state archive, successor, publication or admission",
}
CLAIM = (
    "Bounded non-admitting native-S16-RTZ Q24/P0 CPU software only, pending independent "
    "normal Host review. Q24 residual state is wider than FP16; native G128 INT4 weights "
    "and FP16 scales, operator boundaries and KV are unchanged. Both integer endpoints "
    "remain distinct even when H agrees. No minimality or exhaustive-search claim. "
    "Historical failures and bounded acceptances remain unchanged. No L0-L8 replay, RTL, "
    "hardware, GPU, FPGA, simulation, synthesis, PPA, strict-FP16-state W4A16, new-token "
    "or full-model PASS."
)


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 2, "interface": ["--check", "--execute"],
        "focused_tests": EXPECTED_TESTS, "executor": EXECUTOR,
        "input": str(INPUT.relative_to(ROOT)), "input_sha256": PINS,
        "policy_id": prepared.gates.POLICY_ID,
        "local_gate": upstream.producer.LOCAL_GATE, "excess_budget": "1/8",
        "reference_policy": upstream.suffix.REFERENCE_POLICY,
        "reproduction": COMMAND, "execution": EXECUTE_COMMAND, **FLAGS,
    }
    for key, value in expected.items():
        equal(document[key], value, f"v2 contract mismatch: {key}")


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "v2 source/contract origin mismatch")
    records = v1.origins()
    for name in (MODULE, TEST_MODULE, v1.MODULE, v1.TEST_MODULE, upstream.LEGACY_TEST):
        require(name in records and records[name]["path"]
                == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"required repository origin mismatch: {name}")
    return records


def check_frozen(document, history, preparation, endpoints, baseline):
    expected = {
        "diagnostic_id": v1.ID, "status": "ENDPOINT_EXECUTOR_PREFLIGHT_VALIDATED",
        "executor": v1.EXECUTOR, "candidate_endpoint_count": 2, "candidate_set": endpoints,
        "baseline_control": {"delta_Q24_units": 0, "status": "NOT_EXECUTED",
                             "state_binding": baseline},
        "historical_sensitivity": v1.HISTORY,
        "historical_preparation": {
            "status": "BLOCKED", "reason": "PREPARED_DIAGNOSTIC_HAS_NO_NATIVE_EXECUTION_INTERFACE",
        },
        "native_layers": [], "native_search_executed": False,
        "minimality_proven": False, "search_exhausted": False, **FLAGS,
    }
    for key in ("analysis", "parent_binding", "original_L20_reference", "lineage",
                "reference_policy", "frozen_first_failure"):
        expected[key] = preparation[key]
    for key, value in expected.items():
        equal(document[key], value, f"frozen v1 preflight mismatch: {key}")
    upstream.check_validation(document["validation"], v1.EXPECTED_TESTS)
    expected_history = {
        "status": "BLOCKED", "reason": "ACCEPTED_EXECUTOR_HAS_NO_NATIVE_EXECUTION_INTERFACE",
        "diagnostic_status": document["status"], "preflight_validated": True,
        "returncode": 0, "phase": "native_execution_interface", "executor_invocations": 1,
        "native_layers": [], "native_search_executed": False,
        "native_L0_L8_invocations": 0, "native_layer_invocations": 0,
        "native_L21_invocations": 0, "native_L22_invocations": 0, "native_L23_invocations": 0,
        "native_L9_plus_invocations": 0, "rtl_invocations": 0,
        "candidate_admitted": False, "successor_published": False, "policy_adopted": False,
        "scientific_result_claim": False, "normal_host_review": "REQUIRED",
    }
    for key in ("candidate_set", "analysis", "baseline_control", "parent_binding",
                "original_L20_reference", "lineage", "reference_policy", "validation",
                "historical_sensitivity", "historical_preparation"):
        expected_history[key] = document[key]
    for key, value in expected_history.items():
        equal(history[key], value, f"frozen v1 BLOCKED record mismatch: {key}")


def bind_v1_output(inputs, record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256", "size"},
            "invalid frozen v1 output record schema")
    require(type(record["size"]) is int and record["size"] >= 0,
            "invalid frozen v1 output record size")
    return inputs.bind({"path": record["path"], "sha256": record["sha256"],
                        "bytes": record["size"]})


def authenticate():
    with v1.no_execution():
        inputs, preparation, endpoints, baseline = v1.authenticate()
        documents = {name: v1.pinned(inputs, INPUT / name, digest)
                     for name, digest in PINS.items()}
        document, history = documents["diagnostic.stdout.json"], documents["result.json"]
        check_frozen(document, history, preparation, endpoints, baseline)
        v1.check_contract(json.loads(v1.CONTRACT.read_text()))
        current = origins()
        validation = document["validation"]
        equal(validation["contract"], retained.record(v1.CONTRACT), "v1 contract changed")
        equal(sorted(validation["compiled"], key=lambda record: record["path"]),
              sorted(validation["origins"].values(), key=lambda record: record["path"]),
              "v1 compiled closure mismatch")
        for name, record in validation["origins"].items():
            equal(current[name], record, f"v1 source origin changed: {name}")
        for record in (*document["authenticated_inputs"], *history["authenticated_inputs"],
                       *validation["compiled"], validation["contract"]):
            inputs.bind(record)
        for record in history["outputs"]:
            bind_v1_output(inputs, record)
        parent = inputs.archive(document["parent_binding"])
        freeze_record = retained.record(engine.margin.INPUT / "freeze.json")
        require(freeze_record in inputs.records.values(), "unbound original reference freeze")
        freeze = inputs.read(freeze_record)
        require(freeze["reference_extension"] in inputs.records.values(),
                "unbound original reference extension")
        extension = inputs.read(freeze["reference_extension"])
        equal(extension["layers"]["20"], document["original_L20_reference"],
              "L20 original reference substitution")
        for layer in (20, *LAYERS):
            item = extension["layers"][str(layer)]
            for key in ("fp16", "binary64"):
                require(item[key] in inputs.records.values(), "unbound original layer reference")
        require(extension["checkpoint"] in inputs.records.values(), "unbound model operands")
    return inputs, document, history, parent, extension


@contextmanager
def native_only(audit, layer):
    require(type(layer) is int and layer in LAYERS, "only native L21-L23 permitted")
    require(len(audit) < EXECUTOR["maximum_native_layer_invocations"],
            "endpoint invocation budget exceeded")
    raw = native.candidate._stages
    called = False

    def forbidden(*args, **kwargs):
        raise RuntimeError("native replay, external execution or publication forbidden")

    def guarded(tensors, requested, parent, arrays):
        nonlocal called
        require(type(requested) is int and requested == layer and not called,
                "native dispatch mismatch or duplicate invocation")
        called = True
        audit.append(requested)
        yield from raw(tensors, requested, parent, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates."):
                for attribute in ("stages", "continuation_stages", "native_layer", "run",
                                  "_stages", "execute", "dispatch", "save", "save_state", "write"):
                    if callable(getattr(module, attribute, None)):
                        replacement = (guarded if module is native.candidate
                                       and attribute == "_stages" else forbidden)
                        stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        for attribute in ("save", "savez", "savez_compressed"):
            stack.enter_context(patch.object(np, attribute, forbidden))
        yield


def check_layer_result(layer, entry, output):
    reports = entry["reports"]
    require(0 < len(reports) <= 19, "incomplete native gate reports")
    for stage, report in enumerate(reports):
        require(report["stage"] == stage and report["node"] == [layer, 0, stage]
                and report["policy_id"] == prepared.gates.POLICY_ID
                and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
                and report["local_reference_independent"] is (stage < 18),
                "native report source/state/KV/policy mismatch")
        gate = report["local_operator_fp16"] if stage < 18 else report["binary64_v1"]
        require(type(gate["passed"]) is bool
                and report["status"] == ("PASS" if gate["passed"] else "FAIL"),
                "mandatory native gate/status mismatch")
        require(stage == len(reports) - 1 or report["status"] == "PASS",
                "native execution continued after first failure")
        if stage == 18:
            equal(gate["coordinates"], 896, "global gate must cover full vector")
    if reports[-1]["status"] == "FAIL":
        require(entry["status"] == "FAIL" and output is None
                and entry["failure"]["node"] == reports[-1]["node"]
                and entry["failure"]["gate"]
                == ("binary64_v1" if len(reports) == 19 else "local_operator_fp16"),
                "native first-failure boundary mismatch")
    else:
        require(len(reports) == 19 and entry["status"] == "PASS"
                and entry["failure"] is None and output is not None,
                "incomplete passing native layer")
        retained.verify_parent(output, output)


def accounting(result):
    layers = result["native_layers"]
    require(all(type(layer) is int and layer in LAYERS for layer in layers)
            and len(layers) <= 6, "invalid native accounting")
    result.update(native_invocations=len(layers), native_layer_invocations=len(layers),
                  native_L9_plus_invocations=len(layers), native_search_executed=bool(layers))
    for layer in LAYERS:
        result[f"native_L{layer}_invocations"] = layers.count(layer)


def execute_endpoints(result, parent, analysis, trajectory20, reference20, run_layer):
    endpoints = v1.candidates(analysis, analysis, parent)
    equal(result["candidate_set"], endpoints, "authenticated endpoint set changed")
    baseline = v1.state_binding(parent)
    result["status"] = "BLOCKED"
    try:
        for row in result["candidate_set"]:
            state = prepared.change_coordinate(parent, row["Q24_integer"])
            equal(v1.state_binding(state), row["state_binding"], "endpoint state substitution")
            row.update(status="BLOCKED", layers=[])
            start = len(result["native_layers"])
            try:
                gate = upstream.parent_gate(state, trajectory20, reference20)
                row["L20_global_gate"] = gate
                if gate["status"] != "PASS":
                    require(gate["status"] == "FAIL", "L20 precondition blocked")
                    row.update(status="FAIL", first_failure={"node": [20, 0, 18],
                                                            "gate": "binary64_v1"})
                    result.update(status="ENDPOINT_GATE_FAIL", reason="L20_GLOBAL_GATE_BOUNDARY")
                    return
                for layer in LAYERS:
                    before = v1.state_binding(state)
                    entry = {"layer": layer, "position": 0, "input_state_binding": before,
                             "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
                             "status": "BLOCKED", "reports": []}
                    row["layers"].append(entry)
                    output = run_layer(layer, state, entry)
                    equal(v1.state_binding(state), before, "native driver mutated its input state")
                    check_layer_result(layer, entry, output)
                    if entry["status"] == "FAIL":
                        row.update(status="FAIL", first_failure=entry["failure"])
                        break
                    state = {key: value.copy() for key, value in output.items()}
                    entry["output_state_binding"] = v1.state_binding(state)
                else:
                    row.update(status="PASS", first_failure=None)
            finally:
                row["native_layers"] = result["native_layers"][start:]
                row["native_L9_plus_invocations"] = len(row["native_layers"])
                row["native_gate_outcomes"] = [entry["reports"] for entry in row["layers"]]
                accounting(result)
            equal(v1.state_binding(parent), baseline, "frozen parent was mutated")
        result["status"] = ("BOUNDED_ENDPOINTS_PASS"
                            if all(row["status"] == "PASS" for row in result["candidate_set"])
                            else "ENDPOINT_GATE_FAIL")
    finally:
        accounting(result)


def evaluate_layer(inputs, model, extension, layer, parent, audit, entry):
    require(type(layer) is int and layer in LAYERS, "only native L21-L23 permitted")
    item = extension["layers"][str(layer)]
    entry["original_reference"] = item
    tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
    local.authenticate_tensors(tensors, item["canonical"], layer)
    trajectory = inputs.archive(item["fp16"])
    reference = engine.global_reference(inputs, item)
    arrays, local_refs = {}, {}
    try:
        with native_only(audit, layer):
            failure, timings = engine.drive_layer(
                tensors, layer, parent, trajectory, reference, arrays, local_refs, entry["reports"])
        entry.update(status="FAIL" if failure else "PASS", failure=failure, phase_timings=timings)
        if failure:
            return None
        output = retained.state_from(arrays, "output", "stage18")
        retained.verify_parent(output, output)
        kv = {kind: arrays["output_cache_" + kind] for kind in ("k", "v")}
        for value in kv.values():
            local.finite_words(value, (1, 128))
        entry["own_P0_FP16_KV_binding"] = v1.state_binding(kv)
        return output
    finally:
        entry["actual_stage_bindings"] = v1.state_binding(arrays)
        entry["local_reference_bindings"] = v1.state_binding(local_refs)


def check_test_result(result, collected):
    equal(collected, EXPECTED_TESTS, "focused test collection mismatch")
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and not result.skipped,
            "focused tests failed, skipped or incomplete")


def validate(result):
    context = upstream.producer.context()
    tests = importlib.import_module(TEST_MODULE)
    for name in (v1.TEST_MODULE, prepared.TEST_MODULE, upstream.TEST_MODULE, upstream.LEGACY_TEST):
        importlib.import_module(name)
    records = origins()
    contract_record = retained.record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    validation = {**context, "command": COMMAND, "origins": records, "compiled": compiled,
                  "contract": contract_record, "contract_parsed": True,
                  "legacy_19_tests_executed": False, "upstream_tests_executed": False,
                  "native_L0_L8_invocations": 0, "native_layer_invocations": 0,
                  "rtl_invocations": 0}
    result["validation"] = validation
    for record in records.values():
        path = Path(record["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    equal(count, EXPECTED_TESTS, "focused test collection mismatch")
    with v1.no_execution():
        outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    validation.update(collected=count, executed=outcome.testsRun, failures=len(outcome.failures),
                      errors=len(outcome.errors), skipped=len(outcome.skipped))
    check_test_result(outcome, count)
    equal(origins(), records, "source origins changed during validation")
    equal(retained.record(CONTRACT), contract_record, "contract changed during validation")


def check_inputs(inputs, validation):
    equal(origins(), validation["origins"], "source origins changed")
    for record in inputs.records.values():
        equal(retained.record(record["path"]), record,
              f"input/source changed: {record['path']}")


def checked_execution(result, execute_requested):
    validate(result)
    inputs, document, history, parent, extension = authenticate()
    for record in (*result["validation"]["origins"].values(), result["validation"]["contract"]):
        inputs.bind(record)
    result.update(
        status="EXECUTOR_READY", executor=EXECUTOR, candidate_endpoint_count=2,
        candidate_set=document["candidate_set"], baseline_control=document["baseline_control"],
        analysis=document["analysis"], parent_binding=document["parent_binding"],
        original_L20_reference=document["original_L20_reference"], lineage=document["lineage"],
        reference_policy=document["reference_policy"], frozen_first_failure=document["frozen_first_failure"],
        historical_sensitivity=document["historical_sensitivity"],
        historical_preparation=document["historical_preparation"],
        historical_v1={"status": history["status"], "reason": history["reason"],
                       "evidence": {name: retained.record(INPUT / name) for name in PINS}},
        authenticated_inputs=list(inputs.records.values()))
    check_inputs(inputs, result["validation"])
    if not execute_requested:
        return
    result["status"] = "BLOCKED"
    try:
        trajectory20 = inputs.archive(extension["layers"]["20"]["fp16"])
        reference20 = engine.global_reference(inputs, extension["layers"]["20"])
        with engine.safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
            def run_layer(layer, state, entry):
                return evaluate_layer(inputs, model, extension, layer, state,
                                      result["native_layers"], entry)

            execute_endpoints(result, parent, document["analysis"], trajectory20, reference20, run_layer)
    finally:
        accounting(result)
        result["authenticated_inputs"] = list(inputs.records.values())
        check_inputs(inputs, result["validation"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = {
        "diagnostic_id": ID, "status": "BLOCKED", **FLAGS,
        "command": EXECUTE_COMMAND if args.execute else COMMAND,
        "native_layers": [], "native_search_executed": False,
        "minimality_proven": False, "search_exhausted": False, "claim_boundary": CLAIM,
    }
    try:
        checked_execution(result, args.execute)
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, SyntaxError, ImportError) as exc:
        result.update(status="BLOCKED", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    accounting(result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("EXECUTOR_READY", "BOUNDED_ENDPOINTS_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
