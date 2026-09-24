"""Repository-bound, non-admitting L20 endpoint / L21-L23 CPU executor.

--check compiles, tests and authenticates without native execution. --execute
additionally evaluates the two distinct in-memory endpoints with the existing
guarded suffix engine. Neither mode publishes state or admits a candidate.
"""

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import (
    diagnose_q24_s16_l20_to_l21_coordinate62_admissible_threshold_v1 as prepared,
)


ROOT, PYTHON = prepared.ROOT, prepared.PYTHON
retained, require, equal = prepared.retained, prepared.require, prepared.equal
NAME = "q24_s16_l20_endpoint_l21_l23_executor_v1"
ID = "ace3-q24-s16-l20-endpoint-l21-l23-executor-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = ROOT / "build/q24_s16_l20_to_l21_coordinate62_admissible_threshold_v1_attempt001"
PINS = {
    "diagnostic.stdout.json": "7be3b69940585cb1ab5797f0acf18cbdcb6372348c21d0586090313856748500",
    "result.json": "2d946c38863e905b5eb009f721e12388b53b24b8206e8317493d33c278e1af08",
}
EXPECTED_TESTS = 28
FLAGS = dict(prepared.FLAGS)
RUNTIME_MODULE = "ace3.model.candidates.diagnose_q24_s16_l20_endpoint_l21_l23_execute_v2"
EXECUTION = {
    "version": 2,
    "native_layers": [21, 22, 23],
    "maximum_native_layer_invocations": 6,
    "runtime_module": RUNTIME_MODULE,
    "baseline": "unchanged frozen control; not executed",
    "stop": "first mandatory failure per endpoint; BLOCKED or failed L20 precondition aborts batch",
    "output": "stdout evidence only; no state archive, successor, publication or admission",
}
EXECUTOR = {
    "version": 1,
    "mode": "no-native-preflight",
    "candidate_endpoint_count": 2,
    "candidate_order": ["minimum", "maximum"],
    "boundary": "authenticated frozen L20 output / L21 input",
    "coordinate": 62,
    "position": 0,
    "history": [9707],
    "prospective_suffix": [21, 22, 23],
    "native_layers": [],
    "state": "copy I/Z/H; change only I[62]; preserve Z; recompute H[62] by RNE16",
    "baseline": "separate unchanged frozen control; not a third endpoint; not executed",
    "output": "stdout evidence only; no state archive, successor, publication or admission",
}
HISTORY = {
    "status": "INCONCLUSIVE", "reason": "L20_GLOBAL_GATE_BOUNDARY",
    "native_layer_invocations": 43, "native_layers": [21] * 43,
    "samples": 44, "search_exhausted": False, "minimality_proven": False,
}
COMMAND = (
    "cd /home/argustest/ace3-argus && "
    "PYTHONPATH=/home/argustest/ace3-argus /home/argustest/miniconda3/bin/python -B "
    "-m ace3.model.candidates.diagnose_q24_s16_l20_endpoint_l21_l23_executor_v1 --check"
)
EXECUTE_COMMAND = COMMAND.removesuffix("--check") + "--execute"
pinned = prepared.scalar_math.pinned


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 2, "interface": ["--check", "--execute"],
        "focused_tests": EXPECTED_TESTS, "executor": EXECUTOR,
        "input": str(INPUT.relative_to(ROOT)), "input_sha256": PINS,
        "policy_id": prepared.gates.POLICY_ID,
        "local_gate": prepared.upstream.producer.LOCAL_GATE,
        "excess_budget": "1/8",
        "reference_policy": prepared.upstream.suffix.REFERENCE_POLICY,
        "native_layers": [], "reproduction": COMMAND,
        "execution": EXECUTE_COMMAND, "execution_surface": EXECUTION, **FLAGS,
    }
    for key, value in expected.items():
        equal(document[key], value, f"versioned executor contract mismatch: {key}")


@contextmanager
def no_execution():
    def forbidden(*args, **kwargs):
        raise RuntimeError("native/external execution or state publication forbidden")

    with prepared.no_execution(), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates."):
                for attribute in ("execute", "dispatch", "save_state"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        for attribute in ("save", "savez", "savez_compressed"):
            stack.enter_context(patch.object(np, attribute, forbidden))
        yield


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "executor source/contract origin mismatch")
    records = prepared.origins()
    for name in (MODULE, TEST_MODULE, RUNTIME_MODULE, prepared.MODULE, prepared.TEST_MODULE,
                 prepared.upstream.LEGACY_TEST):
        require(name in records and records[name]["path"]
                == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"required repository origin mismatch: {name}")
    return records


def check_preparation(document, parent_binding, reference, failure, analysis):
    expected = {
        "diagnostic_id": prepared.ID, "status": "ADMISSIBLE_INTERVAL_PREPARED",
        "method": prepared.METHOD, "native_layers": [], "native_search_executed": False,
        "parent_binding": parent_binding, "original_L20_reference": reference,
        "frozen_first_failure": failure, "analysis": analysis,
        "frozen_sensitivity": HISTORY,
        "reference_policy": prepared.upstream.suffix.REFERENCE_POLICY,
        "lineage": prepared.upstream.producer.LINEAGE,
        "input": retained.record(prepared.EVIDENCE), **FLAGS,
    }
    for key, value in expected.items():
        equal(document[key], value, f"prepared evidence mismatch: {key}")
    prepared.upstream.check_validation(document["validation"], prepared.EXPECTED_TESTS)
    equal(document["validation"]["contract_parsed"], True, "unparsed prepared contract")


def check_history(document, preparation):
    expected = {
        "status": "BLOCKED", "reason": "PREPARED_DIAGNOSTIC_HAS_NO_NATIVE_EXECUTION_INTERFACE",
        "preparation_validated": True, "returncode": 0,
        "candidate_set": preparation["analysis"]["candidate_endpoints"],
        "analysis": preparation["analysis"],
        "parent_binding": preparation["parent_binding"],
        "original_L20_reference": preparation["original_L20_reference"],
        "reference_policy": preparation["reference_policy"],
        "historical_sensitivity": HISTORY,
        "baseline_control": {"delta_Q24_units": 0, "status": "NOT_EXECUTED"},
        "native_layers": [], "native_search_executed": False,
        "native_L0_L8_invocations": 0, "native_layer_invocations": 0, "rtl_invocations": 0,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED",
    }
    for key, value in expected.items():
        equal(document[key], value, f"historical preparation mismatch: {key}")
    accounting = [{
        **row, "native_L0_L8_invocations": 0, "native_L21_invocations": 0,
        "native_L22_L23_invocations": 0, "rtl_invocations": 0,
        "native_gate_outcomes": None, "status": "NOT_EXECUTED",
    } for row in preparation["analysis"]["candidate_endpoints"]]
    equal(document["native_probe_accounting"], accounting, "historical accounting mismatch")
    require(retained.record(INPUT / "diagnostic.stdout.json") in document["outputs"],
            "historical output binding mismatch")


def state_binding(state):
    return {
        key: {"dtype": value.dtype.str, "shape": list(value.shape),
              "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        for key, value in state.items()
    }


def candidates(analysis, verified_analysis, parent):
    equal(analysis, verified_analysis, "endpoint analysis/reference/threshold mismatch")
    rows = analysis["candidate_endpoints"]
    equal(analysis["candidate_endpoint_count"], 2, "endpoint count mismatch")
    require(len(rows) == 2 and len(analysis["cells"]) == 1, "endpoint/cell count mismatch")
    equal([row["cell_endpoint"] for row in rows], ["minimum", "maximum"],
          "endpoint order mismatch")
    require(rows[0]["Q24_integer"] < rows[1]["Q24_integer"], "endpoints must be distinct")
    result = []
    for row in rows:
        state = prepared.change_coordinate(parent, row["Q24_integer"])
        require(row["delta_Q24_units"] == int(state["i"][62]) - int(parent["i"][62])
                and row["input_FP16_word"] == f"{int(state['h'][62]):04x}",
                "endpoint operand/state mismatch")
        result.append({
            **row, "state_binding": state_binding(state), "status": "NOT_EXECUTED",
            "native_layers": [], "native_L0_L8_invocations": 0,
            "native_L9_plus_invocations": 0, "rtl_invocations": 0,
            "native_gate_outcomes": None,
        })
    return result


def authenticate():
    inputs = prepared.margin.margin.prior.BoundInputs()
    documents = {name: pinned(inputs, INPUT / name, digest) for name, digest in PINS.items()}
    document, history = documents["diagnostic.stdout.json"], documents["result.json"]
    # Reconstruct with the existing independent frozen operand/reference checks,
    # not the prior CLI, prior tests, or any native layer.
    verified, parent_binding, reference, failure, analysis = prepared.authenticate()
    for record in verified.records.values():
        inputs.bind(record)
    check_preparation(document, parent_binding, reference, failure, analysis)
    check_history(history, document)
    for record in (*document["authenticated_inputs"], *history["outputs"]):
        inputs.bind(record)
    prepared.upstream.bind_validation(
        inputs, document["validation"], prepared.SOURCE, prepared.CONTRACT)
    prepared.check_contract(json.loads(prepared.CONTRACT.read_text()))
    parent = inputs.archive(parent_binding)
    endpoints = candidates(document["analysis"], analysis, parent)
    return inputs, document, endpoints, state_binding(parent)


def check_test_result(result, collected):
    equal(collected, EXPECTED_TESTS, "focused test collection mismatch")
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and not result.skipped,
            "focused tests failed, skipped or incomplete")


def validate():
    context = prepared.upstream.producer.context()
    runtime = importlib.import_module(RUNTIME_MODULE)
    equal(list(runtime.LAYERS), EXECUTION["native_layers"], "runtime layer scope changed")
    equal(runtime.EXECUTOR["maximum_native_layer_invocations"],
          EXECUTION["maximum_native_layer_invocations"], "runtime budget changed")
    tests = importlib.import_module(TEST_MODULE)
    for name in (prepared.TEST_MODULE, prepared.upstream.TEST_MODULE,
                 prepared.upstream.LEGACY_TEST):
        importlib.import_module(name)
    records = origins()
    contract_record = retained.record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for record in records.values():
        path = Path(record["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    equal(count, EXPECTED_TESTS, "focused test collection mismatch")
    with no_execution():
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    check_test_result(result, count)
    equal(origins(), records, "source origins changed during validation")
    equal(retained.record(CONTRACT), contract_record, "contract changed during validation")
    return {
        **context, "command": COMMAND, "origins": records, "compiled": compiled,
        "contract": contract_record, "contract_parsed": True,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, "upstream_tests_executed": False,
    }


def check_inputs(inputs, validation):
    equal(origins(), validation["origins"], "source origins changed")
    for record in inputs.records.values():
        equal(retained.record(record["path"]), record,
              f"input/source changed: {record['path']}")


def check():
    validation = validate()
    with no_execution():
        inputs, document, endpoints, baseline = authenticate()
        for record in (*validation["origins"].values(), validation["contract"]):
            inputs.bind(record)
        check_inputs(inputs, validation)
    return {
        "diagnostic_id": ID, "status": "ENDPOINT_EXECUTOR_PREFLIGHT_VALIDATED", **FLAGS,
        "executor": EXECUTOR, "validation": validation,
        "prepared_evidence": {name: retained.record(INPUT / name) for name in PINS},
        "candidate_endpoint_count": len(endpoints), "candidate_set": endpoints,
        "baseline_control": {"delta_Q24_units": 0, "status": "NOT_EXECUTED",
                             "state_binding": baseline},
        "analysis": document["analysis"], "parent_binding": document["parent_binding"],
        "original_L20_reference": document["original_L20_reference"],
        "lineage": document["lineage"], "reference_policy": document["reference_policy"],
        "authenticated_inputs": list(inputs.records.values()),
        "historical_sensitivity": HISTORY,
        "historical_preparation": {
            "status": "BLOCKED", "reason": "PREPARED_DIAGNOSTIC_HAS_NO_NATIVE_EXECUTION_INTERFACE",
        },
        "frozen_first_failure": document["frozen_first_failure"],
        "native_layers": [], "native_search_executed": False,
        "minimality_proven": False, "search_exhausted": False,
        "claim_boundary": (
            "No-native CPU-software executor preflight only, pending independent Host review. "
            "Both prepared endpoints remain distinct even when H agrees. No L21-L23 numerical "
            "result, execution authorization, state publication or admission. Historical failures "
            "and bounded acceptances remain unchanged. Native S16 RTZ Q24 residual state is wider "
            "than FP16; INT4 weights and FP16 scales/operator boundaries/KV are unchanged. "
            "No native execution in this check; --execute is a separate bounded L21-L23 mode. "
            "No native L0-L8 replay, RTL, hardware, GPU, FPGA, simulation, synthesis, "
            "PPA, strict-FP16-state W4A16, new-token or full-model PASS."
        ),
    }


def execution_inputs(inputs, document):
    runtime = importlib.import_module(RUNTIME_MODULE)
    engine = runtime.engine
    freeze_record = retained.record(engine.margin.INPUT / "freeze.json")
    require(freeze_record in inputs.records.values(), "unbound original reference freeze")
    freeze = inputs.read(freeze_record)
    require(freeze["reference_extension"] in inputs.records.values(),
            "unbound original reference extension")
    extension = inputs.read(freeze["reference_extension"])
    equal(extension["layers"]["20"], document["original_L20_reference"],
          "L20 original reference substitution")
    for layer in (20, *EXECUTION["native_layers"]):
        for key in ("fp16", "binary64"):
            require(extension["layers"][str(layer)][key] in inputs.records.values(),
                    "unbound original layer reference")
    require(extension["checkpoint"] in inputs.records.values(), "unbound model operands")
    return inputs.archive(document["parent_binding"]), extension


def checked_execution(result, execute_requested):
    result.update(check())
    if not execute_requested:
        return
    runtime = importlib.import_module(RUNTIME_MODULE)
    result.update(status="BLOCKED", execution_surface=EXECUTION, claim_boundary=runtime.CLAIM)
    with no_execution():
        inputs = prepared.margin.margin.prior.BoundInputs()
        for record in result["authenticated_inputs"]:
            inputs.bind(record)
        parent, extension = execution_inputs(inputs, result)
        trajectory20 = inputs.archive(extension["layers"]["20"]["fp16"])
        reference20 = runtime.engine.global_reference(inputs, extension["layers"]["20"])
        check_inputs(inputs, result["validation"])
    try:
        with runtime.engine.safe_open(
                str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
            def run_layer(layer, state, entry):
                return runtime.evaluate_layer(
                    inputs, model, extension, layer, state, result["native_layers"], entry)

            runtime.execute_endpoints(
                result, parent, result["analysis"], trajectory20, reference20, run_layer)
    finally:
        runtime.accounting(result)
        result["authenticated_inputs"] = list(inputs.records.values())
        check_inputs(inputs, result["validation"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = {"diagnostic_id": ID, "status": "BLOCKED", **FLAGS, "native_layers": [],
              "command": EXECUTE_COMMAND if args.execute else COMMAND}
    try:
        checked_execution(result, args.execute)
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, SyntaxError, ImportError) as exc:
        result.update(status="BLOCKED", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    layers = result["native_layers"]
    result.update(native_invocations=len(layers), native_layer_invocations=len(layers),
                  native_L9_plus_invocations=len(layers), native_search_executed=bool(layers))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in (
        "ENDPOINT_EXECUTOR_PREFLIGHT_VALIDATED", "BOUNDED_ENDPOINTS_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
