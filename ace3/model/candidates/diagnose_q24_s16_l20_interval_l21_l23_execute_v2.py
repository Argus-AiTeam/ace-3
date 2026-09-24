"""Repository-bound, non-admitting strict-interior L21-L23 CPU executor.

--check compiles the repository closure, runs only this version's focused tests,
and authenticates the prepared schedule without native execution. --execute
additionally evaluates all 15 probes, stopping each at its first mandatory
failure. The adjacent contract publishes both command contexts and boundaries.
"""

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

from ace3.model.candidates import diagnose_q24_s16_l20_interval_l21_l23_interior_scan_v1 as interior


endpoint, v1 = interior.endpoint, interior.v1
prepared, upstream = interior.prepared, interior.upstream
engine = endpoint.engine
retained, require, equal = interior.retained, interior.require, interior.equal
ROOT, PYTHON = interior.ROOT, interior.PYTHON
NAME = "q24_s16_l20_interval_l21_l23_execute_v2"
ID = "ace3-q24-s16-l20-interval-l21-l23-execute-v2"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = ROOT / "build/q24_s16_l20_interval_l21_l23_interior_scan_v1_attempt001"
PINS = {
    "check_array_comparison_v1/check.stdout.json":
        "1c10d6520dbdb078944ade5b4d47954a3e6852d2b2260e72913bf64ccf1f0e2b",
    "check.stdout.json": "2951c4ff65d54f1b62f426d5f2507d9b97f430bf8e016e693175ee8c04d8c31a",
    "execution_blocked_v1/terminal.json":
        "23ab21238c22268b069312068d65e0b117337e32dc125e250e82c8e3394df329",
}
LAYERS = (21, 22, 23)
EXPECTED_TESTS = 26
FLAGS = dict(interior.FLAGS)
COMMAND = (
    "cd /home/argustest/ace3-argus && "
    "PYTHONPATH=/home/argustest/ace3-argus /home/argustest/miniconda3/bin/python -B "
    "-m ace3.model.candidates.diagnose_q24_s16_l20_interval_l21_l23_execute_v2 --check"
)
EXECUTE_COMMAND = COMMAND.removesuffix("--check") + "--execute"
EXECUTOR = {
    "version": 2, "coordinate": 62, "position": 0, "history": [9707],
    "probe_count": 15, "native_layers": list(LAYERS),
    "maximum_native_layer_invocations": 45,
    "schedule": interior.METHOD,
    "baseline": "unchanged frozen control; not executed",
    "stop": "first mandatory failure per probe; BLOCKED or failed L20 precondition aborts batch",
    "output": "stdout evidence only; no state archive, successor, publication or admission",
}
CLAIM = (
    "Bounded non-admitting native-S16-RTZ Q24/P0 CPU software; independent normal Host "
    "Reviewer required. Signed int64 Q24 residual state is wider than FP16. Native G128 "
    "asymmetric packed INT4, GEMM nibble order, no qzero plus-one, FP16 scales/operator "
    "boundaries/KV are unchanged. Only 15 prepared strict-interior probes, not an exhaustive "
    "search or minimality/monotonicity proof. Historical failures and bounded acceptances "
    "are preserved. No native L0-L8 replay, RTL, GPU, FPGA, hardware, simulation, synthesis, "
    "PPA, strict-FP16-state W4A16, new-token or full-model PASS."
)


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 2, "interface": ["--check", "--execute"],
        "focused_tests": EXPECTED_TESTS, "executor": EXECUTOR,
        "input": str(INPUT.relative_to(ROOT)), "input_sha256": PINS,
        "policy_id": prepared.gates.POLICY_ID, "local_gate": upstream.producer.LOCAL_GATE,
        "excess_budget": "1/8", "reference_policy": upstream.suffix.REFERENCE_POLICY,
        "reproduction": COMMAND, "execution": EXECUTE_COMMAND, **FLAGS,
    }
    for key, value in expected.items():
        equal(document[key], value, f"interior executor contract mismatch: {key}")


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "interior executor source/contract origin mismatch")
    records = interior.origins()
    for name in (MODULE, TEST_MODULE, interior.MODULE, interior.TEST_MODULE, upstream.LEGACY_TEST):
        require(name in records and records[name]["path"]
                == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"required repository origin mismatch: {name}")
    return records


def check_frozen(document, endpoints, terminal, parent):
    probes = interior.schedule(endpoints["analysis"], parent)
    expected = {
        "diagnostic_id": interior.ID, "status": "INTERIOR_SCAN_PREFLIGHT_VALIDATED",
        "command": interior.COMMAND, "method": interior.METHOD,
        "probe_count": 15, "probes": probes, "endpoint_exclusion_proven": True,
        "excluded_executed_endpoints": [row["Q24_integer"] for row in endpoints["candidate_set"]],
        "native_layers": [], "native_search_executed": False,
        "minimality_proven": False, "search_exhausted": False, **FLAGS,
        "historical_endpoint_execution": {
            "status": endpoints["status"], "terminal_status": terminal["status"],
            "native_layers": endpoints["native_layers"], "native_layer_invocations": 2,
            "first_failures": [row["first_failure"] for row in endpoints["candidate_set"]],
            "evidence": {name: retained.record(interior.INPUT / name) for name in interior.PINS},
        },
    }
    for key in ("analysis", "parent_binding", "baseline_control", "original_L20_reference",
                "lineage", "reference_policy", "historical_preparation",
                "historical_sensitivity", "historical_v1"):
        expected[key] = endpoints[key]
    for key, value in expected.items():
        equal(document[key], value, f"frozen interior mismatch: {key}")


def check_prior(failed, blocked, document):
    equal(failed["status"], "BLOCKED", "historical failed check relabeled")
    equal(failed["error"], "ValueError: focused tests failed, skipped or incomplete",
          "historical failed check changed")
    for key, value in {"collected": 22, "executed": 22, "failures": 0, "errors": 1,
                       "skipped": 0}.items():
        equal(failed["validation"][key], value, f"historical failed validation changed: {key}")
    for key, value in {
        "status": "BLOCKED", "candidate_count": 15, "native_layers": [],
        "native_executor_invocations": 0, "native_search_executed": False,
        "source_matches_frozen_preflight": True, "historical_artifacts_unchanged": True,
        "frozen_preflight_status": document["status"], **FLAGS,
    }.items():
        equal(blocked[key], value, f"historical missing-executor record changed: {key}")
    equal(len(blocked["candidate_schedule"]), 15, "historical probe count mismatch")
    for old, probe in zip(blocked["candidate_schedule"], document["probes"]):
        for key, value in probe.items():
            equal(old[key], value, f"historical probe binding mismatch: {key}")
    for key in ("analysis", "parent_binding", "baseline_control", "original_L20_reference",
                "lineage", "reference_policy", "method", "historical_endpoint_execution",
                "historical_preparation", "historical_sensitivity", "historical_v1"):
        equal(blocked["frozen_" + key], document[key], f"historical frozen mismatch: {key}")


def authenticate():
    with v1.no_execution():
        inputs, endpoints, terminal, parent = interior.authenticate()
        documents = {name: v1.pinned(inputs, INPUT / name, digest) for name, digest in PINS.items()}
        document, failed, blocked = (documents[name] for name in PINS)
        interior.check_contract(json.loads(interior.CONTRACT.read_text()))
        check_frozen(document, endpoints, terminal, parent)
        check_prior(failed, blocked, document)
        validation = document["validation"]
        upstream.check_validation(validation, interior.EXPECTED_TESTS)
        equal(validation["command"], interior.COMMAND, "interior check command mismatch")
        equal(validation["contract"], retained.record(interior.CONTRACT), "interior contract changed")
        equal(sorted(validation["compiled"], key=lambda row: row["path"]),
              sorted(validation["origins"].values(), key=lambda row: row["path"]),
              "interior compiled closure mismatch")
        current = origins()
        for name, record in validation["origins"].items():
            equal(current[name], record, f"interior source origin changed: {name}")
        for record in (*document["authenticated_inputs"], *validation["compiled"],
                       validation["contract"], blocked["source"], blocked["contract"],
                       blocked["frozen_preflight"], *blocked["historical_before"]):
            inputs.bind(record)
        freeze_record = retained.record(engine.margin.INPUT / "freeze.json")
        require(freeze_record in inputs.records.values(), "unbound original reference freeze")
        freeze = inputs.read(freeze_record)
        require(freeze["reference_extension"] in inputs.records.values(),
                "unbound original reference extension")
        extension = inputs.read(freeze["reference_extension"])
        equal(extension["layers"]["20"], document["original_L20_reference"],
              "L20 original reference substitution")
        for layer in (20, *LAYERS):
            for key in ("fp16", "binary64"):
                require(extension["layers"][str(layer)][key] in inputs.records.values(),
                        "unbound original layer reference")
        require(extension["checkpoint"] in inputs.records.values(), "unbound model operands")
    return inputs, document, parent, extension


def accounting(result):
    layers = result["native_layers"]
    require(all(type(layer) is int and layer in LAYERS for layer in layers)
            and len(layers) <= EXECUTOR["maximum_native_layer_invocations"],
            "invalid interior native accounting")
    result.update(native_invocations=len(layers), native_layer_invocations=len(layers),
                  native_L9_plus_invocations=len(layers), native_search_executed=bool(layers))
    for layer in LAYERS:
        result[f"native_L{layer}_invocations"] = layers.count(layer)


def execute_probes(result, parent, analysis, trajectory20, reference20, run_layer):
    equal(result["probes"], interior.schedule(analysis, parent), "authenticated probe set changed")
    baseline = v1.state_binding(parent)
    result["status"] = "BLOCKED"
    try:
        for row in result["probes"]:
            state = prepared.change_coordinate(parent, row["Q24_integer"])
            equal(v1.state_binding(state), row["state_binding"], "interior state substitution")
            row.update(status="BLOCKED", layers=[])
            start = len(result["native_layers"])
            try:
                gate = upstream.parent_gate(state, trajectory20, reference20)
                row["L20_global_gate"] = gate
                if gate["status"] != "PASS":
                    require(gate["status"] == "FAIL", "L20 precondition blocked")
                    row.update(status="FAIL", first_failure={"node": [20, 0, 18],
                                                            "gate": "binary64_v1"})
                    result.update(status="INTERIOR_GATE_FAIL", reason="L20_GLOBAL_GATE_BOUNDARY")
                    return
                for layer in LAYERS:
                    before = v1.state_binding(state)
                    entry = {"layer": layer, "position": 0, "input_state_binding": before,
                             "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
                             "status": "BLOCKED", "reports": []}
                    row["layers"].append(entry)
                    invocation = len(result["native_layers"])
                    output = run_layer(layer, state, entry)
                    equal(result["native_layers"][invocation:], [layer],
                          "native dispatch accounting mismatch")
                    equal(v1.state_binding(state), before, "native driver mutated its input state")
                    endpoint.check_layer_result(layer, entry, output)
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
                row["native_L0_L8_invocations"] = row["rtl_invocations"] = 0
                row["native_gate_outcomes"] = [entry["reports"] for entry in row["layers"]]
                accounting(result)
        result["status"] = ("BOUNDED_INTERIOR_PROBES_PASS"
                            if all(row["status"] == "PASS" for row in result["probes"])
                            else "INTERIOR_GATE_FAIL")
    finally:
        equal(v1.state_binding(parent), baseline, "frozen parent was mutated")
        accounting(result)
        result["sampled_set_complete"] = all(row["status"] in ("PASS", "FAIL")
                                            for row in result["probes"])
        result["first_passing_probe_index"] = next(
            (row["probe_index"] for row in result["probes"] if row["status"] == "PASS"), None)


def check_test_result(outcome, collected):
    equal(collected, EXPECTED_TESTS, "focused test collection mismatch")
    require(outcome.testsRun == EXPECTED_TESTS and outcome.wasSuccessful() and not outcome.skipped,
            "focused tests failed, skipped or incomplete")


def validate(result):
    context = upstream.producer.context()
    tests = importlib.import_module(TEST_MODULE)
    for name in (interior.TEST_MODULE, endpoint.TEST_MODULE, v1.TEST_MODULE,
                 prepared.TEST_MODULE, upstream.TEST_MODULE, upstream.LEGACY_TEST):
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


def checked_execution(result, execute_requested):
    validate(result)
    inputs, document, parent, extension = authenticate()
    for record in (*result["validation"]["origins"].values(), result["validation"]["contract"]):
        inputs.bind(record)
    for key in ("probes", "analysis", "parent_binding", "baseline_control", "original_L20_reference",
                "lineage", "reference_policy", "excluded_executed_endpoints",
                "endpoint_exclusion_proven", "historical_endpoint_execution",
                "historical_preparation", "historical_sensitivity", "historical_v1"):
        result[key] = copy.deepcopy(document[key])
    result.update(
        status="INTERIOR_EXECUTOR_READY", executor=EXECUTOR, probe_count=15,
        historical_interior={
            "preflight_status": document["status"], "failed_check_status": "BLOCKED",
            "failed_check_errors": 1, "missing_executor_status": "BLOCKED",
            "evidence": {name: retained.record(INPUT / name) for name in PINS},
        },
        sampled_set_complete=False, first_passing_probe_index=None,
        authenticated_inputs=list(inputs.records.values()))
    endpoint.check_inputs(inputs, result["validation"])
    if not execute_requested:
        return
    result["status"] = "BLOCKED"
    try:
        trajectory20 = inputs.archive(extension["layers"]["20"]["fp16"])
        reference20 = engine.global_reference(inputs, extension["layers"]["20"])
        with engine.safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
            def run_layer(layer, state, entry):
                # The existing guard authenticates one dispatch; the batch accounts all 45.
                audit = []
                try:
                    return endpoint.evaluate_layer(inputs, model, extension, layer, state, audit, entry)
                finally:
                    result["native_layers"].extend(audit)

            execute_probes(result, parent, document["analysis"], trajectory20, reference20, run_layer)
    finally:
        accounting(result)
        result["authenticated_inputs"] = list(inputs.records.values())
        endpoint.check_inputs(inputs, result["validation"])


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
    return 0 if result["status"] in ("INTERIOR_EXECUTOR_READY", "BOUNDED_INTERIOR_PROBES_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
