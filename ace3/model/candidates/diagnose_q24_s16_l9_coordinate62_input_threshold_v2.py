"""Versioned L9/P0 input-threshold surface; no scientific execution in --check.

Run the adjacent contract's repository-bound command. --out accepts only a
fresh direct build/q24_s16_l9_coordinate62_input_threshold_v2_<name> directory
and validates once before the unchanged, separately authorized L9-L13 sweep.
Preflight and dispatch-bound validation publish to separate exclusive files.
Check JSON includes unchanged thresholds and authenticated original-input
global-reference bindings, without recomputing the scientific trajectory.
Accepted L0-L8 and selected L9/L10 evidence remain read-only. Independent Host
review is required; wide Q24 software is not strict-FP16-state W4A16.
"""

import argparse
from contextlib import redirect_stderr
import importlib
import json
from pathlib import Path
import sys
import traceback
from types import FunctionType
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l10_coordinate62_producer_cone_v2 as producer
from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_input_threshold_v1 as legacy


entry = legacy.entry
require = entry.require
ROOT = legacy.ROOT
ID = "ace3-q24-s16-l9-coordinate62-input-threshold-v2"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_input_threshold_v2"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_input_threshold_v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l9_coordinate62_input_threshold_v2.json"
REVIEWED_L9 = producer.SELECTED
REVIEWED_L10 = (
    "q24_s16_l10_coordinate62_producer_cone_ba86857338f4_attempt001",
    "e932505aca45f1ee7bbbeb18b14dfba6110b872ff64a4822201703edd4860492",
    producer.ID,
)
OUTPUT_PREFIX = "q24_s16_l9_coordinate62_input_threshold_v2_"
LAYERS = legacy.LAYERS
plan = legacy.plan
input_parent = legacy.input_parent
threshold_intervals = legacy.threshold_intervals


def authenticate_l10(data):
    result, reviewed = producer.read_result(REVIEWED_L10, data["bound"])
    expected = {
        "diagnostic_id": producer.ID, "status": "DIAGNOSED", "index": 62,
        "policy_id": entry.prior.gates.POLICY_ID,
        "native_retained_bitwise_reproduction": True,
        "original_L10_S18_bitwise_reproduction": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "accepted_L0_L8_execution": False, "native_L0_L8_invocations": 0,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "control_count": 13,
    }
    for key, value in expected.items():
        require(type(reviewed[key]) is type(value) and reviewed[key] == value,
                f"reviewed L10 v2 metadata mismatch: {key}")
    require(reviewed["selected_L9_evidence"] == data["selected_L9_evidence"],
            "reviewed L10 selected L9 linkage mismatch")
    bind = lambda item: entry.upstream.bind_input(item, data["bound"])
    origins = reviewed["origins_after_execution"]
    for module in (entry, producer, legacy):
        name = module.MODULE
        require(name in origins and origins[name]["path"] == str(Path(module.__file__).resolve()),
                f"reviewed L10 source origin mismatch: {name}")
        bind(origins[name])
    for item in origins.values():
        if "path" in item:
            bind(item)
    require(reviewed["contract"] == entry.record(producer.CONTRACT),
            "reviewed L10 v2 contract mismatch")
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [
            reviewed["validation"], reviewed["contract"]]:
        bind(item)
    policy = legacy.authenticate_policy(reviewed, data["bound"])
    require(policy["contract"] == reviewed["contract"], "L10 validation contract mismatch")
    dispatches = [10]
    for label, _ in producer.plan():
        if label == "inherited_native":
            dispatches.append(10)
        dispatches.extend((11, 12, 13))
    require(reviewed["native_layer_dispatches"] == reviewed["native_stage_dispatches"] == dispatches,
            "reviewed L10 dispatch accounting mismatch")

    def artifact(name):
        path = str(ROOT / "build" / REVIEWED_L10[0] / name)
        matches = [item for item in reviewed["artifacts"] if item["path"] == path]
        require(len(matches) == 1, f"missing or duplicate L10 artifact: {name}")
        return matches[0]

    layer = data["layers"][10]
    arrays = entry.native.load_state(artifact("actual_L10.npz"), bind)
    locals_ = entry.native.load_state(artifact("actual_L10_local_references.npz"), bind)
    entry.paired.same_arrays(arrays, layer["arrays"])
    entry.paired.same_arrays(locals_, layer["locals"])
    entry.coordinate.check_reports(
        json.loads(bind(artifact("actual_L10_gates.json")).read_text()), layer["reports"])
    entry.prior.retained.verify_parent(
        entry.native.load_state(artifact("actual_L10_parent.npz"), bind),
        entry.prior.retained.state_from(arrays, "output", "stage18"))
    return {"result": result, "read_only": True, "policy_binding": policy}


def authenticate():
    data = producer.authenticate()
    selected = data["selected_L9_evidence"]
    require(selected["result"]["sha256"] == REVIEWED_L9[1], "selected L9 pin mismatch")
    data["input_threshold_evidence"] = [
        {"result": selected["result"], "read_only": True,
         "policy_binding": selected["policy_binding"]},
        authenticate_l10(data),
    ]
    return data


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "input-threshold v2 source origin mismatch")
    importlib.import_module(TEST_MODULE)
    origins = legacy.runtime.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing input-threshold v2 origins")
    return origins


def check_contract(contract):
    expected = {
        "diagnostic_id": ID, "policy_id": entry.prior.gates.POLICY_ID,
        "cwd": str(ROOT),
        "command": f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
                   f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check",
        "selected_L9": list(REVIEWED_L9), "reviewed_L10_v2": list(REVIEWED_L10),
        "output_prefix": OUTPUT_PREFIX, "fresh_exclusive_directory": True,
        "controls": plan(), "native_layers": list(LAYERS), "position": 0, "coordinate": 62,
        "max_native_layer_evaluations": len(plan()) * len(LAYERS),
        "native_layer_invocations": 0, "rtl_invocations": 0,
        "normal_host_review": "REQUIRED", "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
        "scientific_result_claim": False, "accepted_L0_L8_execution": False,
        "thresholds": json.loads(legacy.CONTRACT.read_text())["thresholds"],
    }
    for key, value in expected.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"input-threshold v2 contract mismatch: {key}")


def validation_evidence(contract, data):
    extension = data["extension"]
    references = {
        "original_specification": extension["original_specification"],
        "original_binary64_parent": extension["original_binary64_parent"],
        "layers": [
            {"layer": layer, "position": 0,
             "input_binary64": extension["layers"][str(layer)]["input_binary64"],
             "binary64": extension["layers"][str(layer)]["binary64"]}
            for layer in LAYERS
        ],
    }
    bindings = [references["original_specification"], references["original_binary64_parent"]]
    for row in references["layers"]:
        bindings.extend((row["input_binary64"], row["binary64"]))
    for binding in bindings:
        require(binding["path"] in data["bound"]
                and data["bound"][binding["path"]] == binding,
                f"unauthenticated original-input global-reference binding: {binding['path']}")
    predecessor = references["original_binary64_parent"]
    for row in references["layers"]:
        require(row["input_binary64"] == predecessor,
                f"original-input global-reference predecessor mismatch: L{row['layer']}")
        predecessor = row["binary64"]
    return {
        "thresholds": contract["thresholds"],
        "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True,
        "original_input_global_reference_bindings": references,
    }


def validate():
    origins = source_context()
    compiled = []
    for name in (MODULE, TEST_MODULE):
        path = Path(origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(origins[name])
    require(CONTRACT.resolve().is_relative_to(ROOT), "contract outside repository")
    contract_binding = entry.record(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    check_contract(contract)
    tests = sys.modules[TEST_MODULE]
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with patch.object(entry.upstream, "execute_layer",
                      side_effect=AssertionError("native dispatch forbidden during validation")) as dispatch, \
            patch.object(entry.native, "stages",
                         side_effect=AssertionError("native stages forbidden during validation")) as stages, \
            patch.object(Path, "mkdir",
                         side_effect=AssertionError("output creation forbidden during validation")) as mkdir, \
            patch.object(entry, "write",
                         side_effect=AssertionError("output write forbidden during validation")) as write:
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not any(mock.called for mock in (dispatch, stages, mkdir, write)),
                "native execution or output attempted during validation")
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    require(source_context() == origins and entry.record(CONTRACT) == contract_binding,
            "source/contract changed during validation")
    return {
        "status": "VALIDATED_SOFTWARE_ONLY", "diagnostic_id": ID,
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": str(ROOT),
        "origins": origins, "compiled": compiled, "contract": contract_binding,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "reviewed_inputs": tests.InputThresholdV2Tests.data["input_threshold_evidence"],
        "authentication_scope": "Full entry and selected L9/L10 v2 bindings, read-only",
        **validation_evidence(contract, tests.InputThresholdV2Tests.data),
        "native_layer_invocations": 0, "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED",
    }


def output_directory(path):
    require(not path.is_symlink(), "output must not be a symlink")
    out = path.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith(OUTPUT_PREFIX)
            and len(out.name) > len(OUTPUT_PREFIX), "output outside versioned bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def execute_sweep(out=None, log=None):
    # Rebind orchestration only; v1 arithmetic, reference and dispatch globals stay frozen.
    runner = FunctionType(legacy.execute_sweep.__code__, {
        **legacy.execute_sweep.__globals__, "authenticate": authenticate,
        "source_context": source_context, "ID": ID, "CONTRACT": CONTRACT,
    }, legacy.execute_sweep.__name__)
    return runner(out, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(validate()))
        return 0
    out = output_directory(args.out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({
            "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
            "PYTHONPATH": entry.os.environ.get("PYTHONPATH"),
            "PYTHONDONTWRITEBYTECODE": entry.os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "loadavg": entry.os.getloadavg(), "cpu_affinity": sorted(entry.os.sched_getaffinity(0)),
            "torch_threads_before": entry.torch.get_num_threads(),
        }) + "\n")
        log.flush()
        try:
            with redirect_stderr(log):
                validation = validate()
            entry.write(out / "preflight_validation.json", validation)
            native_stages = entry.native.stages
            native_records = []

            def guarded_stages(tensors, layer, parent, arrays):
                require(type(layer) is int and layer in LAYERS,
                        "native stage execution outside L9-L13")
                native_records.append({"layer": layer, "position": 0})
                return native_stages(tensors, layer, parent, arrays)

            with patch.object(entry.native, "stages", side_effect=guarded_stages):
                result = execute_sweep(out, log)
            require(native_records == result["native_execution_records"],
                    "native stage/dispatch accounting mismatch")
            result["native_stage_execution_records"] = native_records
            require(result["origins_after_execution"] == validation["origins"]
                    and result["contract"] == validation["contract"],
                    "source/contract changed after validation")
            validation["sweep_execution"] = {
                "native_L0_L8_invocations": result["native_L0_L8_invocations"],
                "rtl_invocations": result["rtl_invocations"],
                "native_layer_count": result["native_layer_count"],
                "native_execution_records": result["native_execution_records"],
                "native_stage_execution_records": native_records,
            }
            entry.write(out / "validation.json", validation)
            result["validation"] = entry.record(out / "validation.json")
            entry.write(out / "result.json", result)
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, AssertionError):
            traceback.print_exc(file=log)
            raise
        summary = {"status": result["status"], "result": str(out / "result.json")}
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
