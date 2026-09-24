"""Read-only L11/P0 binding preflight. Only --check is available; no execution."""

import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l11_coordinate62_producer_cone_v1 as legacy
from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v2 as retained


ROOT = Path("/home/argustest/ace3-argus")
ID = "ace3-q24-s16-l11-coordinate62-producer-cone-v2"
MODULE = "ace3.model.candidates.preflight_q24_s16_l11_coordinate62_producer_cone_v2"
TEST_MODULE = "tests.test_q24_s16_l11_coordinate62_producer_cone_v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l11_coordinate62_producer_cone_v2.json"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check")
EXPECTED_TESTS = 20
FAILED = ROOT / "build/q24_s16_l11_coordinate62_producer_cone_4340d090a3b2_attempt001"
HISTORY = {
    "ace3/model/candidates/diagnose_q24_s16_l11_coordinate62_producer_cone_v1.py":
        "837cba394b032e3fdadbcdc5a8b9d6d9022b4f8c46d008f37d4c817ace522b38",
    "ace3/contracts/candidates/q24_s16_l11_coordinate62_producer_cone_v1.json":
        "7a56450c59110ee1b4a307f68deb5760eae7b5b1a350617c6e2538c49ba0b0ec",
    "tests/test_q24_s16_l11_coordinate62_producer_cone_v1.py":
        "73663ee78fe24180f8897077b331978e96f6c20db06547ec83e3cf10e27a8d2c",
    str((FAILED / "validation.json").relative_to(ROOT)):
        "151c81f9d367c671787bc5dcc7ae2d67fce9408fa41f10fc4684fc4860832e7a",
    str((FAILED / "unittest.log").relative_to(ROOT)):
        "6836dacbc9e92d233c9f220701cfb5fc994e72170a21cac4ab6dbcb256208bb6",
    str((FAILED / "compiled_0.pyc").relative_to(ROOT)):
        "166ef5550fed21b8e18215c85bfda2d11978aa8cc78ecc58e2d781276562e709",
    str((FAILED / "compiled_1.pyc").relative_to(ROOT)):
        "261e84214f3cec135a3ece006cef0d4a566b8da83f0f5271110e288f0b6844f8",
}
PRESERVED_FIELDS = (
    "policy_id", "position", "coordinate", "thresholds", "mapping", "branch_definitions",
    "controls", "classification", "claim_boundary", "candidate_admitted", "policy_adopted",
    "successor_published", "scientific_result_claim", "normal_host_review", "rtl_invocations",
)
require, record = retained.require, retained.record
upstream, prior, native, paired = retained.upstream, retained.prior, retained.native, retained.paired


def history_records():
    items = [record(ROOT / path) for path in HISTORY]
    for item, digest in zip(items, HISTORY.values(), strict=True):
        require(item["sha256"] == digest, f"L11 v1 history changed: {item['path']}")
    failed = json.loads((FAILED / "validation.json").read_text())
    require(failed["collected"] == 20 and failed["executed"] == 0
            and failed["errors"] == 1 and failed["failures"] == failed["skipped"] == 0,
            "v1 terminal authentication failure changed")
    require(not (FAILED / "result.json").exists() and not (FAILED / "command.log").exists(),
            "v1 failed diagnostic was resumed")
    return items


def check_contract(contract):
    history_records()
    original = json.loads(legacy.CONTRACT.read_text())
    for key in PRESERVED_FIELDS:
        require(type(contract[key]) is type(original[key]) and contract[key] == original[key],
                f"L11 scientific contract changed: {key}")
    expected = {
        "diagnostic_id": ID, "cwd": str(ROOT), "command": COMMAND, "interface": ["--check"],
        "focused_tests": EXPECTED_TESTS, "native_layer_invocations": 0,
        "native_diagnostic_invocations": 0, "accepted_prefix_replay": False,
        "execution_authorized": False, "history_sha256": HISTORY,
        "binding_provider": retained.MODULE,
        "reviewed_result_sha256": retained.REVIEWED_SHA,
        "retained_freeze_sha256": prior.FREEZE_SHA,
        "source_updates": retained.coordinate.v2.SOURCE_UPDATES,
    }
    require(set(contract) == set(PRESERVED_FIELDS) | set(expected) | {
        "authentication", "reproduction", "execution_boundary"},
        "unexpected L11 v2 contract fields")
    for key, value in expected.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"L11 v2 contract mismatch: {key}")
    retained.check_contract(json.loads(retained.CONTRACT.read_text()))
    retained.coordinate.check_contract(json.loads(retained.coordinate.CONTRACT.read_text()))


def check_l11_lineage(data):
    bound = data["bound"]
    bind = lambda item: upstream.bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    freeze_record = data["authentication"]["retained_freeze"]
    freeze = read(freeze_record)
    accepted = freeze["state_lineage"]["prior_residual"]
    legacy.entry.check_review(accepted["review"])
    parent = read(accepted["receipt"])
    native.validate_parent_scope(parent)
    require(parent == accepted["parent"] and accepted["prior_layer_kv_consumed"] is False,
            "accepted L8 receipt changed")
    accepted_freeze, accepted_result = read(parent["arithmetic_lineage"]), read(accepted["result"])
    require(accepted_result["status"] == "PASS" and accepted_result["candidate_admitted"] is True
            and accepted_result["first_failure"] is None and accepted_result["rtl_invocations"] == 0
            and accepted_result["policy_id"] == prior.gates.POLICY_ID
            and accepted_result["scope"] == accepted_freeze["contract"]["scope"]
            and accepted_result["scope"]["layers"] == list(range(3, 9)),
            "accepted L8 result scope mismatch")
    final = accepted_result["layers"][-1]
    require(final["layer"] == 8 and final["output_parent"] == parent["state"]
            and final["kv_state"] == parent["kv"]
            and final["input_parent"] == parent["state_lineage_parent"]
            and final["reports"] == parent["numerical_report"], "accepted L8 endpoint mismatch")
    arrays, incoming = archive(final["actual_stages"]), archive(parent["state_lineage_parent"])
    for stage in range(19):
        prior.retained.check_stage_state(stage, arrays, incoming)
    paired.same_arrays(archive(parent["kv"]),
                       {key: arrays["output_cache_" + key] for key in ("k", "v")})
    actual = archive(parent["state"])
    prior.retained.verify_parent(actual, prior.retained.state_from(arrays, "output", "stage18"))
    prior.retained.verify_parent(actual, data["layers"][9]["parent"])
    reports = read(parent["numerical_report"])
    require(len(reports) == 19 and all(
        report["status"] == "PASS" and report["node"] == [8, 0, stage]
        for stage, report in enumerate(reports)), "accepted L8 stage evidence changed")

    result = read(record(prior.INPUT / "result.json"))
    entries = {item["layer"]: item for item in result["layers"]}
    l10, l11 = entries[10], entries[11]
    require(l10["output_parent"] == l11["input_parent"], "L10-output/L11-input binding mismatch")
    paired.same_arrays(data["layers"][10]["output"], data["layers"][11]["parent"])
    paired.same_arrays(archive(l11["input_parent"]), data["layers"][11]["parent"])
    retained.coordinate.check_reference_chain(data["extension"])
    for layer in (11, 12, 13):
        refs = data["extension"]["layers"]
        require(refs[str(layer)]["input_binary64"] == refs[str(layer - 1)]["binary64"],
                f"L{layer} original reference predecessor mismatch")
    with retained.legacy.safe_open(str(bind(data["extension"]["checkpoint"])), framework="numpy") as model:
        tensors = {key: legacy.np.ascontiguousarray(model.get_tensor(key))
                   for key in prior.local.tensor_shapes(11)}
    prior.local.authenticate_tensors(tensors, data["extension"]["layers"]["11"]["canonical"], 11)
    data["layers"][11]["tensors"] = tensors
    return {
        "L10_output": l10["output_parent"], "L11_input": l11["input_parent"],
        "L10_receipt": record(prior.INPUT / "layer10/software_parent.json"),
        "L11_receipt": record(prior.INPUT / "layer11/software_parent.json"),
        "L11_original_input": data["extension"]["layers"]["11"]["input_binary64"],
        "L11_original_output": data["extension"]["layers"]["11"]["binary64"],
        "accepted_L8_review": accepted["review"],
    }


def authenticate():
    data = retained.authenticate()
    data["L11_binding"] = check_l11_lineage(data)
    return data


def source_context():
    require(Path.cwd() == ROOT and os.environ.get("PYTHONPATH") == str(ROOT)
            and Path(sys.executable) == prior.PYTHON and sys.dont_write_bytecode
            and not sys.flags.optimize, "use the published repository-bound command")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
            and CONTRACT.resolve() == ROOT / "ace3/contracts/candidates/q24_s16_l11_coordinate62_producer_cone_v2.json",
            "L11 v2 source/contract origin mismatch")
    for name in (MODULE, TEST_MODULE, legacy.ANCESTOR_TEST):
        importlib.import_module(name)
    origins = retained.source_context()
    for name in (MODULE, TEST_MODULE, legacy.ANCESTOR_TEST):
        require(origins[name]["path"] == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"L11 v2 module origin mismatch: {name}")
    return origins


def dispatch_guards(stack):
    return [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
        f"dispatch forbidden in L11 v2 preflight: {name}"))) for owner, name in (
            (legacy, "diagnose"), (legacy, "execute_layer"),
            (upstream, "execute_layer"), (native, "stages"), (native, "execute_layers"),
            (subprocess, "Popen"), (os, "system"))]


def validate():
    origins = source_context()
    contract = record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for item in origins.values():
        if "path" in item:
            path = Path(item["path"])
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            compiled.append(item)
    tests = sys.modules[TEST_MODULE]
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with ExitStack() as stack:
        guards = dispatch_guards(stack)
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not any(guard.called for guard in guards), "preflight dispatch attempted")
    require(count == EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    data = tests.BindingTests.data
    retained.coordinate.recheck(data["bound"].values())
    legacy.entry.check_review(data["L11_binding"]["accepted_L8_review"])
    history = history_records()
    require(source_context() == origins and record(CONTRACT) == contract,
            "source/contract changed during validation")
    return {
        "status": "VALIDATED_CHECK_ONLY", "diagnostic_id": ID,
        "command": COMMAND, "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": os.environ["PYTHONPATH"], "origins": origins, "compiled": compiled,
        "contract": contract, "preserved_history": history,
        "authentication": data["authentication"], "input_bindings": list(data["bound"].values()),
        "L11_binding": data["L11_binding"], "reviewed_result": data["reviewed_result"],
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "native_diagnostic_invocations": 0, "native_layer_invocations": 0, "rtl_invocations": 0,
        "source_operand_state_KV_lineage_checks": "AUTHENTICATED_RETAINED_ONLY",
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "execution_authorized": False, "normal_host_review": "REQUIRED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
