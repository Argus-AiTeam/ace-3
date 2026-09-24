"""Check-only L12 coordinate-62 producer-cone surface on reviewed v3 evidence.

The adjacent contract publishes the only command. Producer arithmetic is reused
unchanged from v1; authentication reads retained evidence without native dispatch.
This surface neither executes nor authorizes a counterfactual suffix.
"""

import argparse
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v1 as legacy
from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v3 as coordinate


ROOT = coordinate.ROOT
ID = "ace3-q24-s16-l12-coordinate62-producer-cone-v2"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l12_coordinate62_producer_cone_v2"
TEST_MODULE = "tests.test_q24_s16_l12_coordinate62_producer_cone_v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v2.json"
REVIEWED = ROOT / "build/q24_s16_l13_l12_parent_coordinate_cut_v3_1d567e59cfac_attempt001"
REVIEWED_SHA = "4073e5da85af77e0770a399e724c64f8326abb1a636f21554f4966b50b7299a9"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check")
EXPECTED_TESTS = 26
HISTORY = {
    "ace3/model/candidates/diagnose_q24_s16_l12_coordinate62_producer_cone_v1.py":
        "cd5ab1a02301c2477b3ca860962a9f35bc68ed21ff4323c75d94c9774e5c02d7",
    "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v1.json":
        "e958abc05369640a91e63f73c9ce2298c3401b482dccd97f5603e59b2f4468b1",
    "tests/test_q24_s16_l12_coordinate62_producer_cone_v1.py":
        "92cba95112c6633a99e1cc70d16093e016b912adff82f546faca6d68b65c5e47",
}
PRESERVED_FIELDS = (
    "policy_id", "reviewed_upstream_sha256", "candidate_admitted", "policy_adopted",
    "successor_published", "rtl_invocations", "normal_host_review", "controls",
    "branch_definitions", "mapping", "thresholds", "classification", "claim_boundary",
)
require, record = coordinate.require, coordinate.record
upstream, prior, native, paired = coordinate.upstream, coordinate.prior, coordinate.native, coordinate.paired
plan, compose, frozen_parent = legacy.plan, legacy.compose, legacy.frozen_parent
classify, original_branches = legacy.classify, legacy.original_branches


def history_records():
    records = [record(ROOT / relative) for relative in HISTORY]
    for item, digest in zip(records, HISTORY.values(), strict=True):
        require(item["sha256"] == digest, f"L12 v1 history changed: {item['path']}")
    return records


def check_contract(contract):
    original = json.loads(legacy.CONTRACT.read_text())
    for key in PRESERVED_FIELDS:
        require(type(contract[key]) is type(original[key]) and contract[key] == original[key],
                f"v1 scientific contract changed: {key}")
    expected = {
        "diagnostic_id": ID, "cwd": str(ROOT), "command": COMMAND,
        "interface": ["--check"], "expected_tests": EXPECTED_TESTS,
        "reviewed_result_sha256": REVIEWED_SHA, "reviewed_result": str(REVIEWED / "result.json"),
        "reviewed_task": "1d567e59cfac", "history_sha256": HISTORY,
        "retained_freeze_sha256": prior.FREEZE_SHA,
        "source_updates": coordinate.v2.SOURCE_UPDATES,
        "inputs": [str(REVIEWED.relative_to(ROOT)),
                   str(coordinate.legacy.REVIEWED.relative_to(ROOT)),
                   str(prior.INPUT.relative_to(ROOT))],
        "native_layer_invocations": 0, "scientific_result_claim": False,
        "accepted_prefix_replay": False,
    }
    require(set(contract) == set(PRESERVED_FIELDS) | set(expected) | {
        "authentication", "execution_boundary", "reproduction", "gate_scope"},
        "unexpected v2 contract fields")
    for key, value in expected.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"v2 contract mismatch: {key}")


def check_reviewed(reviewed):
    require(reviewed["diagnostic_id"] == coordinate.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["control_count"] == 954 and reviewed["native_layer_invocations"] == 954
            and reviewed["node"] == [13, 0, 18] and reviewed["index"] == 62
            and reviewed["sufficient_single_coordinates"] == [62]
            and reviewed["single_coordinate_662f_rescues"] == [62]
            and reviewed["retained_status"] == "FAIL"
            and reviewed["native_retained_bitwise_reproduction"] is True
            and reviewed["reviewed_L12_cut_reproduced"] is True
            and reviewed["unchanged_baseline_gates"] is True
            and reviewed["original_global_reference_unchanged"] is True
            and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
            and reviewed["normal_host_review"] == "REQUIRED"
            and reviewed["native_L0_L8_invocations"] == 0 and reviewed["rtl_invocations"] == 0
            and all(reviewed[key] is False for key in (
                "candidate_admitted", "policy_adopted", "successor_published",
                "unique_upstream_producer_attributed")),
            "reviewed v3 coordinate scope mismatch")
    for label, bits, accepted in (("actual", "6630", False), ("mapped", "662f", True)):
        endpoint = reviewed["selected_coordinate"][label]
        require(endpoint["actual_fp16_bits"] == bits and endpoint["accepted"] is accepted
                and endpoint["excess_budget"] == "1/8"
                and endpoint["reference_binary64_hex"] == "0x1.8bd7b2092532cp+10",
                "reviewed coordinate endpoint changed")


def authenticate():
    authentication = coordinate.v2.authenticate()
    data = coordinate.load_data(authentication)
    bound = data["bound"]
    bind = lambda item: upstream.bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    result_record = record(REVIEWED / "result.json")
    require(result_record["sha256"] == REVIEWED_SHA, "wrong reviewed v3 result")
    reviewed = read(result_record)
    check_reviewed(reviewed)
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    for item in reviewed["origins_after_execution"].values():
        if "path" in item:
            bind(item)

    def artifact(name):
        matches = [item for item in reviewed["artifacts"]
                   if item["path"] == str(REVIEWED / name)]
        require(len(matches) == 1, f"missing or duplicate reviewed endpoint: {name}")
        return archive(matches[0])

    for label, expected in (("actual", data["actual_arrays"]), ("mapped", data["mapped_arrays"])):
        observed = artifact(label + "_native.npz")
        paired.same_arrays(observed, expected)
        data[label + "_reviewed"] = observed
    data["single_062_reviewed"] = artifact("single_062_native.npz")
    paired.same_arrays(artifact("actual_L12_parent.npz"), data["actual"])
    paired.same_arrays(artifact("mapped_L12_parent.npz"), data["mapped"])
    freeze_record = authentication["retained_freeze"]
    freeze = read(freeze_record)
    extension = read(freeze["reference_extension"])
    coordinate.check_reference_chain(extension)
    retained = read(record(prior.INPUT / "result.json"))
    entries = {entry["layer"]: entry for entry in retained["layers"]}
    expected = freeze["state_lineage"]["prior_residual"]["parent"]["state"]
    require(freeze["state_lineage"]["prior_kv"] == "own empty P0"
            and freeze["state_lineage"]["prior_residual"]["prior_layer_kv_consumed"] is False,
            "wrong retained KV boundary")
    layers = {}
    for layer in range(9, 13):
        entry, reference = entries[layer], extension["layers"][str(layer)]
        require(entry["input_parent"] == expected, f"spliced actual parent at L{layer}")
        parent, arrays = archive(entry["input_parent"]), archive(entry["actual_stages"])
        for stage in range(19):
            prior.retained.check_stage_state(stage, arrays, parent)
        reports = read(entry["reports"])
        require(len(reports) == 19 and all(
            report["node"] == [layer, 0, stage] and report["status"] == "PASS"
            and report["policy_id"] == prior.gates.POLICY_ID
            and report["residual_state_lineage"] == report["kv_lineage"] == "PASS"
            for stage, report in enumerate(reports)), "retained prefix gates changed")
        output = archive(entry["output_parent"])
        prior.retained.verify_parent(output, prior.retained.state_from(arrays, "output", "stage18"))
        paired.same_arrays(archive(entry["kv_state"]),
                           {key: arrays["output_cache_" + key] for key in ("k", "v")})
        receipt = read(record(prior.INPUT / f"layer{layer:02d}/software_parent.json"))
        require(receipt["state"] == entry["output_parent"]
                and receipt["state_lineage_parent"] == entry["input_parent"]
                and receipt["numerical_report"] == entry["reports"]
                and receipt["kv"] == entry["kv_state"] and receipt["arithmetic_lineage"] == freeze_record
                and receipt["candidate_id"] == freeze["contract"]["candidate_id"]
                and receipt["state_id"] == freeze["contract"]["state_id"]
                and receipt["policy_id"] == prior.gates.POLICY_ID
                and receipt["evidence_kind"] == "cpu_software_q24"
                and receipt["position"] == 0 and receipt["history"] == [9707]
                and receipt["next_layer"] == layer + 1 and receipt["rtl_admissible"] is False
                and receipt["normal_host_review"] == "REQUIRED", "retained receipt mismatch")
        layers[layer] = {
            "parent": parent, "arrays": arrays, "output": output, "reports": reports,
            "locals": archive(entry["local_references"]),
            "reference": legacy.np.load(bind(reference["binary64"]), allow_pickle=False),
            "trajectory": archive(reference["fp16"]),
        }
        expected = entry["output_parent"]
    require(entries[13]["input_parent"] == expected, "spliced L13 parent")
    with legacy.safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: legacy.np.ascontiguousarray(model.get_tensor(key))
                   for key in prior.local.tensor_shapes(12)}
    prior.local.authenticate_tensors(tensors, extension["layers"]["12"]["canonical"], 12)
    data.update(layers=layers, tensors12=tensors, extension=extension, reviewed_result=result_record)
    return data


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "L12 v2 source origin mismatch")
    require(CONTRACT.resolve() == ROOT / "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v2.json",
            "L12 v2 contract origin mismatch")
    importlib.import_module(TEST_MODULE)
    origins = coordinate.origins()
    require(MODULE in origins and TEST_MODULE in origins, "missing L12 v2 candidate/test origins")
    return origins


def validate():
    origins = source_context()
    compiled = []
    for name in (MODULE, TEST_MODULE):
        path = Path(origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(origins[name])
    history = history_records()
    contract_record = record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    coordinate.check_contract(json.loads(coordinate.CONTRACT.read_text()))
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(module, name, side_effect=AssertionError(
            "native dispatch forbidden in L12 v2 check"))) for module, name in (
                (upstream, "execute_layer"), (native, "stages"),
                (coordinate, "execute_layer"), (legacy, "diagnose"))]
        data = authenticate()
        tests = sys.modules[TEST_MODULE]
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == EXPECTED_TESTS == tests.EXPECTED_TESTS, "focused test count mismatch")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not any(guard.called for guard in guards), "native dispatch attempted")
    require(result.testsRun == count and result.wasSuccessful() and not result.skipped,
            "focused validation failed")
    coordinate.recheck(data["bound"].values())
    require(source_context() == origins and record(CONTRACT) == contract_record,
            "source/contract changed during validation")
    return {
        "status": "VALIDATED_CHECK_ONLY", "diagnostic_id": ID,
        "command": COMMAND, "cwd": str(ROOT), "executable": sys.executable,
        "PYTHONPATH": str(ROOT), "origins": origins, "compiled": compiled,
        "contract": contract_record, "preserved_history": history,
        "reviewed_result": data["reviewed_result"],
        "reviewed_upstream": data["authentication"]["reviewed_parent_cut"],
        "retained_freeze": data["authentication"]["retained_freeze"],
        "authenticated_binding_count": len(data["bound"]),
        "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "source_operand_state_KV_lineage_checks": "AUTHENTICATED_RETAINED_ONLY",
        "original_global_reference_unchanged": True, "unchanged_baseline_gates": True,
        "native_layer_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    importlib.import_module(MODULE)
    print(json.dumps(validate(), sort_keys=True))


if __name__ == "__main__":
    main()
