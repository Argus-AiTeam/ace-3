"""Versioned L12 coordinate controls; --check never executes native controls.

The adjacent contract publishes separate check and future execution commands.
Execution is only the 954 isolated, non-admitting L13/P0 counterfactual controls.
The accepted v2 retained/current source closure is not a historical relabelling.
"""

import argparse
from contextlib import ExitStack
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v2 as v2


legacy, upstream, prior, native, paired = v2.legacy, v2.upstream, v2.prior, v2.native, v2.paired
ROOT = v2.ROOT
ID = "ace3-q24-s16-l13-l12-parent-coordinate-cut-v3"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l13_l12_parent_coordinate_cut_v3"
TEST_MODULE = "tests.test_q24_s16_l13_l12_parent_coordinate_cut_v3"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v3.json"
OUTPUT_PREFIX = "q24_s16_l13_l12_parent_coordinate_cut_v3_"
CONTEXT = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE}")
COMMAND = CONTEXT + " --check"
EXECUTION_COMMAND = CONTEXT + f" --execute --out build/{OUTPUT_PREFIX}fresh"
EXPECTED_TESTS = 30
require, record, write = legacy.require, legacy.record, legacy.write
interventions, intervene, classify = v2.interventions, v2.intervene, v2.classify
compare_parents, check_reports = v2.compare_parents, v2.check_reports
HISTORY = {
    "ace3/model/candidates/diagnose_q24_s16_l13_l12_parent_coordinate_cut_v1.py":
        "d736d5a2ff7324ed0a103c27a2a9a8fdb0b78c6f057d7c35c19804b171200d6e",
    "ace3/model/candidates/diagnose_q24_s16_l13_l12_parent_coordinate_cut_v2.py":
        "8ceccf15cea6f571b28cc81d9ce8db9df2ae7a21d40c4ddaf663d675fe72ea67",
    "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v1.json":
        "c9ac0e723ee5b482712f9307130e40c106eff6ae7ec3b6ce45f78f4588d38dfa",
    "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v2.json":
        "ef52ebd67cc3f836c6be1704457afe0883ea85d5250e34a008b0b1884dabdbc6",
    "tests/test_q24_s16_l13_l12_parent_coordinate_cut_v1.py":
        "cda6da51416647e5281eef11417e7cdf35cfa0c9a2d16f0402b46aceeab90229",
    "tests/test_q24_s16_l13_l12_parent_coordinate_cut_v2.py":
        "15b1b503b4a7c8c6319289165540d12331e65fb01a4d9ed9ad71cfa5cbb54f59",
}


def history_records():
    records = [record(ROOT / relative) for relative in HISTORY]
    for item, digest in zip(records, HISTORY.values(), strict=True):
        require(item["sha256"] == digest, f"v1/v2 history changed: {item['path']}")
    return records


def check_contract(contract):
    history_records()
    base = json.loads(v2.CONTRACT.read_text())
    v2.check_contract(base)
    expected = {key: base[key] for key in v2.PRESERVED_FIELDS}
    expected.update(
        diagnostic_id=ID, cwd=str(ROOT), command=COMMAND, execution_command=EXECUTION_COMMAND,
        interface=["--check", "--execute --out FRESH_DIRECTORY"],
        inherits=str(v2.CONTRACT.relative_to(ROOT)), history_sha256=HISTORY,
        source_updates=v2.SOURCE_UPDATES, retained_freeze_sha256=prior.FREEZE_SHA,
        historical_blocked_result=str(v2.BLOCKED), historical_blocked_sha256=v2.BLOCKED_SHA,
        native_L0_L8_invocations=0, native_layers_per_control=[13],
        maximum_native_layer_invocations=954, scientific_result_claim=False,
        execution_boundary="--check compiles the repository sources and runs only the v3 and "
        "v2 focused suites, then authenticates retained/current bindings without native "
        "dispatch. --execute additionally runs exactly the 954 isolated L13/P0 controls "
        "into a fresh v3 build directory; never replay the accepted prefix or admission "
        "run. This delivery task does not execute those controls. Future execution and "
        "its non-admission evidence require a separate bounded task and normal Host review.",
        reproduction="Run command once from cwd with the explicit absolute interpreter and "
        "command-local PYTHONPATH. No files are written by --check. The execution command "
        "is a template: replace only the fresh output basename, never reuse an existing "
        "path, and use the operator-required durable runner for the long execution. "
        "Independent Reviewer must reproduce the check with zero failures/errors/skips. "
        "No historical bare-Python command is certified.",
        authentication="Inherit the hash-pinned v2 check contract and its exact retained/current "
        "source-update closure. Bind all reviewed artifacts and original snapshots without "
        "relabeling current sources as historical execution. Consume the authenticated "
        "retained operand/state/KV evidence read-only; recheck the L12 receipt, paired "
        "state, own-layer KV, RTZ operands and original-input reference chain before "
        "dispatch. Every control uses the unchanged native L13 stage/gate implementation. "
        "Reproduce both reviewed endpoints and recheck all bindings and origins at completion.",
    )
    require(set(contract) == set(expected), "v3 contract fields changed")
    for key, value in expected.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"v3 contract mismatch: {key}")


def origins():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "v3 entry source origin mismatch")
    require(CONTRACT.resolve() == ROOT / "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v3.json",
            "v3 contract origin mismatch")
    importlib.import_module(TEST_MODULE)
    importlib.import_module("tests.test_q24_s16_toward_zero_l3_l8_v1")
    result = v2.origins()
    require(MODULE in result and TEST_MODULE in result, "missing v3 candidate/test origins")
    return result


def recheck(items):
    for item in items:
        require(record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")


def run_suite(module, expected):
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    count = suite.countTestCases()
    require(count == expected, f"wrong focused collection: {module.__name__}: {count}")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(result.testsRun == count and result.wasSuccessful() and not result.skipped,
            f"focused validation failed:\n{log.getvalue()}")
    return {"module": module.__name__, "collected": count, "executed": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "unittest_log": log.getvalue()}


def validate():
    source_origins = origins()
    contract_record = record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for name in (MODULE, TEST_MODULE, v2.MODULE, v2.TEST_MODULE):
        path = Path(source_origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(source_origins[name])
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
            "native dispatch forbidden during v3 validation")))
            for owner, name in ((legacy, "execute"), (legacy, "diagnose"),
                                (upstream, "execute_layer"), (native, "stages"),
                                (native, "execute_layers"))]
        suites = [run_suite(importlib.import_module(name), count)
                  for name, count in ((TEST_MODULE, EXPECTED_TESTS), (v2.TEST_MODULE, 26))]
        authentication = v2.authenticate()
        load_data(authentication)
        require(not any(guard.called for guard in guards), "native dispatch attempted during check")
    require(origins() == source_origins and record(CONTRACT) == contract_record,
            "source/contract changed during check")
    history = history_records()
    return {
        "diagnostic_id": ID, "status": "VALIDATED_SOFTWARE_ONLY",
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "command": COMMAND, "origins": source_origins, "compiled": compiled,
        "contract": contract_record, "history": history, "suites": suites,
        "authentication": authentication, "native_layer_invocations": 0,
        "execution_input_preflight": "PASS; read-only parent/state/KV/reference checks, no native controls",
        "native_L0_L8_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED",
    }


def check_reference_chain(extension):
    require(extension["reference_only"] is True
            and extension["policy_id"] == prior.gates.POLICY_ID
            and extension["global_reference_policy"] ==
            "legacy-binary64-AWQ-fully-independent-propagation",
            "reference substituted or re-anchored")
    expected64, expected16 = extension["original_binary64_parent"], extension["original_fp16_parent"]
    for layer in range(9, 14):
        item = extension["layers"][str(layer)]
        require(item["input_binary64"] == expected64 and item["input_fp16"] == expected16
                and item["prior_kv"] == "own empty P0", "spliced original-input reference")
        expected64, expected16 = item["binary64"], item["fp16"]


def check_parent_receipt(receipt, freeze_record, freeze, previous, entry):
    require(receipt["next_layer"] == 13 and receipt["position"] == 0
            and receipt["history"] == [9707] and receipt["rtl_admissible"] is False
            and receipt["evidence_kind"] == "cpu_software_q24"
            and receipt["candidate_id"] == freeze["contract"]["candidate_id"]
            and receipt["state_id"] == freeze["contract"]["state_id"]
            and receipt["policy_id"] == prior.gates.POLICY_ID
            and receipt["arithmetic_lineage"] == freeze_record
            and receipt["state"] == previous["output_parent"] == entry["input_parent"]
            and receipt["state_lineage_parent"] == previous["input_parent"]
            and receipt["numerical_report"] == previous["reports"]
            and receipt["kv"] == previous["kv_state"], "L12 parent receipt lineage mismatch")


def load_data(authentication):
    bound = {item["path"]: item for item in authentication["input_bindings"]}
    recheck(bound.values())
    updates = {}
    bind = lambda item: v2.bind_input(item, bound, updates)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    reviewed = read(authentication["reviewed_parent_cut"])

    def artifact(name, arrays=False):
        matches = [item for item in reviewed["artifacts"]
                   if item["path"] == str(legacy.REVIEWED / name)]
        require(len(matches) == 1, f"missing reviewed artifact: {name}")
        return archive(matches[0]) if arrays else read(matches[0])

    retained_authentication = artifact("retained_authentication.json")
    for item in retained_authentication["input_bindings"]:
        bind(item)
    freeze_record = authentication["retained_freeze"]
    freeze = read(freeze_record)
    require(freeze["rtl_invocations"] == 0
            and freeze["contract"]["policy_id"] == prior.gates.POLICY_ID
            and freeze["contract"]["policy_adopted"] is False
            and freeze["global_reference_lineage"]["candidate_data_used"] is False,
            "wrong retained software/reference scope")
    extension = read(freeze["reference_extension"])
    check_reference_chain(extension)
    for layer in range(9, 14):
        item = extension["layers"][str(layer)]
        for key in ("input_binary64", "input_fp16", "binary64", "fp16"):
            bind(item[key])
    retained = read(record(prior.INPUT / "result.json"))
    entries = retained["layers"]
    require(retained["status"] == "FAIL" and retained["candidate_admitted"] is False
            and retained["rtl_invocations"] == 0
            and [item["layer"] for item in entries] == list(range(9, 14))
            and all(item["status"] == "PASS" for item in entries[:-1])
            and entries[-1]["status"] == "FAIL", "unexpected retained traversal")
    previous, entry = entries[-2:]
    require("output_parent" not in entry
            and not (prior.INPUT / "layer13/software_parent.json").exists(),
            "failed layer published a successor")
    receipt = read(record(prior.INPUT / "layer12/software_parent.json"))
    check_parent_receipt(receipt, freeze_record, freeze, previous, entry)
    parent_reports = read(receipt["numerical_report"])
    require(len(parent_reports) == 19 and all(
        report["node"] == [12, 0, stage] and report["status"] == "PASS"
        for stage, report in enumerate(parent_reports)), "L12 parent not passing")
    actual, actual_arrays = archive(receipt["state"]), archive(entry["actual_stages"])
    prior.check_lineage(actual_arrays, actual, archive(previous["actual_stages"]),
                        archive(receipt["kv"]))
    reports = read(entry["reports"])
    prior.select_failure(retained, reports)
    require(retained["first_failure"]["evidence"] == entry["reports"]["path"],
            "wrong failure report path")
    item = extension["layers"]["13"]
    original = np.load(bind(item["input_binary64"]), allow_pickle=False)
    reference = np.load(bind(item["binary64"]), allow_pickle=False)
    paired.drift.same_binary64(reference, reference)
    mapped = paired.mapped_parent(original)
    paired.same_arrays(mapped, artifact("cut12_mapped_parent.npz", True))
    prior.retained.verify_parent(actual, prior.retained.state_from(
        artifact("actual_layer12.npz", True), "output", "stage18"))
    paired.same_arrays(actual_arrays, artifact("actual_layer13.npz", True))
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                   for key in prior.local.tensor_shapes(13)}
    prior.local.authenticate_tensors(tensors, item["canonical"], 13)
    return {
        "bound": bound, "authentication": authentication, "actual": actual, "mapped": mapped,
        "original": original, "reference": reference, "tensors": tensors,
        "trajectory": archive(item["fp16"]), "actual_arrays": actual_arrays,
        "actual_locals": archive(entry["local_references"]), "actual_reports": reports,
        "mapped_arrays": artifact("cut12_mapped_layer13.npz", True),
        "mapped_reports": artifact("cut12_mapped_layer13_gates.json"),
    }


def execute_layer(tensors, layer, parent, trajectory, reference):
    require(type(layer) is int and layer == 13, "v3 permits only isolated native L13/P0")
    return upstream.execute_layer(tensors, layer, parent, trajectory, reference)


def diagnose(out, log, contract, validation):
    started = time.monotonic()
    torch.set_num_threads(1)
    data = load_data(validation["authentication"])
    timing = {"authentication": time.monotonic() - started}
    artifacts, rows = [], []

    def document(name, value):
        write(out / name, value)
        item = record(out / name)
        artifacts.append(item)
        return item

    document("retained_authentication.json", data["authentication"])
    document("parent_comparison.json", compare_parents(data["actual"], data["mapped"], data["original"]))
    for label in ("actual", "mapped"):
        artifacts.append(prior.retained.save(out / f"{label}_L12_parent.npz", data[label]))
    endpoint_rows = {}
    for label, kind, indices in interventions():
        require(len(rows) < legacy.EXPECTED_CONTROLS, "native control budget exceeded")
        began = time.monotonic()
        parent = intervene(data["actual"], data["mapped"], indices)
        arrays, locals_, reports = execute_layer(
            data["tensors"], 13, parent, data["trajectory"], data["reference"])
        native_seconds = time.monotonic() - began
        if label in ("actual", "mapped"):
            paired.same_arrays(arrays, data[label + "_arrays"])
            check_reports(reports, data[label + "_reports"])
            if label == "actual":
                paired.same_arrays(locals_, data["actual_locals"])
        gate = prior.measure(int(arrays["stage18"][62]), float(data["reference"][62]))
        vectors = prior.retained.save(out / f"{label}_native.npz", arrays)
        artifacts.append(vectors)
        gates = document(f"{label}_gates.json", reports)
        failures = [item["index"] for item in reports[18]["binary64_v1"]["failures"]]
        require((62 not in failures) == gate["accepted"], "scalar/full-vector gate disagreement")
        if label == "actual":
            require(failures == [62] and gate["actual_fp16_bits"] == "6630"
                    and gate["accepted"] is False, "retained rejection changed")
        elif label == "mapped":
            require(gate["actual_fp16_bits"] == "662f" and gate["accepted"] is True
                    and all(r["status"] == "PASS" for r in reports), "mapped control changed")
        row = {
            "label": label, "kind": kind, "indices": indices, "index62": gate,
            "vectors": vectors, "gates": gates, "mandatory_statuses": [r["status"] for r in reports],
            "L13_S18_failure_indices": failures,
            "L13_S18_changed_indices": np.flatnonzero(
                arrays["stage18"] != data["actual_arrays"]["stage18"]).tolist(),
            "all_L13_gates_pass": all(r["status"] == "PASS" for r in reports),
            "source_operand_state_KV_lineage_RTZ_checks": "PASS", "candidate_admitted": False,
        }
        if kind in ("baseline", "full_parent") or gate["accepted"]:
            full = [dict(index=i, **prior.measure(int(arrays["stage18"][i]),
                                                 float(data["reference"][i])))
                    for i in range(legacy.WIDTH)]
            require([r["index"] for r in full if not r["accepted"]] == failures,
                    "full-vector scalar gate disagreement")
            row["full_vector_outcomes"] = document(f"{label}_full_vector.json", full)
        if label in ("actual", "mapped"):
            endpoint_rows[label] = gate
        row["timing_seconds"] = {"native_and_all_gates": native_seconds,
                                 "total_control": time.monotonic() - began}
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
    document("interventions.json", rows)
    require(len(rows) == legacy.EXPECTED_CONTROLS, "missing planned controls")
    require(origins() == validation["origins"] and record(CONTRACT) == validation["contract"],
            "module/source/contract origins changed during execution")
    recheck(data["bound"].values())
    recheck(validation["history"])
    timing["total_diagnosis"] = time.monotonic() - started
    timing["native_and_all_gates"] = sum(r["timing_seconds"]["native_and_all_gates"] for r in rows)
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "native_L0_L8_invocations": 0, "native_layer_invocations": len(rows),
        "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "native_retained_bitwise_reproduction": True, "reviewed_L12_cut_reproduced": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "origins_after_execution": validation["origins"],
        "control_count": len(rows), "singleton_count": legacy.WIDTH, "shard_count": 28,
        "shard_complement_count": 28, "selected_coordinate": endpoint_rows,
        "all_gate_passing_controls": [r["label"] for r in rows if r["all_L13_gates_pass"]],
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "timing_seconds": timing, **classify(rows), "claim_boundary": contract["claim_boundary"],
    }


def output_path(value):
    out = value.absolute()
    require(out == out.resolve() and out.parent == ROOT / "build"
            and out.name.startswith(OUTPUT_PREFIX) and len(out.name) > len(OUTPUT_PREFIX),
            "output outside fresh v3 build scope")
    require(not out.exists() and not out.is_symlink(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check and args.out is not None or args.execute and args.out is None:
        parser.error("--out is required only with --execute")
    if args.check:
        print(json.dumps(validate(), sort_keys=True), flush=True)
        return 0
    out = output_path(args.out)
    validation = validate()
    out.mkdir(exist_ok=False)
    write(out / "validation.json", validation)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": os.environ.get("PYTHONPATH")}) + "\n")
        log.flush()
        try:
            result = diagnose(out, log, json.loads(CONTRACT.read_text()), validation)
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError) as exc:
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
                "rtl_invocations": 0, "native_L0_L8_invocations": 0,
                "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
    print(json.dumps({"status": result["status"], "result": str(out / "result.json")}), flush=True)
    return 0 if result["status"] == "DIAGNOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
