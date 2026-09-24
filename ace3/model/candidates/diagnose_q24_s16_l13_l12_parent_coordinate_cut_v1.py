"""Non-admitting, exhaustive singleton and bounded shard L12 parent controls.

The adjacent contract specifies inputs, mapping, scope and one-shot validation.
Only the native L13/P0 suffix executes; no successor state is published.
"""

import argparse
from fractions import Fraction
import importlib
import json
import os
from pathlib import Path
import py_compile
import sys
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_upstream_parent_cut_v1 as upstream


paired = upstream.paired
prior = upstream.prior
native = upstream.native
ROOT = prior.ROOT
ID = "ace3-q24-s16-l13-l12-parent-coordinate-cut-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l13_l12_parent_coordinate_cut_v1"
TEST_MODULE = "tests.test_q24_s16_l13_l12_parent_coordinate_cut_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v1.json"
REVIEWED = ROOT / "build/q24_s16_l13_upstream_parent_cut_a81669925e0b_attempt001"
REVIEWED_SHA = "f9a98abf3c266ecf79ee1822223617436d91037851137b8d25f0fa33ee5eb732"
WIDTH = 896
SHARD_SIZE = 32
EXPECTED_CONTROLS = 954
require = prior.require
record = paired.record
write = paired.write


def interventions():
    yield "actual", "baseline", []
    yield "mapped", "full_parent", list(range(WIDTH))
    for index in range(WIDTH):
        yield f"single_{index:03d}", "singleton", [index]
    for start in range(0, WIDTH, SHARD_SIZE):
        shard = list(range(start, start + SHARD_SIZE))
        yield f"shard_{start:03d}", "shard", shard
        yield f"complement_{start:03d}", "shard_complement", [
            index for index in range(WIDTH) if index not in shard]


def intervene(actual, mapped, indices):
    prior.retained.verify_parent(actual, actual)
    prior.retained.verify_parent(mapped, mapped)
    require(type(indices) is list and all(type(i) is int and 0 <= i < WIDTH for i in indices)
            and indices == sorted(set(indices)), "invalid coordinate intervention")
    result = {key: value.copy() for key, value in actual.items()}
    for key in ("i", "z", "h"):
        result[key][indices] = mapped[key][indices]
    prior.retained.verify_parent(result, result)
    return result


def classify(rows):
    plan = list(interventions())
    require(len(rows) == len(plan) and all(
        row["label"] == label and row["kind"] == kind and row["indices"] == indices
        for row, (label, kind, indices) in zip(rows, plan, strict=True)),
        "incomplete or reordered coordinate experiment")
    require(rows[0]["index62"]["actual_fp16_bits"] == "6630"
            and rows[0]["index62"]["accepted"] is False, "retained index62 baseline changed")
    require(rows[1]["index62"]["actual_fp16_bits"] == "662f"
            and rows[1]["index62"]["accepted"] is True, "reviewed mapped endpoint changed")
    singles = [row["indices"][0] for row in rows
               if row["kind"] == "singleton" and row["index62"]["accepted"]]
    shards = [row["label"] for row in rows if row["kind"] in ("shard", "shard_complement")
              and row["index62"]["accepted"]]
    return {
        "classification": ("single_coordinate_sufficiency_under_disclosed_mapping" if singles
                           else "bounded_shard_sufficiency_without_singleton_rescue" if shards
                           else "full_parent_rescue_with_unresolved_subset_interactions"),
        "sufficient_single_coordinates": singles,
        "sufficient_shard_controls": shards,
        "single_coordinate_662f_rescues": [row["indices"][0] for row in rows
            if row["kind"] == "singleton" and row["index62"]["accepted"]
            and row["index62"]["actual_fp16_bits"] == "662f"],
        "unique_sufficient_singleton_in_tested_mapping": len(singles) == 1,
        "unique_upstream_producer_attributed": False,
        "missing_evidence": [
            "Sufficiency is conditional on replacing complete I/Z/H coordinates with the "
            "nearest-Q24 ties-even mapped original-input L12 parent, not an admitted repair.",
            "Only singletons, 32-coordinate contiguous shards and their complements are "
            "tested. Other subsets, necessity of single coordinates on the mapped background, "
            "minimal interaction sets and alternative parent mappings are not established.",
            "No upstream producer is isolated: L12 parent drift combines inherited and "
            "own-layer effects. A scalar rescue is not a full-vector or full-model PASS.",
        ],
    }


def origins():
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        if name not in ("ace3", "tests") and not name.startswith(("ace3.", "tests.")):
            continue
        expected = ROOT.joinpath(*name.split("."))
        filename = getattr(module, "__file__", None)
        if filename is None:
            locations = list(getattr(module, "__path__", ()))
            require(locations and all(Path(p).resolve() == expected for p in locations),
                    f"namespace origin mismatch: {name}")
            result[name] = {"namespace_paths": locations}
        else:
            path = Path(filename).resolve()
            require(path.is_relative_to(ROOT)
                    and path in (expected.with_suffix(".py"), expected / "__init__.py"),
                    f"module origin mismatch: {name}")
            result[name] = record(path)
    require(MODULE in result and TEST_MODULE in result, "missing candidate/test origins")
    return result


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "use the published repository-bound command")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "entry source origin mismatch")
    tests = importlib.import_module(TEST_MODULE)
    source_origins = origins()
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["reviewed_result_sha256"] == REVIEWED_SHA
            and contract["policy_id"] == prior.gates.POLICY_ID
            and contract["width"] == WIDTH and contract["shard_size"] == SHARD_SIZE
            and contract["controls"] == EXPECTED_CONTROLS
            and contract["normal_host_review"] == "REQUIRED"
            and contract["rtl_invocations"] == 0
            and all(contract[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "coordinate-cut contract mismatch")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(out / "validation.json", {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "contract": record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "accepted_L0_L8_tests_executed": False,
    })
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return contract


def authenticate(contract):
    bound = {}
    bind = lambda item: upstream.bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    reviewed_record = record(REVIEWED / "result.json")
    require(reviewed_record["sha256"] == contract["reviewed_result_sha256"],
            "wrong reviewed parent-cut result")
    reviewed = read(reviewed_record)
    require(reviewed["diagnostic_id"] == upstream.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["reviewed_L12_cut_reproduced"] is True
            and reviewed["native_retained_bitwise_reproduction"] is True
            and reviewed["unchanged_baseline_gates"] is True
            and reviewed["original_global_reference_unchanged"] is True
            and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
            and reviewed["rtl_invocations"] == 0
            and all(reviewed[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "reviewed parent-cut scope mismatch")
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    for item in reviewed["origins_after_execution"].values():
        if "path" in item:
            bind(item)
    authentication = prior.diagnose(prior.INPUT)
    for item in authentication["input_bindings"]:
        bind(item)

    def reviewed_artifact(filename, arrays=False):
        matches = [item for item in reviewed["artifacts"]
                   if item["path"] == str(REVIEWED / filename)]
        require(len(matches) == 1, f"missing reviewed artifact: {filename}")
        return archive(matches[0]) if arrays else read(matches[0])

    freeze = read(record(prior.INPUT / "freeze.json"))
    extension = read(freeze["reference_extension"])
    retained = read(record(prior.INPUT / "result.json"))
    entry = retained["layers"][-1]
    require(entry["layer"] == 13 and entry["status"] == "FAIL", "wrong retained endpoint")
    expected64, expected16 = extension["original_binary64_parent"], extension["original_fp16_parent"]
    for layer in range(9, 14):
        item = extension["layers"][str(layer)]
        require(item["input_binary64"] == expected64 and item["input_fp16"] == expected16
                and item["prior_kv"] == "own empty P0", "spliced original-input reference")
        for key in ("input_binary64", "input_fp16", "binary64", "fp16"):
            bind(item[key])
        expected64, expected16 = item["binary64"], item["fp16"]
    item = extension["layers"]["13"]
    original = np.load(bind(item["input_binary64"]), allow_pickle=False)
    reference = np.load(bind(item["binary64"]), allow_pickle=False)
    paired.drift.same_binary64(reference, reference)
    actual = archive(entry["input_parent"])
    mapped = paired.mapped_parent(original)
    paired.same_arrays(mapped, reviewed_artifact("cut12_mapped_parent.npz", True))
    prior.retained.verify_parent(actual, prior.retained.state_from(
        reviewed_artifact("actual_layer12.npz", True), "output", "stage18"))
    actual_arrays = archive(entry["actual_stages"])
    paired.same_arrays(actual_arrays, reviewed_artifact("actual_layer13.npz", True))
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                   for key in prior.local.tensor_shapes(13)}
    prior.local.authenticate_tensors(tensors, item["canonical"], 13)
    return {
        "bound": bound, "authentication": authentication, "actual": actual, "mapped": mapped,
        "original": original, "reference": reference, "tensors": tensors,
        "trajectory": archive(item["fp16"]), "actual_arrays": actual_arrays,
        "actual_locals": archive(entry["local_references"]), "actual_reports": read(entry["reports"]),
        "mapped_arrays": reviewed_artifact("cut12_mapped_layer13.npz", True),
        "mapped_reports": reviewed_artifact("cut12_mapped_layer13_gates.json"),
    }


def compare_parents(actual, mapped, original):
    rows = []
    for index in range(WIDTH):
        a, m = Fraction(int(actual["i"][index]), 1 << 24), Fraction(int(mapped["i"][index]), 1 << 24)
        reference = paired.drift.exact(original[index])
        require(abs(m - reference) <= Fraction(1, 1 << 25), "mapping exceeds half-grid")
        rows.append({
            "index": index, "actual": {key: int(value[index]) for key, value in actual.items()},
            "mapped": {key: int(value[index]) for key, value in mapped.items()},
            "original_binary64_hex": float(original[index]).hex(),
            "actual_minus_original": str(a - reference), "mapped_minus_original": str(m - reference),
            "mapped_minus_actual_Q24_units": int(mapped["i"][index]) - int(actual["i"][index]),
            "changed_fields": [key for key in ("i", "z", "h")
                               if actual[key][index] != mapped[key][index]],
        })
    return rows


def execute(parent, data):
    return upstream.execute_layer(data["tensors"], 13, parent, data["trajectory"], data["reference"])


def check_reports(fresh, saved):
    require(len(fresh) == len(saved) == 19, "incomplete endpoint gate reports")
    for stage, (a, b) in enumerate(zip(fresh, saved, strict=True)):
        for key in ("node", "policy_id", "status", "residual_state_lineage", "kv_lineage",
                    "local_operator_fp16" if stage < 18 else "binary64_v1"):
            require(a[key] == b[key], f"reviewed endpoint gate changed: S{stage}/{key}")


def diagnose(out, log, contract):
    started = time.monotonic()
    torch.set_num_threads(1)
    data = authenticate(contract)
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
        began = time.monotonic()
        parent = intervene(data["actual"], data["mapped"], indices)
        arrays, locals_, reports = execute(parent, data)
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
            "source_operand_state_KV_lineage_RTZ_checks": "PASS",
            "candidate_admitted": False,
        }
        # All controls retain complete native vectors and all 19 gate reports.
        # Decisive endpoints/rescues also expose every scalar exact-excess result.
        if kind in ("baseline", "full_parent") or gate["accepted"]:
            full = [dict(index=i, **prior.measure(int(arrays["stage18"][i]),
                                                 float(data["reference"][i])))
                    for i in range(WIDTH)]
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
    require(len(rows) == EXPECTED_CONTROLS, "missing planned controls")
    source_origins = origins()
    validation = json.loads((out / "validation.json").read_text())
    require(source_origins == validation["origins"], "module/source origins changed during execution")
    for item in data["bound"].values():
        require(record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    timing["total_diagnosis"] = time.monotonic() - started
    timing["native_and_all_gates"] = sum(r["timing_seconds"]["native_and_all_gates"] for r in rows)
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "native_retained_bitwise_reproduction": True, "reviewed_L12_cut_reproduced": True,
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS", "origins_after_execution": source_origins,
        "control_count": len(rows), "singleton_count": WIDTH, "shard_count": 28,
        "shard_complement_count": 28, "selected_coordinate": endpoint_rows,
        "all_gate_passing_controls": [r["label"] for r in rows if r["all_L13_gates_pass"]],
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "timing_seconds": timing, **classify(rows), "claim_boundary": contract["claim_boundary"],
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l13_l12_parent_coordinate_cut_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.input.resolve() == prior.INPUT and args.reviewed.resolve() == REVIEWED,
            "unselected authenticated inputs")
    out = output_path(args.out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({
            "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
            "PYTHONPATH": os.environ.get("PYTHONPATH"), "loadavg": os.getloadavg(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_threads_before": torch.get_num_threads(),
            "rtl_invocations": 0,
        }) + "\n")
        log.flush()
        try:
            contract = validate(out)
            result = diagnose(out, log, contract)
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                "candidate_admitted": False, "policy_adopted": False,
                "successor_published": False, "rtl_invocations": 0, "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
        summary = {key: result[key] for key in
                   ("status", "classification", "candidate_admitted", "missing_evidence")}
        summary["result"] = str(out / "result.json")
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
