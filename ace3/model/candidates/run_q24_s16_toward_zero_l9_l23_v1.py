"""Isolated L9-L23/P0 CPU continuation of the reviewed native Q24 L8 parent.

Published command context (no Host/global configuration changes):
    cd /home/argustest/ace3-argus
    /home/argustest/miniconda3/bin/python -I -B \
        ace3/model/candidates/run_q24_s16_toward_zero_l9_l23_v1.py \
        --out build/q24_software_l9_l23_<task>_attempt001 \
        --resume-parent build/q24_software_l3_l8_dc98a86e7c02_attempt001/layer08/software_parent.json

The script explicitly prepends this repository to sys.path. Compilation,
origin records and focused regressions precede authentication and compute.
No accepted candidate prefix is executed. Independent Host review is required.
"""

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import platform
import py_compile
import re
import shutil
import sys
import time
import unittest


ROOT = Path("/home/argustest/ace3-argus")
PYTHON = Path("/home/argustest/miniconda3/bin/python")
if __name__ == "__main__":
    if Path.cwd() != ROOT or Path(sys.executable) != PYTHON:
        raise RuntimeError("use the published repository-bound cwd and interpreter")
    sys.path.insert(0, str(ROOT))

import numpy as np
import safetensors
from safetensors import safe_open
import torch

from ace3.model.candidates import run_q24_s16_toward_zero_l3_l8_v1 as prefix
from ace3.model.candidates import remaining_layers_v3 as references


runner = prefix.runner
candidate = runner.candidate
retained = runner.retained
local = runner.local
gates = runner.gates
require = runner.require
LAYERS = tuple(range(9, 24))
SCOPE = {"layers": list(LAYERS), "position": 0, "history": [9707]}
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_toward_zero_l9_l23_v1.json"
ACCEPTED = ROOT / "build/q24_software_l3_l8_dc98a86e7c02_attempt001"
SELECTED_PARENT = ACCEPTED / "layer08/software_parent.json"
REVIEW = retained.HANDOFFS / "dc98a86e7c02/round-0003.json"
ACCEPTED_FREEZE_SHA = "2f90c70717f30a03ee8040470ad8ff1fc1de01a8211d6515ebebb6642044acf3"
TEST_MODULE = "tests.test_q24_s16_toward_zero_l9_l23_v1"


def stages(tensors, layer, parent, arrays):
    require(type(layer) is int and layer in LAYERS, "continuation scope is L9-L23/P0")
    yield from candidate._stages(tensors, layer, parent, arrays)


def bind_record(record, bound):
    signature = {key: record[key] for key in ("path", "bytes", "sha256")}
    path = signature["path"]
    require(path not in bound or bound[path] == signature, f"conflicting binding: {path}")
    if path not in bound:
        require(retained.record(path) == signature, f"binding mismatch: {path}")
        bound[path] = signature
    return Path(path)


def bind_tree(value, bind):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            bind(value)
        else:
            for child in value.values():
                bind_tree(child, bind)
    elif isinstance(value, list):
        for child in value:
            bind_tree(child, bind)


def load_state(record, bind):
    with np.load(bind(record), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def validate_parent_scope(parent):
    require(parent["candidate_id"] == "ace3-q24-s16-toward-zero-native-v1"
            and parent["state_id"] == "ace3-q24-software-paired-state-v1"
            and parent["policy_id"] == gates.POLICY_ID
            and parent["model"] == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
            and parent["history"] == [9707]
            and type(parent["position"]) is int and parent["position"] == 0
            and type(parent["next_layer"]) is int and parent["next_layer"] == 9
            and parent["evidence_kind"] == "cpu_software_q24"
            and parent["rtl_admissible"] is False
            and parent["normal_host_review"] == "REQUIRED",
            "wrong reviewed L8 Q24 continuation scope")


def authenticate_resume(path, bind, read):
    require(path == SELECTED_PARENT, "only the Planner-selected accepted L8 parent is supported")
    receipt = retained.record(path)
    parent = read(receipt)
    validate_parent_scope(parent)
    review_record = retained.record(REVIEW)
    review = read(review_record)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "dc98a86e7c02"
            and review["review"]["status"] == "done", "missing independent accepted L8 review")
    require(parent["arithmetic_lineage"]["path"] == str(ACCEPTED / "freeze.json")
            and parent["arithmetic_lineage"]["sha256"] == ACCEPTED_FREEZE_SHA,
            "wrong accepted L3-L8 arithmetic freeze")
    freeze = read(parent["arithmetic_lineage"])
    require(freeze["contract"]["scope"] == {
        "layers": list(range(3, 9)), "position": 0, "history": [9707]},
        "wrong accepted L3-L8 scope")
    source_paths = {source["original"]["path"] for source in freeze["sources"]}
    for record in freeze["input_bindings"]:
        if record["path"] not in source_paths:
            bind(record)
    for source in freeze["sources"]:
        bind(source["snapshot"])
        require(source["snapshot"]["sha256"] == source["original"]["sha256"],
                "accepted source snapshot mismatch")
    for module in (candidate, candidate.state, retained, local, gates, runner, prefix):
        matches = [source for source in freeze["sources"]
                   if source["original"]["path"] == str(Path(module.__file__).resolve())]
        require(len(matches) == 1, "missing accepted arithmetic/evaluator source")
        bind(matches[0]["original"])
    root = prefix.authenticate_resume(prefix.SELECTED_PARENT, bind, read,
                                      lambda value: bind_tree(value, bind))
    require(freeze["state_lineage"]["prior_residual"] == root,
            "accepted L3 input differs from reviewed root lineage")
    result_record = retained.record(ACCEPTED / "result.json")
    result = read(result_record)
    require(result["status"] == "PASS" and result["candidate_admitted"] is True
            and result["candidate_id"] == parent["candidate_id"]
            and result["policy_id"] == parent["policy_id"]
            and result["scope"] == freeze["contract"]["scope"]
            and result["normal_host_review"] == "REQUIRED"
            and result["rtl_invocations"] == result["l9_invocations"] == 0
            and result["policy_adopted"] is False and result["historical_fail_preserved"] is True
            and result["first_failure"] is None
            and [entry["layer"] for entry in result["layers"]] == list(range(3, 9)),
            "accepted L3-L8 result is not a complete CPU PASS")
    bind_tree(result["layers"], bind)
    bind_tree(parent, bind)
    previous = root["parent"]["state"]
    for layer, entry in zip(range(3, 9), result["layers"], strict=True):
        reports = read(entry["reports"])
        require(entry["status"] == "PASS" and entry["position"] == 0
                and entry["input_parent"] == previous and len(reports) == 19
                and all(report["stage"] == stage and report["node"] == [layer, 0, stage]
                        and report["status"] == "PASS" for stage, report in enumerate(reports)),
                "accepted stage coverage or predecessor lineage mismatch")
        incoming = load_state(previous, bind)
        arrays = load_state(entry["actual_stages"], bind)
        retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), incoming)
        for stage in range(19):
            retained.check_stage_state(stage, arrays, incoming)
        outgoing = load_state(entry["output_parent"], bind)
        retained.verify_parent(outgoing, retained.state_from(arrays, "output", "stage18"))
        cache = load_state(entry["kv_state"], bind)
        require(set(cache) == {"k", "v"}, "accepted KV keys changed")
        for kind in ("k", "v"):
            local.finite_words(cache[kind], (1, 128))
            require(np.array_equal(cache[kind], arrays["output_cache_" + kind]),
                    "accepted KV differs from its own layer output")
        previous = entry["output_parent"]
    final = result["layers"][-1]
    require(parent["state"] == final["output_parent"]
            and parent["state_lineage_parent"] == final["input_parent"]
            and parent["numerical_report"] == final["reports"]
            and parent["kv"] == final["kv_state"], "L8 receipt differs from reviewed output")
    bind(retained.record(ACCEPTED / "command.log"))
    return {"receipt": receipt, "parent": parent, "review": review_record,
            "result": result_record, "prior_layer_kv_consumed": False}


def prepare_references(out, bind, read):
    policy = read(retained.record(gates.CONTRACT))
    model_record = retained.record(retained.B_FREEZE)
    require(model_record["sha256"] == policy["trusted_independent_freeze_sha256"],
            "wrong independent original model freeze")
    model = read(model_record)
    control_record = retained.record(retained.CONTROL)
    require(control_record["sha256"] == retained.CONTROL_SHA, "wrong original global control")
    control = read(control_record)
    require(model["checkpoint"] == control["checkpoint"]
            and model["embeddings"] == control["embeddings"]
            and model["reference_source"] == control["fp16_reference_root"],
            "original-input reference lineage mismatch")
    base = references.original_context()
    bind_tree(base["reference_bindings"], bind)
    specification = read(base["reference_bindings"]["independent"])
    require(specification["checkpoint"] == model["checkpoint"], "reference checkpoint mismatch")
    bind(model["checkpoint"])
    read(control["binary64_reference_freeze"])
    read(control["binary64_array_binding_freeze"])
    bind(control["binary64_csv"])
    cases = [case for key, case in zip(control["ordered_cases"], control["cases"], strict=True)
             if key == [8, 0]]
    require(len(cases) == 1, "missing original L8 binary64 parent")
    original = np.load(bind(cases[0]["binary64_reference_array"]), allow_pickle=False)
    require(np.array_equal(original, base["binary64"]), "reference extension re-anchored at L8")
    transactions = [entry for entry in model["reference_transactions"]
                    if entry["layer"] == 8 and entry["position"] == 0]
    require(len(transactions) == 1, "missing original L8 FP16 parent")
    words = retained.read_words(bind(transactions[0]["stages"]["18"]), 896)
    require(np.array_equal(words, base["trajectory"][18]), "FP16 diagnostic reference re-anchored")
    extension_record, extension = references.bind_references(out, base, model["checkpoint"])
    bind_tree(extension, bind)
    return extension_record, extension, model_record, control_record


def first_failure_index(report):
    if report["status"] != "FAIL":
        return None
    mandatory = report["binary64_v1" if report["stage"] == 18 else "local_operator_fp16"]
    require(mandatory["passed"] is False and mandatory["failures"],
            "failing mandatory gate has no coordinate evidence")
    return mandatory["failures"][0]["index"]


def execute_layers(out, contract, resume, extension, bind, before_publish):
    layers = []
    parent_record = resume["parent"]["state"]
    parent = load_state(parent_record, bind)
    retained.verify_parent(parent, parent)
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        for layer in LAYERS:
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            node, phase = [layer, 0, 0], "operand_authentication"
            arrays, local_refs, reports = {}, {}, []
            failure = None
            status = "PASS"
            compute_seconds = oracle_seconds = 0.0
            producer = None
            try:
                item = extension["layers"][str(layer)]
                tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors, item["canonical"], layer)
                loaded = load_state(parent_record, bind)
                retained.verify_parent(loaded, parent)
                parent = loaded
                trajectories = load_state(item["fp16"], bind)
                binary64 = np.load(bind(item["binary64"]), allow_pickle=False)
                require(binary64.dtype == np.dtype("<f8") and binary64.shape == (896,)
                        and np.all(np.isfinite(binary64)) and np.all(np.abs(binary64) <= 65504),
                        "invalid independent global reference")
                producer = stages(tensors, layer, parent, arrays)
                for stage in range(19):
                    node, phase = [layer, 0, stage], "candidate_arithmetic"
                    stamp = time.monotonic()
                    require(next(producer) == stage, "out-of-order native stage")
                    compute_seconds += time.monotonic() - stamp
                    stamp, phase = time.monotonic(), "state_lineage"
                    expected = retained.check_stage_state(stage, arrays, parent)
                    phase = "independent_local_reference"
                    if stage < 18 and stage != 12:
                        expected = local.local_reference(stage, {
                            key: arrays[key].copy() for key in local.OPERANDS[stage]}, tensors, layer)
                    if stage < 18:
                        local_refs[f"stage{stage:02d}"] = expected
                    phase = "global_numerical" if stage == 18 else "local_operator_numerical"
                    report = gates.evaluate_decoder_stage(
                        stage=stage, actual=arrays[f"stage{stage:02d}"],
                        reference=trajectories[f"stage{stage:02d}"], policy=gates.POLICY_ID,
                        local_reference=expected, reference_binary64=binary64 if stage == 18 else None)
                    report.update(node=node, residual_state_lineage="PASS", kv_lineage="PASS",
                                  local_reference_independent=stage < 18)
                    reports.append(report)
                    oracle_seconds += time.monotonic() - stamp
                    if report["status"] != "PASS":
                        require(report["status"] in ("FAIL", "BLOCKED"), "invalid gate status")
                        status = report["status"]
                        failure = {"node": node, "index": first_failure_index(report),
                                   "failure_taxonomy": phase,
                                   "reason": "unchanged mandatory numerical gate failed",
                                   "evidence": str(directory / "reports.json")}
                        break
                if failure is None:
                    phase = "prepublication_binding_integrity"
                    before_publish()
            except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                    StopIteration, RuntimeError) as exc:
                status = "BLOCKED"
                failure = {"node": node, "index": None, "failure_taxonomy": phase,
                           "reason": f"{type(exc).__name__}: {exc}"}
            finally:
                if producer is not None:
                    producer.close()
                archive = retained.save(directory / "actual_stages.npz", arrays)
                locals_record = retained.save(directory / "local_references.npz", local_refs)
                retained.write(directory / "reports.json", reports)
            entry = {"layer": layer, "position": 0, "status": status,
                     "input_parent": parent_record, "actual_stages": archive,
                     "local_references": locals_record,
                     "reports": retained.record(directory / "reports.json"),
                     "candidate_seconds": compute_seconds, "oracle_seconds": oracle_seconds,
                     "local_vectors": sum(report["stage"] < 18 for report in reports),
                     "global_coordinates": 896 if len(reports) == 19 else 0}
            layers.append(entry)
            if failure is not None:
                return status, layers, failure
            parent = retained.state_from(arrays, "output", "stage18")
            parent_record = retained.save(directory / "parent_state.npz", parent)
            entry["output_parent"] = parent_record
            entry["kv_state"] = retained.save(directory / "kv_state.npz",
                {kind: arrays["output_cache_" + kind] for kind in ("k", "v")})
            retained.write(directory / "software_parent.json", {
                "candidate_id": contract["candidate_id"], "state_id": contract["state_id"],
                "policy_id": gates.POLICY_ID, "model": contract["model"], "history": [9707],
                "position": 0, "next_layer": layer + 1, "state": parent_record, "kv": entry["kv_state"],
                "state_lineage_parent": entry["input_parent"],
                "arithmetic_lineage": retained.record(out / "freeze.json"),
                "numerical_report": entry["reports"], "evidence_kind": "cpu_software_q24",
                "rtl_admissible": False, "normal_host_review": "REQUIRED"})
            print(json.dumps({"layer": layer, "status": status}), flush=True)
    return "PASS", layers, None


def validate_repository(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON,
            "repository-bound cwd/interpreter required")
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,
            "published validation requires -I -B and enabled assertions")
    sys.pycache_prefix = str(out / "bytecode")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module("tests.test_q24_s16_toward_zero_l3_l8_v1")
    origins = {}
    for name, module in sorted(list(sys.modules.items())):
        if name.startswith("ace3.model.candidates.") or name in (
                TEST_MODULE, "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            expected = ROOT.joinpath(*name.split(".")).with_suffix(".py")
            origin = Path(module.__file__).resolve()
            require(origin == expected and origin.is_file(), f"module origin mismatch: {name}: {origin}")
            origins[name] = str(origin)
    compiled = []
    paths = {Path(path) for path in origins.values()} | {Path(__file__).resolve()}
    for index, path in enumerate(sorted(paths)):
        py_compile.compile(str(path), cfile=str(out / "bytecode" / f"{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    json.loads(CONTRACT.read_text())
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == tests.EXPECTED_TESTS, "focused unittest collection count mismatch")
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    record = {"cwd": str(ROOT), "executable": str(PYTHON),
              "isolated": True, "dont_write_bytecode": True, "sys_path": sys.path,
              "origins": origins, "compiled": compiled, "collected": count,
              "executed": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "test_log": retained.record(out / "unittest.log")}
    retained.write(out / "validation.json", record)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped,
            "focused repository-bound regressions failed")
    return record


def run(out, resume_parent):
    started = time.monotonic()
    bound, layers, phase_times = {}, [], []
    phase, status, failure = "repository_validation", "BLOCKED", None
    bind = lambda record: bind_record(record, bound)
    read = lambda record: json.loads(bind(record).read_text())

    def verify_bindings():
        for record in bound.values():
            require(retained.record(record["path"]) == record,
                    f"bound artifact changed: {record['path']}")

    try:
        validation = validate_repository(out)
        phase = "resume_authentication"
        contract = read(retained.record(CONTRACT))
        require(contract["scope"] == SCOPE and contract["policy_id"] == gates.POLICY_ID
                and contract["candidate_id"] == "ace3-q24-s16-toward-zero-native-v1"
                and contract["state_id"] == "ace3-q24-software-paired-state-v1"
                and contract["normal_host_review"] == "REQUIRED"
                and contract["rtl_admissible"] is False and contract["policy_adopted"] is False,
                "L9-L23 contract scope mismatch")
        resume = authenticate_resume(resume_parent, bind, read)
        source_dir = out / "source"
        source_dir.mkdir()
        paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
                 if getattr(module, "__file__", None)
                 and str(module.__file__).endswith(".py")
                 and Path(module.__file__).resolve().is_relative_to(ROOT / "ace3/model")}
        paths.update((Path(__file__).resolve(), CONTRACT, gates.CONTRACT, runner.CONTRACT,
                      ROOT / "ace3/contracts/candidates/binary64_fp16_excess_v1.json",
                      ROOT / "tests/test_q24_s16_toward_zero_l9_l23_v1.py"))
        sources = []
        for index, path in enumerate(sorted(paths)):
            original = retained.record(path)
            bind(original)
            snapshot = source_dir / f"{index:03d}_{path.name}"
            shutil.copyfile(path, snapshot)
            snapshot_record = retained.record(snapshot)
            require(snapshot_record["sha256"] == original["sha256"], "source snapshot drift")
            sources.append({"original": original, "snapshot": snapshot_record})
        tools = {"python": retained.record(sys.executable), "python_version": sys.version,
                 "numpy": np.__version__, "torch": torch.__version__,
                 "safetensors": safetensors.__version__, "platform": platform.platform(),
                 "thread_environment": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        retained.write(out / "preexecution.json", {
            "contract": contract, "sources": sources, "tools": tools,
            "validation": validation, "resume": resume, "input_bindings": list(bound.values()),
            "command": [str(PYTHON), "-I", "-B", str(Path(__file__).resolve()),
                        "--out", str(out), "--resume-parent", str(resume_parent)],
            "cwd": str(ROOT), "repository_search_context": str(ROOT),
            "normal_host_review": "REQUIRED", "native_L0_L8_invocations": 0, "rtl_invocations": 0})
        phase = "independent_reference_extension"
        stamp = time.monotonic()
        extension_record, extension, model_record, control_record = prepare_references(out, bind, read)
        phase_times.append({"phase": phase, "seconds": time.monotonic() - stamp})
        verify_bindings()
        retained.write(out / "freeze.json", {
            "contract": contract, "sources": sources, "tools": tools,
            "preexecution": retained.record(out / "preexecution.json"),
            "input_bindings": list(bound.values()), "reference_extension": extension_record,
            "state_lineage": {"prior_residual": resume, "prior_kv": "own empty P0"},
            "global_reference_lineage": {"model": model_record, "control": control_record,
                                         "candidate_data_used": False},
            "normal_host_review": "REQUIRED", "native_L0_L8_invocations": 0, "rtl_invocations": 0})
        bind(retained.record(out / "freeze.json"))
        phase = "suffix_execution"
        status, layers, failure = execute_layers(out, contract, resume, extension, bind, verify_bindings)
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError, RuntimeError) as exc:
        failure = {"node": None, "index": None, "failure_taxonomy": phase,
                   "reason": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(failure), flush=True)
    result = {
        "status": status, "candidate_admitted": status == "PASS",
        "candidate_id": "ace3-q24-s16-toward-zero-native-v1", "policy_id": gates.POLICY_ID,
        "scope": SCOPE, "layers": layers, "first_failure": failure,
        "phase_times": phase_times, "seconds": time.monotonic() - started,
        "evidence_kind": "cpu_software_q24", "native_L0_L8_invocations": 0,
        "rtl_invocations": 0, "policy_adopted": False,
        "normal_host_review": "REQUIRED",
        "candidate_admission_scope": "bounded CPU numerical/state gates only; Host review still required",
        "claim_boundary": "L9-L23/P0 from reviewed L8 software parent only; "
                          "no production, strict-FP16-state, RTL, token or full-model admission"}
    retained.write(out / "result.json", result)
    print(json.dumps({"status": status, "first_failure": failure}), flush=True)
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-parent", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == ROOT / "build"
            and re.fullmatch(r"q24_software_l9_l23_[a-z0-9]+_attempt[0-9]+", out.name),
            "fresh ignored L9-L23 build attempt required")
    require(args.resume_parent.resolve() == SELECTED_PARENT, "only the selected L8 parent is supported")
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return run(out, args.resume_parent.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
