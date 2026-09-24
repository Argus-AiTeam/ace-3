"""Non-admitting L16-L23/P0 software suffix from the authenticated L15 cut.

The adjacent contract publishes the single repo-bound compile/test/run command.
Frozen ancestors and original-input references are consumed read-only.
"""

import argparse
from contextlib import ExitStack, contextmanager
import importlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l15_cut_l16_suffix_preflight_v1 as preflight


upstream = preflight.margin.upstream
native, retained, require = upstream.native, upstream.retained, upstream.require
local, gates = upstream.local, upstream.gates
ROOT, PYTHON = preflight.ROOT, preflight.PYTHON
NAME = "q24_s16_l15_cut_l16_suffix_v1"
ID = "ace3-q24-s16-l15-cut-l16-suffix-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = preflight.OUTPUT / "array_equality"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PINS = {
    "result.json": "9307ef34309418bd95da01b3b3843246f1190b915e4b489ab263a693547e294e",
    "validation.json": "5f5fc7b59fdbccb6d049433657048405ad097f0031f28a76bcea4b805908f6b6",
    "command.json": "529435becb3d49c62cbd4362abf4cd4b6f19661b709ca9e6266f667a8dd8d79b",
}
LAYERS = tuple(range(16, 24))
EXPECTED_TESTS = 20
FLAGS = dict(upstream.FLAGS)


def check_preflight(document, validation, command):
    require(document["diagnostic_id"] == preflight.ID
            and document["status"] == "PREFLIGHT_READY"
            and document["node"] == [15, 0, 18] and document["index"] == 62
            and document["prospective_next_layer"] == 16
            and document["upstream_sha256"] == preflight.PINS,
            "wrong accepted preflight dependency")
    preflight.check_flags(document)
    preflight.check_flags(validation)
    require(validation["collected"] == validation["executed"] == preflight.EXPECTED_TESTS
            and document["tests_executed"] == preflight.EXPECTED_TESTS
            and validation["failures"] == validation["errors"] == validation["skipped"] == 0
            and validation["legacy_19_tests_executed"] is False
            and validation["upstream_tests_executed"] is False,
            "invalid preflight validation counts")
    for context in (command, validation):
        require(context["cwd"] == str(ROOT) and context["executable"] == str(PYTHON)
                and context["PYTHONPATH"] == str(ROOT)
                and context["dont_write_bytecode"] is True,
                "invalid preflight command context")
    require(document["validation"] == retained.record(INPUT / "validation.json")
            and document["command"] == retained.record(INPUT / "command.json")
            and validation["contract"] == retained.record(preflight.CONTRACT)
            and command["argv"] ==
            [str(preflight.SOURCE), "--out", str(INPUT.relative_to(ROOT))],
            "preflight evidence linkage mismatch")


def authenticate():
    inputs = upstream.prior.BoundInputs()
    documents = {name: upstream.margin.pinned(inputs, INPUT / name, digest)
                 for name, digest in PINS.items()}
    document, validation = documents["result.json"], documents["validation.json"]
    check_preflight(document, validation, documents["command.json"])
    for record in document["authenticated_inputs"]:
        inputs.bind(record)
    preflight.bind_origins(inputs, validation["origins"])
    for record in (*validation["compiled"], validation["contract"]):
        inputs.bind(record)
    with preflight.no_execution():
        verified, reconstructed = preflight.preflight()
    require(all(document[key] == value for key, value in reconstructed.items()),
            "accepted preflight reconstruction mismatch")
    for record in verified.records.values():
        inputs.bind(record)
    arrays = inputs.archive(document["dependency"]["frozen_L15_actual"])
    parent = preflight.minimal_cut(
        retained.state_from(arrays, "output", "stage18"), document["minimal_residual_cut"])
    freeze = inputs.read(retained.record(upstream.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    upstream.bind_reference_inputs(inputs, extension)
    preflight.check_boundary(parent, {kind: np.empty((0, 128), dtype="<u2")
                                      for kind in ("k", "v")}, extension)
    return inputs, document, extension, parent


@contextmanager
def suffix_only(audit):
    raw = native.candidate._stages

    def forbidden(*args, **kwargs):
        raise RuntimeError("native replay, external execution or admission forbidden")

    def guarded(tensors, layer, parent, arrays):
        require(type(layer) is int and layer in LAYERS, "only native L16-L23 permitted")
        require(layer == 16 + len(audit["native_layers"]), "nonsequential native suffix")
        audit["native_layers"].append(layer)
        yield from raw(tensors, layer, parent, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates."):
                for attribute in ("stages", "continuation_stages", "native_layer", "run", "_stages"):
                    if callable(getattr(module, attribute, None)):
                        replacement = (guarded if module is native.candidate
                                       and attribute == "_stages" else forbidden)
                        stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def drive_layer(tensors, layer, parent, trajectory, reference, arrays, local_refs, reports):
    require(type(layer) is int and layer in LAYERS, "only native L16-L23 permitted")
    return upstream.drive_layer(tensors, layer, parent, trajectory, reference,
                                arrays, local_refs, reports)


def execute_suffix(out, inputs, extension, parent, audit, layers):
    parent_binding = retained.save(out / "counterfactual_L15_state.npz", parent)
    with upstream.safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
        with suffix_only(audit):
            for layer in LAYERS:
                item = extension["layers"][str(layer)]
                tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors, item["canonical"], layer)
                trajectory = inputs.archive(item["fp16"])
                reference = upstream.global_reference(inputs, item)
                retained.verify_parent(parent, parent)
                arrays, local_refs, reports = {}, {}, []
                directory = out / f"layer{layer:02d}"
                directory.mkdir()
                entry = {
                    "layer": layer, "position": 0, "input_state_evidence": parent_binding,
                    "original_reference": item, "status": "BLOCKED",
                    "prior_kv": "own empty P0", "prior_layer_kv_consumed": False,
                }
                layers.append(entry)
                try:
                    failure, timings = drive_layer(tensors, layer, parent, trajectory, reference,
                                                   arrays, local_refs, reports)
                    entry.update(status="FAIL" if failure else "PASS", **timings)
                finally:
                    entry["actual_stages"] = retained.save(directory / "actual_stages.npz", arrays)
                    entry["local_references"] = retained.save(
                        directory / "local_references.npz", local_refs)
                    retained.write(directory / "reports.json", reports)
                    entry["reports"] = retained.record(directory / "reports.json")
                print(json.dumps({"layer": layer, "status": entry["status"]}), flush=True)
                if failure:
                    failure["evidence"] = entry["reports"]
                    return failure
                parent = {key: value.copy() for key, value in
                          retained.state_from(arrays, "output", "stage18").items()}
                retained.verify_parent(parent, parent)
                parent_binding = retained.save(directory / "counterfactual_state.npz", parent)
                entry["output_state_evidence"] = parent_binding
                entry["own_kv_evidence"] = retained.save(directory / "own_kv.npz", {
                    kind: arrays["output_cache_" + kind] for kind in ("k", "v")})
    return None


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "candidate/contract origin mismatch")
    records = preflight.margin.origins()
    require(records[MODULE]["path"] == str(SOURCE)
            and records[TEST_MODULE]["path"] ==
            str(ROOT / "tests" / (TEST_MODULE.split(".")[-1] + ".py"))
            and upstream.prior.LEGACY_TEST in records, "required repository origin missing")
    return records


def check_contract(contract):
    require(contract["diagnostic_id"] == ID and contract["version"] == 1
            and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["preflight_sha256"] == PINS
            and contract["layers"] == list(LAYERS) and contract["parent_node"] == [15, 0, 18]
            and contract["focused_tests"] == EXPECTED_TESTS
            and contract["policy_id"] == gates.POLICY_ID and contract["excess_budget"] == "1/8",
            "versioned suffix contract mismatch")
    for key, value in FLAGS.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"changed non-admitting boundary: {key}")


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "published repo-bound command required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(upstream.prior.LEGACY_TEST)
    source_origins = origins()
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for index, path in enumerate((SOURCE, Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with (out / "unittest.log").open("x") as log:
        with preflight.no_execution():
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "dont_write_bytecode": sys.dont_write_bytecode,
        "origins": source_origins, "compiled": compiled, "contract": retained.record(CONTRACT),
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, "upstream_tests_executed": False,
        "native_layer_invocations": 0, **FLAGS,
    }
    retained.write(out / "validation.json", report)
    require(count == result.testsRun == EXPECTED_TESTS and result.wasSuccessful()
            and not result.skipped, "focused tests failed")
    return report


def output_path(path):
    out = path.absolute()
    require((out == OUTPUT or out.parent == OUTPUT) and out.resolve() == out,
            "output outside bounded writable scope")
    require(not out.exists(), "output already exists; preserve historical evidence")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out)
    out.mkdir(exist_ok=False)
    retained.write(out / "command.json", {
        "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
        "PYTHONPATH": os.environ.get("PYTHONPATH"), "dont_write_bytecode": sys.dont_write_bytecode})
    audit, layers, timings = {"native_layers": []}, [], {}
    result = {
        "diagnostic_id": ID, **FLAGS, "layers": layers, "audit": audit,
        "history": [9707], "position": 0, "retained_status": "DOWNSTREAM_GATE_FAIL",
        "preflight_sha256": PINS, "timing_seconds": timings,
        "reference_policy": "unchanged independently propagated original-input global reference",
        "claim_boundary": "Non-admitting bounded counterfactual CPU evidence only. Wide Q24 "
        "residuals are not strict-FP16-state W4A16. No new-token, full-model, RTL, simulation, "
        "synthesis, PPA, GPU, FPGA or hardware PASS. Historical failures and acceptances stand.",
    }
    phase, started = "compile_and_focused_tests", time.monotonic()
    try:
        validation = validate(out)
        result["validation"] = retained.record(out / "validation.json")
        result["command"] = retained.record(out / "command.json")
        timings[phase] = time.monotonic() - started
        phase, started = "authenticate_preflight_and_parent", time.monotonic()
        inputs, document, extension, parent = authenticate()
        for record in (*validation["origins"].values(), *validation["compiled"],
                       validation["contract"]):
            inputs.bind(record)
        result.update(preflight_dependency=retained.record(INPUT / "result.json"),
                      exact_L15_cut_parent=document["dependency"],
                      minimal_residual_cut=document["minimal_residual_cut"],
                      frozen_lineage=document["lineage"],
                      reconstructed_L15_global_gate=document["reconstructed_L15_global_gate"],
                      review_provenance=document["review_provenance"])
        timings[phase] = time.monotonic() - started
        phase, started = "native_suffix", time.monotonic()
        failure = execute_suffix(out, inputs, extension, parent, audit, layers)
        timings[phase] = time.monotonic() - started
        phase, started = "final_binding_integrity", time.monotonic()
        origins()
        for record in inputs.records.values():
            require(retained.record(record["path"]) == record,
                    f"input/source changed during diagnostic: {record['path']}")
        result.update(authenticated_inputs=list(inputs.records.values()), first_failure=failure,
                      tests_executed=validation["executed"],
                      status="DOWNSTREAM_GATE_FAIL" if failure else "BOUNDED_COUNTERFACTUAL_PASS")
        timings[phase] = time.monotonic() - started
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            StopIteration, RuntimeError, py_compile.PyCompileError) as exc:
        timings[phase] = time.monotonic() - started
        result.update(status="BLOCKED", phase=phase, error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr, flush=True)
    result["native_layer_invocations"] = len(audit["native_layers"])
    result["first_native_layer"] = audit["native_layers"][0] if audit["native_layers"] else None
    retained.write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in
                      ("status", "native_layer_invocations", "first_native_layer", *FLAGS)}),
          flush=True)
    return 1 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
