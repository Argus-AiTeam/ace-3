"""Non-admitting L22-L23/P0 CPU suffix from the labeled L21 coordinate-62 cut.

The adjacent contract publishes the repository-bound compile/test/run command.
Accepted ancestors and independently propagated original references stay read-only.
"""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
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

from ace3.model.candidates import diagnose_q24_s16_l21_s18_margin_sensitivity_v1 as margin


upstream = margin.upstream
native, retained, require = upstream.native, upstream.retained, upstream.require
local, gates, rational = upstream.local, upstream.gates, margin.rational
ROOT, PYTHON = upstream.ROOT, upstream.PYTHON
NAME = "q24_s16_l21_coordinate62_adjusted_l22_l23_suffix_v1"
ID = "ace3-q24-s16-l21-coordinate62-adjusted-l22-l23-suffix-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = margin.OUTPUT / "array_equality"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PINS = {
    "result.json": "334b47246ce4f8d66257f1cdbe6265fe462da6eb58923e020b5c8ac4d2bcd5bd",
    "validation.json": "127757c73cf6784b1fa62f120c43479f42c70d9e9b6c6b2663810861f723cbdc",
    "command.json": "6bf2ebb19047236cf9976e46543c9d146d552f9cb487bc7cc9ba502ebe64aafc",
}
LAYERS = (22, 23)
DELTA = 10103041
EXPECTED_TESTS = 24
FLAGS = dict(upstream.FLAGS)
REFERENCE_POLICY = "unchanged independently propagated original-input global reference"


def check_margin(document, validation, command):
    require(document["diagnostic_id"] == margin.ID and document["status"] == "DIAGNOSED"
            and document["node"] == [21, 0, 18] and document["index"] == 62
            and document["retained_status"] == "DOWNSTREAM_GATE_FAIL"
            and document["upstream_sha256"] == margin.PINS,
            "wrong accepted L21 sensitivity dependency")
    for context in (document, validation):
        for key, value in margin.FLAGS.items():
            require(type(context[key]) is type(value) and context[key] == value,
                    f"changed frozen boundary: {key}")
    require(document["tests_executed"] == validation["collected"]
            == validation["executed"] == margin.EXPECTED_TESTS
            and validation["failures"] == validation["errors"] == validation["skipped"] == 0
            and validation["legacy_19_tests_executed"] is False
            and validation["upstream_tests_executed"] is False,
            "invalid frozen validation counts")
    for context in (validation, command):
        require(context["cwd"] == str(ROOT) and context["executable"] == str(PYTHON)
                and context["PYTHONPATH"] == str(ROOT)
                and context["dont_write_bytecode"] is True,
                "invalid frozen command context")
    require(document["validation"] == retained.record(INPUT / "validation.json")
            and document["command"] == retained.record(INPUT / "command.json")
            and validation["contract"] == retained.record(margin.CONTRACT)
            and command["argv"] == [str(margin.SOURCE), "--out", str(INPUT.relative_to(ROOT))]
            and document["upstream_result"] == retained.record(margin.INPUT / "result.json"),
            "frozen evidence linkage mismatch")
    require(document["reference_policy"] == REFERENCE_POLICY
            and document["lineage"] == {
                "L15_cut_to_L21_input_IZH": "PASS", "own_empty_P0_FP16_KV": "PASS",
                "S12_S18_exact_IZH": "PASS", "S16_retained_operand_RTZ": "PASS",
                "original_input_reference_recurrence": "PASS"},
            "changed frozen reference/state lineage")


def minimal_cut(parent, cut):
    retained.verify_parent(parent, parent)
    delta = cut["delta_Q24_units"]
    require(type(delta) is int and delta == DELTA
            and int(parent["i"][62]) == cut["baseline_Q24_integer"] == 1592645376
            and type(cut["adjusted_Q24_integer"]) is int
            and cut["adjusted_Q24_integer"] == 1602748417
            and Fraction(cut["delta"]) == Fraction(DELTA, margin.Q)
            and cut["output_word"] == "55f9"
            and cut["one_unit_less_delta_Q24_units"] == DELTA - 1
            and cut["one_unit_less_Q24_integer"] == 1602748416
            and cut["one_unit_less_word"] == "55f8"
            and cut["scalar_gate"]["accepted"] is True
            and cut["one_unit_less_gate"]["accepted"] is False,
            "wrong exact L21 coordinate-62 minimal adjustment")
    _, _, minimum, _ = margin.scalar_math.passing_cell(0x55F9)
    require(minimum == int(parent["i"][62]) + delta == 1602748417
            and rational.project(minimum, 0) == 0x55F9
            and rational.project(minimum - 1, 0) == 0x55F8,
            "wrong exclusive RNE midpoint ownership")
    changed = {key: value.copy() for key, value in parent.items()}
    changed["i"][62] = minimum
    changed["h"][62] = rational.project(minimum, int(changed["z"][62]))
    retained.verify_parent(changed, changed)
    require(np.array_equal(changed["z"], parent["z"])
            and np.flatnonzero(changed["i"] != parent["i"]).tolist() == [62]
            and np.flatnonzero(changed["h"] != parent["h"]).tolist() == [62],
            "adjustment changed another coordinate or zero-sign tag")
    return changed


def check_boundary(parent, kv, extension):
    retained.verify_parent(parent, parent)
    require(set(kv) == {"k", "v"}, "invalid own P0 KV keys")
    for kind in ("k", "v"):
        local.finite_words(kv[kind], (0, 128))
    upstream.upstream.check_reference_suffix(extension)
    for layer in LAYERS:
        previous, item = (extension["layers"][str(n)] for n in (layer - 1, layer))
        require(item["input_binary64"] == previous["binary64"]
                and item["input_fp16"] == previous["fp16"]
                and item["prior_kv"] == "own empty P0",
                "L22-L23 original reference recurrence mismatch")


def authenticate():
    inputs = margin.margin.prior.BoundInputs()
    documents = {name: margin.scalar_math.pinned(inputs, INPUT / name, digest)
                 for name, digest in PINS.items()}
    document, validation = documents["result.json"], documents["validation.json"]
    check_margin(document, validation, documents["command.json"])
    for record in document["authenticated_inputs"]:
        inputs.bind(record)
    upstream.preflight.bind_origins(inputs, validation["origins"])
    for record in (*validation["compiled"], validation["contract"]):
        inputs.bind(record)
    with margin.no_execution():
        verified, frozen, failure, analysis = margin.authenticate()
    for record in verified.records.values():
        margin.margin.bind_closure(inputs, record)
    require(document["first_failure"] == failure == frozen["first_failure"]
            and failure["node"] == [21, 0, 18] and failure["index"] == 62
            and failure["gate"] == "binary64_v1"
            and all(document[key] == value for key, value in analysis.items()),
            "accepted L21 failure/minimality reconstruction mismatch")
    arrays = inputs.archive(frozen["layers"][-1]["actual_stages"])
    parent = minimal_cut(retained.state_from(arrays, "output", "stage18"),
                         document["minimal_residual_cut"])
    freeze = inputs.read(retained.record(upstream.upstream.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    upstream.upstream.bind_reference_inputs(inputs, extension)
    check_boundary(parent, {kind: np.empty((0, 128), dtype="<u2")
                            for kind in ("k", "v")}, extension)
    return inputs, document, extension, parent


@contextmanager
def suffix_only(audit):
    raw = native.candidate._stages

    def forbidden(*args, **kwargs):
        raise RuntimeError("native replay, external execution or admission forbidden")

    def guarded(tensors, layer, parent, arrays):
        require(type(layer) is int and layer in LAYERS, "only native L22-L23 permitted")
        require(layer == 22 + len(audit["native_layers"]), "nonsequential native suffix")
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
    require(type(layer) is int and layer in LAYERS, "only native L22-L23 permitted")
    return upstream.drive_layer(tensors, layer, parent, trajectory, reference,
                                arrays, local_refs, reports)


def execute_suffix(out, inputs, extension, parent, audit, layers):
    parent_binding = retained.save(out / "counterfactual_L21_state.npz", parent)
    with upstream.upstream.safe_open(
            str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
        with suffix_only(audit):
            for layer in LAYERS:
                item = extension["layers"][str(layer)]
                tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors, item["canonical"], layer)
                trajectory = inputs.archive(item["fp16"])
                reference = upstream.upstream.global_reference(inputs, item)
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
    records = margin.margin.origins()
    require(records[MODULE]["path"] == str(SOURCE)
            and records[TEST_MODULE]["path"] ==
            str(ROOT / "tests" / (TEST_MODULE.split(".")[-1] + ".py"))
            and upstream.upstream.prior.LEGACY_TEST in records,
            "required repository origin missing")
    return records


def check_contract(contract):
    require(contract["diagnostic_id"] == ID and contract["version"] == 1
            and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["margin_sha256"] == PINS
            and contract["suffix_sha256"] == margin.PINS
            and contract["layers"] == list(LAYERS) and contract["parent_node"] == [21, 0, 18]
            and contract["index"] == 62 and type(contract["delta_Q24_units"]) is int
            and contract["delta_Q24_units"] == DELTA
            and contract["focused_tests"] == EXPECTED_TESTS
            and contract["policy_id"] == gates.POLICY_ID and contract["excess_budget"] == "1/8"
            and contract["reference_policy"] == REFERENCE_POLICY,
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
    importlib.import_module(upstream.upstream.prior.LEGACY_TEST)
    source_origins = origins()
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for index, path in enumerate((SOURCE, Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with (out / "unittest.log").open("x") as log:
        with margin.no_execution():
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "dont_write_bytecode": sys.dont_write_bytecode,
        "origins": source_origins, "compiled": compiled, "contract": retained.record(CONTRACT),
        "contract_parsed": True, "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
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
        "margin_sha256": PINS, "suffix_sha256": margin.PINS, "timing_seconds": timings,
        "reference_policy": REFERENCE_POLICY,
        "intervention": {"node": [21, 0, 18], "index": 62, "delta_Q24_units": DELTA,
                         "label": "counterfactual only; frozen native L21 failure unchanged"},
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
        phase, started = "authenticate_frozen_L21_and_minimal_adjustment", time.monotonic()
        with margin.no_execution():
            inputs, document, extension, parent = authenticate()
        for record in (*validation["origins"].values(), *validation["compiled"],
                       validation["contract"]):
            inputs.bind(record)
        result.update(margin_dependency=retained.record(INPUT / "result.json"),
                      suffix_dependency=document["upstream_result"],
                      frozen_first_failure=document["first_failure"],
                      minimal_residual_cut=document["minimal_residual_cut"],
                      projected_L21_global_gate=document["projected_full_vector_gate"],
                      frozen_lineage=document["lineage"],
                      review_provenance=document["review_provenance"])
        timings[phase] = time.monotonic() - started
        phase, started = "native_L22_L23_suffix", time.monotonic()
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
