"""Non-admitting L14-L23/P0 suffix after the reviewed minimal L13 Q24 cut.

The adjacent contract publishes the repo-bound command and reproduction scope.
Frozen L0-L13 evidence is read-only; no software-parent receipt is published.
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
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_s18_margin_sensitivity_v1 as margin
from ace3.model.candidates import run_q24_s16_toward_zero_l9_l23_v1 as native


ROOT, PYTHON = margin.ROOT, margin.PYTHON
NAME = "q24_s16_l13_minimal_cut_l14_suffix_v1"
ID = "ace3-q24-s16-l13-minimal-cut-l14-suffix-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
MARGIN = ROOT / "build/q24_s16_l13_s18_margin_sensitivity_v1_attempt001"
MARGIN_SHA = "5acc6ca0c7bc0765f1aa373b953715c1da743b8b3bca3a9d9a19d4ce33b591ef"
MARGIN_VALIDATION_SHA = "a9307433315ef1782ed759774d5db6760aef3c28db69d7379fa5163e5bc93664"
LAYERS = tuple(range(14, 24))
EXPECTED_TESTS = 18
prior, retained, require = margin.prior, margin.retained, margin.require
local, gates = native.local, native.gates
FLAGS = {
    "native_L0_L8_invocations": 0, "rtl_invocations": 0,
    "candidate_admitted": False, "policy_adopted": False,
    "successor_published": False, "normal_host_review": "REQUIRED",
}


@contextmanager
def suffix_only(audit):
    raw = native.candidate._stages

    def forbidden(*args, **kwargs):
        raise RuntimeError("native replay, external execution or publication forbidden")

    def guarded(tensors, layer, parent, arrays):
        require(type(layer) is int and layer in LAYERS,
                "only native L14-L23/P0 suffix execution is permitted")
        require(layer == 14 + len(audit["native_layers"]), "nonsequential native suffix")
        audit["native_layers"].append(layer)
        yield from raw(tensors, layer, parent, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if not name.startswith("ace3.model.candidates."):
                continue
            for attribute in ("stages", "continuation_stages", "native_layer", "run", "_stages"):
                if callable(getattr(module, attribute, None)):
                    replacement = (guarded if module is native.candidate and attribute == "_stages"
                                   else forbidden)
                    stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def check_reference_suffix(extension):
    prior.check_reference_chain(extension)
    previous = extension["layers"]["13"]
    for layer in LAYERS:
        item = extension["layers"][str(layer)]
        require(item["input_binary64"] == previous["binary64"]
                and item["input_fp16"] == previous["fp16"]
                and item["prior_kv"] == "own empty P0",
                f"original reference recurrence/KV mismatch at L{layer}")
        previous = item


def check_margin(document, validation):
    require(document["diagnostic_id"] == margin.ID and document["status"] == "DIAGNOSED"
            and document["node"] == [13, 0, 18] and document["index"] == 62
            and document["retained_status"] == "FAIL", "wrong reviewed margin result")
    for key, value in margin.FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed reviewed margin boundary: {key}")
    require(validation["collected"] == validation["executed"] == margin.EXPECTED_TESTS
            and validation["errors"] == validation["failures"] == validation["skipped"] == 0,
            "invalid reviewed margin validation")
    require(document["validation"] == retained.record(MARGIN / "validation.json"),
            "margin validation linkage mismatch")


def bind_reference_inputs(inputs, extension):
    check_reference_suffix(extension)
    for key in ("checkpoint", "original_binary64_parent", "original_fp16_parent",
                "original_specification", "input_freeze"):
        inputs.bind(extension[key])
    for record in extension["generators"]:
        inputs.bind(record)
    native.bind_tree(extension["layers"], inputs.bind)


def authenticate():
    inputs = prior.BoundInputs()
    document = margin.pinned(inputs, MARGIN / "result.json", MARGIN_SHA)
    validation = margin.pinned(inputs, MARGIN / "validation.json", MARGIN_VALIDATION_SHA)
    check_margin(document, validation)
    for record in document["authenticated_inputs"]:
        inputs.bind(record)
    for record in validation["origins"].values():
        inputs.bind(record)
    inputs.bind(validation["contract"])
    for record in validation["compiled"]:
        inputs.bind(record)
    with margin.no_execution():
        attribution, exact, lineage, bindings = margin.authenticate(margin.INPUT, margin.ATTRIBUTION)
    for record in bindings:
        inputs.bind(record)
    require(exact == document["exact_coordinate_decomposition"]
            and lineage == document["lineage"], "reviewed margin operand/lineage mismatch")
    freeze = inputs.read(retained.record(margin.INPUT / "freeze.json"))
    result = inputs.read(retained.record(margin.INPUT / "result.json"))
    extension = inputs.read(freeze["reference_extension"])
    bind_reference_inputs(inputs, extension)
    arrays = inputs.archive(attribution["operand_bindings"]["L13_actual"])
    require(attribution["operand_bindings"]["L13_actual"] == result["layers"][-1]["actual_stages"]
            and result["layers"][-1]["layer"] == 13
            and result["layers"][-1]["status"] == "FAIL"
            and "output_parent" not in result["layers"][-1],
            "not the frozen failed actual L13 parent")
    parent = retained.state_from(arrays, "output", "stage18")
    retained.verify_parent(parent, parent)
    return inputs, document, extension, parent, lineage


def minimal_cut(parent, cut):
    retained.verify_parent(parent, parent)
    integer = int(parent["i"][62])
    delta = cut["delta_Q24_units"]
    require(type(delta) is int and delta < 0
            and integer == cut["baseline_Q24_integer"]
            and Fraction(delta, 1 << 24) == Fraction(cut["delta"])
            and cut["output_word"] == "662f"
            and cut["one_Q24_unit_less_reduction_word"] == "6630",
            "wrong reviewed coordinate-62 cut")
    _, _, minimum, maximum = margin.passing_cell(0x662F)
    require(integer + delta == maximum and minimum <= maximum
            and margin.rational.project(integer, 0) == 0x6630
            and margin.rational.project(maximum + 1, 0) == 0x6630,
            "cut is not the exact minimal Q24 decrement")
    changed = {key: value.copy() for key, value in parent.items()}
    changed["i"][62] = maximum
    changed["h"][62] = margin.rational.project(maximum, int(changed["z"][62]))
    retained.verify_parent(changed, changed)
    require(np.array_equal(changed["z"], parent["z"])
            and np.flatnonzero(changed["i"] != parent["i"]).tolist() == [62]
            and np.flatnonzero(changed["h"] != parent["h"]).tolist() == [62],
            "cut altered another coordinate or zero-sign tag")
    return changed


def global_reference(inputs, item):
    reference = np.load(inputs.bind(item["binary64"]), allow_pickle=False)
    require(reference.dtype == np.dtype("<f8") and reference.shape == (896,)
            and np.all(np.isfinite(reference)) and np.all(np.abs(reference) <= 65504),
            "invalid independently propagated original-input reference")
    return reference


def drive_layer(tensors, layer, parent, trajectory, reference, arrays, local_refs, reports):
    require(type(layer) is int and layer in LAYERS, "suffix layer outside L14-L23")
    producer = native.candidate._stages(tensors, layer, parent, arrays)
    timings = {"native_seconds": 0.0, "oracle_seconds": 0.0}
    try:
        for stage in range(19):
            started = time.monotonic()
            require(next(producer) == stage, "out-of-order native stage")
            timings["native_seconds"] += time.monotonic() - started
            started = time.monotonic()
            expected = retained.check_stage_state(stage, arrays, parent)
            if stage == 16:
                require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                                       arrays["stage16"]), "changed S16 operand RTZ")
            if stage < 18 and stage != 12:
                expected = local.local_reference(
                    stage, {key: arrays[key].copy() for key in local.OPERANDS[stage]},
                    tensors, layer)
            if stage < 18:
                local_refs[f"stage{stage:02d}"] = expected
            report = gates.evaluate_decoder_stage(
                stage=stage, actual=arrays[f"stage{stage:02d}"],
                reference=trajectory[f"stage{stage:02d}"], policy=gates.POLICY_ID,
                local_reference=expected, reference_binary64=reference if stage == 18 else None)
            report.update(node=[layer, 0, stage], residual_state_lineage="PASS",
                          kv_lineage="PASS", local_reference_independent=stage < 18)
            reports.append(report)
            timings["oracle_seconds"] += time.monotonic() - started
            if report["status"] != "PASS":
                require(report["status"] == "FAIL", "mandatory gate blocked")
                return {"node": [layer, 0, stage], "index": native.first_failure_index(report),
                        "gate": "binary64_v1" if stage == 18 else "local_operator_fp16"}, timings
    finally:
        producer.close()
    return None, timings


def execute_suffix(out, inputs, extension, parent, audit, layers):
    parent_binding = retained.save(out / "counterfactual_L13_state.npz", parent)
    with safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
        with suffix_only(audit):
            for layer in LAYERS:
                item = extension["layers"][str(layer)]
                tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors, item["canonical"], layer)
                trajectory = inputs.archive(item["fp16"])
                reference = global_reference(inputs, item)
                retained.verify_parent(parent, parent)
                arrays, local_refs, reports = {}, {}, []
                directory = out / f"layer{layer:02d}"
                directory.mkdir()
                entry = {"layer": layer, "position": 0, "input_state_evidence": parent_binding,
                         "original_reference": item, "status": "BLOCKED",
                         "prior_kv": "own empty P0", "prior_layer_kv_consumed": False}
                layers.append(entry)
                try:
                    failure, timings = drive_layer(tensors, layer, parent, trajectory, reference,
                                                   arrays, local_refs, reports)
                    entry.update(status="FAIL" if failure else "PASS", **timings)
                finally:
                    entry["actual_stages"] = retained.save(directory / "actual_stages.npz", arrays)
                    entry["local_references"] = retained.save(directory / "local_references.npz",
                                                              local_refs)
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
                entry["own_kv_evidence"] = retained.save(directory / "own_kv.npz",
                    {kind: arrays["output_cache_" + kind] for kind in ("k", "v")})
    return None


def origins():
    records = {}
    for name, module in sorted(list(sys.modules.items())):
        if name == "ace3" or name.startswith("ace3.") or name in (
                "tests", TEST_MODULE, prior.LEGACY_TEST):
            path = getattr(module, "__file__", None)
            expected = ROOT.joinpath(*name.split("."))
            if path is None:
                paths = list(getattr(module, "__path__", ()))
                require(paths and all(Path(p).resolve() == expected for p in paths),
                        f"namespace origin mismatch: {name}")
                continue
            expected = expected / "__init__.py" if hasattr(module, "__path__") else expected.with_suffix(".py")
            require(Path(path).resolve() == expected, f"module origin mismatch: {name}")
            records[name] = retained.record(expected)
    return records


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT) and sys.dont_write_bytecode
            and not sys.flags.optimize, "published repo-bound command required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(prior.LEGACY_TEST)
    source_origins = origins()
    compiled = []
    for index, path in enumerate((ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
                                  Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["layers"] == list(LAYERS)
            and contract["focused_tests"] == EXPECTED_TESTS
            and contract["margin_sha256"] == MARGIN_SHA
            and contract["margin_validation_sha256"] == MARGIN_VALIDATION_SHA
            and contract["upstream_result_sha256"] == prior.RESULT_SHA
            and contract["attribution_sha256"] == margin.ATTRIBUTION_SHA,
            "versioned contract mismatch")
    for key, value in FLAGS.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"contract boundary mismatch: {key}")
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with (out / "unittest.log").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "contract": retained.record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, "native_layer_invocations": 0, **FLAGS,
    }
    retained.write(out / "validation.json", report)
    require(count == result.testsRun == EXPECTED_TESTS and result.wasSuccessful()
            and not result.skipped, "focused tests failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.absolute()
    require((out == OUTPUT or out.parent == OUTPUT) and out.resolve() == out,
            "output outside bounded writable scope")
    out.mkdir(exist_ok=False)
    retained.write(out / "command.json", {
        "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")})
    audit, layers, timings = {"native_layers": []}, [], {}
    phase, started = "compile_and_focused_tests", time.monotonic()
    result = {"diagnostic_id": ID, **FLAGS, "layers": layers, "audit": audit,
              "retained_status": "FAIL", "history": [9707], "position": 0,
              "reference_policy": "unchanged independently propagated original-input global reference",
              "timing_seconds": timings,
              "claim_boundary": "Bounded non-admitting CPU counterfactual only. Wide Q24 residual "
              "state is not strict-FP16-state W4A16. No token, full-model, RTL or hardware PASS."}
    try:
        validation = validate(out)
        result["validation"] = retained.record(out / "validation.json")
        timings[phase] = time.monotonic() - started
        phase, started = "authentication", time.monotonic()
        inputs, document, extension, frozen, lineage = authenticate()
        for record in validation["origins"].values():
            inputs.bind(record)
        inputs.bind(validation["contract"])
        result.update(upstream_results={
            "attempt": retained.record(margin.INPUT / "result.json"),
            "attribution": retained.record(margin.ATTRIBUTION / "result.json"),
            "margin": retained.record(MARGIN / "result.json")},
            frozen_lineage=lineage, minimal_residual_cut=document["minimal_residual_cut"],
            review_provenance=document["review_provenance"])
        parent = minimal_cut(frozen, document["minimal_residual_cut"])
        original = extension["layers"]["13"]
        cut_gate = gates.evaluate_decoder_stage(
            stage=18, actual=parent["h"], reference=inputs.archive(original["fp16"])["stage18"],
            policy=gates.POLICY_ID, local_reference=None,
            reference_binary64=global_reference(inputs, original))
        result["counterfactual_L13_global_gate"] = cut_gate
        require(cut_gate["status"] == "PASS", "minimal cut does not pass full-vector L13 global gate")
        timings[phase] = time.monotonic() - started
        phase, started = "native_suffix", time.monotonic()
        failure = execute_suffix(out, inputs, extension, parent, audit, layers)
        timings[phase] = time.monotonic() - started
        phase, started = "final_binding_integrity", time.monotonic()
        for record in inputs.records.values():
            require(retained.record(record["path"]) == record, "input/source changed during diagnostic")
        result["authenticated_inputs"] = list(inputs.records.values())
        result["first_failure"] = failure
        result["status"] = "DOWNSTREAM_GATE_FAIL" if failure else "BOUNDED_COUNTERFACTUAL_PASS"
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
