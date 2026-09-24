"""Repo-bound, read-only L15 minimal-cut preflight for a prospective L16/P0.

The adjacent contract documents the single compile/test/preflight command.
The reconstructed counterfactual stays in memory: no native execution, saved
parent, admission receipt, reference propagation, or suffix publication.
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

from ace3.model.candidates import diagnose_q24_s16_l15_s18_margin_sensitivity_v1 as margin


ROOT, PYTHON, Q = margin.ROOT, margin.PYTHON, margin.Q
retained, require, rational = margin.retained, margin.require, margin.rational
NAME = "q24_s16_l15_cut_l16_suffix_preflight_v1"
ID = "ace3-q24-s16-l15-cut-l16-suffix-preflight-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = margin.OUTPUT / "closure_signature"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PINS = {
    "result.json": "237ccabf99aec92b4d724b1c3a7aba2d873dbd137b628f7d0d2af235a9ed42a6",
    "validation.json": "6e2c6df460608ae15b2e0ba3525128a502d1240d0d3558c37b7c8d81ccf8dc9f",
    "command.json": "19e4a787281b4c30153e67e9b4e7806cd5dc4b2dd79df60aad4b799dfc66c935",
}
FLAGS = dict(margin.FLAGS)
EXPECTED_TESTS = 20
no_execution = margin.no_execution


def check_flags(document):
    for key, value in FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed non-admitting boundary: {key}")


def check_margin(document, validation, command):
    require(document["diagnostic_id"] == margin.ID and document["status"] == "DIAGNOSED"
            and document["node"] == [15, 0, 18] and document["index"] == 62
            and document["retained_status"] == "DOWNSTREAM_GATE_FAIL"
            and document["upstream_sha256"] == margin.PINS,
            "wrong corrected L15 sensitivity evidence")
    check_flags(document)
    check_flags(validation)
    require(validation["collected"] == validation["executed"] == margin.EXPECTED_TESTS
            and validation["failures"] == validation["errors"] == validation["skipped"] == 0
            and validation["legacy_19_tests_executed"] is False
            and document["tests_executed"] == margin.EXPECTED_TESTS,
            "invalid corrected validation counts")
    for context in (validation, command):
        require(context["cwd"] == str(ROOT) and context["executable"] == str(PYTHON)
                and context["PYTHONPATH"] == str(ROOT)
                and context["dont_write_bytecode"] is True,
                "invalid corrected repo-bound command context")
    require(document["validation"] == retained.record(INPUT / "validation.json")
            and document["command"] == retained.record(INPUT / "command.json")
            and validation["contract"] == retained.record(margin.CONTRACT)
            and command["argv"] == [str(margin.SOURCE), "--out", str(INPUT.relative_to(ROOT))],
            "corrected evidence linkage mismatch")
    require(document["reference_policy"] ==
            "unchanged independently propagated original-input global reference"
            and document["lineage"] == {
                "L14_output_to_L15_input_IZH": "PASS", "L14_own_KV": "PASS",
                "L15_empty_prior_and_own_KV": "PASS", "S12_S18_exact_IZH": "PASS",
                "S16_retained_operand_RTZ": "PASS",
                "original_input_reference_recurrence": "PASS"},
            "changed corrected reference/state lineage")


def bind_origins(inputs, origins):
    for name, record in origins.items():
        require(name == "ace3" or name.startswith("ace3.")
                or name == "tests" or name.startswith("tests."),
                f"unexpected repository module: {name}")
        path = ROOT.joinpath(*name.split("."))
        require(record["path"] in (str(path.with_suffix(".py")), str(path / "__init__.py")),
                f"module origin mismatch: {name}")
        inputs.bind(record)


def origins():
    require(Path(__file__).resolve() == SOURCE, "candidate source origin mismatch")
    require(CONTRACT.resolve() == CONTRACT, "contract origin mismatch")
    records = margin.origins()
    require(records[MODULE]["path"] == str(SOURCE)
            and records[TEST_MODULE]["path"] ==
            str(ROOT / "tests" / (TEST_MODULE.split(".")[-1] + ".py"))
            and margin.prior.LEGACY_TEST in records, "required module origin missing")
    return records


def minimal_cut(parent, cut):
    retained.verify_parent(parent, parent)
    integer, delta = int(parent["i"][62]), cut["delta_Q24_units"]
    require(type(delta) is int and delta == -365567
            and integer == cut["baseline_Q24_integer"] == 26617418751
            and Fraction(cut["delta"]) == Fraction(delta, Q)
            and cut["output_word"] == "6632"
            and cut["one_Q24_unit_less_reduction_word"] == "6633"
            and cut["continuous_cut_is_strict"] is False,
            "wrong exact L15 coordinate-62 minimal cut")
    _, _, minimum, maximum = margin.margin.passing_cell(0x6632)
    require(integer + delta == maximum == 26617053184 and minimum <= maximum
            and rational.project(integer, 0) == 0x6633
            and rational.project(maximum, 0) == 0x6632
            and rational.project(maximum + 1, 0) == 0x6633,
            "cut does not reach the inclusive RNE boundary minimally")
    changed = {key: value.copy() for key, value in parent.items()}
    changed["i"][62] = maximum
    changed["h"][62] = rational.project(maximum, int(changed["z"][62]))
    retained.verify_parent(changed, changed)
    require(np.array_equal(changed["z"], parent["z"])
            and np.flatnonzero(changed["i"] != parent["i"]).tolist() == [62]
            and np.flatnonzero(changed["h"] != parent["h"]).tolist() == [62],
            "cut changed another coordinate or zero-sign tag")
    return changed


def check_boundary(parent, kv, extension):
    retained.verify_parent(parent, parent)
    require(set(kv) == {"k", "v"}, "invalid prospective P0 KV keys")
    for kind in ("k", "v"):
        margin.local.finite_words(kv[kind], (0, 128))
    margin.upstream.check_reference_suffix(extension)
    l15, l16 = (extension["layers"][str(layer)] for layer in (15, 16))
    require(l16["input_binary64"] == l15["binary64"]
            and l16["input_fp16"] == l15["fp16"]
            and l16["prior_kv"] == "own empty P0",
            "prospective L16 original-input reference recurrence mismatch")


def preflight():
    inputs = margin.prior.BoundInputs()
    documents = {name: margin.margin.pinned(inputs, INPUT / name, digest)
                 for name, digest in PINS.items()}
    document, validation = documents["result.json"], documents["validation.json"]
    check_margin(document, validation, documents["command.json"])
    for record in document["authenticated_inputs"]:
        inputs.bind(record)
    bind_origins(inputs, validation["origins"])
    for record in (*validation["compiled"], validation["contract"]):
        inputs.bind(record)

    # Reuse the read-only operand/lineage checks, never the producer or its CLI.
    verified, suffix, failure, exact = margin.authenticate()
    for record in verified.records.values():
        margin.bind_closure(inputs, record)
    require(failure == document["first_failure"], "changed frozen first failure")
    analysis = margin.sensitivity(exact)
    require(all(document[key] == value for key, value in analysis.items()),
            "corrected sensitivity does not match authenticated operands")
    freeze = inputs.read(retained.record(margin.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    for key in ("checkpoint", "original_binary64_parent", "original_fp16_parent",
                "original_specification", "input_freeze"):
        margin.bind_closure(inputs, extension[key])
    for record in extension["generators"]:
        margin.bind_closure(inputs, record)
    entry = suffix["layers"][-1]
    require(entry["layer"] == 15 and entry["position"] == 0
            and entry["status"] == "FAIL", "not the frozen failed L15 output")
    arrays = inputs.archive(entry["actual_stages"])
    parent = retained.state_from(arrays, "output", "stage18")
    changed = minimal_cut(parent, document["minimal_residual_cut"])
    kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
    check_boundary(changed, kv, extension)
    reference = margin.upstream.global_reference(inputs, extension["layers"]["15"])
    require(Fraction.from_float(float(reference[62])) ==
            Fraction(exact["original_L15_reference"]), "original reference scalar mismatch")
    trajectory = inputs.archive(extension["layers"]["15"]["fp16"])
    report = margin.gates.evaluate_decoder_stage(
        stage=18, actual=changed["h"], reference=trajectory["stage18"],
        policy=margin.gates.POLICY_ID, local_reference=None, reference_binary64=reference)
    require(report["binary64_v1"]["passed"] is True and report["status"] == "PASS",
            "reconstructed L15 cut still fails the unchanged global gate")
    return inputs, {
        "status": "PREFLIGHT_READY",
        "node": [15, 0, 18], "index": 62, "history": [9707], "position": 0,
        "prospective_next_layer": 16, "retained_status": "DOWNSTREAM_GATE_FAIL",
        "upstream_result": retained.record(INPUT / "result.json"),
        "first_failure": failure, "minimal_residual_cut": document["minimal_residual_cut"],
        "dependency": {
            "kind": "non_admitting_in_memory_cut_recipe",
            "frozen_L15_actual": entry["actual_stages"],
            "state_keys": ["i", "z", "h"], "shape": [896],
            "dtypes": {"i": "<i8", "z": "|u1", "h": "<u2"},
            "coordinate": 62, "baseline_Q24_integer": int(parent["i"][62]),
            "delta_Q24_units": -365567, "prospective_Q24_integer": int(changed["i"][62]),
            "baseline_FP16_word": "6633", "prospective_FP16_word": "6632",
            "changed_I_coordinates": [62], "changed_H_coordinates": [62],
            "all_Z_and_other_coordinates_unchanged": True,
            "state_archive_published": False, "execution_authorized": False,
            "prior_kv": "own empty P0", "kv_shape": [0, 128], "kv_dtype": "<u2",
            "prior_layer_kv_consumed": False,
            "original_L15_reference": extension["layers"]["15"],
            "original_L16_reference": extension["layers"]["16"],
        },
        "lineage": {**document["lineage"], "L15_cut_IZH": "PASS",
                    "L16_empty_prior_KV": "PASS", "L15_to_L16_original_reference": "PASS"},
        "reconstructed_L15_global_gate": report["binary64_v1"],
        "original_FP16_trajectory_diagnostic": report["fp16"],
        "reference_policy": document["reference_policy"],
        "review_provenance": document["review_provenance"],
        "claim_boundary": "Non-admitting Q24 CPU preflight only, not native L16 execution or "
                          "a saved successor. Mandatory independent Host review remains required. "
                          "No strict-FP16-state W4A16, new-token, full-model, RTL, simulation, "
                          "synthesis, PPA, GPU, FPGA or hardware PASS. Historical failures and "
                          "bounded Reviewer acceptances are unchanged.",
    }


def check_contract(contract):
    require(contract["diagnostic_id"] == ID and contract["version"] == 1
            and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["upstream_sha256"] == PINS
            and contract["focused_tests"] == EXPECTED_TESTS
            and contract["policy_id"] == margin.gates.POLICY_ID
            and contract["excess_budget"] == "1/8"
            and contract["node"] == [15, 0, 18] and contract["index"] == 62
            and contract["prospective_next_layer"] == 16,
            "versioned preflight contract mismatch")
    check_flags(contract)


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "published repo-bound command required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(margin.prior.LEGACY_TEST)
    source_origins = origins()
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for index, path in enumerate((SOURCE, Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with (out / "unittest.log").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "dont_write_bytecode": sys.dont_write_bytecode,
        "origins": source_origins, "compiled": compiled, "contract": retained.record(CONTRACT),
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, "upstream_tests_executed": False, **FLAGS,
    }
    retained.write(out / "validation.json", report)
    require(count == result.testsRun == EXPECTED_TESTS and result.wasSuccessful()
            and not result.skipped, "focused tests failed")
    return report


def output_path(path):
    out = path.absolute()
    require((out == OUTPUT or out.parent == OUTPUT) and out.resolve() == out,
            "output outside bounded writable scope")
    require(not out.exists(), "output already exists; historical evidence is read-only")
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
    result = {"diagnostic_id": ID, **FLAGS, "timing_seconds": {}}
    phase, started = "compile_and_focused_tests", time.monotonic()
    try:
        with no_execution():
            validation = validate(out)
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "authenticate_and_reconstruct", time.monotonic()
            inputs, diagnostic = preflight()
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "final_binding_integrity", time.monotonic()
            for record in (*validation["origins"].values(), *validation["compiled"],
                           validation["contract"]):
                inputs.bind(record)
            origins()
            for record in inputs.records.values():
                require(retained.record(record["path"]) == record,
                        f"input/source changed during preflight: {record['path']}")
            result["timing_seconds"][phase] = time.monotonic() - started
        result.update(diagnostic, authenticated_inputs=list(inputs.records.values()),
                      validation=retained.record(out / "validation.json"),
                      command=retained.record(out / "command.json"),
                      tests_executed=validation["executed"], upstream_sha256=PINS)
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, py_compile.PyCompileError) as exc:
        result.update(status="BLOCKED", phase=phase, error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    retained.write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", *FLAGS)}, sort_keys=True))
    return 0 if result["status"] == "PREFLIGHT_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
