"""Read-only L21/P0/S18 margin diagnostic; never execution or admission.

The adjacent contract publishes the repository-bound compile/test/diagnose
command. Frozen local/global reports and paired state are authenticated before
an exact scalar boundary calculation; no residual state archive is published.
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

from ace3.model.candidates import diagnose_q24_s16_l15_cut_l16_suffix_v1 as upstream


margin = upstream.preflight.margin
scalar_math = margin.margin
retained, require, gates = upstream.retained, upstream.require, upstream.gates
rational, Q = margin.rational, margin.Q
ROOT, PYTHON = upstream.ROOT, upstream.PYTHON
NAME = "q24_s16_l21_s18_margin_sensitivity_v1"
ID = "ace3-q24-s16-l21-s18-margin-sensitivity-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = upstream.OUTPUT
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
LAYERS = tuple(range(16, 22))
EXPECTED_TESTS = 20
PINS = {
    "result.json": "6d827ad412e70d59ba7bfcb28b9da056fdfc66c75299852c2645eeeab684290b",
    "validation.json": "7555a88af5ec76546eb18288df2a9a2adbf12a8a2ceea2de5654093804c9e0ae",
    "command.json": "0bac35d13e1aeadb5989319cf0150abf32ac48fb8dd1bbb60b35eeed71d5cebc",
    "layer21/actual_stages.npz": "784c5af93e09f97fced8fae1fed66bcccc0ddb5f181f76bebeb9d7543ce66e49",
    "layer21/reports.json": "b713d6a112768889915172eaa74cf8fb49faf184a4ae4a903f59f2d2f467f30c",
}
FLAGS = {**margin.FLAGS, "native_invocations": 0}
no_execution = margin.no_execution


def check_upstream(document, validation, command):
    require(document["diagnostic_id"] == upstream.ID
            and document["status"] == document["retained_status"] == "DOWNSTREAM_GATE_FAIL"
            and document["history"] == [9707] and document["position"] == 0
            and document["audit"] == {"native_layers": list(LAYERS)}
            and document["first_native_layer"] == 16
            and type(document["native_layer_invocations"]) is int
            and document["native_layer_invocations"] == len(LAYERS)
            and document["preflight_sha256"] == upstream.PINS,
            "wrong frozen suffix scope")
    for key, value in upstream.FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed frozen boundary: {key}")
    require(document["tests_executed"] == validation["collected"]
            == validation["executed"] == upstream.EXPECTED_TESTS
            and validation["failures"] == validation["errors"] == validation["skipped"] == 0
            and validation["legacy_19_tests_executed"] is False
            and validation["upstream_tests_executed"] is False,
            "invalid frozen validation counts")
    for context in (command, validation):
        require(context["cwd"] == str(ROOT) and context["executable"] == str(PYTHON)
                and context["PYTHONPATH"] == str(ROOT)
                and context["dont_write_bytecode"] is True,
                "invalid frozen command context")
    require(document["validation"] == retained.record(INPUT / "validation.json")
            and document["command"] == retained.record(INPUT / "command.json")
            and validation["contract"] == retained.record(upstream.CONTRACT)
            and command["argv"] == [str(upstream.SOURCE), "--out", str(INPUT.relative_to(ROOT))],
            "frozen command/source linkage mismatch")


def select_failure(entries, reports_by_layer):
    require([entry["layer"] for entry in entries] == list(LAYERS),
            "unexpected frozen layer sequence")
    for entry in entries:
        layer = entry["layer"]
        require(entry["position"] == 0 and entry["prior_kv"] == "own empty P0"
                and entry["prior_layer_kv_consumed"] is False,
                "changed frozen P0 KV boundary")
        reports = reports_by_layer[layer]
        require(len(reports) == 19, "incomplete frozen stage reports")
        for stage, report in enumerate(reports):
            require(report["stage"] == stage and report["node"] == [layer, 0, stage]
                    and report["policy_id"] == gates.POLICY_ID
                    and report["residual_state_lineage"] == report["kv_lineage"] == "PASS",
                    "wrong frozen node, policy or lineage")
            mandatory = report["binary64_v1"] if stage == 18 else report["local_operator_fp16"]
            if layer == 21 and stage == 18:
                failures = mandatory["failures"]
                require(entry["status"] == report["status"] == "FAIL"
                        and mandatory["passed"] is False
                        and len(failures) == mandatory["failure_count"] == 1
                        and failures == [row for row in mandatory["rows"]
                                         if row["accepted"] is False],
                        "missing or inconsistent mandatory failure")
                first = failures[0]
                require(first["index"] == 62 and first["actual_fp16_bits"] == "55ef"
                        and first["nearest_fp16_bits"] == "55fb"
                        and first["excess_error"] == "3/4"
                        and first["excess_budget"] == "1/8",
                        "wrong first mandatory scalar failure")
                return {"node": [21, 0, 18], "index": 62, "gate": "binary64_v1",
                        "evidence": entry["reports"]}
            require(report["status"] == "PASS" and mandatory["passed"] is True,
                    "earlier mandatory gate failure")
        require(entry["status"] == "PASS", "earlier layer did not pass")
    raise ValueError("missing L21 first failure")


def check_reference(entry, item):
    require(entry["original_reference"] == item, "original reference substitution")


def sensitivity(integer, reference):
    require(type(integer) is int and type(reference) is Fraction
            and Fraction.from_float(float(reference)) == reference
            and 0 < reference < 65504, "exact interior binary64 scalar required")
    baseline_word = rational.project(integer, 0)
    baseline = scalar_math.scalar(baseline_word, reference)
    nearest = int(baseline["nearest_fp16_bits"], 16)
    require(baseline["accepted"] is False and 1 < nearest < 0x7BFE,
            "failing interior scalar required")
    first = last = nearest
    while scalar_math.scalar(first - 1, reference)["accepted"]:
        first -= 1
    while scalar_math.scalar(last + 1, reference)["accepted"]:
        last += 1
    # The gate is a closed real interval; positive FP16 words and their RNE
    # cells are ordered. Its closest passing cell proves global minimality.
    lower, _, minimum, _ = scalar_math.passing_cell(first)
    _, upper, _, maximum = scalar_math.passing_cell(last)
    adjusted = minimum if integer < minimum else maximum
    delta = adjusted - integer
    require(delta != 0, "baseline unexpectedly inside passing interval")
    one_less = adjusted - (1 if delta > 0 else -1)
    word, adjacent_word = rational.project(adjusted, 0), rational.project(one_less, 0)
    passing = scalar_math.scalar(word, reference)
    adjacent = scalar_math.scalar(adjacent_word, reference)
    require(passing["accepted"] is True and adjacent["accepted"] is False,
            "minimal Q24 one-unit boundary mismatch")
    radius = Fraction(baseline["q"]) + Fraction(1, 8)
    require(reference - radius <= rational.fp16_value(first)
            <= rational.fp16_value(last) <= reference + radius
            and rational.fp16_value(first - 1) < reference - radius
            and rational.fp16_value(last + 1) > reference + radius,
            "passing FP16 interval is not exhaustive")
    return {
        "reference_binary64_hex": float(reference).hex(),
        "reference_exact": str(reference), "baseline": baseline,
        "passing_FP16_interval": {
            "first_word": f"{first:04x}", "last_word": f"{last:04x}",
            "gate_lower": str(reference - radius), "gate_upper": str(reference + radius),
            "RNE_lower_midpoint": str(lower), "RNE_upper_midpoint": str(upper),
            "lower_inclusive": not bool(first & 1), "upper_inclusive": not bool(last & 1),
            "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
        },
        "minimal_residual_cut": {
            "baseline_Q24_integer": integer, "adjusted_Q24_integer": adjusted,
            "delta_Q24_units": delta, "delta": str(Fraction(delta, Q)),
            "output_word": f"{word:04x}", "scalar_gate": passing,
            "one_unit_less_delta_Q24_units": one_less - integer,
            "one_unit_less_Q24_integer": one_less,
            "one_unit_less_word": f"{adjacent_word:04x}", "one_unit_less_gate": adjacent,
            "scope": "scalar sensitivity only; no state is modified or published",
        },
    }


def vector_boundary(arrays, trajectory, reference, analysis):
    cut = analysis["minimal_residual_cut"]
    summaries = {}
    for name, word, accepted in (
        ("minimum", cut["output_word"], True),
        ("one_unit_less", cut["one_unit_less_word"], False),
    ):
        actual = arrays["stage18"].copy()
        actual[62] = int(word, 16)
        require(np.flatnonzero(actual != arrays["stage18"]).tolist() == [62],
                "counterfactual changed wrong coordinates")
        report = gates.evaluate_decoder_stage(
            stage=18, actual=actual, reference=trajectory["stage18"],
            policy=gates.POLICY_ID, local_reference=None, reference_binary64=reference)
        gate = report["binary64_v1"]
        require(gate["passed"] is accepted and (report["status"] == "PASS") is accepted
                and gate["failure_count"] == (0 if accepted else 1),
                "projected full-vector global boundary mismatch")
        summaries[name] = {
            "status": report["status"], "passed": gate["passed"],
            "coordinates": gate["coordinates"], "failure_count": gate["failure_count"],
            "failed_indices": [row["index"] for row in gate["failures"]],
            "changed_coordinates": [62],
        }
    return summaries


def authenticate():
    inputs = margin.prior.BoundInputs()
    documents = {name: scalar_math.pinned(inputs, INPUT / name, PINS[name])
                 for name in ("result.json", "validation.json", "command.json")}
    document, validation = documents["result.json"], documents["validation.json"]
    check_upstream(document, validation, documents["command.json"])
    for record in document["authenticated_inputs"]:
        inputs.bind(record)
    upstream.preflight.bind_origins(inputs, validation["origins"])
    for record in (*validation["compiled"], validation["contract"]):
        inputs.bind(record)
    for name, digest in PINS.items():
        record = retained.record(INPUT / name)
        require(record["sha256"] == digest, f"pinned evidence mismatch: {name}")
        inputs.bind(record)
    verified, preflight, extension, parent = upstream.authenticate()
    for record in verified.records.values():
        margin.bind_closure(inputs, record)
    require(document["preflight_dependency"] == retained.record(upstream.INPUT / "result.json")
            and document["minimal_residual_cut"] == preflight["minimal_residual_cut"]
            and document["exact_L15_cut_parent"] == preflight["dependency"]
            and document["frozen_lineage"] == preflight["lineage"]
            and document["reconstructed_L15_global_gate"]
            == preflight["reconstructed_L15_global_gate"],
            "changed frozen L15 cut dependency")
    entries, reports_by_layer = document["layers"], {}
    require([entry["layer"] for entry in entries] == list(LAYERS),
            "unexpected frozen suffix")
    parent_record = retained.record(INPUT / "counterfactual_L15_state.npz")
    retained.verify_parent(inputs.archive(parent_record), parent)
    for entry in entries:
        layer = entry["layer"]
        directory = INPUT / f"layer{layer}"
        item = extension["layers"][str(layer)]
        check_reference(entry, item)
        require(entry["input_state_evidence"] == parent_record,
                "paired I/Z/H parent evidence linkage mismatch")
        for key, name in (("actual_stages", "actual_stages.npz"),
                          ("local_references", "local_references.npz"),
                          ("reports", "reports.json")):
            require(entry[key] == retained.record(directory / name),
                    "selected frozen artifact substitution")
        arrays = inputs.archive(entry["actual_stages"])
        local_refs = inputs.archive(entry["local_references"])
        reports = inputs.read(entry["reports"])
        trajectory = inputs.archive(item["fp16"])
        reference = upstream.upstream.global_reference(inputs, item)
        margin.check_layer(arrays, parent)
        require(set(local_refs) == {f"stage{stage:02d}" for stage in range(18)}
                and len(reports) == 19, "incomplete frozen operator evidence")
        for stage in range(19):
            expected = local_refs[f"stage{stage:02d}"] if stage < 18 else None
            if stage == 12:
                require(np.array_equal(expected, retained.transition_reference(
                    parent, arrays["stage11"])["h"]), "changed exact S12 reference")
            report = gates.evaluate_decoder_stage(
                stage=stage, actual=arrays[f"stage{stage:02d}"],
                reference=trajectory[f"stage{stage:02d}"], policy=gates.POLICY_ID,
                local_reference=expected, reference_binary64=reference if stage == 18 else None)
            report.update(node=[layer, 0, stage], residual_state_lineage="PASS",
                          kv_lineage="PASS", local_reference_independent=stage < 18)
            require(report == reports[stage], f"frozen gate reproduction mismatch L{layer}/S{stage}")
        reports_by_layer[layer] = reports
        if layer < 21:
            parent_record = retained.record(directory / "counterfactual_state.npz")
            require(entry["output_state_evidence"] == parent_record,
                    "frozen output state substitution")
            parent = inputs.archive(parent_record)
            retained.verify_parent(parent, retained.state_from(arrays, "output", "stage18"))
            require(entry["own_kv_evidence"] == retained.record(directory / "own_kv.npz"),
                    "own KV evidence substitution")
            kv = inputs.archive(entry["own_kv_evidence"])
            require(set(kv) == {"k", "v"}, "invalid own KV archive")
            for kind in ("k", "v"):
                upstream.local.finite_words(kv[kind], (1, 128))
                require(np.array_equal(kv[kind], arrays["output_cache_" + kind]),
                        "own KV evidence mismatch")
        else:
            require(not any(key in entry for key in
                            ("output_state_evidence", "output_parent", "own_kv_evidence")),
                    "failed L21 published successor evidence")
    failure = select_failure(entries, reports_by_layer)
    require(failure == document["first_failure"], "wrong frozen first failure")
    analysis = sensitivity(int(arrays["output_i"][62]),
                           Fraction.from_float(float(reference[62])))
    require(all(analysis["baseline"][key] == value
                for key, value in reports[-1]["binary64_v1"]["failures"][0].items()
                if key != "index"), "independent scalar gate reproduction mismatch")
    analysis["projected_full_vector_gate"] = vector_boundary(arrays, trajectory, reference, analysis)
    return inputs, document, failure, analysis


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "candidate/contract origin mismatch")
    records = margin.origins()
    require(records[MODULE]["path"] == str(SOURCE)
            and records[TEST_MODULE]["path"] ==
            str(ROOT / "tests" / (TEST_MODULE.split(".")[-1] + ".py"))
            and margin.prior.LEGACY_TEST in records, "required repository origin missing")
    return records


def check_contract(contract):
    require(contract["diagnostic_id"] == ID and contract["version"] == 1
            and contract["focused_tests"] == EXPECTED_TESTS
            and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["upstream_sha256"] == PINS and contract["node"] == [21, 0, 18]
            and contract["index"] == 62 and contract["policy_id"] == gates.POLICY_ID
            and contract["excess_budget"] == "1/8" and contract["baseline_excess"] == "3/4",
            "versioned contract mismatch")
    for key, value in FLAGS.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"contract boundary mismatch: {key}")


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
        "contract_parsed": True, "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
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
    result = {"diagnostic_id": ID, **FLAGS, "timing_seconds": {}}
    phase, started = "compile_and_focused_tests", time.monotonic()
    try:
        with no_execution():
            validation = validate(out)
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "frozen_authentication_and_scalar_boundary", time.monotonic()
            inputs, document, failure, analysis = authenticate()
            result.update(analysis)
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "final_binding_integrity", time.monotonic()
            for record in (*origins().values(), validation["contract"]):
                inputs.bind(record)
            for record in inputs.records.values():
                require(retained.record(record["path"]) == record,
                        f"input/source changed during diagnostic: {record['path']}")
            result["timing_seconds"][phase] = time.monotonic() - started
        result.update(
            status="DIAGNOSED", node=[21, 0, 18], index=62, first_failure=failure,
            retained_status=document["status"], upstream_sha256=PINS,
            upstream_result=retained.record(INPUT / "result.json"),
            authenticated_inputs=list(inputs.records.values()),
            review_provenance=document["review_provenance"],
            lineage={"L15_cut_to_L21_input_IZH": "PASS", "own_empty_P0_FP16_KV": "PASS",
                     "S12_S18_exact_IZH": "PASS", "S16_retained_operand_RTZ": "PASS",
                     "original_input_reference_recurrence": "PASS"},
            reference_policy=document["reference_policy"],
            validation=retained.record(out / "validation.json"),
            command=retained.record(out / "command.json"), tests_executed=validation["executed"],
            claim_boundary="Non-admitting frozen Q24 CPU scalar diagnostic only. No native "
            "execution or state publication; not a propagated intervention or root-cause proof. "
            "No strict-FP16-state W4A16, new-token, full-model, RTL, simulation, synthesis, PPA, "
            "GPU, FPGA or hardware PASS. Historical failures and bounded acceptances stand.")
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, py_compile.PyCompileError) as exc:
        result.update(status="BLOCKED", phase=phase, error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    retained.write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", *FLAGS)}, sort_keys=True))
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
