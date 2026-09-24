"""Frozen L15/P0/S18 scalar sensitivity, never native execution or admission.

The adjacent contract publishes the repository-bound verification command.
Only the selected scientific_binding result and its authenticated closure are
consumed. Compilation and focused tests precede one read-only diagnostic.
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

from ace3.model.candidates import diagnose_q24_s16_l13_minimal_cut_l14_suffix_v1 as upstream


margin = upstream.margin
prior, retained, rational = margin.prior, margin.retained, margin.rational
require, gates, local = margin.require, upstream.gates, upstream.local
ROOT, PYTHON, Q = margin.ROOT, margin.PYTHON, margin.Q
NAME = "q24_s16_l15_s18_margin_sensitivity_v1"
ID = "ace3-q24-s16-l15-s18-margin-sensitivity-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
INPUT = upstream.OUTPUT / "scientific_binding"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
PINS = {
    "result.json": "331d2536e78faf9e51258da1e614754a57e22c2e39e640473680ac02d041f491",
    "validation.json": "912fa21e61f3e9a00efa8485ad15f818beb46a8e6fd81bb854b474e01791973c",
    "layer15/actual_stages.npz": "205f43ce2783958e492ce7908204f35255f990c4d2ccb6b6da34059f874652a5",
    "layer15/reports.json": "219aa11dedc492cd8186ae082a34854fb5305bcd8f5df95a8b976cf566a52c8f",
}
EXPECTED_TESTS = 18
FLAGS = dict(margin.FLAGS)
no_execution = margin.no_execution


def check_upstream(document, validation):
    require(document["diagnostic_id"] == upstream.ID
            and document["status"] == "DOWNSTREAM_GATE_FAIL"
            and document["retained_status"] == "FAIL"
            and document["history"] == [9707] and document["position"] == 0
            and document["audit"] == {"native_layers": [14, 15]}
            and document["first_native_layer"] == 14
            and type(document["native_layer_invocations"]) is int
            and document["native_layer_invocations"] == 2,
            "wrong frozen suffix scope")
    for key, value in upstream.FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed frozen boundary: {key}")
    require(validation["collected"] == validation["executed"] == upstream.EXPECTED_TESTS
            and validation["errors"] == validation["failures"] == validation["skipped"] == 0
            and validation["cwd"] == str(ROOT) and validation["executable"] == str(PYTHON)
            and validation["PYTHONPATH"] == str(ROOT),
            "invalid frozen validation context or counts")


def select_failure(entries, reports_by_layer):
    require([entry["layer"] for entry in entries] == [14, 15],
            "unexpected frozen layer sequence")
    for entry in entries:
        layer = entry["layer"]
        require(entry["position"] == 0 and entry["prior_kv"] == "own empty P0"
                and entry["prior_layer_kv_consumed"] is False,
                "changed frozen P0 KV boundary")
        reports = reports_by_layer[layer]
        require(len(reports) == 19, "incomplete stage reports")
        for stage, report in enumerate(reports):
            require(report["stage"] == stage and report["node"] == [layer, 0, stage]
                    and report["policy_id"] == gates.POLICY_ID
                    and report["residual_state_lineage"] == report["kv_lineage"] == "PASS",
                    "wrong report node, policy or lineage")
            if layer == 15 and stage == 18:
                mandatory = report["binary64_v1"]
                failures = mandatory["failures"]
                require(entry["status"] == report["status"] == "FAIL"
                        and mandatory["passed"] is False
                        and len(failures) == mandatory["failure_count"] > 0
                        and failures == [row for row in mandatory["rows"]
                                         if row["accepted"] is False],
                        "missing or inconsistent mandatory failures")
                first = failures[0]
                require(first["index"] == 62 and first["actual_fp16_bits"] == "6633"
                        and first["nearest_fp16_bits"] == "6632"
                        and first["excess_budget"] == "1/8",
                        "wrong first mandatory scalar failure")
                return {"node": [15, 0, 18], "index": 62, "gate": "binary64_v1",
                        "evidence": entry["reports"]}
            require(report["status"] == "PASS"
                    and (report["binary64_v1"]["passed"] is True if stage == 18
                         else report["local_operator_fp16"]["passed"] is True),
                    "earlier mandatory gate failure")
        require(entry["status"] == "PASS", "earlier layer did not pass")
    raise ValueError("missing first mandatory failure")


def check_layer(arrays, parent):
    retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parent)
    for stage in range(19):
        retained.check_stage_state(stage, arrays, parent)
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "changed retained S16 RTZ operand")


def check_cut(frozen, cut_parent, cut):
    retained.verify_parent(frozen, frozen)
    retained.verify_parent(cut_parent, cut_parent)
    delta = cut["delta_Q24_units"]
    require(type(delta) is int and delta == -1866689
            and int(frozen["i"][62]) == cut["baseline_Q24_integer"] == 26568588224
            and int(cut_parent["i"][62]) == 26566721535
            and Fraction(delta, Q) == Fraction(cut["delta"])
            and np.array_equal(cut_parent["z"], frozen["z"])
            and np.flatnonzero(cut_parent["i"] != frozen["i"]).tolist() == [62]
            and np.flatnonzero(cut_parent["h"] != frozen["h"]).tolist() == [62],
            "changed reviewed L13 minimal-cut lineage")


def bind_closure(inputs, record):
    signature = {key: record[key] for key in ("path", "bytes", "sha256")}
    require(inputs.records.get(signature["path"]) == signature,
            f"upstream evidence outside selected frozen closure: {signature['path']}")
    return inputs.bind(record)


def decomposition(arrays, parent, parent_ref, reference, local_s17):
    index = 62
    values = {
        "parent_I_over_2p24": Fraction(int(parent["i"][index]), Q),
        "S11": rational.fp16_value(int(arrays["stage11"][index])),
        "scratch_I_over_2p24": Fraction(int(arrays["scratch_i"][index]), Q),
        "S17_actual": rational.fp16_value(int(arrays["stage17"][index])),
        "S17_local": rational.fp16_value(int(local_s17[index])),
        "output_I_over_2p24": Fraction(int(arrays["output_i"][index]), Q),
        "original_L14_reference": Fraction.from_float(float(parent_ref[index])),
        "original_L15_reference": Fraction.from_float(float(reference[index])),
    }
    values["L14_signed_Q24_drift"] = (
        values["parent_I_over_2p24"] - values["original_L14_reference"])
    values["L15_net_increment_error"] = (
        values["S11"] + values["S17_actual"]
        - (values["original_L15_reference"] - values["original_L14_reference"]))
    values["S18_projection_error"] = (
        rational.fp16_value(int(arrays["stage18"][index])) - values["output_I_over_2p24"])
    values["signed_global_error"] = (
        rational.fp16_value(int(arrays["stage18"][index])) - values["original_L15_reference"])
    require(values["parent_I_over_2p24"] + values["S11"] == values["scratch_I_over_2p24"]
            and values["scratch_I_over_2p24"] + values["S17_actual"]
            == values["output_I_over_2p24"], "exact residual addition identity mismatch")
    return {key: str(value) for key, value in values.items()}


def authenticate():
    inputs = prior.BoundInputs()
    document = margin.pinned(inputs, INPUT / "result.json", PINS["result.json"])
    validation = margin.pinned(inputs, INPUT / "validation.json", PINS["validation.json"])
    check_upstream(document, validation)
    require(document["validation"] == retained.record(INPUT / "validation.json"),
            "frozen validation linkage mismatch")
    for record in document["authenticated_inputs"]:
        inputs.bind(record)
    for record in (*validation["origins"].values(), *validation["compiled"],
                   validation["contract"], *document["upstream_results"].values()):
        inputs.bind(record)
    for name, digest in PINS.items():
        record = retained.record(INPUT / name)
        require(record["sha256"] == digest, f"pinned evidence mismatch: {name}")
        inputs.bind(record)

    # All older evidence must already belong to this selected result's closure.
    def closure(record):
        return bind_closure(inputs, record)

    freeze_record = retained.record(margin.INPUT / "freeze.json")
    closure(freeze_record)
    require(freeze_record["sha256"] == prior.FREEZE_SHA, "changed original freeze")
    freeze = inputs.read(freeze_record)
    require(freeze["global_reference_lineage"]["candidate_data_used"] is False,
            "candidate-substituted original reference")
    for source in freeze["sources"]:
        closure(source["original"])
        closure(source["snapshot"])
        require(source["original"]["sha256"] == source["snapshot"]["sha256"],
                "source/snapshot mismatch")
    closure(freeze["reference_extension"])
    extension = inputs.read(freeze["reference_extension"])
    upstream.check_reference_suffix(extension)
    upstream.native.bind_tree(extension["layers"], closure)
    original_result = inputs.read(document["upstream_results"]["attempt"])
    require(document["upstream_results"]["attempt"]["sha256"] == prior.RESULT_SHA,
            "wrong original attempt")
    original_arrays = inputs.archive(original_result["layers"][-1]["actual_stages"])
    frozen = retained.state_from(original_arrays, "output", "stage18")
    entries = document["layers"]
    require([entry["layer"] for entry in entries] == [14, 15],
            "unexpected frozen suffix")
    parent = inputs.archive(entries[0]["input_state_evidence"])
    require(entries[0]["input_state_evidence"] ==
            retained.record(INPUT / "counterfactual_L13_state.npz"), "wrong frozen cut input")
    check_cut(frozen, parent, document["minimal_residual_cut"])
    reports_by_layer, operands = {}, {}
    for entry in entries:
        layer = entry["layer"]
        item = extension["layers"][str(layer)]
        require(entry["original_reference"] == item, "original reference substitution")
        directory = INPUT / f"layer{layer}"
        for key, name in (("actual_stages", "actual_stages.npz"),
                          ("local_references", "local_references.npz"),
                          ("reports", "reports.json")):
            require(entry[key] == retained.record(directory / name),
                    "selected frozen artifact substitution")
        arrays = inputs.archive(entry["actual_stages"])
        local_refs = inputs.archive(entry["local_references"])
        reports = inputs.read(entry["reports"])
        trajectory = inputs.archive(item["fp16"])
        reference = upstream.global_reference(inputs, item)
        check_layer(arrays, parent)
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
        operands[layer] = (arrays, parent, reference, local_refs)
        if layer == 14:
            require(entry["output_state_evidence"] == entries[1]["input_state_evidence"]
                    == retained.record(directory / "counterfactual_state.npz"),
                    "L14-to-L15 state evidence linkage mismatch")
            parent = inputs.archive(entry["output_state_evidence"])
            retained.verify_parent(parent, retained.state_from(arrays, "output", "stage18"))
            require(entry["own_kv_evidence"] == retained.record(directory / "own_kv.npz"),
                    "L14 own KV evidence substitution")
            kv = inputs.archive(entry["own_kv_evidence"])
            require(set(kv) == {"k", "v"}, "invalid L14 own KV archive")
            for kind in ("k", "v"):
                local.finite_words(kv[kind], (1, 128))
                require(np.array_equal(kv[kind], arrays["output_cache_" + kind]),
                        "L14 own KV archive mismatch")
        else:
            require(not any(key in entry for key in (
                "output_state_evidence", "output_parent", "own_kv_evidence")),
                "failed L15 published successor evidence")
    failure = select_failure(entries, reports_by_layer)
    require(failure == document["first_failure"], "wrong frozen first failure")
    arrays, parent, reference, local_refs = operands[15]
    exact = decomposition(arrays, parent, operands[14][2], reference, local_refs["stage17"])
    return inputs, document, failure, exact


def sensitivity(exact):
    v = {key: Fraction(value) for key, value in exact.items()}
    output, reference = v["output_I_over_2p24"], v["original_L15_reference"]
    integer = margin.q24_units(output)
    drift, increment, projection = (v["L14_signed_Q24_drift"],
                                   v["L15_net_increment_error"], v["S18_projection_error"])
    require(drift + increment + projection == v["signed_global_error"]
            and v["parent_I_over_2p24"] + v["S11"] == v["scratch_I_over_2p24"]
            and v["scratch_I_over_2p24"] + v["S17_actual"] == output
            and output + projection == rational.fp16_value(0x6633)
            and v["signed_global_error"] == rational.fp16_value(0x6633) - reference,
            "exact signed decomposition identity mismatch")
    baseline, target = 0x6633, 0x6632
    base, passing = margin.scalar(baseline, reference), margin.scalar(target, reference)
    require(rational.project(integer, 0) == baseline and base["accepted"] is False
            and passing["accepted"] is True and passing["nearest_fp16_bits"] == "6632"
            and not margin.scalar(target - 1, reference)["accepted"],
            "selected baseline/passing word neighbourhood mismatch")
    lower, upper, minimum, maximum = margin.passing_cell(target)
    delta = maximum - integer
    require(delta < 0 and rational.project(maximum, 0) == target
            and rational.project(maximum + 1, 0) == baseline,
            "minimal Q24 crossing mismatch")
    rows = {}
    for name, correction in (
        ("remove_L14_drift", -drift), ("remove_L15_net_increment_error", -increment),
        ("remove_both", -drift - increment),
        ("canonical_S17_on_retained_S16", v["S17_local"] - v["S17_actual"]),
    ):
        hypothetical = output + correction
        scaled = hypothetical * Q
        row = {
            "correction": str(correction), "unprojected_value": str(hypothetical),
            "on_Q24_grid": scaled.denominator == 1,
            "inside_passing_RNE_cell": lower <= hypothetical <= upper,
            "margin_to_inclusive_upper_midpoint": str(upper - hypothetical),
            "scope": "frozen-component algebra only; not a propagated intervention",
        }
        if row["on_Q24_grid"]:
            word = rational.project(scaled.numerator, 0)
            row.update(output_word=f"{word:04x}", scalar_gate=margin.scalar(word, reference))
        rows[name] = row
    for name, component in (("L14_drift", drift), ("L15_net_increment_error", increment)):
        rows[name + "_single_component_boundary"] = {
            "current": str(component),
            "inclusive_upper_with_others_fixed": str(component + upper - output),
            "required_delta_Q24_units": delta,
        }
    rows["S18_projection"] = {
        "signed_contribution": str(projection),
        "signed_error_before_projection": str(drift + increment),
        "RNE_correct_for_frozen_Q24_input": True,
        "scope": "projection is determined by the residual, not an independent adjustable operand",
    }
    radius = Fraction(base["q"]) + Fraction(1, 8)
    return {
        "baseline": base, "nearest_pass": passing, "exact_coordinate_decomposition": exact,
        "unchanged_scalar_acceptance_interval": {
            "lower": str(reference - radius), "upper": str(reference + radius),
            "endpoints_included": True, "only_accepting_FP16_word": "6632",
        },
        "passing_cell": {
            "word": "6632", "lower_midpoint": str(lower), "upper_midpoint": str(upper),
            "midpoints_included": True, "upper_tie_word": "6632",
            "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
        },
        "minimal_residual_cut": {
            "baseline_Q24_integer": integer, "delta_Q24_units": delta,
            "delta": str(Fraction(delta, Q)),
            "continuous_distance_to_upper_midpoint": str(output - upper),
            "continuous_cut_is_strict": False, "output_word": "6632",
            "one_Q24_unit_less_reduction_word": f"{rational.project(maximum + 1, 0):04x}",
            "scope": "scalar sensitivity only; no state is modified or published",
        },
        "component_sensitivities": rows,
        "S11_FP16_cut": margin.component_word_cut(
            v["S11"], v["parent_I_over_2p24"] + v["S17_actual"], target, reference),
        "S17_FP16_cut": margin.component_word_cut(
            v["S17_actual"], v["scratch_I_over_2p24"], target, reference),
        "classification": "inconclusive_unique_cause",
        "attribution": "Exact additive accounting and frozen-component sensitivity, not causal "
                       "isolation. A changed parent can also change L15 increments. Correct RNE "
                       "projection is not a rounding defect or a policy recommendation.",
    }


def origins():
    require(Path(__file__).resolve() == SOURCE, "candidate source origin mismatch")
    records = {}
    for name, module in sorted(list(sys.modules.items())):
        if name == "ace3" or name.startswith("ace3.") or name == "tests" or name.startswith("tests."):
            expected = ROOT.joinpath(*name.split("."))
            path = getattr(module, "__file__", None)
            if path is None:
                paths = list(getattr(module, "__path__", ()))
                require(paths and all(Path(p).resolve() == expected for p in paths),
                        f"namespace origin mismatch: {name}")
                continue
            expected = (expected / "__init__.py" if hasattr(module, "__path__")
                        else expected.with_suffix(".py"))
            require(Path(path).resolve() == expected, f"module origin mismatch: {name}")
            records[name] = retained.record(expected)
    require(CONTRACT.resolve() == CONTRACT, "contract origin mismatch")
    return records


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "published repo-bound command required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(prior.LEGACY_TEST)
    source_origins = origins()
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["focused_tests"] == EXPECTED_TESTS
            and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["upstream_sha256"] == PINS
            and contract["policy_id"] == gates.POLICY_ID
            and contract["excess_budget"] == "1/8", "versioned contract mismatch")
    for key, value in FLAGS.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"contract boundary mismatch: {key}")
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
        "legacy_19_tests_executed": False, **FLAGS,
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
        "PYTHONPATH": os.environ.get("PYTHONPATH"), "dont_write_bytecode": sys.dont_write_bytecode})
    result = {"diagnostic_id": ID, **FLAGS, "timing_seconds": {}}
    phase, started = "compile_and_focused_tests", time.monotonic()
    try:
        with no_execution():
            validation = validate(out)
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "frozen_authentication_and_gate_reproduction", time.monotonic()
            inputs, document, failure, exact = authenticate()
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "exact_scalar_sensitivity", time.monotonic()
            result.update(sensitivity(exact))
            result["timing_seconds"][phase] = time.monotonic() - started
            phase, started = "final_binding_integrity", time.monotonic()
            for record in (*validation["origins"].values(), validation["contract"]):
                inputs.bind(record)
            for record in inputs.records.values():
                require(retained.record(record["path"]) == record,
                        f"input/source changed during diagnostic: {record['path']}")
            result["timing_seconds"][phase] = time.monotonic() - started
        result.update(
            status="DIAGNOSED", node=[15, 0, 18], index=62, first_failure=failure,
            retained_status=document["status"], authenticated_inputs=list(inputs.records.values()),
            upstream_sha256=PINS, review_provenance=document["review_provenance"],
            lineage={"L14_output_to_L15_input_IZH": "PASS", "L14_own_KV": "PASS",
                     "L15_empty_prior_and_own_KV": "PASS", "S12_S18_exact_IZH": "PASS",
                     "S16_retained_operand_RTZ": "PASS",
                     "original_input_reference_recurrence": "PASS"},
            reference_policy="unchanged independently propagated original-input global reference",
            validation=retained.record(out / "validation.json"),
            command=retained.record(out / "command.json"), tests_executed=validation["executed"],
            claim_boundary="Non-admitting frozen Q24 CPU scalar sensitivity only. No native layer "
                           "execution, strict-FP16-state W4A16, token, full-model, RTL, simulation, "
                           "synthesis, PPA, GPU, FPGA or hardware PASS. Historical evidence and "
                           "bounded Reviewer acceptances remain unchanged.")
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, py_compile.PyCompileError) as exc:
        result.update(status="BLOCKED", phase=phase, error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    retained.write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", *FLAGS)}, sort_keys=True))
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
