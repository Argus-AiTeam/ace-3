"""Frozen-evidence scalar sensitivity; never execute a native layer or publish state.

Run the command in the adjacent versioned contract from the isolated repository.
The CLI compiles these sources, runs only its focused tests once, authenticates
the reviewed attribution and its input closure, and writes exclusive evidence.
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

from ace3.model.candidates import diagnose_q24_s16_l13_s18_first_failure_v2 as prior


ROOT, PYTHON, INPUT = prior.ROOT, prior.PYTHON, prior.INPUT
NAME = "q24_s16_l13_s18_margin_sensitivity_v1"
ID = "ace3-q24-s16-l13-s18-margin-sensitivity-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
ATTRIBUTION = ROOT / "build/q24_s16_l13_s18_first_failure_v2_attempt001"
ATTRIBUTION_SHA = "c98d21b7a4f49686567de6cb22de765dbb0b45e9d27143df6b19856289069538"
VALIDATION_SHA = "c28c5ccf311794ca29ccb892d90834ac08894398b36eeaaa5225074c801c5d3f"
Q = 1 << 24
EXPECTED_TESTS = 14
require, retained, rational = prior.require, prior.retained, prior.rational
FLAGS = {
    "native_L0_L8_invocations": 0, "native_layer_invocations": 0,
    "rtl_invocations": 0, "candidate_admitted": False, "policy_adopted": False,
    "successor_published": False, "normal_host_review": "REQUIRED",
}


@contextmanager
def no_execution():
    def forbidden(*args, **kwargs):
        raise RuntimeError("native execution, external execution or successor publication forbidden")

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if not name.startswith("ace3.model.candidates."):
                continue
            for attribute in ("stages", "continuation_stages", "_stages", "native_layer",
                              "run", "save"):
                if callable(getattr(module, attribute, None)):
                    stack.enter_context(patch.object(module, attribute, forbidden))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def pinned(inputs, path, digest):
    # Check containment before opening, including symlink targets.
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(ROOT),
            f"input outside isolated repository: {path}")
    record = retained.record(path)
    require(record["sha256"] == digest, f"pinned evidence mismatch: {path}")
    return inputs.read(record)


def check_attribution(document, validation):
    require(document["diagnostic_id"] == prior.ID and document["status"] == "DIAGNOSED"
            and document["node"] == [13, 0, 18] and document["index"] == 62
            and document["retained_status"] == "FAIL", "wrong accepted attribution")
    for key, value in FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed attribution boundary: {key}")
    require(validation["collected"] == validation["executed"] == 16
            and validation["failures"] == validation["errors"] == validation["skipped"] == 0,
            "invalid retained attribution validation")
    require(document["baseline"]["actual_fp16_bits"] == "6630"
            and document["baseline"]["nearest_fp16_bits"] == "662f"
            and document["baseline"]["excess_budget"] == "1/8"
            and document["baseline"]["accepted"] is False, "changed retained scalar gate")


def authenticate(input_dir, attribution_dir):
    require(input_dir.resolve() == INPUT and attribution_dir.resolve() == ATTRIBUTION,
            "only the selected reviewed inputs are supported")
    inputs = prior.BoundInputs()
    document = pinned(inputs, ATTRIBUTION / "result.json", ATTRIBUTION_SHA)
    validation = pinned(inputs, ATTRIBUTION / "validation.json", VALIDATION_SHA)
    check_attribution(document, validation)
    for record in document["input_bindings"]:
        inputs.bind(record)
    for record in validation["origins"].values():
        inputs.bind(record)
    inputs.bind(validation["contract"])
    for record in validation["compiled"]:
        inputs.bind(record)
    require(document["validation"] == retained.record(ATTRIBUTION / "validation.json"),
            "attribution validation linkage mismatch")
    freeze = pinned(inputs, INPUT / "freeze.json", prior.FREEZE_SHA)
    result = pinned(inputs, INPUT / "result.json", prior.RESULT_SHA)
    extension = inputs.read(freeze["reference_extension"])
    prior.check_reference_chain(extension)
    for source in freeze["sources"]:
        inputs.bind(source["original"])
        inputs.bind(source["snapshot"])
        require(source["original"]["sha256"] == source["snapshot"]["sha256"],
                "source/snapshot mismatch")
    for record in document["baseline"]["sources"]:
        path = ROOT / record["path"]
        require(path.resolve() == path and path.is_relative_to(ROOT),
                "gate source escaped repository")
        require(retained.record(path)["sha256"] == record["sha256"], "changed scalar gate source")
    bindings = document["operand_bindings"]
    arrays = inputs.archive(bindings["L13_actual"])
    parent = inputs.archive(bindings["L12_parent"])
    local = inputs.archive(bindings["L13_local"])
    parent_arrays = inputs.archive(result["layers"][-2]["actual_stages"])
    parent_kv = inputs.archive(result["layers"][-2]["kv_state"])
    lineage = prior.check_lineage(arrays, parent, parent_arrays, parent_kv)
    reports = inputs.read(result["layers"][-1]["reports"])
    failure = prior.select_failure(result, reports)
    require(failure == document["mandatory_failure"], "retained failure selection mismatch")
    refs = {}
    for key in ("L12_global", "original_global"):
        ref = np.load(inputs.bind(bindings[key]), allow_pickle=False)
        require(ref.dtype == np.dtype("<f8") and ref.shape == (896,)
                and np.all(np.isfinite(ref)), "invalid original global reference")
        refs[key] = ref
    exact = prior.decomposition(
        arrays, parent, refs["L12_global"], refs["original_global"], local["stage17"], 62)
    require(exact == document["exact_coordinate_decomposition"],
            "retained exact decomposition mismatch")
    baseline = prior.measure(int(arrays["stage18"][62]), float(refs["original_global"][62]))
    require(baseline == document["baseline"], "retained baseline gate mismatch")
    require(bindings["original_global"] == extension["layers"]["13"]["binary64"]
            and bindings["L12_global"] == extension["layers"]["13"]["input_binary64"]
            and bindings["L13_actual"] == result["layers"][-1]["actual_stages"]
            and bindings["L12_parent"] == result["layers"][-1]["input_parent"],
            "operand/reference lineage substitution")
    return document, exact, lineage, list(inputs.records.values())


def q24_units(value):
    scaled = value * Q
    require(scaled.denominator == 1, "value is not on the Q24 residual grid")
    return scaled.numerator


def passing_cell(word):
    require(type(word) is int and 1 <= word < 0x7BFF,
            "positive interior finite FP16 target required")
    value = rational.fp16_value(word)
    lower = (rational.fp16_value(word - 1) + value) / 2
    upper = (value + rational.fp16_value(word + 1)) / 2
    # An odd significand loses both midpoint ties to the even neighbours.
    lo = -((-lower.numerator * Q) // lower.denominator)
    hi = (upper.numerator * Q) // upper.denominator
    if word & 1:
        lo += Fraction(lo, Q) == lower
        hi -= Fraction(hi, Q) == upper
    return lower, upper, lo, hi


def scalar(word, reference):
    return prior.measure(word, float(reference))


def component_word_cut(component, other, target, reference):
    lower, upper, _, _ = passing_cell(target)
    low_value, high_value = lower - other, upper - other
    choices = []
    for word in range(65536):
        if word & 0x7C00 == 0x7C00:
            continue
        value = rational.fp16_value(word)
        in_cell = (low_value < value < high_value if target & 1
                   else low_value <= value <= high_value)
        if in_cell:
            choices.append((abs(value - component), word, value))
    require(bool(choices), "no finite FP16 component can reach selected passing cell")
    _, word, value = min(choices)
    output = rational.project(q24_units(other + value), 0)
    require(output == target and scalar(output, reference)["accepted"],
            "component cut did not meet unchanged scalar gate")
    return {
        "word": f"{word:04x}", "value": str(value), "delta": str(value - component),
        "delta_Q24_units": q24_units(value - component), "output_word": f"{output:04x}",
        "scalar_accepted": True,
        "scope": "minimum absolute single-component FP16 substitution; all other operands held fixed",
    }


def sensitivity(exact):
    values = {key: Fraction(value) for key, value in exact.items()}
    output = values["output_I_over_2p24"]
    reference = values["original_L13_reference"]
    integer = q24_units(output)
    baseline, target = 0x6630, 0x662F
    require(rational.project(integer, 0) == baseline, "not selected baseline projection")
    require(not scalar(baseline, reference)["accepted"] and scalar(target, reference)["accepted"],
            "baseline/pass word gate mismatch")
    require(not scalar(target - 1, reference)["accepted"]
            and scalar(target, reference)["nearest_fp16_bits"] == "662f",
            "unexpected selected passing-word neighbourhood")
    lower, upper, minimum, maximum = passing_cell(target)
    delta_units = maximum - integer
    require(delta_units < 0 and rational.project(maximum, 0) == target
            and rational.project(maximum + 1, 0) == baseline,
            "minimal Q24 boundary crossing mismatch")
    drift = values["L12_signed_Q24_drift"]
    net = values["L13_net_increment_error"]
    projection = values["S18_projection_error"]
    require(drift + net + projection == values["signed_global_error"],
            "signed error identity broken")
    rows = {}
    for name, correction in (
        ("remove_L12_drift", -drift), ("remove_L13_net_increment_error", -net),
        ("remove_both", -drift - net),
        ("canonical_S17_on_retained_S16", values["S17_local"] - values["S17_actual"]),
    ):
        hypothetical = output + correction
        scaled = hypothetical * Q
        on_grid = scaled.denominator == 1
        rows[name] = {
            "correction": str(correction), "unprojected_value": str(hypothetical),
            "on_Q24_grid": on_grid,
            "inside_passing_RNE_cell": lower < hypothetical < upper,
            "margin_below_upper_midpoint": str(upper - hypothetical),
            "scope": "algebraic frozen-component sensitivity, not a propagated trajectory",
        }
        if on_grid:
            word = rational.project(scaled.numerator, 0)
            rows[name]["output_word"] = f"{word:04x}"
            rows[name]["scalar_gate"] = scalar(word, reference)
    for name, component in (("L12_drift", drift), ("L13_net_increment_error", net)):
        rows[name + "_single_component_boundary"] = {
            "current": str(component),
            "strict_upper_for_pass_with_others_fixed": str(component + upper - output),
            "largest_passing_value_on_current_Q24_delta_grid": str(
                component + Fraction(delta_units, Q)),
            "required_delta_Q24_units": delta_units,
        }
    return {
        "baseline": scalar(baseline, reference), "nearest_pass": scalar(target, reference),
        "exact_coordinate_decomposition": exact,
        "passing_cell": {
            "word": "662f", "value": str(rational.fp16_value(target)),
            "lower_midpoint": str(lower), "upper_midpoint": str(upper),
            "midpoints_included": False, "upper_tie_word": "6630",
            "minimum_Q24_integer": minimum, "maximum_Q24_integer": maximum,
        },
        "minimal_residual_cut": {
            "baseline_Q24_integer": integer, "continuous_distance_to_upper_midpoint": str(output - upper),
            "continuous_cut_is_strict": True, "delta_Q24_units": delta_units,
            "delta": str(Fraction(delta_units, Q)), "output_word": "662f",
            "one_Q24_unit_less_reduction_word": f"{rational.project(maximum + 1, 0):04x}",
        },
        "component_sensitivities": rows,
        "S17_FP16_cut": component_word_cut(
            values["S17_actual"], values["scratch_I_over_2p24"], target, reference),
        "S11_FP16_cut": component_word_cut(
            values["S11"], values["parent_I_over_2p24"] + values["S17_actual"], target, reference),
        "S18_output_word_delta": str(rational.fp16_value(target) - rational.fp16_value(baseline)),
        "classification": "inconclusive_unique_cause",
        "attribution": (
            "L12 drift and S18 RNE projection contribute positive signed error; L13 net "
            "increment error opposes it. Removing L12 drift alone algebraically crosses "
            "the cell; removing L13 net error alone does not. Same-input canonical S17 "
            "equals actual S17. S18 RNE is correct for its retained Q24 input. These are "
            "frozen-component sensitivities, not independent upstream interventions: "
            "changing a parent can also change downstream increments. No unique causal "
            "producer, policy recommendation or admitted trajectory follows."
        ),
    }


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and sys.dont_write_bytecode and not sys.flags.optimize
            and os.environ.get("PYTHONPATH") == str(ROOT), "published repo-bound command required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(prior.LEGACY_TEST)
    origins = {}
    for name, module in sorted(list(sys.modules.items())):
        if name.startswith("ace3.model.") or name in (TEST_MODULE, prior.LEGACY_TEST):
            path = getattr(module, "__file__", None)
            if path is None:
                continue
            expected = ROOT.joinpath(*name.split("."))
            expected = expected / "__init__.py" if hasattr(module, "__path__") else expected.with_suffix(".py")
            require(Path(path).resolve() == expected, f"module origin mismatch: {name}")
            origins[name] = retained.record(expected)
    compiled = []
    for index, path in enumerate((ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
                                  Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["focused_tests"] == EXPECTED_TESTS
            and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["attribution_sha256"] == ATTRIBUTION_SHA
            and contract["attribution_validation_sha256"] == VALIDATION_SHA
            and contract["freeze_sha256"] == prior.FREEZE_SHA
            and contract["result_sha256"] == prior.RESULT_SHA, "versioned contract mismatch")
    for key, value in FLAGS.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"contract boundary mismatch: {key}")
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    with (out / "unittest.log").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": origins, "compiled": compiled,
        "contract": retained.record(CONTRACT), "collected": suite.countTestCases(),
        "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, **FLAGS,
    }
    retained.write(out / "validation.json", report)
    require(report["collected"] == result.testsRun == EXPECTED_TESTS
            and result.wasSuccessful() and not result.skipped, "focused tests failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out == ROOT / "build" / (NAME + "_attempt001"), "output outside bounded writable scope")
    out.mkdir(exist_ok=False)
    context = {"cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
               "PYTHONPATH": os.environ.get("PYTHONPATH"),
               "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")}
    retained.write(out / "command.json", context)
    started = time.monotonic()
    phase = "compile_and_focused_tests"
    timings = {}
    try:
        with no_execution():
            validation = validate(out)
            timings[phase] = time.monotonic() - started
            phase, started = "frozen_input_authentication", time.monotonic()
            document, exact, lineage, bindings = authenticate(args.input, args.attribution)
            timings[phase] = time.monotonic() - started
            phase, started = "exact_scalar_sensitivity", time.monotonic()
            analysis = sensitivity(exact)
            timings[phase] = time.monotonic() - started
        result = {
            "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": 62,
            "retained_status": "FAIL", **FLAGS, **analysis, "lineage": lineage,
            "authenticated_inputs": bindings, "retained_controls": document["controls"],
            "review_provenance": document["accepted_review_provenance"],
            "reference_policy": "unchanged independently propagated original-input global reference",
            "authentication_scope": "Pinned reviewed attribution and all 102 frozen input bindings; "
                                    "current sources, snapshots, operands, I/Z/H, KV and recurrence. "
                                    "No original-reference propagation or native execution.",
            "validation": retained.record(out / "validation.json"),
            "command": retained.record(out / "command.json"),
            "tests_executed": validation["executed"], "timing_seconds": timings,
            "claim_boundary": "Wide-residual Q24 CPU scalar sensitivity only; not strict-FP16-state "
                              "W4A16, new-token, L9+ execution, RTL, hardware or full-model PASS.",
        }
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError, RuntimeError,
            py_compile.PyCompileError) as exc:
        result = {"diagnostic_id": ID, "status": "BLOCKED", **FLAGS, "phase": phase,
                  "error": f"{type(exc).__name__}: {exc}", "timing_seconds": timings}
    retained.write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", *FLAGS)}, sort_keys=True))
    if result["status"] == "BLOCKED":
        print(result["error"], file=sys.stderr)
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
