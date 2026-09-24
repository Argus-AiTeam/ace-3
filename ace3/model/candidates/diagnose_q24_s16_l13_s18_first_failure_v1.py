"""Read-only L13/P0/S18 diagnosis; never an admission or continuation runner.

Run from /home/argustest/ace3-argus:
  PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 \
    /home/argustest/miniconda3/bin/python -B -m \
    ace3.model.candidates.diagnose_q24_s16_l13_s18_first_failure_v1 \
    --input build/q24_software_l9_l23_4d1cfdfc15c0_attempt002 \
    --out build/q24_s16_l13_s18_first_failure_attempt001

The command compiles the touched Python, runs only this diagnostic's tests,
and writes exclusive JSON/log files. Counterfactuals never replace the global
reference, publish a parent, or adopt a rounding policy.
"""

import argparse
import importlib
import json
import math
from fractions import Fraction
from pathlib import Path
import py_compile
import sys
import time
import unittest

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import binary64_fp16_excess_v1 as excess
from ace3.model.candidates import decoder_gate_policy_v3 as gates
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_software_root_v3 as retained


ROOT = Path("/home/argustest/ace3-argus")
PYTHON = Path("/home/argustest/miniconda3/bin/python")
INPUT = ROOT / "build/q24_software_l9_l23_4d1cfdfc15c0_attempt002"
FREEZE_SHA = "a87f01cdb117e4eefa4f9f2c9d27eb9c01c124b61053f3f3d85738308ebe7395"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_s18_first_failure_attribution_v1.json"
TEST_MODULE = "tests.test_q24_s16_l13_s18_first_failure_v1"
ID = "ace3-q24-s16-l13-s18-first-failure-attribution-v1"
require = local.require
rational = retained.rational


def select_failure(result, reports):
    """Only the mandatory failures list selects a coordinate, never rows/argmax."""
    require(isinstance(reports, list) and len(reports) == 19, "expected 19 L13 reports")
    for stage, report in enumerate(reports):
        require(report["stage"] == stage and report["node"] == [13, 0, stage],
                "out-of-order or wrong report node")
        require(report["policy_id"] == gates.POLICY_ID, "wrong retained gate policy")
        require(report["residual_state_lineage"] == report["kv_lineage"] == "PASS",
                "retained lineage not passing")
        if stage < 18:
            require(report["status"] == "PASS"
                    and report["local_operator_fp16"]["passed"] is True,
                    "earlier mandatory local failure")
    mandatory = reports[18]["binary64_v1"]
    require(reports[18]["status"] == "FAIL" and mandatory["passed"] is False,
            "missing mandatory S18 rejection")
    failures = mandatory["failures"]
    require(isinstance(failures, list) and len(failures) == mandatory["failure_count"]
            and len(failures) > 0, "missing or inconsistent mandatory failures list")
    first = failures[0]
    require(type(first["index"]) is int and first["index"] == 62
            and first["accepted"] is False, "not the selected index 62 rejection")
    require(result["first_failure"]["node"] == [13, 0, 18]
            and result["first_failure"]["index"] == first["index"]
            and result["first_failure"]["failure_taxonomy"] == "global_numerical",
            "result/report first-failure mismatch")
    return first


def measure(word, reference):
    require(type(word) is int and 0 <= word <= 65535, "invalid FP16 word")
    require(math.isfinite(reference) and abs(reference) <= 65504, "invalid binary64 reference")
    row = excess.evaluate_layer_final_output(
        actual_fp16_bits=word, reference_binary64_hex=float(reference).hex())
    actual = rational.fp16_value(word)
    ref = Fraction.from_float(float(reference))
    floor = Fraction(row["q"])
    error = abs(actual - ref)
    require(error >= floor and Fraction(row["actual_error"]) == error
            and Fraction(row["excess_error"]) == error - floor
            and Fraction(row["excess_budget"]) == Fraction(1, 8),
            "exact error decomposition mismatch")
    return {**row, "signed_error": str(actual - ref),
            "excess_over_budget": str(error - floor - Fraction(1, 8)),
            "representation_floor_already_subtracted": True}


def rtz_reference(values):
    """Independent RNE conversion followed by a directed adjacent-value step."""
    require(values.dtype == np.dtype("<f8") and values.ndim == 1
            and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504),
            "invalid retained RTZ operands")
    nearest = values.astype("<f2")
    words = nearest.view("<u2").copy()
    overshot = np.abs(nearest.astype("<f8")) > np.abs(values)
    words[overshot] -= np.uint16(1)
    return words


def check_lineage(arrays, parent, parent_arrays, parent_kv):
    retained.verify_parent(parent, retained.state_from(parent_arrays, "output", "stage18"))
    retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parent)
    for kind, stage in (("k", 5), ("v", 3)):
        local.finite_words(parent_kv[kind], (1, 128))
        require(np.array_equal(parent_kv[kind], parent_arrays["output_cache_" + kind])
                and np.array_equal(parent_kv[kind][0], parent_arrays[f"stage{stage:02d}"]),
                "L12 KV receipt/archive mismatch")
    for stage in range(19):
        retained.check_stage_state(stage, arrays, parent)
    require(np.array_equal(rtz_reference(arrays["s16_unrounded_binary64"]), arrays["stage16"]),
            "S16 does not implement RTZ of its retained unrounded operand")
    return {"L12_output_to_L13_input_IZH": "PASS", "L12_own_KV": "PASS",
            "L13_empty_prior_and_own_KV": "PASS", "S12_S18_exact_rational_IZH": "PASS",
            "S16_retained_operand_RTZ": "PASS",
            "residual_projection": "RNE16; RTZ applies only to S16, not Q24 addition"}


def classify(baseline, s17_control, rtz_control):
    require(baseline["accepted"] is False, "diagnostic requires a baseline rejection")
    s17_rescue = s17_control["accepted"] is True
    rtz_rescue = rtz_control["accepted"] is True
    return {
        "classification": "inconclusive",
        "representation_floor": "excluded as sole cause: mandatory v1 already subtracts q",
        "residual_addition_or_projection_defect": "excluded for the authenticated retained operands",
        "S17_local_reference_substitution_rescues": s17_rescue,
        "S16_RNE_local_reference_suffix_rescues": rtz_rescue,
        "rationale": (
            "Exact Q24 I/Z/H identities and RTZ of the recorded S16 operand hold. "
            "These same-parent suffix controls measure conditional sensitivity, not "
            "independent trajectory admission or a unique upstream producer. "
            + ("A canonical S17 substitution clears this coordinate. " if s17_rescue else
               "Canonical S17 substitution does not clear this coordinate. ")
            + ("The canonical RNE S16/S17 suffix clears this coordinate. " if rtz_rescue else
               "The canonical RNE S16/S17 suffix does not clear this coordinate. ")
            + "L12 drift is measured but is not causally isolated."
        ),
        "missing_evidence": [
            "Independent binary64 L13 intermediate residual/S11/S17 values on the "
            "unchanged original-input recurrence; the retained global archive is S18 only.",
            "A source-bound same-L13-operator control isolating the complete incoming "
            "L12 trajectory drift from L13 local rounding, without inventing an admitted "
            "Q24 parent or replacing the original global reference.",
        ],
    }


def diagnose(input_dir):
    require(input_dir.resolve() == INPUT, "only the selected retained attempt002 is supported")
    bound = {}

    def bind(record):
        signature = {key: record[key] for key in ("path", "bytes", "sha256")}
        path = Path(signature["path"])
        require(path.resolve().is_relative_to(ROOT), f"input outside isolated repository: {path}")
        if str(path) in bound:
            require(bound[str(path)] == signature, f"conflicting binding: {path}")
        else:
            require(retained.record(path) == signature, f"binding mismatch: {path}")
            bound[str(path)] = signature
        return path

    def read(record):
        return json.loads(bind(record).read_text())

    def archive(record):
        with np.load(bind(record), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    freeze_record = retained.record(input_dir / "freeze.json")
    require(freeze_record["sha256"] == FREEZE_SHA, "wrong selected arithmetic freeze")
    freeze = read(freeze_record)
    require(freeze["rtl_invocations"] == 0
            and freeze["contract"]["policy_id"] == gates.POLICY_ID
            and freeze["contract"]["policy_adopted"] is False
            and freeze["global_reference_lineage"]["candidate_data_used"] is False,
            "wrong retained software/reference scope")
    for source in freeze["sources"]:
        original, snapshot = bind(source["original"]), bind(source["snapshot"])
        require(source["original"]["sha256"] == source["snapshot"]["sha256"],
                f"source snapshot differs: {original}: {snapshot}")
    for key in ("model", "control"):
        bind(freeze["global_reference_lineage"][key])
    extension = read(freeze["reference_extension"])
    require(extension["reference_only"] is True
            and extension["policy_id"] == gates.POLICY_ID
            and extension["global_reference_policy"] ==
            "legacy-binary64-AWQ-fully-independent-propagation",
            "reference substituted or re-anchored")
    for key in ("original_binary64_parent", "original_fp16_parent",
                "original_specification", "input_freeze"):
        bind(extension[key])
    for generator in extension["generators"]:
        bind(generator)
    result = read(retained.record(input_dir / "result.json"))
    require(result["status"] == "FAIL" and result["candidate_admitted"] is False
            and result["rtl_invocations"] == 0, "retained run is not the selected CPU failure")
    entries = result["layers"]
    require([entry["layer"] for entry in entries] == list(range(9, 14))
            and all(entry["status"] == "PASS" for entry in entries[:-1])
            and entries[-1]["status"] == "FAIL", "unexpected retained traversal")
    previous, entry = entries[-2:]
    require("output_parent" not in entry
            and not (input_dir / "layer13/software_parent.json").exists(),
            "failed layer published a successor")
    reports = read(entry["reports"])
    failure = select_failure(result, reports)
    require(result["first_failure"]["evidence"] == entry["reports"]["path"],
            "wrong failure report path")
    receipt = read(retained.record(input_dir / "layer12/software_parent.json"))
    require(receipt["next_layer"] == 13 and receipt["position"] == 0
            and receipt["history"] == [9707] and receipt["rtl_admissible"] is False
            and receipt["evidence_kind"] == "cpu_software_q24"
            and receipt["candidate_id"] == freeze["contract"]["candidate_id"]
            and receipt["state_id"] == freeze["contract"]["state_id"]
            and receipt["policy_id"] == gates.POLICY_ID
            and receipt["arithmetic_lineage"] == freeze_record
            and receipt["state"] == previous["output_parent"] == entry["input_parent"]
            and receipt["state_lineage_parent"] == previous["input_parent"]
            and receipt["numerical_report"] == previous["reports"]
            and receipt["kv"] == previous["kv_state"], "L12 parent receipt lineage mismatch")
    parent_reports = read(receipt["numerical_report"])
    require(len(parent_reports) == 19 and all(
        report["node"] == [12, 0, stage] and report["status"] == "PASS"
        for stage, report in enumerate(parent_reports)), "L12 parent not passing")
    parent = archive(receipt["state"])
    arrays = archive(entry["actual_stages"])
    parent_arrays = archive(previous["actual_stages"])
    lineage = check_lineage(arrays, parent, parent_arrays, archive(receipt["kv"]))
    saved_local = archive(entry["local_references"])
    item = extension["layers"]["13"]
    require(item["input_binary64"] == extension["layers"]["12"]["binary64"]
            and item["input_fp16"] == extension["layers"]["12"]["fp16"]
            and item["prior_kv"] == "own empty P0", "global L12-to-L13 reference lineage mismatch")
    global_ref = np.load(bind(item["binary64"]), allow_pickle=False)
    parent_ref = np.load(bind(item["input_binary64"]), allow_pickle=False)
    for ref in (global_ref, parent_ref):
        require(ref.dtype == np.dtype("<f8") and ref.shape == (896,)
                and np.all(np.isfinite(ref)) and np.all(np.abs(ref) <= 65504),
                "invalid original binary64 vector")
    trajectory = archive(item["fp16"])
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(13)}
    local.authenticate_tensors(tensors, item["canonical"], 13)
    for stage in range(18):
        expected = (retained.transition_reference(parent, arrays["stage11"])["h"]
                    if stage == 12 else local.local_reference(
                        stage, {key: arrays[key].copy() for key in local.OPERANDS[stage]},
                        tensors, 13))
        require(np.array_equal(expected, saved_local[f"stage{stage:02d}"]),
                f"retained local reference not reproducible on bound operands: S{stage:02d}")
        fresh = gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=gates.POLICY_ID,
            local_reference=expected)
        require(fresh["status"] == reports[stage]["status"], f"local gate differs at S{stage:02d}")
    index = failure["index"]
    baseline = measure(int(arrays["stage18"][index]), float(global_ref[index]))
    for key in ("actual_fp16_bits", "nearest_fp16_bits", "reference_binary64_hex",
                "q", "actual_error", "excess_error", "excess_budget", "accepted"):
        require(baseline[key] == failure[key], f"mandatory coordinate differs: {key}")
    scratch = retained.state_from(arrays, "scratch", "stage12")
    s17_state = retained.transition_reference(scratch, saved_local["stage17"])
    s17_control = measure(int(s17_state["h"][index]), float(global_ref[index]))
    rne_s17 = local.local_reference(17, {"stage16": saved_local["stage16"].copy()}, tensors, 13)
    rne_state = retained.transition_reference(scratch, rne_s17)
    rtz_control = measure(int(rne_state["h"][index]), float(global_ref[index]))
    # Isolate rounding mode on the *same* retained binary64 S16 operand as well.
    retained_rne = arrays["s16_unrounded_binary64"].astype("<f2").view("<u2")
    same_operand_s17 = local.local_reference(17, {"stage16": retained_rne.copy()}, tensors, 13)
    same_operand_state = retained.transition_reference(scratch, same_operand_s17)
    same_operand_control = measure(int(same_operand_state["h"][index]), float(global_ref[index]))
    values = {
        "parent_I_over_2p24": Fraction(int(parent["i"][index]), 1 << 24),
        "S11": rational.fp16_value(int(arrays["stage11"][index])),
        "scratch_I_over_2p24": Fraction(int(scratch["i"][index]), 1 << 24),
        "S17_actual": rational.fp16_value(int(arrays["stage17"][index])),
        "S17_local": rational.fp16_value(int(saved_local["stage17"][index])),
        "output_I_over_2p24": Fraction(int(arrays["output_i"][index]), 1 << 24),
    }
    values["S18_projection_error"] = (
        rational.fp16_value(int(arrays["stage18"][index])) - values["output_I_over_2p24"])
    values["L12_signed_Q24_drift"] = values["parent_I_over_2p24"] - Fraction.from_float(
        float(parent_ref[index]))
    require(values["parent_I_over_2p24"] + values["S11"] + values["S17_actual"]
            == values["output_I_over_2p24"], "coordinate residual decomposition mismatch")
    drift = parent["i"].astype("<f8") / (1 << 24) - parent_ref
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": index,
        "retained_status": "FAIL", "candidate_admitted": False,
        "rtl_invocations": 0, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED", "lineage": lineage,
        "mandatory_failure": failure, "baseline": baseline,
        "exact_coordinate_decomposition": {key: str(value) for key, value in values.items()},
        "coordinate_state_words": {
            "parent": {key: int(value[index]) for key, value in parent.items()},
            "scratch": {key: int(value[index]) for key, value in scratch.items()},
            "output": {key: int(value[index]) for key, value in
                       retained.state_from(arrays, "output", "stage18").items()}},
        "L12_drift": {"coordinate_binary64_hex": float(parent_ref[index]).hex(),
                      "nonzero_Q24_coordinates": int(np.count_nonzero(drift)),
                      "max_absolute_Q24_drift": float(np.max(np.abs(drift))),
                      "coordinate_fp16_gate": measure(int(parent["h"][index]), float(parent_ref[index])),
                      "causal_attribution": "not established by drift alone"},
        "controls": {
            "scope": "retained actual parent/scratch; same original global reference; not admission",
            "canonical_S17": s17_control,
            "canonical_RNE_S16_then_S17": rtz_control,
            "same_unrounded_operand_RNE_S16_then_canonical_S17": same_operand_control,
            "rounding_only_comparison": "compare same_unrounded_operand control with canonical_S17",
            "rounding_only_S18_word_changed":
                same_operand_control["actual_fp16_bits"] != s17_control["actual_fp16_bits"],
            "canonical_vs_retained_RNE_S16_word_differences":
                int(np.count_nonzero(saved_local["stage16"] != retained_rne)),
            "S17_actual_vs_local_word_differences":
                int(np.count_nonzero(arrays["stage17"] != saved_local["stage17"])),
            "S16_RTZ_vs_same_operand_RNE_word_differences":
                int(np.count_nonzero(arrays["stage16"] != retained_rne)),
        },
        **classify(baseline, s17_control, rtz_control),
        "operand_bindings": {"L12_parent": receipt["state"], "L13_actual": entry["actual_stages"],
                             "L13_local": entry["local_references"], "weights": item["canonical"],
                             "original_global": item["binary64"], "L12_global": item["input_binary64"],
                             "S17_operands": list(local.OPERANDS[17])},
        "input_bindings": list(bound.values()),
        "claim_boundary": "CPU retained-coordinate diagnosis only; wider-than-FP16 Q24 state. "
                          "No RTL, hardware, strict-FP16-state W4A16, new-token or full-model PASS.",
    }


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON,
            "use the published absolute interpreter and repository cwd")
    require(sys.dont_write_bytecode and not sys.flags.optimize
            and str(ROOT) in sys.path, "visible repository module search and -B required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module("tests.test_q24_s16_toward_zero_l3_l8_v1")
    origins = {}
    for name, module in sorted(list(sys.modules.items())):
        if name.startswith("ace3.model.candidates.") or name in (
                TEST_MODULE, "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            expected = ROOT.joinpath(*name.split(".")).with_suffix(".py")
            require(Path(module.__file__).resolve() == expected, f"module origin mismatch: {name}")
            origins[name] = str(expected)
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["policy_adopted"] is False
            and contract["rtl_invocations"] == 0, "invalid diagnostic contract")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    validation = {"cwd": str(ROOT), "executable": sys.executable, "sys_path": sys.path,
                  "origins": origins, "compiled": compiled, "contract": retained.record(CONTRACT),
                  "collected": count, "executed": result.testsRun,
                  "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": len(result.skipped), "legacy_19_tests_executed": False}
    retained.write(out / "validation.json", validation)
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == ROOT / "build" and out.name.startswith("q24_s16_l13_s18_first_failure_"),
            "output must be a fresh bounded build directory")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    with (out / "command.log").open("x") as log:
        context = {"cwd": str(Path.cwd()), "executable": sys.executable,
                   "argv": sys.argv, "sys_path": sys.path, "rtl_invocations": 0}
        log.write(json.dumps(context) + "\n")
        log.flush()
        try:
            validation = validate(out)
            validation_seconds = time.monotonic() - started
            diagnosis = diagnose(args.input)
            diagnosis["validation"] = retained.record(out / "validation.json")
            diagnosis["timing_seconds"] = {
                "validation": validation_seconds,
                "diagnosis": time.monotonic() - started - validation_seconds}
            diagnosis["tests_executed"] = validation["executed"]
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            diagnosis = {"diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                         "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                         "rtl_invocations": 0, "policy_adopted": False, "successor_published": False,
                         "candidate_admitted": False, "normal_host_review": "REQUIRED"}
        with (out / "result.json").open("x") as output:
            json.dump(diagnosis, output, indent=2, sort_keys=True)
            output.write("\n")
        summary = {key: diagnosis[key] for key in (
            "status", "classification", "rtl_invocations", "policy_adopted", "successor_published")}
        summary["result"] = str(out / "result.json")
        if diagnosis["status"] == "BLOCKED":
            summary["missing_evidence"] = diagnosis["missing_evidence"]
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return 0 if diagnosis["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
