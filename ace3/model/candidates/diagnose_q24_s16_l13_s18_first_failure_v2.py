"""Read-only attribution of the selected reviewed L13/P0/S18 failure.

From /home/argustest/ace3-argus:
  PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 \
    /home/argustest/miniconda3/bin/python -B -m \
    ace3.model.candidates.diagnose_q24_s16_l13_s18_first_failure_v2 \
    --input build/q24_software_l9_l23_0c02042f0e21_attempt001 \
    --out build/q24_s16_l13_s18_first_failure_v2_attempt001

Compiles repository Python and runs the focused tests once before diagnosis.
No native layer execution, reference propagation, or successor publication.
"""

import argparse
import importlib
import json
import os
from fractions import Fraction
from pathlib import Path
import py_compile
import sys
import time
import unittest

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_s18_first_failure_v1 as prior


ROOT = prior.ROOT
PYTHON = prior.PYTHON
INPUT = ROOT / "build/q24_software_l9_l23_0c02042f0e21_attempt001"
FREEZE_SHA = "93a46d54a9c1b8fa1eeca21e58bd3ea2109780b4d64fbbf036a627e90750ae2b"
RESULT_SHA = "0aa53205f23fbf79bb4d00b011fac3850e5809819d8e543804579978d1d29996"
ID = "ace3-q24-s16-l13-s18-first-failure-attribution-v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_s18_first_failure_attribution_v2.json"
TEST_MODULE = "tests.test_q24_s16_l13_s18_first_failure_v2"
LEGACY_TEST = "tests.test_q24_s16_toward_zero_l3_l8_v1"
local, gates, retained, rational = prior.local, prior.gates, prior.retained, prior.rational
require = local.require
measure = prior.measure
select_failure = prior.select_failure
check_lineage = prior.check_lineage
rtz_reference = prior.rtz_reference
classify = prior.classify


class BoundInputs:
    def __init__(self):
        self.records = {}

    def bind(self, record):
        signature = {key: record[key] for key in ("path", "bytes", "sha256")}
        path = Path(signature["path"])
        require(path.is_absolute() and path.resolve() == path
                and path.is_relative_to(ROOT), f"input outside isolated repository: {path}")
        if str(path) in self.records:
            require(self.records[str(path)] == signature, f"conflicting binding: {path}")
        else:
            require(retained.record(path) == signature, f"binding mismatch: {path}")
            self.records[str(path)] = signature
        return path

    def read(self, record):
        return json.loads(self.bind(record).read_text())

    def archive(self, record):
        with np.load(self.bind(record), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}


def check_counters(document):
    for name in ("native_L0_L8_invocations", "rtl_invocations"):
        require(type(document[name]) is int and document[name] == 0,
                f"forbidden or invalid retained counter: {name}")


def check_reference_chain(extension):
    require(extension["reference_only"] is True
            and extension["accepted_ancestor_oracle_replays"] == 0
            and extension["policy_id"] == gates.POLICY_ID
            and extension["global_reference_policy"] ==
            "legacy-binary64-AWQ-fully-independent-propagation",
            "reference substituted, replayed or re-anchored")
    previous64 = extension["original_binary64_parent"]
    previous16 = extension["original_fp16_parent"]
    for layer in range(9, 14):
        item = extension["layers"][str(layer)]
        require(item["input_binary64"] == previous64
                and item["input_fp16"] == previous16
                and item["prior_kv"] == "own empty P0",
                f"broken original reference recurrence at L{layer}")
        previous64, previous16 = item["binary64"], item["fp16"]


def decomposition(arrays, parent, parent_ref, global_ref, local_s17, index):
    scratch = retained.state_from(arrays, "scratch", "stage12")
    values = {
        "parent_I_over_2p24": Fraction(int(parent["i"][index]), 1 << 24),
        "S11": rational.fp16_value(int(arrays["stage11"][index])),
        "scratch_I_over_2p24": Fraction(int(scratch["i"][index]), 1 << 24),
        "S17_actual": rational.fp16_value(int(arrays["stage17"][index])),
        "S17_local": rational.fp16_value(int(local_s17[index])),
        "output_I_over_2p24": Fraction(int(arrays["output_i"][index]), 1 << 24),
        "original_L12_reference": Fraction.from_float(float(parent_ref[index])),
        "original_L13_reference": Fraction.from_float(float(global_ref[index])),
    }
    values["S18_projection_error"] = (
        rational.fp16_value(int(arrays["stage18"][index])) - values["output_I_over_2p24"])
    values["L12_signed_Q24_drift"] = (
        values["parent_I_over_2p24"] - values["original_L12_reference"])
    values["L13_net_increment_error"] = (
        values["S11"] + values["S17_actual"]
        - (values["original_L13_reference"] - values["original_L12_reference"]))
    values["signed_global_error"] = (
        rational.fp16_value(int(arrays["stage18"][index])) - values["original_L13_reference"])
    require(values["parent_I_over_2p24"] + values["S11"] == values["scratch_I_over_2p24"]
            and values["scratch_I_over_2p24"] + values["S17_actual"] ==
            values["output_I_over_2p24"]
            and values["L12_signed_Q24_drift"] + values["L13_net_increment_error"]
            + values["S18_projection_error"] == values["signed_global_error"],
            "exact signed residual decomposition mismatch")
    return {key: str(value) for key, value in values.items()}


def diagnose(input_dir):
    require(input_dir.resolve() == INPUT, "only the selected 0c02042f0e21 attempt001 is supported")
    inputs = BoundInputs()
    bind, read, archive = inputs.bind, inputs.read, inputs.archive
    freeze_record = retained.record(INPUT / "freeze.json")
    result_record = retained.record(INPUT / "result.json")
    require(freeze_record["sha256"] == FREEZE_SHA and result_record["sha256"] == RESULT_SHA,
            "selected freeze/result identity mismatch")
    freeze, result = read(freeze_record), read(result_record)
    preexecution = read(freeze["preexecution"])
    for document in (freeze, preexecution, result):
        check_counters(document)
    require(freeze["contract"] == preexecution["contract"]
            and freeze["contract"]["policy_id"] == result["policy_id"] == gates.POLICY_ID
            and freeze["contract"]["policy_adopted"] is result["policy_adopted"] is False
            and result["candidate_admitted"] is False and result["status"] == "FAIL"
            and freeze["global_reference_lineage"]["candidate_data_used"] is False,
            "wrong retained software/reference scope")
    for source in freeze["sources"]:
        bind(source["original"])
        bind(source["snapshot"])
        require(source["original"]["sha256"] == source["snapshot"]["sha256"],
                "source snapshot mismatch")
    for record in freeze["global_reference_lineage"].values():
        if isinstance(record, dict):
            bind(record)
    extension = read(freeze["reference_extension"])
    check_reference_chain(extension)
    for key in ("original_binary64_parent", "original_fp16_parent",
                "original_specification", "input_freeze"):
        bind(extension[key])
    for generator in extension["generators"]:
        bind(generator)
    for layer in range(9, 14):
        for key in ("binary64", "fp16"):
            bind(extension["layers"][str(layer)][key])

    resume = freeze["state_lineage"]["prior_residual"]
    require(resume == preexecution["resume"] and resume["prior_layer_kv_consumed"] is False
            and freeze["state_lineage"]["prior_kv"] == "own empty P0",
            "changed accepted-parent provenance")
    receipt = read(resume["receipt"])
    require(receipt == resume["parent"] and receipt["next_layer"] == 9,
            "frozen accepted L8 receipt mismatch")
    parent = archive(receipt["state"])
    retained.verify_parent(parent, parent)
    accepted_result = read(resume["result"])
    accepted_entry = accepted_result["layers"][-1]
    require(accepted_result["status"] == "PASS" and accepted_entry["layer"] == 8
            and accepted_entry["output_parent"] == receipt["state"]
            and accepted_entry["kv_state"] == receipt["kv"], "wrong retained accepted L8 output")
    accepted_arrays = archive(accepted_entry["actual_stages"])
    retained.verify_parent(parent, retained.state_from(accepted_arrays, "output", "stage18"))
    accepted_kv = archive(receipt["kv"])
    for kind, stage in (("k", 5), ("v", 3)):
        local.finite_words(accepted_kv[kind], (1, 128))
        require(np.array_equal(accepted_kv[kind], accepted_arrays["output_cache_" + kind])
                and np.array_equal(accepted_kv[kind][0], accepted_arrays[f"stage{stage:02d}"]),
                "accepted L8 own-KV mismatch")

    entries = result["layers"]
    require([entry["layer"] for entry in entries] == list(range(9, 14)),
            "unexpected retained traversal")
    lineage = {}
    for entry in entries:
        layer = entry["layer"]
        require(entry["position"] == 0 and entry["input_parent"] == receipt["state"],
                "broken immediate-predecessor state lineage")
        arrays = archive(entry["actual_stages"])
        reports = read(entry["reports"])
        require(len(reports) == 19, "missing retained stage reports")
        retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parent)
        for stage, report in enumerate(reports):
            retained.check_stage_state(stage, arrays, parent)
            require(report["node"] == [layer, 0, stage] and report["stage"] == stage
                    and report["policy_id"] == gates.POLICY_ID
                    and report["residual_state_lineage"] == report["kv_lineage"] == "PASS",
                    "retained report lineage mismatch")
        require(np.array_equal(rtz_reference(arrays["s16_unrounded_binary64"]), arrays["stage16"]),
                "retained S16 RTZ operand mismatch")
        if layer == 13:
            require(entry["status"] == "FAIL" and "output_parent" not in entry
                    and not (INPUT / "layer13/software_parent.json").exists(),
                    "failed layer published successor")
            lineage["L13"] = check_lineage(arrays, parent, parent_arrays, parent_kv)
            break
        require(entry["status"] == "PASS" and all(r["status"] == "PASS" for r in reports),
                "nonpassing retained predecessor")
        next_receipt = read(retained.record(INPUT / f"layer{layer:02d}/software_parent.json"))
        require(next_receipt["arithmetic_lineage"] == freeze_record
                and next_receipt["state"] == entry["output_parent"]
                and next_receipt["state_lineage_parent"] == entry["input_parent"]
                and next_receipt["kv"] == entry["kv_state"]
                and next_receipt["numerical_report"] == entry["reports"]
                and next_receipt["next_layer"] == layer + 1
                and next_receipt["history"] == [9707] and next_receipt["position"] == 0
                and next_receipt["rtl_admissible"] is False
                and next_receipt["evidence_kind"] == "cpu_software_q24"
                and next_receipt["candidate_id"] == freeze["contract"]["candidate_id"]
                and next_receipt["state_id"] == freeze["contract"]["state_id"]
                and next_receipt["policy_id"] == gates.POLICY_ID,
                "retained software-parent receipt mismatch")
        parent_kv = archive(next_receipt["kv"])
        for kind in ("k", "v"):
            require(np.array_equal(parent_kv[kind], arrays["output_cache_" + kind]),
                    "retained own-layer KV receipt mismatch")
        parent = archive(next_receipt["state"])
        retained.verify_parent(parent, retained.state_from(arrays, "output", "stage18"))
        parent_arrays, receipt = arrays, next_receipt
        lineage[f"L{layer}"] = "authenticated retained I/Z/H, own KV and reports; no native execution"

    failure = select_failure(result, reports)
    require(result["first_failure"]["evidence"] == entry["reports"]["path"],
            "wrong mandatory failure report")
    saved_local = archive(entry["local_references"])
    item = extension["layers"]["13"]
    global_ref = np.load(bind(item["binary64"]), allow_pickle=False)
    parent_ref = np.load(bind(item["input_binary64"]), allow_pickle=False)
    for ref in (global_ref, parent_ref):
        require(ref.dtype == np.dtype("<f8") and ref.shape == (896,)
                and np.all(np.isfinite(ref)) and np.all(np.abs(ref) <= 65504),
                "invalid independent global reference")
    trajectory = archive(item["fp16"])
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(13)}
    local.authenticate_tensors(tensors, item["canonical"], 13)
    for stage in range(19):
        expected = None
        if stage < 18:
            expected = (retained.transition_reference(parent, arrays["stage11"])["h"]
                        if stage == 12 else local.local_reference(
                            stage, {key: arrays[key].copy() for key in local.OPERANDS[stage]},
                            tensors, 13))
            require(np.array_equal(expected, saved_local[f"stage{stage:02d}"]),
                    f"local reference not reproducible: S{stage}")
        fresh = gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=gates.POLICY_ID,
            local_reference=expected, reference_binary64=global_ref if stage == 18 else None)
        for key in ("status", "fp16", "binary64_v1" if stage == 18 else "local_operator_fp16"):
            require(fresh[key] == reports[stage][key], f"mandatory report differs: S{stage} {key}")

    index = failure["index"]
    baseline = measure(int(arrays["stage18"][index]), float(global_ref[index]))
    for key in ("actual_fp16_bits", "nearest_fp16_bits", "reference_binary64_hex",
                "q", "actual_error", "excess_error", "excess_budget", "accepted"):
        require(baseline[key] == failure[key], f"mandatory coordinate differs: {key}")
    scratch = retained.state_from(arrays, "scratch", "stage12")
    retained_rne = arrays["s16_unrounded_binary64"].astype("<f2").view("<u2")
    controls = {}
    suffixes = {
        "canonical_S17": saved_local["stage17"],
        "canonical_RNE_S16_then_S17": local.local_reference(
            17, {"stage16": saved_local["stage16"].copy()}, tensors, 13),
        "same_unrounded_operand_RNE_S16_then_canonical_S17": local.local_reference(
            17, {"stage16": retained_rne.copy()}, tensors, 13),
    }
    for name, words in suffixes.items():
        state = retained.transition_reference(scratch, words)
        controls[name] = measure(int(state["h"][index]), float(global_ref[index]))
    drift = parent["i"].astype("<f8") / (1 << 24) - parent_ref
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": index,
        "retained_status": "FAIL", "candidate_admitted": False, "policy_adopted": False,
        "successor_published": False, "normal_host_review": "REQUIRED",
        "native_L0_L8_invocations": 0, "native_layer_invocations": 0, "rtl_invocations": 0,
        "lineage": lineage, "mandatory_failure": failure, "baseline": baseline,
        "mandatory_reports_recomputed": 19,
        "exact_coordinate_decomposition": decomposition(
            arrays, parent, parent_ref, global_ref, saved_local["stage17"], index),
        "controls": {
            **controls,
            "scope": "same retained actual parent/scratch and identical original global coordinate; not admission",
            "rounding_only_comparison": "same_unrounded_operand_RNE_S16_then_canonical_S17 versus canonical_S17",
            "canonical_vs_retained_RNE_S16_word_differences":
                int(np.count_nonzero(saved_local["stage16"] != retained_rne)),
            "S17_actual_vs_local_word_differences":
                int(np.count_nonzero(arrays["stage17"] != saved_local["stage17"])),
            "S16_RTZ_vs_same_operand_RNE_word_differences":
                int(np.count_nonzero(arrays["stage16"] != retained_rne)),
        },
        "L12_drift": {"nonzero_Q24_coordinates": int(np.count_nonzero(drift)),
                      "max_absolute_Q24_drift": float(np.max(np.abs(drift))),
                      "coordinate_fp16_gate": measure(int(parent["h"][index]), float(parent_ref[index])),
                      "causal_attribution": "not established by drift alone"},
        **classify(baseline, controls["canonical_S17"], controls["canonical_RNE_S16_then_S17"]),
        "operand_bindings": {"L12_parent": receipt["state"], "L13_actual": entry["actual_stages"],
                             "L13_local": entry["local_references"], "weights": item["canonical"],
                             "original_global": item["binary64"], "L12_global": item["input_binary64"]},
        "input_bindings": list(inputs.records.values()),
        "accepted_review_provenance": {
            "record_in_pinned_freeze": resume["review"],
            "mode": "inherited frozen provenance; external Host review file not reopened or re-adjudicated"},
        "claim_boundary": "Retained bounded CPU attribution only; Q24 is wider than FP16. "
                          "No strict-FP16-state W4A16, RTL, hardware, new-token or full-model PASS.",
    }


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON,
            "use the published absolute interpreter and repository cwd")
    require(sys.dont_write_bytecode and not sys.flags.optimize
            and os.environ.get("PYTHONPATH") == str(ROOT) and str(ROOT) in sys.path,
            "explicit command-local repository PYTHONPATH and -B required")
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(LEGACY_TEST)
    origins = {}
    for name, module in sorted(list(sys.modules.items())):
        if name.startswith("ace3.model.candidates.") or name in (
                TEST_MODULE, LEGACY_TEST, "tests.test_q24_s16_l13_s18_first_failure_v1"):
            expected = ROOT.joinpath(*name.split(".")).with_suffix(".py")
            require(Path(module.__file__).resolve() == expected, f"module origin mismatch: {name}")
            origins[name] = retained.record(expected)
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["input"] == str(INPUT.relative_to(ROOT))
            and contract["freeze_sha256"] == FREEZE_SHA and contract["result_sha256"] == RESULT_SHA
            and contract["policy_adopted"] is contract["candidate_admitted"] is
            contract["successor_published"] is False, "invalid versioned diagnostic contract")
    check_counters(contract)
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    validation = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": origins, "compiled": compiled,
        "contract": retained.record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, "native_L0_L8_invocations": 0, "rtl_invocations": 0,
    }
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
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l13_s18_first_failure_v2_"),
            "output must be a fresh bounded build directory")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": os.environ.get("PYTHONPATH")}) + "\n")
        log.flush()
        try:
            validation = validate(out)
            validated = time.monotonic()
            diagnosis = diagnose(args.input)
            diagnosis["validation"] = retained.record(out / "validation.json")
            diagnosis["tests_executed"] = validation["executed"]
            diagnosis["timing_seconds"] = {
                "compile_and_tests": validated - started,
                "authentication_and_diagnosis": time.monotonic() - validated}
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            diagnosis = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
                "native_L0_L8_invocations": 0, "native_layer_invocations": 0, "rtl_invocations": 0,
                "normal_host_review": "REQUIRED",
            }
        retained.write(out / "result.json", diagnosis)
        summary = {key: diagnosis[key] for key in (
            "status", "classification", "native_L0_L8_invocations", "rtl_invocations",
            "candidate_admitted", "policy_adopted", "successor_published")}
        summary["result"] = str(out / "result.json")
        if diagnosis["status"] == "BLOCKED":
            summary["missing_evidence"] = diagnosis["missing_evidence"]
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return 0 if diagnosis["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
