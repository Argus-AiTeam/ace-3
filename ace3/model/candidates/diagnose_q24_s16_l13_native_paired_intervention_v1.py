"""Non-admitting, full-vector L13/P0 native-S16-RTZ paired CPU controls.

The adjacent versioned contract publishes the single repo-bound command, parent
mapping, interventions and limits. This module never executes a successor.
"""

import argparse
from fractions import Fraction
import importlib
import json
import math
import os
from pathlib import Path
import py_compile
import sys
import time
import unittest

import numpy as np
import torch
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_original_intermediate_drift_v1 as drift
from ace3.model.candidates import run_q24_s16_toward_zero_l9_l23_v1 as native


prior = drift.prior
ROOT = prior.ROOT
ID = "ace3-q24-s16-l13-native-paired-intervention-v1"
REVIEWED = ROOT / "build/q24_s16_l13_original_intermediate_drift_e6409c8e059a_attempt001"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_native_paired_intervention_v1.json"
TEST_MODULE = "tests.test_q24_s16_l13_native_paired_intervention_v1"
MODES = ("native", "local_s11", "local_s17", "local_both",
         "original_s11", "original_s17", "original_both",
         "rational_residual", "rational_final")
require = prior.require
record = prior.retained.record
write = prior.retained.write


def mapped_parent(values):
    require(values.dtype == np.dtype("<f8") and values.shape == (896,)
            and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504),
            "invalid original parent vector")
    integers = np.asarray([round(drift.exact(v) * (1 << 24)) for v in values], dtype="<i8")
    tags = ((integers == 0) & np.signbit(values)).astype("u1")
    hidden = np.asarray([prior.rational.project(int(i), int(z))
                         for i, z in zip(integers, tags, strict=True)], dtype="<u2")
    result = {"i": integers, "z": tags, "h": hidden}
    prior.retained.verify_parent(result, result)
    require(all(abs(Fraction(int(i), 1 << 24) - drift.exact(v)) <= Fraction(1, 1 << 25)
                for i, v in zip(integers, values, strict=True)), "Q24 mapping exceeds half-grid")
    return result


def same_arrays(actual, expected):
    require(set(actual) == set(expected), "native archive key mismatch")
    for key in actual:
        a, b = actual[key], expected[key]
        require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                f"native retained reproduction mismatch: {key}")


def local_words(stage, arrays, tensors):
    return prior.local.local_reference(
        stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]}, tensors, 13)


def execute(tensors, parent, mode, original):
    require(mode in MODES, "unknown intervention")
    arrays, changes = {}, {}
    for stage in native.stages(tensors, 13, parent, arrays):
        old = arrays[f"stage{stage:02d}"]
        replacement = None
        for source in ("local", "original"):
            if stage in (11, 17) and mode in (f"{source}_s{stage}", f"{source}_both"):
                replacement = (local_words(stage, arrays, tensors) if source == "local"
                               else native.candidate.rne(torch.from_numpy(original[f"s{stage}"].copy())))
        if (stage, mode) in ((12, "rational_residual"), (18, "rational_final")):
            incoming = (parent if stage == 12 else
                        prior.retained.state_from(arrays, "scratch", "stage12"))
            expected = prior.retained.transition_reference(
                incoming, arrays["stage11" if stage == 12 else "stage17"])
            prefix = "scratch" if stage == 12 else "output"
            prior.retained.verify_parent(
                prior.retained.state_from(arrays, prefix, f"stage{stage:02d}"), expected)
            # Preserve the native generator's scratch aliases; independent exact
            # arithmetic must agree before this diagnostic replacement is used.
            arrays[prefix + "_i"][:] = expected["i"]
            arrays[prefix + "_z"][:] = expected["z"]
            replacement = expected["h"]
        if replacement is not None:
            changes[str(stage)] = int(np.count_nonzero(old != replacement))
            arrays[f"stage{stage:02d}"] = replacement.copy()
        prior.retained.check_stage_state(stage, arrays, parent)
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "paired control changed S16 RTZ semantics")
    return arrays, changes


def parts(parent, arrays, original, index):
    exact = drift.exact
    word = prior.rational.fp16_value
    p = Fraction(int(parent["i"][index]), 1 << 24)
    s11 = word(int(arrays["stage11"][index]))
    s17 = word(int(arrays["stage17"][index]))
    scratch = Fraction(int(arrays["scratch_i"][index]), 1 << 24)
    output = Fraction(int(arrays["output_i"][index]), 1 << 24)
    final = word(int(arrays["stage18"][index]))
    require(scratch == p + s11 and output == scratch + s17, "residual exact sum mismatch")
    terms = {
        "incoming_Q24_minus_original": p - exact(original["input"][index]),
        "S11_minus_original": s11 - exact(original["s11"][index]),
        "S17_minus_original": s17 - exact(original["s17"][index]),
        "final_FP16_projection": final - output,
        "negative_original_addition_roundoff":
            exact(original["input"][index]) + exact(original["s11"][index])
            + exact(original["s17"][index]) - exact(original["s18"][index]),
    }
    error = final - exact(original["s18"][index])
    require(sum(terms.values()) == error, "full-vector signed decomposition does not close")
    return {
        "signed_error": str(error),
        "signed_parts": {key: str(value) for key, value in terms.items()},
        "input_FP16_projection": str(word(int(parent["h"][index])) - p),
        "scratch_FP16_projection": str(word(int(arrays["stage12"][index])) - scratch),
        "gate": prior.measure(int(arrays["stage18"][index]), float(original["s18"][index])),
    }


def interaction(a, a_base, b, b_base):
    return a - a_base - b + b_base


def classification(actual_gate, mapped_gate):
    require(actual_gate["accepted"] is False, "expected actual index-62 failure")
    return {
        "classification": ("incoming_parent_sensitive_under_explicit_Q24_mapping"
                           if mapped_gate["accepted"] else
                           "native_failure_persists_under_explicit_Q24_mapping"),
        "mapped_native_index62_passes": mapped_gate["accepted"],
        "claim": "A conditional native parent intervention, not unique producer attribution, "
                 "rounding-policy adoption or an admitted mapped parent.",
        "missing_evidence": [
            "A unique causal attribution within the upstream L0-L12 producers is not tested.",
            "The original binary64 parent is not a native Q24 state. This result is conditional "
            "on the disclosed nearest-Q24 mapping (maximum error 2^-25), not an exact "
            "binary64-parent native execution or proof over all parent mappings.",
            "S11/S17 original-output cuts combine trajectory and boundary effects. "
            "Their rescue does not prove a local arithmetic defect or a deployable repair.",
        ],
    }


def origins():
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        if name == "ace3" or name.startswith("ace3.") or name in (TEST_MODULE, "tests"):
            path = getattr(module, "__file__", None)
            if path is None:
                locations = list(getattr(module, "__path__", ()))
                require(locations and all(Path(p).resolve().is_relative_to(ROOT) for p in locations),
                        f"namespace origin mismatch: {name}")
                result[name] = {"namespace_paths": locations}
            else:
                path = Path(path).resolve()
                expected = ROOT.joinpath(*name.split("."))
                require(path in (expected.with_suffix(".py"), expected / "__init__.py"),
                        f"module origin mismatch: {name}")
                result[name] = record(path)
    return result


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT) and sys.dont_write_bytecode
            and not sys.flags.optimize, "use the published repository-bound command")
    tests = importlib.import_module(TEST_MODULE)
    source_origins = origins()
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and tuple(contract["modes"]) == MODES
            and contract["candidate_admitted"] is False and contract["policy_adopted"] is False
            and contract["successor_published"] is False and contract["rtl_invocations"] == 0
            and contract["normal_host_review"] == "REQUIRED", "diagnostic contract mismatch")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "contract": record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "accepted_L0_L8_tests_executed": False,
    }
    write(out / "validation.json", report)
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return report


def diagnose(out, log):
    start = time.monotonic()
    torch.set_num_threads(1)
    baseline = prior.diagnose(prior.INPUT)
    bound = {r["path"]: r for r in baseline["input_bindings"]}

    def bind(item):
        path = Path(item["path"])
        require(path.resolve().is_relative_to(ROOT), f"binding outside repository: {path}")
        if str(path) in bound:
            require(bound[str(path)] == item, f"conflicting input binding: {path}")
        else:
            require(record(path) == item, f"binding mismatch: {path}")
            bound[str(path)] = item
        return path

    def read(item):
        return json.loads(bind(item).read_text())

    def archive(item):
        with np.load(bind(item), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    reviewed = read(record(REVIEWED / "result.json"))
    require(reviewed["diagnostic_id"] == drift.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["original_S18_bitwise_reproduction"] is True
            and reviewed["candidate_admitted"] is False and reviewed["rtl_invocations"] == 0
            and reviewed["policy_adopted"] is False and reviewed["successor_published"] is False,
            "wrong reviewed drift evidence")
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    controls = archive(next(r for r in reviewed["artifacts"]
                            if r["path"] == str(REVIEWED / "reference_intermediates.npz")))
    original = {key: controls["original_" + key] for key in ("input", "s11", "s17", "residual", "s18")}
    bindings = baseline["operand_bindings"]
    parent = archive(bindings["L12_parent"])
    retained = archive(bindings["L13_actual"])
    global_reference = np.load(bind(bindings["original_global"]), allow_pickle=False)
    drift.same_binary64(original["s18"], global_reference)
    drift.same_binary64(original["input"], np.load(bind(bindings["L12_global"]), allow_pickle=False))
    drift.same_binary64(original["input"] + original["s11"], original["residual"])
    drift.same_binary64(original["residual"] + original["s17"], original["s18"])
    freeze = read(record(prior.INPUT / "freeze.json"))
    extension = read(freeze["reference_extension"])
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {key: np.ascontiguousarray(model.get_tensor(key))
                   for key in prior.local.tensor_shapes(13)}
    prior.local.authenticate_tensors(tensors, bindings["weights"], 13)
    trajectory = archive(extension["layers"]["13"]["fp16"])
    reports = read(record(prior.INPUT / "layer13/reports.json"))
    parents = {"actual": parent, "original_Q24_RNE": mapped_parent(original["input"])}
    timing = {"authentication": time.monotonic() - start}
    results, vectors, gate_reports = {}, {}, {}
    for parent_name, incoming in parents.items():
        results[parent_name], gate_reports[parent_name] = {}, {}
        for mode in MODES:
            begun = time.monotonic()
            arrays, changes = execute(tensors, incoming, mode, original)
            if parent_name == "actual" and mode == "native":
                same_arrays(arrays, retained)
            stage_reports = []
            for stage in range(18):
                local = (prior.retained.transition_reference(incoming, arrays["stage11"])["h"]
                         if stage == 12 else local_words(stage, arrays, tensors))
                report = prior.gates.evaluate_decoder_stage(
                    stage=stage, actual=arrays[f"stage{stage:02d}"],
                    reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
                    local_reference=local)
                if parent_name == "actual" and mode == "native":
                    require(report["status"] == reports[stage]["status"],
                            f"baseline gate changed at stage {stage}")
                stage_reports.append(report)
            rows = [parts(incoming, arrays, original, index) for index in range(896)]
            failures = [i for i, row in enumerate(rows) if not row["gate"]["accepted"]]
            if parent_name == "actual" and mode == "native":
                require(failures == [r["index"] for r in reports[18]["binary64_v1"]["failures"]],
                        "full-vector retained mandatory failures changed")
            results[parent_name][mode] = rows
            gate_reports[parent_name][mode] = {
                "S0_S17": stage_reports, "S18_failure_indices": failures,
                "replacement_word_changes": changes,
                "state_KV_RTZ_checks": "PASS", "candidate_admitted": False,
            }
            vectors.update({f"{parent_name}__{mode}__{key}": value for key, value in arrays.items()})
            timing[parent_name + "/" + mode] = time.monotonic() - begun
            log.write(json.dumps({"parent": parent_name, "mode": mode, "index62": rows[62],
                                  "S18_failure_indices": failures,
                                  "seconds": timing[parent_name + "/" + mode]}) + "\n")
            log.flush()
    paired = []
    for index in range(896):
        a, b = results["actual"], results["original_Q24_RNE"]
        effects = {}
        for mode in MODES:
            effects[mode] = {
                "actual_minus_mapped_signed_error":
                    str(Fraction(a[mode][index]["signed_error"]) - Fraction(b[mode][index]["signed_error"])),
                "parent_by_intervention_interaction": str(interaction(
                    Fraction(a[mode][index]["signed_error"]), Fraction(a["native"][index]["signed_error"]),
                    Fraction(b[mode][index]["signed_error"]), Fraction(b["native"][index]["signed_error"]))),
            }
            for key in ("input_FP16_projection", "scratch_FP16_projection"):
                effects[mode][key + "_paired_delta"] = str(
                    Fraction(a[mode][index][key]) - Fraction(b[mode][index][key]))
            effects[mode]["final_FP16_projection_paired_delta"] = str(
                Fraction(a[mode][index]["signed_parts"]["final_FP16_projection"])
                - Fraction(b[mode][index]["signed_parts"]["final_FP16_projection"]))
        joint = {}
        for parent_name, modes in results.items():
            joint[parent_name] = {}
            for source in ("local", "original"):
                error = lambda mode: Fraction(modes[mode][index]["signed_error"])
                joint[parent_name][source] = str(
                    error(source + "_both") - error(source + "_s11")
                    - error(source + "_s17") + error("native"))
        paired.append({"index": index, "effects": effects, "S11_by_S17_interaction": joint})
    write(out / "coordinate_controls.json", results)
    write(out / "paired_effects.json", paired)
    write(out / "gate_reports.json", gate_reports)
    np.savez(out / "native_vectors.npz", **vectors)
    write(out / "retained_authentication.json", baseline)
    mapping_errors = [Fraction(int(i), 1 << 24) - drift.exact(v)
                      for i, v in zip(parents["original_Q24_RNE"]["i"], original["input"], strict=True)]
    timing["total_diagnosis"] = time.monotonic() - start
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "native_retained_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "lineage": baseline["lineage"],
        "material_set": {"parents": list(parents), "modes": list(MODES),
                         "coordinates_per_control": 896, "native_L13_executions": 18},
        "parent_mapping": {"rule": "exact rational nearest Q24, ties even, signed zero, RNE16 view",
                           "candidate_admitted": False, "error_bound": "1/33554432",
                           "max_absolute_error": str(max(map(abs, mapping_errors))),
                           "index62_error": str(mapping_errors[62])},
        "selected_coordinate": {name: {mode: rows[62] for mode, rows in modes.items()}
                                for name, modes in results.items()},
        "selected_paired_effects": paired[62],
        **classification(results["actual"]["native"][62]["gate"],
                         results["original_Q24_RNE"]["native"][62]["gate"]),
        "input_bindings": list(bound.values()), "origins_after_execution": origins(),
        "timing_seconds": timing,
        "artifacts": [record(out / name) for name in (
            "coordinate_controls.json", "paired_effects.json", "gate_reports.json",
            "native_vectors.npz", "retained_authentication.json")],
        "claim_boundary": "CPU diagnostic only. Q24 residual wider than FP16; INT4 G128 "
                          "asymmetric native AWQ, FP16 scales/operators/KV unchanged. "
                          "No strict-FP16-state W4A16, RTL, hardware, new-token or full-model PASS. "
                          "No successor, policy change or runtime bottleneck claim.",
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l13_native_paired_intervention_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.input.resolve() == prior.INPUT and args.reviewed.resolve() == REVIEWED,
            "unselected source artifacts")
    out = output_path(args.out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": os.environ.get("PYTHONPATH"),
                              "loadavg": os.getloadavg(), "torch_threads_before": torch.get_num_threads(),
                              "cpu_affinity": sorted(os.sched_getaffinity(0))}) + "\n")
        log.flush()
        try:
            validation = validate(out)
            result = diagnose(out, log)
            result["validation"] = record(out / "validation.json")
            result["tests_executed"] = validation["executed"]
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            result = {"diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                      "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                      "candidate_admitted": False, "policy_adopted": False,
                      "successor_published": False, "rtl_invocations": 0,
                      "normal_host_review": "REQUIRED"}
        write(out / "result.json", result)
        summary = {key: result[key] for key in ("status", "classification", "candidate_admitted")}
        summary.update(result=str(out / "result.json"), missing_evidence=result["missing_evidence"])
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
