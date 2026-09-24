"""Read-only exact S18 boundary diagnosis of the reviewed full-stage17 capture."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as policy
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational
from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_suffix_intervention_v1 as previous


ROOT, PYTHON = previous.ROOT, previous.PYTHON
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_s18_boundary_diagnostic_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
MISSION = "f3e815be2d42"
CONTROLS, FLAGS, ENVIRONMENT = previous.CONTROLS, previous.FLAGS, previous.ENVIRONMENT
capture, require, same, bound_bytes = (
    previous.capture, previous.require, previous.same, previous.bound_bytes)
verify_members = previous.verify_members
COUNTS = {**dict.fromkeys(previous.COUNTS, 0),
          "exact_coordinate_accounts": 8064, "independent_scalar_oracles": 8064}
PARENT_ROOT = ROOT / "build/mlp17-full-vector-f96d2447f5f5-attempt002"
PINS = {
    "stdout": {"path": str(PARENT_ROOT / "run/check.stdout"), "bytes": 10897300,
               "sha256": "d90552a0d0801653ca50af7fd43bd5296b8abfde008161c4e95e4cc31d142358"},
    "capture": {"path": str(PARENT_ROOT / "run/capture.json"), "bytes": 19572,
                "sha256": "78aaf1a021e87c3b411975434635bf46441de343f1e75edc08aea4564930e58f"},
    "outer": {"path": str(PARENT_ROOT / "launcher.capture.json"), "bytes": 1441,
              "sha256": "3a3f4a0189d12f8bf52e8d0557ebf6742535b4d428e3d4ad5164412a49a3f33b"},
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/f96d2447f5f5/round-0001.json",
        "bytes": 690, "sha256": "e342e5f5190b16dbd99b8a08a64e259987c37a34f4beff1e1b11b4976359ff07"},
}
BOUNDARY = (
    "One read-only CPU S18 boundary diagnosis of reviewed f96d2447f5f5 bytes. "
    "Q24 scratch state is wider than FP16: its exact integer, not the rounded "
    "stage12 view, is the left S18 operand. Native S16 RTZ, asymmetric G128 packed "
    "INT4 native GEMM nibble order without qzero plus-one, FP16 scales/operator "
    "and KV boundaries, original-input independently propagated FP16/binary64 "
    "references and excess <= 1/8 remain unchanged. No decoder, prefix, admission, "
    "reference or accepted/closed producer/intervention/review replay; no "
    "passable-word intervention or final-margin bridge; no hardware/RTL/GPU/ACE2 "
    "action, precision/scale change, strict-FP16-state W4A16, new-token or full-model "
    "admission. Stop this branch after this diagnostic and normal Host review."
)
PREREGISTRATION = {
    "SUPPORTED": "Exactly 83 retained failures per control, all exact Q24/RNE and "
                 "independent oracle agreements, intact operands/state/KV/lineage, "
                 "and every failed fixed-operand unrounded sum outside the "
                 "unchanged binary64 authority radius.",
    "REJECTED": "Any fixed-operand RNE can pass, an unrounded sum is inside the "
                "radius while its word fails, or an arithmetic, rounding, "
                "operand/state/KV/lineage/oracle mismatch exists. Localize the "
                "remaining question to S18 arithmetic/rounding/integrity, not a "
                "final-head repair.",
    "UNKNOWN": "Authentication, source/test/capture, environment/account/budget "
               "or execution-integrity defect only.",
}


def authenticate():
    review = json.loads(bound_bytes(PINS["review"]))
    same((review["kind"], review["mission_id"], review["producer_role"],
          review["round"], review["review"]["status"]),
         ("round_reviewed_handoff", previous.MISSION, "reviewer", 1, "done"),
         "independent f96d review")
    inner, outer = (json.loads(bound_bytes(PINS[k])) for k in ("capture", "outer"))
    require(inner["success"] is True and inner["failure"] is None
            and outer["success"] is True and outer["exit_status"] == 0
            and outer["timed_out"] is False, "parent capture failure")
    preflight = inner["preflight"]
    same((preflight["mission_id"], preflight["role"], preflight["uid"],
          preflight["cwd"], preflight["python"], preflight["environment"]),
         (previous.MISSION, "engineer", 1000, str(ROOT), PYTHON, ENVIRONMENT),
         "parent account/environment")
    same(preflight["command_budget"], {"compile": 1, "pytest": 1, "check": 1},
         "parent budget")
    same((preflight["independent_host_review"], preflight["model_or_service_calls_authorized"]),
         ("REQUIRED", 0), "parent review/service authority")
    same(preflight["sources"], [capture.binding(previous.SOURCE), capture.binding(previous.TEST)],
         "f96d source/test pins")
    same(inner["sources_after"], preflight["sources"], "parent source drift")
    for pins in (preflight["capture_implementation"], inner["capture_implementation_after"],
                 outer["capture_implementation_after"]):
        same(pins, capture.implementation_pins(), "shared sealer source/test pins")
    bound_bytes(preflight["executable"])
    same([r["label"] for r in inner["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"],
         "parent command census")
    verify_members(PARENT_ROOT / "run", inner)
    same(inner["results"][-1]["argv"], [PYTHON, "-B", "-m", previous.MODULE, "--check"],
         "parent check command")
    same(inner["results"][-1]["files"][3], PINS["stdout"], "parent stdout splice")
    same([p["path"] for p in outer["files"]],
         [str(PARENT_ROOT / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "outer paths")
    identity, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    launch = json.loads(identity)
    same((launch["argv"], launch["cwd"], launch["uid"], launch["environment"]),
         ([PYTHON, "-B", str(previous.TEST), "--run", str(PARENT_ROOT / "run")],
          str(ROOT), 1000, ENVIRONMENT), "outer identity")
    same(launch["launcher"], preflight["sources"][1], "outer source")
    same(launch["capture_implementation"], capture.implementation_pins(), "outer helper")
    same(launch["command"], capture.environment_command(ENVIRONMENT, launch["argv"]),
         "outer command")
    same(error, b"", "outer stderr")
    same(whole, b"IDENTITY\n" + identity + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "outer whole-command bytes")
    same(json.loads(output), {"capture": PINS["capture"], "success": True, "failure": None},
         "outer/inner splice")
    result = json.loads(bound_bytes(PINS["stdout"]))
    same((result["diagnostic_id"], result["version"], result["artifact_authentication"],
          result["status"]), (previous.NAME, 1, "AUTHENTICATED", "REJECTED"),
         "reviewed full-vector result")
    same(result["flags"], FLAGS, "parent non-admission flags")
    same(result["dispatch_and_write_audit"], previous.COUNTS, "parent operator census")
    same(result["capture_preflight"], preflight, "parent preflight splice")
    same(result["tests"]["compiled"], preflight["sources"], "parent compile pins")
    suites = ET.fromstring(bound_bytes(result["tests"]["pytest"])).findall("testsuite")
    require(sum(int(s.attrib["tests"]) for s in suites) == result["tests"]["executed"] == 32
            and not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
            "f96d focused tests")
    for pin in (*result["diagnostic_sources"].values(), *result["authenticated_files"]):
        bound_bytes(pin)
    same(result["parent_pins"],
         {"hotspot": previous.hotspot.PINS, "direct_hidden": previous.direct.PINS,
          "single_coordinate": previous.PINS}, "parent lineage pins")
    for group in result["parent_pins"].values():
        for pin in group.values():
            bound_bytes(pin)
    return result


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("boundary diagnostic forbids model/producer dispatch")

    with previous.guards.no_writes(audit), previous.parent.no_dispatch(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates.") and name != MODULE:
                for attribute in ("check", "run_tests", "focused_tests", "collect", "report",
                                  "measure", "validate", "run_control", "suffix_stages",
                                  "projection", "rmsnorm", "logits", "top_k", "expected_stage",
                                  "stage_report", "preregister"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        stack.enter_context(patch.object(previous.native.state, "add", forbidden))
        yield


def capture_preflight():
    # Reuse the reviewed account, capture, budget and completed-test gates verbatim.
    with patch.multiple(previous, SOURCE=SOURCE, TEST=TEST, MODULE=MODULE, MISSION=MISSION):
        return previous.capture_preflight()


def coordinate(index, scratch_i, scratch_z, stage12, stage17, output_i, output_z,
               word, reference16, reference64):
    left = Fraction(scratch_i, 1 << 24)
    right, view = rational.fp16_value(stage17), rational.fp16_value(stage12)
    total = left + right
    integer = int(total * (1 << 24))
    zero = int(integer == 0 and scratch_i == 0 and scratch_z == 1 and stage17 == 0x8000)
    state_error = None
    try:
        rational.validate_state(scratch_i, scratch_z)
        rational.validate_state(output_i, output_z)
        expected = rational.project(integer, zero)
        scratch_word = rational.project(scratch_i, scratch_z)
    except rational.PrimitiveFault as error:
        state_error, expected, scratch_word = str(error), None, None
    gate = policy.evaluate_layer_final_output(
        actual_fp16_bits=word, reference_binary64_hex=reference64.hex())
    ref = Fraction.from_float(reference64)
    floor = Fraction(gate["q"])
    radius = floor + policy.EXCESS_BUDGET
    rounded = rational.fp16_value(word)
    return {
        "index": index, "scratch_i": scratch_i, "scratch_z": scratch_z,
        "stage12_fp16_word": f"{stage12:04x}", "stage12_fp16_value": str(view),
        "scratch_exact_value": str(left), "scratch_minus_fp16_view": str(left - view),
        "stage17_fp16_word": f"{stage17:04x}", "stage17_fp16_value": str(right),
        "unrounded_fixed_operand_sum": str(total),
        "fp16_view_operand_sum_not_state_operand": str(view + right),
        "expected_q24_i": integer, "expected_q24_z": zero,
        "actual_q24_i": output_i, "actual_q24_z": output_z,
        "expected_rne_word": None if expected is None else f"{expected:04x}",
        "actual_fp16_word": f"{word:04x}", "actual_fp16_value": str(rounded),
        "state_error": state_error,
        "scratch_view_matches": scratch_word == stage12,
        "state_matches": state_error is None and (output_i, output_z) == (integer, zero),
        "rne_matches": expected == word,
        "original_input_fp16_word": f"{reference16:04x}",
        "original_input_fp16_value": str(rational.fp16_value(reference16)),
        "fp16_reference_error": str(rounded - rational.fp16_value(reference16)),
        "original_input_binary64_hex": reference64.hex(),
        "original_input_binary64_value": str(ref),
        "nearest_fp16_word": gate["nearest_fp16_bits"], "nearest_fp16_floor": str(floor),
        "actual_error": gate["actual_error"], "excess": gate["excess_error"],
        "excess_budget": str(policy.EXCESS_BUDGET),
        "threshold_margin": str(policy.EXCESS_BUDGET - Fraction(gate["excess_error"])),
        "binary64_authority_radius": str(radius),
        "accepted": gate["accepted"],
        "fixed_operand_rne_passable": expected is not None
        and abs(rational.fp16_value(expected) - ref) <= radius,
        "unrounded_outside_radius": abs(total - ref) > radius,
        "unrounded_threshold_margin": str(radius - abs(total - ref)),
        "unrounded_minus_binary64": str(total - ref),
        "rounding_delta": str(rounded - total),
        "binary64_gate": gate,
    }


def classify(rows, issues):
    reasons = list(issues)
    if len(rows) != 83:
        reasons.append("retained_failure_census")
    for row in rows:
        for key in ("scratch_view_matches", "state_matches", "rne_matches", "oracle_matches"):
            if not row[key]:
                reasons.append(f"{row['index']}:{key}")
        if row["accepted"] or row["fixed_operand_rne_passable"]:
            reasons.append(f"{row['index']}:fixed_operand_passable")
        if not row["unrounded_outside_radius"]:
            reasons.append(f"{row['index']}:unrounded_inside_radius")
    return ("REJECTED" if reasons else "SUPPORTED"), reasons


def diagnose_control(control, actual, output, reference, binary64, report, oracle, audit):
    issues, rows, all_failures = [], [], []
    for key in ("residual_state_lineage", "kv_lineage"):
        if report[key] != "PASS":
            issues.append(key)
    for kind, stage in (("k", "05"), ("v", "03")):
        cache = actual["output_cache_" + kind]
        if cache.dtype.str != "<u2" or cache.shape != (1, 128) or not np.array_equal(
                cache[0], actual["stage" + stage]):
            issues.append("kv_" + kind)
    for key in ("input_i", "input_z", "input_hidden", "scratch_i", "scratch_z", "stage12"):
        require(actual[key].shape == (896,), "retained state shape")
    for key in ("stage17", "stage18", "output_i", "output_z"):
        require(len(output[key]) == 896 and all(type(v) is int for v in output[key]),
                "captured output encoding")
    gate = report["binary64_v1"]
    require(len(gate["rows"]) == 896 and [r["index"] for r in gate["rows"]] == list(range(896)),
            "retained gate coordinate encoding")
    retained_failures = [r["index"] for r in gate["failures"]]
    if (retained_failures != [r["index"] for r in gate["rows"] if not r["accepted"]]
            or gate["failure_count"] != 83 or len(retained_failures) != 83
            or gate["passed"] is not False or report["stage"] != 18 or report["status"] != "FAIL"):
        issues.append("retained_failure_gate_mismatch")
    for k in range(896):
        try:
            input_view = rational.project(int(actual["input_i"][k]), int(actual["input_z"][k]))
        except rational.PrimitiveFault:
            input_view = None
        if input_view != int(actual["input_hidden"][k]):
            issues.append(f"{k}:input_state_view")
        if output["stage17"][k] != int(reference["stage17"][k]):
            issues.append(f"{k}:original_stage17_operand")
        row = coordinate(k, int(actual["scratch_i"][k]), int(actual["scratch_z"][k]),
                         int(actual["stage12"][k]), output["stage17"][k], output["output_i"][k],
                         output["output_z"][k], output["stage18"][k],
                         int(reference["stage18"][k]), float(binary64[k]))
        audit["exact_coordinate_accounts"] += 1
        row["oracle_matches"] = oracle.verify_coordinate(row)
        audit["independent_scalar_oracles"] += 1
        expected_gate = {**row["binary64_gate"], "index": k}
        if expected_gate != gate["rows"][k]:
            issues.append(f"{k}:retained_binary64_oracle")
        if not all(row[key] for key in
                   ("scratch_view_matches", "state_matches", "rne_matches", "oracle_matches")):
            issues.append(f"{k}:S18_arithmetic_state_oracle")
        if not row["accepted"]:
            all_failures.append(k)
        if k in retained_failures:
            rows.append(row)
    if all_failures != retained_failures:
        issues.append("failure_indices_changed")
    decision, reasons = classify(rows, issues)
    return {"control": control, "status": decision, "rejection_reasons": reasons,
            "coordinates_checked": 896, "retained_failure_indices": retained_failures,
            "failed_coordinates": rows, "failure_count": len(rows),
            "unrounded_outside_radius_count": sum(r["unrounded_outside_radius"] for r in rows),
            "q24_rne_agreement_count": sum(r["state_matches"] and r["rne_matches"] for r in rows),
            "oracle_agreement_count": sum(r["oracle_matches"] for r in rows),
            "fixed_operand_rne_passable_count": sum(r["fixed_operand_rne_passable"] for r in rows)}


def check():
    preflight, tests = capture_preflight()
    oracle = previous.load_module(TEST, NAME + "_oracle")
    audit = dict.fromkeys(COUNTS, 0)
    start = time.monotonic()
    with read_only(audit):
        result = authenticate()
        authority = result["L23_reference_authority"]
        same((authority["status"], authority["missing"]), ("BOUND_ORIGINAL_INPUT_L23", []),
             "original-input reference authority")
        bound_bytes(authority["manifest"])
        reference = previous.layer.archive(authority["reference"]["fp16"])
        binary64 = np.load(io.BytesIO(bound_bytes(authority["reference"]["binary64"])),
                           allow_pickle=False)
        require(binary64.dtype.str == "<f8" and binary64.shape == (896,)
                and np.isfinite(binary64).all(), "binary64 reference encoding")
        require(all(reference[k].dtype.str == "<u2" and reference[k].shape == (896,)
                    for k in ("stage17", "stage18")), "FP16 reference encoding")
        retained = json.loads(bound_bytes(result["authenticated_files"][0]))
        same(result["original_thresholds"], retained["preflight"]["thresholds"],
             "original threshold authority")
        same(authority, retained["preflight"]["L23_original_reference"], "reference lineage splice")
        same([r["control"] for r in retained["controls"]], list(CONTROLS), "retained control order")
        same(sorted(result["outputs"]), sorted(CONTROLS), "captured control census")
        same(sorted(result["stage_reports"]), sorted(CONTROLS), "captured gate census")
        controls, operands = [], {}
        authenticated_seconds = time.monotonic() - start
        for record in retained["controls"]:
            control = record["control"]
            pin = record["parent"]["terminal_archive"]
            require(pin in result["authenticated_files"], "retained operand pin splice")
            operands[control] = pin
            actual = previous.layer.archive(pin)
            controls.append(diagnose_control(
                control, actual, result["outputs"][control], reference, binary64,
                result["stage_reports"][control], oracle, audit))
        same(audit, COUNTS, "read-only arithmetic/forbidden dispatch census")
        for pin in (*PINS.values(), *result["diagnostic_sources"].values(),
                    *result["authenticated_files"], *preflight["sources"]):
            bound_bytes(pin)
    return {
        "diagnostic_id": NAME, "version": 1, "artifact_authentication": "AUTHENTICATED",
        "status": "SUPPORTED" if all(c["status"] == "SUPPORTED" for c in controls) else "REJECTED",
        "preregistration": PREREGISTRATION, "controls": controls, "parent_pins": PINS,
        "operand_pins": operands, "L23_reference_authority": authority,
        "original_thresholds": result["original_thresholds"], "flags": FLAGS,
        "dispatch_and_write_audit": audit, "capture_preflight": preflight, "tests": tests,
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "retained_lineage_separation": result["retained_lineage_separation"],
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "timing_seconds": {"authentication": authenticated_seconds,
                           "exact_accounting_and_final_binding": time.monotonic() - start
                           - authenticated_seconds},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError,
            ArithmeticError, ET.ParseError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN", "flags": FLAGS,
                          "error_type": type(error).__name__, "error": str(error),
                          "normal_host_review": "REQUIRED"}, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    from importlib import import_module

    raise SystemExit(import_module(MODULE).main())
