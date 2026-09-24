"""Read-only coordinate-761 RNE cell anatomy, not an S18 or final-head repair."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import io
import json
from pathlib import Path
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_full_vector_s18_boundary_diagnostic_v1 as previous


NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mlp_stage17_s18_coordinate761_rne_cell_anatomy_v1"
MODULE = "ace3.model.candidates." + NAME
MISSION = "456967d2260e"
ROOT, PYTHON = previous.ROOT, previous.PYTHON
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
CONTROLS, FLAGS, ENVIRONMENT = previous.CONTROLS, previous.FLAGS, previous.ENVIRONMENT
capture, require, same, bound_bytes = (
    previous.capture, previous.require, previous.same, previous.bound_bytes)
verify_members, rational = previous.verify_members, previous.rational
COUNTS = {**dict.fromkeys(previous.COUNTS, 0),
          "exact_coordinate_accounts": 9, "independent_scalar_oracles": 9}
PARENT_ROOT = ROOT / "build/mlp17-full-vector-s18-boundary-f3e815be2d42-attempt001"
PINS = {
    "stdout": {"path": str(PARENT_ROOT / "run/check.stdout"), "bytes": 1859998,
               "sha256": "0d49fb0298bc16360721173dd7997edf28c1b3fff32632b5a36eab254a4b8362"},
    "capture": {"path": str(PARENT_ROOT / "run/capture.json"), "bytes": 20252,
                "sha256": "89a2360880a55e54d2917730aa541f3e012169f7826fb03a4a5ee2827ab08aaf"},
    "outer": {"path": str(PARENT_ROOT / "launcher.capture.json"), "bytes": 1493,
              "sha256": "018aef496efb9949af123e4e71dc3fe18010a01d96718958e574fbac44f4e83f"},
    "review": {
        "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/f3e815be2d42/round-0001.json",
        "bytes": 689, "sha256": "5501e5c3c9b8343275ea993315c171e1122576587bc889bf6f32eb65b4ada3e8"},
}
BOUNDARY = (
    "One stdout-only read-only CPU coordinate-761 RNE-cell diagnostic of reviewed "
    "f3 and authenticated f96 bytes. Exact scalar arithmetic only; no prefix, "
    "accepted/closed producer, native decoder, admission, original reference, "
    "intervention, passable-word or final-margin bridge execution. Preserve Q24-wide "
    "scratch, native S16 RTZ, G128 asymmetric packed INT4 native GEMM nibble order "
    "without qzero plus-one, FP16 scales/operator/KV boundaries, independently "
    "propagated original-input global references and excess <= 1/8. No hardware, "
    "RTL, GPU, ACE2, precision/scale change, strict-FP16-state W4A16, new-token or "
    "full-model admission. Terminate this mechanism branch after this diagnostic "
    "and normal independent Host review; no successor scheduled."
)
PREREGISTRATION = {
    "SUPPORTED": "Eight authenticated inside-before/fail-after cases have ordinary "
                 "non-anomalous outward RNE-cell crossings; mapped_all alone retains "
                 "83/83 failed-coordinate unrounded sums outside. Close only this "
                 "coordinate-761 mechanism branch as non-admission evidence.",
    "REJECTED": "Any arithmetic/state/KV/oracle, operand-view, cell/tie/zero/subnormal "
                "anomaly or failure of the eight-case prediction. Localize to S18 "
                "arithmetic/integrity, not downstream repair.",
    "UNKNOWN": "Missing or unauthenticated review/source/test/capture or failed "
               "execution/account/environment/budget gates. Localize to evidence availability.",
}


def authenticate():
    documents = {key: json.loads(bound_bytes(pin)) for key, pin in PINS.items()}
    review, inner, outer, result = (documents[k] for k in ("review", "capture", "outer", "stdout"))
    same((review["kind"], review["mission_id"], review["producer_role"],
          review["round"], review["review"]["status"]),
         ("round_reviewed_handoff", previous.MISSION, "reviewer", 1, "done"), "f3 review")
    require(inner["success"] is True and inner["failure"] is None
            and outer["success"] is True and outer["exit_status"] == 0
            and outer["timed_out"] is False, "f3 capture failure")
    preflight = inner["preflight"]
    same((preflight["mission_id"], preflight["role"], preflight["uid"], preflight["cwd"],
          preflight["python"], preflight["environment"], preflight["scope"]),
         (previous.MISSION, "engineer", 1000, str(ROOT), PYTHON, ENVIRONMENT, previous.BOUNDARY),
         "f3 identity/environment/scope")
    same(preflight["command_budget"], {"compile": 1, "pytest": 1, "check": 1}, "f3 budget")
    same((preflight["independent_host_review"], preflight["model_or_service_calls_authorized"]),
         ("REQUIRED", 0), "f3 review/service authority")
    same(preflight["sources"], [capture.binding(previous.SOURCE), capture.binding(previous.TEST)],
         "f3 source/test pins")
    same(inner["sources_after"], preflight["sources"], "f3 source drift")
    for pins in (preflight["capture_implementation"], inner["capture_implementation_after"],
                 outer["capture_implementation_after"]):
        same(pins, capture.implementation_pins(), "f3 capture implementation drift")
    bound_bytes(preflight["executable"])
    same([r["label"] for r in inner["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"], "f3 budget census")
    verify_members(PARENT_ROOT / "run", inner)
    same(inner["results"][-1]["argv"], [PYTHON, "-B", "-m", previous.MODULE, "--check"],
         "f3 command")
    same(inner["results"][-1]["files"][3], PINS["stdout"], "f3 stdout splice")
    same([p["path"] for p in outer["files"]],
         [str(PARENT_ROOT / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "f3 outer paths")
    identity, output, error, whole = [bound_bytes(pin) for pin in outer["files"]]
    launch = json.loads(identity)
    same((launch["argv"], launch["cwd"], launch["uid"], launch["environment"]),
         ([PYTHON, "-B", str(previous.TEST), "--run", str(PARENT_ROOT / "run")],
          str(ROOT), 1000, ENVIRONMENT), "f3 outer identity")
    same(launch["launcher"], preflight["sources"][1], "f3 launcher source")
    same(launch["capture_implementation"], capture.implementation_pins(), "f3 launcher helper")
    same(launch["command"], capture.environment_command(ENVIRONMENT, launch["argv"]), "f3 outer command")
    same(error, b"", "f3 outer stderr")
    same(whole, b"IDENTITY\n" + identity + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "f3 outer whole-command")
    same(json.loads(output), {"capture": PINS["capture"], "success": True, "failure": None},
         "f3 inner/outer splice")
    same((result["diagnostic_id"], result["version"], result["artifact_authentication"],
          result["status"]), (previous.NAME, 1, "AUTHENTICATED", "REJECTED"), "f3 result")
    same(result["flags"], FLAGS, "f3 non-admission flags")
    same(result["dispatch_and_write_audit"], previous.COUNTS, "f3 arithmetic-only census")
    same(result["capture_preflight"], preflight, "f3 preflight splice")
    same(result["tests"]["compiled"], preflight["sources"], "f3 compiled pins")
    suites = ET.fromstring(bound_bytes(result["tests"]["pytest"])).findall("testsuite")
    require(sum(int(s.attrib["tests"]) for s in suites) == result["tests"]["executed"] == 42
            and not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
            "f3 focused tests")
    same(result["parent_pins"], previous.PINS, "f3/f96 lineage pins")
    parent = previous.authenticate()  # Authentication only, never the closed check or producer.
    for key in ("L23_reference_authority", "original_thresholds", "retained_lineage_separation"):
        same(result[key], parent[key], "f3/f96 " + key)
    same((result["historical_failures_preserved"], result["original_global_reference_unchanged"]),
         (True, True), "f3 preserved authority")
    for pin in result["operand_pins"].values():
        require(pin in parent["authenticated_files"], "f3/f96 operand splice")
        bound_bytes(pin)
    return result, parent


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("closed diagnostic/producer execution forbidden")

    with previous.read_only(audit), patch.object(previous, "check", forbidden), patch.object(
            previous, "diagnose_control", forbidden):
        yield


def capture_preflight():
    with patch.multiple(previous.previous, SOURCE=SOURCE, TEST=TEST, MODULE=MODULE, MISSION=MISSION):
        return previous.previous.capture_preflight()


def rne_cell(word):
    center = rational.fp16_value(word)
    magnitude = word & 0x7fff
    if magnitude == 0:
        lower, upper = -Fraction(1, 1 << 25), Fraction(1, 1 << 25)
    else:
        value = abs(center)
        below = rational.fp16_value(magnitude - 1)
        above = rational.fp16_value(magnitude + 1) if magnitude < 0x7bff else Fraction(65536)
        lower, upper = (below + value) / 2, (value + above) / 2
        if word & 0x8000:
            lower, upper = -upper, -lower
    return {"lower": str(lower), "upper": str(upper), "center": str(center),
            "significand_parity": "odd" if word & 1 else "even",
            "lower_tie_owned": not bool(word & 1), "upper_tie_owned": not bool(word & 1),
            "zero_sign_rule": "negative only for two negative-zero operands; cancellation is positive"}


def anatomy(row, oracle):
    flags = dict.fromkeys((
        "malformed_operands", "signed_zero", "subnormal", "overflow", "unexpected_tie",
        "scratch_view_substitution", "state_mismatch", "oracle_mismatch",
        "cell_geometry_mismatch", "retained_account_mismatch", "capture_source_drift"), False)
    result = {"account": row, "anomaly_flags": flags, "outward_radius_crossing": False}
    try:
        word = int(row["actual_fp16_word"], 16)
        total = Fraction(row["unrounded_fixed_operand_sum"])
        rounded = rational.fp16_value(word)
        reference = Fraction(row["original_input_binary64_value"])
        radius = Fraction(row["binary64_authority_radius"])
        cell = rne_cell(word)
        lower, upper = Fraction(cell["lower"]), Fraction(cell["upper"])
        in_cell = ((lower < total < upper)
                   or (total == lower and cell["lower_tie_owned"])
                   or (total == upper and cell["upper_tie_owned"]))
        before, after = radius - abs(total - reference), radius - abs(rounded - reference)
        movement = abs(rounded - reference) - abs(total - reference)
        flags.update(
            signed_zero=total == 0 or any(row[k] == "8000" for k in
                                         ("stage12_fp16_word", "stage17_fp16_word", "actual_fp16_word")),
            subnormal=any(0 < (int(row[k], 16) & 0x7fff) < 0x400 for k in
                          ("stage12_fp16_word", "stage17_fp16_word", "actual_fp16_word")),
            overflow=abs(total) >= 65520,
            unexpected_tie=total in (lower, upper),
            scratch_view_substitution=not row["scratch_view_matches"]
            or Fraction(row["scratch_exact_value"]) != Fraction(row["scratch_i"], 1 << 24)
            or total != Fraction(row["scratch_i"], 1 << 24)
            + rational.fp16_value(int(row["stage17_fp16_word"], 16)),
            state_mismatch=not row["state_matches"] or not row["rne_matches"]
            or row["state_error"] is not None,
            cell_geometry_mismatch=not in_cell or not lower < rounded < upper
            or before != Fraction(row["unrounded_threshold_margin"])
            or after != Fraction(row["threshold_margin"]) or before - after != movement,
        )
        result.update(
            rne_cell=cell, unrounded_to_cell_lower=str(total - lower),
            cell_upper_to_unrounded=str(upper - total),
            authority_lower=str(reference - radius), authority_upper=str(reference + radius),
            before_rounding_radius_margin=str(before), after_rounding_radius_margin=str(after),
            outward_error_movement=str(movement), rounding_delta=str(rounded - total),
            outward_radius_crossing=before >= 0 and after < 0 and movement > 0
            and (total - reference) * (rounded - reference) > 0,
        )
        flags["oracle_mismatch"] = not oracle.verify_account(row, cell)
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        flags["malformed_operands"] = True
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def classify(controls):
    reasons = []
    if [c["control"] for c in controls] != list(CONTROLS):
        reasons.append("nine_control_identity")
    for control in controls:
        name = control["control"]
        if control["integrity_issues"] or any(control["anomaly_flags"].values()):
            reasons.append(name + ":arithmetic_or_integrity")
        if control["failure_count"] != 83 or control["unrounded_outside_radius_count"] != (
                83 if name == "mapped_all" else 82):
            reasons.append(name + ":retained_failure_geometry")
        if name != "mapped_all" and not control["outward_radius_crossing"]:
            reasons.append(name + ":missing_outward_crossing")
    return ("REJECTED" if reasons else "SUPPORTED"), reasons


def check():
    preflight, tests = capture_preflight()
    oracle = previous.previous.load_module(TEST, NAME + "_oracle")
    audit = dict.fromkeys(COUNTS, 0)
    start = time.monotonic()
    with read_only(audit):
        retained, parent = authenticate()
        authority = retained["L23_reference_authority"]
        same((authority["status"], authority["missing"]), ("BOUND_ORIGINAL_INPUT_L23", []),
             "original-input authority")
        bound_bytes(authority["manifest"])
        reference = previous.previous.layer.archive(authority["reference"]["fp16"])
        binary64 = np.load(io.BytesIO(bound_bytes(authority["reference"]["binary64"])),
                           allow_pickle=False)
        require(binary64.dtype.str == "<f8" and binary64.shape == (896,)
                and np.isfinite(binary64).all(), "binary64 authority encoding")
        require(all(reference[k].dtype.str == "<u2" and reference[k].shape == (896,)
                    for k in ("stage17", "stage18")), "FP16 authority encoding")
        same([c["control"] for c in retained["controls"]], list(CONTROLS), "retained controls")
        controls = []
        authenticated_seconds = time.monotonic() - start
        for old in retained["controls"]:
            name, issues = old["control"], []
            actual = previous.previous.layer.archive(retained["operand_pins"][name])
            output, report = parent["outputs"][name], parent["stage_reports"][name]
            for key in ("residual_state_lineage", "kv_lineage"):
                if report[key] != "PASS":
                    issues.append(key)
            for kind, stage in (("k", "05"), ("v", "03")):
                cache = actual["output_cache_" + kind]
                if cache.dtype.str != "<u2" or cache.shape != (1, 128) or not np.array_equal(
                        cache[0], actual["stage" + stage]):
                    issues.append("kv_" + kind)
            for key in ("scratch_i", "scratch_z", "stage12", "input_i", "input_z", "input_hidden"):
                require(actual[key].shape == (896,), "operand shape")
            for key in ("stage17", "stage18", "output_i", "output_z"):
                require(len(output[key]) == 896 and all(type(v) is int for v in output[key]),
                        "f96 output encoding")
            k = 761
            if output["stage17"][k] != int(reference["stage17"][k]):
                issues.append("original_stage17_operand")
            if rational.project(int(actual["input_i"][k]), int(actual["input_z"][k])) != int(
                    actual["input_hidden"][k]):
                issues.append("input_state_view")
            # This is the shared pure scalar account, not a producer or vector diagnostic.
            row = previous.coordinate(
                k, int(actual["scratch_i"][k]), int(actual["scratch_z"][k]),
                int(actual["stage12"][k]), output["stage17"][k], output["output_i"][k],
                output["output_z"][k], output["stage18"][k], int(reference["stage18"][k]),
                float(binary64[k]))
            account = anatomy(row, oracle)
            audit["exact_coordinate_accounts"] += 1
            audit["independent_scalar_oracles"] += 1
            row["oracle_matches"] = not account["anomaly_flags"]["oracle_mismatch"]
            old_rows = [r for r in old["failed_coordinates"] if r["index"] == k]
            account["anomaly_flags"]["retained_account_mismatch"] = (
                len(old_rows) != (0 if name == "mapped_all" else 1)
                or bool(old_rows and old_rows[0] != row))
            if {**row["binary64_gate"], "index": k} != report["binary64_v1"]["rows"][k]:
                issues.append("f96_binary64_oracle")
            failed = old["failed_coordinates"]
            if (len(failed) != 83 or old["failure_count"] != 83
                    or old["coordinates_checked"] != 896
                    or old["q24_rne_agreement_count"] != 83 or old["oracle_agreement_count"] != 83
                    or old["fixed_operand_rne_passable_count"] != 0
                    or [r["index"] for r in failed] != old["retained_failure_indices"]
                    or old["retained_failure_indices"] !=
                    [r["index"] for r in report["binary64_v1"]["failures"]]
                    or any(not all(r[key] for key in ("state_matches", "rne_matches",
                                                     "scratch_view_matches", "oracle_matches"))
                           or r["accepted"] or r["fixed_operand_rne_passable"] for r in failed)
                    or sum(r["unrounded_outside_radius"] for r in failed)
                    != old["unrounded_outside_radius_count"]):
                issues.append("f3_retained_failure_integrity")
            controls.append({**account, "control": name, "integrity_issues": issues,
                             "retained_f3_account_present": bool(old_rows),
                             "account_origin": "f3_and_f96" if old_rows else "f96_raw_operands",
                             "operand_pin": retained["operand_pins"][name],
                             "failure_count": old["failure_count"],
                             "unrounded_outside_radius_count": old["unrounded_outside_radius_count"]})
        same(audit, COUNTS, "read-only scalar/forbidden operator census")
        for pin in (*PINS.values(), *previous.PINS.values(), *preflight["sources"],
                    *retained["tests"]["compiled"], *parent["diagnostic_sources"].values(),
                    *parent["authenticated_files"]):
            bound_bytes(pin)
    status, reasons = classify(controls)
    return {
        "diagnostic_id": NAME, "version": 1, "artifact_authentication": "AUTHENTICATED",
        "status": status, "rejection_reasons": reasons, "preregistration": PREREGISTRATION,
        "controls": controls, "parent_pins": {"f3": PINS, "f96": previous.PINS},
        "L23_reference_authority": authority, "original_thresholds": parent["original_thresholds"],
        "retained_lineage_separation": parent["retained_lineage_separation"],
        "flags": FLAGS, "dispatch_and_write_audit": audit,
        "capture_preflight": preflight, "tests": tests,
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "branch_terminated": True, "successor_scheduled": False,
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
        print(json.dumps({"diagnostic_id": NAME, "status": "UNKNOWN",
                          "artifact_authentication": "UNAUTHENTICATED", "flags": FLAGS,
                          "anomaly_flags": {"capture_source_or_execution_drift": True},
                          "error_type": type(error).__name__, "error": str(error),
                          "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY},
                         sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    from importlib import import_module

    raise SystemExit(import_module(MODULE).main())
