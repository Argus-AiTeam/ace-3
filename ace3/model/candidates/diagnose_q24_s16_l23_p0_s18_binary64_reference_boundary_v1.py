"""Read-only exact S18 boundary diagnosis; never dispatch a retained operator.

The command-only bootstrap directory is consumed once. All arithmetic below is
rational accounting of retained words, not a trajectory or counterfactual replay.
"""

import argparse
from bisect import bisect_left, bisect_right
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import binary64_fp16_excess_v1 as policy
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational
from ace3.model.candidates import q24_s16_l23_p0_s18_immediate_operand_combined_sufficiency_v1 as parent


ROOT, final, base, layer = parent.ROOT, parent.final, parent.base, parent.layer
require, same, read_bound, record = parent.require, parent.same, parent.read_bound, parent.record
NAME = "diagnose_q24_s16_l23_p0_s18_binary64_reference_boundary_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
PREVIOUS_OUTPUT = ROOT / "build/q24_s16_l23_p0_s18_binary64_reference_boundary_v1_attempt001"
OUTPUT = ROOT / "build/q24_s16_l23_p0_s18_binary64_reference_boundary_v1_attempt002"
REVISION = 2
TASK = "209cf7a65bdc"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {final.PYTHON} -B -m {MODULE} --execute"
RESULT_PIN = {
    "path": str(parent.OUTPUT / "result.json"), "bytes": 15877095,
    "sha256": "30746da7c96c1647352c0f550e8a023451a3808e9bd6ae11e44cf30d59877740",
}
CONTROLS = parent.CONTROLS
EXPECTED_TESTS = 27
BOUNDARY = (
    "Read-only rational accounting of reviewed combined S12/S17 FP16 operands and "
    "S18 output, against separate original-input FP16 and independently propagated "
    "binary64 L23 references. Fixed-operand incompatibility is not inability of "
    "FP16 words to pass: nearest-reference FP16 words always have zero excess. "
    "No upstream binary64 stages are reconstructed and no operand change is executed. "
    "All historical failures and thresholds remain unchanged. Final-head rescue is "
    "NOT_DEFINED; no final-head computation or new threshold. Q24 residual state is "
    "wider than FP16; native S16 RTZ, G128 asymmetric packed INT4 GEMM nibble ordering, "
    "no qzero plus-one, FP16 scales/operator boundaries/KV remain unchanged. "
    "No prefix/admission/reference/counterfactual replay, hardware/RTL/GPU/FPGA/"
    "simulation, ACE2 changes, precision/scale expansion, strict-FP16-state W4A16, "
    "new-token or full-model admission. Normal independent Host review required."
)


def finite_table():
    pairs = sorted((rational.fp16_value(word), word)
                   for word in range(65536) if word & 0x7c00 != 0x7c00)
    return tuple(value for value, _ in pairs), tuple(word for _, word in pairs)


def nearest_words(value, table, low=0, high=None):
    values, words = table
    high = len(values) if high is None else high
    require(low < high, "empty passable interval")
    index = bisect_left(values, value, low, high)
    candidates = {max(low, min(high - 1, index)),
                  max(low, min(high - 1, index - 1))}
    distance = min(abs(values[i] - value) for i in candidates)
    selected = set()
    for i in candidates:
        if abs(values[i] - value) == distance:
            selected.update(range(bisect_left(values, values[i], low, high),
                                  bisect_right(values, values[i], low, high)))
    return [words[i] for i in sorted(selected)], distance


def scalar(left_word, right_word, actual_word, reference_word, reference64, table):
    left, right, actual, fp16 = map(
        rational.fp16_value, (left_word, right_word, actual_word, reference_word))
    reference = Fraction.from_float(reference64)
    require(np.isfinite(reference64) and abs(reference) <= 65504,
            "nonfinite or out-of-range binary64 reference")
    exact = left + right
    scaled = exact * (1 << 24)
    require(scaled.denominator == 1, "S18 sum outside exact Q24 grid")
    tag = int(left_word == right_word == 0x8000)
    expected_word = rational.project(scaled.numerator, tag)
    expected = rational.fp16_value(expected_word)
    nearest, floor = nearest_words(reference, table)
    radius = floor + policy.EXCESS_BUDGET
    values, words = table
    low, high = (bisect_left(values, reference - radius),
                 bisect_right(values, reference + radius))
    passable, correction = nearest_words(actual, table, low, high)
    error = abs(actual - reference)
    excess = error - floor
    require(excess >= 0, "negative excess")
    return {
        "actual_substituted_fp16_word": f"{actual_word:04x}",
        "actual_substituted_value": str(actual),
        "fp16_reference_word": f"{reference_word:04x}",
        "fp16_reference_value": str(fp16),
        "operand_stage12_word": f"{left_word:04x}",
        "operand_stage17_word": f"{right_word:04x}",
        "operand_stage12_value": str(left), "operand_stage17_value": str(right),
        "independent_rational_s18": str(exact),
        "independent_q24_i": scaled.numerator, "independent_q24_z": tag,
        "independent_rne_word": f"{expected_word:04x}",
        "local_word_matches": actual_word == expected_word,
        "fp16_reference_matches_independent_rne": reference_word == expected_word,
        "reference_binary64_hex": reference64.hex(),
        "reference_binary64_rational": str(reference),
        "nearest_binary64_fp16_words": [f"{word:04x}" for word in nearest],
        "representation_floor": str(floor), "actual_error": str(error),
        "excess_error": str(excess), "excess_budget": str(policy.EXCESS_BUDGET),
        "threshold_margin": str(policy.EXCESS_BUDGET - excess),
        "accepted": excess <= policy.EXCESS_BUDGET,
        "independent_rne_accepted": abs(expected - reference) <= radius,
        "unrounded_sum_within_binary64_error_radius": abs(exact - reference) <= radius,
        "fixed_operand_minus_binary64": str(exact - reference),
        "actual_minus_fixed_operand": str(actual - exact),
        "actual_minus_binary64": str(actual - reference),
        "fp16_reference_minus_binary64": str(fp16 - reference),
        "passable_interval": {
            "order": "all finite FP16 words ascending exact value then unsigned word",
            "first_index": low, "last_index": high - 1, "count": high - low,
            "first_word": f"{words[low]:04x}", "last_word": f"{words[high - 1]:04x}",
            "lower_inclusive": str(reference - radius),
            "upper_inclusive": str(reference + radius),
        },
        "nearest_passable_words_to_actual": [f"{word:04x}" for word in passable],
        "minimum_passable_word_change": str(correction),
        "passable_signed_changes": [str(rational.fp16_value(word) - actual)
                                    for word in passable],
    }


def decide(rows):
    failures = [row for row in rows if not row["accepted"]]
    local = [row["index"] for row in rows if not row["local_word_matches"]
             or not row["local_state_matches"]]
    reference = [row["index"] for row in rows
                 if not row["fp16_reference_matches_independent_rne"]]
    if local:
        decision = "LOCAL_S18_INTEGRATION_DISCREPANCY"
    elif reference:
        decision = "FP16_REFERENCE_S18_BOUNDARY_DISCREPANCY"
    elif failures:
        decision = "FIXED_FP16_OPERAND_REFERENCE_BOUNDARY_INCOMPATIBILITY"
    else:
        decision = "NO_REMAINING_S18_FAILURE"
    return {
        "status": "UNKNOWN" if local or reference else "COMPLETE",
        "decision": decision, "coordinates": len(rows),
        "binary64_failures": len(failures),
        "failure_indices": [row["index"] for row in failures],
        "local_discrepancy_indices": local,
        "fp16_reference_discrepancy_indices": reference,
        "failure_passable_word_counts": sorted({row["passable_interval"]["count"]
                                              for row in failures}),
        "failure_unrounded_sum_outside_budget_count": sum(
            not row["unrounded_sum_within_binary64_error_radius"] for row in failures),
        "fp16_representation_cannot_pass": False,
        "actionable_local_s18_discrepancy": bool(local),
        "fixed_operand_correct_rne_can_rescue": not any(
            not row["independent_rne_accepted"] for row in rows),
    }


def terminal_review():
    directory = layer.preflight.HANDOFFS / parent.TASK
    latest = json.loads((directory / "latest.json").read_bytes())
    require(latest["kind"] == "handoff_ref", "combined evidence lacks terminal review")
    path = Path(latest["handoff"]["path"])
    require(path.parent == directory and path.resolve() == path, "review path escaped")
    pin = record(path)
    review = json.loads(read_bound(pin))
    backlog = [json.loads(line) for line in layer.preflight.BACKLOG.read_text().splitlines()
               if line.strip()]
    final.preflight.check_review(review, latest, backlog, mission=parent.TASK,
                                 round_number=review["round"], pin=pin)
    mission_path = directory / "mission.json"
    same(latest["mission"]["path"], str(mission_path), "review mission path changed")
    mission = json.loads(mission_path.read_bytes())
    require(mission["mission_id"] == parent.TASK
            and mission["execution_workdir"] == str(ROOT), "review scope changed")
    return {"pin": pin, "record": review, "backlog_status": "done",
            "outcome_review_status": "done", "mission": record(mission_path)}


def check_combined_header(result):
    require(result["task_id"] == parent.TASK and result["revision"] == 1
            and result["diagnostic_id"] == parent.NAME
            and result["execution_status"] == "COMPLETE"
            and result["execution_count"] == 1 and result["status"] == "FAIL",
            "combined evidence identity/execution changed")
    same(result["flags"], parent.FLAGS, "combined non-admission flags changed")
    same(result["audit"]["file_write_opens"], 12, "combined write census changed")
    # The predecessor checks its census before its twelfth (result.json) write.
    parent.check_counts({**result["audit"], "file_write_opens": 11})
    require(result["final_rescued"] == "NOT_DEFINED", "final-head boundary changed")


def authenticate():
    review = terminal_review()
    result = json.loads(read_bound(RESULT_PIN))
    check_combined_header(result)
    for pin in result["origins"].values():
        read_bound(pin)
    pins = {pin["path"]: pin for pin in result["artifacts"]}
    expected = {"command.json", "compile_tests.json"} | {c + ".npz" for c in CONTROLS}
    same(sorted(pins), sorted(str(parent.OUTPUT / name) for name in expected),
         "combined artifact manifest changed")
    same(len(result["artifacts"]), len(expected), "duplicate combined artifact")
    same(sorted(p.name for p in parent.OUTPUT.iterdir()), sorted(expected | {"result.json"}),
         "combined artifact census changed")
    for name in ("command.json", "compile_tests.json"):
        read_bound(pins[str(parent.OUTPUT / name)])
    policy_sources = {}
    for row in result["controls"]:
        for gate in row["reports"][0]["binary64_v1"]["rows"]:
            for pin in gate["sources"]:
                if pin["path"] in policy_sources:
                    same(pin, policy_sources[pin["path"]], "conflicting policy source pin")
                policy_sources[pin["path"]] = pin
    same(sorted(policy_sources), sorted(policy.SOURCE_PATHS), "policy source census changed")
    for pin in policy_sources.values():
        read_bound({**pin, "path": str(ROOT / pin["path"])})
    baseline, _, _, authenticated_pins = base.retained.authenticate()
    same(result["preflight"], baseline["preflight"], "combined reference/lineage splice")
    same(result["thresholds"], baseline["preflight"]["thresholds"], "threshold change")
    _, trajectory, reference64, checkpoint = layer.load_inputs(
        {"summary": result["preflight"]})
    same(final.preflight.bind_assets(checkpoint), result["preflight"]["assets"],
         "model operand identity changed")
    item = result["preflight"]["L23_original_reference"]["reference"]
    require(item["prior_kv"] == "own empty P0", "reference KV/source changed")
    same([row["control"] for row in result["controls"]], list(CONTROLS),
         "combined control order changed")
    payloads = []
    for row, original in zip(result["controls"], baseline["controls"], strict=True):
        same(row["parent"], original["parent"], "combined actual parent lineage changed")
        same(row["raw_payload"], pins[str(parent.OUTPUT / (row["control"] + ".npz"))],
             "raw payload splice")
        same(row["substitution"]["source"], item["fp16"], "operand source splice")
        same(row["substitution"]["fields"], ["stage12", "stage17"], "operand selection changed")
        require(row["source_operand_state_KV_lineage_checks"] == "PASS",
                "retained lineage gate defect")
        actual = base.archive(original["parent"]["terminal_archive"])
        arrays = base.archive(row["raw_payload"])
        checks = parent.prior.protected(actual, arrays, trajectory)
        same({key: row["preservation_checks"][key] for key in checks}, checks,
             "protected source/state/KV fields changed")
        for key, dtype in (("stage18", "<u2"), ("output_i", "<i8"), ("output_z", "|u1"),
                           ("stage18_operand_h", "<u2"), ("stage18_operand_i", "<i8"),
                           ("stage18_operand_z", "|u1")):
            require(arrays[key].shape == (896,) and arrays[key].dtype.str == dtype,
                    "invalid S18 array: " + key)
        for i in range(896):
            word = int(trajectory["stage12"][i])
            integer, tag = rational.root(word)
            same((int(arrays["stage18_operand_i"][i]), int(arrays["stage18_operand_z"][i]),
                  int(arrays["stage18_operand_h"][i])), (integer, tag, word),
                 "S12 operand authentication/encoding defect")
        payloads.append(arrays)
    return result, trajectory, reference64, payloads, {
        "combined_result": RESULT_PIN, "combined_review": review,
        "original_input_reference": item, "checkpoint": checkpoint,
        "baseline_authenticated_pins": authenticated_pins,
        "source_operand_state_KV_lineage": "AUTHENTICATED",
    }


@contextmanager
def write_scope(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_mutations"] += 1
        raise RuntimeError("mutation outside exclusive diagnostic writes forbidden")

    with base.write_scope(OUTPUT, audit), ExitStack() as stack:
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "rename",
                     "replace", "link", "symlink", "chmod", "truncate", "utime"):
            stack.enter_context(patch.object(os, name, forbidden))
        yield


@contextmanager
def no_dispatch(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("read-only S18 diagnosis forbids operator/replay dispatch")

    with final.no_dispatch(audit), ExitStack() as stack:
        for module, names in (
            (final, ("rmsnorm", "logits", "top_k", "run_control", "load_operands")),
            (final.norm, ("rmsnorm",)),
            (parent, ("run_control", "prepare", "focused_tests")),
            (parent.prior, ("run_control", "prepare", "focused_tests")),
            (base, ("suffix_stages", "prepare", "focused_tests")),
            (parent.native, ("projection", "rne", "toward_zero")),
            (parent.native.state, ("add", "lift")),
            (layer, ("stage_report", "expected_stage")),
            (layer.prior.local, ("local_reference",)),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def fresh_output():
    require(OUTPUT.resolve() == OUTPUT and not OUTPUT.is_symlink()
            and OUTPUT.is_dir(), "missing canonical command-only output bootstrap")
    names = {path.name for path in OUTPUT.iterdir()}
    failed_bootstrap = {
        "command.sh", "bootstrap_failure.json", "preserve_failed_source.sh", "failed_source.py"}
    require(names == {"command.sh"} or names == failed_bootstrap,
            "occupied output; zero diagnostic dispatch")
    if names == failed_bootstrap:
        failure = json.loads((OUTPUT / "bootstrap_failure.json").read_bytes())
        require(failure["status"] == "UNKNOWN" and failure["command"] == COMMAND
                and failure["phase"] == "fresh_output_before_exclusive_command_reservation"
                and failure["command_json_created"] is False
                and failure["authentication_invocations"] == failure["compiled_files"]
                == failure["tests_executed"] == failure["diagnostic_controls"]
                == failure["native_dispatches"] == failure["reference_producer_invocations"]
                == failure["counterfactual_invocations"] == 0,
                "bootstrap failure is not a zero-execution pre-reservation failure")
    require(all(not (OUTPUT / name).is_symlink() for name in names),
            "bootstrap artifact symlink")
    require(not (OUTPUT / "command.sh").is_symlink()
            and (OUTPUT / "command.sh").read_text() == COMMAND + "\n",
            "command sidecar differs from exact launch command")
    check = final.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(OUTPUT)],
        check=False, capture_output=True)
    require(check.returncode == 0, "diagnostic output not ignored")
    return [record(OUTPUT / name) for name in sorted(names)]


def focused_tests(origins):
    compiled = []
    for pin in {pin["path"]: pin for pin in origins.values()}.values():
        compile(read_bound(pin), pin["path"], "exec", dont_inherit=True)
        compiled.append(pin)
    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    require(spec is not None and spec.loader is not None, "test loader missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    require(result.wasSuccessful() and not result.skipped, "focused tests failed:\n" + output.getvalue())
    return {"compiled": compiled, "executed": result.testsRun, "failures": 0, "errors": 0,
            "skipped": 0, "raw_output": output.getvalue(), "retained_operator_dispatches": 0}


def control_report(row, arrays, trajectory, reference64, table):
    reports = row["reports"]
    require(len(reports) == 1 and reports[0]["node"] == [23, 0, 18],
            "retained report boundary changed")
    stored = reports[0]["binary64_v1"]["rows"]
    same([r["index"] for r in stored], list(range(896)), "retained coordinate census changed")
    failures = {r["index"] for r in stored if not r["accepted"]}
    same(len(failures), 110, "retained 110-failure scope changed")
    neighbors = {i + offset for i in failures for offset in (-1, 1)
                 if 0 <= i + offset < 896} - failures
    rows = []
    for i, gate in enumerate(stored):
        value = scalar(int(trajectory["stage12"][i]), int(trajectory["stage17"][i]),
                       int(arrays["stage18"][i]), int(trajectory["stage18"][i]),
                       float(reference64[i]), table)
        for key, field in (("actual_fp16_bits", "actual_substituted_fp16_word"),
                           ("reference_binary64_hex", "reference_binary64_hex"),
                           ("q", "representation_floor"), ("actual_error", "actual_error"),
                           ("excess_error", "excess_error"), ("excess_budget", "excess_budget"),
                           ("accepted", "accepted")):
            same(gate[key], value[field], "retained scalar gate defect: " + key)
        require(gate["nearest_fp16_bits"] in value["nearest_binary64_fp16_words"],
                "retained nearest word defect")
        integer, tag = int(arrays["output_i"][i]), int(arrays["output_z"][i])
        rational.validate_state(integer, tag)
        value.update({
            "index": i, "retained_gate": "FAIL" if i in failures else "PASS",
            "is_failure_neighbor": i in neighbors,
            "actual_q24_i": integer, "actual_q24_z": tag,
            "local_state_matches": (integer, tag) == (
                value["independent_q24_i"], value["independent_q24_z"]),
        })
        rows.append(value)
    return {"control": row["control"], **decide(rows), "rows": rows,
            "surrounding_indices": sorted(neighbors), "parent": row["parent"],
            "substitution": row["substitution"], "raw_input": row["raw_payload"],
            "final_rescued": "NOT_DEFINED", "historical_S18_status": row["status"],
            "source_operand_state_KV_lineage": "AUTHENTICATED"}


def execute():
    bootstrap = fresh_output()
    previous_artifacts = [record(path) for path in sorted(PREVIOUS_OUTPUT.iterdir())]
    previous_failure = json.loads(read_bound(next(
        pin for pin in previous_artifacts if pin["path"] == str(PREVIOUS_OUTPUT / "failure.json"))))
    require(previous_failure["status"] == "UNKNOWN"
            and previous_failure["audit"]["diagnostic_controls"] == 0
            and previous_failure["audit"]["rational_coordinate_comparisons"] == 0,
            "recovery requires the preserved pre-authentication UNKNOWN attempt")
    recovery = {
        "attempt": 2, "previous_output": str(PREVIOUS_OUTPUT),
        "previous_status": previous_failure["status"], "previous_artifacts": previous_artifacts,
        "authorization": "mission round 2 live session-rotation directive: execute one recovery now",
    }
    audit = {"file_write_opens": 0, "forbidden_calls": 0, "forbidden_mutations": 0,
             "native_dispatches": 0, "reference_producer_invocations": 0,
             "counterfactual_invocations": 0, "final_head_invocations": 0,
             "external_invocations": 0, "retained_evidence_writes": 0,
             "diagnostic_controls": 0, "rational_coordinate_comparisons": 0,
             "preexisting_bootstrap_artifact_count": len(bootstrap)}

    def save(name, value):
        with (OUTPUT / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    started = time.monotonic()
    with write_scope(audit):
        save("command.json", {
            "command": COMMAND, "command_sidecar": record(OUTPUT / "command.sh"),
            "argv": sys.argv, "task_id": TASK, "revision": REVISION, "recovery": recovery,
            "bootstrap_artifacts": bootstrap,
            "cwd": str(ROOT), "uid": os.getuid(), "executable": sys.executable,
            "execution_allowance": "one read-only diagnostic; exclusive command.json reservation",
            "host_managed_controls": "unchanged role/model/account/budget/access/concurrency",
        })
    try:
        origins = {**final.source_context(), "diagnostic_source": record(SOURCE),
                   "diagnostic_test": record(TEST)}
        require(Path(__file__).resolve() == SOURCE, "diagnostic source origin changed")
        with write_scope(audit), no_dispatch(audit):
            tests = focused_tests(origins)
            save("compile_tests.json", tests)
            baseline, trajectory, reference64, payloads, authentication = authenticate()
            auth_end = time.monotonic()
            table = finite_table()
            rows = []
            with (OUTPUT / "references.npz").open("xb") as stream:
                np.savez(stream, binary64=reference64, **trajectory)
            for row, arrays in zip(baseline["controls"], payloads, strict=True):
                entry = control_report(row, arrays, trajectory, reference64, table)
                with (OUTPUT / (row["control"] + ".npz")).open("xb") as stream:
                    np.savez(stream, **arrays)
                entry["raw_payload"] = record(OUTPUT / (row["control"] + ".npz"))
                rows.append(entry)
                audit["diagnostic_controls"] += 1
                audit["rational_coordinate_comparisons"] += len(entry["rows"])
                print(json.dumps({k: v for k, v in entry.items()
                                  if k not in ("rows", "parent", "substitution")}), flush=True)
            calculation_end = time.monotonic()
        same({**final.source_context(), "diagnostic_source": record(SOURCE),
              "diagnostic_test": record(TEST)}, origins, "source changed during diagnosis")
        same([record(path) for path in sorted(PREVIOUS_OUTPUT.iterdir())],
             previous_artifacts, "preserved UNKNOWN attempt changed during recovery")
        decisions = {row["decision"] for row in rows}
        status = "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "UNKNOWN"
        result = {
            "task_id": TASK, "revision": REVISION, "diagnostic_id": NAME, "status": status,
            "recovery": recovery,
            "decision": next(iter(decisions)) if len(decisions) == 1 else "MIXED",
            "execution_count": 1, "controls": rows, "authentication": authentication,
            "thresholds": baseline["thresholds"], "origins": origins, "tests": tests,
            "audit": {**audit, "file_write_opens": audit["file_write_opens"] + 1},
            "total_artifact_creations": audit["file_write_opens"] + 1 + len(bootstrap),
            "bootstrap_artifacts": bootstrap,
            "output": str(OUTPUT), "expected_file_count": 13 + len(bootstrap),
            "timing_seconds": {"compile_test_authentication": auth_end - started,
                               "rational_comparison_and_payload_writes": calculation_end - auth_end},
            "flags": parent.FLAGS, "claim_boundary": BOUNDARY,
            "final_rescued": "NOT_DEFINED", "automatic_replay_authorized": False,
            "normal_host_review": "REQUIRED",
        }
        with write_scope(audit):
            save("result.json", result)
        same(audit, result["audit"], "write/dispatch census mismatch")
        expected_names = {Path(pin["path"]).name for pin in bootstrap} | {
            "command.json", "compile_tests.json", "result.json", "references.npz"
        } | {control + ".npz" for control in CONTROLS}
        same(sorted(path.name for path in OUTPUT.iterdir()), sorted(expected_names),
             "output file census mismatch")
        print(json.dumps({key: result[key] for key in (
            "task_id", "status", "decision", "audit", "output",
            "total_artifact_creations", "claim_boundary")}), flush=True)
        return result
    except Exception as error:
        with write_scope(audit):
            save("failure.json", {"task_id": TASK, "status": "UNKNOWN",
                                  "error_type": type(error).__name__, "error": str(error),
                                  "audit": {**audit, "file_write_opens": audit["file_write_opens"] + 1},
                                  "automatic_replay_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    from importlib import import_module

    import_module(MODULE).main()
