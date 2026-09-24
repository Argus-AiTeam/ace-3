"""Fixed-row final-RMSNorm vector substitution, CPU-only and non-admission."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_accumulation_boundary_cutoff_classifier_v1 as previous


bridge, base, parent = previous.bridge, previous.base, previous.parent
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_vector_cutoff_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
TASK = "875bdd31dc4d"
PREFIX = "q24_s16_final_head_final_rmsnorm_vector_cutoff_classifier_v1_"
PARENT_OUTPUT = ROOT / "build/q24_s16_final_head_accumulation_boundary_cutoff_classifier_v1_b288dca598b7_attempt004"
REVIEW_PIN = {
    "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/b288dca598b7/round-0002.json",
    "bytes": 690, "sha256": "1ef32e18c00bab2c3343daf93fac3b17f95b04b2dc18c6ec8c2fcac3df64adc4",
}
PINS = (
    REVIEW_PIN,
    {"path": str(previous.SOURCE), "bytes": 24621,
     "sha256": "0da83caec002365fa2073da6491a2789a6dd3c93fb57d5174fecb90855494d50"},
    {"path": str(previous.TEST), "bytes": 9081,
     "sha256": "7cc4a60382fdd08eb93ed764c1a017a0b3ed822822f17cc89dc49b9929909fb4"},
    {"path": str(PARENT_OUTPUT / "result.json"), "bytes": 256446,
     "sha256": "94f433d39210745c9976691810a6fbf4a454b21e727b2b2ffd53e33588eeda78"},
    {"path": str(PARENT_OUTPUT / "raw.capture"), "bytes": 272442,
     "sha256": "3a75bde1d01e39768883c397b8fea5ab154f20dd6c661e24afc3cb25a2868902"},
    {"path": str(PARENT_OUTPUT / "capture.json"), "bytes": 3532,
     "sha256": "f8b89af7e832fcb8f84d708804447d1d89e661429f1791833e868773f913bdb8"},
)
IDS, PAIRS = previous.IDS, previous.PAIRS
require, same, sign = base.require, base.same, previous.sign
SUCCESSORS = {
    "SUPPORTED": "final-RMSNorm vector component localization: direct-hidden, scalar-scale or interaction",
    "REJECTED": "upstream retained hidden-vector or RMSNorm scalar-boundary classifier, not head accumulation",
    "UNKNOWN": "integrity repair only; no scientific successor",
}
BOUNDARY = (
    "Only the actual final-RMSNorm input vector of a fixed selected-row exact "
    "head is replaced by the independently retained original-input FP16 vector. "
    "Numeric rows 13/319/34319, row weights and original logit references stay "
    "fixed. No scalar RNE or RMSNorm execution, reference mutation, prefix/"
    "admission/reference-producer/native decoder replay, full-vocabulary head/"
    "ranking/token publication/selection, row319 provenance, middle/outer/dose "
    "branch, GPU/RTL/FPGA/ACE2, precision or scale expansion. Q24 residual state "
    "is wider than FP16; native-S16-RTZ, INT4 weights, FP16 operator boundaries/"
    "KV, exact thresholds and source/operand/state/KV/lineage gates are unchanged. "
    "Historical FAIL/UNKNOWN and non-admission limits remain intact. This "
    "reference-guided contrast is not an algorithm repair, causal dominance "
    "claim, strict-FP16-state W4A16, new-token or full-model admission."
)
ENVIRONMENT = {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
ARGV = [parent.PYTHON, "-B", "-m", MODULE, "--check"]
COMMAND = " ".join(f"{k}={shlex.quote(v)}" for k, v in ENVIRONMENT.items()) + " " + shlex.join(ARGV)
EXPECTED_TESTS = 10


@contextmanager
def selected_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("vector classifier forbids prior classifier or scalar-RNE replay")

    with previous.selected_only(audit), patch.multiple(
        previous, execute_check=forbidden, run_selected=forbidden,
        run_tests=forbidden, authenticate=forbidden, round_half=forbidden,
    ):
        yield


def identity():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items())
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/environment gate failed")


def key(row):
    return row["control"], row["left_id"], row["right_id"], row["branch"]


def census(records):
    expected = {(c, a, b, branch) for c in parent.CONTROLS
                for a, b in PAIRS for branch in ("fp16", "binary64")}
    keys = [key(row) for row in records]
    require(len(keys) == len(set(keys)) == 54 and set(keys) == expected,
            "retained control/pair/branch census changed")
    for row in records:
        previous.account_closure(row)


def prediction(row):
    previous.account_closure(row)
    a, r = (Fraction(row[k]) for k in ("retained_actual_margin", "retained_reference_margin"))
    return {
        "control": row["control"], "left_id": row["left_id"], "right_id": row["right_id"],
        "branch": row["branch"], "role": previous.ROLES[row["left_id"], row["right_id"]],
        "predicted_margin_movement": str(r-a),
        "predicted_movement_sign": sign(r-a),
        "predicted_crossing_pattern": [sign(a), sign(r)],
    }


def preregister(records):
    census(records)
    return {
        "version": 1, "task": TASK, "frozen_selected_row_ids": list(IDS),
        "primary_branch": "fp16", "pair_count": 27, "prediction_source": PINS[3],
        "prediction": (
            "Before selected arithmetic: for all nine controls and three ordered "
            "fixed-cutoff/exchanged pairs, substituting only the independent FP16 "
            "final-RMSNorm vector in the same exact selected head reproduces the "
            "retained logit movement sign sign(R-A) and crossing pattern "
            "(sign(A),sign(R)). Compare observed sign(ER-EA) and (sign(EA),sign(ER)). "
            "Exact zero is distinct from both directions. No thresholds are fitted."
        ),
        "decision_rule": "SUPPORTED iff all 27 movement signs and endpoint crossing patterns agree; otherwise REJECTED",
        "unknown_rule": "Integrity, closure, runtime, dispatch or test failure is UNKNOWN",
        "successors": SUCCESSORS,
        "rows": [prediction(r) for r in records if r["branch"] == "fp16"],
        "not_independent_samples": True,
    }


def classify_pair(row, actual_dots, swapped_dots):
    previous.account_closure(row)
    require(row["branch"] == "fp16" and len(actual_dots) == len(swapped_dots) == 2
            and all(isinstance(v, Fraction) for v in (*actual_dots, *swapped_dots)),
            "vector swap requires two exact FP16-branch row dots")
    a, r, delta, ea, er, vector, ba, nr, h = (Fraction(row[k]) for k in previous.COLUMNS[4:])
    before, after = actual_dots[0]-actual_dots[1], swapped_dots[0]-swapped_dots[1]
    movement = after-before
    residuals = {
        "actual_dot_margin_minus_parent": before-ea,
        "swapped_dot_margin_minus_parent": after-er,
        "movement_plus_vector_sum": movement+vector,
        "movement_minus_retained_movement_minus_H": movement-(r-a)-h,
        "vector_plus_H_minus_retained_total": vector+h-delta,
        "actual_anchor_closure": a-before-ba,
        "reference_anchor_closure": r-after+nr,
    }
    require(not any(residuals.values()), "vector swap does not close retained parent totals")
    p = prediction(row)
    pattern = [sign(before), sign(after)]
    return {
        **p, "parent_account": row,
        "actual_exact_dots": list(map(str, actual_dots)),
        "swapped_exact_dots": list(map(str, swapped_dots)),
        "before_exact_margin": str(before), "after_exact_margin": str(after),
        "vector_swap_margin_movement": str(movement),
        "retained_actual_margin": str(a), "immutable_reference_margin": str(r),
        "before_minus_immutable_reference": str(before-r),
        "after_minus_immutable_reference": str(after-r),
        "separate_head_boundary_change": str(h),
        "observed_movement_sign": sign(movement), "observed_crossing_pattern": pattern,
        "direction_agrees": sign(movement) == p["predicted_movement_sign"],
        "crossing_pattern_agrees": pattern == p["predicted_crossing_pattern"],
        "exact_closure_residuals": {k: str(v) for k, v in residuals.items()},
    }


def decision(accounts):
    expected = {(c, a, b, "fp16") for c in parent.CONTROLS for a, b in PAIRS}
    keys = [key(row) for row in accounts]
    require(len(keys) == len(set(keys)) == 27 and set(keys) == expected,
            "incomplete/duplicated classifier matrix")
    require(all(type(row[k]) is bool for row in accounts
                for k in ("direction_agrees", "crossing_pattern_agrees")),
            "invalid classifier agreement type")
    return ("SUPPORTED" if all(r["direction_agrees"] and r["crossing_pattern_agrees"]
                               for r in accounts) else "REJECTED")


def authenticate():
    for pin in PINS:
        base.read_bound(pin)
    review = json.loads(base.read_bound(REVIEW_PIN))
    require(review["kind"] == "round_reviewed_handoff"
            and review["mission_id"] == previous.TASK
            and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done",
            "cutoff classifier lacks terminal independent review")
    saved = json.loads(base.read_bound(PINS[3]))
    same(previous.validate_stored(PARENT_OUTPUT)["decision"], "REJECTED",
         "reviewed head-boundary successor changed")
    records = [r["parent_account"] for r in saved["parent_closures"]]
    census(records)
    result, arrays, references, files, assets = bridge.contributions.authenticate()
    same(result["controls"], saved["retained_controls_and_failure_gates"],
         "historical failure/control gates changed")
    same(result["preflight"]["thresholds"], saved["thresholds"], "threshold splice")
    same(result["preflight"]["final_reference"], saved["final_reference_authority"],
         "independent original-input reference changed")
    same(assets, saved["assets"], "row weight/tokenizer identity changed")
    same(previous.runtime_pins(), saved["runtime_pins"], "reviewed runtime changed")
    base.check_history(result)
    return {"saved": saved, "records": records, "arrays": arrays,
            "references": references, "files": files, "assets": assets}


def run_vectors(evidence, audit):
    rows = bridge.contributions.load_rows(evidence["assets"], set(IDS))
    vectors = {c: evidence["arrays"][c]["rmsnorm"].view("<f2") for c in parent.CONTROLS}
    vectors["reference"] = evidence["references"]["rmsnorm_fp16"].view("<f2")
    operands, dots = {}, {}
    for label, vector in vectors.items():
        require(vector.dtype.str == "<f2", "vector substitution changed precision")
        for index in IDS:
            operands[label, index] = vector, rows[index]
            dots[label, index] = bridge.selected_row_dot(vector, rows[index], audit)
    accounts = []
    for row in evidence["records"]:
        if row["branch"] != "fp16":
            continue
        control, left, right, _ = key(row)
        a = evidence["arrays"][control]["logits"].view("<f2")
        r = evidence["references"]["logits_fp16"].view("<f2")
        same(str(Fraction(float(a[left]))-Fraction(float(a[right]))),
             row["retained_actual_margin"], "actual retained logit anchor splice")
        same(str(Fraction(float(r[left]))-Fraction(float(r[right]))),
             row["retained_reference_margin"], "independent logit anchor splice")
        accounts.append(classify_pair(
            row, tuple(dots[control, i] for i in (left, right)),
            tuple(dots["reference", i] for i in (left, right))))
    same(audit["local_exact_row_dot_computations"], 30, "selected dot budget changed")
    same(audit["selected_row_scalar_products"], 26880, "selected scalar budget changed")
    evidence.update(operands=operands, dots=dots, accounts=accounts)
    return accounts


def run_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("vector_cutoff_tests", TEST)
    require(spec is not None and spec.loader is not None, "test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and not result.skipped, "focused tests failed/errored/skipped")
    return {"compiled": compiled, "executed": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped)}


def execute_check():
    identity()
    origins = [*parent.source_context().values(), parent.record(SOURCE), parent.record(TEST)]
    audit = {"forbidden_calls": 0, "local_exact_row_dot_computations": 0,
             "selected_row_scalar_products": 0}
    with selected_only(audit):
        start = time.monotonic()
        evidence = authenticate()
        authenticated_at = time.monotonic()
        runtime = previous.runtime_pins()
        preregistration = {**preregister(evidence["records"]), "runtime_pins": runtime}
        print("PREREGISTRATION="+json.dumps(preregistration, sort_keys=True),
              file=sys.stderr, flush=True)
        require(os.path.isfile("/proc/self/fd/2"), "persist raw preregistration before arithmetic")
        os.fsync(2)
        accounts = run_vectors(evidence, audit)
        arithmetic_at = time.monotonic()
        tests = run_tests(evidence)
        tested_at = time.monotonic()
        pins = [*PINS, *origins, *evidence["saved"]["authenticated_pins"], *evidence["files"]]
        # The parent already preserved the original temporary capture byte-for-byte.
        pins = [p if p["path"] != previous.CAPTURE_PIN["path"] else {
            **p, "path": str(PARENT_OUTPUT / "parent.capture")} for p in pins]
        for pin in pins:
            base.read_bound(pin)
        same(previous.runtime_pins(), runtime, "runtime changed during diagnostic")
        same(audit["forbidden_calls"], 0, "forbidden dispatch attempted")
    selected = decision(accounts)
    return {
        "diagnostic_id": NAME, "version": 1, "task": TASK, "command": COMMAND,
        "execution_valid": True, "decision": selected, "successor": SUCCESSORS[selected],
        "preregistration": preregistration, "accounts": accounts,
        "direction_agreements": sum(r["direction_agrees"] for r in accounts),
        "crossing_pattern_agreements": sum(r["crossing_pattern_agrees"] for r in accounts),
        "parent_records": evidence["records"],
        "binary64_scope": "54 retained parent identities checked; no binary64 dot or rounding computation",
        "authenticated_pins": pins, "runtime_pins": runtime, "tests": tests,
        "assets": evidence["assets"],
        "thresholds": evidence["saved"]["thresholds"],
        "final_reference_authority": evidence["saved"]["final_reference_authority"],
        "retained_controls_and_failure_gates": evidence["saved"]["retained_controls_and_failure_gates"],
        "original_execution_sources": evidence["saved"]["original_execution_sources"],
        "dispatch_and_write_audit": {
            **bridge.FLAGS, **audit, "artifact_overwrites": 0, "scalar_rne_computations": 0,
            "reference_trajectory_changes": 0, "full_vocabulary_ranking_calls": 0,
        },
        "phase_seconds": {"authentication": authenticated_at-start,
                          "preregistration_and_vector_arithmetic": arithmetic_at-authenticated_at,
                          "compile_and_tests": tested_at-arithmetic_at},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
    }


def validate_result(result):
    same(result["diagnostic_id"], NAME, "stored diagnostic changed")
    require(result["execution_valid"] and result["task"] == TASK
            and result["command"] == COMMAND and result["normal_host_review"] == "REQUIRED",
            "stored identity/status/review changed")
    census(result["parent_records"])
    same(result["preregistration"],
         {**preregister(result["parent_records"]), "runtime_pins": result["runtime_pins"]},
         "stored preregistration changed")
    records = {key(r): r for r in result["parent_records"]}
    for account in result["accounts"]:
        row = records[key(account)]
        # Validate retained scalar identities only; never load vectors or recompute dots.
        expected = classify_pair(row, tuple(map(Fraction, account["actual_exact_dots"])),
                                 tuple(map(Fraction, account["swapped_exact_dots"])))
        same(account, expected, "stored vector account changed")
    selected = decision(result["accounts"])
    same(result["decision"], selected, "stored decision changed")
    same(result["successor"], SUCCESSORS[selected], "stored successor changed")
    for field, agreement in (("direction_agreements", "direction_agrees"),
                             ("crossing_pattern_agreements", "crossing_pattern_agrees")):
        same(result[field], sum(r[agreement] for r in result["accounts"]), "stored count changed")
    tests, audit = result["tests"], result["dispatch_and_write_audit"]
    require(tests["executed"] == EXPECTED_TESTS and len(tests["compiled"]) == 2
            and all(tests[k] == 0 for k in ("failures", "errors", "skipped")),
            "stored compile/test gate failed")
    for name, value in {"forbidden_calls": 0, "artifact_overwrites": 0,
                        "scalar_rne_computations": 0, "reference_trajectory_changes": 0,
                        "full_vocabulary_ranking_calls": 0,
                        "local_exact_row_dot_computations": 30,
                        "selected_row_scalar_products": 26880}.items():
        same(audit[name], value, "stored dispatch budget changed")


def validate_stored(output):
    identity()
    output = output.resolve(strict=True)
    require(output.parent == ROOT / "build" and output.name.startswith(PREFIX),
            "stored evidence outside allowed output prefix")
    metadata = json.loads((output / "capture.json").read_bytes())
    required_files = {"command.json", "source.py", "test.py", "raw.capture", "result.json"}
    require(set(metadata["files"]) == required_files, "capture manifest incomplete")
    for name, pin in metadata["files"].items():
        path = output / name
        require(pin["path"] == str(path) and not path.is_symlink(), "capture artifact escaped output")
        previous.pin_bytes(path.read_bytes(), pin)
    raw = (output / "raw.capture").read_bytes()
    result_bytes = (output / "result.json").read_bytes()
    result = json.loads(result_bytes)
    command = json.loads((output / "command.json").read_bytes())
    same(metadata["whole_capture"], metadata["files"]["raw.capture"], "whole/stream hash conflation")
    require(metadata["returncode"] == 0 and metadata["task"] == command["task"] == TASK
            and type(command["attempt"]) is int and command["attempt"] >= 1
            and metadata["attempt"] == command["attempt"]
            and command["command"] == metadata["command"] == COMMAND
            and command["argv"] == ARGV and command["cwd"] == str(ROOT)
            and command["uid"] == 1000 and command["branch"] == "argus/full-projection"
            and command["environment"] == ENVIRONMENT
            and metadata["original_source_path"] == command["original_source_path"]
            == str(output / "raw.capture")
            and command["classifier_source_path"] == str(SOURCE)
            and metadata["separate_stream_hashes"] is None,
            "captured command/task/attempt/source/streams mismatch")
    previous.validate_runtime_pins(result, command, metadata)
    prereg = b"PREREGISTRATION="+json.dumps(result["preregistration"], sort_keys=True).encode()+b"\n"
    require(raw.startswith(prereg) and raw.endswith(result_bytes)
            and raw.splitlines(keepends=True)[-1] == result_bytes,
            "preregistration/result not byte-exact raw command capture")
    validate_result(result)
    for pin in result["authenticated_pins"]:
        base.read_bound(pin)
    source_test = [parent.record(SOURCE), parent.record(TEST)]
    same(result["tests"]["compiled"], source_test, "compiled source/test pins changed")
    same(command["source_test_pins"], source_test, "command source/test pins changed")
    for name, pin in zip(("source.py", "test.py"), source_test, strict=True):
        previous.pin_bytes((output / name).read_bytes(), pin)
    saved = json.loads(base.read_bound(PINS[3]))
    same(result["parent_records"], [r["parent_account"] for r in saved["parent_closures"]],
         "stored original parent records changed")
    for field in ("thresholds", "assets", "final_reference_authority",
                  "retained_controls_and_failure_gates", "original_execution_sources"):
        same(result[field], saved[field], "stored parent authority changed: "+field)
    return {"status": "VALID_STORED_EVIDENCE", "decision": result["decision"],
            "scientific_recomputation": 0, "normal_host_review": "REQUIRED",
            "result": parent.record(output / "result.json"),
            "whole_capture": parent.record(output / "raw.capture")}


def write_new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(document):
    return (json.dumps(document, sort_keys=True, allow_nan=False)+"\n").encode()


def capture_run(attempt):
    identity()
    require(type(attempt) is int and attempt >= 1, "attempt must be a positive integer")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    same(branch, "argus/full-projection", "isolated branch changed")
    output = ROOT / "build" / f"{PREFIX}{TASK}_attempt{attempt:03d}"
    ignored = subprocess.run(["git", "check-ignore", "-q", str(output)], cwd=ROOT, check=False)
    same(ignored.returncode, 0, "evidence output is not ignored")
    output.mkdir(exist_ok=False)
    runtime = previous.runtime_pins()
    command = {
        "task": TASK, "attempt": attempt, "command": COMMAND, "argv": ARGV,
        "cwd": str(ROOT), "uid": os.getuid(), "branch": branch, "environment": ENVIRONMENT,
        "classifier_source_path": str(SOURCE), "original_source_path": str(output / "raw.capture"),
        "runtime_pins": runtime, "source_test_pins": [parent.record(SOURCE), parent.record(TEST)],
        "streams": "stdout/stderr merged at child descriptors; separate stream hashes unavailable",
    }
    write_new(output / "command.json", json_bytes(command))
    for name, source in (("source.py", SOURCE), ("test.py", TEST)):
        write_new(output / name, source.read_bytes())
    with (output / "raw.capture").open("xb") as raw:
        completed = subprocess.run(ARGV, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                                   stdout=raw, stderr=subprocess.STDOUT, check=False)
        raw.flush()
        os.fsync(raw.fileno())
    payload = (output / "raw.capture").read_bytes()
    lines = payload.splitlines(keepends=True)
    require(bool(lines), "empty scientific capture retained at "+str(output))
    result_bytes = lines[-1]
    result = json.loads(result_bytes)
    write_new(output / "result.json", result_bytes)
    files = {name: parent.record(output / name) for name in
             ("command.json", "source.py", "test.py", "raw.capture", "result.json")}
    metadata = {
        "task": TASK, "attempt": attempt, "command": COMMAND,
        "original_source_path": str(output / "raw.capture"), "runtime_pins": runtime,
        "returncode": completed.returncode, "whole_capture": files["raw.capture"],
        "separate_stream_hashes": None, "files": files,
    }
    write_new(output / "capture.json", json_bytes(metadata))
    if completed.returncode:
        print(json.dumps({"output": str(output), "decision": "UNKNOWN",
                          "error": result.get("error"), "returncode": completed.returncode}))
        return completed.returncode
    validation_argv = [parent.PYTHON, "-B", "-m", MODULE, "--validate", str(output)]
    write_new(output / "validation.command.json", json_bytes({
        **command, "argv": validation_argv,
        "command": COMMAND.rsplit("--check", 1)[0]+shlex.join(validation_argv[4:]),
        "original_source_path": str(output / "validation.raw.capture"),
    }))
    with (output / "validation.raw.capture").open("xb") as raw:
        validation = subprocess.run(validation_argv, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                                    stdout=raw, stderr=subprocess.STDOUT, check=False)
        raw.flush()
        os.fsync(raw.fileno())
    write_new(output / "validation.metadata.json", json_bytes({
        "task": TASK, "attempt": attempt, "returncode": validation.returncode,
        "scientific_recomputation": 0, "separate_stream_hashes": None,
        "command": parent.record(output / "validation.command.json"),
        "whole_capture": parent.record(output / "validation.raw.capture"),
    }))
    print(json.dumps({
        "output": str(output), "decision": result["decision"], "successor": result["successor"],
        "direction_agreements": result["direction_agreements"],
        "crossing_pattern_agreements": result["crossing_pattern_agreements"],
        "tests": result["tests"], "dispatch": result["dispatch_and_write_audit"],
        "validation_returncode": validation.returncode, "whole_capture": files["raw.capture"],
        "normal_host_review": "REQUIRED",
    }, sort_keys=True))
    return validation.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--validate", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args(argv)
    if args.run:
        raise SystemExit(capture_run(args.attempt))
    if args.validate:
        audit = {"forbidden_calls": 0}
        with selected_only(audit):
            result = validate_stored(args.validate)
        same(audit["forbidden_calls"], 0, "validation attempted forbidden dispatch")
        print(json.dumps(result, sort_keys=True))
        return
    try:
        result = execute_check()
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr, flush=True)
        print(json.dumps({"diagnostic_id": NAME, "execution_valid": False, "decision": "UNKNOWN",
                          "successor": SUCCESSORS["UNKNOWN"], "error": str(error),
                          "runtime_pins": previous.runtime_pins(),
                          "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY}), flush=True)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]
    main()
