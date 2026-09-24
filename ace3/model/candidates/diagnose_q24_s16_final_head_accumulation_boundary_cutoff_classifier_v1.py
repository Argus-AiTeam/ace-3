"""Selected-row, non-admission head-boundary counterfactual; no producer replay."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_accumulation_boundary_residual_margin_bridge_v1 as bridge


base, parent = bridge.base, bridge.parent
ROOT = bridge.ROOT
NAME = "diagnose_q24_s16_final_head_accumulation_boundary_cutoff_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
TASK = "b288dca598b7"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
CAPTURE_PIN = {
    "path": "/tmp/1789246357000-copilot-tool-output-2412893-99c7784c-1a7c-42c8-9c66-da46e9226c82.txt",
    "bytes": 23733,
    "sha256": "a78a7edc9da0327952af2ab19de3f3ef2cf2cd5adeeb958febcf7c4e390a0937",
}
REVIEW_PIN = {
    "path": "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/f4cf7bfb2c22/round-0001.json",
    "bytes": 690,
    "sha256": "67bd261f15f81c19706beb7918e891cff1ffad8f3aa8918ad17f0a8a0bd7bd76",
}
PARENT_PINS = (
    {"path": str(bridge.SOURCE), "bytes": 17318,
     "sha256": "028b254a002ae3d9db206a92a51aa4fd0b8b525f139edb7fd5795912ce4c8ed3"},
    {"path": str(bridge.TEST), "bytes": 17056,
     "sha256": "5c98829427a039d56bea9dd33878388ab65fbeabd36ea2bbbb7a7f02a2bf0f3f"},
    {"path": str(bridge.mixed.SOURCE), "bytes": 20433,
     "sha256": "aecb39a7d509e8e89fe99d3cbcfcf1b67384383e759c4924a1a0f01f6643e138"},
    {"path": str(bridge.mixed.TEST), "bytes": 12117,
     "sha256": "c9a213db2a071c2fedf73eec9e041c8112e8862ae1fb4ba46949a901e8f0c1cd"},
)
PAIRS = ((34319, 319), (34319, 13), (319, 34319))
IDS = (13, 319, 34319)
ROLES = {
    (34319, 319): "fixed exchanged pair versus independent FP16",
    (34319, 13): "fixed actual/binary64 cutoff neighbors",
    (319, 34319): "fixed independent-FP16 cutoff neighbors",
}
COLUMNS = (
    "control", "left_id", "right_id", "branch", "retained_actual_margin",
    "retained_reference_margin", "retained_margin_change",
    "actual_exact_row_dot_margin", "reference_exact_row_dot_margin",
    "vector_coordinate_sum", "actual_head_boundary",
    "negative_reference_head_boundary", "head_boundary_remainder_change",
)
SUCCESSORS = {
    "SUPPORTED": "narrower selected-head accumulation/order/RNE-cell successor",
    "REJECTED": "final-RMSNorm-vector rather than head-boundary successor",
    "UNKNOWN": "integrity/availability repair only; no scientific successor",
}
BOUNDARY = (
    "CPU-only diagnostic shadow heads on three already diagnosed numeric rows. "
    "Both retained actual and independent FP16 RMSNorm inputs receive exact dots "
    "and one FP16 RNE; the independent reference trajectory/logits are never "
    "replaced. Binary64 is a frozen closure comparator, not rounded into a new "
    "profile. This reference-guided contrast is not an algorithm repair or a "
    "root-cause attribution. No prefix/admission/native decoder/reference producer/"
    "RMSNorm/full-vocabulary replay, token ranking/publication/selection, row319 "
    "operand-provenance, middle/outer/dose intervention, GPU/RTL/FPGA/ACE2 work, "
    "or precision/scale expansion. Q24 residual state remains wider than FP16; "
    "native-S16-RTZ, INT4 weights, FP16 operator boundaries/KV and all retained "
    "thresholds, histories and source/operand/state/KV/lineage gates are unchanged. "
    "No strict-FP16-state W4A16, new-token or full-model admission."
)
require, same = base.require, base.same
EXPECTED_TESTS = 14


@contextmanager
def selected_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("selected classifier forbids parent replay or ranking")

    with bridge.read_only(audit), ExitStack() as stack:
        for module, names in (
            (bridge, ("measure", "report", "check", "focused_tests")),
            (bridge.margin.cutoff, ("profile", "report", "cutoff_control")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def pin_bytes(payload, pin):
    require(len(payload) == pin["bytes"]
            and hashlib.sha256(payload).hexdigest() == pin["sha256"],
            "byte-bound evidence mismatch: " + pin["path"])
    return payload


def runtime_pins():
    return {
        "python_version": sys.version,
        "python": parent.record(Path(sys.executable)),
        "numpy_version": bridge.mixed.np.__version__,
        "torch_version": str(bridge.mixed.torch.__version__),
    }


def validate_runtime_pins(result, command, metadata):
    expected = runtime_pins()
    for label, document in (
        ("result", result), ("preregistration", result.get("preregistration", {})),
        ("command", command), ("capture", metadata),
    ):
        same(document.get("runtime_pins"), expected,
             label + " runtime/dependency pins missing or changed")
    return expected


def decode_capture(payload):
    pin_bytes(payload, CAPTURE_PIN)
    marker = b'{"branch_totals"'
    require(payload.count(marker) == 1, "ambiguous retained receipt")
    receipt = json.loads(payload[payload.index(marker):])
    same(receipt["retained_value_columns"], list(COLUMNS), "parent columns changed")
    same(receipt["tests"]["compiled"], list(PARENT_PINS[:2]), "reviewed bridge sources changed")
    same(receipt["control_count"], 9, "parent control count changed")
    same(receipt["pair_branch_count"], 54, "parent pair/branch count changed")
    require(receipt["dispatch_and_write_audit"]["forbidden_calls"] == 0
            and receipt["exact_closure_residuals_zero"] == 270
            and receipt["tests"]["executed"] == 18
            and all(receipt["tests"][k] == 0 for k in ("errors", "failures", "skipped")),
            "parent execution was not a closed valid result")
    records = [dict(zip(COLUMNS, row, strict=True)) for row in receipt["retained_values"]]
    expected = {(c, a, b, branch) for c in parent.CONTROLS
                for a, b in PAIRS for branch in ("fp16", "binary64")}
    keys = [(r["control"], r["left_id"], r["right_id"], r["branch"]) for r in records]
    require(len(keys) == len(set(keys)) == 54 and set(keys) == expected,
            "parent account missing, duplicated or relabeled")
    for row in records:
        account_closure(row)
    return receipt, records


def account_closure(row):
    a, r, delta, ea, er, vector, ba, nr, h = (Fraction(row[k]) for k in COLUMNS[4:])
    require(a-r == delta and ea-er == vector and a-ea == ba
            and er-r == nr and ba+nr == h and vector+h == delta,
            "parent head-boundary account does not close")


def sign(value):
    return (value > 0) - (value < 0)


def prediction(row):
    account_closure(row)
    a, r, delta, ea, er, vector, ba, nr, h = (Fraction(row[k]) for k in COLUMNS[4:])
    return {
        "control": row["control"], "left_id": row["left_id"], "right_id": row["right_id"],
        "role": ROLES[row["left_id"], row["right_id"]],
        "predicted_relative_margin_movement": str(-h),
        "predicted_movement_sign": sign(-h),
        "retained_sign_pattern": [sign(a), sign(r), sign(delta)],
        "predicted_sign_pattern": [sign(ea), sign(er), sign(vector)],
        "predicted_actual_margin": str(ea), "predicted_reference_shadow_margin": str(er),
        "predicted_relative_margin": str(vector),
    }


def preregister(records):
    rows = [prediction(r) for r in records if r["branch"] == "fp16"]
    require(len(rows) == 27, "preregistered FP16 account census changed")
    return {
        "version": 1, "task": TASK, "primary_branch": "fp16",
        "frozen_selected_row_ids": list(IDS), "pair_count": 27,
        "prediction_source": CAPTURE_PIN,
        "prediction": (
            "Removing parent H predicts relative movement -H and endpoint signs "
            "(EA, ER, EA-ER). Exact dot followed by FP16 RNE must preserve that "
            "movement sign and all three endpoint signs for every fixed pair/control. "
            "A zero is a distinct outcome, not directional agreement. Original "
            "margins remain anchors. No rankings or new cutoff IDs are computed."
        ),
        "decision_rule": "SUPPORTED iff all 27 direction and crossing-pattern tests agree; otherwise REJECTED",
        "unknown_rule": "Missing/mismatched evidence, closure, or runtime invalidates execution: UNKNOWN",
        "successors": SUCCESSORS, "rows": rows,
        "ties": "Report exact zero separately; never use ties to publish/select a token",
        "not_independent_samples": True,
    }


def half_value(word):
    require(type(word) is int and 0 <= word < 65536, "invalid half word")
    exponent, mantissa = (word >> 10) & 31, word & 1023
    require(exponent != 31, "nonfinite half word")
    value = Fraction(mantissa if exponent == 0 else 1024+mantissa)
    value *= Fraction(2) ** (-24 if exponent == 0 else exponent-25)
    return -value if word & 32768 else value


def round_half(value):
    require(isinstance(value, Fraction) and abs(value) < 65520,
            "selected-row FP16 overflow or non-exact operand")
    negative, value = value < 0, abs(value)
    low, high = 0, 0x7bff
    while low < high:
        mid = (low+high+1)//2
        if half_value(mid) <= value:
            low = mid
        else:
            high = mid-1
    chosen = low
    if low < 0x7bff:
        dl, du = value-half_value(low), half_value(low+1)-value
        if du < dl or (du == dl and low & 1):
            chosen += 1
    return chosen | (32768 if negative else 0)


def classify_pair(row, actual_dots, reference_dots, actual_words, reference_words):
    account_closure(row)
    a, r, delta, ea, er, vector, ba, nr, h = (Fraction(row[k]) for k in COLUMNS[4:])
    require(actual_dots[0]-actual_dots[1] == ea
            and reference_dots[0]-reference_dots[1] == er,
            "new selected dots do not close retained parent accounts")
    ac = half_value(actual_words[0])-half_value(actual_words[1])
    rc = half_value(reference_words[0])-half_value(reference_words[1])
    observed = ac-rc
    movement = observed-delta
    rne_remainder = (ac-ea)-(rc-er)
    require(movement == -h+rne_remainder, "head replacement/RNE closure failed")
    p = prediction(row)
    signs = [sign(ac), sign(rc), sign(observed)]
    return {
        **p, "branch": "fp16", "parent_account": row,
        "actual_exact_dots": list(map(str, actual_dots)),
        "reference_shadow_exact_dots": list(map(str, reference_dots)),
        "actual_rne_words": list(actual_words), "reference_shadow_rne_words": list(reference_words),
        "actual_counterfactual_margin": str(ac), "reference_shadow_counterfactual_margin": str(rc),
        "relative_counterfactual_margin": str(observed),
        "actual_margin_movement": str(ac-a), "reference_shadow_margin_movement": str(rc-r),
        "relative_margin_movement": str(movement), "observed_sign_pattern": signs,
        "direction_agrees": sign(movement) == p["predicted_movement_sign"],
        "crossing_pattern_agrees": signs == p["predicted_sign_pattern"],
        "rne_boundary_remainder": str(rne_remainder),
        "exact_closure_residuals": {
            "actual_dot_margin_minus_parent": str(actual_dots[0]-actual_dots[1]-ea),
            "reference_dot_margin_minus_parent": str(reference_dots[0]-reference_dots[1]-er),
            "margin_movement_plus_parent_H_minus_rne": str(movement+h-rne_remainder),
        },
    }


def decision(rows):
    require(len(rows) == 27, "incomplete classifier matrix")
    return ("SUPPORTED" if all(r["direction_agrees"] and r["crossing_pattern_agrees"]
                               for r in rows) else "REJECTED")


def authenticate():
    for pin in (REVIEW_PIN, *PARENT_PINS, *bridge.margin.rows.PINS.values()):
        base.read_bound(pin)
    review = json.loads(base.read_bound(REVIEW_PIN))
    require(review["kind"] == "round_reviewed_handoff"
            and review["mission_id"] == "f4cf7bfb2c22"
            and review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done",
            "independent bridge review is not terminal")
    receipt, records = decode_capture(base.read_bound(CAPTURE_PIN))
    result, arrays, references, files, assets = bridge.contributions.authenticate()
    manifest = json.loads(base.read_bound(result["preflight"]["final_reference"]["manifest"]))
    same(manifest["arithmetic"]["runtime"],
         {"torch": bridge.mixed.torch.__version__, "numpy": bridge.mixed.np.__version__},
         "reviewed mixed-head runtime changed")
    for c in result["controls"]:
        top = c["comparisons"]["top_k"]
        require(top["actual_diagnostic_only"][-1]["token_id"] == 34319
                and top["fp16"]["reference"][-1]["token_id"] == 319
                and top["binary64"]["reference"][-1]["token_id"] == 34319,
                "retained cutoff IDs changed")
    return {"result": result, "arrays": arrays, "references": references,
            "files": files, "assets": assets, "records": records, "receipt": receipt}


def run_selected(evidence, audit):
    rows = bridge.contributions.load_rows(evidence["assets"], set(IDS))
    dots, rounded, operands = {}, {}, {}
    trajectories = [(("actual", c), a["rmsnorm"].view("<f2"))
                    for c, a in evidence["arrays"].items()]
    trajectories += [(("reference", b), evidence["references"]["rmsnorm_"+b].view("<f2")
                      if b == "fp16" else evidence["references"]["rmsnorm_"+b])
                     for b in ("fp16", "binary64")]
    for trajectory, vector in trajectories:
        for index in IDS:
            key = (*trajectory, index)
            operands[key] = (vector, rows[index])
            dots[key] = bridge.selected_row_dot(vector, rows[index], audit)
            if trajectory != ("reference", "binary64"):
                rounded[key] = round_half(dots[key])
                audit["selected_scalar_rne_computations"] += 1
            if trajectory[0] == "actual":
                same(rounded[key], int(evidence["arrays"][trajectory[1]]["logits"][index]),
                     "exact/RNE actual row does not reproduce retained FP16 word")
    accounts, closures = [], []
    for row in evidence["records"]:
        c, left, right, b = row["control"], row["left_id"], row["right_id"], row["branch"]
        ad = tuple(dots["actual", c, i] for i in (left, right))
        rd = tuple(dots["reference", b, i] for i in (left, right))
        al = evidence["arrays"][c]["logits"].view("<f2")
        rl = evidence["references"]["logits_"+b]
        if b == "fp16":
            rl = rl.view("<f2")
        am = Fraction(float(al[left]))-Fraction(float(al[right]))
        rm = Fraction(float(rl[left]))-Fraction(float(rl[right]))
        same(str(am), row["retained_actual_margin"], "retained actual margin splice")
        same(str(rm), row["retained_reference_margin"], "retained reference margin splice")
        ea, er = ad[0]-ad[1], rd[0]-rd[1]
        residuals = [ea-Fraction(row["actual_exact_row_dot_margin"]),
                     er-Fraction(row["reference_exact_row_dot_margin"]),
                     am-ea+er-rm-Fraction(row["head_boundary_remainder_change"])]
        require(not any(residuals), "exact selected-row parent closure failed")
        closures.append({"control": c, "left_id": left, "right_id": right, "branch": b,
                         "residuals": list(map(str, residuals)), "parent_account": row})
        if b == "fp16":
            accounts.append(classify_pair(
                row, ad, rd, tuple(rounded["actual", c, i] for i in (left, right)),
                tuple(rounded["reference", b, i] for i in (left, right))))
    same(audit["local_exact_row_dot_computations"], 33, "selected dot budget changed")
    same(audit["selected_row_scalar_products"], 29568, "selected product budget changed")
    same(audit["selected_scalar_rne_computations"], 30, "selected RNE budget changed")
    evidence.update(operands=operands, dots=dots, rounded=rounded, accounts=accounts)
    return accounts, closures


def run_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("head_boundary_cutoff_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
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
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "classifier_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0, "local_exact_row_dot_computations": 0,
             "selected_row_scalar_products": 0, "selected_scalar_rne_computations": 0}
    with selected_only(audit):
        runtime = runtime_pins()
        evidence = authenticate()
        evidence["runtime_pins"] = runtime
        preregistration = preregister(evidence["records"])
        preregistration["runtime_pins"] = runtime
        print("PREREGISTRATION="+json.dumps(preregistration, sort_keys=True),
              file=sys.stderr, flush=True)
        require(os.path.isfile("/proc/self/fd/2"), "raw capture must precede selected arithmetic")
        os.fsync(2)
        accounts, closures = run_selected(evidence, audit)
        tests = run_tests(evidence)
        pins = [*origins.values(), *PARENT_PINS, REVIEW_PIN, CAPTURE_PIN,
                *bridge.margin.rows.PINS.values(), *evidence["files"]]
        for pin in pins:
            base.read_bound(pin)
        same(runtime_pins(), runtime, "runtime/dependency pins changed during execution")
        same(audit["forbidden_calls"], 0, "forbidden scientific dispatch attempted")
    selected = decision(accounts)
    return {
        "diagnostic_id": NAME, "version": 1, "task": TASK, "command": COMMAND,
        "execution_valid": True, "decision": selected, "successor": SUCCESSORS[selected],
        "preregistration": preregistration, "accounts": accounts, "parent_closures": closures,
        "direction_agreements": sum(r["direction_agrees"] for r in accounts),
        "crossing_pattern_agreements": sum(r["crossing_pattern_agrees"] for r in accounts),
        "authenticated_pins": pins, "assets": evidence["assets"], "runtime_pins": runtime,
        "original_execution_sources": parent.RETAINED_SOURCES,
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "thresholds": evidence["result"]["preflight"]["thresholds"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "parent_capture_scope": (
            "Original complete command capture of the parent wrapper's 54-account receipt; "
            "not the parent's unretained 3514472-byte full stdout JSON. No reconstruction."
        ),
        "dispatch_and_write_audit": {**bridge.FLAGS, **audit, "artifact_overwrites": 0,
                                     "reference_trajectory_changes": 0,
                                     "full_vocabulary_ranking_calls": 0},
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
    }


def validate_stored(output):
    output = output.resolve(strict=True)
    require(output.parent == ROOT / "build", "stored evidence must be a direct build child")
    metadata = json.loads((output / "capture.json").read_bytes())
    for pin in metadata["files"].values():
        path = Path(pin["path"])
        require(path.parent == output and not path.is_symlink(), "capture artifact escaped output")
        pin_bytes(path.read_bytes(), pin)
    raw = (output / "raw.capture").read_bytes()
    result_bytes = (output / "result.json").read_bytes()
    require(raw.endswith(result_bytes), "result is not the byte-exact captured stdout document")
    result = json.loads(result_bytes)
    command = json.loads((output / "command.json").read_bytes())
    require(command["task"] == TASK and type(command["attempt"]) is int
            and command["attempt"] >= 1 and metadata["attempt"] == command["attempt"]
            and command["command"] == COMMAND and command["cwd"] == str(ROOT)
            and command["uid"] == 1000 and metadata["returncode"] == 0,
            "captured execution identity/status mismatch")
    require(result["execution_valid"] and result["normal_host_review"] == "REQUIRED",
            "result is invalid or bypasses review")
    runtime = validate_runtime_pins(result, command, metadata)
    prereg = b"PREREGISTRATION="+json.dumps(result["preregistration"], sort_keys=True).encode()+b"\n"
    require(raw.startswith(prereg), "preregistration not captured before execution")
    require(len(result["accounts"]) == 27 and len(result["parent_closures"]) == 54,
            "stored account census incomplete")
    require(all(set(r["residuals"]) == {"0"} for r in result["parent_closures"]),
            "stored parent closure failed")
    require(all(set(r["exact_closure_residuals"].values()) == {"0"} for r in result["accounts"]),
            "stored counterfactual closure failed")
    same(result["decision"], decision(result["accounts"]), "stored decision disagrees with tests")
    same(result["successor"], SUCCESSORS[result["decision"]], "stored successor changed")
    require(result["tests"]["executed"] == EXPECTED_TESTS
            and len(result["tests"]["compiled"]) == 2
            and all(result["tests"][k] == 0 for k in ("failures", "errors", "skipped"))
            and result["dispatch_and_write_audit"]["forbidden_calls"] == 0
            and result["dispatch_and_write_audit"]["artifact_overwrites"] == 0,
            "stored compile/test/dispatch gates failed")
    for pin in result["authenticated_pins"]:
        if pin["path"] == CAPTURE_PIN["path"]:
            pin_bytes((output / "parent.capture").read_bytes(), CAPTURE_PIN)
        else:
            base.read_bound(pin)
    return {"status": "VALID_STORED_EVIDENCE", "decision": result["decision"],
            "scientific_recomputation": 0, "normal_host_review": "REQUIRED",
            "runtime_pins": runtime,
            "result": parent.record(output / "result.json"),
            "whole_capture": parent.record(output / "raw.capture")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        print(json.dumps(validate_stored(args.validate), sort_keys=True))
        return
    try:
        result = execute_check()
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr, flush=True)
        print(json.dumps({"diagnostic_id": NAME, "execution_valid": False, "decision": "UNKNOWN",
                          "successor": SUCCESSORS["UNKNOWN"], "error": str(error),
                          "runtime_pins": runtime_pins(),
                          "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY},
                         sort_keys=True), flush=True)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    sys.modules[MODULE] = sys.modules[__name__]
    main()
