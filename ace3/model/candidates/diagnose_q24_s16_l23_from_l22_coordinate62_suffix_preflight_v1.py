"""Read-only preparation of nine raw failed-parent L23/P0 consumers."""

import argparse
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1 as parent


census = parent.census
require = census.require
read_bound = census.read_bound
record = parent.matrix.record
ROOT = parent.ROOT
NAME = "q24_s16_l23_from_l22_coordinate62_suffix_preflight_v1"
ID = "ace3-" + NAME.replace("_", "-")
SOURCE = ROOT / f"ace3/model/candidates/diagnose_{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
BASE = ROOT / "build/q24_s16_l22_from_l21_coordinate62_suffix_execution_v1_attempt001"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
HANDOFFS = parent.matrix.HANDOFFS
MISSION = "bc782e353b8c"
BACKLOG = HANDOFFS.parent / "backlog.jsonl"
REFERENCE_MANIFEST = parent.REFERENCE_MANIFEST
CONTROLS = census.CONTROLS
EXPECTED_TESTS = 30
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m "
           f"ace3.model.candidates.diagnose_{NAME} --check")
PINS = {
    "review": {
        "path": str(HANDOFFS / MISSION / "round-0001.json"),
        "sha256": "81e34683ce427070284a7991dd2e50c4a04f176ac6cba08f5759f3b4abd01ddb",
    },
    "mission": {
        "path": str(HANDOFFS / MISSION / "mission.json"),
        "sha256": "b85ff50f05431e49c7fbd6c2162b37381919a38f4ab98535e119a1b6f2dcb040",
    },
    "result": {
        "path": str(BASE / "result.json"),
        "sha256": "fde641e5006e2d89e8925dbca140adbd55be0a023e9067ee2fb69c882e3d4d9f",
    },
    "validation": {
        "path": str(BASE / "validation.json"),
        "sha256": "6877676c7f7f14b1158673dfc40f11dcdecef1727b6b066973805c2557deb77a",
    },
}
FLAGS = {
    "native_layer_invocations": 0, "native_L0_L22_invocations": 0,
    "native_L23_P0_invocations": 0, "dispatch_guard_calls": 0,
    "rtl_invocations": 0, "hardware_invocations": 0, "gpu_invocations": 0,
    "simulation_invocations": 0, "external_service_invocations": 0,
    "evidence_writes": 0, "execution_authorized": False,
    "accepted_prefix_replay": False, "original_prefix_replay": False,
    "admission_replay": False, "reference_recomputation": False,
    "reference_reanchoring": False, "candidate_admitted": False,
    "policy_adopted": False, "successor_published": False,
    "strict_FP16_state_claim": False, "new_token_claim": False, "full_model_claim": False,
}
STATE = {**parent.STATE,
         "transformation": "none; entire raw L22 output I/Z/H, not a coordinate cut"}
KV = parent.KV
FUTURE_OUTPUT = {
    **parent.FUTURE_OUTPUT, "default": str(OUTPUT),
    "basename_pattern": NAME + r"_attempt[0-9]{3,}",
    "per_control_artifacts": [
        "{control}_L23.npz", "{control}_L23_parent.npz",
        "{control}_L23_local_references.npz", "{control}_L23_gates.json",
    ],
}
BOUNDARY = (
    "Preparation only, not L23 numerical evidence. Exactly nine complete raw failed L22 "
    "states are isolated counterfactual inputs, not admission parents or adjusted sparse-cut "
    "replacements. No native execution, L0-L22 or accepted/original-prefix/admission replay, "
    "RTL, hardware, GPU, simulation or evidence writes. Q24 residual state is wider than "
    "FP16; G128 asymmetric packed INT4, native GEMM nibble ordering, no qzero plus-one, "
    "FP16 scales/operator boundaries/KV and native S16 RTZ remain unchanged. Original-input "
    "references, exact thresholds, source/operand/state/KV/lineage gates and all historical "
    "failures remain mandatory. No strict-FP16-state W4A16, new-token or full-model admission. "
    "This preflight requires independent Host Reviewer validation; later execution requires "
    "a separately reviewed executor, exclusive fresh output creation and current "
    "source/identity/role-model/account/budget/access/lock/concurrency preflights."
)
EXPECTED_CONTRACT = {
    "diagnostic_id": ID, "version": 1, "schema_version": 1, "cwd": str(ROOT),
    "evidence": PINS, "controls": list(CONTROLS),
    "parent_node": [22, 0, 18], "consumer_node": [23, 0], "history": [9707],
    "state": STATE, "consumer_kv": KV, "future_output": FUTURE_OUTPUT,
    "reference_manifest": str(REFERENCE_MANIFEST),
    "reference_missing_status": "BLOCKED_MISSING_L23_REFERENCE",
    "policy_id": census.POLICY, "profile_id": census.PROFILE,
    "reference_policy": census.REFERENCE_POLICY, "excess_budget": "1/8",
    "local_threshold": census.EXPECTED_CONTRACT["local_threshold"],
    "interface": ["--check", "--out"], "output": "stdout JSON only; no file writes",
    "flags": FLAGS, "expected_tests": EXPECTED_TESTS, "command_sidecar": COMMAND,
    "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
}


def same(left, right, message):
    require(json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True), message)


def check_contract(document):
    same(document, EXPECTED_CONTRACT, "L23 preflight contract mismatch")


def output_path(value):
    path = Path(value)
    out = path if path.is_absolute() else ROOT / path
    require(out.resolve() == out and out.parent == ROOT / "build"
            and re.fullmatch(FUTURE_OUTPUT["basename_pattern"], out.name) is not None
            and int(out.name.rsplit("attempt", 1)[1]) > 0,
            "output must be a canonical versioned direct child of build")
    require(not out.exists() and not out.is_symlink(), "output occupied or symlinked")
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored: " +
            ignored.stderr.decode(errors="replace"))
    return out


def check_review(review, latest, backlog):
    parent.matrix.check_review(review, MISSION, 1)
    require(latest["kind"] == "handoff_ref"
            and latest["handoff"]["path"] == PINS["review"]["path"],
            "terminal review is no longer the resolved review")
    rows = [row for row in backlog if row["id"] == MISSION]
    require(len(rows) == 1 and rows[0]["status"] == "done"
            and rows[0]["outcome"]["review_status"] == "done"
            and rows[0]["finished_ts"] >= review["created_at"],
            "terminal review lacks backlog corroboration")


def archive(pin):
    with np.load(io.BytesIO(read_bound(pin)), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def check_reference_manifest(extension, previous):
    original22 = parent.check_reference_manifest(
        extension, extension["layers"]["21"])
    same(original22, previous, "reviewed original L22 reference changed")
    current = extension["layers"].get("23")
    if current is None:
        return None
    require(current["input_binary64"] == previous["binary64"]
            and current["input_fp16"] == previous["fp16"]
            and current["prior_kv"] == "own empty P0",
            "L23 reference re-anchored or KV changed")
    for key, suffix in (("binary64", "binary64.npy"), ("fp16", "fp16.npz")):
        require(current[key]["path"] == str(REFERENCE_MANIFEST.parent / f"layer23_{suffix}"),
                "L23 reference archive substituted")
    require(bool(current["canonical"])
            and all(key.startswith("model.layers.23.") for key in current["canonical"]),
            "L23 official operand binding missing")
    return current


def bind_reference(manifest, previous):
    def blocked(missing, current=None):
        return {"status": "BLOCKED_MISSING_L23_REFERENCE", "manifest": manifest,
                "reference": current, "missing": missing}

    require(manifest["path"] == str(REFERENCE_MANIFEST), "reference manifest substituted")
    try:
        extension = json.loads(read_bound(manifest))
    except FileNotFoundError:
        return blocked([str(REFERENCE_MANIFEST)])
    current = check_reference_manifest(extension, previous)
    if current is None:
        return blocked([str(REFERENCE_MANIFEST) + "#layers/23"])
    payloads, missing = {}, []
    for key in ("binary64", "fp16", "input_binary64", "input_fp16"):
        try:
            payloads[key] = read_bound(current[key])
        except FileNotFoundError:
            missing.append(current[key]["path"])
    if missing:
        return blocked(missing, current)
    binary = np.load(io.BytesIO(payloads["binary64"]), allow_pickle=False)
    require(binary.shape == (896,) and binary.dtype.str == "<f8"
            and np.all(np.isfinite(binary)), "invalid L23 binary64 reference")
    sizes = (896, 896, 128, 128, 896, 128, 128, 128, 14, 14,
             896, 896, 896, 896, 4864, 4864, 4864, 896, 896)
    with np.load(io.BytesIO(payloads["fp16"]), allow_pickle=False) as fp16:
        for key, size in [(f"stage{s:02d}", size) for s, size in enumerate(sizes)] + [
                ("input_hidden", 896)]:
            words = fp16[key]
            require(words.shape == (size,) and words.dtype.str == "<u2"
                    and np.all(np.isfinite(words.view("<f2"))), "invalid L23 FP16 reference")
    return {"status": "BOUND_ORIGINAL_INPUT_L23", "manifest": manifest,
            "reference": current, "missing": []}


def check_reports(row, reports, state, reference):
    statuses = ["PASS"] * 18 + ["FAIL"]
    require(row["mandatory_statuses"] == statuses and row["L22_status"] == "FAIL"
            and row["S18_failure_indices"] == [62]
            and row["source_operand_state_KV_RTZ_checks"] == "PASS",
            "raw L22 failure or retained gates changed")
    parent.matrix.check_false_flags(row)
    require(len(reports) == 19, "incomplete L22 stage reports")
    for stage, report in enumerate(reports):
        require(report["stage"] == stage and report["node"] == [22, 0, stage]
                and report["policy_id"] == census.POLICY and report["status"] == statuses[stage]
                and report["kv_lineage"] == report["residual_state_lineage"] == "PASS",
                "L22 source/state/KV/lineage gate changed")
        if stage < 18:
            gate = report["local_operator_fp16"]
            require(gate["role"] == "mandatory" and gate["passed"] is True
                    and gate["failure_count"] == 0 and gate["failures"] == [],
                    "retained same-operand local gate changed")
    gate = reports[18]["binary64_v1"]
    require(gate["profile_id"] == census.PROFILE and gate["role"] == "mandatory"
            and gate["coordinates"] == len(gate["rows"]) == 896,
            "L22 global profile changed")
    failures, sources = [], {}
    for index, metric in enumerate(gate["rows"]):
        require(type(metric["index"]) is int and metric["index"] == index
                and metric["reference_policy"] == census.REFERENCE_POLICY
                and metric["profile_id"] == census.PROFILE
                and metric["boundary"] == "layer-final-output"
                and metric["reference_binary64_hex"] == float(reference[index]).hex()
                and int(metric["actual_fp16_bits"], 16) == int(state["stage18"][index]),
                "L22 original reference/raw state splice")
        actual = Fraction(float(state["stage18"][index:index + 1].view("<f2")[0]))
        original = Fraction(float(reference[index]))
        nearest = np.array([reference[index]], dtype="<f2")
        floor = abs(Fraction(float(nearest[0])) - original)
        excess = abs(actual - original) - floor
        require(metric["nearest_fp16_bits"] == f"{int(nearest.view('<u2')[0]):04x}"
                and Fraction(metric["actual_error"]) == abs(actual - original)
                and Fraction(metric["q"]) == floor
                and Fraction(metric["excess_error"]) == excess
                and Fraction(metric["excess_budget"]) == Fraction(1, 8)
                and metric["accepted"] is (excess <= Fraction(1, 8)),
                "L22 exact threshold metric changed")
        if excess > Fraction(1, 8):
            failures.append(metric)
        for pin in metric["sources"]:
            if pin["path"] in sources:
                same(sources[pin["path"]], pin, "conflicting metric source")
            sources[pin["path"]] = pin
    require(gate["passed"] is False and gate["failure_count"] == 1
            and gate["failures"] == failures and [m["index"] for m in failures] == [62],
            "L22 coordinate-62 failure lost")
    for pin in sources.values():
        read_bound(pin)
    return {**failures[0], "threshold_margin":
            str(Fraction(1, 8) - Fraction(failures[0]["excess_error"]))}


def authenticate(out=OUTPUT):
    check_contract(json.loads(CONTRACT.read_bytes()))
    out = output_path(out)
    pinned = {key: json.loads(read_bound(pin)) for key, pin in PINS.items()}
    latest = json.loads((HANDOFFS / MISSION / "latest.json").read_bytes())
    backlog = [json.loads(line) for line in BACKLOG.read_text().splitlines() if line.strip()]
    check_review(pinned["review"], latest, backlog)
    require(pinned["mission"]["mission_id"] == MISSION
            and pinned["mission"]["node_key"] == "execute-l22-from-l21-failed-parent-suffix"
            and pinned["mission"]["execution_workdir"] == str(ROOT), "review scope mismatch")
    result, validation = pinned["result"], pinned["validation"]
    require(result["diagnostic_id"] == "q24-s16-l22-from-l21-coordinate62-suffix-execution-v1"
            and result["status"] == "DIAGNOSED" and result["output"] == str(BASE)
            and result["native_layer_invocations"] == result["native_L22_P0_invocations"] == 9
            and result["audit"] == {"native_controls": list(CONTROLS), "forbidden_calls": 0}
            and validation["status"] == "VALIDATED_EXECUTION_EVIDENCE"
            and validation["stage_reports_verified"] == 171
            and validation["source_operand_state_KV_lineage_checks"] == "PASS"
            and validation["native_layer_invocations"] == validation["forbidden_calls"] == 0,
            "reviewed L22 execution identity/gates mismatch")
    same(validation["result"], record(BASE / "result.json"), "validation/result splice")
    parent.check_summary(result["preflight"])
    retained = census.authenticate()
    same(result["preflight"]["certified_census"], retained["summary"],
         "reviewed raw L21 history changed")
    for pin in list(result["origins"].values()) + list(result["preflight_pins"].values()):
        read_bound(pin)
    tests = result["tests"]
    require(tests["collected"] == tests["executed"] == 24
            and all(tests[key] == 0 for key in ("failures", "errors", "skipped",
                                               "native_layer_invocations")),
            "reviewed L22 compile/test receipt changed")
    for pin in tests["compiled"]:
        read_bound(pin)
    for key, value in result["flags"].items():
        require(type(result[key]) is type(value) and result[key] == value,
                "L22 execution boundary changed")
    parent.matrix.check_false_flags(result)
    bound = {pin["path"]: pin for pin in result["artifacts"]}
    expected = {str(BASE / name) for name in ("command.json", "interventions.json")}
    expected.update(str(BASE / f"{label}_L22{suffix}") for label in CONTROLS
                    for suffix in (".npz", "_parent.npz", "_local_references.npz", "_gates.json"))
    require(len(result["artifacts"]) == len(bound) == len(expected)
            and set(bound) == expected, "L22 artifact set changed")
    for pin in bound.values():
        read_bound(pin)
    same(json.loads(read_bound(bound[str(BASE / "interventions.json")])),
         result["controls"], "L22 intervention order or content changed")
    require([row["control"] for row in result["controls"]] == list(CONTROLS),
            "L22 controls reordered or duplicated")
    original = result["preflight"]["L22_original_reference"]
    same(validation["original_input_L22_reference_binding"], original,
         "L22 original reference validation splice")
    for key in ("binary64", "fp16", "input_binary64", "input_fp16"):
        read_bound(original["reference"][key])
    reference22 = np.load(io.BytesIO(read_bound(original["reference"]["binary64"])),
                          allow_pickle=False)
    plans, states, kvs = [], {}, {}
    for row, old in zip(result["controls"], retained["summary"]["controls"], strict=True):
        label = row["control"]
        same(row["retained_L21"], old, "L21 historical failure changed")
        state = archive(bound[str(BASE / f"{label}_L22.npz")])
        parent.check_state(state)
        prior_state = archive(row["parent_archive"])
        stored_parent = archive(bound[str(BASE / f"{label}_L22_parent.npz")])
        for key, field in STATE["fields"].items():
            require(np.array_equal(prior_state[field], stored_parent[key])
                    and prior_state[field].dtype == stored_parent[key].dtype,
                    "raw L21 to L22 parent lineage changed")
        same(row["gates"], bound[str(BASE / f"{label}_L22_gates.json")],
             "L22 control/gate splice")
        failure = check_reports(row, json.loads(read_bound(row["gates"])), state, reference22)
        states[label] = {field: state[field] for field in STATE["fields"].values()}
        kvs[label] = parent.empty_kv()
        plans.append({
            "control": label, "parent_node": [22, 0, 18], "consumer_node": [23, 0],
            "parent_archive": bound[str(BASE / f"{label}_L22.npz")],
            "parent_state": STATE, "consumer_kv": KV,
            "retained_L21": old, "retained_L22": row, "L22_failure": failure,
            "L23_status": "NOT_EXECUTED", "candidate_admitted": False,
            "policy_adopted": False, "successor_published": False,
        })
    reference23 = bind_reference(original["manifest"], original["reference"])
    summary = {
        "diagnostic_id": ID, "schema_version": 1, "contract": record(CONTRACT),
        "status": "PREFLIGHT_READY" if not reference23["missing"] else reference23["status"],
        "L23_status": "NOT_EXECUTED", "control_count": 9, "controls": plans,
        "evidence": PINS, "terminal_parent_review": {
            "mission_id": MISSION, "round": 1, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done",
        },
        "future_output": {**FUTURE_OUTPUT, "selected": str(out)},
        "L23_original_reference": reference23,
        "retained_L22_stage_reports": {"PASS": 162, "FAIL": 9},
        "retained_L21_history": retained["summary"],
        "thresholds": retained["summary"]["thresholds"],
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "old_sparse_cut_used_as_parent": False,
        "gate_evidence_scope": "AUTHENTICATED_RETAINED_ONLY; fresh raw I/Z/H and exact S18 metrics",
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY, **FLAGS,
    }
    check_summary(summary)
    return {"summary": summary, "states": states, "kvs": kvs, "result": result}


def check_summary(summary):
    require(summary["diagnostic_id"] == ID and type(summary["schema_version"]) is int
            and summary["schema_version"] == 1 and summary["contract"] == record(CONTRACT),
            "L23 result schema/contract mismatch")
    same(summary["evidence"], PINS, "L23 evidence pins changed")
    same(summary["terminal_parent_review"], {
        "mission_id": MISSION, "round": 1, "producer_role": "reviewer",
        "review_status": "done", "backlog_status": "done", "outcome_review_status": "done",
    }, "terminal parent review changed")
    same(summary["thresholds"], summary["retained_L21_history"]["thresholds"],
         "inherited thresholds changed")
    same(summary["retained_L22_stage_reports"], {"PASS": 162, "FAIL": 9},
         "retained L22 stage census changed")
    for key, value in FLAGS.items():
        require(type(summary[key]) is type(value) and summary[key] == value,
                f"L23 check boundary changed: {key}")
    require(summary["L23_status"] == "NOT_EXECUTED" and summary["control_count"] == 9
            and [row["control"] for row in summary["controls"]] == list(CONTROLS)
            and summary["historical_failures_preserved"] is True
            and summary["original_global_reference_unchanged"] is True
            and summary["old_sparse_cut_used_as_parent"] is False
            and summary["claim_boundary"] == BOUNDARY
            and summary["normal_host_review"] == "REQUIRED", "L23 schedule or claim changed")
    for row in summary["controls"]:
        require(row["parent_node"] == [22, 0, 18] and row["consumer_node"] == [23, 0]
                and row["parent_archive"]["path"] == str(BASE / f"{row['control']}_L22.npz")
                and row["parent_state"] == STATE and row["consumer_kv"] == KV
                and row["L23_status"] == "NOT_EXECUTED"
                and row["retained_L21"]["control"] == row["retained_L22"]["control"] == row["control"]
                and row["retained_L21"]["S18"] == row["retained_L22"]["L22_status"] == "FAIL"
                and row["retained_L21"]["failing_coordinates"] == [62]
                and row["retained_L21"]["failures"][0]["threshold_margin"] == "-3/4"
                and row["retained_L22"]["source_operand_state_KV_RTZ_checks"] == "PASS"
                and row["retained_L22"]["S18_failure_indices"] == [62]
                and row["retained_L22"]["mandatory_statuses"] == ["PASS"] * 18 + ["FAIL"]
                and row["L22_failure"]["index"] == 62
                and Fraction(row["L22_failure"]["excess_budget"]) == Fraction(1, 8)
                and Fraction(row["L22_failure"]["threshold_margin"]) ==
                Fraction(1, 8) - Fraction(row["L22_failure"]["excess_error"]) < 0,
                "L23 raw failed parent plan changed")
        parent.matrix.check_false_flags(row)
        parent.matrix.check_false_flags(row["retained_L21"])
        parent.matrix.check_false_flags(row["retained_L22"])
    reference = summary["L23_original_reference"]
    require((summary["status"] == "PREFLIGHT_READY"
             and reference["status"] == "BOUND_ORIGINAL_INPUT_L23"
             and reference["reference"] is not None and reference["missing"] == [])
            or (summary["status"] == reference["status"] == "BLOCKED_MISSING_L23_REFERENCE"
                and bool(reference["missing"])), "L23 reference readiness mismatch")
    same({key: value for key, value in summary["future_output"].items() if key != "selected"},
         FUTURE_OUTPUT, "future output contract changed")


def validate(out=OUTPUT):
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "disclosed repository/account-bound unoptimized -B command required")
    branch = subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                            check=True, capture_output=True, text=True).stdout.strip()
    require(branch == "argus/full-projection", "isolated branch changed")
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(record(path))
    evidence = authenticate(out)
    spec = importlib.util.spec_from_file_location(f"test_{NAME}", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader missing")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count mismatch")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful() and not result.skipped,
            "focused tests failed, errored or skipped")
    output_path(out)
    return {**evidence["summary"], "command": COMMAND, "compiled": compiled,
            "collected": suite.countTestCases(), "executed": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.out), sort_keys=True))


if __name__ == "__main__":
    main()
