"""Read-only contrast of certified L21 failures and a distinct sparse-cut suffix."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import unittest

from ace3.model.candidates import diagnose_q24_s16_l21_from_l20_s18_failure_census_v1 as census


ROOT = census.ROOT
NAME = "q24_s16_l21_sparse_cut_localization_matrix_v1"
ID = "ace3-q24-s16-l21-sparse-cut-localization-matrix-v1"
SOURCE = ROOT / f"ace3/model/candidates/diagnose_{NAME}.py"
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
TEST = ROOT / f"tests/test_{NAME}.py"
CLOSURE_NAME = "q24_s16_l13_l15_l21_sparse_cut_suffix_closure_v1"
CLOSURE_ID = "ace3-q24-s16-l13-l15-l21-sparse-cut-suffix-closure-v1-r2"
BASE = ROOT / "build" / (CLOSURE_NAME + "_attempt002")
REVIEW_BASE = ROOT / "build" / (CLOSURE_NAME + "_review002")
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
EXPECTED_TESTS = 22
FLAGS = {
    **census.FLAGS, "native_L0_L20_invocations": 0,
    "native_L0_L23_invocations": 0, "hardware_invocations": 0,
    "simulation_invocations": 0, "original_prefix_replay": False,
}
PINS = {
    "census_source": {
        "path": str(census.SOURCE),
        "sha256": "a8faef5d1deb15803357bc952b0bc4c177b23e044a6a30299d5f29d79599c2f7",
    },
    "census_review": {
        "path": str(HANDOFFS / "ecf478e49bef/round-0001.json"),
        "sha256": "4ccd955f1460ad654f3ce190ec4e9ac28605923f8591e6dfa01ec2ad7dd7defc",
    },
    "closure_review": {
        "path": str(HANDOFFS / "beb6029db7e6/round-0001.json"),
        "sha256": "54a9c456f9d5974d3f068107d0db5a1c713e4ece86155c1f837a41d9bfc69862",
    },
    "closure_execution_review": {
        "path": str(HANDOFFS / "7ea8d7751111/round-0002.json"),
        "sha256": "7cca77afcdf5152def3af188d714a25398e16cef611810070acca406fb1fb183",
    },
    "closure_result": {
        "path": str(BASE / "result.json"),
        "sha256": "60c8fde2c388d1ce696b92b8741287098791a5bf0376cf81624b8c73b82f1f14",
    },
    "closure_read_only_result": {
        "path": str(REVIEW_BASE / "result.json"),
        "sha256": "9aa36bd44774071263c0ef9eb9fc4098b148ba3c3d7de0ac23b14e293d0b5246",
    },
}
COMMAND = (
    "PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 "
    "/home/argustest/miniconda3/bin/python -B -m "
    "ace3.model.candidates.diagnose_q24_s16_l21_sparse_cut_localization_matrix_v1 --check"
)
COMPARISON = (
    "Shared separately executed sparse-cut suffix comparator, not a paired "
    "intervention on any census control. Margin differences are arithmetic "
    "contrasts, not causal effects or per-control closure certifications."
)
AUTHENTICATION = (
    "Hash-bound certified census source and independent reviews; reuse only "
    "the census read-only authenticator, not its tests. Authenticate revision2 "
    "attempt002 and review002 results, validation, source/contract origins, "
    "all recorded input files, raw reports/state/local-reference/KV artifacts "
    "and separate adjusted states. Source/operand/state/KV/RTZ gate PASS facts "
    "are authenticated retained evidence, not newly executed scientific gates."
)
BOUNDARY = census.BOUNDARY + " " + COMPARISON
EXPECTED_CONTRACT = {
    "diagnostic_id": ID, "version": 1, "schema_version": 1,
    "cwd": str(ROOT), "evidence": PINS, "controls": list(census.CONTROLS),
    "node": [21, 0, 18], "coordinate": 62, "coordinates_per_control": 896,
    "policy_id": census.POLICY, "profile_id": census.PROFILE,
    "reference_policy": census.REFERENCE_POLICY, "excess_budget": "1/8",
    "local_threshold": census.EXPECTED_CONTRACT["local_threshold"],
    "closure_revision": 2, "closure_attempt": "attempt002",
    "interface": ["--check"], "output": "stdout JSON only; no file writes",
    "command_sidecar": COMMAND, "expected_tests": EXPECTED_TESTS,
    "flags": FLAGS, "normal_host_review": "REQUIRED",
    "comparison_scope": COMPARISON, "authentication_scope": AUTHENTICATION,
    "claim_boundary": BOUNDARY,
}
require = census.require


def check_contract(document):
    require(json.dumps(document, sort_keys=True) ==
            json.dumps(EXPECTED_CONTRACT, sort_keys=True), "matrix contract mismatch")


def check_review(document, mission, round_number):
    require(document["kind"] == "round_reviewed_handoff"
            and document["producer_role"] == "reviewer"
            and document["mission_id"] == mission
            and type(document["round"]) is int and document["round"] == round_number
            and document["review"]["status"] == "done", "independent review mismatch")


def check_false_flags(document):
    for key in ("candidate_admitted", "policy_adopted", "successor_published"):
        require(document[key] is False, f"non-admission flag changed: {key}")


def coordinate(metric):
    def half(word):
        require(isinstance(word, str) and len(word) == 4, "invalid FP16 word")
        return struct.unpack(">e", bytes.fromhex(word))[0]

    actual = half(metric["actual_fp16_bits"])
    nearest = half(metric["nearest_fp16_bits"])
    reference = float.fromhex(metric["reference_binary64_hex"])
    require(all(math.isfinite(x) for x in (actual, nearest, reference)),
            "nonfinite coordinate")
    reference = Fraction(reference)
    error = abs(Fraction(actual) - reference)
    floor = abs(Fraction(nearest) - reference)
    excess = error - floor
    budget = Fraction(metric["excess_budget"])
    require(budget == Fraction(1, 8) and excess >= 0
            and error == Fraction(metric["actual_error"])
            and floor == Fraction(metric["q"])
            and excess == Fraction(metric["excess_error"]), "coordinate arithmetic mismatch")
    margin = budget - excess
    if "accepted" in metric:
        require(metric["accepted"] is (margin >= 0), "coordinate status mismatch")
    return {
        **{key: metric[key] for key in (
            "index", "actual_fp16_bits", "nearest_fp16_bits", "reference_binary64_hex")},
        "actual_error": str(error), "q": str(floor), "excess_error": str(excess),
        "excess_budget": str(budget), "threshold_margin": str(margin),
        "excess_over_budget": str(-margin), "status": "PASS" if margin >= 0 else "FAIL",
    }


def check_gate(report, layer, failures):
    gate = report["binary64_v1"]
    require(report["node"] == [layer, 0, 18] and report["stage"] == 18
            and report["policy_id"] == census.POLICY
            and report["status"] == ("FAIL" if failures else "PASS"),
            "global stage/status mismatch")
    require(gate["profile_id"] == census.PROFILE and gate["role"] == "mandatory"
            and gate["coordinates"] == len(gate["rows"]) == 896,
            "global profile/shape mismatch")
    found = []
    for index, metric in enumerate(gate["rows"]):
        require(type(metric["index"]) is int and metric["index"] == index
                and metric["profile_id"] == census.PROFILE
                and metric["reference_policy"] == census.REFERENCE_POLICY
                and metric["boundary"] == "layer-final-output", "coordinate identity mismatch")
        if coordinate(metric)["status"] == "FAIL":
            found.append(metric)
    require([m["index"] for m in found] == failures
            and gate["failures"] == found and gate["failure_count"] == len(found)
            and gate["passed"] is (not failures), "global failures/status mismatch")
    return coordinate(gate["rows"][62])


def check_reference_pair(closure, original):
    files = ("binary64", "fp16", "input_binary64", "input_fp16")
    require({k: v for k, v in closure.items() if k not in files}
            == {k: v for k, v in original.items() if k not in files},
            "original reference metadata changed")
    for key in files:
        left, right = closure[key], original[key]
        require(left["sha256"] == right["sha256"] and left["bytes"] == right["bytes"],
                "original global reference re-anchored")
        require(Path(left["path"]).parent == ROOT / (
            "build/q24_software_l9_l23_0c02042f0e21_attempt001/references")
            and Path(right["path"]).parent == ROOT / (
                "build/q24_software_l9_l23_4d1cfdfc15c0_attempt002/references")
            and Path(left["path"]).name == Path(right["path"]).name,
            "original reference archive path changed")


def check_closure(result, validation, reviewed, contract, reports, original_reference):
    check_false_flags(result)
    check_false_flags(validation)
    check_false_flags(reviewed)
    require(result["diagnostic_id"] == reviewed["diagnostic_id"] == CLOSURE_ID
            and contract["diagnostic_id"] == CLOSURE_ID and contract["version"] == 2
            and contract["policy_id"] == census.POLICY
            and contract["excess_budget"] == "1/8", "closure identity/threshold mismatch")
    require(result["status"] == "BOUNDED_COMBINED_SUFFIX_PASS"
            and result["first_failure"] is None and result["position"] == 0
            and result["history"] == [9707]
            and result["historical_attempt002_used_as_parent"] is False,
            "closure result mismatch")
    require(result["cuts"] == contract["cuts"] and set(result["cuts"]) == {"13", "15", "21"},
            "closure cut set mismatch")
    require(result["audit"]["native_layers"] == list(range(14, 24))
            and result["native_layer_invocations"] == 10
            and result["native_per_layer"] == {
                str(layer): int(layer >= 14) for layer in range(24)},
            "historical native schedule mismatch")
    for document in (result, validation, reviewed):
        require(all(type(document[k]) is int and document[k] == 0 for k in (
            "rtl_invocations", "native_L0_L13_invocations", "native_L0_L8_invocations",
            "L20_endpoint_interval_replays")), "forbidden historical invocation")
    require(validation["collected"] == validation["executed"] == result["tests_executed"] == 29
            and all(validation[k] == 0 for k in (
                "failures", "errors", "skipped", "native_layer_invocations"))
            and validation["contract_parsed"] is True
            and validation["legacy_19_tests_executed"] is False
            and validation["upstream_tests_executed"] is False, "closure validation mismatch")
    require(reviewed["status"] == "READ_ONLY_CHECK_PASS" and reviewed["layers"] == []
            and reviewed["audit"]["native_layers"] == []
            and reviewed["native_layer_invocations"] == 0
            and reviewed["tests_executed"] == 29, "read-only review evidence mismatch")
    require([entry["layer"] for entry in result["layers"]] == list(range(14, 24)),
            "closure layer set/order mismatch")
    previous = None
    raw62 = adjusted62 = None
    for entry in result["layers"]:
        layer = entry["layer"]
        expected = [62] if layer in (15, 21) else []
        require(entry["position"] == 0 and entry["status"] == "PASS"
                and entry["first_failure"] is None and entry["prior_kv"] == "own empty P0"
                and entry["prior_layer_kv_consumed"] is False, "closure state/KV mismatch")
        if previous is not None:
            require(entry["input_state_evidence"] == previous, "spliced suffix state lineage")
        previous = entry["output_state_evidence"]
        require(entry["raw_first_failure"] == (
            {"gate": "binary64_v1", "index": 62, "node": [layer, 0, 18]}
            if expected else None), "historical raw failure changed")
        stages = reports[layer]
        require(len(stages) == 19, "incomplete closure stage reports")
        for stage, report in enumerate(stages):
            require(report["node"] == [layer, 0, stage] and report["stage"] == stage
                    and report["policy_id"] == census.POLICY
                    and report["kv_lineage"] == report["residual_state_lineage"] == "PASS",
                    "closure source/state/KV gate mismatch")
            if stage < 18:
                gate = report["local_operator_fp16"]
                require(report["status"] == "PASS" and gate["role"] == "mandatory"
                        and gate["passed"] is True and gate["failure_count"] == 0
                        and gate["failures"] == [] and report["local_reference_independent"] is True,
                        "closure local gate mismatch")
        raw = check_gate(stages[18], layer, expected)
        boundary = entry["adjusted_boundary_gate"]
        if expected:
            require(entry["cut"] == result["cuts"][str(layer)]
                    and entry["adjusted_state_evidence"]["sha256"]
                    == entry["output_state_evidence"]["sha256"], "adjusted state/cut mismatch")
            adjusted = check_gate(boundary, layer, [])
        else:
            require(boundary is None, "unplanned adjusted boundary")
        if layer == 21:
            check_reference_pair(entry["original_reference"], original_reference)
            raw62, adjusted62 = raw, adjusted
    require(raw62["reference_binary64_hex"] == adjusted62["reference_binary64_hex"],
            "raw/adjusted reference mismatch")
    return {
        "comparator_id": CLOSURE_ID + ":attempt002:L21",
        "raw_L21_coordinate62": raw62, "adjusted_L21_coordinate62": adjusted62,
        "adjusted_L21_full_vector_status": "PASS", "L22_status": "PASS", "L23_status": "PASS",
        "retained_raw_failure_nodes": [[15, 0, 18], [21, 0, 18]],
        "retained_native_layer_invocations": 10,
        "original_reference": result["layers"][7]["original_reference"],
        "comparison_scope": COMPARISON,
    }


def build_matrix(controls, closure):
    require([row["control"] for row in controls] == list(census.CONTROLS),
            "matrix control set/order mismatch")
    adjusted = closure["adjusted_L21_coordinate62"]
    rows = []
    for control in controls:
        check_false_flags(control)
        require(control["S0_S17"] == "PASS" and control["S18"] == "FAIL"
                and control["failing_coordinates"] == [62] and len(control["failures"]) == 1,
                "retained census status mismatch")
        metric = control["failures"][0]
        raw = coordinate(metric)
        require(raw["index"] == 62 and raw["status"] == "FAIL"
                and raw["threshold_margin"] == metric["threshold_margin"]
                and raw["excess_over_budget"] == metric["excess_over_budget"],
                "retained census margin mismatch")
        require(all(raw[k] == adjusted[k] for k in (
            "reference_binary64_hex", "nearest_fp16_bits", "q", "excess_budget")),
            "census/closure reference mismatch")
        rows.append({
            "control": control["control"], "node": [21, 0, 18],
            "retained_raw_coordinate62": raw, "retained_S0_S17_status": "PASS",
            "adjusted_sparse_cut_L21_status": adjusted["status"],
            "adjusted_sparse_cut_threshold_margin": adjusted["threshold_margin"],
            "margin_contrast_not_causal_effect": str(
                Fraction(adjusted["threshold_margin"]) - Fraction(raw["threshold_margin"])),
            "closure_comparator_id": closure["comparator_id"],
            "per_control_adjusted_execution_performed": False,
            "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        })
    return rows


def authenticate():
    check_contract(json.loads(CONTRACT.read_bytes()))
    records, payloads = {}, {}

    def bind(record, load=False):
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        require(path.resolve() == path, f"noncanonical evidence path: {path}")
        key = str(path)
        if key not in records:
            if load:
                payloads[key] = census.read_bound(record)
                size = len(payloads[key])
            else:
                digest, size = hashlib.sha256(), 0
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                require(digest.hexdigest() == record["sha256"], f"evidence hash mismatch: {path}")
            records[key] = {"path": key, "sha256": record["sha256"], "bytes": size}
        require(records[key]["sha256"] == record["sha256"]
                and ("bytes" not in record or records[key]["bytes"] == record["bytes"]),
                f"conflicting evidence binding: {path}")
        if load:
            require(key in payloads, f"evidence was not loaded as JSON: {path}")
            return json.loads(payloads[key])

    bind(PINS["census_source"])
    reviews = {}
    for label, mission, number in (
        ("census_review", "ecf478e49bef", 1),
        ("closure_review", "beb6029db7e6", 1),
        ("closure_execution_review", "7ea8d7751111", 2),
    ):
        reviews[label] = bind(PINS[label], True)
        check_review(reviews[label], mission, number)
    retained = census.authenticate()
    result = bind(PINS["closure_result"], True)
    reviewed = bind(PINS["closure_read_only_result"], True)
    validation = bind(result["validation"], True)
    require(bind(reviewed["validation"], True) == validation, "review validation binding mismatch")
    contract = bind(validation["contract"], True)
    reports = {}
    for entry in result["layers"]:
        reports[entry["layer"]] = bind(entry["reports"], True)
    closure = check_closure(result, validation, reviewed, contract, reports,
                            retained["summary"]["original_reference"])
    for document in (result, reviewed):
        bind(document["command"])
        bind(document["frozen_L13_parent"])
        for record in document["authenticated_inputs"]:
            bind(record)
    for record in (*validation["origins"].values(), *validation["compiled"]):
        bind(record)
    for entry in result["layers"]:
        for key in ("actual_stages", "local_references", "input_state_evidence",
                    "output_state_evidence", "own_kv_evidence"):
            bind(entry[key])
        if entry["adjusted_boundary_gate"] is not None:
            bind(entry["adjusted_state_evidence"])
        for key in ("binary64", "fp16", "input_binary64", "input_fp16"):
            bind(entry["original_reference"][key])
    summary = {
        "diagnostic_id": ID, "schema_version": 1, "status": "BOUNDED_LOCALIZATION_MATRIX",
        "node": [21, 0, 18], "control_count": 9,
        "matrix": build_matrix(retained["summary"]["controls"], closure),
        "closure_comparator": closure, "evidence": PINS,
        "certified_census_evidence": census.ANCHORS,
        "closure_authenticated_file_count": len(records),
        "census_authenticated_file_count": len(retained["summary"]["authenticated_files"]),
        "policy_id": census.POLICY, "profile_id": census.PROFILE,
        "thresholds": retained["summary"]["thresholds"],
        "original_reference": retained["summary"]["original_reference"],
        "original_global_reference_unchanged": True,
        "retained_L20_failing_controls": retained["summary"]["retained_L20_failing_controls"],
        "historical_failures_preserved": True,
        "gate_evidence_scope": "AUTHENTICATED_RETAINED_ONLY",
        "comparison_scope": COMPARISON, "authentication_scope": AUTHENTICATION,
        "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED", **FLAGS,
    }
    return {
        "summary": summary, "census": retained, "closure": result, "reviewed": reviewed,
        "validation": validation, "closure_contract": contract, "reports": reports,
        "reviews": reviews,
    }


def record(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def validate():
    require(Path(__file__).resolve() == SOURCE and Path.cwd().resolve() == ROOT
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "repository-bound unoptimized -B command required")
    compiled = []
    for path in (SOURCE, TEST):
        data = path.read_bytes()
        compile(data, str(path), "exec", dont_inherit=True)
        compiled.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                         "bytes": len(data)})
    evidence = authenticate()
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
    return {
        **evidence["summary"], "command": COMMAND, "contract": record(CONTRACT),
        "compiled": compiled, "collected": suite.countTestCases(), "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True,
                        help="compile, authenticate and run focused read-only tests; emit JSON")
    parser.parse_args(argv)
    print(json.dumps(validate(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
