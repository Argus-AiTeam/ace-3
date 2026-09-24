"""Read-only census of the independently reviewed L21/P0 S18 failures."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path("/home/argustest/ace3-argus")
NAME = "q24_s16_l21_from_l20_s18_failure_census_v1"
ID = "ace3-q24-s16-l21-from-l20-s18-failure-census-v1"
SOURCE = ROOT / "ace3/model/candidates" / f"diagnose_{NAME}.py"
CONTRACT = ROOT / "ace3/contracts/candidates" / f"{NAME}.json"
TEST = ROOT / "tests" / f"test_{NAME}.py"
BASE = ROOT / "build/q24_s16_l21_from_l20_coordinate62_suffix_v1_fresh_e0ffefcd1586_attempt001"
REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "e0ffefcd1586/round-0003.json")
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
HISTORICAL_FAILURES = ["actual", "frozen_o", "frozen_down", "frozen_o_down"]
POLICY = "ace3-w4a16-local-operator-global-binary64-authority-v3"
PROFILE = "ace3-w4a16-layer-final-binary64-fp16-excess-v1"
REFERENCE_POLICY = "legacy-binary64-AWQ-fully-independent-propagation"
COMMAND = (
    "PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 "
    "/home/argustest/miniconda3/bin/python -B -m "
    "ace3.model.candidates.diagnose_q24_s16_l21_from_l20_s18_failure_census_v1 --check"
)
EXPECTED_TESTS = 18
FLAGS = {
    "native_layer_invocations": 0, "native_L0_L21_invocations": 0,
    "rtl_invocations": 0, "reference_recomputation": False,
    "reference_reanchoring": False, "accepted_prefix_replay": False,
    "candidate_admitted": False, "policy_adopted": False,
    "successor_published": False,
}
ANCHORS = {
    "review": {
        "path": str(REVIEW),
        "sha256": "d8a961a4eb0ea09526febb557b2ddc0f74d1ff4094cd2b2b2101237b023cf12f",
    },
    "result": {
        "path": str(BASE / "result.json"), "bytes": 723870,
        "sha256": "32fe8fa45a34770e326ba3763fd032a227a33b621eda7732fa9b45ff7b91353c",
    },
    "validation": {
        "path": str(BASE / "validation.json"), "bytes": 745760,
        "sha256": "7ee2d13f60eaffbecd5905a81a2cfa53cc0e9babf123a95bec46dd1d8ed27102",
    },
}
BOUNDARY = (
    "Bounded CPU-software non-admission evidence only. Q24 residual state is "
    "wider than FP16; native G128 asymmetric packed INT4 weights, GEMM nibble "
    "ordering, no qzero plus-one, FP16 scales/operator boundaries/KV and S16 "
    "RTZ are unchanged. No prefix replay, native dispatch, RTL, hardware, "
    "simulation, reference recomputation/re-anchoring or successor publication. "
    "No unique cause, repair, strict-FP16-state W4A16, new-token or full-model "
    "admission is established. Independent Host Reviewer acceptance is required."
)
EXPECTED_CONTRACT = {
    "diagnostic_id": ID, "version": 1, "cwd": str(ROOT),
    "evidence": ANCHORS, "controls": list(CONTROLS),
    "node": [21, 0, 18], "coordinates_per_control": 896,
    "policy_id": POLICY, "profile_id": PROFILE,
    "excess_budget": "1/8",
    "local_threshold": (
        "finite AND (absolute_error <= 0.125 OR (relative_error < 0.001 AND "
        "ordered_FP16_ULP <= 1)); denominator max(abs(reference), 2^-14)"
    ),
    "interface": ["--check"], "command_sidecar": COMMAND,
    "expected_tests": EXPECTED_TESTS, "flags": FLAGS,
    "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
    "authentication_scope": (
        "Pinned round-3 review, L21 result/validation, all 37 L21 artifacts, "
        "retained source/history files, L20 parent result/review/stage archives "
        "and original L20/L21 reference files. The 2593 input-binding records "
        "are preserved by the pinned result/validation; their entire ancestral "
        "payload closure is not replayed or reread. Source/operand/state/KV/RTZ "
        "gates are authenticated retained facts, not newly executed checks."
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_contract(document):
    require(document == EXPECTED_CONTRACT, "census contract mismatch")


def read_bound(record):
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    require(path.resolve() == path, f"noncanonical evidence path: {path}")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == record["sha256"],
            f"evidence hash mismatch: {path}")
    if "bytes" in record:
        require(len(data) == record["bytes"], f"evidence size mismatch: {path}")
    return data


def check_review(review):
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "e0ffefcd1586"
            and type(review["round"]) is int and review["round"] == 3
            and review["review"]["status"] == "done", "L21 review not terminal DONE")


def check_lineage(result, validation, parent):
    require(result["diagnostic_id"] == validation["diagnostic_id"]
            == "ace3-q24-s16-l21-from-l20-coordinate62-suffix-v1",
            "L21 diagnostic identity mismatch")
    require(result["status"] == "DIAGNOSED" and result["retained_status"] == "FAIL"
            and result["node"] == [21, 0, 18] and result["control_count"] == 9
            and result["all_L21_gate_passing_controls"] == [], "L21 result mismatch")
    require(validation["status"] == "VALIDATED_SOFTWARE_SURFACE"
            and validation["L21_status"] == "NOT_EXECUTED"
            and validation["collected"] == validation["executed"] == 32
            and all(validation[k] == 0 for k in (
                "failures", "errors", "skipped", "native_layer_invocations",
                "rtl_invocations", "dispatch_guard_calls")), "L21 validation mismatch")
    require(result["validation"] == ANCHORS["validation"], "validation binding mismatch")
    for key in ("candidate_admitted", "policy_adopted", "successor_published",
                "accepted_prefix_replay"):
        require(result[key] is False and validation[key] is False,
                f"retained boundary changed: {key}")
    require(result["native_layer_invocations"] == result["native_L21_P0_invocations"] == 9
            and result["native_L0_L20_invocations"] == result["rtl_invocations"] == 0,
            "historical execution schedule mismatch")
    require(result["source_operand_state_KV_lineage_checks"] == "PASS"
            and result["original_global_reference_unchanged"] is True,
            "retained gates/reference failed")
    require(result["origins_after_execution"] == validation["origins"]
            and result["input_bindings"] == validation["input_bindings"]
            and len(result["input_bindings"]) == 2593, "source/input lineage mismatch")
    for key in ("parent_review", "selected_result", "L21_original_reference",
                "retained_L20_failing_controls", "retained_L20_gate_passing_controls"):
        require(result[key] == validation[key], f"spliced lineage: {key}")
    failures = result["retained_L20_failing_controls"]
    require([row["control"] for row in failures] == HISTORICAL_FAILURES
            and result["retained_L20_gate_passing_controls"] == list(CONTROLS),
            "historical L20 controls changed")
    for row in failures:
        require(row["L20_status"] == "FAIL" and row["all_L20_gates_pass"] is False
                and row["L20_S18_failure_indices"] == [62]
                and row["mandatory_statuses"] == ["PASS"] * 18 + ["FAIL"]
                and row["retained_L15_S18_failure_indices"] == [62]
                and row["retained_L18_S18_failure_indices"] == [62]
                and row["candidate_admitted"] is False, "historical failure changed")
    reference = result["L21_original_reference"]
    previous = parent["L20_original_reference"]
    require(reference["input_binary64"] == previous["binary64"]
            and reference["input_fp16"] == previous["fp16"]
            and reference["prior_kv"] == "own empty P0", "re-anchored reference/KV")
    require(reference["binary64"]["sha256"]
            == "462a59d1c4b250fa65c04becff13fb3a424c9e1de4263b694a73be2b5eaf09b9"
            and reference["fp16"]["sha256"]
            == "a2df2b4e70a65278a79dff65616ac649905004d1bb959bdf474ffc95d966d391",
            "original L21 reference changed")


def artifact_manifest(records):
    expected = {str(BASE / "interventions.json")}
    for label in CONTROLS:
        expected.update(str(BASE / f"{label}_L21{suffix}") for suffix in (
            ".npz", "_parent.npz", "_local_references.npz", "_gates.json"))
    require(len(records) == 37 and {r["path"] for r in records} == expected,
            "L21 artifact set incomplete, duplicated or substituted")
    return {r["path"]: r for r in records}


def summarize_control(row, reports):
    require(row["node"] == [21, 0] and row["prior_kv"] == "own empty P0"
            and row["prior_layer_kv_consumed"] is False
            and row["parent_fields"] == {"h": "stage18", "i": "output_i", "z": "output_z"}
            and row["L21_source_operand_state_KV_RTZ_checks"] == "PASS",
            "retained parent/state/KV/RTZ gate mismatch")
    require(row["L21_status"] == "FAIL" and row["all_L21_gates_pass"] is False
            and row["conditional_acceptance_retained"] is False
            and row["candidate_admitted"] is False, "control admission/status mismatch")
    for layer in range(15, 21):
        require(row[f"retained_L{layer}_all_gates_pass"] is True
                and row[f"retained_L{layer}_S18_failure_indices"] == [],
                f"retained L{layer} parent failed")
    statuses = ["PASS"] * 18 + ["FAIL"]
    require(len(reports) == 19 and row["mandatory_statuses"] == statuses,
            "incomplete stage census")
    for stage, report in enumerate(reports):
        require(report["stage"] == stage and report["node"] == [21, 0, stage]
                and report["status"] == statuses[stage] and report["policy_id"] == POLICY
                and report["kv_lineage"] == report["residual_state_lineage"] == "PASS",
                f"retained S{stage} status/lineage mismatch")
        if stage < 18:
            gate = report["local_operator_fp16"]
            require(gate["role"] == "mandatory" and gate["passed"] is True
                    and gate["failure_count"] == 0 and gate["failures"] == [],
                    f"retained S{stage} local gate mismatch")
    gate = reports[18]["binary64_v1"]
    metrics = gate["rows"]
    require(gate["profile_id"] == PROFILE and gate["role"] == "mandatory"
            and gate["coordinates"] == len(metrics) == 896 and gate["passed"] is False,
            "S18 profile/shape mismatch")
    failures, margins = [], []
    for index, metric in enumerate(metrics):
        require(type(metric["index"]) is int and metric["index"] == index
                and metric["profile_id"] == PROFILE
                and metric["reference_policy"] == REFERENCE_POLICY
                and metric["boundary"] == "layer-final-output",
                "coordinate/reference profile mismatch")
        error, floor, excess, budget = (
            Fraction(metric[key]) for key in ("actual_error", "q", "excess_error", "excess_budget"))
        require(error >= 0 and floor >= 0 and excess >= 0 and error - floor == excess
                and budget == Fraction(1, 8), "invalid exact error/excess/budget")
        margin = budget - excess
        require(metric["accepted"] is (margin >= 0), "coordinate acceptance mismatch")
        margins.append(margin)
        if margin < 0:
            failures.append(metric)
    require(gate["failure_count"] == len(failures) == 1
            and gate["failures"] == failures
            and row["L21_S18_failure_indices"] == [m["index"] for m in failures] == [62],
            "certified failure census mismatch")
    scalar = row["index62"]
    require(all(scalar[k] == value for k, value in failures[0].items() if k != "index")
            and Fraction(scalar["excess_over_budget"]) == -margins[62],
            "scalar/vector failure mismatch")
    nearest = min(abs(margin) for margin in margins)
    return {
        "control": row["control"], "node": [21, 0, 18],
        "mandatory_statuses": statuses, "S0_S17": "PASS", "S18": "FAIL",
        "failing_coordinates": [62],
        "failures": [{
            **{k: m[k] for k in (
                "index", "actual_fp16_bits", "nearest_fp16_bits", "reference_binary64_hex",
                "actual_error", "q", "excess_error", "excess_budget")},
            "excess_over_budget": str(-margins[m["index"]]),
            "threshold_margin": str(margins[m["index"]]),
        } for m in failures],
        "nearest_threshold": {
            "absolute_margin": str(nearest),
            "coordinates": [
                {"index": i, "threshold_margin": str(margin), "accepted": margin >= 0}
                for i, margin in enumerate(margins) if abs(margin) == nearest
            ],
        },
        "retained_state_KV_lineage_RTZ": "PASS",
        "gate_evidence_scope": "AUTHENTICATED_RETAINED_ONLY",
        "retained_L15_L20_all_gates_pass": True,
        "prior_kv": row["prior_kv"], "prior_layer_kv_consumed": False,
        "parent_stages": row["parent_stages"], "parent_fields": row["parent_fields"],
        "gates": row["gates"], "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False,
    }


def summarize(rows, gates):
    require([r["control"] for r in rows] == list(CONTROLS), "control order/set mismatch")
    return [summarize_control(row, gates[row["control"]]) for row in rows]


def authenticate():
    check_contract(json.loads(CONTRACT.read_bytes()))
    authenticated = {}
    payloads = {}

    def bind(record):
        path = str(Path(record["path"]) if Path(record["path"]).is_absolute()
                   else ROOT / record["path"])
        if path not in payloads:
            payloads[path] = read_bound(record)
            authenticated[path] = {
                "path": path, "bytes": len(payloads[path]), "sha256": record["sha256"]}
        require(authenticated[path]["sha256"] == record["sha256"]
                and ("bytes" not in record or authenticated[path]["bytes"] == record["bytes"]),
                f"conflicting evidence binding: {path}")
        return payloads[path]

    review, result, validation = (json.loads(bind(ANCHORS[key]))
                                  for key in ("review", "result", "validation"))
    check_review(review)
    parent = json.loads(bind(result["selected_result"]))
    parent_review = json.loads(bind(result["parent_review"]))
    require(parent_review["producer_role"] == "reviewer"
            and parent_review["mission_id"] == "dc9d3eb787bd"
            and parent_review["review"]["status"] == "done", "L20 review lineage mismatch")
    check_lineage(result, validation, parent)
    old_contract = json.loads(bind(validation["contract"]))
    require(old_contract["thresholds"] == validation["thresholds"]
            and old_contract["policy_id"] == POLICY, "threshold/policy lineage changed")
    for item in list(validation["origins"].values()) + validation["history"]:
        if "path" in item:
            bind(item)
        else:
            require(all(Path(p).is_dir() and Path(p).is_relative_to(ROOT)
                        for p in item["namespace_paths"]), "source namespace mismatch")
    manifest = artifact_manifest(result["artifacts"])
    for record in manifest.values():
        bind(record)
    reference = result["L21_original_reference"]
    for key in ("binary64", "fp16", "input_binary64", "input_fp16"):
        bind(reference[key])
    rows = json.loads(bind(manifest[str(BASE / "interventions.json")]))
    gates = {}
    for row in rows:
        expected = manifest[str(BASE / f"{row['control']}_L21_gates.json")]
        require(row["gates"] == expected, "control/gate binding mismatch")
        bind(row["parent_stages"])
        gates[row["control"]] = json.loads(bind(expected))
        for metric in gates[row["control"]][18]["binary64_v1"]["rows"]:
            for source in metric["sources"]:
                bind(source)
    controls = summarize(rows, gates)
    return {
        "result": result, "validation": validation, "parent": parent,
        "review": review, "rows": rows, "gates": gates,
        "summary": {
            "diagnostic_id": ID, "status": "BOUNDED_FAILURE_CENSUS",
            "reviewed_mission": "e0ffefcd1586", "review_round": 3,
            "control_count": 9, "S0_S17_PASS_reports": 162, "S18_FAIL_reports": 9,
            "failure_coordinate_counts": {"62": 9}, "controls": controls,
            "evidence": ANCHORS, "authenticated_files": list(authenticated.values()),
            "artifact_count": len(manifest), "policy_id": POLICY,
            "thresholds": validation["thresholds"],
            "original_reference": reference,
            "original_global_reference_unchanged": True,
            "retained_L20_failing_controls": HISTORICAL_FAILURES,
            "historical_failures_preserved": True,
            "authentication_scope": EXPECTED_CONTRACT["authentication_scope"],
            "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED", **FLAGS,
        },
    }


def validate():
    require(Path(__file__).resolve() == SOURCE and Path.cwd().resolve() == ROOT,
            "census source/workdir mismatch")
    compiled = []
    for path in (SOURCE, TEST):
        data = path.read_bytes()
        compile(data, str(path), "exec", dont_inherit=True)
        compiled.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
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
        **evidence["summary"], "command": COMMAND, "compiled": compiled,
        "collected": suite.countTestCases(), "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True,
                        help="compile repository files, authenticate evidence and run focused tests")
    parser.parse_args(argv)
    print(json.dumps(validate(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
