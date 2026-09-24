"""Read-only cross-control census of reviewed final-head attempt001.

--check compiles this diagnostic and its tests, runs only its focused tests,
and emits one JSON document to stdout. No model operator is executed.
"""

import argparse
import builtins
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1 as parent


ROOT = parent.ROOT
NAME = "diagnose_q24_s16_final_head_impact_census_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON} -B -m {MODULE} --check"
)
MISSION = "4a3869d59053"
HANDOFFS = parent.preflight.preflight.HANDOFFS
PINS = {
    "result": parent.RETAINED_RESULT,
    "review": {
        "path": str(HANDOFFS / MISSION / "round-0002.json"), "bytes": 717,
        "sha256": "de570e42b4536809ae7caefe60d5a8f9831b153b457581be9dd196160451c0c9",
    },
    "mission": {
        "path": str(HANDOFFS / MISSION / "mission.json"), "bytes": 2516,
        "sha256": "a0796c73bc45bf6d9da7995012c4482c66d0710edd3233ffd3eb73e51cdc5408",
    },
    "reviewed_validator_source": {
        "path": str(parent.SOURCE), "bytes": 23207,
        "sha256": "b406b5f3bc89eebadc89156c3a09ad5b7d3a25d3bf1da55f3c81186835e915e1",
    },
    "reviewed_validator_test": {
        "path": str(parent.TEST), "bytes": 8786,
        "sha256": "14fd0ef07e79ed554dbbdeb96de9603cfd77181d3796701b8a7b9a953d1107a8",
    },
}
KINDS = (("rmsnorm", (896,)), ("logits", (151936,)))
EXPECTED_TESTS = 24
require, same, read_bound = parent.require, parent.same, parent.read_bound
FLAGS = {
    **parent.FLAGS, "final_rmsnorm_invocations": 0, "lm_head_invocations": 0,
    "decoder_invocations": 0, "evidence_writes": 0,
}
BOUNDARY = (
    "Read-only CPU-software cross-control statistics of reviewed final-head attempt001. "
    "No operator, decoder, native layer, prefix/admission, reference, RTL, hardware, "
    "GPU or simulation replay; no evidence writes or token publication. All retained "
    "L21/L22/L23 failures, thresholds, original-input global references and "
    "source/operand/state/KV/lineage gates remain unchanged. Q24 state is wider than "
    "FP16; native G128 asymmetric packed INT4, GEMM nibble ordering, no qzero plus-one, "
    "FP16 scales/operator boundaries/KV and native S16 RTZ remain unchanged. "
    "No new final-head threshold, causal attribution, candidate/policy/successor "
    "adoption, strict-FP16-state W4A16, new-token or full-model admission. "
    "Normal independent Reviewer validation of this new census is REQUIRED."
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("read-only census forbids dispatch or filesystem mutation")

    def checked_open(original):
        def opening(file, mode="r", *args, **kwargs):
            if not isinstance(mode, str) or any(flag in mode for flag in "wax+"):
                return forbidden()
            return original(file, mode, *args, **kwargs)
        return opening

    original_os_open = os.open

    def os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return forbidden()
        return original_os_open(path, flags, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(parent.no_dispatch(audit))
        for name in ("rmsnorm", "logits", "load_operands", "run_control", "execute",
                     "validate", "authenticate", "focused_tests", "write"):
            stack.enter_context(patch.object(parent, name, forbidden))
        stack.enter_context(patch.object(parent.norm, "rmsnorm", forbidden))
        for module in (builtins, io):
            stack.enter_context(patch.object(module, "open", checked_open(module.open)))
        stack.enter_context(patch.object(os, "open", os_open))
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "rename",
                     "replace", "link", "symlink", "chmod", "truncate", "utime"):
            stack.enter_context(patch.object(os, name, forbidden))
        yield


def terminal_review(review, latest, backlog, mission):
    parent.preflight.check_review(review, latest, backlog, mission=MISSION,
                                  round_number=2, pin=PINS["review"])
    require(latest["mission"]["path"] == PINS["mission"]["path"]
            and mission["mission_id"] == MISSION
            and mission["node_key"] == "execute-final-head-from-reviewed-l23-suffix"
            and mission["execution_workdir"] == str(ROOT), "review scope changed")


def artifact_manifest(result, names):
    expected = {"command.json", "result.json"} | {
        f"{label}_{kind}.npy" for label in parent.CONTROLS for kind, _ in KINDS}
    require(set(names) == expected, "unexpected or missing final-head evidence")
    pins = {pin["path"]: pin for pin in result["artifacts"]}
    require(len(pins) == len(result["artifacts"]) == 19
            and set(pins) == {str(parent.OUTPUT / name) for name in expected - {"result.json"}},
            "duplicate, missing or substituted artifact")
    return pins


def check_history(result):
    summary = result["preflight"]
    parent.check_summary(summary)
    parent.check_result(result, {"summary": summary}, parent.OUTPUT)
    require(summary["historical_failures_preserved"] is True
            and summary["original_global_reference_unchanged"] is True,
            "retained failure/reference boundary changed")
    for row in result["controls"]:
        label, plan = row["control"], row["parent"]
        l23 = plan["retained_L23"]
        l22, l21 = l23["retained_L22"], l23["retained_L21"]
        same(l22["retained_L21"], l21, "L21 lineage splice")
        same(l23["S18_failure_indices"], [62] if label == "mapped_all" else [62, 241],
             "L23 failure coordinates changed")
        for history in (l21, l22, l23):
            same(history["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"],
                 "retained stage history changed")
            require(history["control"] == label
                    and all(history[key] is False for key in
                            ("candidate_admitted", "policy_adopted", "successor_published")),
                    "retained admission boundary changed")
        require(l23["source_operand_state_KV_RTZ_checks"] == "PASS"
                and l22["source_operand_state_KV_RTZ_checks"] == "PASS"
                and l21["retained_state_KV_lineage_RTZ"] == "PASS",
                "retained source/operand/state/KV/lineage gates changed")
        for failure in plan["L23_failures"] + l21["failures"] + [l23["L22_failure"]]:
            require(Fraction(failure["excess_budget"]) == Fraction(1, 8)
                    and Fraction(failure["threshold_margin"]) < 0,
                    "retained threshold/failure changed")


def authenticate():
    verified, payload_cache = {}, {}

    def bound(pin):
        path = pin["path"]
        if path in verified:
            for key in ("sha256", "bytes"):
                if key in pin:
                    same(pin[key], verified[path][key], "conflicting input pins")
            return payload_cache[path]
        payload = read_bound(pin)
        verified[path] = {"path": path, "sha256": pin["sha256"], "bytes": len(payload)}
        payload_cache[path] = payload
        return payload

    pinned = {key: bound(pin) for key, pin in PINS.items()}
    review, mission = (json.loads(pinned[key]) for key in ("review", "mission"))
    latest = json.loads((HANDOFFS / MISSION / "latest.json").read_bytes())
    backlog = [json.loads(line) for line in parent.preflight.preflight.BACKLOG.read_text().splitlines()
               if line.strip()]
    terminal_review(review, latest, backlog, mission)
    result = json.loads(pinned["result"])
    check_history(result)
    for name, pin in result["origins"].items():
        if name in parent.RETAINED_SOURCES:
            same(pin, parent.RETAINED_SOURCES[name], "original execution source pin changed")
        else:
            bound(pin)
    for pin in result["preflight_pins"].values():
        bound(pin)
    summary = result["preflight"]
    for pin in summary["evidence"].values():
        bound(pin)
    final = summary["final_reference"]
    for pin in final["authority"].values():
        bound(pin)
    manifest = json.loads(bound(final["manifest"]))
    same(manifest["final"], final["reference"], "final reference manifest splice")
    reference = final["reference"]
    original = summary["L23_original_reference"]
    bound(original["manifest"])
    for key in ("binary64", "fp16"):
        same(reference["input_" + key], original["reference"][key],
             "original-input global reference reanchored")
        bound(reference["input_" + key])
    for row in result["controls"]:
        plan = row["parent"]
        bound(plan["terminal_archive"])
        l23 = plan["retained_L23"]
        for history in (l23, l23["retained_L22"], l23["retained_L21"]):
            bound(history["gates"])
            bound(history["parent_stages"] if "parent_stages" in history
                  else history["parent_archive"])
    pins = artifact_manifest(result, (path.name for path in parent.OUTPUT.iterdir()))
    payloads = {path: bound(pin) for path, pin in pins.items()}
    command = json.loads(payloads[str(parent.OUTPUT / "command.json")])
    require(command["command"] == parent.command_for(parent.OUTPUT)
            and command["cwd"] == str(ROOT) and command["uid"] == 1000
            and command["executable"] == parent.PYTHON, "execution command changed")
    references = {
        key: parent.checked_array(bound(reference[key]), shape, dtype)
        for key, shape, dtype in parent.preflight.FINAL_ARRAYS}
    arrays = {}
    for row in result["controls"]:
        control = {}
        for kind, shape in KINDS:
            path = str(parent.OUTPUT / f"{row['control']}_{kind}.npy")
            same(row["arrays"][kind], pins[path], "control array splice")
            control[kind] = parent.checked_array(payloads[path], shape, "<u2")
        arrays[row["control"]] = control
    return result, arrays, references, list(verified.values())


def metric(actual, binary64, fp16):
    result = parent.comparison(actual, binary64, fp16)
    fp16_error = parent.comparison(actual, fp16.view("<f2").astype("<f8"), fp16)
    return {
        **result,
        "binary64_mismatch_count": int(np.count_nonzero(actual.view("<f2") != binary64)),
        "reviewed_fp16_max_absolute_error": fp16_error["binary64_max_absolute_error"],
        "reviewed_fp16_worst_index": fp16_error["binary64_worst_index"],
        "mismatch_semantics": "binary64 numerical inequality; FP16 word inequality",
    }


def equivalence_classes(rows, kinds):
    groups = {}
    for row in rows:
        key = tuple(row["arrays"][kind]["sha256"] for kind in kinds)
        groups.setdefault(key, []).append(row["control"])
    return [{"sha256": dict(zip(kinds, key, strict=True)), "controls": controls,
             "control_count": len(controls)} for key, controls in groups.items()]


def overlap(left, right):
    return {"overlap_count": len(set(left) & set(right)), "same_order": left == right}


def extrema(rows, kind):
    metrics = [(row["control"], row["metrics"][kind]) for row in rows]
    counts = {}
    for field in ("binary64_mismatch_count", "reviewed_fp16_mismatch_count"):
        values = [value[field] for _, value in metrics]
        counts[field] = {"min": min(values), "max": max(values)}
    errors = {}
    for prefix in ("binary64", "reviewed_fp16"):
        field = prefix + "_max_absolute_error"
        values = [Fraction(value[field]) for _, value in metrics]
        maximum = max(values)
        errors[prefix] = {
            "min_max_absolute_error": str(min(values)), "max_absolute_error": str(maximum),
            "worst_controls": [label for label, value in metrics
                               if Fraction(value[field]) == maximum],
            "worst_indices_by_control": {
                label: value[prefix + "_worst_index"] for label, value in metrics},
        }
    return {"mismatch_counts": counts, "errors": errors}


def census(result, arrays, references):
    rows, ranked = [], {}
    for row in result["controls"]:
        label, actual = row["control"], arrays[row["control"]]
        measured = parent.comparisons(actual, references)
        same(measured, row["comparisons"], "retained final-head comparison changed")
        top = measured["top_k"]
        ranked[label] = [entry["token_id"] for entry in top["actual_diagnostic_only"]]
        l23 = row["parent"]["retained_L23"]
        rows.append({
            "control": label,
            "metrics": {kind: metric(actual[kind], references[kind + "_binary64"],
                                     references[kind + "_fp16"]) for kind, _ in KINDS},
            "top_k": {key: {field: top[key][field] for field in ("overlap_count", "same_order")}
                      for key in ("binary64", "fp16")},
            "retained_failures": {
                "L21": l23["retained_L21"]["failures"], "L22": [l23["L22_failure"]],
                "L23": row["parent"]["L23_failures"],
            },
            "L21_L22_L23_status": ["FAIL", "FAIL", "FAIL"],
        })
    pairs = []
    for index, left in enumerate(parent.CONTROLS):
        for right in parent.CONTROLS[index + 1:]:
            pair = {"left": left, "right": right, "top_k": overlap(ranked[left], ranked[right])}
            for kind, _ in KINDS:
                a, b = arrays[left][kind], arrays[right][kind]
                comparison = parent.comparison(a, b.view("<f2").astype("<f8"), b)
                pair[kind] = {
                    "mismatch_count": comparison["reviewed_fp16_mismatch_count"],
                    "max_absolute_error": comparison["binary64_max_absolute_error"],
                    "worst_index": comparison["binary64_worst_index"],
                }
            pairs.append(pair)
    reference_ranks = {
        key: [entry["token_id"] for entry in parent.top_k(
            references["logits_" + key], words=key == "fp16")] for key in ("binary64", "fp16")}
    return {
        "controls": rows, "control_count": len(rows), "arrays_verified": 2 * len(rows),
        "artifact_equivalence_classes": {
            **{kind: equivalence_classes(result["controls"], (kind,)) for kind, _ in KINDS},
            "joint": equivalence_classes(result["controls"], ("rmsnorm", "logits")),
        },
        "extrema": {kind: extrema(rows, kind) for kind, _ in KINDS},
        "cross_control": {"pair_count": len(pairs), "pairs": pairs,
                          "all_top_k_orders_stable": all(p["top_k"]["same_order"] for p in pairs)},
        "top_k": {
            "k": 10, "order": "descending finite value, ascending diagnostic ID on ties",
            "binary64_vs_fp16_reference": overlap(reference_ranks["binary64"], reference_ranks["fp16"]),
            "per_reference": {
                key: {
                    "overlap_min": min(row["top_k"][key]["overlap_count"] for row in rows),
                    "overlap_max": max(row["top_k"][key]["overlap_count"] for row in rows),
                    "same_order_controls": [row["control"] for row in rows
                                            if row["top_k"][key]["same_order"]],
                } for key in ("binary64", "fp16")},
        },
        "retained_L23_failures": 9,
        "retained_L23_stage_reports": result["preflight"]["retained_L23_stage_reports"],
        "thresholds": result["preflight"]["thresholds"],
        "original_global_reference": result["preflight"]["final_reference"],
        "retained_flags": result["flags"],
    }


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("final_head_impact_census_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.wasSuccessful() and result.testsRun == EXPECTED_TESTS and not result.skipped,
            "focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "census_focused_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        result, arrays, references, files = authenticate()
        report = census(result, arrays, references)
        tests = focused_tests((result, arrays, references, report))
        for pin in origins.values():
            read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "census attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1, "status": "READ_ONLY_IMPACT_CENSUS",
        "command": COMMAND, "pins": PINS, "original_execution_sources": parent.RETAINED_SOURCES,
        "input_terminal_review": {
            "mission_id": MISSION, "round": 2, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done",
        },
        "authentication_scope": (
            "Exact result, original execution source/test pins retained in that result, "
            "live reviewed validator/test and other execution sources, all 20 attempt001 "
            "files, four final reference arrays and their manifest/input pins, and retained "
            "L21/L22/L23 gate/state archives. Operand and ancestral gates remain authenticated "
            "retained facts; no model loading or gate/operator replay."
        ),
        "authenticated_files": files, "diagnostic_sources": origins,
        "controls_verified": 9, "attempt001_files_verified": 20, "reference_arrays_verified": 4,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS, "tests": tests,
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY, **report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
