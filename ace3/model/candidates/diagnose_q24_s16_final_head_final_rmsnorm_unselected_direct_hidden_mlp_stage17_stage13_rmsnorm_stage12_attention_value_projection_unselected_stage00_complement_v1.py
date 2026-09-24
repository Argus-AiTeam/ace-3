"""Stdout-only exact unselected stage00 accounts for the middle-pair V bridge."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_projection_input_bridge_v1 as previous


base, parent, bridge = previous.base, previous.parent, previous.bridge
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_projection_unselected_stage00_complement_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
WIDTH, LIMIT = 896, 8
FIELDS, GROUPS = previous.FIELDS, previous.GROUPS
PARENT_PINS = (
    {"path": str(previous.SOURCE), "bytes": 18430,
     "sha256": "c07a1bb1c015e2643ed2b6819cfb17367b6c6233b117213306dc08dc1f4cd137"},
    {"path": str(previous.TEST), "bytes": 22756,
     "sha256": "ffd6f27b9675d95a578496228ef4c2f56b034e6e74a09a1b36a5f6fc4a4a6e1e"},
)
FLAGS = {**previous.FLAGS, "stage00_complement_causal_allocation": False}
BOUNDARY = (
    "Only the unselected stage00 complement of every retained selected value "
    "projection account for pair (34319,13) is expanded. The eight selected "
    "coordinates and all downstream norm/gate/up/down/row/opposite-reference "
    "fields remain unchanged. Seven contiguous G128 groups in input order "
    "partition 888 coordinates, with exact signed/absolute/cancellation totals. "
    "Shared control/value-coordinate complement tables use the retained "
    "input_coordinate_multipliers for every logical account and group; repeated "
    "accounts are not independent samples. Selected plus unselected plus the "
    "separate actual and negative original projection boundaries closes every "
    "downstream field. This accounting is not a dominant-cause, counterfactual, "
    "new-token, full-model or strict-FP16-state W4A16 admission. " + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("stage00 complement forbids predecessor checks, replay and writes")

    with previous.read_only(audit), patch.object(previous, "check", forbidden), \
            patch.object(previous, "run_tests", forbidden), \
            patch.object(previous, "collect", forbidden):
        yield


def authenticate_parent():
    for pin in PARENT_PINS:
        base.read_bound(pin)
    previous.authenticate_parent()


def bind_parent(inputs, retained):
    authenticate_parent()
    same(retained["internal_reference"], "original_input_L23_fp16",
         "original-input internal reference changed")
    same(retained["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
         "binary64 internal attention reconstruction forbidden")
    same(retained, previous.report(*inputs), "retained V bridge binding changed")


def split_unselected(actual, reference, weights, core):
    expected = previous.projection_account(
        actual, reference, weights, Fraction(core["actual_v"]),
        Fraction(core["original_input_fp16_v"]), Fraction(core["shared_fp16_bias"]),
        core["output_coordinate"])
    same({k: v for k, v in core.items() if k != "control"}, expected,
         "retained selected stage00/projection account changed")
    selected = [entry["coordinate"] for entry in core["largest_absolute_input_coordinates"]]
    excluded = set(selected)
    require(len(selected) == len(excluded) == LIMIT, "selected coordinate census changed")
    terms = [(a-r)*w for a, r, w in zip(actual, reference, weights, strict=True)]
    groups = []
    for group in range(7):
        indices = [i for i in range(group*128, (group+1)*128) if i not in excluded]
        groups.append({
            "input_group": group, "start_coordinate": group*128,
            "end_coordinate_exclusive": (group+1)*128, "coordinate_count": len(indices),
            "excluded_ranked_coordinate_count": 128-len(indices),
            **bridge.mass(terms[i] for i in indices),
        })
    total = previous.previous.merged_mass(groups)
    same(total, core["input_coordinate_totals"]["unselected"],
         "unselected coordinate mass does not close")
    require(sum(g["coordinate_count"] for g in groups) == WIDTH-LIMIT,
            "G128 complement partition changed")
    same(previous.previous.merged_mass(
        (core["input_coordinate_totals"]["selected"], total)),
        core["input_coordinate_totals"]["full"], "selected/unselected mass does not close")
    return {
        "attention_reference": "original_input_fp16", "source_position": 0,
        "source_token_id": 9707, "coordinate_count": WIDTH-LIMIT,
        "excluded_ranked_coordinates": selected, "groups_in_input_order": groups,
        "totals": total, "closure_residual": "0",
    }


def accounts(report):
    for row in report["rows"]:
        for hotspot in row["hotspots"]:
            for selected in hotspot["selected_stage16_coordinates"]:
                for kind in ("gate_proj", "up_proj"):
                    for source in selected[kind]["ranked_stage13_rmsnorm_sources"]:
                        yield from source["attention_value_projection_input"]["selected_value_components"]


def weighted_closure(core, complement, account):
    same(list(account["input_coordinate_multipliers"]), list(FIELDS),
         "downstream multiplier census changed")
    same([t["term"] for t in account["terms"]], list(previous.TERMS),
         "projection boundary term census changed")
    for field in FIELDS:
        multiplier = Fraction(account["input_coordinate_multipliers"][field])
        masses = account["input_coordinate_totals"]
        for group in GROUPS:
            local = complement["totals"] if group == "unselected" else core["input_coordinate_totals"][group]
            same(masses[group][field], previous.previous.scaled_mass(local, multiplier),
                 "weighted coordinate mass changed: " + field)
        terms = [Fraction(core[term])*multiplier for term in previous.TERMS]
        same([t[field] for t in account["terms"]], [str(t) for t in terms],
             "weighted projection boundaries changed: " + field)
        closed = (Fraction(masses["selected"][field]["signed"])
                  + Fraction(masses["unselected"][field]["signed"]) + sum(terms[1:]))
        same(str(closed), account["retained_parent_contributions"][field],
             "selected/unselected/boundary identity failed: " + field)
        same(account["closure_residuals"][field], "0", "parent field closure changed")


def report(inputs, retained):
    bind_parent(inputs, retained)
    actual, reference, tensors = inputs[2]
    columns, cores = {}, {}
    for key, core in retained["value_projection_local_accounts"].items():
        coordinate = core["output_coordinate"]
        same(key, f'{core["control"]}:{coordinate}', "V projection account key changed")
        if coordinate not in columns:
            columns[coordinate] = previous.value.weight_column(tensors, coordinate)
        complement = split_unselected(actual[core["control"]], reference, columns[coordinate], core)
        cores[key] = {**core, "unselected_stage00_inputs": complement}
    count, used = 0, set()
    for account in accounts(retained):
        key = account["projection_account_key"]
        require(key in cores, "selected V account refers to missing complement")
        core = cores[key]
        weighted_closure(core, core["unselected_stage00_inputs"], account)
        count += 1
        used.add(key)
    same(sorted(used), sorted(cores), "unreferenced V complement account")
    same(count, retained["attention_value_projection_input_summary"]
         ["selected_value_component_count"], "logical selected V census changed")
    return {
        **retained, "value_projection_local_accounts": cores,
        "unselected_stage00_complement_summary": {
            "selected_value_component_count": count,
            "unselected_input_coordinate_count": count*(WIDTH-LIMIT),
            "unselected_input_group_count": count*7,
            "shared_projection_account_count": len(cores),
            "selected_entries_unchanged": True,
            "downstream_fields": list(FIELDS),
            "closure_residuals": {field: "0" for field in FIELDS},
        },
        "unselected_stage00_complement_weighting":
            "each shared G128 group signed * input_coordinate_multipliers[field]; "
            "absolute * abs(multiplier); cancellation=absolute-abs(signed). "
            "Existing per-account selected/unselected/full input_coordinate_totals "
            "and separate actual/negative-original terms remain unchanged and close "
            "retained_parent_contributions in every field.",
    }


def collect(audit):
    authenticate_parent()
    inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*inputs)
    return inputs, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("stage11_v_complement_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "stage00 complement tests failed, errored or skipped")
    return {"executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "stage11_v_complement_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    e = previous.previous.evidence(inputs[0][0][0])
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *PARENT_PINS, *previous.PARENT_PINS,
                    *base.PINS.values(), *bridge.margin.rows.PINS.values(), *e["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "stage00 complement attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_MIDDLE_PAIR_STAGE11_V_PROJECTION_UNSELECTED_STAGE00_COMPLEMENT",
        "command": COMMAND, "diagnostic_sources": origins, "parent_source_pins": PARENT_PINS,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": e["files"], "hidden_pins": e["hidden_pins"], "assets": e["assets"],
        "L23_reference_authority": e["result"]["preflight"]["L23_original_reference"],
        "final_reference_authority": e["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": e["result"]["controls"],
        "retained_thresholds": e["result"]["preflight"]["thresholds"],
        "L23_archives_verified": 10, "v_projection_tensors_verified": 4,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": {"compiled": compiled, **tests}, "normal_host_review": "REQUIRED",
        "claim_boundary": BOUNDARY, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
