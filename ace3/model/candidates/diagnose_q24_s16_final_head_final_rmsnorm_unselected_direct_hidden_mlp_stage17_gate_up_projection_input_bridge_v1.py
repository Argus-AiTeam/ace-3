"""Stdout-only stage13 accounts for retained middle-pair stage16 gate/up deltas."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_down_input_silu_product_bridge_v1 as previous


gate_up = previous.silu.gate_up
bridge, hotspot, down = previous.bridge, previous.hotspot, previous.down
base, parent = previous.base, previous.parent
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_gate_up_projection_input_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 17
FLAGS = {**previous.FLAGS, "mlp_gate_up_projection_replay": 0,
         "stage13_input_causal_allocation": False}
BOUNDARY = (
    "Only the gate/up operands of the retained middle-pair (34319,13) "
    "stage16 selections are bridged, without reselection. Each stage14 gate "
    "and stage15 up delta closes as all 896 exact "
    "(stage13_actual-stage13_original_input_FP16)*W terms plus its own "
    "projection-boundary remainder. The remainder is not solely rounding. "
    "All seven contiguous G128 groups and eight descending-absolute top inputs "
    "with ascending-coordinate tie breaks retain signed, absolute and "
    "cancellation totals. The six canonical gate/up tensors use native G128 "
    "asymmetric INT4 GEMM nibble ordering, no qzero plus-one, FP16 scales and "
    "no bias. This is post-RMSNorm FP16 operand accounting, not propagation "
    "of raw Q24 state through RMSNorm, projection or nonlinear/product replay, "
    "nor causal allocation to stage16, stage17 or the final margin. The "
    "reviewed four-term stage16 bridge and its separate down-projection "
    "remainder are unchanged. Repeated selections are logical accounts, not "
    "independent samples. Both final-reference branches retain the same "
    "original-input FP16 internal operands; no binary64 internal stage is "
    "reconstructed. This bounded diagnostic does not pass the common UNKNOWN "
    "stop gate or authorize further expansion. " + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("stage13 bridge forbids predecessor execution, replay and writes")

    with previous.read_only(audit), ExitStack() as stack:
        for module, names in (
            (previous, ("check", "run_tests", "collect")),
            (gate_up, ("check", "focused_tests", "measure", "report")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def load_tensors(evidence):
    canonical = evidence["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
    for kind in gate_up.STAGES:
        gate_up.tensor_names(canonical, kind)
    with parent.preflight.parent.producer.legacy.safe_open(
            evidence["assets"]["checkpoint"]["path"], framework="numpy") as model:
        return {
            kind: {
                suffix: gate_up.bind_tensor(
                    model.get_tensor(prefix + suffix), canonical[prefix + suffix], kind, suffix)
                for suffix in gate_up.SPECS
            } for kind, prefix in gate_up.PREFIXES.items()
        }


def bind_inputs(evidence, tensors):
    binding = evidence["result"]["preflight"]["L23_original_reference"]["reference"]
    canonical = binding["canonical"]
    same(list(evidence["archives"]), list(hotspot.census.CONTROLS),
         "stage13 archive control order changed")
    same(sorted(tensors), sorted(gate_up.STAGES), "gate/up projection census changed")
    for kind in gate_up.STAGES:
        gate_up.tensor_names(canonical, kind)
        same(sorted(tensors[kind]), sorted(gate_up.SPECS), "gate/up tensor census changed")
        for suffix in gate_up.SPECS:
            gate_up.bind_tensor(tensors[kind][suffix],
                                canonical[gate_up.PREFIXES[kind] + suffix], kind, suffix)
    sources = [(None, evidence["reference_archive"], binding["fp16"])]
    sources.extend((row["control"], evidence["archives"][row["control"]],
                    row["parent"]["terminal_archive"]) for row in evidence["result"]["controls"])
    actual, reference = {}, None
    for control, archive, pin in sources:
        authenticated = down.residual.read_archive(pin)
        for stage in ("stage13", "stage14", "stage15"):
            require(archive[stage].dtype == authenticated[stage].dtype
                    and archive[stage].shape == authenticated[stage].shape
                    and np.array_equal(archive[stage], authenticated[stage]),
                    "retained stage13/gate/up archive binding changed")
        values = gate_up.operands(archive)
        if control is None:
            reference = values
        else:
            actual[control] = values
    same(list(actual), list(hotspot.census.CONTROLS), "stage13 operand census changed")
    require(reference is not None, "original L23 stage13 reference missing")
    return actual, reference


def account(actual, reference, weights, kind, selected):
    coordinate = selected["selected_down_input"]["coordinate"]
    raw = gate_up.account(actual, reference, weights, kind, coordinate)
    operand = {"gate_proj": "gate", "up_proj": "up"}[kind]
    local = selected["bridge"]
    for field, expected in (
        ("actual_projection_output", local["actual"][operand]),
        ("reference_projection_output", local["reference_operands"][operand]),
        ("retained_projection_delta", local[operand + "_delta"]),
    ):
        same(raw[field], expected, "retained stage16 gate/up operand splice")
    delta = [a-r for a, r in zip(actual["stage13"], reference["stage13"], strict=True)]
    terms = [value*weight for value, weight in zip(delta, weights, strict=True)]
    top = [row["coordinate"] for row in raw["largest_absolute_coordinates"]]
    selected_ids = set(top)
    total = bridge.mass(terms)
    ranked = bridge.mass([terms[i] for i in top])
    unranked = bridge.mass([term for i, term in enumerate(terms) if i not in selected_ids])
    same(total["signed"], raw["exact_input_delta"], "exact stage13 input sum changed")
    same(total["absolute"], raw["sum_absolute_contributions"], "stage13 absolute sum changed")
    remainder = Fraction(raw["projection_boundary_remainder"])
    retained_delta = Fraction(raw["retained_projection_delta"])
    closure = sum(terms, Fraction())+remainder-retained_delta
    require(closure == 0, "stage13 to retained gate/up delta closure changed")
    return {
        **raw,
        "input_coordinates": [
            {"coordinate": i, "input_group": i//128, "input_delta": str(delta[i]),
             "weight": str(weights[i]), "signed_contribution": str(value),
             "absolute_contribution": str(abs(value))}
            for i, value in enumerate(terms)],
        "largest_absolute_coordinates": [
            {**row, "absolute_contribution": str(abs(Fraction(row["signed_contribution"])))}
            for row in raw["largest_absolute_coordinates"]],
        "groups_in_input_order": [
            {**row, "cancellation_absolute_mass": str(
                Fraction(row["sum_absolute_contributions"])
                - abs(Fraction(row["signed_contribution"])))}
            for row in raw["groups_in_input_order"]],
        "input_totals": total, "ranked_input_totals": ranked, "unranked_input_totals": unranked,
        "input_and_boundary_cancellation_mass": str(
            Fraction(total["absolute"])+abs(remainder)-abs(retained_delta)),
        "projection_closure_residual": str(closure),
        "retained_stage16_operand_delta_unchanged": True,
    }


def report(previous_inputs, tensors, retained):
    previous.validate_report(previous_inputs, retained)
    evidence = previous_inputs[0]
    actual, reference = bind_inputs(evidence, tensors)
    rows, columns, accounts = [], {}, {}
    for row in retained["rows"]:
        hotspots = []
        for old in row["hotspots"]:
            selections = []
            for selected in old["selected_stage16_coordinates"]:
                coordinate = selected["selected_down_input"]["coordinate"]
                projections = {}
                for kind in gate_up.STAGES:
                    if (kind, coordinate) not in columns:
                        columns[kind, coordinate] = gate_up.weight_column(tensors[kind], coordinate)
                    key = row["control"], kind, coordinate
                    if key not in accounts:
                        accounts[key] = account(actual[row["control"]], reference,
                                                columns[kind, coordinate], kind, selected)
                    projections[kind] = accounts[key]
                selections.append({**selected, **projections})
            hotspots.append({**old, "selected_stage16_coordinates": selections})
        rows.append({**row, "hotspots": hotspots})
    count = sum(len(h["selected_stage16_coordinates"]) for row in rows for h in row["hotspots"])
    same(count, retained["bridge_account_count"], "retained stage16 account count changed")
    projections = count*len(gate_up.STAGES)
    canonical = evidence["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
    return {
        "left_id": hotspot.PAIR[0], "right_id": hotspot.PAIR[1], "layer": 23, "position": 0,
        "row_count": len(rows), "hotspot_count": retained["projection_account_count"],
        "selected_stage16_coordinate_count": count, "projection_account_count": projections,
        "input_coordinate_term_count": projections*gate_up.WIDTH,
        "selected_input_coordinate_count": projections*gate_up.LIMIT,
        "input_group_count": projections*gate_up.GROUPS,
        "input_coordinate_order_rule": "descending absolute contribution, then ascending input coordinate",
        "rows": rows, "retained_stage16_bridge": retained,
        "gate_up_projection_tensor_bindings": {
            name: canonical[name] for kind in gate_up.STAGES
            for name in gate_up.tensor_names(canonical, kind)},
        "common_component": retained["common_component"],
        "stop_nested_bridge_expansion": retained["stop_nested_bridge_expansion"],
        "prior_pair_and_common_gates_unchanged": True,
        "internal_reference": "original_input_L23_fp16",
        "binary64_internal_stages": "NOT_RETAINED_NO_RECONSTRUCTION",
    }


def validate_report(inputs, candidate):
    same(candidate, report(*inputs), "stage13 input/selection/closure/gate report binding changed")


def collect(audit):
    previous_inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*previous_inputs)
        tensors = load_tensors(previous_inputs[0])
    return previous_inputs, tensors, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("mlp_stage17_stage13_bridge_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "stage13 bridge tests failed, errored or skipped")
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
    origins = {
        **parent.source_context(), MODULE: parent.record(SOURCE),
        "mlp_stage17_stage13_bridge_tests": parent.record(TEST),
        **{module.MODULE: parent.record(module.SOURCE)
           for module in (previous, previous.previous, hotspot, down, previous.silu, gate_up)},
    }
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    evidence = inputs[0][0]
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "stage13 bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_GATE_UP_PROJECTION_INPUT_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "L23_reference_authority": evidence["result"]["preflight"]["L23_original_reference"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "retained_thresholds": evidence["result"]["preflight"]["thresholds"],
        "L23_archives_verified": 10, "projection_tensors_verified": 6,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": {"compiled": compiled, **tests},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "authenticated retained middle-pair stage16 bridge and original L23 stage13/14/15 operands",
        "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
