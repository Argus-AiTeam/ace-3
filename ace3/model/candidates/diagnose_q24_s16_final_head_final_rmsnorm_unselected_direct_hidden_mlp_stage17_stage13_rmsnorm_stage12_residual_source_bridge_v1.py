"""Stdout-only retained stage12 residual sources for ranked direct RMSNorm terms."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_source_bridge_v1 as previous


base, parent, bridge = previous.base, previous.parent, previous.bridge
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_residual_source_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 14
TERMS = (
    "input_hidden_delta", "attention_stage11_delta", "actual_input_q24_carry",
    "actual_attention_state_update", "actual_stage12_fp16_conversion",
    "negative_original_input_fp16_attention_residual_boundary",
)
FIELDS = (
    "stage12_delta_contribution", "stage13_direct_input_contribution",
    "projection_contribution", "coordinate_linear_operand_contribution",
)
FLAGS = {**previous.FLAGS, "stage12_residual_replay": 0,
         "stage12_residual_source_causal_allocation": False}
BOUNDARY = (
    "Only every retained ranked direct_stage12_input term for pair (34319,13) "
    "is expanded, with no reselection. Six exact retained-operand accounts "
    "separate input_hidden delta, attention_stage11 delta, actual input-Q24 "
    "carry, actual attention-state update, actual stage12 FP16 conversion and "
    "the negative original-input FP16 attention residual boundary. The actual "
    "and original boundaries are disclosed separately, not solely as rounding "
    "and not as identified causes. Multiplication through the unchanged "
    "post_attention RMSNorm weight and original scalar anchor, native gate/up "
    "weight, down weight, row factor and opposite reference operand closes "
    "each parent's direct-input contribution exactly. Global scalar-scale, "
    "interaction and stage13 output remainders, unranked inputs and all later "
    "boundaries remain unchanged. Signed/absolute/cancellation totals describe "
    "logical accounts, not independent samples or causal allocation. No "
    "attention, RMSNorm, native, decoder, prefix, admission, reference, head, "
    "row-dot or local operator is replayed and no evidence is written. "
    + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("stage12 residual source bridge forbids prior execution, replay and writes")

    with previous.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "collect"):
            stack.enter_context(patch.object(previous, name, forbidden))
        yield


def account(actual, reference, source, norm_weight, reference_anchor, weight, factor):
    i = source["retained_ranked_input"]["coordinate"]
    require(type(i) is int and 0 <= i < previous.gate_up.WIDTH
            and all(isinstance(v, Fraction)
                    for v in (norm_weight, reference_anchor, weight, factor)),
            "invalid retained stage12 coordinate or exact multiplier")
    a, r = actual["stage12"][i], reference["stage12"][i]
    ha, hr = actual["input_hidden"][i], reference["input_hidden"][i]
    aa, ar = actual["stage11"][i], reference["stage11"][i]
    iq, sq = actual["input_q24"][i], actual["scratch_q24"][i]
    carry, update, conversion = iq-ha, sq-iq-aa, a-sq
    reference_boundary = r-hr-ar
    values = (ha-hr, aa-ar, carry, update, conversion, -reference_boundary)
    direct = [v*norm_weight*reference_anchor for v in values]
    projected = [v*weight for v in direct]
    weighted = [v*factor for v in projected]
    same([term["term"] for term in source["terms"]], list(previous.TERMS),
         "parent RMSNorm source term order changed")
    retained = source["terms"][0]
    for field, expected in (
        ("actual_stage12", a), ("original_input_fp16_stage12", r),
        ("stage12_delta", a-r), ("norm_weight", norm_weight),
        ("actual_raw_scratch_q24", sq), ("actual_q24_to_stage12_conversion", conversion),
        ("gate_up_weight", weight), ("coordinate_linear_operand_factor", factor),
    ):
        same(source[field], str(expected), "parent stage12 operand or multiplier changed: " + field)
    same(retained["stage13_delta_contribution"], str(sum(direct, Fraction())),
         "parent direct stage13 input contribution changed")
    same(retained["projection_contribution"], str(sum(projected, Fraction())),
         "parent direct projection contribution changed")
    same(retained["coordinate_linear_operand_contribution"], str(sum(weighted, Fraction())),
         "parent direct coordinate contribution changed")
    require(sum(values, Fraction()) == a-r
            and carry+update+conversion == a-ha-aa,
            "stage12 residual source identity failed")
    return {
        "retained_parent_direct_stage12_input": retained,
        "actual_operands": {
            "input_hidden": str(ha), "attention_stage11": str(aa),
            "input_q24": str(iq), "scratch_q24": str(sq), "stage12_fp16": str(a)},
        "original_input_fp16_operands": {
            "input_hidden": str(hr), "attention_stage11": str(ar), "stage12_fp16": str(r)},
        "actual_attention_residual_boundary": str(a-ha-aa),
        "actual_input_q24_carry": str(carry),
        "actual_attention_state_update": str(update),
        "actual_stage12_fp16_conversion": str(conversion),
        "original_input_fp16_attention_residual_boundary": str(reference_boundary),
        "actual_minus_original_attention_residual_boundary": str(a-ha-aa-reference_boundary),
        "norm_weight": str(norm_weight), "original_input_fp16_scalar_anchor": str(reference_anchor),
        "gate_up_weight": str(weight), "coordinate_linear_operand_factor": str(factor),
        "terms": [
            {"term": name, **{field: str(value) for field, value in zip(FIELDS, row, strict=True)}}
            for name, row in zip(TERMS, zip(values, direct, projected, weighted, strict=True), strict=True)],
        "totals": {field: bridge.mass(v) for field, v in
                   zip(FIELDS, (values, direct, projected, weighted), strict=True)},
        "stage12_closure_residual": "0", "actual_boundary_closure_residual": "0",
        "parent_direct_stage13_closure_residual": "0",
        "parent_direct_projection_closure_residual": "0",
        "parent_direct_coordinate_closure_residual": "0",
    }


def summary(accounts):
    return {
        "account_count": len(accounts), "component_count": len(accounts)*len(TERMS),
        "totals": {field: bridge.mass(Fraction(t[field]) for a in accounts for t in a["terms"])
                   for field in FIELDS},
        "component_totals": {
            name: {field: bridge.mass(Fraction(a["terms"][i][field]) for a in accounts)
                   for field in FIELDS}
            for i, name in enumerate(TERMS)},
    }


def report(previous_inputs, retained):
    previous.validate_report(previous_inputs, retained)
    evidence = previous_inputs[0][0][0]
    actual, reference, norm = previous.bind_inputs(evidence, previous_inputs[1])
    sr = Fraction(previous_inputs[2]["original_input_fp16"]["inverse_norm_anchor"])
    rows, columns, all_accounts = [], {}, []
    for row in retained["rows"]:
        hotspots, row_accounts = [], []
        for old in row["hotspots"]:
            row_factor = Fraction(old["retained_weighted_row_factor"])
            selections = []
            for selected in old["selected_stage16_coordinates"]:
                output = selected["selected_down_input"]["coordinate"]
                down_weight = Fraction(selected["selected_down_input"]["weight"])
                projections = {}
                for kind, opposite in (("gate_proj", "up"), ("up_proj", "gate")):
                    if (kind, output) not in columns:
                        columns[kind, output] = previous.gate_up.weight_column(
                            previous_inputs[0][1][kind], output)
                    factor = row_factor*down_weight*Fraction(
                        selected["bridge"]["reference_operands"][opposite])
                    projection = selected[kind]
                    sources, accounts = [], []
                    for source in projection["ranked_stage13_rmsnorm_sources"]:
                        i = source["retained_ranked_input"]["coordinate"]
                        result = account(actual[row["control"]], reference, source,
                                         norm[i], sr, columns[kind, output][i], factor)
                        accounts.append(result)
                        sources.append({**source, "direct_stage12_residual_source": result})
                    projections[kind] = {
                        **projection, "ranked_stage13_rmsnorm_sources": sources,
                        "stage12_residual_source_summary": summary(accounts)}
                    row_accounts.extend(accounts)
                selections.append({**selected, **projections})
            hotspots.append({**old, "selected_stage16_coordinates": selections})
        rows.append({**row, "hotspots": hotspots,
                     "stage12_residual_source_summary": summary(row_accounts)})
        all_accounts.extend(row_accounts)
    same(len(all_accounts), retained["ranked_stage13_source_account_count"],
         "retained direct stage12 source census changed")
    return {
        **retained, "rows": rows,
        "stage12_residual_source_summary": summary(all_accounts),
        "stage12_residual_source_identity":
            "delta_x=(h_a-h_r)+(att_a-att_r)+(q_a-h_a)+(s_a-q_a-att_a)"
            "+(x_a-s_a)-(x_r-h_r-att_r)",
        "stage12_residual_source_weighting":
            "component*w_norm*s_ref*w_gate_or_up*w_down*row_factor*opposite_reference_operand",
        "stage12_residual_reference":
            "original_input_L23_fp16_for_both_final_reference_branches",
    }


def validate_report(inputs, candidate):
    same(candidate, report(*inputs), "stage12 residual source selection/operand/closure/gate report changed")


def collect(audit):
    previous_inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*previous_inputs)
    return previous_inputs, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("mlp_stage17_stage12_residual_source_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "stage12 residual source tests failed, errored or skipped")
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
        "mlp_stage17_stage12_residual_source_tests": parent.record(TEST),
        **{module.MODULE: parent.record(module.SOURCE) for module in (
            previous, previous.previous, previous.previous.previous,
            previous.previous.previous.previous, previous.previous.hotspot,
            previous.previous.down, previous.previous.previous.silu,
            previous.gate_up, previous.hidden)},
    }
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    evidence = inputs[0][0][0][0]
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "stage12 residual source attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_STAGE12_RESIDUAL_SOURCE_BRIDGE",
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
        "post_attention_rmsnorm_tensors_verified": 1,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": {"compiled": compiled, **tests},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "authenticated retained direct_stage12_input terms and original L23 residual operands",
        "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
