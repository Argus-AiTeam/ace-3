"""Stdout-only retained stage12 RMSNorm source accounts for ranked stage13 terms."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_gate_up_projection_input_bridge_v1 as previous


base, parent = previous.base, previous.parent
bridge, gate_up, residual = previous.bridge, previous.gate_up, previous.down.residual
hidden = bridge.hidden
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_source_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
NORM = "model.layers.23.post_attention_layernorm.weight"
TERMS = ("direct_stage12_input", "global_scalar_scale", "interaction",
         "output_boundary_remainder_delta")
FLAGS = {**previous.FLAGS, "stage13_rmsnorm_replay": 0,
         "binary64_internal_stage_reconstruction": False,
         "rmsnorm_source_causal_allocation": False}
BOUNDARY = (
    "Only the parent's retained ranked stage13 terms are expanded, without "
    "reselection. The inputs are retained stage12 FP16 operator-boundary words, "
    "not raw Q24 state fed through a reconstructed RMSNorm. Raw scratch Q24 "
    "and its stage12 conversion difference are disclosed separately. Scalar "
    "anchors use s=1/math.sqrt(float(exact_mean_square+1/1000000)); they are "
    "binary64 accounting anchors, not replayed RMSNorm outputs. Each retained "
    "stage13 delta closes through direct stage12 input, global scalar-scale, "
    "interaction and actual-minus-reference output/boundary remainders. These "
    "remainders combine implementation, epsilon, normalization, multiplication, "
    "conversion and scalar-anchor effects; they are not solely rounding. "
    "Unchanged native gate/up weights restore each parent ranked contribution. "
    "Additional row-weighted accounts use the retained down weight, row factor "
    "and opposite reference gate/up operand, only for the parent's linear "
    "operand term. They do not allocate its gate/up interaction or joint "
    "SiLU/product boundary, or establish causality. Signed/absolute/cancellation "
    "totals are logical accounts, not independent samples. Both final-reference "
    "branches use the original-input FP16 internal trajectory, with no binary64 "
    "internal reconstruction. The common UNKNOWN stop gate is unchanged. "
    + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("RMSNorm source bridge forbids prior execution, replay and writes")

    with previous.read_only(audit), ExitStack() as stack:
        for name in ("check", "run_tests", "collect"):
            stack.enter_context(patch.object(previous, name, forbidden))
        yield


def bind_weight(array, pin):
    require(pin["name"] == NORM, "post-attention RMSNorm tensor name changed")
    return hidden.validate_weight(
        array, {key: pin[key] for key in ("shape", "dtype", "sha256")})


def bind_inputs(evidence, weight):
    binding = evidence["result"]["preflight"]["L23_original_reference"]["reference"]
    weights = bind_weight(weight, binding["canonical"][NORM])
    same(list(evidence["archives"]), list(previous.hotspot.census.CONTROLS),
         "RMSNorm source control order changed")
    sources = [(None, evidence["reference_archive"], binding["fp16"])]
    sources.extend((row["control"], evidence["archives"][row["control"]],
                    row["parent"]["terminal_archive"])
                   for row in evidence["result"]["controls"])
    actual, reference = {}, None
    for control, archive, pin in sources:
        authenticated = residual.read_archive(pin)
        same(sorted(archive), sorted(authenticated), "RMSNorm source archive census changed")
        for key, expected in authenticated.items():
            value = archive[key]
            require(isinstance(value, np.ndarray) and value.dtype == expected.dtype
                    and value.shape == expected.shape and np.array_equal(value, expected),
                    "retained stage12/stage13/state/KV archive binding changed")
        values = residual.operands(archive, actual=control is not None)
        values["stage13"] = residual.words(archive["stage13"])
        if control is None:
            reference = values
        else:
            actual[control] = values
    same(list(actual), list(previous.hotspot.census.CONTROLS),
         "RMSNorm source operand census changed")
    require(reference is not None, "original L23 RMSNorm operands missing")
    return actual, reference, weights


def scalar_anchors(actual, reference):
    require(hidden.EPSILON == Fraction(1, 1000000), "scalar epsilon contract changed")
    same(parent.norm.EPSILON_Q48, 281474977, "native RMSNorm epsilon changed")
    return {
        "original_input_fp16": hidden.norm_summary(reference["stage12"]),
        "actual": {control: hidden.norm_summary(values["stage12"])
                   for control, values in actual.items()},
    }


def bind_anchors(actual, reference, anchors):
    same(anchors, scalar_anchors(actual, reference),
         "stage12 scalar anchor/energy/defect binding changed")


def account(actual, reference, weights, anchors, control, pin, weight, factor):
    i = pin["coordinate"]
    require(type(i) is int and 0 <= i < gate_up.WIDTH
            and isinstance(weight, Fraction) and isinstance(factor, Fraction),
            "invalid ranked stage13 coordinate or exact factor")
    a, r = actual["stage12"][i], reference["stage12"][i]
    ya, yr = actual["stage13"][i], reference["stage13"][i]
    sa = Fraction(anchors["actual"][control]["inverse_norm_anchor"])
    sr = Fraction(anchors["original_input_fp16"]["inverse_norm_anchor"])
    w, dh, ds = weights[i], a-r, sa-sr
    ba, br = ya-w*a*sa, yr-w*r*sr
    values = (w*dh*sr, w*r*ds, w*dh*ds, ba-br)
    projected = [value*weight for value in values]
    weighted = [value*factor for value in projected]
    same(pin["input_delta"], str(ya-yr), "parent ranked stage13 delta changed")
    same(pin["weight"], str(weight), "parent ranked gate/up weight changed")
    same(pin["signed_contribution"], str((ya-yr)*weight),
         "parent ranked stage13 contribution changed")
    same(pin["absolute_contribution"], str(abs((ya-yr)*weight)),
         "parent ranked stage13 absolute contribution changed")
    require(sum(values, Fraction()) == ya-yr
            and sum(projected, Fraction()) == Fraction(pin["signed_contribution"])
            and sum(weighted, Fraction()) == Fraction(pin["signed_contribution"])*factor,
            "ranked stage13 RMSNorm source identity failed")
    return {
        "retained_ranked_input": pin,
        "actual_stage12": str(a), "original_input_fp16_stage12": str(r),
        "stage12_delta": str(dh), "norm_weight": str(w),
        "actual_raw_scratch_q24": str(actual["scratch_q24"][i]),
        "actual_q24_to_stage12_conversion": str(a-actual["scratch_q24"][i]),
        "actual_stage13": str(ya), "original_input_fp16_stage13": str(yr),
        "retained_stage13_delta": str(ya-yr), "scalar_anchor_delta": str(ds),
        "actual_output_boundary_remainder": str(ba),
        "original_output_boundary_remainder": str(br),
        "gate_up_weight": str(weight), "coordinate_linear_operand_factor": str(factor),
        "terms": [
            {"term": name, "stage13_delta_contribution": str(value),
             "projection_contribution": str(p), "coordinate_linear_operand_contribution": str(c)}
            for name, value, p, c in zip(TERMS, values, projected, weighted, strict=True)],
        "stage13_totals": bridge.mass(values),
        "projection_totals": bridge.mass(projected),
        "coordinate_linear_operand_totals": bridge.mass(weighted),
        "stage13_closure_residual": "0", "parent_ranked_term_closure_residual": "0",
        "coordinate_linear_operand_closure_residual": "0",
    }


def projection_summary(accounts, projection, factor, retained_linear):
    projected = [Fraction(t["projection_contribution"]) for a in accounts for t in a["terms"]]
    weighted = [Fraction(t["coordinate_linear_operand_contribution"])
                for a in accounts for t in a["terms"]]
    ranked = Fraction(projection["ranked_input_totals"]["signed"])
    unranked = Fraction(projection["unranked_input_totals"]["signed"])
    boundary = Fraction(projection["projection_boundary_remainder"])
    delta = Fraction(projection["retained_projection_delta"])
    require(sum(projected, Fraction()) == ranked
            and ranked+unranked+boundary == delta
            and (sum(projected, Fraction())+unranked+boundary)*factor == retained_linear,
            "ranked/unranked/boundary to retained linear operand closure failed")
    return {
        "ranked_projection_totals": bridge.mass(projected),
        "ranked_coordinate_linear_operand_totals": bridge.mass(weighted),
        "component_projection_totals": {
            term: bridge.mass([Fraction(a["terms"][i]["projection_contribution"]) for a in accounts])
            for i, term in enumerate(TERMS)},
        "component_coordinate_linear_operand_totals": {
            term: bridge.mass([Fraction(a["terms"][i]["coordinate_linear_operand_contribution"])
                               for a in accounts])
            for i, term in enumerate(TERMS)},
        "retained_ranked_input_totals": projection["ranked_input_totals"],
        "retained_unranked_input_totals": projection["unranked_input_totals"],
        "retained_projection_boundary_remainder": str(boundary),
        "coordinate_linear_operand_factor": str(factor),
        "weighted_unranked_remainder": str(unranked*factor),
        "weighted_projection_boundary_remainder": str(boundary*factor),
        "retained_coordinate_linear_operand_term": str(retained_linear),
        "expanded_projection_and_boundary_totals": bridge.mass([*projected, unranked, boundary]),
        "expanded_coordinate_linear_operand_totals": bridge.mass(
            [*weighted, unranked*factor, boundary*factor]),
        "projection_closure_residual": "0", "retained_linear_operand_closure_residual": "0",
    }


def report(previous_inputs, norm_weight, anchors, retained):
    previous.validate_report(previous_inputs, retained)
    evidence = previous_inputs[0][0]
    actual, reference, weights = bind_inputs(evidence, norm_weight)
    bind_anchors(actual, reference, anchors)
    rows, columns, count = [], {}, 0
    for row in retained["rows"]:
        hotspots = []
        for old in row["hotspots"]:
            row_factor = Fraction(old["retained_weighted_row_factor"])
            selections = []
            for selected in old["selected_stage16_coordinates"]:
                output = selected["selected_down_input"]["coordinate"]
                down_weight = Fraction(selected["selected_down_input"]["weight"])
                projections = {}
                for kind, opposite, term_index in (("gate_proj", "up", 0), ("up_proj", "gate", 1)):
                    if (kind, output) not in columns:
                        columns[kind, output] = gate_up.weight_column(previous_inputs[1][kind], output)
                    reference_operand = Fraction(selected["bridge"]["reference_operands"][opposite])
                    factor = row_factor*down_weight*reference_operand
                    projection = selected[kind]
                    accounts = [
                        {"retained_input_rank": rank, **account(
                            actual[row["control"]], reference, weights, anchors, row["control"],
                            pin, columns[kind, output][pin["coordinate"]], factor)}
                        for rank, pin in enumerate(projection["largest_absolute_coordinates"], 1)]
                    count += len(accounts)
                    linear = selected["coordinate_weighted_terms_in_operand_order"][term_index]
                    projections[kind] = {
                        **projection, "ranked_stage13_rmsnorm_sources": accounts,
                        "rmsnorm_source_factor_bindings": {
                            "retained_weighted_row_factor": str(row_factor),
                            "down_weight": str(down_weight), "opposite_reference_operand": opposite,
                            "opposite_reference_operand_value": str(reference_operand),
                            "retained_linear_operand_term": linear["term"]},
                        "rmsnorm_source_summary": projection_summary(
                            accounts, projection, factor, Fraction(linear["signed_contribution"])),
                    }
                selections.append({**selected, **projections})
            hotspots.append({**old, "selected_stage16_coordinates": selections})
        rows.append({**row, "hotspots": hotspots})
    same(count, retained["selected_input_coordinate_count"], "ranked source account census changed")
    return {
        **retained, "rows": rows, "ranked_stage13_source_account_count": count,
        "rmsnorm_source_component_count": count*len(TERMS),
        "post_attention_rmsnorm_tensor_binding": evidence["result"]["preflight"]
            ["L23_original_reference"]["reference"]["canonical"][NORM],
        "stage12_scalar_anchors": anchors,
        "scalar_anchor": "s=1/math.sqrt(float(exact_mean_square+1/1000000)); binary64 accounting only",
        "native_epsilon_q48": parent.norm.EPSILON_Q48,
        "native_epsilon_minus_contract": str(Fraction(parent.norm.EPSILON_Q48, 1 << 48)-hidden.EPSILON),
        "rmsnorm_source_identity": "delta_y=w*delta_x*s_ref+w*x_ref*delta_s+w*delta_x*delta_s+b_a-b_r; b=y-w*x*s",
        "row_weighting_scope": "retained linear gate/up operand terms only; interaction and SiLU/product boundary unchanged",
    }


def validate_report(inputs, candidate):
    same(candidate, report(*inputs), "RMSNorm source selection/anchor/closure/gate report changed")


def collect(audit):
    previous_inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*previous_inputs)
        evidence = previous_inputs[0][0]
        with parent.preflight.parent.producer.legacy.safe_open(
                evidence["assets"]["checkpoint"]["path"], framework="numpy") as model:
            weight = model.get_tensor(NORM)
        actual, reference, _ = bind_inputs(evidence, weight)
        anchors = scalar_anchors(actual, reference)
    return previous_inputs, weight, anchors, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("mlp_stage17_rmsnorm_source_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "RMSNorm source tests failed, errored or skipped")
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
        "mlp_stage17_rmsnorm_source_tests": parent.record(TEST),
        **{module.MODULE: parent.record(module.SOURCE) for module in (
            previous, previous.previous, previous.previous.previous,
            previous.hotspot, previous.down, previous.previous.silu, gate_up, hidden)},
    }
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    evidence = inputs[0][0][0]
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "RMSNorm source attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_STAGE13_RMSNORM_SOURCE_BRIDGE",
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
        "input_mode": "authenticated retained ranked stage13 terms and original L23 stage12/13 operands",
        "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
