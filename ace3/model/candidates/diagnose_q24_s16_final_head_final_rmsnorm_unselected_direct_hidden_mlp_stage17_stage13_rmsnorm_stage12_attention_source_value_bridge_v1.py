"""Stdout-only exact P0 source/value accounts for retained stage11 attention deltas."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_residual_source_bridge_v1 as previous
from ace3.model.candidates import diagnose_q24_s16_final_head_attention_source_value_unselected_component_complement_v1 as complement


attention = complement.attribution
base, parent, bridge = previous.base, previous.parent, previous.bridge
rmsnorm = previous.previous
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_source_value_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
FIELDS = previous.FIELDS
PARTS = (*attention.PARTS, "signed_contribution")
TERMS = (*attention.PARTS, "av_boundary_remainder", "o_projection_boundary_remainder")
GROUPS = ("selected", "unselected", "full")
FLAGS = {**previous.FLAGS, **attention.FLAGS,
         "stage11_attention_source_value_causal_allocation": False}
BOUNDARY = (
    "Only retained attention_stage11_delta terms for pair (34319,13) are "
    "expanded. Original-input FP16 L23 P0 source position 0/token 9707, empty "
    "prior KV, unit singleton probabilities, 14 query heads and two 64-channel "
    "KV heads are authenticated. Exact probability/value/interaction accounts "
    "over all 128 GQA value components and separate AV/o_projection boundary "
    "remainders close every parent term through its unchanged norm/scalar/"
    "gate/up/down/row/opposite-reference multipliers. Shared local accounts "
    "are referenced by control and output coordinate; every local component "
    "and remainder is multiplied by the disclosed per-field multiplier. "
    "Eight descending-absolute local value components, with ascending-coordinate "
    "tie breaks, and their 120-component complement have separate signed/"
    "absolute/cancellation totals after GQA aggregation, not absolute query "
    "products. Logical repeated accounts are not independent samples. No "
    "binary64 internal-attention reconstruction, operator replay, evidence "
    "writes, causal allocation or admission follows. Other parent components "
    "and the common UNKNOWN stop gate remain unchanged. " + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("stage11 source/value bridge forbids prior execution, replay and writes")

    with previous.read_only(audit), ExitStack() as stack:
        for module, names in (
            (previous, ("check", "run_tests", "collect")),
            (attention, ("check", "focused_tests", "measure", "report")),
            (complement, ("check", "focused_tests", "measure", "report")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def evidence(inputs):
    return inputs[0][0][0][0]


def bind_authority(e):
    same(e["result"], json.loads(base.read_bound(base.PINS["result"])),
         "retained source/token/reference/KV/tensor/threshold authority changed")


def bind_attention(inputs, tensors):
    e = evidence(inputs)
    bind_authority(e)
    rmsnorm.bind_inputs(e, inputs[0][1])
    actual, reference, expected, _ = attention.load_attention(e["result"], e["assets"])
    same(sorted(tensors), sorted(expected), "o_proj tensor census changed")
    for name, value in expected.items():
        candidate = tensors[name]
        require(isinstance(candidate, np.ndarray) and candidate.dtype == value.dtype
                and candidate.shape == value.shape
                and candidate.tobytes() == value.tobytes(),
                "canonical o_proj tensor operand changed")
    for control in actual:
        require(attention.attention_operands(e["archives"][control], actual=True)
                == actual[control], "actual attention operand binding changed")
    require(attention.attention_operands(e["reference_archive"], actual=False) == reference,
            "original-input FP16 attention operand binding changed")
    return actual, reference


def scaled_mass(total, factor):
    signed = Fraction(total["signed"])*factor
    absolute = Fraction(total["absolute"])*abs(factor)
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


def merged_mass(totals):
    totals = list(totals)
    signed = sum((Fraction(t["signed"]) for t in totals), Fraction())
    absolute = sum((Fraction(t["absolute"]) for t in totals), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


def local_account(actual, reference, weights, control, coordinate):
    result = attention.account(actual, reference, weights, coordinate)
    partition = complement.split_unselected(result)
    ranking = result["full_value_component_ranking"]
    sets = (ranking[:attention.LIMIT], ranking[attention.LIMIT:], ranking)
    return {
        **result, "control": control, "output_coordinate": coordinate,
        "value_component_partition": partition,
        "value_component_totals": {
            group: {part: bridge.mass(Fraction(e[part]) for e in entries) for part in PARTS}
            for group, entries in zip(GROUPS, sets, strict=True)},
    }


def account(local, source):
    coordinate = source["retained_ranked_input"]["coordinate"]
    same(local["output_coordinate"], coordinate, "attention output/ranked input splice")
    retained = source["direct_stage12_residual_source"]
    same([t["term"] for t in retained["terms"]], list(previous.TERMS),
         "parent residual term selection changed")
    target = retained["terms"][1]
    norm = Fraction(retained["norm_weight"])
    anchor = Fraction(retained["original_input_fp16_scalar_anchor"])
    weight = Fraction(retained["gate_up_weight"])
    factor = Fraction(retained["coordinate_linear_operand_factor"])
    multipliers = (Fraction(1), norm*anchor, norm*anchor*weight, norm*anchor*weight*factor)
    values = [Fraction(local["source_tokens"][0][name]) for name in attention.PARTS]
    values.extend(Fraction(local[name]) for name in TERMS[3:])
    same(local["retained_o_projection_delta"],
         str(Fraction(retained["actual_operands"]["attention_stage11"])
             - Fraction(retained["original_input_fp16_operands"]["attention_stage11"])),
         "parent attention operand delta changed")
    terms = [
        {"term": name, **{field: str(value*m) for field, m in zip(FIELDS, multipliers, strict=True)}}
        for name, value in zip(TERMS, values, strict=True)]
    for field in FIELDS:
        same(str(sum((Fraction(t[field]) for t in terms), Fraction())), target[field],
             "parent attention contribution does not close: " + field)
    return {
        "local_account_key": f"{local['control']}:{coordinate}",
        "retained_parent_attention_stage11_delta": target,
        "multipliers": {field: str(m) for field, m in zip(FIELDS, multipliers, strict=True)},
        "terms": terms,
        "totals": {field: bridge.mass(Fraction(t[field]) for t in terms) for field in FIELDS},
        "value_component_totals": {
            group: {field: {part: scaled_mass(local["value_component_totals"][group][part], m)
                            for part in PARTS}
                    for field, m in zip(FIELDS, multipliers, strict=True)}
            for group in GROUPS},
        "closure_residuals": {field: "0" for field in FIELDS},
    }


def summary(accounts):
    return {
        "account_count": len(accounts), "value_component_count": len(accounts)*128,
        "selected_value_component_count": len(accounts)*8,
        "unselected_value_component_count": len(accounts)*120,
        "query_value_product_count": len(accounts)*896,
        "totals": {field: bridge.mass(Fraction(t[field]) for a in accounts for t in a["terms"])
                   for field in FIELDS},
        "term_totals": {
            name: {field: bridge.mass(Fraction(a["terms"][i][field]) for a in accounts)
                   for field in FIELDS}
            for i, name in enumerate(TERMS)},
        "value_component_totals": {
            group: {field: {part: merged_mass(
                a["value_component_totals"][group][field][part] for a in accounts)
                for part in PARTS} for field in FIELDS} for group in GROUPS},
        "component_and_boundary_totals": {
            field: merged_mass(
                total for a in accounts for total in (
                    a["value_component_totals"]["full"][field]["signed_contribution"],
                    bridge.mass(Fraction(t[field]) for t in a["terms"][3:])))
            for field in FIELDS},
    }


def report(inputs, retained, tensors):
    previous.validate_report(inputs, retained)
    same(retained["internal_reference"], "original_input_L23_fp16",
         "internal attention reference changed")
    same(retained["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
         "binary64 internal-attention reconstruction forbidden")
    actual, reference = bind_attention(inputs, tensors)
    rows, locals_by_key, columns, all_accounts = [], {}, {}, []
    for row in retained["rows"]:
        hotspots, row_accounts = [], []
        for old in row["hotspots"]:
            selections = []
            for selected in old["selected_stage16_coordinates"]:
                projections = {}
                for kind in ("gate_proj", "up_proj"):
                    projection = selected[kind]
                    sources, accounts = [], []
                    for source in projection["ranked_stage13_rmsnorm_sources"]:
                        coordinate = source["retained_ranked_input"]["coordinate"]
                        key = f"{row['control']}:{coordinate}"
                        if key not in locals_by_key:
                            if coordinate not in columns:
                                columns[coordinate] = attention.weight_column(tensors, coordinate)
                            locals_by_key[key] = local_account(
                                actual[row["control"]], reference, columns[coordinate],
                                row["control"], coordinate)
                        result = account(locals_by_key[key], source)
                        accounts.append(result)
                        sources.append({**source, "attention_stage11_source_value": result})
                    projections[kind] = {
                        **projection, "ranked_stage13_rmsnorm_sources": sources,
                        "attention_stage11_source_value_summary": summary(accounts)}
                    row_accounts.extend(accounts)
                selections.append({**selected, **projections})
            hotspots.append({**old, "selected_stage16_coordinates": selections})
        rows.append({**row, "hotspots": hotspots,
                     "attention_stage11_source_value_summary": summary(row_accounts)})
        all_accounts.extend(row_accounts)
    same(len(all_accounts), retained["stage12_residual_source_summary"]["account_count"],
         "retained attention term census changed")
    return {
        **retained, "rows": rows, "attention_stage11_local_accounts": locals_by_key,
        "attention_stage11_source_value_summary": summary(all_accounts),
        "attention_stage11_source_value_identity":
            "delta_att=sum_value_components(delta_P*V_ref*W+P_ref*delta_V*W"
            "+delta_P*delta_V*W)+AV_boundary+o_projection_boundary",
        "attention_stage11_source_value_weighting":
            "each referenced local component and boundary * per-field multiplier; "
            "w_norm*s_ref*w_gate_or_up*w_down*row_factor*opposite_reference_operand",
    }


def validate_report(inputs, candidate):
    same(candidate, report(*inputs), "attention selection/operand/closure/gate report changed")


def collect(audit):
    inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*inputs)
        e = evidence(inputs)
        bind_authority(e)
        _, _, tensors, _ = attention.load_attention(e["result"], e["assets"])
    return inputs, retained, tensors


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("stage11_attention_source_value_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "stage11 source/value tests failed, errored or skipped")
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
    modules = (
        previous, rmsnorm, rmsnorm.previous, rmsnorm.previous.previous,
        rmsnorm.previous.previous.previous, rmsnorm.previous.hotspot,
        rmsnorm.previous.down, rmsnorm.previous.previous.silu,
        rmsnorm.gate_up, rmsnorm.hidden, attention, complement,
    )
    origins = {
        **parent.source_context(), MODULE: parent.record(SOURCE),
        "stage11_attention_source_value_tests": parent.record(TEST),
        **{module.MODULE: parent.record(module.SOURCE) for module in modules},
    }
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    e = evidence(inputs[0])
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *base.PINS.values(), *bridge.margin.rows.PINS.values(),
                    *e["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "attention source/value attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_STAGE11_ATTENTION_SOURCE_VALUE_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": e["files"], "hidden_pins": e["hidden_pins"],
        "assets": e["assets"],
        "L23_reference_authority": e["result"]["preflight"]["L23_original_reference"],
        "final_reference_authority": e["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": e["result"]["controls"],
        "retained_thresholds": e["result"]["preflight"]["thresholds"],
        "L23_archives_verified": 10, "projection_tensors_verified": 9,
        "post_attention_rmsnorm_tensors_verified": 1,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": {"compiled": compiled, **tests},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "input_mode": "authenticated parent attention_stage11_delta and original L23 P0 FP16 attention",
        "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
