"""Stdout-only native AWQ stage00 accounts for selected retained P0 value terms."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_source_value_bridge_v1 as previous
from ace3.model.candidates import diagnose_q24_s16_final_head_attention_value_projection_input_coordinates_v1 as value


base, parent, bridge = previous.base, previous.parent, previous.bridge
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_projection_input_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
FIELDS = previous.FIELDS
GROUPS = previous.GROUPS
TERMS = ("exact_stage00_input_delta", "actual_projection_boundary",
         "negative_original_input_fp16_projection_boundary")
PARENT_PINS = (
    {"path": str(previous.SOURCE), "bytes": 16762,
     "sha256": "3869042e21ad0c81152fcbedbb3b38b1bae0a1fcc713e918c4cdf9b9e005ca32"},
    {"path": str(previous.TEST), "bytes": 23216,
     "sha256": "9a4ea6a85ba2c358c402c3271547b1396356ccf70a4fb5b63bb18a445af34e73"},
)
FLAGS = {**previous.FLAGS, **value.FLAGS,
         "stage11_value_projection_input_causal_allocation": False}
BOUNDARY = (
    "Only the eight already selected value components of each retained stage11 "
    "source/value account for pair (34319,13) are expanded, with no reselection. "
    "All 896 original-input FP16 stage00 delta-input times native G128 AWQ "
    "weights are accounted for, with eight ranked coordinates and their full "
    "complement, seven contiguous groups, exact actual/reference sums including "
    "shared bias, and separate actual and negative original projection boundaries. "
    "The boundaries are not solely rounding. Shared control/value-coordinate "
    "tables and disclosed GQA and downstream multipliers define every logical "
    "input-coordinate term, including unranked coordinates; repeated logical "
    "accounts are not independent samples. Original-input FP16 internal attention "
    "is used for both final-reference branches, never binary64 reconstruction. "
    "No predecessor checks, operators, prefix/admission replay or evidence writes. "
    + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("V input bridge forbids predecessor checks, replay and writes")

    with previous.read_only(audit), ExitStack() as stack:
        for module, names in (
            (previous, ("check", "run_tests", "collect")),
            (value, ("check", "focused_tests", "measure", "report")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def authenticate_parent():
    for pin in PARENT_PINS:
        base.read_bound(pin)


def load_values(inputs):
    actual, reference = previous.bind_attention(inputs[0], inputs[2])
    e = previous.evidence(inputs[0])
    return value.load_inputs(((e["result"],), actual, reference), e["assets"])


def bind_values(inputs, operands):
    expected = load_values(inputs)
    actual, reference, tensors = operands
    same(list(actual), list(expected[0]), "stage00 control census changed")
    for control in actual:
        require(actual[control] == expected[0][control], "actual stage00 operand changed")
    require(reference == expected[1], "original-input FP16 stage00 operand changed")
    same(sorted(tensors), sorted(expected[2]), "v_proj tensor census changed")
    e = previous.evidence(inputs[0])
    canonical = e["result"]["preflight"]["L23_original_reference"]["reference"]["canonical"]
    for suffix, tensor in tensors.items():
        value.bind_tensor(tensor, canonical[value.PROJECTION + suffix], suffix)
        require(tensor.tobytes() == expected[2][suffix].tobytes(), "v_proj tensor splice")


def projection_account(actual, reference, weights, actual_v, reference_v, bias, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < 128
            and all(len(v) == value.WIDTH and all(isinstance(x, Fraction) for x in v)
                    for v in (actual, reference, weights))
            and all(isinstance(v, Fraction) for v in (actual_v, reference_v, bias)),
            "invalid exact V input account")
    actual_sum = sum((x*w for x, w in zip(actual, weights, strict=True)), bias)
    reference_sum = sum((x*w for x, w in zip(reference, weights, strict=True)), bias)
    deltas = [a-r for a, r in zip(actual, reference, strict=True)]
    terms = [x*w for x, w in zip(deltas, weights, strict=True)]
    order = sorted(range(value.WIDTH), key=lambda i: (-abs(terms[i]), i))
    exact = sum(terms, Fraction())
    actual_boundary, reference_boundary = actual_v-actual_sum, reference_v-reference_sum
    require(exact == actual_sum-reference_sum
            and exact+actual_boundary-reference_boundary == actual_v-reference_v,
            "V stage00/projection identity failed")
    groups = [
        {"input_group": start//128, "start_coordinate": start,
         "end_coordinate_exclusive": start+128, "coordinate_count": 128,
         **bridge.mass(terms[start:start+128])}
        for start in range(0, value.WIDTH, 128)]
    groups.sort(key=lambda g: (-abs(Fraction(g["signed"])), g["input_group"]))
    return {
        "output_coordinate": coordinate, "input_stage": "stage00_fp16",
        "output_stage": "stage03_fp16", "source_position": 0, "source_token_id": 9707,
        "coordinate_count": value.WIDTH, "shared_fp16_bias": str(bias),
        "actual_v": str(actual_v), "original_input_fp16_v": str(reference_v),
        "actual_exact_input_weight_sum_with_bias": str(actual_sum),
        "original_exact_input_weight_sum_with_bias": str(reference_sum),
        "exact_stage00_input_delta": str(exact),
        "actual_projection_boundary": str(actual_boundary),
        "negative_original_input_fp16_projection_boundary": str(-reference_boundary),
        "retained_projection_delta": str(actual_v-reference_v),
        "projection_boundary_remainder": str(actual_boundary-reference_boundary),
        "largest_absolute_input_coordinates": [
            {"coordinate": i, "input_group": i//128, "actual_input": str(actual[i]),
             "original_input_fp16_input": str(reference[i]), "input_delta": str(deltas[i]),
             "weight": str(weights[i]), "signed_contribution": str(terms[i])}
            for i in order[:value.LIMIT]],
        "groups_by_absolute_signed_sum": groups,
        "input_coordinate_totals": {
            group: bridge.mass(terms[i] for i in indices)
            for group, indices in zip(GROUPS, (order[:8], order[8:], order), strict=True)},
        "closure_residual": "0",
    }


def component_account(core, component, factor, downstream, key, rank):
    require(isinstance(factor, Fraction), "invalid GQA value multiplier")
    coordinate = core["output_coordinate"]
    same(component["value_coordinate"], coordinate, "selected V coordinate splice")
    kv, dim = divmod(coordinate, 64)
    same((component["source_position"], component["source_token_id"], component["kv_head"],
          component["head_dimension"], component["query_heads"]),
         (0, 9707, kv, dim, list(range(kv*7, (kv+1)*7))), "V source/GQA binding changed")
    delta = Fraction(core["retained_projection_delta"])
    require(Fraction(component["probability"]) == Fraction(component["interaction"]) == 0
            and Fraction(component["value"]) == Fraction(component["signed_contribution"])
            == delta*factor, "selected parent value contribution changed")
    same(list(downstream), list(FIELDS), "downstream field census changed")
    multipliers = {field: factor*Fraction(downstream[field]) for field in FIELDS}
    terms = [{"term": term, **{field: str(Fraction(core[term])*multipliers[field])
                             for field in FIELDS}} for term in TERMS]
    targets = {field: str(Fraction(component["value"])*Fraction(downstream[field]))
               for field in FIELDS}
    for field in FIELDS:
        same(str(sum((Fraction(t[field]) for t in terms), Fraction())), targets[field],
             "selected value term does not close: " + field)
    return {
        "value_hotspot_rank": rank, "retained_parent_value_component": component,
        "projection_account_key": key, "value_component_factor": str(factor),
        "input_coordinate_multipliers": {f: str(m) for f, m in multipliers.items()},
        "retained_parent_contributions": targets, "terms": terms,
        "totals": {f: bridge.mass(Fraction(t[f]) for t in terms) for f in FIELDS},
        "input_coordinate_totals": {
            group: {f: previous.scaled_mass(core["input_coordinate_totals"][group], m)
                    for f, m in multipliers.items()} for group in GROUPS},
        "closure_residuals": {f: "0" for f in FIELDS},
    }


def summary(accounts):
    return {
        "selected_value_component_count": len(accounts),
        "input_coordinate_term_count": len(accounts)*value.WIDTH,
        "input_group_count": len(accounts)*7,
        "totals": {f: bridge.mass(Fraction(t[f]) for a in accounts for t in a["terms"])
                   for f in FIELDS},
        "term_totals": {
            term: {f: bridge.mass(Fraction(a["terms"][i][f]) for a in accounts) for f in FIELDS}
            for i, term in enumerate(TERMS)},
        "input_coordinate_totals": {
            group: {f: previous.merged_mass(a["input_coordinate_totals"][group][f]
                                           for a in accounts) for f in FIELDS}
            for group in GROUPS},
    }


def report(inputs, retained, operands):
    authenticate_parent()
    same(retained["internal_reference"], "original_input_L23_fp16", "internal reference changed")
    same(retained["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION",
         "binary64 internal attention reconstruction forbidden")
    previous.validate_report(inputs, retained)
    bind_values(inputs, operands)
    actual, reference, tensors = operands
    e = previous.evidence(inputs[0])
    cores, columns, output_columns, rows, all_accounts = {}, {}, {}, [], []
    for row in retained["rows"]:
        control = row["control"]
        hotspots, row_accounts = [], []
        for hotspot in row["hotspots"]:
            selections = []
            for selected in hotspot["selected_stage16_coordinates"]:
                projections = {}
                for kind in ("gate_proj", "up_proj"):
                    projection = selected[kind]
                    sources, accounts = [], []
                    for source in projection["ranked_stage13_rmsnorm_sources"]:
                        inherited = source["attention_stage11_source_value"]
                        local = retained["attention_stage11_local_accounts"][inherited["local_account_key"]]
                        output = local["output_coordinate"]
                        if output not in output_columns:
                            output_columns[output] = previous.attention.weight_column(inputs[2], output)
                        components = []
                        for rank, component in enumerate(local["largest_absolute_value_components"], 1):
                            v = component["value_coordinate"]
                            key = f"{control}:{v}"
                            if key not in cores:
                                if v not in columns:
                                    columns[v] = value.weight_column(tensors, v)
                                av = previous.attention.words(e["archives"][control]["stage03"], (128,))[v]
                                rv = previous.attention.words(e["reference_archive"]["stage03"], (128,))[v]
                                core = projection_account(actual[control], reference, columns[v], av, rv,
                                                          Fraction(float(tensors["bias"][v])), v)
                                cores[key] = {**core, "control": control}
                            kv, dim = divmod(v, 64)
                            factor = sum((output_columns[output][h*64+dim]
                                          for h in range(kv*7, (kv+1)*7)), Fraction())
                            result = component_account(cores[key], component, factor,
                                                       inherited["multipliers"], key, rank)
                            components.append(result)
                        accounts.extend(components)
                        sources.append({**source, "attention_value_projection_input": {
                            "local_account_key": inherited["local_account_key"],
                            "downstream_multipliers": inherited["multipliers"],
                            "selected_value_components": components}})
                    projections[kind] = {
                        **projection, "ranked_stage13_rmsnorm_sources": sources,
                        "attention_value_projection_input_summary": summary(accounts)}
                    row_accounts.extend(accounts)
                selections.append({**selected, **projections})
            hotspots.append({**hotspot, "selected_stage16_coordinates": selections})
        rows.append({**row, "hotspots": hotspots,
                     "attention_value_projection_input_summary": summary(row_accounts)})
        all_accounts.extend(row_accounts)
    same(len(all_accounts), retained["attention_stage11_source_value_summary"]
         ["selected_value_component_count"], "selected V component census changed")
    return {
        **retained, "rows": rows, "value_projection_local_accounts": cores,
        "attention_value_projection_input_summary": summary(all_accounts),
        "attention_value_projection_input_identity":
            "selected_value=GQA_factor*(sum_stage00(delta_input*native_weight)"
            "+actual_projection_boundary-original_projection_boundary)",
        "attention_value_projection_input_weighting":
            "each referenced stage00 coordinate and boundary * input_coordinate_multipliers[field]; "
            "GQA_factor=sum_original_P*o_proj_weight; downstream fields unchanged",
    }


def collect(audit):
    authenticate_parent()
    inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*inputs)
        operands = load_values(inputs)
    return inputs, retained, operands


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("stage11_value_projection_input_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "V projection input tests failed, errored or skipped")
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
               "stage11_value_projection_input_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    e = previous.evidence(inputs[0][0])
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        for pin in (*origins.values(), *PARENT_PINS, *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *e["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "V input bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_MIDDLE_PAIR_MLP_STAGE17_STAGE11_VALUE_PROJECTION_INPUT_BRIDGE",
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
