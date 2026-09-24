"""Stdout-only exact stage10 input accounts for retained middle-pair o_proj."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_value_projection_unselected_stage00_complement_v1 as previous


source = previous.previous.previous
attention = source.attention
base, parent, bridge = previous.base, previous.parent, previous.bridge
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_o_projection_input_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
WIDTH, LIMIT = 896, 8
FIELDS, GROUPS = previous.FIELDS, previous.GROUPS
TERMS = ("exact_stage10_weighted_delta", "actual_projection_boundary",
         "negative_original_input_fp16_projection_boundary")
PARENT_PINS = (
    {"path": str(previous.SOURCE), "bytes": 12829,
     "sha256": "50bef3ac80db46ab73e98f3ca6da28fc91c7b8bad2e534b651b697d1591fe65c"},
    {"path": str(previous.TEST), "bytes": 18691,
     "sha256": "19fa6b6acd59f75dda63577e1dc18fb49ad90c5eb14063617fec4a7e6e0ae4ea"},
)
FLAGS = {**previous.FLAGS, "stage10_o_projection_input_causal_allocation": False}
BOUNDARY = (
    "Every retained local stage11 attention account for pair (34319,13) gains "
    "896 exact stage10 delta-input times native G128 o_proj weight accounts. "
    "Eight descending-absolute coordinates use ascending-coordinate tie breaks; "
    "seven contiguous input-ordered G128 groups partition the 888-coordinate "
    "complement. Signed, absolute and cancellation masses are exact. Separate "
    "actual and negative original-input FP16 projection boundaries, not solely "
    "rounding, close stage11 delta through unchanged norm/gate/up/down/row/"
    "opposite-reference multipliers. Existing selected, source/value and V "
    "projection accounts remain unchanged. Shared accounts are not independent "
    "samples; stage10 channel masses precede GQA value-component aggregation. "
    "No predecessor checks, operator/prefix/admission replay, evidence writes or "
    "binary64 internal-attention reconstruction. Q24 residual state is wider "
    "than FP16; INT4 weights, S16 RTZ and FP16 operator/KV boundaries remain "
    "unchanged. No dominant cause, strict-FP16-state W4A16, new-token or "
    "full-model admission is established. " + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("o_proj input bridge forbids predecessor checks, replay and writes")

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
    same(retained, previous.report(*inputs), "reviewed V complement binding changed")


def projection_account(actual, reference, weights, actual_o, reference_o, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < WIDTH
            and all(len(v) == WIDTH and all(isinstance(x, Fraction) for x in v)
                    for v in (actual, reference, weights))
            and isinstance(actual_o, Fraction) and isinstance(reference_o, Fraction),
            "invalid exact stage10/o_proj account")
    actual_sum = sum((x*w for x, w in zip(actual, weights, strict=True)), Fraction())
    reference_sum = sum((x*w for x, w in zip(reference, weights, strict=True)), Fraction())
    terms = [(a-r)*w for a, r, w in zip(actual, reference, weights, strict=True)]
    order = sorted(range(WIDTH), key=lambda i: (-abs(terms[i]), i))
    selected = order[:LIMIT]
    excluded = set(selected)
    groups = []
    for start in range(0, WIDTH, 128):
        indices = [i for i in range(start, start+128) if i not in excluded]
        groups.append({
            "input_group": start//128, "start_coordinate": start,
            "end_coordinate_exclusive": start+128, "coordinate_count": len(indices),
            "excluded_ranked_coordinate_count": 128-len(indices),
            **bridge.mass(terms[i] for i in indices),
        })
    totals = {
        group: bridge.mass(terms[i] for i in indices)
        for group, indices in zip(GROUPS, (selected, order[LIMIT:], order), strict=True)}
    same(source.merged_mass(groups), totals["unselected"], "G128 complement mass changed")
    same(source.merged_mass((totals["selected"], totals["unselected"])), totals["full"],
         "selected/complement mass does not close")
    exact = sum(terms, Fraction())
    actual_boundary = actual_o-actual_sum
    negative_original_boundary = reference_sum-reference_o
    require(exact == actual_sum-reference_sum
            and exact+actual_boundary+negative_original_boundary == actual_o-reference_o,
            "stage10/stage11 projection identity failed")
    return {
        "output_coordinate": coordinate, "input_stage": "stage10_fp16",
        "output_stage": "stage11_fp16", "source_position": 0, "source_token_id": 9707,
        "coordinate_count": WIDTH, "attention_reference": "original_input_fp16",
        "actual_o_projection": str(actual_o), "original_input_fp16_o_projection": str(reference_o),
        "actual_exact_input_weight_sum": str(actual_sum),
        "original_exact_input_weight_sum": str(reference_sum),
        "exact_stage10_weighted_delta": str(exact),
        "actual_projection_boundary": str(actual_boundary),
        "negative_original_input_fp16_projection_boundary": str(negative_original_boundary),
        "retained_o_projection_delta": str(actual_o-reference_o),
        "largest_absolute_input_coordinates": [
            {"coordinate": i, "input_group": i//128, "actual_input": str(actual[i]),
             "original_input_fp16_input": str(reference[i]),
             "input_delta": str(actual[i]-reference[i]), "weight": str(weights[i]),
             "signed_contribution": str(terms[i])} for i in selected],
        "input_coordinate_totals": totals,
        "unselected_stage10_inputs": {
            "coordinate_count": WIDTH-LIMIT, "excluded_ranked_coordinates": selected,
            "groups_in_input_order": groups, "totals": totals["unselected"],
            "closure_residual": "0"},
        "closure_residual": "0",
    }


def bind_local(core, retained):
    same((core["control"], core["output_coordinate"]),
         (retained["control"], retained["output_coordinate"]), "local account lineage changed")
    same(core["retained_o_projection_delta"], retained["retained_o_projection_delta"],
         "retained stage11 delta changed")
    same(core[TERMS[0]], str(Fraction(retained["exact_attention_sum"])
                           + Fraction(retained["av_boundary_remainder"])),
         "retained stage10 weighted delta changed")
    same(str(Fraction(core[TERMS[1]])+Fraction(core[TERMS[2]])),
         retained["o_projection_boundary_remainder"], "retained o_proj boundary changed")


def sources(retained):
    for row_index, row in enumerate(retained["rows"]):
        for hotspot_index, hotspot in enumerate(row["hotspots"]):
            for selected_index, selected in enumerate(hotspot["selected_stage16_coordinates"]):
                for kind in ("gate_proj", "up_proj"):
                    for source_index, entry in enumerate(selected[kind]["ranked_stage13_rmsnorm_sources"]):
                        yield {
                            "row_index": row_index, "hotspot_index": hotspot_index,
                            "selected_stage16_index": selected_index, "projection": kind,
                            "ranked_stage13_index": source_index,
                        }, entry


def weighted_account(core, inherited, location):
    same(inherited["local_account_key"], f'{core["control"]}:{core["output_coordinate"]}',
         "retained attention account key changed")
    same(list(inherited["multipliers"]), list(FIELDS), "downstream multiplier census changed")
    same([t["term"] for t in inherited["terms"]], list(source.TERMS),
         "retained source/value term census changed")
    multipliers = {f: Fraction(inherited["multipliers"][f]) for f in FIELDS}
    terms = [{"term": term, **{f: str(Fraction(core[term])*m)
                             for f, m in multipliers.items()}} for term in TERMS]
    target = inherited["retained_parent_attention_stage11_delta"]
    totals = {
        group: {f: source.scaled_mass(core["input_coordinate_totals"][group], m)
                for f, m in multipliers.items()} for group in GROUPS}
    for field in FIELDS:
        same(str(sum((Fraction(t[field]) for t in terms), Fraction())), target[field],
             "stage10/boundaries do not close retained parent: " + field)
        same(terms[0][field],
             str(sum((Fraction(t[field]) for t in inherited["terms"][:4]), Fraction())),
             "weighted stage10/source-value bridge changed: " + field)
        same(str(Fraction(terms[1][field])+Fraction(terms[2][field])),
             inherited["terms"][4][field], "weighted o_proj boundary changed: " + field)
        closed = (Fraction(totals["selected"][field]["signed"])
                  + Fraction(totals["unselected"][field]["signed"])
                  + Fraction(terms[1][field])+Fraction(terms[2][field]))
        same(str(closed), target[field], "selected/complement/boundary closure failed: " + field)
        same(inherited["closure_residuals"][field], "0", "retained closure changed")
    return {
        **location, "local_account_key": inherited["local_account_key"],
        "multipliers": inherited["multipliers"], "retained_parent_contributions": target,
        "terms": terms, "input_coordinate_totals": totals,
        "closure_residuals": {f: "0" for f in FIELDS},
    }


def report(inputs, retained):
    bind_parent(inputs, retained)
    attention_inputs = inputs[0][0]
    actual, reference = source.bind_attention(attention_inputs[0], attention_inputs[2])
    cores, columns = {}, {}
    for key, local in retained["attention_stage11_local_accounts"].items():
        control, coordinate = local["control"], local["output_coordinate"]
        same(key, f"{control}:{coordinate}", "retained local attention key changed")
        if coordinate not in columns:
            columns[coordinate] = attention.weight_column(attention_inputs[2], coordinate)
        core = projection_account(actual[control][2], reference[2], columns[coordinate],
                                  actual[control][3][coordinate], reference[3][coordinate], coordinate)
        cores[key] = {**core, "control": control}
        bind_local(cores[key], local)
    accounts, used = [], set()
    for location, entry in sources(retained):
        inherited = entry["attention_stage11_source_value"]
        key = inherited["local_account_key"]
        require(key in cores, "missing retained o_proj input account")
        accounts.append(weighted_account(cores[key], inherited, location))
        used.add(key)
    same(sorted(used), sorted(cores), "unreferenced local o_proj input account")
    same(len(accounts), retained["attention_stage11_source_value_summary"]["account_count"],
         "retained stage11 account census changed")
    return {
        **retained, "attention_o_projection_local_accounts": cores,
        "attention_o_projection_input_accounts": accounts,
        "attention_o_projection_input_summary": {
            "account_count": len(accounts), "shared_local_account_count": len(cores),
            "input_coordinate_count": len(accounts)*WIDTH,
            "selected_input_coordinate_count": len(accounts)*LIMIT,
            "unselected_input_coordinate_count": len(accounts)*(WIDTH-LIMIT),
            "unselected_input_group_count": len(accounts)*7,
            "downstream_fields": list(FIELDS),
            "term_totals": {
                term: {f: bridge.mass(Fraction(a["terms"][i][f]) for a in accounts)
                       for f in FIELDS} for i, term in enumerate(TERMS)},
            "input_coordinate_totals": {
                group: {f: source.merged_mass(a["input_coordinate_totals"][group][f]
                                              for a in accounts) for f in FIELDS}
                for group in GROUPS},
            "closure_residuals": {f: "0" for f in FIELDS},
            "inherited_entries_unchanged": True,
        },
        "attention_o_projection_input_weighting":
            "each shared stage10 coordinate, G128 complement group and boundary uses "
            "multipliers[field]; signed *= multiplier, absolute *= abs(multiplier), "
            "cancellation = absolute-abs(signed). Logical locations index unchanged "
            "parent rows/hotspots/selections/projections/ranked sources.",
    }


def collect(audit):
    authenticate_parent()
    inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*inputs)
    return inputs, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("stage11_o_projection_input_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "o_proj input bridge tests failed, errored or skipped")
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
               "stage11_o_projection_input_tests": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    e = source.evidence(inputs[0][0][0][0])
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        authenticate_parent()
        for pin in (*origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *e["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "o_proj input bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_MIDDLE_PAIR_STAGE11_O_PROJECTION_INPUT_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins, "parent_source_pins": PARENT_PINS,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": e["files"], "hidden_pins": e["hidden_pins"], "assets": e["assets"],
        "L23_reference_authority": e["result"]["preflight"]["L23_original_reference"],
        "final_reference_authority": e["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": e["result"]["controls"],
        "retained_thresholds": e["result"]["preflight"]["thresholds"],
        "L23_archives_verified": 10, "o_projection_tensors_verified": 3,
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
