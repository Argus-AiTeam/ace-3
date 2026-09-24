"""Stdout-only retained middle-pair P0 score/probability versus value contrasts."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_o_projection_input_bridge_v1 as previous
from ace3.model.candidates import diagnose_q24_s16_final_head_attention_score_vs_value_v1 as contrast


source, attention = previous.source, previous.attention
base, parent, bridge = previous.base, previous.parent, previous.bridge
require, same = base.require, base.same
ROOT = previous.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_attention_score_value_contrast_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
FIELDS, GROUPS = previous.FIELDS, previous.GROUPS
PRODUCTS, EFFECTS = contrast.PRODUCTS, contrast.EFFECTS
ZERO_EFFECTS = ("probability_at_reference_value", "probability_at_actual_value", "interaction")
PARENT_PINS = (
    {"path": str(previous.SOURCE), "bytes": 17272,
     "sha256": "d5aefd34950d1d938a99be553373330a225767d8982080a46eeddb154f8cc4a3"},
    {"path": str(previous.TEST), "bytes": 20901,
     "sha256": "f35463b53d35b0e1dd3e13a5604330f1439db5b40f7222664812da23266b2fe8"},
)
FLAGS = {**previous.FLAGS, "score_replay": 0, "softmax_replay": 0,
         "score_value_causal_allocation": False}
BOUNDARY = (
    "Only retained pair (34319,13) stage11 accounts gain P0 stage08 score and "
    "stage09 probability versus value-content contrasts. Finite FP16 scores "
    "are ranked by descending absolute delta then ascending query head. Actual "
    "and original-input FP16 singleton probabilities remain one. All 128 GQA "
    "value components disclose four exact retained-operand products, both "
    "swap orders and zero probability/interaction effects. No softmax or "
    "counterfactual operator is executed. Unchanged AV and separate actual/"
    "original o_proj boundaries close every downstream field through the "
    "unchanged norm/gate/up/down/row/opposite-reference multipliers. Existing "
    "V-projection/o_proj/source-value entries are unchanged; repeated logical "
    "accounts are not independent samples. Score changes are descriptive, "
    "not Q/K attribution, profiling or a dominant-cause explanation. No "
    "predecessor checks, operator/prefix/admission replay, evidence writes or "
    "binary64 internal-attention reconstruction. Q24 residual state remains "
    "wider than FP16, with unchanged INT4 weights and FP16 operator/KV "
    "boundaries. No strict-FP16-state W4A16, new-token or full-model admission. "
    + previous.BOUNDARY
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("score/value bridge forbids predecessor checks, replay and writes")

    with previous.read_only(audit), \
            patch.object(previous, "check", forbidden), \
            patch.object(previous, "run_tests", forbidden), \
            patch.object(previous, "collect", forbidden), \
            patch.object(contrast, "check", forbidden), \
            patch.object(contrast, "focused_tests", forbidden), \
            patch.object(contrast, "measure", forbidden), \
            patch.object(contrast, "report", forbidden):
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
    same(retained, previous.report(*inputs), "reviewed o_proj input bridge binding changed")


def bind_scores(inputs, tensors):
    actual, reference = source.bind_attention(inputs, tensors)
    e = source.evidence(inputs)
    scores = {control: attention.words(e["archives"][control]["stage08"], (14,))
              for control in parent.CONTROLS}
    reference_scores = attention.words(e["reference_archive"]["stage08"], (14,))
    return actual, reference, scores, reference_scores


def score_rows(actual, reference, probabilities, reference_probabilities):
    require(all(len(values) == 14 and all(isinstance(x, Fraction) for x in values)
                for values in (actual, reference, probabilities, reference_probabilities)),
            "invalid exact score/probability operands")
    return contrast.score_rows(actual, reference, probabilities, reference_probabilities)


def singleton_identity(item):
    require(all(item[name] == "0" for name in ZERO_EFFECTS),
            "singleton probability/interaction effect changed")
    same(item[PRODUCTS[0]], item[PRODUCTS[1]], "reference value product changed")
    same(item[PRODUCTS[2]], item[PRODUCTS[3]], "actual value product changed")
    for name in ("value_at_actual_probability", "value_at_reference_probability"):
        same(item[name], item["signed_contribution"], "singleton value contrast changed")


def local_account(actual, reference, weights, inherited, o_projection):
    previous.bind_local(o_projection, inherited)
    require(actual[0] == reference[0] == [Fraction(1)]*14
            and all(isinstance(x, Fraction) for values in (*actual[:2], *reference[:2], weights)
                    for x in values), "invalid exact singleton product operands")
    products = contrast.product_contrasts(actual, reference, weights)
    retained_source = contrast.source_account(inherited, products)
    components = []
    for old in inherited["full_value_component_ranking"]:
        item = contrast.contrast(products[old["value_coordinate"]])
        singleton_identity(item)
        components.append({
            **{key: old[key] for key in (
                "value_coordinate", "kv_head", "head_dimension", "query_heads")},
            "contrasts": item,
        })
    groups = {}
    for group, entries in zip(GROUPS, (components[:8], components[8:], components), strict=True):
        groups[group] = contrast.contrast(tuple(
            sum((Fraction(entry["contrasts"][name]) for entry in entries), Fraction())
            for name in PRODUCTS))
        singleton_identity(groups[group])
    same(groups["full"], retained_source["contrasts"], "source/component products do not close")
    for name in (*PRODUCTS, *EFFECTS):
        same(str(Fraction(groups["selected"][name])+Fraction(groups["unselected"][name])),
             groups["full"][name], "selected/complement contrast identity failed")
    return {
        "control": inherited["control"], "output_coordinate": inherited["output_coordinate"],
        "score_control": inherited["control"], "attention_reference": "original_input_fp16",
        "source_tokens": [retained_source], "value_component_contrasts": components,
        "value_component_group_contrasts": groups,
        **{name: inherited[name] for name in (
            "av_boundary_remainder", "o_projection_boundary_remainder",
            "retained_o_projection_delta")},
        **{name: o_projection[name] for name in previous.TERMS[1:]},
        "closure_residual": "0",
    }


def weighted_account(core, inherited, o_projection, location):
    key = f'{core["control"]}:{core["output_coordinate"]}'
    same(inherited["local_account_key"], key, "source/value local key changed")
    same(o_projection["local_account_key"], key, "o_proj local key changed")
    same(list(inherited["multipliers"]), list(FIELDS), "downstream multiplier census changed")
    same(inherited["multipliers"], o_projection["multipliers"], "downstream multiplier changed")
    same([entry["term"] for entry in inherited["terms"]], list(source.TERMS),
         "source/value term census changed")
    for name in location:
        same(location[name], o_projection[name], "logical account location changed")
    multipliers = {field: Fraction(inherited["multipliers"][field]) for field in FIELDS}
    total = core["source_tokens"][0]["contrasts"]
    singleton_identity(total)
    weighted = {name: {f: str(Fraction(total[name])*m) for f, m in multipliers.items()}
                for name in (*PRODUCTS, *EFFECTS)}
    boundaries = {name: {f: str(Fraction(core[name])*m) for f, m in multipliers.items()}
                  for name in ("av_boundary_remainder", *previous.TERMS[1:])}
    target = inherited["retained_parent_attention_stage11_delta"]
    same(target, o_projection["retained_parent_contributions"], "parent attention target changed")
    for field in FIELDS:
        expected = (
            weighted["probability_at_reference_value"][field],
            weighted["value_at_reference_probability"][field],
            weighted["interaction"][field], boundaries["av_boundary_remainder"][field],
            str(Fraction(core["o_projection_boundary_remainder"])*multipliers[field]))
        same([entry[field] for entry in inherited["terms"]], list(expected),
             "retained source/value downstream terms changed: " + field)
        same(str(Fraction(boundaries[previous.TERMS[1]][field])
                 + Fraction(boundaries[previous.TERMS[2]][field])), expected[-1],
             "separate o_proj boundaries do not close: " + field)
        same(str(Fraction(weighted["signed_contribution"][field])
                 + Fraction(boundaries["av_boundary_remainder"][field])),
             o_projection["terms"][0][field], "o_proj exact stage10 product mismatch: " + field)
        same([boundaries[name][field] for name in previous.TERMS[1:]],
             [entry[field] for entry in o_projection["terms"][1:]],
             "separate downstream o_proj boundaries changed: " + field)
        same(str(Fraction(weighted["signed_contribution"][field])
                 + sum((Fraction(value[field]) for value in boundaries.values()), Fraction())),
             target[field], "downstream contrast/boundary closure failed: " + field)
        for old in (inherited, o_projection):
            same(old["closure_residuals"][field], "0", "inherited closure changed")
    return {
        **location, "local_account_key": key, "multipliers": inherited["multipliers"],
        "contrasts": weighted, "boundaries": boundaries, "retained_parent_contributions": target,
        "value_component_group_contrasts": {
            group: {name: {f: str(Fraction(value)*m) for f, m in multipliers.items()}
                    for name, value in values.items()}
            for group, values in core["value_component_group_contrasts"].items()},
        "closure_residuals": {f: "0" for f in FIELDS},
    }


def report(inputs, retained):
    bind_parent(inputs, retained)
    attention_inputs = inputs[0][0][0]
    actual, reference, scores, reference_scores = bind_scores(
        attention_inputs[0], attention_inputs[2])
    controls = []
    for control in parent.CONTROLS:
        rows = score_rows(scores[control], reference_scores, actual[control][0], reference[0])
        controls.append({
            "control": control, "source_position": 0, "source_token_id": 9707,
            "changed_score_head_count": sum(Fraction(row["score_delta"]) != 0 for row in rows),
            "changed_probability_head_count": 0, "scores_by_absolute_delta": rows,
        })
    cores, columns = {}, {}
    for key, local in retained["attention_stage11_local_accounts"].items():
        control, coordinate = local["control"], local["output_coordinate"]
        same(key, f"{control}:{coordinate}", "retained local key changed")
        if coordinate not in columns:
            columns[coordinate] = attention.weight_column(attention_inputs[2], coordinate)
        cores[key] = local_account(actual[control], reference, columns[coordinate], local,
                                  retained["attention_o_projection_local_accounts"][key])
    sources = list(previous.sources(retained))
    same(len(sources), len(retained["attention_o_projection_input_accounts"]),
         "logical o_proj account census changed")
    accounts, used = [], set()
    for (location, entry), o_projection in zip(
            sources, retained["attention_o_projection_input_accounts"], strict=True):
        inherited = entry["attention_stage11_source_value"]
        key = inherited["local_account_key"]
        require(key in cores, "missing retained contrast local account")
        accounts.append(weighted_account(cores[key], inherited, o_projection, location))
        used.add(key)
    same(sorted(used), sorted(cores), "unreferenced local contrast account")
    return {
        **retained, "attention_score_value_controls": controls,
        "attention_score_value_local_accounts": cores,
        "attention_score_value_accounts": accounts,
        "attention_score_value_summary": {
            "classification": "SINGLETON_PROBABILITY_INVARIANT",
            "control_count": len(controls), "score_pair_count": len(controls)*14,
            "reference_score_count": 14, "changed_probability_head_count": 0,
            "account_count": len(accounts), "shared_local_account_count": len(cores),
            "value_component_count": len(accounts)*128,
            "query_value_product_count": len(accounts)*896,
            "downstream_fields": list(FIELDS),
            "contrast_totals": {
                name: {f: bridge.mass(Fraction(a["contrasts"][name][f]) for a in accounts)
                       for f in FIELDS} for name in (*PRODUCTS, *EFFECTS)},
            "closure_residuals": {f: "0" for f in FIELDS}, "inherited_entries_unchanged": True,
        },
        "attention_score_value_weighting":
            "Each shared local product, effect and boundary is multiplied by "
            "multipliers[field] at each indexed logical parent location. Group "
            "contrasts partition the unchanged selected eight and other 120 "
            "GQA components; no causal allocation or independent-sample claim.",
    }


def collect(audit):
    authenticate_parent()
    inputs = previous.collect(audit)
    with read_only(audit):
        retained = previous.report(*inputs)
    return inputs, retained


def run_tests(inputs, result):
    spec = importlib.util.spec_from_file_location("stage11_score_value_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = inputs, result
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "score/value bridge tests failed, errored or skipped")
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
               "stage11_score_value_tests": parent.record(TEST),
               contrast.MODULE: parent.record(contrast.SOURCE)}
    audit = {"forbidden_calls": 0}
    inputs = collect(audit)
    attention_inputs = inputs[0][0][0][0]
    e = source.evidence(attention_inputs[0])
    with read_only(audit):
        result = report(*inputs)
        tests = run_tests(inputs, result)
        authenticate_parent()
        for pin in (*origins.values(), *base.PINS.values(),
                    *bridge.margin.rows.PINS.values(), *e["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "score/value bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_MIDDLE_PAIR_STAGE11_SCORE_VALUE_CONTRAST_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins, "parent_source_pins": PARENT_PINS,
        "pins": {**base.PINS, **bridge.margin.rows.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": e["files"], "hidden_pins": e["hidden_pins"], "assets": e["assets"],
        "L23_reference_authority": e["result"]["preflight"]["L23_original_reference"],
        "final_reference_authority": e["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": e["result"]["controls"],
        "retained_thresholds": e["result"]["preflight"]["thresholds"],
        "L23_archives_verified": 10, "score_probability_archive_pairs_verified": 10,
        "o_projection_tensors_verified": 3,
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
