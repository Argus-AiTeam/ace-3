"""Read-only retained P0 score/probability versus value-content contrasts.

--check compiles the two new Python files in memory, runs focused tests, and
emits JSON to stdout. No attention operator or earlier check is executed.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_source_value_attribution_v1 as attribution


base, parent, channels = attribution.base, attribution.parent, attribution.channels
ROOT = attribution.ROOT
NAME = "diagnose_q24_s16_final_head_attention_score_vs_value_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
require, same = base.require, base.same
FLAGS = {**attribution.FLAGS, "score_replay": 0, "softmax_replay": 0}
BOUNDARY = (
    attribution.BOUNDARY + " This version additionally reads retained stage08 scores "
    "and stage09 probabilities, and evaluates exact weighted P/V product contrasts "
    "with either operand held fixed, NOT counterfactual operator executions. "
    "Singleton P0 softmax has probability one independently of its finite score; "
    "changed scores need not change weighting. Both swap orders and their interaction "
    "are reported without choosing a causal allocation. There is only one source "
    "token, not a multi-token source ranking. Score changes are descriptive, not "
    "Q/K attribution, runtime profiling or a final-head error explanation."
)
obj, RATIONAL = channels.object_schema, channels.RATIONAL
PRODUCTS = ("reference_product", "actual_probability_reference_value",
            "reference_probability_actual_value", "actual_product")
EFFECTS = ("probability_at_reference_value", "value_at_actual_probability",
           "value_at_reference_probability", "probability_at_actual_value",
           "interaction", "signed_contribution")
CONTRAST_SCHEMA = obj({key: RATIONAL for key in (*PRODUCTS, *EFFECTS)})
SCORE_SCHEMA = obj({
    "query_head": {"type": "integer", "minimum": 0, "maximum": 13},
    "kv_head": {"type": "integer", "minimum": 0, "maximum": 1},
    **{key: RATIONAL for key in ("actual_score", "reference_score", "score_delta")},
    "actual_probability": {"const": "1"}, "reference_probability": {"const": "1"},
    "probability_delta": {"const": "0"},
})
SOURCE_SCHEMA = obj({
    "source_rank": {"const": 1}, "source_position": {"const": 0},
    "source_token_id": {"const": 9707}, "contrasts": CONTRAST_SCHEMA,
    "largest_absolute_value_components": {
        "type": "array", "minItems": 8, "maxItems": 8, "items": obj({
            "value_coordinate": attribution.COMPONENT_SCHEMA["properties"]["value_coordinate"],
            "kv_head": attribution.COMPONENT_SCHEMA["properties"]["kv_head"],
            "head_dimension": attribution.COMPONENT_SCHEMA["properties"]["head_dimension"],
            "query_heads": attribution.COMPONENT_SCHEMA["properties"]["query_heads"],
            "contrasts": CONTRAST_SCHEMA,
        })},
})
REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "score_pair_count": {"const": 126},
    "reference_score_count": {"const": 14}, "hotspot_count": {"const": 144},
    "source_account_count": {"const": 144}, "selected_value_component_count": {"const": 1152},
    "attention_reference": {"const": "original_input_fp16"},
    "classification": {"const": "SINGLETON_PROBABILITY_INVARIANT"},
    "controls": {"type": "array", "minItems": 9, "maxItems": 9, "items": obj({
        "control": {"enum": list(parent.CONTROLS)},
        "changed_score_head_count": {"type": "integer", "minimum": 0, "maximum": 14},
        "changed_probability_head_count": {"const": 0},
        "scores_by_absolute_delta": {
            "type": "array", "minItems": 14, "maxItems": 14, "items": SCORE_SCHEMA},
        "branches": obj({branch: obj({
            "numeric_id": channels.ROW_SCHEMA["properties"]["numeric_id"],
            "hotspots": {"type": "array", "minItems": 8, "maxItems": 8, "items": obj({
                "hotspot_rank": attribution.HOTSPOT_SCHEMA["properties"]["hotspot_rank"],
                "final_head_channel": channels.ENTRY_SCHEMA,
                "source_tokens": {"type": "array", "minItems": 1, "maxItems": 1,
                                  "items": SOURCE_SCHEMA},
                **{key: RATIONAL for key in (
                    "av_boundary_remainder", "o_projection_boundary_remainder",
                    "retained_o_projection_delta")},
                "exact_local_identity": {"const": True},
            })},
        }) for branch in channels.BRANCHES}),
    })},
    "source_value_attribution_report": attribution.REPORT_SCHEMA,
})
OUTPUT_SCHEMA = obj({
    "diagnostic_id": {"const": NAME}, "version": {"const": 1},
    "status": {"const": "READ_ONLY_ATTENTION_SCORE_VS_VALUE"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "normal_host_review": {"const": "REQUIRED"},
    "original_execution_sources": {"const": parent.RETAINED_SOURCES},
    "diagnostic_sources": {"type": "object"},
    "authenticated_files": {"type": "array", "minItems": 1},
    "assets": {"type": "object"},
    "attention_archives_verified": {"const": 10},
    "projection_tensors_verified": {"const": 3},
    "dispatch_and_write_audit": {"const": {**FLAGS, "forbidden_calls": 0}},
    "tests": obj({
        "compiled": {"type": "array", "minItems": 2, "maxItems": 2},
        "executed": {"const": EXPECTED_TESTS},
        **{key: {"const": 0} for key in ("failures", "errors", "skipped")},
    }),
    "report": REPORT_SCHEMA,
})


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("score/value diagnostic forbids prior checks and replay")

    with attribution.read_only(audit), \
            patch.object(attribution, "check", forbidden), \
            patch.object(attribution, "focused_tests", forbidden):
        yield


def retained_scores(pin):
    with np.load(io.BytesIO(base.read_bound(pin)), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate attention fields")
        return attribution.words(archive["stage08"], (14,))


def score_rows(actual, reference, probabilities, reference_probabilities):
    require(len(actual) == len(reference) == 14, "invalid score count")
    require(probabilities == [Fraction(1)] * 14, "actual singleton probabilities changed")
    require(reference_probabilities == [Fraction(1)] * 14, "reference singleton probabilities changed")
    return [{
        "query_head": head, "kv_head": head // 7,
        "actual_score": str(actual[head]), "reference_score": str(reference[head]),
        "score_delta": str(actual[head] - reference[head]),
        "actual_probability": "1", "reference_probability": "1", "probability_delta": "0",
    } for head in sorted(range(14), key=lambda h: (-abs(actual[h] - reference[h]), h))]


def contrast(products):
    c00, c10, c01, c11 = products
    effects = (c10 - c00, c11 - c10, c01 - c00, c11 - c01,
               c11 - c10 - c01 + c00, c11 - c00)
    return {key: str(value) for key, value in zip(
        (*PRODUCTS, *EFFECTS), (*products, *effects), strict=True)}


def product_contrasts(actual, reference, weights):
    pa, va = actual[:2]
    pr, vr = reference[:2]
    require(len(pa) == len(pr) == 14 and len(va) == len(vr) == 128
            and len(weights) == 896, "invalid bounded product operands")
    products = []
    for index in range(128):
        kv, dim = divmod(index, 64)
        heads = range(kv * 7, (kv + 1) * 7)
        a = sum(pa[h] * weights[h * 64 + dim] for h in heads)
        r = sum(pr[h] * weights[h * 64 + dim] for h in heads)
        products.append((r * vr[index], a * vr[index], r * va[index], a * va[index]))
    return products


def source_account(account, products):
    same(len(products), 128, "value component count changed")
    components = account["full_value_component_ranking"]
    same(sorted(e["value_coordinate"] for e in components), list(range(128)),
         "missing or duplicate value coordinate")
    same(components, sorted(components, key=lambda e: (
        -abs(Fraction(e["signed_contribution"])), e["value_coordinate"])),
        "value ranking changed")
    selected = []
    for entry in components:
        index = entry["value_coordinate"]
        item = contrast(products[index])
        for old, new in (("probability", "probability_at_reference_value"),
                         ("value", "value_at_reference_probability"),
                         ("interaction", "interaction"),
                         ("signed_contribution", "signed_contribution")):
            same(item[new], entry[old], "retained attribution/product contrast mismatch")
        if len(selected) < 8:
            selected.append({
                **{key: entry[key] for key in (
                    "value_coordinate", "kv_head", "head_dimension", "query_heads")},
                "contrasts": item,
            })
    total = contrast(tuple(sum(row[i] for row in products) for i in range(4)))
    same(total["signed_contribution"], account["exact_attention_sum"],
         "source sum mismatch")
    require(Fraction(total["signed_contribution"])
            + Fraction(account["av_boundary_remainder"])
            + Fraction(account["o_projection_boundary_remainder"])
            == Fraction(account["retained_o_projection_delta"]), "local boundary identity failed")
    return {"source_rank": 1, "source_position": 0, "source_token_id": 9707,
            "contrasts": total, "largest_absolute_value_components": selected}


def report(evidence, scores, reference_scores):
    _, actual, reference, tensors, previous = evidence
    same(list(actual), list(parent.CONTROLS), "attention control order changed")
    same(list(scores), list(parent.CONTROLS), "score control order changed")
    same([row["control"] for row in previous["controls"]], list(parent.CONTROLS),
         "attribution control order changed")
    rows, columns, cache = [], {}, {}
    for row in previous["controls"]:
        label = row["control"]
        ordered_scores = score_rows(scores[label], reference_scores, actual[label][0], reference[0])
        same(list(row["branches"]), list(channels.BRANCHES), "reference branch order changed")
        branches = {}
        for name, branch in row["branches"].items():
            same([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, 9)),
                 "hotspot rank/count changed")
            hotspots = []
            for hotspot in branch["hotspots"]:
                coordinate = hotspot["final_head_channel"]["coordinate"]
                if coordinate not in columns:
                    columns[coordinate] = attribution.weight_column(tensors, coordinate)
                key = (label, coordinate)
                if key not in cache:
                    cache[key] = product_contrasts(actual[label], reference, columns[coordinate])
                account = hotspot["attention"]
                source = source_account(account, cache[key])
                require(all(source["contrasts"][key] == "0" for key in (
                    "probability_at_reference_value", "probability_at_actual_value", "interaction")),
                    "singleton source weighting effect changed")
                hotspots.append({
                    **{key: hotspot[key] for key in ("hotspot_rank", "final_head_channel")},
                    "source_tokens": [source],
                    **{key: account[key] for key in (
                        "av_boundary_remainder", "o_projection_boundary_remainder",
                        "retained_o_projection_delta", "exact_local_identity")},
                })
            branches[name] = {"numeric_id": branch["numeric_id"], "hotspots": hotspots}
        rows.append({
            "control": label, "changed_score_head_count": sum(
                Fraction(s["score_delta"]) != 0 for s in ordered_scores),
            "changed_probability_head_count": 0,
            "scores_by_absolute_delta": ordered_scores, "branches": branches,
        })
    return {
        "control_count": len(rows), "score_pair_count": len(rows) * 14,
        "reference_score_count": len(reference_scores), "hotspot_count": len(rows) * 16,
        "source_account_count": len(rows) * 16, "selected_value_component_count": len(rows) * 128,
        "attention_reference": "original_input_fp16",
        "classification": "SINGLETON_PROBABILITY_INVARIANT",
        "controls": rows, "source_value_attribution_report": previous,
    }


def measure():
    evidence, files, assets = attribution.measure()
    result = evidence[0][0]
    reference = retained_scores(result["preflight"]["L23_original_reference"]["reference"]["fp16"])
    scores = {row["control"]: retained_scores(row["parent"]["terminal_archive"])
              for row in result["controls"]}
    return (evidence, scores, reference, report(evidence, scores, reference)), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("attention_score_value_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "score/value focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors),
            "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "score_value_focused_test": parent.record(TEST),
               attribution.MODULE: parent.record(attribution.SOURCE),
               channels.MODULE: parent.record(channels.SOURCE)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *channels.PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or evidence write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1, "status": "READ_ONLY_ATTENTION_SCORE_VS_VALUE",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 3,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "tests": tests, "report": evidence[3],
        }
        jsonschema.Draft202012Validator.check_schema(OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
