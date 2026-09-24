"""Read-only exact Q/K component accounting for retained P0 score hotspots.

--check compiles this module and its focused tests in memory, runs only those
tests, validates the JSON schema, and emits JSON to stdout without evidence writes.
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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_score_vs_value_v1 as scores


base, parent, channels = scores.base, scores.parent, scores.channels
ROOT = scores.ROOT
NAME = "diagnose_q24_s16_final_head_attention_qk_logit_attribution_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
require, same = base.require, base.same
FLAGS = {**scores.FLAGS, "qk_projection_replay": 0, "rope_replay": 0}
BOUNDARY = (
    scores.attribution.BOUNDARY
    + " This version attributes the eight largest absolute retained score deltas "
    "per control to exact scaled Q/K component-product differences, not operator "
    "execution. Each head has only source position 0/token 9707; all retained "
    "probabilities are one, so probability hotspots are tied with zero difference. "
    "No multi-token ranking or nonzero probability effect is inferred. Query-only, "
    "key-only and interaction terms use the original-input FP16 Q/K reference; "
    "neither swap order is a causal allocation. Retained score minus exact QK/8 "
    "has an explicit boundary remainder, not attributed solely to rounding. "
    "Dimension groups are the two contiguous 32-component halves, not upstream "
    "projection provenance. No Q/K, RoPE, score or softmax replay; no binary64 "
    "attention reconstruction, reference reanchoring or final-head causal claim."
)
obj, RATIONAL = channels.object_schema, channels.RATIONAL
PARTS = ("query_at_reference_key", "key_at_reference_query", "interaction",
         "signed_contribution")
COMPONENT_SCHEMA = obj({
    "head_dimension": {"type": "integer", "minimum": 0, "maximum": 63},
    "query_coordinate": {"type": "integer", "minimum": 0, "maximum": 895},
    "key_coordinate": {"type": "integer", "minimum": 0, "maximum": 127},
    **{key: RATIONAL for key in ("actual_query", "reference_query",
                               "actual_key", "reference_key", *PARTS)},
})
ACCOUNT_SCHEMA = obj({
    "component_count": {"const": 64}, "score_scale": {"const": "1/8"},
    "full_absolute_component_ranking": {
        "type": "array", "minItems": 64, "maxItems": 64, "items": COMPONENT_SCHEMA},
    "largest_absolute_components": {
        "type": "array", "minItems": 8, "maxItems": 8, "items": COMPONENT_SCHEMA},
    "component_groups": {"type": "array", "minItems": 2, "maxItems": 2, "items": obj({
        "group": {"enum": ["dimensions_0_31", "dimensions_32_63"]},
        "start_dimension": {"enum": [0, 32]}, "end_dimension_exclusive": {"enum": [32, 64]},
        **{key: RATIONAL for key in PARTS},
    })},
    **{key: RATIONAL for key in (
        *PARTS, "query_at_actual_key", "key_at_actual_query",
        "actual_exact_scaled_dot", "reference_exact_scaled_dot",
        "actual_score_boundary_remainder", "reference_score_boundary_remainder",
        "score_boundary_delta", "retained_score_delta",
        "ranked_signed_sum", "unranked_signed_remainder")},
    "exact_local_identity": {"const": True},
})
HOTSPOT_SCHEMA = obj({
    "score_hotspot_rank": {"type": "integer", "minimum": 1, "maximum": 8},
    "retained_score": scores.SCORE_SCHEMA,
    "source_tokens": {"type": "array", "minItems": 1, "maxItems": 1, "items": obj({
        "source_rank": {"const": 1}, "source_position": {"const": 0},
        "source_token_id": {"const": 9707}, "qk_account": ACCOUNT_SCHEMA,
    })},
})
REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "score_pair_count": {"const": 126},
    "selected_score_hotspot_count": {"const": 72}, "source_account_count": {"const": 72},
    "dimension_account_count": {"const": 4608},
    "selected_dimension_count": {"const": 576}, "component_group_count": {"const": 144},
    "attention_reference": {"const": "original_input_fp16"},
    "classification": {"const": "SINGLETON_PROBABILITY_INVARIANT"},
    "controls": {"type": "array", "minItems": 9, "maxItems": 9, "items": obj({
        "control": {"enum": list(parent.CONTROLS)},
        "score_hotspots": {
            "type": "array", "minItems": 8, "maxItems": 8, "items": HOTSPOT_SCHEMA},
    })},
    "score_vs_value_report": scores.REPORT_SCHEMA,
})
OUTPUT_SCHEMA = obj({
    "diagnostic_id": {"const": NAME}, "version": {"const": 1},
    "status": {"const": "READ_ONLY_ATTENTION_QK_LOGIT_ATTRIBUTION"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "normal_host_review": {"const": "REQUIRED"},
    "original_execution_sources": {"const": parent.RETAINED_SOURCES},
    "diagnostic_sources": {"type": "object"},
    "authenticated_files": {"type": "array", "minItems": 1},
    "assets": {"type": "object"}, "attention_archives_verified": {"const": 10},
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
        raise RuntimeError("Q/K diagnostic forbids prior checks and native replay")

    with scores.read_only(audit), patch.object(scores, "check", forbidden), \
            patch.object(scores, "focused_tests", forbidden):
        yield


def qk_operands(archive, *, actual):
    q = scores.attribution.words(archive["stage04"], (896,))
    k = scores.attribution.words(archive["stage06"], (128,))
    for earlier, later, shape in (
            ("stage01", "stage04", (896,)), ("stage02", "stage05", (128,)),
            ("stage05", "stage06", (128,))):
        scores.attribution.words(archive[earlier], shape)
        scores.attribution.words(archive[later], shape)
        require(np.array_equal(archive[earlier], archive[later]),
                "retained P0 Q/K projection/RoPE/cache lineage changed")
    if actual:
        scores.attribution.words(archive["input_cache_k"], (0, 128))
        scores.attribution.words(archive["output_cache_k"], (1, 128))
        require(np.array_equal(archive["output_cache_k"][0], archive["stage06"]),
                "retained P0 K cache lineage changed")
    return q, k


def retained_qk(pin, *, actual, expected_scores):
    with np.load(io.BytesIO(base.read_bound(pin)), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate attention fields")
        require(scores.attribution.words(archive["stage08"], (14,)) == expected_scores,
                "Q/K and retained score archive splice")
        return qk_operands(archive, actual=actual)


def account(actual, reference, retained):
    for q, k in (actual, reference):
        require(len(q) == 896 and len(k) == 128
                and all(isinstance(x, Fraction) for x in (*q, *k)),
                "invalid exact bounded Q/K operands")
    jsonschema.Draft202012Validator(scores.SCORE_SCHEMA).validate(retained)
    head, kv = retained["query_head"], retained["kv_head"]
    same(kv, head // 7, "GQA head mapping changed")
    sa, sr = Fraction(retained["actual_score"]), Fraction(retained["reference_score"])
    same(str(sa - sr), retained["score_delta"], "retained score delta changed")
    qa, ka = actual
    qr, kr = reference
    components = []
    actual_dot, reference_dot = Fraction(), Fraction()
    for dim in range(64):
        qi, ki = head * 64 + dim, kv * 64 + dim
        a, b, r, s = qa[qi], ka[ki], qr[qi], kr[ki]
        terms = ((a - r) * s / 8, r * (b - s) / 8, (a - r) * (b - s) / 8)
        actual_dot += a * b / 8
        reference_dot += r * s / 8
        components.append({
            "head_dimension": dim, "query_coordinate": qi, "key_coordinate": ki,
            "actual_query": str(a), "reference_query": str(r),
            "actual_key": str(b), "reference_key": str(s),
            **{key: str(value) for key, value in zip(
                PARTS, (*terms, sum(terms)), strict=True)},
        })

    def totals(entries):
        return {key: str(sum(Fraction(e[key]) for e in entries)) for key in PARTS}

    total = totals(components)
    delta = Fraction(total["signed_contribution"])
    ar, rr = sa - actual_dot, sr - reference_dot
    require(delta == actual_dot - reference_dot and delta + ar - rr == sa - sr,
            "Q/K local score identity failed")
    ordered = sorted(components, key=lambda e: (
        -abs(Fraction(e["signed_contribution"])), e["head_dimension"]))
    ranked = sum(Fraction(e["signed_contribution"]) for e in ordered[:8])
    return {
        "component_count": 64, "score_scale": "1/8",
        "full_absolute_component_ranking": ordered, "largest_absolute_components": ordered[:8],
        "component_groups": [
            {"group": f"dimensions_{start}_{start + 31}",
             "start_dimension": start, "end_dimension_exclusive": start + 32,
             **totals(components[start:start + 32])} for start in (0, 32)],
        **total,
        "query_at_actual_key": str(Fraction(total["query_at_reference_key"])
                                   + Fraction(total["interaction"])),
        "key_at_actual_query": str(Fraction(total["key_at_reference_query"])
                                   + Fraction(total["interaction"])),
        "actual_exact_scaled_dot": str(actual_dot), "reference_exact_scaled_dot": str(reference_dot),
        "actual_score_boundary_remainder": str(ar), "reference_score_boundary_remainder": str(rr),
        "score_boundary_delta": str(ar - rr), "retained_score_delta": str(sa - sr),
        "ranked_signed_sum": str(ranked), "unranked_signed_remainder": str(delta - ranked),
        "exact_local_identity": True,
    }


def report(previous, actual, reference):
    jsonschema.Draft202012Validator(scores.REPORT_SCHEMA).validate(previous)
    same(list(actual), list(parent.CONTROLS), "Q/K control order changed")
    same([row["control"] for row in previous["controls"]], list(parent.CONTROLS),
         "score control order changed")
    controls = []
    for row in previous["controls"]:
        ordered = row["scores_by_absolute_delta"]
        same(sorted(e["query_head"] for e in ordered), list(range(14)),
             "missing or duplicate score head")
        same(ordered, sorted(ordered, key=lambda e: (
            -abs(Fraction(e["score_delta"])), e["query_head"])), "score ranking changed")
        same(row["changed_score_head_count"],
             sum(Fraction(e["score_delta"]) != 0 for e in ordered), "changed score count changed")
        hotspots = []
        for rank, entry in enumerate(ordered[:8], 1):
            hotspots.append({
                "score_hotspot_rank": rank, "retained_score": entry,
                "source_tokens": [{
                    "source_rank": 1, "source_position": 0, "source_token_id": 9707,
                    "qk_account": account(actual[row["control"]], reference, entry),
                }],
            })
        controls.append({"control": row["control"], "score_hotspots": hotspots})
    return {
        "control_count": len(controls), "score_pair_count": previous["score_pair_count"],
        "selected_score_hotspot_count": len(controls) * 8,
        "source_account_count": len(controls) * 8,
        "dimension_account_count": len(controls) * 8 * 64,
        "selected_dimension_count": len(controls) * 8 * 8,
        "component_group_count": len(controls) * 8 * 2,
        "attention_reference": "original_input_fp16",
        "classification": "SINGLETON_PROBABILITY_INVARIANT",
        "controls": controls, "score_vs_value_report": previous,
    }


def measure():
    previous, files, assets = scores.measure()
    result = previous[0][0][0]
    reference = retained_qk(
        result["preflight"]["L23_original_reference"]["reference"]["fp16"],
        actual=False, expected_scores=previous[2])
    actual = {
        row["control"]: retained_qk(row["parent"]["terminal_archive"], actual=True,
                                   expected_scores=previous[1][row["control"]])
        for row in result["controls"]}
    return (previous, actual, reference, report(previous[3], actual, reference)), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("attention_qk_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "Q/K focused tests failed, errored or skipped")
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
               "qk_focused_test": parent.record(TEST)}
    for module in (scores, scores.attribution, channels):
        origins[module.MODULE] = parent.record(module.SOURCE)
        origins[module.MODULE + ".focused_test"] = parent.record(module.TEST)
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *channels.PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or evidence write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_ATTENTION_QK_LOGIT_ATTRIBUTION",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10,
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
