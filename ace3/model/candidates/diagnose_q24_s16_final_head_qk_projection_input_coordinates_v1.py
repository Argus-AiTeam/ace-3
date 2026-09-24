"""Read-only input-coordinate accounting for retained P0 Q/K hotspots.

--check compiles the two new Python files in memory, runs only their focused
tests and emits schema-checked JSON to stdout. No evidence output mode exists.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_qk_logit_attribution_v1 as qk


base, parent, channels = qk.base, qk.parent, qk.channels
attribution = qk.scores.attribution
ROOT = qk.ROOT
NAME = "diagnose_q24_s16_final_head_qk_projection_input_coordinates_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 16
WIDTH, LIMIT = 896, 8
OUTPUT_WIDTHS = {"query": 896, "key": 128}
PREFIXES = {kind: f"model.layers.23.self_attn.{letter}_proj."
            for kind, letter in (("query", "q"), ("key", "k"))}
require, same = base.require, base.same
FLAGS = dict(qk.FLAGS)
BOUNDARY = attribution.BOUNDARY + (
    " This version follows the eight selected dimensions of each of the eight "
    "retained QK score hotspots to query/key stage00 FP16 projection-input "
    "actual-minus-original-reference residual coordinates. These are differences "
    "at the post-RMSNorm operator input, not an attribution of raw Q24 state "
    "through nonlinear RMSNorm. Exact delta-input times native AWQ weight terms "
    "are ranked by absolute signed contribution then coordinate. The seven "
    "contiguous G128 input groups are ranked by absolute net sum then group. "
    "Shared authenticated FP16 biases cancel. Retained projection delta minus "
    "the exact input sum is an explicit boundary remainder, not solely rounding. "
    "Reference-partner and actual-partner QK/8 weights retain both swap orders; "
    "interaction is not allocated to either projection. No projection, RoPE, "
    "score, softmax, native or original-prefix replay. No multi-token ranking: "
    "all retained singleton probabilities remain one. No final-head causal claim."
)
obj, RATIONAL = channels.object_schema, channels.RATIONAL


def array_schema(items, count):
    return {"type": "array", "minItems": count, "maxItems": count, "items": items}


TERM_SCHEMA = obj({
    "coordinate": {"type": "integer", "minimum": 0, "maximum": 895},
    "input_group": {"type": "integer", "minimum": 0, "maximum": 6},
    **{key: RATIONAL for key in ("input_delta", "weight", "signed_contribution",
                                "qk_at_reference_partner", "qk_at_actual_partner")},
})
GROUP_SCHEMA = obj({
    "input_group": {"type": "integer", "minimum": 0, "maximum": 6},
    "start_coordinate": {"enum": list(range(0, WIDTH, 128))},
    "end_coordinate_exclusive": {"enum": list(range(128, WIDTH + 1, 128))},
    "coordinate_count": {"const": 128},
    **{key: RATIONAL for key in ("signed_contribution", "sum_absolute_contributions",
                                "qk_at_reference_partner", "qk_at_actual_partner")},
})
ACCOUNT_SCHEMA = obj({
    "projection": {"enum": list(OUTPUT_WIDTHS)},
    "output_coordinate": {"type": "integer", "minimum": 0, "maximum": 895},
    "input_stage": {"const": "stage00_fp16"},
    "coordinate_count": {"const": WIDTH},
    "largest_absolute_coordinates": array_schema(TERM_SCHEMA, LIMIT),
    "groups_by_absolute_signed_sum": array_schema(GROUP_SCHEMA, 7),
    **{key: RATIONAL for key in (
        "exact_input_delta", "retained_projection_delta", "projection_boundary_delta",
        "ranked_signed_sum", "unranked_signed_remainder", "sum_absolute_contributions",
        "reference_partner_over_8", "actual_partner_over_8",
        "qk_input_at_reference_partner", "qk_input_at_actual_partner",
        "qk_boundary_at_reference_partner", "qk_boundary_at_actual_partner",
        "qk_retained_at_reference_partner", "qk_retained_at_actual_partner")},
    "exact_local_identity": {"const": True},
})
ACCOUNT_SCHEMA["allOf"] = [{
    "if": {"properties": {"projection": {"const": "key"}}},
    "then": {"properties": {"output_coordinate": {"maximum": 127}}},
}]
DIMENSION_SCHEMA = obj({
    "dimension_hotspot_rank": {"type": "integer", "minimum": 1, "maximum": LIMIT},
    "retained_qk_component": qk.COMPONENT_SCHEMA,
    "query": ACCOUNT_SCHEMA, "key": ACCOUNT_SCHEMA,
    "both_swap_orders_close": {"const": True},
})
REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "score_hotspot_count": {"const": 72},
    "selected_dimension_count": {"const": 576},
    "projection_account_count": {"const": 1152},
    "input_coordinate_term_count": {"const": 1032192},
    "selected_input_coordinate_count": {"const": 9216},
    "input_group_count": {"const": 8064},
    "attention_reference": {"const": "original_input_fp16"},
    "controls": array_schema(obj({
        "control": {"enum": list(parent.CONTROLS)},
        "score_hotspots": array_schema(obj({
            "score_hotspot_rank": {"type": "integer", "minimum": 1, "maximum": LIMIT},
            "retained_score": qk.scores.SCORE_SCHEMA,
            "source_position": {"const": 0}, "source_token_id": {"const": 9707},
            "dimensions": array_schema(DIMENSION_SCHEMA, LIMIT),
        }), LIMIT),
    }), 9),
    "qk_logit_attribution_report": qk.REPORT_SCHEMA,
})
OUTPUT_SCHEMA = obj({
    **{key: value for key, value in qk.OUTPUT_SCHEMA["properties"].items()
       if key not in ("diagnostic_id", "status", "command", "claim_boundary", "tests", "report")},
    "diagnostic_id": {"const": NAME},
    "status": {"const": "READ_ONLY_QK_PROJECTION_INPUT_COORDINATES"},
    "command": {"const": COMMAND}, "claim_boundary": {"const": BOUNDARY},
    "projection_tensors_verified": {"const": 8},
    "tests": obj({
        "compiled": array_schema({"type": "object"}, 2),
        "executed": {"const": EXPECTED_TESTS},
        **{key: {"const": 0} for key in ("failures", "errors", "skipped")},
    }),
    "report": REPORT_SCHEMA,
})


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("projection-input diagnostic forbids prior checks and native replay")

    with qk.read_only(audit), patch.object(qk, "check", forbidden), \
            patch.object(qk, "focused_tests", forbidden):
        yield


def retained_input(pin, *, actual, expected_qk):
    with np.load(io.BytesIO(base.read_bound(pin)), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate projection fields")
        require(qk.qk_operands(archive, actual=actual) == expected_qk, "projection/QK archive splice")
        return attribution.words(archive["stage00"], (WIDTH,))


def tensor_spec(kind, suffix):
    require(kind in OUTPUT_WIDTHS and suffix in ("qweight", "qzeros", "scales", "bias"),
            "unsupported Q/K tensor")
    width = OUTPUT_WIDTHS[kind]
    return {
        "qweight": ((WIDTH, width // 8), "<i4"),
        "qzeros": ((7, width // 8), "<i4"),
        "scales": ((7, width), "<f2"), "bias": ((width,), "<f2"),
    }[suffix]


def bind_tensor(value, pin, kind, suffix):
    shape, dtype = tensor_spec(kind, suffix)
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.str == dtype
            and pin["name"] == PREFIXES[kind] + suffix and pin["shape"] == list(shape)
            and pin["dtype"] == str(value.dtype) and np.all(np.isfinite(value))
            and hashlib.sha256(value.tobytes()).hexdigest() == pin["sha256"],
            "original AWQ Q/K operand binding changed")
    return value


def load_inputs(previous, assets):
    result = previous[0][0][0][0]
    binding = result["preflight"]["L23_original_reference"]["reference"]
    reference = retained_input(binding["fp16"], actual=False, expected_qk=previous[2])
    actual = {
        row["control"]: retained_input(row["parent"]["terminal_archive"], actual=True,
                                       expected_qk=previous[1][row["control"]])
        for row in result["controls"]}
    tensors = {}
    with channels.contributions.preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        for kind in OUTPUT_WIDTHS:
            tensors[kind] = {}
            for suffix in ("qweight", "qzeros", "scales", "bias"):
                name = PREFIXES[kind] + suffix
                tensors[kind][suffix] = bind_tensor(
                    model.get_tensor(name), binding["canonical"][name], kind, suffix)
    return actual, reference, tensors


def weight_column(tensors, kind, coordinate):
    require(kind in OUTPUT_WIDTHS and type(coordinate) is int
            and 0 <= coordinate < OUTPUT_WIDTHS[kind], "Q/K output coordinate out of bounds")
    return attribution.weight_column(tensors[kind], coordinate)


def account(actual, reference, weights, component, kind):
    require(kind in OUTPUT_WIDTHS and all(
        len(values) == WIDTH and all(isinstance(v, Fraction) for v in values)
        for values in (actual, reference, weights)), "invalid exact projection inputs")
    jsonschema.Draft202012Validator(qk.COMPONENT_SCHEMA).validate(component)
    coordinate = component[kind + "_coordinate"]
    require(coordinate < OUTPUT_WIDTHS[kind], "Q/K output coordinate out of bounds")
    other = "key" if kind == "query" else "query"
    retained = Fraction(component["actual_" + kind]) - Fraction(component["reference_" + kind])
    factors = [Fraction(component[side + "_" + other]) / 8 for side in ("reference", "actual")]
    deltas = [a - r for a, r in zip(actual, reference, strict=True)]
    values = [delta * weight for delta, weight in zip(deltas, weights, strict=True)]
    order = sorted(range(WIDTH), key=lambda i: (-abs(values[i]), i))
    exact = sum(values, Fraction())
    boundary = retained - exact
    ranked = sum((values[i] for i in order[:LIMIT]), Fraction())

    def products(value):
        return {key: str(value * factor) for key, factor in zip(
            ("qk_at_reference_partner", "qk_at_actual_partner"), factors, strict=True)}

    groups = []
    for start in range(0, WIDTH, 128):
        terms = values[start:start + 128]
        total = sum(terms, Fraction())
        groups.append({
            "input_group": start // 128, "start_coordinate": start,
            "end_coordinate_exclusive": start + 128, "coordinate_count": 128,
            "signed_contribution": str(total),
            "sum_absolute_contributions": str(sum(map(abs, terms), Fraction())),
            **products(total),
        })
    groups.sort(key=lambda e: (-abs(Fraction(e["signed_contribution"])), e["input_group"]))
    require(exact + boundary == retained, "projection local identity failed")
    return {
        "projection": kind, "output_coordinate": coordinate, "input_stage": "stage00_fp16",
        "coordinate_count": WIDTH,
        "largest_absolute_coordinates": [
            {"coordinate": i, "input_group": i // 128, "input_delta": str(deltas[i]),
             "weight": str(weights[i]), "signed_contribution": str(values[i]), **products(values[i])}
            for i in order[:LIMIT]],
        "groups_by_absolute_signed_sum": groups,
        "exact_input_delta": str(exact), "retained_projection_delta": str(retained),
        "projection_boundary_delta": str(boundary), "ranked_signed_sum": str(ranked),
        "unranked_signed_remainder": str(exact - ranked),
        "sum_absolute_contributions": str(sum(map(abs, values), Fraction())),
        **{side + "_partner_over_8": str(factor)
           for side, factor in zip(("reference", "actual"), factors, strict=True)},
        **{f"qk_{name}_at_{side}_partner": str(value * factor)
           for name, value in (("input", exact), ("boundary", boundary), ("retained", retained))
           for side, factor in zip(("reference", "actual"), factors, strict=True)},
        "exact_local_identity": True,
    }


def report(previous, actual, reference, tensors):
    jsonschema.Draft202012Validator(qk.REPORT_SCHEMA).validate(previous)
    same(list(actual), list(parent.CONTROLS), "projection input control order changed")
    same([row["control"] for row in previous["controls"]], list(parent.CONTROLS),
         "Q/K control order changed")
    controls, columns = [], {}
    for row, score_row in zip(previous["controls"],
                              previous["score_vs_value_report"]["controls"], strict=True):
        same(row["control"], score_row["control"], "score/QK control splice")
        same([h["retained_score"] for h in row["score_hotspots"]],
             score_row["scores_by_absolute_delta"][:LIMIT], "score hotspot selection changed")
        hotspots = []
        same([h["score_hotspot_rank"] for h in row["score_hotspots"]], list(range(1, 9)),
             "Q/K hotspot order changed")
        for hotspot in row["score_hotspots"]:
            same(hotspot["retained_score"]["kv_head"],
                 hotspot["retained_score"]["query_head"] // 7, "GQA head mapping changed")
            source = hotspot["source_tokens"][0]
            prior = source["qk_account"]
            ordered = prior["full_absolute_component_ranking"]
            same(sorted(e["head_dimension"] for e in ordered), list(range(64)),
                 "Q/K dimension census changed")
            same(ordered, sorted(ordered, key=lambda e: (
                -abs(Fraction(e["signed_contribution"])), e["head_dimension"])),
                 "Q/K dimension order changed")
            same(prior["largest_absolute_components"], ordered[:LIMIT], "Q/K selection changed")
            dimensions = []
            for rank, component in enumerate(ordered[:LIMIT], 1):
                head = hotspot["retained_score"]["query_head"]
                dim = component["head_dimension"]
                same((component["query_coordinate"], component["key_coordinate"]),
                     (head * 64 + dim, (head // 7) * 64 + dim), "Q/K coordinate mapping changed")
                entry = {"dimension_hotspot_rank": rank, "retained_qk_component": component}
                for kind in OUTPUT_WIDTHS:
                    key = (kind, component[kind + "_coordinate"])
                    if key not in columns:
                        columns[key] = weight_column(tensors, *key)
                    entry[kind] = account(actual[row["control"]], reference,
                                          columns[key], component, kind)
                q, k = entry["query"], entry["key"]
                qr, qa = (Fraction(q["qk_retained_at_" + side + "_partner"])
                          for side in ("reference", "actual"))
                kr, ka = (Fraction(k["qk_retained_at_" + side + "_partner"])
                          for side in ("reference", "actual"))
                require(tuple(Fraction(component[part]) for part in qk.PARTS)
                        == (qr, kr, qa - qr, qr + ka), "retained Q/K component identity changed")
                require(qr + ka == kr + qa, "Q/K swap order identity failed")
                entry["both_swap_orders_close"] = True
                dimensions.append(entry)
            hotspots.append({
                "score_hotspot_rank": hotspot["score_hotspot_rank"],
                "retained_score": hotspot["retained_score"],
                "source_position": source["source_position"],
                "source_token_id": source["source_token_id"], "dimensions": dimensions,
            })
        controls.append({"control": row["control"], "score_hotspots": hotspots})
    dimensions = sum(len(h["dimensions"]) for c in controls for h in c["score_hotspots"])
    accounts = dimensions * 2
    return {
        "control_count": len(controls), "score_hotspot_count": dimensions // LIMIT,
        "selected_dimension_count": dimensions, "projection_account_count": accounts,
        "input_coordinate_term_count": accounts * WIDTH,
        "selected_input_coordinate_count": accounts * LIMIT, "input_group_count": accounts * 7,
        "attention_reference": "original_input_fp16", "controls": controls,
        "qk_logit_attribution_report": previous,
    }


def measure():
    previous, files, assets = qk.measure()
    actual, reference, tensors = load_inputs(previous, assets)
    measured = report(previous[3], actual, reference, tensors)
    return (previous, actual, reference, tensors, measured), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("projection_input_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "projection input focused tests failed, errored or skipped")
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
               "projection_input_focused_test": parent.record(TEST)}
    for module in (qk, qk.scores, attribution, channels):
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
            "status": "READ_ONLY_QK_PROJECTION_INPUT_COORDINATES",
            "command": COMMAND, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 8,
            "dispatch_and_write_audit": {**FLAGS, **audit}, "tests": tests, "report": evidence[4],
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
