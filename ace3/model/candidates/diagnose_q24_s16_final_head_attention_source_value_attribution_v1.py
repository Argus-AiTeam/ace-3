"""Read-only P0 attention-source/value accounting at retained final-head hotspots.

--check compiles this module and its focused tests in memory and emits JSON only.
"""

import argparse
from contextlib import ExitStack, contextmanager
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

from ace3.model.candidates import diagnose_q24_s16_final_head_output_projection_channel_hotspots_v1 as channels


base, parent = channels.base, channels.parent
ROOT = channels.ROOT
NAME = "diagnose_q24_s16_final_head_attention_source_value_attribution_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
EXPECTED_TESTS = 18
WIDTH, LIMIT = 896, 8
PROJECTION = "model.layers.23.self_attn.o_proj."
SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
PARTS = ("probability", "value", "interaction")
require, same = base.require, base.same
FLAGS = {**channels.FLAGS, "attention_replay": 0, "o_projection_replay": 0,
         "binary64_attention_reference_reconstructed": False}
BOUNDARY = (
    "Read-only native-S16-RTZ Q24 CPU-software diagnostic of retained final-head "
    "attempt001. Top final tied-head input channels are output coordinates of the "
    "L23 attention o_proj, not attention query channels. For each selected coordinate, "
    "exact signed P*V*W differences attribute the L23 o_proj component to source "
    "position/token and GQA value components; explicit AV and o_proj boundary "
    "remainders close the retained local difference. This is upstream coordinate "
    "accounting, NOT propagation through residual/MLP/final RMSNorm or a causal "
    "decomposition of final-head channel/logit error. FP16 and binary64 final-head "
    "branches keep their independently propagated original-input references. Only "
    "the original-input FP16 attention trajectory is retained; binary64 attention "
    "stages are unavailable and are neither substituted nor reconstructed. P0 has "
    "one source token 9707, 14 query heads, two KV heads, 64 channels/head and empty "
    "prior KV; no other token/position/layer is supported. No native, prefix, "
    "admission, decoder, attention, o_proj, RMSNorm, head or reference replay; no "
    "evidence writes, token publication/selection, RTL, GPU, hardware, simulation "
    "or ACE2 changes. Original global references, exact thresholds, all nine "
    "L21/L22/L23 failures and source/operand/state/KV/lineage gates are unchanged. "
    "Old sparse-cut closure does not certify or propagate these parents. Q24 state "
    "is wider than FP16; G128 asymmetric packed INT4 GEMM ordering, no qzero "
    "plus-one, FP16 scales/operator boundaries/KV and native S16 RTZ remain unchanged. "
    "No root-cause, runtime-bottleneck, candidate admission, policy adoption, "
    "successor publication, strict-FP16-state W4A16, new-token or full-model claim. "
    "Normal independent Reviewer validation REQUIRED."
)

obj = channels.object_schema
RATIONAL = channels.RATIONAL
COMPONENT_SCHEMA = obj({
    "source_position": {"const": 0}, "source_token_id": {"const": 9707},
    "value_coordinate": {"type": "integer", "minimum": 0, "maximum": 127},
    "kv_head": {"type": "integer", "minimum": 0, "maximum": 1},
    "head_dimension": {"type": "integer", "minimum": 0, "maximum": 63},
    "query_heads": {"type": "array", "minItems": 7, "maxItems": 7,
                    "items": {"type": "integer", "minimum": 0, "maximum": 13}},
    **{key: RATIONAL for key in (*PARTS, "signed_contribution")},
})
ACCOUNT_SCHEMA = obj({
    "attention_reference": {"const": "original_input_fp16"},
    "layer": {"const": 23}, "position": {"const": 0},
    "query_value_product_count": {"const": WIDTH},
    "value_component_count": {"const": 128},
    "source_token_count": {"const": 1},
    "source_tokens": {"type": "array", "minItems": 1, "maxItems": 1, "items": obj({
        "source_position": {"const": 0}, "source_token_id": {"const": 9707},
        **{key: RATIONAL for key in (*PARTS, "signed_contribution")},
    })},
    "full_value_component_ranking": {
        "type": "array", "minItems": 128, "maxItems": 128, "items": COMPONENT_SCHEMA},
    "largest_absolute_value_components": {
        "type": "array", "minItems": LIMIT, "maxItems": LIMIT, "items": COMPONENT_SCHEMA},
    **{key: RATIONAL for key in (
        "exact_attention_sum", "ranked_signed_sum", "unranked_signed_remainder",
        "av_boundary_remainder", "o_projection_boundary_remainder",
        "retained_o_projection_delta")},
    "exact_local_identity": {"const": True},
})
HOTSPOT_SCHEMA = obj({
    "hotspot_rank": {"type": "integer", "minimum": 1, "maximum": LIMIT},
    "final_head_channel": channels.ENTRY_SCHEMA,
    "attention": ACCOUNT_SCHEMA,
})
REPORT_SCHEMA = obj({
    "control_count": {"const": 9}, "selected_row_count": {"const": 18},
    "hotspot_count": {"const": 144},
    "query_value_product_count": {"const": 129024},
    "value_component_count": {"const": 18432},
    "controls": {"type": "array", "minItems": 9, "maxItems": 9, "items": obj({
        "control": {"enum": list(parent.CONTROLS)},
        "branches": obj({branch: obj({
            "numeric_id": channels.ROW_SCHEMA["properties"]["numeric_id"],
            "retained_signed_residual": RATIONAL,
            "hotspots": {"type": "array", "minItems": LIMIT, "maxItems": LIMIT,
                         "items": HOTSPOT_SCHEMA},
        }) for branch in channels.BRANCHES}),
    })},
    "final_head_channel_report": channels.REPORT_SCHEMA,
    "attention_reference": obj({
        "fp16": {"type": "object"}, "manifest": {"type": "object"},
        "binary64_stage_status": {"const": "NOT_RETAINED_NO_RECONSTRUCTION"},
        "final_head_branches_reanchored": {"const": False},
    }),
})
OUTPUT_SCHEMA = obj({
    "diagnostic_id": {"const": NAME}, "version": {"const": 1},
    "status": {"const": "READ_ONLY_ATTENTION_SOURCE_VALUE_ATTRIBUTION"},
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
        raise RuntimeError("attention diagnostic forbids prior checks and native replay")

    with channels.read_only(audit), ExitStack() as stack:
        for module_name, module in list(sys.modules.items()):
            if module_name.startswith("ace3.model.candidates.") and module_name != MODULE:
                for attribute in ("projection", "local_reference", "attention_value"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        stack.enter_context(patch.object(channels, "check", forbidden))
        stack.enter_context(patch.object(channels, "focused_tests", forbidden))
        yield


def words(array, shape):
    require(isinstance(array, np.ndarray) and array.dtype.str == "<u2"
            and array.shape == shape
            and np.all((array & 0x7c00) != 0x7c00), "invalid retained attention FP16 array")
    return [Fraction(float(value)) for value in array.view("<f2").reshape(-1)]


def attention_operands(archive, *, actual):
    p = words(archive["stage09"], (14,))
    v = words(archive["stage07"], (128,))
    av = words(archive["stage10"], (WIDTH,))
    projected = words(archive["stage11"], (WIDTH,))
    words(archive["stage03"], (128,))
    require(np.array_equal(archive["stage03"], archive["stage07"]),
            "V projection/cache lineage changed")
    require(p == [Fraction(1)] * 14, "non-singleton P0 probabilities")
    if actual:
        for kind, stage in (("k", "06"), ("v", "07")):
            words(archive["input_cache_" + kind], (0, 128))
            words(archive["output_cache_" + kind], (1, 128))
            require(np.array_equal(archive["output_cache_" + kind][0],
                                   archive["stage" + stage]), "P0 cache lineage changed")
    require(av == [v[(head // 7) * 64 + dim] for head in range(14) for dim in range(64)],
            "retained P0 GQA AV identity changed")
    return p, v, av, projected


def load_attention(result, assets):
    binding = result["preflight"]["L23_original_reference"]
    require(binding["status"] == "BOUND_ORIGINAL_INPUT_L23" and not binding["missing"]
            and binding["reference"]["prior_kv"] == "own empty P0",
            "original attention reference scope changed")
    reference_pin = binding["reference"]["fp16"]
    same(reference_pin, result["preflight"]["final_reference"]["reference"]["input_fp16"],
         "attention/final-head FP16 lineage splice")
    pins = [reference_pin] + [row["parent"]["terminal_archive"] for row in result["controls"]]

    def archive(pin):
        with np.load(io.BytesIO(base.read_bound(pin)), allow_pickle=False) as arrays:
            require(len(arrays.files) == len(set(arrays.files)), "duplicate archive fields")
            return {key: arrays[key] for key in arrays.files}

    reference_archive = archive(reference_pin)
    reference = attention_operands(reference_archive, actual=False)
    actual = {}
    for row, pin in zip(result["controls"], pins[1:], strict=True):
        retained = archive(pin)
        actual[row["control"]] = attention_operands(retained, actual=True)
    tensors = {}
    with channels.contributions.preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        for suffix, shape, dtype in (
                ("qweight", (896, 112), "<i4"),
                ("qzeros", (7, 112), "<i4"),
                ("scales", (7, 896), "<f2")):
            name = PROJECTION + suffix
            value = model.get_tensor(name)
            pin = binding["reference"]["canonical"][name]
            require(value.shape == shape and value.dtype.str == dtype
                    and pin["name"] == name and pin["shape"] == list(shape)
                    and pin["dtype"] == str(value.dtype)
                    and np.all(np.isfinite(value))
                    and hashlib.sha256(value.tobytes()).hexdigest() == pin["sha256"],
                    "original AWQ o_proj operand binding changed")
            tensors[suffix] = value
    return actual, reference, tensors, pins


def weight_column(tensors, coordinate):
    require(type(coordinate) is int and 0 <= coordinate < WIDTH,
            "o_proj output coordinate outside bounded width")
    shift = SHIFTS[coordinate % 8]
    group = coordinate // 8
    return [
        Fraction(float(tensors["scales"][j // 128, coordinate])) * (
            ((int(tensors["qweight"][j, group]) >> shift) & 15)
            - ((int(tensors["qzeros"][j // 128, group]) >> shift) & 15))
        for j in range(WIDTH)
    ]


def account(actual, reference, weights, coordinate):
    pa, va, aa, oa = actual
    pr, vr, ar, ort = reference
    require(type(coordinate) is int and 0 <= coordinate < WIDTH
            and len(weights) == WIDTH, "invalid attribution coordinate/weights")
    components = [[Fraction() for _ in PARTS] for _ in range(128)]
    av_remainder = Fraction()
    for j, weight in enumerate(weights):
        head, dim = divmod(j, 64)
        value = (head // 7) * 64 + dim
        dp, dv = pa[head] - pr[head], va[value] - vr[value]
        terms = (dp * vr[value] * weight, pr[head] * dv * weight, dp * dv * weight)
        for index, term in enumerate(terms):
            components[value][index] += term
        av_remainder += (aa[j] - ar[j]) * weight - sum(terms)
    totals = [sum(parts) for parts in components]
    order = sorted(range(128), key=lambda i: (-abs(totals[i]), i))

    def entry(i):
        kv, dim = divmod(i, 64)
        return {
            "source_position": 0, "source_token_id": 9707,
            "value_coordinate": i, "kv_head": kv, "head_dimension": dim,
            "query_heads": list(range(kv * 7, (kv + 1) * 7)),
            **{key: str(value) for key, value in zip(PARTS, components[i], strict=True)},
            "signed_contribution": str(totals[i]),
        }

    exact, retained = sum(totals), oa[coordinate] - ort[coordinate]
    remainder = retained - exact - av_remainder
    ranked = sum(totals[i] for i in order[:LIMIT])
    require(exact + av_remainder + remainder == retained, "local attention identity failed")
    return {
        "attention_reference": "original_input_fp16", "layer": 23, "position": 0,
        "query_value_product_count": WIDTH, "value_component_count": 128,
        "source_token_count": 1,
        "source_tokens": [{
            "source_position": 0, "source_token_id": 9707,
            **{key: str(sum(row[i] for row in components)) for i, key in enumerate(PARTS)},
            "signed_contribution": str(exact),
        }],
        "full_value_component_ranking": [entry(i) for i in order],
        "largest_absolute_value_components": [entry(i) for i in order[:LIMIT]],
        "exact_attention_sum": str(exact), "ranked_signed_sum": str(ranked),
        "unranked_signed_remainder": str(exact - ranked),
        "av_boundary_remainder": str(av_remainder),
        "o_projection_boundary_remainder": str(remainder),
        "retained_o_projection_delta": str(retained), "exact_local_identity": True,
    }


def report(channel_report, actual, reference, tensors, binding):
    same(list(actual), list(parent.CONTROLS), "attention control order changed")
    same([row["control"] for row in channel_report["controls"]], list(parent.CONTROLS),
         "final-head control order changed")
    rows, cache, columns = [], {}, {}
    for row in channel_report["controls"]:
        branches = {}
        same(list(row["branches"]), list(channels.BRANCHES), "reference branch order changed")
        for name, branch in row["branches"].items():
            selected = branch["channels"]["largest_absolute_signed"]
            same(len(selected), LIMIT, "hotspot selection count changed")
            hotspots = []
            for rank, channel in enumerate(selected, 1):
                i = channel["coordinate"]
                if i not in columns:
                    columns[i] = weight_column(tensors, i)
                key = (row["control"], i)
                if key not in cache:
                    cache[key] = account(actual[row["control"]], reference, columns[i], i)
                hotspots.append({"hotspot_rank": rank, "final_head_channel": channel,
                                 "attention": cache[key]})
            branches[name] = {
                "numeric_id": branch["numeric_id"],
                "retained_signed_residual": branch["retained_signed_residual"],
                "hotspots": hotspots,
            }
        rows.append({"control": row["control"], "branches": branches})
    return {
        "control_count": 9, "selected_row_count": 18, "hotspot_count": 144,
        "query_value_product_count": 144 * WIDTH, "value_component_count": 144 * 128,
        "controls": rows, "final_head_channel_report": channel_report,
        "attention_reference": {
            "fp16": binding["reference"]["fp16"], "manifest": binding["manifest"],
            "binary64_stage_status": "NOT_RETAINED_NO_RECONSTRUCTION",
            "final_head_branches_reanchored": False,
        },
    }


def measure():
    evidence, files, assets = channels.measure()
    result = evidence[0]
    actual, reference, tensors, pins = load_attention(result, assets)
    measured = report(evidence[5], actual, reference, tensors,
                      result["preflight"]["L23_original_reference"])
    return (evidence, actual, reference, tensors, measured), files + pins, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("attention_source_value_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "attention diagnostic focused tests failed, errored or skipped")
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
               "attention_focused_test": parent.record(TEST),
               channels.MODULE: parent.record(channels.SOURCE)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *channels.PINS.values(), *files):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "forbidden dispatch or evidence write attempted")
        output = {
            "diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_ATTENTION_SOURCE_VALUE_ATTRIBUTION",
            "command": COMMAND, "claim_boundary": BOUNDARY,
            "normal_host_review": "REQUIRED",
            "original_execution_sources": parent.RETAINED_SOURCES,
            "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
            "attention_archives_verified": 10, "projection_tensors_verified": 3,
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
