"""Read-only tied-head row geometry for retained numeric diagnostic pairs.

--check compiles both new Python files in memory, runs focused tests, and
emits one JSON document. No RMSNorm, row-dot or head operator is replayed.
"""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_rmsnorm_delta_hotspots_v1 as hotspots


contributions = hotspots.contributions
base, cutoff, parent = hotspots.base, hotspots.cutoff, hotspots.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_row_difference_hotspots_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
    f" -B -m {MODULE} --check"
)
PINS = {
    **hotspots.PINS,
    "rmsnorm_hotspot_source": {
        "path": str(hotspots.SOURCE),
        "sha256": "68e9b31d22832e0f969ac0db8be3ad3e509f4b0787f56afd19901a7f52df2beb",
    },
    "rmsnorm_hotspot_test": {
        "path": str(hotspots.TEST),
        "sha256": "a0eb4a7d45381075d4c4b139c3a07d1a5617f87230731c826696394ba09272d6",
    },
}
RANK_LIMIT = hotspots.RANK_LIMIT
RANK_FIELDS = hotspots.RANK_FIELDS
BRANCHES = hotspots.BRANCHES
EXPECTED_TESTS = 24
FLAGS = dict(hotspots.FLAGS)
require, same = base.require, base.same
BOUNDARY = (
    "Read-only CPU-software tied-head row differences for already exposed numeric "
    "diagnostic pairs on reviewed final-head attempt001 and original-input final "
    "attempt003 independent FP16/binary64 references. Exact row subtraction and "
    "coordinate contribution identities only: no row-dot, RMSNorm/head/native/decoder/"
    "prefix/admission/reference replay, full-vocabulary recomputation, evidence writes, "
    "token decode/publication/selection, RTL/GPU/hardware/simulation dispatch. All nine "
    "L21/L22/L23 failures, exact thresholds and source/operand/state/KV/lineage gates "
    "remain intact. Old sparse-cut adjusted closure PASS cannot certify, substitute "
    "for or propagate the new nine-control parents. Row geometry and hotspot overlap "
    "do not establish an upstream root cause, runtime bottleneck or causal shares. "
    "Q24 residual state is wider than FP16; native S16 RTZ, G128 asymmetric packed "
    "INT4 GEMM ordering without qzero plus-one, FP16 scales/operator boundaries/KV "
    "and tied-head identities remain unchanged. No candidate admission, policy "
    "adoption, successor publication, strict-FP16-state W4A16, new-token or full-model "
    "claim. Normal independent Reviewer validation REQUIRED."
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("row geometry diagnostic forbids prior-check or operator replay")

    with hotspots.read_only(audit), ExitStack() as stack:
        for name in ("measure", "check", "focused_tests"):
            stack.enter_context(patch.object(hotspots, name, forbidden))
        yield


def authenticate():
    for pin in PINS.values():
        base.read_bound(pin)
    return hotspots.authenticate()


def row_summary(left, right, forced, limit=RANK_LIMIT):
    require(left.dtype.str == right.dtype.str == "<f2"
            and left.shape == right.shape == (896,), "tied-head row shape/dtype changed")
    summary = hotspots.delta_summary(left, right, forced, limit)
    wl, wr = contributions.exact_vector(left), contributions.exact_vector(right)
    values = [a - b for a, b in zip(wl, wr, strict=True)]
    # This second subtraction path uses integer ratios, not vector subtraction.
    for a, b, value in zip(left, right, values, strict=True):
        an, ad = float(a).as_integer_ratio()
        bn, bd = float(b).as_integer_ratio()
        require(value == Fraction(an * bd - bn * ad, ad * bd),
                "independent exact row-difference identity failed")
    for field in (*RANK_FIELDS, "full_absolute_signed_ranking", "forced_coordinates"):
        summary[field] = [
            {**{key: value for key, value in entry.items()
                if key not in ("signed_delta", "actual_exact", "reference_exact", "actual_hex", "reference_hex")},
             "signed_row_delta": entry["signed_delta"],
             "absolute_magnitude": str(abs(Fraction(entry["signed_delta"]))),
             "left_exact": str(wl[entry["coordinate"]]),
             "right_exact": str(wr[entry["coordinate"]])}
            for entry in summary[field]
        ]
    summary["exact_sum_of_absolute_row_deltas"] = summary.pop("exact_sum_of_absolute_contributions")
    summary["independent_exact_row_difference_identity"] = True
    return summary, values, wl, wr


def relationship(row_values, multipliers, left, right, retained_summary):
    require(len(row_values) == len(multipliers) == len(left) == len(right) == 896,
            "coordinate relationship width changed")
    products = [h * d for h, d in zip(multipliers, row_values, strict=True)]
    expanded = [h * l - h * r for h, l, r in zip(multipliers, left, right, strict=True)]
    require(products == expanded, "expanded coordinate-contribution identity failed")
    ranked = contributions.ranked(products)
    for key, value in ranked.items():
        same(value, retained_summary[key], "retained coordinate contribution summary changed")
    row_rank = contributions.ranked(row_values)
    overlap = {
        field: hotspots.overlap(
            [entry["coordinate"] for entry in row_rank[field]],
            [entry["coordinate"] for entry in ranked[field]])
        for field in RANK_FIELDS
    }
    nonzero = [i for i, value in enumerate(row_values) if value]
    erased = [i for i in nonzero if multipliers[i] == 0]
    reversed_sign = [i for i in nonzero if multipliers[i] < 0]
    preserved_sign = [i for i in nonzero if multipliers[i] > 0]
    same_rank = all(value["same_order"] for value in overlap.values())
    return {
        "exact_coordinate_identity": True, "coordinate_count": len(products),
        "coordinate_formula": "multiplier[i] * (left[i] - right[i]) = multiplier[i]*left[i] - multiplier[i]*right[i]",
        "exact_contribution_sum": ranked["exact_sum"],
        "row_vs_contribution_hotspots": overlap,
        "row_sign_preserved_coordinates": preserved_sign,
        "row_sign_reversed_coordinates": reversed_sign,
        "nonzero_row_erased_coordinates": erased,
        "top_rankings_match_row_geometry": same_rank,
        "interpretation": (
            "All reported absolute/positive/negative row and contribution rankings coincide "
            "for this multiplier; this is descriptive agreement, not geometry-only causation."
            if same_rank else
            "Row geometry alone does not reproduce the reported signed contribution rankings. "
            "Contributions also depend on coordinate-dependent multiplier magnitudes; negative "
            "multipliers reverse signs and zero multipliers erase nonzero row differences."
        ),
    }


def report(result, arrays, references, geometry, weights, alignment):
    base.check_history(result)
    same(list(arrays), list(parent.CONTROLS), "row geometry control census changed")
    same([row["control"] for row in geometry["controls"]], list(parent.CONTROLS),
         "cutoff control census changed")
    same([row["control"] for row in alignment["controls"]], list(parent.CONTROLS),
         "RMSNorm hotspot control census changed")
    rows = []
    for control, aligned in zip(geometry["controls"], alignment["controls"], strict=True):
        label = control["control"]
        expected = contributions.pairs_for(control)
        pairs = aligned["selected_pair_coordinate_alignment"]
        same([(pair["left_id"], pair["right_id"]) for pair in pairs], list(expected),
             "selected diagnostic pair census changed")
        hidden = {
            "actual_fp16": arrays[label]["rmsnorm"].view("<f2"),
            "fp16": references["rmsnorm_fp16"].view("<f2"),
            "binary64": references["rmsnorm_binary64"],
        }
        vectors = {name: contributions.exact_vector(value) for name, value in hidden.items()}
        measured_pairs = []
        for pair in pairs:
            left, right = pair["left_id"], pair["right_id"]
            same(pair["roles"], expected[left, right], "selected diagnostic pair roles changed")
            forced = hotspots.forced_coordinates([pair])
            for branch, summary in aligned["rmsnorm_delta_hotspots"].items():
                for field in RANK_FIELDS:
                    for entry in summary[field]:
                        forced.setdefault(entry["coordinate"], []).append(f"rmsnorm_delta:{branch}:{field}")
            summary, values, wl, wr = row_summary(weights[left], weights[right], forced)
            branches = {
                name: relationship(values, vector, wl, wr, pair["branches"][name])
                for name, vector in vectors.items()
            }
            effects = {}
            for name in BRANCHES:
                delta = [a - r for a, r in zip(vectors["actual_fp16"], vectors[name], strict=True)]
                effect = relationship(values, delta, wl, wr, pair["actual_minus_independent_reference"][name])
                independently_expanded = [
                    (a * l - a * r) - (ref * l - ref * r)
                    for a, ref, l, r in zip(vectors["actual_fp16"], vectors[name], wl, wr, strict=True)
                ]
                require([h * d for h, d in zip(delta, values, strict=True)] == independently_expanded,
                        "actual/reference expanded contribution identity failed")
                retained = pair["actual_minus_independent_reference"][name]
                require(Fraction(retained["retained_margin_change"])
                        == sum(independently_expanded, Fraction()) + Fraction(retained["boundary_remainder_change"]),
                        "retained margin/contribution remainder identity failed")
                effects[name] = {**effect, "actual_minus_reference_identity": True}
            for entry in summary["forced_coordinates"]:
                i = entry["coordinate"]
                entry["coordinate_relationships"] = {
                    "branches": {
                        name: {"multiplier": str(vector[i]), "signed_contribution": str(vector[i] * values[i])}
                        for name, vector in vectors.items()
                    },
                    "actual_minus_independent_reference": {
                        name: {
                            "multiplier": str(vectors["actual_fp16"][i] - vectors[name][i]),
                            "signed_contribution": str((vectors["actual_fp16"][i] - vectors[name][i]) * values[i]),
                        } for name in BRANCHES
                    },
                }
            measured_pairs.append({
                **pair, "row_difference_hotspots": summary,
                "row_vs_rmsnorm_delta_hotspots": {
                    name: {
                        field: hotspots.overlap(
                            [entry["coordinate"] for entry in summary[field]],
                            [entry["coordinate"] for entry in delta[field]])
                        for field in RANK_FIELDS
                    } for name, delta in aligned["rmsnorm_delta_hotspots"].items()
                },
                "contribution_relationships": {
                    "branches": branches, "actual_minus_independent_reference": effects},
            })
        rows.append({**control, "rmsnorm_delta_hotspots": aligned["rmsnorm_delta_hotspots"],
                     "row_difference_pairs": measured_pairs})
    return {
        **geometry, "controls": rows, "lineage_separation": contributions.LINEAGE,
        "selected_pair_count": sum(len(row["row_difference_pairs"]) for row in rows),
        "distinct_numeric_pairs": sorted({(pair["left_id"], pair["right_id"])
                                         for row in rows for pair in row["row_difference_pairs"]}),
        "coordinate_product_vectors": alignment["coordinate_product_vectors"],
        "arithmetic": {
            "row_delta": "W[left_numeric_id,i] - W[right_numeric_id,i], exact rational",
            "ranking": "absolute magnitude descending, coordinate ascending on ties; positive descending, negative ascending",
            "rank_limit": RANK_LIMIT, "rank_base": 1, "coordinate_count": 896,
            "forced": "coordinate 62 plus all signed/absolute RMSNorm delta and selected-pair contribution hotspots",
            "overlap": "top-eight sets and order only, not new thresholds or nonnegative causal shares",
            "branch_multiplier": "retained actual or independently propagated reference RMSNorm[i]",
            "change_multiplier": "retained actual RMSNorm[i] minus independent reference RMSNorm[i]",
            "boundary_remainder": "retained pair-margin change minus coordinate-product sum; not attributed solely to rounding",
            "evaluation": "coordinate subtraction/products only; no row-dot/head or prior diagnostic check replay",
        },
    }


def measure():
    result, arrays, references, files, assets = authenticate()
    geometry = cutoff.report(result, arrays, references)
    ids = {index for control in geometry["controls"] for pair in contributions.pairs_for(control) for index in pair}
    weights = contributions.load_rows(assets, ids)
    alignment = hotspots.report(result, arrays, references, geometry, weights)
    measured = report(result, arrays, references, geometry, weights, alignment)
    return (result, arrays, references, geometry, weights, alignment, measured), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("row_difference_hotspot_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "row geometry tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun, "failures": len(outcome.failures),
            "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "row_difference_focused_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *PINS.values()):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "row geometry diagnostic attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1, "status": "READ_ONLY_ROW_DIFFERENCE_HOTSPOTS",
        "command": COMMAND, "pins": {**base.PINS, **PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
        "head_rows": [{"numeric_id": index, "sha256": hashlib.sha256(row.tobytes()).hexdigest()}
                      for index, row in evidence[4].items()],
        "input_terminal_review": {
            "mission_id": base.MISSION, "round": 2, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done"},
        "attempt001_files_verified": 20, "arrays_verified": 18, "reference_arrays_verified": 4,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        **evidence[6],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
