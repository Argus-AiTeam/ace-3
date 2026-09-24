"""Read-only RMSNorm delta hotspots against independent original-input references.

--check compiles both new Python files in memory, runs only the focused tests,
and emits one JSON document. No RMSNorm, row-dot or head operator is replayed.
"""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_coordinate_contributions_v1 as contributions


base = contributions.base
cutoff = contributions.cutoff
parent = contributions.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_rmsnorm_delta_hotspots_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
    f" -B -m {MODULE} --check"
)
PINS = {
    **contributions.PINS,
    "contribution_source": {
        "path": str(contributions.SOURCE),
        "sha256": "f3e32abfe8bdca843a52a0775e776626f7fc4dabc0599ff8816c34d68aada36c",
    },
    "contribution_test": {
        "path": str(contributions.TEST),
        "sha256": "a0494da50f76bbabc4f61de534d2102b4a9b5a32d4dd87d139ee355868814cd0",
    },
}
RANK_LIMIT = contributions.RANK_LIMIT
EXPECTED_TESTS = 24
BRANCHES = ("fp16", "binary64")
RANK_FIELDS = ("largest_absolute_signed", "largest_positive", "most_negative")
require, same = base.require, base.same
FLAGS = {
    **contributions.FLAGS,
    "selected_row_head_replay": 0,
    "local_exact_row_dot_computations": 0,
    "simulation_dispatch": 0,
}
BOUNDARY = (
    "Read-only CPU-software RMSNorm delta localization on reviewed final-head "
    "attempt001 and independent original-input final attempt003 FP16/binary64 vectors. "
    "Exact coordinate subtraction and selected diagnostic-pair coordinate products "
    "are descriptive arithmetic, not RMSNorm, row-dot, head, full-vocabulary, native, "
    "decoder, prefix, admission or reference replay. No evidence writes, token decode, "
    "publication or selection. All nine L21/L22/L23 failures, exact thresholds and "
    "source/operand/state/KV/lineage gates remain unchanged. Old sparse-cut adjusted "
    "closure PASS cannot certify, substitute for or propagate the new nine-control "
    "parents. Hotspot overlap is not an upstream root-cause or runtime-bottleneck claim; "
    "signed contributions are not nonnegative causal shares. Q24 residual state is "
    "wider than FP16; native S16 RTZ, G128 asymmetric packed INT4 GEMM ordering without "
    "qzero plus-one, FP16 scales/operator boundaries/KV and tied-head identities remain "
    "unchanged. No RTL, GPU, hardware or simulation dispatch, candidate admission, "
    "policy adoption, successor publication, strict-FP16-state W4A16, new-token or "
    "full-model claim. Normal independent Reviewer validation REQUIRED."
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("hotspot diagnostic forbids operator or prior-check replay")

    with contributions.read_only(audit), ExitStack() as stack:
        for module, names in (
            (contributions, ("analyze_pair", "dyadic_dot", "report", "measure", "check", "focused_tests")),
            (cutoff, ("check", "focused_tests")),
            (base, ("check", "focused_tests", "census")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        yield


def authenticate():
    for pin in PINS.values():
        base.read_bound(pin)
    return contributions.authenticate()


def indices(summary):
    return [entry["coordinate"] for entry in summary["largest_absolute_signed"]]


def overlap(left, right):
    common = sorted(set(left) & set(right))
    union = set(left) | set(right)
    return {
        "shared_coordinates": common,
        "left_only": sorted(set(left) - set(right)),
        "right_only": sorted(set(right) - set(left)),
        "overlap_count": len(common),
        "same_order": left == right,
        "jaccard": str(Fraction(len(common), len(union))) if union else "1",
        "classification": (
            "IDENTICAL_HOTSPOT_SET" if set(left) == set(right)
            else "PARTIAL_HOTSPOT_OVERLAP" if common else "DISJOINT_HOTSPOTS"
        ),
    }


def delta_summary(actual, reference, forced, limit=RANK_LIMIT):
    a, r = contributions.exact_vector(actual), contributions.exact_vector(reference)
    require(len(a) == len(r), "RMSNorm delta vector shapes differ")
    require(isinstance(forced, dict) and all(
        type(i) is int and 0 <= i < len(a) and isinstance(reasons, list)
        and reasons and all(isinstance(reason, str) and reason for reason in reasons)
        for i, reasons in forced.items()), "invalid forced hotspot coordinates")
    values = [x - y for x, y in zip(a, r, strict=True)]
    ranked = contributions.ranked(values, limit)
    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    ranks = {i: rank for rank, i in enumerate(order, 1)}

    def entry(i):
        return {"coordinate": i, "signed_delta": str(values[i])}

    return {
        **{key: value for key, value in ranked.items() if key not in RANK_FIELDS},
        **{key: [entry(row["coordinate"]) for row in ranked[key]] for key in RANK_FIELDS},
        "full_absolute_signed_ranking": [entry(i) for i in order],
        "forced_coordinates": [
            {**entry(i), "absolute_rank": ranks[i],
             "actual_exact": str(a[i]), "reference_exact": str(r[i]),
             "actual_hex": float(actual[i]).hex(), "reference_hex": float(reference[i]).hex(),
             "in_largest_absolute": i in order[:limit], "reasons": sorted(set(forced[i]))}
            for i in sorted(forced)
        ],
    }


def pair_products(left, right, vectors, weights, retained, roles):
    wl, wr = contributions.exact_vector(weights[left]), contributions.exact_vector(weights[right])
    contrast = [a - b for a, b in zip(wl, wr, strict=True)]
    products = {
        name: [h * w for h, w in zip(hidden, contrast, strict=True)]
        for name, hidden in vectors.items()
    }
    summaries = {name: contributions.ranked(values) for name, values in products.items()}
    effects = {}
    for name in BRANCHES:
        values = [(a - r) * w for a, r, w in zip(
            vectors["actual_fp16"], vectors[name], contrast, strict=True)]
        require(values == [a - r for a, r in zip(
            products["actual_fp16"], products[name], strict=True)],
            "coordinate delta/contribution identity failed")
        summary = contributions.ranked(values)
        actual_margin = cutoff.exact(retained["actual_fp16"][left]) - cutoff.exact(retained["actual_fp16"][right])
        reference_margin = cutoff.exact(retained[name][left]) - cutoff.exact(retained[name][right])
        effects[name] = {
            **summary, "exact_coordinate_identity": True,
            "retained_actual_margin": str(actual_margin),
            "retained_reference_margin": str(reference_margin),
            "retained_margin_change": str(actual_margin - reference_margin),
            "boundary_remainder_change": str(actual_margin - reference_margin - sum(values, Fraction())),
        }
    return {
        "left_id": left, "right_id": right, "roles": roles,
        "branches": summaries, "actual_minus_independent_reference": effects,
    }


def forced_coordinates(pairs):
    forced = {62: ["required_coordinate_62"]}
    for pair in pairs:
        for kind in ("branches", "actual_minus_independent_reference"):
            for branch, summary in pair[kind].items():
                for field in RANK_FIELDS:
                    reason = f"{pair['left_id']}-{pair['right_id']}:{kind}:{branch}:{field}"
                    for entry in summary[field]:
                        forced.setdefault(entry["coordinate"], []).append(reason)
    return forced


def report(result, arrays, references, geometry, weights):
    base.check_history(result)
    same(list(arrays), list(parent.CONTROLS), "hotspot control census changed")
    same([row["control"] for row in geometry["controls"]], list(parent.CONTROLS),
         "cutoff control census changed")
    rows = []
    for control in geometry["controls"]:
        label = control["control"]
        hidden = {
            "actual_fp16": arrays[label]["rmsnorm"].view("<f2"),
            "fp16": references["rmsnorm_fp16"].view("<f2"),
            "binary64": references["rmsnorm_binary64"],
        }
        require(all(value.shape == (896,) for value in hidden.values()), "RMSNorm width changed")
        vectors = {name: contributions.exact_vector(value) for name, value in hidden.items()}
        retained = {
            "actual_fp16": arrays[label]["logits"].view("<f2"),
            "fp16": references["logits_fp16"].view("<f2"),
            "binary64": references["logits_binary64"],
        }
        pairs = [pair_products(left, right, vectors, weights, retained, roles)
                 for (left, right), roles in contributions.pairs_for(control).items()]
        forced = forced_coordinates(pairs)
        summaries = {name: delta_summary(hidden["actual_fp16"], hidden[name], forced) for name in BRANCHES}
        for pair in pairs:
            for name in BRANCHES:
                effect = pair["actual_minus_independent_reference"][name]
                effect["hidden_delta_hotspot_overlap"] = overlap(indices(summaries[name]), indices(effect))
                effect["branch_contribution_hotspot_overlap"] = {
                    branch: overlap(indices(summaries[name]), indices(summary))
                    for branch, summary in pair["branches"].items()
                }
        rows.append({
            **control, "rmsnorm_delta_hotspots": summaries, "selected_pair_coordinate_alignment": pairs,
            "reference_branch_hotspot_overlap": overlap(indices(summaries["fp16"]), indices(summaries["binary64"])),
        })
    by_control = {row["control"]: row for row in rows}
    classes = {
        kind: base.equivalence_classes(result["controls"], kinds)
        for kind, kinds in (("rmsnorm", ("rmsnorm",)), ("joint", ("rmsnorm", "logits")))
    }
    for kind, groups in classes.items():
        for group in groups:
            first = group["controls"][0]
            for label in group["controls"][1:]:
                for array_kind in (("rmsnorm",) if kind == "rmsnorm" else ("rmsnorm", "logits")):
                    require(arrays[first][array_kind].tobytes() == arrays[label][array_kind].tobytes(),
                            "byte-identical artifact group changed")
                for branch in BRANCHES:
                    same(by_control[first]["rmsnorm_delta_hotspots"][branch]["full_absolute_signed_ranking"],
                         by_control[label]["rmsnorm_delta_hotspots"][branch]["full_absolute_signed_ranking"],
                         "byte-identical RMSNorm group has different deltas")
            group["per_reference_hotspots"] = {
                branch: indices(by_control[first]["rmsnorm_delta_hotspots"][branch]) for branch in BRANCHES}
            group["byte_identity_and_delta_rankings_verified"] = True
    cross = [
        {"left": left["control"], "right": right["control"],
         "per_reference": {
             branch: overlap(indices(left["rmsnorm_delta_hotspots"][branch]),
                             indices(right["rmsnorm_delta_hotspots"][branch])) for branch in BRANCHES}}
        for index, left in enumerate(rows) for right in rows[index + 1:]
    ]
    return {
        **geometry, "controls": rows, "artifact_equivalence_classes": classes,
        "cross_control_hotspots": {"pair_count": len(cross), "pairs": cross},
        "lineage_separation": contributions.LINEAGE,
        "coordinate_product_vectors": sum(5 * len(row["selected_pair_coordinate_alignment"]) for row in rows),
        "arithmetic": {
            "delta": "actual RMSNorm[i] minus independent original-input reference RMSNorm[i], exact rational",
            "ranking": "absolute magnitude descending, coordinate ascending on ties; full signed ranking",
            "rank_limit": RANK_LIMIT, "rank_base": 1,
            "forced": "coordinate 62 and all reported signed/absolute cutoff and exchanged-pair contribution hotspots",
            "pair_contribution": "h[i] * (W[left,i] - W[right,i]); existing selected-row diagnostic definition",
            "pair_delta": "(actual_h[i] - reference_h[i]) * (W[left,i] - W[right,i])",
            "alignment": "top-eight set overlap, not a causal percentage or new acceptance threshold",
            "boundary_remainder": "retained pair-margin change minus exact coordinate-product sum; not solely rounding",
            "references": "independently propagated FP16 and binary64, never cast from each other or reanchored",
            "pair_evaluation": "coordinate arithmetic only; no prior diagnostic check, row-dot, head or vector-splice replay",
        },
    }


def measure():
    result, arrays, references, files, assets = authenticate()
    geometry = cutoff.report(result, arrays, references)
    ids = {index for control in geometry["controls"] for pair in contributions.pairs_for(control) for index in pair}
    weights = contributions.load_rows(assets, ids)
    measured = report(result, arrays, references, geometry, weights)
    return (result, arrays, references, geometry, weights, measured), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("rmsnorm_delta_hotspot_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "hotspot tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "hotspot_focused_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *PINS.values()):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "hotspot diagnostic attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1, "status": "READ_ONLY_RMSNORM_DELTA_HOTSPOTS",
        "command": COMMAND, "pins": {**base.PINS, **PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "diagnostic_sources": origins, "authenticated_files": files, "assets": assets,
        "input_terminal_review": {
            "mission_id": base.MISSION, "round": 2, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done"},
        "attempt001_files_verified": 20, "arrays_verified": 18, "reference_arrays_verified": 4,
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        **evidence[5],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
