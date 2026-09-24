"""Read-only exact coordinate contributions for retained final-head diagnostic IDs.

--check compiles both new Python files in memory, runs the focused tests, and
emits one JSON document. No RMSNorm or full-vocabulary head is recomputed.
"""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_cutoff_margin_v1 as cutoff


base = cutoff.base
parent = base.parent
preflight = parent.preflight
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_coordinate_contributions_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
    f" -B -m {MODULE} --check"
)
PINS = {
    "authenticator": cutoff.AUTHENTICATOR_PIN,
    "cutoff_source": {
        "path": str(cutoff.SOURCE),
        "sha256": "40006dd48fe9604519e412185d7fd0cac9cd2c7ca7d89be759db0c04c9a08f4e",
    },
    "cutoff_test": {
        "path": str(cutoff.TEST),
        "sha256": "faa1502c64daec36ed1e2e1cf0068df2926f4c1475a7470546f05a8ad6ab4e0b",
    },
}
EXPECTED_TESTS = 24
RANK_LIMIT = 8
require, same = base.require, base.same
FLAGS = {
    **cutoff.FLAGS,
    "native_dispatch": 0, "decoder_dispatch": 0, "prefix_dispatch": 0,
    "admission_dispatch": 0, "RTL_dispatch": 0, "GPU_dispatch": 0,
    "hardware_dispatch": 0, "reference_producer_dispatch": 0,
    "full_vocabulary_head_recomputation": False,
    "rmsnorm_recomputation": False, "reference_reanchored": False,
    "token_selection": False, "full_model_claim": False,
}
LINEAGE = {
    "analyzed": "reviewed final-head attempt001, nine-control L21/L22/L23 lineage",
    "historical_sparse_cut": "old sparse-cut attempt002/revision2 pins only",
    "old_adjusted_closure_certifies_new_parents": False,
    "old_states_substituted_or_propagated": False,
    "old_sparse_cut_replayed_or_revalidated": False,
}
BOUNDARY = (
    "Read-only CPU-software exact coordinate arithmetic for numeric diagnostic IDs only. "
    "Retained actual, independent FP16 and independent binary64 RMSNorm vectors remain "
    "distinct original-input trajectories, never recomputed or reanchored. Only selected "
    "tied-head row dot products are regenerated locally; no full-vocabulary head, native, "
    "decoder, prefix, admission, reference producer, RTL, GPU, hardware or simulation "
    "dispatch and no evidence writes. All nine L21/L22/L23 failures, exact thresholds, "
    "source/operand/state/KV/lineage gates and non-admission flags remain unchanged. "
    "The old sparse-cut adjusted closure PASS cannot certify, substitute for or propagate "
    "the new nine-control parent states. No token decode, publication or selection, "
    "upstream root-cause or runtime-bottleneck claim. Q24 residual state is wider than "
    "FP16; native S16 RTZ, G128 asymmetric packed INT4 GEMM ordering without qzero "
    "plus-one, FP16 scales/operator boundaries/KV and the FP16 tied head are unchanged. "
    "No candidate admission, policy adoption, successor publication, strict-FP16-state "
    "W4A16, new-token or full-model claim. Normal independent Reviewer validation REQUIRED."
)


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("coordinate diagnostic forbids GPU dispatch")

    torch = preflight.parent.producer.legacy.torch
    with base.read_only(audit), patch.object(torch.cuda, "_lazy_init", forbidden):
        yield


def authenticate():
    for pin in PINS.values():
        base.read_bound(pin)
    result, arrays, references, files = base.authenticate()
    summary = result["preflight"]
    assets = preflight.bind_assets(summary["assets"]["checkpoint"])
    same(assets, summary["assets"], "live tokenizer/tied-head identities changed")
    require(assets["tokenizer_status"] == "BOUND" and not assets["missing_tokenizer"],
            "tokenizer identity unavailable")
    final = preflight.bind_final_reference(summary["L23_original_reference"], assets)
    same(final, summary["final_reference"], "attempt003 reference authority changed")
    return result, arrays, references, files, assets


def pairs_for(control):
    pairs = {}

    def add(left, right, role):
        require(type(left) is int and type(right) is int and left != right,
                "invalid diagnostic pair")
        pairs.setdefault((left, right), []).append(role)

    for name, comparison in control["comparisons"].items():
        for pair in comparison["cutoff_reversals"]:
            add(pair["actual_only_id"], pair["reference_only_id"], "exchanged_vs_" + name)
    for name, summary in control["summaries"].items():
        add(summary["lowest_included"]["token_id"], summary["highest_excluded"]["token_id"],
            name + "_cutoff_neighbors")
    for entry in control["differing_token_diagnostics"]:
        for name in control["summaries"]:
            add(entry["token_id"], entry[name]["nearest_opposite_membership"]["token_id"],
                name + "_differing_id_opposite_neighbor")
    return pairs


def load_rows(assets, ids):
    require(bool(ids) and all(type(i) is int and 0 <= i < 151936 for i in ids),
            "head row IDs outside diagnostic vocabulary")
    with preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        tensor = model.get_tensor("lm_head.weight")
        require(tensor.shape == (151936, 896) and tensor.dtype.str == "<f2"
                and np.all(np.isfinite(tensor)), "invalid authenticated tied head")
        same(hashlib.sha256(tensor.tobytes()).hexdigest(),
             assets["tensors"]["lm_head.weight"]["sha256"],
             "head row bytes changed after asset authentication")
        return {index: tensor[index].copy() for index in sorted(ids)}


def exact_vector(values):
    require(isinstance(values, np.ndarray) and values.ndim == 1 and len(values) > 0
            and values.dtype.str in ("<f2", "<f8") and np.all(np.isfinite(values)),
            "coordinate operand must be a finite FP16 or binary64 vector")
    return [Fraction.from_float(float(value)) for value in values]


def dyadic_dot(hidden, weights):
    """Independent row-dot path: align integer ratios, then sum Python integers."""
    require(len(hidden) == len(weights) and len(hidden) > 0, "dot operand shape changed")
    terms = []
    for x, w in zip(hidden, weights, strict=True):
        xn, xd = float(x).as_integer_ratio()
        wn, wd = float(w).as_integer_ratio()
        terms.append((xn * wn, xd * wd))
    denominator = max(d for _, d in terms)
    return Fraction(sum(n * (denominator // d) for n, d in terms), denominator)


def ranked(values, limit=RANK_LIMIT):
    require(bool(values) and all(isinstance(value, Fraction) for value in values)
            and type(limit) is int and 0 < limit <= len(values),
            "invalid exact coordinate ranking")
    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))

    def entries(indices):
        return [{"coordinate": i, "signed_contribution": str(values[i])} for i in indices]

    chosen = order[:limit]
    selected = sum((values[i] for i in chosen), Fraction())
    total = sum(values, Fraction())
    return {
        "coordinate_count": len(values), "largest_absolute_signed": entries(chosen),
        "largest_positive": entries(sorted(
            (i for i in order if values[i] > 0), key=lambda i: (-values[i], i))[:limit]),
        "most_negative": entries(sorted(
            (i for i in order if values[i] < 0), key=lambda i: (values[i], i))[:limit]),
        "exact_sum": str(total), "ranked_signed_sum": str(selected),
        "unranked_exact_remainder": str(total - selected),
        "exact_sum_of_absolute_contributions": str(sum(map(abs, values), Fraction())),
        "zero_coordinate_count": sum(value == 0 for value in values),
    }


def analyze_pair(left_id, right_id, vectors, weights, retained, actual_words, limit=RANK_LIMIT):
    require(set(vectors) == set(retained) == {"actual_fp16", "fp16", "binary64"},
            "independent vector branches changed")
    left, right = weights[left_id], weights[right_id]
    require(left.dtype.str == right.dtype.str == "<f2" and left.shape == right.shape,
            "tied-head row shape/dtype changed")
    wl, wr = exact_vector(left), exact_vector(right)
    differences = [a - b for a, b in zip(wl, wr, strict=True)]
    branches, contributions = {}, {}
    for name, hidden in vectors.items():
        h = exact_vector(hidden)
        require(len(h) == len(differences), "RMSNorm/head dimensions differ")
        values = [x * w for x, w in zip(h, differences, strict=True)]
        summary = ranked(values, limit)
        dl, dr = dyadic_dot(hidden, left), dyadic_dot(hidden, right)
        require(Fraction(summary["exact_sum"]) == dl - dr,
                "contribution sum differs from independent row-dot computation")
        margin = cutoff.exact(retained[name][left_id]) - cutoff.exact(retained[name][right_id])
        if name == "actual_fp16":
            for index, value in ((left_id, dl), (right_id, dr)):
                q48 = value * (1 << 48)
                require(q48.denominator == 1, "actual row sum is not exact Q48")
                word, saturated = parent.head.fixed_to_f16(q48.numerator, 48)
                require(not saturated and word == int(actual_words[index]),
                        "local exact-head row does not reproduce retained actual logit")
        branches[name] = {
            **summary, "independent_left_dot": str(dl), "independent_right_dot": str(dr),
            "independent_dot_identity": True, "retained_margin": str(margin),
            "head_accumulation_rounding_remainder": str(margin - (dl - dr)),
            "retained_margin_tie": margin == 0, "exact_margin_tie": dl == dr,
        }
        contributions[name] = values
    effects = {}
    for name in ("fp16", "binary64"):
        values = [a - r for a, r in zip(
            contributions["actual_fp16"], contributions[name], strict=True)]
        summary = ranked(values, limit)
        a, r = branches["actual_fp16"], branches[name]
        am, rm = Fraction(a["retained_margin"]), Fraction(r["retained_margin"])
        vector_change = Fraction(summary["exact_sum"])
        boundary_change = (Fraction(a["head_accumulation_rounding_remainder"])
                           - Fraction(r["head_accumulation_rounding_remainder"]))
        require(am - rm == vector_change + boundary_change,
                "coordinate/boundary retained-margin decomposition failed")
        effects[name] = {
            **summary, "reference_retained_margin": str(rm),
            "margin_after_ranked_coordinate_changes": str(rm + Fraction(summary["ranked_signed_sum"])),
            "margin_after_all_coordinate_changes": str(rm + vector_change),
            "boundary_remainder_change": str(boundary_change),
            "retained_actual_margin": str(am), "retained_relative_change": str(am - rm),
            "exact_additive_identity": True,
        }
    return {"left_id": left_id, "right_id": right_id, "branches": branches,
            "actual_minus_independent_reference": effects}


def report(result, arrays, references, geometry, weights):
    base.check_history(result)
    same(list(arrays), list(parent.CONTROLS), "coordinate control census changed")
    rows, row_dots = [], 0
    for control in geometry["controls"]:
        label = control["control"]
        vectors = {"actual_fp16": arrays[label]["rmsnorm"].view("<f2"),
                   "fp16": references["rmsnorm_fp16"].view("<f2"),
                   "binary64": references["rmsnorm_binary64"]}
        retained = {"actual_fp16": arrays[label]["logits"].view("<f2"),
                    "fp16": references["logits_fp16"].view("<f2"),
                    "binary64": references["logits_binary64"]}
        pairs = []
        for (left, right), roles in pairs_for(control).items():
            pairs.append({**analyze_pair(left, right, vectors, weights, retained,
                                        arrays[label]["logits"]), "roles": roles})
            row_dots += 2 * len(vectors)
        rows.append({**control, "coordinate_pairs": pairs})
    return {
        **geometry, "controls": rows, "local_exact_row_dot_computations": row_dots,
        "lineage_separation": LINEAGE,
        "arithmetic": {
            "coordinate": "h[i] * (W[left_id,i] - W[right_id,i]), exact rational",
            "vector_change": "(actual_h[i] - independent_h[i]) * (W[left_id,i] - W[right_id,i])",
            "margin_orientation": "left numeric diagnostic ID minus right numeric diagnostic ID",
            "ranking": "absolute magnitude descending, coordinate ascending on ties; signed values",
            "rank_limit": RANK_LIMIT,
            "identity": "actual retained margin = reference retained margin + coordinate changes + boundary remainder change",
            "boundary_remainder": "retained margin minus exact row-dot margin; not assigned solely to rounding",
            "partial_margin": "additive accounting only, not an executed vector splice or token selection",
            "reference_semantics": "independently propagated FP16 and binary64 RMSNorms, not casts of each other",
            "threshold_semantics": "rational logit units; no new acceptance threshold or causal percentage",
        },
    }


def measure():
    result, arrays, references, files, assets = authenticate()
    geometry = cutoff.report(result, arrays, references)
    ids = {index for control in geometry["controls"] for pair in pairs_for(control) for index in pair}
    weights = load_rows(assets, ids)
    measured = report(result, arrays, references, geometry, weights)
    return (result, arrays, references, geometry, weights, measured), files, assets


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("coordinate_contribution_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "coordinate tests failed, errored or skipped")
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
               "coordinate_focused_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence, files, assets = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *PINS.values()):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "coordinate diagnostic attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_COORDINATE_CONTRIBUTION_DIAGNOSTIC",
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
        **evidence[5],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
