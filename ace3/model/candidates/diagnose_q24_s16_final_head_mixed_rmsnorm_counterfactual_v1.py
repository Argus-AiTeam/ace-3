"""Read-only CPU final-head counterfactuals from authenticated retained RMSNorms.

--check compiles both new Python files in memory, runs only the focused tests,
and emits one JSON document. No RMSNorm, decoder or original prefix is replayed.
"""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_cutoff_margin_v1 as cutoff


base = cutoff.base
parent = base.parent
preflight = parent.preflight
torch = preflight.parent.producer.legacy.torch
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_mixed_rmsnorm_counterfactual_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (
    f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
    f" -B -m {MODULE} --check"
)
CUTOFF_PIN = {
    "path": str(cutoff.SOURCE),
    "sha256": "40006dd48fe9604519e412185d7fd0cac9cd2c7ca7d89be759db0c04c9a08f4e",
}
EXPECTED_TESTS = 24
require, same = base.require, base.same
POLICIES = ("exact_rne", "torch_forward", "torch_reverse")
FLAGS = {
    **{key: value for key, value in cutoff.FLAGS.items() if key != "lm_head_invocations"},
    "counterfactual_final_head_only": True,
    "rmsnorm_recomputation": False,
    "original_reference_recomputation": False,
    "reference_reanchored": False,
    "token_selection": False,
    "full_model_claim": False,
}
BOUNDARY = (
    "Read-only CPU-software counterfactual tied-head computation from retained actual "
    "and independently propagated FP16 RMSNorm vectors, not a new reference or admission "
    "run. Retained binary64 RMSNorm/logits are comparison anchors, never repropagated. "
    "All nine L21/L22/L23 failures, exact thresholds and source/operand/state/KV/lineage "
    "gates remain unchanged. No RMSNorm, native/decoder/prefix/admission/reference "
    "producer, RTL/GPU/hardware/simulation or external dispatch; no evidence writes, "
    "token decoding, publication or selection. Numeric IDs are diagnostic only. "
    "Q24 state is wider than FP16; native S16 RTZ, G128 asymmetric packed INT4 GEMM "
    "ordering without qzero plus-one, FP16 scales/operator boundaries/KV are unchanged. "
    "The official tied head is FP16. No strict-FP16-state W4A16, new-token, full-model, "
    "candidate admission, policy adoption or successor publication claim. Controlled "
    "effects are scoped to this final-head comparison, not an upstream root cause or "
    "runtime bottleneck. Normal independent Reviewer validation is REQUIRED."
)


def authenticate():
    base.read_bound(cutoff.AUTHENTICATOR_PIN)
    base.read_bound(CUTOFF_PIN)
    result, arrays, references, files = base.authenticate()
    summary = result["preflight"]
    assets = preflight.bind_assets(summary["assets"]["checkpoint"])
    same(assets, summary["assets"], "live tied-head/tokenizer identity changed")
    final = preflight.bind_final_reference(summary["L23_original_reference"], assets)
    same(final, summary["final_reference"], "final reference authority changed")
    manifest = json.loads(base.read_bound(final["manifest"]))
    same(manifest["arithmetic"]["runtime"],
         {"torch": torch.__version__, "numpy": np.__version__},
         "reviewed CPU numerical runtime changed")
    return result, arrays, references, files, assets


def exact_head(words, weights):
    require(words.dtype.str == "<u2" and words.ndim == 1
            and weights.dtype.str == "<f2" and weights.ndim == 2
            and weights.shape[1] == len(words)
            and np.all(np.isfinite(words.view("<f2")))
            and np.all(np.isfinite(weights)), "invalid mixed-head operands")
    hidden = parent.head.decode_array_q24(words)
    decoded = parent.head.decode_array_q24(weights.view("<u2"))
    bound = int(np.max(np.abs(decoded)))
    require(bound * sum(map(abs, map(int, hidden))) <= (1 << 63) - 1,
            "counterfactual exact Q48 int64 accumulation bound exceeded")
    sums = np.sum(decoded * hidden, axis=1, dtype=np.int64)
    rounded = np.empty(len(weights), dtype="<u2")
    for index, total in enumerate(sums):
        bits, saturated = parent.head.fixed_to_f16(int(total), 48)
        require(not saturated, "counterfactual final-head saturation")
        rounded[index] = bits
    return sums, rounded


def float_head(words, weights, *, reverse=False):
    x = words.view("<f2").astype("<f8")
    w = weights.astype("<f8")
    if reverse:
        x, w = x[::-1].copy(), w[:, ::-1].copy()
    values = (torch.from_numpy(w) @ torch.from_numpy(x)).numpy()
    rounded = torch.from_numpy(values).to(torch.float16).numpy().view("<u2")
    direct = values.astype("<f2").view("<u2")
    require(np.all(np.isfinite(values)) and np.all(np.isfinite(rounded.view("<f2")))
            and np.all(np.isfinite(direct.view("<f2"))), "nonfinite counterfactual head")
    return values, rounded, direct


def inputs_for(arrays, references):
    inputs, aliases, seen = {}, {}, {}
    for label in parent.CONTROLS:
        words = arrays[label]["rmsnorm"]
        key = words.tobytes()
        if key not in seen:
            seen[key] = label
            inputs[label] = words
        aliases[label] = seen[key]
    inputs["independent_fp16"] = references["rmsnorm_fp16"]
    return inputs, aliases


def compute(inputs, assets, witness_ids):
    outputs = {
        name: {
            "q48": np.empty(151936, dtype="<i8"),
            **{key: np.empty(151936, dtype="<u2") for key in POLICIES},
            **{key: np.empty(151936, dtype="<f8")
               for key in ("forward_unrounded", "reverse_unrounded")},
            **{key: np.empty(151936, dtype="<u2")
               for key in ("forward_direct_rne", "reverse_direct_rne")},
        } for name in inputs}
    timings = {key: 0.0 for key in ("operand_read", *POLICIES)}
    digest, witnesses = hashlib.sha256(), {}
    with preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        tensor = model.get_slice("lm_head.weight")
        same(tensor.get_shape(), [151936, 896], "counterfactual head shape changed")
        for start in range(0, 151936, 2048):
            stop = min(start + 2048, 151936)
            began = time.monotonic()
            weights = np.asarray(tensor[start:stop])
            require(weights.dtype.str == "<f2" and np.all(np.isfinite(weights)),
                    "invalid tied-head chunk")
            digest.update(weights.tobytes())
            for index in witness_ids:
                if start <= index < stop:
                    witnesses[index] = weights[index - start].copy()
            timings["operand_read"] += time.monotonic() - began
            for name, words in inputs.items():
                target = outputs[name]
                began = time.monotonic()
                target["q48"][start:stop], target["exact_rne"][start:stop] = exact_head(words, weights)
                timings["exact_rne"] += time.monotonic() - began
                for reverse, policy, prefix in (
                        (False, "torch_forward", "forward"),
                        (True, "torch_reverse", "reverse")):
                    began = time.monotonic()
                    values, rounded, direct = float_head(words, weights, reverse=reverse)
                    target[policy][start:stop] = rounded
                    target[prefix + "_unrounded"][start:stop] = values
                    target[prefix + "_direct_rne"][start:stop] = direct
                    timings[policy] += time.monotonic() - began
    same(digest.hexdigest(), assets["tensors"]["lm_head.weight"]["sha256"],
         "computed tied-head bytes changed after authentication")
    return outputs, witnesses, timings


def difference(a, b):
    require(a.shape == b.shape and a.ndim == 1
            and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
            "invalid difference vectors")
    errors = np.abs(a - b)
    index = int(np.argmax(errors))
    return {"mismatch_count": int(np.count_nonzero(a != b)), "worst_index": index,
            "max_absolute_error": str(cutoff.exact(a[index]) - cutoff.exact(b[index])
                                      if a[index] >= b[index]
                                      else cutoff.exact(b[index]) - cutoff.exact(a[index]))}


def pair_effect(gained, lost, actual, reference, actual_words, reference_words):
    def exact_margin(output):
        return Fraction(int(output["q48"][gained]) - int(output["q48"][lost]), 1 << 48)

    def margin(words):
        values = words.view("<f2")
        return cutoff.exact(values[gained]) - cutoff.exact(values[lost])

    ea, er = exact_margin(actual), exact_margin(reference)
    a, r = margin(actual_words), margin(reference_words)
    vector_effect = ea - er
    actual_head_effect, reference_head_effect = a - ea, r - er
    combined_head_effect = actual_head_effect - reference_head_effect
    require(a - r == vector_effect + combined_head_effect, "mixed-head decomposition failed")
    return {
        "actual_only_id": gained, "reference_only_id": lost,
        "actual_exact_margin": str(ea), "reference_exact_margin": str(er),
        "retained_actual_margin": str(a), "retained_reference_margin": str(r),
        "rmsnorm_vector_effect_fixed_exact_head": str(vector_effect),
        "actual_head_rounding_effect": str(actual_head_effect),
        "reference_accumulation_and_rounding_effect": str(reference_head_effect),
        "combined_head_effect": str(combined_head_effect),
        "total_retained_relative_change": str(a - r),
        "exact_additive_identity": True,
        "vector_swap_reverses_pair_under_exact_head": ea > 0 > er,
        "vector_share_of_signed_change": str(vector_effect / (a - r)) if a != r else None,
        "head_share_of_signed_change": str(combined_head_effect / (a - r)) if a != r else None,
        "exact_margin_tie": ea == 0 or er == 0,
        "scope": "signed additive effects for this pair, not nonnegative causal percentages",
    }


def branch_summary(output, reference_profiles):
    profiles = {key: cutoff.profile(output[key].view("<f2").astype("<f8"), 10)
                for key in (*POLICIES, "forward_direct_rne", "reverse_direct_rne")}
    exact_values = np.ldexp(output["q48"].astype("<f8"), -48)
    summary = {
        key: {
            **ranked["summary"],
            "vs_retained_references": {
                ref: cutoff.compare_profiles(ranked, profile, 10)
                for ref, profile in reference_profiles.items()},
            "word_mismatches_vs_exact_rne": int(np.count_nonzero(output[key] != output["exact_rne"])),
        } for key, ranked in profiles.items()}
    # An exact binary64 comparison is valid here only if the Q48 integer converts exactly.
    require(all(int(float(value)) == int(value) for value in output["q48"]),
            "Q48 sums are not exactly representable for binary64 order comparison")
    return {
        "policies": summary,
        "forward_accumulation_vs_exact": difference(output["forward_unrounded"], exact_values),
        "reverse_accumulation_vs_exact": difference(output["reverse_unrounded"], exact_values),
        "order_effect_before_rounding": difference(
            output["forward_unrounded"], output["reverse_unrounded"]),
        "order_effect_fp16_word_count": int(np.count_nonzero(
            output["torch_forward"] != output["torch_reverse"])),
        "forward_torch_vs_direct_rounding_word_count": int(np.count_nonzero(
            output["torch_forward"] != output["forward_direct_rne"])),
        "reverse_torch_vs_direct_rounding_word_count": int(np.count_nonzero(
            output["torch_reverse"] != output["reverse_direct_rne"])),
    }


def logit_reproduction(actual, retained):
    require(actual.dtype.str == retained.dtype.str == "<u2"
            and actual.shape == retained.shape and actual.ndim == 1,
            "invalid retained-logit comparison operands")
    mismatches = np.flatnonzero(actual != retained)
    first = int(mismatches[0]) if len(mismatches) else None
    return {
        "word_count": len(actual), "word_mismatch_count": len(mismatches),
        "first_counterexample": None if first is None else {
            "index": first, "computed_word": int(actual[first]),
            "retained_word": int(retained[first])},
    }


def classify_mechanism(rows, branch_count, reproduction=None):
    pairs = [pair for row in rows for pair in row["pair_effects"]]
    fields = (
        "actual_exact_margin", "reference_exact_margin", "total_retained_relative_change",
        "rmsnorm_vector_effect_fixed_exact_head", "combined_head_effect",
    )
    closed, reversed_pairs = 0, 0
    for pair in pairs:
        require(all(key in pair for key in (*fields, "actual_only_id", "reference_only_id",
                                            "exact_additive_identity",
                                            "vector_swap_reverses_pair_under_exact_head")),
                "malformed mixed-head pair account")
        require(all(type(pair[key]) is int and 0 <= pair[key] < 151936
                    for key in ("actual_only_id", "reference_only_id"))
                and pair["actual_only_id"] != pair["reference_only_id"],
                "malformed mixed-head pair orientation")
        ea, er, total, vector, head = (Fraction(pair[key]) for key in fields)
        identity = total == vector + head and vector == ea - er
        reversal = ea > 0 > er
        require(pair["exact_additive_identity"] is identity
                and pair["vector_swap_reverses_pair_under_exact_head"] is reversal,
                "mixed-head pair flags contradict exact margins")
        closed += identity
        reversed_pairs += reversal
    complete = ([row["control"] for row in rows] == list(parent.CONTROLS)
                and branch_count == 3 and all(len(row["pair_effects"]) == 1 for row in rows))
    counterexamples = {
        name: comparison for name, comparison in (reproduction or {}).items()
        if comparison["word_mismatch_count"]}
    supported = (complete and closed == reversed_pairs == len(parent.CONTROLS)
                 and not counterexamples)
    return {
        "classification": "SUPPORTED" if supported else "REJECTED",
        "complete_nine_control_three_branch_pair_census": complete,
        "exchanged_pair_count": len(pairs),
        "closed_additive_pair_count": closed,
        "exact_head_vector_swap_reversal_count": reversed_pairs,
        "logit_policy_counterexamples": counterexamples,
        "reason": ("all retained pair changes close and reverse under the fixed exact head"
                   if supported else "retained logit/policy reproduction failed"
                   if counterexamples else
                   "pair census, additive closure or exact-head vector reversal absent"),
        "scope": "bounded final-head mechanism only; no upstream root cause or admission",
    }


def report(result, arrays, references, inputs, aliases, outputs):
    base.check_history(result)
    same(list(arrays), list(parent.CONTROLS), "mixed control census changed")
    reference_profiles = {
        "binary64": cutoff.profile(references["logits_binary64"], 10),
        "fp16": cutoff.profile(references["logits_fp16"].view("<f2").astype("<f8"), 10)}
    ref = outputs["independent_fp16"]
    reproduction = {
        "independent_fp16_torch_forward": logit_reproduction(
            ref["torch_forward"], references["logits_fp16"])}
    branches = {name: branch_summary(output, reference_profiles) for name, output in outputs.items()}
    rows = []
    for retained in result["controls"]:
        label = retained["control"]
        actual = outputs[aliases[label]]
        reproduction[label + "_exact_rne"] = logit_reproduction(
            actual["exact_rne"], arrays[label]["logits"])
        ranked = cutoff.profile(arrays[label]["logits"].view("<f2").astype("<f8"), 10)
        comparisons = {
            name: cutoff.compare_profiles(ranked, profile, 10)
            for name, profile in reference_profiles.items()}
        for name, comparison in comparisons.items():
            for key in ("overlap_count", "same_order"):
                same(comparison[key], retained["comparisons"]["top_k"][name][key],
                     "retained top-k boundary changed")
        effects = [
            pair_effect(gained, lost, actual, ref, arrays[label]["logits"],
                        references["logits_fp16"])
            for gained in comparisons["fp16"]["actual_only_ids"]
            for lost in comparisons["fp16"]["reference_only_ids"]]
        l23 = retained["parent"]["retained_L23"]
        rows.append({
            "control": label, "actual_input_branch": aliases[label],
            "reference_input_branch": "independent_fp16",
            "rmsnorm": base.metric(arrays[label]["rmsnorm"], references["rmsnorm_binary64"],
                                   references["rmsnorm_fp16"]),
            "retained_top_k": {"actual": ranked["summary"], "comparisons": comparisons},
            "mixed_top_k": {
                "actual_rmsnorm": branches[aliases[label]]["policies"],
                "independent_fp16_rmsnorm": branches["independent_fp16"]["policies"]},
            "pair_effects": effects,
            "L21_L22_L23_status": ["FAIL"] * 3,
            "retained_failures": {
                "L21": l23["retained_L21"]["failures"], "L22": [l23["L22_failure"]],
                "L23": retained["parent"]["L23_failures"]},
            "retained_L23_and_ancestral_lineage": l23,
        })
    return {
        "control_count": len(rows), "controls": rows, "k": 10,
        "counterfactual_input_count": len(inputs), "counterfactual_branches": branches,
        "mechanism": classify_mechanism(rows, len(inputs), reproduction),
        "retained_logit_reproduction": reproduction,
        "lane_terminated": True,
        "reference_top_k": {key: p["summary"] for key, p in reference_profiles.items()},
        "reference_rmsnorm_fp16_vs_binary64": difference(
            references["rmsnorm_fp16"].view("<f2").astype("<f8"), references["rmsnorm_binary64"]),
        "retained_L23_failures": 9,
        "retained_L23_stage_reports": result["preflight"]["retained_L23_stage_reports"],
        "thresholds": result["preflight"]["thresholds"],
        "original_global_reference": result["preflight"]["final_reference"],
        "retained_flags": result["flags"],
        "arithmetic": {
            "exact_rne": "exact FP16 products, bounded int64 Q48 sum, one FP16 RNE",
            "torch_forward": "CPU binary64 matvec, original 2048-row chunks, torch FP16 boundary",
            "torch_reverse": "same matvec with both operands' 896 coordinates reversed",
            "direct_rne": "NumPy direct binary64-to-FP16 conversion of each same matvec",
            "reference_fp16_policy": "independent retained RMSNorm; frozen torch cast via float32",
            "order": "descending finite value, ascending numeric diagnostic ID on ties",
            "margin_units": "exact rational logit differences, not acceptance thresholds",
            "binary64_reference": "retained anchors only; no binary64 RMSNorm/head regeneration",
        },
    }


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("mixed_rmsnorm_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    require(suite.countTestCases() == EXPECTED_TESTS, "focused test count changed")
    progress = io.StringIO()
    outcome = unittest.TextTestRunner(stream=progress, verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        print(progress.getvalue(), file=sys.stderr, end="")
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "mixed RMSNorm tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors),
            "skipped": len(outcome.skipped), "output": progress.getvalue()}


def measure():
    audit, started = {"forbidden_calls": 0}, time.monotonic()
    with base.read_only(audit):
        result, arrays, references, files, assets = authenticate()
        auth_done = time.monotonic()
        inputs, aliases = inputs_for(arrays, references)
        witnesses = {0, 151935}
        for values in [a["logits"].view("<f2").astype("<f8") for a in arrays.values()] + [
                references["logits_fp16"].view("<f2").astype("<f8"),
                references["logits_binary64"]]:
            witnesses.update(map(int, cutoff.profile(values, 10)["order"][:11]))
        outputs, weights, phases = compute(inputs, assets, witnesses)
        computed = time.monotonic()
        measured = report(result, arrays, references, inputs, aliases, outputs)
        same(audit, {"forbidden_calls": 0}, "mixed diagnostic attempted forbidden dispatch/write")
    evidence = (result, arrays, references, inputs, aliases, outputs, weights, measured)
    return measured, evidence, {
        "authenticated_files": files, "assets": assets,
        "pins": {**base.PINS, "authenticator": cutoff.AUTHENTICATOR_PIN, "cutoff_helper": CUTOFF_PIN},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "input_terminal_review": {
            "mission_id": base.MISSION, "round": 2, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done"},
        "dispatch_and_write_audit": {
            **audit, **FLAGS, "counterfactual_final_head_invocations": len(inputs) * len(POLICIES)},
        "phase_seconds": {
            "authentication": auth_done - started, "counterfactual_compute": computed - auth_done,
            "summaries": time.monotonic() - computed, "compute_components": phases},
    }


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    torch.set_num_threads(1)
    require(torch.get_num_threads() == 1, "one CPU thread required")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "mixed_rmsnorm_focused_test": parent.record(TEST)}
    measured, evidence, authentication = measure()
    audit = {"forbidden_calls": 0}
    with base.read_only(audit):
        tests = focused_tests(evidence)
        for pin in origins.values():
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "focused tests attempted unguarded work")
    return {
        "diagnostic_id": NAME, "version": 1, "status": "READ_ONLY_MIXED_RMSNORM_COUNTERFACTUAL",
        "command": COMMAND, "diagnostic_sources": origins, **authentication, **measured,
        "flags": FLAGS, "tests": tests, "normal_host_review": "REQUIRED",
        "claim_boundary": BOUNDARY,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        result = check()
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({
            "diagnostic_id": NAME, "version": 1, "status": "UNKNOWN",
            "mechanism": {"classification": "UNKNOWN"},
            "failure": {"type": type(error).__name__, "message": str(error)},
            "flags": FLAGS, "lane_terminated": False,
            "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        }, sort_keys=True, allow_nan=False))
        print(f"{NAME}: UNKNOWN: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
