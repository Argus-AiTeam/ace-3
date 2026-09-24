"""Retained-only conversion-policy top-k stability; no numerical producers."""

import argparse
import hashlib
import json
import math
import shlex
from fractions import Fraction
from pathlib import Path

from ace3.model.candidates import diagnostic_capture_v1 as capture

ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_mixed_rmsnorm_conversion_topk_stability_v1"
PARENT = "diagnose_q24_s16_final_head_mixed_rmsnorm_counterfactual_v1"
RETAINED = ROOT / "build/mixed-rmsnorm-final-head-a399e916489f-attempt002"
OUTER_SHA = "799a2485234b4a0057f4052315cffe18bb744cb9bb065f94c0048d8df37b57dc"
RUN_SHA = "f4085f2d71f1a3e88ae91180484d4bf23ad949df592e591efef86babbda64b6b"
POLICIES = ("exact_rne", "torch_forward", "torch_reverse",
            "forward_direct_rne", "reverse_direct_rne")
BRANCHES = ("frozen_inherited", "independent_fp16", "mapped_all")
REFERENCES = ("fp16", "binary64")
FALSE_FLAGS = (
    "accepted_prefix_replay admission_replay candidate_admitted "
    "direct_binary64_to_fp16_rounding_claimed full_model_claim new_token_claim "
    "original_prefix_replay original_reference_recomputation policy_adopted "
    "reference_reanchored reference_reanchoring reference_recomputation "
    "rmsnorm_recomputation strict_FP16_state_claim successor_published "
    "token_published token_selected_for_feedback token_selection upstream_cause_identified"
).split()
ZERO_FLAGS = (
    "admission_invocations decoder_invocations evidence_writes external_invocations "
    "external_service_invocations final_rmsnorm_invocations gpu_invocations "
    "hardware_invocations native_L0_L22_invocations native_L0_L23_invocations "
    "native_layer_invocations prefix_invocations reference_lm_head_invocations "
    "reference_rmsnorm_invocations retained_evidence_writes rtl_invocations "
    "simulation_invocations tokenizer_decode_invocations"
).split()
PARENT_FLAGS = {**dict.fromkeys(FALSE_FLAGS, False), **dict.fromkeys(ZERO_FLAGS, 0),
                "counterfactual_final_head_only": True}
FLAGS = {**PARENT_FLAGS, "counterfactual_final_head_only": False,
         "counterfactual_final_head_invocations": 0, "head_matvec_invocations": 0,
         "full_vocabulary_producer_invocations": 0, "retained_capture_only": True}
PRESERVED = (
    "claim_boundary", "flags", "retained_flags", "dispatch_and_write_audit",
    "controls", "retained_L23_failures", "retained_L23_stage_reports", "thresholds",
    "original_global_reference", "original_execution_sources", "diagnostic_sources",
    "pins", "authenticated_files", "input_terminal_review", "mechanism",
)
BOUNDARY = (
    "CPU retained-check.stdout-only conversion-policy relevance, not policy adoption "
    "or admission. No producer, prefix, RMSNorm, head matvec, full-vocabulary, "
    "reference, model, token, service or hardware computation. Original-input "
    "references, thresholds and source/operand/state/KV/lineage gates are unchanged. "
    "Q24 residual state is wider than FP16; native-S16-RTZ, INT4 weights and FP16 "
    "operator boundaries/KV remain unchanged. No strict-FP16-state W4A16, new-token "
    "or full-model claim. Normal independent Reviewer validation is REQUIRED."
)


class BindingError(ValueError):
    pass


def require(condition, fact):
    if not condition:
        raise BindingError(fact)


def same(actual, expected, fact):
    require(capture.encoded(actual) == capture.encoded(expected), fact)


def read_pin(pin, path):
    require(pin["path"] == str(path), f"capture member path: {path}")
    require(path.resolve() == path, f"capture member symlink: {path}")
    raw = path.read_bytes()
    same(pin, {"path": str(path), "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest()}, f"byte binding: {path}")
    return raw


def fixed_pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


def validate_command(command, argv, environment, fact):
    tokens = shlex.split(command)
    count = len(environment)
    require(len(tokens) >= count and all("=" in token for token in tokens[:count]), fact)
    # JSON object serialization sorts keys; shell assignment order is not identity.
    same(dict(token.split("=", 1) for token in tokens[:count]), environment, fact)
    same(tokens[count:], argv, fact)


def load_capture():
    """Authenticate transport/source bindings, never import the retained producer."""
    outer_pin = fixed_pin(RETAINED / "launcher.capture.json", 1470, OUTER_SHA)
    outer = json.loads(read_pin(outer_pin, RETAINED / "launcher.capture.json"))
    identity = json.loads(read_pin(outer["files"][0], RETAINED / "launcher.identity.json"))
    for pin in (*identity["capture_implementation"], identity["source"],
                identity["launcher"], identity["executable"]):
        read_pin(pin, Path(pin["path"]))
    try:
        capture.verify_launcher(RETAINED, identity, outer_pin)
    except RuntimeError as error:
        raise BindingError(str(error)) from error
    run_dir = RETAINED / "run"
    run_pin = fixed_pin(run_dir / "capture.json", 18491, RUN_SHA)
    link = json.loads(read_pin(outer["files"][1], RETAINED / "launcher.stdout"))
    same(link["capture"], run_pin, "launcher-to-run receipt binding")
    require(link["success"] is True and link["failure"] is None,
            "launcher-to-run terminal state")
    receipt = json.loads(read_pin(run_pin, run_dir / "capture.json"))
    preflight = receipt["preflight"]
    same(receipt["sources_after"], preflight["sources"], "retained source drift")
    same(preflight["sources"], [identity["source"], identity["launcher"]],
         "launcher/run source and test pins")
    for key in ("cwd", "uid", "environment", "executable", "scope",
                "capture_implementation", "independent_host_review",
                "model_or_service_calls_authorized"):
        same(preflight[key], identity[key], f"launcher/run {key}")
    same(preflight["cwd"], str(ROOT), "isolated retained cwd")
    same(preflight["uid"], 1000, "retained account")
    same(preflight["mission_id"], "a399e916489f", "retained mission")
    same(preflight["role"], "engineer", "retained role")
    same(preflight["scope"], "ace3.model.candidates." + PARENT, "retained scope")
    same(preflight["model_or_service_calls_authorized"], 0, "retained service budget")
    same(preflight["independent_host_review"], "REQUIRED", "retained review gate")
    same(preflight["command_budget"], {"compile": 1, "pytest": 1, "check": 1},
         "retained command budget")
    require(receipt["success"] is True and receipt["failure"] is None,
            "retained run terminal state")
    same(receipt["classification"], link["classification"], "run classification binding")
    labels = ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"]
    same([step["label"] for step in receipt["results"]], labels, "retained command census")
    check_bytes = None
    for step in receipt["results"]:
        label = step["label"]
        same(step["exit_status"], 0, f"{label} exit status")
        require(step["timed_out"] is False and "launch_error" not in step,
                f"{label} completion")
        paths = [run_dir / (label + suffix) for suffix in capture.SUFFIXES]
        require(len(step["files"]) == len(paths), f"{label} capture member census")
        command, argv, environment, out, err, whole = [
            read_pin(pin, path) for pin, path in zip(step["files"], paths, strict=True)]
        validate_command(step["command"], step["argv"], preflight["environment"],
                         f"{label} command/environment identity")
        require(command == (step["command"] + "\n").encode()
                and argv == capture.encoded(step["argv"])
                and environment == capture.encoded(preflight) and not err,
                f"{label} command/argv/environment/streams")
        require(whole == b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
                + b"\nSTDOUT\n" + out + b"\nSTDERR\n" + err
                + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", f"{label} whole capture framing")
        if label == "branch":
            require(out == b"argus/full-projection\n", "retained branch")
        elif label == "concurrency":
            require(out == b"[]\n", "retained same-scope concurrency")
        elif label == "check":
            same(step["argv"], [preflight["python"], "-B", "-m",
                                preflight["scope"], "--check"], "retained diagnostic argv")
            check_bytes = out
    require(check_bytes is not None, "retained check.stdout")
    document = json.loads(check_bytes)
    same(document["diagnostic_id"], PARENT, "retained diagnostic identity")
    same(document["mechanism"]["classification"], receipt["classification"],
         "retained scientific/transport classification")
    same(document["tests"]["compiled"], preflight["sources"], "retained compile pins")
    same(document["tests"]["executed"], 24, "retained test census")
    for key in ("errors", "failures", "skipped"):
        same(document["tests"][key], 0, f"retained tests {key}")
    for key in ("diagnostic_sources", "original_execution_sources"):
        for pin in document[key].values():
            path = Path(pin["path"])
            require(path.is_relative_to(ROOT), f"retained source outside isolated root: {path}")
            read_pin(pin, path)
    same(document["diagnostic_sources"]["ace3.model.candidates." + PARENT],
         identity["source"], "retained producer source pin")
    same(document["diagnostic_sources"]["mixed_rmsnorm_focused_test"],
         identity["launcher"], "retained producer test pin")
    return document, {"launcher": outer_pin, "run": run_pin,
                      "check_stdout": receipt["results"][-1]["files"][3]}


def fraction(value, fact):
    require(isinstance(value, str), fact)
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise BindingError(fact) from error


def known_rank(profile, identifier):
    ids = profile["top_k_ids_diagnostic_only"]
    if identifier in ids:
        return ids.index(identifier) + 1
    if profile["highest_excluded"]["token_id"] == identifier:
        return 11
    return None


def validate_profile(profile, fact):
    ids = profile["top_k_ids_diagnostic_only"]
    require(isinstance(ids, list) and len(ids) == 10
            and all(type(i) is int and 0 <= i < 151936 for i in ids)
            and len(set(ids)) == 10, fact + " top-10 IDs")
    require(profile["all_finite"] is True and profile["elements"] == 151936,
            fact + " finite vocabulary")
    low, high = profile["lowest_included"], profile["highest_excluded"]
    same(low["rank"], 10, fact + " included rank")
    same(high["rank"], 11, fact + " excluded rank")
    require(low["token_id"] == ids[-1] and type(high["token_id"]) is int
            and 0 <= high["token_id"] < 151936 and high["token_id"] not in ids,
            fact + " cutoff IDs")
    values = [float.fromhex(entry["value_hex"]) for entry in (low, high)]
    require(all(map(math.isfinite, values)), fact + " finite cutoff values")
    gap = fraction(profile["cutoff_gap"], fact + " cutoff gap")
    require(gap == Fraction(values[0]) - Fraction(values[1]) and gap >= 0,
            fact + " exact cutoff gap")
    same(profile["cutoff_tie"], gap == 0, fact + " cutoff tie")
    require(gap != 0 or low["token_id"] < high["token_id"], fact + " numeric tie order")


def validate_comparison(profile, reference, comparison, fact):
    a, r = profile["top_k_ids_diagnostic_only"], reference["top_k_ids_diagnostic_only"]
    gained, lost = sorted(set(a) - set(r)), sorted(set(r) - set(a))
    same(comparison["actual_only_ids"], gained, fact + " exchanged actual IDs")
    same(comparison["reference_only_ids"], lost, fact + " exchanged reference IDs")
    same(comparison["overlap_count"], len(set(a) & set(r)), fact + " overlap")
    same(comparison["same_order"], a == r, fact + " order")
    reversals = comparison["cutoff_reversals"]
    same([(p["actual_only_id"], p["reference_only_id"]) for p in reversals],
         [(g, l) for g in gained for l in lost], fact + " reversal census")
    for pair in reversals:
        am, rm, gc, lc, delta, closure = [
            fraction(pair[key], fact + " " + key) for key in (
                "actual_preference_margin", "reference_preference_margin",
                "actual_only_value_change", "reference_only_value_change",
                "relative_value_change", "relative_change_minus_reference_margin")]
        require(am >= 0 and rm >= 0 and delta == gc - lc == am + rm and closure == am,
                fact + " reversal rational closure")
        require(pair["exact_reversal_identity"] is True, fact + " reversal identity")
        same(pair["tie_at_either_boundary"], am == 0 or rm == 0, fact + " reversal tie")
        require(type(pair["adjacent_cutoff_in_both"]) is bool
                and type(pair["max_distance_from_cutoff_rank"]) is int
                and 1 <= pair["max_distance_from_cutoff_rank"] < 151936,
                fact + " reversal rank constraints")


def selected_rank(profile, references, identifier, fact):
    rank = known_rank(profile, identifier)
    if rank is not None:
        return {"present": rank <= 10, "rank": rank, "basis": "retained top-10/cutoff"}
    candidates = set()
    for name, reference in references.items():
        for pair in profile["vs_retained_references"][name]["cutoff_reversals"]:
            gained, lost = pair["actual_only_id"], pair["reference_only_id"]
            if identifier != lost:
                continue
            other = [known_rank(profile, gained), known_rank(reference, gained),
                     known_rank(reference, lost)]
            distance = pair["max_distance_from_cutoff_rank"]
            # An excluded non-cutoff ID is >=12. If all three known distances
            # are smaller, it alone attains the retained maximum.
            if all(r is not None for r in other) and max(abs(r - 10) for r in other) < distance:
                require(distance >= 2, fact + " excluded rank constraint")
                candidates.add(10 + distance)
    require(len(candidates) == 1, fact + f" unique retained rank binding for ID {identifier}")
    return {"present": False, "rank": candidates.pop(),
            "basis": "unique retained maximum cutoff-distance constraint"}


def classification_signature(profile):
    return {
        "cutoff_tie": profile["cutoff_tie"],
        "lowest_included_id": profile["lowest_included"]["token_id"],
        "highest_excluded_id": profile["highest_excluded"]["token_id"],
        "references": {
            name: {**{key: comparison[key] for key in (
                "actual_only_ids", "reference_only_ids", "overlap_count", "same_order")},
                "cutoff_reversals": [
                    {**{key: pair[key] for key in (
                        "actual_only_id", "reference_only_id", "exact_reversal_identity",
                        "tie_at_either_boundary", "adjacent_cutoff_in_both",
                        "max_distance_from_cutoff_rank")},
                     "actual_margin_sign": int(Fraction(pair["actual_preference_margin"]) > 0),
                     "reference_margin_sign": int(Fraction(pair["reference_preference_margin"]) > 0)}
                    for pair in comparison["cutoff_reversals"]]}
            for name, comparison in profile["vs_retained_references"].items()}
    }


def compare(left, right):
    a, b = left["top_k_ids_diagnostic_only"], right["top_k_ids_diagnostic_only"]
    ca, cb = classification_signature(left), classification_signature(right)
    return {
        "top_k_membership_equal": set(a) == set(b), "top_k_order_equal": a == b,
        "classification_equal": ca == cb,
        "cutoff_gap_equal": Fraction(left["cutoff_gap"]) == Fraction(right["cutoff_gap"]),
        "retained_reference_accounts_equal":
            left["vs_retained_references"] == right["vs_retained_references"],
        "left_only_ids": sorted(set(a) - set(b)), "right_only_ids": sorted(set(b) - set(a)),
        "order_differences": [{"rank": i + 1, "left_id": x, "right_id": y}
                              for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y],
        "classification_difference": None if ca == cb else {"left": ca, "right": cb},
        "cutoff": {name: {key: profile[key] for key in (
            "lowest_included", "highest_excluded", "cutoff_gap", "cutoff_tie")}
            for name, profile in (("left", left), ("right", right))},
    }


def classify(document):
    same(document["flags"], PARENT_FLAGS, "retained non-admission flags")
    for key, value in document["retained_flags"].items():
        require(key in PARENT_FLAGS, f"unknown retained non-admission flag: {key}")
        same(value, PARENT_FLAGS[key], f"ancestral non-admission flag: {key}")
    for key in ("candidate_admitted", "policy_adopted", "strict_FP16_state_claim",
                "new_token_claim", "full_model_claim"):
        require(document["retained_flags"][key] is False, f"missing ancestral flag: {key}")
    same(document["dispatch_and_write_audit"],
         {**PARENT_FLAGS, "counterfactual_final_head_invocations": 9, "forbidden_calls": 0},
         "retained dispatch/write audit")
    same(document["k"], 10, "retained k")
    same(document["counterfactual_input_count"], 3, "retained input count")
    same(document["control_count"], 9, "retained control count")
    same(document["retained_L23_failures"], 9, "retained L23 failures")
    require(document["normal_host_review"] == "REQUIRED", "retained independent review flag")
    controls = document["controls"]
    require(len(controls) == 9 and len({c["control"] for c in controls}) == 9,
            "retained nine-control history census")
    for control in controls:
        same(control["L21_L22_L23_status"], ["FAIL"] * 3, "retained L21/L22/L23 failures")
        require(control["retained_failures"] and control["retained_L23_and_ancestral_lineage"],
                "retained failure/lineage fields")
    references = document["reference_top_k"]
    same(sorted(references), sorted(REFERENCES), "retained reference census")
    for name, profile in references.items():
        validate_profile(profile, name)
    branches = document["counterfactual_branches"]
    same(sorted(branches), sorted(BRANCHES), "retained branch census")
    reports, comparisons, exact_comparisons = {}, [], []
    for branch in BRANCHES:
        data = branches[branch]
        policies = data["policies"]
        same(sorted(policies), sorted(POLICIES), branch + " policy census")
        reports[branch] = {}
        for policy in POLICIES:
            profile = policies[policy]
            fact = branch + "/" + policy
            validate_profile(profile, fact)
            same(sorted(profile["vs_retained_references"]), sorted(REFERENCES),
                 fact + " reference comparison census")
            for name in REFERENCES:
                validate_comparison(profile, references[name],
                                    profile["vs_retained_references"][name], fact + "/" + name)
            count = profile["word_mismatches_vs_exact_rne"]
            require(type(count) is int and 0 <= count <= 151936, fact + " exact mismatch count")
            reports[branch][policy] = {
                **profile, "selected_ids": {
                    str(i): selected_rank(profile, references, i, fact) for i in (319, 34319)},
                "cutoff_reversal_classification": classification_signature(profile)}
            if policy != "exact_rne":
                exact_comparisons.append({"branch": branch, "policy": policy,
                                          **compare(policies["exact_rne"], profile)})
        for order in ("forward", "reverse"):
            count = data[order + "_torch_vs_direct_rounding_word_count"]
            require(type(count) is int and 0 < count <= 151936,
                    branch + "/" + order + " nonzero conversion mismatch count")
            left, right = "torch_" + order, order + "_direct_rne"
            comparisons.append({"branch": branch, "left_policy": left, "right_policy": right,
                                "conversion_word_mismatch_count": count,
                                **compare(policies[left], policies[right])})
    failures = [row for row in comparisons if not all(row[key] for key in (
        "top_k_membership_equal", "top_k_order_equal", "classification_equal"))]
    return {
        "classification": "REJECTED" if failures else "SUPPORTED",
        "hypothesis": "Nonzero torch/direct conversion word changes preserve top-k stability",
        "lane_terminated": True, "branches": reports,
        "torch_vs_direct": comparisons, "vs_exact_rne_diagnostic_only": exact_comparisons,
        "counterexamples": failures, "reference_top_k": references,
        "retained_context": {key: document[key] for key in PRESERVED},
    }


def check():
    result = {"diagnostic_id": NAME, "version": 1, "flags": FLAGS,
              "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED"}
    try:
        document, bindings = load_capture()
        result.update(classify(document))
        result["capture_bindings"] = bindings
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result.update(classification="UNKNOWN", missing_binding=str(error), lane_terminated=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = check()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return int(result["classification"] == "UNKNOWN")


if __name__ == "__main__":
    raise SystemExit(main())
