"""Stdout-only, retained modal source-only product-difference classifier."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_only_product_term_collapse_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_factor_decomposition_classifier_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-modal-factor-decomposition-b388bceee6eb-attempt001/run"
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "factor_stdout": pin(CAPTURE_ROOT / "check.stdout", 58906,
                         "fcf50af92d5d6e3046fc50606496b1bcfc391992f795d02af48599b37c5ac99c"),
    "factor_capture": pin(CAPTURE_ROOT / "capture.json", 18438,
                          "11fd2dfed65bc85cb5d8785fd846adae7498a9c7a055fc72600885f4e0a8a37d"),
    "factor_review": pin(REVIEWS / "b388bceee6eb/round-0001.json", 690,
                         "34f6d5206ab8d218b8c5afae0181180ff93629cf75b6aff85236e7bd63e3fc39"),
    "factor_source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 25716,
                         "54c2de866ca3fd4710e112ab0197e947ae4b798177c7e35998faaf4cb6a2277a"),
    "factor_test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 12134,
                       "edc8382cd9fc0b81a6a33542be3bcf95a33a46eb29bf03042b6730a836289a04"),
}
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62", "inherited_native",
)
FACTORS = ("norm_weight", "reference_inverse_norm_anchor", "row_difference")
FACTOR = "weight_times_reference_anchor_times_row_difference"
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17", "actual_residual_boundary",
    "negative_reference_residual_boundary", "q24_to_fp16_conversion",
    "fp16_to_branch_terminal_remainder",
)
SCOPE_FIELDS = ("table", "left_id", "right_id", "branch", "coordinate")
SCOPES = (
    ("coordinate", 34319, 319, "binary64", 62),
    ("coordinate", 34319, 319, "binary64", 241),
    ("coordinate", 319, 34319, "binary64", 62),
    ("coordinate", 319, 34319, "binary64", 241),
    ("coordinate_component", 34319, 13, "binary64", 241),
)
MASSES = ("signed", "absolute", "cancellation_absolute_mass")
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)
BOUNDARY = (
    "Retained exact-rational CPU accounting only; descriptive, not a causal explanation, "
    "performance diagnosis, repair, intervention or admission. Original-input global "
    "references, exact thresholds, source/operand/state/KV/lineage gates, accepted evidence "
    "and historical failures remain unchanged. No prefix/admission/reference producer, "
    "accepted producer, native decoder/operator/RMSNorm/head/row-dot replay; no row319 "
    "availability/census/recheck or missing-896-element reconstruction; no closed branch "
    "reopening. Binary64 internal stages remain NOT_RETAINED_NO_RECONSTRUCTION. Native "
    "S16 RTZ/Q24-wide residual state is not strict-FP16-state W4A16. G128 asymmetric packed "
    "INT4 native GEMM ordering, no qzero plus-one, FP16 scales/operator boundaries/KV "
    "remain unchanged. No precision/scale/hardware/GPU/RTL/FPGA/ACE2 expansion or "
    "new-token/full-model admission. Normal independent Host Reviewer is required."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def rational(text):
    require(type(text) is str, "canonical rational string required")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational")
    return value


def bound_bytes(binding):
    path = Path(binding["path"])
    require(not path.is_symlink(), "symlinked retained artifact")
    data = path.read_bytes()
    same(len(data), binding["bytes"], "retained byte count: " + str(path))
    same(hashlib.sha256(data).hexdigest(), binding["sha256"], "retained hash: " + str(path))
    return data


def load_helper():
    # Execute only authenticated definitions, never an ancestor check/classify/main.
    data = bound_bytes(PINS["factor_source"])
    helper = types.ModuleType("_authenticated_modal_factor_authentication")
    helper.__file__ = PINS["factor_source"]["path"]
    exec(compile(data, helper.__file__, "exec"), helper.__dict__)
    return helper


def authenticate(helper):
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    factor = helper.decode(data["factor_stdout"], metadata=True)
    capture = helper.decode(data["factor_capture"], metadata=True)
    review = helper.decode(data["factor_review"], metadata=True)
    helper.terminal_review(review, "b388bceee6eb")
    same((factor["diagnostic_id"], factor["version"], factor["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_FACTOR_DECOMPOSITION_CLASSIFIER"),
         "b388 identity")
    require(capture["success"] is True and capture["failure"] is None, "b388 capture failure")
    sources = [PINS["factor_source"], PINS["factor_test"]]
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "b388 account/runtime")
    same(preflight["sources"], sources, "b388 preflight sources")
    same(capture["sources_after"], sources, "b388 source drift")
    same(factor["source_test_pins"], sources, "b388 stdout source pins")
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"],
         "b388 command census")
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "b388 command failure")
        same([p["path"] for p in result["files"]],
             [str(CAPTURE_ROOT / (result["label"] + s)) for s in helper.SUFFIXES],
             "b388 capture member pointer")
        command, argv, environment, output, error, whole = [
            bound_bytes(p) for p in result["files"]
        ]
        same(command, (result["command"] + "\n").encode(), "b388 command bytes")
        same(helper.decode(argv), result["argv"], "b388 argv")
        same(helper.decode(environment), {k: preflight[k] for k in
                                         ("cwd", "uid", "python", "python_version", "environment")},
             "b388 environment bytes")
        same(error, b"", "b388 stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "b388 whole capture bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "b388 branch")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "b388 concurrency")
        if result["label"] == "check":
            same(output, data["factor_stdout"], "b388 stdout splice")
            same(result["command"], helper.COMMAND, "b388 disclosed command")
            same(result["argv"], [PYTHON, "-B", "-m", helper.MODULE, "--check"], "b388 check argv")
    same(factor["command"], helper.COMMAND, "b388 stdout command")
    same(factor["runtime"]["environment"],
         {k: preflight["environment"][k] for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "b388 runtime environment")
    helper.zero_counters(factor["dispatch_and_write_audit"])
    modal, raw, upstream = helper.authenticate()
    same(factor["artifact_authentication"], upstream, "b388/184/50f authentication splice")
    return factor, modal, raw, {
        "status": "AUTHENTICATED", "input_pins": PINS, "factor_review": review,
        "factor_capture_success": True, "retained_184_50f": upstream,
    }


def resolve(document, pointer):
    require(type(pointer) is str and pointer.startswith("/report/"), "retained JSON pointer")
    value = document
    for part in pointer.split("/")[1:]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def raw_row(raw, pointer, scope, control):
    require(type(pointer) is str, "raw pointer must be a string")
    require("/branches/binary64/selected_coordinates/" in pointer, "raw selected-row pointer")
    same(resolve(raw, pointer.split("/pairs/")[0])["control"], control, "raw control binding")
    pair = resolve(raw, pointer.split("/branches/")[0])
    same((pair["left_id"], pair["right_id"]), scope[1:3], "raw ordered-pair binding")
    same(pair["branches"]["binary64"]["hidden_reference"], "original_input_L23_binary64",
         "original-input reference binding")
    row = resolve(raw, pointer)
    same(row["coordinate"], scope[4], "raw coordinate binding")
    return row


def decomposition(hidden, factors, product, base_hidden, base_factors, base_product):
    w, a, d = factors
    w0, a0, d0 = base_factors
    dh = hidden - base_hidden
    dw, da, dd = w - w0, a - a0, d - d0
    f, f0 = w * a * d, w0 * a0 * d0
    terms = (hidden * dw * a0 * d0, hidden * w * da * d0, hidden * w * a * dd)
    combined = (base_hidden * (f - f0), dh * (f - f0))
    source = dh * f0
    movement = product - base_product
    return {
        "hidden": str(hidden), "canonical_hidden": str(base_hidden), "source_delta": str(dh),
        "weighted": str(product), "canonical_weighted": str(base_product),
        "weighted_movement": str(movement), "source_only_term": str(source),
        "individual_factor_delta_terms": dict(zip(FACTORS, map(str, terms), strict=True)),
        "combined_factor_delta_term": str(combined[0]), "source_factor_interaction_term": str(combined[1]),
        "product_identity_residual": str(product - hidden * f),
        "canonical_product_identity_residual": str(base_product - base_hidden * f0),
        "four_term_identity_residual": str(movement - source - sum(terms)),
        "combined_identity_residual": str(movement - source - sum(combined)),
        "source_only_residual": str(movement - source),
    }


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [
        {"phase": phase, "error_type": type(error).__name__, "message": str(error)}
    ], "claim_boundary": BOUNDARY}


def classify(factor, modal, raw):
    try:
        return report(factor, modal, raw)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(factor, modal, raw):
    parent, cancellation = factor["report"], modal["report"]
    for payload in (parent, cancellation):
        same([tuple(a[k] for k in SCOPE_FIELDS) for a in payload["coordinate_accounts"]],
             list(SCOPES), "coordinate domain")
    same(parent["canonical_control"], CONTROLS[0], "canonical control")
    same(parent["modal_controls"], list(CONTROLS), "modal domain")
    same(parent["reference_scope"], cancellation["reference_scope"], "reference scope splice")
    same(parent["lineage_separation"], cancellation["lineage_separation"], "lineage splice")
    same(parent["reference_scope"], raw["report"]["reference_scope"], "50f reference scope splice")
    same(parent["lineage_separation"], raw["report"]["lineage_separation"], "50f lineage splice")
    failures, accounts = [], []
    checks = 0
    counts = dict.fromkeys((
        "component_account_count", "nonzero_weighted_movement_count", "source_only_movement_count",
        "zero_source_count", "zero_source_movement_count", "zero_factor_delta_term_count",
        "nonzero_factor_delta_term_count", "nonzero_individual_factor_delta_count",
        "nonzero_combined_factor_delta_count", "retained_field_movement_count",
        "reversed_component_checks",
    ), 0)

    def verify(field, actual, expected, **identity):
        nonlocal checks
        checks += 1
        if actual != expected:
            failures.append({**identity, "field": field, "actual": str(actual), "expected": str(expected)})

    verify("parent_factor_decision", parent["decision"], "SUPPORTED")
    verify("parent_cancellation_status", cancellation["cancellation_topology_status"], "SUPPORTED")
    verify("parent_cancellation_failures", cancellation["failure_count"], 0)
    same(cancellation["input_unknown_count"], 0, "retained unknown fields")
    same(cancellation["input_equivalence_status"], "REJECTED", "historical equivalence boundary")
    verify("historical_mismatches", cancellation["input_mismatch_count"], 80)
    verify("accounted_mismatches", cancellation["accounted_mismatch_count"], 80)
    verify("unaccounted_mismatches", cancellation["unaccounted_mismatch_rows"], [])
    for ai, scope in enumerate(SCOPES):
        fa, ma = parent["coordinate_accounts"][ai], cancellation["coordinate_accounts"][ai]
        for account in (fa, ma):
            same([r["control"] for r in account["controls"]], list(CONTROLS), "control census")
        same(fa["canonical_control"], CONTROLS[0], "account canonical control")
        base_fields = {k: rational(v) for k, v in ma["canonical_values"].items()}
        base_factors = tuple(rational(fa["controls"][0]["factors"][k]) for k in FACTORS)
        rows = []
        for ci, control in enumerate(CONTROLS):
            fr, mr = fa["controls"][ci], ma["controls"][ci]
            identity = {**dict(zip(SCOPE_FIELDS, scope, strict=True)), "control": control}
            binding = fr["retained_bindings"]
            same(binding["modal_stdout_pin"], "modal_stdout", "184 pin identity")
            same(binding["raw_stdout_pin"], "raw_stdout", "50f pin identity")
            same(binding["modal_account_pointer"], f"/report/coordinate_accounts/{ai}", "184 account pointer")
            same(binding["modal_control_pointer"],
                 f"/report/coordinate_accounts/{ai}/controls/{ci}", "184 control pointer")
            require(binding["raw_selected_row_pointer"].startswith("/report/controls/"),
                    "50f selected row root")
            require(binding["raw_factor_row_pointer"].startswith(
                "/report/retained_final_rmsnorm_logit_margin_bridge/controls/"), "50f factor row root")
            rr = raw_row(raw, binding["raw_selected_row_pointer"], scope, control)
            detail = raw_row(raw, binding["raw_factor_row_pointer"], scope, control)
            same(set(fr["factors"]), set(FACTORS), "individual factor census")
            factors = tuple(rational(fr["factors"][k]) for k in FACTORS)
            w, a, d = factors
            same(factors, (rational(detail["hidden_bridge"]["weight"]), rational(rr[FACTORS[1]]),
                           rational(detail["row_difference"])), "b388/50f factor binding")
            same(fr["left_weight"], detail["left_weight"], "left tied row binding")
            same(fr["right_weight"], detail["right_weight"], "right tied row binding")
            verify("row_difference", d, rational(fr["left_weight"]) - rational(fr["right_weight"]), **identity)
            same(set(mr["field_deltas"]), set(base_fields), "184 field census")
            values = {k: base_fields[k] + rational(v) for k, v in mr["field_deltas"].items()}
            fields = {FACTOR: rr[FACTOR], FACTORS[1]: rr[FACTORS[1]],
                      "direct_hidden_weighted_term": rr["direct_hidden_weighted_term"]}
            for family in ("hidden", "weighted"):
                same(set(rr[family + "_components"]), set(COMPONENTS), "50f component census")
                same(set(rr[family + "_component_mass"]), set(MASSES), "50f mass census")
                fields.update({family + "." + k: v for k, v in rr[family + "_components"].items()})
                fields.update({family + ".mass." + k: v for k, v in rr[family + "_component_mass"].items()})
                fields[family + ".sum"] = rr[family + "_component_mass"]["signed"]
            same(values, {k: rational(v) for k, v in fields.items()}, "184/50f source binding")
            same(fr[FACTOR], rr[FACTOR], "b388/50f combined factor binding")
            same(fr["retained_184_factor"], str(values[FACTOR]), "184 factor binding")
            same(fr["retained_50f_factor"], rr[FACTOR], "50f factor binding")
            same(detail["weighted_terms"]["direct_hidden"], rr["direct_hidden_weighted_term"],
                 "nested direct-hidden binding")
            verify("retained_direct_identity", rr["exact_direct_hidden_identity"], True, **identity)
            verify("retained_weighted_identity", detail["exact_weighted_identity"], True, **identity)
            verify("factor_product", w * a * d, values[FACTOR], **identity)
            verify("factor_product_flag", fr["product_closure"], True, **identity)
            deltas = tuple(v - v0 for v, v0 in zip(factors, base_factors, strict=True))
            df = values[FACTOR] - base_fields[FACTOR]
            counts["nonzero_combined_factor_delta_count"] += int(df != 0)
            verify("combined_factor_equality", df, 0, **identity)
            w0, a0, d0 = base_factors
            retained_terms = (deltas[0] * a0 * d0, w * deltas[1] * d0, w * a * deltas[2])
            verify("retained_product_delta", rational(fr["product_delta"]), df, **identity)
            for name, delta in zip(FACTORS, deltas, strict=True):
                counts["nonzero_individual_factor_delta_count"] += int(delta != 0)
                verify(name + ".equality", delta, 0, **identity)
                verify(name + ".retained_delta", rational(fr["factor_deltas_from_frozen_inherited"][name]),
                       delta, **identity)
            for name, term in zip(FACTORS, retained_terms, strict=True):
                verify("retained_telescoping." + name,
                       rational(fr["ordered_telescoping_product_delta_terms"][name]), term, **identity)
            counts["retained_field_movement_count"] += sum(v != "0" for v in mr["field_deltas"].values())
            terms = {}
            for component in COMPONENTS:
                h, h0 = values["hidden." + component], base_fields["hidden." + component]
                p, p0 = values["weighted." + component], base_fields["weighted." + component]
                term = decomposition(h, factors, p, h0, base_factors, p0)
                for key in ("product_identity_residual", "canonical_product_identity_residual",
                            "four_term_identity_residual", "combined_identity_residual", "source_only_residual"):
                    verify(component + "." + key, term[key], "0", **identity)
                factor_terms = [*term["individual_factor_delta_terms"].values(),
                                term["combined_factor_delta_term"], term["source_factor_interaction_term"]]
                counts["component_account_count"] += 1
                counts["nonzero_weighted_movement_count"] += int(p != p0)
                counts["source_only_movement_count"] += int(
                    p != p0 and p - p0 == (h - h0) * base_fields[FACTOR]
                    and not any(deltas) and df == 0 and all(v == "0" for v in factor_terms))
                counts["zero_source_count"] += int(h == h0)
                counts["zero_source_movement_count"] += int(h == h0 and p != p0)
                for value in factor_terms:
                    counts["zero_factor_delta_term_count" if value == "0"
                           else "nonzero_factor_delta_term_count"] += 1
                    verify(component + ".factor_delta_term", value, "0", **identity)
                terms[component] = term
            mass = {}
            for family in ("hidden", "weighted"):
                components = [values[family + "." + c] for c in COMPONENTS]
                signed, absolute = sum(components), sum(map(abs, components))
                expected = (signed, absolute, absolute - abs(signed))
                mass[family] = {}
                verify(family + ".sum", values[family + ".sum"], signed, **identity)
                for field, computed in zip(MASSES, expected, strict=True):
                    key = family + ".mass." + field
                    verify(key, values[key], computed, **identity)
                    mass[family][field] = {
                        "value": str(computed), "canonical": str(base_fields[key]),
                        "movement": str(computed - base_fields[key]),
                    }
                delta_sum = sum(values[family + "." + c] - base_fields[family + "." + c]
                                for c in COMPONENTS)
                verify(family + ".source_delta_sum", delta_sum, 0, **identity)
                verify(family + ".retained_source_delta_sum", rational(mr[family + "_source_delta_sum"]),
                       delta_sum, **identity)
            verify("direct_hidden_closure", values["weighted.sum"],
                   values["direct_hidden_weighted_term"], **identity)
            for field in MASSES:
                scale = values[FACTOR] if field == "signed" else abs(values[FACTOR])
                verify("weighted_mass." + field, rational(mass["weighted"][field]["value"]),
                       scale * rational(mass["hidden"][field]["value"]), **identity)
                verify("weighted_mass_delta." + field, rational(mass["weighted"][field]["movement"]),
                       scale * rational(mass["hidden"][field]["movement"]), **identity)
            rows.append({"control": control, "factors": fr["factors"], FACTOR: str(values[FACTOR]),
                         "left_weight": fr["left_weight"], "right_weight": fr["right_weight"],
                         "factor_deltas": dict(zip(FACTORS, map(str, deltas), strict=True)),
                         "combined_factor_delta": str(df), "components": terms, "mass": mass,
                         "retained_bindings": {
                             "factor_stdout_pin": "factor_stdout",
                             "factor_control_pointer": f"/report/coordinate_accounts/{ai}/controls/{ci}",
                             **binding,
                         }})
        accounts.append({**dict(zip(SCOPE_FIELDS, scope, strict=True)), "controls": rows})
    for fi, ri in ((0, 2), (1, 3)):
        for forward, reverse in zip(accounts[fi]["controls"], accounts[ri]["controls"], strict=True):
            identity = {"control": forward["control"], "coordinate": SCOPES[fi][4]}
            for name in FACTORS:
                sign = -1 if name == "row_difference" else 1
                verify("reverse." + name, rational(forward["factors"][name]),
                       sign * rational(reverse["factors"][name]), **identity)
            verify("reverse.combined_factor", rational(forward[FACTOR]), -rational(reverse[FACTOR]), **identity)
            verify("reverse.left_weight", forward["left_weight"], reverse["right_weight"], **identity)
            verify("reverse.right_weight", forward["right_weight"], reverse["left_weight"], **identity)
            for c in COMPONENTS:
                f, r = forward["components"][c], reverse["components"][c]
                for key in ("hidden", "canonical_hidden", "source_delta"):
                    verify("reverse." + c + "." + key, f[key], r[key], **identity)
                for key in ("weighted", "canonical_weighted", "weighted_movement", "source_only_term",
                            "combined_factor_delta_term", "source_factor_interaction_term"):
                    verify("reverse." + c + "." + key, rational(f[key]), -rational(r[key]), **identity)
                for name in FACTORS:
                    verify("reverse.term." + name, rational(f["individual_factor_delta_terms"][name]),
                           -rational(r["individual_factor_delta_terms"][name]), **identity)
                counts["reversed_component_checks"] += 1
            for family in ("hidden", "weighted"):
                for field in MASSES:
                    sign = -1 if family == "weighted" and field == "signed" else 1
                    for key in ("value", "canonical", "movement"):
                        verify("reverse.mass." + family + "." + field + "." + key,
                               rational(forward["mass"][family][field][key]),
                               sign * rational(reverse["mass"][family][field][key]), **identity)
    verify("source_only_movement_census", counts["source_only_movement_count"],
           counts["nonzero_weighted_movement_count"])
    verify("zero_source_movement_census", counts["zero_source_movement_count"], 0)
    verify("retained_mismatch_census", counts["retained_field_movement_count"], 80)
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "unknown_reasons": [],
        "failure_count": len(failures), "failures": failures, "exact_check_count": checks,
        "counts": counts, "coordinate_account_count": len(accounts),
        "control_coordinate_account_count": sum(len(a["controls"]) for a in accounts),
        "coordinate_accounts": accounts,
        "product_difference_identity": "dh*w0*a0*d0 + h*dw*a0*d0 + h*w*da*d0 + h*w*a*dd",
        "combined_product_difference_identity": "dh*F0 + h0*dF + dh*dF",
        "factor_term_counting": "three individual telescoping terms plus two combined-factor terms per component",
        "canonical_control": CONTROLS[0], "modal_controls": list(CONTROLS),
        "parent_source_equivalence_status_preserved": cancellation["input_equivalence_status"],
        "reference_scope": parent["reference_scope"], "lineage_separation": parent["lineage_separation"],
        "claim_boundary": BOUNDARY,
    }


def check():
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    audit = {"forbidden_calls": 0}
    source_pins = []
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    try:
        helper = load_helper()
        audit = dict.fromkeys(helper.COUNTERS, 0)
        allowed = helper.allowed_paths() | {
            str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
            *(str(CAPTURE_ROOT / (label + suffix))
              for label in (*helper.LABELS, "concurrency") for suffix in helper.SUFFIXES),
        }
        helper.allowed_paths = lambda: allowed
        with helper.read_only(audit):
            same((runtime["cwd"], runtime["uid"], sys.executable), (str(ROOT), 1000, PYTHON),
                 "current account/interpreter gate")
            same(runtime["environment"], {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
                 "current command-local environment")
            for path in (SOURCE, TEST, Path(sys.executable).resolve()):
                data = path.read_bytes()
                binding = pin(path, len(data), hashlib.sha256(data).hexdigest())
                if path in (SOURCE, TEST):
                    source_pins.append(binding)
                else:
                    runtime["executable_pin"] = binding
            factor, modal, raw, authentication = authenticate(helper)
            result = classify(factor, modal, raw)
            helper.zero_counters(audit)
    except ERRORS as error:
        result = unknown(error, "authentication_or_runtime")
        authentication["error"] = result["unknown_reasons"][0]
    except RuntimeError as error:
        if type(error).__name__ != "ForbiddenOperation":
            raise
        result = unknown(error, "forbidden_operation")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_SOURCE_ONLY_PRODUCT_TERM_COLLAPSE_CLASSIFIER",
        "decision": result["decision"], "command": COMMAND,
        "artifact_authentication": authentication, "source_test_pins": source_pins,
        "runtime": runtime, "report": result, "dispatch_and_write_audit": audit,
        "claim_boundary": BOUNDARY,
        "normal_host_review": "Independent Reviewer required; not asserted by Engineer.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = check()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 2 if result["decision"] == "UNKNOWN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
