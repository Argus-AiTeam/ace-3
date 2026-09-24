"""Stdout-only exact dose/rank classification of reviewed 8525 retained accounts."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_only_dose_rank_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_only_product_term_collapse_classifier_v1"
CAPTURE = ROOT / "build/selected-direct-hidden-modal-source-only-product-collapse-8525fd988f9c-attempt001"
RUN = CAPTURE / "run"
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 275995,
                  "505d05e3e1e69cf97ed6c3f54146c074ba294da29a5f50dd7cc0bfb7097dc356"),
    "capture": pin(RUN / "capture.json", 18979,
                   "203f9264cc742df1045b7ef6542861ee77a2b602a3eacb12f7087588480312eb"),
    "review": pin(REVIEWS / "8525fd988f9c/round-0001.json", 690,
                  "f62ba4df0be287357359b6bb655ec2fc5fdff03ac99a8512fd9d9076ffbd774a"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 29020,
                  "37c70f7eab6a46243d7fb39e0b76ba1ebdc82d58b3b0d09cced71e4e7d134868"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 11014,
                "949deaa3b46f5499bcc5fec6e0f6431d14973a5912947511985e486d30b8e912"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1114,
                         "1511887964484b2c796258ab9e39f7e845e621aa4e7b681c5069ec44afc777ab"),
    "launcher": pin(CAPTURE / "launch.py", 7889,
                    "7aee2dc74607aab284cfdc195e1d2c981ba686277fa5863b66a6d5dae0cfbae9"),
}
GROUPS = {
    "frozen_inherited_o": ("frozen_inherited_o", "frozen_inherited_o_down"),
    "scratch": ("scratch", "scratch_down"),
    "mapped62": ("mapped62",),
}
DOSES = {"frozen_inherited_o": "3/32768", "scratch": "321/4194304", "mapped62": "933/16777216"}
GAPS = {
    ("frozen_inherited_o", "scratch"): "63/4194304",
    ("frozen_inherited_o", "mapped62"): "603/16777216",
    ("scratch", "mapped62"): "351/16777216",
}
ZERO_CONTROLS = ("frozen_inherited", "frozen_inherited_down", "inherited_native")
MOVING = ("actual_residual_boundary", "q24_to_fp16_conversion")
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def rational(text):
    require(type(text) is str, "missing or non-string retained rational")
    result = Fraction(text)
    same(str(result), text, "ambiguous noncanonical rational")
    return result


def bound_bytes(binding):
    path = Path(binding["path"])
    require(not path.is_symlink(), "symlinked retained artifact")
    data = path.read_bytes()
    same(len(data), binding["bytes"], "artifact byte count: " + str(path))
    same(hashlib.sha256(data).hexdigest(), binding["sha256"], "artifact hash: " + str(path))
    return data


def load_helpers():
    # Only pinned definitions and authentication helpers; no ancestor semantic entrypoint.
    data = bound_bytes(PINS["source"])
    parent = types.ModuleType("_authenticated_8525_read_only_helpers")
    parent.__file__ = PINS["source"]["path"]
    exec(compile(data, parent.__file__, "exec"), parent.__dict__)
    return parent, parent.load_helper()


def allowed_paths(parent, helper):
    return helper.allowed_paths() | {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        *(p["path"] for p in parent.PINS.values()),
        *(str(parent.CAPTURE_ROOT / (label + suffix))
          for label in (*helper.LABELS, "concurrency") for suffix in helper.SUFFIXES),
        *(str(RUN / (label + suffix))
          for label in (*helper.LABELS, "concurrency") for suffix in helper.SUFFIXES),
        *(str(CAPTURE / ("launcher." + name))
          for name in ("identity.json", "stdout", "stderr", "whole-command.log")),
    }


def authenticate(parent, helper):
    data = {key: bound_bytes(value) for key, value in PINS.items()}
    collapse, capture, review, outer = [
        helper.decode(data[key], metadata=True)
        for key in ("stdout", "capture", "review", "outer_capture")
    ]
    helper.terminal_review(review, "8525fd988f9c")
    same((collapse["diagnostic_id"], collapse["version"], collapse["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_SOURCE_ONLY_PRODUCT_TERM_COLLAPSE_CLASSIFIER"),
         "8525 identity")
    same((collapse["artifact_authentication"]["status"], collapse["decision"],
          collapse["report"]["decision"]), ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"),
         "8525 authenticated supported prerequisite")
    require(capture["success"] is True and capture["failure"] is None, "8525 capture failure")
    sources = [PINS["source"], PINS["test"]]
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "8525 account/runtime")
    for actual in (preflight["sources"], capture["sources_after"], collapse["source_test_pins"]):
        same(actual, sources, "8525 source/test binding")
    same(preflight["launcher"], PINS["launcher"], "8525 launcher pin")
    same(collapse["command"], parent.COMMAND, "8525 disclosed command")
    runtime = collapse["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "8525 runtime splice")
    same(runtime["environment"], {k: preflight["environment"][k]
                                 for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "8525 runtime environment splice")
    bound_bytes(runtime["executable_pin"])
    same(runtime["executable_pin"]["path"], str(Path(PYTHON).resolve()), "8525 executable identity")
    same(set(collapse["dispatch_and_write_audit"]), set(helper.COUNTERS), "8525 counter census")
    helper.zero_counters(collapse["dispatch_and_write_audit"])
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"],
         "8525 command census")
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "8525 command failure")
        same([p["path"] for p in result["files"]],
             [str(RUN / (result["label"] + s)) for s in helper.SUFFIXES], "8525 member paths")
        command, argv, environment, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "8525 command bytes")
        same(result["command"], f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
             + shlex.join(result["argv"]), "8525 command/argv splice")
        same(helper.decode(argv), result["argv"], "8525 argv bytes")
        same(helper.decode(environment), {k: preflight[k] for k in
                                         ("cwd", "uid", "python", "python_version", "environment")},
             "8525 environment bytes")
        same(error, b"", "8525 stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "8525 whole-command bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "8525 branch")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "8525 concurrency")
        if result["label"] == "check":
            same(output, data["stdout"], "8525 stdout splice")
            same(result["argv"], [PYTHON, "-B", "-m", parent.MODULE, "--check"], "8525 check argv")
    same(outer["exit_status"], 0, "8525 outer exit")
    same([p["path"] for p in outer["files"]],
         [str(CAPTURE / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "8525 outer paths")
    identity_bytes, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    identity = helper.decode(identity_bytes)
    same(identity["launcher"], PINS["launcher"], "8525 outer launcher")
    same(identity["argv"], [PYTHON, "-B", PINS["launcher"]["path"], "--run"], "8525 outer argv")
    same(identity["command"], " ".join(f"{k}={shlex.quote(preflight['environment'][k])}"
         for k in ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD")) + " " + shlex.join(identity["argv"]),
         "8525 outer command")
    same({k: identity[k] for k in ("cwd", "uid", "environment")},
         {k: preflight[k] for k in ("cwd", "uid", "environment")}, "8525 outer environment")
    same(error, b"", "8525 outer stderr")
    same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\n", "8525 outer whole bytes")
    receipt = helper.decode(output.splitlines()[-1], metadata=True)
    same(receipt, {"capture": PINS["capture"], "success": True, "failure": None}, "8525 outer receipt")
    factor, modal, raw, upstream = parent.authenticate(helper)
    same(collapse["artifact_authentication"], upstream, "8525 upstream authentication splice")
    return collapse, factor, modal, raw, {
        "status": "AUTHENTICATED", "reviewed_mission": "8525fd988f9c", "input_pins": PINS,
        "review": review, "complete_command_capture": capture, "outer_capture": outer,
        "retained_predecessor_raw_bindings": upstream,
    }


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def classify(collapse, factor, modal, raw, parent):
    try:
        return report(collapse, factor, modal, raw, parent)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(collapse, factor, modal, raw, parent):
    retained = collapse["report"]
    same((collapse["artifact_authentication"]["status"], collapse["decision"], retained["decision"]),
         ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "8525 prerequisite")
    same(retained["canonical_control"], parent.CONTROLS[0], "canonical binding")
    same(retained["modal_controls"], list(parent.CONTROLS), "modal domain binding")
    for field in ("reference_scope", "lineage_separation"):
        for document in (factor, modal, raw):
            same(retained[field], document["report"][field], field + " binding")
    same(retained["parent_source_equivalence_status_preserved"], "REJECTED", "equivalence boundary")
    failures, movements, zeros, reverse, records, signatures = [], [], [], [], {}, {}
    checks = 0

    def verify(field, actual, expected, **identity):
        nonlocal checks
        checks += 1
        if actual != expected:
            failures.append({**identity, "field": field, "actual": str(actual), "expected": str(expected)})

    accounts = retained["coordinate_accounts"]
    scopes = [tuple(a[k] for k in parent.SCOPE_FIELDS) for a in accounts]
    require(len(set(scopes)) == len(scopes), "ambiguous duplicate coordinate")
    verify("coordinate_account_count", len(accounts), 5)
    verify("retained_coordinate_account_count", retained["coordinate_account_count"], len(accounts))
    verify("coordinate_domain", set(scopes), set(parent.SCOPES))
    component_count, control_count, proportionality_count = 0, 0, 0
    for account, scope in zip(accounts, scopes, strict=True):
        require(scope in parent.SCOPES, "incompatible coordinate binding")
        ai = parent.SCOPES.index(scope)
        controls = account["controls"]
        names = [c["control"] for c in controls]
        require(len(set(names)) == len(names), "ambiguous duplicate modal control")
        verify("control_count", len(controls), 8, coordinate=scope[4])
        verify("control_domain", set(names), set(parent.CONTROLS))
        base = next((c for c in controls if c["control"] == "frozen_inherited"), None)
        require(base is not None, "missing canonical control")
        common = rational(base[parent.FACTOR])
        for control in controls:
            name = control["control"]
            require(name in parent.CONTROLS, "incompatible control binding")
            ci = parent.CONTROLS.index(name)
            identity = {**dict(zip(parent.SCOPE_FIELDS, scope, strict=True)), "control": name}
            pointer = f"/report/coordinate_accounts/{ai}/controls/{ci}"
            binding = control["retained_bindings"]
            fr = parent.resolve(factor, pointer)
            same(binding, {**fr["retained_bindings"], "factor_stdout_pin": "factor_stdout",
                           "factor_control_pointer": pointer}, "factor/raw pointer splice")
            same(parent.resolve(modal, binding["modal_control_pointer"])["control"], name,
                 "modal control binding")
            rr = parent.raw_row(raw, binding["raw_selected_row_pointer"], scope, name)
            detail = parent.raw_row(raw, binding["raw_factor_row_pointer"], scope, name)
            for field in ("factors", "left_weight", "right_weight", parent.FACTOR):
                same(control[field], fr[field], "factor field binding: " + field)
            same(control[parent.FACTOR], rr[parent.FACTOR], "raw common factor binding")
            for field in ("left_weight", "right_weight"):
                same(control[field], detail[field], "raw tied-row binding")
            current_factor = rational(control[parent.FACTOR])
            verify("common_factor", current_factor, common, **identity)
            verify("combined_factor_delta", rational(control["combined_factor_delta"]), 0, **identity)
            verify("factor_product", current_factor,
                   rational(control["factors"]["norm_weight"])
                   * rational(control["factors"]["reference_inverse_norm_anchor"])
                   * rational(control["factors"]["row_difference"]), **identity)
            same(set(control["components"]), set(parent.COMPONENTS), "component field census")
            control_count += 1
            deltas = {}
            for component in parent.COMPONENTS:
                term = control["components"][component]
                baseline = base["components"][component]
                h, h0, w, w0, dh, dw = [rational(term[k]) for k in (
                    "hidden", "canonical_hidden", "weighted", "canonical_weighted",
                    "source_delta", "weighted_movement")]
                same(term["hidden"], rr["hidden_components"][component], "raw hidden binding")
                same(term["weighted"], rr["weighted_components"][component], "raw weighted binding")
                item = {**identity, "component": component}
                verify("canonical_hidden", h0, rational(baseline["hidden"]), **item)
                verify("canonical_weighted", w0, rational(baseline["weighted"]), **item)
                verify("source_delta", dh, h - h0, **item)
                verify("weighted_movement", dw, w - w0, **item)
                verify("weighted_factor_proportionality", dw, dh * common, **item)
                verify("retained_source_only_term", rational(term["source_only_term"]), dh * common, **item)
                verify("retained_source_only_residual", rational(term["source_only_residual"]), dw - dh * common,
                       **item)
                group = next((g for g, members in GROUPS.items() if name in members), None)
                expected = Fraction(0)
                if scope[4] == 62 and group and component in MOVING:
                    expected = rational(DOSES[group]) * (-1 if component == MOVING[0] else 1)
                verify("expected_signed_source_dose", dh, expected, **item)
                deltas[component] = dh
                records[(scope, name, component)] = (h, dh, w, dw, current_factor)
                component_count += 1
                proportionality_count += int(dw == dh * common)
                if dh:
                    movements.append({**item, "group": group, "source_delta": str(dh),
                                      "absolute_dose": str(abs(dh)), "weighted_movement": str(dw),
                                      "common_factor": str(common), "source_only_term": str(dh * common),
                                      "retained_8525_pointer": pointer + "/components/" + component,
                                      "retained_bindings": binding})
                else:
                    verify("zero_source_weighted_movement", dw, 0, **item)
                    zeros.append({**item, "source_delta": str(dh), "weighted_movement": str(dw)})
            verify("source_pair_opposition", deltas[MOVING[0]], -deltas[MOVING[1]], **identity)
            verify("source_signed_sum", sum(deltas.values(), Fraction(0)), 0, **identity)
            if scope[4] == 62:
                signatures[(scope[1], scope[2], name)] = abs(deltas[MOVING[0]])

    for (scope, name, component), forward in records.items():
        if scope[1:3] != (34319, 319):
            continue
        reverse_scope = (scope[0], 319, 34319, scope[3], scope[4])
        key = (reverse_scope, name, component)
        if key not in records:
            verify("missing_reverse_account", False, True)
            continue
        backward = records[key]
        hidden_same = forward[:2] == backward[:2]
        weighted_opposite = forward[2:] == tuple(-v for v in backward[2:])
        verify("reverse_hidden_signature", hidden_same, True, control=name, component=component)
        verify("reverse_weighted_orientation", weighted_opposite, True, control=name, component=component)
        reverse.append({"coordinate": scope[4], "control": name, "component": component,
                        "hidden_and_dose_equal": hidden_same, "weighted_and_factor_opposite": weighted_opposite})

    groups, measured = [], {}
    for group, members in GROUPS.items():
        require((34319, 319, members[0]) in signatures, "missing retained group representative")
        dose = signatures[(34319, 319, members[0])]
        measured[group] = dose
        group_signatures = []
        for left, right in ((34319, 319), (319, 34319)):
            for name in members:
                if (left, right, name) not in signatures:
                    verify("missing_group_member", name, "present")
                    continue
                actual = signatures[(left, right, name)]
                verify("group_dose_equality", actual, dose, control=name)
                group_signatures.append({"left_id": left, "right_id": right, "control": name,
                                         "absolute_dose": str(actual)})
        verify("exact_group_dose", dose, rational(DOSES[group]), group=group)
        groups.append({"group": group, "controls": list(members), "absolute_dose": str(dose),
                       "ordered_pair_signatures": group_signatures})
    rank = sorted(measured, key=measured.get, reverse=True)
    verify("strict_descending_rank", rank, list(GROUPS))
    gaps = []
    for (larger, smaller), expected in GAPS.items():
        gap = measured[larger] - measured[smaller]
        verify("exact_pairwise_gap", gap, rational(expected), larger=larger, smaller=smaller)
        verify("strict_pairwise_gap", gap > 0, True, larger=larger, smaller=smaller)
        gaps.append({"larger": larger, "smaller": smaller, "gap": str(gap), "positive": gap > 0})
    counts = {"component_accounts": component_count, "control_coordinate_accounts": control_count,
              "nonzero_source_only_movements": len(movements), "zero_source_accounts": len(zeros),
              "factor_proportionality_checks": proportionality_count, "reverse_component_checks": len(reverse)}
    expected_counts = {"component_accounts": 280, "control_coordinate_accounts": 40,
                       "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
                       "factor_proportionality_checks": 280, "reverse_component_checks": 112}
    for key, expected in expected_counts.items():
        verify(key, counts[key], expected)
    for key, actual in (("component_account_count", component_count), ("source_only_movement_count", len(movements)),
                        ("zero_source_count", len(zeros)), ("nonzero_weighted_movement_count", len(movements))):
        verify("8525_" + key, retained["counts"][key], actual)
    verify("retained_control_coordinate_count", retained["control_coordinate_account_count"], control_count)
    return {"decision": "REJECTED" if failures else "SUPPORTED", "unknown_reasons": [],
            "failure_count": len(failures), "failures": failures, "exact_check_count": checks,
            "counts": counts, "dose_groups": groups, "absolute_dose_rank_descending": rank,
            "pairwise_rational_gaps": gaps, "nonzero_source_only_movements": movements,
            "zero_source_accounts": zeros, "zero_dose_controls": list(ZERO_CONTROLS),
            "reverse_pair_checks": reverse, "canonical_control": "frozen_inherited",
            "reference_scope": retained["reference_scope"], "lineage_separation": retained["lineage_separation"],
            "parent_source_equivalence_status_preserved": "REJECTED", "claim_boundary": parent.BOUNDARY}


def check():
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    audit, source_pins = {"forbidden_calls": 0}, []
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    boundary = "Retained-only CPU non-admission accounting; independent Host Reviewer required."
    try:
        parent, helper = load_helpers()
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        allowed = allowed_paths(parent, helper)
        helper.allowed_paths = lambda: allowed
        with helper.read_only(audit):
            same((runtime["cwd"], runtime["uid"], runtime["python"]), (str(ROOT), 1000, PYTHON),
                 "current account/interpreter")
            same(runtime["environment"], {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
                 "command-local environment")
            require(sys.dont_write_bytecode, "bytecode writing must be disabled")
            for path in (SOURCE, TEST, Path(sys.executable).resolve()):
                data = path.read_bytes()
                binding = pin(path, len(data), hashlib.sha256(data).hexdigest())
                if path in (SOURCE, TEST):
                    source_pins.append(binding)
                else:
                    runtime["executable_pin"] = binding
            collapse, factor, modal, raw, authentication = authenticate(parent, helper)
            result = classify(collapse, factor, modal, raw, parent)
            helper.zero_counters(audit)
    except ERRORS as error:
        result = unknown(error, "authentication_or_runtime")
        authentication["error"] = result["unknown_reasons"][0]
    except RuntimeError as error:
        if type(error).__name__ != "ForbiddenOperation":
            raise
        result = unknown(error, "forbidden_operation")
    return {"diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_SOURCE_ONLY_DOSE_RANK_CLASSIFIER",
            "decision": result["decision"], "command": COMMAND, "artifact_authentication": authentication,
            "source_test_pins": source_pins, "runtime": runtime, "report": result,
            "dispatch_and_write_audit": audit, "claim_boundary": boundary,
            "normal_host_review": "Independent Reviewer required; not asserted by Engineer."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    result = check()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 2 if result["decision"] == "UNKNOWN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
