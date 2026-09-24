"""Stdout-only retained modal down-axis and native/canonical alias classifier."""

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
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_down_axis_invariance_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = NAME.replace("down_axis_invariance", "source_only_dose_rank")
CAPTURE = ROOT / "build/selected-direct-hidden-modal-source-only-dose-rank-12ea641e7a89-attempt001"
RUN = CAPTURE / "run"
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 122944,
                  "1fb30820ae07279348a3d92e45328a22e9a827d51258d493bcc70b85373ad5ee"),
    "capture": pin(RUN / "capture.json", 18456,
                   "10f118506032adde4228afa48a59deb8f5d3ba201905547deed91df090c5a820"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1108,
                         "db3be4b66fb8b9d618a6cf4854619524925bd47e55eaab580d3264745cbd93ee"),
    "launcher": pin(CAPTURE / "launch.py", 7761,
                    "f95b6f41f6c410e754debcdfce8e165afc74f073137586290bd155e5c59079d0"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 24832,
                  "74b61e38cee9b84165e418b7b49f9277a20a2703737060c146162a2e4d59db85"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 11424,
                "3504407a81b17ac0187ce97e551599cd7db564776debddb5c7544fe878452558"),
    "review": pin(REVIEWS / "12ea641e7a89/round-0001.json", 690,
                  "f23965527aec4cb377c2f4662d840a0dd2adda16cd877cd0dbd50ca04e702382"),
}
ALIASES = (
    ("frozen_inherited_down", "frozen_inherited", "0"),
    ("frozen_inherited_o_down", "frozen_inherited_o", "3/32768"),
    ("scratch_down", "scratch", "321/4194304"),
    ("inherited_native", "frozen_inherited", "0"),
)
DOSES = {
    "frozen_inherited": "0", "frozen_inherited_down": "0", "inherited_native": "0",
    "frozen_inherited_o": "3/32768", "frozen_inherited_o_down": "3/32768",
    "scratch": "321/4194304", "scratch_down": "321/4194304", "mapped62": "933/16777216",
}
GROUPS = {
    "frozen_inherited_o": ["frozen_inherited_o", "frozen_inherited_o_down"],
    "scratch": ["scratch", "scratch_down"], "mapped62": ["mapped62"],
}
COUNTS = {
    "component_accounts": 280, "control_coordinate_accounts": 40,
    "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
    "factor_proportionality_checks": 280, "reverse_component_checks": 112,
}
TERM_FIELDS = {
    "hidden", "canonical_hidden", "weighted", "canonical_weighted", "source_delta",
    "weighted_movement", "source_only_term", "source_only_residual",
    "combined_factor_delta_term", "source_factor_interaction_term",
    "product_identity_residual", "canonical_product_identity_residual",
    "combined_identity_residual", "four_term_identity_residual",
    "individual_factor_delta_terms",
}
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def rational(text):
    require(type(text) is str, "missing or non-string rational")
    value = Fraction(text)
    same(str(value), text, "ambiguous noncanonical rational")
    return value


def bound_bytes(binding):
    path = Path(binding["path"])
    require(not path.is_symlink(), "symlinked retained artifact")
    data = path.read_bytes()
    same(len(data), binding["bytes"], "artifact byte count: " + str(path))
    same(hashlib.sha256(data).hexdigest(), binding["sha256"], "artifact hash: " + str(path))
    return data


def load_helpers():
    # Load pinned definitions only; no ancestor semantic entrypoint is called.
    data = bound_bytes(PINS["source"])
    dose = types.ModuleType("_authenticated_12ea_retained_helpers")
    dose.__file__ = PINS["source"]["path"]
    exec(compile(data, dose.__file__, "exec"), dose.__dict__)
    parent, helper = dose.load_helpers()
    return dose, parent, helper


def allowed_paths(dose, parent, helper):
    return dose.allowed_paths(parent, helper) | {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        *(str(RUN / (label + suffix))
          for label in (*helper.LABELS, "concurrency") for suffix in helper.SUFFIXES),
        *(str(CAPTURE / ("launcher." + suffix))
          for suffix in ("identity.json", "stdout", "stderr", "whole-command.log")),
    }


def authenticate(dose, parent, helper):
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    retained, capture, outer, review = [
        helper.decode(data[key], metadata=True)
        for key in ("stdout", "capture", "outer_capture", "review")
    ]
    helper.terminal_review(review, "12ea641e7a89")
    same((retained["diagnostic_id"], retained["version"], retained["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_SOURCE_ONLY_DOSE_RANK_CLASSIFIER"),
         "12ea identity")
    same((retained["artifact_authentication"]["status"], retained["decision"],
          retained["report"]["decision"]), ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"),
         "12ea authenticated supported prerequisite")
    require(capture["success"] is True and capture["failure"] is None, "12ea capture failure")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "12ea account/runtime")
    for sources in (preflight["sources"], capture["sources_after"], retained["source_test_pins"]):
        same(sources, [PINS["source"], PINS["test"]], "12ea source/test pins")
    same(preflight["launcher"], PINS["launcher"], "12ea launcher binding")
    same(retained["command"], dose.COMMAND, "12ea disclosed command")
    runtime = retained["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "12ea runtime splice")
    same(runtime["environment"], {k: preflight["environment"][k]
                                 for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "12ea environment splice")
    same(runtime["executable_pin"]["path"], str(Path(PYTHON).resolve()), "12ea executable identity")
    bound_bytes(runtime["executable_pin"])
    same(set(retained["dispatch_and_write_audit"]), set(helper.COUNTERS), "12ea counter census")
    helper.zero_counters(retained["dispatch_and_write_audit"])
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"],
         "12ea command census")
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "12ea command failure")
        same([p["path"] for p in result["files"]],
             [str(RUN / (result["label"] + s)) for s in helper.SUFFIXES], "12ea member paths")
        command, argv, env, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "12ea command bytes")
        same(result["command"], f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
             + shlex.join(result["argv"]), "12ea command/argv splice")
        same(helper.decode(argv), result["argv"], "12ea argv bytes")
        same(helper.decode(env), {k: preflight[k] for k in
                                 ("cwd", "uid", "python", "python_version", "environment")},
             "12ea environment bytes")
        same(error, b"", "12ea stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
             + output + b"\nSTDERR\n" + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n",
             "12ea whole-command bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "12ea branch")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "12ea concurrency")
        if result["label"] == "check":
            same(output, data["stdout"], "12ea stdout splice")
            same(result["argv"], [PYTHON, "-B", "-m", dose.MODULE, "--check"], "12ea check argv")
    same((outer["exit_status"], outer["timed_out"]), (0, False), "12ea outer completion")
    same([p["path"] for p in outer["files"]],
         [str(CAPTURE / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "12ea outer paths")
    identity_bytes, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    identity = helper.decode(identity_bytes)
    same(identity["launcher"], PINS["launcher"], "12ea outer launcher")
    same(identity["argv"], [PYTHON, "-B", PINS["launcher"]["path"], "--run"], "12ea outer argv")
    same(identity["command"], " ".join(f"{k}={shlex.quote(preflight['environment'][k])}"
         for k in ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD")) + " " + shlex.join(identity["argv"]),
         "12ea outer command")
    same({k: identity[k] for k in ("cwd", "uid", "environment")},
         {k: preflight[k] for k in ("cwd", "uid", "environment")}, "12ea outer environment")
    same(error, b"", "12ea outer stderr")
    same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "12ea outer whole bytes")
    same(helper.decode(output.splitlines()[-1], metadata=True),
         {"capture": PINS["capture"], "success": True, "failure": None}, "12ea outer receipt")
    collapse, factor, modal, raw, upstream = dose.authenticate(parent, helper)
    same(retained["artifact_authentication"], upstream, "12ea predecessor authentication splice")
    return (retained, collapse, factor, modal, raw), {
        "status": "AUTHENTICATED", "reviewed_mission": "12ea641e7a89", "input_pins": PINS,
        "review": review, "complete_command_capture": capture, "outer_capture": outer,
        "retained_predecessor_raw_bindings": upstream,
    }


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, prefix + "/" + key)
    else:
        yield prefix, value


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        require(key not in result, "ambiguous duplicate retained account")
        result[key] = row
    return result


def classify(documents, parent):
    try:
        return report(documents, parent)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(documents, parent):
    dose, collapse, factor, modal, raw = documents
    retained, predecessor = dose["report"], collapse["report"]
    for document in (dose, collapse):
        same((document["artifact_authentication"]["status"], document["decision"],
              document["report"]["decision"]), ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"),
             "authenticated supported prerequisite")
    same(retained["canonical_control"], "frozen_inherited", "canonical binding")
    same(predecessor["modal_controls"], list(parent.CONTROLS), "modal domain binding")
    for field in ("reference_scope", "lineage_separation"):
        for document in (collapse, factor, modal, raw):
            same(retained[field], document["report"][field], field + " binding")
    same(retained["parent_source_equivalence_status_preserved"], "REJECTED", "closed equivalence boundary")
    failures, checks = [], 0

    def verify(field, actual, expected, **identity):
        nonlocal checks
        checks += 1
        if type(actual) is not type(expected) or actual != expected:
            failures.append({**identity, "field": field, "actual": str(actual), "expected": str(expected)})

    verify("parent_failure_count", retained["failure_count"], 0)
    verify("parent_failures", retained["failures"], [])
    verify("parent_unknown_reasons", retained["unknown_reasons"], [])
    fields = (*parent.SCOPE_FIELDS, "control", "component")
    moving = indexed(retained["nonzero_source_only_movements"], fields)
    zeros = indexed(retained["zero_source_accounts"], fields)
    require(not moving.keys() & zeros.keys(), "ambiguous zero/nonzero membership")
    accounts = indexed(predecessor["coordinate_accounts"], parent.SCOPE_FIELDS)
    same(set(accounts), set(parent.SCOPES), "missing/incompatible coordinate bindings")
    matrix, records, seen, proportional = [], {}, set(), 0
    nonzero_count = 0
    for ai, scope in enumerate(parent.SCOPES):
        account = accounts[scope]
        controls = indexed(account["controls"], ("control",))
        same(set(controls), {(c,) for c in parent.CONTROLS}, "missing/incompatible control bindings")
        base = controls[("frozen_inherited",)]
        common = rational(base[parent.FACTOR])
        identity = dict(zip(parent.SCOPE_FIELDS, scope, strict=True))
        for ci, name in enumerate(parent.CONTROLS):
            control = controls[(name,)]
            item = {**identity, "control": name}
            pointer = f"/report/coordinate_accounts/{ai}/controls/{ci}"
            binding = control["retained_bindings"]
            fr = parent.resolve(factor, pointer)
            same(binding, {**fr["retained_bindings"], "factor_stdout_pin": "factor_stdout",
                           "factor_control_pointer": pointer}, "factor/raw pointer splice")
            same(parent.resolve(modal, binding["modal_control_pointer"])["control"], name,
                 "modal identity binding")
            rr = parent.raw_row(raw, binding["raw_selected_row_pointer"], scope, name)
            detail = parent.raw_row(raw, binding["raw_factor_row_pointer"], scope, name)
            for field in ("factors", "left_weight", "right_weight", parent.FACTOR):
                verify("retained_factor/" + field, control[field], fr[field], **item)
            for field in ("left_weight", "right_weight"):
                verify("raw_factor/" + field, control[field], detail[field], **item)
            f = rational(control[parent.FACTOR])
            verify("common_factor", f, common, **item)
            verify("raw_common_factor", f, rational(rr[parent.FACTOR]), **item)
            same(set(control["factors"]), set(parent.FACTORS), "missing/ambiguous factors")
            same(set(control["factor_deltas"]), set(parent.FACTORS), "missing/ambiguous factor deltas")
            for field in parent.FACTORS:
                verify("factor_equality/" + field, control["factors"][field], base["factors"][field], **item)
                verify("factor_delta/" + field, rational(control["factor_deltas"][field]), Fraction(0), **item)
            verify("row_difference", rational(control["factors"]["row_difference"]),
                   rational(control["left_weight"]) - rational(control["right_weight"]), **item)
            verify("factor_product", f, rational(control["factors"]["norm_weight"])
                   * rational(control["factors"]["reference_inverse_norm_anchor"])
                   * rational(control["factors"]["row_difference"]), **item)
            verify("combined_factor_delta", rational(control["combined_factor_delta"]), f - common, **item)
            same(set(control["components"]), set(parent.COMPONENTS), "missing/ambiguous component")
            for component in parent.COMPONENTS:
                term = control["components"][component]
                same(set(term), TERM_FIELDS, "missing/ambiguous component fields")
                same(set(term["individual_factor_delta_terms"]), set(parent.FACTORS),
                     "missing/ambiguous factor delta terms")
                values = {k: rational(v) for k, v in term.items() if k != "individual_factor_delta_terms"}
                h, h0 = values["hidden"], rational(base["components"][component]["hidden"])
                w, w0 = values["weighted"], rational(base["components"][component]["weighted"])
                dh, dw = values["source_delta"], values["weighted_movement"]
                detail_id = {**item, "component": component}
                expected = {
                    "canonical_hidden": h0, "canonical_weighted": w0,
                    "source_delta": h - h0, "weighted_movement": w - w0,
                    "source_only_term": dh * common, "source_only_residual": dw - dh * common,
                    "combined_factor_delta_term": h0 * (f - common),
                    "source_factor_interaction_term": dh * (f - common),
                    "product_identity_residual": w - h * f,
                    "canonical_product_identity_residual": w0 - h0 * common,
                    "combined_identity_residual": dw - dh * common - h0 * (f - common) - dh * (f - common),
                    "four_term_identity_residual": dw - dh * common,
                }
                for field, value in expected.items():
                    verify(field, values[field], value, **detail_id)
                for field in ("source_only_residual", "product_identity_residual",
                              "canonical_product_identity_residual", "combined_identity_residual",
                              "four_term_identity_residual"):
                    verify("zero/" + field, values[field], Fraction(0), **detail_id)
                for field, value in term["individual_factor_delta_terms"].items():
                    verify("individual_factor_delta_terms/" + field, rational(value), Fraction(0), **detail_id)
                verify("raw_hidden", term["hidden"], rr["hidden_components"][component], **detail_id)
                verify("raw_weighted", term["weighted"], rr["weighted_components"][component], **detail_id)
                expected_dose = Fraction(0)
                if scope[4] == 62 and component in ("actual_residual_boundary", "q24_to_fp16_conversion"):
                    expected_dose = rational(DOSES[name]) * (-1 if component == "actual_residual_boundary" else 1)
                verify("signed_source_dose", dh, expected_dose, **detail_id)
                verify("proportionality", dw, dh * f, **detail_id)
                proportional += int(dw == dh * f)
                key = (*scope, name, component)
                require(key in moving or key in zeros, "missing retained 12ea component")
                row = moving[key] if key in moving else zeros[key]
                if key in moving:
                    same(row["retained_bindings"], binding, "12ea retained binding splice")
                    same(row["retained_8525_pointer"], pointer + "/components/" + component,
                         "12ea component pointer splice")
                expected_row = {**detail_id, "source_delta": str(dh), "weighted_movement": str(dw)}
                if dh:
                    group = next((g for g, names in GROUPS.items() if name in names), None)
                    expected_row.update({
                        "group": group, "absolute_dose": str(abs(dh)), "common_factor": str(f),
                        "source_only_term": str(dh * f), "retained_bindings": binding,
                        "retained_8525_pointer": pointer + "/components/" + component,
                    })
                    nonzero_count += 1
                verify("12ea_component_account", row, expected_row, **detail_id)
                verify("12ea_zero_nonzero_classification", key in moving, bool(dh), **detail_id)
                records[key] = (h, h0, dh, w, w0, dw, f)
                seen.add(key)
            same(set(control["mass"]), {"hidden", "weighted"}, "missing mass branch")
            for branch in ("hidden", "weighted"):
                values = [rational(control["components"][c][branch]) for c in parent.COMPONENTS]
                baseline = [rational(base["components"][c][branch]) for c in parent.COMPONENTS]
                total, total0 = sum(values, Fraction(0)), sum(baseline, Fraction(0))
                absolute, absolute0 = sum(map(abs, values), Fraction(0)), sum(map(abs, baseline), Fraction(0))
                masses = {"signed": (total, total0), "absolute": (absolute, absolute0),
                          "cancellation_absolute_mass": (absolute - abs(total), absolute0 - abs(total0))}
                same(set(control["mass"][branch]), set(masses), "missing mass fields")
                for field, (value, canonical) in masses.items():
                    verify("mass/" + branch + "/" + field, control["mass"][branch][field],
                           {"value": str(value), "canonical": str(canonical),
                            "movement": str(value - canonical)}, **item)
        for alias, canonical, dose_value in ALIASES:
            left, right = controls[(alias,)], controls[(canonical,)]
            left_fields, right_fields = dict(leaves(left)), dict(leaves(right))
            same(set(left_fields), set(right_fields), "missing/ambiguous alias fields")
            comparisons = []
            for field in sorted(left_fields):
                actual, expected = left_fields[field], right_fields[field]
                # Identities/pointers must remain distinct; each was authenticated above.
                identity_field = field == "/control" or field.startswith("/retained_bindings/")
                equal = True if identity_field else type(actual) is type(expected) and actual == expected
                if not identity_field:
                    verify("alias" + field, actual, expected, **identity, alias=alias, canonical=canonical)
                comparisons.append({"field": field, "alias_value": actual, "canonical_value": expected,
                                    "rule": "authenticated_control_specific_binding" if identity_field
                                    else "exact_equality", "equal_under_alias_map": equal})
            matrix.append({**identity, "alias": alias, "canonical": canonical,
                           "coordinate62_absolute_dose": dose_value,
                           "coordinate_absolute_dose": dose_value if scope[4] == 62 else "0",
                           "retained_12ea_stdout_pin": PINS["stdout"],
                           "alias_bindings": left["retained_bindings"],
                           "canonical_bindings": right["retained_bindings"],
                           "fields": comparisons, "equal_under_alias_map": all(c["equal_under_alias_map"]
                                                                              for c in comparisons)})
    verify("12ea_component_census", set(moving) | set(zeros), seen)
    reverse = []
    retained_reverse = indexed(retained["reverse_pair_checks"], ("coordinate", "control", "component"))
    for key, forward in records.items():
        table, left, right, branch, coordinate, name, component = key
        if (left, right) != (34319, 319):
            continue
        back = (table, right, left, branch, coordinate, name, component)
        require(back in records, "missing retained reverse orientation")
        backward = records[back]
        hidden_same = forward[:3] == backward[:3]
        weighted_opposite = forward[3:] == tuple(-v for v in backward[3:])
        item = {"coordinate": coordinate, "control": name, "component": component}
        verify("reverse_hidden_and_dose", hidden_same, True, **item)
        verify("reverse_weighted_and_factor", weighted_opposite, True, **item)
        row = {**item, "hidden_and_dose_equal": hidden_same,
               "weighted_and_factor_opposite": weighted_opposite}
        require((coordinate, name, component) in retained_reverse, "missing retained reverse check")
        verify("12ea_reverse_check", retained_reverse[(coordinate, name, component)], row)
        reverse.append(row)
    verify("reverse_check_census", len(retained_reverse), 112)
    groups = [{"group": group, "controls": names, "absolute_dose": DOSES[group],
               "ordered_pair_signatures": [
                   {"left_id": left, "right_id": right, "control": name, "absolute_dose": DOSES[group]}
                   for left, right in ((34319, 319), (319, 34319)) for name in names]}
              for group, names in GROUPS.items()]
    verify("dose_groups", retained["dose_groups"], groups)
    verify("dose_rank", retained["absolute_dose_rank_descending"], list(GROUPS))
    gaps = [{"larger": left, "smaller": right,
             "gap": str(rational(DOSES[left]) - rational(DOSES[right])), "positive": True}
            for left, right in (("frozen_inherited_o", "scratch"), ("frozen_inherited_o", "mapped62"),
                                ("scratch", "mapped62"))]
    verify("dose_gaps", retained["pairwise_rational_gaps"], gaps)
    verify("zero_dose_controls", retained["zero_dose_controls"],
           ["frozen_inherited", "frozen_inherited_down", "inherited_native"])
    counts = {"component_accounts": len(records), "control_coordinate_accounts": len(records) // 7,
              "nonzero_source_only_movements": nonzero_count, "zero_source_accounts": len(records) - nonzero_count,
              "factor_proportionality_checks": proportional, "reverse_component_checks": len(reverse)}
    verify("expected_counts", counts, COUNTS)
    verify("12ea_counts_unchanged", retained["counts"], counts)
    for field, expected in (("component_account_count", 280), ("source_only_movement_count", 20),
                            ("zero_source_count", 260), ("nonzero_weighted_movement_count", 20)):
        verify("8525_counts/" + field, predecessor["counts"][field], expected)
    verify("8525_coordinate_count", predecessor["coordinate_account_count"], 5)
    verify("8525_control_count", predecessor["control_coordinate_account_count"], 40)
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "failure_count": len(failures),
        "failures": failures, "unknown_reasons": [], "exact_check_count": checks,
        "alias_map": {alias: canonical for alias, canonical, _ in ALIASES},
        "alias_pair_matrix": matrix, "alias_coordinate_comparisons": len(matrix),
        "alias_component_comparisons": len(matrix) * 7, "counts": counts,
        "retained_12ea_counts": retained["counts"], "dose_groups": retained["dose_groups"],
        "absolute_dose_rank_descending": retained["absolute_dose_rank_descending"],
        "pairwise_rational_gaps": retained["pairwise_rational_gaps"],
        "zero_dose_controls": retained["zero_dose_controls"], "reverse_pair_checks": reverse,
        "reference_scope": retained["reference_scope"], "lineage_separation": retained["lineage_separation"],
        "parent_source_equivalence_status_preserved": "REJECTED", "claim_boundary": parent.BOUNDARY,
    }


def check():
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    audit, sources = {"forbidden_calls": 0}, []
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    boundary = "Retained-only CPU non-admission accounting; independent Host Reviewer required."
    try:
        dose, parent, helper = load_helpers()
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        allowed = allowed_paths(dose, parent, helper)
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
                    sources.append(binding)
                else:
                    runtime["executable_pin"] = binding
            documents, authentication = authenticate(dose, parent, helper)
            result = classify(documents, parent)
            helper.zero_counters(audit)
    except ERRORS as error:
        result = unknown(error, "authentication_or_runtime")
        authentication["error"] = result["unknown_reasons"][0]
    except RuntimeError as error:
        if type(error).__name__ != "ForbiddenOperation":
            raise
        result = unknown(error, "forbidden_operation")
    return {"diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_DOWN_AXIS_INVARIANCE_CLASSIFIER",
            "decision": result["decision"], "command": COMMAND, "artifact_authentication": authentication,
            "source_test_pins": sources, "runtime": runtime, "report": result,
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
