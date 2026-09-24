"""Stdout-only exact affine normal forms of reviewed retained canonical dose accounts."""

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
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_normal_form_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = NAME.replace("canonical_dose_affine_normal_form", "modal_canonical_dose_class_quotient")
CAPTURE = ROOT / "build/selected-direct-hidden-modal-dose-class-quotient-b4b9b98aaea7-attempt001"
RUN = CAPTURE / "run"
ENV_KEYS = ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)
CLASSES = {
    "frozen_inherited": ("0", ["frozen_inherited", "frozen_inherited_down", "inherited_native"]),
    "frozen_inherited_o": ("3/32768", ["frozen_inherited_o", "frozen_inherited_o_down"]),
    "scratch": ("321/4194304", ["scratch", "scratch_down"]),
    "mapped62": ("933/16777216", ["mapped62"]),
}
COUNTS = {
    "component_accounts": 280, "control_coordinate_accounts": 40,
    "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
    "factor_proportionality_checks": 280, "reverse_component_checks": 112,
}
TERM_FIELDS = (
    "hidden", "canonical_hidden", "weighted", "canonical_weighted", "source_delta",
    "weighted_movement", "source_only_term", "source_only_residual",
    "combined_factor_delta_term", "source_factor_interaction_term", "product_identity_residual",
    "canonical_product_identity_residual", "combined_identity_residual", "four_term_identity_residual",
)


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 1711087,
                  "e105964af002912b9af0c07813e15eda7b6449858b3ba61114bc2d2ff2cbb34b"),
    "capture": pin(RUN / "capture.json", 19248,
                   "8e03c7e51764b45ea74b2430c3c834ae4d3c72b4b335ceca726283b21b5c33f4"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1100,
                         "15f6cf962ce725d8d0cae021aceec6aaddb9d687a5889dedc9c3762e1262159f"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 34949,
                  "0d2b6fcd5b60632e04a55826c81c434cc10cc55ae7e969f067f4fabaf23d0337"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 19502,
                "36c1a893efeae5e28ef27ade1df791f0ad7d433d48a23fe12e6fd02d5d7f7c77"),
    "review": pin(Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
                  / "b4b9b98aaea7/round-0001.json", 688,
                  "78648cfa632e327c9bdfd7a0bbd36a2c80e40ba3ee39573f5aa6be2153675e6c"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(type(actual) is type(expected) and actual == expected, message)


def rational(text):
    same(type(text), str, "missing/non-string rational")
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
    # Only authenticated definitions and byte/pointer helpers, never ancestor classifiers.
    data = bound_bytes(PINS["source"])
    quotient = types.ModuleType("_authenticated_b4_retained_helpers")
    quotient.__file__ = PINS["source"]["path"]
    exec(compile(data, quotient.__file__, "exec"), quotient.__dict__)
    return quotient, *quotient.load_helpers()


def allowed_paths(quotient, *modules):
    helper = modules[-1]
    return quotient.allowed_paths(*modules) | {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        *(str(RUN / (label + suffix))
          for label in (*helper.LABELS, "concurrency") for suffix in helper.SUFFIXES),
        *(str(CAPTURE / ("launcher." + suffix))
          for suffix in ("identity.json", "stdout", "stderr", "whole-command.log")),
    }


def environment_command(environment, argv):
    return " ".join(f"{k}={shlex.quote(environment[k])}" for k in ENV_KEYS) + " " + shlex.join(argv)


def authenticate(quotient, *modules):
    helper = modules[-1]
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    retained, capture, outer, review = [
        helper.decode(data[key], metadata=True)
        for key in ("stdout", "capture", "outer_capture", "review")
    ]
    helper.terminal_review(review, "b4b9b98aaea7")
    same((retained["diagnostic_id"], retained["version"], retained["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_CANONICAL_DOSE_CLASS_QUOTIENT_CLASSIFIER"),
         "b4 identity")
    same((retained["artifact_authentication"]["status"], retained["decision"], retained["report"]["decision"]),
         ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "b4 authenticated supported prerequisite")
    require(capture["success"] is True and capture["failure"] is None, "b4 capture failure")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "b4 account/runtime")
    for sources in (preflight["sources"], capture["sources_after"], retained["source_test_pins"]):
        same(sources, [PINS["source"], PINS["test"]], "b4 source/test pins")
    same(preflight["independent_host_review"], "REQUIRED", "b4 review gate")
    same(preflight["model_or_service_calls_authorized"], 0, "b4 service budget")
    same(retained["command"], quotient.COMMAND, "b4 disclosed command")
    runtime = retained["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "b4 runtime splice")
    same(runtime["environment"], {k: preflight["environment"][k]
                                 for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "b4 environment splice")
    same(runtime["environment"], {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
         "b4 command-local environment")
    same(runtime["executable_pin"], preflight["executable"], "b4 executable pin")
    same(runtime["executable_pin"]["path"], str(Path(PYTHON).resolve()), "b4 executable identity")
    bound_bytes(runtime["executable_pin"])
    same(set(retained["dispatch_and_write_audit"]), set(helper.COUNTERS), "b4 counter census")
    require(all(type(v) is int and v == 0 for v in retained["dispatch_and_write_audit"].values()),
            "b4 forbidden counter")
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"], "b4 command census")
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "b4 command failure")
        same([p["path"] for p in result["files"]],
             [str(RUN / (result["label"] + s)) for s in helper.SUFFIXES], "b4 member paths")
        command, argv, env, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "b4 command bytes")
        same(result["command"], environment_command(preflight["environment"], result["argv"]),
             "b4 command/argv splice")
        same(helper.decode(argv), result["argv"], "b4 argv bytes")
        same(helper.decode(env), preflight, "b4 environment bytes")
        same(error, b"", "b4 stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
             + output + b"\nSTDERR\n" + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n",
             "b4 whole-command bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "b4 branch")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "b4 concurrency")
        if result["label"] == "check":
            same(output, data["stdout"], "b4 stdout splice")
            same(result["argv"], [PYTHON, "-B", "-m", quotient.MODULE, "--check"], "b4 check argv")
    same(outer["exit_status"], 0, "b4 outer exit")
    same(outer["timed_out"], False, "b4 outer timeout")
    same([p["path"] for p in outer["files"]],
         [str(CAPTURE / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "b4 outer paths")
    identity_bytes, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    identity = helper.decode(identity_bytes)
    same(identity["launcher"], PINS["test"], "b4 outer launcher")
    same(identity["argv"], [PYTHON, "-B", PINS["test"]["path"], "--run", str(RUN)], "b4 outer argv")
    same(identity["command"], environment_command(preflight["environment"], identity["argv"]),
         "b4 outer command")
    same({k: identity[k] for k in ("cwd", "uid", "environment")},
         {k: preflight[k] for k in ("cwd", "uid", "environment")}, "b4 outer environment")
    same(error, b"", "b4 outer stderr")
    same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "b4 outer whole bytes")
    same(helper.decode(output, metadata=True),
         {"capture": PINS["capture"], "success": True, "failure": None}, "b4 outer receipt")
    documents, upstream = quotient.authenticate(*modules)
    same(retained["artifact_authentication"], upstream, "b4 predecessor authentication splice")
    return (retained, *documents), {
        "status": "AUTHENTICATED", "reviewed_mission": "b4b9b98aaea7", "input_pins": PINS,
        "review": review, "complete_command_capture": capture, "outer_capture": outer,
        "retained_predecessor_raw_bindings": upstream,
    }


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


def numeric_fields(control, parent):
    schema = {
        "factors": dict.fromkeys(parent.FACTORS, "0"),
        "factor_deltas": dict.fromkeys(parent.FACTORS, "0"),
        "left_weight": "0", "right_weight": "0", parent.FACTOR: "0", "combined_factor_delta": "0",
        "components": {c: {**dict.fromkeys(TERM_FIELDS, "0"),
                            "individual_factor_delta_terms": dict.fromkeys(parent.FACTORS, "0")}
                       for c in parent.COMPONENTS},
        "mass": {b: {m: dict.fromkeys(("value", "canonical", "movement"), "0")
                     for m in ("signed", "absolute", "cancellation_absolute_mass")}
                 for b in ("hidden", "weighted")},
    }
    same(set(control), {*schema, "control", "retained_bindings"}, "missing/ambiguous control fields")
    fields = dict(leaves({k: control[k] for k in schema}))
    same(set(fields), set(dict(leaves(schema))), "missing/ambiguous numeric fields")
    return {k: rational(v) for k, v in fields.items()}


def reversal(field):
    target = {"/left_weight": "/right_weight", "/right_weight": "/left_weight"}.get(field, field)
    parts = field.split("/")
    preserve = (
        field in ("/left_weight", "/right_weight")
        or parts[1] in ("factors", "factor_deltas") and parts[2] != "row_difference"
        or parts[1] == "components" and parts[3] in ("hidden", "canonical_hidden", "source_delta")
        or parts[1] == "mass" and (parts[2] == "hidden" or parts[3] != "signed")
    )
    return target, 1 if preserve else -1


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def classify(documents, parent):
    try:
        return report(documents, parent)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(documents, parent):
    quotient, down, dose, collapse, factor, modal, raw = documents
    q, ranked, accounts_report = quotient["report"], dose["report"], collapse["report"]
    for document in (quotient, down, dose, collapse):
        same((document["artifact_authentication"]["status"], document["decision"], document["report"]["decision"]),
             ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "authenticated supported prerequisite")
    for field in ("reference_scope", "lineage_separation"):
        for document in documents[1:]:
            same(q[field], document["report"][field], field + " binding")
    same(q["parent_source_equivalence_status_preserved"], "REJECTED", "closed branch boundary")
    same(accounts_report["modal_controls"], list(parent.CONTROLS), "modal domain binding")
    failures = []

    def verify(field, actual, expected, **identity):
        equal = type(actual) is type(expected) and actual == expected
        if not equal:
            failures.append({**identity, "field": field, "actual": str(actual), "expected": str(expected)})
        return equal

    classes = [{"canonical": c, "absolute_dose": d, "controls": names} for c, (d, names) in CLASSES.items()]
    membership = {c: names for c, (_, names) in CLASSES.items()}
    aliases = {name: c for c, (_, names) in CLASSES.items() for name in names if name != c}
    verify("dose_classes", q["dose_classes"], classes)
    verify("quotient_membership", q["observed_quotient_membership"], membership)
    verify("alias_map", q["alias_map"], aliases)
    for previous in (q, down["report"], ranked, accounts_report):
        for field, expected in (("failure_count", 0), ("failures", []), ("unknown_reasons", [])):
            verify("predecessor/" + field, previous[field], expected)
    for key, size in (("within_class_equality_matrix", 90), ("cross_class_proportionality_checks", 210),
                      ("reverse_numeric_orientation_matrix", 16), ("reverse_pair_checks", 112)):
        verify(key + "/count", len(q[key]), size)
    accounts = indexed(accounts_report["coordinate_accounts"], parent.SCOPE_FIELDS)
    same(set(accounts), set(parent.SCOPES), "missing/incompatible coordinate bindings")
    qrows = indexed(q["coordinate_account_checks"], (*parent.SCOPE_FIELDS, "control"))
    same(set(qrows), {(*s, c) for s in parent.SCOPES for c in parent.CONTROLS},
         "missing/incompatible quotient account bindings")
    moving = indexed(ranked["nonzero_source_only_movements"], (*parent.SCOPE_FIELDS, "control", "component"))
    zeros = indexed(ranked["zero_source_accounts"], (*parent.SCOPE_FIELDS, "control", "component"))
    require(not moving.keys() & zeros.keys(), "ambiguous zero/nonzero membership")
    dose_by_control = {n: rational(d) for d, names in CLASSES.values() for n in names}
    canonical_by_control = {n: c for c, (_, names) in CLASSES.items() for n in names}
    coefficients, numeric, controls_by_scope = {}, {}, {}
    coefficient_matrix, reconstruction, closures, source_matrix = [], [], [], []
    component_keys, nonzero_count, proportional = set(), 0, 0
    for ai, scope in enumerate(parent.SCOPES):
        identity = dict(zip(parent.SCOPE_FIELDS, scope, strict=True))
        controls = indexed(accounts[scope]["controls"], ("control",))
        same(set(controls), {(c,) for c in parent.CONTROLS}, "missing/incompatible control bindings")
        controls_by_scope[scope] = controls
        fields_by_control = {n: numeric_fields(controls[(n,)], parent) for n in parent.CONTROLS}
        base = fields_by_control["frozen_inherited"]
        coeffs = {}
        for field, baseline in base.items():
            candidates = {c: (fields_by_control[c][field] - baseline) / rational(d)
                          for c, (d, _) in CLASSES.items() if rational(d)}
            slope = candidates["frozen_inherited_o"]
            unique = len(set(candidates.values())) == 1
            verify("coefficient_uniqueness" + field, unique, True, **identity)
            if scope[4] == 241:
                verify("context_zero_slope" + field, slope, Fraction(0), **identity)
            coeffs[field] = {
                "baseline": str(baseline), "slope": str(slope),
                "candidate_slopes": {k: str(v) for k, v in candidates.items()},
                "candidate_slope_residuals": {k: str(v - slope) for k, v in candidates.items()},
                "unique": unique, "design_rank": 2,
            }
        coefficients[scope] = coeffs
        coefficient_matrix.append({**identity, "baseline_control": "frozen_inherited",
                                   "abscissa": "canonical_class_coordinate62_absolute_dose",
                                   "fields": coeffs})
        f0 = base["/" + parent.FACTOR]
        for ci, name in enumerate(parent.CONTROLS):
            item = {**identity, "control": name}
            control, actual = controls[(name,)], fields_by_control[name]
            numeric[(*scope, name)] = actual
            d = dose_by_control[name]
            canonical = canonical_by_control[name]
            pointer = f"/report/coordinate_accounts/{ai}/controls/{ci}"
            binding = control["retained_bindings"]
            fr = parent.resolve(factor, pointer)
            same(binding, {**fr["retained_bindings"], "factor_stdout_pin": "factor_stdout",
                           "factor_control_pointer": pointer}, "factor/raw pointer splice")
            same(parent.resolve(modal, binding["modal_control_pointer"])["control"], name,
                 "modal control identity")
            rr = parent.raw_row(raw, binding["raw_selected_row_pointer"], scope, name)
            detail = parent.raw_row(raw, binding["raw_factor_row_pointer"], scope, name)
            qr = qrows[(*scope, name)]
            same(qr["retained_bindings"], binding, "quotient distinct retained bindings")
            same(qr["retained_8525_pointer"], pointer, "quotient account pointer")
            same(qr["retained_390_stdout_pin"], quotient["artifact_authentication"]["input_pins"]["stdout"],
                 "quotient predecessor stdout pointer")
            same(set(qr["numeric_field_equality"]), set(actual), "quotient numeric field census")
            verify("quotient_fields", qr["numeric_field_equality"], dict.fromkeys(actual, True), **item)
            verify("quotient_all_fields", qr["all_numeric_fields_dose_determined"], True, **item)
            verify("quotient_class", qr["canonical_class"], canonical, **item)
            verify("quotient_dose", qr["coordinate62_absolute_dose"], str(d), **item)
            verify("quotient_coordinate_dose", qr["coordinate_absolute_dose"],
                   str(d if scope[4] == 62 else Fraction(0)), **item)
            verify("quotient_factor", qr["common_factor"], control[parent.FACTOR], **item)
            for field in ("factors", "left_weight", "right_weight", parent.FACTOR):
                verify("retained_factor/" + field, control[field], fr[field], **item)
            for field in ("left_weight", "right_weight"):
                verify("raw_factor/" + field, control[field], detail[field], **item)
            verify("raw_common_factor", control[parent.FACTOR], rr[parent.FACTOR], **item)
            residuals = {}
            for field, observed in actual.items():
                c = coeffs[field]
                predicted = rational(c["baseline"]) + d * rational(c["slope"])
                residual = observed - predicted
                verify("reconstruction" + field, residual, Fraction(0), **item)
                residuals[field] = {"retained": str(observed), "reconstructed": str(predicted),
                                    "residual": str(residual)}
            reconstruction.append({**item, "canonical_class": canonical, "dose": str(d),
                                   "retained_bindings": binding, "retained_component_account_pointer": pointer,
                                   "fields": residuals})
            closure = {}

            def close(field, observed, expected):
                residual = observed - expected
                closure[field] = str(residual)
                verify("arithmetic/" + field, residual, Fraction(0), **item)

            f = actual["/" + parent.FACTOR]
            close("factor_product", f, actual["/factors/norm_weight"]
                  * actual["/factors/reference_inverse_norm_anchor"] * actual["/factors/row_difference"])
            close("row_difference", actual["/factors/row_difference"],
                  actual["/left_weight"] - actual["/right_weight"])
            close("factor_invariance", f, f0)
            close("combined_factor_delta", actual["/combined_factor_delta"], Fraction(0))
            for factor_name in parent.FACTORS:
                close("factor_delta/" + factor_name, actual["/factor_deltas/" + factor_name], Fraction(0))
                close("factor_invariance/" + factor_name, actual["/factors/" + factor_name],
                      base["/factors/" + factor_name])
            for component in parent.COMPONENTS:
                prefix = "/components/" + component + "/"
                h, h0 = actual[prefix + "hidden"], base[prefix + "hidden"]
                w, w0 = actual[prefix + "weighted"], base[prefix + "weighted"]
                dh, dw = actual[prefix + "source_delta"], actual[prefix + "weighted_movement"]
                sign = (-1 if component == "actual_residual_boundary" else
                        1 if component == "q24_to_fp16_conversion" else 0) if scope[4] == 62 else 0
                expected = {
                    "canonical_hidden": h0, "canonical_weighted": w0, "source_delta": h - h0,
                    "weighted": h * f, "weighted_movement": w - w0, "source_only_term": dh * f0,
                    **dict.fromkeys(TERM_FIELDS[7:], Fraction(0)),
                }
                for field, value in expected.items():
                    close(component + "/" + field, actual[prefix + field], value)
                close(component + "/canonical_product", w0, h0 * f0)
                close(component + "/expected_source_dose", dh, sign * d)
                close(component + "/source_only_proportionality", dw, dh * f)
                proportional += int(dw == dh * f)
                for factor_name in parent.FACTORS:
                    close(component + "/individual_factor_delta_terms/" + factor_name,
                          actual[prefix + "individual_factor_delta_terms/" + factor_name], Fraction(0))
                verify("raw_hidden/" + component, str(h), rr["hidden_components"][component], **item)
                verify("raw_weighted/" + component, str(w), rr["weighted_components"][component], **item)
                key = (*scope, name, component)
                component_keys.add(key)
                require(key in moving or key in zeros, "missing retained component classification")
                row = moving[key] if key in moving else zeros[key]
                expected_row = {**item, "component": component, "source_delta": str(dh),
                                "weighted_movement": str(dw)}
                if sign * d:
                    expected_row.update({
                        "group": canonical, "absolute_dose": str(abs(dh)), "common_factor": str(f),
                        "source_only_term": str(dw), "retained_bindings": binding,
                        "retained_8525_pointer": pointer + "/components/" + component,
                    })
                    if key in moving:
                        same(row["retained_bindings"], binding, "dose binding splice")
                        same(row["retained_8525_pointer"], expected_row["retained_8525_pointer"],
                             "dose component pointer splice")
                verify("retained_source_account", row, expected_row, **item, component=component)
                verify("zero_nonzero_classification", key in moving, bool(dh), **item, component=component)
                nonzero_count += int(bool(dh))
                source_matrix.append({**item, "component": component, "expected_source_slope": str(sign),
                                      "source_slope": coeffs[prefix + "source_delta"]["slope"],
                                      "source_delta": str(dh), "weighted_movement": str(dw),
                                      "dose_residual": str(dh - sign * d),
                                      "proportionality_residual": str(dw - dh * f)})
            for branch in ("hidden", "weighted"):
                values = [actual["/components/" + c + "/" + branch] for c in parent.COMPONENTS]
                baselines = [base["/components/" + c + "/" + branch] for c in parent.COMPONENTS]
                total, total0 = sum(values, Fraction(0)), sum(baselines, Fraction(0))
                absolute, absolute0 = sum(map(abs, values), Fraction(0)), sum(map(abs, baselines), Fraction(0))
                for mass, (value, value0) in {
                    "signed": (total, total0), "absolute": (absolute, absolute0),
                    "cancellation_absolute_mass": (absolute - abs(total), absolute0 - abs(total0)),
                }.items():
                    for field, expected_value in (("value", value), ("canonical", value0),
                                                  ("movement", value - value0)):
                        key = "/mass/" + branch + "/" + mass + "/" + field
                        close(key, actual[key], expected_value)
            closures.append({**item, "residuals": closure})
    reverse_coefficients, reverse_numeric, reverse_components = [], [], []
    for scope in parent.SCOPES:
        table, left, right, branch, coordinate = scope
        if (left, right) != (34319, 319):
            continue
        back = (table, right, left, branch, coordinate)
        oriented = {}
        for field, c in coefficients[scope].items():
            target, sign = reversal(field)
            rc = coefficients[back][target]
            residual = {k: str(rational(rc[k]) - sign * rational(c[k])) for k in ("baseline", "slope")}
            for k, value in residual.items():
                verify("reverse_coefficient/" + k + field, value, "0", coordinate=coordinate)
            oriented[field] = {"reverse_field": target, "multiplier": sign, "residuals": residual}
        reverse_coefficients.append({"coordinate": coordinate, "forward_pair": [left, right],
                                     "reverse_pair": [right, left], "fields": oriented})
        for name in parent.CONTROLS:
            a, b = numeric[(*scope, name)], numeric[(*back, name)]
            residual = {}
            for field, value in a.items():
                target, sign = reversal(field)
                residual[field] = str(b[target] - sign * value)
                verify("reverse_numeric" + field, residual[field], "0", coordinate=coordinate, control=name)
            reverse_numeric.append({"coordinate": coordinate, "control": name, "residuals": residual})
            for component in parent.COMPONENTS:
                prefix = "/components/" + component + "/"
                hidden = ("hidden", "canonical_hidden", "source_delta")
                reverse_components.append({
                    "coordinate": coordinate, "control": name, "component": component,
                    "hidden_and_dose_equal": all(a[prefix + k] == b[prefix + k] for k in hidden),
                    "weighted_and_factor_opposite": all(a[prefix + k] == -b[prefix + k]
                                                       for k in TERM_FIELDS if k not in hidden)
                    and a["/" + parent.FACTOR] == -b["/" + parent.FACTOR],
                })
    verify("retained_reverse_orientation", q["reverse_pair_checks"], reverse_components)
    verify("component_census", set(moving) | set(zeros), component_keys)
    counts = {"component_accounts": len(component_keys), "control_coordinate_accounts": len(numeric),
              "nonzero_source_only_movements": nonzero_count, "zero_source_accounts": len(component_keys) - nonzero_count,
              "factor_proportionality_checks": proportional, "reverse_component_checks": len(reverse_components)}
    verify("expected_counts", counts, COUNTS)
    for previous in (q["counts"], q["retained_390_counts"], down["report"]["counts"], ranked["counts"]):
        verify("unchanged_counts", previous, counts)
    for field, expected in (("component_account_count", 280), ("source_only_movement_count", 20),
                            ("zero_source_count", 260), ("nonzero_weighted_movement_count", 20)):
        verify("collapse_counts/" + field, accounts_report["counts"][field], expected)
    for field, expected in (("coordinate_account_count", 5), ("control_coordinate_account_count", 40)):
        verify("collapse_counts/" + field, accounts_report[field], expected)
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "failure_count": len(failures),
        "failures": failures, "unknown_reasons": [], "counts": counts,
        "normal_form": "field(d) = zero_dose_baseline + slope * canonical_class_absolute_dose",
        "coefficient_uniqueness": "Three nonzero canonical-class slopes must agree; fixed distinct doses give rank 2.",
        "context_rule": "Coordinate 241 uses the varying coordinate-62 class-dose abscissa and has slope zero.",
        "coefficient_matrix": coefficient_matrix, "reconstruction_residual_matrix": reconstruction,
        "arithmetic_closure_residual_matrix": closures, "source_movement_matrix": source_matrix,
        "reverse_coefficient_matrix": reverse_coefficients, "reverse_numeric_residual_matrix": reverse_numeric,
        "reverse_pair_checks": reverse_components,
        "retained_quotient_bindings": {k: q[k] for k in (
            "alias_map", "observed_quotient_membership", "dose_classes", "counts", "retained_390_counts",
            "coordinate_account_checks", "within_class_equality_matrix", "cross_class_proportionality_checks",
            "reverse_pair_checks", "reverse_numeric_orientation_matrix", "retained_390_stdout_pin")},
        "retained_b4_stdout_pin": PINS["stdout"], "reference_scope": q["reference_scope"],
        "lineage_separation": q["lineage_separation"], "parent_source_equivalence_status_preserved": "REJECTED",
        "claim_boundary": parent.BOUNDARY,
    }


def check():
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    audit, sources = {"forbidden_calls": 0}, []
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    boundary = "Retained-only CPU non-admission accounting; independent Host Reviewer required."
    try:
        modules = load_helpers()
        parent, helper = modules[-2:]
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        allowed = allowed_paths(*modules)
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
            documents, authentication = authenticate(*modules)
            result = classify(documents, parent)
            require(all(type(v) is int and v == 0 for v in audit.values()), "forbidden counter")
    except ERRORS as error:
        result = unknown(error, "authentication_or_runtime")
        authentication["error"] = result["unknown_reasons"][0]
    except RuntimeError as error:
        if type(error).__name__ != "ForbiddenOperation":
            raise
        result = unknown(error, "forbidden_operation")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_NORMAL_FORM_CLASSIFIER",
        "decision": result["decision"], "command": COMMAND, "artifact_authentication": authentication,
        "source_test_pins": sources, "runtime": runtime, "report": result,
        "dispatch_and_write_audit": audit, "claim_boundary": boundary,
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
