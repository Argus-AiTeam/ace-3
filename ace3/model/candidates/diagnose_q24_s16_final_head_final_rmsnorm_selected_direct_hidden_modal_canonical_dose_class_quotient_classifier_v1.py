"""Stdout-only exact canonical dose quotient of authenticated retained modal controls."""

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_canonical_dose_class_quotient_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = NAME.replace("canonical_dose_class_quotient", "down_axis_invariance")
CAPTURE = ROOT / "build/selected-direct-hidden-modal-down-axis-invariance-390eb4740186-attempt001"
RUN = CAPTURE / "run"
REVIEWS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
ENV_KEYS = ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 683012,
                  "89a991bc9d52b08e0abf391793e1723e37a36c095dcdd703ada1d8249bc1e193"),
    "capture": pin(RUN / "capture.json", 19004,
                   "f7d706ec91626944d5aece95afc925ce3925b4cb1c85082ef1b630e239e49d5d"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1104,
                         "a75380ab965a9e67e5f7db2c78560c47c24fd14bec0a737e6e8dc7bff5493029"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 30293,
                  "659c8401fe80d404c5acd8c03b939c593a4ddf3b522abcec5372f53860e01bed"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 19314,
                "a0019a4c530359a847c289e33c111bca3a29db8705b1c350327ebecc1694354c"),
    "review": pin(REVIEWS / "390eb4740186/round-0001.json", 690,
                  "565e844e16961ba40c7f2afc497abc5b4d0d26388cd71677dab632426da857e4"),
}
ALIASES = {
    "frozen_inherited_down": "frozen_inherited",
    "frozen_inherited_o_down": "frozen_inherited_o",
    "scratch_down": "scratch",
    "inherited_native": "frozen_inherited",
}
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
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(type(actual) is type(expected) and actual == expected, message)


def rational(text):
    require(type(text) is str, "missing/non-string rational")
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
    # Pinned definitions/authentication only, never ancestor semantic entrypoints.
    data = bound_bytes(PINS["source"])
    down = types.ModuleType("_authenticated_390_retained_helpers")
    down.__file__ = PINS["source"]["path"]
    exec(compile(data, down.__file__, "exec"), down.__dict__)
    return down, *down.load_helpers()


def allowed_paths(down, dose, parent, helper):
    return down.allowed_paths(dose, parent, helper) | {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        *(str(RUN / (label + suffix))
          for label in (*helper.LABELS, "concurrency") for suffix in helper.SUFFIXES),
        *(str(CAPTURE / ("launcher." + suffix))
          for suffix in ("identity.json", "stdout", "stderr", "whole-command.log")),
    }


def environment_command(environment, argv):
    return " ".join(f"{k}={shlex.quote(environment[k])}" for k in ENV_KEYS) + " " + shlex.join(argv)


def authenticate(down, dose, parent, helper):
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    retained, capture, outer, review = [
        helper.decode(data[key], metadata=True)
        for key in ("stdout", "capture", "outer_capture", "review")
    ]
    helper.terminal_review(review, "390eb4740186")
    same((retained["diagnostic_id"], retained["version"], retained["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_DOWN_AXIS_INVARIANCE_CLASSIFIER"),
         "390 identity")
    same((retained["artifact_authentication"]["status"], retained["decision"],
          retained["report"]["decision"]), ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"),
         "390 authenticated supported prerequisite")
    require(capture["success"] is True and capture["failure"] is None, "390 capture failure")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "390 account/runtime")
    for sources in (preflight["sources"], capture["sources_after"], retained["source_test_pins"]):
        same(sources, [PINS["source"], PINS["test"]], "390 source/test pins")
    same(preflight["independent_host_review"], "REQUIRED", "390 review gate")
    same(retained["command"], down.COMMAND, "390 disclosed command")
    runtime = retained["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "390 runtime splice")
    same(runtime["environment"], {k: preflight["environment"][k]
                                 for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "390 environment splice")
    same(runtime["environment"], {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
         "390 command-local environment")
    same(runtime["executable_pin"], preflight["executable"], "390 runtime executable pin")
    same(runtime["executable_pin"]["path"], str(Path(PYTHON).resolve()), "390 executable identity")
    bound_bytes(runtime["executable_pin"])
    same(set(retained["dispatch_and_write_audit"]), set(helper.COUNTERS), "390 counter census")
    require(all(type(v) is int and v == 0 for v in retained["dispatch_and_write_audit"].values()),
            "390 forbidden counter")
    same([r["label"] for r in capture["results"]],
         ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"],
         "390 command census")
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "390 command failure")
        same([p["path"] for p in result["files"]],
             [str(RUN / (result["label"] + s)) for s in helper.SUFFIXES], "390 member paths")
        command, argv, env, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "390 command bytes")
        same(result["command"], environment_command(preflight["environment"], result["argv"]),
             "390 command/argv splice")
        same(helper.decode(argv), result["argv"], "390 argv bytes")
        same(helper.decode(env), preflight, "390 environment bytes")
        same(error, b"", "390 stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
             + output + b"\nSTDERR\n" + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n",
             "390 whole-command bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "390 branch")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "390 concurrency")
        if result["label"] == "check":
            same(output, data["stdout"], "390 stdout splice")
            same(result["argv"], [PYTHON, "-B", "-m", down.MODULE, "--check"], "390 check argv")
    same((outer["exit_status"], outer["timed_out"]), (0, False), "390 outer completion")
    same([p["path"] for p in outer["files"]],
         [str(CAPTURE / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "390 outer paths")
    identity_bytes, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    identity = helper.decode(identity_bytes)
    same(identity["launcher"], PINS["test"], "390 outer launcher")
    same(identity["argv"], [PYTHON, "-B", PINS["test"]["path"], "--run", str(RUN)], "390 outer argv")
    same(identity["command"], environment_command(preflight["environment"], identity["argv"]),
         "390 outer command")
    same({k: identity[k] for k in ("cwd", "uid", "environment")},
         {k: preflight[k] for k in ("cwd", "uid", "environment")}, "390 outer environment")
    same(error, b"", "390 outer stderr")
    same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "390 outer whole bytes")
    same(helper.decode(output, metadata=True),
         {"capture": PINS["capture"], "success": True, "failure": None}, "390 outer receipt")
    documents, upstream = down.authenticate(dose, parent, helper)
    same(retained["artifact_authentication"], upstream, "390 predecessor authentication splice")
    return (retained, *documents), {
        "status": "AUTHENTICATED", "reviewed_mission": "390eb4740186", "input_pins": PINS,
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


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def classify(documents, parent):
    try:
        return report(documents, parent)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(documents, parent):
    down, dose, collapse, factor, modal, raw = documents
    retained, ranked, accounts_report = down["report"], dose["report"], collapse["report"]
    for document in (down, dose, collapse):
        same((document["artifact_authentication"]["status"], document["decision"],
              document["report"]["decision"]), ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"),
             "authenticated supported prerequisite")
    same(accounts_report["modal_controls"], list(parent.CONTROLS), "modal domain binding")
    for field in ("reference_scope", "lineage_separation"):
        for document in (dose, collapse, factor, modal, raw):
            same(retained[field], document["report"][field], field + " binding")
    same(retained["parent_source_equivalence_status_preserved"], "REJECTED", "closed branch boundary")
    failures, checks = [], 0

    def verify(field, actual, expected, **identity):
        nonlocal checks
        checks += 1
        equal = type(actual) is type(expected) and actual == expected
        if not equal:
            failures.append({**identity, "field": field, "actual": str(actual), "expected": str(expected)})
        return equal

    for previous in (retained, ranked, accounts_report):
        for field, expected in (("failure_count", 0), ("failures", []), ("unknown_reasons", [])):
            verify("predecessor/" + field, previous[field], expected)
    verify("alias_map", retained["alias_map"], ALIASES)
    membership = {}
    for name in parent.CONTROLS:
        canonical = retained["alias_map"].get(name, name)
        require(type(canonical) is str, "ambiguous alias target")
        membership.setdefault(canonical, []).append(name)
    verify("quotient_membership", membership, {k: v[1] for k, v in CLASSES.items()})
    groups = [
        {"group": canonical, "controls": names, "absolute_dose": dose_value,
         "ordered_pair_signatures": [
             {"left_id": left, "right_id": right, "control": name, "absolute_dose": dose_value}
             for left, right in ((34319, 319), (319, 34319)) for name in names]}
        for canonical, (dose_value, names) in CLASSES.items() if dose_value != "0"
    ]
    nonzero_classes = list(CLASSES)[1:]
    gaps = [{"larger": a, "smaller": b,
             "gap": str(rational(CLASSES[a][0]) - rational(CLASSES[b][0])), "positive": True}
            for a, b in itertools.combinations(nonzero_classes, 2)]
    for previous in (retained, ranked):
        verify("dose_groups", previous["dose_groups"], groups)
        verify("zero_dose_controls", previous["zero_dose_controls"], CLASSES["frozen_inherited"][1])
        verify("dose_rank", previous["absolute_dose_rank_descending"], nonzero_classes)
        verify("dose_gaps", previous["pairwise_rational_gaps"], gaps)
    verify("canonical_control", ranked["canonical_control"], "frozen_inherited")
    verify("canonical_control", accounts_report["canonical_control"], "frozen_inherited")
    fields = (*parent.SCOPE_FIELDS, "control", "component")
    moving = indexed(ranked["nonzero_source_only_movements"], fields)
    zeros = indexed(ranked["zero_source_accounts"], fields)
    require(not moving.keys() & zeros.keys(), "ambiguous zero/nonzero membership")
    accounts = indexed(accounts_report["coordinate_accounts"], parent.SCOPE_FIELDS)
    same(set(accounts), set(parent.SCOPES), "missing/incompatible coordinate bindings")
    alias_rows = indexed(retained["alias_pair_matrix"], (*parent.SCOPE_FIELDS, "alias", "canonical"))
    expected_alias_keys = {(*s, a, c) for s in parent.SCOPES for a, c in ALIASES.items()}
    same(set(alias_rows), expected_alias_keys, "missing/incompatible retained alias rows")
    numeric, controls_by_scope, records = {}, {}, {}
    within, cross, coordinate_checks = [], [], []
    nonzero_count, proportional = 0, 0
    for ai, scope in enumerate(parent.SCOPES):
        identity = dict(zip(parent.SCOPE_FIELDS, scope, strict=True))
        controls = indexed(accounts[scope]["controls"], ("control",))
        same(set(controls), {(c,) for c in parent.CONTROLS}, "missing/incompatible control bindings")
        controls_by_scope[scope] = controls
        base = controls[("frozen_inherited",)]
        f = rational(base[parent.FACTOR])
        same(set(base["factors"]), set(parent.FACTORS), "missing/ambiguous factors")
        verify("factor_product", f, rational(base["factors"]["norm_weight"])
               * rational(base["factors"]["reference_inverse_norm_anchor"])
               * rational(base["factors"]["row_difference"]), **identity)
        verify("row_difference", rational(base["factors"]["row_difference"]),
               rational(base["left_weight"]) - rational(base["right_weight"]), **identity)
        for ci, name in enumerate(parent.CONTROLS):
            control = controls[(name,)]
            item = {**identity, "control": name}
            canonical = ALIASES.get(name, name)
            dose_value = rational(CLASSES[canonical][0])
            pointer = f"/report/coordinate_accounts/{ai}/controls/{ci}"
            binding = control["retained_bindings"]
            fr = parent.resolve(factor, pointer)
            same(binding, {**fr["retained_bindings"], "factor_stdout_pin": "factor_stdout",
                           "factor_control_pointer": pointer}, "factor/raw pointer splice")
            same(parent.resolve(modal, binding["modal_control_pointer"])["control"], name,
                 "modal control identity")
            rr = parent.raw_row(raw, binding["raw_selected_row_pointer"], scope, name)
            detail = parent.raw_row(raw, binding["raw_factor_row_pointer"], scope, name)
            for field in ("factors", "left_weight", "right_weight", parent.FACTOR):
                verify("retained_factor/" + field, control[field], fr[field], **item)
            for field in ("left_weight", "right_weight"):
                verify("raw_factor/" + field, control[field], detail[field], **item)
            verify("raw_common_factor", control[parent.FACTOR], rr[parent.FACTOR], **item)
            expected = {
                "factors": base["factors"], "left_weight": base["left_weight"],
                "right_weight": base["right_weight"], parent.FACTOR: str(f),
                "factor_deltas": dict.fromkeys(parent.FACTORS, "0"), "combined_factor_delta": "0",
                "components": {}, "mass": {},
            }
            same(set(control["components"]), set(parent.COMPONENTS), "missing/ambiguous components")
            for component in parent.COMPONENTS:
                sign = (-1 if component == "actual_residual_boundary" else
                        1 if component == "q24_to_fp16_conversion" else 0) if scope[4] == 62 else 0
                dh = dose_value * sign
                h0 = rational(base["components"][component]["hidden"])
                h, w0, dw = h0 + dh, h0 * f, dh * f
                term = {
                    "hidden": str(h), "canonical_hidden": str(h0), "weighted": str(h * f),
                    "canonical_weighted": str(w0), "source_delta": str(dh),
                    "weighted_movement": str(dw), "source_only_term": str(dw),
                    "source_only_residual": "0", "combined_factor_delta_term": "0",
                    "source_factor_interaction_term": "0", "product_identity_residual": "0",
                    "canonical_product_identity_residual": "0", "combined_identity_residual": "0",
                    "four_term_identity_residual": "0",
                    "individual_factor_delta_terms": dict.fromkeys(parent.FACTORS, "0"),
                }
                expected["components"][component] = term
                actual = control["components"][component]
                verify("raw_hidden", actual["hidden"], rr["hidden_components"][component], **item)
                verify("raw_weighted", actual["weighted"], rr["weighted_components"][component], **item)
                proportional += int(verify("source_only_proportionality",
                                           rational(actual["weighted_movement"]),
                                           rational(actual["source_delta"]) * rational(control[parent.FACTOR]),
                                           **item, component=component))
                key = (*scope, name, component)
                require(key in moving or key in zeros, "missing retained component classification")
                row = moving[key] if key in moving else zeros[key]
                expected_row = {**item, "component": component, "source_delta": str(dh),
                                "weighted_movement": str(dw)}
                if dh:
                    expected_row.update({
                        "group": canonical, "absolute_dose": str(abs(dh)), "common_factor": str(f),
                        "source_only_term": str(dw), "retained_bindings": binding,
                        "retained_8525_pointer": pointer + "/components/" + component,
                    })
                    if key in moving:
                        same(row["retained_bindings"], binding, "dose retained binding splice")
                        same(row["retained_8525_pointer"], expected_row["retained_8525_pointer"],
                             "dose component pointer splice")
                verify("retained_component_account", row, expected_row, **item, component=component)
                verify("zero_nonzero_classification", key in moving, bool(dh), **item, component=component)
                nonzero_count += int(bool(rational(actual["source_delta"])))
                records[key] = {k: rational(v) for k, v in actual.items()
                                if k != "individual_factor_delta_terms"}
            for branch in ("hidden", "weighted"):
                canonical_field = "canonical_" + branch
                values = [rational(t[branch]) for t in expected["components"].values()]
                baselines = [rational(t[canonical_field]) for t in expected["components"].values()]
                total, total0 = sum(values, Fraction(0)), sum(baselines, Fraction(0))
                absolute, absolute0 = sum(map(abs, values), Fraction(0)), sum(map(abs, baselines), Fraction(0))
                masses = {"signed": (total, total0), "absolute": (absolute, absolute0),
                          "cancellation_absolute_mass": (absolute - abs(total), absolute0 - abs(total0))}
                expected["mass"][branch] = {
                    k: {"value": str(v), "canonical": str(v0), "movement": str(v - v0)}
                    for k, (v, v0) in masses.items()
                }
            same(set(control), {*expected, "control", "retained_bindings"}, "missing/ambiguous control fields")
            actual_fields = dict(leaves({k: v for k, v in control.items()
                                        if k not in ("control", "retained_bindings")}))
            expected_fields = dict(leaves(expected))
            same(set(actual_fields), set(expected_fields), "missing/ambiguous numeric fields")
            equality = {field: verify("dose_determined" + field, rational(value),
                                      rational(expected_fields[field]), **item)
                        for field, value in actual_fields.items()}
            numeric[(*scope, name)] = actual_fields
            coordinate_checks.append({
                **item, "canonical_class": canonical, "coordinate62_absolute_dose": str(dose_value),
                "coordinate_absolute_dose": str(dose_value if scope[4] == 62 else Fraction(0)),
                "common_factor": str(f), "numeric_field_equality": equality,
                "all_numeric_fields_dose_determined": all(equality.values()),
                "retained_bindings": binding, "retained_8525_pointer": pointer,
                "retained_390_stdout_pin": PINS["stdout"],
            })
        for alias, canonical in ALIASES.items():
            row = alias_rows[(*scope, alias, canonical)]
            left, right = controls[(alias,)], controls[(canonical,)]
            same(row["alias_bindings"], left["retained_bindings"], "390 alias binding")
            same(row["canonical_bindings"], right["retained_bindings"], "390 canonical binding")
            same(row["retained_12ea_stdout_pin"], down["artifact_authentication"]["input_pins"]["stdout"],
                 "390 dose stdout binding")
            expected_fields = []
            right_fields = dict(leaves(right))
            for field, value in sorted(leaves(left)):
                identity_field = field == "/control" or field.startswith("/retained_bindings/")
                expected_fields.append({
                    "field": field, "alias_value": value, "canonical_value": right_fields[field],
                    "rule": "authenticated_control_specific_binding" if identity_field else "exact_equality",
                    "equal_under_alias_map": True if identity_field else value == right_fields[field],
                })
            present = indexed(row["fields"], ("field",))
            same(set(present), {(v["field"],) for v in expected_fields}, "missing 390 alias fields")
            for expected_field in expected_fields:
                field = expected_field["field"]
                if field == "/control" or field.startswith("/retained_bindings/"):
                    same(present[(field,)], expected_field, "390 distinct identity/binding")
                else:
                    verify("390_alias" + field, present[(field,)], expected_field, **identity, alias=alias)
            verify("390_alias_equality", row["equal_under_alias_map"], True, **identity, alias=alias)
            verify("390_alias_dose", row["coordinate62_absolute_dose"], CLASSES[canonical][0])
            verify("390_coordinate_dose", row["coordinate_absolute_dose"],
                   CLASSES[canonical][0] if scope[4] == 62 else "0")
        for canonical, (_, names) in CLASSES.items():
            for left, right in itertools.product(names, repeat=2):
                lf, rf = numeric[(*scope, left)], numeric[(*scope, right)]
                equality = {field: verify("within_class" + field, value, rf[field], **identity,
                                          left_control=left, right_control=right)
                            for field, value in lf.items()}
                lb, rb = controls[(left,)]["retained_bindings"], controls[(right,)]["retained_bindings"]
                require(left == right or lb != rb, "quotient erased distinct retained bindings")
                within.append({**identity, "canonical_class": canonical, "left_control": left,
                               "right_control": right, "field_equality": equality,
                               "all_numeric_fields_equal": all(equality.values()),
                               "left_bindings": lb, "right_bindings": rb})
        for left, right in itertools.combinations(CLASSES, 2):
            dose_delta = rational(CLASSES[right][0]) - rational(CLASSES[left][0])
            for component in parent.COMPONENTS:
                a, b = records[(*scope, left, component)], records[(*scope, right, component)]
                sign = (-1 if component == "actual_residual_boundary" else
                        1 if component == "q24_to_fp16_conversion" else 0) if scope[4] == 62 else 0
                dh, dw = b["source_delta"] - a["source_delta"], b["weighted_movement"] - a["weighted_movement"]
                equal = verify("cross_class_source_dose", dh, sign * dose_delta, **identity,
                               left_class=left, right_class=right, component=component)
                equal &= verify("cross_class_dose_scaled_movement", dw, sign * dose_delta * f, **identity,
                                left_class=left, right_class=right, component=component)
                cross.append({**identity, "left_class": left, "right_class": right, "component": component,
                              "left_dose": CLASSES[left][0], "right_dose": CLASSES[right][0],
                              "dose_difference": str(dose_delta), "unit_source_delta": str(sign),
                              "source_delta_difference": str(dh), "weighted_movement_difference": str(dw),
                              "common_factor": str(f), "dose_scaled_source_only_movement": str(sign * dose_delta * f),
                              "equal": bool(equal)})
    verify("component_census", set(moving) | set(zeros), set(records))
    reverse, reverse_numeric = [], []
    for scope in parent.SCOPES:
        table, left, right, branch, coordinate = scope
        if (left, right) != (34319, 319):
            continue
        back = (table, right, left, branch, coordinate)
        for name in parent.CONTROLS:
            lf, rf = numeric[(*scope, name)], numeric[(*back, name)]
            equality = {}
            for field, text in lf.items():
                target = {"/left_weight": "/right_weight", "/right_weight": "/left_weight"}.get(field, field)
                parts = field.split("/")
                preserve = (field in ("/left_weight", "/right_weight")
                            or parts[1] in ("factors", "factor_deltas") and parts[2] != "row_difference"
                            or parts[1] == "components" and parts[3] in ("hidden", "canonical_hidden", "source_delta")
                            or parts[1] == "mass" and (parts[2] == "hidden" or parts[3] != "signed"))
                equality[field] = verify("reverse_numeric" + field, rational(rf[target]),
                                         rational(text) * (1 if preserve else -1),
                                         coordinate=coordinate, control=name)
            reverse_numeric.append({"coordinate": coordinate, "control": name,
                                    "field_orientation_preserved": equality,
                                    "all_numeric_orientations_preserved": all(equality.values())})
            for component in parent.COMPONENTS:
                a, b = records[(*scope, name, component)], records[(*back, name, component)]
                hidden = ("hidden", "canonical_hidden", "source_delta")
                reverse.append({
                    "coordinate": coordinate, "control": name, "component": component,
                    "hidden_and_dose_equal": all(a[k] == b[k] for k in hidden),
                    "weighted_and_factor_opposite": all(a[k] == -b[k] for k in a if k not in hidden)
                    and rational(lf["/" + parent.FACTOR]) == -rational(rf["/" + parent.FACTOR]),
                })
    for previous in (retained, ranked):
        verify("retained_reverse_orientation", previous["reverse_pair_checks"], reverse)
    counts = {"component_accounts": len(records), "control_coordinate_accounts": len(numeric),
              "nonzero_source_only_movements": nonzero_count, "zero_source_accounts": len(records) - nonzero_count,
              "factor_proportionality_checks": proportional, "reverse_component_checks": len(reverse)}
    verify("expected_counts", counts, COUNTS)
    for previous in (retained["counts"], ranked["counts"], retained["retained_12ea_counts"]):
        verify("unchanged_counts", previous, counts)
    for field, expected in (("component_account_count", 280), ("source_only_movement_count", 20),
                            ("zero_source_count", 260), ("nonzero_weighted_movement_count", 20)):
        verify("collapse_counts/" + field, accounts_report["counts"][field], expected)
    for field, expected in (("coordinate_account_count", 5), ("control_coordinate_account_count", 40)):
        verify("collapse_counts/" + field, accounts_report[field], expected)
    verify("390_alias_coordinate_count", retained["alias_coordinate_comparisons"], 20)
    verify("390_alias_component_count", retained["alias_component_comparisons"], 140)
    verify("within_class_matrix_count", len(within), 90)
    verify("cross_class_check_count", len(cross), 210)
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "failure_count": len(failures),
        "failures": failures, "unknown_reasons": [], "exact_check_count": checks,
        "quotient_scope": "Within each retained coordinate account; not coordinate independence.",
        "alias_map": retained["alias_map"], "observed_quotient_membership": membership,
        "dose_classes": [{"canonical": c, "controls": membership.get(c, []), "absolute_dose": d}
                         for c, (d, _) in CLASSES.items()],
        "counts": counts, "retained_390_counts": retained["counts"],
        "coordinate_account_checks": coordinate_checks,
        "within_class_equality_matrix": within, "cross_class_proportionality_checks": cross,
        "reverse_pair_checks": reverse, "reverse_numeric_orientation_matrix": reverse_numeric,
        "retained_390_stdout_pin": PINS["stdout"],
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
        down, dose, parent, helper = load_helpers()
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        allowed = allowed_paths(down, dose, parent, helper)
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
            documents, authentication = authenticate(down, dose, parent, helper)
            result = classify(documents, parent)
            require(all(type(v) is int and v == 0 for v in audit.values()), "forbidden counter")
    except ERRORS as error:
        result = unknown(error, "authentication_or_runtime")
        authentication["error"] = result["unknown_reasons"][0]
    except RuntimeError as error:
        if type(error).__name__ != "ForbiddenOperation":
            raise
        result = unknown(error, "forbidden_operation")
    return {"diagnostic_id": NAME, "version": 1,
            "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_CANONICAL_DOSE_CLASS_QUOTIENT_CLASSIFIER",
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
