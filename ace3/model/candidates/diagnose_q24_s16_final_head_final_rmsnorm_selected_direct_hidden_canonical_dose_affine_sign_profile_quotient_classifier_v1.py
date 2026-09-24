"""Stdout-only sign-profile quotient of authenticated retained affine root evidence."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import types
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_sign_profile_quotient_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = NAME.replace("affine_sign_profile_quotient", "affine_root_margin")
CAPTURE = ROOT / "build/selected-direct-hidden-canonical-dose-affine-root-margin-e8328d151459-attempt001"
RUN = CAPTURE / "run"
SCOPE = "e8328d151459 retained-only affine sign-profile quotient"
ENV_KEYS = ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
ENVIRONMENT = dict(zip(ENV_KEYS, (
    "/home/argustest", "/home/argustest/miniconda3/bin:/usr/bin:/bin", "C.UTF-8",
    str(ROOT), "1", "1",
), strict=True))
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout", ".stderr", ".whole-command.log")
LABELS = ("branch", "ignored-build", "concurrency", "compile", "pytest", "check")
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, ET.ParseError)
COUNTS = {"coefficient_records": 735, "class_evaluations": 2940, "classes": 4, "scopes": 5,
          "paired_fields": 588, "orientation_pairs": 294, "unpaired_fields": 147}
ROOT_COUNTS = {"LOWER_ENDPOINT": 20, "ABOVE": 16, "ALL_DOSES": 485, "NO_ROOT": 214}
STABILITY_COUNTS = {"sign_stable": 715, "strict_sign_stable": 230, "no_sign_reversal": 735}
PROFILE_FIELDS = ("sign_vector", "root_kind", "root", "root_location", "sign_stable",
                  "strict_sign_stable", "no_sign_reversal", "nearest_classes", "orientation_rule")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 1530061,
                  "a69ad0052fce076a3ab17bbc5704739d1a23fc2643f1b89a1b6d9b58c6bd5bc1"),
    "capture": pin(RUN / "capture.json", 19842,
                   "bc1a8c1900e7946fd1a72487b621722b1822be01f286b7ad5c984a3dfa18cd59"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1132,
                         "4cbf7fb6ba8309705d46ee1305d91b732f706cddf7349d94fef667fa3270b6a9"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 29857,
                  "4637b65024ce17f1cc053b408aa7ce97452ac0bab59b74b7628f161d5d921a62"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 24041,
                "27fc0f7885b640b5412f3ed5af2c1f1610f88ec38dd4e01ec6373d82d0bfb1d7"),
    "review": pin(Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
                  / "e8328d151459/round-0001.json", 690,
                  "347975c81ec41b4279bbd44a260f16b9d4a69df69d8977dc96652950ae70da26"),
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
    data = bound_bytes(PINS["source"])
    root = types.ModuleType("_authenticated_e832_retained_helpers")
    root.__file__ = PINS["source"]["path"]
    exec(compile(data, root.__file__, "exec"), root.__dict__)
    return root, *root.load_helpers()


def environment_command(environment, argv):
    return " ".join(f"{k}={shlex.quote(environment[k])}" for k in ENV_KEYS) + " " + shlex.join(argv)


def verify_capture(directory, capture, outer, stdout, sources, module, helper):
    return load_helpers()[0].verify_capture(directory, capture, outer, stdout, sources, module, helper)


def authenticate(*modules):
    root, helper = modules[0], modules[-1]
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    retained, capture, outer, review = [
        helper.decode(data[key], metadata=True) for key in ("stdout", "capture", "outer_capture", "review")
    ]
    helper.terminal_review(review, "e8328d151459")
    same((retained["diagnostic_id"], retained["version"], retained["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_ROOT_MARGIN_CLASSIFIER"),
         "e832 identity")
    same((retained["artifact_authentication"]["status"], retained["decision"], retained["report"]["decision"]),
         ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "e832 supported prerequisite")
    sources = [PINS["source"], PINS["test"]]
    preflight, inputs = root.verify_capture(CAPTURE, capture, outer, data["stdout"], sources, root.MODULE, helper)
    same(retained["source_test_pins"], sources, "e832 source/test pins")
    same(retained["command"], root.COMMAND, "e832 command")
    runtime = retained["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "e832 runtime splice")
    same(runtime["environment"], {k: ENVIRONMENT[k] for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "e832 runtime environment")
    same(runtime["executable_pin"], preflight["executable"], "e832 runtime executable")
    same(retained["byte_exact_capture"]["status"], "BYTE_EXACT_STDOUT_STDERR_FILES_OPEN", "e832 capture gate")
    same(retained["byte_exact_capture"]["command_inputs"], inputs, "e832 capture identity")
    same(set(retained["dispatch_and_write_audit"]), set(helper.COUNTERS), "e832 counter census")
    require(all(type(v) is int and v == 0 for v in retained["dispatch_and_write_audit"].values()),
            "e832 forbidden counter")
    documents, upstream = root.authenticate(*modules[1:])
    same(retained["artifact_authentication"], upstream, "e832 predecessor authentication splice")
    same(retained["claim_boundary"], modules[-2].BOUNDARY, "e832 non-admission boundary")
    return (retained, *documents), {
        "status": "AUTHENTICATED", "reviewed_mission": "e8328d151459", "input_pins": PINS,
        "review": review, "complete_command_capture": capture, "outer_capture": outer,
        "retained_predecessor_raw_bindings": upstream,
    }


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def classify(documents, modules):
    try:
        return report(documents, modules)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(documents, modules):
    root, slope, affine, parent = modules[0], modules[1], modules[2], modules[-2]
    retained, support, normal, quotient, down, dose, collapse, factor, modal, raw = documents
    r, s, n = retained["report"], support["report"], normal["report"]
    for document in (retained, support, normal):
        same((document["artifact_authentication"]["status"], document["decision"], document["report"]["decision"]),
             ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "supported prerequisite")
    same(r["retained_2861_pins"], root.PINS, "root predecessor pins")
    same(s["retained_70fb_pins"], slope.PINS, "support predecessor pins")
    same(r["retained_coefficient_matrix"], n["coefficient_matrix"], "retained coefficient splice")
    same(s["retained_quotient_bindings"], n["retained_quotient_bindings"], "quotient splice")
    same(n["retained_quotient_bindings"], {k: quotient["report"][k] for k in slope.QUOTIENT_FIELDS},
         "retained quotient pointers")
    for key in ("reference_scope", "lineage_separation"):
        for document in documents[1:]:
            same(r[key], document["report"][key], key + " binding")
    for document in (retained, support, normal):
        same(document["report"]["parent_source_equivalence_status_preserved"], "REJECTED", "closed branch")
        same(document["report"]["claim_boundary"], parent.BOUNDARY, "non-admission boundary")
    declared = [{"canonical": c, "absolute_dose": d, "controls": names}
                for c, (d, names) in affine.CLASSES.items()]
    same(r["dose_classes"], declared, "dose class bindings")
    same(n["retained_quotient_bindings"]["dose_classes"], declared, "predecessor dose bindings")
    classes = sorted(declared, key=lambda c: rational(c["absolute_dose"]))
    order = [c["canonical"] for c in classes]
    doses = {c["canonical"]: rational(c["absolute_dose"]) for c in classes}
    lower, upper = min(doses.values()), max(doses.values())
    same(r["retained_interval"], [str(lower), str(upper)], "retained interval")
    scope_fields = parent.SCOPE_FIELDS
    keys = (*scope_fields, "field")
    coefficients = affine.indexed(n["coefficient_matrix"], scope_fields)
    matrix = affine.indexed(r["root_margin_matrix"], keys)
    same(set(coefficients), set(parent.SCOPES), "coefficient scope census")
    names = {
        "/left_weight", "/right_weight", "/" + parent.FACTOR, "/combined_factor_delta",
        *(f"/{group}/{f}" for group in ("factors", "factor_deltas") for f in parent.FACTORS),
        *(f"/components/{c}/{f}" for c in parent.COMPONENTS for f in affine.TERM_FIELDS),
        *(f"/components/{c}/individual_factor_delta_terms/{f}"
          for c in parent.COMPONENTS for f in parent.FACTORS),
        *(f"/mass/{b}/{m}/{v}" for b in ("hidden", "weighted")
          for m in parent.MASSES for v in ("value", "canonical", "movement")),
    }
    for row in coefficients.values():
        same(set(row["fields"]), names, "coefficient field census")
        same(row["abscissa"], "canonical_class_coordinate62_absolute_dose", "abscissa")
        same(row["baseline_control"], "frozen_inherited", "baseline control")
    same(set(matrix), {(*scope, f) for scope in parent.SCOPES for f in names}, "root field census")
    reconstructions = affine.indexed(n["reconstruction_residual_matrix"], (*scope_fields, "control"))
    same(set(reconstructions), {(*scope, c) for scope in parent.SCOPES for c in parent.CONTROLS},
         "retained control census")
    for key, row in reconstructions.items():
        scope, control = key[:-1], key[-1]
        original = parent.resolve(collapse, row["retained_component_account_pointer"])
        binding = row["retained_bindings"]
        same(binding, original["retained_bindings"], "operand/lineage binding")
        same(tuple(original[k] for k in ("control",)), (control,), "control pointer")
        same(set(row["fields"]), names, "retained value field census")
        original_factor = parent.resolve(factor, binding["factor_control_pointer"])
        same(binding, {**original_factor["retained_bindings"], "factor_stdout_pin": "factor_stdout",
                       "factor_control_pointer": binding["factor_control_pointer"]}, "factor pointer")
        same(parent.resolve(modal, binding["modal_control_pointer"])["control"], control, "modal pointer")
        parent.raw_row(raw, binding["raw_selected_row_pointer"], scope, control)
        parent.raw_row(raw, binding["raw_factor_row_pointer"], scope, control)
    failures = []

    def verify(field, actual, expected, identity=None):
        if type(actual) is not type(expected) or actual != expected:
            failures.append({**(identity or {}), "field": field, "actual": str(actual), "expected": str(expected)})

    for previous in (r, s, n):
        for field, expected in (("failure_count", 0), ("failures", []), ("unknown_reasons", [])):
            verify("prerequisite/" + field, previous[field], expected)
    verify("retained_counts", r["counts"], root.COUNTS)
    verify("retained_expected_counts", r["expected_counts"], root.COUNTS)
    oriented = affine.indexed(s["orientation_closure_matrix"], ("coordinate",))
    same(set(oriented), {(62,), (241,)}, "orientation scope census")
    for entry in oriented.values():
        verify("orientation/forward_pair", entry["forward_pair"], [34319, 319])
        verify("orientation/reverse_pair", entry["reverse_pair"], [319, 34319])
        same(set(entry["fields"]), names, "orientation field census")
    records, indexed, groups, profile_ids = [], {}, [], {}
    for index, row in enumerate(r["root_margin_matrix"]):
        identity = {k: row[k] for k in keys}
        key = tuple(row[k] for k in keys)
        scope, field = key[:-1], row["field"]
        coefficient = coefficients[scope]["fields"][field]
        baseline, derivative = rational(row["baseline"]), rational(row["slope"])
        verify("baseline", row["baseline"], coefficient["baseline"], identity)
        verify("slope", row["slope"], coefficient["slope"], identity)
        same(set(row["classes"]), set(order), "class value census")
        values, signs, margins = {}, {}, {}
        unique = row["root"] is not None
        location = rational(row["root"]) if unique else None
        kind = "UNIQUE" if derivative else "NO_ROOT" if baseline else "ALL_DOSES"
        verify("root_kind", row["root_kind"], kind, identity)
        verify("unique_root_presence", unique, bool(derivative), identity)
        if unique:
            verify("root_equation", baseline + derivative * location, Fraction(0), identity)
            verify("root_equation_residual", row["root_equation_residual"], "0", identity)
            where = ("BELOW" if location < lower else "LOWER_ENDPOINT" if location == lower else
                     "INTERIOR" if location < upper else "UPPER_ENDPOINT" if location == upper else "ABOVE")
        else:
            where = kind
            verify("root_equation_residual", row["root_equation_residual"], None, identity)
        verify("root_location", row["root_location"], where, identity)
        inside = kind == "ALL_DOSES" or unique and lower <= location <= upper
        verify("root_in_interval", row["root_in_retained_interval"], inside, identity)
        same(set(row["retained_value_residuals"]), set(parent.CONTROLS), "control residual census")
        for canonical in order:
            entry, dose_value = row["classes"][canonical], doses[canonical]
            value = rational(entry["value"])
            values[canonical], signs[canonical], margins[canonical] = value, (value > 0) - (value < 0), abs(value)
            verify(canonical + "/dose", rational(entry["dose"]), dose_value, identity)
            verify(canonical + "/sign", entry["sign"], signs[canonical], identity)
            verify(canonical + "/absolute_margin", rational(entry["absolute_margin"]), abs(value), identity)
            verify(canonical + "/affine_equation", value - baseline - derivative * dose_value, Fraction(0), identity)
            distance = dose_value - location if unique else None
            verify(canonical + "/signed_distance", entry["signed_root_distance"],
                   str(distance) if unique else None, identity)
            verify(canonical + "/absolute_distance", entry["absolute_root_distance"],
                   str(abs(distance)) if unique else "0" if kind == "ALL_DOSES" else None, identity)
            verify(canonical + "/distance_residual", entry["root_distance_identity_residual"],
                   "0" if unique else None, identity)
            if unique:
                verify(canonical + "/distance_equation", abs(value) - abs(derivative * distance), Fraction(0), identity)
            for control in affine.CLASSES[canonical][1]:
                reconstruction = reconstructions[(*scope, control)]
                same(reconstruction["canonical_class"], canonical, "retained class pointer")
                same(reconstruction["dose"], str(dose_value), "retained dose pointer")
                saved = reconstruction["fields"][field]
                for name in ("retained", "reconstructed"):
                    verify(control + "/" + name, rational(saved[name]), value, identity)
                verify(control + "/residual", saved["residual"], "0", identity)
                verify(control + "/root_residual", row["retained_value_residuals"][control], "0", identity)
        minimum = min(margins.values())
        nearest = [c for c in order if margins[c] == minimum]
        verify("nearest_classes", row["nearest_classes"],
               [c for c in affine.CLASSES if c in nearest], identity)
        verify("minimum_margin", rational(row["minimum_retained_absolute_margin"]), minimum, identity)
        verify("interval_margin", rational(row["interval_minimum_absolute_margin"]),
               Fraction(0) if inside else minimum, identity)
        sign_set = set(signs.values())
        stability = {"sign_stable": len(sign_set) == 1,
                     "strict_sign_stable": len(sign_set) == 1 and 0 not in sign_set,
                     "no_sign_reversal": not {-1, 1} <= sign_set}
        for name, expected in stability.items():
            verify(name, row[name], expected, identity)
        paired = row["table"] == "coordinate"
        target, multiplier = affine.reversal(field)
        rule = ("SWAP_WEIGHTS" if target != field else "PRESERVE" if multiplier == 1 else "NEGATE"
                ) if paired else "UNPAIRED_RETAINED_SCOPE"
        profile = {"sign_vector": [signs[c] for c in order], "root_kind": kind, "root": row["root"],
                   "root_location": where, **stability, "nearest_classes": nearest, "orientation_rule": rule}
        signature = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        if signature not in profile_ids:
            profile_ids[signature] = len(groups)
            groups.append({"profile": profile, "members": []})
        group = profile_ids[signature]
        groups[group]["members"].append(index)
        record = {
            **identity, **profile, "group": group,
            "retained_root_pointer": f"/report/root_margin_matrix/{index}",
            "class_values": {c: str(values[c]) for c in order},
            "reverse_field": target if paired else None, "orientation_multiplier": multiplier if paired else None,
        }
        records.append(record)
        indexed[key] = index
    pairs = []
    for key, index in indexed.items():
        row, record = matrix[key], records[index]
        if row["table"] != "coordinate":
            continue
        identity = {k: row[k] for k in keys}
        back = {**identity, "left_id": row["right_id"], "right_id": row["left_id"],
                "field": record["reverse_field"]}
        back_key = tuple(back[k] for k in keys)
        reverse_index = indexed[back_key]
        reverse, reverse_record = matrix[back_key], records[reverse_index]
        multiplier = record["orientation_multiplier"]
        verify("orientation/involution", reverse_record["reverse_field"], row["field"], identity)
        verify("orientation/multiplier", reverse_record["orientation_multiplier"], multiplier, identity)
        for field in ("baseline", "slope"):
            verify("orientation/" + field, rational(reverse[field]), multiplier * rational(row[field]), identity)
        for c in order:
            verify("orientation/value/" + c, rational(reverse["classes"][c]["value"]),
                   multiplier * rational(row["classes"][c]["value"]), identity)
        verify("orientation/sign_vector", reverse_record["sign_vector"],
               [multiplier * sign for sign in record["sign_vector"]], identity)
        for field in PROFILE_FIELDS[1:-1]:
            verify("orientation/" + field, reverse_record[field], record[field], identity)
        if row["left_id"] == 34319:
            saved = oriented[(row["coordinate"],)]["fields"][row["field"]]
            verify("retained_orientation/target", saved["reverse_field"], record["reverse_field"], identity)
            verify("retained_orientation/multiplier", saved["multiplier"], multiplier, identity)
            verify("retained_orientation/residuals", saved["residuals"], {"baseline": "0", "slope": "0"}, identity)
            pairs.append({"forward": index, "reverse": reverse_index, "multiplier": multiplier,
                          "forward_group": record["group"], "reverse_group": reverse_record["group"]})
    unpaired = [i for i, r in enumerate(records) if r["orientation_rule"] == "UNPAIRED_RETAINED_SCOPE"]
    counts = {"coefficient_records": len(records), "class_evaluations": sum(len(r["sign_vector"]) for r in records),
              "classes": len(order), "scopes": len(coefficients), "paired_fields": 2 * len(pairs),
              "orientation_pairs": len(pairs), "unpaired_fields": len(unpaired)}
    histogram = dict(Counter(r["root_location"] for r in records))
    stable = {k: sum(r[k] for r in records) for k in STABILITY_COUNTS}
    verify("quotient_counts", counts, COUNTS)
    verify("root_accounting", histogram, ROOT_COUNTS)
    verify("retained_root_accounting", r["root_location_counts"], histogram)
    verify("stability_counts", stable, STABILITY_COUNTS)
    verify("retained_stability_counts", r["sign_stability_counts"], stable)
    verify("partition_coverage", sorted(i for g in groups for i in g["members"]), list(range(735)))
    verify("orientation_coverage", sorted([i for p in pairs for i in (p["forward"], p["reverse"])] + unpaired),
           list(range(735)))
    for group in groups:
        group["count"] = len(group["members"])
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "counts": counts, "expected_counts": COUNTS,
        "class_order": order, "dose_classes": classes, "retained_interval": r["retained_interval"],
        "quotient": groups, "profile_count": len(groups), "field_membership_matrix": records,
        "orientation_pairs": pairs, "unpaired_fields": unpaired,
        "root_location_counts": histogram, "sign_stability_counts": stable,
        "lower_endpoint_departures": [i for i, row in enumerate(records) if row["root_location"] == "LOWER_ENDPOINT"],
        "above_interval_roots": [i for i, row in enumerate(records) if row["root_location"] == "ABOVE"],
        "identically_zero_fields": [i for i, row in enumerate(records) if row["root_kind"] == "ALL_DOSES"],
        "constant_nonzero_fields": [i for i, row in enumerate(records) if row["root_kind"] == "NO_ROOT"],
        "retained_e832_pins": PINS, "retained_coefficient_pointer": "/report/retained_coefficient_matrix",
        "retained_predecessor_bindings": n["retained_quotient_bindings"],
        "failure_count": len(failures), "failures": failures, "unknown_reasons": [],
        "reference_scope": r["reference_scope"], "lineage_separation": r["lineage_separation"],
        "parent_source_equivalence_status_preserved": "REJECTED", "claim_boundary": r["claim_boundary"],
        "root_domain": r["root_domain"], "margin_definition": r["margin_definition"],
        "quotient_boundary": "Exact retained sign profiles, not numeric equality or model-dose extrapolation. "
                             "Unpaired retained scopes do not imply a missing reverse producer is available.",
    }


def check():
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    audit, sources, capture = {"forbidden_calls": 0}, [], {"status": "UNKNOWN"}
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    boundary = "Retained-only CPU non-admission accounting; independent Host Reviewer required."
    try:
        modules = load_helpers()
        root, slope, parent, helper = modules[0], modules[1], modules[-2], modules[-1]
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        paths = slope.capture_paths()
        allowed = slope.allowed_paths(*modules[2:]) | {
            str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()), str(RUN / "pytest.xml"),
            *(str(RUN / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
            *(str(CAPTURE / ("launcher." + suffix))
              for suffix in ("identity.json", "stdout", "stderr", "whole-command.log")),
            *(p["path"] for p in root.PINS.values()), str(root.RUN / "pytest.xml"),
            *(str(root.RUN / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
            *(str(root.CAPTURE / ("launcher." + suffix))
              for suffix in ("identity.json", "stdout", "stderr", "whole-command.log")),
            *(str(p) for p in paths),
        }
        helper.allowed_paths = lambda: allowed
        with helper.read_only(audit):
            same((runtime["cwd"], runtime["uid"], runtime["python"]), (str(ROOT), 1000, PYTHON),
                 "current account/interpreter")
            same(runtime["environment"], {k: ENVIRONMENT[k] for k in runtime["environment"]},
                 "command-local environment")
            require(sys.dont_write_bytecode, "bytecode writing must be disabled")
            for path in (SOURCE, TEST, Path(sys.executable).resolve()):
                data = path.read_bytes()
                binding = pin(path, len(data), hashlib.sha256(data).hexdigest())
                if path in (SOURCE, TEST):
                    sources.append(binding)
                else:
                    runtime["executable_pin"] = binding
            command, argv, environment = [p.read_bytes() for p in paths]
            preflight = helper.decode(environment, metadata=True)
            same(helper.decode(argv), [PYTHON, "-B", "-m", MODULE, "--check"], "capture argv")
            same(command, (environment_command(ENVIRONMENT, helper.decode(argv)) + "\n").encode(),
                 "capture command bytes")
            same(preflight["sources"], sources, "capture source/test binding")
            same((preflight["cwd"], preflight["uid"], preflight["python"], preflight["python_version"]),
                 (str(ROOT), 1000, PYTHON, sys.version), "capture runtime binding")
            same(preflight["environment"], ENVIRONMENT, "capture environment binding")
            same(preflight["executable"], runtime["executable_pin"], "capture executable binding")
            same(preflight["independent_host_review"], "REQUIRED", "capture review gate")
            same(preflight["model_or_service_calls_authorized"], 0, "capture service budget")
            same(preflight["scope"], SCOPE, "current scope")
            same(preflight["attempt"], str(paths[0].parent.parent), "current attempt")
            capture = {"status": "BYTE_EXACT_STDOUT_STDERR_FILES_OPEN",
                       "command_inputs": [pin(p, len(b), hashlib.sha256(b).hexdigest())
                                          for p, b in zip(paths, (command, argv, environment), strict=True)],
                       "terminal_whole_capture": "Outer launcher must seal and validate before review."}
            documents, authentication = authenticate(*modules)
            result = classify(documents, modules)
            require(all(type(v) is int and v == 0 for v in audit.values()), "forbidden counter")
    except ERRORS as error:
        result = unknown(error, "authentication_runtime_or_capture")
        authentication["error"] = result["unknown_reasons"][0]
    except RuntimeError as error:
        if type(error).__name__ != "ForbiddenOperation":
            raise
        result = unknown(error, "forbidden_operation")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_SIGN_PROFILE_QUOTIENT_CLASSIFIER",
        "decision": result["decision"], "command": COMMAND, "artifact_authentication": authentication,
        "source_test_pins": sources, "runtime": runtime, "report": result,
        "dispatch_and_write_audit": audit, "byte_exact_capture": capture, "claim_boundary": boundary,
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
