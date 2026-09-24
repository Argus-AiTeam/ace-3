"""Stdout-only exact affine roots and sign margins of reviewed retained coefficients."""

import argparse
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
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_root_margin_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = NAME.replace("affine_root_margin", "affine_slope_support")
CAPTURE = ROOT / "build/selected-direct-hidden-canonical-dose-affine-slope-support-2861c02a943b-attempt001"
RUN = CAPTURE / "run"
ENV_KEYS = ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
ENVIRONMENT = dict(zip(ENV_KEYS, (
    "/home/argustest", "/home/argustest/miniconda3/bin:/usr/bin:/bin", "C.UTF-8",
    str(ROOT), "1", "1",
), strict=True))
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout", ".stderr", ".whole-command.log")
LABELS = ("branch", "ignored-build", "concurrency", "compile", "pytest", "check")
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, ET.ParseError)
COUNTS = {"coefficient_records": 735, "nonzero_slopes": 36, "zero_slopes": 699,
          "class_evaluations": 2940, "retained_control_evaluations": 5880, "scopes": 5, "classes": 4}


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 2159375,
                  "e459a28a503c1464d86827ee785e4db220c7e34b91b38b23948faa49aa24c7ca"),
    "capture": pin(RUN / "capture.json", 19960,
                   "e459b7b7053b729c4e766220873313af183df73219e9147359155325646f9310"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1140,
                         "e94e88f5f894e4fdaf283aa8645b7ab285939815d8af2d5632f2ef59c9dad954"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 26479,
                  "569d0dc3cba102344f71131b63b65abcf2e47cf5885896b4b953574da2d7d958"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 27992,
                "05d67a60ea7a870fd66b3f8856fb0d5629e4bf3a2818ad7d615b860544cdd9de"),
    "review": pin(Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
                  / "2861c02a943b/round-0001.json", 689,
                  "cbd938b63d8fcf11497c9e0f3e5c1bce9a702998e6b60ff694f318557d600b19"),
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
    slope = types.ModuleType("_authenticated_2861_retained_helpers")
    slope.__file__ = PINS["source"]["path"]
    exec(compile(data, slope.__file__, "exec"), slope.__dict__)
    return slope, *slope.load_helpers()


def environment_command(environment, argv):
    return " ".join(f"{k}={shlex.quote(environment[k])}" for k in ENV_KEYS) + " " + shlex.join(argv)


def verify_capture(directory, capture, outer, stdout, sources, module, helper):
    """Authenticate complete stored commands, including the outer launch, without execution."""
    run = directory / "run"
    same((capture["success"], capture["failure"]), (True, None), "capture failure")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "capture account/runtime")
    same(preflight["environment"], ENVIRONMENT, "capture environment")
    same(preflight["sources"], sources, "capture source/test pins")
    same(capture["sources_after"], sources, "capture source/test drift")
    same(preflight["attempt"], str(directory), "capture attempt")
    same(preflight["independent_host_review"], "REQUIRED", "capture review gate")
    same(preflight["model_or_service_calls_authorized"], 0, "capture service budget")
    same(preflight["executable"]["path"], str(Path(PYTHON).resolve()), "executable identity")
    bound_bytes(preflight["executable"])
    same([r["label"] for r in capture["results"]], list(LABELS), "capture command census")
    check_inputs = None
    for result in capture["results"]:
        same((result["exit_status"], result["timed_out"]), (0, False), "command exit/timeout")
        same([p["path"] for p in result["files"]],
             [str(run / (result["label"] + s)) for s in SUFFIXES], "capture member paths")
        command, argv, env, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "command bytes")
        same(result["command"], environment_command(ENVIRONMENT, result["argv"]), "command/argv splice")
        same(helper.decode(argv), result["argv"], "argv bytes")
        same(helper.decode(env), preflight, "environment bytes")
        same(error, b"", "command stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
             + output + b"\nSTDERR\n" + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n",
             "whole-command bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "branch")
        if result["label"] == "ignored-build":
            same(output, (str(run) + "\n").encode(), "ignored build path")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "concurrency")
        if result["label"] == "pytest":
            same(result["argv"], [PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                 "--basetemp", str(run / "pytest-tmp"), "--junitxml",
                                 str(run / "pytest.xml"), sources[1]["path"]], "targeted pytest argv")
        if result["label"] == "check":
            same(output, stdout, "stdout splice")
            same(result["argv"], [PYTHON, "-B", "-m", module, "--check"], "check argv")
            check_inputs = result["files"][:3]
    same(capture["pytest_xml"]["path"], str(run / "pytest.xml"), "pytest XML path")
    suites = ET.fromstring(bound_bytes(capture["pytest_xml"])).findall("testsuite")
    require(bool(suites) and sum(int(s.attrib["tests"]) for s in suites) > 0, "absent tests")
    require(not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
            "test failure/error/skip")
    same((outer["exit_status"], outer["timed_out"]), (0, False), "outer exit/timeout")
    same([p["path"] for p in outer["files"]],
         [str(directory / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "outer paths")
    identity_bytes, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    identity = helper.decode(identity_bytes)
    same(identity["launcher"], sources[1], "outer launcher")
    same(identity["argv"], [PYTHON, "-B", sources[1]["path"], "--run", str(run)], "outer argv")
    same(identity["command"], environment_command(ENVIRONMENT, identity["argv"]), "outer command")
    for key in ("cwd", "uid", "environment"):
        same(identity[key], preflight[key], "outer " + key)
    same(error, b"", "outer stderr")
    same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "outer whole bytes")
    receipt = helper.decode(output)
    same((receipt["success"], receipt["failure"]), (True, None), "outer receipt")
    same(receipt["capture"]["path"], str(run / "capture.json"), "outer capture pointer")
    same(helper.decode(bound_bytes(receipt["capture"]), metadata=True), capture, "outer capture splice")
    return preflight, check_inputs


def authenticate(*modules):
    slope, helper = modules[0], modules[-1]
    data = {k: bound_bytes(p) for k, p in PINS.items()}
    retained, capture, outer, review = [
        helper.decode(data[k], metadata=True) for k in ("stdout", "capture", "outer_capture", "review")
    ]
    helper.terminal_review(review, "2861c02a943b")
    same((retained["diagnostic_id"], retained["version"], retained["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_SLOPE_SUPPORT_CLASSIFIER"),
         "2861 identity")
    same((retained["artifact_authentication"]["status"], retained["decision"], retained["report"]["decision"]),
         ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "2861 supported prerequisite")
    sources = [PINS["source"], PINS["test"]]
    preflight, inputs = verify_capture(CAPTURE, capture, outer, data["stdout"], sources, slope.MODULE, helper)
    same(retained["source_test_pins"], sources, "2861 source/test pins")
    same(retained["command"], slope.COMMAND, "2861 command")
    runtime = retained["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "2861 runtime splice")
    same(runtime["environment"], {k: ENVIRONMENT[k] for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "2861 runtime environment")
    same(runtime["executable_pin"], preflight["executable"], "2861 runtime executable")
    same(retained["byte_exact_capture"]["status"], "BYTE_EXACT_STDOUT_STDERR_FILES_OPEN", "2861 capture gate")
    same(retained["byte_exact_capture"]["command_inputs"], inputs, "2861 capture identity")
    same(set(retained["dispatch_and_write_audit"]), set(helper.COUNTERS), "2861 counter census")
    require(all(type(v) is int and v == 0 for v in retained["dispatch_and_write_audit"].values()),
            "2861 forbidden counter")
    documents, upstream = slope.authenticate(*modules[1:])
    same(retained["artifact_authentication"], upstream, "2861 predecessor authentication splice")
    return (retained, *documents), {
        "status": "AUTHENTICATED", "reviewed_mission": "2861c02a943b", "input_pins": PINS,
        "review": review, "complete_command_capture": capture, "outer_capture": outer,
        "retained_predecessor_raw_bindings": upstream,
    }


def sign(value):
    return (value > 0) - (value < 0)


def root_margin(baseline, slope, doses):
    values = {c: baseline + slope * d for c, d in doses.items()}
    lower, upper = min(doses.values()), max(doses.values())
    root = -baseline / slope if slope else None
    kind = "UNIQUE" if slope else "ALL_DOSES" if not baseline else "NO_ROOT"
    location = (
        "BELOW" if root < lower else "LOWER_ENDPOINT" if root == lower else
        "INTERIOR" if root < upper else "UPPER_ENDPOINT" if root == upper else "ABOVE"
    ) if root is not None else kind
    entries = {}
    for c, dose in doses.items():
        value = values[c]
        distance = abs(dose - root) if root is not None else Fraction(0) if not baseline else None
        entries[c] = {
            "dose": str(dose), "value": str(value), "sign": sign(value), "absolute_margin": str(abs(value)),
            "signed_root_distance": str(dose - root) if root is not None else None,
            "absolute_root_distance": str(distance) if distance is not None else None,
            "root_distance_identity_residual": str(abs(value) - abs(slope) * distance)
            if root is not None else None,
        }
    minimum = min(abs(v) for v in values.values())
    signs = set(map(sign, values.values()))
    root_in_interval = kind == "ALL_DOSES" or root is not None and lower <= root <= upper
    return {
        "baseline": str(baseline), "slope": str(slope), "root_kind": kind,
        "root": str(root) if root is not None else None, "root_location": location,
        "root_equation_residual": str(baseline + slope * root) if root is not None else None,
        "classes": entries, "minimum_retained_absolute_margin": str(minimum),
        "nearest_classes": [c for c, v in values.items() if abs(v) == minimum],
        "interval_minimum_absolute_margin": str(Fraction(0) if root_in_interval else minimum),
        "root_in_retained_interval": root_in_interval,
        "sign_stable": len(signs) == 1, "strict_sign_stable": len(signs) == 1 and 0 not in signs,
        "no_sign_reversal": not {-1, 1} <= signs,
    }


def nearest(records, value):
    if not records:
        return {"minimum": None, "fields": []}
    minimum = min(value(r) for r in records)
    return {"minimum": str(minimum), "fields": [
        {k: r[k] for k in ("table", "branch", "left_id", "right_id", "coordinate", "field", "root")}
        for r in records if value(r) == minimum
    ]}


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def classify(documents, modules):
    try:
        return report(documents, modules)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(documents, modules):
    slope, affine, parent = modules[0], modules[1], modules[-2]
    support, normal, quotient, down, dose, collapse, factor, modal, raw = documents
    s, r = support["report"], normal["report"]
    for document in (support, normal):
        same((document["artifact_authentication"]["status"], document["decision"], document["report"]["decision"]),
             ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "supported prerequisite")
    same(s["retained_70fb_pins"], slope.PINS, "coefficient source pins")
    same(s["retained_quotient_bindings"], r["retained_quotient_bindings"], "quotient splice")
    same(s["retained_quotient_bindings"],
         {k: quotient["report"][k] for k in slope.QUOTIENT_FIELDS}, "retained quotient bindings")
    for field in ("reference_scope", "lineage_separation"):
        for document in documents[1:]:
            same(s[field], document["report"][field], field + " binding")
    same(s["parent_source_equivalence_status_preserved"], "REJECTED", "closed branch boundary")
    same(r["parent_source_equivalence_status_preserved"], "REJECTED", "coefficient closed branch boundary")
    doses = {c: rational(d) for c, (d, _) in affine.CLASSES.items()}
    classes = [{"canonical": c, "absolute_dose": d, "controls": names}
               for c, (d, names) in affine.CLASSES.items()]
    same(s["retained_quotient_bindings"]["dose_classes"], classes, "retained dose classes")
    rows = affine.indexed(r["coefficient_matrix"], parent.SCOPE_FIELDS)
    accounts = affine.indexed(collapse["report"]["coordinate_accounts"], parent.SCOPE_FIELDS)
    reconstructions = affine.indexed(r["reconstruction_residual_matrix"], (*parent.SCOPE_FIELDS, "control"))
    same(set(rows), set(parent.SCOPES), "coefficient scope census")
    same(set(accounts), set(rows), "retained account scope census")
    same(set(reconstructions), {(*scope, c) for scope in rows for c in parent.CONTROLS},
         "reconstruction scope/control census")
    failures = []

    def verify(field, actual, expected, identity=None):
        if type(actual) is not type(expected) or actual != expected:
            failures.append({**(identity or {}), "field": field, "actual": str(actual), "expected": str(expected)})

    for predecessor in (s, r):
        for key, expected in (("failure_count", 0), ("failures", []), ("unknown_reasons", [])):
            verify("prerequisite/" + key, predecessor[key], expected)
    verify("support/counts", s["counts"], slope.COUNTS)
    verify("support/expected_counts", s["expected_counts"], slope.COUNTS)
    verify("normal/counts", r["counts"], affine.COUNTS)
    verify("support/retained_counts", s["retained_counts"], affine.COUNTS)
    for key in ("sign_closure_matrix", "mass_closure_matrix", "arithmetic_closure_matrix"):
        matrix = affine.indexed(s[key], parent.SCOPE_FIELDS)
        same(set(matrix), set(rows), key + " scope census")
        for scope, entry in matrix.items():
            require(bool(entry["residuals"]), key + " missing residuals")
            for field, residual in entry["residuals"].items():
                verify(key + field, rational(residual), Fraction(0), dict(zip(parent.SCOPE_FIELDS, scope)))
    verify("orientation/count", len(s["orientation_closure_matrix"]), 2)
    for row in s["orientation_closure_matrix"]:
        verify("orientation/field_count", len(row["fields"]), 147)
        for field, entry in row["fields"].items():
            for name, residual in entry["residuals"].items():
                verify("orientation" + field + "/" + name, rational(residual), Fraction(0))
    support_rows = affine.indexed(s["support_matrix"], parent.SCOPE_FIELDS)
    same(set(support_rows), set(rows), "support scope census")
    matrix, nonzero, zero, control_evaluations = [], [], [], 0
    for scope, row in rows.items():
        identity = dict(zip(parent.SCOPE_FIELDS, scope, strict=True))
        same(set(row), {*parent.SCOPE_FIELDS, "fields", "abscissa", "baseline_control"}, "coefficient schema")
        same(row["abscissa"], "canonical_class_coordinate62_absolute_dose", "abscissa")
        same(row["baseline_control"], "frozen_inherited", "baseline control")
        controls = affine.indexed(accounts[scope]["controls"], ("control",))
        same(set(controls), {(c,) for c in parent.CONTROLS}, "retained control census")
        actual = {c: affine.numeric_fields(controls[(c,)], parent) for c in parent.CONTROLS}
        names = set(actual["frozen_inherited"])
        same(set(row["fields"]), names, "coefficient field census")
        verify("field_count", len(names), 147, identity)
        for c in parent.CONTROLS:
            reconstruction = reconstructions[(*scope, c)]
            control = controls[(c,)]
            same(reconstruction["retained_bindings"], control["retained_bindings"], "operand/lineage binding")
            same(parent.resolve(collapse, reconstruction["retained_component_account_pointer"]), control,
                 "retained control pointer")
            binding = control["retained_bindings"]
            original = parent.resolve(factor, binding["factor_control_pointer"])
            same(binding, {**original["retained_bindings"], "factor_stdout_pin": "factor_stdout",
                           "factor_control_pointer": binding["factor_control_pointer"]}, "factor binding")
            same(parent.resolve(modal, binding["modal_control_pointer"])["control"], c, "modal binding")
            parent.raw_row(raw, binding["raw_selected_row_pointer"], scope, c)
            parent.raw_row(raw, binding["raw_factor_row_pointer"], scope, c)
            same(set(reconstruction["fields"]), names, "reconstruction field census")
        active = []
        for field, coefficient in row["fields"].items():
            same(set(coefficient), {"baseline", "slope", "candidate_slopes", "candidate_slope_residuals",
                                    "unique", "design_rank"}, "coefficient field schema")
            baseline, derivative = [rational(coefficient[k]) for k in ("baseline", "slope")]
            verify(field + "/unique", coefficient["unique"], True, identity)
            verify(field + "/rank", coefficient["design_rank"], 2, identity)
            for key in ("candidate_slopes", "candidate_slope_residuals"):
                same(set(coefficient[key]), set(doses) - {"frozen_inherited"}, "candidate slope census")
            for c in coefficient["candidate_slopes"]:
                verify(field + "/candidate/" + c, rational(coefficient["candidate_slopes"][c]), derivative, identity)
                verify(field + "/candidate_residual/" + c,
                       rational(coefficient["candidate_slope_residuals"][c]), Fraction(0), identity)
            record = {**identity, "field": field, "slope": str(derivative)}
            (nonzero if derivative else zero).append(record)
            if derivative:
                active.append(field)
            result = root_margin(baseline, derivative, doses)
            residuals = {}
            for canonical, (d, members) in affine.CLASSES.items():
                predicted = rational(result["classes"][canonical]["value"])
                for c in members:
                    reconstruction = reconstructions[(*scope, c)]
                    verify("canonical_class/" + c, reconstruction["canonical_class"], canonical, identity)
                    verify("dose/" + c, reconstruction["dose"], d, identity)
                    retained = reconstruction["fields"][field]
                    same(set(retained), {"retained", "reconstructed", "residual"}, "retained value schema")
                    observed = actual[c][field]
                    residuals[c] = str(observed - predicted)
                    verify(field + "/retained/" + c, rational(retained["retained"]), observed, identity)
                    verify(field + "/reconstruction/" + c, rational(retained["reconstructed"]), predicted, identity)
                    verify(field + "/residual/" + c, rational(retained["residual"]), Fraction(0), identity)
                    verify(field + "/value/" + c, observed, predicted, identity)
                    verify(field + "/sign/" + c, sign(observed), result["classes"][canonical]["sign"], identity)
                    control_evaluations += 1
            matrix.append({**identity, "field": field, **result, "retained_value_residuals": residuals})
        sr = support_rows[scope]
        verify("support/exact", sr["exact"], True, identity)
        verify("support/nonzero", sr["nonzero_fields"], sorted(active), identity)
        verify("support/expected_nonzero", sr["expected_nonzero_fields"], sorted(active), identity)
        verify("support/zero", sr["zero_fields"], sorted(names - set(active)), identity)
    verify("support/nonzero_list", s["nonzero_slope_fields"], nonzero)
    verify("support/zero_list", s["zero_slope_fields"], zero)
    counts = {"coefficient_records": len(matrix), "nonzero_slopes": len(nonzero), "zero_slopes": len(zero),
              "class_evaluations": sum(len(r["classes"]) for r in matrix),
              "retained_control_evaluations": control_evaluations, "scopes": len(rows), "classes": len(doses)}
    verify("root_margin_census", counts, COUNTS)
    unique = [r for r in matrix if r["root_kind"] == "UNIQUE"]
    lower, upper = min(doses.values()), max(doses.values())
    histogram = {}
    for row in matrix:
        key = row["root_location"]
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "counts": counts, "expected_counts": COUNTS,
        "root_margin_matrix": matrix, "root_location_counts": histogram,
        "sign_stability_counts": {k: sum(row[k] for row in matrix)
                                 for k in ("sign_stable", "strict_sign_stable", "no_sign_reversal")},
        "nearest_unique_roots_by_class": {
            c: nearest(unique, lambda r: abs(rational(r["root"]) - d)) for c, d in doses.items()
        },
        "nearest_off_interval_roots": {
            "below": nearest([r for r in unique if rational(r["root"]) < lower],
                             lambda r: lower - rational(r["root"])),
            "above": nearest([r for r in unique if rational(r["root"]) > upper],
                             lambda r: rational(r["root"]) - upper),
        },
        "nearest_nonzero_margins_by_class": {
            c: nearest([r for r in matrix if rational(r["classes"][c]["absolute_margin"])],
                       lambda r: rational(r["classes"][c]["absolute_margin"])) for c in doses
        },
        "dose_classes": classes, "retained_interval": [str(lower), str(upper)],
        "retained_coefficient_matrix": r["coefficient_matrix"], "retained_2861_pins": PINS,
        "failure_count": len(failures), "failures": failures, "unknown_reasons": [],
        "reference_scope": s["reference_scope"], "lineage_separation": s["lineage_separation"],
        "parent_source_equivalence_status_preserved": "REJECTED", "claim_boundary": s["claim_boundary"],
        "root_domain": "Roots outside the retained interval are algebraic locations only, not evaluated model doses.",
        "margin_definition": "Absolute affine value; root distance is in canonical-dose units, not an admission threshold.",
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
        slope, parent, helper = modules[0], modules[-2], modules[-1]
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        paths = slope.capture_paths()
        allowed = slope.allowed_paths(*modules[1:]) | {
            str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()), str(RUN / "pytest.xml"),
            *(str(RUN / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
            *(str(CAPTURE / ("launcher." + suffix))
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
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_ROOT_MARGIN_CLASSIFIER",
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
