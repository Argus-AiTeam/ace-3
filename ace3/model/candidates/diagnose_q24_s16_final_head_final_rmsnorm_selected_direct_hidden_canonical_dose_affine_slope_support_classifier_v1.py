"""Stdout-only exact sparse derivative support of reviewed retained affine coefficients."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import types
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_slope_support_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT = NAME.replace("affine_slope_support", "affine_normal_form")
CAPTURE = ROOT / "build/selected-direct-hidden-canonical-dose-affine-normal-form-70fbef52693d-attempt001"
RUN = CAPTURE / "run"
ENV_KEYS = ("HOME", "PATH", "LC_ALL", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout", ".stderr", ".whole-command.log")
LABELS = ("branch", "ignored-build", "concurrency", "compile", "pytest", "check")
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, ET.ParseError)
COUNTS = {"coefficient_records": 735, "nonzero_slopes": 36, "zero_slopes": 699}
MOVING = {"actual_residual_boundary": -1, "q24_to_fp16_conversion": 1}
SOURCE_FIELDS = ("hidden", "source_delta", "source_only_term", "weighted", "weighted_movement")
QUOTIENT_FIELDS = (
    "alias_map", "coordinate_account_checks", "counts", "cross_class_proportionality_checks",
    "dose_classes", "observed_quotient_membership", "retained_390_counts", "retained_390_stdout_pin",
    "reverse_numeric_orientation_matrix", "reverse_pair_checks", "within_class_equality_matrix",
)


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(RUN / "check.stdout", 3410516,
                  "b786aa598b0c9308bd15108985361e1184caba40d2d7551d74d201bcbfa25b6f"),
    "capture": pin(RUN / "capture.json", 19838,
                   "ced7bf4055aff3172194d25278395f4461724fadb38dea01889df25e521f49a8"),
    "outer_capture": pin(CAPTURE / "launcher.capture.json", 1132,
                         "68e22cc832a6d006136aa508a17e0f0c4bd3e57b775c4b54f1821457f4c28bcf"),
    "source": pin(ROOT / "ace3/model/candidates" / (PARENT + ".py"), 34047,
                  "600f1c62c24e5eafed2a99d0d15bba5495ee74a4209bd1991675384310de8205"),
    "test": pin(ROOT / "tests" / ("test_" + PARENT + ".py"), 27666,
                "06670aa3cca8d65e2ebab3494cc01e2ff93d99fdcdeafcb57315c5716fa60899"),
    "review": pin(Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
                  / "70fbef52693d/round-0001.json", 689,
                  "9e16486ba0a0a7a77409501c97c889dc72bd9dd6775ab6a92c238d235d353e72"),
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
    # Authenticate definitions before loading; no ancestor semantic entrypoint is called.
    data = bound_bytes(PINS["source"])
    affine = types.ModuleType("_authenticated_70fb_retained_helpers")
    affine.__file__ = PINS["source"]["path"]
    exec(compile(data, affine.__file__, "exec"), affine.__dict__)
    return affine, *affine.load_helpers()


def allowed_paths(affine, *modules):
    return affine.allowed_paths(*modules) | {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()), str(RUN / "pytest.xml"),
        *(str(RUN / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
        *(str(CAPTURE / ("launcher." + suffix))
          for suffix in ("identity.json", "stdout", "stderr", "whole-command.log")),
    }


def environment_command(environment, argv):
    return " ".join(f"{k}={shlex.quote(environment[k])}" for k in ENV_KEYS) + " " + shlex.join(argv)


def authenticate(affine, *modules):
    helper = modules[-1]
    data = {key: bound_bytes(binding) for key, binding in PINS.items()}
    retained, capture, outer, review = [
        helper.decode(data[key], metadata=True)
        for key in ("stdout", "capture", "outer_capture", "review")
    ]
    helper.terminal_review(review, "70fbef52693d")
    same((retained["diagnostic_id"], retained["version"], retained["status"]),
         (PARENT, 1, "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_NORMAL_FORM_CLASSIFIER"),
         "70fb identity")
    same((retained["artifact_authentication"]["status"], retained["decision"], retained["report"]["decision"]),
         ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "70fb authenticated supported prerequisite")
    same((capture["success"], capture["failure"]), (True, None), "70fb capture failure")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "70fb account/runtime")
    for sources in (preflight["sources"], capture["sources_after"], retained["source_test_pins"]):
        same(sources, [PINS["source"], PINS["test"]], "70fb source/test pins")
    same(preflight["independent_host_review"], "REQUIRED", "70fb review gate")
    same(preflight["model_or_service_calls_authorized"], 0, "70fb service budget")
    same(retained["command"], affine.COMMAND, "70fb disclosed command")
    runtime = retained["runtime"]
    same((runtime["cwd"], runtime["uid"], runtime["python"], runtime["version"]),
         (str(ROOT), 1000, PYTHON, preflight["python_version"]), "70fb runtime splice")
    same(runtime["environment"], {k: preflight["environment"][k]
                                 for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")},
         "70fb environment splice")
    same(runtime["environment"], {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
         "70fb command-local environment")
    same(runtime["executable_pin"], preflight["executable"], "70fb executable pin")
    same(runtime["executable_pin"]["path"], str(Path(PYTHON).resolve()), "70fb executable identity")
    bound_bytes(runtime["executable_pin"])
    same(set(retained["dispatch_and_write_audit"]), set(helper.COUNTERS), "70fb counter census")
    require(all(type(v) is int and v == 0 for v in retained["dispatch_and_write_audit"].values()),
            "70fb forbidden counter")
    same([r["label"] for r in capture["results"]], list(LABELS), "70fb command census")
    for result in capture["results"]:
        same(result["exit_status"], 0, "70fb command failure")
        same(result["timed_out"], False, "70fb timeout")
        same([p["path"] for p in result["files"]],
             [str(RUN / (result["label"] + s)) for s in SUFFIXES], "70fb member paths")
        command, argv, env, output, error, whole = [bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "70fb command bytes")
        same(result["command"], environment_command(preflight["environment"], result["argv"]),
             "70fb command/argv splice")
        same(helper.decode(argv), result["argv"], "70fb argv bytes")
        same(helper.decode(env), preflight, "70fb environment bytes")
        same(error, b"", "70fb stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
             + output + b"\nSTDERR\n" + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n",
             "70fb whole-command bytes")
        if result["label"] == "branch":
            same(output, b"argus/full-projection\n", "70fb branch")
        if result["label"] == "concurrency":
            same(helper.decode(output), [], "70fb concurrency")
        if result["label"] == "check":
            same(output, data["stdout"], "70fb stdout splice")
            same(result["argv"], [PYTHON, "-B", "-m", affine.MODULE, "--check"], "70fb check argv")
    same(capture["pytest_xml"]["path"], str(RUN / "pytest.xml"), "70fb test result path")
    suites = ET.fromstring(bound_bytes(capture["pytest_xml"])).findall("testsuite")
    require(bool(suites) and sum(int(s.attrib["tests"]) for s in suites) > 0, "70fb absent tests")
    require(not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
            "70fb test failure/error/skip")
    same(outer["exit_status"], 0, "70fb outer exit")
    same(outer["timed_out"], False, "70fb outer timeout")
    same([p["path"] for p in outer["files"]],
         [str(CAPTURE / ("launcher." + s)) for s in
          ("identity.json", "stdout", "stderr", "whole-command.log")], "70fb outer paths")
    identity_bytes, output, error, whole = [bound_bytes(p) for p in outer["files"]]
    identity = helper.decode(identity_bytes)
    same(identity["launcher"], PINS["test"], "70fb outer launcher")
    same(identity["argv"], [PYTHON, "-B", PINS["test"]["path"], "--run", str(RUN)], "70fb outer argv")
    same(identity["command"], environment_command(preflight["environment"], identity["argv"]),
         "70fb outer command")
    same({k: identity[k] for k in ("cwd", "uid", "environment")},
         {k: preflight[k] for k in ("cwd", "uid", "environment")}, "70fb outer environment")
    same(error, b"", "70fb outer stderr")
    same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
         + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "70fb outer whole bytes")
    same(helper.decode(output, metadata=True),
         {"capture": PINS["capture"], "success": True, "failure": None}, "70fb outer receipt")
    documents, upstream = affine.authenticate(*modules)
    same(retained["artifact_authentication"], upstream, "70fb predecessor authentication splice")
    return (retained, *documents), {
        "status": "AUTHENTICATED", "reviewed_mission": "70fbef52693d", "input_pins": PINS,
        "review": review, "complete_command_capture": capture, "outer_capture": outer,
        "retained_predecessor_raw_bindings": upstream,
    }


def unknown(error, phase):
    return {"decision": "UNKNOWN", "unknown_reasons": [f"{phase}: {type(error).__name__}: {error}"]}


def classify(documents, affine, parent):
    try:
        return report(documents, affine, parent)
    except ERRORS as error:
        return unknown(error, "retained_fields_or_bindings")


def report(documents, affine, parent):
    retained, quotient = documents[:2]
    r = retained["report"]
    same((retained["artifact_authentication"]["status"], retained["decision"], r["decision"]),
         ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "70fb supported prerequisite")
    same(r["retained_b4_stdout_pin"], affine.PINS["stdout"], "retained quotient pointer")
    same(set(r["retained_quotient_bindings"]), set(QUOTIENT_FIELDS), "missing quotient bindings")
    same(r["retained_quotient_bindings"],
         {k: quotient["report"][k] for k in QUOTIENT_FIELDS},
         "retained quotient bindings")
    for field in ("reference_scope", "lineage_separation"):
        for document in documents[1:]:
            same(r[field], document["report"][field], field + " binding")
    same(r["parent_source_equivalence_status_preserved"], "REJECTED", "closed branch boundary")
    names = {
        "/left_weight", "/right_weight", "/" + parent.FACTOR, "/combined_factor_delta",
        *(f"/{group}/{f}" for group in ("factors", "factor_deltas") for f in parent.FACTORS),
        *(f"/components/{c}/{f}" for c in parent.COMPONENTS for f in affine.TERM_FIELDS),
        *(f"/components/{c}/individual_factor_delta_terms/{f}"
          for c in parent.COMPONENTS for f in parent.FACTORS),
        *(f"/mass/{b}/{m}/{v}" for b in ("hidden", "weighted")
          for m in ("signed", "absolute", "cancellation_absolute_mass")
          for v in ("value", "canonical", "movement")),
    }
    rows = {}
    for row in r["coefficient_matrix"]:
        scope = tuple(row[k] for k in parent.SCOPE_FIELDS)
        require(scope not in rows, "ambiguous duplicate coefficient scope")
        same(set(row), {*parent.SCOPE_FIELDS, "fields", "abscissa", "baseline_control"}, "scope schema")
        same(row["abscissa"], "canonical_class_coordinate62_absolute_dose", "abscissa binding")
        same(row["baseline_control"], "frozen_inherited", "baseline binding")
        same(set(row["fields"]), names, "missing/ambiguous coefficient fields")
        rows[scope] = row
    same(set(rows), set(parent.SCOPES), "missing/incompatible coefficient scopes")
    failures, support, arithmetic, signs, masses, orientation = [], [], [], [], [], []
    zero, nonzero = [], []

    def verify(field, actual, expected, identity):
        if type(actual) is not type(expected) or actual != expected:
            failures.append({**identity, "field": field, "actual": str(actual), "expected": str(expected)})

    for field, expected in (("failure_count", 0), ("failures", []), ("unknown_reasons", [])):
        verify("70fb/" + field, r[field], expected, {})
    verify("retained_counts", r["counts"], affine.COUNTS, {})
    candidates = set(affine.CLASSES) - {"frozen_inherited"}
    values = {}
    max_dose = max(rational(d) for d, _ in affine.CLASSES.values())
    for scope, row in rows.items():
        identity = {k: row[k] for k in parent.SCOPE_FIELDS}
        fields, residuals = {}, {}
        for field, coefficient in row["fields"].items():
            same(set(coefficient), {"baseline", "slope", "candidate_slopes", "candidate_slope_residuals",
                                    "design_rank", "unique"}, "coefficient schema")
            baseline, slope = [rational(coefficient[k]) for k in ("baseline", "slope")]
            fields[field] = (baseline, slope)
            verify(field + "/unique", coefficient["unique"], True, identity)
            verify(field + "/design_rank", coefficient["design_rank"], 2, identity)
            for key in ("candidate_slopes", "candidate_slope_residuals"):
                same(set(coefficient[key]), candidates, "missing/ambiguous candidate slopes")
            for candidate in sorted(candidates):
                residual = rational(coefficient["candidate_slopes"][candidate]) - slope
                residuals[field + "/" + candidate] = str(residual)
                verify(field + "/" + candidate, residual, Fraction(0), identity)
                verify(field + "/retained_residual/" + candidate,
                       rational(coefficient["candidate_slope_residuals"][candidate]), Fraction(0), identity)
            record = {**identity, "field": field, "slope": str(slope)}
            (nonzero if slope else zero).append(record)
        values[scope] = fields
        arithmetic.append({**identity, "residuals": residuals})
        expected_support = {
            *(f"/components/{c}/{f}" for c in MOVING for f in SOURCE_FIELDS),
            *(f"/mass/{b}/{m}/{v}" for b in ("hidden", "weighted")
              for m in ("absolute", "cancellation_absolute_mass") for v in ("value", "movement")),
        } if row["coordinate"] == 62 else set()
        actual_support = {f for f, (_, slope) in fields.items() if slope}
        verify("exact_nonzero_support", actual_support, expected_support, identity)
        support.append({**identity, "nonzero_fields": sorted(actual_support),
                        "zero_fields": sorted(names - actual_support),
                        "expected_nonzero_fields": sorted(expected_support),
                        "exact": actual_support == expected_support})
        factor = fields["/" + parent.FACTOR][0]
        product = Fraction(1)
        for f in parent.FACTORS:
            product *= fields["/factors/" + f][0]
        verify("common_factor_product", factor, product, identity)
        verify("row_difference", fields["/factors/row_difference"][0],
               fields["/left_weight"][0] - fields["/right_weight"][0], identity)
        sign_residuals = {"common_factor_product": str(factor - product)}
        for component in parent.COMPONENTS:
            source_slope = Fraction(MOVING.get(component, 0) if row["coordinate"] == 62 else 0)
            for field in SOURCE_FIELDS:
                expected = source_slope if field in ("hidden", "source_delta") else source_slope * factor
                name = f"/components/{component}/{field}"
                residual = fields[name][1] - expected
                sign_residuals[name] = str(residual)
                verify(name + "/source_factor_sign", residual, Fraction(0), identity)
        signs.append({**identity, "common_factor": str(factor), "residuals": sign_residuals})

        def absolute_slope(baseline, slope, field):
            end = baseline + max_dose * slope
            verify(field + "/no_sign_crossing", baseline * end >= 0, True, identity)
            return slope if baseline > 0 else -slope if baseline < 0 else abs(slope)

        mass_residuals = {}
        for branch in ("hidden", "weighted"):
            components = [fields[f"/components/{c}/{branch}"] for c in parent.COMPONENTS]
            signed = sum((s for _, s in components), Fraction(0))
            absolute = sum((absolute_slope(b, s, f"{branch}/{c}")
                            for c, (b, s) in zip(parent.COMPONENTS, components)), Fraction(0))
            base_signed = sum((b for b, _ in components), Fraction(0))
            cancellation = absolute - absolute_slope(base_signed, signed, branch + "/signed")
            for mass, slope in (("signed", signed), ("absolute", absolute),
                                ("cancellation_absolute_mass", cancellation)):
                for value in ("value", "canonical", "movement"):
                    field = f"/mass/{branch}/{mass}/{value}"
                    expected = Fraction(0) if value == "canonical" else slope
                    residual = fields[field][1] - expected
                    mass_residuals[field] = str(residual)
                    verify(field + "/closure", residual, Fraction(0), identity)
        masses.append({**identity, "residuals": mass_residuals})
    for scope, row in rows.items():
        if (row["left_id"], row["right_id"]) != (34319, 319):
            continue
        reverse = {k: row[k] for k in parent.SCOPE_FIELDS}
        reverse.update(left_id=319, right_id=34319)
        other = values[tuple(reverse[k] for k in parent.SCOPE_FIELDS)]
        entries = {}
        for field, (baseline, slope) in values[scope].items():
            parts = field.split("/")
            target = {"/left_weight": "/right_weight", "/right_weight": "/left_weight"}.get(field, field)
            preserved = (
                field in ("/left_weight", "/right_weight")
                or parts[1] in ("factors", "factor_deltas") and parts[2] != "row_difference"
                or parts[1] == "components" and parts[3] in ("hidden", "canonical_hidden", "source_delta")
                or parts[1] == "mass" and (parts[2] == "hidden" or parts[3] != "signed")
            )
            multiplier = 1 if preserved else -1
            residuals = {"baseline": str(other[target][0] - multiplier * baseline),
                         "slope": str(other[target][1] - multiplier * slope)}
            for key, residual in residuals.items():
                verify(field + "/reverse/" + key, rational(residual), Fraction(0), reverse)
            entries[field] = {"reverse_field": target, "multiplier": multiplier, "residuals": residuals}
        orientation.append({"coordinate": row["coordinate"], "forward_pair": [34319, 319],
                            "reverse_pair": [319, 34319], "fields": entries})
    counts = {"coefficient_records": len(zero) + len(nonzero),
              "nonzero_slopes": len(nonzero), "zero_slopes": len(zero)}
    verify("slope_support_census", counts, COUNTS, {})
    return {
        "decision": "REJECTED" if failures else "SUPPORTED", "counts": counts,
        "expected_counts": COUNTS, "zero_slope_fields": zero, "nonzero_slope_fields": nonzero,
        "support_matrix": support, "sign_closure_matrix": signs, "mass_closure_matrix": masses,
        "orientation_closure_matrix": orientation, "arithmetic_closure_matrix": arithmetic,
        "failure_count": len(failures), "failures": failures, "unknown_reasons": [],
        "retained_70fb_pins": PINS, "retained_counts": r["counts"],
        "retained_quotient_bindings": r["retained_quotient_bindings"],
        "reference_scope": r["reference_scope"], "lineage_separation": r["lineage_separation"],
        "parent_source_equivalence_status_preserved": "REJECTED", "claim_boundary": r["claim_boundary"],
        "derivative_domain": "Retained canonical nonnegative dose interval only; no extrapolation or causal claim.",
    }


def capture_paths():
    paths = [Path(os.readlink(f"/proc/self/fd/{fd}")) for fd in (1, 2)]
    out, err = paths
    require(all(stat.S_ISREG(os.fstat(fd).st_mode) for fd in (1, 2)), "byte-exact file capture required")
    same(out.name, "check.stdout", "stdout capture member")
    same(err, out.with_name("check.stderr"), "stderr capture member")
    same(out.parent.name, "run", "capture run directory")
    same(out.parent.parent.parent, ROOT / "build", "isolated capture directory")
    return [out.parent / ("check" + suffix) for suffix in SUFFIXES[:3]]


def check():
    authentication = {"status": "UNKNOWN", "input_pins": PINS}
    audit, sources, capture = {"forbidden_calls": 0}, [], {"status": "UNKNOWN"}
    runtime = {"python": sys.executable, "version": sys.version, "implementation": sys.implementation.name,
               "cwd": str(Path.cwd()), "uid": os.getuid(),
               "environment": {k: os.environ.get(k) for k in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}}
    boundary = "Retained-only CPU non-admission accounting; independent Host Reviewer required."
    try:
        modules = load_helpers()
        affine, parent, helper = modules[0], modules[-2], modules[-1]
        boundary = parent.BOUNDARY
        audit = dict.fromkeys(helper.COUNTERS, 0)
        paths = capture_paths()
        allowed = allowed_paths(*modules) | {str(p) for p in paths}
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
            command, argv, environment = [p.read_bytes() for p in paths]
            preflight = helper.decode(environment, metadata=True)
            same(helper.decode(argv), [PYTHON, "-B", "-m", MODULE, "--check"], "capture argv")
            same(command, (environment_command(preflight["environment"], helper.decode(argv)) + "\n").encode(),
                 "capture command bytes")
            same(preflight["sources"], sources, "capture source/test binding")
            same((preflight["cwd"], preflight["uid"], preflight["python"], preflight["python_version"]),
                 (str(ROOT), 1000, PYTHON, sys.version), "capture runtime binding")
            same(preflight["executable"], runtime["executable_pin"], "capture executable binding")
            same({k: preflight["environment"][k] for k in runtime["environment"]},
                 runtime["environment"], "capture environment binding")
            same(preflight["independent_host_review"], "REQUIRED", "capture review gate")
            same(preflight["model_or_service_calls_authorized"], 0, "capture service budget")
            capture = {"status": "BYTE_EXACT_STDOUT_STDERR_FILES_OPEN",
                       "command_inputs": [pin(p, len(b), hashlib.sha256(b).hexdigest())
                                          for p, b in zip(paths, (command, argv, environment))],
                       "terminal_whole_capture": "Outer launcher must seal and validate before review."}
            documents, authentication = authenticate(*modules)
            result = classify(documents, affine, parent)
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
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_CANONICAL_DOSE_AFFINE_SLOPE_SUPPORT_CLASSIFIER",
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
