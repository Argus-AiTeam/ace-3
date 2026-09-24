"""Independent integer root/margin oracle, hostile controls and exactly-once byte capture."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_root_margin_classifier_v1 as diagnostic,
)


def pair(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, d=1):
    divisor = math.gcd(n, d)
    n, d = n // divisor, d // divisor
    if d < 0:
        n, d = -n, -d
    return str(n) if d == 1 else f"{n}/{d}"


def add(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * d + c * b, b * d)


def multiply(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * c, b * d)


def magnitude(value):
    n, d = pair(value)
    return exact(abs(n), d)


def less(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return a * d < c * b


def minimum(values):
    result = values[0]
    for value in values[1:]:
        if less(value, result):
            result = value
    return result


@pytest.fixture(scope="module")
def retained():
    modules = diagnostic.load_helpers()

    def forbidden(*args, **kwargs):
        raise AssertionError("ancestor semantic execution forbidden")

    for module in modules:
        for name in ("check", "classify", "report", "_classify", "main"):
            setattr(module, name, forbidden)
    documents, authentication = diagnostic.authenticate(*modules)
    assert authentication["status"] == "AUTHENTICATED"
    assert authentication["review"]["producer_role"] == "reviewer"
    return documents, modules


def test_independent_integer_oracle_every_field_and_class(retained):
    documents, modules = retained
    result = diagnostic.classify(documents, modules)
    assert result["decision"] == "SUPPORTED", result.get("unknown_reasons", result.get("failures"))
    assert result["counts"] == diagnostic.COUNTS
    assert len(result["root_margin_matrix"]) == 735
    coefficients = {
        (*[row[k] for k in modules[-2].SCOPE_FIELDS], field): value
        for row in documents[1]["report"]["coefficient_matrix"] for field, value in row["fields"].items()
    }
    locations, stable = {}, dict.fromkeys(("sign_stable", "strict_sign_stable", "no_sign_reversal"), 0)
    for row in result["root_margin_matrix"]:
        key = (*[row[k] for k in modules[-2].SCOPE_FIELDS], row["field"])
        coefficient = coefficients[key]
        baseline, slope = coefficient["baseline"], coefficient["slope"]
        a, b = pair(baseline)
        c, d = pair(slope)
        root = exact(-a * d, b * c) if c else None
        assert row["root"] == root
        assert row["root_kind"] == ("UNIQUE" if c else "NO_ROOT" if a else "ALL_DOSES")
        expected_location = (
            "BELOW" if less(root, "0") else "LOWER_ENDPOINT" if root == "0" else
            "INTERIOR" if less(root, "3/32768") else "UPPER_ENDPOINT" if root == "3/32768" else "ABOVE"
        ) if root is not None else row["root_kind"]
        assert row["root_location"] == expected_location
        locations[expected_location] = locations.get(expected_location, 0) + 1
        margins, signs = {}, set()
        for name, (dose, _) in modules[1].CLASSES.items():
            value = add(baseline, multiply(slope, dose))
            entry = row["classes"][name]
            numerator, _ = pair(value)
            sign = (numerator > 0) - (numerator < 0)
            assert entry["dose"] == dose
            assert entry["value"] == value
            assert entry["absolute_margin"] == magnitude(value)
            assert entry["sign"] == sign
            margins[name] = magnitude(value)
            signs.add(sign)
            if root is not None:
                distance = add(dose, multiply("-1", root))
                assert entry["signed_root_distance"] == distance
                assert entry["absolute_root_distance"] == magnitude(distance)
                assert magnitude(value) == multiply(magnitude(slope), magnitude(distance))
                assert entry["root_distance_identity_residual"] == "0"
                assert add(baseline, multiply(slope, root)) == "0"
                assert row["root_equation_residual"] == "0"
            else:
                assert entry["absolute_root_distance"] == ("0" if not a else None)
                assert entry["signed_root_distance"] is None
                assert entry["root_distance_identity_residual"] is None
                assert row["root_equation_residual"] is None
        closest = minimum(list(margins.values()))
        assert row["minimum_retained_absolute_margin"] == closest
        assert row["nearest_classes"] == [k for k, v in margins.items() if v == closest]
        inside = expected_location in ("ALL_DOSES", "LOWER_ENDPOINT", "INTERIOR", "UPPER_ENDPOINT")
        assert row["root_in_retained_interval"] is inside
        assert row["interval_minimum_absolute_margin"] == ("0" if inside else closest)
        expected = {"sign_stable": len(signs) == 1, "strict_sign_stable": len(signs) == 1 and 0 not in signs,
                    "no_sign_reversal": not {-1, 1} <= signs}
        for k, v in expected.items():
            assert row[k] is v
            stable[k] += v
        assert len(row["retained_value_residuals"]) == 8
        assert set(row["retained_value_residuals"].values()) == {"0"}
    assert result["root_location_counts"] == locations
    assert result["sign_stability_counts"] == stable
    assert sum(locations.values()) == 735
    for canonical, (dose, _) in modules[1].CLASSES.items():
        unique = [row for row in result["root_margin_matrix"] if row["root"] is not None]
        distances = [magnitude(add(dose, multiply("-1", row["root"]))) for row in unique]
        closest = minimum(distances)
        summary = result["nearest_unique_roots_by_class"][canonical]
        assert summary["minimum"] == closest
        assert summary["fields"] == [
            {k: row[k] for k in ("table", "branch", "left_id", "right_id", "coordinate", "field", "root")}
            for row, distance in zip(unique, distances, strict=True) if distance == closest
        ]
        margins = [row["classes"][canonical]["absolute_margin"] for row in result["root_margin_matrix"]
                   if row["classes"][canonical]["absolute_margin"] != "0"]
        assert result["nearest_nonzero_margins_by_class"][canonical]["minimum"] == minimum(margins)
    for side, endpoint in (("below", "0"), ("above", "3/32768")):
        outside = [row for row in unique if (
            less(row["root"], endpoint) if side == "below" else less(endpoint, row["root"])
        )]
        distances = [magnitude(add(row["root"], multiply("-1", endpoint))) for row in outside]
        summary = result["nearest_off_interval_roots"][side]
        closest = minimum(distances) if distances else None
        assert summary["minimum"] == closest
        assert summary["fields"] == [
            {k: row[k] for k in ("table", "branch", "left_id", "right_id", "coordinate", "field", "root")}
            for row, distance in zip(outside, distances, strict=True) if distance == closest
        ]
    assert result["failure_count"] == 0 and result["failures"] == [] and result["unknown_reasons"] == []
    assert result["retained_2861_pins"] == diagnostic.PINS
    assert result["parent_source_equivalence_status_preserved"] == "REJECTED"
    for key in ("reference_scope", "lineage_separation", "claim_boundary"):
        assert result[key] == documents[0]["report"][key]


@pytest.mark.parametrize(("baseline", "slope", "root", "location", "stable", "strict", "margin"), [
    ("0", "0", None, "ALL_DOSES", True, False, "0"),
    ("2", "0", None, "NO_ROOT", True, True, "2"),
    ("-2", "0", None, "NO_ROOT", True, True, "2"),
    ("0", "1", "0", "LOWER_ENDPOINT", False, False, "0"),
    ("-2", "1", "2", "UPPER_ENDPOINT", False, False, "0"),
    ("-1", "1", "1", "INTERIOR", False, False, "0"),
    ("1", "-1", "1", "INTERIOR", False, False, "0"),
    ("1", "1", "-1", "BELOW", True, True, "1"),
    ("3", "-1", "3", "ABOVE", True, True, "1"),
])
def test_exact_root_degeneracies(baseline, slope, root, location, stable, strict, margin):
    f = diagnostic.rational
    row = diagnostic.root_margin(f(baseline), f(slope), {"lower": f("0"), "upper": f("2")})
    assert (row["root"], row["root_location"]) == (root, location)
    assert row["sign_stable"] is stable and row["strict_sign_stable"] is strict
    assert row["interval_minimum_absolute_margin"] == margin
    assert row["no_sign_reversal"] is (location != "INTERIOR")


@pytest.mark.parametrize("kind", [
    "baseline", "slope", "sign", "count", "support", "candidate", "rank", "unique",
    "arithmetic", "reconstruction", "retained_value", "orientation",
])
def test_arithmetic_count_sign_mismatch_rejected(retained, kind):
    documents, modules = retained
    changed = list(documents)
    for index in (0, 1):
        changed[index] = {**changed[index], "report": copy.deepcopy(changed[index]["report"])}
    support, normal = changed[0]["report"], changed[1]["report"]
    coefficient = normal["coefficient_matrix"][0]["fields"]["/components/actual_residual_boundary/hidden"]
    if kind in ("baseline", "slope"):
        coefficient[kind] = add(coefficient[kind], "1")
    elif kind in ("sign", "arithmetic"):
        key = "sign_closure_matrix" if kind == "sign" else "arithmetic_closure_matrix"
        residuals = support[key][0]["residuals"]
        residuals[next(iter(residuals))] = "1"
    elif kind == "count":
        support["counts"]["zero_slopes"] -= 1
    elif kind == "support":
        support["support_matrix"][0]["exact"] = False
    elif kind == "candidate":
        coefficient["candidate_slopes"]["scratch"] = "1"
    elif kind in ("rank", "unique"):
        coefficient["design_rank" if kind == "rank" else "unique"] = 1 if kind == "rank" else False
    elif kind in ("reconstruction", "retained_value"):
        item = normal["reconstruction_residual_matrix"][0]["fields"]["/components/actual_residual_boundary/hidden"]
        item["reconstructed" if kind == "reconstruction" else "retained"] = "0"
    else:
        fields = support["orientation_closure_matrix"][0]["fields"]
        fields[next(iter(fields))]["residuals"]["baseline"] = "1"
    result = diagnostic.classify(changed, modules)
    assert result["decision"] == "REJECTED", (kind, result)
    assert result["failure_count"] > 0 and result["unknown_reasons"] == []


@pytest.mark.parametrize("kind", ["missing", "duplicate", "noncanonical", "dose", "lineage", "binding"])
def test_missing_ambiguous_or_incompatible_unknown(retained, kind):
    documents, modules = retained
    changed = list(documents)
    changed[1] = {**changed[1], "report": copy.deepcopy(changed[1]["report"])}
    normal = changed[1]["report"]
    if kind == "missing":
        normal["coefficient_matrix"][0]["fields"].pop("/left_weight")
    elif kind == "duplicate":
        normal["coefficient_matrix"].append(normal["coefficient_matrix"][0])
    elif kind == "noncanonical":
        normal["coefficient_matrix"][0]["fields"]["/left_weight"]["baseline"] = "2/2"
    elif kind == "dose":
        normal["retained_quotient_bindings"]["dose_classes"][0]["absolute_dose"] = "1"
    elif kind == "lineage":
        normal["lineage_separation"] = "spliced"
    else:
        normal["reconstruction_residual_matrix"][0]["retained_bindings"] = {}
    result = diagnostic.classify(changed, modules)
    assert result["decision"] == "UNKNOWN", (kind, result)
    assert result["unknown_reasons"]


@pytest.mark.parametrize("target", list(diagnostic.PINS))
def test_all_root_pins_authenticated(target, monkeypatch):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[target]["sha256"] = "0" * 64
    modules = diagnostic.load_helpers()
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError, match="artifact hash"):
        diagnostic.authenticate(*modules)


@pytest.mark.parametrize("kind", ["whole", "stderr", "argv", "environment", "exit", "review", "counter"])
def test_capture_review_runtime_tamper(retained, monkeypatch, kind):
    original = diagnostic.bound_bytes
    paths = diagnostic.PINS
    capture = json.loads(original(paths["capture"]))
    check = capture["results"][-1]
    target = {
        "whole": check["files"][5]["path"], "stderr": check["files"][4]["path"],
        "argv": check["files"][1]["path"], "environment": check["files"][2]["path"],
        "exit": paths["capture"]["path"], "review": paths["review"]["path"],
        "counter": paths["stdout"]["path"],
    }[kind]

    def corrupted(binding):
        data = original(binding)
        if binding["path"] != target:
            return data
        if kind in ("whole", "stderr"):
            return data + b"tampered"
        value = json.loads(data)
        if kind == "argv":
            value[-1] = "--other"
        elif kind == "environment":
            value["uid"] = 0
        elif kind == "exit":
            value["results"][-1]["exit_status"] = 1
        elif kind == "review":
            value["producer_role"] = "engineer"
        else:
            value["dispatch_and_write_audit"]["prefix_dispatch"] = 1
        return json.dumps(value).encode()

    monkeypatch.setattr(diagnostic, "bound_bytes", corrupted)
    with pytest.raises(ValueError):
        diagnostic.authenticate(*retained[1])


def test_read_only_guard_and_no_ancestor_semantics(retained, tmp_path):
    documents, modules = retained
    helper = modules[-1]
    audit = dict.fromkeys(helper.COUNTERS, 0)
    with helper.read_only(audit):
        assert diagnostic.classify(documents, modules)["decision"] == "SUPPORTED"
        with pytest.raises(helper.ForbiddenOperation):
            (tmp_path / "forbidden").write_text("no")
        with pytest.raises(helper.ForbiddenOperation):
            subprocess.run([diagnostic.PYTHON, "-V"], check=True)
    assert audit["forbidden_calls"] == 2
    assert not (tmp_path / "forbidden").exists()


def test_authentication_unknown_and_cli_exit(retained, monkeypatch, capsys):
    def rejected_load():
        raise ValueError("authentication pin mismatch")

    monkeypatch.setattr(diagnostic, "load_helpers", rejected_load)
    assert diagnostic.main(["--check"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "UNKNOWN"
    assert "authentication pin mismatch" in result["report"]["unknown_reasons"][0]
    with pytest.raises(SystemExit) as error:
        diagnostic.main(["--check", "--output", "forbidden"])
    assert error.value.code == 2


def test_uncaptured_stdout_unknown():
    result = diagnostic.check()
    assert result["decision"] == "UNKNOWN"
    assert result["byte_exact_capture"]["status"] == "UNKNOWN"


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"


def binding(path):
    data = path.read_bytes()
    return diagnostic.pin(path, len(data), hashlib.sha256(data).hexdigest())


def save(directory, name, data):
    path = directory / name
    with path.open("xb") as stream:
        stream.write(data)
    return binding(path)


def capture(directory):
    """Execute the task-native compile/pytest/check sequence once, retaining failures."""
    root, python = diagnostic.ROOT, diagnostic.PYTHON
    if (Path.cwd(), os.getuid(), sys.executable) != (root, 1000, python):
        raise RuntimeError("account/workdir/interpreter gate")
    directory = Path(directory).absolute()
    if directory.parent.parent != root / "build" or directory.name != "run":
        raise RuntimeError("fresh build/<attempt>/run required")
    directory.mkdir(exist_ok=False)
    sources = [binding(diagnostic.SOURCE), binding(diagnostic.TEST)]
    preflight = {
        "cwd": str(root), "uid": os.getuid(), "python": python, "python_version": sys.version,
        "environment": diagnostic.ENVIRONMENT, "sources": sources, "executable": binding(Path(python).resolve()),
        "independent_host_review": "REQUIRED", "scope": "2861 retained-only affine root/sign-margin classification",
        "model_or_service_calls_authorized": 0, "attempt": str(directory.parent),
    }
    results, failure, pytest_xml = [], None, None

    def run(label, argv):
        command = diagnostic.environment_command(diagnostic.ENVIRONMENT, argv)
        command_bytes, env_bytes = (command + "\n").encode(), encoded(preflight)
        files = [save(directory, label + ".command.txt", command_bytes),
                 save(directory, label + ".argv.json", encoded(argv)),
                 save(directory, label + ".environment.json", env_bytes)]
        started, timed_out = time.monotonic(), False
        with (directory / (label + ".stdout")).open("xb") as out, (directory / (label + ".stderr")).open("xb") as err:
            try:
                status = subprocess.run(argv, cwd=root, env=diagnostic.ENVIRONMENT,
                                        stdout=out, stderr=err, timeout=90).returncode
            except subprocess.TimeoutExpired:
                status, timed_out = 124, True
        output, error = [(directory / (label + suffix)).read_bytes() for suffix in (".stdout", ".stderr")]
        files += [binding(directory / (label + suffix)) for suffix in (".stdout", ".stderr")]
        files.append(save(directory, label + ".whole-command.log", b"COMMAND\n" + command_bytes
                          + b"ENVIRONMENT\n" + env_bytes + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
                          + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode()))
        results.append({"label": label, "command": command, "argv": argv, "files": files,
                        "exit_status": status, "timed_out": timed_out,
                        "elapsed_seconds": time.monotonic() - started})
        if status != 0 or timed_out or error:
            raise RuntimeError(label + " failed; complete bytes retained")
        return output

    try:
        if run("branch", ["git", "branch", "--show-current"]) != b"argus/full-projection\n":
            raise RuntimeError("branch gate")
        run("ignored-build", ["git", "check-ignore", str(directory)])
        probe = (
            "import json,os,subprocess; "
            "rows=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines(); "
            f"module={diagnostic.MODULE!r}; test={str(diagnostic.TEST)!r}; "
            "matches=[r for r in rows if int(r.split(None,1)[0]) != os.getpid() "
            "and ((' -m '+module+' ') in r or (' -m pytest ' in r and test in r))]; print(json.dumps(matches))"
        )
        if json.loads(run("concurrency", [python, "-B", "-c", probe])) != []:
            raise RuntimeError("same-scope concurrency")
        code = ("from pathlib import Path; "
                f"paths={[str(diagnostic.SOURCE), str(diagnostic.TEST)]!r}; "
                "[compile(Path(p).read_bytes(),p,'exec') for p in paths]; print('compiled 2 Python files')")
        run("compile", [python, "-B", "-c", code])
        run("pytest", [python, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--basetemp", str(directory / "pytest-tmp"), "--junitxml", str(directory / "pytest.xml"),
                       str(diagnostic.TEST)])
        pytest_xml = binding(directory / "pytest.xml")
        suites = ET.fromstring(diagnostic.bound_bytes(pytest_xml)).findall("testsuite")
        if not suites or sum(int(s.attrib["tests"]) for s in suites) == 0 or any(
            int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")
        ):
            raise RuntimeError("test failure/error/skip")
        result = json.loads(run("check", [python, "-B", "-m", diagnostic.MODULE, "--check"]))
        if (result["artifact_authentication"]["status"], result["decision"]) != ("AUTHENTICATED", "SUPPORTED"):
            raise RuntimeError("diagnostic not authenticated SUPPORTED")
        if set(result["dispatch_and_write_audit"]) != set(diagnostic.load_helpers()[-1].COUNTERS) or any(
            type(v) is not int or v != 0 for v in result["dispatch_and_write_audit"].values()
        ):
            raise RuntimeError("forbidden counter")
        if [binding(diagnostic.SOURCE), binding(diagnostic.TEST)] != sources:
            raise RuntimeError("source/test drift")
    except (OSError, ValueError, KeyError, RuntimeError, ET.ParseError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    receipt = {"preflight": preflight, "results": results, "failure": failure, "pytest_xml": pytest_xml,
               "success": failure is None, "sources_after": [binding(diagnostic.SOURCE), binding(diagnostic.TEST)]}
    captured = save(directory, "capture.json", encoded(receipt))
    print(json.dumps({"capture": captured, "success": failure is None, "failure": failure}))
    return 1 if failure else 0


def launch(directory):
    directory = Path(directory).absolute()
    if directory.parent != diagnostic.ROOT / "build":
        raise RuntimeError("fresh direct build child required")
    directory.mkdir(exist_ok=False)
    argv = [diagnostic.PYTHON, "-B", str(diagnostic.TEST), "--run", str(directory / "run")]
    identity = {
        "argv": argv, "command": diagnostic.environment_command(diagnostic.ENVIRONMENT, argv),
        "cwd": str(Path.cwd()), "uid": os.getuid(), "environment": diagnostic.ENVIRONMENT,
        "launcher": binding(diagnostic.TEST), "invocation_argv": sys.orig_argv,
    }
    identity_bytes = encoded(identity)
    save(directory, "launcher.identity.json", identity_bytes)
    timed_out = False
    with (directory / "launcher.stdout").open("xb") as out, (directory / "launcher.stderr").open("xb") as err:
        try:
            status = subprocess.run(argv, cwd=diagnostic.ROOT, env=diagnostic.ENVIRONMENT,
                                    stdout=out, stderr=err, timeout=115).returncode
        except subprocess.TimeoutExpired:
            status, timed_out = 124, True
    output, error = [(directory / ("launcher." + suffix)).read_bytes() for suffix in ("stdout", "stderr")]
    save(directory, "launcher.whole-command.log", b"IDENTITY\n" + identity_bytes + b"STDOUT\n"
         + output + b"\nSTDERR\n" + error + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
    outer = {"exit_status": status, "timed_out": timed_out,
             "files": [binding(directory / ("launcher." + suffix)) for suffix in
                       ("identity.json", "stdout", "stderr", "whole-command.log")]}
    save(directory, "launcher.capture.json", encoded(outer))
    if status == 0:
        helper = diagnostic.load_helpers()[-1]
        receipt = helper.decode(output)
        captured = helper.decode(diagnostic.bound_bytes(receipt["capture"]), metadata=True)
        stdout = (directory / "run/check.stdout").read_bytes()
        preflight, inputs = diagnostic.verify_capture(
            directory, captured, outer, stdout, [binding(diagnostic.SOURCE), binding(diagnostic.TEST)],
            diagnostic.MODULE, helper,
        )
        result = helper.decode(stdout, metadata=True)
        diagnostic.same(result["byte_exact_capture"]["command_inputs"], inputs, "current capture identity")
        diagnostic.same(result["source_test_pins"], preflight["sources"], "current source pins")
        diagnostic.same(result["report"]["counts"], diagnostic.COUNTS, "current census")
        diagnostic.same(result["report"]["failure_count"], 0, "current failures")
    print(output.decode(), end="")
    print(json.dumps(outer))
    return status


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--launch", "--run"):
        raise SystemExit({"--launch": launch, "--run": capture}[sys.argv[1]](Path(sys.argv[2])))
    raise SystemExit("use --launch or --run <fresh build attempt>")
