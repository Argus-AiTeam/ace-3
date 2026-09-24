"""Independent integer affine oracle, hostile controls, and exactly-once byte capture."""

import ast
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
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_normal_form_classifier_v1 as diagnostic,
)


def pair(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, d):
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


def divide(left, right):
    c, d = pair(right)
    return multiply(left, exact(d, c))


@pytest.fixture(scope="module")
def retained():
    modules = diagnostic.load_helpers()

    def forbidden(*args, **kwargs):
        raise AssertionError("ancestor semantic execution is forbidden")

    for module in modules:
        for name in ("check", "classify", "report", "_classify", "main"):
            setattr(module, name, forbidden)
    documents, authentication = diagnostic.authenticate(*modules)
    assert authentication["status"] == "AUTHENTICATED"
    return documents, modules


def classify(retained, documents=None):
    return diagnostic.classify(retained[0] if documents is None else documents, retained[1][-2])


def test_independent_integer_affine_oracle(retained):
    result = classify(retained)
    assert result["decision"] == "SUPPORTED", result
    assert result["counts"] == {
        "component_accounts": 280, "control_coordinate_accounts": 40,
        "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
        "factor_proportionality_checks": 280, "reverse_component_checks": 112,
    }
    assert result["retained_quotient_bindings"]["dose_classes"] == [
        {"canonical": "frozen_inherited", "absolute_dose": "0",
         "controls": ["frozen_inherited", "frozen_inherited_down", "inherited_native"]},
        {"canonical": "frozen_inherited_o", "absolute_dose": "3/32768",
         "controls": ["frozen_inherited_o", "frozen_inherited_o_down"]},
        {"canonical": "scratch", "absolute_dose": "321/4194304", "controls": ["scratch", "scratch_down"]},
        {"canonical": "mapped62", "absolute_dose": "933/16777216", "controls": ["mapped62"]},
    ]
    keys = retained[1][-2].SCOPE_FIELDS
    by_scope = {tuple(row[k] for k in keys): row for row in result["coefficient_matrix"]}
    accounts = {tuple(row[k] for k in keys): row for row in retained[0][3]["report"]["coordinate_accounts"]}
    assert len(by_scope) == 5 and len(result["reconstruction_residual_matrix"]) == 40
    for row in result["reconstruction_residual_matrix"]:
        scope = tuple(row[k] for k in keys)
        account = accounts[scope]
        base = dict(diagnostic.leaves(account["controls"][0]))
        anchor = dict(diagnostic.leaves(account["controls"][1]))
        actual = dict(diagnostic.leaves(next(c for c in account["controls"] if c["control"] == row["control"])))
        coefficients = by_scope[scope]["fields"]
        assert len(row["fields"]) == len(coefficients) == 147
        for field, values in row["fields"].items():
            c = coefficients[field]
            slope = divide(add(anchor[field], multiply("-1", base[field])), "3/32768")
            assert c["baseline"] == base[field] and c["slope"] == slope
            assert c["unique"] is True and c["design_rank"] == 2
            assert set(c["candidate_slopes"].values()) == {slope}
            assert set(c["candidate_slope_residuals"].values()) == {"0"}
            expected = add(base[field], multiply(slope, row["dose"]))
            assert values == {"retained": actual[field], "reconstructed": expected, "residual": "0"}
            assert expected == actual[field]
            if row["coordinate"] == 241:
                assert slope == "0"
        for component in retained[1][-2].COMPONENTS:
            prefix = "/components/" + component + "/"
            sign = ("-1" if component == "actual_residual_boundary" else
                    "1" if component == "q24_to_fp16_conversion" else "0") if row["coordinate"] == 62 else "0"
            f = actual["/weight_times_reference_anchor_times_row_difference"]
            assert coefficients[prefix + "source_delta"]["slope"] == sign
            assert coefficients[prefix + "hidden"]["slope"] == sign
            assert coefficients[prefix + "weighted_movement"]["slope"] == multiply(sign, f)
            assert actual[prefix + "weighted"] == multiply(actual[prefix + "hidden"], f)
        for branch in ("hidden", "weighted"):
            values = [actual["/components/" + c + "/" + branch] for c in retained[1][-2].COMPONENTS]
            total, absolute = "0", "0"
            for value in values:
                total = add(total, value)
                absolute = add(absolute, value.lstrip("-"))
            assert actual[f"/mass/{branch}/signed/value"] == total
            assert actual[f"/mass/{branch}/absolute/value"] == absolute
            assert actual[f"/mass/{branch}/cancellation_absolute_mass/value"] == add(
                absolute, multiply("-1", total.lstrip("-")))
    assert len(result["arithmetic_closure_residual_matrix"]) == 40
    assert all(set(row["residuals"].values()) == {"0"} for row in result["arithmetic_closure_residual_matrix"])
    assert len(result["source_movement_matrix"]) == 280
    assert sum(row["source_delta"] != "0" for row in result["source_movement_matrix"]) == 20
    assert len(result["reverse_pair_checks"]) == 112
    assert all(row["hidden_and_dose_equal"] and row["weighted_and_factor_opposite"]
               for row in result["reverse_pair_checks"])
    assert len(result["reverse_numeric_residual_matrix"]) == 16
    assert all(set(row["residuals"].values()) == {"0"} for row in result["reverse_numeric_residual_matrix"])
    for row in result["reverse_coefficient_matrix"]:
        for field, value in row["fields"].items():
            assert set(value["residuals"].values()) == {"0"}
            if field.startswith("/mass/weighted/absolute/") or field.startswith("/mass/weighted/cancellation"):
                assert value["multiplier"] == 1
            if field.startswith("/mass/weighted/signed/"):
                assert value["multiplier"] == -1
        assert row["fields"]["/left_weight"]["reverse_field"] == "/right_weight"


def test_every_numeric_field_nonaffinity_rejected(retained):
    control = retained[0][3]["report"]["coordinate_accounts"][0]["controls"][4]
    numeric = {k: v for k, v in control.items() if k not in ("control", "retained_bindings")}
    for field, value in diagnostic.leaves(numeric):
        target = control
        parts = field.strip("/").split("/")
        for part in parts[:-1]:
            target = target[part]
        try:
            target[parts[-1]] = add(value, "1")
            result = classify(retained)
            assert result["decision"] == "REJECTED", (field, result)
            assert any(f["field"].startswith("coefficient_uniqueness") for f in result["failures"]), field
        finally:
            target[parts[-1]] = value


@pytest.mark.parametrize("kind", [
    "dose", "membership", "count", "mass", "reverse", "context", "source_sign", "classification", "matrix",
])
def test_semantic_failures_rejected(retained, kind):
    documents = copy.deepcopy(retained[0])
    q = documents[0]["report"]
    accounts = documents[3]["report"]["coordinate_accounts"]
    if kind == "dose":
        q["dose_classes"][1]["absolute_dose"] = "0"
    elif kind == "membership":
        q["observed_quotient_membership"]["mapped62"] = ["scratch"]
    elif kind == "count":
        q["counts"]["component_accounts"] = 279
    elif kind == "reverse":
        q["reverse_pair_checks"][0]["hidden_and_dose_equal"] = False
    elif kind == "classification":
        documents[2]["report"]["zero_source_accounts"].append(
            documents[2]["report"]["nonzero_source_only_movements"].pop())
    elif kind == "matrix":
        q["coordinate_account_checks"][0]["numeric_field_equality"]["/left_weight"] = False
    elif kind == "mass":
        accounts[0]["controls"][0]["mass"]["hidden"]["absolute"]["value"] = "0"
    elif kind == "source_sign":
        accounts[0]["controls"][1]["components"]["actual_residual_boundary"]["source_delta"] = "3/32768"
    else:
        accounts[1]["controls"][1]["components"]["input_hidden"]["hidden"] = "1"
    assert classify(retained, documents)["decision"] == "REJECTED"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "rational", "binding", "scope", "prerequisite"])
def test_missing_ambiguous_incompatible_unknown(retained, kind):
    documents = copy.deepcopy(retained[0])
    control = documents[3]["report"]["coordinate_accounts"][0]["controls"][4]
    if kind == "missing":
        del control["components"]["input_hidden"]["hidden"]
    elif kind == "duplicate":
        documents[3]["report"]["coordinate_accounts"].append(documents[3]["report"]["coordinate_accounts"][0])
    elif kind == "rational":
        control["components"]["input_hidden"]["hidden"] = "2/2"
    elif kind == "binding":
        control["retained_bindings"]["factor_control_pointer"] = "/report/coordinate_accounts/0/controls/0"
    elif kind == "scope":
        documents[0]["report"]["lineage_separation"] = "other"
    else:
        documents[0]["decision"] = "UNKNOWN"
    assert classify(retained, documents)["decision"] == "UNKNOWN"


@pytest.mark.parametrize("target", list(diagnostic.PINS))
def test_root_pin_authentication(retained, monkeypatch, target):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[target]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError, match="artifact hash"):
        diagnostic.authenticate(*retained[1])


@pytest.mark.parametrize("kind", ["argv", "exit", "review", "counter", "outer", "environment", "whole"])
def test_capture_integrity(retained, monkeypatch, kind):
    read = diagnostic.bound_bytes

    def corrupt(binding):
        data = read(binding)
        key = {"argv": "capture", "exit": "capture", "review": "review", "counter": "stdout",
               "outer": "outer_capture", "environment": "capture"}.get(kind)
        if key and binding == diagnostic.PINS[key]:
            value = json.loads(data)
            if kind == "argv":
                value["results"][-1]["argv"][-1] = "--other"
            elif kind == "exit":
                value["results"][-1]["exit_status"] = 1
            elif kind == "review":
                value["producer_role"] = "engineer"
            elif kind == "counter":
                value["dispatch_and_write_audit"]["accepted_producer_replay"] = 1
            elif kind == "outer":
                value["timed_out"] = True
            else:
                value["preflight"]["environment"]["PYTHONPATH"] = "/other"
            return json.dumps(value).encode()
        if kind == "whole" and binding["path"].endswith("check.whole-command.log"):
            return data + b"\n"
        return data

    monkeypatch.setattr(diagnostic, "bound_bytes", corrupt)
    with pytest.raises(ValueError):
        diagnostic.authenticate(*retained[1])


def test_guard_no_ancestor_dispatch_and_cli(retained, monkeypatch, tmp_path, capsys):
    helper = retained[1][-1]
    allowed = diagnostic.allowed_paths(*retained[1])
    monkeypatch.setattr(helper, "allowed_paths", lambda: allowed)
    audit = dict.fromkeys(helper.COUNTERS, 0)
    with helper.read_only(audit):
        assert classify(retained)["decision"] == "SUPPORTED"
    assert all(type(v) is int and v == 0 for v in audit.values())
    with pytest.raises(helper.ForbiddenOperation), helper.read_only(audit):
        (tmp_path / "forbidden").write_text("not authorized")
    with pytest.raises(helper.ForbiddenOperation), helper.read_only(audit):
        subprocess.run(["false"], check=True)
    assert audit["forbidden_calls"] == 2
    for node in ast.walk(ast.parse(diagnostic.SOURCE.read_bytes())):
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] in sys.stdlib_module_names for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in sys.stdlib_module_names
    with pytest.raises(ValueError, match="duplicate JSON"):
        helper.decode(b'{"decision":"SUPPORTED","decision":"UNKNOWN"}')
    for decision, status in (("SUPPORTED", 0), ("REJECTED", 0), ("UNKNOWN", 2)):
        monkeypatch.setattr(diagnostic, "check", lambda: {"decision": decision})
        assert diagnostic.main(["--check"]) == status
        assert json.loads(capsys.readouterr().out) == {"decision": decision}


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


ENVIRONMENT = {
    "HOME": "/home/argustest", "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
    "LC_ALL": "C.UTF-8", "PYTHONPATH": str(diagnostic.ROOT), "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout", ".stderr", ".whole-command.log")


def capture(directory):
    """One compile/test/check sequence; only the outer harness writes evidence."""
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
        "environment": ENVIRONMENT, "sources": sources, "executable": binding(Path(python).resolve()),
        "independent_host_review": "REQUIRED", "scope": "b4 retained-only canonical dose affine normal form",
        "model_or_service_calls_authorized": 0, "attempt": str(directory.parent),
    }
    results, failure, pytest_xml = [], None, None

    def run(label, argv):
        command = diagnostic.environment_command(ENVIRONMENT, argv)
        command_bytes, env_bytes = (command + "\n").encode(), encoded(preflight)
        files = [save(directory, label + ".command.txt", command_bytes),
                 save(directory, label + ".argv.json", encoded(argv)),
                 save(directory, label + ".environment.json", env_bytes)]
        started, timed_out = time.monotonic(), False
        with (directory / (label + ".stdout")).open("xb") as out, (directory / (label + ".stderr")).open("xb") as err:
            try:
                status = subprocess.run(argv, cwd=root, env=ENVIRONMENT, stdout=out, stderr=err,
                                        timeout=90).returncode
            except subprocess.TimeoutExpired:
                status, timed_out = 124, True
        output, error = [(directory / (label + suffix)).read_bytes() for suffix in (".stdout", ".stderr")]
        files += [binding(directory / (label + suffix)) for suffix in (".stdout", ".stderr")]
        files.append(save(directory, label + ".whole-command.log",
                          b"COMMAND\n" + command_bytes + b"ENVIRONMENT\n" + env_bytes
                          + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
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
        if any(type(v) is not int or v != 0 for v in result["dispatch_and_write_audit"].values()):
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
        "argv": argv, "command": diagnostic.environment_command(ENVIRONMENT, argv),
        "cwd": str(Path.cwd()), "uid": os.getuid(), "environment": ENVIRONMENT,
        "launcher": binding(diagnostic.TEST), "invocation_argv": sys.orig_argv,
    }
    identity_bytes = encoded(identity)
    save(directory, "launcher.identity.json", identity_bytes)
    timed_out = False
    with (directory / "launcher.stdout").open("xb") as out, (directory / "launcher.stderr").open("xb") as err:
        try:
            status = subprocess.run(argv, cwd=diagnostic.ROOT, env=ENVIRONMENT, stdout=out, stderr=err,
                                    timeout=115).returncode
        except subprocess.TimeoutExpired:
            status, timed_out = 124, True
    output, error = [(directory / ("launcher." + suffix)).read_bytes() for suffix in ("stdout", "stderr")]
    save(directory, "launcher.whole-command.log", b"IDENTITY\n" + identity_bytes + b"STDOUT\n"
         + output + b"\nSTDERR\n" + error + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
    receipt = {"exit_status": status, "timed_out": timed_out,
               "files": [binding(directory / ("launcher." + suffix)) for suffix in
                         ("identity.json", "stdout", "stderr", "whole-command.log")]}
    save(directory, "launcher.capture.json", encoded(receipt))
    print(output.decode(), end="")
    print(json.dumps(receipt))
    return status


def validate_capture(directory):
    """Read stored bytes only; no authentication helper load or semantic recomputation."""
    directory = Path(directory).absolute()
    run = directory / "run"
    outer = json.loads((directory / "launcher.capture.json").read_bytes())
    diagnostic.same(outer["exit_status"], 0, "outer exit")
    diagnostic.same(outer["timed_out"], False, "outer timeout")
    diagnostic.same([p["path"] for p in outer["files"]],
                    [str(directory / ("launcher." + s)) for s in
                     ("identity.json", "stdout", "stderr", "whole-command.log")], "outer paths")
    identity_bytes, output, error, whole = [diagnostic.bound_bytes(p) for p in outer["files"]]
    identity = json.loads(identity_bytes)
    diagnostic.same(error, b"", "outer stderr")
    diagnostic.same(whole, b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n"
                    + error + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "outer whole bytes")
    receipt = json.loads(output)
    diagnostic.same(receipt["capture"]["path"], str(run / "capture.json"), "receipt path")
    capture_bytes = diagnostic.bound_bytes(receipt["capture"])
    captured = json.loads(capture_bytes)
    diagnostic.same(receipt["success"], True, "outer success")
    diagnostic.same(receipt["failure"], None, "outer failure")
    diagnostic.same(captured["success"], True, "capture success")
    diagnostic.same(captured["failure"], None, "capture failure")
    preflight = captured["preflight"]
    sources = [binding(diagnostic.SOURCE), binding(diagnostic.TEST)]
    diagnostic.same(preflight["sources"], sources, "source pins")
    diagnostic.same(captured["sources_after"], sources, "source drift")
    diagnostic.same(identity["launcher"], sources[1], "launcher pin")
    diagnostic.same(identity["argv"], [diagnostic.PYTHON, "-B", str(diagnostic.TEST), "--run", str(run)],
                    "outer argv")
    diagnostic.same(identity["command"], diagnostic.environment_command(ENVIRONMENT, identity["argv"]),
                    "outer command")
    diagnostic.same(preflight["environment"], ENVIRONMENT, "environment")
    for key in ("cwd", "uid", "environment"):
        diagnostic.same(identity[key], preflight[key], "outer identity splice")
    diagnostic.same((preflight["cwd"], preflight["uid"], preflight["python"]),
                    (str(diagnostic.ROOT), 1000, diagnostic.PYTHON), "runtime identity")
    diagnostic.bound_bytes(preflight["executable"])
    diagnostic.same(preflight["independent_host_review"], "REQUIRED", "review required")
    diagnostic.same(preflight["model_or_service_calls_authorized"], 0, "service budget")
    diagnostic.same([r["label"] for r in captured["results"]],
                    ["branch", "ignored-build", "concurrency", "compile", "pytest", "check"], "command census")
    result = None
    for row in captured["results"]:
        diagnostic.same(row["exit_status"], 0, "command exit")
        diagnostic.same(row["timed_out"], False, "command timeout")
        diagnostic.same([p["path"] for p in row["files"]],
                        [str(run / (row["label"] + s)) for s in SUFFIXES], "member paths")
        command, argv, env, out, err, all_bytes = [diagnostic.bound_bytes(p) for p in row["files"]]
        diagnostic.same(command, (row["command"] + "\n").encode(), "command bytes")
        diagnostic.same(row["command"], diagnostic.environment_command(ENVIRONMENT, row["argv"]), "command")
        diagnostic.same(json.loads(argv), row["argv"], "argv")
        diagnostic.same(json.loads(env), preflight, "environment bytes")
        diagnostic.same(err, b"", "stderr")
        diagnostic.same(all_bytes, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
                        + out + b"\nSTDERR\n" + err + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "whole bytes")
        if row["label"] == "check":
            diagnostic.same(row["argv"], [diagnostic.PYTHON, "-B", "-m", diagnostic.MODULE, "--check"],
                            "disclosed invocation")
            result = json.loads(out)
    suites = ET.fromstring(diagnostic.bound_bytes(captured["pytest_xml"])).findall("testsuite")
    diagnostic.require(bool(suites) and sum(int(s.attrib["tests"]) for s in suites) > 0, "tests absent")
    diagnostic.require(not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
                       "test failure/error/skip")
    diagnostic.same((result["artifact_authentication"]["status"], result["decision"], result["report"]["decision"]),
                    ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "result status")
    diagnostic.same(result["source_test_pins"], sources, "result sources")
    diagnostic.same(result["command"], diagnostic.COMMAND, "disclosed command")
    diagnostic.same(result["runtime"]["executable_pin"], preflight["executable"], "result runtime")
    diagnostic.require(bool(result["dispatch_and_write_audit"]) and all(
        type(v) is int and v == 0 for v in result["dispatch_and_write_audit"].values()), "forbidden counters")
    report = result["report"]
    diagnostic.same(report["counts"], diagnostic.COUNTS, "counts")
    diagnostic.same(report["failure_count"], 0, "failures")
    diagnostic.require(all(c["unique"] is True for row in report["coefficient_matrix"]
                           for c in row["fields"].values()), "nonunique coefficient")
    diagnostic.require(all(c["residual"] == "0" for row in report["reconstruction_residual_matrix"]
                           for c in row["fields"].values()), "reconstruction residual")
    for matrix in ("arithmetic_closure_residual_matrix", "reverse_numeric_residual_matrix"):
        diagnostic.require(all(v == "0" for row in report[matrix] for v in row["residuals"].values()), matrix)
    print(json.dumps({"stored_bytes_validated_without_semantic_recomputation": True,
                      "capture": receipt["capture"], "stdout": binding(run / "check.stdout"),
                      "tests": sum(int(s.attrib["tests"]) for s in suites),
                      "decision": result["decision"], "counts": report["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--capture", "--run", "--validate"):
        raise SystemExit("expected --capture build/<fresh-output>, --run build/<attempt>/run, or --validate build/<attempt>")
    raise SystemExit({"--capture": launch, "--run": capture, "--validate": validate_capture}[sys.argv[1]](sys.argv[2]))
