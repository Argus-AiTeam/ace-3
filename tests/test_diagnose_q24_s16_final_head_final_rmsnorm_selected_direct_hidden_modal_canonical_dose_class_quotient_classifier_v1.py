"""Independent integer quotient oracle, mutation coverage, and byte-exact one-shot capture."""

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_canonical_dose_class_quotient_classifier_v1 as diagnostic,
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
    return diagnostic.classify(retained[0] if documents is None else documents, retained[1][2])


def test_independent_integer_quotient_oracle(retained):
    result = classify(retained)
    assert result["decision"] == "SUPPORTED", result
    assert result["dose_classes"] == [
        {"canonical": "frozen_inherited", "absolute_dose": "0",
         "controls": ["frozen_inherited", "frozen_inherited_down", "inherited_native"]},
        {"canonical": "frozen_inherited_o", "absolute_dose": "3/32768",
         "controls": ["frozen_inherited_o", "frozen_inherited_o_down"]},
        {"canonical": "scratch", "absolute_dose": "321/4194304", "controls": ["scratch", "scratch_down"]},
        {"canonical": "mapped62", "absolute_dose": "933/16777216", "controls": ["mapped62"]},
    ]
    assert result["counts"] == {
        "component_accounts": 280, "control_coordinate_accounts": 40,
        "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
        "factor_proportionality_checks": 280, "reverse_component_checks": 112,
    }
    assert len(result["within_class_equality_matrix"]) == 90
    assert len(result["cross_class_proportionality_checks"]) == 210
    assert len(result["reverse_numeric_orientation_matrix"]) == 16
    for row in result["within_class_equality_matrix"]:
        assert row["all_numeric_fields_equal"] and all(row["field_equality"].values())
        if row["left_control"] != row["right_control"]:
            assert row["left_bindings"] != row["right_bindings"]
    for row in result["cross_class_proportionality_checks"]:
        delta = add(row["right_dose"], multiply("-1", row["left_dose"]))
        dh = multiply(delta, row["unit_source_delta"])
        assert row["dose_difference"] == delta
        assert row["source_delta_difference"] == dh
        assert row["weighted_movement_difference"] == multiply(dh, row["common_factor"])
        assert row["dose_scaled_source_only_movement"] == row["weighted_movement_difference"]
        assert row["equal"]
    for row in result["reverse_numeric_orientation_matrix"]:
        assert row["all_numeric_orientations_preserved"]
        assert all(row["field_orientation_preserved"].values())
    doses = {name: row["absolute_dose"] for row in result["dose_classes"] for name in row["controls"]}
    nonzero = 0
    for account in retained[0][2]["report"]["coordinate_accounts"]:
        base = account["controls"][0]
        for control in account["controls"]:
            for component, term in control["components"].items():
                sign = ("-1" if component == "actual_residual_boundary" else
                        "1" if component == "q24_to_fp16_conversion" else "0")
                dh = multiply(doses[control["control"]], sign) if account["coordinate"] == 62 else "0"
                assert term["source_delta"] == dh
                assert term["hidden"] == add(base["components"][component]["hidden"], dh)
                assert term["weighted_movement"] == multiply(
                    dh, control["weight_times_reference_anchor_times_row_difference"])
                nonzero += dh != "0"
    assert nonzero == 20


def test_every_retained_numeric_field_mutation_rejected(retained):
    control = retained[0][2]["report"]["coordinate_accounts"][0]["controls"][4]
    numeric = {k: v for k, v in control.items() if k not in ("control", "retained_bindings")}
    for field, value in diagnostic.leaves(numeric):
        documents = copy.deepcopy(retained[0])
        target = documents[2]["report"]["coordinate_accounts"][0]["controls"][4]
        parts = field.strip("/").split("/")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = add(value, "1")
        result = classify(retained, documents)
        assert result["decision"] == "REJECTED", (field, result)


@pytest.mark.parametrize("kind", [
    "alias", "membership", "dose", "counts", "zero_count", "mass", "reverse", "matrix",
    "matrix_dose", "rank", "source_classification", "singleton", "context", "canonical",
])
def test_semantic_mismatches_rejected(retained, kind):
    documents = copy.deepcopy(retained[0])
    down, dose, collapse = [d["report"] for d in documents[:3]]
    if kind == "alias":
        down["alias_map"]["scratch_down"] = "mapped62"
    elif kind == "membership":
        down["dose_groups"][1]["controls"].append("mapped62")
    elif kind == "dose":
        down["dose_groups"][2]["absolute_dose"] = "1"
    elif kind == "counts":
        down["counts"]["nonzero_source_only_movements"] = 19
    elif kind == "zero_count":
        dose["counts"]["zero_source_accounts"] = 259
    elif kind == "mass":
        collapse["coordinate_accounts"][0]["controls"][6]["mass"]["weighted"]["absolute"]["movement"] = "1"
    elif kind == "reverse":
        down["reverse_pair_checks"][0]["hidden_and_dose_equal"] = False
    elif kind == "matrix":
        row = down["alias_pair_matrix"][0]["fields"][0]
        row["alias_value"] = "1"
    elif kind == "matrix_dose":
        down["alias_pair_matrix"][0]["coordinate_absolute_dose"] = "1"
    elif kind == "rank":
        dose["absolute_dose_rank_descending"].reverse()
    elif kind == "source_classification":
        dose["zero_source_accounts"].append(dose["nonzero_source_only_movements"].pop())
    else:
        ai, ci = {"singleton": (0, 6), "context": (4, 0), "canonical": (0, 0)}[kind]
        collapse["coordinate_accounts"][ai]["controls"][ci]["components"]["input_hidden"]["hidden"] = "1"
    result = classify(retained, documents)
    assert result["decision"] == "REJECTED", result


@pytest.mark.parametrize("kind", ["missing", "duplicate", "rational", "binding", "scope", "prerequisite"])
def test_missing_ambiguous_or_incompatible_unknown(retained, kind):
    documents = copy.deepcopy(retained[0])
    control = documents[2]["report"]["coordinate_accounts"][0]["controls"][4]
    if kind == "missing":
        del control["components"]["input_hidden"]["hidden"]
    elif kind == "duplicate":
        documents[2]["report"]["coordinate_accounts"].append(documents[2]["report"]["coordinate_accounts"][0])
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
    assert audit["forbidden_calls"] == 1
    with pytest.raises(helper.ForbiddenOperation), helper.read_only(audit):
        subprocess.run(["false"], check=True)
    assert audit["forbidden_calls"] == 2
    tree = ast.parse(diagnostic.SOURCE.read_bytes())
    for node in ast.walk(tree):
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
        "independent_host_review": "REQUIRED", "scope": "390 retained-only canonical dose quotient",
        "model_or_service_calls_authorized": 0, "attempt": str(directory.parent),
    }
    results, failure = [], None

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
        suites = ET.parse(directory / "pytest.xml").getroot().findall("testsuite")
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
    receipt = {"preflight": preflight, "results": results, "failure": failure,
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


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--capture", "--run"):
        raise SystemExit("expected --capture build/<fresh-output> or --run build/<attempt>/run")
    raise SystemExit((launch if sys.argv[1] == "--capture" else capture)(sys.argv[2]))
