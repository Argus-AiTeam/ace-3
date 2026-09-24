"""Independent integer oracle, retained-field mutations and bounded capture runner."""

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
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_down_axis_invariance_classifier_v1 as diagnostic,
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


def subtract(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * d - c * b, b * d)


def multiply(left, right):
    a, b = pair(left)
    c, d = pair(right)
    return exact(a * c, b * d)


@pytest.fixture(scope="module")
def retained():
    dose, parent, helper = diagnostic.load_helpers()

    def forbidden(*args, **kwargs):
        raise AssertionError("ancestor semantic execution is forbidden")

    for module in (dose, parent, helper):
        for name in ("check", "classify", "report", "_classify", "main"):
            setattr(module, name, forbidden)
    documents, authentication = diagnostic.authenticate(dose, parent, helper)
    assert authentication["status"] == "AUTHENTICATED"
    return documents, dose, parent, helper


def classify(retained, documents=None):
    return diagnostic.classify(retained[0] if documents is None else documents, retained[2])


def test_independent_alias_and_integer_oracle(retained):
    result = classify(retained)
    assert result["decision"] == "SUPPORTED", result
    assert result["failure_count"] == 0
    assert result["counts"] == {
        "component_accounts": 280, "control_coordinate_accounts": 40,
        "nonzero_source_only_movements": 20, "zero_source_accounts": 260,
        "factor_proportionality_checks": 280, "reverse_component_checks": 112,
    }
    aliases = {"frozen_inherited_down": "frozen_inherited",
               "frozen_inherited_o_down": "frozen_inherited_o", "scratch_down": "scratch",
               "inherited_native": "frozen_inherited"}
    assert result["alias_map"] == aliases
    assert result["alias_coordinate_comparisons"] == 20
    assert result["alias_component_comparisons"] == 140
    counts, observed = {"zero": 0, "nonzero": 0}, {}
    for account in retained[0][1]["report"]["coordinate_accounts"]:
        controls = {c["control"]: c for c in account["controls"]}
        for alias, canonical in aliases.items():
            for field in set(controls[alias]) - {"control", "retained_bindings"}:
                assert controls[alias][field] == controls[canonical][field]
            assert controls[alias]["retained_bindings"] != controls[canonical]["retained_bindings"]
        for name, control in controls.items():
            for component, term in control["components"].items():
                dh = subtract(term["hidden"], term["canonical_hidden"])
                dw = subtract(term["weighted"], term["canonical_weighted"])
                assert dh == term["source_delta"]
                assert dw == term["weighted_movement"] == multiply(dh, control[retained[2].FACTOR])
                counts["zero" if dh == "0" else "nonzero"] += 1
                observed[(account["left_id"], account["right_id"], account["coordinate"], name, component)] = (dh, dw)
    assert counts == {"zero": 260, "nonzero": 20}
    for (left, right, coordinate, name, component), (dh, dw) in observed.items():
        if (left, right) == (34319, 319):
            assert observed[(right, left, coordinate, name, component)] == (dh, subtract("0", dw))
    assert len(result["alias_pair_matrix"]) == 20
    assert all(r["equal_under_alias_map"] and all(f["equal_under_alias_map"] for f in r["fields"])
               for r in result["alias_pair_matrix"])
    assert all(r["hidden_and_dose_equal"] and r["weighted_and_factor_opposite"]
               for r in result["reverse_pair_checks"])


@pytest.mark.parametrize("field", sorted(diagnostic.TERM_FIELDS - {"individual_factor_delta_terms"}))
def test_every_component_field_mismatch_rejected(retained, field):
    documents = copy.deepcopy(retained[0])
    documents[1]["report"]["coordinate_accounts"][0]["controls"][2]["components"]["input_hidden"][field] = "1"
    result = classify(retained, documents)
    assert result["decision"] == "REJECTED", result
    assert any(f["field"] == "alias/components/input_hidden/" + field for f in result["failures"])


@pytest.mark.parametrize("kind", ["factor", "factor_delta", "individual_delta", "mass",
                                 "counts", "orientation", "dose", "rank", "gap", "zero", "nonzero"])
def test_inconsistencies_rejected(retained, kind):
    documents = copy.deepcopy(retained[0])
    dose = documents[0]["report"]
    control = documents[1]["report"]["coordinate_accounts"][0]["controls"][2]
    if kind == "factor":
        control["factors"]["norm_weight"] = "0"
    elif kind == "factor_delta":
        control["factor_deltas"]["norm_weight"] = "1"
    elif kind == "individual_delta":
        control["components"]["input_hidden"]["individual_factor_delta_terms"]["norm_weight"] = "1"
    elif kind == "mass":
        control["mass"]["hidden"]["absolute"]["movement"] = "1"
    elif kind == "counts":
        dose["counts"]["zero_source_accounts"] = 259
    elif kind == "orientation":
        dose["reverse_pair_checks"][0]["weighted_and_factor_opposite"] = False
    elif kind == "dose":
        dose["dose_groups"][0]["absolute_dose"] = "0"
    elif kind == "rank":
        dose["absolute_dose_rank_descending"].reverse()
    elif kind == "gap":
        dose["pairwise_rational_gaps"][0]["gap"] = "0"
    elif kind == "zero":
        dose["zero_source_accounts"][0]["weighted_movement"] = "1"
    else:
        dose["nonzero_source_only_movements"][0]["source_delta"] = "0"
    assert classify(retained, documents)["decision"] == "REJECTED"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "pointer", "scope", "unsupported",
                                 "component", "control", "ambiguous_rational"])
def test_missing_or_incompatible_fields_unknown(retained, kind):
    documents = copy.deepcopy(retained[0])
    dose = documents[0]["report"]
    controls = documents[1]["report"]["coordinate_accounts"][0]["controls"]
    if kind == "missing":
        dose["zero_source_accounts"].pop()
    elif kind == "duplicate":
        dose["zero_source_accounts"].append(copy.deepcopy(dose["zero_source_accounts"][0]))
    elif kind == "pointer":
        controls[2]["retained_bindings"]["raw_selected_row_pointer"] = "/report/missing"
    elif kind == "scope":
        dose["reference_scope"] = "replacement reference"
    elif kind == "unsupported":
        documents[0]["decision"] = "UNKNOWN"
    elif kind == "component":
        del controls[2]["components"]["input_hidden"]["source_delta"]
    elif kind == "control":
        controls.pop()
    else:
        controls[2]["components"]["input_hidden"]["source_delta"] = "0/1"
    assert classify(retained, documents)["decision"] == "UNKNOWN"


@pytest.mark.parametrize("target", sorted(diagnostic.PINS))
def test_root_pins_enforced(retained, monkeypatch, target):
    pins = copy.deepcopy(diagnostic.PINS)
    pins[target]["sha256"] = "0" * 64
    monkeypatch.setattr(diagnostic, "PINS", pins)
    with pytest.raises(ValueError, match="artifact hash"):
        diagnostic.authenticate(*retained[1:])


@pytest.mark.parametrize("kind", ["argv", "exit", "review", "counter", "outer", "environment", "whole"])
def test_capture_integrity_enforced(retained, monkeypatch, kind):
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
        diagnostic.authenticate(*retained[1:])


def test_guard_and_no_ancestor_dispatch(retained, monkeypatch, tmp_path):
    dose, parent, helper = retained[1:]
    allowed = diagnostic.allowed_paths(dose, parent, helper)
    monkeypatch.setattr(helper, "allowed_paths", lambda: allowed)
    audit = dict.fromkeys(helper.COUNTERS, 0)
    with helper.read_only(audit):
        assert classify(retained)["decision"] == "SUPPORTED"
    assert all(type(v) is int and v == 0 for v in audit.values())
    with pytest.raises(helper.ForbiddenOperation), helper.read_only(audit):
        (tmp_path / "forbidden").write_text("no")
    assert audit["forbidden_calls"] == 1


def test_schema_and_cli(monkeypatch, capsys, retained):
    tree = ast.parse(diagnostic.SOURCE.read_bytes())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module.split(".")[0])
    assert set(modules) <= sys.stdlib_module_names
    with pytest.raises(ValueError, match="duplicate JSON"):
        retained[3].decode(b'{"decision":"SUPPORTED","decision":"UNKNOWN"}')
    for decision, status in (("SUPPORTED", 0), ("REJECTED", 0), ("UNKNOWN", 2)):
        monkeypatch.setattr(diagnostic, "check", lambda: {"decision": decision})
        assert diagnostic.main(["--check"]) == status
        assert json.loads(capsys.readouterr().out) == {"decision": decision}


def capture(directory):
    """Capture one compile/test/check sequence; the diagnostic itself never writes."""
    root, python = diagnostic.ROOT, diagnostic.PYTHON
    if (Path.cwd(), os.getuid(), sys.executable) != (root, 1000, python):
        raise RuntimeError("account/workdir/interpreter gate")
    directory = Path(directory).absolute()
    if directory.parent.parent != root / "build" or directory.name != "run":
        raise RuntimeError("fresh build/<attempt>/run required")
    directory.mkdir(exist_ok=False)
    environment = {"HOME": "/home/argustest", "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
                   "LC_ALL": "C.UTF-8", "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

    def encoded(value):
        return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"

    def binding(path):
        data = path.read_bytes()
        return diagnostic.pin(path, len(data), hashlib.sha256(data).hexdigest())

    def save(name, data):
        path = directory / name
        with path.open("xb") as stream:
            stream.write(data)
        return binding(path)

    sources = [binding(diagnostic.SOURCE), binding(diagnostic.TEST)]
    preflight = {"cwd": str(root), "uid": os.getuid(), "python": python, "python_version": sys.version,
                 "environment": environment, "sources": sources,
                 "executable": binding(Path(python).resolve()), "independent_host_review": "REQUIRED",
                 "scope": "12ea retained-only modal down-axis alias; no model/service dispatch"}
    results = []
    failure = None

    def run(label, argv):
        command = " ".join(f"{k}={shlex.quote(v)}" for k, v in environment.items()) + " " + shlex.join(argv)
        command_bytes, env_bytes = (command + "\n").encode(), encoded(preflight)
        files = [save(label + ".command.txt", command_bytes), save(label + ".argv.json", encoded(argv)),
                 save(label + ".environment.json", env_bytes)]
        started, timed_out = time.monotonic(), False
        with (directory / (label + ".stdout")).open("xb") as out, (directory / (label + ".stderr")).open("xb") as err:
            try:
                status = subprocess.run(argv, cwd=root, env=environment, stdout=out, stderr=err,
                                        timeout=90).returncode
            except subprocess.TimeoutExpired:
                status, timed_out = 124, True
        output, error = [(directory / (label + suffix)).read_bytes() for suffix in (".stdout", ".stderr")]
        files += [binding(directory / (label + suffix)) for suffix in (".stdout", ".stderr")]
        files.append(save(label + ".whole-command.log", b"COMMAND\n" + command_bytes + b"ENVIRONMENT\n"
                          + env_bytes + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
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
    captured = save("capture.json", encoded(receipt))
    print(json.dumps({"capture": captured, "success": failure is None, "failure": failure}))
    return 1 if failure else 0


def launch(directory):
    directory = Path(directory).absolute()
    if directory.parent != diagnostic.ROOT / "build":
        raise RuntimeError("fresh direct build child required")
    directory.mkdir(exist_ok=False)
    argv = [diagnostic.PYTHON, "-B", str(diagnostic.TEST), "--run", str(directory / "run")]
    environment = {"HOME": "/home/argustest", "PATH": "/home/argustest/miniconda3/bin:/usr/bin:/bin",
                   "LC_ALL": "C.UTF-8", "PYTHONPATH": str(diagnostic.ROOT), "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

    def encoded(value):
        return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"

    def binding(path):
        data = path.read_bytes()
        return diagnostic.pin(path, len(data), hashlib.sha256(data).hexdigest())

    def save(name, data):
        with (directory / name).open("xb") as stream:
            stream.write(data)
        return binding(directory / name)

    command = " ".join(f"{k}={shlex.quote(v)}" for k, v in environment.items()) + " " + shlex.join(argv)
    identity = {"argv": argv, "command": command, "cwd": str(Path.cwd()), "uid": os.getuid(),
                "environment": environment, "launcher": binding(diagnostic.TEST),
                "invocation_argv": sys.orig_argv}
    identity_bytes = encoded(identity)
    save("launcher.identity.json", identity_bytes)
    timed_out = False
    with (directory / "launcher.stdout").open("xb") as out, (directory / "launcher.stderr").open("xb") as err:
        try:
            status = subprocess.run(argv, cwd=diagnostic.ROOT, env=environment, stdout=out, stderr=err,
                                    timeout=115).returncode
        except subprocess.TimeoutExpired:
            status, timed_out = 124, True
    output, error = [(directory / ("launcher." + suffix)).read_bytes() for suffix in ("stdout", "stderr")]
    save("launcher.whole-command.log", b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output
         + b"\nSTDERR\n" + error + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
    receipt = {"exit_status": status, "timed_out": timed_out,
               "files": [binding(directory / ("launcher." + suffix)) for suffix in
                         ("identity.json", "stdout", "stderr", "whole-command.log")]}
    save("launcher.capture.json", encoded(receipt))
    print(output.decode(), end="")
    print(json.dumps(receipt))
    return status


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--capture", "--run"):
        raise SystemExit("expected --capture build/<fresh-output> or --run build/<attempt>/run")
    raise SystemExit((launch if sys.argv[1] == "--capture" else capture)(sys.argv[2]))
