"""Independent integer slope oracle, hostile controls, and exactly-once byte capture."""

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
    diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_slope_support_classifier_v1 as diagnostic,
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
    assert authentication["review"]["producer_role"] == "reviewer"
    return documents, modules


def classify(retained, documents=None):
    documents = retained[0] if documents is None else documents
    return diagnostic.classify(documents, retained[1][0], retained[1][-2])


def changed(retained):
    documents = list(retained[0])
    documents[0] = dict(documents[0])
    documents[0]["report"] = copy.deepcopy(documents[0]["report"])
    return documents, documents[0]["report"]


def test_independent_integer_support_oracle(retained):
    result = classify(retained)
    assert result["decision"] == "SUPPORTED", result
    assert result["counts"] == {"coefficient_records": 735, "nonzero_slopes": 36, "zero_slopes": 699}
    assert result["retained_counts"] == {
        "component_accounts": 280, "control_coordinate_accounts": 40,
        "factor_proportionality_checks": 280, "nonzero_source_only_movements": 20,
        "reverse_component_checks": 112, "zero_source_accounts": 260,
    }
    census = {False: [], True: []}
    for row in retained[0][0]["report"]["coefficient_matrix"]:
        fields = row["fields"]
        factor = fields["/weight_times_reference_anchor_times_row_difference"]["baseline"]
        product = "1"
        for name in ("norm_weight", "reference_inverse_norm_anchor", "row_difference"):
            product = multiply(product, fields["/factors/" + name]["baseline"])
        assert factor == product
        if row["coordinate"] == 62:
            assert (pair(factor)[0] < 0) == (row["left_id"] == 34319)
        assert len(fields) == 147
        active = row["coordinate"] == 62
        for name, coefficient in fields.items():
            expected = "0"
            parts = name.split("/")
            if active and parts[1] == "components":
                component, field = parts[2:4]
                if component in ("actual_residual_boundary", "q24_to_fp16_conversion"):
                    source = "-1" if component == "actual_residual_boundary" else "1"
                    if field in ("hidden", "source_delta"):
                        expected = source
                    elif field in ("source_only_term", "weighted", "weighted_movement"):
                        expected = multiply(source, factor)
            if active and parts[1] == "mass":
                branch, mass, value = parts[2:]
                if mass in ("absolute", "cancellation_absolute_mass") and value in ("value", "movement"):
                    numerator, denominator = pair(factor)
                    expected = "-2" if branch == "hidden" else exact(-2 * abs(numerator), denominator)
            assert coefficient["slope"] == expected, (row, name)
            identity = {k: row[k] for k in ("table", "branch", "left_id", "right_id", "coordinate")}
            census[expected != "0"].append({**identity, "field": name, "slope": expected})
    assert result["zero_slope_fields"] == census[False]
    assert result["nonzero_slope_fields"] == census[True]
    assert len(census[False]) == 699 and len(census[True]) == 36
    assert [len(r["nonzero_fields"]) for r in result["support_matrix"]] == [18, 0, 18, 0, 0]
    assert all(row["exact"] is True for row in result["support_matrix"])
    for matrix in ("sign_closure_matrix", "mass_closure_matrix", "arithmetic_closure_matrix"):
        assert len(result[matrix]) == 5
        assert all(v == "0" for row in result[matrix] for v in row["residuals"].values())
    assert len(result["orientation_closure_matrix"]) == 2
    for row in result["orientation_closure_matrix"]:
        assert len(row["fields"]) == 147
        assert all(v == "0" for f in row["fields"].values() for v in f["residuals"].values())
    assert result["failure_count"] == 0 and result["failures"] == [] and result["unknown_reasons"] == []
    assert result["retained_70fb_pins"] == diagnostic.PINS
    assert result["parent_source_equivalence_status_preserved"] == "REJECTED"


def test_every_field_support_mutation_rejected(retained):
    # Keep candidate slopes coherent so the support/sign gates, not fit residuals, decide.
    for name in retained[0][0]["report"]["coefficient_matrix"][0]["fields"]:
        documents, report = changed(retained)
        for row in report["coefficient_matrix"]:
            coefficient = row["fields"][name]
            coefficient["slope"] = add(coefficient["slope"], "1")
            coefficient["candidate_slopes"] = dict.fromkeys(coefficient["candidate_slopes"], coefficient["slope"])
        result = classify(retained, documents)
        assert result["decision"] == "REJECTED", name
        assert result["unknown_reasons"] == []
        scopes = {tuple(f[k] for k in ("table", "branch", "left_id", "right_id", "coordinate"))
                  for f in result["failures"] if "coordinate" in f}
        assert len(scopes) == 5, name


@pytest.mark.parametrize("kind", [
    "sign", "mass", "orientation", "arithmetic", "candidate", "residual", "rank", "unique", "count",
])
def test_semantic_mismatch_rejected(retained, kind):
    documents, report = changed(retained)
    fields = report["coefficient_matrix"][0]["fields"]
    coefficient = fields["/components/actual_residual_boundary/hidden"]
    if kind == "sign":
        coefficient["slope"] = "1"
        coefficient["candidate_slopes"] = dict.fromkeys(coefficient["candidate_slopes"], "1")
    elif kind == "mass":
        coefficient = fields["/mass/weighted/cancellation_absolute_mass/value"]
        coefficient["slope"] = multiply("-1", coefficient["slope"])
        coefficient["candidate_slopes"] = dict.fromkeys(coefficient["candidate_slopes"], coefficient["slope"])
    elif kind == "orientation":
        report["coefficient_matrix"][2]["fields"]["/left_weight"]["baseline"] = "0"
    elif kind == "arithmetic":
        fields["/weight_times_reference_anchor_times_row_difference"]["baseline"] = "1"
    elif kind == "candidate":
        coefficient["candidate_slopes"]["scratch"] = "3"
    elif kind == "residual":
        coefficient["candidate_slope_residuals"]["scratch"] = "3"
    elif kind == "rank":
        coefficient["design_rank"] = 1
    elif kind == "unique":
        coefficient["unique"] = False
    else:
        report["counts"]["component_accounts"] = 279
    result = classify(retained, documents)
    assert result["decision"] == "REJECTED", result
    assert result["failure_count"] > 0 and result["unknown_reasons"] == []


@pytest.mark.parametrize("kind", [
    "missing_field", "duplicate_scope", "missing_scope", "noncanonical", "missing_candidate",
    "abscissa", "pointer", "lineage", "prerequisite", "missing_quotient",
])
def test_missing_ambiguous_incompatible_unknown(retained, kind):
    documents, report = changed(retained)
    row = report["coefficient_matrix"][0]
    if kind == "missing_field":
        del row["fields"]["/left_weight"]
    elif kind == "duplicate_scope":
        report["coefficient_matrix"].append(copy.deepcopy(row))
    elif kind == "missing_scope":
        report["coefficient_matrix"].pop()
    elif kind == "noncanonical":
        row["fields"]["/left_weight"]["slope"] = "0/1"
    elif kind == "missing_candidate":
        del row["fields"]["/left_weight"]["candidate_slopes"]["scratch"]
    elif kind == "abscissa":
        row["abscissa"] = "context_constant_zero"
    elif kind == "pointer":
        report["retained_b4_stdout_pin"]["sha256"] = "0" * 64
    elif kind == "lineage":
        report["lineage_separation"]["old_states_substituted_or_propagated"] = True
    elif kind == "missing_quotient":
        del report["retained_quotient_bindings"]["counts"]
    else:
        documents[0]["decision"] = "REJECTED"
    result = classify(retained, documents)
    assert result["decision"] == "UNKNOWN" and result["unknown_reasons"], kind


@pytest.mark.parametrize("target", list(diagnostic.PINS))
def test_root_pin_authentication(target):
    binding = dict(diagnostic.PINS[target], sha256="0" * 64)
    with pytest.raises(ValueError, match="artifact hash"):
        diagnostic.bound_bytes(binding)


@pytest.mark.parametrize("kind", [
    "review", "source", "runtime", "budget", "command", "counter", "predecessor", "stderr", "whole", "outer",
])
def test_capture_integrity(retained, monkeypatch, kind):
    target = "review" if kind == "review" else "capture" if kind in ("source", "budget") else "stdout"
    if kind == "outer":
        target = "outer_capture"
    binding = diagnostic.PINS[target]
    document = json.loads(diagnostic.bound_bytes(binding))
    extra = {}
    if kind == "review":
        document["producer_role"] = "engineer"
    elif kind == "source":
        document["preflight"]["sources"] = []
    elif kind == "runtime":
        document["runtime"]["uid"] = 0
    elif kind == "budget":
        document["preflight"]["model_or_service_calls_authorized"] = 1
    elif kind == "command":
        document["command"] += " --replay"
    elif kind == "counter":
        document["dispatch_and_write_audit"]["decoder_dispatch"] = 1
    elif kind == "predecessor":
        document["artifact_authentication"]["status"] = "UNKNOWN"
    elif kind in ("stderr", "whole"):
        capture = json.loads(diagnostic.bound_bytes(diagnostic.PINS["capture"]))
        member = capture["results"][-1]["files"][4 if kind == "stderr" else 5]
        extra[member["path"]] = b"capture tamper"
    else:
        document["exit_status"] = 1
    original = diagnostic.bound_bytes
    replacements = extra if kind in ("stderr", "whole") else {binding["path"]: json.dumps(document).encode()}

    def tampered(pin):
        return replacements[pin["path"]] if pin["path"] in replacements else original(pin)

    monkeypatch.setattr(diagnostic, "bound_bytes", tampered)
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


def test_uncaptured_stdout_rejected(monkeypatch):
    monkeypatch.setattr(os, "readlink", lambda path: "pipe:[123]")
    with pytest.raises(ValueError, match="capture"):
        diagnostic.capture_paths()


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
        "independent_host_review": "REQUIRED", "scope": "70fb retained-only canonical dose affine slope support",
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
    """Read stored bytes only; never load authentication helpers or rerun semantics."""
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
    captured = json.loads(diagnostic.bound_bytes(receipt["capture"]))
    for value in (receipt, captured):
        diagnostic.same(value["success"], True, "capture success")
        diagnostic.same(value["failure"], None, "capture failure")
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
    diagnostic.same([r["label"] for r in captured["results"]], list(diagnostic.LABELS), "command census")
    result = None
    for row in captured["results"]:
        diagnostic.same(row["exit_status"], 0, "command exit")
        diagnostic.same(row["timed_out"], False, "command timeout")
        diagnostic.same([p["path"] for p in row["files"]],
                        [str(run / (row["label"] + s)) for s in diagnostic.SUFFIXES], "member paths")
        command, argv, env, out, err, all_bytes = [diagnostic.bound_bytes(p) for p in row["files"]]
        diagnostic.same(command, (row["command"] + "\n").encode(), "command bytes")
        diagnostic.same(row["command"], diagnostic.environment_command(ENVIRONMENT, row["argv"]), "command")
        diagnostic.same(json.loads(argv), row["argv"], "argv")
        diagnostic.same(json.loads(env), preflight, "environment bytes")
        diagnostic.same(err, b"", "stderr")
        diagnostic.same(all_bytes, b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
                        + out + b"\nSTDERR\n" + err + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "whole bytes")
        if row["label"] == "branch":
            diagnostic.same(out, b"argus/full-projection\n", "branch")
        if row["label"] == "concurrency":
            diagnostic.same(json.loads(out), [], "concurrency")
        if row["label"] == "check":
            diagnostic.same(row["argv"], [diagnostic.PYTHON, "-B", "-m", diagnostic.MODULE, "--check"],
                            "disclosed invocation")
            result = json.loads(out)
            diagnostic.same(result["byte_exact_capture"]["command_inputs"], row["files"][:3],
                            "current capture input pins")
    suites = ET.fromstring(diagnostic.bound_bytes(captured["pytest_xml"])).findall("testsuite")
    diagnostic.require(bool(suites) and sum(int(s.attrib["tests"]) for s in suites) > 0, "tests absent")
    diagnostic.require(not any(int(s.attrib[k]) for s in suites for k in ("errors", "failures", "skipped")),
                       "test failure/error/skip")
    diagnostic.same((result["artifact_authentication"]["status"], result["decision"], result["report"]["decision"]),
                    ("AUTHENTICATED", "SUPPORTED", "SUPPORTED"), "result status")
    diagnostic.same(result["artifact_authentication"]["input_pins"], diagnostic.PINS, "70fb input pins")
    for p in diagnostic.PINS.values():
        diagnostic.bound_bytes(p)
    diagnostic.same(result["source_test_pins"], sources, "result sources")
    diagnostic.same(result["command"], diagnostic.COMMAND, "disclosed command")
    diagnostic.same(result["runtime"]["executable_pin"], preflight["executable"], "result runtime")
    diagnostic.require(bool(result["dispatch_and_write_audit"]) and all(
        type(v) is int and v == 0 for v in result["dispatch_and_write_audit"].values()), "forbidden counters")
    diagnostic.same(result["byte_exact_capture"]["status"], "BYTE_EXACT_STDOUT_STDERR_FILES_OPEN",
                    "stdout/stderr capture contract")
    report = result["report"]
    diagnostic.same(report["counts"], diagnostic.COUNTS, "counts")
    diagnostic.same(len(report["zero_slope_fields"]), 699, "zero census")
    diagnostic.same(len(report["nonzero_slope_fields"]), 36, "nonzero census")
    diagnostic.same(report["failure_count"], 0, "failures")
    diagnostic.require(all(r["exact"] is True for r in report["support_matrix"]), "support mismatch")
    for matrix in ("sign_closure_matrix", "mass_closure_matrix", "arithmetic_closure_matrix"):
        diagnostic.require(all(v == "0" for r in report[matrix] for v in r["residuals"].values()), matrix)
    diagnostic.require(all(v == "0" for r in report["orientation_closure_matrix"]
                           for f in r["fields"].values() for v in f["residuals"].values()), "orientation")
    print(json.dumps({"stored_bytes_validated_without_semantic_recomputation": True,
                      "capture": receipt["capture"], "stdout": binding(run / "check.stdout"),
                      "tests": sum(int(s.attrib["tests"]) for s in suites),
                      "decision": result["decision"], "counts": report["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--launch", "--run", "--validate"):
        raise SystemExit({"--launch": launch, "--run": capture, "--validate": validate_capture}[sys.argv[1]](
            Path(sys.argv[2])))
    raise SystemExit("use --launch, --run or --validate <fresh build attempt>")
