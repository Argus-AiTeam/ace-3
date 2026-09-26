"""Transport oracles and synthetic representative wiring, not scientific evidence."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import sys
import subprocess
import tempfile
import traceback
from types import SimpleNamespace

import pytest

from ace3.model.candidates import diagnostic_capture_v1 as capture


@pytest.fixture
def legacy():
    from ace3.model.candidates import (
        diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_root_margin_classifier_v1,
    )
    return diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_root_margin_classifier_v1


@pytest.fixture
def diagnostic():
    from ace3.model.candidates import (
        diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_sign_profile_quotient_classifier_v1,
    )
    return diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_canonical_dose_affine_sign_profile_quotient_classifier_v1


def preflight():
    return {"cwd": str(Path.cwd()), "environment": {"PATH": os.environ["PATH"], "LC_ALL": "C.UTF-8"},
            "sources": [capture.binding(Path(__file__))], "independent_host_review": "REQUIRED",
            "model_or_service_calls_authorized": 0}


def assert_pins(pins):
    for pin in pins:
        data = Path(pin["path"]).read_bytes()
        assert pin["bytes"] == len(data)
        assert pin["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("status,stderr", [(0, b""), (7, b"\xfe\x00stderr\r\n"), (0, b"warning")])
def test_byte_exact_streams_and_legacy_frame(tmp_path, status, stderr):
    stdout = bytes(range(256)) * 4096
    code = f"import os; os.write(1,bytes(range(256))*4096); os.write(2,{stderr!r}); raise SystemExit({status})"
    argv, before, results = [sys.executable, "-B", "-c", code], preflight(), []
    if status or stderr:
        with pytest.raises(RuntimeError, match="complete bytes retained"):
            capture.run_command(tmp_path, "check", argv, before, results)
    else:
        assert capture.run_command(tmp_path, "check", argv, before, results) == stdout
    result, = results
    command = capture.environment_command(before["environment"], argv).encode() + b"\n"
    assert result["exit_status"] == status and result["timed_out"] is False
    assert (tmp_path / "check.stdout").read_bytes() == stdout
    assert (tmp_path / "check.stderr").read_bytes() == stderr
    assert (tmp_path / "check.whole-command.log").read_bytes() == (
        b"COMMAND\n" + command + b"ENVIRONMENT\n" + capture.encoded(before)
        + b"\nSTDOUT\n" + stdout + b"\nSTDERR\n" + stderr
        + f"\nEXIT_STATUS={status}\nTIMED_OUT=False\n".encode()
    )
    assert json.loads((tmp_path / "check.argv.json").read_bytes()) == argv
    assert json.loads((tmp_path / "check.environment.json").read_bytes()) == before
    assert_pins(result["files"])


def test_preflight_cwd_records_and_executes_captured_command_path(tmp_path, monkeypatch):
    workdir, directory = tmp_path / "command-workdir", tmp_path / "capture"
    workdir.mkdir()
    directory.mkdir()
    (workdir / "command.py").write_text(
        "import os\nos.write(1, os.fsencode(os.getcwd()) + b'\\x00\\xff')\n")
    with monkeypatch.context() as context:
        context.chdir(workdir)
        before = preflight()
    monkeypatch.chdir(tmp_path)
    assert before["cwd"] == str(workdir)
    argv, results = [sys.executable, "-B", "command.py"], []
    output = os.fsencode(workdir) + b"\x00\xff"
    assert capture.run_command(directory, "check", argv, before, results) == output
    result, = results
    assert result["exit_status"] == 0 and result["timed_out"] is False
    assert "launch_error" not in result
    assert (directory / "check.environment.json").read_bytes() == capture.encoded(before)
    command = (capture.environment_command(before["environment"], argv) + "\n").encode()
    assert (directory / "check.whole-command.log").read_bytes() == (
        b"COMMAND\n" + command + b"ENVIRONMENT\n" + capture.encoded(before)
        + b"\nSTDOUT\n" + output + b"\nSTDERR\n\nEXIT_STATUS=0\nTIMED_OUT=False\n")
    assert_pins(result["files"])


@pytest.mark.parametrize("defect", [
    "missing", None, "", 0, True, [], {}, "invalid\0cwd",
])
def test_preflight_cwd_rejects_malformed_records_before_dispatch(tmp_path, monkeypatch, defect):
    before, results = preflight(), []
    if defect == "missing":
        del before["cwd"]
    else:
        before["cwd"] = defect

    def forbidden(*args):
        pytest.fail("dispatch with missing/invalid cwd")

    monkeypatch.setattr(capture, "_execute", forbidden)
    with pytest.raises(ValueError, match="preflight requires a nonempty cwd string without NUL"):
        capture.run_command(tmp_path, "check", [sys.executable, "-B", "-c", "pass"], before, results)
    assert results == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind,error_type", [
    ("missing", "FileNotFoundError"), ("file", "NotADirectoryError"),
])
def test_preflight_cwd_filesystem_failure_is_sealed(tmp_path, kind, error_type):
    cwd = tmp_path / "invalid-cwd"
    if kind == "file":
        cwd.write_bytes(b"not a directory")
    before, results = preflight(), []
    before["cwd"] = str(cwd)
    with pytest.raises(RuntimeError, match="complete bytes retained"):
        capture.run_command(tmp_path, "check", [sys.executable, "-B", "-c", "print('unexpected')"],
                            before, results)
    result, = results
    assert result["exit_status"] == 127 and result["timed_out"] is False
    assert result["launch_error"]["type"] == error_type
    assert result["launch_error"]["message"]
    assert (tmp_path / "check.stdout").read_bytes() == b""
    assert (tmp_path / "check.stderr").read_bytes() == b""
    assert (tmp_path / "check.environment.json").read_bytes() == capture.encoded(before)
    assert (tmp_path / "check.whole-command.log").read_bytes().endswith(
        b"\nSTDOUT\n\nSTDERR\n\nEXIT_STATUS=127\nTIMED_OUT=False\n")
    assert_pins(result["files"])


def test_timeout_retains_partial_bytes_and_stops_descendant(tmp_path):
    descendant = "import time,os; time.sleep(2); os.write(1,b'late bytes')"
    code = ("import os,subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
            "os.write(1,b'partial\\x00'); os.write(2,b'failure\\xff'); time.sleep(60)")
    results = []
    with pytest.raises(RuntimeError, match="complete bytes retained"):
        capture.run_command(tmp_path, "check", [sys.executable, "-B", "-c", code],
                            preflight(), results, timeout=1)
    assert results[0]["exit_status"] == 124 and results[0]["timed_out"] is True
    assert (tmp_path / "check.stdout").read_bytes() == b"partial\x00"
    assert (tmp_path / "check.stderr").read_bytes() == b"failure\xff"
    # Waiting past the descendant's write deadline tests the seal, not a stale receipt.
    import time
    time.sleep(1.2)
    assert_pins(results[0]["files"])


def test_launch_error_is_sealed(tmp_path):
    results = []
    with pytest.raises(RuntimeError, match="complete bytes retained"):
        capture.run_command(tmp_path, "check", [str(tmp_path / "missing-executable")],
                            preflight(), results)
    result, = results
    assert result["exit_status"] == 127
    assert result["launch_error"]["type"] == "FileNotFoundError"
    assert (tmp_path / "check.stdout").read_bytes() == b""
    assert (tmp_path / "check.stderr").read_bytes() == b""
    assert_pins(result["files"])


@pytest.mark.parametrize("suffix", capture.SUFFIXES)
def test_any_existing_member_prevents_dispatch_and_overwrite(tmp_path, monkeypatch, suffix):
    existing = tmp_path / ("check" + suffix)
    existing.write_bytes(b"historical failure")

    def forbidden(*args):
        pytest.fail("dispatch before exclusive member reservation")

    monkeypatch.setattr(capture, "_execute", forbidden)
    with pytest.raises(FileExistsError):
        capture.run_command(tmp_path, "check", ["unused"], preflight(), [])
    assert existing.read_bytes() == b"historical failure"


def test_scope_gate_and_release(tmp_path):
    (tmp_path / "build").mkdir()
    with capture.same_scope(tmp_path, "representative"):
        with pytest.raises(RuntimeError, match="same-scope concurrency"):
            with capture.same_scope(tmp_path, "representative"):
                pytest.fail("overlapping scope admitted")
        with capture.same_scope(tmp_path, "independent"):
            pass
    with capture.same_scope(tmp_path, "representative"):
        pass


@pytest.mark.parametrize("status,stderr,timeout", [(0, b"", False), (5, b"bad", False), (0, b"warning", False), (124, b"partial", True)])
def test_outer_full_bytes_and_failure_status(tmp_path, status, stderr, timeout):
    code = ("import os,time; os.write(1,b'outer\\xff\\x00'); "
            f"os.write(2,{stderr!r}); " + ("time.sleep(60)" if timeout else f"raise SystemExit({status})"))
    identity = {**preflight(), "argv": [sys.executable, "-B", "-c", code], "uid": os.getuid()}
    identity["command"] = capture.environment_command(identity["environment"], identity["argv"])
    output, outer = capture.seal_launcher(tmp_path, identity, timeout=1)
    assert output == b"outer\xff\x00"
    assert (outer["exit_status"], outer["timed_out"]) == (status, timeout)
    assert outer["success"] is (status == 0 and not stderr and not timeout)
    identity_bytes = (tmp_path / "launcher.identity.json").read_bytes()
    assert json.loads(identity_bytes)["command"] == identity["command"]
    assert (tmp_path / "launcher.whole-command.log").read_bytes() == (
        b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output + b"\nSTDERR\n" + stderr
        + f"\nEXIT_STATUS={status}\nTIMED_OUT={timeout}\n".encode()
    )
    assert json.loads((tmp_path / "launcher.capture.json").read_bytes()) == outer
    assert_pins(outer["files"])
    with pytest.raises(FileExistsError):
        capture.seal_launcher(tmp_path, identity)
    assert_pins(outer["files"])


@pytest.fixture
def representative(diagnostic):
    spec = importlib.util.spec_from_file_location("_representative_capture_test", diagnostic.TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", ["SUPPORTED", "UNKNOWN", "REJECTED", "counter", "branch", "concurrency"])
def test_representative_capture_legacy_identity_without_producer_replay(
        tmp_path, monkeypatch, capsys, representative, outcome, legacy, diagnostic):
    """Synthetic process adapter: real launch/run/main/sealing and stored-byte validator."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build").mkdir()
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    helper = SimpleNamespace(COUNTERS=("prefix_dispatch", "state_write", "kv_write"),
                             decode=lambda data, **kwargs: json.loads(data))
    monkeypatch.setattr(diagnostic, "load_helpers", lambda: (helper,))
    monkeypatch.setattr(diagnostic, "verify_capture", legacy.verify_capture)
    attempted = []

    def execute(argv, cwd, environment, out, err, timeout):
        attempted.append(argv)
        if "--run" in argv:
            text = io.StringIO()
            with redirect_stdout(text):
                status = representative.capture(Path(argv[-1]))
            out.write(text.getvalue().encode())
        elif argv[:2] == ["git", "branch"]:
            out.write(b"wrong\n" if outcome == "branch" else b"argus/full-projection\n")
            status = 0
        elif argv[:2] == ["git", "check-ignore"]:
            out.write((argv[-1] + "\n").encode())
            status = 0
        elif "-m" in argv and argv[argv.index("-m") + 1] == "pytest":
            Path(argv[argv.index("--junitxml") + 1]).write_bytes(
                b'<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0"/></testsuites>')
            out.write(b"synthetic test-command output\n")
            status = 0
        elif "--check" in argv:
            run = Path(out.name).parent
            result = {
                "decision": outcome if outcome in ("UNKNOWN", "REJECTED") else "SUPPORTED",
                "artifact_authentication": {"status": "AUTHENTICATED"},
                "source_test_pins": [capture.binding(diagnostic.SOURCE), capture.binding(diagnostic.TEST)],
                "byte_exact_capture": {"command_inputs": [
                    capture.binding(run / ("check" + suffix)) for suffix in capture.SUFFIXES[:3]]},
                "dispatch_and_write_audit": dict.fromkeys(helper.COUNTERS, 0),
                "report": {"counts": diagnostic.COUNTS, "failure_count": 0},
                "fixture_only": True,
            }
            if outcome == "counter":
                result["dispatch_and_write_audit"]["prefix_dispatch"] = 1
            monkeypatch.setattr(diagnostic, "check", lambda: result)
            text = io.StringIO()
            with redirect_stdout(text):
                status = diagnostic.main(["--check"])
            out.write(text.getvalue().encode())
        else:
            out.write(b'["same scope"]\n' if outcome == "concurrency" else b"[]\n")
            status = 0
        out.flush()
        err.flush()
        return status, False, None

    monkeypatch.setattr(capture, "_execute", execute)
    directory = tmp_path / "build/synthetic-transport"
    status = representative.launch(directory)
    capsys.readouterr()
    assert status == (0 if outcome == "SUPPORTED" else 1)
    receipt = json.loads((directory / "run/capture.json").read_bytes())
    assert receipt["success"] is (outcome == "SUPPORTED")
    assert receipt["capture_implementation_after"] == receipt["preflight"]["capture_implementation"]
    assert receipt["sources_after"] == receipt["preflight"]["sources"]
    for result in receipt["results"]:
        assert_pins(result["files"])
    outer = json.loads((directory / "launcher.capture.json").read_bytes())
    assert_pins(outer["files"])
    if outcome == "SUPPORTED":
        stdout = (directory / "run/check.stdout").read_bytes()
        pre, inputs = legacy.verify_capture(directory, receipt, outer, stdout,
                                            receipt["preflight"]["sources"], diagnostic.MODULE, helper)
        report = json.loads(stdout)
        assert report["byte_exact_capture"]["command_inputs"] == inputs
        assert all(type(v) is int and v == 0 for v in report["dispatch_and_write_audit"].values())
        assert pre["independent_host_review"] == "REQUIRED"
        assert pre["model_or_service_calls_authorized"] == 0
    elif outcome in ("branch", "concurrency"):
        assert not any("--check" in argv for argv in attempted)
    else:
        assert (directory / "run/check.stdout").stat().st_size > 0
        assert receipt["failure"] is not None
    with pytest.raises(FileExistsError):
        representative.launch(directory)


def test_representative_account_gate_before_creation(tmp_path, monkeypatch, representative):
    monkeypatch.setattr(representative.os, "getuid", lambda: -1)
    with pytest.raises(RuntimeError, match="account/workdir/interpreter"):
        representative.capture(tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


@pytest.fixture
def gate_attempt(record_property):
    # Retain these synthetic attempts for review, including intentionally damaged copies.
    parent = Path(tempfile.mkdtemp(prefix="stdout-capture-synthetic-", dir=capture.ROOT / "build"))
    attempt = parent / "attempt"
    record_property("capture_attempt", str(attempt))
    return attempt


def gate_identity(attempt, code):
    return {**preflight(), "uid": os.getuid(), "scope": str(attempt.parent),
            "argv": [sys.executable, "-B", "-c", code], "synthetic_only": True}


def test_enforced_cli_large_stdout_is_not_a_summary(gate_attempt):
    stdout = bytes(range(256)) * 32768 + b"\nSTDERR\nEXIT_STATUS=17\n"
    identity = gate_identity(
        gate_attempt, "import os; os.write(1, bytes(range(256))*32768"
        "+b'\\nSTDERR\\nEXIT_STATUS=17\\n')")
    request = gate_attempt.parent / "request.json"
    request.write_bytes(capture.encoded(identity))
    argv = [sys.executable, "-B", "-m", capture.__name__, "--attempt", str(gate_attempt),
            "--identity", str(request), "--timeout", "10"]
    process = subprocess.run(argv, cwd=capture.ROOT, env=identity["environment"],
                             capture_output=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert process.stderr == b""
    summary = json.loads(process.stdout)
    assert summary["success"] is True and summary["independent_host_review"] == "REQUIRED"
    assert len(process.stdout) < 2048 < len(stdout)
    expected_identity = {**identity, "attempt": str(gate_attempt), "timeout_seconds": 10.0,
                         "command": capture.environment_command(
                             dict(sorted(identity["environment"].items())), identity["argv"]),
                         "capture_implementation": capture.implementation_pins()}
    outer = capture.verify_launcher(gate_attempt, expected_identity, summary["capture"])
    assert (gate_attempt / "launcher.stdout").read_bytes() == stdout
    assert (gate_attempt / "launcher.stderr").read_bytes() == b""
    whole = (b"IDENTITY\n" + capture.encoded(expected_identity) + b"STDOUT\n" + stdout
             + b"\nSTDERR\n\nEXIT_STATUS=0\nTIMED_OUT=False\n")
    assert (gate_attempt / "launcher.whole-command.log").read_bytes() == whole
    assert outer["files"][1]["sha256"] == hashlib.sha256(stdout).hexdigest()
    assert outer["files"][3]["sha256"] == hashlib.sha256(whole).hexdigest()
    assert outer["files"][1]["sha256"] != outer["files"][3]["sha256"]
    assert_pins(outer["files"] + [summary["capture"]])
    with pytest.raises(FileExistsError):
        capture.launch_diagnostic(gate_attempt, identity, timeout=10)
    assert_pins(outer["files"] + [summary["capture"]])


@pytest.mark.parametrize("kind,status,stderr", [
    ("exit", 7, b"failure\xff\r\n"), ("stderr", 0, b"warning\x00"),
    ("timeout", 124, b"partial\xff"), ("launch", 127, b""),
])
def test_enforced_failures_are_retained_not_admitted(gate_attempt, kind, status, stderr):
    stdout = b"" if kind == "launch" else b"retained\x00\xff"
    code = (f"import os,time; os.write(1,{stdout!r}); os.write(2,{stderr!r}); "
            + ("time.sleep(60)" if kind == "timeout" else f"raise SystemExit({status})"))
    identity = gate_identity(gate_attempt, code)
    if kind == "launch":
        identity["argv"] = [str(gate_attempt.parent / "missing-executable")]
    with pytest.raises(RuntimeError, match="launcher failed; complete bytes retained"):
        capture.launch_diagnostic(gate_attempt, identity, timeout=1)
    expected_identity = {**identity, "attempt": str(gate_attempt), "timeout_seconds": 1,
                         "command": capture.environment_command(
                             dict(sorted(identity["environment"].items())), identity["argv"]),
                         "capture_implementation": capture.implementation_pins()}
    receipt_path = gate_attempt / "launcher.capture.json"
    receipt_pin = capture.binding(receipt_path)
    outer = capture.verify_launcher(
        gate_attempt, expected_identity, receipt_pin, require_success=False)
    assert outer["success"] is False and outer["exit_status"] == status
    assert outer["timed_out"] is (kind == "timeout")
    assert (gate_attempt / "launcher.stdout").read_bytes() == stdout
    assert (gate_attempt / "launcher.stderr").read_bytes() == stderr
    if kind == "launch":
        assert outer["launch_error"]["type"] == "FileNotFoundError"
    assert_pins(outer["files"] + [receipt_pin])
    with pytest.raises(RuntimeError, match="launcher failed"):
        capture.verify_launcher(gate_attempt, expected_identity, receipt_pin)


def test_launcher_cannot_skip_whole_capture_validation(gate_attempt, monkeypatch):
    seal = capture.seal_launcher

    def summary_only(directory, identity, timeout):
        output, outer = seal(directory, identity, timeout)
        outer["files"].pop()
        (directory / "launcher.capture.json").write_bytes(capture.encoded(outer))
        return output, outer

    monkeypatch.setattr(capture, "seal_launcher", summary_only)
    with pytest.raises(RuntimeError, match="whole capture bindings required"):
        capture.launch_diagnostic(gate_attempt, gate_identity(gate_attempt, "print('synthetic')"))


@pytest.mark.parametrize("mutation", [
    "missing-whole-binding", "missing-whole-file", "stdout-hash-as-whole",
    "stdout-binding-as-whole", "rebound-whole-is-stdout", "rebound-stream",
    "missing-stderr-binding", "rebound-identity", "rebound-status", "receipt-hash",
])
def test_enforced_gate_rejects_summary_splices_and_conflation(gate_attempt, mutation):
    result = capture.launch_diagnostic(
        gate_attempt, gate_identity(gate_attempt, "import os; os.write(1,b'full\\x00\\xff')"))
    receipt_path = gate_attempt / "launcher.capture.json"
    outer = json.loads(receipt_path.read_bytes())
    if mutation == "missing-whole-binding":
        outer["files"].pop()
    elif mutation == "missing-whole-file":
        (gate_attempt / "launcher.whole-command.log").unlink()
    elif mutation == "stdout-hash-as-whole":
        outer["files"][3]["sha256"] = outer["files"][1]["sha256"]
    elif mutation == "stdout-binding-as-whole":
        outer["files"][3] = outer["files"][1]
    elif mutation == "rebound-whole-is-stdout":
        path = gate_attempt / "launcher.whole-command.log"
        path.write_bytes((gate_attempt / "launcher.stdout").read_bytes())
        outer["files"][3] = capture.binding(path)
    elif mutation == "rebound-stream":
        path = gate_attempt / "launcher.stdout"
        path.write_bytes(b"summary")
        outer["files"][1] = capture.binding(path)
    elif mutation == "missing-stderr-binding":
        outer["files"].pop(2)
    elif mutation == "rebound-identity":
        path = gate_attempt / "launcher.identity.json"
        path.write_bytes(capture.encoded({**result["identity"], "environment": {}}))
        outer["files"][0] = capture.binding(path)
    elif mutation == "rebound-status":
        outer.update(exit_status=9, success=False)
    else:
        outer["unbound_summary"] = True
    receipt_path.write_bytes(capture.encoded(outer))
    pin = result["capture"] if mutation == "receipt-hash" else capture.binding(receipt_path)
    with pytest.raises(RuntimeError, match="capture"):
        capture.verify_launcher(gate_attempt, result["identity"], pin, require_success=False)


def test_empty_streams_may_share_a_hash_but_not_a_binding(gate_attempt):
    result = capture.launch_diagnostic(gate_attempt, gate_identity(gate_attempt, "pass"))
    outer = capture.verify_launcher(gate_attempt, result["identity"], result["capture"])
    assert outer["files"][1]["sha256"] == outer["files"][2]["sha256"]
    assert outer["files"][1]["path"] != outer["files"][2]["path"]


@pytest.mark.parametrize("gate", ["uid", "cwd", "review", "budget", "timeout", "command", "scope"])
def test_new_launcher_preflight_rejects_before_attempt_or_dispatch(gate_attempt, monkeypatch, gate):
    identity = gate_identity(gate_attempt, "raise AssertionError('must not execute')")
    timeout = 1
    if gate == "uid":
        identity["uid"] = -1
    elif gate == "cwd":
        identity["cwd"] = str(gate_attempt.parent)
    elif gate == "review":
        identity["independent_host_review"] = "SKIPPED"
    elif gate == "budget":
        identity["model_or_service_calls_authorized"] = 1
    elif gate == "timeout":
        timeout = float("nan")
    elif gate == "command":
        identity["command"] = "a different command"
    else:
        identity["scope"] = ""
    monkeypatch.setattr(capture, "_execute", lambda *args: pytest.fail("dispatch past failed gate"))
    with pytest.raises((RuntimeError, ValueError)):
        capture.launch_diagnostic(gate_attempt, identity, timeout)
    assert not gate_attempt.exists()


@pytest.mark.parametrize("environment", [
    {}, {"Z": "space = value", "A": ""}, {"A": "", "Z": "space = value"},
])
def test_retained_authentication_command_order(environment):
    argv = ["/synthetic/python", "space argument", "K=an argv element"]
    expected = dict(reversed(list(environment.items())))
    command = capture.environment_command(environment, argv)
    assert capture.verify_command(command, argv, environment, argv, expected) == (
        capture.environment_command(dict(sorted(environment.items())), argv))


@pytest.mark.parametrize("mutation", [
    "duplicate", "value", "argv", "missing-key", "extra-key", "quoting",
    "malformed", "nul", "invalid-env", "empty-argv",
])
def test_retained_authentication_command_rejections(mutation):
    argv, environment = ["/synthetic/python", "arg"], {"A": "1", "B": "2"}
    command = capture.environment_command(environment, argv)
    retained_argv, retained_env = argv[:], environment.copy()
    if mutation == "duplicate":
        command = "A=1 A=1 " + shlex.join(argv)
    elif mutation == "value":
        retained_env["A"] = "changed"
        command = capture.environment_command(retained_env, argv)
    elif mutation == "argv":
        retained_argv[-1] = "changed"
        command = capture.environment_command(environment, retained_argv)
    elif mutation == "missing-key":
        retained_env.pop("B")
        command = capture.environment_command(retained_env, argv)
    elif mutation == "extra-key":
        retained_env["C"] = "3"
        command = capture.environment_command(retained_env, argv)
    elif mutation == "quoting":
        command = command.replace("A=1", "A='1'")
    elif mutation == "malformed":
        command += "'"
    elif mutation == "nul":
        command += "\0"
    elif mutation == "invalid-env":
        retained_env["A"] = 1
    else:
        retained_argv = []
    with pytest.raises(RuntimeError):
        capture.verify_command(command, retained_argv, retained_env, argv, environment)


@pytest.mark.parametrize("data", [
    b'{"environment":{"A":"1","A":"1"}}', b'{"argv":[],"argv":[]}',
    b'{"historical_generation":{},"historical_generation":{}}',
    b'{"value":NaN}', b'{', b'\xff',
])
def test_retained_authentication_json_rejections(data):
    with pytest.raises(RuntimeError):
        capture.decode_retained(data)


@pytest.fixture
def retained_authentication_reservation(tmp_path, monkeypatch):
    (tmp_path / "build").mkdir()

    def ignored(argv, **kwargs):
        assert argv[:3] == ["git", "check-ignore", "--quiet"]
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(capture.subprocess, "run", ignored)
    reservation = capture.reserve_authentication_attempt(tmp_path / "build/fresh", root=tmp_path)
    return tmp_path, reservation


@pytest.mark.parametrize("empty", [True, False])
def test_retained_authentication_write_positive(retained_authentication_reservation, empty):
    root, reservation = retained_authentication_reservation
    writes = [] if empty else [reservation["path"], reservation["path"] + "/result.json"]
    result = capture.verify_write_audit(
        {"attempt": reservation["path"], "writes": writes}, reservation, root=root)
    assert result["declared_write_count"] == len(writes)
    assert result["symlink_derived_writes"] == []
    with pytest.raises(FileExistsError):
        capture.reserve_authentication_attempt(reservation["path"], root=root)


@pytest.mark.parametrize("kind", ["directory", "file", "chain", "relative-target"])
def test_retained_authentication_write_symlink_positive(retained_authentication_reservation, kind):
    root, reservation = retained_authentication_reservation
    attempt = Path(reservation["path"])
    target, link = attempt / "target", attempt / "link"
    if kind == "file":
        target.write_bytes(b"retained temporary write\n")
    else:
        target.mkdir()
    link.symlink_to("target" if kind == "relative-target" else target,
                    target_is_directory=kind != "file")
    if kind == "chain":
        chained = attempt / "chained"
        chained.symlink_to(link, target_is_directory=True)
        link = chained
    path = link if kind == "file" else link / "result"
    path.write_bytes(b"confined\n")
    result = capture.verify_write_audit(
        {"attempt": str(attempt), "writes": [str(link), str(path)]}, reservation, root=root)
    assert result["declared_write_count"] == 2
    assert result["symlink_derived_writes"] == [
        {"path": str(item), "resolved": str(item.resolve())} for item in (link, path)]
    with pytest.raises(RuntimeError, match="noncanonical"):
        capture._canonical_path(str(link))


@pytest.mark.parametrize("mutation", [
    "missing", "missing-writes", "null-writes", "string-writes", "outside", "prefix-sibling",
    "relative", "dot", "dotdot", "double-slash", "trailing-slash", "nul", "not-string",
    "wrong-attempt", "wrong-inode", "unignored", "leading-double-slash",
    "symlink-outside", "symlink-sibling", "symlink-outside-in", "symlink-dotdot",
])
def test_retained_authentication_write_rejections(retained_authentication_reservation, monkeypatch, mutation):
    root, reservation = retained_authentication_reservation
    reservation = reservation.copy()
    attempt = reservation["path"]
    audit = {"attempt": attempt, "writes": [attempt + "/result.json"]}
    link = None
    if mutation == "missing":
        audit = None
    elif mutation == "missing-writes":
        audit.pop("writes")
    elif mutation == "null-writes":
        audit["writes"] = None
    elif mutation == "string-writes":
        audit["writes"] = attempt
    elif mutation == "wrong-attempt":
        audit["attempt"] = str(root)
    elif mutation == "wrong-inode":
        reservation["inode"] += 1
    elif mutation == "unignored":
        monkeypatch.setattr(capture.subprocess, "run",
                            lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, b"", b""))
    else:
        paths = {"outside": str(root / "outside"), "prefix-sibling": attempt + "-other/result",
                 "relative": "result.json", "dot": attempt + "/./result",
                 "dotdot": attempt + "/child/../result", "double-slash": attempt + "//result",
                 "trailing-slash": attempt + "/", "nul": attempt + "/\0", "not-string": 42,
                 "leading-double-slash": "/" + attempt + "/result"}
        if mutation.startswith("symlink-"):
            target = (Path(attempt + "-other") if mutation == "symlink-sibling"
                      else root / "outside")
            target.mkdir()
            link = Path(attempt) / "link"
            if mutation in ("symlink-outside-in", "symlink-dotdot"):
                target = Path(attempt) / "target"
                target.mkdir()
            if mutation == "symlink-outside-in":
                link = root / "outside/link"
            link.symlink_to(target, target_is_directory=True)
            paths[mutation] = str(link) + ("/../result" if mutation == "symlink-dotdot" else "/result")
        audit["writes"] = [paths[mutation]]
    try:
        with pytest.raises(RuntimeError):
            capture.verify_write_audit(audit, reservation, root=root)
    finally:
        if link is not None:
            link.unlink()


@pytest.mark.parametrize("status", ["PASS", "FAIL", "UNKNOWN"])
@pytest.mark.parametrize("mutation", ["none", "outside-write", "source-drift", "command"])
def test_retained_authentication_terminal_sealing(retained_authentication_reservation, status, mutation):
    root, reservation = retained_authentication_reservation
    attempt = Path(reservation["path"])
    source = attempt / "source.py"
    source.write_bytes(b"retained validation source\n")
    argv, environment = ["/synthetic/python", "validation.py"], {"TMPDIR": str(attempt)}
    command = capture.environment_command(environment, argv) + "\n"
    if mutation == "command":
        command = "TMPDIR=changed /synthetic/python validation.py\n"
    for name, data in {
            "command.txt": command.encode(), "argv.json": capture.encoded(argv),
            "environment.json": capture.encoded(environment),
            "source-roles.json": capture.encoded({"current_validation": [capture.binding(source)]}),
            "stdout": b"\xffstdout\r\n", "stderr": b"failure\n" if status == "FAIL" else b"",
    }.items():
        capture.save(attempt, "validation." + name, data)
    audit = {"attempt": str(attempt), "writes": [str(path) for path in attempt.iterdir()],
             "scientific_invocations": 0, "producer_invocations": 0, "service_invocations": 0}
    if mutation == "outside-write":
        audit["writes"].append(str(root / "not-an-attempt-member"))
    elif mutation == "source-drift":
        source.write_bytes(b"changed source\n")
    result = {"status": status}
    if status == "UNKNOWN":
        result["unavailable_binding"] = "missing retained original: /synthetic/original"
    sealed = capture.seal_authentication_validation(attempt, result, audit, reservation, root=root)
    expected_status = status if mutation == "none" else "FAIL"
    assert sealed["status"] == expected_status
    assert sealed["exit_status"] == (1 if expected_status == "FAIL" else 0)
    assert sealed["timed_out"] is False
    assert all(sealed[key] == 0 for key in (
        "scientific_invocations", "producer_invocations", "service_invocations"))
    assert capture.decode_retained((attempt / "validation.results.json").read_bytes()) == sealed
    receipt = capture.decode_retained((attempt / "validation.capture.json").read_bytes())
    assert receipt["status"] == expected_status
    assert len(receipt["files"]) == 9
    assert all(capture.binding(pin["path"]) == pin for pin in receipt["files"])
    frame = b""
    for label, name in (("COMMAND", "command.txt"), ("ARGV", "argv.json"),
                        ("ENVIRONMENT", "environment.json"), ("SOURCE_ROLES", "source-roles.json"),
                        ("STDOUT", "stdout")):
        frame += label.encode() + b"\n" + (attempt / ("validation." + name)).read_bytes()
    frame += b"\nSTDERR\n" + (attempt / "validation.stderr").read_bytes()
    frame += f"\nEXIT_STATUS={sealed['exit_status']}\nTIMED_OUT=False\n".encode()
    assert (attempt / "validation.whole-command.log").read_bytes() == frame
    if mutation != "none":
        assert sealed["sealing_failure"]["message"]
    with pytest.raises(FileExistsError):
        capture.seal_authentication_validation(attempt, result, audit, reservation, root=root)


@pytest.mark.skipif("ACE3_PREFLIGHT_ATTEMPT" not in os.environ, reason="explicit confined validation")
def test_retained_authentication_tempfile_confinement(tmp_path):
    attempt = Path(os.environ["ACE3_PREFLIGHT_ATTEMPT"])
    assert Path(tempfile.gettempdir()) == attempt / "tmp"
    assert tmp_path.is_relative_to(attempt / "pytest-tmp")
    with tempfile.TemporaryDirectory() as directory:
        assert Path(directory).is_relative_to(attempt / "tmp")
        with tempfile.NamedTemporaryFile(dir=directory) as stream:
            stream.write(b"confined temporary validation bytes\n")
            stream.flush()
            assert Path(stream.name).read_bytes() == b"confined temporary validation bytes\n"


@pytest.fixture
def retained_authentication_roles(tmp_path):
    live, old = tmp_path / "source.py", tmp_path / "historical.py"
    live.write_bytes(b"current diagnostic\n")
    old.write_bytes(b"historical generation\n")
    old_pin, live_pin = capture.binding(old), capture.binding(live)
    return {
        "historical_generation": {"source": {**old_pin, "path": str(live)}, "retained": old_pin},
        "current_diagnostic": {"source": live_pin, "retained": live_pin},
    }


def test_retained_authentication_source_positive(retained_authentication_roles):
    roles = retained_authentication_roles
    assert capture.verify_source_roles(roles) == roles
    assert roles["historical_generation"]["source"]["path"] == roles["current_diagnostic"]["source"]["path"]
    assert roles["historical_generation"]["source"]["sha256"] != roles["current_diagnostic"]["source"]["sha256"]


@pytest.mark.parametrize("mutation", [
    "missing-role", "renamed-role", "hash-only", "wrong-original", "current-for-history",
    "swapped", "current-snapshot", "changed-bytes", "missing-original", "hardlink",
])
def test_retained_authentication_source_rejections(retained_authentication_roles, tmp_path, mutation):
    roles = copy.deepcopy(retained_authentication_roles)
    old, current = roles["historical_generation"], roles["current_diagnostic"]
    if mutation == "missing-role":
        roles.pop("historical_generation")
    elif mutation == "renamed-role":
        roles["generation"] = roles.pop("historical_generation")
    elif mutation == "hash-only":
        old["retained"].pop("bytes")
    elif mutation == "wrong-original":
        old["source"]["sha256"] = "0" * 64
    elif mutation == "current-for-history":
        old["retained"] = current["retained"]
    elif mutation == "swapped":
        roles["historical_generation"], roles["current_diagnostic"] = current, old
    elif mutation == "current-snapshot":
        path = tmp_path / "current-copy.py"
        path.write_bytes(Path(current["source"]["path"]).read_bytes())
        current["retained"] = capture.binding(path)
    elif mutation == "changed-bytes":
        Path(old["retained"]["path"]).write_bytes(b"summary")
    elif mutation == "missing-original":
        Path(old["retained"]["path"]).unlink()
    else:
        path = tmp_path / "hardlink.py"
        path.hardlink_to(current["retained"]["path"])
        old.update(source=current["source"], retained=capture.binding(path))
    with pytest.raises(RuntimeError):
        capture.verify_source_roles(roles)


def retained_authentication_run_fixture(directory, status=0, timed_out=False):
    environment = {"Z": "a value", "A": ""}
    argv = ["/synthetic/python", "retained-only"]
    source = capture.binding(Path(__file__).resolve())
    pre = {"environment": environment, "sources": [source], "cwd": str(directory), "uid": os.getuid(),
           "capture_implementation": [source]}
    command = capture.environment_command(environment, argv)
    output, error = bytes(range(256)) + b"\nSTDERR\n", b"" if status == 0 else b"failed\xff"
    command_bytes, env_bytes = (command + "\n").encode(), capture.encoded(pre)
    whole = (b"COMMAND\n" + command_bytes + b"ENVIRONMENT\n" + env_bytes
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
    data = (command_bytes, capture.encoded(argv), env_bytes, output, error, whole)
    files = [capture.save(directory, "read" + suffix, raw)
             for suffix, raw in zip(capture.SUFFIXES, data, strict=True)]
    row = {"label": "read", "argv": argv, "command": command, "files": files,
           "exit_status": status, "timed_out": timed_out}
    receipt = {"preflight": pre, "results": [row], "sources_after": [source],
               "success": status == 0, "failure": None if status == 0 else {
                   "type": "RuntimeError", "message": "synthetic failure"}}
    pin = capture.save(directory, "capture.json", capture.encoded(receipt))
    return pre, {"read": argv}, pin, receipt


@pytest.mark.parametrize("status,timed_out", [(0, False), (7, False), (124, True)])
def test_retained_authentication_run_positive(tmp_path, status, timed_out):
    pre, commands, pin, receipt = retained_authentication_run_fixture(tmp_path, status, timed_out)
    assert capture.verify_retained_run(tmp_path, pin, pre, commands, require_success=False) == receipt
    if status:
        with pytest.raises(RuntimeError, match="run failed"):
            capture.verify_retained_run(tmp_path, pin, pre, commands)


@pytest.mark.parametrize("mutation", [
    "receipt-pin", "missing-member", "rebound-stream", "whole-is-stdout", "wrong-argv",
    "duplicate-environment", "timeout", "boolean-exit", "success", "source-drift",
])
def test_retained_authentication_run_rejections(tmp_path, mutation):
    pre, commands, pin, receipt = retained_authentication_run_fixture(tmp_path)
    row = receipt["results"][0]
    if mutation == "receipt-pin":
        pin["sha256"] = "0" * 64
    elif mutation == "missing-member":
        row["files"].pop()
    elif mutation in ("rebound-stream", "whole-is-stdout", "duplicate-environment"):
        index = {"rebound-stream": 3, "whole-is-stdout": 5, "duplicate-environment": 2}[mutation]
        path = Path(row["files"][index]["path"])
        data = b"summary" if index == 3 else (tmp_path / "read.stdout").read_bytes()
        if index == 2:
            data = capture.encoded(pre).replace(b'"A": ""', b'"A": "", "A": ""')
        path.write_bytes(data)
        row["files"][index] = capture.binding(path)
    elif mutation == "wrong-argv":
        commands["read"] = ["/different"]
    elif mutation == "timeout":
        row["timed_out"] = True
    elif mutation == "boolean-exit":
        row["exit_status"] = False
    elif mutation == "success":
        receipt["success"] = False
    else:
        receipt["sources_after"] = []
    if mutation != "receipt-pin":
        Path(pin["path"]).write_bytes(capture.encoded(receipt))
        pin = capture.binding(pin["path"])
    with pytest.raises(RuntimeError):
        capture.verify_retained_run(tmp_path, pin, pre, commands, require_success=False)


@pytest.mark.parametrize("mutation", [
    "none", "omitted_command", "command", "null_command", "link", "frame", "identity", "review",
    "retained_missing_command", "retained_bad_command",
])
def test_retained_authentication_launcher_and_review(tmp_path, mutation):
    run = tmp_path / "run"
    run.mkdir()
    pre, commands, run_pin, _ = retained_authentication_run_fixture(run)
    identity = {**pre, "argv": ["/synthetic/python", "launcher"]}
    identity["command"] = capture.environment_command(identity["environment"], identity["argv"])
    retained_identity = dict(identity)
    if mutation == "retained_missing_command":
        del retained_identity["command"]
    elif mutation == "retained_bad_command":
        retained_identity["command"] = "different"
    raw_identity = json.dumps(retained_identity, indent=2).encode() + b"\n"
    link = {"capture": run_pin, "success": True, "failure": None}
    if mutation == "link":
        link["capture"] = {**run_pin, "sha256": "0" * 64}
    output = capture.encoded(link)
    whole = b"IDENTITY\n" + raw_identity + b"STDOUT\n" + output + b"\nSTDERR\n\nEXIT_STATUS=0\nTIMED_OUT=False\n"
    if mutation == "frame":
        whole = output
    files = [capture.save(tmp_path, "launcher." + suffix, data) for suffix, data in zip(
        ("identity.json", "stdout", "stderr", "whole-command.log"),
        (raw_identity, output, b"", whole), strict=True)]
    outer = {"files": files, "exit_status": 0, "timed_out": False, "success": True,
             "capture_implementation_after": pre["capture_implementation"]}
    outer_pin = capture.save(tmp_path, "launcher.capture.json", capture.encoded(outer))
    checkpoint_pin = capture.save(tmp_path, "checkpoint.txt",
                                  ("launcher.capture.json " + outer_pin["sha256"]).encode())
    review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer", "mission_id": "synthetic",
              "review": {"status": "done"}, "checkpoint": {"path": checkpoint_pin["path"]}}
    if mutation == "review":
        review["review"]["status"] = "pending"
    review_pin = capture.save(tmp_path, "review.json", capture.encoded(review))
    expected = {**identity, "command": capture.environment_command(
        dict(sorted(identity["environment"].items())), identity["argv"])}
    if mutation == "identity":
        expected["argv"] = ["/different"]
    elif mutation == "omitted_command":
        del expected["command"]
    elif mutation in ("command", "null_command"):
        expected["command"] = None if mutation == "null_command" else "different"

    def authenticate():
        capture.verify_review_links(review_pin, checkpoint_pin, "synthetic", [outer_pin])
        return capture.verify_launcher(tmp_path, expected, outer_pin,
                                       run_expectation={"preflight": pre, "commands": commands})

    if mutation in ("none", "omitted_command"):
        assert authenticate() == outer
    else:
        with pytest.raises(RuntimeError):
            authenticate()


def test_captured_claim_failure_retains_json_diff_and_canonical_command(tmp_path):
    module = ("ace3.model.candidates.diagnose_q24_s16_final_head_final_rmsnorm_"
              "unselected_direct_hidden_mlp_stage17_stage11_attention_output_suffix_candidate_v1")
    current = {"id": "synthetic-claim", "status": "running",
               "started_ts": 1700000000.0, "attempt": 1}
    backlog = tmp_path / "backlog.jsonl"
    backlog.write_text(json.dumps({**current, "attempt": 2, "notes": "Host update"}) + "\n")
    supplied = {"mission_id": current["id"], "normal_running_claim": current,
                "native_backlog": str(backlog), "execution_authorization": {}}
    code = (
        f"import json\nfrom {module} import validate_launch_claim\n"
        f"try:\n    validate_launch_claim({supplied!r})\n"
        "except RuntimeError as error:\n"
        "    print(json.dumps({'status': 'UNKNOWN', 'failure': json.loads(str(error)), "
        "'candidate_check_invocations': 0}))\n"
        "    raise SystemExit(1)\n"
        "raise AssertionError('claim drift admitted')\n"
    )
    identity = {**preflight(), "argv": [sys.executable, "-B", "-c", code],
                "capture_implementation": capture.implementation_pins()}
    output, sealed = capture.seal_launcher(tmp_path, identity, timeout=30)
    pin = capture.binding(tmp_path / "launcher.capture.json")
    verified = capture.verify_launcher(tmp_path, identity, pin, require_success=False)
    assert verified == sealed and verified["success"] is False
    assert verified["exit_status"] == 1 and not verified["timed_out"]
    assert (tmp_path / "launcher.stderr").read_bytes() == b""
    payload = json.loads(output)
    assert payload["status"] == "UNKNOWN" and payload["candidate_check_invocations"] == 0
    assert payload["failure"]["claim_differences"]["attempt"] == {
        "authorized_present": True, "live_present": True, "authorized": 1, "live": 2}
    assert payload["failure"]["claim_differences"]["notes"]["live"] == "Host update"
    retained = json.loads((tmp_path / "launcher.identity.json").read_bytes())
    assert retained["command"] == capture.environment_command(
        dict(sorted(identity["environment"].items())), identity["argv"])
    assert "command" not in identity
    assert_pins(verified["files"] + [pin])
    with pytest.raises(RuntimeError, match="launcher failed"):
        capture.verify_launcher(tmp_path, identity, pin)


@pytest.mark.skipif("ACE3_PREFLIGHT_ATTEMPT" not in os.environ, reason="explicit retained-only integration")
def test_retained_authentication_real_integration():
    attempt = Path(os.environ["ACE3_PREFLIGHT_ATTEMPT"])
    root = capture.ROOT
    historical = root / "build/mixed-rmsnorm-conversion-topk-aef19b73e212-attempt001"
    launcher = root / "build/mixed-rmsnorm-conversion-topk-92c53aa23eae-actual-03"
    handoffs = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")

    def pinned(path, digest, size=None):
        if size is None:
            try:
                size = path.stat().st_size
            except OSError as error:
                raise capture.UnavailableBinding(f"missing/unreadable capture member: {path}") from error
        return {"path": str(path), "bytes": size, "sha256": digest}

    def document(pin):
        return capture.decode_retained(capture._retained_bytes(pin, Path(pin["path"])))

    result = {"status": "FAIL", "failure": "retained integration did not complete"}
    try:
        historical_pin = pinned(historical / "capture.json",
                                "8897cedbbe4f1a72186ba9dbe48c0bbf10c9076908727a3c317e29b29087f8bb")
        outer_pin = pinned(launcher / "launcher.capture.json",
                           "2353b356a3dbf3e68b9046a7c4190d909dd4ee6273ffb29aea6a671377721371", 1487)
        run_pin = pinned(launcher / "run/capture.json",
                         "e4ab543b7ffb27578a4e8f14170172b23ae701e25d6f1ea72fe06ce3224591c4", 20104)
        old_run, outer, current_run = document(historical_pin), document(outer_pin), document(run_pin)
        identity = document(outer["files"][0])
        roles = {
            "historical_generation": {
                "source": old_run["preflight"]["sources"][0],
                "retained": pinned(historical / "failed-source.py",
                                   "c05707c352a14014d095d7690a0f6f179ecf90b304b00a6dc58a509ee777a173", 21482)},
            "current_diagnostic": {
                "source": pinned(Path(identity["sources"][0]["path"]),
                                 "371be95be97c6d65590263709529fc3838aebf73c7d1dd28e70eef2c58640694", 21905),
                "retained": identity["sources"][0]},
        }
        capture.verify_source_roles(roles)
        capture.verify_source_snapshot(old_run["preflight"]["sources"][1], pinned(
            historical / "failed-test.py",
            "6d81a52540ce2386b155fab4ddde6cd1508116b16c18ee4e0082cd7230a12b27", 11439))
        for source in identity["sources"][1:]:
            capture.verify_source_snapshot(source, source)
        originals = capture.decode_retained(
            (attempt / "generation-implementation.source-roles.json").read_bytes())
        assert [item["source"] for item in originals] == identity["capture_implementation"]
        assert identity["capture_implementation"] == old_run["preflight"]["capture_implementation"]
        for item in originals:
            capture.verify_source_snapshot(item["source"], item["retained"])
        reviews = []
        for mission, round_number, receipts in (
                ("aef19b73e212", 2, [historical_pin]),
                ("92c53aa23eae", 4, [outer_pin, run_pin])):
            directory = handoffs / mission
            for path in (directory / f"round-{round_number:04d}.json", directory / "CHECKPOINT.md"):
                if not path.is_file():
                    raise capture.UnavailableBinding(f"missing/unreadable review original: {path}")
            reviews.append(capture.verify_review_links(
                capture.binding(directory / f"round-{round_number:04d}.json"),
                capture.binding(directory / "CHECKPOINT.md"), mission, receipts))
        expectations = []
        for directory, pin, receipt in (
                (historical, historical_pin, old_run), (launcher / "run", run_pin, current_run)):
            expected = {"preflight": receipt["preflight"],
                        "commands": {row["label"]: row["argv"] for row in receipt["results"]}}
            expectations.append(expected)
        capture.verify_retained_run(historical, historical_pin, **expectations[0], require_success=False)
        verified = capture.verify_launcher(launcher, identity, outer_pin, require_success=False,
                                           run_expectation=expectations[1])
        assert verified["success"] is False
        capture.save(attempt, "retained.command-environment.json", capture.encoded({
            "launcher": identity, "runs": expectations}))
        capture.save(attempt, "retained.source-roles.json", capture.encoded({
            "roles": roles, "historical_capture_implementation": originals,
            "historical_test": old_run["preflight"]["sources"][1],
            "current_sources": identity["sources"]}))
        capture.save(attempt, "retained.review-links.json", capture.encoded(reviews))
        result = {"status": "PASS", "historical_run": historical_pin, "launcher": outer_pin,
                  "run": run_pin, "retained_launcher_success": False,
                  "retained_exit_status": verified["exit_status"], "retained_timed_out": verified["timed_out"],
                  "scientific_classification_performed": False}
    except capture.UnavailableBinding as error:
        result = {"status": "UNKNOWN", "unavailable_binding": str(error)}
    finally:
        capture.save(attempt, "retained-integration.results.json", capture.encoded(result))
    assert result["status"] in ("PASS", "UNKNOWN")


def _run_retained_authentication_validation(attempt):
    """One compile/test invocation; the shared helper owns every authentication gate."""
    import py_compile

    attempt = capture._canonical_path(str(attempt))
    reservation = capture.decode_retained((attempt / "write-reservation.json").read_bytes())
    writes = [str(attempt), *(str(path) for path in attempt.iterdir())]
    audit = {"attempt": str(attempt), "writes": writes, "scientific_invocations": 0,
             "producer_invocations": 0, "service_invocations": 0, "subprocess_argv": []}
    active = True

    def record_path(value, dir_fd=None):
        if isinstance(value, int):
            value = os.readlink(f"/proc/self/fd/{value}").removesuffix(" (deleted)")
        if not isinstance(value, (str, bytes, os.PathLike)):
            raise RuntimeError(f"unattributed write descriptor: {value}")
        path = Path(os.fsdecode(value))
        if not path.is_absolute() and isinstance(dir_fd, int) and dir_fd >= 0:
            path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / path
        path = path.absolute()
        capture.verify_write_path(str(path), attempt)
        writes.append(str(path))

    def observe(event, args):
        if not active:
            return
        if event == "open" and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            record_path(args[0])
        elif event == "os.mkdir":
            record_path(args[0], args[2])
        elif event in ("os.remove", "os.rmdir"):
            record_path(args[0], args[1])
        elif event in ("os.rename", "os.link"):
            record_path(args[0], args[2])
            record_path(args[1], args[3])
        elif event == "os.symlink":
            record_path(args[1], args[2])
        elif event in ("os.chmod", "os.chown", "os.utime"):
            record_path(args[0], args[-1])
        elif event == "os.truncate":
            record_path(args[0])
        elif event == "subprocess.Popen":
            argv = args[1]
            audit["subprocess_argv"].append(argv)
            if not isinstance(argv, list) or argv[:3] != ["git", "check-ignore", "--quiet"]:
                audit["producer_invocations"] += 1
                raise RuntimeError(f"unauthorized validation subprocess: {argv}")
        elif event in ("socket.connect", "socket.getaddrinfo"):
            audit["service_invocations"] += 1
            raise RuntimeError(f"unauthorized validation service invocation: {event}")

    def profile(frame, event, arg):
        filename = frame.f_code.co_filename
        if (event == "call" and filename.startswith(str(capture.ROOT / "ace3") + "/")
                and filename != str(Path(capture.__file__).resolve())
                and not filename.endswith("/__init__.py")):
            audit["scientific_invocations"] += 1
            raise RuntimeError(f"forbidden repository invocation: {filename}:{frame.f_code.co_name}")

    sys.addaudithook(observe)
    sys.setprofile(profile)
    previous_tempdir = tempfile.tempdir
    try:
        temporary = attempt / "tmp"
        temporary.mkdir()
        tempfile.tempdir = str(temporary)
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--authentication-preflight", str(attempt)]
        capture.save(attempt, "validation.command.txt",
                     (capture.environment_command(dict(os.environ), command) + "\n").encode())
        capture.save(attempt, "validation.argv.json", capture.encoded(command))
        capture.save(attempt, "validation.environment.json", capture.encoded(dict(os.environ)))
        sources = capture.implementation_pins()
        capture.save(attempt, "validation.source-roles.json",
                     capture.encoded({"current_validation": sources}))
        with ExitStack() as stack:
            out = stack.enter_context((attempt / "validation.stdout").open("x"))
            err = stack.enter_context((attempt / "validation.stderr").open("x"))
            result = {"status": "FAIL", "compile_exit_status": None, "compiled_files": sources,
                      "pytest_exit_status": None, "write_confinement": None,
                      "normal_independent_review": "REQUIRED", "scientific_or_admission_claim": False}
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    for index, pin in enumerate(sources):
                        py_compile.compile(pin["path"], cfile=str(attempt / f"compiled-{index}.pyc"), doraise=True)
                    result["compile_exit_status"] = 0
                    print("py_compile: 2 scoped files")
                    pytest_argv = ["-q", "-p", "no:cacheprovider", "--basetemp", str(attempt / "pytest-tmp"),
                                   "--log-file", str(attempt / "pytest.log"),
                                   "--junitxml", str(attempt / "pytest.xml"),
                                   sources[1]["path"], "-k", "retained_authentication"]
                    result["pytest_argv"] = pytest_argv
                    capture.save(attempt, "pytest.argv.json", capture.encoded(pytest_argv))
                    status = int(pytest.main(pytest_argv))
                    result["pytest_exit_status"] = status
                integration_path = attempt / "retained-integration.results.json"
                integration = (capture.decode_retained(integration_path.read_bytes())
                               if integration_path.exists() else {
                                   "status": "FAIL", "failure": "integration did not finish"})
                result["integration"] = integration
                if capture.implementation_pins() != sources:
                    raise RuntimeError("validation source drift")
                result["status"] = integration["status"] if status == 0 else "FAIL"
                if result["status"] == "UNKNOWN":
                    result["unavailable_binding"] = integration["unavailable_binding"]
            except (OSError, RuntimeError, ValueError, py_compile.PyCompileError) as error:
                result["failure"] = {"type": type(error).__name__, "message": str(error)}
                traceback.print_exc(file=err)
        result = capture.seal_authentication_validation(attempt, result, audit, reservation)
        print(json.dumps(result, sort_keys=True))
        return result["exit_status"]
    finally:
        active = False
        sys.setprofile(None)
        tempfile.tempdir = previous_tempdir


@pytest.mark.parametrize("data", [
    b"command", b"command\n\n", b"command\r\n", b"command\xff\n", "command\n",
])
def test_command_file_rejects_malformed_framing(data):
    with pytest.raises(RuntimeError):
        capture.decode_command_file(data)


def test_command_file_decodes_only_one_terminal_lf():
    command = "HOME=/home/example python 'embedded\nnewline' 'trailing space '"
    assert capture.decode_command_file((command + "\n").encode()) == command


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--authentication-preflight":
        raise SystemExit("expected --authentication-preflight FRESH_RESERVED_ATTEMPT")
    raise SystemExit(_run_retained_authentication_validation(Path(sys.argv[2])))
