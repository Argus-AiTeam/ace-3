"""Byte-exact CPU diagnostic transport; callers retain scientific and authority gates.

New stdout-only launchers use launch_diagnostic, not a summary of subprocess output.
It reserves a fresh ignored build attempt and validates the retained identity,
streams, whole-command framing and terminal receipt before returning success.
The CLI accepts --attempt PATH --identity JSON --timeout SECONDS; identity supplies
argv, cwd, the complete child environment, uid, scope and the caller's preflights.
No environment values are implicitly inherited by the child. The new gate spells
commands with sorted environment keys so JSON serialization cannot change identity;
the legacy command and outer byte frames are unchanged.

verify_launcher requires an independently retained expected identity and receipt
binding, never identity reconstructed from a summary. Its failure-inspection mode
authenticates failed captures without admitting them. Neither API grants scientific
execution authority or replaces source/input/model/budget gates or Host review.

Retained preflights use verify_command, verify_write_audit and verify_source_roles.
Command equivalence permits only assignment-key order to vary; byte pins and
frames remain exact. Write audits require a trusted exclusive-creation reservation
(reserve_authentication_attempt supplies one), not a claimed "fresh" flag.
Write spellings must be absolute and lexically canonical; symlink-derived writes
are allowed only when both the spelling and resolved target stay in that attempt.
Source pins and attempt reservations still require fully canonical paths.
Historical snapshots authenticate their original source pins without reading that
original live path. Current diagnostic roles must bind the current source itself.
UnavailableBinding identifies one missing/corrupt original, never a summary fallback.
seal_authentication_validation retains the validation identity, streams, audit and
terminal result even when final confinement or source authentication fails. It
seals declared Python-level writes, not an OS sandbox or scientific evidence.

diagnostic_identity_preflight_v1 consumes these interfaces without dispatching.
Its accepted-novel result is only an identity gate, never execution authority.
diagnostic_identity_custody_v1 emits future proposal/catalog records from pinned
declarations and authenticated terminal captures, without launching a diagnostic.
"""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


SUFFIXES = (".command.txt", ".argv.json", ".environment.json",
            ".stdout", ".stderr", ".whole-command.log")
ROOT = Path(__file__).resolve().parents[3]


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"


def binding(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def implementation_pins():
    return [binding(Path(__file__).resolve()), binding(ROOT / "tests/test_diagnostic_capture_v1.py")]


def _write(stream, data):
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())


def save(directory, name, data):
    if Path(name).name != name:
        raise ValueError("capture member must be a basename")
    path = Path(directory) / name
    with path.open("xb") as stream:
        _write(stream, data)
    return binding(path)


def environment_command(environment, argv):
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) + " " + shlex.join(argv)


class UnavailableBinding(RuntimeError):
    """One required retained original is unavailable or corrupt."""


def decode_retained(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate retained JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise RuntimeError(f"invalid retained JSON constant: {value}")

    try:
        return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("malformed retained JSON") from error


def verify_command(command, argv, environment, expected_argv, expected_environment):
    """Compare complete identities, without executing shell text or inheriting keys."""
    for arguments, variables in ((argv, environment), (expected_argv, expected_environment)):
        if (not isinstance(arguments, list) or not arguments
                or not all(isinstance(arg, str) and "\0" not in arg for arg in arguments)
                or not isinstance(variables, dict)
                or not all(isinstance(k, str) and k and "=" not in k and "\0" not in k
                           and isinstance(v, str) and "\0" not in v for k, v in variables.items())):
            raise RuntimeError("invalid capture argv/complete environment")
    if not isinstance(command, str) or "\0" in command:
        raise RuntimeError("invalid capture command")
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise RuntimeError("malformed capture command") from error
    assignments = {}
    count = len(environment)
    for token in tokens[:count]:
        key, separator, value = token.partition("=")
        if not separator or not key or key in assignments:
            raise RuntimeError("invalid/duplicate capture environment assignment")
        assignments[key] = value
    if (assignments != environment or environment != expected_environment
            or tokens[count:] != argv or argv != expected_argv
            or command != environment_command(assignments, argv)):
        raise RuntimeError("capture command/argv/complete-environment mismatch")
    return environment_command(dict(sorted(environment.items())), argv)


def decode_command_file(data):
    """Remove exactly the one LF used to frame a retained command member."""
    if (not isinstance(data, bytes) or not data.endswith(b"\n")
            or data.endswith((b"\n\n", b"\r\n"))):
        raise RuntimeError("capture command file requires one terminal LF")
    try:
        return data[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("capture command file is not UTF-8") from error


def verify_launch_identity(actual, expected):
    """Preserve every launch field; only environment-key order is immaterial."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise RuntimeError("complete launch identity required")
    normalized = []
    for identity in (actual, expected):
        command = verify_command(
            identity.get("command"), identity.get("argv"), identity.get("environment"),
            expected.get("argv"), expected.get("environment"))
        normalized.append({**identity, "command": command})
    if encoded(normalized[0]) != encoded(normalized[1]):
        raise RuntimeError("command/account/role/budget launch identity mismatch")
    return normalized[0]


def _canonical_path(value, *, allow_symlinks=False):
    if not isinstance(value, str) or not value or "\0" in value:
        raise RuntimeError("invalid canonical path")
    path = Path(value)
    if (not path.is_absolute() or str(path) != value or ".." in path.parts
            or value.startswith("//") or (not allow_symlinks and path.resolve() != path)):
        raise RuntimeError(f"noncanonical path: {value}")
    return path


def _authentication_attempt(directory, root):
    directory, root = _canonical_path(str(directory)), _canonical_path(str(root))
    build = _canonical_path(str(root / "build"))
    if not build.is_dir() or directory == build or not directory.is_relative_to(build):
        raise RuntimeError("authentication attempt must be a canonical child of build")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", str(directory)],
                             cwd=root, capture_output=True, check=False)
    if ignored.returncode != 0:
        raise RuntimeError("authentication attempt is not confirmed ignored")
    return directory


def reserve_authentication_attempt(directory, *, root=ROOT):
    """Reserve once; callers must retain this trusted creation receipt for the audit."""
    directory = _authentication_attempt(directory, root)
    directory.mkdir(mode=0o700)
    stat = directory.stat()
    return {"path": str(directory), "device": stat.st_dev, "inode": stat.st_ino}


def verify_write_path(value, directory):
    """Require both the lexical write location and its target inside the attempt."""
    path = _canonical_path(value, allow_symlinks=True)
    resolved = path.resolve()
    if not path.is_relative_to(directory) or not resolved.is_relative_to(directory):
        raise RuntimeError(f"outside-attempt write: {value}")
    return resolved


def verify_write_audit(audit, reservation, *, root=ROOT):
    """Authenticate declared writes, not an OS sandbox or an inferred empty audit."""
    if not isinstance(reservation, dict) or "path" not in reservation:
        raise RuntimeError("fresh attempt reservation required")
    directory = _authentication_attempt(reservation["path"], root)
    stat = directory.stat()
    if (not directory.is_dir() or type(reservation.get("device")) is not int
            or type(reservation.get("inode")) is not int
            or (reservation["device"], reservation["inode"]) != (stat.st_dev, stat.st_ino)):
        raise RuntimeError("fresh attempt reservation mismatch")
    if (not isinstance(audit, dict) or audit.get("attempt") != str(directory)
            or not isinstance(audit.get("writes"), list)):
        raise RuntimeError("complete declared write audit required")
    symlink_writes = []
    for value in audit["writes"]:
        resolved = verify_write_path(value, directory)
        if str(resolved) != value:
            symlink_writes.append({"path": value, "resolved": str(resolved)})
    return {"attempt": str(directory), "declared_write_count": len(audit["writes"]),
            "symlink_derived_writes": symlink_writes}


def seal_authentication_validation(directory, result, audit, reservation, *, root=ROOT):
    """Seal a non-generative validation, including failures of the final gates."""
    directory = _canonical_path(str(directory))
    identity_names = ("command.txt", "argv.json", "environment.json", "source-roles.json")
    identity = {name: (directory / ("validation." + name)).read_bytes()
                for name in identity_names}
    terminal_names = ("write-audit.json", "results.json", "whole-command.log", "capture.json")
    result = dict(result)
    result["write_confinement"] = None
    with ExitStack() as stack:
        streams = {name: stack.enter_context((directory / ("validation." + name)).open("xb"))
                   for name in terminal_names}
        if isinstance(audit.get("writes"), list):
            audit["writes"].extend(str(directory / ("validation." + name)) for name in terminal_names)
        try:
            argv = decode_retained(identity["argv.json"])
            environment = decode_retained(identity["environment.json"])
            command = identity["command.txt"].decode()
            if not command.endswith("\n"):
                raise RuntimeError("validation command newline missing")
            verify_command(command[:-1], argv, environment, argv, environment)
            roles = decode_retained(identity["source-roles.json"])
            if not isinstance(roles, dict) or set(roles) != {"current_validation"}:
                raise RuntimeError("current validation source role required")
            if not isinstance(roles["current_validation"], list) or not roles["current_validation"]:
                raise RuntimeError("current validation source bindings required")
            for pin in roles["current_validation"]:
                _retained_bytes(pin, _validate_pin(pin))
            result["write_confinement"] = verify_write_audit(audit, reservation, root=root)
            for counter in ("scientific_invocations", "producer_invocations", "service_invocations"):
                if type(audit.get(counter)) is not int or audit[counter] != 0:
                    raise RuntimeError(f"forbidden or unaccounted validation invocation: {counter}")
            if result.get("status") not in ("PASS", "FAIL", "UNKNOWN"):
                raise RuntimeError("invalid validation terminal status")
            if result["status"] == "UNKNOWN" and not result.get("unavailable_binding"):
                raise RuntimeError("UNKNOWN requires one precise unavailable binding")
        except (OSError, RuntimeError, ValueError) as error:
            result["status"] = "FAIL"
            result["sealing_failure"] = {"type": type(error).__name__, "message": str(error)}
        result.update({counter: audit.get(counter) for counter in (
            "scientific_invocations", "producer_invocations", "service_invocations")})
        result.update(exit_status=0 if result["status"] in ("PASS", "UNKNOWN") else 1,
                      timed_out=False, normal_independent_review="REQUIRED",
                      scientific_or_admission_claim=False)
        _write(streams["write-audit.json"], encoded(audit))
        _write(streams["results.json"], encoded(result))
        output = (directory / "validation.stdout").read_bytes()
        error = (directory / "validation.stderr").read_bytes()
        whole = (b"COMMAND\n" + identity["command.txt"] + b"ARGV\n" + identity["argv.json"]
                 + b"ENVIRONMENT\n" + identity["environment.json"]
                 + b"SOURCE_ROLES\n" + identity["source-roles.json"] + b"STDOUT\n" + output
                 + b"\nSTDERR\n" + error
                 + f"\nEXIT_STATUS={result['exit_status']}\nTIMED_OUT=False\n".encode())
        _write(streams["whole-command.log"], whole)
        members = (*identity_names, "stdout", "stderr", *terminal_names[:-1])
        _write(streams["capture.json"], encoded({
            "status": result["status"], "exit_status": result["exit_status"], "timed_out": False,
            "files": [binding(directory / ("validation." + name)) for name in members],
        }))
    return result


def _validate_pin(pin):
    if (not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}
            or type(pin["bytes"]) is not int or pin["bytes"] < 0
            or not isinstance(pin["sha256"], str) or len(pin["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in pin["sha256"])):
        raise RuntimeError("exact path/byte-count/SHA-256 binding required")
    return _canonical_path(pin["path"])


def verify_source_snapshot(source, retained):
    """Bind real snapshot bytes to an original pin; never read source['path'] here."""
    _validate_pin(source)
    data = _retained_bytes(retained, _validate_pin(retained))
    if (source["bytes"], source["sha256"]) != (retained["bytes"], retained["sha256"]):
        raise RuntimeError("source snapshot/original binding mismatch")
    return data


def verify_source_roles(roles):
    names = {"historical_generation", "current_diagnostic"}
    if not isinstance(roles, dict) or set(roles) != names:
        raise RuntimeError("distinct historical_generation/current_diagnostic roles required")
    for role in roles.values():
        if not isinstance(role, dict) or set(role) != {"source", "retained"}:
            raise RuntimeError("each source role requires original and retained byte bindings")
        verify_source_snapshot(role["source"], role["retained"])
    historical, current = roles["historical_generation"], roles["current_diagnostic"]
    old_path, live_path = Path(historical["retained"]["path"]), Path(current["retained"]["path"])
    if (current["source"] != current["retained"]
            or historical["source"]["path"] == str(old_path)
            or old_path.samefile(live_path)):
        raise RuntimeError("historical/current source role conflation")
    return roles


def retained_document(pin):
    """Decode an exact, externally pinned original, not a newly inferred summary."""
    return decode_retained(_retained_bytes(pin, _validate_pin(pin)))


def verify_terminal_review(latest_pin, review_pin, checkpoint_pin, mission_id):
    """Authenticate the latest normal Reviewer identity before consuming its links."""
    latest, review = retained_document(latest_pin), retained_document(review_pin)
    checkpoint = _retained_bytes(checkpoint_pin, _validate_pin(checkpoint_pin))
    if (not isinstance(latest, dict) or latest.get("kind") != "handoff_ref"
            or latest.get("handoff") != {"path": review_pin["path"]}
            or not isinstance(review, dict) or review.get("kind") != "round_reviewed_handoff"
            or review.get("producer_role") != "reviewer" or review.get("mission_id") != mission_id
            or not isinstance(review.get("review"), dict) or review["review"].get("status") != "done"
            or review.get("checkpoint") != {"path": checkpoint_pin["path"]}):
        raise RuntimeError("latest terminal Reviewer identity mismatch")
    if not checkpoint.strip():
        raise UnavailableBinding(
            f"missing retained review-to-capture/source binding: empty {checkpoint_pin['path']}")
    return review


def verify_review_links(review_pin, checkpoint_pin, mission_id, receipt_pins):
    """Read exact review originals and their receipt references, without re-review."""
    review = decode_retained(_retained_bytes(review_pin, _validate_pin(review_pin)))
    checkpoint = _retained_bytes(checkpoint_pin, _validate_pin(checkpoint_pin))
    if (not isinstance(review, dict) or review.get("kind") != "round_reviewed_handoff"
            or review.get("producer_role") != "reviewer" or review.get("mission_id") != mission_id
            or not isinstance(review.get("review"), dict) or review["review"].get("status") != "done"
            or review.get("checkpoint") != {"path": checkpoint_pin["path"]}
            or not isinstance(receipt_pins, list) or not receipt_pins):
        raise RuntimeError("retained independent review link mismatch")
    for pin in receipt_pins:
        data = _retained_bytes(pin, _validate_pin(pin))
        if pin["sha256"].encode() not in checkpoint or Path(pin["path"]).name.encode() not in checkpoint:
            raise RuntimeError(f"review-to-receipt binding missing: {pin['path']}")
        if not data:
            raise RuntimeError("empty reviewed receipt")
    return {"review": review_pin, "checkpoint": checkpoint_pin, "receipts": receipt_pins}


@contextmanager
def same_scope(root, scope):
    """Keep a stable lock inode: unlinking it would allow overlapping owners."""
    build = Path(root) / "build"
    if build.resolve() != build or not build.is_dir():
        raise RuntimeError("canonical existing build directory required")
    name = hashlib.sha256(scope.encode()).hexdigest()
    descriptor = os.open(build / ("diagnostic-capture-" + name + ".lock"),
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("same-scope concurrency") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _execute(argv, cwd, environment, out, err, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("finite positive capture timeout required")
    timed_out, launch_error = False, None
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=out, stderr=err,
                                   start_new_session=True)
    except OSError as error:
        return 127, False, {"type": type(error).__name__, "message": str(error)}
    try:
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status, timed_out = 124, True
    finally:
        # Descendants must not keep writing after the streams have been sealed.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        out.flush()
        err.flush()
        os.fsync(out.fileno())
        os.fsync(err.fileno())
    return status, timed_out, launch_error


def run_command(directory, label, argv, preflight, results, timeout=90):
    """Require explicit cwd; seal command/stderr/launch/timeout failures before raising.

    Malformed cwd records fail before file creation; filesystem cwd failures
    retain the normal launch-error result and byte framing.
    """
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in label):
        raise ValueError("invalid capture label")
    cwd = preflight.get("cwd")
    if not isinstance(cwd, str) or not cwd or "\0" in cwd:
        raise ValueError("capture preflight requires a nonempty cwd string without NUL")
    directory = Path(directory)
    command = environment_command(preflight["environment"], argv)
    command_bytes, env_bytes = (command + "\n").encode(), encoded(preflight)
    paths = [directory / (label + suffix) for suffix in SUFFIXES]
    started = time.monotonic()
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open("xb")) for path in paths]
        for stream, data in zip(streams[:3], (command_bytes, encoded(argv), env_bytes), strict=True):
            _write(stream, data)
        status, timed_out, launch_error = _execute(
            argv, cwd, preflight["environment"], streams[3], streams[4], timeout)
        output, error = paths[3].read_bytes(), paths[4].read_bytes()
        _write(streams[5], b"COMMAND\n" + command_bytes + b"ENVIRONMENT\n" + env_bytes
               + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
               + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
    result = {"label": label, "command": command, "argv": argv, "files": [binding(p) for p in paths],
              "exit_status": status, "timed_out": timed_out,
              "elapsed_seconds": time.monotonic() - started}
    if launch_error is not None:
        result["launch_error"] = launch_error
    results.append(result)
    if status != 0 or timed_out or error or launch_error:
        raise RuntimeError(label + " failed; complete bytes retained")
    return output


def _launcher_command(identity):
    argv, environment = identity.get("argv"), identity.get("environment")
    command = identity.get("command")
    if "command" not in identity:
        try:
            command = environment_command(environment, argv)
        except (AttributeError, TypeError) as error:
            raise RuntimeError("invalid capture argv/complete environment") from error
    return verify_command(command, argv, environment, argv, environment)


def seal_launcher(directory, identity, timeout=115):
    """Seal the entire outer command, including failures, before returning its bytes."""
    directory = Path(directory)
    command = _launcher_command(identity)
    identity = {**identity, "command": identity.get("command", command),
                "capture_implementation": implementation_pins()}
    identity_bytes = encoded(identity)
    paths = [directory / ("launcher." + suffix) for suffix in
             ("identity.json", "stdout", "stderr", "whole-command.log")]
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open("xb")) for path in paths]
        receipt = stack.enter_context((directory / "launcher.capture.json").open("xb"))
        _write(streams[0], identity_bytes)
        status, timed_out, launch_error = _execute(
            identity["argv"], identity["cwd"], identity["environment"], streams[1], streams[2], timeout)
        output, error = paths[1].read_bytes(), paths[2].read_bytes()
        _write(streams[3], b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output
               + b"\nSTDERR\n" + error + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
        after = implementation_pins()
        outer = {"exit_status": status, "timed_out": timed_out,
                 "files": [binding(p) for p in paths],
                 "capture_implementation_after": after,
                 "success": status == 0 and not timed_out and not error and launch_error is None
                 and after == identity["capture_implementation"]}
        if launch_error is not None:
            outer["launch_error"] = launch_error
        if after != identity["capture_implementation"]:
            outer["failure"] = "capture implementation drift"
        _write(receipt, encoded(outer))
    return output, outer


def _retained_bytes(pin, path):
    _validate_pin(pin)
    if not isinstance(pin, dict) or pin.get("path") != str(path):
        raise RuntimeError(f"capture member path mismatch: {path}")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise UnavailableBinding(f"missing/unreadable capture member: {path}") from error
    expected = {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if pin != expected:
        raise UnavailableBinding(f"capture member binding mismatch: {path}")
    return data


def _capture_members(files, paths):
    if not isinstance(files, list) or len(files) != len(paths):
        raise RuntimeError("complete identity/stdout/stderr/whole capture bindings required")
    return [_retained_bytes(pin, path) for pin, path in zip(files, paths, strict=True)]


def _terminal_state(receipt):
    status, timed_out = receipt.get("exit_status"), receipt.get("timed_out")
    if type(status) is not int or type(timed_out) is not bool or (timed_out and status != 124):
        raise RuntimeError("invalid capture exit/timeout state")
    launch_error = receipt.get("launch_error")
    if launch_error is not None and (
            not isinstance(launch_error, dict) or status != 127 or timed_out
            or not isinstance(launch_error.get("type"), str)
            or not isinstance(launch_error.get("message"), str)):
        raise RuntimeError("invalid capture launch failure")
    return status, timed_out, launch_error


def verify_retained_run(directory, receipt_pin, preflight, commands, *, require_success=True):
    """Verify legacy run_command members and their sealed receipt, without dispatch."""
    directory = _canonical_path(str(directory))
    receipt = decode_retained(_retained_bytes(receipt_pin, directory / "capture.json"))
    if (not isinstance(receipt, dict) or encoded(receipt.get("preflight")) != encoded(preflight)
            or not isinstance(preflight, dict) or not preflight.get("sources")
            or receipt.get("sources_after") != preflight["sources"]
            or not isinstance(commands, dict) or not commands
            or not isinstance(receipt.get("results"), list)
            or not all(isinstance(row, dict) for row in receipt["results"])
            or [row.get("label") for row in receipt["results"]] != list(commands)):
        raise RuntimeError("capture preflight/source/command census mismatch")
    successful_commands = True
    for row in receipt["results"]:
        label = row["label"]
        if not isinstance(label, str) or not label or any(
                c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in label):
            raise RuntimeError("invalid capture label")
        paths = [directory / (label + suffix) for suffix in SUFFIXES]
        command, argv, env, output, error, whole = _capture_members(row.get("files"), paths)
        arguments, environment = decode_retained(argv), decode_retained(env)
        if (encoded(environment) != encoded(preflight) or arguments != row.get("argv")
                or not isinstance(row.get("command"), str)
                or command != (row["command"] + "\n").encode()):
            raise RuntimeError("capture command/argv/environment member mismatch")
        verify_command(row["command"], arguments, preflight.get("environment"),
                       commands[label], preflight.get("environment"))
        status, timed_out, launch_error = _terminal_state(row)
        expected = (b"COMMAND\n" + command + b"ENVIRONMENT\n" + env + b"\nSTDOUT\n"
                    + output + b"\nSTDERR\n" + error
                    + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
        if whole != expected:
            raise RuntimeError("capture whole-command/stream framing mismatch")
        successful_commands &= status == 0 and not timed_out and not error and launch_error is None
    success, failure = receipt.get("success"), receipt.get("failure")
    if (type(success) is not bool or (success and (not successful_commands or failure is not None))
            or (not success and (not isinstance(failure, dict)
                                 or not isinstance(failure.get("message"), str)
                                 or not isinstance(failure.get("type"), str)))):
        raise RuntimeError("capture run success/failure mismatch")
    if require_success and not success:
        raise RuntimeError("run failed; complete bytes retained")
    return receipt


def verify_launcher(directory, identity, receipt_pin, *, require_success=True, run_expectation=None):
    """Authenticate stored bytes only; require_success=False never changes the verdict."""
    directory = Path(directory)
    outer = decode_retained(_retained_bytes(receipt_pin, directory / "launcher.capture.json"))
    if not isinstance(outer, dict):
        raise RuntimeError("invalid capture receipt")
    paths = [directory / ("launcher." + suffix) for suffix in
             ("identity.json", "stdout", "stderr", "whole-command.log")]
    identity_bytes, output, error, whole = _capture_members(outer.get("files"), paths)
    retained_identity = decode_retained(identity_bytes)
    if not isinstance(identity, dict) or not isinstance(retained_identity, dict):
        raise RuntimeError("invalid capture identity")
    command = verify_command(retained_identity.get("command"), retained_identity.get("argv"),
                             retained_identity.get("environment"), identity.get("argv"),
                             identity.get("environment"))
    expected_command = _launcher_command(identity)
    if encoded({**retained_identity, "command": command}) != encoded({**identity, "command": expected_command}):
        raise RuntimeError("capture identity mismatch")
    status, timed_out, launch_error = _terminal_state(outer)
    expected_whole = (b"IDENTITY\n" + identity_bytes + b"STDOUT\n" + output
                      + b"\nSTDERR\n" + error
                      + f"\nEXIT_STATUS={status}\nTIMED_OUT={timed_out}\n".encode())
    if whole != expected_whole:
        raise RuntimeError("capture whole-command/stream framing mismatch")
    pins = identity.get("capture_implementation")
    if not isinstance(pins, list) or not pins:
        raise RuntimeError("capture implementation identity required")
    stable = outer.get("capture_implementation_after") == pins
    success = status == 0 and not timed_out and not error and launch_error is None and stable
    if outer.get("success") is not success:
        raise RuntimeError("capture success/terminal state mismatch")
    if not stable and outer.get("failure") != "capture implementation drift":
        raise RuntimeError("missing capture implementation failure state")
    if success and outer.get("failure") is not None:
        raise RuntimeError("capture success/failure mismatch")
    if run_expectation is not None:
        link = decode_retained(output)
        if not isinstance(link, dict) or not isinstance(run_expectation, dict):
            raise RuntimeError("invalid launcher-to-run receipt link")
        run = verify_retained_run(directory / "run", link.get("capture"),
                                  run_expectation["preflight"], run_expectation["commands"],
                                  require_success=require_success)
        for key in ("cwd", "uid", "environment", "sources", "capture_implementation"):
            if encoded(retained_identity.get(key)) != encoded(run["preflight"].get(key)):
                raise RuntimeError(f"launcher/run identity mismatch: {key}")
        if (encoded([link.get("success"), link.get("failure")]) != encoded([run["success"], run["failure"]])
                or (success and not run["success"])):
            raise RuntimeError("launcher/run terminal receipt mismatch")
    if require_success and not success:
        raise RuntimeError("launcher failed; complete bytes retained")
    return outer


def launch_diagnostic(directory, identity, timeout=90):
    """Capture one already-authorized command; existing attempts are never reused."""
    directory = Path(directory).absolute()
    build = ROOT / "build"
    if (directory != directory.resolve() or not directory.is_relative_to(build)
            or directory == build):
        raise ValueError("capture attempt must be a canonical child of build")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("finite positive capture timeout required")
    if (identity.get("cwd") != str(ROOT) or Path.cwd().resolve() != ROOT
            or identity.get("uid") != os.getuid()
            or identity.get("independent_host_review") != "REQUIRED"
            or type(identity.get("model_or_service_calls_authorized")) is not int
            or identity["model_or_service_calls_authorized"] != 0):
        raise RuntimeError("capture account/workdir/review/service preflight mismatch")
    argv, environment, scope = identity.get("argv"), identity.get("environment"), identity.get("scope")
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
            or not Path(argv[0]).is_absolute()):
        raise ValueError("capture requires explicit argv with an absolute executable")
    if (not isinstance(environment, dict)
            or not all(isinstance(k, str) and k and "=" not in k and "\0" not in k
                       and isinstance(v, str) and "\0" not in v for k, v in environment.items())
            or not isinstance(scope, str) or not scope):
        raise ValueError("capture requires explicit environment and original scope")
    command = environment_command(dict(sorted(environment.items())), argv)
    if identity.get("command", command) != command:
        raise RuntimeError("capture command/argv/environment mismatch")
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            check=True, capture_output=True).stdout
    if branch != b"argus/full-projection\n":
        raise RuntimeError("capture branch mismatch")
    subprocess.run(["git", "check-ignore", "--quiet", str(directory)], cwd=ROOT, check=True)
    identity = {**identity, "attempt": str(directory), "timeout_seconds": timeout,
                "command": command, "capture_implementation": implementation_pins()}
    with same_scope(ROOT, scope):
        directory.mkdir(mode=0o700)
        seal_launcher(directory, identity, timeout=timeout)
        receipt_pin = binding(directory / "launcher.capture.json")
        verify_launcher(directory, identity, receipt_pin)
    return {"success": True, "capture": receipt_pin, "identity": identity,
            "independent_host_review": "REQUIRED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args(argv)
    result = launch_diagnostic(args.attempt, json.loads(args.identity.read_bytes()), args.timeout)
    print(json.dumps({key: result[key] for key in
                      ("success", "capture", "independent_host_review")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
