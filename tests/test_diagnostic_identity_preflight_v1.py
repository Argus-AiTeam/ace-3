"""Non-generative shared identity tests and one retained-only validation entry point.

python -B tests/test_diagnostic_identity_preflight_v1.py --validate ATTEMPT
reserves ATTEMPT once under ignored build, compiles the scoped Python, runs these
tests, authenticates an existing failed capture without reclassification, and
records the real closure-binding boundary. No candidate module is imported.
Pytest's FD capture, temporary files and cached tempfile directory are confined
to the reserved attempt; the effective test environment is retained separately.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import json
from io import BytesIO
import os
from pathlib import Path
import py_compile
import sys
import tempfile

import pytest
import numpy as np

from ace3.model.candidates import diagnostic_capture_v1 as capture
from ace3.model.candidates import diagnostic_identity_preflight_v1 as gate


def put(directory, name, value):
    return capture.save(directory, name, value if isinstance(value, bytes) else capture.encoded(value))


@pytest.fixture
def fixture(tmp_path):
    return identity_fixture(tmp_path)


def identity_fixture(tmp_path):
    source = put(tmp_path, "operator.py", b"retained synthetic operator\n")
    historical = put(tmp_path, "historical-wrapper.py", b"historical synthetic wrapper\n")
    current = put(tmp_path, "current-wrapper.py", b"current synthetic wrapper\n")
    roles = {"historical_generation": {
        "source": {**historical, "path": current["path"]}, "retained": historical},
        "current_diagnostic": {"source": current, "retained": current}}
    identity = {
        "diagnostic": "synthetic-mixed-rmsnorm",
        "observation": {"quantity": "mixed-factor-swap", "controls": 9, "branches": 3},
        "inputs": {"operand": put(tmp_path, "operand.bin", b"synthetic operand")},
        "references": {"original": put(tmp_path, "reference.bin", b"fixed original reference")},
        "sources": {"operator": source},
        "command_semantics": {"operation": "final-head-vector-swap", "controls": 9},
        "permitted_invocations": {"head": 9, "prefix": 0, "reference": 0},
        "output_accounts": {"additive_closures": 9, "swap_reversals": 9},
        "termination": {"lane_terminated": True, "condition": "one reviewed engineering result"},
        "boundary": {"profile": "native-S16-RTZ/Q24", "weights": "INT4",
                     "operators": "FP16", "KV": "FP16", "admission": False, "thresholds": "fixed"},
    }
    argv, env = ["/synthetic/python", "-m", "synthetic.wrapper"], {"B": "two", "A": "one"}
    launch = {"argv": argv, "environment": env, "command": capture.environment_command(env, argv),
              "uid": os.getuid(), "cwd": str(capture.ROOT), "role": "engineer",
              "budget": {"head": 9}, "model": "unchanged", "access": "unchanged"}
    preflight = {"environment": env, "sources": [source], "cwd": str(capture.ROOT)}
    command = (launch["command"] + "\n").encode()
    environment = capture.encoded(preflight)
    stdout = capture.encoded({"diagnostic_identity": identity})
    whole = (b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment + b"\nSTDOUT\n"
             + stdout + b"\nSTDERR\n\nEXIT_STATUS=0\nTIMED_OUT=False\n")
    files = [put(tmp_path, "saved" + suffix, data) for suffix, data in zip(
        capture.SUFFIXES, (command, capture.encoded(argv), environment, stdout, b"", whole), strict=True)]
    receipt = put(tmp_path, "capture.json", {
        "preflight": preflight, "sources_after": [source], "success": True, "failure": None,
        "results": [{"label": "saved", "argv": argv, "command": launch["command"],
                     "files": files, "exit_status": 0, "timed_out": False}]})
    mission = "synthetic-closed"
    native = put(tmp_path, "native.json", {"task_id": mission, "status": "done"})
    checkpoint = put(tmp_path, "checkpoint.txt", (
        f"{Path(receipt['path']).name} {receipt['sha256']}\n"
        f"{Path(native['path']).name} {native['sha256']}\n").encode())
    review = put(tmp_path, "review.json", {
        "kind": "round_reviewed_handoff", "producer_role": "reviewer", "mission_id": mission,
        "review": {"status": "done"}, "checkpoint": {"path": checkpoint["path"]}})
    latest = put(tmp_path, "latest.json", {"kind": "handoff_ref", "handoff": {"path": review["path"]}})
    lane = {"mission_id": mission, "latest": latest, "review": review, "checkpoint": checkpoint,
            "native": {"pin": native, "mission_path": ["task_id"], "status_path": ["status"]},
            "capture": {"directory": str(tmp_path), "receipt": receipt, "preflight": preflight,
                        "commands": {"saved": argv}, "label": "saved"},
            "identity_paths": {field: ["diagnostic_identity", field] for field in gate.FIELDS},
            "source_snapshots": {"operator": source}}
    proposal = {"schema_version": 1, "identity": deepcopy(identity), "source_roles": roles,
                "source_snapshots": {"operator": source}, "launch": deepcopy(launch),
                "packaging": {"task_id": "new", "capture_directory": "fresh", "wrapper": "renamed"}}
    return tmp_path, proposal, {"schema_version": 1, "lanes": [lane]}, launch


def decide(fixture):
    directory, proposal, catalog, launch = fixture
    return gate.preflight_identity(put(directory, "proposal.json", proposal),
                                   put(directory, "catalog.json", catalog), launch)


@pytest.mark.parametrize("old,new", [
    ("a1e29a661b91", "a399e916489f"), ("a1d2195ce52f", "7117b5d21aa6")])
def test_closed_equivalent_packaging_is_not_novel(fixture, old, new):
    # Task labels demonstrate packaging invariance, not authentication of real lanes.
    fixture[1]["packaging"].update(task_id=new, previous_task=old)
    fixture[1]["identity"]["diagnostic"] = "renamed-wrapper"
    result = decide(fixture)
    assert result["status"] == "REJECTED" and result["reason"] == "closed_equivalent"
    assert result["scientific_invocations"] == result["producer_invocations"] == 0
    assert result["dispatch_authorized"] is False


def novel(fixture):
    directory, proposal, _, _ = fixture
    proposal["identity"]["observation"] = {"quantity": "unmeasured-suffix-response", "coordinate": 17}
    proposal["identity"]["inputs"] = {"operand": put(directory, "novel.bin", b"different operand bytes")}
    proposal["identity"]["permitted_invocations"] = {"retained_scalar_comparison": 1, "head": 0}


def test_material_novelty_acceptance(fixture):
    novel(fixture)
    result = decide(fixture)
    assert result["status"] == "ACCEPTED_NOVEL"
    assert result["dispatch_authorized"] is False


@pytest.mark.parametrize("field", ["observation", "inputs", "permitted_invocations"])
def test_each_material_distinction_is_required(fixture, field):
    old = deepcopy(fixture[1]["identity"][field])
    novel(fixture)
    fixture[1]["identity"][field] = old
    assert decide(fixture)["status"] == "REJECTED"


@pytest.mark.parametrize("field", ["sources", "command_semantics", "output_accounts", "termination",
                                  "references", "boundary"])
def test_single_surface_mutations_do_not_reopen_lane(fixture, field):
    directory, proposal, _, _ = fixture
    if field in ("sources", "references"):
        name = next(iter(proposal["identity"][field]))
        pin = put(directory, "mutant.bin", b"mutant bytes")
        proposal["identity"][field][name] = pin
        if field == "sources":
            proposal["source_snapshots"][name] = pin
    else:
        proposal["identity"][field]["mutation"] = True
    assert decide(fixture)["status"] == "REJECTED"


@pytest.mark.parametrize("mutation", ["value", "argv", "duplicate", "uid", "budget", "role", "model"])
def test_actual_command_and_account_mutations_fail(fixture, mutation):
    launch = fixture[3]
    if mutation == "value":
        launch["environment"]["B"] = "changed"
    elif mutation == "argv":
        launch["argv"].append("--different")
    elif mutation == "duplicate":
        launch["command"] = "A=one " + launch["command"]
    elif mutation == "uid":
        launch["uid"] += 1
    else:
        launch[mutation] = "changed"
    assert decide(fixture)["status"] == "REJECTED"


def test_only_environment_key_order_is_equivalent(fixture):
    novel(fixture)
    launch = fixture[3]
    launch["environment"] = dict(reversed(list(launch["environment"].items())))
    launch["command"] = capture.environment_command(launch["environment"], launch["argv"])
    assert decide(fixture)["status"] == "ACCEPTED_NOVEL"


@pytest.mark.parametrize("mutation", ["missing_field", "missing_source", "empty_checkpoint",
                                      "missing_capture", "missing_native"])
def test_missing_original_is_precise_unknown(fixture, mutation):
    directory, _, catalog, _ = fixture
    lane = catalog["lanes"][0]
    if mutation == "missing_field":
        lane["identity_paths"]["observation"] = ["absent"]
    elif mutation == "empty_checkpoint":
        lane["checkpoint"] = put(directory, "empty.txt", b"")
        review = capture.retained_document(lane["review"])
        review["checkpoint"] = {"path": lane["checkpoint"]["path"]}
        lane["review"] = put(directory, "empty-review.json", review)
        lane["latest"] = put(directory, "empty-latest.json",
                             {"kind": "handoff_ref", "handoff": {"path": lane["review"]["path"]}})
    else:
        pin = (lane["source_snapshots"]["operator"] if mutation == "missing_source" else
               lane["capture"]["receipt"] if mutation == "missing_capture" else lane["native"]["pin"])
        Path(pin["path"]).unlink()
    result = decide(fixture)
    assert result["status"] == "UNKNOWN" and result["unavailable_binding"]
    assert result["dispatch_authorized"] is False


@pytest.mark.parametrize("mutation", ["native_status", "stale_round", "frame", "source_role", "termination"])
def test_integrity_and_termination_mutations_fail(fixture, mutation):
    directory, proposal, catalog, _ = fixture
    lane = catalog["lanes"][0]
    if mutation == "native_status":
        lane["native"]["status_path"] = ["task_id"]
    elif mutation == "stale_round":
        lane["latest"] = put(directory, "stale.json", {"kind": "handoff_ref", "handoff": {"path": "other"}})
    elif mutation == "frame":
        receipt = capture.retained_document(lane["capture"]["receipt"])
        Path(receipt["results"][0]["files"][-1]["path"]).write_bytes(b"corrupt original")
    elif mutation == "source_role":
        proposal["source_roles"]["historical_generation"]["retained"] = (
            proposal["source_roles"]["current_diagnostic"]["retained"])
    else:
        proposal["identity"]["termination"]["lane_terminated"] = False
    assert decide(fixture)["status"] != "ACCEPTED_NOVEL"


def test_same_bytes_new_input_path_is_not_material(fixture):
    original = fixture[1]["identity"]["inputs"]["operand"]
    novel(fixture)
    fixture[1]["identity"]["inputs"]["operand"] = put(
        fixture[0], "copied.bin", Path(original["path"]).read_bytes())
    assert decide(fixture)["status"] == "REJECTED"


def test_renamed_input_role_is_not_material(fixture):
    original = fixture[1]["identity"]["inputs"]["operand"]
    novel(fixture)
    fixture[1]["identity"]["inputs"] = {"renamed-operand": original}
    assert decide(fixture)["status"] == "REJECTED"


def test_missing_lane_never_allows_novel_candidate(fixture):
    novel(fixture)
    fixture[2]["lanes"] = []
    assert decide(fixture)["status"] == "UNKNOWN"


def test_all_closed_lanes_are_checked(fixture):
    novel(fixture)
    lane = deepcopy(fixture[2]["lanes"][0])
    lane["identity_paths"]["observation"] = ["missing-second-lane-binding"]
    fixture[2]["lanes"].append(lane)
    assert decide(fixture)["status"] == "UNKNOWN"


def test_write_audit_rejects_outside_attempt(fixture):
    directory = fixture[0]
    stat = directory.stat()
    reservation = {"path": str(directory), "device": stat.st_dev, "inode": stat.st_ino}
    # Validation puts pytest's tmp_path inside its ignored attempt.
    if directory.is_relative_to(capture.ROOT / "build"):
        assert capture.verify_write_audit(
            {"attempt": str(directory), "writes": [str(directory / "inside")]}, reservation)
        with pytest.raises(RuntimeError, match="outside-attempt"):
            capture.verify_write_audit(
                {"attempt": str(directory), "writes": [str(capture.ROOT / "outside")]}, reservation)


@contextmanager
def _pytest_temporary_files(directory):
    directory.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        for name in ("TMPDIR", "TEMP", "TMP"):
            patch.setenv(name, str(directory))
        # FD capture uses tempfile's cached choice, independently of --basetemp.
        patch.setattr(tempfile, "tempdir", str(directory))
        yield


def test_pytest_temporary_files_override_cached_outside_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, "/tmp")
    directory = tmp_path / "capture-tmp"
    with _pytest_temporary_files(directory):
        assert tempfile.gettempdir() == str(directory)
        assert all(os.environ[name] == str(directory) for name in ("TMPDIR", "TEMP", "TMP"))
        with tempfile.TemporaryFile(mode="w+b") as stream:
            target = Path(os.readlink(f"/proc/self/fd/{stream.fileno()}"))
            assert target.is_relative_to(directory)
            stream.write(b"capture bytes")
            stream.seek(0)
            assert stream.read() == b"capture bytes"
        with tempfile.TemporaryDirectory() as temporary:
            assert Path(temporary).is_relative_to(directory)
    assert tempfile.tempdir == "/tmp"
    assert all(os.environ[name] == "/tmp" for name in ("TMPDIR", "TEMP", "TMP"))


def _retained_capture_validation(attempt):
    root = capture.ROOT
    handoffs = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
    launcher = root / "build/mixed-rmsnorm-conversion-topk-92c53aa23eae-actual-03"

    def pinned(path, digest, size):
        return {"path": str(path), "bytes": size, "sha256": digest}

    outer_pin = pinned(launcher / "launcher.capture.json",
                       "2353b356a3dbf3e68b9046a7c4190d909dd4ee6273ffb29aea6a671377721371", 1487)
    run_pin = pinned(launcher / "run/capture.json",
                     "e4ab543b7ffb27578a4e8f14170172b23ae701e25d6f1ea72fe06ce3224591c4", 20104)
    receipt = capture.retained_document(run_pin)
    identity = capture.retained_document(capture.retained_document(outer_pin)["files"][0])
    snapshots = root / "build/retained-capture-authentication-preflight-8197a9dd10a0-attempt001"
    roles = []
    for source, filename in zip(identity["capture_implementation"],
                                ("generation-capture-helper.py", "generation-capture-test.py"), strict=True):
        retained = {**source, "path": str(snapshots / filename)}
        capture.verify_source_snapshot(source, retained)
        roles.append({"source": source, "retained": retained})
    for source in identity["sources"]:
        capture.verify_source_snapshot(source, source)
    source_roles = {
        "historical_generation": roles[0],
        "current_diagnostic": {"source": capture.binding(Path(capture.__file__).resolve()),
                               "retained": capture.binding(Path(capture.__file__).resolve())}}
    capture.verify_source_roles(source_roles)
    expectation = {"preflight": receipt["preflight"],
                   "commands": {row["label"]: row["argv"] for row in receipt["results"]}}
    verified = capture.verify_launcher(launcher, identity, outer_pin, require_success=False,
                                       run_expectation=expectation)
    review_dir = handoffs / "92c53aa23eae"
    review_pins = [capture.binding(review_dir / name)
                   for name in ("latest.json", "round-0004.json", "CHECKPOINT.md")]
    capture.verify_terminal_review(*review_pins, "92c53aa23eae")
    review_links = capture.verify_review_links(review_pins[1], review_pins[2],
                                               "92c53aa23eae", [outer_pin, run_pin])
    positive = {
        "capture": outer_pin, "run": run_pin, "identity": identity, "run_expectation": expectation,
        "source_roles": source_roles, "generation_snapshots": roles, "review": review_links,
        "latest": review_pins[0],
        "historical_success": verified["success"], "historical_exit": verified["exit_status"],
        "historical_timed_out": verified["timed_out"], "reclassification_performed": False}
    put(attempt, "retained-positive.json", positive)
    return positive


def _retained_validation(attempt):
    _retained_capture_validation(attempt)
    handoffs = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
    directory = handoffs / "a399e916489f"
    pins = [capture.binding(directory / name)
            for name in ("latest.json", "round-0001.json", "CHECKPOINT.md")]
    put(attempt, "closed-lane-review-pins.json", pins)
    capture.verify_terminal_review(*pins, "a399e916489f")
    raise capture.UnavailableBinding(
        "missing reviewed schema-1 identity field bindings for the two real closed-lane pairs")


def validate(attempt, *, retained_validation=_retained_validation, extra_sources=(), test_paths=None,
             forbidden_functions=()):
    """One guarded, non-generative compile/test/retained-read attempt."""
    reservation = capture.reserve_authentication_attempt(attempt)
    writes = [str(attempt)]
    counts = {"scientific_invocations": 0, "model_invocations": 0, "producer_invocations": 0,
              "service_invocations": 0, "outside_write_attempts": 0}
    counts.update({counter: 0 for _, _, counter in forbidden_functions})
    active = True
    allowed_sources = {str(Path(capture.__file__).resolve()), str(Path(gate.__file__).resolve()),
                       *(str(path) for path in extra_sources)}

    def observe(event, args):
        if not active:
            return
        paths = []
        if event == "open" and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            paths = [args[0]]
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime", "os.truncate"):
            paths = [args[0]]
        elif event in ("os.rename", "os.link"):
            paths = list(args[:2])
        elif event == "os.symlink":
            paths = [args[1]]
        elif event == "subprocess.Popen":
            if not isinstance(args[1], list) or args[1][:3] != ["git", "check-ignore", "--quiet"]:
                counts["producer_invocations"] += 1
                raise RuntimeError(f"forbidden validation subprocess: {args[1]}")
        elif event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            counts["service_invocations"] += 1
            raise RuntimeError(f"forbidden validation service call: {event}")
        elif event in ("os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec"):
            counts["producer_invocations"] += 1
            raise RuntimeError(f"forbidden validation dispatch: {event}")
        for value in paths:
            if isinstance(value, int):
                value = os.readlink(f"/proc/self/fd/{value}")
            path = Path(os.fsdecode(value)).absolute()
            if not path.is_relative_to(attempt) or not path.resolve().is_relative_to(attempt):
                counts["outside_write_attempts"] += 1
                raise RuntimeError(f"outside-attempt validation write: {path}")
            writes.append(str(path))

    def profile(frame, event, arg):
        filename = frame.f_code.co_filename
        for source, name, counter in forbidden_functions:
            if event == "call" and filename == source and frame.f_code.co_name == name:
                counts[counter] += 1
                raise RuntimeError(f"forbidden validation function: {name}")
        if (event == "call" and filename.startswith(str(capture.ROOT / "ace3") + "/")
                and filename not in allowed_sources and not filename.endswith("/__init__.py")):
            counts["scientific_invocations"] += 1
            counts["model_invocations"] += 1
            raise RuntimeError(f"forbidden scientific invocation: {filename}")
        if event == "call" and filename == str(Path(capture.__file__).resolve()) and (
                frame.f_code.co_name in {"_execute", "run_command", "seal_launcher", "launch_diagnostic"}):
            counts["producer_invocations"] += 1
            raise RuntimeError("forbidden capture dispatch")

    sys.addaudithook(observe)
    sys.setprofile(profile)
    result = {"status": "FAIL", "normal_independent_review": "REQUIRED",
              "scientific_or_admission_claim": False}
    try:
        put(attempt, "write-reservation.json", reservation)
        argv = [sys.executable, "-B", *sys.argv]
        env = dict(os.environ)
        put(attempt, "validation.argv.json", argv)
        put(attempt, "validation.environment.json", env)
        command = (capture.environment_command(env, argv) + "\n").encode()
        put(attempt, "validation.command.txt", command)
        if "DIAGNOSTIC_VALIDATION_SHELL_COMMAND" in env:
            put(attempt, "validation.shell-command.txt",
                env["DIAGNOSTIC_VALIDATION_SHELL_COMMAND"].encode())
        sources = [Path(capture.__file__).resolve(), Path(gate.__file__).resolve(),
                   Path(__file__).resolve(), *extra_sources]
        source_pins = [capture.binding(path) for path in sources]
        put(attempt, "validation.sources.json", source_pins)
        with (attempt / "validation.stdout").open("x") as out, (
                attempt / "validation.stderr").open("x") as err:
            with redirect_stdout(out), redirect_stderr(err):
                try:
                    for index, path in enumerate(sources):
                        py_compile.compile(str(path), cfile=str(attempt / f"compiled-{index}.pyc"), doraise=True)
                    result["compile_exit_status"] = 0
                    pytest_args = ["-q", "--capture=fd", "-p", "no:cacheprovider",
                                   "--basetemp", str(attempt / "pytest-tmp"),
                                   "--log-file", str(attempt / "pytest.log"),
                                   "--junitxml", str(attempt / "pytest.xml"),
                                   *(str(path) for path in (test_paths or [Path(__file__).resolve()]))]
                    put(attempt, "pytest.argv.json", pytest_args)
                    with _pytest_temporary_files(attempt / "capture-tmp"):
                        put(attempt, "pytest.environment.json", dict(os.environ))
                        result["pytest_exit_status"] = int(pytest.main(pytest_args))
                    if result["pytest_exit_status"] != 0:
                        raise RuntimeError("focused identity tests failed")
                    retained_validation(attempt)
                    result["status"] = "PASS"
                except capture.UnavailableBinding as error:
                    result.update(status="UNKNOWN", unavailable_binding=str(error))
                except (RuntimeError, OSError, py_compile.PyCompileError) as error:
                    result.update(status="FAIL", failure=f"{type(error).__name__}: {error}")
                    print(result["failure"], file=sys.stderr)
                if [capture.binding(path) for path in sources] != source_pins:
                    result.update(status="FAIL", failure="validation source drift")
        result["counters"] = counts
        if any(counts.values()):
            result.update(status="FAIL", failure="nonzero forbidden validation counters")
        result["exit_status"], result["timed_out"] = int(result["status"] == "FAIL"), False
        stdout, stderr = (attempt / "validation.stdout").read_bytes(), (attempt / "validation.stderr").read_bytes()
        put(attempt, "validation.whole-command.log",
            b"COMMAND\n" + command + b"ENVIRONMENT\n" + capture.encoded(env) + b"\nSTDOUT\n"
            + stdout + b"\nSTDERR\n" + stderr
            + f"\nEXIT_STATUS={result['exit_status']}\nTIMED_OUT=False\n".encode())
        with (attempt / "validation.results.json").open("xb") as result_stream, (
                attempt / "validation.write-audit.json").open("xb") as audit_stream:
            result["write_confinement"] = capture.verify_write_audit(
                {"attempt": str(attempt), "writes": writes}, reservation)
            audit_stream.write(capture.encoded({"reservation": reservation, "writes": writes, **counts}))
            audit_stream.flush()
            os.fsync(audit_stream.fileno())
            result["members"] = [capture.binding(path) for path in sorted(attempt.iterdir())
                                 if path.is_file() and path.name != "validation.results.json"]
            result_stream.write(capture.encoded(result))
            result_stream.flush()
            os.fsync(result_stream.fileno())
        return result
    finally:
        active = False
        sys.setprofile(None)


@pytest.mark.parametrize("mutation", ["none", "path", "digest", "snapshot", "missing"])
def test_generation_origin_requires_original_pin(tmp_path, mutation):
    source = put(tmp_path, "generation.py", b"historical source\n")
    snapshot = put(tmp_path, "snapshot.py", b"historical source\n")
    current = put(tmp_path, "current.py", b"current source\n")
    roles = {"historical_generation": {"source": source, "retained": snapshot},
             "current_diagnostic": {"source": current, "retained": current}}
    runs = [{"preflight": {"sources": [deepcopy(source)]}}]
    if mutation == "path":
        roles["historical_generation"]["source"] = {**source, "path": current["path"]}
    elif mutation == "digest":
        roles["historical_generation"]["source"] = {**source, "sha256": "0" * 64}
    elif mutation == "snapshot":
        roles["historical_generation"]["retained"] = current
    elif mutation == "missing":
        runs[0]["preflight"]["sources"] = []
    if mutation == "none":
        gate.verify_generation_origin(roles, runs)
    else:
        with pytest.raises(RuntimeError):
            gate.verify_generation_origin(roles, runs)


@pytest.mark.parametrize("mutation", [
    "none", "missing", "member", "encoding", "reencoded", "mutated", "shape", "dtype",
])
def test_current_declaration_stage_member(tmp_path, mutation):
    words = np.arange(896, dtype="<u2")
    stream = BytesIO()
    np.savez(stream, stage13=(words[:895] if mutation == "shape" else
                             words.astype(">u2") if mutation == "dtype" else words))
    archive = put(tmp_path, "reference.npz", stream.getvalue())
    data = words.tobytes()
    if mutation == "reencoded":
        data = words.astype(">u2").tobytes()
    elif mutation == "mutated":
        data = b"\xff\xff" + data[2:]
    member = put(tmp_path, "stage13.raw", data)
    spec = {"source_archive": archive, "source_member": "stage13",
            "encoding": gate.MEMBER_ENCODING, "derived_pin": member}
    if mutation == "missing":
        Path(member["path"]).unlink()
    elif mutation == "member":
        spec["source_member"] = "stage11"
    elif mutation == "encoding":
        spec["encoding"] = "numpy-native"
    if mutation == "none":
        assert gate.verify_stage_member(spec, "stage13").tolist() == list(range(896))
    else:
        with pytest.raises(RuntimeError):
            gate.verify_stage_member(spec, "stage13")


@pytest.mark.parametrize("mutation", ["none", "missing_field", "empty_checkpoint", "snapshot", "native"])
def test_current_declaration_preserves_embedded_mode(fixture, mutation):
    lane = fixture[2]["lanes"][0]
    if mutation == "missing_field":
        lane["identity_paths"]["observation"] = ["missing"]
    elif mutation == "empty_checkpoint":
        Path(lane["checkpoint"]["path"]).write_bytes(b"")
    elif mutation == "snapshot":
        Path(lane["source_snapshots"]["operator"]["path"]).write_bytes(b"changed")
    elif mutation == "native":
        Path(lane["native"]["pin"]["path"]).write_bytes(b"{}")
    if mutation == "none":
        assert gate._closed_lane(lane)["diagnostic"] == "synthetic-mixed-rmsnorm"
    else:
        with pytest.raises(RuntimeError):
            gate._closed_lane(lane)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--validate":
        raise SystemExit("expected --validate FRESH_IGNORED_ATTEMPT")
    outcome = validate(Path(sys.argv[2]).absolute())
    print(json.dumps({key: value for key, value in outcome.items() if key != "members"}, sort_keys=True))
    raise SystemExit(outcome["exit_status"])
