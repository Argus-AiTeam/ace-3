"""Task-native, in-memory synthetic tests plus one frozen-package observation.

This test sidecar retains the exact single-round compile/pytest command below.
No evidence output directory, pytest cache, or bytecode cache is written.
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


VALIDATION_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
    "python3 -B tests/test_stage11_operational_release_consumption_preflight_v1.py"
)
SOURCE = Path(__file__).resolve().parents[1] / (
    "ace3/model/candidates/stage11_operational_release_consumption_preflight_v1.py"
)
SPEC = importlib.util.spec_from_file_location("stage11_release_preflight", SOURCE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def encoded(value):
    return json.dumps(value, sort_keys=True).encode() if not isinstance(value, bytes) else value


def pin(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def synthetic(monkeypatch):
    """Construct independent synthetic receipts, never execute a copied member."""
    repository = Path("/synthetic/ace3-argus")
    root = repository / gate.PACKAGE_RELATIVE
    files = {}

    def put(path, value):
        files[path] = encoded(value)
        return pin(path, files[path])

    def make(change=None):
        files.clear()
        interpreter = put(repository / "python", b"synthetic interpreter; never execute")
        sources = [
            {**put(repository / f"source-{index}.py", b"never import"), "role": role}
            for index, role in enumerate(sorted(gate.SOURCE_ROLES))
        ]
        authority = {
            "kind": "manager_stage11_operational_release_build_authorization",
            "issuer": "manager", "mission_id": gate.MISSION_ID, "expected_attempt": 1,
            "authorized_science": False, "scientific_run_budget": 0,
            "runtime_authorization_fields": {"authorization_id": "", "authorization_action": ""},
            "prospective_contract": {
                "builder_identity": gate.MISSION_ID, "output_root": gate.PACKAGE_RELATIVE,
                "source_inputs_read_only": sources,
            },
        }
        authority_pin = put(repository / "authority.json", authority)
        put(root / "build-authorization.json", authority)
        for name in (
            "builder.identity.json", "build.command.txt", "README.md", "focused-tests.txt",
            "operational.py", "test_operational.py", "build_release.py",
        ):
            put(root / name, b"synthetic inert member")
        implementation = [pin(root / "operational.py", files[root / "operational.py"])]
        history = {
            "count_required_by_authority": 6, "status": "CLOSED", "reason": "synthetic only",
            "individual_failure_evidence": [],
        }
        for index in range(6):
            failure_id = f"synthetic-historical-{index}"
            failure = put(repository / f"failure-{index}.json", {
                "failure_id": failure_id, "status": "FAIL",
            })
            closure = put(repository / f"closure-{index}.json", {
                "failure_id": failure_id, "status": "PASS", "original_failure": failure,
                "test_node": f"test_identity[{index}]",
            })
            history["individual_failure_evidence"].append({
                "failure_id": failure_id, "failure": failure, "closure": closure,
            })
        inputs = {
            "schema_version": 1, "mission_id": gate.MISSION_ID, "attempt": 1,
            "authority": authority_pin, "authorization_id": "", "authorization_action": "",
            "scientific_run_budget": 0, "cwd": str(root / "candidate"),
            "environment": {"PYTHONDONTWRITEBYTECODE": "1"}, "uid": 1000,
            "interpreter": interpreter, "historical_six_gate_failures": history,
            "members": implementation, "origins": [], "review": "REQUIRED", "sources": sources,
        }
        phases = {}
        processes = []
        for phase in gate.PHASES:
            argv = [interpreter["path"], "-P", "-B", str(root / "operational.py"), phase]
            identity = {
                "argv": argv, "cwd": inputs["cwd"], "environment": inputs["environment"],
                "interpreter": interpreter, "no_bytecode": True, "optimization": 0,
                "safe_path": True, "uid": inputs["uid"],
            }
            processes.append({key: identity[key] for key in ("argv", "cwd", "environment")})
            phases[phase] = {
                "actual_identity": identity, "candidate_check_invocations": 0,
                "scientific_invocations": 0, "dependency_boundary": "synthetic only",
                "dispatch_attempts": [], "numpy_import": {}, "phase": phase, "status": "PASS",
            }
        phases["compile"].update(build="synthetic in-memory compilation", compiled=implementation)
        phases["tests"].update(errors=0, failures=0, skipped=0, tests=6)
        phases["validate"].update(
            historical_six_gate_failures=history, normal_independent_review="REQUIRED",
            prior_phases={p: phases[p] for p in ("compile", "tests")},
            reason="synthetic admissibility only", release_consumption_authorized=False,
        )
        result = {
            "mission_id": gate.MISSION_ID, "attempt": 1, "authority": authority_pin,
            "candidate_check_invocations": 0, "scientific_invocations": 0,
            "package_validation_invocations": 1, "commands": {}, "members_unchanged": True,
            "sources_unchanged": True, "normal_independent_review": "REQUIRED",
            "observed_subprocesses": processes, "operational_result": phases["validate"],
            "scope": "synthetic operational only",
        }
        audits = {
            name: {
                "attempt": str(root), "scope": (
                    gate.BUILDER_AUDIT_SCOPE if name == "builder" else gate.AUDIT_SCOPE
                ), "writes": [], **(
                    {"observed_subprocesses": processes} if name == "builder"
                    else {"dispatch_attempts": []}
                ),
            } for name in gate.AUDITS
        }
        review = {
            "checkpoint": {"path": "/synthetic/checkpoint"},
            "created_at": 1, "frontier": {"path": "/synthetic/frontier"},
            "kind": "round_reviewed_handoff", "mission_context": "/synthetic/mission",
            "mission_id": gate.MISSION_ID, "producer_role": "reviewer",
            "review": {
                "next_action": "", "operator_question": "", "reason": "synthetic only",
                "status": "done",
            }, "round": 1, "schema_version": 3,
            "release_consumption": {
                "decision": "ADMISSIBLE", "package_census": None,
                "historical_failure_ids": [
                    r["failure_id"] for r in history["individual_failure_evidence"]
                ],
                "write_audit_limits": {
                    "accepted": True, "scope": gate.AUDIT_SCOPE,
                    "unobserved_finalization_writes": [
                        str(root / name) for name in gate.FINALIZATION_WRITES
                    ],
                },
            },
        }
        if change:
            change(inputs, result, phases, audits, review)
        for phase in gate.PHASES:
            directory = root / f"{phase}-capture"
            streams = [
                put(directory / "launcher.identity.json", phases[phase]["actual_identity"]),
                put(directory / "launcher.stdout", phases[phase]),
                put(directory / "launcher.stderr", b""),
                put(directory / "launcher.whole-command.log", b"synthetic retained log"),
            ]
            terminal = {
                "capture_implementation_after": implementation, "exit_status": 0,
                "files": streams, "success": True, "timed_out": False,
            }
            put(directory / "launcher.capture.json", terminal)
            put(directory / "argv.json", phases[phase]["actual_identity"]["argv"])
            put(directory / "environment.json", inputs["environment"])
            put(directory / "command.txt", b"synthetic retained command")
            verification = {
                "started_at": 1, "finished_at": 2, "implementation_before": implementation,
                "implementation_after": implementation, "implementation_unchanged": True,
                "stdout": streams[1], "terminal": terminal,
            }
            put(directory / "verification.json", verification)
            result["commands"][phase] = verification
        put(root / "release-inputs.json", inputs)
        put(root / "operational-result.json", result)
        for name, audit in audits.items():
            put(root / f"{name}.write-audit.json", audit)
        census = put(root / "final-census.json", {
            "mission_id": gate.MISSION_ID,
            "members": [pin(p, raw) for p, raw in files.items() if p.is_relative_to(root)],
            "excludes": ["final-census.json (its exact pin belongs in the Host checkpoint)"],
            "review_status": "PENDING_NORMAL_INDEPENDENT_REVIEW",
        })
        if "release_consumption" in review:
            review["release_consumption"]["package_census"] = census
        reviewed = put(repository / "review.json", review)
        return dict(package_root=root, repository_root=repository, census_pin=census, review_pin=reviewed)

    def read(path):
        if path not in files:
            raise FileNotFoundError(str(path))
        return files[path]

    monkeypatch.setattr(gate, "_read_bytes", read)
    monkeypatch.setattr(gate, "_inventory", lambda directory: {
        p for p in files if p.is_relative_to(directory)
    })
    return make, files, root, repository


def assert_blocked(report, check):
    assert report["status"] == "UNKNOWN"
    assert report["blocked"] is True
    assert report["release_consumption_admissible"] is False
    assert report["release_consumption_authorized"] is False
    assert report["checks"][check] is False
    assert any(b["gate"] == check and b["reason"] for b in report["blockers"])


def test_synthetic_pass_is_not_authority(synthetic):
    make, files, _, _ = synthetic
    arguments = make()
    before = deepcopy(files)
    report = gate.preflight_release(**arguments)
    assert report["status"] == "PASS", report
    assert not report["blocked"]
    assert report["release_consumption_admissible"] is True
    assert report["release_consumption_authorized"] is False
    assert all(report["checks"].values())
    assert report["blockers"] == []
    assert report["scientific_invocations"] == report["candidate_check_invocations"] == 0
    assert report["package_validation_invocations"] == 0
    assert report["dispatch_attempts"] == []
    assert files == before


@pytest.mark.parametrize("target", ["census", "review", "member", "source", "capture"])
@pytest.mark.parametrize("change", ["missing", "changed"])
def test_missing_or_changed_pins(synthetic, target, change):
    make, files, root, repository = synthetic
    arguments = make()
    path = {
        "census": root / "final-census.json", "review": repository / "review.json",
        "member": root / "operational.py", "source": repository / "source-0.py",
        "capture": root / "validate-capture/launcher.stderr",
    }[target]
    if change == "missing":
        del files[path]
    else:
        files[path] += b"\nchanged"
    expected = {"review": "independent_review", "source": "source_pins"}.get(target, "package_pins")
    assert_blocked(gate.preflight_release(**arguments), expected)


@pytest.mark.parametrize("value", [1, -1, True, False, "0", 0.0, None])
@pytest.mark.parametrize("counter", ["scientific_invocations", "candidate_check_invocations"])
@pytest.mark.parametrize("level", ["result", "compile", "tests", "validate"])
def test_nonzero_or_noninteger_counters(synthetic, value, counter, level):
    make, _, _, _ = synthetic

    def change(inputs, result, phases, audits, review):
        (result if level == "result" else phases[level])[counter] = value

    assert_blocked(gate.preflight_release(**make(change)), "zero_science_and_candidate_checks")


@pytest.mark.parametrize("case", [
    "unknown", "no-review", "engineer-review", "review-not-done", "no-release-review",
    "unresolved-six", "five-closures", "duplicate-failure", "unreviewed-failure",
    "audit-not-accepted", "audit-overclaim", "audit-omission", "outside-write",
    "validation-write", "candidate-dispatch", "science-dispatch", "source-role",
    "runtime-field", "unknown-field", "identity", "compile-failed",
])
def test_semantically_inadmissible_pinned_evidence(synthetic, case):
    make, files, root, repository = synthetic
    expected = {
        "unknown": "package_status", "no-review": "independent_review",
        "engineer-review": "independent_review", "review-not-done": "independent_review",
        "no-release-review": "release_review", "unresolved-six": "historical_failure_closure",
        "five-closures": "historical_failure_closure", "duplicate-failure": "historical_failure_closure",
        "unreviewed-failure": "historical_failure_closure",
        "audit-not-accepted": "write_audit_limits", "audit-overclaim": "write_audit_limits",
        "audit-omission": "write_audit_limits", "outside-write": "write_audits",
        "validation-write": "write_audits", "candidate-dispatch": "write_audits",
        "science-dispatch": "zero_science_and_candidate_checks", "source-role": "source_pins",
        "runtime-field": "source_pins", "unknown-field": "schemas",
        "identity": "retained_captures", "compile-failed": "package_status",
    }[case]

    def change(inputs, result, phases, audits, review):
        history = inputs["historical_six_gate_failures"]
        if case == "unknown":
            phases["validate"]["status"] = "UNKNOWN"
        elif case == "engineer-review":
            review["producer_role"] = "engineer"
        elif case == "review-not-done":
            review["review"]["status"] = "blocked"
        elif case == "no-release-review":
            del review["release_consumption"]
        elif case == "unresolved-six":
            history.update(status="RETAINED_UNRESOLVED", individual_failure_evidence=None)
        elif case == "five-closures":
            history["individual_failure_evidence"].pop()
        elif case == "duplicate-failure":
            history["individual_failure_evidence"][-1] = history["individual_failure_evidence"][0]
        elif case == "unreviewed-failure":
            review["release_consumption"]["historical_failure_ids"][-1] = "another-failure"
        elif case == "audit-not-accepted":
            review["release_consumption"]["write_audit_limits"]["accepted"] = False
        elif case == "audit-overclaim":
            review["release_consumption"]["write_audit_limits"]["scope"] = "OS-wide complete"
        elif case == "audit-omission":
            review["release_consumption"]["write_audit_limits"]["unobserved_finalization_writes"].pop()
        elif case == "outside-write":
            audits["builder"]["writes"] = [{"event": "open", "path": str(root) + "-other/file"}]
        elif case == "validation-write":
            audits["validate"]["writes"] = [{"event": "open", "path": str(root / "file")}]
        elif case == "candidate-dispatch":
            audits["validate"]["dispatch_attempts"] = ["candidate --check"]
        elif case == "science-dispatch":
            phases["tests"]["dispatch_attempts"] = ["suffix"]
        elif case == "source-role":
            inputs["sources"].pop()
        elif case == "runtime-field":
            inputs["authorization_action"] = "activate"
        elif case == "unknown-field":
            result["runtime_activation"] = True
        elif case == "identity":
            phases["compile"]["actual_identity"]["uid"] = False
        elif case == "compile-failed":
            phases["compile"]["status"] = "FAIL"

    arguments = make(change)
    if case == "no-review":
        del files[repository / "review.json"]
    assert_blocked(gate.preflight_release(**arguments), expected)


def test_unpinned_extra_file(synthetic):
    make, files, root, _ = synthetic
    arguments = make()
    files[root / "unreviewed.py"] = b"extra"
    assert_blocked(gate.preflight_release(**arguments), "package_pins")


@pytest.mark.parametrize("event,name,admissible", [
    ("open", "focused-tests.txt", True),
    ("open", "tmp/test-output", True),
    ("open", "unrelated.txt", False),
    ("os.mkdir", "focused-tests.txt", False),
])
def test_focused_test_write_limits(synthetic, event, name, admissible):
    make, _, root, _ = synthetic

    def change(inputs, result, phases, audits, review):
        audits["tests"]["writes"] = [{"event": event, "path": str(root / name)}]

    report = gate.preflight_release(**make(change))
    if admissible:
        assert report["status"] == "PASS", report
    else:
        assert_blocked(report, "write_audits")


@pytest.mark.parametrize("key", ["bytes", "sha256"])
def test_absent_source_pin_field(synthetic, key):
    make, _, _, _ = synthetic

    def change(inputs, result, phases, audits, review):
        del inputs["sources"][0][key]

    assert_blocked(gate.preflight_release(**make(change)), "source_pins")


def test_duplicate_json_field_rejected(synthetic):
    make, files, root, _ = synthetic
    arguments = make()
    path = root / "final-census.json"
    files[path] = files[path][:-1] + b', "mission_id": "016efb266000"}'
    arguments["census_pin"] = pin(path, files[path])
    assert_blocked(gate.preflight_release(**arguments), "package_pins")


def test_current_frozen_unknown_and_no_forbidden_dispatch(monkeypatch, capsys):
    import builtins
    import os
    import socket
    import subprocess

    paths = gate._inventory(gate.PACKAGE_ROOT)
    before = {path: pin(path, path.read_bytes()) for path in paths}
    original_import = builtins.__import__

    def deny(*args, **kwargs):
        pytest.fail("read-only preflight attempted a forbidden dispatch or write")

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("argus_skill", "numpy", "torch", "ace3")), name
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        guard.setattr(subprocess, "Popen", deny)
        guard.setattr(os, "system", deny)
        guard.setattr(os, "posix_spawn", deny)
        guard.setattr(socket, "socket", deny)
        guard.setattr(Path, "write_bytes", deny)
        guard.setattr(Path, "write_text", deny)
        report = gate.preflight_release()
    assert report["package_status"] == "UNKNOWN", report
    assert report["independent_review_status"] == "done", report
    for check in ("package_status", "historical_failure_closure", "release_review", "write_audit_limits"):
        assert_blocked(report, check)
    for check in (
        "package_pins", "schemas", "source_pins", "zero_science_and_candidate_checks",
        "retained_captures", "independent_review", "write_audits",
    ):
        assert report["checks"][check], report
    assert report["dispatch_attempts"] == []
    assert report["scientific_invocations"] == report["candidate_check_invocations"] == 0
    assert report["package_validation_invocations"] == 0
    assert paths == gate._inventory(gate.PACKAGE_ROOT)
    assert before == {path: pin(path, path.read_bytes()) for path in paths}
    with capsys.disabled():
        print("Frozen 016: blocked/UNKNOWN; six-failure closure and release review absent; bytes unchanged.")


def test_cli_unknown_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(gate, "preflight_release", lambda: {"status": "UNKNOWN", "blocked": True})
    assert gate.main([]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "UNKNOWN"


@pytest.mark.parametrize("arguments", [["--check"], ["validate"], ["--run"], ["--output", "file"]])
def test_cli_has_no_dispatch_or_write_options(monkeypatch, arguments):
    def deny():
        pytest.fail("invalid CLI action reached evidence consumption")

    monkeypatch.setattr(gate, "preflight_release", deny)
    with pytest.raises(SystemExit) as error:
        gate.main(arguments)
    assert error.value.code == 2


if __name__ == "__main__":
    for source in (SOURCE, Path(__file__).resolve()):
        compile(source.read_bytes(), str(source), "exec")
    print("Targeted compilation: PASS (two files, in memory)")
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "-p", "no:cacheprovider",
        "--capture=sys", "--assert=plain", "--noconftest",
    ]))
