"""Synthetic data-only handoff tests; no operational build, observer, or candidate."""

import ast
import builtins
from copy import deepcopy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "ace3/model/stage11_forward_handoff_repair.py"
SPEC = importlib.util.spec_from_file_location("forward_handoff", SOURCE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
NOW = datetime.datetime(2026, 9, 22, 9, tzinfo=datetime.timezone.utc)


def authority():
    return {
        "schema": "argus.stage11-dataonly-forward-handoff-authorization.v1",
        "authority_id": gate.AUTHORITY, "mission_id": gate.MISSION,
        "authorized_root": gate.ROOT, "issuer": "manager",
        "decision": "AUTHORIZE_ZERO_SCIENCE_DATA_ONLY_FORWARD_HANDOFF_REPAIR",
        "expected_attempt": 1, "authorized_science": False, "scientific_run_budget": 0,
        "issued_at_utc": "2026-09-22T08:00:00+00:00",
        "valid_until_utc": "2026-09-22T10:00:00+00:00",
    }


def test_exact_authority():
    gate.check_authority(authority(), NOW)


@pytest.mark.parametrize("field,value", [
    ("authority_id", "other"), ("mission_id", "other"),
    ("authorized_root", "build/other"), ("expected_attempt", True),
    ("authorized_science", True), ("scientific_run_budget", False),
    ("scientific_run_budget", 1), ("issuer", "engineer"),
    ("valid_until_utc", "2026-09-22T08:30:00+00:00"),
])
def test_authority_rejects_conflicts(field, value):
    document = authority()
    document[field] = value
    with pytest.raises(ValueError):
        gate.check_authority(document, NOW)


def frozen_report():
    return {
        "mission_id": "016efb266000",
        "status": "UNKNOWN", "package_status": "UNKNOWN", "blocked": True,
        "release_consumption_admissible": False, "release_consumption_authorized": False,
        "scientific_invocations": 0, "candidate_check_invocations": 0,
        "package_validation_invocations": 0, "dispatch_attempts": [],
        "blockers": [
            {"gate": name, "reason": name}
            for name in (
                "package_status", "release_review",
                "historical_failure_closure", "write_audit_limits",
            )
        ],
    }


def test_exact_blocker_does_not_promote_history():
    report = frozen_report()
    original = deepcopy(report)
    result = gate.terminal_blocker(report)
    assert result["code"] == "frozen_016_release_consumption_inadmissible"
    assert result["frozen_016_blockers"] == report["blockers"]
    assert report == original


@pytest.mark.parametrize("field,value", [
    ("status", "PASS"), ("package_status", "PASS"),
    ("release_consumption_admissible", True),
    ("release_consumption_authorized", True), ("blockers", []),
    ("mission_id", "other"), ("scientific_invocations", 1),
    ("candidate_check_invocations", 1), ("package_validation_invocations", True),
    ("dispatch_attempts", ["attempt"]),
])
def test_changed_frozen_evidence_is_not_silently_accepted(field, value):
    report = frozen_report()
    report[field] = value
    with pytest.raises(ValueError):
        gate.terminal_blocker(report)


def claim():
    return {
        "schema": "argus.manager-postclaim-owner.v2",
        "state": "BOUND_TO_NATIVE_RUNNING_CLAIM",
        "authority_id": gate.AUTHORITY, "mission_id": gate.MISSION,
        "authorized_root": gate.ROOT, "issuer": "manager", "executing_role": "engineer",
        "manager_inputs": {}, "science_issuance": None, "scientific_run_budget": 0,
        "preflight_status": "UNKNOWN", "release_consumption_authorized": False,
        "release_consumption_admissible": False, "published_at": 3,
        "native_claim": {
            "status": "running", "attempt": 1, "started_ts": 1, "mission_started_ts": 2,
            "node_key": "stage11-dataonly-handoff-correction", "independent_review_required": True,
        },
    }


def test_postclaim_is_required_before_edits():
    document = dict(authority(), manager_inputs={})
    gate.check_claim(claim(), document, [4_000_000_000, 5_000_000_000])
    with pytest.raises(ValueError, match="predate"):
        gate.check_claim(claim(), document, [2_000_000_000, 5_000_000_000])


@pytest.mark.parametrize("field,value", [
    ("state", "PREPARED_BEFORE_PLANNER_HANDOFF"), ("mission_id", "558580af5d32"),
    ("science_issuance", {}), ("scientific_run_budget", True),
    ("release_consumption_authorized", True), ("manager_inputs", {"other": {}}),
])
def test_claim_conflicts_fail_closed(field, value):
    owner = claim()
    owner[field] = value
    with pytest.raises(ValueError):
        gate.check_claim(owner, dict(authority(), manager_inputs={}), [4_000_000_000])


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def manager_input(tmp_path):
    report = frozen_report()
    retained = tmp_path / "retained-preflight.json"
    retained.write_text(json.dumps(report, indent=3) + "\n")
    document = {
        "schema": "argus.manager-supplied-preflight-report.v1",
        "mission_id": gate.MISSION, "generated_by": "manager",
        "decision": "RETAIN_UNKNOWN_AS_DATA_ONLY", "status": "UNKNOWN",
        "source": pin(retained), "report": report,
    }
    path = tmp_path / "manager-preflight-report.json"
    path.write_text(json.dumps(document))
    return {"manager_inputs": {"preflight_report": pin(path)}}, document, path, retained


def test_consumer_has_no_candidate_or_dynamic_preflight_loader():
    tree = ast.parse(SOURCE.read_bytes())
    allowed_imports = {
        "datetime", "hashlib", "json", "os", "pathlib", "subprocess", "sys", "time",
    }
    forbidden_calls = {
        "preflight_release", "exec_module", "load_module", "spec_from_file_location",
        "module_from_spec", "import_module", "__import__", "exec", "eval",
        "run_path", "run_module",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed_imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed_imports
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in forbidden_calls, f"forbidden executable loader: {name}"
            if name in {"run", "Popen", "system", "popen"}:
                assert name == "run" and ast.unparse(node.func) == "subprocess.run"
                assert ast.unparse(node.args[0]) == "command"
    commands = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets)
    ]
    assert len(commands) == 1 and isinstance(commands[0], ast.List)
    assert [ast.unparse(item) for item in commands[0].elts] == [
        "sys.executable", "'-B'", "'-m'", "'pytest'", "'-q'", "'-p'", "'no:cacheprovider'", "TEST",
    ]


def test_preflight_is_only_authenticated_json_no_import_or_execution(tmp_path, monkeypatch):
    document, _, path, retained = manager_input(tmp_path)
    original_import = builtins.__import__
    events = []

    def guarded_import(name, *args, **kwargs):
        assert "candidate" not in name and "preflight" not in name, name
        return original_import(name, *args, **kwargs)

    def forbidden_loader(*args, **kwargs):
        pytest.fail("candidate/preflight dynamic module execution attempted")

    def trace(frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            assert "/candidates/" not in filename
            assert frame.f_code.co_name != "preflight_release"
            assert filename not in {str(path), str(retained)}
            events.append(frame.f_code.co_name)
        return trace

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", forbidden_loader)
    monkeypatch.setattr(importlib.util, "module_from_spec", forbidden_loader)
    monkeypatch.setattr(gate.subprocess, "run", forbidden_loader)
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        raw, retained_raw, report = gate.load_manager_preflight(document)
    finally:
        sys.settrace(previous_trace)
    assert "load_manager_preflight" in events
    assert raw == path.read_bytes() and retained_raw == retained.read_bytes()
    assert report == frozen_report()
    assert gate.terminal_blocker(report)["frozen_016_blockers"] == report["blockers"]


@pytest.mark.parametrize("target", ["manager", "retained"])
def test_report_byte_tampering_is_rejected(tmp_path, target):
    document, _, path, retained = manager_input(tmp_path)
    changed = path if target == "manager" else retained
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="authentication"):
        gate.load_manager_preflight(document)


def test_authenticated_wrapper_cannot_replace_retained_report(tmp_path):
    document, wrapper, path, _ = manager_input(tmp_path)
    wrapper["report"]["status"] = "PASS"
    path.write_text(json.dumps(wrapper))
    document["manager_inputs"]["preflight_report"] = pin(path)
    with pytest.raises(ValueError, match="differs from retained JSON"):
        gate.load_manager_preflight(document)
