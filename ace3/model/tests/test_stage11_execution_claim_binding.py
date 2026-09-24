"""Non-generative Manager selection and mandatory captured-launch claim binding."""

import ast
import copy
import inspect
import json
import os
import socket
import subprocess

import pytest

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage11_attention_output_suffix_candidate_v1 as d


@pytest.fixture(autouse=True)
def no_dispatch(monkeypatch, record_property):
    calls = {"scientific": 0, "model": 0, "producer": 0, "service": 0}

    def deny(kind):
        def forbidden(*args, **kwargs):
            calls[kind] += 1
            raise AssertionError("forbidden invocation: " + kind)
        return forbidden

    monkeypatch.setattr(d, "check", deny("scientific"))
    monkeypatch.setattr(d, "_load_runtime", deny("model"))
    monkeypatch.setattr(d.np, "load", deny("model"))
    monkeypatch.setattr(subprocess, "Popen", deny("producer"))
    monkeypatch.setattr(os, "system", deny("producer"))
    monkeypatch.setattr(socket, "socket", deny("service"))
    yield
    for kind, count in calls.items():
        record_property("forbidden_" + kind, count)
    assert not any(calls.values())


@pytest.fixture
def authority(tmp_path):
    current = {"id": "current-mission", "status": "running", "started_ts": 1700000000.0,
               "node_key": "synthetic-stage11", "attempt": 2}
    backlog = tmp_path / "backlog.jsonl"
    backlog.write_text(json.dumps(current) + "\n")
    constraints = {"budget": {"scientific_suffix_runs": 1}, "concurrency": {"max_active": 1}}
    snapshot = {"schema_version": 1, "kind": "native_running_claim_snapshot",
                "mission_id": current["id"], "status": "running",
                "started_ts": current["started_ts"], "constraints": constraints,
                "observed_at_utc": "2023-11-14T22:13:21+00:00",
                "source": {"backlog": str(backlog)}}

    def save(name, value):
        return d.capture.save(tmp_path, name, d.capture.encoded(value))

    claim = save("native-running-claim.json", snapshot)
    accepted = save("synthetic-accepted.json", {"synthetic": True})
    declaration = {
        "schema_version": 1, "kind": "stage11_suffix_execution_authorization",
        "issuer": "manager", "authorized": True, "scientific_run_budget": 1,
        "issued_at_utc": "2023-11-14T22:13:22+00:00",
        "launch": {"mission_id": current["id"], "constraints": constraints},
        "claim": {"pin": claim, "mission_path": ["mission_id"], "status_path": ["status"],
                  "constraints_path": ["constraints"]},
        "accepted_identity": {"receipt": accepted}, "accepted_proposal": accepted}
    issuance = {
        "schema_version": 1, "kind": "manager_stage11_execution_issuance", "issuer": "manager",
        "authorized_science": True, "authorized_before_running_claim": False,
        "scientific_run_budget": 1, "declaration": save("execution-authorization.json", declaration),
        "claim": claim, "mission_id": current["id"],
        "accepted_identity_receipt": accepted, "accepted_proposal": accepted}
    envelope = {"declaration": issuance["declaration"],
                "issuance": save("execution-issuance.json", issuance)}
    return current, backlog, snapshot, declaration, issuance, envelope


def rebind(pin, value):
    d.Path(pin["path"]).write_bytes(d.capture.encoded(value))
    return d.capture.binding(pin["path"])


def select(authority, catalog=None):
    current, backlog, _, _, _, envelope = authority
    return d.select_execution_authorization(
        {current["id"]: envelope} if catalog is None else catalog,
        current_claim=current, backlog=backlog)


def test_current_claim_selects_exact_pins_without_writes_or_budget_reset(authority):
    current, backlog, _, declaration, issuance, envelope = authority
    before = {p: p.read_bytes() for p in backlog.parent.iterdir()}
    assert select(authority, {"old-mission": {}, current["id"]: envelope}) is envelope
    assert declaration["scientific_run_budget"] == issuance["scientific_run_budget"] == 1
    assert before == {p: p.read_bytes() for p in backlog.parent.iterdir()}
    assert d.validate_launch_claim({
        "mission_id": current["id"], "normal_running_claim": current,
        "native_backlog": str(backlog), "execution_authorization": envelope}) is envelope


def test_no_latest_authority_fallback(authority):
    with pytest.raises(ValueError, match="unavailable for current mission"):
        select(authority, {"old-mission": authority[-1]})


@pytest.mark.parametrize("defect", [
    "mission", "snapshot_mission", "retry", "done", "missing", "context_start",
    "context_metadata", "snapshot_start", "backlog", "before_claim", "claim_pin",
    "issuance_mission", "issuance_claim",
])
def test_stale_mismatched_or_rebound_claim_is_rejected(authority, defect):
    current, backlog, snapshot, declaration, issuance, envelope = authority
    if defect in ("retry", "done", "missing"):
        row = copy.deepcopy(current)
        row.update({"retry": {"started_ts": current["started_ts"] + 100},
                    "done": {"status": "done"}, "missing": {"id": "other"}}[defect])
        backlog.write_text(json.dumps(row) + "\n")
    elif defect == "context_start":
        current["started_ts"] += 100
    elif defect == "context_metadata":
        current["attempt"] += 1
    elif defect == "mission":
        declaration["launch"]["mission_id"] = issuance["mission_id"] = "old-mission"
    elif defect in ("snapshot_mission", "snapshot_start", "backlog"):
        if defect == "snapshot_mission":
            snapshot["mission_id"] = "old-mission"
        elif defect == "snapshot_start":
            snapshot["started_ts"] -= 100
        else:
            other = backlog.with_name("other-backlog.jsonl")
            other.write_bytes(backlog.read_bytes())
            snapshot["source"]["backlog"] = str(other)
        declaration["claim"]["pin"] = rebind(declaration["claim"]["pin"], snapshot)
        issuance["claim"] = declaration["claim"]["pin"]
    elif defect == "before_claim":
        declaration["issued_at_utc"] = "2023-11-14T22:13:20+00:00"
    elif defect == "claim_pin":
        d.Path(declaration["claim"]["pin"]["path"]).write_bytes(b"{}\n")
    elif defect == "issuance_mission":
        issuance["mission_id"] = "old-mission"
    else:
        issuance["claim"] = declaration["accepted_proposal"]
    envelope["declaration"] = rebind(envelope["declaration"], declaration)
    issuance["declaration"] = envelope["declaration"]
    envelope["issuance"] = rebind(envelope["issuance"], issuance)
    with pytest.raises((ValueError, RuntimeError)):
        select(authority)


@pytest.mark.parametrize("owner", ["declaration", "issuance"])
@pytest.mark.parametrize("budget", [None, False, True, 0, 2, 1.0, "1"])
def test_selection_cannot_restore_or_expand_one_shot_budget(authority, owner, budget):
    _, _, _, declaration, issuance, envelope = authority
    document = declaration if owner == "declaration" else issuance
    if budget is None:
        del document["scientific_run_budget"]
    else:
        document["scientific_run_budget"] = budget
    envelope["declaration"] = rebind(envelope["declaration"], declaration)
    issuance["declaration"] = envelope["declaration"]
    envelope["issuance"] = rebind(envelope["issuance"], issuance)
    with pytest.raises((ValueError, KeyError)):
        select(authority)


@pytest.mark.parametrize("missing", ["normal_running_claim", "native_backlog", "mission_id"])
def test_captured_check_requires_runtime_claim_context(authority, missing):
    current, backlog, _, _, _, envelope = authority
    preflight = {"mission_id": current["id"], "normal_running_claim": current,
                 "native_backlog": str(backlog), "execution_authorization": envelope}
    del preflight[missing]
    with pytest.raises(d.LaunchClaimError, match="KeyError"):
        d.validate_launch_claim(preflight)


def test_captured_claim_gate_precedes_remaining_authorization():
    tree = ast.parse(inspect.getsource(d.execution_authorization))
    calls = [node.value.func.id for node in tree.body[0].body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name)]
    assert "validate_launch_claim" in calls
    source = inspect.getsource(d.execution_authorization)
    assert source.index("validate_launch_claim(preflight)") < source.index(
        "validate_execution_authorization(preflight)")


def test_claim_change_during_selection_fails_closed(authority, monkeypatch):
    current, backlog, _, _, _, _ = authority
    original = d._execution_declaration

    def change_claim(envelope):
        result = original(envelope)
        changed = {**current, "attempt": current["attempt"] + 1}
        backlog.write_text(json.dumps(changed) + "\n")
        return result

    monkeypatch.setattr(d, "_execution_declaration", change_claim)
    with pytest.raises(ValueError, match="changed during authorization selection"):
        select(authority)


@pytest.mark.parametrize("field", ["notes", "last_error"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_host_text_bookkeeping_drift_preserves_claim(authority, monkeypatch, field, when):
    current, backlog, _, _, _, envelope = authority
    changed = {**current, field: "Host diagnostic updated"}
    original = d._execution_declaration

    def update(envelope):
        result = original(envelope)
        backlog.write_text(json.dumps(changed) + "\n")
        return result

    if when == "before":
        backlog.write_text(json.dumps(changed) + "\n")
    else:
        monkeypatch.setattr(d, "_execution_declaration", update)
    assert select(authority) is envelope
    assert field not in current
    assert json.loads(backlog.read_text()) == changed


@pytest.mark.parametrize("field,value", [
    ("id", "other"), ("status", "done"), ("started_ts", 1700000001.0),
    ("attempt", 3), ("authorization_id", "other"), ("authorization_action", "other"),
    ("node_key", "other"), ("plan_id", "other"), ("plan_version", 2),
    ("objective", "broader scope"), ("owns_paths", ["other"]),
    ("execution_workdir", "/other"), ("constraints", {"budget": 2}),
    ("budget", {"remaining": 2}), ("iteration_cost_usd", 1),
    ("iteration_cycles_done", 1), ("iteration_max_cycles", 7),
    ("access", {"service": True}), ("concurrency", {"max_active": 2}),
    ("parallel_safe", True), ("running_owner", "other"),
    ("sources", ["changed"]), ("review", {"required": False}),
    ("manager_decision", {"routed": False}), ("operator_decision", {"allow": True}),
    ("outcome", {"review_status": "done"}), ("finished_ts", 1700000001.0),
    ("future_authority_field", None), ("attempt", 2.0),
])
def test_stable_claim_drift_with_bookkeeping_is_rejected(authority, field, value):
    current, backlog, _, _, _, envelope = authority
    changed = {**current, "notes": "Host diagnostic updated", field: value}
    backlog.write_text(json.dumps(changed) + "\n")
    with pytest.raises(RuntimeError) as caught:
        d.validate_launch_claim({
            "mission_id": current["id"], "normal_running_claim": current,
            "native_backlog": str(backlog), "execution_authorization": envelope})
    assert isinstance(caught.value, ValueError)
    failure = json.loads(str(caught.value))
    assert failure["message"]
    if field not in ("id", "status"):
        diff = failure["claim_differences"][field]
        assert diff["authorized_present"] is (field in current)
        assert diff["live_present"] is True
        assert diff["live"] == value
        assert failure["claim_differences"]["notes"]["live"] == changed["notes"]


@pytest.mark.parametrize("field", ["notes", "last_error"])
def test_bookkeeping_cannot_hide_structured_authority(authority, field):
    current, backlog, _, _, _, _ = authority
    current[field] = {"budget": 2}
    backlog.write_text(json.dumps(current) + "\n")
    with pytest.raises(ValueError, match="bookkeeping must be text"):
        select(authority)


@pytest.mark.parametrize("change", ["bookkeeping", "attempt", "missing"])
def test_retained_native_record_binds_stable_identity(authority, change):
    current, backlog, snapshot, declaration, issuance, envelope = authority
    snapshot["native_record"] = copy.deepcopy(current)
    if change == "bookkeeping":
        current["last_error"] = "Host diagnostic updated"
    elif change == "attempt":
        current["attempt"] += 1
    else:
        del current["node_key"]
    backlog.write_text(json.dumps(current) + "\n")
    declaration["claim"]["pin"] = rebind(declaration["claim"]["pin"], snapshot)
    issuance["claim"] = declaration["claim"]["pin"]
    envelope["declaration"] = rebind(envelope["declaration"], declaration)
    issuance["declaration"] = envelope["declaration"]
    envelope["issuance"] = rebind(envelope["issuance"], issuance)
    if change == "bookkeeping":
        assert select(authority) is envelope
    else:
        with pytest.raises(d.LaunchClaimError, match="Manager snapshot native claim changed"):
            select(authority)
