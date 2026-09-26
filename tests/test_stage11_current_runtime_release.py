import copy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

import pytest

from ace3.model import stage11_current_runtime_release as release


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    before = {name for name in sys.modules if ".candidates." in name}

    def forbidden(*args, **kwargs):
        raise AssertionError("emitter, science, service, or subprocess dispatch is forbidden")

    monkeypatch.setattr(release, "emit_after_claim", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    yield
    assert {name for name in sys.modules if ".candidates." in name} == before


@pytest.fixture(scope="session")
def production():
    inputs = release.proposal_context()
    contract = release.document(inputs["accepted_construction"]["contract"])
    run = contract["accepted_lineage"]
    current = release.current_source_pins()
    contract["current_sources"] = current
    contract["capture_snapshots"] = {
        key: release.pin_bytes(release.REPOSITORY / release.ROOT / name,
                               release.authenticate(current[key]))
        for key, name in (("capture_helper", "capture-source.py"),
                          ("capture_tests", "capture-tests.py"))
    }
    contract["manager_proposal"] = release.CONSUMPTION_INPUT
    contract["runtime_sources"] = inputs["expected_runtime_sources"]
    contract["reference_members"] = inputs["expected_reference_members"]
    contract["construction_provenance"] = inputs["accepted_construction"]
    contract["artifact_review_provenance"] = inputs["accepted_artifact_review"]
    for role, key in (("current_diagnostic", "candidate"), ("current_tests", "tests")):
        contract["execution_contract"]["sources"][role] = current[key]
        contract["source_snapshots"][role] = current[key]
    contract["source_roles"]["current_diagnostic"] = {
        "source": current["candidate"], "retained": current["candidate"],
    }
    # Extract pinned provenance once for isolated tests; the integration test
    # still authenticates the complete real native histories without caching.
    native_bytes = {}
    terminal = release.construction_input(inputs)["accepted_provenance"]["native_terminal"]
    for name, key, digest_key in (
        ("backlog.jsonl", "id", "backlog_row_sha256"),
        ("events.jsonl", "item_id", "mission_completed_event_sha256"),
    ):
        path = release.LIFE / name
        matches = []
        for raw in path.read_bytes().splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get(key) == release.PROVENANCE_MISSION and (
                key == "id" or value.get("type") == "life.mission.completed"
            ):
                matches.append(raw)
        assert len(matches) == 1
        assert release.digest(matches[0]) == terminal[digest_key]
        retained = []
        for raw in path.read_bytes().splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get(key) in (
                release.CONSTRUCTION_MISSION, release.ARTIFACT_REVIEW_MISSION, "ab65cb0e6bea"
            ) or release.digest(raw) == inputs["accepted_artifact_review"]["review_completed_event_sha256"]:
                retained.append(raw)
        native_bytes[path] = b"\n".join(matches + retained) + b"\n"
    return run, contract, native_bytes


def test_actual_authenticated_33_file_production_lineage(production):
    run, contract, _ = production
    assert len(contract["accepted_files"]) == 33
    assert release.accepted_lineage(run, contract) == release.document(run["proposal"])
    assert set(run) == {
        "authority_bundle", "launcher_capture", "launcher_identity",
        "launcher_expected_identity", "terminal_receipt", "proposal", "receipt_index",
    }
    assert release.document(run["launcher_capture"])["success"] is True


def test_fresh_proposed_contract_consumes_production(production, tmp_path):
    proposal = release.CONSUMPTION_INPUT
    accepted = release.proposal_context()["accepted_construction"]
    for (_, _, name), source in zip(release.SOURCE_LAYOUT, release.implementation_pins(), strict=True):
        (tmp_path / name).write_bytes(release.authenticate(source))
    contract = release.proposed_contract(
        release.document(accepted["contract"]), proposal, tmp_path, time.time(),
        production[1]["runtime"])
    assert contract["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
    assert contract["scientific_run_budget"] == 0
    assert contract["mission_id"] == release.MISSION
    assert contract["inherits_operational_admission"] is False
    assert contract["current_sources"] == production[1]["current_sources"]
    assert contract["implementation_sources"] == release.implementation_pins()
    assert len(contract["implementation_sources"]) == 6
    for key in ("capture_helper", "capture_tests"):
        assert release.authenticate(contract["capture_snapshots"][key]) == release.authenticate(
            contract["current_sources"][key])
    previous = release.document(accepted["contract"])
    for key in ("identity", "launch", "launch_constraints", "accepted_lineage",
                "original_scientific_lineage", "operational_provenance"):
        assert contract[key] == previous[key]
    execution = copy.deepcopy(previous["execution_contract"])
    execution["sources"]["current_diagnostic"] = contract["current_sources"]["candidate"]
    execution["sources"]["current_tests"] = contract["current_sources"]["tests"]
    assert contract["execution_contract"] == execution
    assert contract["runtime_sources"] == release.proposal_context()["expected_runtime_sources"]
    assert contract["reference_members"] == release.REFERENCE_MEMBERS
    assert contract["accepted_current_runtime"]["mission_id"] == "04c9ee209234"
    assert release.accepted_lineage(production[0], contract)["identity"] == contract["identity"]
    properties = "".join(
        '<property name="forbidden_' + kind + '" value="0"/>'
        for kind in release.ZERO_KINDS)
    raw = ('<testsuites><testsuite tests="293" failures="0" errors="0" skipped="0">'
           '<testcase classname="test_diagnose_candidate"><properties>' + properties
           + '</properties></testcase>' + '<testcase classname="release"/>' * 292
           + '</testsuite></testsuites>').encode()
    assert len(release.test_summary(raw, b"293 passed in 1.00s\n")) == len(release.ZERO_KINDS)
    assert len(release.test_summary(raw, b"293 passed in 61.00s (0:01:01)\n")) == len(release.ZERO_KINDS)
    for before, after in (
        (b'tests="293"', b'tests="292"'), (b'tests="293"', b'tests="294"'),
        (b'failures="0"', b'failures="1"'), (b'errors="0"', b'errors="1"'),
        (b'skipped="0"', b'skipped="1"'), (b'value="0"', b'value="1"'),
        (b'forbidden_model', b'missing_model'),
    ):
        with pytest.raises(ValueError):
            release.test_summary(raw.replace(before, after), b"293 passed in 1.00s\n")
    with pytest.raises(ValueError):
        release.test_summary(raw, b"293 passed, 1 error in 1.00s\n")


@pytest.fixture
def case(production, tmp_path, monkeypatch):
    run, original, provenance_native_bytes = production
    contract = copy.deepcopy(original)
    now = round(time.time(), 6)
    mission = "abcdef123456"
    row = {"id": mission, "status": "running", "running_owner": "primary", "attempt": 1,
           "started_ts": now - 20, "authorization_id": "", "authorization_action": "",
           "notes": "Host diagnostic", "custom_binding": {"budget": 1}}
    event = {"type": "life.mission.started", "item_id": mission, "attempt": 1,
             "independent_review_required": True, "usage_attempt_id": mission + ":attempt:1",
             "ts": now - 19}
    native = tmp_path / "native"
    native.mkdir()
    (native / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    (native / "events.jsonl").write_bytes(release.encoded(event) + b"\n")
    original_claim = release.native_claim
    monkeypatch.setattr(release, "native_claim",
                        lambda life, task, now: original_claim(native, task, now))
    runtime = copy.deepcopy(contract["runtime"])
    monkeypatch.setattr(release, "observe_runtime", lambda: copy.deepcopy(runtime))
    # Only future claim/review/request documents are virtual. All retained
    # production members go through the real byte authenticator and validator.
    virtual = {}
    authenticate = release.authenticate

    def put(path, value):
        raw = release.json_bytes(value)
        virtual[str(path)] = raw
        return release.pin_bytes(path, raw)

    def read(record):
        if record["path"] not in virtual:
            return authenticate(record)
        raw = virtual[record["path"]]
        release.require(release.same(record, release.pin_bytes(record["path"], raw)),
                        "authenticated bytes changed")
        return raw

    monkeypatch.setattr(release, "authenticate", read)
    consumption_input = release.proposal_context()
    consumption_decision = release.document(release.CONSUMPTION_DECISION)
    contract.update(mission_id=release.MISSION, created_at=now - 30, runtime=runtime,
                    manager_proposal=release.CONSUMPTION_INPUT,
                    construction_provenance=consumption_input["accepted_construction"],
                    artifact_review_provenance=consumption_input["accepted_artifact_review"],
                    implementation_sources=release.implementation_pins())
    root = release.REPOSITORY / release.ROOT
    contract["capture_snapshots"] = {}
    for key, name in (("capture_helper", "capture-source.py"),
                      ("capture_tests", "capture-tests.py")):
        raw = release.authenticate(contract["current_sources"][key])
        virtual[str(root / name)] = raw
        contract["capture_snapshots"][key] = release.pin_bytes(root / name, raw)
    contract_pin = put(root / "current-runtime-contract.json", contract)
    seal_pin = put(root / "seal.json", {
        "mission_id": release.MISSION, "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
        "scientific_run_budget": 0,
        "members": [dict(contract_pin, path="current-runtime-contract.json")],
    })
    review_root = release.LIFE / "handoffs" / release.MISSION
    checkpoint = put(review_root / "CHECKPOINT.md", {"future_review_fixture": True})
    review = put(review_root / "round-0001.json", {
        "kind": "round_reviewed_handoff", "mission_id": release.MISSION,
        "producer_role": "reviewer", "created_at": now - 25,
        "schema_version": 3, "round": 1,
        "mission_context": str(review_root / "mission.json"),
        "checkpoint": {"path": checkpoint["path"]},
        "review": {"status": "done", "reason": "requested outcome is materially complete",
                   "operator_question": "", "next_action": ""},
    })
    latest = put(review_root / "latest.json",
                 {"kind": "handoff_ref", "schema_version": 3,
                  "handoff": {"path": review["path"]},
                  "mission": {"path": str(review_root / "mission.json")}})
    selected_mission = put(review_root / "mission.json", {
        "kind": "mission_context", "schema_version": 3, "mission_id": release.MISSION,
        "fixture_only": True, "tags": ["stage_transition:skip"],
    })
    terminal_row = {
        "id": release.MISSION, "status": "done", "attempt": 1, "finished_ts": now - 24,
        "tags": ["stage_transition:skip"],
        "outcome": {"review_status": "done", "resumable": False,
                    "stage_certification": "intentionally_skipped"},
    }
    terminal_event = {
        "type": "life.mission.completed", "item_id": release.MISSION, "status": "done",
        "ts": now - 23, "independent_review_required": True, "resumable": False,
        "outcome": {"stage_certification": "intentionally_skipped"},
    }
    terminal = {
        "backlog_row_sha256": release.digest(release.encoded(terminal_row)),
        "mission_completed_event_sha256": release.digest(release.encoded(terminal_event)),
        "finished_ts": terminal_row["finished_ts"],
    }
    native_bytes = {}
    for name, value in (("backlog.jsonl", terminal_row), ("events.jsonl", terminal_event)):
        path = release.LIFE / name
        native_bytes[path] = provenance_native_bytes[path] + release.encoded(value) + b"\n"
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: native_bytes[self] if self in native_bytes else read_bytes(self))
    lineage = contract["original_scientific_lineage"]
    accepted_review = {
        "mission_id": contract["accepted_mission"],
        "latest": release.pin(release.LIFE / "handoffs" / contract["accepted_mission"]
                              / "latest.json"),
        "review": contract["accepted_review"], "checkpoint": lineage["accepted_checkpoint"],
    }
    request = {
        "kind": "manager_stage11_postclaim_request", "issuer": "manager",
        "mission_id": mission, "attempt": 1, "scientific_run_budget": 1,
        "issued_at_utc": release.utc(now - 10), "native_record": row,
        "mission_started_event": event, "contract": contract_pin, "seal": seal_pin,
        "contract_review": {"mission_id": release.MISSION, "latest": latest,
                            "review": review, "checkpoint": checkpoint},
        "accepted_review": accepted_review, "accepted_identity": copy.deepcopy(run),
        "launch": dict(contract["launch"], mission_id=mission,
                       constraints=copy.deepcopy(contract["launch_constraints"])),
    }
    authority = release.REPOSITORY / ".argus/live/manager-authority" / (
        "stage11-science-" + mission + "-attempt001")
    request_path = authority / "request.json"
    selected = {
        "mission_id": release.MISSION, "status": "NORMAL_REVIEWER_DONE_NATIVE_DONE",
        "mission": selected_mission, "native_terminal": terminal,
    }
    selection = {
        "kind": "manager_stage11_operational_release_selection", "issuer": "manager",
        "status": "SELECTED_AFTER_NORMAL_REVIEW_AND_NATIVE_DONE",
        "decision": "ADMISSIBLE_POST_REVIEW_CONSUMPTION", "scientific_run_budget": 0,
        "issued_at_utc": release.utc(now - 22), "selected": selected,
        "implementation_sources": contract["implementation_sources"],
        "provenance": {
            "construction": consumption_input["accepted_construction"],
            "artifact_review": consumption_input["accepted_artifact_review"],
        },
        "manager_input": release.CONSUMPTION_INPUT, "manager_decision": release.CONSUMPTION_DECISION,
        "admission": release.ADMISSION, "audit_limit": release.AUDIT_LIMIT,
    }
    selection_path = release.REPOSITORY / ".argus/live/manager-authority" / (
        "stage11-operational-release-" + release.MISSION) / "selection.json"

    def bind_consumption():
        selected.update(contract=request["contract"], seal=request["seal"])
        selected.update({key: request["contract_review"][key]
                         for key in ("latest", "review", "checkpoint")})
        request["operational_release_selection"] = put(selection_path, selection)

    def rebind_contract(changed):
        contract_pin = put(root / "current-runtime-contract.json", changed)
        seal_pin = put(root / "seal.json", {
            "mission_id": release.MISSION, "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
            "scientific_run_budget": 0,
            "members": [dict(contract_pin, path="current-runtime-contract.json")],
        })
        review_value = release.document(request["contract_review"]["review"])
        request.update(contract=contract_pin, seal=seal_pin,
                       accepted_identity=changed["accepted_lineage"])
        request["contract_review"]["review"] = put(review["path"], review_value)
        bind_consumption()
        consumption_input["accepted_operational_successor"] = selected

    bind_consumption()
    consumption_input["accepted_operational_successor"] = selected
    return {"request": request, "request_path": request_path, "put": put, "virtual": virtual,
            "contract": contract, "runtime": runtime, "native": native,
            "output": authority / "issued", "rebind_contract": rebind_contract,
            "bind_consumption": bind_consumption, "consumption_input": consumption_input,
            "consumption_decision": consumption_decision, "selection": selection,
            "selection_path": selection_path}


def check(case):
    return release.validate_after_claim(case["put"](case["request_path"], case["request"]))


def test_production_shaped_postclaim_request_is_read_only(case):
    output, documents, unchanged = check(case)
    assert output == case["output"]
    assert not output.exists()
    assert len(documents) == len(release.OUTPUTS) == 4
    values = dict(zip(release.OUTPUTS, map(json.loads, documents)))
    for value in values.values():
        assert type(value["scientific_run_budget"]) is int
        assert value["scientific_run_budget"] == 1
        assert value["attempt"] == 1
    declaration = values["execution-authorization.json"]
    assert declaration["runtime_sources"] == case["contract"]["runtime_sources"]
    assert declaration["reference_members"] == case["contract"]["reference_members"]
    issuance = values["manager-issuance.json"]
    assert declaration["launch"] == case["request"]["launch"]
    assert declaration["accepted_identity"] == case["request"]["accepted_identity"]
    assert issuance["accepted_identity_receipt"] == case["request"]["accepted_identity"][
        "launcher_capture"]
    assert issuance["authorized_before_running_claim"] is False
    assert issuance["claim"] == release.pin_bytes(output / release.OUTPUTS[0], documents[0])
    assert issuance["declaration"] == release.pin_bytes(output / release.OUTPUTS[1], documents[1])
    assert values["envelope.json"]["issuance"] == release.pin_bytes(
        output / release.OUTPUTS[2], documents[2])
    unchanged()
    assert not output.exists()


def test_obsolete_identity_schema_rejected(case):
    case["request"]["accepted_identity"] = {
        key: {} for key in ("directory", "receipt", "preflight", "commands",
                           "label", "decision_path", "proposal_path", "payload_source")
    }
    with pytest.raises(ValueError, match="production lineage"):
        check(case)


@pytest.mark.parametrize("budget", [0, 2, True, False, 1.0, "1", None])
def test_noninteger_or_wrong_budget(case, budget):
    case["request"]["scientific_run_budget"] = budget
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("key,value", [
    ("attempt", 2), ("attempt", True), ("status", "done"), ("running_owner", "secondary"),
    ("authorization_id", "reserved"), ("authorization_action", "validator-repair"),
    ("validator_expected_result", "PASS"), ("started_ts", 0), ("started_ts", True),
    ("started_ts", 9999999999),
])
def test_wrong_normal_native_claim(case, key, value):
    row = copy.deepcopy(case["request"]["native_record"])
    row[key] = value
    (case["native"] / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_attempt", "no_review", "preclaim"])
def test_started_event_gate(case, defect):
    event = copy.deepcopy(case["request"]["mission_started_event"])
    if defect == "wrong_attempt":
        event["attempt"] = 2
    if defect == "no_review":
        event["independent_review_required"] = False
    if defect == "preclaim":
        event["ts"] = case["request"]["native_record"]["started_ts"] - 1
    raw = release.encoded(event) + b"\n"
    if defect == "missing":
        raw = b""
    if defect == "duplicate":
        raw *= 2
    (case["native"] / "events.jsonl").write_bytes(raw)
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("defect", [
    "engineer", "non_done", "wrong_mission", "missing_audit", "wrong_contract",
    "wrong_seal", "missing_gates", "missing_zero", "future_review", "old_review", "absent",
])
def test_exact_independent_contract_review(case, defect):
    bundle = case["request"]["contract_review"]
    review = release.document(bundle["review"])
    if defect == "engineer":
        review["producer_role"] = "engineer"
    elif defect == "non_done":
        review["review"]["status"] = "replan_requested"
    elif defect == "wrong_mission":
        review["mission_id"] = "222222222222"
    elif defect == "future_review":
        review["created_at"] = time.time() + 1000
    elif defect == "old_review":
        review["created_at"] = 1
    else:
        review["review"]["reason"] = release.review_acceptance(
            case["request"]["contract"], case["request"]["seal"]) + "\n" + defect
    bundle["review"] = case["put"](bundle["review"]["path"], review)
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("target", ["candidate", "tests", "issuer", "proposal", "seal", "review"])
def test_byte_drift(case, target):
    records = {
        "candidate": case["contract"]["current_sources"]["candidate"],
        "tests": case["contract"]["current_sources"]["tests"],
        "issuer": case["contract"]["implementation_sources"][0],
        "proposal": case["contract"]["accepted_proposal"],
        "seal": case["request"]["seal"],
        "review": case["request"]["contract_review"]["review"],
    }
    case["virtual"][records[target]["path"]] = b"changed bytes\n"
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("target", ["command", "environment", "candidate", "budget", "constraints"])
def test_exact_command_environment_and_launch_lineage(case, target):
    launch = case["request"]["launch"]
    if target == "command":
        launch["command"] += " --extra"
    elif target == "environment":
        launch["environment"]["INJECTED"] = "1"
    elif target == "candidate":
        launch["argv"][3] = "other.candidate"
    elif target == "budget":
        launch["budget"]["scientific_suffix_runs"] = 2
    else:
        launch["constraints"]["unexpected"] = True
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("defect", ["rebound", "preclaim", "runtime", "retry_root", "unknown_field"])
def test_remaining_postclaim_gates(case, defect, monkeypatch):
    if defect == "rebound":
        case["request"]["native_record"]["custom_binding"]["budget"] = 2
    elif defect == "preclaim":
        case["request"]["issued_at_utc"] = release.utc(1)
    elif defect == "runtime":
        case["runtime"]["start_ticks"] += 1
    elif defect == "retry_root":
        exists = Path.exists
        monkeypatch.setattr(Path, "exists", lambda p: p == case["output"] or exists(p))
    else:
        case["request"]["authorization_id"] = "reserved"
    with pytest.raises(ValueError):
        check(case)


def test_bookkeeping_is_text_only_and_not_stable_authority(case):
    row = copy.deepcopy(case["request"]["native_record"])
    row["notes"] = "Updated Host diagnostic"
    (case["native"] / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    assert check(case)
    case["request"]["native_record"]["notes"] = {"budget": 1}
    with pytest.raises(ValueError):
        check(case)


def test_recheck_rejects_claim_drift_without_emitting(case):
    output, _, unchanged = check(case)
    row = copy.deepcopy(case["request"]["native_record"])
    row["custom_binding"]["budget"] = 2
    (case["native"] / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    with pytest.raises(ValueError, match="claim changed"):
        unchanged()
    assert not output.exists()


def test_one_shot_reservation_and_exclusive_overwrite(tmp_path):
    output = tmp_path / "issued"
    release.reserve_issuance(output)
    release.exclusive(output / "sentinel", b"partial")
    with pytest.raises(FileExistsError):
        release.reserve_issuance(output)
    with pytest.raises(FileExistsError):
        release.exclusive(output / "sentinel", b"replacement")
    assert (output / "sentinel").read_bytes() == b"partial"
    assert not (output / "envelope.json").exists()


def test_canonical_authority_cannot_be_rebound(case):
    other = case["put"](case["request_path"].with_name("other-request.json"), case["request"])
    with pytest.raises(ValueError, match="noncanonical"):
        release.validate_after_claim(other)


def test_pin_symlink_rejected(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        release.authenticate(dict(release.pin(target), path=str(link)))
    link.unlink()


@pytest.fixture
def clone(production, tmp_path):
    """Physical, repinned production copies for semantic corruption tests."""
    run, contract = copy.deepcopy(production[:2])
    old_root = Path(run["terminal_receipt"]["path"]).parent
    new_root = tmp_path / "retained"
    records = contract["accepted_files"] + [run["receipt_index"], run["terminal_receipt"]]
    templates = {}
    for record in records:
        relative = str(Path(record["path"]).relative_to(old_root))
        raw = release.authenticate(record)
        templates[relative] = json.loads(raw) if relative.endswith(".json") else raw

    def materialize(edit=None):
        values = copy.deepcopy(templates)
        if edit:
            edit(values)
        pins = {}
        active = set()

        def relocate(value):
            if isinstance(value, dict):
                if set(value) == {"path", "bytes", "sha256"}:
                    path = Path(value["path"])
                    if path.is_relative_to(old_root):
                        relative = str(path.relative_to(old_root))
                        if relative in values:
                            return write(relative)
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v) for v in value]
            if isinstance(value, str):
                return value.replace(str(old_root), str(new_root))
            return value

        def write(relative):
            if relative in pins:
                return pins[relative]
            assert relative not in active, "production evidence must not contain circular pins"
            active.add(relative)
            value = values[relative]
            raw = (value.replace(str(old_root).encode(), str(new_root).encode())
                   if isinstance(value, bytes) else release.json_bytes(relocate(value)))
            path = new_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            pins[relative] = release.pin(path)
            active.remove(relative)
            return pins[relative]

        for relative in values:
            write(relative)
        copied_run = {k: pins[str(Path(v["path"]).relative_to(old_root))] for k, v in run.items()}
        copied_contract = copy.deepcopy(contract)
        copied_contract.update(
            accepted_lineage=copied_run, accepted_proposal=copied_run["proposal"],
            accepted_terminal=copied_run["terminal_receipt"],
            accepted_files=release.document(copied_run["receipt_index"])["files"])
        copied_contract["original_scientific_lineage"].update(
            proposal=copied_run["proposal"], accepted_terminal=copied_run["terminal_receipt"])
        return copied_run, copied_contract

    return materialize


def test_production_copy_preserves_validator_semantics(clone):
    run, contract = clone()
    assert release.accepted_lineage(run, contract)["launch"] == contract["launch"]


@pytest.mark.parametrize("index", range(35))
@pytest.mark.parametrize("defect", ["missing", "altered"])
def test_every_retained_member_requires_exact_bytes(clone, index, defect):
    run, contract = clone()
    records = contract["accepted_files"] + [run["terminal_receipt"], run["receipt_index"]]
    path = Path(records[index]["path"])
    if defect == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        release.accepted_lineage(run, contract)


@pytest.mark.parametrize("name", [
    "authority_bundle", "launcher_capture", "launcher_identity", "launcher_expected_identity",
    "terminal_receipt", "proposal", "receipt_index",
])
def test_cross_spliced_production_members(production, clone, name):
    run, contract = clone()
    run[name] = production[0][name]
    with pytest.raises(ValueError):
        release.accepted_lineage(run, contract)


SEMANTIC_DEFECTS = [
    ("bundle/authority-bundle.json", ["status"], "FAIL"),
    ("bundle/authority-bundle.json", ["identity_preflight", "status"], "UNKNOWN"),
    ("bundle/authority-bundle.json", ["dispatch_authorized"], True),
    ("bundle/authority-bundle.json", ["scientific_invocations"], True),
    ("bundle/authority-bundle.json", ["proposal"], "splice"),
    ("bundle/authority-bundle.json", ["origins", "proposal", "pin"], "splice"),
    ("launcher.capture.json", ["exit_status"], 1),
    ("launcher.capture.json", ["exit_status"], False),
    ("launcher.capture.json", ["success"], False),
    ("launcher.capture.json", ["timed_out"], True),
    ("launcher.capture.json", ["files"], []),
    ("launcher.capture.json", ["capture_implementation_after"], []),
    ("launcher.identity.json", ["command"], "other"),
    ("launcher.identity.json", ["argv"], ["other"]),
    ("launcher.identity.json", ["cwd"], "/other"),
    ("launcher.identity.json", ["uid"], -1),
    ("launcher.identity.json", ["scope"], "other"),
    ("launcher.identity.json", ["environment"], {"PATH": "/other"}),
    ("launcher.expected-identity.json", ["environment"], {"PATH": "/other"}),
    ("terminal-receipt.json", ["status"], "UNKNOWN"),
    ("terminal-receipt.json", ["dispatch_authorized"], True),
    ("terminal-receipt.json", ["scientific_or_admission_claim"], True),
    ("terminal-receipt.json", ["closure_check", "status"], "FAIL"),
    ("terminal-receipt.json", ["manager_authority", "proposal_id"], "other"),
    ("terminal-receipt.json", ["manager_authority", "issuance"], "splice"),
    ("terminal-receipt.json", ["manager_authority", "review", "mission_id"], "other"),
    ("terminal-receipt.json", ["receipt_index"], "splice"),
    ("closure-check.json", ["status"], "FAIL"),
    ("closure-check.json", ["counters", "scientific_invocations"], 1),
    ("closure-check.json", ["counters", "service_invocations"], False),
    ("receipt-index.json", ["files"], []),
    ("receipt-index.json", ["terminal_receipt_is_sealed_after_this_index"], False),
    ("bundle/proposal.json", ["identity", "boundary"], "other"),
    ("bundle/proposal.json", ["launch", "command"], "other"),
    ("bundle/proposal.json", ["launch", "environment"], {"PATH": "/other"}),
    ("bundle/proposal.json", ["source_roles"], {}),
    ("bundle/proposal.json", ["source_snapshots"], {}),
    ("consumption.inputs.json", ["launch", "budget", "check"], 2),
    ("custody/custody-receipt.json", ["launch", "command"], "other"),
    ("consumption.result.json", ["identity_decision"], "UNKNOWN"),
    ("input-census.after.json", ["all_equal_to_before"], False),
]


@pytest.mark.parametrize("relative,keys,value", SEMANTIC_DEFECTS)
def test_repinned_production_semantic_rejections(clone, relative, keys, value):
    def edit(documents):
        node = documents[relative]
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = (copy.deepcopy(documents["launcher.capture.json"]["files"][0])
                          if value == "splice" else value)

    run, contract = clone(edit)
    with pytest.raises(ValueError):
        release.accepted_lineage(run, contract)


@pytest.mark.parametrize("key", ["source_roles", "source_snapshots", "execution_contract", "identity"])
def test_current_contract_source_splices(production, key):
    run, contract = copy.deepcopy(production[:2])
    contract[key] = {}
    with pytest.raises(ValueError):
        release.accepted_lineage(run, contract)


def test_index_duplicate_and_membership_splice(clone):
    def edit(documents):
        files = documents["receipt-index.json"]["files"]
        files[0] = copy.deepcopy(files[1])

    run, contract = clone(edit)
    with pytest.raises(ValueError, match="duplicate"):
        release.accepted_lineage(run, contract)


def test_recheck_rejects_retained_lineage_drift(case):
    _, _, unchanged = check(case)
    case["virtual"][case["request"]["accepted_identity"]["launcher_capture"]["path"]] = b"{}"
    with pytest.raises(ValueError):
        unchanged()


def test_actual_pinned_native_review_consumption():
    accepted = release.construction_input(
        release.document(release.CONSUMPTION_INPUT))["accepted_provenance"]
    bundle = dict(mission_id=release.PROVENANCE_MISSION,
                  **{key: accepted[key] for key in ("latest", "review", "checkpoint")})
    review = release.native_review_consumption(bundle, accepted["contract"], accepted["seal"])
    assert review["producer_role"] == "reviewer"
    assert review["round"] == 1
    assert review["review"]["reason"] == "requested outcome is materially complete"
    assert release.document(accepted["latest"])["handoff"] == {
        "path": accepted["review"]["path"]}
    expected = release.review_acceptance(accepted["contract"], accepted["seal"])
    assert expected.splitlines() == [
        "Stage11-Admission: " + release.ADMISSION,
        "Stage11-Audit-Limit: " + release.AUDIT_LIMIT,
        "Stage11-Contract-SHA256: " + accepted["contract"]["sha256"],
        "Stage11-Seal-SHA256: " + accepted["seal"]["sha256"],
        "Stage11-Emitter-Gates: ACCEPTED_EXACT_POSTCLAIM_ONE_SHOT",
        "Stage11-Zero-Science: ACCEPTED",
    ]


@pytest.mark.parametrize("target,keys,value", [
    ("review", ["producer_role"], "engineer"),
    ("review", ["mission_id"], "94e2e47338e2"),
    ("review", ["round"], 2),
    ("review", ["round"], True),
    ("review", ["schema_version"], 2),
    ("review", ["mission_context"], "/wrong/mission.json"),
    ("review", ["checkpoint", "path"], "/wrong/CHECKPOINT.md"),
    ("review", ["review", "status"], "replan_requested"),
    ("review", ["review", "reason"], "done"),
    ("review", ["review", "operator_question"], "approve?"),
    ("review", ["review", "next_action"], "science"),
    ("latest", ["handoff", "path"], "/wrong/round-0001.json"),
    ("latest", ["handoff", "sha256"], "invented"),
    ("latest", ["mission", "path"], "/wrong/mission.json"),
    ("latest", ["mission", "sha256"], "invented"),
    ("latest", ["schema_version"], 2),
])
def test_repinned_native_review_schema_rejected(case, target, keys, value):
    bundle = case["request"]["contract_review"]
    record = release.document(bundle[target])
    node = record
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    bundle[target] = case["put"](bundle[target]["path"], record)
    case["bind_consumption"]()
    with pytest.raises(ValueError):
        check(case)


@pytest.fixture
def later_review_case(case):
    bundle = case["request"]["contract_review"]
    review = release.document(bundle["review"])
    first = copy.deepcopy(review)
    first["created_at"] -= 1
    first["review"].update(status="continue", reason="validation pending")
    first_pin = case["put"](bundle["review"]["path"], first)
    bundle["review"] = first_pin
    case["bind_consumption"]()
    with pytest.raises(ValueError):
        check(case)
    review["round"] = 3
    bundle["review"] = case["put"](
        Path(first_pin["path"]).with_name("round-0003.json"), review)
    latest = release.document(bundle["latest"])
    latest["handoff"]["path"] = bundle["review"]["path"]
    bundle["latest"] = case["put"](bundle["latest"]["path"], latest)
    case["bind_consumption"]()
    assert release.document(first_pin) == first
    return case


def test_later_native_done_after_round_one_continue(later_review_case):
    case = later_review_case
    output, documents, unchanged = check(case)
    assert not output.exists()
    assert len(documents) == len(release.OUTPUTS)
    assert release.document(case["selection"]["selected"]["review"])["round"] == 3
    unchanged()


@pytest.mark.parametrize("target,keys,value", [
    ("review", ["round"], 1), ("review", ["round"], 4),
    ("review", ["round"], 0), ("review", ["round"], -1),
    ("review", ["round"], True), ("review", ["round"], 3.0),
    ("review", ["round"], "3"),
    ("review", ["mission_id"], "b38bfad5e968"),
    ("review", ["mission_context"], "/wrong/mission.json"),
    ("latest", ["handoff", "path"],
     str(release.LIFE / "handoffs" / release.MISSION / "round-0001.json")),
    ("latest", ["mission", "path"], "/wrong/mission.json"),
])
def test_later_native_done_rejects_rebound_handoff(later_review_case, target, keys, value):
    case = later_review_case
    bundle = case["request"]["contract_review"]
    record = release.document(bundle[target])
    node = record
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    bundle[target] = case["put"](bundle[target]["path"], record)
    case["bind_consumption"]()
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("filename", ["round-3.json", "round-0004.json", "other.json"])
def test_later_native_done_requires_canonical_round_path(later_review_case, filename):
    case = later_review_case
    bundle = case["request"]["contract_review"]
    bundle["review"] = case["put"](
        Path(bundle["review"]["path"]).with_name(filename), release.document(bundle["review"]))
    latest = release.document(bundle["latest"])
    latest["handoff"]["path"] = bundle["review"]["path"]
    bundle["latest"] = case["put"](bundle["latest"]["path"], latest)
    case["bind_consumption"]()
    with pytest.raises(ValueError, match="normalized native Reviewer DONE"):
        check(case)


@pytest.mark.parametrize("key,value", [
    ("issuer", "engineer"), ("mission_id", "94e2e47338e2"),
    ("decision", "DONE"), ("status", "done"),
    ("admission", "INHERITED"), ("scientific_run_budget", 1),
    ("scientific_run_budget", False), ("accepted_root", "build/other"),
    ("failed_predecessor", "other"), ("source_contract_sha256", "changed"),
    ("source_seal_sha256", "changed"), ("native_acceptance", {}),
    ("review_acceptance", {}),
])
def test_repinned_manager_consumption_decision_rejected(case, monkeypatch, key, value):
    decision = copy.deepcopy(case["consumption_decision"])
    decision[key] = value
    monkeypatch.setattr(release, "CONSUMPTION_DECISION",
                        case["put"](release.CONSUMPTION_DECISION["path"], decision))
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("target", [
    "latest", "review", "mission", "checkpoint", "contract", "seal",
    "manager_input", "manager_decision",
])
def test_native_consumption_byte_changes_rejected(case, target):
    records = dict(case["consumption_input"]["accepted_operational_successor"],
                   manager_input=release.CONSUMPTION_INPUT,
                   manager_decision=release.CONSUMPTION_DECISION)
    case["virtual"][records[target]["path"]] = b"changed\n"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        check(case)


@pytest.mark.parametrize("name,defect", [
    ("backlog.jsonl", "status"), ("backlog.jsonl", "missing"),
    ("backlog.jsonl", "duplicate"), ("backlog.jsonl", "finished_ts"),
    ("events.jsonl", "status"), ("events.jsonl", "missing"),
    ("events.jsonl", "duplicate"), ("events.jsonl", "independent_review_required"),
])
def test_changed_native_terminal_rejected(case, monkeypatch, name, defect):
    read_bytes = Path.read_bytes
    path = release.LIFE / name
    key = "id" if name == "backlog.jsonl" else "item_id"
    lines = []
    for raw in path.read_bytes().splitlines():
        value = json.loads(raw)
        if value.get(key) == release.MISSION and (
            key == "id" or value.get("type") == "life.mission.completed"
        ):
            if defect == "missing":
                continue
            if defect == "duplicate":
                lines.append(raw)
            else:
                value[defect] = "changed"
                raw = release.encoded(value)
        lines.append(raw)
    changed = b"\n".join(lines) + b"\n"
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: changed if self == path else read_bytes(self))
    with pytest.raises(ValueError):
        check(case)


def test_native_consumption_rechecked_before_issuance(case):
    _, _, unchanged = check(case)
    case["virtual"][release.CONSUMPTION_DECISION["path"]] = b"{}"
    with pytest.raises(ValueError):
        unchanged()


def test_native_acceptance_does_not_rebind_implementation_sources():
    accepted = release.construction_input(
        release.document(release.CONSUMPTION_INPUT))["accepted_provenance"]
    contract = release.document(accepted["contract"])
    for (role, path, _), record in zip(
            release.SOURCE_LAYOUT, contract["implementation_sources"], strict=True):
        if role in ("capture_helper", "capture_tests"):
            assert record == release.pin(release.REPOSITORY / path)
            release.authenticate(record)
        else:
            with pytest.raises(ValueError, match="authenticated bytes changed"):
                release.authenticate(record)


@pytest.mark.parametrize("name", [
    "candidate-source.py", "candidate-tests.py", "capture-source.py", "capture-tests.py",
])
def test_proposed_contract_requires_materialized_snapshots(production, tmp_path, name):
    inputs = release.proposal_context()
    for (_, _, filename), source in zip(release.SOURCE_LAYOUT, release.implementation_pins(), strict=True):
        if filename != name:
            (tmp_path / filename).write_bytes(release.authenticate(source))
    with pytest.raises(FileNotFoundError):
        release.proposed_contract(release.document(inputs["accepted_construction"]["contract"]),
                                  release.CONSUMPTION_INPUT, tmp_path, time.time(),
                                  production[1]["runtime"])


@pytest.mark.parametrize("key,value", [
    ("issuer", "engineer"), ("kind", "done"), ("status", "done"),
    ("decision", "DONE"), ("scientific_run_budget", False), ("scientific_run_budget", 1),
    ("admission", "INHERITED"), ("audit_limit", "OS_WIDE"),
    ("implementation_sources", []), ("provenance", {}), ("manager_input", {}),
    ("manager_decision", {}), ("issued_at_utc", "2000-01-01T00:00:00+00:00"),
    ("issued_at_utc", "2100-01-01T00:00:00+00:00"),
])
def test_repinned_operational_selection_rejected(case, key, value):
    case["selection"][key] = value
    case["bind_consumption"]()
    with pytest.raises(ValueError):
        check(case)


def test_selection_cannot_use_arbitrary_path_or_done_blob(case):
    case["request"]["operational_release_selection"] = case["put"](
        case["selection_path"].with_name("other.json"), case["selection"])
    with pytest.raises(ValueError, match="noncanonical"):
        check(case)
    case["request"]["operational_release_selection"] = case["put"](
        case["selection_path"], {"status": "done"})
    with pytest.raises(ValueError, match="selection fields"):
        check(case)


def test_old_implementation_pins_cannot_be_selected(case):
    provenance = release.proposal_context()["accepted_construction"]
    old_sources = release.document(provenance["contract"])["implementation_sources"]
    contract = copy.deepcopy(case["contract"])
    contract["implementation_sources"] = old_sources
    case["selection"]["implementation_sources"] = old_sources
    case["rebind_contract"](contract)
    with pytest.raises(ValueError, match="source version"):
        check(case)


@pytest.mark.parametrize("key", ["candidate", "tests", "capture_helper", "capture_tests"])
def test_every_current_source_is_bound_in_selection(case, key):
    contract = copy.deepcopy(case["contract"])
    contract["current_sources"][key]["sha256"] = "0" * 64
    case["rebind_contract"](contract)
    with pytest.raises(ValueError, match="source version"):
        check(case)


@pytest.mark.parametrize("key", ["capture_helper", "capture_tests"])
def test_capture_snapshot_cannot_be_spliced(case, key):
    case["virtual"][case["contract"]["capture_snapshots"][key]["path"]] = b"changed\n"
    with pytest.raises(ValueError, match="authenticated bytes"):
        check(case)


@pytest.mark.parametrize("index", range(6))
def test_each_implementation_source_is_required(case, index):
    contract = copy.deepcopy(case["contract"])
    contract["implementation_sources"][index]["sha256"] = "0" * 64
    case["selection"]["implementation_sources"] = contract["implementation_sources"]
    case["rebind_contract"](contract)
    with pytest.raises(ValueError, match="source version"):
        check(case)


def test_old_ff_candidate_pins_are_not_current_acceptance(case):
    provenance = release.construction_input(
        release.document(release.CONSUMPTION_INPUT))["accepted_provenance"]
    previous = release.document(provenance["contract"])
    contract = copy.deepcopy(case["contract"])
    for key in ("candidate", "tests"):
        contract["current_sources"][key] = previous["current_sources"][key]
    case["rebind_contract"](contract)
    with pytest.raises(ValueError, match="source version"):
        check(case)


def test_selection_must_predate_science_claim(case):
    case["selection"]["issued_at_utc"] = release.utc(time.time() - 1)
    case["bind_consumption"]()
    with pytest.raises(ValueError, match="pre-claim"):
        check(case)


@pytest.mark.parametrize("offset,accepted", [
    (-1.5, False), (-0.5, False), (0, True), (0.5, True),
])
def test_selection_must_follow_native_completion_event(case, offset, accepted):
    events = [
        json.loads(raw)
        for raw in (release.LIFE / "events.jsonl").read_bytes().splitlines()
        if raw.strip()
    ]
    completed = next(
        event for event in events
        if event.get("type") == "life.mission.completed"
        and event.get("item_id") == release.MISSION
    )
    case["selection"]["issued_at_utc"] = release.utc(completed["ts"] + offset)
    case["bind_consumption"]()
    if accepted:
        check(case)
    else:
        with pytest.raises(ValueError, match="selection must follow native DONE"):
            check(case)


@pytest.mark.parametrize("enabled", [False, True, 0, 1, "false", None])
def test_runtime_observation_binds_continuous_mode(tmp_path, monkeypatch, enabled):
    source_root = tmp_path / "candidate" / "argus_skill"
    source_root.mkdir(parents=True)
    source = {
        "pid": 12345, "source_root": str(source_root), "source_fingerprint": "test-source",
    }
    (tmp_path / "daemon.source.json").write_bytes(release.json_bytes(source))
    (tmp_path / "continuous.json").write_bytes(
        release.json_bytes({"enabled": enabled, "generation": 51})
    )
    monkeypatch.setattr(release, "LIFE", tmp_path)
    read_bytes, read_text = Path.read_bytes, Path.read_text
    command = b"python\0" + str(source_root.parent).encode() + b"\0"
    monkeypatch.setattr(
        Path, "read_bytes",
        lambda path: command if path == Path("/proc/12345/cmdline") else read_bytes(path),
    )
    monkeypatch.setattr(
        Path, "read_text",
        lambda path: "12345 (python) " + " ".join(["S", *["0"] * 18, "123"])
        if path == Path("/proc/12345/stat") else read_text(path),
    )
    if type(enabled) is not bool:
        with pytest.raises(ValueError, match="continuous enabled flag must be boolean"):
            release.observe_runtime()
        return
    observed = release.observe_runtime()
    assert observed["continuous_enabled"] is enabled
    assert observed["continuous_generation"] == 51
    assert observed["start_ticks"] == 123
    assert observed["cmdline_sha256"] == release.digest(command)


def test_selection_is_bound_through_pure_envelope_consumption(case):
    output, documents, _ = check(case)
    for name, raw in zip(release.OUTPUTS, documents):
        case["virtual"][str(output / name)] = raw
    envelope = json.loads(documents[3])
    assert release.validate_execution_envelope(envelope) == json.loads(documents[1])
    for raw in documents[1:]:
        assert json.loads(raw)["operational_release_selection"] == case["request"][
            "operational_release_selection"]
    case["virtual"][case["selection_path"].as_posix()] += b"\n"
    with pytest.raises(ValueError, match="authenticated bytes"):
        release.validate_execution_envelope(envelope)


@pytest.mark.parametrize("defect", [
    None, "command", "source_root", "pythonpath", "cwd", "missing_environment",
])
def test_runtime_observation_binds_handoff_source(tmp_path, monkeypatch, defect):
    source_root = tmp_path / "candidate" / "argus_skill"
    source_root.mkdir(parents=True)
    (tmp_path / "daemon.source.json").write_bytes(release.json_bytes({
        "pid": 12345, "source_root": str(source_root), "source_fingerprint": "test-source",
    }))
    (tmp_path / "continuous.json").write_bytes(
        release.json_bytes({"enabled": True, "generation": 51}))
    command = [
        sys.executable.encode(), b"-c",
        b"from argus_skill.daemon.life_worker import run_handoff_child; "
        b"raise SystemExit(run_handoff_child())", b"",
    ]
    if defect == "command":
        command[2] += b"# unrecognized bootstrap"
    environment = {
        b"ARGUS_SKILL_SOURCE_ROOT": str(source_root.parent).encode(),
        b"PYTHONPATH": str(source_root.parent).encode() + b":/retained",
    }
    if defect in ("source_root", "pythonpath"):
        environment[b"ARGUS_SKILL_SOURCE_ROOT" if defect == "source_root"
                    else b"PYTHONPATH"] = b"/rebound"
    if defect == "missing_environment":
        environment.clear()
    process = Path("/proc/12345")
    virtual = {
        process / "cmdline": b"\0".join(command),
        process / "environ": b"\0".join(key + b"=" + value
                                      for key, value in environment.items()),
    }
    read_bytes, read_text, resolve = Path.read_bytes, Path.read_text, Path.resolve
    monkeypatch.setattr(release, "LIFE", tmp_path)
    monkeypatch.setattr(Path, "read_bytes", lambda p: virtual[p] if p in virtual else read_bytes(p))
    monkeypatch.setattr(Path, "read_text",
                        lambda p: "12345 (python) " + " ".join(["S", *["0"] * 18, "123"])
                        if p == process / "stat" else read_text(p))
    monkeypatch.setattr(Path, "resolve",
                        lambda p: Path("/rebound" if defect == "cwd" else "/")
                        if p == process / "cwd" else resolve(p))
    if defect is not None:
        with pytest.raises(ValueError, match="running (handoff )?process"):
            release.observe_runtime()
    else:
        observed = release.observe_runtime()
        assert observed["source_root"] == str(source_root)
        assert observed["continuous_enabled"] is True
        assert observed["continuous_generation"] == 51
        assert observed["start_ticks"] == 123
        assert observed["cmdline_sha256"] == release.digest(virtual[process / "cmdline"])


def test_actual_successor_v6_owner_and_source_bindings():
    proposal_pin, proposal, _, owner, _ = release.successor_inputs()
    assert release.MISSION == "287d994f2fe4"
    assert release.AUTHORITY == "s11-reviewed-interface-release-20260926-287d994f2fe4-r1"
    assert str(release.ROOT) == "build/argus-stage11-reviewed-interface-release-287d994f2fe4-attempt001"
    assert str(release.VALIDATION_ROOT) == (
        ".argus/live/manager-validation/stage11-reviewed-interface-release-287d994f2fe4")
    assert release.PROVENANCE_MISSION == "f1c261d97727"
    assert proposal_pin == release.CONSUMPTION_INPUT
    assert owner["schema"] == "argus.manager-stage11-reviewed-interface-release-owner.v6"
    assert owner["accepted_release_mission"] == "d15533af5bdf"
    assert owner["accepted_repository_repair_mission"] == "ed9ef3afb3ad"
    assert owner["reviewed_proposed_bytes"] == proposal["repository_repair"]["proposed_bytes"]
    assert owner["failed_science_mission"] == "b7d95bd6e7f2"
    assert release.source_edit_bindings(proposal, owner) == {
        role: "POSTCLAIM_EDIT" if role in ("release_builder", "release_tests")
        else "UNCHANGED_MANAGER_BASELINE" for role, _, _ in release.SOURCE_LAYOUT
    }


@pytest.mark.parametrize("schema", [
    "argus.manager-stage11-operational-successor-owner.v1",
    "argus.manager-stage11-continuation-acceptance-owner.v3",
    "argus.manager-stage11-continuous-release-owner.v4",
    "argus.manager-stage11-execution-contract-repair-owner.v5",
])
def test_successor_rejects_obsolete_owner_schema(monkeypatch, schema):
    original = release.document

    def obsolete_owner(record):
        result = original(record)
        if Path(record["path"]).name == "manager-owner-claim.json":
            result["schema"] = schema
        return result

    monkeypatch.setattr(release, "document", obsolete_owner)
    with pytest.raises(ValueError, match="operational-transition authority differs"):
        release.successor_inputs()


@pytest.mark.parametrize("key,value", [
    ("pid", 1), ("release_id", "obsolete"), ("source_root", "/obsolete"),
    ("source_digest", "0" * 64), ("source_fingerprint", "0" * 64),
    ("source_file_count", 1), ("started_at_iso", "obsolete"),
    ("continuous_enabled", False), ("continuous_enabled", 1),
    ("continuous_generation", 50), ("continuous_open_ended", False),
    ("continuous_objective_sha256", "0" * 64),
])
def test_successor_rejects_owner_runtime_drift(monkeypatch, key, value):
    original = release.document

    def changed_owner(record):
        result = original(record)
        if Path(record["path"]).name == "manager-owner-claim.json":
            result["runtime"][key] = value
        return result

    monkeypatch.setattr(release, "document", changed_owner)
    with pytest.raises(ValueError, match="v4 owner runtime or continuous scheduling differs"):
        release.successor_inputs()


@pytest.mark.parametrize("filename,keys,value", [
    ("daemon.status.json", ("pid",), 1),
    ("daemon.status.json", ("started_at_iso",), "obsolete"),
    ("daemon.status.json", ("runtime", "release_id"), "obsolete"),
    ("daemon.status.json", ("runtime", "source_root"), "/obsolete"),
    ("daemon.status.json", ("runtime", "runtime_source_digest"), "0" * 64),
    ("daemon.status.json", ("runtime", "manifest_source_digest"), "0" * 64),
    ("daemon.status.json", ("runtime", "release_matches_source"), False),
    ("daemon.source.json", ("source_fingerprint",), "0" * 64),
    ("daemon.source.json", ("source_file_count",), 1),
    ("continuous.json", ("enabled",), False),
    ("continuous.json", ("generation",), 50),
    ("continuous.json", ("open_ended",), False),
    ("continuous.json", ("objective",), "obsolete"),
])
def test_successor_rejects_live_runtime_drift(monkeypatch, filename, keys, value):
    path = release.LIFE / filename
    changed = json.loads(path.read_bytes())
    target = changed
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: release.json_bytes(changed)
                        if p == path else read_bytes(p))
    with pytest.raises(ValueError, match="v4 owner runtime or continuous scheduling differs"):
        release.successor_inputs()


@pytest.mark.parametrize("changed,mtime_ns,accepted", [
    (False, 1_000_000_000, True), (True, 3_000_000_000, True),
    (True, 1_000_000_000, False), (True, 2_000_000_000, False),
])
def test_source_binding_requires_baseline_or_postclaim_bytes(
        tmp_path, monkeypatch, changed, mtime_ns, accepted):
    source = tmp_path / "source.py"
    source.write_bytes(b"baseline\n")
    proposal = {"source_baseline": {"release_builder": release.pin(source)}}
    if changed:
        source.write_bytes(b"changed\n")
    os.utime(source, ns=(mtime_ns, mtime_ns))
    monkeypatch.setattr(release, "SOURCE_LAYOUT", (("release_builder", source, "source.py"),))
    owner = {"published_at": 2.0}
    if accepted:
        assert release.source_edit_bindings(proposal, owner) == {
            "release_builder": "POSTCLAIM_EDIT" if changed else "UNCHANGED_MANAGER_BASELINE"
        }
    else:
        with pytest.raises(ValueError, match="native owner claim must precede changed source bytes"):
            release.source_edit_bindings(proposal, owner)


@pytest.mark.parametrize("index", range(2, 6))
def test_reviewed_candidate_capture_cannot_be_edited(monkeypatch, index):
    proposal = release.proposal_context()
    path = release.REPOSITORY / release.SOURCE_LAYOUT[index][1]
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: b"postclaim edit" if self == path else original(self))
    with pytest.raises(ValueError, match="reviewed ed9 candidate/capture drift"):
        release.source_edit_bindings(proposal, {"published_at": 0})


@pytest.mark.parametrize("key", [
    "review", "proposed_bytes", "compile_receipt", "focused_tests_receipt",
    "focused_tests_xml", "mission_completed_event_sha256",
])
def test_reviewed_repair_rejects_changed_evidence(key):
    proposal = release.proposal_context()
    repair = proposal["repository_repair"]
    if key == "mission_completed_event_sha256":
        repair[key] = "0" * 64
    else:
        repair[key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authenticated bytes changed|immutable native completion"):
        release.reviewed_repair_evidence(proposal)


@pytest.fixture
def preseal_case(tmp_path, monkeypatch):
    """Isolated prospective review data; never write a native Host verdict."""
    monkeypatch.setattr(release, "LIFE", tmp_path / "life")
    base = release.LIFE / "handoffs" / release.MISSION
    base.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_bytes(b"unchanged reviewed source\n")
    now = time.time()
    os.utime(source, (now - 5, now - 5))
    sources = [release.pin(source)]
    prevalidation = {"finished_at": now - 2}
    review = {
        "kind": "round_reviewed_handoff", "schema_version": 3,
        "mission_id": release.MISSION, "producer_role": "reviewer",
        "round": 1, "created_at": now - 1,
        "mission_context": str(base / "mission.json"),
        "review": {
            "status": "continue", "reason": "isolated test-only preseal acceptance",
            "next_action": "BUILD_FINAL_FROM_REVIEWED_BYTES",
        },
    }
    review_path = base / "round-0001.json"
    release.write_json(base / "latest.json", {
        "kind": "handoff_ref", "schema_version": 3,
        "handoff": {"path": str(review_path)},
        "mission": {"path": str(base / "mission.json")},
    })

    def check_review():
        review_path.write_bytes(release.json_bytes(review))
        return release.authenticate_preseal_review(prevalidation, sources)

    return review, sources, prevalidation, check_review, review_path


@pytest.mark.parametrize("status", ["continue", "replan_requested"])
def test_preseal_verdict_matches_real_runtime_classification(
        preseal_case, monkeypatch, record_property, status):
    import importlib

    runtime_root = Path(release.proposal_context()["expected_runtime"]["source_root"])
    monkeypatch.syspath_prepend(str(runtime_root))
    settlement = importlib.import_module("argus_skill.engineer.round_settlement")
    models = importlib.import_module("argus_skill.core.models")
    path = runtime_root / "argus_skill/engineer/round_settlement.py"
    assert Path(settlement.__file__).resolve() == path
    record_property("runtime_classifier_source", json.dumps(release.pin(path), sort_keys=True))
    decision = models.ReviewDecision(
        status=status, reason="isolated lifecycle regression",
        next_action="BUILD_FINAL_FROM_REVIEWED_BYTES")
    terminal, _ = settlement.RoundSettlementMixin._classify(
        review=decision, no_progress_streak=0, no_progress_threshold=3,
        round_index=1, max_rounds=8)
    review, _, _, check_review, review_path = preseal_case
    review["review"]["status"] = decision.status
    review["review"]["next_action"] = decision.next_action
    if status == "continue":
        assert terminal is None
        assert check_review() == release.pin(review_path)
    else:
        assert terminal == "replan_requested"
        with pytest.raises(ValueError, match="independent preseal review"):
            check_review()


@pytest.mark.parametrize("defect", [
    "mission", "role", "kind", "round", "context", "action", "done", "before-validation",
    "future-review", "changed-bytes", "edit-after-review",
])
def test_preseal_rejects_rebound_review(preseal_case, defect):
    review, sources, prevalidation, check_review, _ = preseal_case
    if defect == "mission":
        review["mission_id"] = "other"
    elif defect == "role":
        review["producer_role"] = "engineer"
    elif defect == "kind":
        review["kind"] = "mission_context"
    elif defect == "round":
        review["round"] = 2
    elif defect == "context":
        review["mission_context"] = "/other/mission.json"
    elif defect == "action":
        review["review"]["next_action"] = "not approved"
    elif defect == "done":
        review["review"]["status"] = "done"
    elif defect == "before-validation":
        review["created_at"] = prevalidation["finished_at"] - 1
    elif defect == "future-review":
        review["created_at"] = time.time() + 60
    else:
        path = Path(sources[0]["path"])
        path.write_bytes(b"changed source\n")
        if defect == "edit-after-review":
            sources[0] = release.pin(path)
    with pytest.raises(ValueError, match="independent preseal review|authenticated bytes changed"):
        check_review()


@pytest.mark.parametrize("failing_command", ["py_compile", "focused-tests"])
def test_failed_validation_is_retained_without_delivery(production, tmp_path, monkeypatch,
                                                       failing_command):
    inputs = release.proposal_context()
    previous = release.document(inputs["accepted_construction"]["contract"])
    owner = {"published_at": time.time() - 10}
    monkeypatch.setattr(release, "successor_inputs",
                        lambda: (release.CONSUMPTION_INPUT, inputs, previous, owner, {}))
    monkeypatch.setattr(release, "preserved_trees", lambda *args: {})
    monkeypatch.setattr(release, "observe_runtime", lambda: production[1]["runtime"])
    monkeypatch.setattr(release, "ROOT", tmp_path / "delivery")
    monkeypatch.setattr(release, "VALIDATION_ROOT", tmp_path / "validation")

    def failed_command(scratch, name, argv):
        assert not release.ROOT.exists()
        result = {"name": name, "argv": argv, "environment": release.validation_environment(scratch),
                  "exit_code": 1 if name == failing_command else 0,
                  "test_only": True, "test_summary": None}
        release.write_json(scratch / (name + ".receipt.json"), result)
        release.exclusive(scratch / (name + ".stdout"), b"test-only failed validation fixture\n")
        release.exclusive(scratch / (name + ".stderr"), b"")
        return result

    monkeypatch.setattr(release, "run_validation_command", failed_command)
    for number in (1, 2):
        with pytest.raises(ValueError, match=failing_command + " failed"):
            release.prevalidate()
        assert not release.ROOT.exists()
        for prior in range(1, number + 1):
            attempt = release.VALIDATION_ROOT / f"attempt-{prior:03d}"
            evidence = release.document(release.pin(attempt / "prevalidation.json"))
            assert evidence["status"] == "FAIL"
            assert evidence["failure"]["type"] == "ValueError"
            assert evidence["commands"][-1]["exit_code"] == 1
            for member in evidence["members"]:
                release.authenticate(member)
            for record, (_, _, name) in zip(evidence["sources"], release.SOURCE_LAYOUT, strict=True):
                assert (attempt / name).read_bytes() == release.authenticate(record)


@pytest.fixture
def validation_case(production, tmp_path, monkeypatch):
    """Prospective command receipts are isolated test data, never native evidence."""
    validation = tmp_path / "validation"
    scratch = validation / "attempt-001"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(release, "VALIDATION_ROOT", validation)
    monkeypatch.setattr(release, "preserved_trees", lambda *args: {})
    inputs = release.proposal_context()
    sources = release.implementation_pins()
    for i, ((_, _, name), source) in enumerate(zip(release.SOURCE_LAYOUT, sources, strict=True)):
        (scratch / name).write_bytes(release.authenticate(source))
        (scratch / f"compile-{i}.pyc").write_bytes(b"test-only compile placeholder")
    xml = (
        '<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0">'
        '<testcase classname="release" name="test_fresh_proposed_contract_consumes_production"/>'
        '<testcase classname="release" name="test_actual_successor_v6_owner_and_source_bindings"/>'
        '<testcase classname="release" '
        'name="test_preseal_verdict_matches_real_runtime_classification[continue]"/>'
        '<testcase classname="release" '
        'name="test_preseal_verdict_matches_real_runtime_classification[replan_requested]"/>'
        '</testsuite></testsuites>')
    (scratch / "focused-tests.xml").write_text(xml)
    now = time.time()
    commands = []
    for name, argv in release.validation_commands(scratch, sources):
        environment = release.validation_environment(scratch)
        (scratch / (name + ".command.txt")).write_text(release.shlex.join(argv) + "\n")
        release.write_json(scratch / (name + ".argv.json"), argv)
        release.write_json(scratch / (name + ".environment.json"), environment)
        (scratch / (name + ".stdout")).write_bytes(b"4 passed in 1.00s\n" if name == "focused-tests" else b"")
        (scratch / (name + ".stderr")).write_bytes(b"")
        result = {
            "name": name, "argv": argv, "environment": environment, "cwd": str(release.REPOSITORY),
            "exit_code": 0, "timed_out": False, "started_at": now - 2, "finished_at": now - 1,
            "stdout": release.pin(scratch / (name + ".stdout")),
            "stderr": release.pin(scratch / (name + ".stderr")),
            "test_summary": {"tests": 4, "failures": 0, "errors": 0, "skipped": 0}
            if name == "focused-tests" else None,
        }
        release.write_json(scratch / (name + ".receipt.json"), result)
        commands.append(result)
    evidence = {
        "mission_id": release.MISSION, "status": "PASS", "manager_proposal": release.CONSUMPTION_INPUT,
        "owner_sha256": release.CLAIM_SHA256, "started_at": now - 3, "finished_at": now,
        "sources": sources, "terminal_609": {}, "preserved_trees": {}, "commands": commands,
        "repository_repair": inputs["repository_repair"],
        "members": [release.pin(p) for p in sorted(scratch.iterdir())],
    }

    def check_evidence():
        path = scratch / "prevalidation.json"
        path.write_bytes(release.json_bytes(evidence))
        return release.authenticate_prevalidation(
            release.pin(path), sources, {"published_at": now - 10}, inputs, {})

    return evidence, check_evidence


def test_authenticated_validation_receipt_requires_exact_evidence(validation_case):
    evidence, check_evidence = validation_case
    assert check_evidence()[0] == evidence


def test_validation_receipt_requires_release_binding_coverage(validation_case):
    evidence, check_evidence = validation_case
    member = next(record for record in evidence["members"]
                  if Path(record["path"]).name == "focused-tests.xml")
    path = Path(member["path"])
    path.write_bytes(path.read_bytes().replace(
        b"test_preseal_verdict_matches_real_runtime_classification[continue]", b"unrelated_test"))
    member.update(release.pin(path))
    with pytest.raises(ValueError, match="required release-binding validation coverage missing"):
        check_evidence()


@pytest.mark.parametrize("key,value", [
    ("mission_id", "b38bfad5e968"), ("status", "FAIL"), ("owner_sha256", "other"),
    ("manager_proposal", {}), ("started_at", 0), ("finished_at", 9999999999),
    ("sources", []), ("terminal_609", {"status": "done"}), ("commands", []),
    ("preserved_trees", {"changed": []}), ("repository_repair", {}),
])
def test_validation_receipt_rejects_repinned_bindings(validation_case, key, value):
    evidence, check_evidence = validation_case
    evidence[key] = value
    with pytest.raises(ValueError):
        check_evidence()


@pytest.mark.parametrize("key,value", [
    ("exit_code", 1), ("exit_code", False), ("timed_out", True), ("argv", []), ("environment", {}),
    ("test_summary", {"tests": 3, "failures": 1, "errors": 0, "skipped": 0}),
])
def test_validation_receipt_rejects_changed_command(validation_case, key, value):
    evidence, check_evidence = validation_case
    evidence["commands"][1][key] = value
    with pytest.raises(ValueError):
        check_evidence()


def test_continuation_provenance_is_split():
    inputs = release.proposal_context()
    provenance = release.continuation_provenance(inputs)
    assert provenance == {
        "construction": inputs["accepted_construction"],
        "artifact_review": inputs["accepted_artifact_review"],
    }
    assert provenance["construction"]["mission_id"] == "640a6d6db0cf"
    assert provenance["artifact_review"]["mission_id"] == "4127f6bb1e4a"
    assert provenance["artifact_review"]["native_status"] == "failed_stage_hold_after_reviewer_done"
    assert release.MISSION not in ("640a6d6db0cf", "4127f6bb1e4a")


@pytest.mark.parametrize("lineage,member", [
    ("accepted_construction", "contract"), ("accepted_construction", "seal"),
    ("accepted_construction", "owner"), ("accepted_construction", "authority"),
    ("accepted_construction", "construction_receipt"), ("accepted_construction", "prevalidation"),
    ("accepted_construction", "manager_input"), ("accepted_construction", "zero_invocations"),
    ("accepted_construction", "termination"), ("accepted_construction", "review"),
    ("accepted_artifact_review", "review"), ("accepted_artifact_review", "latest"),
    ("accepted_artifact_review", "mission"), ("accepted_artifact_review", "checkpoint"),
])
@pytest.mark.parametrize("defect", ["missing", "changed"])
def test_continuation_evidence_byte_drift(case, monkeypatch, lineage, member, defect):
    path = case["consumption_input"][lineage][member]["path"]
    if defect == "changed":
        case["virtual"][path] = b"changed\n"
    else:
        original = release.authenticate

        def missing(record):
            if record["path"] == path:
                raise FileNotFoundError(path)
            return original(record)

        monkeypatch.setattr(release, "authenticate", missing)
    with pytest.raises((ValueError, FileNotFoundError)):
        check(case)


@pytest.mark.parametrize("lineage,mission", [
    ("construction", "4127f6bb1e4a"), ("artifact_review", "640a6d6db0cf"),
    ("construction", "fd53d72da1f9"), ("artifact_review", "fd53d72da1f9"),
])
def test_continuation_rejects_conflated_lineage(case, lineage, mission):
    case["selection"]["provenance"][lineage]["mission_id"] = mission
    case["bind_consumption"]()
    with pytest.raises(ValueError, match="provenance"):
        check(case)


@pytest.mark.parametrize("defect", ["done", "missing", "duplicate", "stage_hold", "review_event"])
def test_continuation_rejects_relabelled_review_terminal(case, monkeypatch, defect):
    path = release.LIFE / ("events.jsonl" if defect == "review_event" else "backlog.jsonl")
    original = Path.read_bytes
    lines = []
    for raw in original(path).splitlines():
        row = json.loads(raw)
        if row.get("id") == release.ARTIFACT_REVIEW_MISSION:
            if defect == "missing":
                continue
            if defect == "duplicate":
                lines.append(raw)
            if defect == "done":
                row["status"] = "done"
                raw = release.encoded(row)
            if defect == "stage_hold":
                row["last_error"] = "some other failure"
                raw = release.encoded(row)
        if defect == "review_event" and row.get("type") == "round.review.completed":
            row["status"] = "continue"
            raw = release.encoded(row)
        lines.append(raw)
    changed = b"\n".join(lines) + b"\n"
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: changed if self == path else original(self))
    with pytest.raises(ValueError):
        check(case)


@pytest.mark.parametrize("target", ["mission", "row", "event"])
def test_continuation_requires_stage_skip(case, monkeypatch, target):
    selected = case["selection"]["selected"]
    if target == "mission":
        mission = release.document(selected["mission"])
        mission["tags"] = []
        selected["mission"] = case["put"](selected["mission"]["path"], mission)
    else:
        name = "backlog.jsonl" if target == "row" else "events.jsonl"
        path = release.LIFE / name
        original = Path.read_bytes
        lines = []
        for raw in original(path).splitlines():
            value = json.loads(raw)
            if value.get("id" if target == "row" else "item_id") == release.MISSION:
                if target == "row":
                    value["tags"] = []
                else:
                    value["outcome"]["stage_certification"] = "certified"
                raw = release.encoded(value)
                key = "backlog_row_sha256" if target == "row" else "mission_completed_event_sha256"
                selected["native_terminal"][key] = release.digest(raw)
            lines.append(raw)
        changed = b"\n".join(lines) + b"\n"
        monkeypatch.setattr(Path, "read_bytes",
                            lambda self: changed if self == path else original(self))
    case["bind_consumption"]()
    with pytest.raises(ValueError, match="skip campaign stage transition"):
        check(case)


@pytest.mark.parametrize("index", range(2, 6))
def test_continuation_rejects_live_candidate_capture_drift(case, monkeypatch, index):
    path = release.REPOSITORY / release.SOURCE_LAYOUT[index][1]
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: b"changed\n" if self == path else original(self))
    with pytest.raises(ValueError, match="authenticated bytes|candidate/capture drift"):
        check(case)


@pytest.mark.parametrize("key", [
    "identity", "launch", "launch_constraints", "execution_contract", "accepted_lineage",
    "original_scientific_lineage", "operational_provenance",
])
def test_continuation_preserves_scientific_contract(case, key):
    changed = copy.deepcopy(case["contract"])
    changed[key] = {}
    case["rebind_contract"](changed)
    with pytest.raises(ValueError, match="immutable construction scientific contract"):
        check(case)


@pytest.mark.parametrize("defect", [
    "missing-members", "extra-member", "wrong-member", "missing-reference",
    "npy-member", "npz-type", "npy-type", "missing-stage11", "duplicate-stage11",
])
def test_preparation_reference_formats_and_census(production, tmp_path, defect):
    contract = copy.deepcopy(production[1])
    references = contract["execution_contract"]["references"]
    if defect == "missing-members":
        del contract["reference_members"]
    elif defect == "extra-member":
        contract["reference_members"]["extra"] = None
    elif defect == "wrong-member":
        contract["reference_members"]["original_input_L23_fp16"] = "stage12"
    elif defect == "missing-reference":
        del references["original_input_final_fp16"]
    elif defect == "npy-member":
        contract["reference_members"]["original_input_final_fp16"] = "stage11"
    else:
        role = "original_input_L23_fp16"
        if defect == "npy-type":
            role = "original_input_final_fp16"
        source = release.authenticate(references[role])
        if defect in ("npz-type", "npy-type"):
            path = tmp_path / ("reference.npy" if defect == "npz-type" else "reference.npz")
            path.write_bytes(source)
        else:
            with zipfile.ZipFile(io.BytesIO(source)) as original:
                member = original.read("stage11.npy")
            path = tmp_path / "reference.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("stage12.npy" if defect == "missing-stage11" else "stage11.npy", member)
                if defect == "duplicate-stage11":
                    with pytest.warns(UserWarning, match="Duplicate name"):
                        archive.writestr("stage11.npy", member)
        references[role] = release.pin(path)
    with pytest.raises(ValueError, match="reference|stage11"):
        release.validate_preparation_contract(contract)


@pytest.mark.parametrize("defect", ["missing", "extra", "path", "bytes", "sha256", "drift"])
@pytest.mark.parametrize("role", [
    "ace3.model.awq_bit_oracle", "ace3.model.stage11_current_runtime_release",
    "ace3.model.candidates." + release.CANDIDATE_NAME, "stage13_suffix", "stage13_tests",
])
def test_preparation_runtime_pin_authentication(production, monkeypatch, defect, role):
    contract = copy.deepcopy(production[1])
    sources = contract["runtime_sources"]
    if defect == "missing":
        del sources[role]
    elif defect == "extra":
        sources["extra"] = sources[role]
    elif defect in ("path", "bytes", "sha256"):
        sources[role][defect] = {"path": "/wrong", "bytes": 0, "sha256": "0" * 64}[defect]
    else:
        path = Path(sources[role]["path"])
        original = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes",
                            lambda self: b"drift" if self == path else original(self))
    with pytest.raises(ValueError, match="runtime_sources"):
        release.validate_preparation_contract(contract)
