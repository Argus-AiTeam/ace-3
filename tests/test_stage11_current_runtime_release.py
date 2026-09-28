import copy
import io
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
import zipfile

import pytest

from ace3.model import stage11_current_runtime_release as release
from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as value_candidate


@pytest.fixture(scope="session", autouse=True)
def retained_release_context():
    """Historical protocol fixtures use live imports, not immutable review pins."""
    path = (release.REPOSITORY / ".argus/live/manager-authority"
            / release.HISTORICAL_AUTHORITY / "build-authorization.json")
    binding = release.prepared_release_binding(path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(release, "ACTIVE_BINDING", binding)
        for name, value in binding.items():
            if name.isupper():
                patch.setattr(release, name, value)
        proposal_context = release.proposal_context

        def fixture_context():
            context = proposal_context()
            if "candidate_binding" not in context:
                context["expected_runtime_sources"].update({
                    value_candidate.__name__: release.pin(value_candidate.SOURCE),
                    "value_sufficiency_source": release.pin(value_candidate.SOURCE),
                    "value_sufficiency_test": release.pin(value_candidate.TEST),
                })
            return context

        patch.setattr(release, "proposal_context", fixture_context)
        yield binding


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    import numpy.testing  # Initialize lazy platform probes before the dispatch guard.

    before = {name for name in sys.modules if ".candidates." in name}
    popen = subprocess.Popen

    def forbidden(*args, **kwargs):
        raise AssertionError("emitter, science, service, or subprocess dispatch is forbidden")

    monkeypatch.setattr(release, "emit_after_claim", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    yield popen
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


def test_historical_fixture_keeps_reviewed_baseline_separate(production):
    baseline = release.document(release.CONSUMPTION_INPUT)
    contract = copy.deepcopy(production[1])
    sources = contract["runtime_sources"]
    candidate = release.pin(value_candidate.SOURCE)
    assert sources[value_candidate.__name__] == sources["value_sufficiency_source"] == candidate
    assert sources["value_sufficiency_test"] == release.pin(value_candidate.TEST)
    retained = baseline["expected_runtime_sources"][value_candidate.__name__]
    assert retained != candidate
    release.validate_preparation_contract(contract)
    sources[value_candidate.__name__] = retained
    with pytest.raises(ValueError, match="runtime_sources"):
        release.validate_preparation_contract(contract)
    assert release.document(release.CONSUMPTION_INPUT) == baseline


@pytest.mark.parametrize("role", [
    value_candidate.__name__, "value_sufficiency_source", "value_sufficiency_test",
])
@pytest.mark.parametrize("defect", ["splice", "drift"])
def test_historical_fixture_rejects_changed_value_sources(production, monkeypatch, role, defect):
    contract = copy.deepcopy(production[1])
    sources = contract["runtime_sources"]
    if defect == "splice":
        sources[role] = sources[
            "value_sufficiency_source" if role == "value_sufficiency_test"
            else "value_sufficiency_test"]
    else:
        path = Path(sources[role]["path"])
        read_bytes = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes",
                            lambda self: b"changed\n" if self == path else read_bytes(self))
    with pytest.raises(ValueError, match="runtime_sources"):
        release.validate_preparation_contract(contract)


def test_fresh_proposed_contract_consumes_production(production, tmp_path, value_case):
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
    assert contract["current_sources"] == release.current_source_pins()
    assert contract["implementation_sources"] == release.implementation_pins()
    assert len(contract["implementation_sources"]) == 6
    for key in ("capture_helper", "capture_tests"):
        assert release.authenticate(contract["capture_snapshots"][key]) == release.authenticate(
            contract["current_sources"][key])
    previous = release.document(accepted["contract"])
    for key in ("identity", "launch_constraints", "accepted_lineage",
                "original_scientific_lineage", "operational_provenance"):
        assert contract[key] == previous[key]
    assert contract["launch"] == release.proposed_launch(previous, release.proposal_context())
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
    assert len(release.test_summary(
        raw.replace(b'tests="293"', b'tests="295"'),
        b"293 passed, 2 subtests passed in 61.00s (0:01:01)\n")) == len(release.ZERO_KINDS)
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
    for stdout in (
        b"293 passed, 2 subtests passed in 1.00s\n",
        b"291 passed, 2 subtests passed in 1.00s\n",
        b"290 passed, 2 subtests passed in 1.00s\n",
        b"291 passed, 2 subtests failed in 1.00s\n",
    ):
        with pytest.raises(ValueError):
            release.test_summary(raw, stdout)


@pytest.fixture
def case(production, tmp_path, monkeypatch):
    run, original, provenance_native_bytes = production
    contract = copy.deepcopy(original)
    now = round(time.time(), 6)
    mission = release.proposal_context().get("candidate_binding", {}).get(
        "future_science_mission", "abcdef123456")
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

    def materialize(edit=None, *, extra_records=None):
        values = copy.deepcopy(templates)
        locations = {str(old_root / relative): relative for relative in values}
        for relative, record in (extra_records or {}).items():
            assert relative not in values
            values[relative] = release.document(record)
            locations[record["path"]] = relative
        if edit:
            edit(values)
        pins = {}
        active = set()

        def relocate(value):
            if isinstance(value, dict):
                if set(value) == {"path", "bytes", "sha256"}:
                    if value["path"] in locations:
                        return write(locations[value["path"]])
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
        copied_contract["original_scientific_lineage"] = relocate(
            copied_contract["original_scientific_lineage"])
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
    ("review", ["review", "reason"], ""),
    ("review", ["review", "reason"], 1),
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


def test_native_acceptance_allows_authentic_done_reason_variation(case):
    bundle = case["request"]["contract_review"]
    review = release.document(bundle["review"])
    review["review"]["reason"] = (
        "sole zero-science release is sealed from reviewed live bytes; "
        "six-source compilation and validation passed"
    )
    bundle["review"] = case["put"](bundle["review"]["path"], review)
    case["bind_consumption"]()
    check(case)


def test_selection_document_can_be_fully_prevalidated_before_write(case):
    selected = case["selection"]["selected"]
    review, checked = release.validate_operational_release_selection(
        copy.deepcopy(case["selection"]),
        case["selection_path"],
        case["request"]["contract_review"],
        selected["contract"],
        selected["seal"],
    )
    assert review["producer_role"] == "reviewer"
    assert checked["selected"]["mission_id"] == release.MISSION


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
     str(release.LIFE / "handoffs" / release.HISTORICAL_MISSION / "round-0001.json")),
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


def test_isolated_value_successor_owner_and_source_bindings(value_binding_case, monkeypatch):
    historical = (
        release.REPOSITORY / release.HISTORICAL_ROOT,
        release.REPOSITORY / release.HISTORICAL_VALIDATION_ROOT,
        release.REPOSITORY / ".argus/live/manager-authority" / release.HISTORICAL_AUTHORITY,
    )
    before = {str(path): release.tree_state(path) for path in historical}
    binding = value_binding_case
    assert binding["MISSION"] != release.HISTORICAL_MISSION
    assert binding["AUTHORITY"] != release.HISTORICAL_AUTHORITY
    assert binding["ROOT"] != release.HISTORICAL_ROOT
    assert binding["VALIDATION_ROOT"] != release.HISTORICAL_VALIDATION_ROOT
    assert binding["CONSUMPTION_INPUT"] != release.HISTORICAL_INPUT
    assert binding["CONSUMPTION_DECISION"] != release.HISTORICAL_DECISION
    monkeypatch.setattr(release, "ACTIVE_BINDING", None)
    for name in binding:
        if name.isupper():
            monkeypatch.setattr(release, name, None)
    proposal_pin, proposal, _, owner, _ = release.successor_inputs()
    assert release.PROVENANCE_MISSION == "f1c261d97727"
    assert proposal_pin == release.CONSUMPTION_INPUT
    assert owner["schema"] == "argus.manager-stage11-value-content-release-owner.v7"
    assert owner["accepted_release_mission"] == release.HISTORICAL_MISSION
    assert owner["candidate_review_mission"] == proposal["candidate_review"]["mission_id"]
    assert owner["candidate_review_mission"] == "3362da556040"
    assert owner["candidate_binding"] == proposal["candidate_binding"]
    assert owner["prior_science_mission"] == "b4978b898600"
    assert release.source_edit_bindings(proposal, owner) == {
        role: "POSTCLAIM_EDIT" if role in (
            "release_builder", "release_tests", "candidate", "candidate_tests")
        else "UNCHANGED_MANAGER_BASELINE" for role, _, _ in release.SOURCE_LAYOUT
    }
    assert {str(path): release.tree_state(path) for path in historical} == before


@pytest.fixture
def value_binding_case(prepared_binding_case, production, monkeypatch):
    """Resolve disposable v7 authority/native/runtime state against retained provenance."""
    path, authority, row = prepared_binding_case
    repository = Path(release.__file__).resolve().parents[2]
    donor_path = repository / (
        ".argus/live/manager-authority/"
        "s11-value-content-release-compatibility-20260926-6ef353b3fc6d-r1/"
        "build-authorization.json")
    donor = release.document(release.pin(donor_path))
    life = Path(donor["life_dir"])
    proposal = release.document(donor["manager_inputs"]["source_input"])
    decision = release.document(donor["manager_inputs"]["consumption_decision"])
    owner = release.document(release.pin(donor_path.with_name("manager-owner-claim.json")))
    target = proposal["candidate_binding"]
    target.update(
        future_science_mission="8ee0b3aaeecb",
        output=str(repository / "build" / (target["name"] + "-8ee0b3aaeecb-attempt001")),
        capture_root="build/stage11-value-content-authorized-8ee0b3aaeecb-attempt001",
    )
    launch = target["launch"]
    launch["argv"] = [
        sys.executable, "-B", "-m", target["module"], "--check", "--output", target["output"],
    ]
    launch["command"] = release.shlex.join([
        *(f"{key}={value}" for key, value in sorted(launch["environment"].items())),
        *launch["argv"],
    ])
    decision.update(
        future_science_mission=target["future_science_mission"],
        future_candidate_output=str(Path(target["output"]).relative_to(repository)),
        future_capture_root=target["capture_root"],
    )
    owner["candidate_binding"] = copy.deepcopy(target)
    authority.update(mission_id="012345abcdef",
                     authority_id="s11-value-content-forward-release-isolated",
                     workspace=str(repository), life_dir=str(life),
                     expected_objective="isolated value release")
    row["id"] = authority["mission_id"]
    proposal["mission_id"] = decision["mission_id"] = row["id"]
    for key in ("authorized_root", "validation_root"):
        proposal["repair"][key] = decision[key] = authority[key]
    owner.update(mission_id=row["id"], authority_id=authority["authority_id"],
                 authorized_root=authority["authorized_root"])
    published = min(time.time(), *(Path(record["path"]).stat().st_mtime
                                  for record in proposal["source_baseline"].values())) - 10
    row.update(running_owner="primary", attempt=1, started_ts=published - 2,
               objective=authority["expected_objective"], tags=["stage_transition:skip"])
    event = {
        "type": "life.mission.started", "item_id": row["id"], "attempt": 1,
        "independent_review_required": True, "usage_attempt_id": row["id"] + ":attempt:1",
        "ts": published - 1,
    }
    row_raw, event_raw = release.encoded(row), release.encoded(event)
    owner.update(published_at=published, native_claim={
        "attempt": 1, "backlog_row_sha256": release.digest(row_raw),
        "mission_started_event_sha256": release.digest(event_raw),
        "mission_started_ts": event["ts"], "owner": "primary",
        "started_ts": row["started_ts"], "usage_attempt_id": event["usage_attempt_id"],
    })
    virtual = {
        life / "backlog.jsonl": (life / "backlog.jsonl").read_bytes() + b"\n" + row_raw + b"\n",
        life / "events.jsonl": (life / "events.jsonl").read_bytes() + b"\n" + event_raw + b"\n",
    }
    read_bytes, is_file, glob = Path.read_bytes, Path.is_file, Path.glob
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: virtual[self] if self in virtual else read_bytes(self))
    monkeypatch.setattr(Path, "is_file",
                        lambda self: self in virtual or is_file(self))

    def put(target, value):
        virtual[target] = release.json_bytes(value)
        return release.pin(target)

    expected = proposal["expected_runtime"]
    runtime = copy.deepcopy(production[1]["runtime"])
    runtime.update(
        pid=expected["pid"], source_root=expected["argus_source_root"],
        source_fingerprint=expected["source_fingerprint"],
        continuous_enabled=True, continuous_generation=expected["continuous_generation"],
        source_record=put(life / "daemon.source.json", {
            "pid": expected["pid"], "source_root": expected["argus_source_root"],
            "source_fingerprint": expected["source_fingerprint"],
            "source_file_count": expected["source_file_count"],
        }),
    )
    put(life / "daemon.status.json", {
        "pid": expected["pid"], "started_at_iso": owner["runtime"]["started_at_iso"],
        "runtime": {
            "release_id": expected["release_id"], "source_root": expected["source_root"],
            "runtime_source_digest": expected["source_digest"],
            "manifest_source_digest": expected["source_digest"], "release_matches_source": True,
        },
    })
    put(life / "continuous.json", {
        "enabled": True, "generation": expected["continuous_generation"],
        "open_ended": True, "objective": expected["continuous_objective"],
    })
    directory = repository / ".argus/live/manager-authority"
    authority_path = directory / authority["authority_id"] / path.name
    authority["manager_inputs"] = {
        "source_input": put(repository / ".argus/live/manager-inputs"
                            / f"stage11-reviewed-interface-release-{row['id']}.json", proposal),
        "consumption_decision": put(
            authority_path.with_name("manager-consumption-decision.json"), decision),
        "claim_binder": release.pin(Path(donor["manager_inputs"]["claim_binder"]["path"])),
    }
    put(authority_path.with_name("manager-owner-claim.json"), owner)
    put(authority_path, authority)
    monkeypatch.setattr(Path, "glob", lambda self, pattern, **kwargs: (
        iter([authority_path] if pattern == "s11-value-content-forward-release-*/build-authorization.json"
             else []) if self == directory else glob(self, pattern, **kwargs)))
    monkeypatch.setattr(release, "REPOSITORY", repository)
    monkeypatch.setattr(release, "LIFE", life)
    monkeypatch.setattr(release, "observe_runtime", lambda: copy.deepcopy(runtime))
    binding = release.resolve_release_binding()
    assert binding["MISSION"] == row["id"]
    return binding


@pytest.mark.parametrize("status", ["done", "failed"])
def test_isolated_value_binding_rejects_terminal_owner(value_binding_case, monkeypatch, status):
    path = release.LIFE / "backlog.jsonl"
    rows = [json.loads(raw) for raw in path.read_bytes().splitlines() if raw.strip()]
    for row in rows:
        if row["id"] == value_binding_case["MISSION"]:
            row["status"] = status
    changed = b"\n".join(release.encoded(row) for row in rows) + b"\n"
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: changed if self == path else read_bytes(self))
    with pytest.raises(ValueError, match="completed-or-terminal-owner"):
        release.resolve_release_binding()


@pytest.fixture
def prepared_binding_case(tmp_path, monkeypatch, retained_release_context, request):
    """Synthetic Manager preparation; no native claim or publisher is invoked."""
    repository = tmp_path / "repository"
    life = tmp_path / "life"
    life.mkdir()
    historical_path = Path(retained_release_context["authority"]["path"])
    authority = release.document(retained_release_context["authority"])
    proposal = release.document(release.HISTORICAL_INPUT)
    decision = release.document(release.HISTORICAL_DECISION)
    mission = "test-unfinished-release"
    family = getattr(request, "param", "s11-reviewed-interface-release")
    authority.update(mission_id=mission, authority_id=family + "-test",
                     workspace=str(repository), life_dir=str(life),
                     authorized_root="build/test-unfinished-release",
                     validation_root=".argus/live/manager-validation/test-unfinished-release")
    proposal["mission_id"] = decision["mission_id"] = mission
    for key in ("authorized_root", "validation_root"):
        proposal["repair"][key] = decision[key] = authority[key]
    path = (repository / ".argus/live/manager-authority"
            / authority["authority_id"] / "build-authorization.json")
    path.parent.mkdir(parents=True)
    input_path = (repository / ".argus/live/manager-inputs"
                  / f"stage11-reviewed-interface-release-{mission}.json")
    input_path.parent.mkdir(parents=True)
    binder = repository / ".argus/live/manager-tools/publisher.py"
    binder.parent.mkdir(parents=True)
    binder.write_bytes(b"test-only, never executed\n")
    release.write_json(input_path, proposal)
    release.write_json(path.with_name("manager-consumption-decision.json"), decision)
    path.with_name("manager-owner-claim.json").write_bytes(
        historical_path.with_name("manager-owner-claim.json").read_bytes())
    authority["manager_inputs"] = {
        "source_input": release.pin(input_path),
        "consumption_decision": release.pin(path.with_name("manager-consumption-decision.json")),
        "claim_binder": release.pin(binder),
    }
    release.write_json(path, authority)
    row = {"id": mission, "status": "running"}
    (life / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    monkeypatch.setattr(release, "REPOSITORY", repository)
    monkeypatch.setattr(release, "LIFE", life)
    return path, authority, row


@pytest.mark.parametrize("defect", [
    "none", "pending", "done", "failed", "superseded", "root", "symlink", "overlap",
])
def test_prepared_release_resolution(prepared_binding_case, defect, monkeypatch):
    path, authority, row = prepared_binding_case
    if defect in ("pending", "done", "failed", "superseded"):
        row["status"] = defect
        (release.LIFE / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    root = release.REPOSITORY / authority["authorized_root"]
    if defect in ("root", "symlink"):
        root.parent.mkdir(parents=True)
        if defect == "root":
            root.mkdir()
        else:
            root.symlink_to(root.with_name("absent"))
    if defect == "overlap":
        other = path.parent.with_name("s11-reviewed-interface-release-other") / path.name
        other.parent.mkdir()
        release.write_json(other, authority)
    if defect in ("none", "pending"):
        binding = release.resolve_release_binding()
        assert binding["MISSION"] == row["id"] != release.HISTORICAL_MISSION
        assert binding["ROOT"] == Path(authority["authorized_root"])
        assert binding["CONSUMPTION_INPUT"] == authority["manager_inputs"]["source_input"]
        assert binding["CONSUMPTION_DECISION"] == authority["manager_inputs"]["consumption_decision"]
        assert binding["VALIDATION_ROOT"] == Path(authority["validation_root"])
    else:
        monkeypatch.setattr(release, "prepared_release_binding",
                            lambda *args: pytest.fail("rejected owner reached claim consumption"))
        expected = "ambiguous_objective" if defect == "overlap" else (
            "existing-final-root" if defect in ("root", "symlink") else "completed-or-terminal-owner")
        with pytest.raises(ValueError, match=expected):
            release.successor_inputs()


@pytest.mark.parametrize("key", ["mission_id", "authorized_root", "validation_root"])
def test_prepared_release_binding_splices(prepared_binding_case, key):
    path, authority, _ = prepared_binding_case
    decision_path = path.with_name("manager-consumption-decision.json")
    decision = release.document(authority["manager_inputs"]["consumption_decision"])
    decision[key] = "spliced"
    decision_path.write_bytes(release.json_bytes(decision))
    authority["manager_inputs"]["consumption_decision"] = release.pin(decision_path)
    path.write_bytes(release.json_bytes(authority))
    with pytest.raises(ValueError, match="Manager input/authority/decision/validation binding differs"):
        release.resolve_release_binding()


@pytest.mark.parametrize("prepared_binding_case", [
    "s11-value-content-forward-release", "s11-value-content-selftest-release",
    "s11-value-content-current-census-release",
], indirect=True)
@pytest.mark.parametrize("defect", [
    "none", "pending", "done", "failed", "superseded", "root", "symlink", "overlap",
])
def test_forward_release_resolution(prepared_binding_case, defect, monkeypatch):
    test_prepared_release_resolution(prepared_binding_case, defect, monkeypatch)


@pytest.mark.parametrize("prepared_binding_case", [
    "s11-value-content-forward-release", "s11-value-content-selftest-release",
    "s11-value-content-current-census-release",
], indirect=True)
@pytest.mark.parametrize("key", ["mission_id", "authorized_root", "validation_root"])
def test_forward_release_binding_splices(prepared_binding_case, key):
    test_prepared_release_binding_splices(prepared_binding_case, key)


@pytest.mark.parametrize("prepared_binding_case", [
    "s11-value-content-forward-release", "s11-value-content-selftest-release",
    "s11-value-content-current-census-release",
], indirect=True)
@pytest.mark.parametrize("defect", [
    "duplicate-row", "duplicate-owner", "authorized_root", "validation_root",
    "missing-historical-owner", "duplicate-historical-row", "historical-distinct",
    "superseded-owner",
])
def test_forward_release_rejects_duplicate_or_overlapping_history(prepared_binding_case, defect):
    path, authority, row = prepared_binding_case
    rows = [row]
    if defect == "duplicate-row":
        rows.append(row)
        message = "missing or duplicate prepared release owner"
    else:
        other = copy.deepcopy(authority)
        other["authority_id"] = "s11-value-content-selftest-release-historical"
        if defect == "duplicate-owner":
            root = release.REPOSITORY / authority["authorized_root"]
            root.mkdir(parents=True)
            other["authorized_root"] = "build/other-release"
            message = "duplicate release owner authorities"
        else:
            other["mission_id"] = (
                "f1a28e051c63" if defect == "superseded-owner" else "historical-owner")
            historical = {"id": other["mission_id"],
                          "status": "superseded" if defect == "superseded-owner" else "failed"}
            if defect == "superseded-owner":
                other["authority_id"] = "s11-value-content-current-census-release-historical"
            if defect != "missing-historical-owner":
                rows.append(historical)
            if defect == "duplicate-historical-row":
                rows.append(historical)
            other["authorized_root"] = "build/historical-owner"
            other["validation_root"] = ".argus/live/manager-validation/historical-owner"
            if defect in ("authorized_root", "validation_root"):
                other[defect] = str(Path(authority[defect]) / "nested")
                message = "prepared release overlaps historical " + defect
            else:
                message = "missing or duplicate prepared release owner"
        other_path = path.parent.with_name(other["authority_id"]) / path.name
        other_path.parent.mkdir()
        release.write_json(other_path, other)
    (release.LIFE / "backlog.jsonl").write_bytes(
        b"".join(release.encoded(item) + b"\n" for item in rows))
    if defect in ("historical-distinct", "superseded-owner"):
        assert release.resolve_release_binding()["MISSION"] == row["id"]
        return
    with pytest.raises(ValueError, match=message):
        release.resolve_release_binding()


@pytest.mark.parametrize("defect", ["none", "path", "drift"])
def test_value_publication_focused_command_preserves_complete_suites(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(release, "REPOSITORY", tmp_path)
    monkeypatch.setattr(release, "resolve_release_binding",
                        lambda: pytest.fail("synthetic command must not resolve a live owner"))
    publication = tmp_path / "tests/test_stage11_publication_compatibility.py"
    publication.parent.mkdir()
    publication.write_bytes(b"# isolated reviewed compatibility source\n")
    record = release.pin(publication)
    proposal = {"candidate_binding": {}, "publication_repair": {},
                "candidate_review": {"publication_tests": record}}
    monkeypatch.setattr(release, "proposal_context", lambda: proposal)
    if defect == "path":
        record["path"] = str(publication.with_name("other.py"))
    elif defect == "drift":
        publication.write_bytes(b"# changed after review\n")
    if defect != "none":
        with pytest.raises(ValueError, match="test path differs|authenticated bytes changed"):
            release.focused_command(tmp_path, {})
        return
    argv = release.focused_command(tmp_path, {})
    assert argv[7:10] == [
        str(release.TEST), str(release.SOURCE_LAYOUT[3][1]),
        "tests/test_stage11_publication_compatibility.py",
    ]
    assert not any("::" in arg for arg in argv)
    proposal.pop("publication_repair")
    assert release.focused_command(tmp_path, {})[7:9] == [
        str(release.TEST), str(release.SOURCE_LAYOUT[3][1])]


@pytest.mark.parametrize("defect", [
    "none", "path", "test-drift", "contract-drift", "proposal-drift",
    "contract-identity", "proposal-identity", "missing", "cycle",
])
def test_value_publication_focused_command_authenticates_ancestry(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(release, "REPOSITORY", tmp_path)
    monkeypatch.setattr(release, "resolve_release_binding",
                        lambda: pytest.fail("isolated ancestry must not resolve a live owner"))
    publication = tmp_path / "tests/test_stage11_publication_compatibility.py"
    publication.parent.mkdir()
    publication.write_bytes(b"# isolated accepted compatibility suite\n")
    record = release.pin(publication)
    if defect == "path":
        record["path"] = str(publication.with_name("other.py"))
    ancestor = {"mission_id": "accepted", "candidate_review": {"publication_tests": record}}
    if defect in ("missing", "cycle"):
        ancestor["candidate_review"].clear()
    if defect == "proposal-identity":
        ancestor["mission_id"] = "spliced"
    ancestor_path, contract_path = tmp_path / "proposal.json", tmp_path / "contract.json"
    release.write_json(ancestor_path, ancestor)
    contract = {"mission_id": "spliced" if defect == "contract-identity" else "accepted",
                "manager_proposal": release.pin(ancestor_path)}
    release.write_json(contract_path, contract)
    proposal = {"candidate_binding": {}, "publication_repair": {}, "candidate_review": {},
                "accepted_release": {"mission_id": "accepted",
                                     "contract": release.pin(contract_path)}}
    if defect == "cycle":
        ancestor["accepted_release"] = proposal["accepted_release"]
        document = release.document
        monkeypatch.setattr(release, "document",
                            lambda pin: ancestor if pin == contract["manager_proposal"]
                            else document(pin))
    monkeypatch.setattr(release, "proposal_context", lambda: proposal)
    for name, path in (("test-drift", publication), ("contract-drift", contract_path),
                       ("proposal-drift", ancestor_path)):
        if defect == name:
            path.write_bytes(path.read_bytes() + b"\n")
    if defect != "none":
        with pytest.raises(ValueError, match=(
                "test path differs|authenticated bytes changed|ancestor identity differs"
                "|test ancestry missing|cyclic publication")):
            release.focused_command(tmp_path, {})
        return
    argv = release.focused_command(tmp_path, {})
    assert argv[7:10] == [
        str(release.TEST), str(release.SOURCE_LAYOUT[3][1]),
        "tests/test_stage11_publication_compatibility.py",
    ]
    assert "publication_tests" not in proposal["candidate_review"]
    assert not any("::" in arg for arg in argv)


def test_legacy_focused_command_selects_isolated_owner_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "proposal_context", lambda: {})
    monkeypatch.setattr(release, "resolve_release_binding",
                        lambda: pytest.fail("synthetic command must not resolve a live owner"))
    argv = release.focused_command(tmp_path, {})
    assert str(release.TEST) + "::test_isolated_value_successor_owner_and_source_bindings" in argv
    assert not any("test_actual_successor_v6_owner_and_source_bindings" in arg for arg in argv)


@pytest.fixture
def construction_case(prepared_binding_case, retained_release_context, monkeypatch):
    """Run the builder lifecycle on synthetic data, not on a live release root."""
    authority_path, authority, row = prepared_binding_case
    donor = release.document(retained_release_context["authority"])
    donor_life = Path(donor["life_dir"])
    proposal_path = Path(authority["manager_inputs"]["source_input"]["path"])
    proposal = release.document(release.pin(proposal_path))
    previous = release.document(proposal["accepted_release"]["contract"])
    now = time.time()
    row.update(running_owner="primary", attempt=1, started_ts=now - 20,
               objective=authority["expected_objective"], tags=["stage_transition:skip"])
    event = {
        "type": "life.mission.started", "item_id": row["id"], "attempt": 1,
        "independent_review_required": True, "usage_attempt_id": row["id"] + ":attempt:1",
        "ts": now - 19,
    }
    row_raw, event_raw = release.encoded(row), release.encoded(event)
    failed = previous["terminal_successor_lineage"]["failed_terminal_successor"]
    history = [
        raw for raw in (donor_life / "backlog.jsonl").read_bytes().splitlines()
        if raw.strip() and json.loads(raw).get("id") in (
            release.FAILED_PREDECESSOR_MISSION, failed["mission_id"])
    ]
    (release.LIFE / "backlog.jsonl").write_bytes(b"\n".join([*history, row_raw]) + b"\n")
    (release.LIFE / "events.jsonl").write_bytes(event_raw + b"\n")
    for name in ("daemon.status.json", "continuous.json"):
        (release.LIFE / name).write_bytes((donor_life / name).read_bytes())
    expected = proposal["expected_runtime"]
    runtime = {
        "pid": expected["pid"], "source_root": expected["argus_source_root"],
        "source_fingerprint": expected["source_fingerprint"],
        "source_record": release.pin(donor_life / "daemon.source.json"),
        "continuous_enabled": True, "continuous_generation": expected["continuous_generation"],
    }
    for role, path, _ in release.HISTORICAL_SOURCE_LAYOUT:
        source = release.REPOSITORY / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(("# synthetic " + role + "\n").encode())
        os.utime(source, (now - 6, now - 6))
        proposal["source_baseline"][role] = release.pin(source)
    proposal_path.write_bytes(release.json_bytes(proposal))
    authority["manager_inputs"]["source_input"] = release.pin(proposal_path)
    authority_path.write_bytes(release.json_bytes(authority))
    owner_path = authority_path.with_name("manager-owner-claim.json")
    owner = release.document(release.pin(owner_path))
    owner.update(
        mission_id=row["id"], authority_id=authority["authority_id"],
        authorized_root=authority["authorized_root"], published_at=now - 10,
        native_claim={
            "attempt": 1, "backlog_row_sha256": release.digest(row_raw),
            "mission_started_event_sha256": release.digest(event_raw),
            "mission_started_ts": event["ts"], "owner": "primary",
            "started_ts": row["started_ts"], "usage_attempt_id": row["id"] + ":attempt:1",
        })
    owner_path.write_bytes(release.json_bytes(owner))
    binding = release.prepared_release_binding(authority_path)
    monkeypatch.setattr(release, "ACTIVE_BINDING", None)
    for name in binding:
        if name.isupper():
            monkeypatch.setattr(release, name, getattr(release, name))
    monkeypatch.chdir(release.REPOSITORY)
    monkeypatch.setattr(release, "proposal_context", lambda: release.document(
        authority["manager_inputs"]["source_input"]))
    monkeypatch.setattr(release, "observe_runtime", lambda: copy.deepcopy(runtime))
    # Retained ancestor acceptance is covered separately; lifecycle authentication
    # below uses the real resolver, successor_inputs, source and native validators.
    monkeypatch.setattr(release, "reviewed_repair_evidence", lambda proposal: {})
    monkeypatch.setattr(release, "continuation_provenance", lambda proposal: {})
    monkeypatch.setattr(release, "native_acceptance", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "construction_input", lambda proposal: {
        "failed_science_terminal": {},
        "accepted_operational_release_selection": proposal["accepted_release"]["selection"],
    })
    monkeypatch.setattr(release, "accepted_lineage", lambda *args: None)
    monkeypatch.setattr(release, "preserved_trees", lambda *args: {})
    sources = [proposal["source_baseline"][role] for role, _, _ in binding["SOURCE_LAYOUT"]]
    scratch = release.REPOSITORY / binding["VALIDATION_ROOT"] / "attempt-001"
    scratch.mkdir(parents=True)
    members = {name: release.authenticate(source) for (_, _, name), source in zip(
        binding["SOURCE_LAYOUT"], sources, strict=True)}
    contract = {
        "created_at": now - 4, "runtime": runtime, "accepted_lineage": {},
        "source_snapshots": {}, "capture_snapshots": {}, "historical_state": {},
        "boundary": "synthetic zero-science lifecycle only",
    }
    members["current-runtime-contract.json"] = release.json_bytes(contract)
    for name, raw in members.items():
        (scratch / name).write_bytes(raw)
    receipt = {
        "mission_id": row["id"], "owner_sha256": binding["CLAIM_SHA256"],
        "manager_proposal": binding["CONSUMPTION_INPUT"],
        "started_at": now - 4, "finished_at": now - 2, "preserved_trees": {},
        "members": [release.pin(path) for path in sorted(scratch.iterdir())],
    }
    release.write_json(scratch / "prevalidation.json", receipt)
    receipt_pin = release.pin(scratch / "prevalidation.json")
    monkeypatch.setattr(release, "authenticate_prevalidation",
                        lambda *args: (copy.deepcopy(receipt), dict(members), {}))
    monkeypatch.setattr(release, "proposed_contract", lambda *args: copy.deepcopy(contract))
    base = release.LIFE / "handoffs" / row["id"]
    base.mkdir(parents=True)
    review_path = base / "round-0001.json"
    release.write_json(review_path, {
        "kind": "round_reviewed_handoff", "schema_version": 3,
        "mission_id": row["id"], "producer_role": "reviewer", "round": 1,
        "created_at": now - 1, "mission_context": str(base / "mission.json"),
        "review": {"status": "continue", "next_action": "BUILD_FINAL_FROM_REVIEWED_BYTES"},
    })
    release.write_json(base / "latest.json", {
        "kind": "handoff_ref", "mission": {"path": str(base / "mission.json")},
        "handoff": {"path": str(review_path)},
    })
    root = release.REPOSITORY / binding["ROOT"]
    root.parent.mkdir()
    selections = []
    resolve = release.resolve_release_binding

    def select():
        selections.append(root.exists())
        return resolve()

    monkeypatch.setattr(release, "resolve_release_binding", select)
    # Scope the otherwise permanent process audit hook to this test's real opens.
    hooks = []
    open_path = Path.open
    monkeypatch.setattr(sys, "addaudithook", hooks.append)

    def audited_open(path, mode="r", *args, **kwargs):
        for hook in hooks:
            hook("open", (path, mode, 0))
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    virtual = {}
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda path: virtual[path] if path in virtual else read_bytes(path))
    return {
        "root": root, "receipt": receipt_pin, "sources": sources, "selections": selections,
        "virtual": virtual, "row": row, "event": event, "history": history,
        "authority": authority_path, "owner": owner_path, "proposal": proposal_path,
    }


def test_build_completes_through_termination_and_seal(construction_case):
    case = construction_case
    release.build(case["receipt"])
    root = case["root"]
    assert case["selections"] == [False]
    seal = release.document(release.pin(root / "seal.json"))
    assert seal["mission_id"] == release.MISSION
    assert seal["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
    assert seal["scientific_run_budget"] == 0
    assert seal["excludes"] == ["seal.json"]
    assert {member["path"] for member in seal["members"]} == {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and path.name != "seal.json"
    }
    for member in seal["members"]:
        release.authenticate(dict(member, path=str(root / member["path"])))
    termination = release.document(release.pin(root / "termination-receipt.json"))
    assert termination["scientific_invocations"] == termination["stage_transitions"] == 0
    assert termination["release_consumption_authorized"] is False
    construction = release.document(release.pin(root / "construction-receipt.json"))
    assert construction["sources"] == case["sources"]
    assert construction["builder_invocations"] == 1
    assert construction["prevalidation_completed_before_root"] is True
    audit = release.document(release.pin(root / "audit-disclosure.json"))["builder_audit"]
    assert audit["writes"] and audit["blocked"] == []
    assert all(Path(path).is_relative_to(root) for path in audit["writes"])
    before = release.tree_state(root)
    with pytest.raises(ValueError, match="existing-final-root"):
        release.build(case["receipt"])
    assert release.tree_state(root) == before


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_build_rejects_occupied_root(construction_case, kind):
    root = construction_case["root"]
    if kind == "directory":
        root.mkdir()
        (root / "sentinel").write_bytes(b"preserve occupied root\n")
    else:
        root.symlink_to(root.with_name("absent"))
    before = release.tree_state(root)
    with pytest.raises(ValueError, match="existing-final-root"):
        release.build(construction_case["receipt"])
    assert release.tree_state(root) == before
    assert not (root / "seal.json").exists()


@pytest.mark.parametrize("phase", ["before-root", "after-root"])
@pytest.mark.parametrize("defect", [
    "claim", "event", "other-owner", "source", "candidate", "candidate-tests",
    "authority", "owner", "proposal",
])
def test_build_rejects_construction_drift(construction_case, monkeypatch, phase, defect):
    case = construction_case
    root = case["root"]

    def change():
        if defect in ("claim", "other-owner"):
            row = dict(case["row"])
            row["objective" if defect == "claim" else "running_owner"] = "changed"
            case["virtual"][release.LIFE / "backlog.jsonl"] = b"\n".join(
                [*case["history"], release.encoded(row)]) + b"\n"
        elif defect == "event":
            event = dict(case["event"], ts=case["event"]["ts"] + 0.5)
            case["virtual"][release.LIFE / "events.jsonl"] = release.encoded(event) + b"\n"
        else:
            if defect in ("source", "candidate", "candidate-tests"):
                index = {"source": 0, "candidate": 2, "candidate-tests": 3}[defect]
                path = Path(case["sources"][index]["path"])
            else:
                path = case[defect]
            case["virtual"][path] = path.read_bytes() + b"\n"

    if phase == "before-root":
        propose = release.proposed_contract

        def changed_proposal(*args):
            result = propose(*args)
            change()
            return result

        monkeypatch.setattr(release, "proposed_contract", changed_proposal)
    else:
        exclusive = release.exclusive

        def changed_write(path, raw):
            exclusive(path, raw)
            if path == root / "zero-invocations.json":
                change()

        monkeypatch.setattr(release, "exclusive", changed_write)
    with pytest.raises(ValueError, match=(
            "authenticated bytes changed|native owner claim differs|"
            "normal running claim required|reviewed ed9 candidate/capture drift")):
        release.build(case["receipt"])
    assert case["selections"] == [False]
    assert root.exists() is (phase == "after-root")
    assert not (root / "termination-receipt.json").exists()
    assert not (root / "seal.json").exists()


@pytest.mark.parametrize("schema", [
    "argus.manager-stage11-continuous-release-owner.v4",
    "argus.manager-stage11-execution-contract-repair-owner.v5",
])
def test_prepared_release_rejects_obsolete_owner_schema(prepared_binding_case, schema):
    path, _, _ = prepared_binding_case
    owner_path = path.with_name("manager-owner-claim.json")
    owner = release.document(release.pin(owner_path))
    owner["schema"] = schema
    owner_path.write_bytes(release.json_bytes(owner))
    with pytest.raises(ValueError, match="prepared release requires v6 owner schema"):
        release.resolve_release_binding()


@pytest.mark.parametrize("schema", [
    "argus.manager-stage11-operational-successor-owner.v1",
    "argus.manager-stage11-continuation-acceptance-owner.v3",
    "argus.manager-stage11-continuous-release-owner.v4",
    "argus.manager-stage11-execution-contract-repair-owner.v5",
])
def test_successor_rejects_obsolete_owner_schema(monkeypatch, schema, retained_release_context):
    monkeypatch.setattr(release, "resolve_release_binding", lambda: retained_release_context)
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
def test_successor_rejects_owner_runtime_drift(monkeypatch, key, value, retained_release_context):
    monkeypatch.setattr(release, "resolve_release_binding", lambda: retained_release_context)
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
def test_successor_rejects_live_runtime_drift(
        monkeypatch, filename, keys, value, retained_release_context):
    monkeypatch.setattr(release, "resolve_release_binding", lambda: retained_release_context)
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


@pytest.mark.parametrize("certification", ["deferred", "not_certified"])
@pytest.mark.parametrize("defect", [
    "none", "pin", "certified", "missing-certification", "attempt", "status",
    "review-status", "execution-status", "resumable", "last-error", "extra-field",
])
def test_reviewed_repair_allows_only_stage_certification_reconciliation(
        monkeypatch, certification, defect):
    proposal = release.proposal_context()
    path = release.LIFE / "backlog.jsonl"
    original = Path.read_bytes
    lines = []
    for raw in original(path).splitlines():
        if not raw.strip():
            lines.append(raw)
            continue
        row = json.loads(raw)
        if row.get("id") == "ed9ef3afb3ad":
            row["outcome"]["stage_certification"] = certification
            if defect == "certified":
                row["outcome"]["stage_certification"] = "certified"
            elif defect == "missing-certification":
                del row["outcome"]["stage_certification"]
            elif defect == "attempt":
                row["attempt"] = 2
            elif defect == "status":
                row["status"] = "running"
            elif defect == "review-status":
                row["outcome"]["review_status"] = "continue"
            elif defect == "execution-status":
                row["outcome"]["execution_status"] = "ended"
            elif defect == "resumable":
                row["outcome"]["resumable"] = True
            elif defect == "last-error":
                row["last_error"] = "changed"
            elif defect == "extra-field":
                row["unexpected"] = True
            raw = json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        lines.append(raw)
    changed = b"\n".join(lines) + b"\n"
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: changed if self == path else original(self))
    if defect == "pin":
        proposal["repository_repair"]["backlog_row_sha256"] = "0" * 64
    if defect == "none":
        release.reviewed_repair_evidence(proposal)
    else:
        with pytest.raises(ValueError, match="ed9 backlog differs beyond stage-certification"):
            release.reviewed_repair_evidence(proposal)
    assert path.read_bytes() == changed


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
    inputs = copy.deepcopy(release.proposal_context())
    previous = release.document(inputs["accepted_construction"]["contract"])
    baseline = tmp_path / "source-baseline"
    baseline.mkdir()
    sources = release.implementation_pins()
    inputs["source_baseline"] = {}
    for (role, _, name), source in zip(release.SOURCE_LAYOUT, sources, strict=True):
        (baseline / name).write_bytes(release.authenticate(source))
        inputs["source_baseline"][role] = source
    inputs["repository_repair"]["proposed_bytes"] = release.pin(
        baseline / release.SOURCE_LAYOUT[0][2])
    proposal_path = tmp_path / "proposal.json"
    release.write_json(proposal_path, inputs)
    proposal = release.pin(proposal_path)
    monkeypatch.setattr(release, "CONSUMPTION_INPUT", proposal)
    monkeypatch.setattr(release, "proposal_context", lambda: copy.deepcopy(inputs))
    owner = {"published_at": time.time() - 10}
    monkeypatch.setattr(release, "successor_inputs",
                        lambda: (proposal, inputs, previous, owner, {}))
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
    binding_cases = "".join(
        f'<testcase classname="release" name="{name}[{value}]"/>'
        for name, values in (
            ("test_prepared_release_resolution", ("none", "done", "failed", "root", "symlink", "overlap")),
            ("test_prepared_release_binding_splices", ("mission_id", "authorized_root", "validation_root")),
            ("test_reviewed_repair_allows_only_stage_certification_reconciliation", tuple(
                f"{defect}-{certification}"
                for certification in ("deferred", "not_certified")
                for defect in ("none", "pin", "certified", "missing-certification", "attempt",
                               "status", "review-status", "execution-status", "resumable",
                               "last-error", "extra-field"))),
        ) for value in values)
    xml = (
        '<testsuites><testsuite tests="35" failures="0" errors="0" skipped="0">'
        '<testcase classname="release" name="test_fresh_proposed_contract_consumes_production"/>'
        '<testcase classname="release" name="test_isolated_value_successor_owner_and_source_bindings"/>'
        '<testcase classname="release" '
        'name="test_preseal_verdict_matches_real_runtime_classification[continue]"/>'
        '<testcase classname="release" '
        'name="test_preseal_verdict_matches_real_runtime_classification[replan_requested]"/>'
        + binding_cases + '</testsuite></testsuites>')
    (scratch / "focused-tests.xml").write_text(xml)
    now = time.time()
    commands = []
    for name, argv in release.validation_commands(scratch, sources):
        environment = release.validation_environment(scratch)
        (scratch / (name + ".command.txt")).write_text(release.shlex.join(argv) + "\n")
        release.write_json(scratch / (name + ".argv.json"), argv)
        release.write_json(scratch / (name + ".environment.json"), environment)
        (scratch / (name + ".stdout")).write_bytes(b"35 passed in 1.00s\n" if name == "focused-tests" else b"")
        (scratch / (name + ".stderr")).write_bytes(b"")
        result = {
            "name": name, "argv": argv, "environment": environment, "cwd": str(release.REPOSITORY),
            "exit_code": 0, "timed_out": False, "started_at": now - 2, "finished_at": now - 1,
            "stdout": release.pin(scratch / (name + ".stdout")),
            "stderr": release.pin(scratch / (name + ".stderr")),
            "test_summary": {"tests": 35, "failures": 0, "errors": 0, "skipped": 0}
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


@pytest.mark.parametrize("summary_total", [35, 36, 37, 38])
def test_validation_receipt_authenticates_subtest_totals(validation_case, summary_total):
    evidence, check_evidence = validation_case
    command = evidence["commands"][1]
    scratch = Path(command["stdout"]["path"]).parent
    xml = scratch / "focused-tests.xml"
    xml.write_bytes(xml.read_bytes().replace(b'tests="35"', b'tests="37"'))
    stdout = scratch / "focused-tests.stdout"
    stdout.write_bytes(b"35 passed, 2 subtests passed in 1.00s\n")
    command["stdout"] = release.pin(stdout)
    command["test_summary"]["tests"] = summary_total
    (scratch / "focused-tests.receipt.json").write_bytes(release.json_bytes(command))
    for member in evidence["members"]:
        path = Path(member["path"])
        if path.name in ("focused-tests.xml", "focused-tests.stdout", "focused-tests.receipt.json"):
            member.update(release.pin(path))
    if summary_total == 37:
        assert check_evidence()[0] == evidence
    else:
        with pytest.raises(ValueError, match="validation test summary differs"):
            check_evidence()


@pytest.mark.parametrize("missing_test", [
    "test_isolated_value_successor_owner_and_source_bindings",
    "test_preseal_verdict_matches_real_runtime_classification[continue]",
    *[
        "test_reviewed_repair_allows_only_stage_certification_reconciliation"
        f"[{defect}-{certification}]"
        for certification in ("deferred", "not_certified")
        for defect in ("none", "pin", "certified", "missing-certification", "attempt",
                       "status", "review-status", "execution-status", "resumable",
                       "last-error", "extra-field")
    ],
])
def test_validation_receipt_requires_release_binding_coverage(validation_case, missing_test):
    evidence, check_evidence = validation_case
    member = next(record for record in evidence["members"]
                  if Path(record["path"]).name == "focused-tests.xml")
    path = Path(member["path"])
    path.write_bytes(path.read_bytes().replace(
        missing_test.encode(), b"unrelated_test"))
    member.update(release.pin(path))
    with pytest.raises(ValueError, match="required release-binding validation coverage missing"):
        check_evidence()


def test_validation_receipt_rejects_obsolete_owner_coverage(validation_case):
    evidence, check_evidence = validation_case
    member = next(record for record in evidence["members"]
                  if Path(record["path"]).name == "focused-tests.xml")
    path = Path(member["path"])
    path.write_bytes(path.read_bytes().replace(
        b"test_isolated_value_successor_owner_and_source_bindings",
        b"test_actual_successor_v6_owner_and_source_bindings"))
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
def test_preparation_reference_formats_and_census(value_case, tmp_path, defect):
    contract = copy.deepcopy(value_case["contract"])
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
    "candidate", "stage13_suffix", "stage13_tests",
    "value_sufficiency_source", "value_sufficiency_test",
])
def test_preparation_runtime_pin_authentication(value_case, monkeypatch, defect, role):
    contract = copy.deepcopy(value_case["contract"])
    if role == "candidate":
        role = "ace3.model.candidates." + release.CANDIDATE_NAME
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


@pytest.fixture
def value_case(production, tmp_path, monkeypatch, value_binding_case):
    binding = value_binding_case
    monkeypatch.setattr(release, "ACTIVE_BINDING", binding)
    for name, value in binding.items():
        if name.isupper():
            monkeypatch.setattr(release, name, value)
    proposal = release.proposal_context()
    for (_, _, name), source in zip(release.SOURCE_LAYOUT, release.implementation_pins(), strict=True):
        (tmp_path / name).write_bytes(release.authenticate(source))
    contract = release.proposed_contract(
        release.document(proposal["accepted_release"]["contract"]),
        release.CONSUMPTION_INPUT, tmp_path, time.time() - 30, production[1]["runtime"])
    fixture = case.__wrapped__((production[0], contract, production[2]), tmp_path, monkeypatch)
    fixture["proposal"] = proposal
    return fixture


@pytest.mark.parametrize("defect", [
    "none", "software-review", "superseded-owner", "superseded-status",
    "superseded-by", "missing-owner", "duplicate-owner", "conflicting-owner",
    "unrelated-superseded-owner", "missing-field", "inherited-recovery",
])
def test_current_census_consumption_decision(value_case, monkeypatch, defect):
    inputs = copy.deepcopy(value_case["consumption_input"])
    inputs["current_runtime_sources"] = value_case["contract"]["runtime_sources"]
    inputs["software_review"] = {"mission_id": "5ddff6694c9d"}
    inputs["recovery_context"] = {
        "failed_assignment": "850677a64813", "runtime_failure_canary": "7d79a6ef428a",
    }
    decision = copy.deepcopy(value_case["consumption_decision"])
    decision.update(software_review="5ddff6694c9d",
                    supersedes_preclaim_release="f1a28e051c63")
    row = {"id": "f1a28e051c63", "status": "superseded",
           "superseded_by_plan_id": release.MISSION}
    if defect == "software-review":
        decision["software_review"] = "wrong"
    elif defect == "superseded-owner":
        decision["supersedes_preclaim_release"] = "wrong"
    elif defect == "superseded-status":
        row["status"] = "running"
    elif defect == "superseded-by":
        row["superseded_by_plan_id"] = "other-mission"
    elif defect == "missing-field":
        del decision["software_review"]
    elif defect == "inherited-recovery":
        decision.update(inputs["recovery_context"])
    path = release.LIFE / "backlog.jsonl"
    rows = [raw for raw in path.read_bytes().splitlines()
            if raw.strip() and json.loads(raw).get("id") != row["id"]]
    if defect != "missing-owner":
        rows.append(release.encoded(row))
    if defect == "duplicate-owner":
        rows.append(release.encoded(row))
    elif defect == "conflicting-owner":
        rows.append(release.encoded(dict(row, superseded_by_plan_id="other-mission")))
    elif defect == "unrelated-superseded-owner":
        rows.append(release.encoded(dict(row, id="unrelated-superseded-owner")))
    raw = b"\n".join(rows) + b"\n"
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: raw if self == path else read_bytes(self))
    monkeypatch.setattr(release, "proposal_context", lambda: inputs)
    monkeypatch.setattr(release, "CONSUMPTION_DECISION", value_case["put"](
        release.CONSUMPTION_DECISION["path"], decision))
    accepted = release.construction_input(inputs)["accepted_provenance"]
    bundle = dict(mission_id=release.PROVENANCE_MISSION, **{
        key: accepted[key] for key in ("latest", "review", "checkpoint")})
    if defect in ("none", "unrelated-superseded-owner"):
        release.native_review_consumption(bundle, accepted["contract"], accepted["seal"])
    else:
        with pytest.raises(ValueError, match="consumption decision|superseded owner"):
            release.native_review_consumption(bundle, accepted["contract"], accepted["seal"])


@pytest.mark.parametrize("defect", [
    "none", "failed-assignment", "canary", "relative-output", "wrong-output",
    "missing-assignment", "missing-canary", "extra-field",
])
def test_value_selftest_consumption_decision(value_case, monkeypatch, defect):
    inputs = copy.deepcopy(value_case["consumption_input"])
    inputs["recovery_context"] = {
        "failed_assignment": "111111111111",
        "runtime_failure_canary": "222222222222",
        "skipped_continuation": "333333333333",
        "native_authorization_fields": "EMPTY_FOR_ORDINARY_ARTIFACT_TASK",
    }
    decision = copy.deepcopy(value_case["consumption_decision"])
    decision.update(
        failed_assignment="111111111111",
        runtime_failure_canary="222222222222",
        future_candidate_output=str(
            release.REPOSITORY / "build"
            / "q24_s16_l23_p0_value_content_sufficiency_v1-8ee0b3aaeecb-attempt001"),
    )
    if defect == "failed-assignment":
        decision["failed_assignment"] = "444444444444"
    elif defect == "canary":
        decision["runtime_failure_canary"] = "444444444444"
    elif defect == "relative-output":
        decision["future_candidate_output"] = str(
            Path(decision["future_candidate_output"]).relative_to(release.REPOSITORY))
    elif defect == "wrong-output":
        decision["future_candidate_output"] = str(release.REPOSITORY / "build/wrong")
    elif defect == "missing-assignment":
        del decision["failed_assignment"]
    elif defect == "missing-canary":
        del decision["runtime_failure_canary"]
    elif defect == "extra-field":
        decision["extra"] = True
    monkeypatch.setattr(release, "proposal_context", lambda: inputs)
    monkeypatch.setattr(release, "CONSUMPTION_DECISION", value_case["put"](
        release.CONSUMPTION_DECISION["path"], decision))
    accepted = release.construction_input(inputs)["accepted_provenance"]
    bundle = {key: accepted[key] for key in ("mission_id", "latest", "review", "checkpoint")}
    if defect == "none":
        release.native_review_consumption(bundle, accepted["contract"], accepted["seal"])
    else:
        with pytest.raises(ValueError, match="exact Manager consumption decision differs"):
            release.native_review_consumption(bundle, accepted["contract"], accepted["seal"])


@pytest.mark.parametrize("defect", [
    "none", "wrong-source", "wrong-output", "historical-output", "missing-argv",
    "extra-argv", "spliced-candidate", "spliced-test", "live-candidate", "live-test",
])
def test_value_target_contract_to_generic_envelope(value_case, monkeypatch, defect):
    """Real proposal/issuer/consumer; only future native transport is virtual."""
    fixture = value_case
    target = fixture["proposal"]["candidate_binding"]
    contract = fixture["contract"]
    assert contract["reference_members"] == release.REFERENCE_MEMBERS
    assert contract["launch"]["argv"] == target["launch"]["argv"]
    assert contract["runtime_sources"]["value_sufficiency_source"] == release.pin(value_candidate.SOURCE)
    assert contract["runtime_sources"]["value_sufficiency_test"] == release.pin(value_candidate.TEST)
    put, virtual = fixture["put"], fixture["virtual"]
    issuer_path = release.REPOSITORY / ".argus/live/manager-tools/publish-stage11-science-issuance-v1.py"
    spec = importlib.util.spec_from_file_location("test_generic_science_issuer", issuer_path)
    issuer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(issuer)
    authority_dir = value_candidate.AUTHORITY
    request = fixture["request"]
    review = fixture["proposal"]["candidate_review"]
    selected = fixture["selection"]["selected"]
    inputs = {
        "mission_id": value_candidate.MISSION, "authority_id": "test-only-future-science",
        "scientific_run_budget": 1, "attempt": 1,
        "capture_root": target["capture_root"],
        "candidate": contract["current_sources"]["candidate"],
        "candidate_tests": contract["current_sources"]["tests"],
        "runner": release.pin(release.REPOSITORY / ".argus/live/manager-tools/run-stage11-science-once-v1.py"),
        "capture_implementation": value_candidate.capture.implementation_pins(),
        "release_source": release.pin(Path(release.__file__)),
        "contract": request["contract"], "seal": request["seal"],
        "operational_release_selection": request["operational_release_selection"],
        "selected_release_mission": release.MISSION,
        "contract_review": dict(request["contract_review"], mission=selected["mission"]),
        "runner_review": {key: review[key] for key in ("mission_id", "latest", "review", "checkpoint", "mission")},
        "accepted_review": request["accepted_review"], "accepted_identity": request["accepted_identity"],
    }
    authority = {
        "mission_id": inputs["mission_id"], "authority_id": inputs["authority_id"],
        "life_dir": str(release.LIFE), "issuer": "manager", "expected_attempt": 1,
        "authorized_science": True, "scientific_run_budget": 1,
        "authorized_root": inputs["capture_root"], "expected_objective": "test-only science",
        "manager_inputs": {"issuer": release.pin(issuer_path),
                          "science_input": put(authority_dir / "test-input.json", inputs)},
    }
    authority_pin = put(authority_dir / "build-authorization.json", authority)
    put(authority_dir / "manager-owner-prepared.json", {
        "authority_id": authority["authority_id"], "mission_id": inputs["mission_id"],
        "state": "PREPARED_BEFORE_NATIVE_HANDOFF", "build_authorization": authority_pin})
    row = request["native_record"]
    assert row["id"] == request["mission_id"] == value_candidate.MISSION == "8ee0b3aaeecb"
    assert request["mission_started_event"]["item_id"] == "8ee0b3aaeecb"
    assert request["mission_started_event"]["usage_attempt_id"] == "8ee0b3aaeecb:attempt:1"
    row["objective"] = authority["expected_objective"]
    (fixture["native"] / "backlog.jsonl").write_bytes(release.encoded(row) + b"\n")
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
                        lambda path: virtual[str(path)] if str(path) in virtual else read_bytes(path))
    isolated_roots = (
        authority_dir, Path(target["output"]),
        release.REPOSITORY / target["capture_root"],
    )
    exists, is_file, is_symlink = Path.exists, Path.is_file, Path.is_symlink

    def isolated(path):
        return any(path == root or root in path.parents for root in isolated_roots)

    monkeypatch.setattr(Path, "exists", lambda path: (
        any(path == Path(name) or path in Path(name).parents for name in virtual)
        if isolated(path) else exists(path)))
    monkeypatch.setattr(Path, "is_file", lambda path: (
        str(path) in virtual if isolated(path) else is_file(path)))
    monkeypatch.setattr(Path, "is_symlink", lambda path: (
        False if isolated(path) else is_symlink(path)))
    monkeypatch.setattr(issuer, "pinned_bytes", release.authenticate)
    monkeypatch.setattr(issuer, "exclusive_json", put)
    monkeypatch.setattr(issuer, "jsonl_matches", lambda path, predicate: [
        (item, release.encoded(item)) for item in (
            [row] if path.name == "backlog.jsonl" else [request["mission_started_event"]])
        if predicate(item)])

    def emit(record):
        output, documents, unchanged = release.validate_after_claim(record)
        unchanged()
        return {name: put(output / name, json.loads(raw))
                for name, raw in zip(release.OUTPUTS, documents, strict=True)}

    monkeypatch.setattr(release, "emit_after_claim", emit)
    if defect in ("wrong-source", "wrong-output", "historical-output", "missing-argv", "extra-argv"):
        argv = contract["launch"]["argv"]
        if defect == "wrong-source":
            argv[3] = "ace3.model.candidates.wrong"
        elif defect in ("wrong-output", "historical-output"):
            argv[-1] = (str(value_candidate.OUTPUT.with_name(
                value_candidate.NAME + "-650940c1b279-attempt001")) if defect == "historical-output"
                        else str(release.REPOSITORY / "build/wrong"))
        elif defect == "missing-argv":
            del argv[-2:]
        else:
            argv.append("--extra")
        fixture["rebind_contract"](contract)
        inputs.update(contract=request["contract"], seal=request["seal"],
                      operational_release_selection=request["operational_release_selection"])
    elif defect.startswith("spliced-"):
        key = "candidate" if defect == "spliced-candidate" else "candidate_tests"
        inputs[key] = inputs["candidate_tests" if key == "candidate" else "candidate"]
    elif defect.startswith("live-"):
        path = value_candidate.SOURCE if defect == "live-candidate" else value_candidate.TEST
        virtual[str(path)] = b"unreviewed changed source\n"
    authority["manager_inputs"]["science_input"] = put(authority_dir / "test-input.json", inputs)
    authority_pin = put(authority_dir / "build-authorization.json", authority)
    put(authority_dir / "manager-owner-prepared.json", {
        "authority_id": authority["authority_id"], "mission_id": inputs["mission_id"],
        "state": "PREPARED_BEFORE_NATIVE_HANDOFF", "build_authorization": authority_pin})
    if defect != "none":
        with pytest.raises((ValueError, RuntimeError)):
            issuer.publish(authority_dir)
        assert not (authority_dir / "issued").exists()
        return
    result = issuer.publish(authority_dir)
    assert result["scientific_invocations"] == 0
    bound = value_candidate.manager_binding(Path(target["output"]))
    assert bound["output"] == Path(target["output"])
    assert value_candidate.manager_binding(live=False)["declaration"] == bound["declaration"]
    release.validate_preparation_contract(contract)
    assert not bound["output"].exists()
    assert not bound["capture_root"].exists()


@pytest.mark.parametrize("defect", [
    "none", "missing", "empty", "stale", "spliced", "editable-baseline", "historical-surplus",
])
def test_explicit_current_runtime_census(value_case, monkeypatch, defect):
    """A new Manager declaration replaces membership, never immutable input bytes."""
    proposal = release.document(release.CONSUMPTION_INPUT)
    current = copy.deepcopy(value_case["contract"]["runtime_sources"])
    declared = copy.deepcopy(current)
    for role, source_role in (
        ("ace3.model.stage11_current_runtime_release", "release_builder"),
        ("ace3.model.candidates." + release.CANDIDATE_NAME, "candidate"),
        ("value_sufficiency_source", "candidate"),
        ("value_sufficiency_test", "candidate_tests"),
    ):
        declared[role] = proposal["source_baseline"][source_role]
    proposal["current_runtime_sources"] = declared
    proposal["expected_runtime_sources"]["historical_surplus"] = declared["focused_test"]
    if defect == "missing":
        del declared["stage13_tests"]
    elif defect == "empty":
        declared.clear()
    elif defect == "stale":
        declared["stage13_tests"]["sha256"] = "0" * 64
    elif defect == "spliced":
        declared["stage13_tests"]["path"] = declared["stage13_suffix"]["path"]
    elif defect == "editable-baseline":
        declared["value_sufficiency_test"] = declared["value_sufficiency_source"]
    document = release.document
    monkeypatch.setattr(release, "document", lambda record: (
        copy.deepcopy(proposal) if record == release.CONSUMPTION_INPUT else document(record)))
    contract = copy.deepcopy(value_case["contract"])
    if defect == "historical-surplus":
        contract["runtime_sources"]["historical_surplus"] = current["focused_test"]
    if defect != "none":
        with pytest.raises(ValueError, match="runtime_sources"):
            release.validate_preparation_contract(contract)
        return
    assert release.proposal_context()["expected_runtime_sources"] == current
    assert "historical_surplus" not in current
    release.validate_preparation_contract(contract)
    assert "current_runtime_sources" not in document(release.CONSUMPTION_INPUT)


@pytest.mark.parametrize("status,action", [
    ("continue", "BUILD_FINAL_FROM_REVIEWED_BYTES"),
    ("replan_requested", "BUILD_FINAL_FROM_REVIEWED_BYTES"),
    ("done", "BUILD_FINAL_FROM_REVIEWED_BYTES"),
    ("continue", "not approved"),
])
def test_value_target_preseal_continue(preseal_case, monkeypatch, status, action):
    original = release.proposal_context()
    monkeypatch.setattr(release, "proposal_context",
                        lambda: dict(original, candidate_binding={"test_only": True}))
    review, _, _, check_review, review_path = preseal_case
    review["review"]["status"] = status
    review["review"]["next_action"] = action
    if status == "continue" and action == "BUILD_FINAL_FROM_REVIEWED_BYTES":
        assert check_review() == release.pin(review_path)
    else:
        with pytest.raises(ValueError, match="independent preseal review"):
            check_review()


@pytest.fixture
def isolated_value_selftests(tmp_path, monkeypatch):
    def install(body):
        path = tmp_path / "isolated_value_selftests.py"
        path.write_text("import unittest\n" + body)
        monkeypatch.setattr(value_candidate, "TEST", path)
        return path

    return install


@pytest.mark.parametrize("count", [1, 3])
def test_value_selftests_use_loaded_count(isolated_value_selftests, count):
    isolated_value_selftests(
        "class Passing(unittest.TestCase):\n" + "".join(
            f"    def test_{i}(self):\n        self.assertTrue(True)\n" for i in range(count)))
    report = value_candidate.focused_tests()
    assert report["loaded"] == report["executed"] == count
    assert all(report[key] == 0 for key in (
        "failures", "errors", "skipped", "unexpected_successes"))
    assert f"Ran {count} test" in report["raw_output"]


@pytest.mark.parametrize("defect,body,counts", [
    ("empty", "", (0, 0, 0, 0, 0, 0)),
    ("partial", """
class StopAfterPass(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(True)
    def run(self, result):
        super().run(result)
        result.stop()
        return result
class NotReached(unittest.TestCase):
    def test_not_reached(self):
        self.fail("partial suite reached its second test")
def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        StopAfterPass("test_pass"), NotReached("test_not_reached")])
""", (2, 1, 0, 0, 0, 0)),
    ("skipped", """
class Skipped(unittest.TestCase):
    @unittest.skip("isolated skip")
    def test_skip(self):
        pass
""", (1, 1, 0, 0, 1, 0)),
    ("failure", """
class Failure(unittest.TestCase):
    def test_failure(self):
        self.fail("isolated failure")
""", (1, 1, 1, 0, 0, 0)),
    ("error", """
class Error(unittest.TestCase):
    def test_error(self):
        raise RuntimeError("isolated error")
""", (1, 1, 0, 1, 0, 0)),
    ("unexpected-success", """
class UnexpectedSuccess(unittest.TestCase):
    @unittest.expectedFailure
    def test_unexpected_success(self):
        self.assertTrue(True)
""", (1, 1, 0, 0, 0, 1)),
])
def test_value_selftests_reject_real_incomplete_or_unsuccessful_runs(
        isolated_value_selftests, monkeypatch, defect, body, counts):
    isolated_value_selftests(body)
    observed = []

    class RecordingRunner(unittest.TextTestRunner):
        def run(self, suite):
            loaded = suite.countTestCases()
            result = super().run(suite)
            observed.append((loaded, result, self.stream.getvalue()))
            return result

    monkeypatch.setattr(value_candidate.unittest, "TextTestRunner", RecordingRunner)
    with pytest.raises((RuntimeError, ValueError), match="focused tests") as caught:
        value_candidate.focused_tests()
    if defect == "empty":
        assert not observed
        assert "empty suite" in str(caught.value)
        return
    assert len(observed) == 1
    loaded, result, raw = observed[0]
    report = json.loads(str(caught.value).split("focused tests failed:\n", 1)[1])
    assert tuple(report[key] for key in (
        "loaded", "executed", "failures", "errors", "skipped", "unexpected_successes")) == counts
    assert loaded == counts[0]
    assert (result.testsRun, len(result.failures), len(result.errors),
            len(result.skipped), len(result.unexpectedSuccesses)) == counts[1:]
    assert report["raw_output"] == raw
    if defect in ("partial", "skipped"):
        assert result.wasSuccessful()


def test_value_selftests_real_current_suite(record_property):
    report = value_candidate.focused_tests()
    assert report["loaded"] == report["executed"] == 35
    assert all(report[key] == 0 for key in (
        "failures", "errors", "skipped", "unexpected_successes"))
    assert "Ran 35 tests" in report["raw_output"]
    assert report["raw_output"].endswith("\nOK\n")
    record_property("value_focused_tests", json.dumps(report, sort_keys=True))


@pytest.fixture
def current_value_runtime_sources():
    """Declare the value target independently of inherited or imported surplus."""
    modules = (
        "ace3.model.awq_bit_oracle",
        "ace3.model.candidates.binary64_fp16_excess_v1",
        "ace3.model.candidates.capture_harness_v3",
        "ace3.model.candidates.decoder_gate_policy",
        "ace3.model.candidates.decoder_gate_policy_v3",
        "ace3.model.candidates.diagnose_q24_s16_final_head_impact_census_v1",
        "ace3.model.candidates.diagnose_q24_s16_l12_coordinate62_producer_cone_v1",
        "ace3.model.candidates.diagnose_q24_s16_l12_coordinate62_producer_cone_v2",
        "ace3.model.candidates.diagnose_q24_s16_l12_coordinate62_producer_cone_v3",
        "ace3.model.candidates.diagnose_q24_s16_l13_l12_parent_coordinate_cut_v1",
        "ace3.model.candidates.diagnose_q24_s16_l13_l12_parent_coordinate_cut_v2",
        "ace3.model.candidates.diagnose_q24_s16_l13_l12_parent_coordinate_cut_v3",
        "ace3.model.candidates.diagnose_q24_s16_l13_native_paired_intervention_v1",
        "ace3.model.candidates.diagnose_q24_s16_l13_original_intermediate_drift_v1",
        "ace3.model.candidates.diagnose_q24_s16_l13_s18_first_failure_v1",
        "ace3.model.candidates.diagnose_q24_s16_l13_upstream_parent_cut_v1",
        "ace3.model.candidates.diagnose_q24_s16_l21_from_l20_s18_failure_census_v1",
        "ace3.model.candidates.diagnose_q24_s16_l21_sparse_cut_localization_matrix_v1",
        "ace3.model.candidates.diagnose_q24_s16_l22_from_l21_coordinate62_suffix_execution_v1",
        "ace3.model.candidates.diagnose_q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1",
        "ace3.model.candidates.diagnose_q24_s16_l23_from_l22_coordinate62_suffix_preflight_v1",
        "ace3.model.candidates.diagnostic_capture_v1",
        "ace3.model.candidates.host_capture_v3",
        "ace3.model.candidates.local_operator_reference_v3",
        "ace3.model.candidates.prepare_l6_l8_continuation_v3",
        "ace3.model.candidates.q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1",
        "ace3.model.candidates.q24_s16_final_head_from_l23_coordinate62_suffix_preflight_v1",
        "ace3.model.candidates.q24_s16_l23_from_l22_coordinate62_suffix_execution_v1",
        "ace3.model.candidates.q24_s16_l23_p0_value_content_sufficiency_v1",
        "ace3.model.candidates.q24_s16_toward_zero_native_v1",
        "ace3.model.candidates.q24_software_candidate_v1",
        "ace3.model.candidates.q24_software_root_v3",
        "ace3.model.candidates.remaining_decoder_gate_policy_v3",
        "ace3.model.candidates.remaining_layers_v3",
        "ace3.model.candidates.residual_exact_grid_q24_reference_v1",
        "ace3.model.candidates.run_q24_s16_toward_zero_l3_l8_v1",
        "ace3.model.candidates.run_q24_s16_toward_zero_l9_l23_v1",
        "ace3.model.candidates.run_q24_s16_toward_zero_native_v1",
        "ace3.model.candidates.run_single_round_residual_rtl",
        "ace3.model.candidates.runtime_admission_v3",
        "ace3.model.fp16_adaptation_oracle",
        "ace3.model.stage11_current_runtime_release",
        "ace3.model.streaming_lm_head_reference",
    )
    sources = {
        name: release.pin(release.REPOSITORY / (name.replace(".", "/") + ".py"))
        for name in modules
    }
    sources.update(
        focused_test=release.pin(value_candidate.final.TEST),
        stage13_suffix=release.pin(value_candidate.layer.SOURCE),
        stage13_tests=release.pin(value_candidate.layer.TEST),
        value_sufficiency_source=release.pin(value_candidate.SOURCE),
        value_sufficiency_test=release.pin(value_candidate.TEST),
    )
    assert len(sources) == 48
    return sources


def test_value_real_loading_exact_runtime_census(
        no_execution, record_property, tmp_path, current_value_runtime_sources):
    """Load real tests against the isolated value-candidate source declaration."""
    expected = copy.deepcopy(current_value_runtime_sources)
    assert len(expected) == 48
    script = r'''
import copy
import json
import os
import sys
from unittest.mock import patch
from ace3.model.candidates import q24_s16_l23_p0_value_content_sufficiency_v1 as d

tests = d.focused_tests()
assert tests["loaded"] == tests["executed"] == 35
assert all(tests[key] == 0 for key in (
    "failures", "errors", "skipped", "unexpected_successes"))
assert tests["raw_output"].endswith("\nOK\n")
fixture = json.load(sys.stdin)
expected = fixture["current_runtime_sources"]
assert len(expected) == 48
for pin in expected.values():
    d.read_bound(pin)
output = d.Path(fixture["output"])
capture = d.Path(fixture["capture"])
assert not any(path.exists() or path.is_symlink() for path in (output, capture))
blocked = []

def audit(event, args):
    if event == "os.mkdir" or (event == "open" and (
            isinstance(args[1], str) and any(flag in args[1] for flag in "wax+")
            or isinstance(args[2], int) and args[2] & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
        blocked.append(event)
        raise AssertionError("runtime census attempted output creation")

def forbidden(*args, **kwargs):
    blocked.append("numerical dispatch")
    raise AssertionError("runtime census attempted numerical dispatch")

sys.addaudithook(audit)
rejections = []
with patch.object(d, "suffix_stages", forbidden), patch.object(d, "pair_logits", forbidden), \
        patch.object(d.retained, "authenticate", forbidden):
    actual = d.runtime_sources({"runtime_sources": expected})
    assert actual == expected and len(actual) == 48
    for role in expected:
        for defect in ("missing", "stale"):
            changed = copy.deepcopy(expected)
            if defect == "missing":
                del changed[role]
            else:
                changed[role]["sha256"] = "0" * 64
            try:
                d.runtime_sources({"runtime_sources": changed})
            except (KeyError, RuntimeError, ValueError):
                rejections.append([role, defect])
            else:
                raise AssertionError("runtime census accepted " + role + ": " + defect)
    changed = dict(expected, unexpected_member=expected["value_sufficiency_source"])
    try:
        d.runtime_sources({"runtime_sources": changed})
    except (RuntimeError, ValueError):
        rejections.append(["unexpected_member", "extra"])
    else:
        raise AssertionError("runtime census accepted an extra member")
assert len(rejections) == 97 and not blocked
assert not any(path.exists() or path.is_symlink() for path in (output, capture))
print(json.dumps({"tests": tests, "runtime_sources": actual, "rejections": rejections,
                  "scientific_invocations": 0, "output_creations": 0}))
'''
    with no_execution(
            [sys.executable, "-B", "-c", script], cwd=release.REPOSITORY,
            env={**os.environ, "PYTHONPATH": str(release.REPOSITORY),
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        stdout, stderr = child.communicate(input=json.dumps({
            "current_runtime_sources": expected,
            "output": str(tmp_path / "output"),
            "capture": str(tmp_path / "capture"),
        }), timeout=120)
    assert child.returncode == 0, stdout + stderr
    report = json.loads(stdout)
    assert len(stdout.splitlines()) == 1
    assert report["runtime_sources"] == expected
    assert len(report["rejections"]) == 97
    assert report["scientific_invocations"] == report["output_creations"] == 0
    record_property("value_runtime_census", json.dumps(report, sort_keys=True))


@pytest.mark.parametrize("entry,defect", [
    ("check", "none"), ("main", "none"),
    ("main", "stale-source"), ("main", "wrong-mission"),
    ("main", "wrong-output"), ("main", "wrong-argv"), ("main", "consumed302"),
    ("main", "consumed650"), ("main", "old650-output"), ("main", "old650-argv"),
])
def test_value_selftest_preflight_to_execute_sentinel(
        isolated_value_selftests, monkeypatch, capsys, record_property, entry, defect):
    spec = importlib.util.spec_from_file_location("isolated_value_binding_tests", value_candidate.TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = module.ValueSufficiencyTests()
    isolated_value_selftests(
        "class Passing(unittest.TestCase):\n"
        "    def test_pass(self):\n"
        "        self.assertTrue(True)\n")
    dispatches = []

    def execute_sentinel(tests=None, output=None):
        bound = value_candidate.manager_binding(output)
        assert tests["loaded"] == tests["executed"] == 1
        dispatches.append((tests, bound["output"]))
        print(json.dumps({"status": "TEST_ONLY_EXECUTE_SENTINEL", "scientific_invocations": 0}))

    try:
        authority, inputs, declaration, output, _ = fixture.binding_fixture()
        if defect == "stale-source":
            inputs["candidate"] = dict(inputs["candidate"], sha256="0" * 64)
        elif defect in ("wrong-mission", "consumed302"):
            authority["mission_id"] = "wrong-mission" if defect == "wrong-mission" else "302ad994e84f"
        elif defect == "consumed650":
            authority["mission_id"] = "650940c1b279"
        elif defect in ("old650-output", "old650-argv"):
            old_output = output.with_name(value_candidate.NAME + "-650940c1b279-attempt001")
            if defect == "old650-output":
                output = old_output
            else:
                declaration["launch"]["argv"][-1] = str(old_output)
        elif defect == "wrong-output":
            output = output.with_name("isolated-wrong-output")
        elif defect == "wrong-argv":
            declaration["launch"]["argv"][3] = "ace3.model.candidates.wrong"
        monkeypatch.setattr(value_candidate, "execute", execute_sentinel)
        monkeypatch.setattr(sys, "argv", [str(value_candidate.SOURCE), "--check", "--output", str(output)])
        if entry == "check":
            value_candidate.check(output)
        elif defect == "none":
            value_candidate.main()
        else:
            with pytest.raises(SystemExit) as caught:
                value_candidate.main()
            assert caught.value.code == 1
        stdout = capsys.readouterr().out
        if entry == "main":
            document = json.loads(stdout)
            assert len(stdout.splitlines()) == 1
            events = document["events"]
            assert document["status"] == (
                "TEST_ONLY_EXECUTE_SENTINEL" if defect == "none" else "UNKNOWN")
            if defect != "none":
                assert document["automatic_replay_authorized"] is False
        else:
            events = [json.loads(line) for line in stdout.splitlines()]
            assert events.pop()["status"] == "TEST_ONLY_EXECUTE_SENTINEL"
        assert len(events) == 1
        assert events[0]["phase"] == "compile_tests"
        assert events[0]["loaded"] == events[0]["executed"] == 1
        assert len(events[0]["compiled"]) == 2
        assert len(dispatches) == (1 if defect == "none" else 0)
        assert not output.exists()
        assert value_candidate.MISSION == "8ee0b3aaeecb"
        record_property("value_preflight_output", stdout)
    finally:
        fixture.doCleanups()
