import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

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
    authority = release.REPOSITORY / ".argus/live/manager-authority" / release.AUTHORITY
    record = json.loads((authority / "build-authorization.json").read_bytes())[
        "manager_inputs"]["interface_repair"]
    assert record["sha256"] == release.PROPOSAL_SHA256
    inputs = release.document(record)
    contract = release.document(inputs["accepted_current_runtime"]["contract"])
    run = contract["accepted_lineage"]
    current = {key: release.pin(record["path"])
               for key, record in contract["current_sources"].items()}
    contract["current_sources"] = current
    for role, key in (("current_diagnostic", "candidate"), ("current_tests", "tests")):
        contract["execution_contract"]["sources"][role] = current[key]
        contract["source_snapshots"][role] = current[key]
    contract["source_roles"]["current_diagnostic"] = {
        "source": current["candidate"], "retained": current["candidate"],
    }
    return run, contract


def test_actual_authenticated_33_file_production_lineage(production):
    run, contract = production
    assert len(contract["accepted_files"]) == 33
    assert release.accepted_lineage(run, contract) == release.document(run["proposal"])
    assert set(run) == {
        "authority_bundle", "launcher_capture", "launcher_identity",
        "launcher_expected_identity", "terminal_receipt", "proposal", "receipt_index",
    }
    assert release.document(run["launcher_capture"])["success"] is True


def test_fresh_proposed_contract_consumes_production(production):
    path = release.REPOSITORY / release.ROOT / "current-runtime-contract.json"
    contract = release.document(release.pin(path))
    assert contract["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
    assert contract["scientific_run_budget"] == 0
    assert contract["mission_id"] == release.MISSION
    assert contract["inherits_operational_admission"] is False
    assert contract["current_sources"] == production[1]["current_sources"]
    assert contract["implementation_sources"] == [
        release.pin(release.REPOSITORY / release.SOURCE),
        release.pin(release.REPOSITORY / release.TEST),
    ]
    assert contract["accepted_current_runtime"]["mission_id"] == "04c9ee209234"
    assert release.accepted_lineage(production[0], contract)["identity"] == contract["identity"]


@pytest.fixture
def case(production, tmp_path, monkeypatch):
    run, original = production
    contract = copy.deepcopy(original)
    now = time.time()
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
    contract.update(mission_id=release.MISSION, created_at=now - 30, runtime=runtime,
                    implementation_sources=[
                        release.pin(release.REPOSITORY / release.SOURCE),
                        release.pin(release.REPOSITORY / release.TEST)])
    root = release.REPOSITORY / release.ROOT
    contract_pin = put(root / "current-runtime-contract.json", contract)
    seal_pin = put(root / "seal.json", {
        "mission_id": release.MISSION, "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
        "members": [dict(contract_pin, path="current-runtime-contract.json")],
    })
    review_root = release.LIFE / "handoffs" / release.MISSION
    checkpoint = put(review_root / "CHECKPOINT.md", {"future_review_fixture": True})
    review = put(review_root / "round-0001.json", {
        "kind": "round_reviewed_handoff", "mission_id": release.MISSION,
        "producer_role": "reviewer", "created_at": now - 25,
        "checkpoint": {"path": checkpoint["path"]},
        "review": {"status": "done", "reason": release.review_acceptance(contract_pin, seal_pin)},
    })
    latest = put(review_root / "latest.json",
                 {"kind": "handoff_ref", "handoff": {"path": review["path"]}})
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

    def rebind_contract(changed):
        contract_pin = put(root / "current-runtime-contract.json", changed)
        seal_pin = put(root / "seal.json", {
            "mission_id": release.MISSION, "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
            "members": [dict(contract_pin, path="current-runtime-contract.json")],
        })
        review_value = release.document(request["contract_review"]["review"])
        review_value["review"]["reason"] = release.review_acceptance(contract_pin, seal_pin)
        request.update(contract=contract_pin, seal=seal_pin,
                       accepted_identity=changed["accepted_lineage"])
        request["contract_review"]["review"] = put(review["path"], review_value)

    return {"request": request, "request_path": request_path, "put": put, "virtual": virtual,
            "contract": contract, "runtime": runtime, "native": native,
            "output": authority / "issued", "rebind_contract": rebind_contract}


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
        markers = {
            "missing_audit": release.AUDIT_LIMIT,
            "wrong_contract": case["request"]["contract"]["sha256"],
            "wrong_seal": case["request"]["seal"]["sha256"],
            "missing_gates": "ACCEPTED_EXACT_POSTCLAIM_ONE_SHOT",
            "missing_zero": "Stage11-Zero-Science: ACCEPTED",
            "absent": release.ADMISSION,
        }
        review["review"]["reason"] = review["review"]["reason"].replace(markers[defect], "REJECTED")
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
    run, contract = copy.deepcopy(production)
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
    run, contract = copy.deepcopy(production)
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
