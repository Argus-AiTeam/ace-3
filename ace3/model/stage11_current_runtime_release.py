"""Data-only Stage11 release construction and dormant, post-claim issuance.

No candidate, observer, preflight producer, or daemon module is imported here.
The issuer writes authority, not scientific results, and has no execution CLI.
The candidate consumes its exact serialized envelope through this same
non-executing validator; retained scientific identity and current source pins
remain separate, authenticated bindings.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


REPOSITORY = Path(__file__).resolve().parents[2]
LIFE = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05")
MISSION = "e2ba4dbc8651"
AUTHORITY = "s11-emitter-candidate-lineage-interface-20260924t143250z-e2ba4dbc8651-r1"
ROOT = Path("build/argus-stage11-emitter-candidate-lineage-interface-e2ba4dbc8651-attempt001")
SOURCE = Path("ace3/model/stage11_current_runtime_release.py")
TEST = Path("tests/test_stage11_current_runtime_release.py")
CLAIM_SHA256 = "96e568fb6ddfba4cc3e10b11a1bf9c7cb0972ce5135ced5376f3d22e7d4928dc"
PROPOSAL_SHA256 = "530530060386a6af62f4cc5efc3c9a2bee5219ea1f97ae0d98357c7c3f6f97d9"
ADMISSION = "ADMISSIBLE_CURRENT_RUNTIME_NON_INHERITING"
AUDIT_LIMIT = "ACCEPTED_PYTHON_SCOPED_NOT_OS_WIDE"
OUTPUTS = (
    "native-running-claim.json", "execution-authorization.json",
    "manager-issuance.json", "envelope.json",
)
BUILD_COMMAND = (
    "/home/argustest/miniconda3/bin/python -P -B "
    "ace3/model/stage11_current_runtime_release.py --build"
)
CLAIM_COMMAND = (
    "/home/argustest/miniconda3/bin/python -P -B "
    ".argus/live/manager-tools/publish-native-manager-owner-claim-v1.py "
    "--authority-dir .argus/live/manager-authority/" + AUTHORITY
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def same(left, right):
    return encoded(left) == encoded(right)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin_bytes(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def pin(path):
    path = Path(path)
    return pin_bytes(path, path.read_bytes())


def authenticate(record):
    require(set(record) == {"path", "bytes", "sha256"}, "invalid pin fields")
    path = Path(record["path"])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            "noncanonical or symlinked input")
    require(type(record["bytes"]) is int and record["bytes"] >= 0, "invalid byte count")
    raw = path.read_bytes()
    require(len(raw) == record["bytes"] and digest(raw) == record["sha256"],
            "authenticated bytes changed: " + str(path))
    return raw


def document(record):
    return json.loads(authenticate(record))


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def exclusive(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


def write_json(path, value):
    exclusive(path, json_bytes(value))


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "timestamp must be timezone-aware")
    return parsed.timestamp()


def utc(value):
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat()


def number(value):
    return type(value) in (int, float) and 0 < value < float("inf")


def stable_claim(row):
    result = copy.deepcopy(row)
    for key in ("notes", "last_error"):
        if key in result:
            require(result[key] is None or isinstance(result[key], str),
                    "structured claim bookkeeping is not exempt")
            del result[key]
    return result


def native_claim(life, mission, now):
    rows = []
    for raw in (life / "backlog.jsonl").read_bytes().splitlines():
        if raw.strip():
            row = json.loads(raw)
            if row.get("id") == mission:
                rows.append((row, raw))
    require(len(rows) == 1, "missing, duplicate, or rebound native claim")
    row, raw = rows[0]
    require(row["status"] == "running" and row["running_owner"] == "primary"
            and type(row["attempt"]) is int and row["attempt"] == 1
            and number(row["started_ts"]) and row["started_ts"] <= now,
            "exact first-attempt normal running claim required")
    require(row.get("authorization_id", "") == ""
            and row.get("authorization_action", "") == ""
            and not row.get("validator_repair")
            and not row.get("validator_expected_result"),
            "reserved validator-repair metadata is not ordinary authority")
    events = []
    for event_raw in (life / "events.jsonl").read_bytes().splitlines():
        if event_raw.strip():
            event = json.loads(event_raw)
            if event.get("type") == "life.mission.started" and event.get("item_id") == mission:
                events.append((event, event_raw))
    require(len(events) == 1, "missing, retry, or rebound mission.started event")
    event, event_raw = events[0]
    require(type(event["attempt"]) is int and event["attempt"] == 1
            and event["independent_review_required"] is True
            and event["usage_attempt_id"] == mission + ":attempt:1"
            and number(event["ts"]) and row["started_ts"] <= event["ts"] <= now,
            "mission.started does not bind this normal claim")
    stable_claim(row)
    return row, event, raw, event_raw


def observe_runtime():
    source_path = LIFE / "daemon.source.json"
    source = json.loads(source_path.read_bytes())
    continuous = json.loads((LIFE / "continuous.json").read_bytes())
    require(continuous["enabled"] is False, "continuous execution must remain disabled")
    process = Path("/proc") / str(source["pid"])
    ticks = int((process / "stat").read_text().rsplit(")", 1)[1].split()[19])
    command = (process / "cmdline").read_bytes()
    source_root = Path(source["source_root"])
    require(source_root.is_absolute() and source_root.resolve() == source_root,
            "runtime source root is not canonical")
    require(str(source_root.parent).encode() in command.split(b"\0"),
            "running process does not name the advertised runtime")
    return {
        "pid": source["pid"], "start_ticks": ticks,
        "source_record": pin_bytes(source_path, json_bytes(source))
        if source_path.read_bytes() == json_bytes(source) else pin(source_path),
        "source_root": str(source_root),
        "source_fingerprint": source["source_fingerprint"],
        "cmdline_sha256": digest(command),
        "continuous_enabled": False,
        "continuous_generation": continuous["generation"],
    }


def reviewer_done(record, mission):
    require(record["kind"] == "round_reviewed_handoff"
            and record["mission_id"] == mission
            and record["producer_role"] == "reviewer"
            and record["review"]["status"] == "done",
            "exact independent Reviewer DONE required")


def sealed_members(seal_pin):
    seal = document(seal_pin)
    root = Path(seal_pin["path"]).parent
    records = []
    for member in seal["members"]:
        relative = Path(member["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "seal member escaped its root")
        record = dict(member, path=str(root / relative))
        authenticate(record)
        records.append(record)
    require(len({r["path"] for r in records}) == len(records), "duplicate seal member")
    return seal, records


def review_acceptance(contract_pin, seal_pin):
    return "\n".join((
        "Stage11-Admission: " + ADMISSION,
        "Stage11-Audit-Limit: " + AUDIT_LIMIT,
        "Stage11-Contract-SHA256: " + contract_pin["sha256"],
        "Stage11-Seal-SHA256: " + seal_pin["sha256"],
        "Stage11-Emitter-Gates: ACCEPTED_EXACT_POSTCLAIM_ONE_SHOT",
        "Stage11-Zero-Science: ACCEPTED",
    ))


def review_bundle(bundle, mission, expected_review=None):
    base = LIFE / "handoffs" / mission
    require(bundle["mission_id"] == mission
            and bundle["latest"]["path"] == str(base / "latest.json"),
            "review is not anchored to the native Host handoff")
    latest = document(bundle["latest"])
    review = document(bundle["review"])
    checkpoint = authenticate(bundle["checkpoint"])
    require(latest["kind"] == "handoff_ref"
            and latest["handoff"]["path"] == bundle["review"]["path"]
            and Path(bundle["review"]["path"]).parent == base
            and bundle["checkpoint"]["path"] == str(base / "CHECKPOINT.md")
            and review["checkpoint"]["path"] == bundle["checkpoint"]["path"],
            "latest/review/checkpoint lineage differs")
    if expected_review is not None:
        require(same(bundle["review"], expected_review), "original review was rebound")
    reviewer_done(review, mission)
    require(checkpoint, "empty review checkpoint")
    return review


def command_identity(launch):
    words = shlex.split(launch["command"])
    assignments = words[:len(launch["environment"])]
    require(len(assignments) == len(launch["environment"])
            and all("=" in word for word in assignments)
            and len({word.split("=", 1)[0] for word in assignments}) == len(assignments)
            and dict(word.split("=", 1) for word in assignments) == launch["environment"]
            and words[len(assignments):] == launch["argv"],
            "exact declared command/argv/environment differs")


def zero_counters(value, names):
    require(set(value) == set(names)
            and all(type(v) is int and v == 0 for v in value.values()),
            "zero-science counters differ")


def accepted_lineage(run, contract):
    """Authenticate retained production documents without replaying their code.

    Historical source pins describe the then-current bytes, not today's mutable
    workspace. Bind them to the retained census and Manager declaration; only
    the separately declared current sources/snapshots are authenticated live.
    """
    names = {
        "authority_bundle": "bundle/authority-bundle.json",
        "launcher_capture": "launcher.capture.json",
        "launcher_identity": "launcher.identity.json",
        "launcher_expected_identity": "launcher.expected-identity.json",
        "terminal_receipt": "terminal-receipt.json",
        "proposal": "bundle/proposal.json",
        "receipt_index": "receipt-index.json",
    }
    require(set(run) == set(names) and same(run, contract["accepted_lineage"]),
            "production lineage request differs from reviewed contract")
    require(same(run["proposal"], contract["accepted_proposal"])
            and same(run["terminal_receipt"], contract["accepted_terminal"]),
            "accepted proposal/terminal splice")
    directory = Path(run["terminal_receipt"]["path"]).parent
    for name, relative in names.items():
        require(run[name]["path"] == str(directory / relative),
                "production member path splice: " + name)
    docs = {name: document(record) for name, record in run.items()}
    index = docs["receipt_index"]
    require(index["terminal_receipt_is_sealed_after_this_index"] is True
            and len(index["files"]) == 33
            and same(index["files"], contract["accepted_files"]),
            "original 33-file receipt index differs")
    members = {r["path"]: r for r in index["files"]}
    require(len(members) == 33, "duplicate retained member")
    for record in members.values():
        require(Path(record["path"]).is_relative_to(directory),
                "receipt index member outside retained root")
        authenticate(record)

    def member(relative):
        path = str(directory / relative)
        require(path in members, "missing retained member: " + relative)
        return members[path]

    for name, relative in names.items():
        if name not in ("terminal_receipt", "receipt_index"):
            require(same(run[name], member(relative)), "receipt index membership splice")
    bundle = docs["authority_bundle"]
    proposal = docs["proposal"]
    terminal = docs["terminal_receipt"]
    capture = docs["launcher_capture"]
    identity = docs["launcher_identity"]
    expected = docs["launcher_expected_identity"]
    selectors = document(member("authority-bundle.selectors.json"))
    inputs = document(member("consumption.inputs.json"))
    custody = document(member("custody/custody-receipt.json"))
    result = document(member("consumption.result.json"))
    closure = document(member("closure-check.json"))
    before = document(member("input-census.before.json"))
    after = document(member("input-census.after.json"))
    require(bundle["status"] == "PASS" and bundle["dispatch_authorized"] is False
            and bundle["normal_independent_review"] == "REQUIRED",
            "authority bundle decision differs")
    decision = bundle["identity_preflight"]
    require(decision["status"] == "ACCEPTED_NOVEL"
            and decision["dispatch_authorized"] is False,
            "authority identity decision differs")
    for value in (bundle, decision):
        zero_counters({k: value[k] for k in ("scientific_invocations", "producer_invocations")},
                      ("scientific_invocations", "producer_invocations"))
    require(same(bundle["proposal"], run["proposal"])
            and same(bundle["origins"], selectors)
            and same(selectors["proposal"]["pin"], member("custody/diagnostic_proposal.json"))
            and selectors["proposal"]["paths"] == {
                k: [k] for k in ("identity", "launch", "source_roles", "source_snapshots")}
            and authenticate(selectors["proposal"]["pin"]) == authenticate(run["proposal"])
            and same(bundle["catalog"], member("bundle/terminal-catalog.json"))
            and same(selectors["catalog"]["pin"], member("custody/terminal_catalog.json"))
            and selectors["catalog"]["lanes_path"] == ["lanes"]
            and selectors["catalog"]["scope_path"] == ["scoped_missions"],
            "authority proposal/origin/catalog binding differs")
    require(capture["success"] is True and type(capture["exit_status"]) is int
            and capture["exit_status"] == 0 and capture["timed_out"] is False
            and same(capture["files"], [member("launcher." + suffix) for suffix in (
                "identity.json", "stdout", "stderr", "whole-command.log")]),
            "launcher result/files differ")
    require(set(expected) == {"command", "argv", "cwd", "uid", "environment", "scope"}
            and set(identity) == set(expected) | {"capture_implementation"}
            and same({k: identity[k] for k in expected}, expected)
            and same(identity["capture_implementation"], capture["capture_implementation_after"])
            and len(identity["capture_implementation"]) == 2,
            "launcher identity/expected/implementation differs")
    command_identity(expected)
    require(expected["cwd"] == proposal["launch"]["cwd"]
            and expected["uid"] == proposal["launch"]["uid"]
            and expected["argv"] == [proposal["launch"]["argv"][0], "-B",
                                     member("consumption.py")["path"], "--consume"],
            "consumption launcher command/account differs")
    require(after["all_equal_to_before"] is True and after["prior_89c_members_unchanged"] is True,
            "historical input census changed")
    # Capture's source is in the census. Its test pin is retained in the
    # authenticated launch identity and identical post-capture pin list only.
    implementation = identity["capture_implementation"]
    require([r["path"] for r in implementation] == [
        str(Path(expected["cwd"]) / "ace3/model/candidates/diagnostic_capture_v1.py"),
        str(Path(expected["cwd"]) / "tests/test_diagnostic_capture_v1.py")],
        "capture implementation paths differ")
    for record in implementation:
        require(set(record) == {"path", "bytes", "sha256"}
                and type(record["bytes"]) is int and record["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
                "invalid historical capture implementation pin")
    for record in implementation[:1] + list(proposal["source_snapshots"].values()):
        require(any(same(record, r) for r in before["pins"])
                and any(same(record, r) for r in after["pins"]),
                "historical implementation/source pin absent from retained census")
    counters = (
        "admission_invocations", "candidate_check_invocations", "classifier_invocations",
        "full_vocabulary_invocations", "model_invocations", "native_suffix_invocations",
        "outside_attempt_writes", "prefix_invocations", "producer_invocations",
        "reference_invocations", "scientific_invocations", "service_invocations",
    )
    zero_counters(closure["counters"], counters)
    require(terminal["status"] == "ACCEPTED_NOVEL"
            and terminal["dispatch_authorized"] is False
            and terminal["scientific_or_admission_claim"] is False
            and terminal["unavailable_binding"] is None
            and terminal["mission_id"] == contract["accepted_mission"]
            and same(terminal["receipt_index"], run["receipt_index"])
            and same(terminal["closure_check"], closure)
            and closure["status"] == "PASS"
            and closure["identity_decision"] == "ACCEPTED_NOVEL"
            and closure["exact_inner_and_outer_frames"] is True
            and closure["inputs_unchanged"] is True
            and same([closure[k] for k in ("fresh_command_runs", "native_identity_runs",
                                          "in_memory_interface_compile_count")], [1, 1, 4]),
            "terminal decision/closure differs")
    lineage = contract["original_scientific_lineage"]
    require(same(lineage["proposal"], run["proposal"])
            and same(lineage["accepted_terminal"], run["terminal_receipt"])
            and same(lineage["accepted_review"], contract["accepted_review"])
            and lineage["status"] == "ACCEPTED_NOVEL_IDENTITY_ONLY",
            "original scientific lineage splice")
    manager = terminal["manager_authority"]
    require(same(manager, inputs["manager_authority"])
            and same(manager, custody["manager_authority"])
            and same(manager["issuance"], lineage["manager_origin_issuance"])
            and terminal["proposal_id"] == manager["proposal_id"] == lineage["proposal_id"],
            "terminal Manager authority splice")
    origin = document(manager["issuance"])
    require(origin["issuer"] == "manager"
            and origin["kind"] == "manager_diagnostic_identity_issuance"
            and origin["authorized_science"] is False and origin["authority_root"] is True
            and origin["proposal_id"] == terminal["proposal_id"],
            "Manager origin authority differs")
    zero_counters(origin["zero_actions"], (
        "ace2_actions", "daemon_restarts", "producer_invocations",
        "scientific_invocations", "service_invocations"))
    review_bundle(manager["review"], manager["review"]["mission_id"])
    declaration = document(origin["proposal_declaration"])
    original_contract = document(origin["contract"])
    require(same(inputs["actual_launch"], origin["actual_launch"])
            and same(inputs["origins"], custody["origins"])
            and same(inputs["origins"]["proposal"], origin["proposal_declaration"])
            and same(inputs["origins"]["terminals"], origin["terminal_declarations"])
            and same(custody["proposal"], selectors["proposal"]["pin"])
            and same(custody["source_roles"], proposal["source_roles"])
            and same(bundle["source_roles"], proposal["source_roles"])
            and same(declaration["diagnostic_identity"], proposal["identity"])
            and same(original_contract["sources"], proposal["identity"]["sources"]),
            "Manager proposal/source identity binding differs")
    for key in ("launch", "source_roles", "source_snapshots"):
        require(same(proposal[key], declaration[key]), "proposal declaration differs: " + key)
    for launch in (contract["launch"], inputs["launch"],
                   document(origin["actual_launch"])):
        require(same(launch, proposal["launch"]), "candidate launch differs from accepted proposal")
    # Custody serializes the same launch with sorted environment assignments.
    custody_launch = dict(proposal["launch"], command=" ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in sorted(proposal["launch"]["environment"].items())
    ) + " " + shlex.join(proposal["launch"]["argv"]))
    require(same(custody["launch"], custody_launch), "custody launch serialization differs")
    command_identity(custody["launch"])
    command_identity(proposal["launch"])
    require(custody["status"] == "PASS" and custody["authority_root"] is False
            and custody["dispatch_authorized"] is False
            and custody["scientific_or_admission_claim"] is False,
            "custody decision differs")
    zero_counters({k: custody[k] for k in (
        "scientific_invocations", "producer_invocations", "service_invocations")},
        ("scientific_invocations", "producer_invocations", "service_invocations"))
    require(same(proposal["identity"], contract["identity"])
            and result["status"] == "PASS"
            and result["authority_root"] is False and result["inputs_unchanged"] is True
            and result["mission_id"] == terminal["mission_id"]
            and result["proposal_id"] == terminal["proposal_id"]
            and result["dispatch_authorized"] is False
            and result["identity_decision"] == terminal["status"]
            and same(result["identity_preflight"], decision)
            and same(result["counters"], closure["counters"])
            and same(result["custody"], dict(custody, receipt=member("custody/custody-receipt.json")))
            and same(result["materialization"], dict(bundle, receipt=run["authority_bundle"])),
            "consumption result/proposal differs")
    for role, binding in proposal["source_roles"].items():
        require(same(binding["source"], proposal["identity"]["sources"][role])
                and same(binding["retained"], proposal["source_snapshots"][role]),
                "accepted source role/snapshot splice")
    current = contract["current_sources"]
    roles = copy.deepcopy(proposal["source_roles"])
    roles["current_diagnostic"] = {
        "source": current["candidate"], "retained": current["candidate"],
    }
    execution = copy.deepcopy(original_contract)
    execution["sources"]["current_diagnostic"] = current["candidate"]
    execution["sources"]["current_tests"] = current["tests"]
    require(same(contract["source_roles"], roles)
            and same(contract["execution_contract"], execution)
            and same(contract["launch_constraints"], original_contract["launch_constraints"])
            and set(contract["source_snapshots"]) == set(proposal["source_snapshots"]),
            "current source roles/execution contract splice")
    for role, key in (("current_diagnostic", "candidate"), ("current_tests", "tests")):
        require(authenticate(contract["source_snapshots"][role]) == authenticate(current[key]),
                "current source snapshot bytes differ")
    historical = proposal["source_snapshots"]["historical_generation"]
    require(same(contract["source_snapshots"]["historical_generation"], historical),
            "historical source snapshot splice")
    authenticate(historical)
    return proposal


def validate_after_claim(request_pin):
    """Read-only request validation and serialization; does not issue authority."""
    return _validate_after_claim(request_pin)


def validate_execution_envelope(envelope):
    """Consume the exact serialized emitter output through the same data-only gates.

    Re-serialization uses the authenticated issuance time, not a new claim or
    issuance. It permits the already-reserved output directory only here; the
    live issuer retains its exclusive, non-retryable reservation.
    """
    issuance = document(envelope["issuance"])
    issued_at = issuance["issued_at_utc"]
    require(timestamp(issued_at) <= time.time(), "future Manager issuance")
    output, documents, unchanged = _validate_after_claim(
        issuance["manager_request"], issued_at=issued_at)
    expected = json.loads(documents[3])
    require(same(envelope, expected), "emitter envelope binding differs")
    for name, raw in zip(OUTPUTS[:3], documents[:3]):
        record = pin_bytes(output / name, raw)
        require(authenticate(record) == raw, "emitter document binding differs")
    unchanged()
    return json.loads(documents[1])


def _validate_after_claim(request_pin, *, issued_at=None):
    now = time.time() if issued_at is None else timestamp(issued_at)
    request_raw = authenticate(request_pin)
    request = json.loads(request_raw)
    require(set(request) == {
        "kind", "issuer", "mission_id", "attempt", "scientific_run_budget",
        "issued_at_utc", "native_record", "mission_started_event", "contract",
        "seal", "contract_review", "accepted_identity", "accepted_review", "launch",
    }, "unexpected or missing Manager request fields")
    mission = request["mission_id"]
    require(isinstance(mission, str) and re.fullmatch(r"[0-9a-f]{12}", mission)
            and mission not in (MISSION, "04c9ee209234", "dec0e93fccfd", "208e2ffd3bca", "33543e0d3d9f",
                                "ed8fb34bae3a", "8df9b6fd2f5c"),
            "a distinct fresh science mission is required")
    authority = REPOSITORY / ".argus/live/manager-authority" / (
        "stage11-science-" + mission + "-attempt001"
    )
    require(request_pin["path"] == str(authority / "request.json"),
            "noncanonical Manager science authority")
    require(request["kind"] == "manager_stage11_postclaim_request"
            and request["issuer"] == "manager"
            and type(request["attempt"]) is int and request["attempt"] == 1
            and type(request["scientific_run_budget"]) is int
            and request["scientific_run_budget"] == 1,
            "ordinary Manager first-attempt integer-budget-one request required")
    root = REPOSITORY / ROOT
    require(request["contract"]["path"] == str(root / "current-runtime-contract.json")
            and request["seal"]["path"] == str(root / "seal.json"),
            "wrong current release root")
    contract = document(request["contract"])
    seal, members = sealed_members(request["seal"])
    require(seal["mission_id"] == MISSION
            and seal["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
            and any(same(r, request["contract"]) for r in members)
            and contract["mission_id"] == MISSION
            and contract["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
            and contract["admission_decision_requested"] == ADMISSION
            and contract["scientific_run_budget"] == 0
            and type(contract["scientific_run_budget"]) is int
            and contract["inherits_operational_admission"] is False,
            "unsealed, promoted, or inheriting release contract")
    for record in contract["implementation_sources"] + list(contract["current_sources"].values()):
        authenticate(record)
    original = document(contract["accepted_proposal"])
    require(same(original["identity"], contract["identity"]),
            "accepted scientific identity changed")
    review = review_bundle(request["contract_review"], MISSION)
    require(review["created_at"] >= contract["created_at"]
            and review["created_at"] <= now
            and review["review"]["reason"] == review_acceptance(
                request["contract"], request["seal"]),
            "Reviewer must explicitly accept this seal, audit limit, gates, and zero science")
    review_bundle(request["accepted_review"], contract["accepted_mission"],
                  contract["accepted_review"])
    require(same(request["accepted_review"]["checkpoint"],
                 contract["original_scientific_lineage"]["accepted_checkpoint"]),
            "accepted review checkpoint splice")
    accepted_lineage(request["accepted_identity"], contract)
    require(same(observe_runtime(), contract["runtime"]), "current runtime identity drift")
    row, event, _, _ = native_claim(LIFE, mission, now)
    require(same(stable_claim(row), stable_claim(request["native_record"]))
            and same(event, request["mission_started_event"]),
            "stale, wrong, or rebound post-claim Manager request")
    require(contract["created_at"] < row["started_ts"]
            and event["ts"] <= timestamp(request["issued_at_utc"]) <= now,
            "pre-claim or nonfresh Manager issuance")
    launch = request["launch"]
    expected_launch = copy.deepcopy(contract["launch"])
    expected_launch.update(mission_id=mission, constraints=contract["launch_constraints"])
    require(same(launch, expected_launch), "proposal/command/environment/budget lineage drift")
    require(launch["cwd"] == str(REPOSITORY) and launch["uid"] == os.getuid()
            and launch["role"] == "engineer"
            and launch["access"]["cpu_only"] is True
            and launch["access"]["gpu"] is False
            and launch["access"]["hardware"] is False
            and launch["access"]["ace2"] is False
            and type(launch["budget"]["scientific_suffix_runs"]) is int
            and launch["budget"]["scientific_suffix_runs"] == 1
            and type(launch["budget"]["check"]) is int and launch["budget"]["check"] == 1,
            "account/access/role/science budget differs")
    command_identity(launch)
    require(launch["argv"][1:] == original["identity"]["command_semantics"]["entry"],
            "candidate command recipe changed")
    output = authority / "issued"
    require(output.resolve() == output and not output.is_symlink(), "noncanonical issuance root")
    require(issued_at is not None or not output.exists(), "issuance attempt already consumed")
    snapshot = {
        "schema_version": 1, "kind": "native_running_claim_snapshot",
        "mission_id": mission, "attempt": 1, "status": "running",
        "scientific_run_budget": 1, "started_ts": row["started_ts"],
        "observed_at_utc": utc(now), "native_record": row,
        "mission_started_event": event, "constraints": launch["constraints"],
        "source": {"backlog": str(LIFE / "backlog.jsonl")},
        "current_release_contract": request["contract"],
    }
    claim_raw = json_bytes(snapshot)
    claim_pin = pin_bytes(output / OUTPUTS[0], claim_raw)
    declaration = {
        "schema_version": 1, "kind": "stage11_suffix_execution_authorization",
        "issuer": "manager", "authorized": True, "scientific_run_budget": 1,
        "issued_at_utc": utc(now), "launch": launch,
        "claim": {"pin": claim_pin, "mission_path": ["mission_id"],
                  "status_path": ["status"], "constraints_path": ["constraints"]},
        "contract": contract["execution_contract"], "identity": contract["identity"],
        "source_roles": contract["source_roles"], "source_snapshots": contract["source_snapshots"],
        "accepted_identity": request["accepted_identity"],
        "accepted_proposal": contract["accepted_proposal"],
        "accepted_preflight_mission": contract["accepted_mission"],
        "accepted_preflight_terminal_receipt": contract["accepted_terminal"],
        "current_release_contract": request["contract"],
        "current_release_review": request["contract_review"],
        "audit_limit": AUDIT_LIMIT, "attempt": 1,
    }
    declaration_raw = json_bytes(declaration)
    declaration_pin = pin_bytes(output / OUTPUTS[1], declaration_raw)
    issuance = {
        "schema_version": 1, "kind": "manager_stage11_execution_issuance",
        "issuer": "manager", "authorized_science": True,
        "authorized_before_running_claim": False, "scientific_run_budget": 1,
        "mission_id": mission, "attempt": 1, "issued_at_utc": utc(now),
        "declaration": declaration_pin, "claim": claim_pin,
        "accepted_identity_receipt": request["accepted_identity"]["launcher_capture"],
        "accepted_proposal": contract["accepted_proposal"],
        "current_release_contract": request["contract"],
        "current_release_review": request["contract_review"],
        "manager_request": request_pin, "runtime": contract["runtime"],
    }
    issuance_raw = json_bytes(issuance)
    envelope = {
        "schema_version": 1, "mission_id": mission, "attempt": 1,
        "scientific_run_budget": 1, "declaration": declaration_pin,
        "issuance": pin_bytes(output / OUTPUTS[2], issuance_raw),
        "review": request["accepted_review"],
        "current_release_contract": request["contract"],
        "current_release_review": request["contract_review"],
        "audit_limit": AUDIT_LIMIT,
    }

    def unchanged():
        require(authenticate(request_pin) == request_raw, "Manager request changed during issuance")
        document(request["contract"])
        sealed_members(request["seal"])
        review_bundle(request["contract_review"], MISSION)
        review_bundle(request["accepted_review"], contract["accepted_mission"],
                      contract["accepted_review"])
        accepted_lineage(request["accepted_identity"], contract)
        for record in contract["implementation_sources"] + list(contract["current_sources"].values()):
            authenticate(record)
        current, started, _, _ = native_claim(LIFE, mission, time.time())
        require(same(stable_claim(current), stable_claim(row)) and same(started, event),
                "claim changed during issuance")
        require(same(observe_runtime(), contract["runtime"]), "runtime changed during issuance")

    unchanged()
    return output, (claim_raw, declaration_raw, issuance_raw, json_bytes(envelope)), unchanged


def reserve_issuance(output):
    output.mkdir()  # A partial issuance consumes the canonical task/attempt.


def emit_after_claim(request_pin):
    """Dormant exclusive issuer; the envelope is published only after rechecking."""
    output, documents, unchanged = validate_after_claim(request_pin)
    reserve_issuance(output)
    for name, raw in zip(OUTPUTS[:3], documents[:3]):
        exclusive(output / name, raw)
    unchanged()
    exclusive(output / OUTPUTS[3], documents[3])
    return {name: pin(output / name) for name in OUTPUTS}


def build():
    require(Path.cwd() == REPOSITORY and sys.dont_write_bytecode
            and not sys.flags.optimize, "wrong workdir or interpreter flags")
    authority_path = REPOSITORY / ".argus/live/manager-authority" / AUTHORITY / "build-authorization.json"
    authority_raw = authority_path.read_bytes()
    authority = json.loads(authority_raw)
    owner_path = authority_path.with_name("manager-owner-claim.json")
    owner_raw = owner_path.read_bytes()
    require(digest(owner_raw) == CLAIM_SHA256, "authenticated Manager claim differs")
    owner = json.loads(owner_raw)
    require(authority["authority_id"] == owner["authority_id"] == AUTHORITY
            and authority["mission_id"] == owner["mission_id"] == MISSION
            and authority["authorized_root"] == owner["authorized_root"] == str(ROOT)
            and authority["issuer"] == owner["issuer"] == "manager"
            and authority["authorized_science"] is False
            and type(authority["scientific_run_budget"]) is int
            and authority["scientific_run_budget"] == owner["scientific_run_budget"] == 0
            and owner["state"] == "BOUND_TO_NATIVE_RUNNING_CLAIM"
            and owner["science_issuance"] is None
            and owner["manager_inputs"] == authority["manager_inputs"]
            and authority["native_task_metadata"]["authorization_id"] == ""
            and authority["native_task_metadata"]["authorization_action"] == "",
            "ambiguous_objective: ordinary zero-science authority differs")
    authenticated = {}

    def retain(record):
        raw = authenticate(record)
        authenticated[record["path"]] = (record, raw)
        return raw

    for record in authority["manager_inputs"].values():
        retain(record)
    proposal_pin = authority["manager_inputs"]["interface_repair"]
    require(proposal_pin["sha256"] == PROPOSAL_SHA256, "wrong pinned Manager proposal")
    proposal = json.loads(authenticated[proposal_pin["path"]][1])
    accepted = proposal["accepted_current_runtime"]
    old = json.loads(retain(accepted["contract"]))
    accepted_review = json.loads(retain(accepted["review"]))
    reviewer_done(accepted_review, "04c9ee209234")
    require(accepted["mission_id"] == old["mission_id"] == "04c9ee209234"
            and old["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW",
            "accepted current-runtime evidence differs")
    retain(accepted["termination_proposal"])
    predecessor = old["failed_predecessor"]
    retain(predecessor["contract"])
    failed_review = json.loads(retain(predecessor["review"]))
    retain(predecessor["seal"])
    previous_seal = document(predecessor["seal"])
    previous_root = Path(predecessor["seal"]["path"]).parent
    previous_members = []
    preserved_links = {}

    def preserve_member(member, previous_root):
        relative = Path(member["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "predecessor seal member escaped its root")
        path = previous_root / relative
        record = dict(member, path=str(path))
        # Preserve the predecessor's deliberate symlink-rejection test artifact,
        # without permitting symlinks in any current authority input.
        if path.is_symlink():
            require(path.is_relative_to(previous_root / "pytest-tmp")
                    and path.resolve().is_relative_to(previous_root / "pytest-tmp"),
                    "unexpected predecessor symlink")
            preserved_links[str(path)] = os.readlink(path)
            record = dict(record, path=str(path.resolve()))
        retain(record)

    for member in previous_seal["members"]:
        previous_members.append(dict(member, path=str(previous_root / member["path"])))
        preserve_member(member, previous_root)
    accepted_seal = json.loads(retain(accepted["seal"]))
    require(accepted_seal["mission_id"] == "04c9ee209234"
            and any(same(dict(m, path=str(Path(accepted["seal"]["path"]).parent / m["path"])),
                         accepted["contract"]) for m in accepted_seal["members"]),
            "accepted contract is not sealed")
    for member in accepted_seal["members"]:
        preserve_member(member, Path(accepted["seal"]["path"]).parent)
    require(predecessor["mission_id"] == previous_seal["mission_id"] == "dec0e93fccfd"
            and failed_review["producer_role"] == "reviewer"
            and failed_review["review"]["status"] == "replan_requested"
            and any(same(r, predecessor["contract"]) for r in previous_members),
            "failed predecessor must remain sealed REPLAN")
    report = document(authority["manager_inputs"]["preflight_report"])
    require(report["report"] == json.loads(retain(report["source"]))
            and report["status"] == report["report"]["status"] == "UNKNOWN"
            and report["report"]["release_consumption_admissible"] is False
            and report["report"]["release_consumption_authorized"] is False
            and report["report"]["checks"]["historical_failure_closure"] is False,
            "frozen 016 cannot be promoted")
    for group in ("current_operational_evidence", "original_scientific_lineage"):
        for value in old[group].values():
            if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}:
                retain(value)
    evidence = old["current_operational_evidence"]
    reviewer_done(document(evidence["data_only_review"]), "33543e0d3d9f")
    lineage = old["original_scientific_lineage"]
    reviewer_done(document(lineage["accepted_review"]), "ed8fb34bae3a")
    terminal = document(lineage["accepted_terminal"])
    original = document(lineage["proposal"])
    origin = document(lineage["manager_origin_issuance"])
    require(terminal["status"] == "ACCEPTED_NOVEL"
            and terminal["dispatch_authorized"] is False
            and terminal["closure_check"]["identity_decision"] == "ACCEPTED_NOVEL"
            and terminal["proposal_id"] == origin["proposal_id"] == lineage["proposal_id"]
            and same(terminal["manager_authority"]["issuance"], lineage["manager_origin_issuance"])
            and origin["authorized_science"] is False,
            "original accepted scientific lineage differs")
    origin_review = terminal["manager_authority"]["review"]
    for key in ("latest", "review", "checkpoint"):
        retain(origin_review[key])
    review_bundle(origin_review, origin_review["mission_id"])
    scientific_contract = json.loads(retain(origin["contract"]))
    require(same(scientific_contract["sources"], original["identity"]["sources"]),
            "original identity/source contract differs")
    accepted_index = json.loads(retain(terminal["receipt_index"]))
    for record in accepted_index["files"]:
        retain(record)
    historical_snapshot = original["source_snapshots"]["historical_generation"]
    retain(historical_snapshot)
    production = old["accepted_lineage"]
    for record in production.values():
        retain(record)
    now = time.time()
    row, event, row_raw, event_raw = native_claim(LIFE, MISSION, now)
    claim = owner["native_claim"]
    require(digest(row_raw) == claim["backlog_row_sha256"]
            and digest(event_raw) == claim["mission_started_event_sha256"]
            and row["objective"] == authority["expected_objective"]
            and row["started_ts"] == claim["started_ts"]
            and row["attempt"] == claim["attempt"] == authority["expected_attempt"]
            and row["node_key"] == claim["node_key"] == authority["planner"]["node_key"]
            and event["ts"] == claim["mission_started_ts"]
            and claim["started_ts"] <= event["ts"] <= owner["published_at"] <= now,
            "exact authenticated implementation claim changed")
    runtime = observe_runtime()
    require(runtime["pid"] == authority["runtime"]["daemon_pid"]
            and runtime["start_ticks"] == authority["runtime"]["daemon_start_ticks"]
            and runtime["source_root"] == str(REPOSITORY /
                "build/argus-stage11-terminal-replan-ledger-repair-attempt007/candidate/argus_skill"),
            "current runtime no longer matches ordinary Manager authority")
    source_pins = [pin(REPOSITORY / SOURCE), pin(REPOSITORY / TEST)]
    current = {key: pin(record["path"]) for key, record in old["current_sources"].items()}
    require(all(Path(r["path"]).stat().st_mtime_ns > owner["published_at"] * 1e9
                for r in source_pins + list(current.values())),
            "claim must predate implementation edits")
    require(now < timestamp(authority["expiry"]), "ordinary build authority expired")
    for record in source_pins + list(current.values()):
        retain(record)
    root = REPOSITORY / ROOT
    require(root.resolve() == root and not root.exists() and not root.is_symlink(),
            "sole fresh root already exists; no overwrite or retry")
    root.mkdir()
    audit = {"writes": [], "subprocesses": [], "blocked": []}
    candidate_tests = [
        "test_running_claim_accepts_actual_emitter_declaration_and_retained_lineage",
        "test_launch_authorization_rejects_invalid_bindings",
        "test_launch_requires_exact_one_shot_budget",
        "test_candidate_lineage_member_rejections",
        "test_candidate_obsolete_authorization_schema_rejected",
        "test_candidate_shared_lineage_rejects_repinned_semantics",
    ]
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               str(TEST), *[current["tests"]["path"] + "::" + name for name in candidate_tests],
               "-o", "junit_family=legacy",
               "--basetemp", str(root / "pytest-tmp"),
               "--junitxml", str(root / "focused-tests.xml")]

    def audit_hook(name, args):
        if name == "open":
            path, mode, flags = args
            if not isinstance(path, (str, bytes, os.PathLike)):
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                                    os.O_TRUNC | os.O_APPEND))
            if not writing:
                return
            target = Path(os.fsdecode(path)).resolve()
            if not target.is_relative_to(root):
                audit["blocked"].append([name, str(target)])
                raise PermissionError("builder write outside fresh root")
            audit["writes"].append(str(target))
        elif name == "subprocess.Popen":
            require(args[1] == command and not audit["subprocesses"],
                    "only one focused pytest command is permitted")
            audit["subprocesses"].append(command)
        elif name in ("os.system", "os.exec", "os.posix_spawn", "socket.connect"):
            audit["blocked"].append(name)
            raise PermissionError("dispatch forbidden in zero-science builder")
        elif name == "import" and ".candidates." in args[0]:
            audit["blocked"].append(name)
            raise PermissionError("candidate imports forbidden")

    sys.addaudithook(audit_hook)
    write_json(root / "construction-receipt.json", {
        "mission_id": MISSION, "authority": pin_bytes(authority_path, authority_raw),
        "manager_owner_claim": pin_bytes(owner_path, owner_raw),
        "claim_predates_edits_and_root": True, "runtime": runtime,
        "native_record": row, "mission_started_event": event, "created_at": now,
        "consumption": "AUTHENTICATED_JSON_DATA_ONLY", "executing_role": "engineer",
    })
    for name, raw in (
        ("build.command.txt", (BUILD_COMMAND + "\n").encode()),
        ("claim.command.txt", (CLAIM_COMMAND + "\n").encode()),
        ("claim.stdout", (json.dumps({
            "candidate_imports": 0, "mission_id": MISSION, "receipt": str(owner_path),
            "scientific_invocations": 0, "sha256": CLAIM_SHA256,
            "status": "BOUND_TO_NATIVE_RUNNING_CLAIM",
        }) + "\n").encode()),
        ("build-authorization.json", authority_raw), ("manager-owner-claim.json", owner_raw),
        ("manager-proposal.json", authenticate(proposal_pin)),
        ("manager-preflight-report.json", authenticate(authority["manager_inputs"]["preflight_report"])),
        ("source.py", authenticate(source_pins[0])), ("focused-tests.py", authenticate(source_pins[1])),
        ("candidate-source.py", authenticate(current["candidate"])),
        ("candidate-tests.py", authenticate(current["tests"])),
    ):
        exclusive(root / name, raw)
    execution_contract = copy.deepcopy(scientific_contract)
    execution_contract["sources"]["current_diagnostic"] = current["candidate"]
    execution_contract["sources"]["current_tests"] = current["tests"]
    roles = copy.deepcopy(original["source_roles"])
    roles["current_diagnostic"] = {
        "source": current["candidate"], "retained": current["candidate"],
    }
    snapshots = {"current_diagnostic": pin(root / "candidate-source.py"),
                 "current_tests": pin(root / "candidate-tests.py"),
                 "historical_generation": historical_snapshot}
    contract = {
        "schema_version": 1, "kind": "fresh_non_inheriting_current_runtime",
        "mission_id": MISSION, "created_at": now, "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
        "admission_decision_requested": ADMISSION, "scientific_run_budget": 0,
        "inherits_operational_admission": False, "manager_proposal": proposal_pin,
        "audit_limit": old["audit_limit"], "runtime": runtime,
        "failed_predecessor": predecessor, "accepted_current_runtime": accepted,
        "accepted_lineage": production,
        "implementation_sources": source_pins, "current_sources": current,
        "current_operational_evidence": evidence, "original_scientific_lineage": lineage,
        "identity": original["identity"], "accepted_proposal": lineage["proposal"],
        "accepted_mission": "ed8fb34bae3a", "accepted_review": lineage["accepted_review"],
        "accepted_terminal": lineage["accepted_terminal"], "accepted_files": accepted_index["files"],
        "execution_contract": execution_contract, "source_roles": roles,
        "source_snapshots": snapshots, "launch": original["launch"],
        "launch_constraints": scientific_contract["launch_constraints"],
        "historical_state": dict(old["historical_state"],
                                 observer_4e91="HISTORICAL_STATE_UNCHANGED_NOT_CLOSED"),
        "frozen_016_report": report, "boundary": report["report"]["boundary"],
        "future_science": {"scientific_run_budget": 1, "command_count": 1, "cpu_only": True,
                           "fresh_distinct_normal_claim": True, "attempt": 1, "retry": False},
        "review_policy": {
            "independent_host_reviewer": "REQUIRED", "root_expected_to_exist": True,
            "read_only_integrity_operations_only": True, "engineer_commands_allowed": False,
            "candidate_observer_builder_emitter_tests_allowed": False,
            "stage_transition_allowed": False,
            "acceptance_encoding": "Exact six-line review.reason from review_acceptance(contract_pin, seal_pin)",
        },
        "emitter": {"entry": "ace3.model.stage11_current_runtime_release.emit_after_claim",
                    "live_invocations": 0, "outputs": list(OUTPUTS),
                    "request_path": ".argus/live/manager-authority/stage11-science-<mission>-attempt001/request.json",
                    "output_path": "issued", "envelope_written_last": True,
                    "partial_issuance_consumes_attempt": True},
    }
    accepted_lineage(production, contract)
    write_json(root / "current-runtime-contract.json", contract)
    exclusive(root / "py_compile.command.txt", (
        BUILD_COMMAND + "\n"
        "py_compile.compile(source, cfile=fresh_root/compile-N.pyc, doraise=True)\n"
    ).encode())
    compile_sources = source_pins + list(current.values())
    for index, record in enumerate(compile_sources):
        py_compile.compile(record["path"], cfile=str(root / f"compile-{index}.pyc"),
                           doraise=True)
    write_json(root / "compile-receipt.json", {
        "status": "PASS", "mode": "py_compile explicit fresh-root cfile", "sources": compile_sources,
        "candidate_imports": 0, "candidate_compiles": 1,
    })
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=str(REPOSITORY),
                       STAGE11_TEST_ROOT=str(root / "pytest-tmp"))
    exclusive(root / "focused-tests.command.txt", (shlex.join(command) + "\n").encode())
    write_json(root / "focused-tests.argv.json", command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment,
                               capture_output=True, timeout=60, check=False)
    exclusive(root / "focused-tests.stdout", completed.stdout)
    exclusive(root / "focused-tests.stderr", completed.stderr)
    write_json(root / "focused-tests-receipt.json", {
        "argv": command, "exit_status": completed.returncode, "suite_runs": 1,
        "actual_authenticated_production_lineage_files": 33,
        "synthetic_future_claim_and_review_only": True,
        "candidate_selection": "data-only authority tests; no synthetic arithmetic or arrays",
        "emitter_invocations": 0,
        "environment_overrides": {k: environment[k] for k in (
            "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH",
            "STAGE11_TEST_ROOT")},
    })
    require(completed.returncode == 0, "focused tests failed; retained output is authoritative")
    properties = ET.parse(root / "focused-tests.xml").findall(".//property")
    forbidden = [p.attrib for p in properties if p.attrib["name"].startswith("forbidden_")]
    require(forbidden and all(p["value"] == "0" for p in forbidden),
            "candidate authority tests must record zero forbidden invocations")
    write_json(root / "zero-invocations.json", {
        "candidate_authority_test_properties": forbidden,
        "candidate_check_invocations": 0, "scientific_invocations": 0,
        "model_invocations": 0, "runtime_loads": 0, "array_loads": 0,
        "live_emitter_invocations": 0,
    })
    for record, raw in authenticated.values():
        require(authenticate(record) == raw, "authenticated input drift")
    require(all(Path(path).is_symlink() and os.readlink(path) == target
                for path, target in preserved_links.items()), "predecessor test symlink drift")
    require(authority_path.read_bytes() == authority_raw and owner_path.read_bytes() == owner_raw,
            "ordinary authority or owner receipt drift")
    after, after_event, _, _ = native_claim(LIFE, MISSION, time.time())
    require(same(stable_claim(after), stable_claim(row)) and same(after_event, event)
            and same(observe_runtime(), runtime), "claim/runtime drift during construction")
    write_json(root / "audit-disclosure.json", {
        "scope": "Python builder audit hooks and test process guards; command receipts and member census",
        "not_claimed": "OS-wide write confinement, sandbox equivalence, historical closure, or scientific results",
        "source_edits": "Engineer tool edits preceded builder audit; claim predates those edits",
        "claim_command_sidecar": "Exact operator command recorded after binder returned its authenticated receipt",
        "runtime_limit": "PID/start/cmdline and startup fingerprint record identity, not an OS attestation",
        "independent_acceptance_required": AUDIT_LIMIT, "builder_audit": audit,
    })
    write_json(root / "preserved-inputs.json", {
        "files": [record for record, _ in authenticated.values()],
        "predecessor_test_symlinks": preserved_links,
        "failed_predecessor_unchanged": True, "accepted_lineage_files": 33,
        "accepted_current_runtime_unchanged": True, "frozen_016_unchanged": True,
    })
    write_json(root / "termination-receipt.json", {
        "mission_id": MISSION, "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
        "scientific_run_budget": 0, "scientific_invocations": 0, "candidate_imports": 0,
        "candidate_executions": 0, "observer_dispatches": 0, "preflight_executions": 0,
        "candidate_check_invocations": 0, "model_invocations": 0,
        "candidate_authority_validation_in_tests_only": True,
        "live_emitter_invocations": 0, "historical_replays": 0,
        "runtime_model_budget_profile_continuous_changes": 0, "stage_transitions": 0,
        "emitter_implemented": True, "independent_review": "REQUIRED_PENDING_HOST_REVIEWER",
        "release_consumption_authorized": False, "release_consumption_admissible": False,
        "history": contract["historical_state"], "boundary": contract["boundary"],
    })
    result = json.dumps({
        "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW", "root": str(root),
        "scientific_invocations": 0,
    }) + "\n"
    exclusive(root / "build.stdout", result.encode())
    exclusive(root / "build.stderr", b"")
    write_json(root / "build-receipt.json", {"command": BUILD_COMMAND, "exit_status": 0})
    census = [dict(pin(p), path=str(p.relative_to(root)))
              for p in sorted(root.rglob("*")) if p.is_file()]
    write_json(root / "seal.json", {
        "schema_version": 1, "mission_id": MISSION,
        "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
        "members": census, "excludes": ["seal.json"], "scientific_run_budget": 0,
    })
    print(result, end="")


if __name__ == "__main__":
    require(sys.argv[1:] == ["--build"], "only zero-science --build is exposed")
    build()
