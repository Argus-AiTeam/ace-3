"""Data-only Stage11 release construction and dormant, post-claim issuance.

No candidate, observer, preflight producer, or daemon module is imported here.
The issuer writes authority, not scientific results, and has no execution CLI.
The candidate consumes its exact serialized envelope through this same
non-executing validator; retained scientific identity and current source pins
remain separate, authenticated bindings.

Immutable 640 construction and independent 412 artifact review are separate
provenance: 412 remains native FAILED after its final-delivery stage hold.
Neither accepts changed implementation bytes. Future requests must pin a
separate Manager operational-release
selection, published after normal Reviewer DONE and native DONE. The selection
binds all native evidence and exact implementation bytes without baking a
future review digest into this source. The pinned latest handoff selects its
canonical positive review round, not necessarily round one. It is not
scientific authority. Selection follows the authenticated native completion
event, not the earlier backlog timestamp. Runtime identity binds the observed
continuous-mode flag and generation without requiring scheduling to be disabled.
The v4 construction owner binds the reviewed runtime and continuous ON51;
that build-time authority does not restrict later observation to disabled mode.
Native handoff children bind their source through the exact launch environment,
while controlled children retain their source-path argv binding.

--prevalidate retains six-source compilation and narrow v6 release-binding
checks outside the absent delivery root. The independently reviewed ed9 repair
supplies authenticated retained candidate compatibility and 229-test evidence,
not another repair run. Its complete runtime-source/reference-member interface
is unchanged. --build PATH SHA256 requires an independent CONTINUE preseal
review with BUILD_FINAL_FROM_REVIEWED_BYTES under the same running claim, then
consumes unchanged passing bytes once. REPLAN terminates that claim and cannot
authorize construction. Audits are Python-scoped, not OS-wide; final
construction still requires independent Host review.
"""

from __future__ import annotations

import copy
import datetime as dt
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile


REPOSITORY = Path(__file__).resolve().parents[2]
LIFE = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05")
MISSION = "287d994f2fe4"
AUTHORITY = "s11-reviewed-interface-release-20260926-287d994f2fe4-r1"
ROOT = Path("build/argus-stage11-reviewed-interface-release-287d994f2fe4-attempt001")
FAILED_PREDECESSOR_MISSION = "fd53d72da1f9"
CONSTRUCTION_MISSION = "640a6d6db0cf"
CONSTRUCTION_ROOT = Path("build/argus-stage11-operational-ordering-successor-640a6d6db0cf-attempt001")
ARTIFACT_REVIEW_MISSION = "4127f6bb1e4a"
PROVENANCE_MISSION = "f1c261d97727"
PROVENANCE_ROOT = Path("build/argus-stage11-operational-framing-successor-f1c261d97727-attempt001")
VALIDATION_ROOT = Path(".argus/live/manager-validation/stage11-reviewed-interface-release-287d994f2fe4")
SOURCE = Path("ace3/model/stage11_current_runtime_release.py")
TEST = Path("tests/test_stage11_current_runtime_release.py")
CANDIDATE_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage11_attention_output_suffix_candidate_v1"
SOURCE_LAYOUT = (
    ("release_builder", SOURCE, "source.py"),
    ("release_tests", TEST, "focused-tests.py"),
    ("candidate", Path("ace3/model/candidates") / (CANDIDATE_NAME + ".py"), "candidate-source.py"),
    ("candidate_tests", Path("tests") / ("test_" + CANDIDATE_NAME + ".py"), "candidate-tests.py"),
    ("capture_helper", Path("ace3/model/candidates/diagnostic_capture_v1.py"), "capture-source.py"),
    ("capture_tests", Path("tests/test_diagnostic_capture_v1.py"), "capture-tests.py"),
)
CLAIM_SHA256 = "96f345d706a2418e231c047bc4ff64bcc54bfa068c1350a11314269b4098e4e6"
PROPOSAL_SHA256 = "0f860f4c7d9041845eb95684423247bd09f955fe20ac76bfb64322a4cf198bed"
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
    ".argus/live/manager-tools/publish-stage11-reviewed-interface-release-owner-claim-v6.py "
    "--authority-dir .argus/live/manager-authority/" + AUTHORITY
)
CONSUMPTION_MISSION = MISSION
CONSUMPTION_AUTHORITY = AUTHORITY
CONSUMPTION_ROOT = ROOT
CONSUMPTION_INPUT = {
    "path": str(REPOSITORY / ".argus/live/manager-inputs"
                / "stage11-reviewed-interface-release-287d994f2fe4.json"),
    "bytes": 40209,
    "sha256": PROPOSAL_SHA256,
}
CONSUMPTION_DECISION = {
    "path": str(REPOSITORY / ".argus/live/manager-authority"
                / CONSUMPTION_AUTHORITY / "manager-consumption-decision.json"),
    "bytes": 582,
    "sha256": "ce0974a1b34cb0a2314f39ba8305bfd2ce7b09de871c143d83f3a2b20797143c",
}
REFERENCE_MEMBERS = {
    "original_input_L23_fp16": "stage11",
    "original_input_L23_binary64": None,
    "original_input_final_fp16": None,
    "original_input_final_binary64": None,
}
ZERO_KINDS = ("scientific", "model", "producer", "service",
              "numerical", "prefix", "reference", "admission")


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
    require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"},
            "invalid pin fields")
    require(isinstance(record["path"], str) and bool(record["path"]), "invalid pin path")
    path = Path(record["path"])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            "noncanonical or symlinked input")
    require(type(record["bytes"]) is int and record["bytes"] >= 0, "invalid byte count")
    raw = path.read_bytes()
    require(len(raw) == record["bytes"] and digest(raw) == record["sha256"],
            "authenticated bytes changed: " + str(path))
    return raw


def proposal_context():
    """Bind proposed implementation imports without rewriting Manager authority."""
    repair = document(CONSUMPTION_INPUT)
    previous = document(repair["accepted_release"]["contract"])
    inherited = document(previous["manager_proposal"])
    # The retained Stage13 census predates the Stage11 candidate and consumer.
    # Their reviewed current bytes also appear in parent.source_context().
    repair["expected_runtime_sources"] = {
        **repair["expected_runtime_sources"],
        "ace3.model.stage11_current_runtime_release": pin(REPOSITORY / SOURCE),
        "ace3.model.candidates." + CANDIDATE_NAME: pin(REPOSITORY / SOURCE_LAYOUT[2][1]),
    }
    return {**inherited, **repair}


def validate_preparation_contract(contract):
    """Authenticate the complete data-only interface before any runtime/array load."""
    expected = proposal_context()
    sources = contract.get("runtime_sources")
    require(isinstance(sources, dict)
            and same(sources, expected["expected_runtime_sources"]),
            "runtime_sources missing or differs from retained census plus current implementation")
    require(same(expected["expected_reference_members"], REFERENCE_MEMBERS)
            and same(contract.get("reference_members"), REFERENCE_MEMBERS),
            "reference_members missing or differs from exact reference member census")
    for role, record in sources.items():
        try:
            authenticate(record)
        except (ValueError, OSError) as error:
            raise ValueError("runtime_sources " + role + ": " + str(error)) from error
    references = contract["execution_contract"]["references"]
    require(isinstance(references, dict) and set(references) == set(REFERENCE_MEMBERS),
            "incomplete reference pin census")
    for role, member in REFERENCE_MEMBERS.items():
        record = references[role]
        try:
            raw = authenticate(record)
        except (ValueError, OSError) as error:
            raise ValueError("reference " + role + ": " + str(error)) from error
        path = Path(record["path"])
        if member is None:
            require(path.suffix == ".npy" and raw.startswith(b"\x93NUMPY"),
                    role + ": no-member reference requires a .npy array")
        else:
            require(path.suffix == ".npz", role + ": stage11 reference requires a .npz archive")
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    names = archive.namelist()
                    require(len(names) == len(set(names))
                            and names.count("stage11.npy") == 1 and "stage11" not in names,
                            role + ": missing or ambiguous stage11 archive member")
                    require(archive.read("stage11.npy").startswith(b"\x93NUMPY"),
                            role + ": stage11 archive member is not a .npy array")
            except (zipfile.BadZipFile, RuntimeError) as error:
                raise ValueError(role + ": invalid .npz reference archive: " + str(error)) from error
    return sources, contract["reference_members"]


def implementation_pins():
    return [pin(REPOSITORY / path) for _, path, _ in SOURCE_LAYOUT]


def current_source_pins():
    return dict(zip(("candidate", "tests", "capture_helper", "capture_tests"),
                    implementation_pins()[2:]))


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
    require(type(continuous["enabled"]) is bool, "continuous enabled flag must be boolean")
    process = Path("/proc") / str(source["pid"])
    ticks = int((process / "stat").read_text().rsplit(")", 1)[1].split()[19])
    command = (process / "cmdline").read_bytes()
    source_root = Path(source["source_root"])
    require(source_root.is_absolute() and source_root.resolve() == source_root,
            "runtime source root is not canonical")
    handoff_command = [
        sys.executable.encode(), b"-c",
        b"from argus_skill.daemon.life_worker import run_handoff_child; "
        b"raise SystemExit(run_handoff_child())", b"",
    ]
    if command.split(b"\0") == handoff_command:
        environment = dict(entry.split(b"=", 1)
                           for entry in (process / "environ").read_bytes().split(b"\0")
                           if b"=" in entry)
        require(environment.get(b"ARGUS_SKILL_SOURCE_ROOT") == str(source_root.parent).encode()
                and environment.get(b"PYTHONPATH", b"").split(b":")[0]
                == str(source_root.parent).encode()
                and (process / "cwd").resolve() == Path("/"),
                "running handoff process does not bind the advertised runtime")
    else:
        require(str(source_root.parent).encode() in command.split(b"\0"),
                "running process does not name the advertised runtime")
    return {
        "pid": source["pid"], "start_ticks": ticks,
        "source_record": pin_bytes(source_path, json_bytes(source))
        if source_path.read_bytes() == json_bytes(source) else pin(source_path),
        "source_root": str(source_root),
        "source_fingerprint": source["source_fingerprint"],
        "cmdline_sha256": digest(command),
        "continuous_enabled": continuous["enabled"],
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


def native_review_consumption(bundle, contract_pin, seal_pin):
    """Authenticate f1 provenance only, not its obsolete implementation pins."""
    inputs = proposal_context()
    decision = document(CONSUMPTION_DECISION)
    accepted = construction_input(inputs)["accepted_provenance"]
    require(inputs["mission_id"] == MISSION
            and type(inputs["repair"]["scientific_run_budget"]) is int
            and inputs["repair"]["scientific_run_budget"] == 0
            and accepted["mission_id"] == PROVENANCE_MISSION
            and accepted["status"] == "NORMAL_REVIEWER_DONE_NATIVE_DONE",
            "wrong Manager consumption input")
    require(same(contract_pin, accepted["contract"])
            and same(seal_pin, accepted["seal"])
            and same(bundle, dict(
                mission_id=PROVENANCE_MISSION,
                **{key: accepted[key] for key in ("latest", "review", "checkpoint")})),
            "current release acceptance pins differ")
    review = native_acceptance(accepted, bundle, PROVENANCE_ROOT)
    require(same(decision, {
        "accepted_release": "d15533af5bdf", "authorized_root": str(ROOT),
        "decision": "BUILD_REVIEWED_INTERFACE_RELEASE",
        "failed_science": "b7d95bd6e7f2",
        "issuer": "manager", "mission_id": MISSION,
        "post_review_action": "PUBLISH_CANONICAL_SELECTION_THEN_RELEASE_DEPENDENT_SCIENCE_CONTINUATION",
        "post_review_owner": "manager",
        "repository_repair": "ed9ef3afb3ad",
        "scientific_run_budget": 0,
        "validation_root": str(VALIDATION_ROOT),
    }), "exact Manager consumption decision differs")
    return review


def construction_input(inputs):
    if "accepted_construction" not in inputs:
        previous = document(inputs["accepted_release"]["contract"])
        inputs = document(previous["manager_proposal"])
    return document(inputs["accepted_construction"]["manager_input"])


def continuation_provenance(inputs):
    """Authenticate retained construction and review as data, never execute them."""
    construction = inputs["accepted_construction"]
    artifact_review = inputs["accepted_artifact_review"]
    root = REPOSITORY / CONSTRUCTION_ROOT
    require(inputs["mission_id"] == MISSION
            and construction["mission_id"] == CONSTRUCTION_MISSION
            and construction["root"] == str(root)
            and artifact_review["mission_id"] == ARTIFACT_REVIEW_MISSION
            and artifact_review["native_status"] == "failed_stage_hold_after_reviewer_done",
            "split construction/review identity differs")
    for value in construction.values():
        if isinstance(value, dict):
            authenticate(value)
    seal, members = sealed_members(construction["seal"])
    require(construction["seal"]["path"] == str(root / "seal.json")
            and seal["mission_id"] == CONSTRUCTION_MISSION and len(members) == 81
            and all(any(same(record, member) for member in members)
                    for record in (construction["contract"], construction["construction_receipt"],
                                   construction["manager_input"], construction["termination"],
                                   construction["zero_invocations"], *inputs["construction_sources"])),
            "immutable construction seal differs")
    contract = document(construction["contract"])
    owner = document(construction["owner"])
    receipt = document(construction["construction_receipt"])
    require(contract["mission_id"] == receipt["mission_id"] == CONSTRUCTION_MISSION
            and owner["schema"] == "argus.manager-stage11-operational-successor-owner.v2"
            and owner["mission_id"] == CONSTRUCTION_MISSION
            and owner["state"] == "BOUND_TO_NATIVE_RUNNING_CLAIM"
            and same(receipt["manager_owner_claim"], construction["owner"])
            and same(receipt["authority"], construction["authority"])
            and receipt["builder_invocations"] == 1
            and receipt["claim_predates_root"] is True
            and receipt["prevalidation_completed_before_root"] is True
            and same(receipt["prevalidation"], construction["prevalidation"])
            and owner["native_claim"]["started_ts"] <= owner["published_at"],
            "original v2 owner/construction binding differs")
    by_name = {Path(member["path"]).relative_to(root).as_posix(): member for member in members}
    require(authenticate(by_name["manager-owner-claim.json"]) == authenticate(construction["owner"])
            and authenticate(by_name["claim.command.txt"]) == (
                "/home/argustest/miniconda3/bin/python -P "
                ".argus/live/manager-tools/publish-stage11-operational-ordering-successor-owner-claim-v2.py "
                "--authority-dir .argus/live/manager-authority/" + owner["authority_id"] + "\n").encode(),
            "original owner publication differs")
    build_receipt = document(by_name["build-receipt.json"])
    require(build_receipt["builder_invocations"] == 1 and build_receipt["exit_status"] == 0
            and build_receipt["validation_reruns"] == 0, "original single build differs")
    final_validation = None
    prior_finished = owner["published_at"]
    for attempt, status, failures in ((1, "FAIL", 2), (2, "PASS", 0)):
        evidence_pin = by_name[f"validation/attempt-{attempt:03d}/prevalidation.json"]
        evidence = document(evidence_pin)
        require(evidence["mission_id"] == CONSTRUCTION_MISSION and evidence["status"] == status
                and evidence["owner_sha256"] == construction["owner"]["sha256"]
                and prior_finished < evidence["started_at"] <= evidence["finished_at"]
                < receipt["created_at"], "original retained validation ordering differs")
        for member in evidence["members"]:
            authenticate(member)
        compile_result, tests = evidence["commands"]
        require(compile_result["name"] == "py_compile" and compile_result["exit_code"] == 0
                and compile_result["argv"][-6:] == [r["path"] for r in evidence["sources"]]
                and len(evidence["sources"]) == 6
                and tests["name"] == "focused-tests"
                and tests["exit_code"] == (1 if failures else 0)
                and same(tests["test_summary"], {
                    "tests": 436, "failures": failures, "errors": 0, "skipped": 0}),
                "original six-source compile/436 evidence differs")
        prior_finished = evidence["finished_at"]
        final_validation = evidence
    require(same(final_validation, document(construction["prevalidation"])),
            "original final PASS receipt differs")
    zero = document(construction["zero_invocations"])
    require(all(type(value) is int and value == 0 for key, value in zero.items()
                if key != "candidate_authority_test_properties")
            and zero["candidate_authority_test_properties"]
            and all(p["value"] == "0" for p in zero["candidate_authority_test_properties"]),
            "construction zero-science evidence differs")
    previous_inputs = construction_input(inputs)
    provenance = previous_inputs["accepted_provenance"]
    native_review_consumption(dict(mission_id=PROVENANCE_MISSION, **{
        key: provenance[key] for key in ("latest", "review", "checkpoint")}),
        provenance["contract"], provenance["seal"])
    failed = previous_inputs["failed_science_terminal"]
    for value in failed.values():
        if isinstance(value, dict):
            authenticate(value)
    require(failed["mission_id"] == "445f400d92c4" and failed["classification"] == "UNKNOWN"
            and document(failed["terminal"])["status"] == "UNKNOWN"
            and document(failed["terminal"])["scientific_check_invocations"] == 0,
            "445 trustworthy UNKNOWN changed")
    authenticate(previous_inputs["accepted_operational_release_selection"])
    for (role, path, name), snapshot in zip(SOURCE_LAYOUT, inputs["construction_sources"], strict=True):
        require(snapshot["path"] == str(root / name), "construction source snapshot path differs")
        raw = authenticate(snapshot)
        if role in ("capture_helper", "capture_tests"):
            require(raw == authenticate(inputs["source_baseline"][role])
                    == (REPOSITORY / path).read_bytes(),
                    "candidate/capture drift from sealed construction")
    bundle = dict(mission_id=ARTIFACT_REVIEW_MISSION, **{
        key: artifact_review[key] for key in ("latest", "review", "checkpoint")})
    review = review_bundle(bundle, ARTIFACT_REVIEW_MISSION)
    base = LIFE / "handoffs" / ARTIFACT_REVIEW_MISSION
    mission = document(artifact_review["mission"])
    require(artifact_review["mission"]["path"] == str(base / "mission.json")
            and mission["mission_id"] == ARTIFACT_REVIEW_MISSION
            and review["round"] == 1 and review["schema_version"] == 3
            and artifact_review["review"]["path"] == str(base / "round-0001.json")
            and review["mission_context"] == artifact_review["mission"]["path"]
            and same(document(artifact_review["latest"]), {
                "kind": "handoff_ref", "schema_version": 3,
                "handoff": {"path": artifact_review["review"]["path"]},
                "mission": {"path": artifact_review["mission"]["path"]}})
            and same(review["review"], {
                "status": "done", "reason": "requested outcome is materially complete",
                "operator_question": "", "next_action": ""}),
            "exact 412 normal independent review differs")
    rows = {}
    for raw in (LIFE / "backlog.jsonl").read_bytes().splitlines():
        if raw.strip():
            row = json.loads(raw)
            if row.get("id") in (CONSTRUCTION_MISSION, ARTIFACT_REVIEW_MISSION, "ab65cb0e6bea"):
                require(row["id"] not in rows, "duplicate historical native row")
                rows[row["id"]] = (row, raw)
    require(set(rows) == {CONSTRUCTION_MISSION, ARTIFACT_REVIEW_MISSION, "ab65cb0e6bea"}
            and rows[CONSTRUCTION_MISSION][0]["status"] == "failed"
            and rows["ab65cb0e6bea"][0]["status"] == "aborted",
            "construction FAILED or ab65 ABORTED history changed")
    row, raw = rows[ARTIFACT_REVIEW_MISSION]
    require(digest(raw) == artifact_review["backlog_row_sha256"]
            and row["status"] == "failed" and row["attempt"] == 1
            and row["finished_ts"] == artifact_review["finished_ts"]
            and same(row["outcome"], {
                "execution_status": "ended", "interruption_kind": "none", "resumable": False,
                "review_status": "done", "stage_certification": "not_certified"})
            and row["last_error"] == (
                "manager stage hold: Latest Reviewer accepted the bounded delivery increment, "
                "but this is an open-ended campaign with unresolved follow-on work and no legal "
                "advance or rollback target from the final delivery stage."),
            "exact 412 stage-hold FAILED terminal changed")
    events = [json.loads(raw) for raw in (LIFE / "events.jsonl").read_bytes().splitlines()
              if digest(raw) == artifact_review["review_completed_event_sha256"]]
    require(len(events) == 1 and events[0]["type"] == "round.review.completed"
            and events[0]["review_source"] == "reviewer" and events[0]["status"] == "done"
            and events[0]["review_skipped"] is False
            and events[0]["ts"] == artifact_review["review_completed_ts"]
            and receipt["created_at"] < events[0]["ts"] <= review["created_at"]
            < row["finished_ts"] <= time.time(), "exact 412 review event differs")
    return {"construction": construction, "artifact_review": artifact_review}


def native_acceptance(accepted, bundle, root, *, selection_issued_at=None):
    """Read native terminal evidence and the normal path-only Host handoff."""
    mission_id = accepted["mission_id"]
    contract_pin, seal_pin = accepted["contract"], accepted["seal"]
    require(accepted["status"] == "NORMAL_REVIEWER_DONE_NATIVE_DONE"
            and same(bundle, dict(mission_id=mission_id, **{
                key: accepted[key] for key in ("latest", "review", "checkpoint")})),
            "selected review bundle differs")
    base = LIFE / "handoffs" / mission_id
    require(accepted["mission"]["path"] == str(base / "mission.json")
            and contract_pin["path"] == str(REPOSITORY / root / "current-runtime-contract.json")
            and seal_pin["path"] == str(REPOSITORY / root / "seal.json"),
            "native acceptance paths differ")
    latest = document(accepted["latest"])
    mission = document(accepted["mission"])
    review = review_bundle(bundle, mission_id, accepted["review"])
    require(same(latest, {
        "schema_version": 3, "kind": "handoff_ref",
        "handoff": {"path": accepted["review"]["path"]},
        "mission": {"path": accepted["mission"]["path"]},
    }), "native latest must use the exact path-only schema")
    require(mission["kind"] == "mission_context" and mission["schema_version"] == 3
            and mission["mission_id"] == mission_id
            and review["schema_version"] == 3
            and type(review["round"]) is int and review["round"] > 0
            and accepted["review"]["path"] == str(base / f"round-{review['round']:04d}.json")
            and review["mission_context"] == accepted["mission"]["path"]
            and same(review["review"], {
                "status": "done", "reason": "requested outcome is materially complete",
                "operator_question": "", "next_action": "",
            }), "exact normalized native Reviewer DONE required")
    contract = document(contract_pin)
    seal, members = sealed_members(seal_pin)
    require(contract["mission_id"] == seal["mission_id"] == mission_id
            and any(same(member, contract_pin) for member in members)
            and number(review["created_at"])
            and contract["created_at"] <= review["created_at"] <= time.time(),
            "native review does not bind this sealed contract")
    terminal = accepted["native_terminal"]
    rows = []
    events = []
    for name, matches, key in (
        ("backlog.jsonl", rows, "id"), ("events.jsonl", events, "item_id"),
    ):
        for raw in (LIFE / name).read_bytes().splitlines():
            if raw.strip():
                value = json.loads(raw)
                if value.get(key) == mission_id and (
                    key == "id" or value.get("type") == "life.mission.completed"
                ):
                    matches.append((value, raw))
    require(len(rows) == len(events) == 1, "missing or duplicate native terminal")
    row, row_raw = rows[0]
    event, event_raw = events[0]
    if mission_id == MISSION:
        require("stage_transition:skip" in mission.get("tags", [])
                and "stage_transition:skip" in row.get("tags", [])
                and row["outcome"].get("stage_certification") == "intentionally_skipped"
                and event.get("outcome", {}).get("stage_certification") == "intentionally_skipped",
                "current release must skip campaign stage transition")
    require(digest(row_raw) == terminal["backlog_row_sha256"]
            and digest(event_raw) == terminal["mission_completed_event_sha256"]
            and row["status"] == event["status"] == "done"
            and type(row["attempt"]) is int and row["attempt"] == 1
            and row["finished_ts"] == terminal["finished_ts"]
            and row["outcome"]["review_status"] == "done"
            and row["outcome"]["resumable"] is False
            and event["independent_review_required"] is True
            and event["resumable"] is False
            and number(row["finished_ts"]) and number(event["ts"])
            and review["created_at"] <= row["finished_ts"] <= event["ts"] <= time.time(),
            "exact native DONE terminal changed")
    if selection_issued_at is not None:
        require(event["ts"] <= selection_issued_at <= time.time(),
                "selection must follow native DONE")
    return review


def operational_release_selection(selection_pin, bundle, contract_pin, seal_pin):
    """Authenticate a separately published Manager selection, without issuing it."""
    selection = document(selection_pin)
    require(set(selection) == {
        "kind", "issuer", "status", "decision", "scientific_run_budget", "issued_at_utc",
        "selected", "implementation_sources", "provenance", "manager_input",
        "manager_decision", "admission", "audit_limit",
    }, "unexpected operational-release selection fields")
    selected = selection["selected"]
    mission = selected["mission_id"]
    require(isinstance(mission, str) and re.fullmatch(r"[0-9a-f]{12}", mission)
            and mission == MISSION,
            "provenance is not a changed-byte release")
    authority = REPOSITORY / ".argus/live/manager-authority" / (
        "stage11-operational-release-" + mission)
    require(selection_pin["path"] == str(authority / "selection.json"),
            "noncanonical Manager operational-release selection")
    require(selection["kind"] == "manager_stage11_operational_release_selection"
            and selection["issuer"] == "manager"
            and selection["status"] == "SELECTED_AFTER_NORMAL_REVIEW_AND_NATIVE_DONE"
            and selection["decision"] == "ADMISSIBLE_POST_REVIEW_CONSUMPTION"
            and type(selection["scientific_run_budget"]) is int
            and selection["scientific_run_budget"] == 0
            and selection["admission"] == ADMISSION and selection["audit_limit"] == AUDIT_LIMIT
            and same(selection["manager_input"], CONSUMPTION_INPUT)
            and same(selection["manager_decision"], CONSUMPTION_DECISION),
            "Manager operational-release decision differs")
    inputs = proposal_context()
    provenance = continuation_provenance(inputs)
    require(same(selection["provenance"], provenance),
            "selected provenance differs from Manager input")
    require(same(contract_pin, selected["contract"]) and same(seal_pin, selected["seal"]),
            "request differs from selected release")
    review = native_acceptance(
        selected, bundle, ROOT,
        selection_issued_at=timestamp(selection["issued_at_utc"]),
    )
    contract = document(contract_pin)
    for record in selection["implementation_sources"]:
        try:
            authenticate(record)
        except (ValueError, OSError) as error:
            raise ValueError("selected implementation source version: " + str(error)) from error
    expected = implementation_pins()
    require(same(selection["implementation_sources"], expected)
            and same(contract["implementation_sources"], expected)
            and same(contract["construction_provenance"], provenance["construction"])
            and same(contract["artifact_review_provenance"], provenance["artifact_review"])
            and same(contract["manager_proposal"], CONSUMPTION_INPUT)
            and same(contract["current_sources"], current_source_pins()),
            "selected implementation/provenance/source version differs")
    previous = document(provenance["construction"]["contract"])
    operational_fields = {
        "mission_id", "created_at", "runtime", "manager_proposal", "implementation_sources",
        "validation_policy", "boundary", "post_review_owner", "source_snapshots", "capture_snapshots",
        "current_sources", "source_roles", "execution_contract",
    }
    require(all(key in contract and same(contract[key], value)
                for key, value in previous.items() if key not in operational_fields),
            "immutable construction scientific contract differs")
    execution = copy.deepcopy(previous["execution_contract"])
    execution["sources"]["current_diagnostic"] = contract["current_sources"]["candidate"]
    execution["sources"]["current_tests"] = contract["current_sources"]["tests"]
    roles = copy.deepcopy(previous["source_roles"])
    roles["current_diagnostic"] = {
        "source": contract["current_sources"]["candidate"],
        "retained": contract["current_sources"]["candidate"],
    }
    require(same(contract["execution_contract"], execution)
            and same(contract["source_roles"], roles),
            "immutable construction scientific contract differs")
    validate_preparation_contract(contract)
    require(isinstance(contract.get("capture_snapshots"), dict)
            and set(contract["capture_snapshots"]) == {"capture_helper", "capture_tests"},
            "capture snapshot census differs")
    for key, name in (("capture_helper", "capture-source.py"),
                      ("capture_tests", "capture-tests.py")):
        snapshot = contract["capture_snapshots"][key]
        require(snapshot["path"] == str(REPOSITORY / ROOT / name)
                and authenticate(snapshot) == authenticate(contract["current_sources"][key]),
                "capture snapshot differs")
    return review, selection


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
        "operational_release_selection",
    }, "unexpected or missing Manager request fields")
    mission = request["mission_id"]
    require(isinstance(mission, str) and re.fullmatch(r"[0-9a-f]{12}", mission)
            and mission not in (MISSION, CONSTRUCTION_MISSION, ARTIFACT_REVIEW_MISSION,
                                PROVENANCE_MISSION, "94e2e47338e2",
                                "bc4840101b6f", "fc2f78aa2e3a", "609b6cad06bc", "e2ba4dbc8651",
                                "04c9ee209234", "dec0e93fccfd", "208e2ffd3bca", "33543e0d3d9f",
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
    review, selection = operational_release_selection(
        request["operational_release_selection"], request["contract_review"],
        request["contract"], request["seal"])
    selected_mission = selection["selected"]["mission_id"]
    require(mission != selected_mission, "science cannot reuse the selected release mission")
    contract = document(request["contract"])
    validate_preparation_contract(contract)
    seal, members = sealed_members(request["seal"])
    require(seal["mission_id"] == selected_mission
            and seal["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
            and type(seal["scientific_run_budget"]) is int
            and seal["scientific_run_budget"] == 0
            and any(same(r, request["contract"]) for r in members)
            and contract["mission_id"] == selected_mission
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
    require(review["created_at"] >= contract["created_at"]
            and review["created_at"] <= now,
            "Reviewer acceptance is outside the current contract lifetime")
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
            and timestamp(selection["issued_at_utc"]) < row["started_ts"]
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
        "runtime_sources": contract["runtime_sources"],
        "reference_members": contract["reference_members"],
        "accepted_identity": request["accepted_identity"],
        "accepted_proposal": contract["accepted_proposal"],
        "accepted_preflight_mission": contract["accepted_mission"],
        "accepted_preflight_terminal_receipt": contract["accepted_terminal"],
        "current_release_contract": request["contract"],
        "current_release_review": request["contract_review"],
        "operational_release_selection": request["operational_release_selection"],
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
        "operational_release_selection": request["operational_release_selection"],
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
        "operational_release_selection": request["operational_release_selection"],
        "audit_limit": AUDIT_LIMIT,
    }

    def unchanged():
        require(authenticate(request_pin) == request_raw, "Manager request changed during issuance")
        document(request["contract"])
        validate_preparation_contract(contract)
        sealed_members(request["seal"])
        operational_release_selection(
            request["operational_release_selection"], request["contract_review"],
            request["contract"], request["seal"])
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


def proposed_contract(previous, proposal_pin, snapshot_root, now, runtime):
    """Rebind operational bytes, never the accepted scientific identity."""
    contract = copy.deepcopy(previous)
    proposal = document(proposal_pin)
    require(same(proposal_pin, CONSUMPTION_INPUT), "wrong repair proposal")
    proposal = proposal_context()
    current = current_source_pins()
    for record in current.values():
        authenticate(record)
    contract.update(
        mission_id=MISSION, created_at=now, runtime=runtime, manager_proposal=proposal_pin,
        status="PROPOSED_PENDING_INDEPENDENT_REVIEW", inherits_operational_admission=False,
        scientific_run_budget=0, current_sources=current,
        implementation_sources=implementation_pins())
    require(all(same(current[key], previous["current_sources"][key])
                for key in ("capture_helper", "capture_tests")),
            "capture source version differs from construction")
    contract["runtime_sources"] = proposal["expected_runtime_sources"]
    contract["reference_members"] = proposal["expected_reference_members"]
    contract["construction_provenance"] = proposal["accepted_construction"]
    contract["artifact_review_provenance"] = proposal["accepted_artifact_review"]
    contract["validation_policy"] = proposal["validation_policy"]
    contract["boundary"] = previous["boundary"].replace(
        "No validation rerun or dispatch.",
        "Ordinary software validation may repeat with retained receipts; no scientific dispatch.")
    contract["post_review_owner"] = {
        "owner": "manager", "action": "PUBLISH_AUTHENTICATED_OPERATIONAL_RELEASE_SELECTION",
        "requires_normal_reviewer_done_and_native_done": True, "science_authorized": False,
    }
    for role, key, name in (
        ("current_diagnostic", "candidate", "candidate-source.py"),
        ("current_tests", "tests", "candidate-tests.py"),
    ):
        contract["execution_contract"]["sources"][role] = current[key]
        snapshot = pin(snapshot_root / name)
        require(authenticate(snapshot) == authenticate(current[key]),
                "candidate snapshot must be materialized before proposing")
        contract["source_snapshots"][role] = snapshot
    contract["source_roles"]["current_diagnostic"] = {
        "source": current["candidate"], "retained": current["candidate"],
    }
    contract["capture_snapshots"] = {}
    for key, name in (("capture_helper", "capture-source.py"),
                      ("capture_tests", "capture-tests.py")):
        snapshot = pin(snapshot_root / name)
        require(authenticate(snapshot) == authenticate(current[key]),
                "capture snapshot must be materialized before proposing")
        contract["capture_snapshots"][key] = snapshot
    validate_preparation_contract(contract)
    return contract


def source_edit_bindings(proposal, owner):
    bindings = {}
    for role, path, _ in SOURCE_LAYOUT:
        source = REPOSITORY / path
        unchanged = same(pin(source), proposal["source_baseline"][role])
        require(role in ("release_builder", "release_tests") or unchanged,
                "reviewed ed9 candidate/capture drift")
        require(unchanged or source.stat().st_mtime_ns > owner["published_at"] * 1e9,
                "native owner claim must precede changed source bytes")
        bindings[role] = "UNCHANGED_MANAGER_BASELINE" if unchanged else "POSTCLAIM_EDIT"
    return bindings


def reviewed_repair_evidence(proposal):
    """Consume immutable ed9 review/validation bytes without rerunning the repair."""
    repair = proposal["repository_repair"]
    mission = repair["mission_id"]
    require(mission == "ed9ef3afb3ad", "wrong reviewed repository repair")
    review = review_bundle(dict(mission_id=mission, **{
        key: repair[key] for key in ("latest", "review", "checkpoint")}), mission)
    base = LIFE / "handoffs" / mission
    require(repair["mission"]["path"] == str(base / "mission.json")
            and document(repair["mission"])["mission_id"] == mission
            and review["round"] == 2
            and repair["review"]["path"] == str(base / "round-0002.json")
            and review["mission_context"] == repair["mission"]["path"]
            and same(review["review"], {
                "status": "done", "reason": "requested outcome is materially complete",
                "operator_question": "", "next_action": "",
            }), "exact ed9 round-2 Reviewer DONE required")
    proposed = document(repair["proposed_bytes"])
    root = Path(repair["proposed_bytes"]["path"]).parent
    sources = [proposal["source_baseline"][role] for role, _, _ in SOURCE_LAYOUT]
    require(proposed["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
            and proposed["owner_authority_claimed"] is False
            and proposed["final_root_built"] is False
            and len(sources) == 6 and same(proposed["sources"], sources),
            "reviewed ed9 six-source census differs")
    for (_, path, name), source in zip(SOURCE_LAYOUT, sources, strict=True):
        require(source["path"] == str(REPOSITORY / path)
                and same(pin_bytes(REPOSITORY / path, (root / name).read_bytes()), source),
                "reviewed ed9 snapshot differs")
    commands = [document(repair[key]) for key in ("compile_receipt", "focused_tests_receipt")]
    require(same(commands, proposed["results"])
            and commands[0]["name"] == "py_compile"
            and commands[0]["argv"][-6:] == [source["path"] for source in sources]
            and commands[1]["name"] == "focused-tests"
            and same(commands[1]["test_summary"], {
                "tests": 229, "failures": 0, "errors": 0, "skipped": 0}),
            "reviewed ed9 compile/229 evidence differs")
    for result in commands:
        name = result["name"]
        require(type(result["exit_code"]) is int and result["exit_code"] == 0
                and result["timed_out"] is False and result["cwd"] == str(REPOSITORY)
                and result["started_at"] <= result["finished_at"] <= review["created_at"]
                and same(json.loads((root / (name + ".argv.json")).read_bytes()), result["argv"])
                and (root / (name + ".command.txt")).read_bytes()
                == (shlex.join(result["argv"]) + "\n").encode()
                and same(json.loads((root / (name + ".environment.json")).read_bytes()),
                         result["environment"]),
                "reviewed ed9 command evidence differs")
        authenticate(result["stdout"])
        authenticate(result["stderr"])
    xml = authenticate(repair["focused_tests_xml"])
    forbidden = test_summary(xml, authenticate(commands[1]["stdout"]))
    cases = ET.fromstring(xml).findall(".//testcase")
    require(len(cases) == 229 and all(any(case.attrib["name"] == name for case in cases)
            for name in (
                "test_fresh_proposed_contract_consumes_production",
                "test_running_claim_accepts_actual_emitter_declaration_and_retained_lineage",
                "test_source_authentication_precedes_real_dispatch_guard",
                "test_preparation_consumes_real_emitter_complete_contract",
            )), "retained real emitter/candidate compatibility coverage missing")
    rows, events = [], []
    for filename, matches, key in (
        ("backlog.jsonl", rows, "id"), ("events.jsonl", events, "item_id"),
    ):
        for raw in (LIFE / filename).read_bytes().splitlines():
            if raw.strip():
                value = json.loads(raw)
                if value.get(key) == mission and (
                    key == "id" or value.get("type") == "life.mission.completed"
                ):
                    matches.append((value, raw))
    require(len(rows) == len(events) == 1, "ed9 native terminal missing or duplicate")
    row, _ = rows[0]
    event, event_raw = events[0]
    require(row["status"] == event["status"] == "done"
            and type(row["attempt"]) is int and row["attempt"] == 1
            and row["outcome"]["review_status"] == "done"
            and event["final_review_source"] == "reviewer"
            and event["final_review_status"] == "done"
            and review["created_at"] <= event["ts"] <= time.time()
            and digest(event_raw) == repair["mission_completed_event_sha256"],
            "ed9 immutable native completion differs")
    return forbidden


def successor_inputs():
    authority_path = REPOSITORY / ".argus/live/manager-authority" / AUTHORITY / "build-authorization.json"
    authority = json.loads(authority_path.read_bytes())
    owner_path = authority_path.with_name("manager-owner-claim.json")
    owner = document(dict(pin(owner_path), sha256=CLAIM_SHA256))
    require(owner["authority_id"] == authority["authority_id"] == AUTHORITY
            and owner["mission_id"] == authority["mission_id"] == MISSION
            and owner["authorized_root"] == authority["authorized_root"] == str(ROOT)
            and authority["validation_root"] == str(VALIDATION_ROOT)
            and authority["issuer"] == owner["issuer"] == "manager"
            and owner["schema"] == "argus.manager-stage11-reviewed-interface-release-owner.v6"
            and owner["accepted_release_mission"] == "d15533af5bdf"
            and owner["accepted_repository_repair_mission"] == "ed9ef3afb3ad"
            and owner["failed_science_mission"] == "b7d95bd6e7f2"
            and owner["state"] == "BOUND_TO_NATIVE_RUNNING_CLAIM"
            and authority["authorized_science"] is False
            and type(authority["scientific_run_budget"]) is int
            and type(owner["scientific_run_budget"]) is int
            and authority["scientific_run_budget"] == owner["scientific_run_budget"] == 0
            and authority["scope"]["ordinary_validation_repeatable"] is True
            and authority["scope"]["stage_transition"] is False
            and authority["scope"]["independent_review_required"] is True,
            "operational-transition authority differs")
    for record in authority["manager_inputs"].values():
        authenticate(record)
    proposal_pin = authority["manager_inputs"]["source_input"]
    require(same(proposal_pin, CONSUMPTION_INPUT)
            and same(authority["manager_inputs"]["consumption_decision"], CONSUMPTION_DECISION),
            "wrong operational-transition input")
    proposal = proposal_context()
    require(same(owner["reviewed_proposed_bytes"], proposal["repository_repair"]["proposed_bytes"]),
            "v6 owner reviewed-source binding differs")
    reviewed_repair_evidence(proposal)
    expected = proposal["expected_runtime"]
    observed = observe_runtime()
    startup = document(observed["source_record"])
    status = json.loads((LIFE / "daemon.status.json").read_bytes())
    continuous = json.loads((LIFE / "continuous.json").read_bytes())
    runtime = status["runtime"]
    accepted_root = Path(proposal["accepted_release"]["contract"]["path"]).parent
    runtime_owner = document(pin(accepted_root / "manager-owner-claim.json"))
    require(same(owner["runtime"], runtime_owner["runtime"])
            and same(runtime_owner["runtime"], {
                **{key: expected[key] for key in (
                    "pid", "release_id", "source_root", "source_digest",
                    "source_fingerprint", "source_file_count", "continuous_generation")},
                "started_at_iso": status["started_at_iso"],
                "continuous_enabled": True, "continuous_open_ended": True,
                "continuous_objective_sha256": digest(expected["continuous_objective"].encode()),
            })
            and status["pid"] == observed["pid"] == expected["pid"]
            and runtime["release_id"] == expected["release_id"]
            and runtime["source_root"] == expected["source_root"]
            and runtime["runtime_source_digest"] == runtime["manifest_source_digest"]
            == expected["source_digest"]
            and runtime["release_matches_source"] is True
            and observed["source_root"] == expected["argus_source_root"]
            and observed["source_fingerprint"] == expected["source_fingerprint"]
            and startup["source_file_count"] == expected["source_file_count"]
            and observed["continuous_enabled"] is continuous["enabled"] is True
            and observed["continuous_generation"] == continuous["generation"]
            == expected["continuous_generation"]
            and continuous["open_ended"] is True
            and continuous["objective"] == expected["continuous_objective"],
            "v4 owner runtime or continuous scheduling differs")
    continuation_provenance(proposal)
    accepted = proposal["accepted_release"]
    for value in accepted.values():
        if isinstance(value, dict):
            authenticate(value)
    sealed_members(accepted["seal"])
    selection = document(accepted["selection"])
    native_acceptance(selection["selected"], dict(mission_id=accepted["mission_id"], **{
        key: accepted[key] for key in ("latest", "review", "checkpoint")}), accepted_root,
        selection_issued_at=timestamp(selection["issued_at_utc"]))
    for value in proposal["failed_science"].values():
        if isinstance(value, dict):
            authenticate(value)
    require(document(proposal["failed_science"]["terminal"])["status"] == "UNKNOWN",
            "consumed b7d trustworthy UNKNOWN changed")
    previous = document(accepted["contract"])
    rows = [(json.loads(raw), raw) for raw in (LIFE / "backlog.jsonl").read_bytes().splitlines()
            if raw.strip()]
    predecessor = [row for row, _ in rows if row.get("id") == FAILED_PREDECESSOR_MISSION]
    require(len(predecessor) == 1 and predecessor[0]["status"] == "failed"
            and predecessor[0]["attempt"] == 1
            and predecessor[0]["last_error"] == (
                "Sealing the sole root before independent review left no legal "
                "in-mission repair path for a production-validator defect."),
            "fd53 failed predecessor differs")
    original_inputs = construction_input(proposal)
    for record in original_inputs["failed_science_terminal"].values():
        if isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}:
            authenticate(record)
    authenticate(original_inputs["accepted_operational_release_selection"])
    failed = previous["terminal_successor_lineage"]["failed_terminal_successor"]
    require(not Path(failed["root"]).exists() and not Path(failed["root"]).is_symlink(),
            "609 root must stay absent")
    document(failed["claim"])
    require(document(failed["review"])["review"]["status"] == "replan_requested",
            "609 review changed")
    terminal = [row for row, _ in rows if row.get("id") == failed["mission_id"]]
    require(len(terminal) == 1 and terminal[0]["status"] == "failed"
            and terminal[0]["finished_ts"] == failed["native_finished_ts"], "609 terminal changed")
    now = time.time()
    row, event, raw, event_raw = native_claim(LIFE, MISSION, now)
    require(digest(raw) == owner["native_claim"]["backlog_row_sha256"]
            and digest(event_raw) == owner["native_claim"]["mission_started_event_sha256"]
            and row["objective"] == authority["expected_objective"]
            and "stage_transition:skip" in row.get("tags", [])
            and same(owner["native_claim"], {
                "attempt": 1, "backlog_row_sha256": digest(raw),
                "mission_started_event_sha256": digest(event_raw),
                "mission_started_ts": event["ts"], "owner": "primary",
                "started_ts": row["started_ts"],
                "usage_attempt_id": MISSION + ":attempt:1",
            })
            and row["started_ts"] <= event["ts"] <= owner["published_at"] <= now,
            "native owner claim differs")
    source_edit_bindings(proposal, owner)
    return proposal_pin, proposal, previous, owner, terminal[0]


def tree_state(root):
    members = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            members.append({"path": relative, "symlink": os.readlink(path)})
        elif path.is_file():
            members.append(dict(pin(path), path=relative))
        else:
            require(path.is_dir(), "unexpected predecessor member")
            members.append({"path": relative, "directory": True})
    return members


def preserved_trees(proposal, previous):
    sealed_members(proposal["accepted_construction"]["seal"])
    baseline_path = REPOSITORY / CONSTRUCTION_ROOT / "preserved-inputs.json"
    baseline = json.loads(baseline_path.read_bytes())
    require(all(tree_state(Path(path)) == members for path, members in baseline.items()),
            "immutable predecessor tree changed")
    roots = [
        Path(proposal["accepted_release"]["contract"]["path"]).parent,
        Path(proposal["failed_science"]["terminal"]["path"]).parent,
        LIFE / "handoffs" / "d15533af5bdf",
        LIFE / "handoffs" / "b7d95bd6e7f2",
        REPOSITORY / CONSTRUCTION_ROOT,
        Path(proposal["accepted_construction"]["prevalidation"]["path"]).parent.parent,
        *map(Path, baseline),
    ]
    missions = {PROVENANCE_MISSION, CONSTRUCTION_MISSION, ARTIFACT_REVIEW_MISSION,
                FAILED_PREDECESSOR_MISSION,
                "ab65cb0e6bea", "445f400d92c4",
                "f5fb1a5b3487", "25482bb8477b",
                "9d53694fd875", "6a7d8ff8feea"}
    roots += [path for path in sorted((REPOSITORY / "build").iterdir())
              if path.is_dir() and any(mission in path.name for mission in missions)]
    roots += [LIFE / "handoffs" / mission for mission in sorted(missions)]
    roots += [path for path in sorted((REPOSITORY / ".argus/live/manager-authority").iterdir())
              if path.is_dir() and any(mission in path.name for mission in missions)]
    require(all(root.is_dir() and root.resolve() == root for root in roots),
            "immutable predecessor root missing or rebound")
    return {str(root): tree_state(root) for root in roots}


def focused_command(scratch, current):
    """Check only the new owner/binding bytes; ed9 retains the repair coverage."""
    release_tests = [
        "test_fresh_proposed_contract_consumes_production",
        "test_actual_successor_v6_owner_and_source_bindings",
        "test_successor_rejects_obsolete_owner_schema",
        "test_successor_rejects_owner_runtime_drift",
        "test_source_binding_requires_baseline_or_postclaim_bytes",
        "test_reviewed_candidate_capture_cannot_be_edited",
        "test_reviewed_repair_rejects_changed_evidence",
        "test_preseal_verdict_matches_real_runtime_classification",
        "test_preseal_rejects_rebound_review",
        "test_authenticated_validation_receipt_requires_exact_evidence",
        "test_validation_receipt_requires_release_binding_coverage",
        "test_validation_receipt_rejects_repinned_bindings",
        "test_validation_receipt_rejects_changed_command",
    ]
    return [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            *[str(TEST) + "::" + name for name in release_tests],
            "--durations=10", "-o", "junit_family=legacy", "--basetemp", str(scratch / "pytest-tmp"),
            "--junitxml", str(scratch / "focused-tests.xml")]


def test_summary(raw, stdout, *, require_candidate=True):
    xml = ET.fromstring(raw)
    suites = xml.findall(".//testsuite")
    cases = xml.findall(".//testcase")
    require(suites and sum(int(s.attrib["tests"]) for s in suites) == len(cases) > 0
            and all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
            and not any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped"))
            and re.search(rb"(?:^|\n)" + str(len(cases)).encode()
                          + rb" passed in [0-9.]+s(?: \([0-9]+:[0-9]{2}:[0-9]{2}\))?\s*$", stdout),
            "prevalidation requires all collected tests passed, zero failures, errors, or skips")
    forbidden = []
    for case in cases:
        if "test_diagnose_" in case.attrib["classname"]:
            properties = [p.attrib for p in case.findall("./properties/property")]
            require({p["name"] for p in properties} == {
                "forbidden_" + kind for kind in ZERO_KINDS
            } and len(properties) == len(ZERO_KINDS) and all(p["value"] == "0" for p in properties),
                "candidate authority test lacks exact zero-invocation evidence")
            forbidden.extend(properties)
    require(forbidden or not require_candidate, "candidate authority coverage missing")
    return forbidden


def prevalidate():
    require(Path.cwd() == REPOSITORY and sys.dont_write_bytecode and not sys.flags.optimize,
            "wrong prevalidation interpreter or workdir")
    proposal_pin, proposal, previous, owner, terminal = successor_inputs()
    root = REPOSITORY / ROOT
    require(not root.exists() and not root.is_symlink(), "delivery root must remain absent")
    preserved = preserved_trees(proposal, previous)
    runtime = observe_runtime()
    validation = REPOSITORY / VALIDATION_ROOT
    require(validation.resolve() == validation and not validation.is_symlink(),
            "noncanonical validation storage")
    validation.mkdir(parents=True, exist_ok=True)
    attempts = sorted(validation.glob("attempt-*"))
    require([p.name for p in attempts] == [
        f"attempt-{i:03d}" for i in range(1, len(attempts) + 1)], "validation attempt gap")
    scratch = validation / f"attempt-{len(attempts) + 1:03d}"
    scratch.mkdir()
    started = time.time()
    sources = implementation_pins()
    receipt = {
        "mission_id": MISSION, "status": "FAIL", "manager_proposal": proposal_pin,
        "owner_sha256": CLAIM_SHA256, "started_at": started, "sources": sources,
        "terminal_609": terminal, "preserved_trees": preserved, "commands": [],
        "audit_limit": AUDIT_LIMIT, "repository_repair": proposal["repository_repair"],
    }
    exclusive(scratch / "prevalidation.command.txt", (shlex.join(sys.orig_argv) + "\n").encode())
    write_json(scratch / "prevalidation.argv.json", sys.orig_argv)
    try:
        for index, ((_, _, name), record) in enumerate(zip(SOURCE_LAYOUT, sources, strict=True)):
            raw = authenticate(record)
            if index >= 2:
                snapshot = authenticate(proposal["source_baseline"][SOURCE_LAYOUT[index][0]])
                require(raw == snapshot, "candidate/capture changed before snapshot copy")
                raw = snapshot
            exclusive(scratch / name, raw)
        reviewed_root = Path(proposal["repository_repair"]["proposed_bytes"]["path"]).parent
        differences = []
        for (_, path, name), source in zip(SOURCE_LAYOUT, sources, strict=True):
            differences.extend(difflib.unified_diff(
                (reviewed_root / name).read_text().splitlines(keepends=True),
                authenticate(source).decode().splitlines(keepends=True),
                fromfile="reviewed-ed9/" + str(path), tofile="proposed/" + str(path)))
        exclusive(scratch / "source.diff", "".join(differences).encode())
        contract = proposed_contract(previous, proposal_pin, scratch, started, runtime)
        write_json(scratch / "current-runtime-contract.json", contract)
        for name, argv in validation_commands(scratch, sources):
            result = run_validation_command(scratch, name, argv)
            receipt["commands"].append(result)
            require(result["exit_code"] == 0, name + " failed; delivery root left absent")
        test_summary((scratch / "focused-tests.xml").read_bytes(),
                     (scratch / "focused-tests.stdout").read_bytes(), require_candidate=False)
        require(preserved_trees(proposal, previous) == preserved and not root.exists(),
                "prevalidation altered roots")
        for record in sources:
            authenticate(record)
        receipt["status"] = "PASS"
    except (ValueError, OSError, ET.ParseError) as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["finished_at"] = time.time()
        receipt["members"] = [pin(p) for p in sorted(scratch.iterdir()) if p.is_file()]
        write_json(scratch / "prevalidation.json", receipt)
        print(json.dumps({"status": receipt["status"],
                          "prevalidation": pin(scratch / "prevalidation.json")}))


def validation_commands(scratch, sources):
    return [
        ("py_compile", [
            sys.executable, "-P", "-B", "-c",
            "import py_compile,sys; "
            "[py_compile.compile(p,cfile=sys.argv[1]+'/compile-'+str(i)+'.pyc',doraise=True) "
            "for i,p in enumerate(sys.argv[2:])]",
            str(scratch), *[r["path"] for r in sources],
        ]),
        ("focused-tests", focused_command(scratch, {"tests": sources[3]})),
    ]


def validation_environment(scratch):
    return {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(scratch), "TMPDIR": str(scratch), "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(REPOSITORY), "STAGE11_TEST_ROOT": str(scratch / "pytest-tmp"),
        "STAGE11_PREVALIDATION_ROOT": str(scratch),
    }


def run_validation_command(scratch, name, argv):
    environment = validation_environment(scratch)
    exclusive(scratch / (name + ".command.txt"), (shlex.join(argv) + "\n").encode())
    write_json(scratch / (name + ".argv.json"), argv)
    write_json(scratch / (name + ".environment.json"), environment)
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(argv, cwd=REPOSITORY, env=environment,
                                   capture_output=True, timeout=900, check=False)
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b"", error.stderr or b"", 124
        timed_out = True
    exclusive(scratch / (name + ".stdout"), stdout)
    exclusive(scratch / (name + ".stderr"), stderr)
    summary = None
    if name == "focused-tests" and (scratch / "focused-tests.xml").is_file():
        suites = ET.fromstring((scratch / "focused-tests.xml").read_bytes()).findall(".//testsuite")
        summary = {key: sum(int(s.attrib[key]) for s in suites)
                   for key in ("tests", "failures", "errors", "skipped")}
    result = {"name": name, "argv": argv, "environment": environment, "cwd": str(REPOSITORY),
              "exit_code": code, "timed_out": timed_out,
              "started_at": started, "finished_at": time.time(),
              "stdout": pin(scratch / (name + ".stdout")),
              "stderr": pin(scratch / (name + ".stderr")), "test_summary": summary}
    write_json(scratch / (name + ".receipt.json"), result)
    print(stdout.decode(), end="")
    print(stderr.decode(), end="", file=sys.stderr)
    return result


def authenticate_prevalidation(record, source_pins, owner, proposal, terminal):
    evidence = document(record)
    scratch = Path(record["path"]).parent
    require(scratch.parent == REPOSITORY / VALIDATION_ROOT
            and re.fullmatch(r"attempt-[0-9]{3}", scratch.name)
            and record["path"] == str(scratch / "prevalidation.json")
            and scratch.stat().st_ctime_ns >= owner["published_at"] * 1e9
            and evidence["mission_id"] == MISSION and evidence["status"] == "PASS"
            and evidence["owner_sha256"] == CLAIM_SHA256
            and same(evidence["manager_proposal"], CONSUMPTION_INPUT)
            and same(evidence["repository_repair"], proposal["repository_repair"])
            and owner["published_at"] < evidence["started_at"] <= evidence["finished_at"] <= time.time()
            and same(evidence["sources"], source_pins) and same(evidence["terminal_609"], terminal),
            "prevalidation claim, timing, or source binding differs")
    members = {}
    for member in evidence["members"]:
        path = Path(member["path"])
        require(path.parent == scratch and path.name not in members, "invalid prevalidation member")
        members[path.name] = authenticate(member)
    commands = validation_commands(scratch, source_pins)
    require(len(evidence["commands"]) == len(commands), "required validation command missing")
    for (name, argv), result in zip(commands, evidence["commands"]):
        require(same(result, json.loads(members[name + ".receipt.json"]))
                and result["name"] == name and same(result["argv"], argv)
                and result["cwd"] == str(REPOSITORY)
                and type(result["exit_code"]) is int and result["exit_code"] == 0
                and result["timed_out"] is False
                and evidence["started_at"] <= result["started_at"]
                <= result["finished_at"] <= evidence["finished_at"]
                and same(result["environment"], validation_environment(scratch))
                and same(json.loads(members[name + ".environment.json"]), result["environment"])
                and same(json.loads(members[name + ".argv.json"]), argv)
                and members[name + ".command.txt"] == (shlex.join(argv) + "\n").encode()
                and same(result["stdout"], pin_bytes(scratch / (name + ".stdout"), members[name + ".stdout"]))
                and same(result["stderr"], pin_bytes(scratch / (name + ".stderr"), members[name + ".stderr"])),
                "compile/test command, environment, timing, or result differs")
    require(all("compile-" + str(i) + ".pyc" in members for i in range(len(source_pins))),
            "compiled source census differs")
    for (_, _, name), source in zip(SOURCE_LAYOUT, source_pins, strict=True):
        require(members[name] == authenticate(source), "validation snapshot differs")
    test_summary(members["focused-tests.xml"], members["focused-tests.stdout"],
                 require_candidate=False)
    forbidden = reviewed_repair_evidence(proposal)
    cases = ET.fromstring(members["focused-tests.xml"]).findall(".//testcase")
    require(any(c.attrib["name"] == "test_fresh_proposed_contract_consumes_production" for c in cases)
            and any(c.attrib["name"] == "test_actual_successor_v6_owner_and_source_bindings"
                    for c in cases)
            and all(any(c.attrib["name"] == (
                "test_preseal_verdict_matches_real_runtime_classification[" + status + "]")
                for c in cases) for status in ("continue", "replan_requested")),
            "required release-binding validation coverage missing")
    summary = evidence["commands"][1]["test_summary"]
    require(same(summary, {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}),
            "validation test summary differs")
    previous = document(proposal["accepted_construction"]["contract"])
    require(preserved_trees(proposal, previous) == evidence["preserved_trees"],
            "immutable predecessor tree changed")
    return evidence, members, forbidden


def authenticate_preseal_review(prevalidation, source_pins):
    base = LIFE / "handoffs" / MISSION
    latest = document(pin(base / "latest.json"))
    require(latest["kind"] == "handoff_ref"
            and latest["mission"] == {"path": str(base / "mission.json")},
            "independent preseal review required")
    review_pin = pin(Path(latest["handoff"]["path"]))
    review = document(review_pin)
    for record in source_pins:
        authenticate(record)
    require(review["kind"] == "round_reviewed_handoff" and review["schema_version"] == 3
            and review["mission_id"] == MISSION and review["producer_role"] == "reviewer"
            and type(review["round"]) is int and review["round"] > 0
            and review_pin["path"] == str(base / f"round-{review['round']:04d}.json")
            and review["mission_context"] == str(base / "mission.json")
            and review["review"]["status"] == "continue"
            and review["review"]["next_action"] == "BUILD_FINAL_FROM_REVIEWED_BYTES"
            and prevalidation["finished_at"] <= review["created_at"] <= time.time()
            and all(Path(record["path"]).stat().st_mtime <= review["created_at"]
                    for record in source_pins),
            "independent preseal review of unchanged validated bytes required")
    return review_pin


def build(prevalidation_pin):
    require(Path.cwd() == REPOSITORY and sys.dont_write_bytecode
            and not sys.flags.optimize, "wrong workdir or interpreter flags")
    proposal_pin, proposal, previous, owner, terminal = successor_inputs()
    source_pins = implementation_pins()
    root = REPOSITORY / ROOT
    require(root.resolve() == root and not root.exists() and not root.is_symlink(),
            "sole fresh root already exists; no overwrite or retry")
    prevalidation, members, forbidden = authenticate_prevalidation(
        prevalidation_pin, source_pins, owner, proposal, terminal)
    review_pin = authenticate_preseal_review(prevalidation, source_pins)
    validation = REPOSITORY / VALIDATION_ROOT
    attempts = sorted(validation.glob("attempt-*"))
    require([p.name for p in attempts] == [
        f"attempt-{i:03d}" for i in range(1, len(attempts) + 1)]
        and attempts[-1] == Path(prevalidation_pin["path"]).parent,
        "final validation must be the latest retained attempt")
    retained = {}
    for attempt in attempts:
        receipt_pin = pin(attempt / "prevalidation.json")
        receipt = document(receipt_pin)
        require(receipt["mission_id"] == MISSION and receipt["owner_sha256"] == CLAIM_SHA256
                and same(receipt["manager_proposal"], proposal_pin)
                and owner["published_at"] < receipt["started_at"] <= receipt["finished_at"]
                <= prevalidation["finished_at"], "retained validation claim/timing differs")
        retained[str(attempt.name + "/prevalidation.json")] = authenticate(receipt_pin)
        for member in receipt["members"]:
            path = Path(member["path"])
            require(path.parent == attempt and path.name != "prevalidation.json",
                    "retained validation member escaped its attempt")
            retained[attempt.name + "/" + path.name] = authenticate(member)
    scratch = Path(prevalidation_pin["path"]).parent
    contract = proposed_contract(previous, proposal_pin, scratch,
                                 prevalidation["started_at"], observe_runtime())
    require(same(contract, json.loads(members["current-runtime-contract.json"])),
            "final contract differs from validated bytes or runtime")
    accepted_lineage(contract["accepted_lineage"], contract)
    now = time.time()
    contract["created_at"] = now
    for role, name in (("current_diagnostic", "candidate-source.py"),
                       ("current_tests", "candidate-tests.py")):
        contract["source_snapshots"][role] = pin_bytes(root / name, members[name])
    for key, name in (("capture_helper", "capture-source.py"),
                      ("capture_tests", "capture-tests.py")):
        contract["capture_snapshots"][key] = pin_bytes(root / name, members[name])
    authority_path = REPOSITORY / ".argus/live/manager-authority" / AUTHORITY / "build-authorization.json"
    owner_path = authority_path.with_name("manager-owner-claim.json")
    authority = json.loads(authority_path.read_bytes())
    payload = {
        "current-runtime-contract.json": json_bytes(contract),
        "build-authorization.json": authority_path.read_bytes(),
        "manager-owner-claim.json": authenticate(dict(pin(owner_path), sha256=CLAIM_SHA256)),
        "manager-input.json": authenticate(proposal_pin),
        "preseal-review.json": authenticate(review_pin),
        "manager-consumption-decision.json": authenticate(CONSUMPTION_DECISION),
        "post-review-manager-owner.json": json_bytes(authority["post_review_owner"]),
        "claim.command.txt": (CLAIM_COMMAND + "\n").encode(),
        "build.command.txt": (shlex.join(sys.orig_argv) + "\n").encode(),
        "build.argv.json": json_bytes(sys.orig_argv),
        **{name: members[name] for _, _, name in SOURCE_LAYOUT},
        **{"validation/" + name: raw for name, raw in retained.items()},
    }
    successor_inputs()
    for record in source_pins:
        authenticate(record)
    root.mkdir()
    audit = {"writes": [], "blocked": []}

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
        elif name in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "socket.connect"):
            audit["blocked"].append(name)
            raise PermissionError("dispatch forbidden in zero-science builder")
        elif name == "import" and (".candidates." in args[0]
                                  or args[0].split(".")[0] in ("numpy", "torch", "transformers")):
            audit["blocked"].append(name)
            raise PermissionError("candidate imports forbidden")

    sys.addaudithook(audit_hook)
    for name, raw in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        exclusive(path, raw)
    row, event, _, _ = native_claim(LIFE, MISSION, time.time())
    write_json(root / "construction-receipt.json", {
        "mission_id": MISSION, "authority": pin(authority_path),
        "manager_owner_claim": pin(owner_path),
        "claim_predates_root": True, "source_edit_bindings": source_edit_bindings(proposal, owner),
        "runtime": contract["runtime"],
        "native_record": row, "mission_started_event": event, "created_at": now,
        "consumption": "AUTHENTICATED_JSON_DATA_ONLY", "executing_role": "engineer",
        "prevalidation": prevalidation_pin,
        "retained_candidate_authority_evidence": proposal["repository_repair"]["focused_tests_xml"],
        "retained_prevalidation": pin(root / "validation" / scratch.name / "prevalidation.json"),
        "prevalidation_completed_before_root": prevalidation["finished_at"] <= now,
        "sources": source_pins, "builder_invocations": 1,
        "provenance_is_not_implementation_acceptance": True,
    })
    write_json(root / "zero-invocations.json", {
        "candidate_authority_test_properties": forbidden,
        "candidate_check_invocations": 0, "scientific_invocations": 0,
        "model_invocations": 0, "runtime_loads": 0, "array_loads": 0,
        "producer_invocations": 0, "service_invocations": 0,
        "live_emitter_invocations": 0,
    })
    require(preserved_trees(proposal, previous) == prevalidation["preserved_trees"],
            "immutable predecessor drift")
    successor_inputs()
    require(same(observe_runtime(), contract["runtime"]), "runtime drift during construction")
    write_json(root / "audit-disclosure.json", {
        "scope": "Python builder audit hooks and prevalidation test guards; command receipts and member census",
        "not_claimed": "OS-wide write confinement, sandbox equivalence, historical closure, or scientific results",
        "source_edits": "Each source matches its Manager baseline or its modification time postdates the claim",
        "claim_command_sidecar": "Exact operator command recorded after binder returned its authenticated receipt",
        "runtime_limit": "PID/start/cmdline and startup fingerprint record identity, not an OS attestation",
        "independent_acceptance_required": AUDIT_LIMIT, "builder_audit": audit,
    })
    write_json(root / "preserved-inputs.json", prevalidation["preserved_trees"])
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
    write_json(root / "build-receipt.json", {"command": shlex.join(sys.orig_argv), "exit_status": 0,
                                           "builder_invocations": 1, "validation_reruns": 0})
    census = [dict(pin(p), path=str(p.relative_to(root)))
              for p in sorted(root.rglob("*")) if p.is_file()]
    write_json(root / "seal.json", {
        "schema_version": 1, "mission_id": MISSION,
        "status": "PROPOSED_PENDING_INDEPENDENT_REVIEW",
        "members": census, "excludes": ["seal.json"], "scientific_run_budget": 0,
    })
    print(result, end="")


if __name__ == "__main__":
    if sys.argv[1:] == ["--prevalidate"]:
        prevalidate()
    else:
        require(len(sys.argv) == 4 and sys.argv[1] == "--build",
                "only zero-science --prevalidate or --build PATH SHA256 is exposed")
        evidence_path = Path(sys.argv[2])
        build(dict(pin(evidence_path), sha256=sys.argv[3]))
