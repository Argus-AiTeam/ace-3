"""Bind reviewed Stage11 interface bytes to one fresh operational release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


REPOSITORY = Path("/home/argustest/ace3-argus")
LIFE = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05")
ACCEPTED_RELEASE = "d15533af5bdf"
REPOSITORY_REPAIR = "ed9ef3afb3ad"
FAILED_SCIENCE = "b7d95bd6e7f2"
V4 = REPOSITORY / (
    ".argus/live/manager-tools/"
    "publish-stage11-continuous-release-owner-claim-v4.py"
)
_v4 = runpy.run_path(str(V4))
authenticate_runtime = _v4["authenticate_runtime"]
pin = _v4["pin"]
pinned_bytes = _v4["pinned_bytes"]
require = _v4["require"]
rows = _v4["rows"]


def _json(record: dict[str, object]) -> dict[str, object]:
    return json.loads(pinned_bytes(record))


def authenticate_accepted_release(source: dict[str, object]) -> None:
    """Use the accepted release's own immutable validator, not current MISSION."""
    accepted = source["accepted_release"]
    require(accepted["mission_id"] == ACCEPTED_RELEASE, "wrong accepted release")
    for key in (
        "selection",
        "contract",
        "seal",
        "latest",
        "review",
        "checkpoint",
        "mission",
        "release_source",
    ):
        pinned_bytes(accepted[key])
    historical = runpy.run_path(str(accepted["release_source"]["path"]))
    selection = _json(accepted["selection"])
    selected = selection["selected"]
    review = historical["native_acceptance"](
        selected,
        {
            "mission_id": ACCEPTED_RELEASE,
            "latest": accepted["latest"],
            "review": accepted["review"],
            "checkpoint": accepted["checkpoint"],
        },
        Path(accepted["contract"]["path"]).parent.relative_to(REPOSITORY),
        selection_issued_at=historical["timestamp"](selection["issued_at_utc"]),
    )
    require(
        selected["mission_id"] == ACCEPTED_RELEASE
        and selected["contract"] == accepted["contract"]
        and selected["seal"] == accepted["seal"]
        and selected["latest"] == accepted["latest"]
        and selected["review"] == accepted["review"]
        and selected["checkpoint"] == accepted["checkpoint"]
        and selected["mission"] == accepted["mission"]
        and selection["status"] == "SELECTED_AFTER_NORMAL_REVIEW_AND_NATIVE_DONE"
        and selection["scientific_run_budget"] == 0
        and review["review"]["status"] == "done",
        "accepted d155 selection differs",
    )


def authenticate_repository_repair(source: dict[str, object]) -> None:
    repair = source["repository_repair"]
    require(repair["mission_id"] == REPOSITORY_REPAIR, "wrong repository repair")
    for key in (
        "latest",
        "review",
        "checkpoint",
        "mission",
        "proposed_bytes",
        "compile_receipt",
        "focused_tests_receipt",
        "focused_tests_xml",
    ):
        pinned_bytes(repair[key])
    review = _json(repair["review"])
    latest = _json(repair["latest"])
    mission = _json(repair["mission"])
    proposed = _json(repair["proposed_bytes"])
    compile_receipt = _json(repair["compile_receipt"])
    tests = _json(repair["focused_tests_receipt"])
    require(
        review["mission_id"] == mission["mission_id"] == REPOSITORY_REPAIR
        and review["producer_role"] == "reviewer"
        and review["round"] == 2
        and review["review"]
        == {
            "status": "done",
            "reason": "requested outcome is materially complete",
            "operator_question": "",
            "next_action": "",
        }
        and latest["handoff"] == {"path": repair["review"]["path"]}
        and proposed["status"] == "PROPOSED_PENDING_INDEPENDENT_REVIEW"
        and proposed["owner_authority_claimed"] is False
        and proposed["final_root_built"] is False
        and compile_receipt["exit_code"] == 0
        and compile_receipt["timed_out"] is False
        and tests["exit_code"] == 0
        and tests["timed_out"] is False
        and tests["test_summary"]
        == {"tests": 229, "failures": 0, "errors": 0, "skipped": 0},
        "ed9 review or retained validation differs",
    )
    expected_sources = source["source_baseline"]
    require(
        len(proposed["sources"]) == len(expected_sources) == 6,
        "ed9 six-source census differs",
    )
    for record in proposed["sources"]:
        pinned_bytes(record)
    require(
        {record["path"]: record for record in proposed["sources"]}
        == {record["path"]: record for record in expected_sources.values()},
        "current source bytes differ from reviewed ed9 package",
    )
    backlog = rows(LIFE / "backlog.jsonl", lambda row: row.get("id") == REPOSITORY_REPAIR)
    completed = rows(
        LIFE / "events.jsonl",
        lambda row: row.get("type") == "life.mission.completed"
        and row.get("item_id") == REPOSITORY_REPAIR,
    )
    require(len(backlog) == len(completed) == 1, "ed9 native terminal missing or duplicate")
    row, _ = backlog[0]
    event, event_raw = completed[0]
    # Backlog metadata is mutable; the pinned completion event binds native DONE.
    require(
        row["status"] == event["status"] == "done"
        and row["attempt"] == 1
        and row["outcome"]["review_status"] == "done"
        and event["final_review_source"] == "reviewer"
        and event["final_review_status"] == "done"
        and review["created_at"] <= event["ts"]
        and hashlib.sha256(event_raw).hexdigest()
        == repair["mission_completed_event_sha256"],
        "ed9 native DONE evidence differs",
    )


def authenticate_failed_science(source: dict[str, object]) -> None:
    failed = source["failed_science"]
    require(failed["mission_id"] == FAILED_SCIENCE, "wrong failed science mission")
    for key in ("terminal", "stdout", "review", "mission"):
        pinned_bytes(failed[key])
    terminal = _json(failed["terminal"])
    output = _json(failed["stdout"])
    review = _json(failed["review"])
    require(
        terminal["status"] == "UNKNOWN"
        and terminal["scientific_check_invocations"] == 0
        and output["status"] == "UNKNOWN"
        and output["error_type"] == "KeyError"
        and output["error"] == "'runtime_sources'"
        and output["dispatch_and_write_audit"]["scientific_invocations"] == 0
        and review["producer_role"] == "reviewer"
        and review["review"]["status"] == "done",
        "b7d zero-dispatch UNKNOWN differs",
    )


def publish(authority_dir: Path) -> dict[str, object]:
    authority_dir = authority_dir.resolve()
    authority_path = authority_dir / "build-authorization.json"
    prepared_path = authority_dir / "manager-owner-prepared.json"
    decision_path = authority_dir / "manager-consumption-decision.json"
    output = authority_dir / "manager-owner-claim.json"
    authority = json.loads(authority_path.read_bytes())
    prepared = json.loads(prepared_path.read_bytes())
    decision = json.loads(decision_path.read_bytes())
    source = _json(authority["manager_inputs"]["source_input"])
    mission = str(authority["mission_id"])
    root = REPOSITORY / str(authority["authorized_root"])
    require(Path.cwd().resolve() == REPOSITORY, "wrong workdir")
    require(sys.dont_write_bytecode, "publisher requires -B")
    require(not root.exists() and not root.is_symlink(), "fresh root already exists")
    require(
        pinned_bytes(authority["manager_inputs"]["claim_binder"])
        == Path(__file__).read_bytes()
        and pinned_bytes(authority["manager_inputs"]["consumption_decision"])
        == decision_path.read_bytes(),
        "Manager inputs changed",
    )
    require(
        source["mission_id"] == mission
        and authority["issuer"] == "manager"
        and authority["expected_attempt"] == 1
        and authority["authorized_science"] is False
        and authority["scientific_run_budget"] == 0
        and authority["scope"]["stage_transition"] is False
        and authority["scope"]["independent_review_required"] is True
        and authority["scope"]["preseal_independent_review_required"] is True
        and decision["decision"] == "BUILD_REVIEWED_INTERFACE_RELEASE"
        and decision["accepted_release"] == ACCEPTED_RELEASE
        and decision["repository_repair"] == REPOSITORY_REPAIR
        and decision["failed_science"] == FAILED_SCIENCE
        and decision["scientific_run_budget"] == 0,
        "reviewed-interface release authority differs",
    )
    require(
        prepared["authority_id"] == authority["authority_id"]
        and prepared["mission_id"] == mission
        and prepared["state"] == "PREPARED_BEFORE_NATIVE_HANDOFF"
        and pinned_bytes(prepared["build_authorization"]) == authority_path.read_bytes(),
        "prepared owner differs",
    )
    authenticate_accepted_release(source)
    authenticate_repository_repair(source)
    authenticate_failed_science(source)
    runtime = authenticate_runtime(source, LIFE)
    claims = rows(LIFE / "backlog.jsonl", lambda row: row.get("id") == mission)
    starts = rows(
        LIFE / "events.jsonl",
        lambda row: row.get("type") == "life.mission.started"
        and row.get("item_id") == mission
        and row.get("attempt") == 1,
    )
    require(len(claims) == len(starts) == 1, "exact native running claim is absent")
    claim, claim_raw = claims[0]
    started, started_raw = starts[0]
    tags = {str(tag).strip().lower().replace("-", "_") for tag in claim["tags"]}
    require(
        claim["status"] == "running"
        and claim["running_owner"] == "primary"
        and claim["attempt"] == 1
        and claim["objective"] == authority["expected_objective"]
        and {"review:required", "stage_transition:skip"} <= tags
        and started["independent_review_required"] is True
        and started["usage_attempt_id"] == f"{mission}:attempt:1"
        and claim["started_ts"] <= started["ts"] <= time.time(),
        "native running claim or role/stage scope differs",
    )
    receipt = {
        "schema": "argus.manager-stage11-reviewed-interface-release-owner.v6",
        "issuer": "manager",
        "authority_id": authority["authority_id"],
        "mission_id": mission,
        "authorized_root": authority["authorized_root"],
        "state": "BOUND_TO_NATIVE_RUNNING_CLAIM",
        "published_at": time.time(),
        "scientific_run_budget": 0,
        "accepted_release_mission": ACCEPTED_RELEASE,
        "accepted_repository_repair_mission": REPOSITORY_REPAIR,
        "failed_science_mission": FAILED_SCIENCE,
        "runtime": runtime,
        "reviewed_proposed_bytes": source["repository_repair"]["proposed_bytes"],
        "native_claim": {
            "attempt": claim["attempt"],
            "owner": claim["running_owner"],
            "started_ts": claim["started_ts"],
            "backlog_row_sha256": hashlib.sha256(claim_raw).hexdigest(),
            "mission_started_ts": started["ts"],
            "mission_started_event_sha256": hashlib.sha256(started_raw).hexdigest(),
            "usage_attempt_id": started["usage_attempt_id"],
        },
    }
    raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "status": receipt["state"],
        "mission_id": mission,
        "receipt": pin(output),
        "scientific_invocations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.authority_dir), sort_keys=True))


if __name__ == "__main__":
    main()
