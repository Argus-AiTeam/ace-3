"""Seal the authorized zero-science Stage11 handoff blocker, never a launch.

Engineer entry: python3 -B ace3/model/stage11_forward_handoff_repair.py
The sole root must not exist. Reviewer reads the resulting seal and receipts;
Reviewer must not run this entry, tests, an observer, or a candidate.

The native Manager claim must precede source edits and root creation. The pinned
Manager preflight report and its retained source are JSON data, never executable
inputs. No frozen approval or historical failure closure is inherited.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
MISSION = "33543e0d3d9f"
AUTHORITY = "s11-forward-handoff-dataonly-20260922t085606z-33543e0d3d9f-r1"
ROOT = "build/argus-stage11-forward-runtime-dataonly-handoff-33543e0d3d9f-attempt001"
AUTHORITY_SHA256 = "52e1bf642c23c91987aa478c75deb81510491c7acb506b69aae4f216eb8a5afe"
CLAIM_SHA256 = "8806b747de6ac54f4a6a7aaa956116d87f164d10f43cd3d05224cbcb450c5ffb"
TEST = "tests/test_stage11_forward_handoff_repair.py"
FROZEN_GATES = {
    "package_status", "release_review", "historical_failure_closure",
    "write_audit_limits",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate_pin(record):
    path = Path(record["path"])
    require(path.is_absolute() and path.resolve() == path, f"noncanonical pin: {path}")
    raw = path.read_bytes()
    require(
        len(raw) == record["bytes"]
        and hashlib.sha256(raw).hexdigest() == record["sha256"],
        f"input authentication failed: {path}",
    )
    return raw


def check_authority(authority, now):
    require(
        authority["schema"] == "argus.stage11-dataonly-forward-handoff-authorization.v1"
        and authority["authority_id"] == AUTHORITY
        and authority["mission_id"] == MISSION
        and authority["authorized_root"] == ROOT
        and authority["issuer"] == "manager"
        and authority["decision"] == "AUTHORIZE_ZERO_SCIENCE_DATA_ONLY_FORWARD_HANDOFF_REPAIR"
        and type(authority["expected_attempt"]) is int
        and authority["expected_attempt"] == 1,
        "ambiguous_objective: authority identity, decision, attempt, or root differs",
    )
    require(
        authority["authorized_science"] is False
        and type(authority["scientific_run_budget"]) is int
        and authority["scientific_run_budget"] == 0,
        "ambiguous_objective: science authorization is not integer zero",
    )
    require(
        datetime.datetime.fromisoformat(authority["issued_at_utc"]) <= now
        < datetime.datetime.fromisoformat(authority["valid_until_utc"]),
        "authority outside validity interval",
    )


def check_claim(owner, authority, source_mtimes_ns):
    claim = owner["native_claim"]
    require(
        owner["schema"] == "argus.manager-postclaim-owner.v2"
        and owner["state"] == "BOUND_TO_NATIVE_RUNNING_CLAIM"
        and owner["authority_id"] == AUTHORITY and owner["mission_id"] == MISSION
        and owner["authorized_root"] == ROOT and owner["issuer"] == "manager"
        and owner["executing_role"] == "engineer"
        and owner["manager_inputs"] == authority["manager_inputs"]
        and claim["status"] == "running" and type(claim["attempt"]) is int
        and claim["attempt"] == authority["expected_attempt"]
        and claim["node_key"] == "stage11-dataonly-handoff-correction"
        and claim["independent_review_required"] is True
        and owner["science_issuance"] is None
        and type(owner["scientific_run_budget"]) is int
        and owner["scientific_run_budget"] == 0
        and owner["preflight_status"] == "UNKNOWN"
        and owner["release_consumption_authorized"] is False
        and owner["release_consumption_admissible"] is False,
        "ambiguous_objective: Manager native claim differs",
    )
    require(
        claim["started_ts"] <= claim["mission_started_ts"] <= owner["published_at"]
        and source_mtimes_ns
        and all(owner["published_at"] * 1_000_000_000 < stamp for stamp in source_mtimes_ns),
        "Manager claim must predate source edits",
    )


def check_frozen_report(report):
    require(
        report["mission_id"] == "016efb266000"
        and report["status"] == report["package_status"] == "UNKNOWN"
        and report["blocked"] is True
        and report["release_consumption_admissible"] is False
        and report["release_consumption_authorized"] is False
        and len(report["blockers"]) == 4
        and {item["gate"] for item in report["blockers"]} == FROZEN_GATES,
        "frozen 016 evidence differs from the four authorized blockers",
    )
    require(
        all(type(report[name]) is int and report[name] == 0 for name in (
            "scientific_invocations", "candidate_check_invocations", "package_validation_invocations",
        )) and report["dispatch_attempts"] == [],
        "frozen report does not retain zero-science/non-dispatch evidence",
    )


def load_manager_preflight(authority):
    raw = authenticate_pin(authority["manager_inputs"]["preflight_report"])
    document = json.loads(raw)
    require(
        document["schema"] == "argus.manager-supplied-preflight-report.v1"
        and document["mission_id"] == MISSION and document["generated_by"] == "manager"
        and document["decision"] == "RETAIN_UNKNOWN_AS_DATA_ONLY"
        and document["status"] == "UNKNOWN",
        "ambiguous_objective: Manager preflight data differs",
    )
    retained_raw = authenticate_pin(document["source"])
    report = document["report"]
    require(json.loads(retained_raw) == report, "Manager report differs from retained JSON")
    check_frozen_report(report)
    return raw, retained_raw, report


def terminal_blocker(report):
    check_frozen_report(report)
    return {
        "code": "frozen_016_release_consumption_inadmissible",
        "reason": (
            "The native Manager claim binds this data-only handoff, not science "
            "or release consumption. Frozen 016 remains UNKNOWN with all four "
            "blockers. Current c8/7d9 identity cannot close historical failures. "
            "The rejected 558 EXACT_BLOCKER root remains superseded/replan, "
            "immutable and inadmissible."
        ),
        "frozen_016_blockers": report["blockers"],
    }


def build():
    require(Path.cwd() == REPOSITORY, "wrong Engineer workdir")
    authority_path = REPOSITORY / ".argus/live/manager-authority" / AUTHORITY / "build-authorization.json"
    raw = authority_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == AUTHORITY_SHA256, "authority bytes changed")
    authority = json.loads(raw)
    check_authority(authority, datetime.datetime.now(datetime.timezone.utc))
    authenticated = {
        name: authenticate_pin(pin)
        for name, pin in authority["authenticated_evidence"].items()
    }
    owner_path = authority_path.with_name("manager-owner-claim.json")
    owner_raw = owner_path.read_bytes()
    require(hashlib.sha256(owner_raw).hexdigest() == CLAIM_SHA256, "Manager claim bytes changed")
    owner = json.loads(owner_raw)
    source_paths = [Path(__file__), REPOSITORY / TEST]
    source_mtimes = {str(path): path.stat().st_mtime_ns for path in source_paths}
    check_claim(owner, authority, list(source_mtimes.values()))
    claim_pin = {"path": str(owner_path), "bytes": len(owner_raw), "sha256": CLAIM_SHA256}
    report_raw, retained_raw, report = load_manager_preflight(authority)
    blocker = terminal_blocker(report)
    census = json.loads(authenticated["retained_016_census"])
    for pin in census["members"]:
        authenticate_pin(pin)
    rejected_seal = json.loads(authenticated["rejected_558_seal"])
    require(
        rejected_seal["status"] == "EXACT_BLOCKER"
        and json.loads(authenticated["rejected_558_review"])["review"]["status"] == "replan_requested",
        "rejected 558 history changed",
    )
    rejected_root = Path(authority["authenticated_evidence"]["rejected_558_seal"]["path"]).parent
    for pin in rejected_seal["members"]:
        authenticate_pin(dict(pin, path=str(rejected_root / pin["path"])))
    settlement = json.loads(authenticated["observer_settlement"])
    observer_pin = settlement["execution_evidence"]["seal"]
    observer_seal = json.loads(authenticate_pin(observer_pin))
    observer_root = Path(observer_pin["path"]).parent
    for pin in observer_seal["members"]:
        authenticate_pin(dict(pin, path=str(observer_root / pin["path"])))
    process = Path("/proc") / str(authority["runtime"]["pid"])
    ticks = int((process / "stat").read_text().rsplit(")", 1)[1].split()[19])
    require(ticks == authority["runtime"]["start_ticks"], "Manager process identity changed")
    root = REPOSITORY / ROOT
    require(root.resolve() == root, "noncanonical output root")
    require(not root.exists() and not root.is_symlink(), "authorized root already exists; no retry")
    absent_ns = time.time_ns()
    require(owner["published_at"] * 1_000_000_000 < absent_ns, "claim must predate root creation")
    root.mkdir()  # Exclusive creation preserves the absent-root precondition.

    def write(name, value):
        with (root / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")

    write("construction-receipt.json", {
        "schema": "argus.stage11-forward-handoff-construction.v1",
        "mission_id": MISSION, "authority_id": AUTHORITY, "authorized_root": ROOT,
        "executing_role": "engineer", "authority_authenticated": True,
        "authority": {"path": str(authority_path), "bytes": len(raw), "sha256": AUTHORITY_SHA256},
        "manager_owner_claim": claim_pin, "claim_published_at": owner["published_at"],
        "source_mtimes_ns": source_mtimes,
        "claim_predates_source_edits_and_root": True,
        "authenticated_evidence": authority["authenticated_evidence"],
        "manager_inputs": authority["manager_inputs"],
        "root_absent_before_construction": True, "absent_observed_ns": absent_ns,
        "created_ns": time.time_ns(), "uid": os.getuid(), "pid": os.getpid(),
        "interpreter": sys.executable, "manager_pid": authority["runtime"]["pid"],
        "manager_start_ticks": ticks, "argv": sys.argv,
    })
    for name, content in (
        ("build-authorization.json", raw), ("manager-owner-claim.json", owner_raw),
        ("manager-preflight-report.json", report_raw), ("frozen-016-preflight.json", retained_raw),
        ("builder-source.py", Path(__file__).read_bytes()),
        ("focused-test-source.py", (REPOSITORY / TEST).read_bytes()),
    ):
        with (root / name).open("xb") as stream:
            stream.write(content)
    for name in ("builder-source.py", "focused-test-source.py"):
        compile((root / name).read_bytes(), name, "exec")
    write("compile-receipt.json", {
        "executing_role": "engineer", "status": "PASS",
        "mode": "in-memory syntax compile only", "candidate_imports": 0,
        "sources": ["builder-source.py", "focused-test-source.py"],
    })
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    completed = subprocess.run(
        command, cwd=REPOSITORY, env=environment, capture_output=True, timeout=60,
        check=False,
    )
    for name, content in (("focused-tests.stdout", completed.stdout), ("focused-tests.stderr", completed.stderr)):
        with (root / name).open("xb") as stream:
            stream.write(content)
    write("focused-tests-receipt.json", {
        "executing_role": "engineer", "argv": command, "exit_status": completed.returncode,
        "suite_runs": 1, "synthetic_only": True, "candidate_executions": 0,
    })
    require(completed.returncode == 0, "focused tests failed; retained output is authoritative")
    for name, pin in authority["authenticated_evidence"].items():
        require(authenticate_pin(pin) == authenticated[name], f"input changed during build: {name}")
    require(load_manager_preflight(authority)[:2] == (report_raw, retained_raw),
            "Manager preflight data changed during build")
    require(authority_path.read_bytes() == raw and owner_path.read_bytes() == owner_raw,
            "authority or Manager claim changed during build")
    for path in source_paths:
        snapshot = "builder-source.py" if path == Path(__file__) else "focused-test-source.py"
        require(path.read_bytes() == (root / snapshot).read_bytes(), "source changed during build")
    write("termination-receipt.json", {
        "schema": "argus.stage11-forward-handoff-termination.v1",
        "mission_id": MISSION, "authority_id": AUTHORITY, "executing_role": "engineer",
        "status": "EXACT_BLOCKER", "exact_blocker": blocker,
        "repair_status": "DATA_ONLY_HANDOFF_SEALED", "manager_owner_claim": claim_pin,
        "preflight_consumption": "AUTHENTICATED_JSON_DATA_ONLY",
        "manager_preflight_report": authority["manager_inputs"]["preflight_report"],
        "zero_science": True, "scientific_run_budget": 0, "scientific_invocations": 0,
        "no_candidate_executed": True, "candidate_executions": 0, "candidate_imports": 0,
        "preflight_executions": 0,
        "observer_dispatches": 0, "replays": 0, "emitter_invocations": 0,
        "manager_postclaim_emitter": "NOT_IMPLEMENTED_OR_INVOKED",
        "runtime_model_continuous_changes": 0, "stage_transitions": 0,
        "science_nodes_created": 0, "successor_nodes_created": 0,
        "non_inheriting_overlay": True, "release_consumption_authorized": False,
        "release_consumption_admissible": False,
        "preserved": {
            "frozen_016": "UNKNOWN / INADMISSIBLE; all four blockers retained",
            "558": "superseded/replan_requested/EXACT_BLOCKER; immutable, not repaired in place",
            "c8": "retained current identity evidence only; not replayed, changed, or promoted",
            "7d9": "accepted current identity evidence only; not historical closure",
            "failures_unknowns_lineage_thresholds_global_reference_operand_state_KV": "UNCHANGED",
            "precision": "Native-S16-RTZ Q24 residual remains wider than strict FP16 state",
            "authority_evidence_unchanged": True,
            "frozen_016_census_members_authenticated": True,
            "rejected_558_seal_members_authenticated": True,
            "accepted_7d9_seal_members_authenticated": True,
            "boundary": report["boundary"],
        },
        "reviewer_policy": {
            "independent_review": "REQUIRED_PENDING_HOST_REVIEWER",
            "access": "READ_ONLY", "root_expected_to_exist": True,
            "require_root_absence": False, "root_mutation_allowed": False,
            "engineer_commands_allowed": False,
            "code_or_skill_execution_allowed": False,
            "stage11-budgeted-operational-observer_selection_allowed": False,
            "reviewer_execution_in_this_engineer_run": False,
        },
        "evidence_limit": (
            "Engineer command receipts and retained-byte checks, not an OS-wide "
            "write audit or a completed independent review. No new science "
            "admission, historical closure, runtime readiness, or stage transition."
        ),
    })
    members = []
    for path in sorted(root.iterdir()):
        content = path.read_bytes()
        members.append({
            "path": path.name, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    write("seal.json", {
        "schema": "argus.stage11-forward-handoff-seal.v1", "mission_id": MISSION,
        "executing_role": "engineer", "status": "EXACT_BLOCKER",
        "members": members, "excludes": ["seal.json"],
    })
    seal = json.loads((root / "seal.json").read_bytes())
    require({p.name for p in root.iterdir()} == {m["path"] for m in members} | {"seal.json"},
            "unexpected output member")
    for member in seal["members"]:
        authenticate_pin(dict(member, path=str(root / member["path"])))
    print(json.dumps({
        "status": "EXACT_BLOCKER", "root": ROOT, "seal_verified": True,
        "exact_blocker": blocker["code"], "science": 0, "candidate_executions": 0,
        "seal_sha256": hashlib.sha256((root / "seal.json").read_bytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    build()
