"""Read-only consumption preflight for the frozen 016 operational package.

Run this file with ``python -B``; stdout is the report, exit 2 means blocked.
Only retained bytes are read. No package module, validator, capture helper,
runtime, model or service is imported or executed.

The defaults bind the census recorded in the 016 Host checkpoint and its sealed
round-1 Reviewer handoff. Alternate pins are for independently trusted evidence
(including synthetic tests), not proposer-selected replacements for that trust
root. A Reviewer ``done`` for an UNKNOWN package is not release approval.

Prospective positive evidence must additionally bind ``release_consumption`` in
the reviewed handoff: decision ADMISSIBLE, package_census, historical_failure_ids,
and write_audit_limits. The latter explicitly accepts the Python-only audit
scope and enumerates the five unobserved finalization writes. CLOSED historical
evidence contains six distinct failure_id/failure/closure records, with pinned
original FAIL receipts and PASS closures naming the original pin and test_node.
This is a consumption prerequisite, never Manager issuance or science authority.
No existing frozen evidence is upgraded, repaired, or rewritten by this module.
"""

import argparse
import hashlib
import json
from pathlib import Path


MISSION_ID = "016efb266000"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = (
    "build/argus-stage11-authorized-execution-root-repair-016efb266000-attempt001"
)
PACKAGE_ROOT = REPOSITORY_ROOT / PACKAGE_RELATIVE
FROZEN_CENSUS_PIN = {
    "path": str(PACKAGE_ROOT / "final-census.json"),
    "bytes": 14455,
    "sha256": "098a21e27b37e6133a363002586b9d08e98e2bd5b35a80c9cd865dec3e70bb03",
}
FROZEN_REVIEW_PIN = {
    "path": (
        "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
        "016efb266000/round-0001.json"
    ),
    "bytes": 690,
    "sha256": "9a309d14cf1f531746f7cb70c9387ac0439ab535c159d6719cea3cf81f4de3af",
}
PHASES = ("compile", "tests", "validate")
AUDITS = ("builder",) + PHASES
FINALIZATION_WRITES = tuple(f"{name}.write-audit.json" for name in AUDITS) + (
    "final-census.json",
)
AUDIT_SCOPE = "Python mutation audit only; no OS-wide completeness claim"
BUILDER_AUDIT_SCOPE = (
    "Python audit-hook mutations in builder; child audits retained separately; "
    "not OS-wide write observation or a sandbox"
)
SOURCE_ROLES = {
    "reviewed_30a_repository_candidate", "reviewed_30a_repository_capture_helper",
    "reviewed_57_operational_candidate", "reviewed_57_operational_capture_helper",
}
CAPTURE_FILES = (
    "argv.json", "command.txt", "environment.json", "launcher.capture.json",
    "launcher.identity.json", "launcher.stderr", "launcher.stdout",
    "launcher.whole-command.log", "verification.json",
)
BOUNDARY = (
    "Read-only retained operational evidence, not release or science authority. "
    "No validation rerun or dispatch. Python-only write audits are not OS-wide. "
    "Native-S16-RTZ Q24 residuals remain wider than FP16; INT4 weights and FP16 "
    "operator/KV boundaries are unchanged. No strict-FP16-state W4A16, new-token "
    "or full-model admission. Later consumption requires a separate Manager decision."
)


class UnavailableEvidence(ValueError):
    """An explicit missing, inconsistent, or inadmissible evidence binding."""


def _require(condition, reason):
    if not condition:
        raise UnavailableEvidence(reason)


def _object(value):
    _require(isinstance(value, dict), "expected an evidence object")
    return value


def _fields(value, required, optional=()):
    value = _object(value)
    required = set(required.split())
    allowed = required | set(optional.split() if isinstance(optional, str) else optional)
    _require(required <= value.keys(), f"missing fields: {sorted(required - value.keys())}")
    _require(value.keys() <= allowed, f"unknown fields: {sorted(value.keys() - allowed)}")
    return value


def _list(value):
    _require(isinstance(value, list), "expected an explicit evidence list")
    return value


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _zero(value, label):
    _require(type(value) is int and value == 0, f"{label} must be integer zero")


def _path(value, base):
    _require(isinstance(value, str) and bool(value), "missing evidence path")
    path = Path(value)
    _require(".." not in path.parts, "parent traversal in evidence path")
    return path if path.is_absolute() else base / path


def _read_bytes(path):
    _require(path.resolve() == path, f"noncanonical or symlink evidence path: {path}")
    _require(path.is_file(), f"missing regular evidence file: {path}")
    return path.read_bytes()


def _inventory(root):
    return {path for path in root.rglob("*") if path.is_file() or path.is_symlink()}


def _pin(pin, base, cache):
    _fields(pin, "path bytes sha256", "role")
    path = _path(pin["path"], base)
    _require(type(pin["bytes"]) is int and pin["bytes"] >= 0, "invalid pin byte count")
    digest = pin["sha256"]
    _require(
        isinstance(digest, str) and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "invalid SHA-256 pin",
    )
    if path not in cache:
        cache[path] = _read_bytes(path)
    raw = cache[path]
    _require(
        len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == digest,
        f"changed bytes or pin: {path}",
    )
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(raw):
    def reject_constant(value):
        raise UnavailableEvidence(f"nonfinite JSON constant: {value}")

    return _object(json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=reject_constant,
    ))


def _all_pins(value, base, cache):
    if isinstance(value, dict):
        if "sha256" in value or ("path" in value and "bytes" in value):
            _pin(value, base, cache)
        else:
            for child in value.values():
                _all_pins(child, base, cache)
    elif isinstance(value, list):
        for child in value:
            _all_pins(child, base, cache)


def preflight_release(
    *, package_root=PACKAGE_ROOT, repository_root=REPOSITORY_ROOT,
    census_pin=FROZEN_CENSUS_PIN, review_pin=FROZEN_REVIEW_PIN,
):
    """Authenticate retained evidence and report prerequisites, never dispatch."""
    root = Path(package_root).absolute()
    repository = Path(repository_root).absolute()
    cache = {}
    checks = {}
    blockers = []
    report = {
        "schema_version": 1, "mission_id": MISSION_ID, "status": "UNKNOWN",
        "package_status": None, "blocked": True,
        "release_consumption_admissible": False, "release_consumption_authorized": False,
        "independent_review_status": None, "checks": checks, "blockers": blockers,
        "scientific_invocations": 0, "candidate_check_invocations": 0,
        "package_validation_invocations": 0, "dispatch_attempts": [], "boundary": BOUNDARY,
    }

    def check(name, operation):
        try:
            operation()
        except (UnavailableEvidence, OSError, UnicodeError, json.JSONDecodeError) as error:
            checks[name] = False
            blockers.append({"gate": name, "reason": str(error)})
        else:
            checks[name] = True
        return checks[name]

    def document(name):
        path = root / name
        _require(path in cache, f"missing census pin: {name}")
        return _json(cache[path])

    def authenticate():
        _require(root == repository / PACKAGE_RELATIVE, "wrong package identity/root")
        path = _pin(census_pin, repository, cache)
        _require(path == root / "final-census.json", "wrong census path")
        census = _fields(
            _json(cache[path]), "mission_id members excludes review_status",
        )
        _require(census["mission_id"] == MISSION_ID, "wrong census mission")
        _require(
            census["excludes"] == ["final-census.json (its exact pin belongs in the Host checkpoint)"],
            "unexpected census exclusion",
        )
        members = set()
        for pin in _list(census["members"]):
            path = _path(_object(pin).get("path"), repository)
            _require(path.is_relative_to(root) and path != root, "census member outside package")
            _require(path not in members and path != root / "final-census.json", "duplicate census member")
            _pin(pin, repository, cache)
            members.add(path)
        _require(
            members | {root / "final-census.json"} == _inventory(root),
            "missing, extra, or unpinned package member",
        )
        required = {
            "operational-result.json", "release-inputs.json", "build-authorization.json",
            "builder.identity.json", "build.command.txt", "README.md", "focused-tests.txt",
            "operational.py", "test_operational.py", "build_release.py",
        } | {f"{name}.write-audit.json" for name in AUDITS} | {
            f"{phase}-capture/{name}" for phase in PHASES for name in CAPTURE_FILES
        }
        _require({root / name for name in required} <= members, "missing required capture/package pins")

    if not check("package_pins", authenticate):
        return report

    documents = {}

    def schemas():
        documents["inputs"] = _fields(document("release-inputs.json"), (
            "schema_version mission_id attempt authority authorization_id authorization_action "
            "scientific_run_budget cwd environment uid interpreter historical_six_gate_failures "
            "members origins review sources"
        ))
        documents["result"] = _fields(document("operational-result.json"), (
            "mission_id attempt authority candidate_check_invocations scientific_invocations "
            "package_validation_invocations commands members_unchanged sources_unchanged "
            "normal_independent_review observed_subprocesses operational_result scope"
        ))
        documents["operational"] = _fields(documents["result"]["operational_result"], (
            "actual_identity candidate_check_invocations dependency_boundary dispatch_attempts "
            "historical_six_gate_failures normal_independent_review numpy_import phase prior_phases "
            "reason release_consumption_authorized scientific_invocations status"
        ))
        _fields(documents["operational"]["prior_phases"], "compile tests")
        for phase in ("compile", "tests"):
            _fields(documents["operational"]["prior_phases"][phase], (
                "actual_identity candidate_check_invocations dependency_boundary dispatch_attempts "
                "numpy_import phase scientific_invocations status "
                + ("build compiled" if phase == "compile" else "errors failures skipped tests")
            ))

    if not check("schemas", schemas):
        return report
    inputs = documents["inputs"]
    result = documents["result"]
    operational = documents["operational"]
    phases = {**operational["prior_phases"], "validate": operational}
    report["package_status"] = operational["status"]
    review = {}

    def source_pins():
        for value in (inputs, result):
            _all_pins(value, repository, cache)
        _require(type(inputs["schema_version"]) is int and inputs["schema_version"] == 1,
                 "unsupported release schema")
        _pin(inputs["interpreter"], repository, cache)
        _require(type(inputs["uid"]) is int and inputs["uid"] >= 0, "invalid retained account")
        _require(all(isinstance(k, str) and isinstance(v, str)
                     for k, v in _object(inputs["environment"]).items()),
                 "invalid complete retained environment")
        for value in (inputs, result):
            _require(value["mission_id"] == MISSION_ID, "wrong builder mission")
            _require(type(value["attempt"]) is int and value["attempt"] == 1, "wrong builder attempt")
        _require(result["members_unchanged"] is True and result["sources_unchanged"] is True,
                 "source/member preservation not explicit")
        _require(_same(inputs["authority"], result["authority"]), "authority pin disagreement")
        authority_path = _pin(inputs["authority"], repository, cache)
        _require(cache[authority_path] == cache[root / "build-authorization.json"],
                 "copied authority differs from original")
        authority = _fields(_json(cache[authority_path]), (
            "kind issuer mission_id expected_attempt authorized_science scientific_run_budget "
            "prospective_contract runtime_authorization_fields"
        ), (
            "schema_version issued_at_utc authorization_id decision predecessor_settlement "
            "immutable_history forbidden termination"
        ))
        _require(
            authority.get("kind") == "manager_stage11_operational_release_build_authorization"
            and authority.get("issuer") == "manager" and authority.get("mission_id") == MISSION_ID
            and type(authority.get("expected_attempt")) is int
            and authority["expected_attempt"] == 1,
            "wrong operational build authority",
        )
        _require(authority.get("authorized_science") is False, "build authority permits science")
        _zero(authority.get("scientific_run_budget"), "authority science budget")
        runtime = _fields(
            authority["runtime_authorization_fields"], "authorization_id authorization_action", "reason",
        )
        _require(runtime["authorization_id"] == runtime["authorization_action"] == "",
                 "authority must not populate runtime authorization fields")
        _zero(inputs["scientific_run_budget"], "release science budget")
        _require(inputs["authorization_id"] == inputs["authorization_action"] == "",
                 "runtime authorization fields must remain empty")
        contract = _object(authority.get("prospective_contract"))
        _require(contract.get("builder_identity") == MISSION_ID
                 and contract.get("output_root") == PACKAGE_RELATIVE, "authority ownership mismatch")
        sources = _list(inputs["sources"])
        roles = [_object(p).get("role") for p in sources]
        _require(all(isinstance(role, str) for role in roles), "invalid source role")
        _require(len(sources) == 4 and set(roles) == SOURCE_ROLES,
                 "missing required 30a/57 source roles")
        _require(_same(sources, contract.get("source_inputs_read_only")), "changed authority source pins")
        member_paths = {_pin(p, repository, cache) for p in _list(inputs["members"])}
        _require(bool(member_paths) and len(member_paths) == len(inputs["members"])
                 and all(p.is_relative_to(root) for p in member_paths),
                 "missing or nonlocal implementation members")
        for origin in _list(inputs["origins"]):
            _fields(origin, "member origin changed_bytes inherited_approval reason")
            _require(_pin(origin["member"], repository, cache) in member_paths, "unbound copied member")
            _require(origin["inherited_approval"] is False, "inherited approval is forbidden")
            _require(type(origin["changed_bytes"]) is bool, "missing changed-byte accounting")
            original = origin["origin"]
            if original is not None:
                _pin(original, repository, cache)
            changed = original is None or any(
                origin["member"][key] != original[key] for key in ("bytes", "sha256")
            )
            _require(origin["changed_bytes"] is changed, "incorrect changed-byte provenance")

    check("source_pins", source_pins)

    def package_status():
        for phase, value in phases.items():
            _require(value["phase"] == phase and value["status"] == "PASS",
                     f"{phase} status is {value['status']!r}, not PASS")
        for key in ("errors", "failures", "skipped"):
            _zero(phases["tests"][key], f"focused tests {key}")
        _require(type(phases["tests"]["tests"]) is int and phases["tests"]["tests"] > 0,
                 "no focused tests recorded")
        _require(operational["release_consumption_authorized"] is False,
                 "operational package must not issue consumption authority")

    check("package_status", package_status)

    def counters():
        for value in (result, *phases.values()):
            _zero(value.get("scientific_invocations"), "science counter")
            _zero(value.get("candidate_check_invocations"), "candidate-check counter")
        _require(type(result["package_validation_invocations"]) is int
                 and result["package_validation_invocations"] == 1,
                 "expected exactly one retained package validation")
        for value in phases.values():
            _require(value.get("dispatch_attempts") == [], "recorded forbidden dispatch")

    check("zero_science_and_candidate_checks", counters)

    def captures():
        _require(checks["source_pins"], "source/identity pins unavailable")
        _fields(result["commands"], "compile tests validate")
        _require(inputs["cwd"] == str(root / "candidate"), "wrong retained copied cwd")
        for phase in PHASES:
            command = result["commands"][phase]
            _fields(command, (
                "finished_at implementation_after implementation_before implementation_unchanged "
                "started_at stdout terminal"
            ))
            _require(_same(command, document(f"{phase}-capture/verification.json")),
                     f"{phase} retained verification mismatch")
            _require(command["implementation_unchanged"] is True
                     and _same(command["implementation_before"], command["implementation_after"]),
                     f"{phase} implementation changed")
            terminal = _fields(command["terminal"], (
                "capture_implementation_after exit_status files success timed_out"
            ))
            _require(terminal["success"] is True and terminal["timed_out"] is False,
                     f"{phase} capture not successfully completed")
            _zero(terminal["exit_status"], f"{phase} exit status")
            _require(_same(terminal, document(f"{phase}-capture/launcher.capture.json")),
                     f"{phase} terminal mismatch")
            _require(_same(phases[phase], document(f"{phase}-capture/launcher.stdout")),
                     f"{phase} stdout/result mismatch")
            stdout = _pin(command["stdout"], repository, cache)
            _require(stdout == root / f"{phase}-capture/launcher.stdout", "wrong stdout binding")
            identity = _fields(phases[phase]["actual_identity"], (
                "argv cwd environment interpreter no_bytecode optimization safe_path uid"
            ))
            expected_argv = [
                inputs["interpreter"]["path"], "-P", "-B", str(root / "operational.py"), phase,
            ]
            _require(
                identity["argv"] == expected_argv and identity["cwd"] == inputs["cwd"]
                and _same(identity["environment"], inputs["environment"])
                and _same(identity["interpreter"], inputs["interpreter"])
                and type(identity["uid"]) is int and _same(identity["uid"], inputs["uid"])
                and identity["no_bytecode"] is True and identity["safe_path"] is True,
                f"{phase} retained identity mismatch",
            )
            _zero(identity["optimization"], f"{phase} optimization")

    check("retained_captures", captures)

    def independent_review():
        path = _pin(review_pin, repository, cache)
        _require(not path.is_relative_to(root), "package cannot supply its own independent review")
        review.update(_fields(_json(cache[path]), (
            "checkpoint created_at frontier kind mission_context mission_id producer_role "
            "review round schema_version"
        ), "release_consumption"))
        _require(review["kind"] == "round_reviewed_handoff"
                 and review["producer_role"] == "reviewer"
                 and review["mission_id"] == MISSION_ID, "missing independent Host Reviewer handoff")
        decision = _fields(review["review"], "next_action operator_question reason status")
        report["independent_review_status"] = decision["status"]
        _require(decision["status"] == "done", "independent review not complete")
        _require(result["normal_independent_review"] == operational["normal_independent_review"]
                 == "REQUIRED", "independent-review requirement changed")

    check("independent_review", independent_review)
    admission = {}

    def release_review():
        _require(checks["independent_review"], "no authenticated independent review")
        _require("release_consumption" in review,
                 "completed package review does not provide release-consumption admissibility")
        admission.update(_fields(review.get("release_consumption"), (
            "decision package_census historical_failure_ids write_audit_limits"
        )))
        _require(admission["decision"] == "ADMISSIBLE", "no explicit release admissibility review")
        _require(_same(admission["package_census"], census_pin), "review covers different package bytes")

    check("release_review", release_review)

    def historical_failures():
        history = _fields(inputs["historical_six_gate_failures"], (
            "count_required_by_authority individual_failure_evidence reason status"
        ))
        _require(_same(history, operational["historical_six_gate_failures"]),
                 "historical-failure provenance disagreement")
        _require(type(history["count_required_by_authority"]) is int
                 and history["count_required_by_authority"] == 6, "six historical failures required")
        _require(history["status"] == "CLOSED",
                 "six historical failures remain unresolved; synthetic negatives are not closure")
        records = _list(history["individual_failure_evidence"])
        _require(len(records) == 6, "six individually pinned failure closures required")
        ids = []
        for record in records:
            _fields(record, "failure_id failure closure")
            failure_id = record["failure_id"]
            _require(isinstance(failure_id, str) and bool(failure_id) and failure_id not in ids,
                     "missing or duplicate historical failure identity")
            ids.append(failure_id)
            failure = _json(cache[_pin(record["failure"], repository, cache)])
            closure = _fields(_json(cache[_pin(record["closure"], repository, cache)]), (
                "failure_id status original_failure test_node"
            ))
            _require(failure.get("failure_id") == failure_id and failure.get("status") == "FAIL",
                     "original historical failure not preserved")
            _require(closure["failure_id"] == failure_id and closure["status"] == "PASS"
                     and _same(closure["original_failure"], record["failure"])
                     and isinstance(closure["test_node"], str) and bool(closure["test_node"]),
                     "closure not bound one-to-one to original failure")
        _require(checks["release_review"] and _same(ids, admission["historical_failure_ids"]),
                 "six-failure census is not independently reviewed")

    check("historical_failure_closure", historical_failures)

    def write_audits():
        _require(checks["source_pins"], "source/identity pins unavailable")
        expected_subprocesses = [{
            "argv": [inputs["interpreter"]["path"], "-P", "-B", str(root / "operational.py"), phase],
            "cwd": inputs["cwd"], "environment": inputs["environment"],
        } for phase in PHASES]
        _require(_same(result["observed_subprocesses"], expected_subprocesses),
                 "unexpected retained builder subprocess")
        for name in AUDITS:
            audit = _fields(document(f"{name}.write-audit.json"), (
                "attempt scope writes "
                + ("observed_subprocesses" if name == "builder" else "dispatch_attempts")
            ))
            _require(audit["attempt"] == str(root), "audit belongs to another attempt")
            _require(audit["scope"] == (BUILDER_AUDIT_SCOPE if name == "builder" else AUDIT_SCOPE),
                     "audit completeness/scope changed")
            if name == "builder":
                _require(_same(audit["observed_subprocesses"], expected_subprocesses),
                         "unexpected audited builder subprocess")
            else:
                _require(audit["dispatch_attempts"] == [], "audited forbidden dispatch")
            writes = _list(audit["writes"])
            if name in ("compile", "validate"):
                _require(writes == [], f"{name} was not read-only")
            for write in writes:
                _fields(write, "event path")
                _require(isinstance(write["event"], str)
                         and write["event"] in {"open", "os.mkdir", "os.remove", "os.rmdir"},
                         "unapproved mutation event")
                path = _path(write["path"], repository)
                limit = root / "tmp" if name == "tests" else root
                focused_log = (
                    name == "tests" and write["event"] == "open"
                    and path == root / "focused-tests.txt"
                )
                _require(path.is_relative_to(limit) or focused_log,
                         "write outside permitted audit root")

    check("write_audits", write_audits)

    def audit_limits():
        _require(checks["release_review"], "write-audit limitations lack explicit independent acceptance")
        limits = _fields(admission["write_audit_limits"], (
            "accepted scope unobserved_finalization_writes"
        ))
        _require(limits["accepted"] is True and limits["scope"] == AUDIT_SCOPE,
                 "Python-only audit scope not explicitly accepted")
        _require(limits["unobserved_finalization_writes"] == [
            str(root / name) for name in FINALIZATION_WRITES
        ], "unobserved finalization writes are not exactly acknowledged")

    check("write_audit_limits", audit_limits)
    if not blockers:
        report.update(status="PASS", blocked=False, release_consumption_admissible=True)
    return report


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    report = preflight_release()
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2 if report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
