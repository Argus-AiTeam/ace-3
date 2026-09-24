"""Locate retained Stage11 historical failure receipts without executing them.

Run ``python3 -B ace3/model/candidates/stage11_historical_six_failure_receipt_locator_v1.py``.
Sorted JSON goes to stdout; exit 0 means LOCATED, exit 2 means blocked.

Discovery is limited to the project's authority, handoff, backlog and build
metadata. Build discovery selects immediate trees containing ``stage11`` or
starting with ``s11-``; advertised individual pins may reference other local
build evidence. JSON/JSONL and JSON capture stdout are searchable; Markdown and
logs are not receipt indexes. No source, interpreter, model or service is loaded.
Project handoffs exclude role-session capsules. Symlinks are never followed.

The native index is ``historical_six_gate_failures`` with integer
``count_required_by_authority=6`` and ``individual_failure_evidence``. Each entry
has ``failure_id``, ``failure`` (path/bytes/sha256), and an optional ``closure``.
An original receipt must bind the same failure_id, status FAIL, and an explicit
``gate`` of account, workdir or interpreter. Unstructured error text is not used
to invent IDs, gate classifications or pins. Other original receipt fields are
preserved by the pin, not interpreted as authority by this locator.
Standalone FAIL receipts in Stage11 build trees (or explicitly ``stage=stage11``)
are also located when a separate retained metadata document advertises their
path/bytes/sha256 pin. Merely hashing an unindexed receipt is not authentication.

A closure is eligible only for *independent review* when its pinned JSON has
exactly failure_id/status/original_failure/test_node, status PASS, and the exact
original pin. This mirrors the existing consumption preflight's receipt binding,
but does not run that preflight or approve a closure. Matching hashes establish
retained-byte integrity, not independent attestation or Manager authorization.
Repeated citations across indexes coalesce; duplicate IDs within one index,
conflicting citations, malformed evidence and nonlocal paths block discovery.
ABSENT is always relative to the reported metadata scan scope.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05")
FROZEN_ROOT = REPOSITORY_ROOT / (
    "build/argus-stage11-authorized-execution-root-repair-016efb266000-attempt001"
)
GATES = {"account", "workdir", "interpreter"}
BOUNDARY = (
    "Read-only historical receipt discovery, not failure closure, release "
    "admissibility or execution authority. No validation/runtime/science/candidate "
    "--check/prefix/reference/admission/hardware dispatch. Native-S16-RTZ Q24 "
    "residual state remains wider than FP16; INT4 weights and FP16 operator/KV "
    "boundaries are unchanged. No strict-FP16-state W4A16, new-token or full-model "
    "admission. Independent Reviewer and later Manager decisions remain required."
)


class EvidenceError(ValueError):
    """A retained evidence binding cannot be used."""

    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code


def _require(condition, reason):
    if not condition:
        raise EvidenceError("AMBIGUOUS", reason)


def _json(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            _require(key not in value, f"duplicate JSON field: {key}")
            value[key] = child
        return value

    def constant(value):
        raise EvidenceError("AMBIGUOUS", f"nonfinite JSON constant: {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise EvidenceError("AMBIGUOUS", f"malformed JSON: {error}") from error


def _scopes(repository, project):
    return (
        repository / ".argus/live",
        project / "authorities",
        project / "handoffs",
        project / "backlog.jsonl",
        repository / "build",
    )


def _local_path(value, repository, scopes):
    _require(isinstance(value, str) and bool(value.strip()), "missing evidence path")
    _require("://" not in value, "nonlocal evidence URL")
    path = Path(value)
    _require(".." not in path.parts, "parent traversal in evidence path")
    path = path if path.is_absolute() else repository / path
    _require(
        any(path == root or (root.name != "backlog.jsonl" and root in path.parents)
            for root in scopes),
        f"nonlocal evidence path: {path}",
    )
    _require("role-sessions" not in path.parts, f"excluded role-session capsule: {path}")
    try:
        _require(path.resolve() == path, f"noncanonical or symlink evidence path: {path}")
    except (OSError, RuntimeError) as error:
        raise EvidenceError("AMBIGUOUS", f"unresolvable evidence path: {path}: {error}") from error
    return path


def _read_bytes(path):
    try:
        if not path.exists():
            raise EvidenceError("ABSENT", f"missing evidence file: {path}")
        _require(path.is_file(), f"not a regular evidence file: {path}")
        return path.read_bytes()
    except FileNotFoundError as error:
        raise EvidenceError("ABSENT", f"missing evidence file: {path}") from error
    except OSError as error:
        raise EvidenceError("AMBIGUOUS", f"unreadable evidence file: {path}: {error}") from error


def _inventory(root):
    """Yield metadata paths or explicit scan errors; never descend a symlink."""
    if not root.exists():
        yield root, "ABSENT", "missing scan root"
        return
    if root.is_symlink():
        yield root, "AMBIGUOUS", "symlink scan root"
        return
    if root.is_file():
        yield root, None, None
        return
    errors = []
    for directory, directories, files in os.walk(root, followlinks=False, onerror=errors.append):
        directories.sort()
        if root.name == "build" and Path(directory) == root:
            directories[:] = [
                name for name in directories
                if "stage11" in name.lower() or name.lower().startswith("s11-")
            ]
        for name in directories[:]:
            path = Path(directory) / name
            if name == "role-sessions":
                directories.remove(name)
            elif path.is_symlink():
                directories.remove(name)
                yield path, "AMBIGUOUS", "symlink evidence directory"
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix in {".json", ".jsonl", ".stdout"}:
                yield path, None, None
    for error in errors:
        yield Path(error.filename), "AMBIGUOUS", f"unreadable scan directory: {error.strerror}"


def _pin_shape(pin):
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"},
             "pin requires exactly path/bytes/sha256")
    _require(type(pin["bytes"]) is int and pin["bytes"] >= 0, "invalid pin byte count")
    digest = pin["sha256"]
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest), "invalid SHA-256 pin")


def _authenticated(pin, repository, scopes):
    _pin_shape(pin)
    path = _local_path(pin["path"], repository, scopes)
    raw = _read_bytes(path)
    _require(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"],
             f"failure/closure pin mismatch: {path}")
    value = _json(raw)
    _require(isinstance(value, dict), f"receipt is not an object: {path}")
    return value, {**pin, "path": str(path)}


def _record(record, repository, scopes):
    _require(isinstance(record, dict), "failure entry is not an object")
    _require({"failure_id", "failure"} <= record.keys()
             and record.keys() <= {"failure_id", "failure", "closure"},
             "failure entry requires failure_id/failure and optional closure only")
    failure_id = record["failure_id"]
    _require(isinstance(failure_id, str) and bool(failure_id.strip()),
             "missing failure_id")
    failure, pin = _authenticated(record["failure"], repository, scopes)
    _require(failure.get("failure_id") == failure_id and failure.get("status") == "FAIL",
             f"original receipt does not bind {failure_id} to FAIL")
    _require(isinstance(failure.get("gate"), str) and failure["gate"] in GATES,
             f"{failure_id}: absent or unsupported structured account/workdir/interpreter gate")
    eligibility = {
        "eligible_for_independent_closure_review": False,
        "status": "ABSENT",
        "reason": "No pinned one-to-one PASS closure; failure remains historical.",
    }
    if record.get("closure") is not None:
        closure, closure_pin = _authenticated(record["closure"], repository, scopes)
        _require(set(closure) == {"failure_id", "status", "original_failure", "test_node"},
                 f"{failure_id}: closure fields differ from the native receipt contract")
        _require(
            closure["failure_id"] == failure_id and closure["status"] == "PASS"
            and json.dumps(closure["original_failure"], sort_keys=True)
            == json.dumps(record["failure"], sort_keys=True)
            and isinstance(closure["test_node"], str) and bool(closure["test_node"].strip()),
            f"{failure_id}: closure is not bound one-to-one to the original failure pin",
        )
        eligibility = {
            "eligible_for_independent_closure_review": True,
            "status": "PINNED_PASS",
            "reason": "Retained binding only; independent closure review has not been established.",
            "closure": closure_pin,
            "test_node": closure["test_node"],
        }
    return {"failure_id": failure_id, "failure": pin, "gate": failure["gate"],
            "closure_eligibility": eligibility}


def locate_receipts(*, repository_root=REPOSITORY_ROOT, project_root=PROJECT_ROOT):
    """Read metadata and authenticate advertised individual pins, without replay."""
    repository, project = Path(repository_root).absolute(), Path(project_root).absolute()
    scopes = _scopes(repository, project)
    blockers, unresolved, found = [], [], {}
    originals, advertised, locations = {}, {}, {}
    scanned = 0

    def block(code, path, reason):
        blockers.append({"code": code, "path": str(path), "reason": reason})

    def retain(result, source, *, original_only=False):
        failure_id = result["failure_id"]
        if failure_id in found:
            previous = found[failure_id]
            _require(
                previous == result or (
                    original_only and previous["failure"] == result["failure"]
                    and previous["gate"] == result["gate"]
                ),
                f"conflicting receipts or closures for failure_id: {failure_id}",
            )
        else:
            found[failure_id] = result
        locations.setdefault(failure_id, set()).add(str(source))

    def history(value, path, pointer):
        location = f"{path}#{pointer}"
        _require(isinstance(value, dict), "historical_six_gate_failures is not an object")
        _require(type(value.get("count_required_by_authority")) is int
                 and value["count_required_by_authority"] == 6, "historical count must be six")
        entries = value.get("individual_failure_evidence")
        if entries is None:
            unresolved.append({
                "path": location,
                "reason": "Individual failure IDs/pins absent; retained summary is not a receipt.",
            })
            return
        _require(isinstance(entries, list), "individual_failure_evidence is not a list")
        ids = set()
        for index, entry in enumerate(entries):
            source = f"{location}/individual_failure_evidence/{index}"
            try:
                result = _record(entry, repository, scopes)
                failure_id = result["failure_id"]
                _require(failure_id not in ids, f"duplicate failure_id within index: {failure_id}")
                ids.add(failure_id)
                retain(result, source)
            except EvidenceError as error:
                block(error.code, source, str(error))

    def visit(value, path, pointer=""):
        if isinstance(value, dict):
            if "path" in value and ("sha256" in value or "bytes" in value):
                target = value["path"]
                if isinstance(target, str):
                    target = Path(target)
                    target = target if target.is_absolute() else repository / target
                    advertised.setdefault(str(target), []).append((value, path, pointer))
            if "failure_id" in value and value.get("status") == "FAIL" and (
                value.get("stage") == "stage11" or (
                    repository / "build" in path.parents
                    and any("stage11" in part.lower() or part.lower().startswith("s11-")
                            for part in path.relative_to(repository / "build").parts[:-1])
                )
            ) and pointer in {"", "/line-1"}:
                originals[str(path)] = value["failure_id"]
            for key, child in sorted(value.items()):
                child_pointer = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                if key == "historical_six_gate_failures":
                    try:
                        history(child, path, child_pointer)
                    except EvidenceError as error:
                        block(error.code, f"{path}#{child_pointer}", str(error))
                else:
                    visit(child, path, child_pointer)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path, f"{pointer}/{index}")

    for root in scopes:
        try:
            _local_path(str(root), repository, scopes)
        except EvidenceError as error:
            block(error.code, root, str(error))
            continue
        for path, code, reason in _inventory(root):
            if code:
                block(code, path, reason)
                continue
            try:
                path = _local_path(str(path), repository, scopes)
                raw = _read_bytes(path)
                scanned += 1
                markers = (b"historical_six_gate_failures", b"failure_id", b"sha256", b"\\u")
                if not any(marker in raw for marker in markers):
                    continue
                if path.suffix == ".jsonl":
                    for number, line in enumerate(raw.splitlines(), 1):
                        if any(marker in line for marker in markers):
                            visit(_json(line), path, f"/line-{number}")
                else:
                    visit(_json(raw), path)
            except (EvidenceError, RecursionError) as error:
                block(error.code if isinstance(error, EvidenceError) else "AMBIGUOUS",
                      path, str(error))

    for path, failure_id in sorted(originals.items()):
        references = [(pin, source, pointer)
                      for pin, source, pointer in advertised.get(path, [])
                      if str(source) != path]
        if not references:
            block("ABSENT", path, "Standalone Stage11 FAIL receipt has no independently retained pin.")
        for pin, source, pointer in references:
            try:
                result = _record({"failure_id": failure_id, "failure": pin}, repository, scopes)
                retain(result, f"{source}#{pointer}", original_only=True)
            except EvidenceError as error:
                block(error.code, f"{source}#{pointer}", str(error))

    records = sorted(found.values(), key=lambda value: value["failure_id"])
    if len(records) < 6:
        block("ABSENT", repository, f"Located {len(records)} of six distinct authenticated failures.")
    elif len(records) > 6:
        block("AMBIGUOUS", repository,
              f"Located {len(records)} failures; no authenticated unique six-member selection.")
    pins = [json.dumps(record["failure"], sort_keys=True) for record in records]
    if len(set(pins)) != len(pins):
        block("AMBIGUOUS", repository, "Different failure IDs reuse an original failure pin.")
    return {
        "schema_version": 1,
        "status": "BLOCKED" if blockers else "LOCATED",
        "blocked": bool(blockers),
        "required_count": 6,
        "located_count": len(records),
        "records": [
            {**record, "index_locations": sorted(locations[record["failure_id"]])}
            for record in records
        ],
        "blockers": sorted(blockers, key=lambda value: (value["path"], value["code"], value["reason"])),
        "unresolved_indexes": sorted(unresolved, key=lambda value: value["path"]),
        "scan": {
            "roots": [str(root) for root in scopes],
            "formats": ["json", "jsonl", "JSON capture stdout"],
            "excluded": ["role-sessions", "symlinks", "unstructured prose", "non-metadata files"],
            "build_tree_selector": "immediate directory contains stage11 or starts with s11-",
            "metadata_files_read": scanned,
        },
        "closure_proven": False,
        "release_consumption_admissible": False,
        "release_consumption_authorized": False,
        "normal_independent_review": "REQUIRED",
        "boundary": BOUNDARY,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = locate_receipts()
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2 if report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
